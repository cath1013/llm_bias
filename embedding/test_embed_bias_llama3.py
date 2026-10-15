"""
Tests for the fixed-prompt Bai-style WAT experiment.

    python3 embedding/test_embed_bias_llama3.py            # all tests
    python3 embedding/test_embed_bias_llama3.py span       # name filter

Same plain runner as `test_embed_bias.py` (no pytest in this environment): every
`test_*` either returns or raises.  The historical Scout tests in that file are
untouched.

Two things are under test here and they fail differently:

  * the FROZEN DESIGN (`make_bai_style_prompts.py`) — counts, exhaustive pair
    coverage, order balance, attribute-position balance, id shape, determinism.
    These are checked on the generated design AND on the prompts.csv on disk, so
    a stale CSV is a failure rather than a surprise at run time.

  * the RUNNER (`embed_bias_llama3_hidden.py`) — orientation, the exact
    unsmoothed behavioural arithmetic, the two per-layer representation scores,
    the wide schema, and the fail-fast rules.

Nothing here loads Llama 3.1.  The tokenizer and the model are fakes with exact,
checkable arithmetic.
"""

import os
import re
import sys
import json
import math
import inspect

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embed_bias as eb
import embed_bias_scout_hidden as sc
import embed_bias_llama3_hidden as lh
import make_bai_style_prompts as mk
import wat_fab
import wat_gender_brilliance as wgb


SOURCE = open(lh.__file__, encoding="utf-8").read()
GEN_SOURCE = open(mk.__file__, encoding="utf-8").read()


def _code_only(src):
    """
    Executable lines only: docstrings and comments stripped.

    The runner's prose names the things it refuses to do — `load_sessions`, the
    JSONL replay, SEAT, `max_memory` — so a naive substring scan of the whole
    file would fail on its own documentation.  What these tests must check is
    the CODE.
    """
    import ast as _ast
    tree = _ast.parse(src)
    lines = src.splitlines()
    drop = set()
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Constant)
                and isinstance(node.value.value, str)):
            drop.update(range(node.lineno - 1, node.end_lineno))
    return "\n".join(l for i, l in enumerate(lines)
                     if i not in drop and not l.lstrip().startswith("#"))


CODE = _code_only(SOURCE)

STIM = mk.load_stimuli()
DESIGN = mk.build_prompts(STIM, seed=0)
ON_DISK = mk.read_prompts()


def by_category(rows, category):
    return [r for r in rows if r["category"] == category]


# =========================================================
# Fakes
# =========================================================

class FakeTokenizer:
    """
    Word-piece-ish tokenizer: runs of at most four letters, digits and single
    punctuation marks, whitespace skipped.  Offsets are exact, so `grandpa`
    really does become two tokens and multi-token pooling is really exercised.

    `register_response` / `decode` are the generation half: a fake model appends
    one sentinel id, and decoding it returns the scripted response verbatim.
    Reconstructing text from the regex pieces would lose whitespace, and the
    behavioural parser is sensitive to exactly that.
    """
    TOK = re.compile(r"[A-Za-z]{1,4}|[0-9]+|[^\sA-Za-z0-9]")
    pad_token_id = None
    eos_token_id = 2

    def __init__(self, header="<|sys|>", footer="<|assistant|>"):
        self.header, self.footer = header, footer
        self._responses = {}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        sysmsg = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        return f"{self.header}{sysmsg}<|user|>{user}{self.footer}"

    def _pieces(self, text):
        return [(m.group(0), m.start(), m.end()) for m in self.TOK.finditer(text)]

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        pieces = self._pieces(text)
        ids = [abs(hash(p[0])) % 50000 for p in pieces]
        out = {"input_ids": torch.tensor([ids]),
               "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor([[[s, e] for _, s, e in pieces]])
        self._last = pieces
        return _Enc(out)

    def convert_ids_to_tokens(self, ids):
        lookup = {abs(hash(p[0])) % 50000: p[0] for p in self._last}
        return [lookup.get(i, "<unk>") for i in ids]

    def register_response(self, text):
        tid = 90000 + len(self._responses)
        self._responses[tid] = text
        return tid

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._responses.get(int(i), "") for i in ids)


class _Enc(dict):
    def pop(self, k, *a):
        return dict.pop(self, k, *a)


class FakeModel:
    """
    Returns `n_blocks + 1` hidden states.  Each token's vector is deterministic
    in (token position, layer), so a mean over a span is predictable by hand.
    `bad` injects a NaN/Inf at one hidden-state index.  `generate` replays a
    scripted response and records the sampling kwargs it was handed.
    """

    def __init__(self, n_blocks=4, dim=8, bad=None, bad_kind="nan",
                 response="", tokenizer=None):
        self.n_blocks, self.dim = n_blocks, dim
        self.bad, self.bad_kind = bad, bad_kind
        self.response, self.tokenizer = response, tokenizer
        self.generate_kwargs = None
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.generate_kwargs = dict(kw)
        tid = self.tokenizer.register_response(self.response)
        return torch.cat([input_ids, torch.tensor([[tid]])], dim=1)

    def __call__(self, input_ids=None, attention_mask=None,
                 output_hidden_states=False, **kw):
        if not output_hidden_states:
            raise AssertionError("the runner must always ask for hidden states")
        n = input_ids.shape[1]
        states = []
        for l in range(self.n_blocks + 1):
            g = torch.Generator().manual_seed(1000 + l)
            h = torch.randn(1, n, self.dim, generator=g, dtype=torch.float32)
            h = h + torch.arange(n, dtype=torch.float32)[None, :, None] * 0.01
            if self.bad is not None and l == self.bad:
                h[0, 2, 3] = float("nan") if self.bad_kind == "nan" else float("inf")
            states.append(h)
        return _Out(tuple(states))


class _Out:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class FakeArgs:
    def __init__(self, seed=0, max_new_tokens=64):
        self.seed = seed
        self.max_new_tokens = max_new_tokens


def perfect_response(record, roles):
    """A stereotype-congruent response: every A item to X, every B item to Y."""
    lines = []
    for item in roles["attributes"]:
        target = roles["X"] if item in roles["A"] else roles["Y"]
        lines.append(f"{item} - {target}")
    return "\n".join(lines)


def run_one(record, *, response=None, n_blocks=4, dim=8, model=None,
            tokenizer=None, acc=None, stim=None, args=None):
    """One frozen prompt through the real driver with fakes underneath."""
    stim = stim or STIM
    tok = tokenizer or FakeTokenizer()
    if model is None:
        roles = lh.prompt_roles(record, stim)
        text = perfect_response(record, roles) if response is None else response
        model = FakeModel(n_blocks=n_blocks, dim=dim, response=text, tokenizer=tok)
    rows, failures, meta = lh.run_prompts(
        [record], stim, model, tok, args=args or FakeArgs(), acc=acc,
        verbose=False)
    return rows, failures, meta, model


def extract_one(record, *, n_blocks=4, dim=8, model=None, tokenizer=None):
    """Just the forward half — vectors, meta, roles."""
    roles = lh.prompt_roles(record, STIM)
    tok = tokenizer or FakeTokenizer()
    mod = model or FakeModel(n_blocks=n_blocks, dim=dim, tokenizer=tok)
    templated, window = lh.build_chat_input(tok, lh.SYSTEM_PROMPT, record["prompt"])
    vectors, meta = lh.extract_prompt_vectors(
        templated, roles["phrases"], tok, mod, user_window=window,
        competitors=roles["phrases"])
    return vectors, meta, roles


# =========================================================
# A — the frozen design: counts and exhaustive coverage
# =========================================================

def test_the_design_is_exactly_272_prompts():
    assert len(DESIGN) == 272, len(DESIGN)
    assert len(ON_DISK) == 272, "prompts.csv on disk is not the 272-prompt design"
    assert mk.N_TOTAL == 272


def test_the_design_is_256_gender_plus_16_fab():
    gender = [r for r in DESIGN if r["domain"] == "gender"]
    fab = [r for r in DESIGN if r["domain"] == "fab"]
    assert len(gender) == 256, len(gender)
    assert len(fab) == 16, len(fab)
    assert {r["domain"] for r in DESIGN} == {"gender", "fab"}
    assert {r["category"] for r in fab} == {"fab"}


def test_every_gender_condition_has_64_prompts():
    assert set(mk.GENDER_CONDITIONS) == {"creativity", "warmth", "humor",
                                         "dedication"}
    for condition in mk.GENDER_CONDITIONS:
        assert len(by_category(DESIGN, condition)) == 64, condition


def test_each_gender_condition_covers_the_8x8_crossing_exactly_once():
    male, female = set(STIM["gender"]["male"]), set(STIM["gender"]["female"])
    assert len(male) == len(female) == 8
    for condition in mk.GENDER_CONDITIONS:
        seen = []
        for r in by_category(DESIGN, condition):
            m = r["group0"] if r["group0"] in male else r["group1"]
            f = r["group1"] if r["group0"] in male else r["group0"]
            seen.append((m, f))
        assert len(seen) == 64
        assert len(set(seen)) == 64, f"{condition}: a target pair repeats"
        assert set(seen) == {(m, f) for m in male for f in female}, condition


def test_fab_covers_the_4x4_pole_crossing_exactly_once():
    bri, eff = set(STIM["fab"]["brilliance"]), set(STIM["fab"]["effort"])
    assert len(bri) == len(eff) == 4
    seen = []
    for r in DESIGN:
        if r["domain"] != "fab":
            continue
        b = r["group0"] if r["group0"] in bri else r["group1"]
        e = r["group1"] if r["group0"] in bri else r["group0"]
        seen.append((b, e))
    assert len(seen) == 16 and len(set(seen)) == 16
    assert set(seen) == {(b, e) for b in bri for e in eff}


# =========================================================
# B — the frozen design: counterbalancing
# =========================================================

def test_gender_target_order_is_32_32_in_every_condition():
    male = set(STIM["gender"]["male"])
    for condition in mk.GENDER_CONDITIONS:
        rows = by_category(DESIGN, condition)
        male_first = sum(1 for r in rows if r["group0"] in male)
        assert male_first == 32, f"{condition}: {male_first} male-first"
        assert len(rows) - male_first == 32


def test_every_gender_word_leads_four_times_and_follows_four_times():
    words = STIM["gender"]["male"] + STIM["gender"]["female"]
    for condition in mk.GENDER_CONDITIONS:
        rows = by_category(DESIGN, condition)
        for w in words:
            first = sum(1 for r in rows if r["group0"] == w)
            second = sum(1 for r in rows if r["group1"] == w)
            assert first == 4, f"{condition}: {w!r} leads {first} times"
            assert second == 4, f"{condition}: {w!r} follows {second} times"


def test_fab_target_order_is_8_8():
    bri = set(STIM["fab"]["brilliance"])
    rows = [r for r in DESIGN if r["domain"] == "fab"]
    bri_first = sum(1 for r in rows if r["group0"] in bri)
    assert bri_first == 8, bri_first
    assert len(rows) - bri_first == 8


def test_every_fab_pole_leads_twice_and_follows_twice():
    rows = [r for r in DESIGN if r["domain"] == "fab"]
    for w in STIM["fab"]["brilliance"] + STIM["fab"]["effort"]:
        first = sum(1 for r in rows if r["group0"] == w)
        second = sum(1 for r in rows if r["group1"] == w)
        assert first == 2, f"{w!r} leads {first} times"
        assert second == 2, f"{w!r} follows {second} times"


def test_gender_attribute_positions_are_exactly_balanced():
    """8 words, 8 positions, 64 prompts -> every cell exactly 8."""
    for condition in mk.GENDER_CONDITIONS:
        words = (STIM["gender"]["brilliance"]
                 + STIM["gender"]["control"][condition])
        orders = [tuple(json.loads(r["attributes"]))
                  for r in by_category(DESIGN, condition)]
        table = mk.position_table(orders, words)
        cells = [c for counts in table.values() for c in counts]
        assert set(cells) == {8}, f"{condition}: position counts {sorted(set(cells))}"


def test_fab_field_positions_are_as_even_as_16_prompts_over_20_slots_allow():
    """16 < 20, so exact balance is impossible; no cell may differ by >1."""
    fields = STIM["fab"]["high_fab"] + STIM["fab"]["low_fab"]
    orders = [tuple(json.loads(r["attributes"]))
              for r in DESIGN if r["domain"] == "fab"]
    table = mk.position_table(orders, fields)
    cells = [c for counts in table.values() for c in counts]
    assert max(cells) - min(cells) <= 1, (min(cells), max(cells))
    assert set(cells) == {0, 1}
    for w, counts in table.items():
        assert sum(counts) == 16, f"{w!r} appears {sum(counts)} times"


def test_attribute_order_is_not_an_unvalidated_per_prompt_shuffle():
    """The balance claim must be asserted in the generator, not just hoped for."""
    gen_code = _code_only(GEN_SOURCE)
    assert "assert_position_balance" in gen_code
    assert "balanced_attribute_orders" in gen_code
    for cond in mk.GENDER_CONDITIONS:
        orders = [tuple(json.loads(r["attributes"]))
                  for r in by_category(DESIGN, cond)]
        mk.assert_position_balance(orders, sorted(set(orders[0])), cond)


# =========================================================
# C — the frozen design: stimulus content
# =========================================================

def test_every_gender_prompt_has_four_brilliance_and_four_control_items():
    brilliance = set(STIM["gender"]["brilliance"])
    for condition in mk.GENDER_CONDITIONS:
        control = set(STIM["gender"]["control"][condition])
        for r in by_category(DESIGN, condition):
            items = json.loads(r["attributes"])
            assert len(items) == 8 and len(set(items)) == 8, r["prompt_id"]
            assert len(set(items) & brilliance) == 4, r["prompt_id"]
            assert len(set(items) & control) == 4, r["prompt_id"]
            assert set(items) == brilliance | control, r["prompt_id"]


def test_creativity_is_the_four_word_set_containing_original():
    words = set(STIM["gender"]["control"]["creativity"])
    assert words == set(wgb.CONTROL_SETS["creativity4"])
    assert "original" in words
    assert len(words) == 4
    assert words != set(wgb.CONTROL_SETS["creativity"]), \
        "the 3-word creativity set must not be used"
    for r in by_category(DESIGN, "creativity"):
        assert "original" in json.loads(r["attributes"]), r["prompt_id"]


def test_dedication_is_the_adjectival_control():
    assert (set(STIM["gender"]["control"]["dedication"])
            == set(wgb.CONTROL_SETS["dedication_adj"]))


def test_the_brilliance_set_is_the_four_word_storage_set():
    assert set(STIM["gender"]["brilliance"]) == set(wgb.BRILLIANCE_SETS["storage4"])
    assert len(STIM["gender"]["brilliance"]) == 4


def test_every_fab_prompt_has_ten_high_and_ten_low_and_no_middle_tertile_field():
    high = set(STIM["fab"]["high_fab"])
    low = set(STIM["fab"]["low_fab"])
    middle = set(wat_fab.FIELD_LABELS) - set(wat_fab.HIGH_TERT) - set(wat_fab.LOW_TERT)
    assert len(middle) == 10, "the tertile split should leave 10 middle fields"
    assert high == set(wat_fab.HIGH_TERT) and low == set(wat_fab.LOW_TERT)
    for r in DESIGN:
        if r["domain"] != "fab":
            continue
        fields = json.loads(r["attributes"])
        assert len(fields) == 20 and len(set(fields)) == 20, r["prompt_id"]
        assert len(set(fields) & high) == 10, r["prompt_id"]
        assert len(set(fields) & low) == 10, r["prompt_id"]
        assert not set(fields) & middle, r["prompt_id"]


def test_the_bian_pole_set_is_never_used():
    bian_bri, bian_eff = wat_fab.POLE_SETS["bian"]
    bian = {w.lower() for w in list(bian_bri) + list(bian_eff)}
    for r in DESIGN:
        if r["domain"] != "fab":
            continue
        assert r["group0"].lower() not in bian and r["group1"].lower() not in bian
    leslie_bri, leslie_eff = wat_fab.POLE_SETS["leslie"]
    assert set(STIM["fab"]["brilliance"]) == set(leslie_bri)
    assert set(STIM["fab"]["effort"]) == set(leslie_eff)


def test_no_median_split_prompt_is_generated():
    """tertile only: the design has one FAB category and it is not a split label."""
    cats = {r["category"] for r in DESIGN if r["domain"] == "fab"}
    assert cats == {"fab"}, cats
    assert "median" not in _code_only(GEN_SOURCE).lower()
    assert "MEDIAN" not in CODE and "FAB_SPLITS" not in CODE


# =========================================================
# D — the frozen design: ids, schema, determinism
# =========================================================

def test_prompt_ids_have_no_zero_padding():
    for r in DESIGN:
        trial = r["prompt_id"].rsplit("_", 1)[1]
        assert not trial.startswith("0"), r["prompt_id"]
        assert str(int(trial)) == trial, r["prompt_id"]
        assert int(trial) == r["trial"]


def test_prompt_ids_are_exactly_gb_category_n_and_fab_n():
    expected = []
    for condition in mk.GENDER_CONDITIONS:
        expected += [f"gb_{condition}_{t}" for t in range(1, 65)]
    expected += [f"fab_{t}" for t in range(1, 17)]
    assert [r["prompt_id"] for r in DESIGN] == expected
    assert len(set(expected)) == 272
    for r in DESIGN:
        if r["domain"] == "fab":
            assert re.fullmatch(r"fab_(?:[1-9]|1[0-6])", r["prompt_id"])
            assert "tertile" not in r["prompt_id"]
        else:
            assert re.fullmatch(r"gb_(creativity|warmth|humor|dedication)_"
                                r"(?:[1-9]|[1-5][0-9]|6[0-4])", r["prompt_id"])
            assert "creativity4" not in r["prompt_id"]
            assert "dedication_adj" not in r["prompt_id"]


def test_prompts_csv_has_exactly_the_specified_columns():
    assert mk.COLUMNS == ["prompt_id", "domain", "category", "trial",
                          "group0", "group1", "attributes", "prompt"]
    with open(mk.PROMPTS_CSV, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == mk.COLUMNS, header
    for r in ON_DISK:
        assert set(r) == set(mk.COLUMNS), sorted(set(r) ^ set(mk.COLUMNS))


def test_generation_is_deterministic_under_the_same_seed():
    again = mk.build_prompts(STIM, seed=0)
    assert again == DESIGN, "the same seed produced a different design"
    other = mk.build_prompts(STIM, seed=1)
    assert [r["prompt_id"] for r in other] == [r["prompt_id"] for r in DESIGN]
    assert [r["group0"] for r in other] == [r["group0"] for r in DESIGN], \
        "pairing and order are structural and must not move with the seed"
    assert any(a["attributes"] != b["attributes"] for a, b in zip(other, DESIGN)), \
        "the seed must actually drive the attribute orderings"


def test_the_prompts_csv_on_disk_is_the_seed_zero_design():
    assert ON_DISK == DESIGN, \
        "prompts.csv is stale — rerun make_bai_style_prompts.py"


def test_the_prompt_wording_comes_from_the_behavioural_runners():
    gender = DESIGN[0]
    assert gender["prompt"] == wgb.build_prompt(
        [gender["group0"], gender["group1"]], json.loads(gender["attributes"]))
    fab = next(r for r in DESIGN if r["domain"] == "fab")
    assert fab["prompt"] == wat_fab.build_prompt(
        [fab["group0"], fab["group1"]], json.loads(fab["attributes"]), strict=False)
    gen_code = _code_only(GEN_SOURCE)
    assert "wgb.build_prompt" in gen_code and "wat_fab.build_prompt" in gen_code


def test_every_presented_stimulus_appears_in_its_prompt():
    for r in DESIGN:
        text = r["prompt"].lower()
        for w in [r["group0"], r["group1"]] + json.loads(r["attributes"]):
            assert w.lower() in text, f"{r['prompt_id']}: {w!r} missing"


def test_the_generator_validates_stimuli_against_the_behavioural_runners():
    broken = json.loads(json.dumps(STIM))
    broken["gender"]["male"][0] = "nephew"
    try:
        mk.validate_stimuli(broken)
    except SystemExit:
        return
    raise AssertionError("a stimulus list that disagrees with the runner was accepted")


# =========================================================
# E — the runner reads only the frozen design
# =========================================================

def test_no_behavioural_jsonl_session_dependency_remains():
    for banned in ("load_sessions", "session_D", "arm_is_present", "jsonl",
                   "assignments", "behavioural_model", "attrs_order",
                   "fields_order", "pick_order", "_assigned_items",
                   "eb.DATASETS", "eb.scoped", "IN_SCOPE", "gb_arms",
                   "fab_arms", "xlsx"):
        assert banned not in CODE, f"{banned!r} is still in the primary path"
    for name in ("eligible_sessions", "presented_stimuli", "presented_arm_names",
                 "recover_orders_from_prompt", "layer_correlations",
                 "is_same_model", "aggregate_pairs"):
        assert not hasattr(lh, name), f"{name} survived the refactor"


def test_the_runner_consumes_stimuli_csv_and_prompts_csv():
    assert lh.STIMULI_CSV.endswith("stimuli.csv")
    assert lh.PROMPTS_CSV.endswith("prompts.csv")
    stim, prompts = lh.load_design(lh.STIMULI_CSV, lh.PROMPTS_CSV)
    assert len(prompts) == 272
    assert set(stim) == {"gender", "fab"}


def test_a_prompts_file_of_the_wrong_size_is_refused():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prompts.csv")
        mk.write_prompts(DESIGN[:10], path)
        try:
            lh.load_design(lh.STIMULI_CSV, path)
        except SystemExit as exc:
            assert "272" in str(exc)
            return
    raise AssertionError("a truncated design was accepted")


def test_the_scout_backend_is_not_imported():
    """The span layer is reused; the MoE/offload/quantisation loader is not."""
    imported = re.search(r"from embed_bias_scout_hidden import \(([^)]*)\)", SOURCE)
    assert imported, "the span-layer import disappeared"
    names = {n.strip() for n in imported.group(1).replace("\n", "").split(",")
             if n.strip()}
    assert names == {"SpanError", "find_phrase_char_spans", "token_span_for_chars",
                     "masking_spans", "resolve_span", "pool_span",
                     "build_chat_input"}, names
    for name in ("SCOUT_BEHAVIOURAL_MODELS", "QUANTISATIONS", "load_backend",
                 "parameter_footprint"):
        assert not hasattr(lh, name), f"{name} leaked into the runner"


# =========================================================
# F — orientation
# =========================================================

def test_roles_come_from_stimuli_not_from_presentation_order():
    male = set(STIM["gender"]["male"])
    for r in DESIGN:
        if r["domain"] != "gender":
            continue
        roles = lh.prompt_roles(r, STIM)
        assert roles["X"] in male, r["prompt_id"]
        assert roles["Y"] not in male, r["prompt_id"]
    for r in DESIGN:
        if r["domain"] != "fab":
            continue
        roles = lh.prompt_roles(r, STIM)
        assert roles["X"] in STIM["fab"]["brilliance"], r["prompt_id"]
        assert roles["Y"] in STIM["fab"]["effort"], r["prompt_id"]


def test_the_same_pair_resolves_to_the_same_roles_in_either_presentation_order():
    """
    Each pair is presented once per condition, so the design itself never shows
    a pair both ways round.  The invariance is therefore tested directly: the
    row is mirrored by hand and the roles must not move.
    """
    male = set(STIM["gender"]["male"])
    record = next(r for r in by_category(DESIGN, "creativity")
                  if r["group0"] in male)
    mirrored = dict(record, group0=record["group1"], group1=record["group0"],
                    prompt=wgb.build_prompt(
                        [record["group1"], record["group0"]],
                        json.loads(record["attributes"])))
    a, b = lh.prompt_roles(record, STIM), lh.prompt_roles(mirrored, STIM)
    assert (a["X"], a["Y"]) == (b["X"], b["Y"])
    assert a["A"] == b["A"] and a["B"] == b["B"]
    assert a["X"] in male and b["X"] in male


def test_presentation_order_is_a_fixed_function_of_the_pair():
    """
    A consequence of full coverage with one prompt per pair: order cannot be
    crossed with pair identity, so it is fixed by index parity and is the SAME
    in all four conditions.  Worth asserting, because it is the reason the
    orientation rule has to come from stimuli.csv rather than from the design.
    """
    male = set(STIM["gender"]["male"])

    def oriented(rows):
        out = {}
        for r in rows:
            m = r["group0"] if r["group0"] in male else r["group1"]
            f = r["group1"] if r["group0"] in male else r["group0"]
            out[(m, f)] = r["group0"] in male
        return out

    base = oriented(by_category(DESIGN, "creativity"))
    assert set(base.values()) == {True, False}
    assert sum(base.values()) == 32
    for condition in mk.GENDER_CONDITIONS[1:]:
        assert oriented(by_category(DESIGN, condition)) == base, condition


def test_the_scoring_code_never_reads_group0_or_group1():
    for fn in (lh.representation_scores, lh.behaviour_wat_score, lh.wide_row):
        src = _code_only(inspect.getsource(fn))
        assert "group0" not in src and "group1" not in src, fn.__name__


def test_targets_that_do_not_resolve_to_one_x_and_one_y_are_refused():
    bad = dict(DESIGN[0])
    bad["group1"] = bad["group0"]          # two male targets
    try:
        lh.prompt_roles(bad, STIM)
    except SystemExit as exc:
        assert "exactly one" in str(exc)
        return
    raise AssertionError("a prompt with two same-pool targets was accepted")


def test_a_reduced_attribute_set_is_refused_rather_than_scored():
    bad = dict(DESIGN[0])
    bad["attributes"] = json.dumps(json.loads(bad["attributes"])[:6])
    try:
        lh.prompt_roles(bad, STIM)
    except SystemExit as exc:
        assert "reduced set" in str(exc)
        return
    raise AssertionError("a prompt missing attributes was scored anyway")


# =========================================================
# G — behavioural arithmetic
# =========================================================

def test_the_behavioural_score_is_exact_and_unsmoothed():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    assign = {w: (roles["X"] if w in roles["A"] else roles["Y"])
              for w in roles["attributes"]}
    score, problem = lh.behaviour_wat_score(assign, record, roles)
    assert problem is None
    assert score == 1.0, score          # 4/4 + 4/4 - 1; smoothed would be 0.99501
    assert score != 4 / 4.01 + 4 / 4.01 - 1


def test_the_behavioural_score_matches_the_bai_formula_on_a_mixed_assignment():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    X, Y = roles["X"], roles["Y"]
    A, B = roles["A"], roles["B"]
    assign = {A[0]: X, A[1]: X, A[2]: X, A[3]: Y,
              B[0]: X, B[1]: Y, B[2]: Y, B[3]: Y}
    # male-assigned: 3 brilliance + 1 control -> P(brilliance|male) = 3/4
    # female-assigned: 1 brilliance + 3 control -> P(control|female) = 3/4
    score, problem = lh.behaviour_wat_score(assign, record, roles)
    assert problem is None
    assert abs(score - (3 / 4 + 3 / 4 - 1)) < 1e-12, score
    assert abs(score - 0.5) < 1e-12


def test_the_fab_behavioural_score_is_high_fab_to_brilliance():
    record = next(r for r in DESIGN if r["domain"] == "fab")
    roles = lh.prompt_roles(record, STIM)
    assign = {w: (roles["X"] if w in roles["A"] else roles["Y"])
              for w in roles["attributes"]}
    score, problem = lh.behaviour_wat_score(assign, record, roles)
    assert problem is None and score == 1.0
    flipped = {w: (roles["Y"] if w in roles["A"] else roles["X"])
               for w in roles["attributes"]}
    score2, _ = lh.behaviour_wat_score(flipped, record, roles)
    assert score2 == -1.0, score2


def test_the_behavioural_sign_does_not_follow_presentation_order():
    male = set(STIM["gender"]["male"])
    rows = by_category(DESIGN, "humor")
    male_first = next(r for r in rows if r["group0"] in male)
    female_first = next(r for r in rows if r["group0"] not in male)
    for record in (male_first, female_first):
        roles = lh.prompt_roles(record, STIM)
        assign = {w: (roles["X"] if w in roles["A"] else roles["Y"])
                  for w in roles["attributes"]}
        score, _ = lh.behaviour_wat_score(assign, record, roles)
        assert score == 1.0, record["prompt_id"]


def test_an_empty_denominator_is_nan_not_a_smoothed_number():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    assign = {w: roles["X"] for w in roles["attributes"]}   # all to one target
    score, problem = lh.behaviour_wat_score(assign, record, roles)
    assert math.isnan(score)
    assert problem and roles["Y"] in problem


def test_only_the_two_presented_targets_are_accepted():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    other_male = next(w for w in STIM["gender"]["male"] if w != roles["X"])
    lines = [f"{roles['attributes'][0]} - {other_male}"]
    lines += [f"{w} - {roles['X']}" for w in roles["attributes"][1:]]
    assign, problems = lh.parse_behaviour("\n".join(lines), record, roles)
    assert any(other_male in p for p in problems), problems
    assert roles["attributes"][0] not in assign


def test_an_item_assigned_twice_invalidates_the_response():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    lines = [f"{w} - {roles['X']}" for w in roles["attributes"]]
    lines.append(f"{roles['attributes'][0]} - {roles['Y']}")
    assign, problems = lh.parse_behaviour("\n".join(lines), record, roles)
    assert any("more than once" in p for p in problems), problems


def test_a_missing_item_invalidates_the_response():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    lines = [f"{w} - {roles['X']}" for w in roles["attributes"][:-1]]
    assign, problems = lh.parse_behaviour("\n".join(lines), record, roles)
    assert any("never assigned" in p for p in problems), problems


def test_the_metric_is_not_named_weat():
    assert hasattr(lh, "behaviour_wat_score")
    assert not hasattr(lh, "behaviour_weat")
    assert "behaviour_wat_score" in lh.primary_columns(33)
    src = _code_only(inspect.getsource(lh.behaviour_wat_score))
    assert "weat" not in src.lower().replace("behaviour_wat_score", ""), \
        "WEAT is reserved for the representation metric"


# =========================================================
# H — representation scores
# =========================================================

def _toy_roles():
    """Hand-built vectors with 2 hidden states and a known geometry."""
    rng = np.random.default_rng(7)
    words = ["X", "Y", "a1", "a2", "b1", "b2"]
    vectors = {w: [rng.normal(size=5), rng.normal(size=5)] for w in words}
    roles = dict(X="X", Y="Y", A=["a1", "a2"], B=["b1", "b2"])
    return vectors, roles


def test_delta_is_the_unstandardised_difference_of_the_two_associations():
    vectors, roles = _toy_roles()
    deltas, weats = lh.representation_scores(vectors, roles, 2)
    for i in range(2):
        A = np.asarray([vectors[w][i] for w in roles["A"]])
        B = np.asarray([vectors[w][i] for w in roles["B"]])
        x = np.asarray(vectors["X"][i])[None, :]
        y = np.asarray(vectors["Y"][i])[None, :]
        s_X = float(np.mean(eb.cos_sim(x, A)) - np.mean(eb.cos_sim(x, B)))
        s_Y = float(np.mean(eb.cos_sim(y, A)) - np.mean(eb.cos_sim(y, B)))
        assert abs(deltas[i] - (s_X - s_Y)) < 1e-12


def test_weat_is_the_population_sd_standardisation_with_the_bai_epsilon():
    vectors, roles = _toy_roles()
    deltas, weats = lh.representation_scores(vectors, roles, 2)
    for i in range(2):
        A = np.asarray([vectors[w][i] for w in roles["A"]])
        B = np.asarray([vectors[w][i] for w in roles["B"]])
        x = np.asarray(vectors["X"][i])[None, :]
        y = np.asarray(vectors["Y"][i])[None, :]
        s_X = float(eb.weat_association(x, A, B)[0])
        s_Y = float(eb.weat_association(y, A, B)[0])
        sd = float(np.std(np.array([s_X, s_Y]), ddof=0))
        assert abs(weats[i] - (s_X - s_Y) / (sd + 0.001)) < 1e-12
        # the singleton identity: pop SD of two numbers is half their gap
        assert abs(weats[i] - deltas[i] / (abs(deltas[i]) / 2 + 0.001)) < 1e-9


def test_the_sample_sd_would_give_a_different_answer():
    """Guards the ddof=0 choice against a silent change to ddof=1."""
    vectors, roles = _toy_roles()
    _, weats = lh.representation_scores(vectors, roles, 2)
    A = np.asarray([vectors[w][0] for w in roles["A"]])
    B = np.asarray([vectors[w][0] for w in roles["B"]])
    x = np.asarray(vectors["X"][0])[None, :]
    y = np.asarray(vectors["Y"][0])[None, :]
    s_X = float(eb.weat_association(x, A, B)[0])
    s_Y = float(eb.weat_association(y, A, B)[0])
    sample = (s_X - s_Y) / (float(np.std(np.array([s_X, s_Y]), ddof=1)) + 0.001)
    assert abs(weats[0] - sample) > 1e-6


def test_the_representation_sign_follows_the_roles_not_the_order():
    """Swapping which target is X flips both scores, and nothing else does."""
    vectors, roles = _toy_roles()
    deltas, weats = lh.representation_scores(vectors, roles, 2)
    swapped = dict(roles, X=roles["Y"], Y=roles["X"])
    d2, w2 = lh.representation_scores(vectors, swapped, 2)
    for i in range(2):
        assert abs(deltas[i] + d2[i]) < 1e-12
        assert abs(weats[i] + w2[i]) < 1e-9


def test_both_scores_are_kept_because_the_standardised_one_saturates():
    """A tiny delta still produces a near-+-2 weat; that is why delta is kept."""
    v = np.ones(4) / 2.0
    tiny = v + np.array([1e-6, 0.0, 0.0, 0.0])
    vectors = {"X": [v], "Y": [tiny], "a1": [np.array([1.0, 0, 0, 0])],
               "b1": [np.array([0, 1.0, 0, 0])]}
    roles = dict(X="X", Y="Y", A=["a1"], B=["b1"])
    deltas, weats = lh.representation_scores(vectors, roles, 1)
    assert abs(deltas[0]) < 1e-4, deltas[0]
    assert abs(weats[0]) == abs(deltas[0]) / (abs(deltas[0]) / 2 + 0.001)


# =========================================================
# I — the wide primary table
# =========================================================

def test_the_primary_table_has_the_specified_column_order():
    cols = lh.primary_columns(33)
    assert cols[:7] == ["prompt_id", "domain", "category", "trial", "prompt",
                        "response", "behaviour_wat_score"]
    assert cols[7:40] == [f"weat_layer{i}" for i in range(33)]
    assert cols[40:] == [f"delta_layer{i}" for i in range(33)]
    assert len(cols) == 7 + 66


def test_no_s_group0_or_s_group1_columns_are_written():
    cols = lh.primary_columns(33)
    assert not any(c.startswith(("s_group0", "s_group1")) for c in cols)
    assert "s_group0_layer0" not in CODE and "s_group1_layer0" not in CODE
    rows, _, meta, _ = run_one(DESIGN[0], n_blocks=4)
    assert not any(k.startswith(("s_group0", "s_group1")) for k in rows[0])
    assert set(rows[0]) == set(lh.primary_columns(meta["n_hidden_states"]))


def test_group_and_attribute_columns_stay_in_prompts_csv():
    rows, _, _, _ = run_one(DESIGN[0], n_blocks=4)
    for col in ("group0", "group1", "attributes"):
        assert col not in rows[0], f"{col} duplicated into the primary table"
    assert "prompt_id" in rows[0], "the join key must be present"


def test_one_row_per_prompt_with_the_response_preserved_verbatim():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    text = perfect_response(record, roles)
    rows, failures, meta, _ = run_one(record, response=text, n_blocks=4)
    assert len(rows) == 1 and not failures
    assert rows[0]["response"] == text
    assert rows[0]["prompt"] == record["prompt"]
    assert rows[0]["behaviour_wat_score"] == 1.0
    assert rows[0]["trial"] == record["trial"]


def test_the_wide_row_rejects_a_mismatched_score_vector():
    try:
        lh.wide_row(DESIGN[0], "r", 0.0, [0.1, 0.2], [0.1])
    except ValueError as exc:
        assert "deltas" in str(exc)
        return
    raise AssertionError("mismatched delta/weat lengths were accepted")


def test_the_primary_writer_refuses_an_unexpected_column():
    import tempfile
    rows, _, meta, _ = run_one(DESIGN[0], n_blocks=4)
    rows[0]["group0"] = "man"
    with tempfile.TemporaryDirectory() as d:
        try:
            lh.write_primary_csv(os.path.join(d, "x.csv"), rows,
                                 meta["n_hidden_states"])
        except ValueError as exc:
            assert "group0" in str(exc)
            return
    raise AssertionError("an extra column reached the primary table")


# =========================================================
# J — generation and the paired forward pass
# =========================================================

def test_the_generation_setting_is_the_chosen_low_nonzero_sampling():
    assert lh.GENERATION["do_sample"] is True
    assert lh.GENERATION["temperature"] == 0.3
    _, _, _, model = run_one(DESIGN[0], n_blocks=4)
    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["temperature"] == 0.3
    assert "max_new_tokens" in model.generate_kwargs


def test_the_generation_seed_is_derived_from_the_prompt_id_not_the_row_order():
    a = lh.prompt_seed("gb_creativity_1", 0)
    assert a == lh.prompt_seed("gb_creativity_1", 0)
    assert a != lh.prompt_seed("gb_creativity_2", 0)
    assert a != lh.prompt_seed("gb_creativity_1", 1)
    seeds = {lh.prompt_seed(r["prompt_id"], 0) for r in DESIGN}
    assert len(seeds) == len(DESIGN), "per-prompt seeds collide"


def test_generation_and_extraction_see_the_same_templated_input():
    """`build_chat_input` is called once and the string is reused for both."""
    src = _code_only(inspect.getsource(lh.run_prompts))
    assert src.count("build_chat_input") == 1
    assert "generate_response(" in src and "extract_prompt_vectors(" in src
    assert src.index("templated") < src.index("generate_response(")


def test_hidden_states_are_never_taken_from_the_generated_response():
    src = _code_only(inspect.getsource(lh.extract_prompt_vectors))
    assert "generate" not in src
    assert "output_hidden_states=True" in src
    assert "inference_mode" in src
    gen = _code_only(inspect.getsource(lh.generate_response))
    assert "output_hidden_states" not in gen


# =========================================================
# K — spans, pooling, substring collisions
# =========================================================

def test_multi_token_stimuli_are_mean_pooled_over_every_sub_token():
    record = DESIGN[0]
    tok = FakeTokenizer()
    vectors, meta, roles = extract_one(record, tokenizer=tok)
    multi = [p for p in meta["subtokens"] if len(meta["subtokens"][p]) > 1]
    assert multi, "the fake tokenizer produced no multi-token stimulus"
    assert "super smart" in multi
    assert meta["pooling"] == "mean"

    mod = FakeModel(n_blocks=4, dim=8, tokenizer=tok)
    templated, _ = lh.build_chat_input(tok, lh.SYSTEM_PROMPT, record["prompt"])
    enc = tok(templated, return_tensors="pt", return_offsets_mapping=True)
    enc.pop("offset_mapping")
    out = mod(**enc, output_hidden_states=True)
    phrase = multi[0]
    lo, hi = meta["spans"][phrase]
    assert hi - lo == len(meta["subtokens"][phrase]) > 1
    expected = out.hidden_states[2][0, lo:hi].numpy().astype(float).mean(axis=0)
    assert np.allclose(vectors[phrase][2], expected), \
        "a multi-token phrase was not mean-pooled over its whole span"


def test_first_token_only_extraction_is_not_used():
    record = DESIGN[0]
    vectors, meta, _ = extract_one(record)
    phrase = "super smart"
    lo, hi = meta["spans"][phrase]
    tok = FakeTokenizer()
    mod = FakeModel(n_blocks=4, dim=8, tokenizer=tok)
    templated, _ = lh.build_chat_input(tok, lh.SYSTEM_PROMPT, record["prompt"])
    enc = tok(templated, return_tensors="pt", return_offsets_mapping=True)
    enc.pop("offset_mapping")
    out = mod(**enc, output_hidden_states=True)
    first_token = out.hidden_states[1][0, lo].numpy().astype(float)
    assert not np.allclose(vectors[phrase][1], first_token)


def _resolved_char_start(prompt, phrase, competitors):
    spans = lh.find_phrase_char_spans(prompt, phrase)
    blocked = lh.masking_spans(prompt, phrase, competitors)
    keep = [s for s in spans
            if not any(b0 <= s[0] and s[1] <= b1 for b0, b1 in blocked)]
    assert len(keep) == 1, (phrase, spans, blocked)
    return keep[0][0]


def test_man_does_not_resolve_inside_woman():
    record = next(r for r in DESIGN
                  if {r["group0"], r["group1"]} == {"man", "woman"})
    roles = lh.prompt_roles(record, STIM)
    vectors, meta, _ = extract_one(record)
    tok = FakeTokenizer()
    templated, _ = lh.build_chat_input(tok, lh.SYSTEM_PROMPT, record["prompt"])
    want = _resolved_char_start(templated, "man", roles["phrases"])
    offsets = [(s, e) for _, s, e in tok._pieces(templated)]
    lo, hi = meta["spans"]["man"]
    assert offsets[lo][0] == want, (offsets[lo], want)
    assert meta["spans"]["man"] != meta["spans"]["woman"]


def test_male_does_not_resolve_inside_female():
    record = next(r for r in DESIGN
                  if {r["group0"], r["group1"]} == {"male", "female"})
    roles = lh.prompt_roles(record, STIM)
    vectors, meta, _ = extract_one(record)
    tok = FakeTokenizer()
    templated, _ = lh.build_chat_input(tok, lh.SYSTEM_PROMPT, record["prompt"])
    want = _resolved_char_start(templated, "male", roles["phrases"])
    offsets = [(s, e) for _, s, e in tok._pieces(templated)]
    lo, _ = meta["spans"]["male"]
    assert offsets[lo][0] == want


def test_history_does_not_resolve_inside_art_history():
    """
    `art history` is a middle-tertile field and so is not presented by this
    design, but the masking rule must keep working: it is what stops a longer
    stimulus from swallowing a shorter one, and the next design may present both.
    """
    prompt = ("Here is a list of words. For each word pick a word — innate gift "
              "or hard work — and write it after the word. The words are "
              "['art history', 'history', 'physics'].")
    competitors = ["art history", "history", "physics", "innate gift", "hard work"]
    blocked = lh.masking_spans(prompt, "history", competitors)
    assert blocked, "'art history' did not mask anything"
    start = _resolved_char_start(prompt, "history", competitors)
    assert prompt[start - 1:start] == "'", prompt[start - 5:start + 10]
    assert not any(b0 <= start < b1 for b0, b1 in blocked)


def test_every_stimulus_in_every_frozen_prompt_resolves_exactly_once():
    tok = FakeTokenizer()
    for record in DESIGN:
        roles = lh.prompt_roles(record, STIM)
        templated, window = lh.build_chat_input(tok, lh.SYSTEM_PROMPT,
                                                record["prompt"])
        offsets = [(s, e) for _, s, e in tok._pieces(templated)]
        for phrase in roles["phrases"]:
            lh.resolve_span(templated, offsets, phrase, window=window,
                            competitors=[c for c in roles["phrases"] if c != phrase])


def test_a_stimulus_absent_from_the_prompt_aborts_the_run():
    record = dict(DESIGN[0])
    missing = json.loads(record["attributes"])[0]
    record["prompt"] = record["prompt"].replace(f"'{missing}', ", "")
    try:
        run_one(record, n_blocks=4)
    except lh.SpanError as exc:
        assert missing in str(exc)
        return
    raise AssertionError("a missing stimulus was silently tolerated")


def test_an_ambiguous_stimulus_span_aborts_the_run():
    record = dict(DESIGN[0])
    item = json.loads(record["attributes"])[0]
    record["prompt"] = record["prompt"] + f" Also consider {item}."
    try:
        run_one(record, n_blocks=4)
    except lh.SpanError as exc:
        assert "occurs 2 times" in str(exc) or "refusing to guess" in str(exc)
        return
    raise AssertionError("an ambiguous span was resolved by guessing")


# =========================================================
# L — fail fast on non-finite values
# =========================================================

def test_a_non_finite_hidden_state_is_reported_with_its_index_and_counts():
    h = [torch.zeros(1, 4, 8) for _ in range(5)]
    h[3][0, 1, 2] = float("nan")
    h[3][0, 2, 2] = float("inf")
    try:
        lh.assert_finite_hidden_states(h, "unit test")
    except lh.NonFiniteHiddenState as exc:
        msg = str(exc)
        assert "index 3" in msg and "1 NaN" in msg and "1 Inf" in msg, msg
        assert "aborting the run" in msg
        return
    raise AssertionError("a NaN hidden state was accepted")


def test_the_first_offending_hidden_state_is_reported_not_the_last():
    h = [torch.zeros(1, 4, 8) for _ in range(6)]
    h[2][0, 0, 0] = float("nan")
    h[5][0, 0, 0] = float("nan")
    try:
        lh.assert_finite_hidden_states(h)
    except lh.NonFiniteHiddenState as exc:
        assert "index 2" in str(exc), str(exc)
        return
    raise AssertionError("not raised")


def test_a_non_finite_hidden_state_aborts_the_whole_run_not_just_the_prompt():
    tok = FakeTokenizer()
    roles = lh.prompt_roles(DESIGN[0], STIM)
    bad = FakeModel(n_blocks=4, dim=8, bad=2, tokenizer=tok,
                    response=perfect_response(DESIGN[0], roles))
    try:
        lh.run_prompts(DESIGN[:3], STIM, bad, tok, args=FakeArgs(), verbose=False)
    except lh.NonFiniteHiddenState as exc:
        assert "index 2" in str(exc)
        return
    raise AssertionError("the run continued past a NaN forward pass")


def test_a_non_finite_pooled_vector_aborts_the_run():
    """The second guard: finite hidden states, broken pooling."""
    real_pool = lh.pool_span

    def poisoned(layer_matrix, span, rule="mean"):
        v = np.array(real_pool(layer_matrix, span, rule), dtype=float)
        v[0] = float("nan")
        return v

    lh.pool_span = poisoned
    try:
        run_one(DESIGN[0], n_blocks=4)
    except lh.NonFiniteHiddenState as exc:
        assert "pooled vector" in str(exc)
        assert "aborting the run" in str(exc)
        return
    finally:
        lh.pool_span = real_pool
    raise AssertionError("a NaN pooled vector reached the statistics")


def test_the_runner_never_skips_a_prompt_to_keep_going():
    src = _code_only(inspect.getsource(lh.run_prompts))
    assert "continue" not in src, \
        "a skip in the prompt loop would break exhaustive pair coverage"
    assert "except" not in src or "NonFiniteHiddenState" not in src


# =========================================================
# M — a malformed response keeps its representation row
# =========================================================

def test_a_malformed_response_keeps_the_representation_observation():
    record = DESIGN[0]
    rows, failures, meta, _ = run_one(record, response="I'd rather not.",
                                      n_blocks=4)
    assert len(rows) == 1, "the representation row was dropped"
    row = rows[0]
    assert math.isnan(row["behaviour_wat_score"])
    assert row["response"] == "I'd rather not."
    assert row["prompt"] == record["prompt"]
    n = meta["n_hidden_states"]
    assert all(isinstance(row[f"delta_layer{i}"], float)
               and not math.isnan(row[f"delta_layer{i}"]) for i in range(n))
    assert all(not math.isnan(row[f"weat_layer{i}"]) for i in range(n))
    assert len(failures) == 1
    assert failures[0]["prompt_id"] == record["prompt_id"]
    assert failures[0]["response"] == "I'd rather not."


def test_a_partially_parsed_response_is_also_a_failure_not_a_partial_score():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    half = "\n".join(f"{w} - {roles['X']}" for w in roles["attributes"][:4])
    rows, failures, _, _ = run_one(record, response=half, n_blocks=4)
    assert math.isnan(rows[0]["behaviour_wat_score"])
    assert len(failures) == 1
    assert "never assigned" in failures[0]["problems"]


def test_an_empty_denominator_is_logged_as_a_failure_but_keeps_the_row():
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    everything_male = "\n".join(f"{w} - {roles['X']}"
                                for w in roles["attributes"])
    rows, failures, _, _ = run_one(record, response=everything_male, n_blocks=4)
    assert len(rows) == 1
    assert math.isnan(rows[0]["behaviour_wat_score"])
    assert len(failures) == 1 and roles["Y"] in failures[0]["problems"]


def test_a_well_formed_response_with_noise_around_it_still_parses():
    """The runners' own line parser is reused, hedges and all."""
    record = DESIGN[0]
    roles = lh.prompt_roles(record, STIM)
    body = perfect_response(record, roles)
    rows, failures, _, _ = run_one(record, response=f"Sure!\n{body}\nDone.",
                                   n_blocks=4)
    assert not failures, failures
    assert rows[0]["behaviour_wat_score"] == 1.0


# =========================================================
# N — layer indexing and the secondary outputs
# =========================================================

def test_layer_zero_is_the_input_embedding_not_transformer_layer_zero():
    lab = lh.layer_label(0)
    assert lab["representation_type"] == "input_embedding"
    assert lab["transformer_layer"] == ""
    cols = lh.primary_columns(33)
    assert "weat_layer0" in cols and "delta_layer0" in cols


def test_transformer_outputs_are_numbered_one_to_n():
    labs = [lh.layer_label(i) for i in range(33)]
    blocks = lh.transformer_layers_only(labs)
    assert len(blocks) == 32
    assert [b["transformer_layer"] for b in blocks] == list(range(1, 33))
    assert all(b["hidden_state_index"] == b["transformer_layer"] for b in blocks)


def test_the_score_columns_cover_every_hidden_state():
    vectors, meta, roles = extract_one(DESIGN[0], n_blocks=4)
    deltas, weats = lh.representation_scores(vectors, roles,
                                             meta["n_hidden_states"])
    assert len(deltas) == len(weats) == meta["n_hidden_states"] == 5
    assert meta["n_transformer_blocks"] == 4


def _seeded_acc(records, n_blocks=3, dim=6):
    acc = lh.PrototypeAccumulator()
    tok = FakeTokenizer()
    for record in records:
        roles = lh.prompt_roles(record, STIM)
        model = FakeModel(n_blocks=n_blocks, dim=dim, tokenizer=tok,
                          response=perfect_response(record, roles))
        lh.run_prompts([record], STIM, model, tok, args=FakeArgs(), acc=acc,
                       verbose=False)
    return acc


def test_the_secondary_set_level_weat_uses_the_population_sd():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4, 6)); Y = rng.normal(size=(4, 6))
    A = rng.normal(size=(4, 6)); B = rng.normal(size=(4, 6))
    pop = lh.weat_effect_size_hidden(X, Y, A, B, ddof=0)
    sample = lh.weat_effect_size_hidden(X, Y, A, B, ddof=1)
    sx = eb.weat_association(X, A, B); sy = eb.weat_association(Y, A, B)
    expected = ((np.mean(sx) - np.mean(sy))
                / np.std(np.concatenate((sx, sy)), ddof=0))
    assert abs(pop - expected) < 1e-12
    assert abs(pop - sample) > 1e-9


def test_the_external_embedding_effect_size_is_left_on_the_sample_sd():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(4, 6)); Y = rng.normal(size=(4, 6))
    A = rng.normal(size=(4, 6)); B = rng.normal(size=(4, 6))
    sx = eb.weat_association(X, A, B); sy = eb.weat_association(Y, A, B)
    expected = ((np.mean(sx) - np.mean(sy))
                / np.std(np.concatenate((sx, sy)), ddof=1))
    assert abs(eb.weat_effect_size(X, Y, A, B) - expected) < 1e-12


def test_the_secondary_set_level_sign_convention_is_stereotype_congruent():
    acc = _seeded_acc(by_category(DESIGN, "creativity")[:4])
    arm = lh.arm_for("gender", "creativity", STIM)
    rows, missing = lh.set_level_rows(acc, "gender", "creativity", arm,
                                      n_hidden_states=4)
    assert rows
    assert rows[0]["X_name"] == "male targets"
    assert rows[0]["A_name"].startswith("brilliance")
    assert rows[0]["analysis_role"] == "secondary"
    assert rows[0]["effect_size_sd"] == "population (ddof=0)"


def test_the_secondary_outputs_only_see_this_experiments_prompts():
    acc = _seeded_acc(by_category(DESIGN, "creativity")[:4])
    contexts = {c for ids in acc.contexts.values() for c in ids}
    assert all(c.startswith("gb_creativity_") for c in contexts), contexts
    assert acc.n("gender", "creativity", "brainiac") == 4
    src = _code_only(inspect.getsource(lh.set_level_rows))
    assert "load_sessions" not in src


def test_accumulating_a_prompt_stores_one_block_per_stimulus():
    vectors, meta, roles = extract_one(DESIGN[0], n_blocks=3, dim=6)
    acc = lh.PrototypeAccumulator()
    lh.accumulate_prompt(acc, "gender", "creativity", vectors,
                         roles["phrases"], "gb_creativity_1")
    for w in roles["phrases"]:
        proto = acc.prototype("gender", "creativity", w)
        assert proto.shape == (meta["n_hidden_states"], 6), (w, proto.shape)
        assert acc.n("gender", "creativity", w) == 1


def test_handing_the_accumulator_the_whole_mapping_is_a_named_error():
    vectors, _, roles = extract_one(DESIGN[0], n_blocks=3, dim=6)
    acc = lh.PrototypeAccumulator()
    try:
        acc.add("gender", "creativity", roles["X"], vectors)
    except TypeError as exc:
        assert "whole" in str(exc) and roles["X"] in str(exc)
        return
    raise AssertionError("the accumulator swallowed the whole mapping")


def test_the_prototype_is_the_mean_over_contexts():
    acc = lh.PrototypeAccumulator()
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[3.0, 4.0], [5.0, 6.0]])
    acc.add("gender", "humor", "witty", a, context_id="p1")
    acc.add("gender", "humor", "witty", b, context_id="p2")
    assert np.allclose(acc.prototype("gender", "humor", "witty"), (a + b) / 2)
    assert acc.n("gender", "humor", "witty") == 2


# =========================================================
# O — what the primary runner must NOT do
# =========================================================

def test_no_pca_nullification_is_applied():
    for banned in ("nullify", "top_components", "max_valid_k", "PCA_K", "pca_k"):
        assert banned not in CODE, banned
    assert not hasattr(lh, "PCA_K")


def test_no_seat_or_neutral_template_path_exists():
    for banned in ("SEAT_TEMPLATES", "def seat", "build_seat", "template.format("):
        assert banned not in CODE, banned


def test_no_profile_or_decision_chain_is_implemented():
    for banned in ("bai_core", "fab_decision", "profile_prompt", "decision_prompt"):
        assert banned not in CODE, banned


def test_the_loader_forbids_quantisation_and_offload():
    src = _code_only(inspect.getsource(lh.load_llama3))
    assert 'device_map={"": gpu}' in src
    assert "max_memory=" not in src
    assert "device_map=\"auto\"" not in src
    for banned in ("load_in_4bit", "load_in_8bit", "BitsAndBytesConfig",
                   "offload_folder", "offload_state_dict"):
        assert banned not in CODE, banned
    assert "model.eval()" in src
    assert 'stray' in src and 'cpu' in src


def test_the_checkpoint_default_is_unchanged():
    assert lh.CHECKPOINT == "/home/syryu/models/llama3.1-8b-instruct"


def test_embed_bias_and_the_scout_module_are_not_modified_by_this_runner():
    assert "def weat_effect_size" in open(eb.__file__, encoding="utf-8").read()
    import numpy as _np
    rng = _np.random.default_rng(1)
    X = rng.normal(size=(3, 4)); Y = rng.normal(size=(3, 4))
    A = rng.normal(size=(3, 4)); B = rng.normal(size=(3, 4))
    before = eb.weat_effect_size(X, Y, A, B)
    lh.weat_effect_size_hidden(X, Y, A, B)
    assert eb.weat_effect_size(X, Y, A, B) == before
    assert sc.OUT_DIR.endswith("embed_scout"), "the Scout module was repurposed"


def test_the_output_directory_is_free_of_the_legacy_run():
    """
    The fixed-prompt experiment writes to `responses/embed_llama31/`, the name
    the 3420-prompt replay pilot used to hold.  That is safe only because the
    pilot was moved to `responses/archive/embed_llama31/`; if its files are back
    under the live name, a run would overwrite them.
    """
    assert os.path.basename(lh.OUT_DIR) == "embed_llama31"
    assert lh.PRIMARY_CSV == "prompt_response_hidden.csv"
    legacy = [f for f in ("hidden_prompt_layer.csv", "hidden_manifest.json",
                          "hidden_pair_layer_robustness.csv")
              if os.path.exists(os.path.join(lh.OUT_DIR, f))]
    assert not legacy, (f"legacy replay output still in {lh.OUT_DIR}: {legacy} "
                        f"— move it to responses/archive/embed_llama31/")
    assert set(lh.OUTPUT_FILES).isdisjoint(
        {"hidden_prompt_layer.csv", "hidden_set_layer.csv",
         "hidden_pair_layer_robustness.csv"})


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and pattern in n]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {exc.__class__.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        import traceback
        for name, exc in failed:
            print(f"\n--- {name}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        sys.exit(1)


if __name__ == "__main__":
    main()
