"""
CORE — model-independent scoring, statistics and span logic shared by every
embedding experiment and every model backend.

Nothing here loads a model, knows a model's architecture, or places a tensor on a
device; nothing here imports a backend.  Every function was moved verbatim from
the module named in its section header:

    embed_bias.py                WEAT / WEFAT, statistics, csv writer
    embed_bias_scout_hidden.py   span layer, chat-input reconstruction, BOS record
    embed_bias_llama3_hidden.py  generation contract, roles, behavioural parser
                                 and score, delta / singleton WEAT, prototypes

The only edits are module-qualifier removals (`eb.cos_sim` -> `cos_sim`) and one
signature change: `set_level_rows` / `stimulus_layer_rows` take the caller's
`analysis_version` instead of reading a runner's global.

torch and numpy are imported inside functions, so importing this module loads
neither transformers nor any weights.
"""

import os
import re
import csv
import json
import math
import hashlib
import itertools
from collections import defaultdict

import wat_fab
import wat_gender_brilliance as wgb
from embedding.design import GENDER_CONDITIONS
from embedding.design import file_digest     # re-exported for the backends


# =========================================================
# WEAT / WEFAT  (Caliskan et al. 2017)  — from embed_bias.py
# =========================================================

PERM_CAP = 200_000                # exact permutation test below this many splits



def _unit(v):
    import numpy as np
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cos_sim(v1, v2):
    import numpy as np
    return np.matmul(_unit(np.asarray(v1, dtype=float)),
                     _unit(np.asarray(v2, dtype=float)).T)


def weat_association(W, A, B):
    """s(w, A, B) = mean cos(w, A) - mean cos(w, B), for every w in W."""
    import numpy as np
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)


def weat_test_stat(X, Y, A, B):
    """S(X,Y,A,B) = sum_x s(x,A,B) - sum_y s(y,A,B)  (unstandardised)."""
    import numpy as np
    return float(np.sum(weat_association(X, A, B)) - np.sum(weat_association(Y, A, B)))


def weat_effect_size(X, Y, A, B):
    """
    Caliskan's d = [mean s(X) - mean s(Y)] / SD(s(X u Y)).
    Degenerate when len(X) == len(Y) == 1 (SD of two values) — callers flag it.
    """
    import numpy as np
    sx = weat_association(X, A, B)
    sy = weat_association(Y, A, B)
    sd = np.std(np.concatenate((sx, sy), axis=0), ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float((np.mean(sx) - np.mean(sy)) / sd)


def bai_singleton_weat(X, Y, A, B):
    """
    Bai's standardized score as literally implemented (population SD + eps=0.001).
    Supplemental only: with one X and one Y it collapses toward +-2 regardless of
    the magnitude of the underlying association.
    """
    import numpy as np
    sx = weat_association(X, A, B)
    sy = weat_association(Y, A, B)
    sd = np.std(np.concatenate((sx, sy), axis=0))
    return float((np.mean(sx) - np.mean(sy)) / (sd + 0.001))


def weat_p_value(X, Y, A, B, cap=PERM_CAP, seed=0):
    """
    Two-sided exact permutation p over equal-size splits of X u Y.
    Seeded random subsample past `cap` splits.  Returns (p, n_perms, exact).
    nan when len(X) != len(Y) (no equal-size partition exists).
    """
    import numpy as np
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if len(X) != len(Y):
        return float("nan"), 0, False
    pool = np.concatenate((X, Y), axis=0)
    n, k = len(pool), len(X)
    observed = weat_test_stat(X, Y, A, B)
    total = math.comb(n, k)
    if total <= cap:
        splits = itertools.combinations(range(n), k)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        splits = (tuple(rng.choice(n, k, replace=False)) for _ in range(cap))
        exact = False
    count = tot = 0
    for idx in splits:
        mask = np.zeros(n, dtype=bool); mask[list(idx)] = True
        if abs(weat_test_stat(pool[mask], pool[~mask], A, B)) >= abs(observed) - 1e-12:
            count += 1
        tot += 1
    return count / tot, tot, exact


def wefat_score(w, A, B):
    """
    WEFAT: where ONE item sits on the A<->B dimension, normalised by the SD of
    its own cosines.  Positive = closer to A.  `w` is a single vector.
    """
    import numpy as np
    w = np.asarray(w, dtype=float)[None, :]
    num = weat_association(w, A, B)[0]
    sd = np.std(np.concatenate((cos_sim(w, A).ravel(), cos_sim(w, B).ravel())), ddof=1)
    return float(num / sd) if sd else float("nan")



def weat_effect_size_hidden(X, Y, A, B, ddof=0):
    """
    Caliskan's d with the CANONICAL POPULATION SD (ddof=0), for the secondary
    set-level file.

    Deliberately NOT `eb.weat_effect_size`, which uses the sample SD (ddof=1).
    Changing that function would silently move every historical external-
    embedding number in `responses/embed/`; this module keeps its own.
    """
    import numpy as np
    sx = weat_association(X, A, B)
    sy = weat_association(Y, A, B)
    sd = np.std(np.concatenate((sx, sy), axis=0), ddof=ddof)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float((np.mean(sx) - np.mean(sy)) / sd)



# =========================================================
# Statistics — Spearman (primary) and Pearson (secondary), scipy-free fallback  — from embed_bias.py
# =========================================================

try:                                     # scipy is NOT installed in this env;
    from scipy import stats as _scipy_stats   # the fallback below is the real path
except Exception:
    _scipy_stats = None


def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf2(t, df):
    """Two-sided Student-t p-value."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _clean_pairs(xs, ys):
    import numpy as np
    x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def pearson(xs, ys):
    """(r, p, n).  p from the t approximation on n-2 df."""
    x, y = _clean_pairs(xs, ys)
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), n
    import numpy as np
    r = max(-1.0, min(1.0, float(np.corrcoef(x, y)[0, 1])))
    if abs(r) >= 1.0:
        return r, 0.0, n
    return r, t_sf2(r * math.sqrt((n - 2) / (1 - r * r)), n - 2), n


def rankdata(values):
    """Ranks with ties averaged (scipy.stats.rankdata 'average')."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs, ys):
    """
    (rho, p, n).  Pearson on tie-averaged ranks; p from the t approximation,
    which is what scipy uses by default for n > 3 as well.  PRIMARY statistic.
    """
    x, y = _clean_pairs(xs, ys)
    if len(x) < 3:
        return float("nan"), float("nan"), len(x)
    if _scipy_stats is not None:
        res = _scipy_stats.spearmanr(x, y)
        return float(res.statistic), float(res.pvalue), len(x)
    return pearson(rankdata(x), rankdata(y))


def holm(pvalues):
    """
    Holm-Bonferroni step-down adjusted p-values, in the INPUT order.

    Finite p-values are corrected among themselves; missing / non-finite entries
    stay missing and do not count towards the family size.  The result is
    monotone in the sorted p-values and capped at 1.  Raw p-values are never
    replaced -- the caller stores both.
    """
    def _finite(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    vals = [_finite(v) for v in pvalues]
    idx = [i for i, v in enumerate(vals) if v is not None]
    out = [float("nan")] * len(vals)
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: vals[j])):
        running = max(running, (m - rank) * vals[i])
        out[i] = min(1.0, running)
    return out


def apply_holm(rows, key_of, label_of, p_key, out_key):
    """
    Holm-correct `p_key` within each family and write `out_key` + `holm_family`
    + `holm_n` onto every row.  Rows for which `key_of` returns None are left
    uncorrected (descriptive rows) with blank family fields.

    The grouping key may be finer than the printed family label: families are
    always formed within one embedding model and one contextualisation, which
    the label omits because those are separate columns.
    """
    fams = defaultdict(list)
    for r in rows:
        key = key_of(r)
        if key is None:
            r.setdefault(out_key, "")
            r.setdefault("holm_family", "")
            r.setdefault("holm_n", "")
            continue
        fams[key].append(r)
    for key, rs in fams.items():
        adj = holm([r.get(p_key) for r in rs])
        n_tested = sum(1 for a in adj if a == a)
        for r, a in zip(rs, adj):
            r[out_key] = a
            r["holm_family"] = label_of(rs[0])
            r["holm_n"] = n_tested



# =========================================================
# Span layer — phrase -> token span, pooling, chat input  — from embed_bias_scout_hidden.py
# =========================================================

class SpanError(RuntimeError):
    """Raised when a stimulus phrase cannot be located unambiguously."""


def find_phrase_char_spans(prompt, phrase, window=None):
    """
    All character spans of `phrase` in `prompt`, case-insensitively.
    `window` = (start, end) restricts the search to one section of the prompt
    (e.g. only the sorted-word list), which is how an intended occurrence is
    disambiguated from an incidental one.
    """
    lo, hi = (0, len(prompt)) if window is None else window
    hay, needle = prompt.lower(), phrase.lower()
    spans, i = [], lo
    while True:
        j = hay.find(needle, i, hi)
        if j < 0:
            return spans
        spans.append((j, j + len(phrase)))
        i = j + 1


def token_span_for_chars(offsets, char_span):
    """
    Token indices whose character offsets overlap `char_span`.
    `offsets` is a HF fast-tokenizer `offset_mapping`: one (start, end) per token.
    Zero-width tokens (specials) are skipped.
    """
    start, end = char_span
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not idx:
        raise SpanError(f"no tokens overlap character span {char_span}")
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise SpanError(f"token span for {char_span} is not contiguous: {idx}")
    return idx[0], idx[-1] + 1          # half-open


def masking_spans(prompt, phrase, competitors, window=None):
    """
    Spans where `phrase` appears only because a LONGER stimulus contains it.

    The FAB prompt lists all 30 Leslie fields, among them both 'art history' and
    'history'.  A substring search for the stimulus 'history' therefore also
    matches inside 'art history', which is a different field and a different
    item in the sorting task.  Any stimulus in the same prompt that strictly
    contains the phrase masks the occurrences that fall inside it.
    """
    blocked = []
    needle = phrase.lower()
    for other in competitors:
        other = str(other)
        if len(other) > len(phrase) and needle in other.lower():
            blocked.extend(find_phrase_char_spans(prompt, other, window))
    return blocked


def resolve_span(prompt, offsets, phrase, window=None, occurrence="single",
                 competitors=()):
    """
    Locate ONE intended occurrence of `phrase` and return its half-open token span.

    occurrence:
      'single' (default) — exactly one match must exist, else SpanError.
      'first' / 'last'   — accept several and take that end deliberately.
    `competitors` are the other stimuli the same prompt presents; an occurrence
    inside a longer one of them is not a mention of `phrase` (see
    `masking_spans`) and is discarded before the count is checked.

    Fails loudly rather than guessing: Bai's first-token `.index()` silently
    returns the wrong position for a multi-token phrase, or for a phrase that
    also appears in the instruction sentence.
    """
    spans = find_phrase_char_spans(prompt, phrase, window)
    blocked = masking_spans(prompt, phrase, competitors, window)
    if blocked:
        spans = [sp for sp in spans
                 if not any(b0 <= sp[0] and sp[1] <= b1 for b0, b1 in blocked)]
    if not spans:
        raise SpanError(f"{phrase!r} does not occur in the prompt"
                        + (f" within {window}" if window else ""))
    if len(spans) > 1 and occurrence == "single":
        raise SpanError(f"{phrase!r} occurs {len(spans)} times "
                        f"{[s for s, _ in spans]} — pass a window or an explicit "
                        f"occurrence rule; refusing to guess")
    chosen = spans[0] if occurrence in ("single", "first") else spans[-1]
    return token_span_for_chars(offsets, chosen)


def build_chat_input(tokenizer, system, user_prompt):
    """
    Apply the model's chat template to the same two messages the runner sent, so
    the forward pass sees the Llama-4 header tokens Scout actually read — not the
    bare user string, which is what the jsonl stores.

    Returns (templated_text, user_window) where `user_window` is the character
    span of the user message inside the templated text.  Span lookups are
    restricted to that window so template boilerplate and the system message can
    never supply a match.  Raises if the template does not reproduce the user
    message verbatim, since every char offset downstream depends on it.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    if not isinstance(text, str):
        raise SpanError("apply_chat_template did not return a string; pass "
                        "tokenize=False is required for offset-based spans")
    start = text.find(user_prompt)
    if start < 0:
        raise SpanError(
            "the chat template did not reproduce the saved user prompt "
            "verbatim, so character offsets cannot be trusted. Saved prompt "
            f"starts {user_prompt[:60]!r}; templated text starts {text[:120]!r}")
    if text.find(user_prompt, start + 1) >= 0:
        raise SpanError("the saved user prompt occurs twice in the templated "
                        "text; refusing to guess which copy is the message")
    return text, (start, start + len(user_prompt))


def pool_span(layer_matrix, span, rule="mean"):
    """
    Pool a phrase's token vectors.  `layer_matrix` is (n_tokens, dim) for ONE layer.
    'mean' (default) averages the span; 'last' takes its final token.
    Never silently uses the first token of a multi-token phrase.
    """
    import numpy as np
    lo, hi = span
    block = np.asarray(layer_matrix, dtype=float)[lo:hi]
    if block.size == 0:
        raise SpanError(f"empty span {span}")
    if rule == "mean":
        return block.mean(axis=0)
    if rule == "last":
        return block[-1]
    raise ValueError(f"unknown pooling rule {rule!r} (expected 'mean' or 'last')")



def leading_bos_count(tokenizer, templated):
    """BOS ids opening the tokenised input (template BOS + tokenizer BOS)."""
    ids = tokenizer(templated)["input_ids"]
    bos = tokenizer.bos_token_id
    n = 0
    while n < len(ids) and ids[n] == bos:
        n += 1
    return n


def bos_record(tokenizer, templated):
    """What the manifest says about BOS, measured rather than asserted."""
    bos = tokenizer.bos_token
    in_template = 0
    while templated.startswith(bos * (in_template + 1)):
        in_template += 1
    return {
        "bos_token": bos,
        "bos_token_id": tokenizer.bos_token_id,
        "bos_in_templated_text": in_template,
        "bos_in_model_input": leading_bos_count(tokenizer, templated),
        "how": "apply_chat_template(..., tokenize=False) emits one BOS; "
               "tokenizer(templated) prepends another special BOS",
        "decision": "kept deliberately: the completed Llama 3.1 run fed the "
                    "model the same double BOS, so preprocessing is comparable "
                    "across the two models",
    }


def validate_all_spans(tokenizer, prompts, stim):
    """
    Tokenizer-only: every stimulus of every prompt resolves to exactly one span
    under THIS tokenizer.  Returns (min tokens, max tokens, {leading BOS}).
    """
    n_tok, n_bos = [], set()
    for record in prompts:
        roles = prompt_roles(record, stim)
        templated, window = build_chat_input(tokenizer, SYSTEM_PROMPT,
                                             record["prompt"])
        enc = tokenizer(templated, return_offsets_mapping=True)
        for phrase in roles["phrases"]:
            resolve_span(templated, enc["offset_mapping"], phrase, window=window,
                         competitors=[c for c in roles["phrases"] if c != phrase])
        n_tok.append(len(enc["input_ids"]))
        n_bos.add(leading_bos_count(tokenizer, templated))
    return min(n_tok), max(n_tok), sorted(n_bos)



# =========================================================
# Generation contract — from embed_bias_llama3_hidden.py
# =========================================================

SYSTEM_PROMPT = "You are a helpful assistant."


POOLING = "mean"


# The chosen low-nonzero sampling setting.  Recorded verbatim in the manifest.
GENERATION = {
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 512,
}

SEED_BASE = 0

SEED_RULE = ("per_prompt_seed = int(sha256(f'{seed_base}|{prompt_id}')"
             ".hexdigest()[:8], 16); torch.manual_seed(per_prompt_seed) "
             "immediately before generate(). Derived from the prompt id, not "
             "from execution order, so a resumed or reordered run reproduces "
             "the same responses.")



def prompt_seed(prompt_id, seed_base=SEED_BASE):
    """Deterministic per-row generation seed; see SEED_RULE."""
    digest = hashlib.sha256(f"{seed_base}|{prompt_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)



# =========================================================
# Failure types
# =========================================================

class NonFiniteHiddenState(RuntimeError):
    """A forward pass returned NaN/Inf.  No statistic may be derived from it."""



class ArchitectureMismatch(RuntimeError):
    """A forward pass returned a hidden-state stack the config does not describe."""



# =========================================================
# Layer indexing — from embed_bias_llama3_hidden.py
# =========================================================

def layer_label(hidden_state_index):
    """
    (hidden_state_index, transformer_layer, representation_type) for the long
    secondary files.

    Index 0 is the input/token embedding, NOT "transformer layer 0": it has not
    been through a block, and numbering it as one would invite averaging a
    lookup table into a sequence of residual-stream states.  The wide primary
    table calls it `layer0` and the manifest says what that means.
    """
    i = int(hidden_state_index)
    if i < 0:
        raise ValueError(f"hidden_state_index must be >= 0, got {i}")
    if i == 0:
        return dict(hidden_state_index=0, transformer_layer="",
                    representation_type="input_embedding")
    return dict(hidden_state_index=i, transformer_layer=i,
                representation_type="transformer_block")


def transformer_layers_only(rows):
    """The exploratory trajectory: block outputs, with the embedding dropped."""
    return [r for r in rows if r["representation_type"] == "transformer_block"]


def layer_columns(n_hidden_states):
    """The wide table's score columns, in their fixed order."""
    return ([f"weat_layer{i}" for i in range(n_hidden_states)]
            + [f"delta_layer{i}" for i in range(n_hidden_states)])


def primary_columns(n_hidden_states):
    return (["prompt_id", "domain", "category", "trial", "prompt", "response",
             "behaviour_wat_score"] + layer_columns(n_hidden_states))



# =========================================================
# Roles — resolved from stimuli.csv, never from group0/group1  — from embed_bias_llama3_hidden.py
# =========================================================

def arm_for(domain, category, stim):
    """
    The full concept sets for one condition.  X/Y/A/B here define the sign, and
    they are read out of stimuli.csv by ROLE, so presentation order cannot
    reach them.
    """
    if domain == "gender":
        if category not in GENDER_CONDITIONS:
            raise SystemExit(f"unknown gender condition {category!r} "
                             f"(have {list(GENDER_CONDITIONS)})")
        g = stim["gender"]
        return dict(X=list(g["male"]), Y=list(g["female"]),
                    A=list(g["brilliance"]), B=list(g["control"][category]),
                    X_name="male targets", Y_name="female targets",
                    A_name="brilliance (storage4)",
                    B_name=f"control ({category})")
    if domain == "fab":
        f = stim["fab"]
        return dict(X=list(f["brilliance"]), Y=list(f["effort"]),
                    A=list(f["high_fab"]), B=list(f["low_fab"]),
                    X_name="brilliance poles (leslie)",
                    Y_name="effort poles (leslie)",
                    A_name="high-FAB fields (tertile)",
                    B_name="low-FAB fields (tertile)")
    raise SystemExit(f"unknown domain {domain!r}")


def prompt_roles(record, stim):
    """
    This prompt's X, Y, A, B.

    `group0`/`group1` say only which target was printed first.  Which one is X
    is decided by looking each up in the stimuli.csv role lists; a prompt whose
    two targets fall in the same pool, or in neither, is a corrupt design row
    and stops the run.
    """
    arm = arm_for(record["domain"], record["category"], stim)
    g0, g1 = record["group0"], record["group1"]
    in_X = [g for g in (g0, g1) if g in arm["X"]]
    in_Y = [g for g in (g0, g1) if g in arm["Y"]]
    if len(in_X) != 1 or len(in_Y) != 1:
        raise SystemExit(
            f"{record['prompt_id']}: targets ({g0!r}, {g1!r}) do not resolve to "
            f"exactly one {arm['X_name']} and one {arm['Y_name']}")

    attributes = json.loads(record["attributes"])
    A = [w for w in arm["A"] if w in attributes]
    B = [w for w in arm["B"] if w in attributes]
    if len(A) != len(arm["A"]) or len(B) != len(arm["B"]):
        raise SystemExit(
            f"{record['prompt_id']}: the prompt presents {len(A)}/{len(arm['A'])} "
            f"{arm['A_name']} and {len(B)}/{len(arm['B'])} {arm['B_name']}; "
            f"refusing to score a reduced set")
    if len(attributes) != len(A) + len(B):
        extra = [w for w in attributes if w not in A and w not in B]
        raise SystemExit(f"{record['prompt_id']}: unrecognised presented "
                         f"item(s) {extra}")

    return dict(X=in_X[0], Y=in_Y[0], A=A, B=B,
                attributes=list(attributes), arm=arm,
                phrases=[in_X[0], in_Y[0]] + list(attributes))



# =========================================================
# Behavioural WAT score  (NOT called WEAT)  — from embed_bias_llama3_hidden.py
# =========================================================

def _norm(x):
    """Same normalisation the behavioural parsers apply to both sides."""
    x = str(x).strip().strip("'\"").strip("*").strip().lower()
    return re.sub(r"\s+", " ", x)


def parse_behaviour(raw, record, roles):
    """
    ({item: target}, problems).  Strict: exact matches only, every presented
    item exactly once, and only the two targets THIS prompt presented.

    The line parsers are the behavioural runners' own — `wgb.parse_wat` splits
    on the LAST hyphen so `super-imaginative` survives, `wat_fab.parse_wat`
    splits on the first, which is right for field names that contain none.
    Matching after that is deliberately stricter than the runners': their
    substring fallbacks exist to rescue n from a paid API call, and here they
    would let a response that named a non-presented target still score.
    """
    parser = wgb.parse_wat if record["domain"] == "gender" else wat_fab.parse_wat
    pairs = parser(raw or "")

    items = {_norm(w): w for w in roles["attributes"]}
    targets = {_norm(t): t for t in (roles["X"], roles["Y"])}
    assign, problems = {}, []
    for left, right in pairs:
        item = items.get(_norm(left))
        if item is None:
            problems.append(f"unrecognised item {left!r}")
            continue
        if item in assign:
            problems.append(f"item {item!r} assigned more than once")
            continue
        target = targets.get(_norm(right))
        if target is None:
            problems.append(f"{item!r} assigned to {right!r}, which is not one "
                            f"of the two presented targets")
            continue
        assign[item] = target

    unassigned = [w for w in roles["attributes"] if w not in assign]
    if unassigned:
        problems.append(f"{len(unassigned)} item(s) never assigned: "
                        f"{', '.join(unassigned)}")
    return assign, problems


def behaviour_wat_score(assign, record, roles):
    """
    Bai's assignment score, EXACT — no +0.01 smoothing.  Positive is always
    stereotype-congruent, because the counts are keyed on ROLE (X/Y, A/B), not
    on which target was printed first.

        gender : P(brilliance | male-assigned) + P(control | female-assigned) - 1
        fab    : P(brilliance | high-FAB)      + P(effort  | low-FAB)         - 1

    Returns (score, problem).  An empty denominator — every item sent to one
    side — yields (nan, reason) rather than a smoothed number.
    """
    X, Y = roles["X"], roles["Y"]
    A = set(roles["A"])
    if record["domain"] == "gender":
        # left of the + : items given to the male target
        aX = sum(1 for i, t in assign.items() if t == X and i in A)
        bX = sum(1 for i, t in assign.items() if t == X and i not in A)
        aY = sum(1 for i, t in assign.items() if t == Y and i in A)
        bY = sum(1 for i, t in assign.items() if t == Y and i not in A)
        if (aX + bX) == 0:
            return float("nan"), f"no item assigned to {X!r}"
        if (aY + bY) == 0:
            return float("nan"), f"no item assigned to {Y!r}"
        return aX / (aX + bX) + bY / (aY + bY) - 1, None

    B = set(roles["B"])
    # left of the + : high-FAB fields; right : low-FAB fields
    hX = sum(1 for i, t in assign.items() if i in A and t == X)
    hY = sum(1 for i, t in assign.items() if i in A and t == Y)
    lX = sum(1 for i, t in assign.items() if i in B and t == X)
    lY = sum(1 for i, t in assign.items() if i in B and t == Y)
    if (hX + hY) == 0:
        return float("nan"), "no high-FAB field assigned to either pole"
    if (lX + lY) == 0:
        return float("nan"), "no low-FAB field assigned to either pole"
    return hX / (hX + hY) + lY / (lX + lY) - 1, None



# =========================================================
# Hidden-state checks and representation scores  — from embed_bias_llama3_hidden.py
# =========================================================

def assert_finite_hidden_states(hidden_states, context=""):
    """
    `torch.isfinite(h).all()` on EVERY returned hidden state, before anything is
    pooled.  Raises with the first offending index and its exact NaN/Inf counts.

    Scout went non-finite at hidden state 7 and the residual stream collapsed to
    absmax 0 from there; with no check, that run would have written a CSV full of
    numbers that looked like results.
    """
    import torch
    for i, h in enumerate(hidden_states):
        if torch.isfinite(h).all():
            continue
        n_nan = int(torch.isnan(h).sum().item())
        n_inf = int(torch.isinf(h).sum().item())
        raise NonFiniteHiddenState(
            f"non-finite hidden state at index {i} "
            f"({layer_label(i)['representation_type']}): "
            f"{n_nan} NaN, {n_inf} Inf of {h.numel()} values"
            + (f" [{context}]" if context else "")
            + " — aborting the run; no score may be derived from it")


def representation_scores(vectors, roles, n_hidden_states):
    """
    (deltas, weats), one value per hidden state.

        s(w,A,B) = mean cos(w,A) - mean cos(w,B)
        delta    = s(X,A,B) - s(Y,A,B)
        weat     = [mean s(X) - mean s(Y)] / [pop_SD(s(X) u s(Y)) + 0.001]

    With one X and one Y, `bai_singleton_weat` IS that standardized formula
    (population SD, eps = 0.001) and is already tested, so it is reused rather
    than re-derived.  The output column is `weat_layer*`; the Bai-specific name
    stays inside this module.
    """
    import numpy as np
    A_words, B_words = list(roles["A"]), list(roles["B"])
    deltas, weats = [], []
    for i in range(n_hidden_states):
        xv = np.asarray(vectors[roles["X"]][i], dtype=float)[None, :]
        yv = np.asarray(vectors[roles["Y"]][i], dtype=float)[None, :]
        A = np.asarray([vectors[w][i] for w in A_words], dtype=float)
        B = np.asarray([vectors[w][i] for w in B_words], dtype=float)
        s_X = float(weat_association(xv, A, B)[0])
        s_Y = float(weat_association(yv, A, B)[0])
        deltas.append(s_X - s_Y)
        weats.append(float(bai_singleton_weat(xv, yv, A, B)))
    return deltas, weats


def wide_row(record, response, score, deltas, weats):
    """
    One row of the primary table, in the fixed column order.

    `group0`, `group1` and `attributes` are NOT repeated here — they live in
    prompts.csv and join on `prompt_id`.  Keeping them out is what stops a
    presentation-order column from quietly becoming an analysis variable.
    """
    if len(deltas) != len(weats):
        raise ValueError(f"{len(deltas)} deltas vs {len(weats)} weats")
    row = {
        "prompt_id": record["prompt_id"],
        "domain": record["domain"],
        "category": record["category"],
        "trial": int(record["trial"]),
        "prompt": record["prompt"],
        "response": response,
        "behaviour_wat_score": score,
    }
    for i, v in enumerate(weats):
        row[f"weat_layer{i}"] = v
    for i, v in enumerate(deltas):
        row[f"delta_layer{i}"] = v
    return row


def write_primary_csv(path, rows, n_hidden_states, force=False):
    """Explicit fieldnames: the column order is part of the specification."""
    columns = primary_columns(n_hidden_states)
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite {path} — rerun with --force "
                         f"(nothing written this run)")
    for r in rows:
        extra = set(r) - set(columns)
        if extra:
            raise ValueError(f"{r.get('prompt_id')}: unexpected column(s) "
                             f"{sorted(extra)} in the primary table")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows x {len(columns)} columns)")
    return columns



# =========================================================
# Contextual prototypes and set-level WEAT  — from embed_bias_llama3_hidden.py
# =========================================================

class PrototypeAccumulator:
    """
    Running mean of each stimulus's contextual vector, per (domain, category,
    stimulus, hidden state).

    A running sum rather than a stored list: 272 prompts x 33 hidden states x
    4096 dims is small, but the accumulator predates this design and there is no
    reason to make it worse.
    """

    def __init__(self):
        self.sums = {}
        self.counts = defaultdict(int)
        self.contexts = defaultdict(list)

    def add(self, dataset, arm, stimulus, vectors, context_id=None):
        """
        `vectors` is ONE stimulus's per-hidden-state vectors — the value of
        `extract_prompt_vectors()[0][stimulus]`, not the whole mapping.  The
        shape is checked because passing the mapping instead is an easy slip and
        numpy's own error ("float() argument must be ... not 'dict'") says
        nothing about which call site was wrong.
        """
        import numpy as np
        key = (dataset, arm, stimulus)
        if isinstance(vectors, dict):
            raise TypeError(
                f"PrototypeAccumulator.add({stimulus!r}) was given the whole "
                f"{{phrase: vectors}} mapping; pass vectors[{stimulus!r}]")
        stack = np.asarray(vectors, dtype=np.float64)     # (n_hidden_states, dim)
        if stack.ndim != 2:
            raise ValueError(
                f"expected one (n_hidden_states, dim) block for {stimulus!r}, "
                f"got shape {stack.shape}")
        if key in self.sums:
            if self.sums[key].shape != stack.shape:
                raise ValueError(f"shape change for {key}: "
                                 f"{self.sums[key].shape} vs {stack.shape}")
            self.sums[key] += stack
        else:
            self.sums[key] = stack.copy()
        self.counts[key] += 1
        if context_id is not None:
            self.contexts[key].append(context_id)

    def prototype(self, dataset, arm, stimulus):
        key = (dataset, arm, stimulus)
        if key not in self.sums:
            raise KeyError(f"no contextual observation of {stimulus!r} in "
                           f"{dataset}/{arm}")
        return self.sums[key] / self.counts[key]

    def n(self, dataset, arm, stimulus):
        return self.counts[(dataset, arm, stimulus)]

    def stimuli(self, dataset, arm):
        return sorted(s for (d, a, s) in self.sums if d == dataset and a == arm)


def accumulate_prompt(acc, dataset, arm_name, vectors, phrases, prompt_id):
    """
    Feed one prompt's contextual vectors into the prototype accumulator.

    A named seam rather than a loop inside `main`, so the wiring — one stimulus's
    vectors per call, not the mapping — is covered by a test instead of by the
    first prompt of a full run.
    """
    for w in phrases:
        acc.add(dataset, arm_name, w, vectors[w], context_id=prompt_id)
    return acc



def set_level_rows(acc, dataset, arm_name, arm, *, n_hidden_states, analysis_version):
    """
    SECONDARY.  The full-set WEAT over contextual prototypes, at every hidden
    state, within this 272-prompt experiment only.  `analysis_version` is the
    calling runner's, stamped on every row.

    X/Y/A/B and therefore the sign come from `arm_for()`, i.e. from stimuli.csv,
    so stereotype-congruent stays positive.  Effect size uses the population SD
    (ddof=0) — `weat_effect_size_hidden`, not `weat_effect_size` — so the
    external-embedding numbers never move.
    """
    import numpy as np
    roles = [("X", arm["X"]), ("Y", arm["Y"]), ("A", arm["A"]), ("B", arm["B"])]
    present, missing = {}, defaultdict(list)
    for role, words in roles:
        for w in words:
            if acc.n(dataset, arm_name, w):
                present.setdefault(role, []).append(w)
            else:
                missing[role].append(w)
    if any(len(present.get(r, [])) == 0 for r, _ in roles):
        return [], dict(missing)

    rows = []
    for i in range(n_hidden_states):
        mats = {r: np.asarray([acc.prototype(dataset, arm_name, w)[i]
                               for w in present[r]], dtype=float)
                for r, _ in roles}
        sx = weat_association(mats["X"], mats["A"], mats["B"])
        sy = weat_association(mats["Y"], mats["A"], mats["B"])
        row = dict(domain=dataset, category=arm_name,
                   X_name=arm["X_name"], Y_name=arm["Y_name"],
                   A_name=arm["A_name"], B_name=arm["B_name"])
        row.update(layer_label(i))
        row.update(
            weat_S=weat_test_stat(mats["X"], mats["Y"], mats["A"], mats["B"]),
            weat_d=weat_effect_size_hidden(mats["X"], mats["Y"],
                                           mats["A"], mats["B"], ddof=0),
            effect_size_sd="population (ddof=0)",
            mean_s_X=float(np.mean(sx)), mean_s_Y=float(np.mean(sy)),
            n_X=len(present["X"]), n_Y=len(present["Y"]),
            n_A=len(present["A"]), n_B=len(present["B"]),
            target_s_values=json.dumps(
                {w: float(v) for w, v in
                 list(zip(present["X"], sx)) + list(zip(present["Y"], sy))}),
            n_contexts_per_target=json.dumps(
                {w: acc.n(dataset, arm_name, w)
                 for w in present["X"] + present["Y"]}),
            n_contexts_per_attribute=json.dumps(
                {w: acc.n(dataset, arm_name, w)
                 for w in present["A"] + present["B"]}),
            missing_stimuli=json.dumps({k: v for k, v in missing.items() if v}),
            prototype="mean over the prompts of THIS experiment in which the "
                      "stimulus appeared",
            pooling=POOLING,
            analysis_role="secondary", exploratory_across_layers=True,
            analysis_version=analysis_version)
        rows.append(row)
    return rows, dict(missing)


def stimulus_layer_rows(acc, dataset, arm_name, arm, *, n_hidden_states, analysis_version):
    """
    SECONDARY.  stimulus x hidden state: enough metadata to audit how each
    prototype was built, and where it sits on the A<->B axis.  The 4096-D
    prototypes themselves go to the npz; a CSV is not the place for them.
    """
    import numpy as np
    role_of = {}
    for role, words in (("X", arm["X"]), ("Y", arm["Y"]),
                        ("A", arm["A"]), ("B", arm["B"])):
        for w in words:
            role_of[w] = role
    A_words = [w for w in arm["A"] if acc.n(dataset, arm_name, w)]
    B_words = [w for w in arm["B"] if acc.n(dataset, arm_name, w)]
    rows = []
    for w in acc.stimuli(dataset, arm_name):
        proto = acc.prototype(dataset, arm_name, w)
        for i in range(n_hidden_states):
            v = proto[i]
            A = np.asarray([acc.prototype(dataset, arm_name, a)[i]
                            for a in A_words], dtype=float)
            B = np.asarray([acc.prototype(dataset, arm_name, b)[i]
                            for b in B_words], dtype=float)
            cA = float(np.mean(cos_sim(v[None, :], A)))
            cB = float(np.mean(cos_sim(v[None, :], B)))
            row = dict(domain=dataset, category=arm_name, stimulus=w,
                       role=role_of.get(w, ""))
            row.update(layer_label(i))
            row.update(n_contexts=acc.n(dataset, arm_name, w),
                       prototype_l2_norm=float(np.linalg.norm(v)),
                       mean_cos_to_A=cA, mean_cos_to_B=cB,
                       s_w_A_B=cA - cB,
                       prototype_dim=int(v.shape[-1]),
                       prototype_file="secondary_stimulus_prototypes.npz",
                       pooling=POOLING, analysis_role="secondary",
                       analysis_version=analysis_version)
            rows.append(row)
    return rows



# =========================================================
# Output  — from embed_bias.py
# =========================================================

def write_csv(path, rows, force):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)} — not written)")
        return
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite {path} — rerun with --force "
                         f"(nothing written this run)")
    keys = []
    for r in rows:                       # union of keys, first-seen order
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")

