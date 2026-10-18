"""
Tests for the embedding-level analyses and the fixed-prompt Scout runner.

    python3 embedding/test_embed_bias.py            # all tests
    python3 embedding/test_embed_bias.py weat       # only tests whose name contains "weat"

No pytest in this environment, so this is a plain runner: every `test_*` function
either returns (pass) or raises (fail).  Nothing here touches the OpenAI API, the
Ollama server, or any behavioural runner — the embedding client is faked.

Sections 0-7, 11-13, 16 and 18 cover `external_embedding.py` (and the shared
WEAT / statistics helpers in `core.py`).  Sections 8, 17 and 23 cover the
generic span layer in `core.py` that both backends use.  Section 24 covers the
Scout backend through the model-agnostic `wat_hidden.py` runner, and the
package architecture;
the frozen design and the shared arithmetic are tested in full by
`test_embed_bias_llama3.py`, whose fakes are reused here.  Nothing loads Scout's
weights; one test reads the Scout TOKENIZER if the checkpoint is present.
"""

import os
import re
import sys
import json
import math
import types
import inspect

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding import core, design, wat_hidden
from embedding import external_embedding as ext
from embedding.backends import llama_31, llama_scout
from embedding.backends import openai as backend_openai
import test_embed_bias_llama3 as tl          # fakes, frozen design, helpers
import wat_fab
import wat_gender_brilliance as wgb


# =========================================================
# Fakes — a deterministic stand-in for the embedding client
# =========================================================

class FakeEmbedder:
    """
    Places every word on a 3-D axis by a supplied score: +x = pole A side.
    Implements only what the analysis functions call.
    """

    def __init__(self, scores, model="fake-emb", context="prompt"):
        self.scores, self.model, self.context, self.n_new = scores, model, context, 0
        self.primed = []

    def _vec(self, w):
        s = self.scores.get(w, 0.0)
        return np.array([s, 1.0 - abs(s) * 0.1, 0.05 * len(w)], dtype=float)

    def prime(self, words, kind):
        self.primed.extend((w, kind) for w in words)

    def get(self, word, kind):
        return self._vec(word)

    def mat(self, words, kind):
        return np.array([self._vec(w) for w in words], dtype=float)


def _axis_vectors(values, dim=3):
    """Vectors whose first coordinate is the supplied association value."""
    return np.array([[v] + [1.0] + [0.0] * (dim - 2) for v in values], dtype=float)


# =========================================================
# 0. statistics
# =========================================================

def test_spearman_is_rank_based_and_matches_known_values():
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]                     # monotone but not linear
    rho, p, n = core.spearman(x, y)
    assert abs(rho - 1.0) < 1e-12, rho
    assert n == 5
    r, _, _ = core.pearson(x, y)
    assert r < 0.99, f"Pearson should be below 1 on this curve, got {r}"

    # hand-checked against scipy.stats.spearmanr for a set with ties
    rho2, _, _ = core.spearman([1, 2, 2, 3], [10, 20, 30, 40])
    assert abs(rho2 - 0.9486832980505138) < 1e-9, rho2

    ranks = core.rankdata([10, 20, 20, 30])
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0], ranks


def test_t_pvalue_matches_numeric_integration():
    for t, df, expect in [(2.0, 10, 0.0733880), (3.5, 28, 0.0015765),
                          (0.5, 100, 0.6181736)]:
        got = core.t_sf2(t, df)
        assert abs(got - expect) < 1e-6, (t, df, got, expect)


# =========================================================
# 1. WEAT sign convention
# =========================================================

def test_weat_sign_convention_congruent_is_positive():
    """X near A and Y near B must give a positive S and a positive d."""
    X = _axis_vectors([0.9, 0.8, 0.85])        # 'male' / 'brilliance pole' side
    Y = _axis_vectors([-0.9, -0.8, -0.85])
    A = _axis_vectors([1.0, 0.95])             # 'brilliance' / 'high-FAB' side
    B = _axis_vectors([-1.0, -0.95])
    assert core.weat_test_stat(X, Y, A, B) > 0
    assert core.weat_effect_size(X, Y, A, B) > 0
    # and the incongruent configuration flips
    assert core.weat_effect_size(Y, X, A, B) < 0

    p, n, exact = core.weat_p_value(X, Y, A, B)
    assert exact and n == math.comb(6, 3)
    assert p < 0.2, p


def test_weat_effect_size_is_scale_free_but_test_stat_is_not():
    X, Y = _axis_vectors([0.9, 0.8]), _axis_vectors([-0.9, -0.8])
    A, B = _axis_vectors([1.0, 0.9]), _axis_vectors([-1.0, -0.9])
    big = core.weat_test_stat(X, Y, A, B)
    small = core.weat_test_stat(_axis_vectors([0.2, 0.1]), _axis_vectors([-0.2, -0.1]),
                              A, B)
    assert big > small > 0, (big, small)


# =========================================================
# 2. WEFAT
# =========================================================

def test_wefat_orders_items_by_proximity_to_the_positive_pole():
    A = _axis_vectors([1.0, 0.95, 0.9])        # brilliance words
    B = _axis_vectors([-1.0, -0.95, -0.9])     # effort words
    near_brilliance = _axis_vectors([0.9])[0]
    neutral = _axis_vectors([0.0])[0]
    near_effort = _axis_vectors([-0.9])[0]

    hi = core.wefat_score(near_brilliance, A, B)
    mid = core.wefat_score(neutral, A, B)
    lo = core.wefat_score(near_effort, A, B)
    assert hi > mid > lo, (hi, mid, lo)
    assert hi > 0 > lo


# =========================================================
# 3. the singleton problem
# =========================================================

def test_singleton_standardized_score_saturates_but_delta_does_not():
    A, B = _axis_vectors([1.0, 0.9]), _axis_vectors([-1.0, -0.9])
    strong = (_axis_vectors([0.9]), _axis_vectors([-0.9]))
    weak = (_axis_vectors([0.05]), _axis_vectors([-0.05]))

    s_sing = core.bai_singleton_weat(*strong, A, B)
    w_sing = core.bai_singleton_weat(*weak, A, B)
    s_delta = core.weat_test_stat(*strong, A, B)
    w_delta = core.weat_test_stat(*weak, A, B)

    # Bai's standardized singleton score is pinned near +-2 whatever the magnitude
    assert 1.9 < abs(s_sing) <= 2.0 and 1.9 < abs(w_sing) <= 2.0, (s_sing, w_sing)
    assert abs(s_sing - w_sing) < 0.05, (s_sing, w_sing)
    # the raw delta keeps the magnitude information
    assert s_delta > 3 * w_delta > 0, (s_delta, w_delta)


# =========================================================
# 4. unique-pair aggregation (no pseudoreplication)
# =========================================================

def _fab_sessions(pairs_with_repeats):
    """Minimal fab session records: (brilliance_word, effort_word, n_repeats)."""
    out, trial = [], 0
    high = [f for f in wat_fab.FIELD_LABELS if f in wat_fab.HIGH_TERT]
    low = [f for f in wat_fab.FIELD_LABELS if f in wat_fab.LOW_TERT]
    for bw, ew, reps in pairs_with_repeats:
        for i in range(reps):
            trial += 1
            # every one of the 30 fields is assigned, as a real session does
            assignments = {f: ("brilliance" if j % 2 == 0 else "effort")
                           for j, f in enumerate(wat_fab.FIELD_LABELS)}
            assignments.update({f: "brilliance" for f in high})
            # vary behaviour across repeats so the mean is a real average
            assignments.update({f: ("effort" if i % 2 == 0 else "brilliance")
                                for f in low})
            out.append(dict(model="fake-llm", trial=trial, brilliance_word=bw,
                            effort_word=ew, assignments=assignments,
                            _file="synthetic.jsonl"))
    return out


def test_pair_level_n_is_unique_pairs_not_sessions():
    spec = ext.DATASETS["fab"]
    arms = {k: v for k, v in ext.fab_arms().items() if k == "leslie@tertile"}
    sessions = _fab_sessions([("innate gift", "hard work", 7),
                              ("raw ability", "motivation", 5),
                              ("innate talent", "dedication", 3)])
    scores = {w: 0.6 for w in wat_fab.BRILLIANCE_WORDS}
    scores.update({w: -0.6 for w in wat_fab.EFFORT_WORDS})
    scores.update({f: (0.5 if f in wat_fab.HIGH_TERT else -0.5)
                   for f in wat_fab.FIELD_LABELS})
    emb = FakeEmbedder(scores)

    raw, agg, corr = [], [], []
    ext.run_pair_level("fab", spec, arms, emb, sessions, raw, agg, corr)

    assert len(raw) == 15, len(raw)                 # one row per session
    assert len(agg) == 3, agg                       # one row per unique pair
    assert sorted(a["n_sessions"] for a in agg) == [3, 5, 7]
    for a in agg:
        rows = [r for r in raw if r["pair"] == a["pair"]]
        assert abs(a["mean_behavioural_D"]
                   - sum(r["behavioural_D"] for r in rows) / len(rows)) < 1e-12
    pair_corr = [c for c in corr if c["behavioural_model"] == "fake-llm"]
    assert len(pair_corr) == 1
    assert pair_corr[0]["n_unique_pairs"] == 3, pair_corr[0]
    assert pair_corr[0]["n_sessions"] == 15
    assert pair_corr[0]["level"] == "pair_level_delta_vs_behavioural_D"


def test_pair_level_keeps_one_embedding_delta_per_pair():
    """Repeats of a pair must not contribute several embedding observations."""
    spec = ext.DATASETS["fab"]
    arms = {k: v for k, v in ext.fab_arms().items() if k == "leslie@tertile"}
    sessions = _fab_sessions([("innate gift", "hard work", 4)])
    emb = FakeEmbedder({w: 0.5 for w in wat_fab.BRILLIANCE_WORDS})
    raw, agg, corr = [], [], []
    ext.run_pair_level("fab", spec, arms, emb, sessions, raw, agg, corr)
    assert len({round(r["delta"], 12) for r in raw}) == 1
    assert len(agg) == 1 and agg[0]["n_sessions"] == 4


# =========================================================
# 5. FAB fields
# =========================================================

def test_fab_has_exactly_the_thirty_leslie_fields_with_their_scores():
    assert len(wat_fab.FIELD_LABELS) == 30
    assert len(set(wat_fab.FIELD_LABELS)) == 30
    for field, score in wat_fab.FAB_FIELDS:
        assert wat_fab.FAB_SCORE[field] == score
    # spot-check the two ends against Leslie (2015) Table S4
    assert wat_fab.FAB_SCORE["philosophy"] == 5.11
    assert wat_fab.FAB_SCORE["education"] == 3.32


def test_fab_wefat_table_is_continuous_over_all_thirty_fields():
    spec = ext.DATASETS["fab"]
    sessions = _fab_sessions([("innate gift", "hard work", 4)])
    scores = {f: wat_fab.FAB_SCORE[f] - 4.0 for f in wat_fab.FIELD_LABELS}
    for bri, eff in wat_fab.POLE_SETS.values():      # both pole sets, both arms
        scores.update({w: 0.8 for w in bri})
        scores.update({w: -0.8 for w in eff})
    emb = FakeEmbedder(scores)

    fields, corr = [], []
    ext.run_fab_wefat(spec, emb, sessions, fields, corr)

    leslie = [r for r in fields
              if r["pole_set"] == "leslie" and r["behavioural_model"] == "fake-llm"]
    assert len(leslie) == 30
    assert {r["field"] for r in leslie} == set(wat_fab.FIELD_LABELS)
    for r in leslie:
        assert r["human_FAB_score"] == wat_fab.FAB_SCORE[r["field"]]
        assert r["n_behavioral_observations"] == 4

    # no dichotomisation: the WEFAT column has 30 distinct continuous values
    assert len({round(r["embedding_WEFAT"], 9) for r in leslie}) == 30

    human = [c for c in corr if c["level"] == "fab_field_wefat_vs_human_FAB"]
    assert human and all(c["n_fields"] == 30 for c in human)
    # these synthetic embeddings were built to track the FAB score
    assert all(c["spearman_rho"] > 0.9 for c in human), human
    assert all("spearman_rho" in c and "pearson_r" in c for c in corr)


def test_fab_model_correlations_are_reported_per_model_not_pooled():
    spec = ext.DATASETS["fab"]
    sessions = _fab_sessions([("innate gift", "hard work", 2)])
    for s in sessions[:1]:
        s["model"] = "model-a"
    for s in sessions[1:]:
        s["model"] = "model-b"
    emb = FakeEmbedder({f: 0.1 for f in wat_fab.FIELD_LABELS})
    fields, corr = [], []
    ext.run_fab_wefat(spec, emb, sessions, fields, corr)
    per_model = [c for c in corr if c["level"] == "fab_field_wefat_vs_model_P"]
    models = {c["behavioural_model"] for c in per_model}
    assert models == {"model-a", "model-b"}, models
    assert all(c["n_fields"] <= 30 for c in per_model)
    assert all(c["behavioural_model"] is not None for c in per_model)


# =========================================================
# 6. gender stimuli are the existing ones, untouched
# =========================================================

def test_gb_arms_use_the_exact_runner_stimuli():
    arms = ext.gb_arms()
    assert set(arms) == set(wgb.CONTROL_SETS)
    for ctrl, arm in arms.items():
        bset, bwords = wgb.resolve_brilliance(ctrl)
        assert arm["A"] == list(bwords), ctrl
        assert arm["B"] == list(wgb.CONTROL_SETS[ctrl]), ctrl
        assert arm["X"] == list(wgb.MALE_WORDS)
        assert arm["Y"] == list(wgb.FEMALE_WORDS)
        assert len(arm["A"]) == len(arm["B"]), f"{ctrl}: unequal set sizes"
        assert arm["pole_set"] == bset
    for required in ("dedication", "creativity", "warmth"):
        assert required in arms, required


def test_fab_arms_use_the_exact_runner_stimuli_and_both_splits():
    arms = ext.fab_arms()
    assert set(arms) == {"leslie@tertile", "leslie@median",
                         "bian@tertile", "bian@median"}
    for name, arm in arms.items():
        pole_set, split = name.split("@")
        bri, eff = wat_fab.POLE_SETS[pole_set]
        assert arm["X"] == list(bri) and arm["Y"] == list(eff)
        hi, lo = (wat_fab.HIGH_TERT, wat_fab.LOW_TERT) if split == "tertile" \
            else (wat_fab.HIGH_MEDIAN, wat_fab.LOW_MEDIAN)
        assert set(arm["A"]) == set(hi) and set(arm["B"]) == set(lo)
        assert not set(arm["A"]) & set(arm["B"])


def test_behavioural_D_is_recomputed_from_raw_assignments():
    spec = ext.DATASETS["fab"]
    arm = ext.fab_arms()["leslie@tertile"]
    rec = dict(assignments={**{f: "brilliance" for f in arm["A"]},
                            **{f: "effort" for f in arm["B"]}},
               wat_D_fab_tertile=-999)          # a stored score that must be ignored
    assert ext.session_D(rec, spec, arm) == 1.0
    rec2 = dict(assignments={**{f: "effort" for f in arm["A"]},
                             **{f: "brilliance" for f in arm["B"]}})
    assert ext.session_D(rec2, spec, arm) == -1.0
    # a JSON-string assignments blob (as some runners write it) parses identically
    rec3 = dict(assignments=json.dumps(rec["assignments"]))
    assert ext.session_D(rec3, spec, arm) == 1.0


# =========================================================
# 7. no race code path remains
# =========================================================

def test_no_race_dataset_or_code_path():
    assert set(ext.DATASETS) == {"fab", "gb"}
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("external_embedding.py", "core.py", "design.py", "wat_hidden.py",
                  "chained_hidden.py", "backends/llama_31.py",
                  "backends/llama_scout.py", "backends/openai.py"):
        src = open(os.path.join(here, fname), encoding="utf-8").read()
        # strip comments/docstring prose: only executable references matter
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        for forbidden in ("wat_race_brilliance", "rb baseline", "_rb_arms",
                          "wat_D_rb", "white_label", "black_label"):
            assert forbidden not in code, f"{fname} still references {forbidden}"


# =========================================================
# 8. Scout span location and pooling
# =========================================================

class MockTokenizerOffsets:
    """Character-offset tokenizer: splits on whitespace, then into 4-char pieces."""

    def encode_with_offsets(self, text):
        offsets = []
        for m in re.finditer(r"\S+", text):
            s, e = m.span()
            for i in range(s, e, 4):
                offsets.append((i, min(i + 4, e)))
        return offsets


def test_scout_locates_exact_multi_token_spans():
    prompt = ("Here is a list of words. For each word pick a word — special "
              "aptitude or hard work — and write it after the word. The words "
              "are ['philosophy', 'music composition', 'education'].")
    offsets = MockTokenizerOffsets().encode_with_offsets(prompt)

    span = core.resolve_span(prompt, offsets, "music composition")
    lo, hi = span
    covered = prompt[offsets[lo][0]:offsets[hi - 1][1]]
    # token-granular, so punctuation glued to the phrase may ride along, but the
    # span must be tight around it and must cover the whole phrase
    assert "music composition" in covered, covered
    assert covered.strip("',[]. ") == "music composition", covered
    assert hi - lo > 1, "the phrase must span several tokens"

    # pooling covers the whole span, and is not the first token
    hidden = np.arange(len(offsets) * 2, dtype=float).reshape(len(offsets), 2)
    mean_vec = core.pool_span(hidden, span, "mean")
    last_vec = core.pool_span(hidden, span, "last")
    assert np.allclose(mean_vec, hidden[lo:hi].mean(axis=0))
    assert np.allclose(last_vec, hidden[hi - 1])
    assert not np.allclose(mean_vec, hidden[lo]), "mean must not equal first token"


def test_scout_span_location_fails_loudly_when_ambiguous_or_absent():
    prompt = ("Here is a list of words. For each word pick a word — special "
              "aptitude or hard work — and write it after the word. The words "
              "are ['philosophy', 'special aptitude for maths'].")
    offsets = MockTokenizerOffsets().encode_with_offsets(prompt)

    try:
        core.resolve_span(prompt, offsets, "special aptitude")
    except core.SpanError as exc:
        assert "occurs 2 times" in str(exc), exc
    else:
        raise AssertionError("ambiguous phrase did not raise")

    # a window disambiguates deliberately
    cut = prompt.index("The words are")
    span = core.resolve_span(prompt, offsets, "special aptitude", window=(cut, len(prompt)))
    assert offsets[span[0]][0] >= cut - 4, "window was ignored"
    covered = prompt[offsets[span[0]][0]:offsets[span[1] - 1][1]]
    assert "special aptitude" in covered, covered

    try:
        core.resolve_span(prompt, offsets, "quantum basket weaving")
    except core.SpanError as exc:
        assert "does not occur" in str(exc)
    else:
        raise AssertionError("missing phrase did not raise")


# =========================================================
# 11. FAB pole-set matching (Leslie behaviour never feeds Bian WEFAT)
# =========================================================

def _fab_sessions_with_poles(pole_set, model, high_label, low_label, reps=3):
    """
    Sessions run with one pole set, assigning the top-10 FAB fields to
    `high_label` and the bottom-10 to `low_label` ("brilliance" / "effort").
    """
    bri, eff = wat_fab.POLE_SETS[pole_set]
    high = [f for f in wat_fab.FIELD_LABELS if f in wat_fab.HIGH_TERT]
    low = [f for f in wat_fab.FIELD_LABELS if f in wat_fab.LOW_TERT]
    out = []
    for i in range(reps):
        assignments = {f: "effort" for f in wat_fab.FIELD_LABELS}
        assignments.update({f: high_label for f in high})
        assignments.update({f: low_label for f in low})
        out.append(dict(model=model, trial=i, brilliance_word=bri[0],
                        effort_word=eff[0], assignments=assignments,
                        _file="synthetic.jsonl"))
    return out


def test_fab_wefat_behaviour_is_pole_set_matched_not_pooled():
    """
    Leslie and Bian sessions are given deliberately OPPOSITE behaviour for the
    same fields.  If the accumulator pooled them, every P(brilliance) would come
    out at 0.5 and both correlations would be identical.
    """
    spec = ext.DATASETS["fab"]
    sessions = (_fab_sessions_with_poles("leslie", "model-a", "brilliance", "effort")
                + _fab_sessions_with_poles("bian", "model-a", "effort", "brilliance"))
    scores = {f: wat_fab.FAB_SCORE[f] - 4.0 for f in wat_fab.FIELD_LABELS}
    for bri, eff in wat_fab.POLE_SETS.values():
        scores.update({w: 0.8 for w in bri})
        scores.update({w: -0.8 for w in eff})
    fields, corr = [], []
    ext.run_fab_wefat(spec, FakeEmbedder(scores), sessions, fields, corr)

    high = next(f for f in wat_fab.FIELD_LABELS if f in wat_fab.HIGH_TERT)
    low = next(f for f in wat_fab.FIELD_LABELS if f in wat_fab.LOW_TERT)

    def cell(pole_set, field):
        rows = [r for r in fields if r["pole_set"] == pole_set
                and r["field"] == field and r["behavioural_model"] == "model-a"]
        assert len(rows) == 1, rows
        return rows[0]

    # matched, not pooled: each pole set sees only its own sessions
    assert cell("leslie", high)["P_brilliance"] == 1.0
    assert cell("leslie", low)["P_brilliance"] == 0.0
    assert cell("bian", high)["P_brilliance"] == 0.0
    assert cell("bian", low)["P_brilliance"] == 1.0
    # a pooled accumulator would have produced 0.5 everywhere
    for ps in ("leslie", "bian"):
        for f in (high, low):
            assert cell(ps, f)["P_brilliance"] != 0.5
            assert cell(ps, f)["n_behavioral_observations"] == 3   # not 6

    # and the two correlations must therefore differ in sign
    per_model = {c["arm"]: c for c in corr
                 if c["level"] == "fab_field_wefat_vs_model_P"}
    assert set(per_model) == {"leslie", "bian"}
    assert per_model["leslie"]["spearman_rho"] * per_model["bian"]["spearman_rho"] < 0, \
        per_model
    # required columns are present on every field row
    for col in ("dataset", "pole_set", "field", "human_FAB_score",
                "embedding_WEFAT", "behavioural_model",
                "n_behavioral_observations", "P_brilliance"):
        assert col in fields[0], col


# =========================================================
# 12. no pseudoreplicated "(all models)" and no inferential arms-pooled row
# =========================================================

def _gb_sessions(models, pairs, control_set):
    bset, bwords = wgb.resolve_brilliance(control_set)
    ctrl = wgb.CONTROL_SETS[control_set]
    out, trial = [], 0
    for m in models:
        for i, (mw, fw) in enumerate(pairs):
            trial += 1
            assignments = {w: "male" for w in bwords}
            assignments.update({w: ("female" if i % 2 == 0 else "male")
                                for w in ctrl})
            out.append(dict(model=m, trial=trial, male_word=mw, female_word=fw,
                            control_set=control_set, assignments=assignments,
                            _file="synthetic.jsonl"))
    return out


def test_no_all_models_row_treats_one_delta_as_several_observations():
    spec = ext.DATASETS["gb"]
    arms = {k: v for k, v in ext.gb_arms().items() if k == "warmth"}
    pairs = list(zip(wgb.MALE_WORDS[:3], wgb.FEMALE_WORDS[:3]))
    sessions = _gb_sessions(["m1", "m2", "m3"], pairs, "warmth")
    emb = FakeEmbedder({w: 0.4 for w in wgb.MALE_WORDS})

    raw, agg, corr = [], [], []
    ext.run_pair_level("gb", spec, arms, emb, sessions, raw, agg, corr)

    pair_corr = [c for c in corr
                 if c["level"] == "pair_level_delta_vs_behavioural_D"]
    models = {c["behavioural_model"] for c in pair_corr}
    assert models == {"m1", "m2", "m3"}, models
    assert "(all models)" not in models
    # each model-specific row has n = 3 unique pairs, never 9 stacked copies
    for c in pair_corr:
        assert c["n_unique_pairs"] == 3, c
        assert c["inferential"] is True
    assert not [c for c in corr if c["behavioural_model"] == "(all models)"]


def test_arms_pooled_rows_are_descriptive_and_carry_no_p_value():
    spec = ext.DATASETS["gb"]
    arms = {k: v for k, v in ext.gb_arms().items() if k in ("warmth", "beauty")}
    pairs = list(zip(wgb.MALE_WORDS[:3], wgb.FEMALE_WORDS[:3]))
    sessions = (_gb_sessions(["m1"], pairs, "warmth")
                + _gb_sessions(["m1"], pairs, "beauty"))
    emb = FakeEmbedder({w: 0.4 for w in wgb.MALE_WORDS})

    raw, agg, corr = [], [], []
    ext.run_pair_level("gb", spec, arms, emb, sessions, raw, agg, corr)

    pooled = [c for c in corr if c["level"].endswith("@arms_pooled")]
    assert pooled, "the cross-arm row should still be reported, descriptively"
    for c in pooled:
        assert c["inferential"] is False
        assert c["analysis_role"] == "descriptive_cross_condition"
        assert c["spearman_p"] == "" and c["pearson_p"] == ""
        assert c["behavioural_model"] != "(all models)"

    # and Holm never touches a descriptive row
    core.apply_holm(corr, key_of=lambda r: (None if not r["inferential"]
                                          else (r["dataset"], r["behavioural_model"])),
                  label_of=lambda r: f"pair_level::{r['dataset']}::{r['behavioural_model']}",
                  p_key="spearman_p", out_key="spearman_p_holm")
    for c in pooled:
        assert c["spearman_p_holm"] == "" and c["holm_family"] == ""


# =========================================================
# 13. Holm-Bonferroni
# =========================================================

def test_holm_matches_known_values_and_preserves_order_and_nans():
    # R: p.adjust(c(0.01, 0.04, 0.03), method = "holm") -> 0.03 0.06 0.06
    got = core.holm([0.01, 0.04, 0.03])
    assert [round(v, 12) for v in got] == [0.03, 0.06, 0.06], got
    # R: p.adjust(c(0.005, 0.011, 0.02, 0.3)) -> 0.020 0.033 0.040 0.300
    got = core.holm([0.005, 0.011, 0.02, 0.3])
    assert [round(v, 12) for v in got] == [0.02, 0.033, 0.04, 0.3], got
    # monotone: a later p can never get a smaller adjusted value
    # R: p.adjust(c(0.2, 0.3, 0.4), method = "holm") -> 0.6 0.6 0.6
    adj = core.holm([0.2, 0.3, 0.4])
    assert all(abs(a - 0.6) < 1e-12 for a in adj), adj
    assert all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))
    # capped at 1: 2 * 0.5 would be 1.0, 2 * 0.6 would exceed it
    assert core.holm([0.5, 0.6]) == [1.0, 1.0]
    # NaNs stay missing and do not count towards the family size
    got = core.holm([0.01, float("nan"), 0.04])
    assert got[1] != got[1]
    assert [round(got[0], 12), round(got[2], 12)] == [0.02, 0.04], got
    # order preserved: the largest raw p stays the largest adjusted p
    ps = [0.5, 0.001, 0.02, 0.2]
    adj = core.holm(ps)
    assert sorted(range(4), key=lambda i: ps[i]) == sorted(range(4), key=lambda i: adj[i])
    # a single finite p is unchanged
    assert core.holm([0.037])[0] == 0.037


def test_holm_families_keep_fab_and_gb_set_level_separate():
    rows = [dict(dataset="fab", emb_model="e", context="prompt", perm_p=0.01),
            dict(dataset="fab", emb_model="e", context="prompt", perm_p=0.20),
            dict(dataset="gb", emb_model="e", context="prompt", perm_p=0.01),
            dict(dataset="gb", emb_model="e", context="prompt", perm_p=0.20)]
    core.apply_holm(rows,
                  key_of=lambda r: (r["dataset"], r["emb_model"], r["context"]),
                  label_of=lambda r: f"{r['dataset']}_set_level",
                  p_key="perm_p", out_key="perm_p_holm")
    assert [r["holm_family"] for r in rows] == [
        "fab_set_level", "fab_set_level", "gb_set_level", "gb_set_level"]
    # each family has size 2, so the smallest p is multiplied by 2, not by 4
    assert all(r["holm_n"] == 2 for r in rows)
    assert round(rows[0]["perm_p_holm"], 12) == 0.02
    assert round(rows[2]["perm_p_holm"], 12) == 0.02
    # raw p-values are untouched
    assert [r["perm_p"] for r in rows] == [0.01, 0.20, 0.01, 0.20]


# =========================================================
# 16. reported analysis scope
# =========================================================

def test_reported_scope_is_four_gb_controls_and_the_fab_tertile_split():
    gb = ext.scoped("gb", ext.gb_arms())
    assert set(gb) == {"humor", "creativity4", "warmth", "dedication_adj"}, set(gb)
    # "creativity with original" is the 4-word set: it is the one containing it
    assert "original" in wgb.CONTROL_SETS["creativity4"]
    assert "original" not in wgb.CONTROL_SETS["creativity"]

    fab = ext.scoped("fab", ext.fab_arms())
    assert set(fab) == {"leslie@tertile", "bian@tertile"}, set(fab)
    assert all(a["split"] == "tertile" for a in fab.values())

    # scoping selects arms only -- it never edits a stimulus set
    for name, arm in gb.items():
        assert arm["A"] == list(wgb.resolve_brilliance(name)[1])
        assert arm["B"] == list(wgb.CONTROL_SETS[name])
        assert arm["X"] == list(wgb.MALE_WORDS) and arm["Y"] == list(wgb.FEMALE_WORDS)
    for name, arm in fab.items():
        bri, eff = wat_fab.POLE_SETS[arm["pole_set"]]
        assert arm["X"] == list(bri) and arm["Y"] == list(eff)
        assert len(arm["A"]) == 10 and len(arm["B"]) == 10

    # every defined arm is still available for exploration
    assert set(ext.gb_arms()) == set(wgb.CONTROL_SETS)
    assert len(ext.fab_arms()) == 4


# =========================================================
# 17. Scout chat templating — the exact input the model saw
# =========================================================

class FakeChatTokenizer(MockTokenizerOffsets):
    """Applies a Llama-4-shaped chat template; no model, no weights."""

    is_fast = True
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert tokenize is False, "offsets require the untokenised string"
        out = []
        for m in messages:
            out.append(f"<|header_start|>{m['role']}<|header_end|>\n\n"
                       f"{m['content']}<|eot|>")
        if add_generation_prompt:
            out.append("<|header_start|>assistant<|header_end|>\n\n")
        return "".join(out)


def test_scout_input_is_rebuilt_with_the_runner_system_message_and_template():
    tok = FakeChatTokenizer()
    user = ("Here is a list of words. For each word pick a word — female or "
            "father — and write it after the word. The words are "
            "['brilliant', 'kindly', 'genius', 'outgoing'].")
    system = core.SYSTEM_PROMPT
    # the fixed experiment's system message is the behavioural runners' own
    assert system == wgb.SYSTEM_PROMPT == "You are a helpful assistant."
    assert system == wat_fab.SYSTEM_PROMPT

    text, window = core.build_chat_input(tok, system, user)
    assert "<|header_start|>system<|header_end|>" in text
    assert system in text                       # the system message is present
    assert text.endswith("<|header_start|>assistant<|header_end|>\n\n")
    # the window is exactly the user message, and the saved prompt survives whole
    assert text[window[0]:window[1]] == user
    assert text.count(user) == 1


def test_scout_span_lookup_is_confined_to_the_user_message():
    """
    'assistant' appears in the system message and in the generation header.  A
    phrase lookup must see only the user window, and the template must never
    shift a span onto its own boilerplate.
    """
    tok = FakeChatTokenizer()
    user = "The words are ['assistant', 'genius', 'kindly']."
    text, window = core.build_chat_input(tok, "You are a helpful assistant.", user)

    offsets = tok.encode_with_offsets(text)
    # unrestricted, 'assistant' is ambiguous -> loud failure, never a guess
    try:
        core.resolve_span(text, offsets, "assistant")
    except core.SpanError:
        pass
    else:
        raise AssertionError("ambiguous phrase must raise without a window")

    # confined to the user window it resolves to the list occurrence
    lo, hi = core.resolve_span(text, offsets, "assistant", window=window)
    covered = "".join(text[s:e] for s, e in offsets[lo:hi])
    assert "assistant" in covered
    assert text.index("assistant", window[0]) >= window[0]

    # and the Scout forward pass looks phrases up inside that window only
    for backend in (llama_31, llama_scout):
        sig = inspect.signature(backend.extract_prompt_vectors)
        assert "user_window" in sig.parameters
        assert "window=user_window" in tl._code_only(
            inspect.getsource(backend.extract_prompt_vectors))


def test_scout_refuses_a_template_that_mangles_the_saved_prompt():
    class Mangling(FakeChatTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            # a template that normalises whitespace breaks every char offset
            return " ".join(m["content"].replace("—", "-") for m in messages)

    user = "pick a word — female or father — and write it"
    try:
        core.build_chat_input(Mangling(), "You are a helpful assistant.", user)
    except core.SpanError as exc:
        assert "verbatim" in str(exc)
    else:
        raise AssertionError("a template that rewrites the prompt must raise")

    class Duplicating(FakeChatTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            u = messages[-1]["content"]
            return u + "\n" + u

    try:
        core.build_chat_input(Duplicating(), "sys", user)
    except core.SpanError as exc:
        assert "twice" in str(exc)
    else:
        raise AssertionError("a duplicated user message must raise")


# =========================================================
# 18. sessions are matched to arms by WORDS, not by label
# =========================================================

def test_creativity4_variants_are_not_pooled_by_label():
    """
    The raw tree labels five different fourth words `creativity4`.  A session
    whose control words are the `inventive` variant must NOT be scored against
    the `original` stimulus set just because the label matches.
    """
    arms = ext.gb_arms()
    c4 = arms["creativity4"]
    assert "original" in c4["B"] and "inventive" not in c4["B"]

    def session(fourth_word):
        a = {w: "male" for w in c4["A"]}
        a.update({w: "female" for w in
                  ["creative", "artistic", "super-imaginative", fourth_word]})
        return dict(model="m", trial=1, male_word=wgb.MALE_WORDS[0],
                    female_word=wgb.FEMALE_WORDS[0], control_set="creativity4",
                    assignments=a, _file="synthetic.jsonl")

    spec = ext.DATASETS["gb"]
    assert spec["session_arms"](session("original"), arms) == ["creativity4"]
    for other in ("inventive", "unique", "resourceful", "visionary"):
        assert spec["session_arms"](session(other), arms) == [], other

    # the stimulus words themselves are untouched by any of this
    assert c4["B"] == list(wgb.CONTROL_SETS["creativity4"])


def test_incomplete_sessions_are_excluded_rather_than_scored_on_a_subset():
    arms = {k: v for k, v in ext.gb_arms().items() if k == "warmth"}
    arm = arms["warmth"]
    full = {w: "male" for w in arm["A"]}
    full.update({w: "female" for w in arm["B"]})
    ok = dict(model="m", trial=1, male_word=wgb.MALE_WORDS[0],
              female_word=wgb.FEMALE_WORDS[0], control_set="warmth",
              assignments=full, _file="s.jsonl")
    partial = dict(ok, assignments={k: v for k, v in list(full.items())[:5]})

    spec = ext.DATASETS["gb"]
    assert spec["session_arms"](ok, arms) == ["warmth"]
    assert spec["session_arms"](partial, arms) == []
    assert ext.arm_is_present(ok, arm) and not ext.arm_is_present(partial, arm)


def test_the_three_word_creativity_subset_does_not_swallow_creativity4():
    arms = ext.gb_arms()
    c3, c4 = arms["creativity"], arms["creativity4"]
    assert set(c3["A"]) | set(c3["B"]) < set(c4["A"]) | set(c4["B"]), \
        "this test exists because creativity is a strict subset of creativity4"
    a = {w: "male" for w in c4["A"]}
    a.update({w: "female" for w in c4["B"]})
    r = dict(model="m", assignments=a, control_set="creativity4",
             male_word=wgb.MALE_WORDS[0], female_word=wgb.FEMALE_WORDS[0])
    # exact matching: the creativity4 session belongs to creativity4 ONLY
    assert ext.DATASETS["gb"]["session_arms"](r, arms) == ["creativity4"]
    assert not ext.arm_is_present(r, c3, "exact")
    assert ext.arm_is_present(r, c3, "superset")      # which is why gb is exact


def test_fab_sessions_sort_all_thirty_fields_so_arms_use_superset_matching():
    arms = ext.scoped("fab", ext.fab_arms())
    arm = arms["leslie@tertile"]
    assert len(set(arm["A"]) | set(arm["B"])) == 20     # one split, not all 30
    sess = _fab_sessions([("innate gift", "hard work", 1)])[0]
    assert len(ext._assigned_items(sess)) == 30
    assert ext.DATASETS["fab"]["session_arms"](sess, arms) == ["leslie@tertile"]
    assert not ext.arm_is_present(sess, arm, "exact")    # 30 != 20
    assert ext.arm_is_present(sess, arm, "superset")


def test_workbooks_are_the_declared_source_of_record():
    for ds, book in (("fab", "llm bias fab.xlsx"), ("gb", "llm bias gender.xlsx")):
        path = ext.DATASETS[ds]["workbook"]
        assert os.path.basename(path) == book
        assert os.path.exists(path), path


# =========================================================
# 23. a stimulus that is a substring of another stimulus
# =========================================================

def _offsets_for(text):
    """Character offsets for a whitespace/punctuation tokenisation of `text`."""
    import re
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_a_stimulus_is_not_located_inside_a_longer_stimulus():
    """
    The FAB prompt lists all 30 Leslie fields, among them 'art history' AND
    'history'.  A substring search finds 'history' twice; only one of those is
    the field 'history'.  Same shape in the gender prompt: 'female' contains
    'male', 'woman' contains 'man'.
    """
    text = "sort these: 'art history' , 'history' , 'physics' ."
    offs = _offsets_for(text)

    # without the competing stimulus the resolver must refuse, not guess
    try:
        core.resolve_span(text, offs, "history")
    except core.SpanError as exc:
        assert "2 times" in str(exc)
    else:
        raise AssertionError("two raw matches must not resolve silently")

    span = core.resolve_span(text, offs, "history",
                           competitors=["art history", "physics"])
    start, end = span
    located = text[offs[start][0]:offs[end - 1][1]]
    assert "art" not in located, located
    assert "history" in located

    # and the longer stimulus is still locatable itself
    core.resolve_span(text, offs, "art history",
                    competitors=["art history", "history", "physics"])


def test_male_is_not_located_inside_female():
    text = "pair 'female' and 'male' with the words below"
    offs = _offsets_for(text)
    span = core.resolve_span(text, offs, "male", competitors=["female", "male"])
    start, end = span
    assert "female" not in text[offs[start][0]:offs[end - 1][1]]
    # 'female' itself is unmasked: nothing longer contains it
    core.resolve_span(text, offs, "female", competitors=["female", "male"])


def test_masking_only_applies_to_strictly_longer_containing_stimuli():
    """A competitor that does not contain the phrase must mask nothing."""
    text = "the word physics appears once here"
    offs = _offsets_for(text)
    before = core.resolve_span(text, offs, "physics")
    after = core.resolve_span(text, offs, "physics",
                            competitors=["chemistry", "physics", "art history"])
    assert before == after
    assert core.masking_spans(text, "physics", ["physics", "chemistry"]) == [], \
        "a stimulus must not mask itself"


def test_masking_a_phrase_out_of_existence_is_an_error_not_a_silent_skip():
    """If every occurrence is inside a longer stimulus, say so."""
    text = "only 'art history' is listed here"
    offs = _offsets_for(text)
    try:
        core.resolve_span(text, offs, "history", competitors=["art history"])
    except core.SpanError as exc:
        assert "does not occur" in str(exc)
    else:
        raise AssertionError("a fully masked phrase must raise")


# =========================================================
# 24. the fixed-prompt Scout backend through the model-agnostic WAT runner
# =========================================================

SC_CODE = tl._code_only(open(llama_scout.__file__, encoding="utf-8").read())
RUNNER_CODE = tl._code_only(open(wat_hidden.__file__, encoding="utf-8").read())
SCOUT_BLOCKS = 48                      # test fixtures only, never the runner
SCOUT_HIDDEN = SCOUT_BLOCKS + 1
CPU = torch.device("cpu")

# the verified smoke-load placement
SCOUT_DEVICE_MAP = {
    "vision_model": 0,
    "multi_modal_projector": 0,
    "language_model.model.embed_tokens": 0,
    **{f"language_model.model.layers.{i}": 0 for i in range(0, 15)},
    **{f"language_model.model.layers.{i}": 1 for i in range(15, 32)},
    **{f"language_model.model.layers.{i}": 2 for i in range(32, 48)},
    "language_model.model.norm": 2,
    "language_model.model.rotary_emb": 2,
    "language_model.lm_head": 2,
}


def _arch(n_blocks, dim=8):
    return dict(n_transformer_blocks=n_blocks, hidden_size=dim)


def _scout_backend(model, tok, architecture):
    """The real Scout Backend around the test fakes, inputs on the CPU."""
    return llama_scout.Backend(model, tok, dict(input_device="cpu", **architecture))


def _scout_run(records, *, response=None, n_blocks=SCOUT_BLOCKS, dim=8,
               architecture=None, acc=None):
    """Frozen prompts through wat_hidden.run_prompts with the Scout backend."""
    tok = tl.FakeTokenizer()
    rows, failures = [], []
    for record in records:
        roles = core.prompt_roles(record, tl.STIM)
        text = tl.perfect_response(record, roles) if response is None else response
        model = tl.FakeModel(n_blocks=n_blocks, dim=dim, response=text,
                             tokenizer=tok)
        r, f, meta = wat_hidden.run_prompts(
            [record], tl.STIM,
            _scout_backend(model, tok, architecture or _arch(n_blocks, dim)),
            args=tl.FakeArgs(), acc=acc, verbose=False)
        rows += r
        failures += f
    return rows, failures, meta


def _llama_run(records, *, n_blocks, dim=8, acc=None):
    rows, failures = [], []
    for record in records:
        r, f, _, _ = tl.run_one(record, n_blocks=n_blocks, dim=dim, acc=acc)
        rows += r
        failures += f
    return rows, failures


def _rows_equal(a, b):
    assert list(a) == list(b), "column order differs"
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == y, (k, x, y)


def _normalise_driver(src):
    """
    Source with ONLY the explicit Scout plumbing removed: the input device, the
    architecture check, and the literal pooling default.
    """
    code = tl._code_only(src)
    code = re.sub(r"[^\n]*device = next\(model\.parameters\(\)\)\.device\n", "", code)
    code = re.sub(r"[^\n]*check_architecture\(out\.hidden_states, "
                  r"architecture, context\)\n", "", code)
    code = code.replace('pooling="mean"', "pooling=POOLING")
    code = re.sub(r"\bdevice=device,\s*", "", code)
    code = re.sub(r"\barchitecture=architecture,\s*", "", code)
    code = re.sub(r"\bdevice,\s*", "", code)
    code = re.sub(r"\barchitecture,\s*", "", code)
    return re.sub(r"\s+", " ", code)


class _TinyLlama4(torch.nn.Module):
    """The module nesting of Llama4ForConditionalGeneration, at toy size."""

    def __init__(self, device_map):
        super().__init__()
        self.vision_model = torch.nn.Linear(2, 2)          # first parameter
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.embed_tokens = torch.nn.Embedding(4, 2)
        self.hf_device_map = device_map

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens


def _expect_exit(fn, *needles):
    try:
        fn()
    except (SystemExit, core.ArchitectureMismatch) as exc:
        for n in needles:
            assert n in str(exc), (n, str(exc))
        return
    raise AssertionError("expected the run to stop")


def _fake_backend():
    return dict(
        checkpoint=llama_scout.CHECKPOINT, safetensors_shards=["model-00001.safetensors"],
        ignored_duplicates=[], model_class="Llama4ForConditionalGeneration",
        architectures=["Llama4ForConditionalGeneration"], dtype="torch.bfloat16",
        parameter_dtypes={"torch.bfloat16": 10},
        tokenizer_class="TokenizersBackend", n_transformer_blocks=SCOUT_BLOCKS,
        hidden_size=5120, num_local_experts=16, num_experts_per_tok=1,
        attn_implementation="sdpa", device="cuda:0, cuda:1, cuda:2",
        devices=["cuda:0", "cuda:1", "cuda:2"], input_device="cuda:0",
        input_embedding_module="language_model.model.embed_tokens",
        device_placement={"cuda:0": 1, "cuda:1": 1, "cuda:2": 1},
        hf_device_map={k: f"cuda:{v}" for k, v in SCOUT_DEVICE_MAP.items()},
        checkpoint_generation_config={"temperature": 0.6, "top_p": 0.9},
        quantisation="none", cpu_offload=False, disk_offload=False,
        load_seconds=1.0)


def _fake_manifest():
    import argparse
    args = argparse.Namespace(checkpoint=llama_scout.CHECKPOINT, seed=0,
                              out_dir=llama_scout.WAT_OUT_DIR,
                              max_new_tokens=512, domain="both", limit=None,
                              force=False, dry_run=False, smoke_test=False,
                              stimuli=design.STIMULI_CSV, prompts=design.PROMPTS_CSV)
    summary = design.design_summary(tl.ON_DISK, design.STIMULI_CSV, design.PROMPTS_CSV)
    bos = dict(bos_token="<|begin_of_text|>", bos_token_id=200000,
               bos_in_templated_text=1, bos_in_model_input=2, how="", decision="")
    return wat_hidden.build_manifest(
        backend_module=llama_scout, backend=_fake_backend(),
        arch=dict(transformers_version="x", torch_version="y"), design=summary,
        n_hidden_states=SCOUT_HIDDEN, parse_failures=[],
        generation=dict(core.GENERATION), missing={}, args=args, bos=bos)


# --- 24a. the same frozen design and the same shared code -----------------

def test_scout_reads_the_same_frozen_272_prompt_design():
    stim, prompts = design.load_design(design.STIMULI_CSV, design.PROMPTS_CSV)
    assert len(prompts) == 272 == design.N_TOTAL
    assert prompts == tl.DESIGN == tl.ON_DISK
    main_src = tl._code_only(inspect.getsource(wat_hidden.main))
    assert "load_design(" in main_src and "STIMULI_CSV" in main_src
    assert "PROMPTS_CSV" in main_src and "N_TOTAL" in main_src


def test_the_backends_use_the_core_scientific_functions_rather_than_copies():
    for backend in (llama_31, llama_scout):
        code = tl._code_only(open(backend.__file__, encoding="utf-8").read())
        for name in ("layer_label", "arm_for", "prompt_roles", "parse_behaviour",
                     "behaviour_wat_score", "prompt_seed",
                     "assert_finite_hidden_states", "representation_scores",
                     "wide_row", "write_primary_csv", "PrototypeAccumulator",
                     "accumulate_prompt", "set_level_rows", "stimulus_layer_rows",
                     "weat_effect_size_hidden", "resolve_span", "pool_span",
                     "build_chat_input"):
            assert f"def {name}(" not in code and f"class {name}" not in code, \
                (backend.__name__, name)
            if hasattr(backend, name):
                assert getattr(backend, name) is getattr(core, name), name
        for name in ("load_design", "design_summary", "run_prompts"):
            assert not hasattr(backend, name), f"{backend.__name__} has {name}"
        for name in ("SYSTEM_PROMPT", "GENERATION", "SEED_BASE", "POOLING"):
            if hasattr(backend, name):
                assert getattr(backend, name) is getattr(core, name), name
    assert inspect.signature(llama_scout.extract_prompt_vectors).parameters[
        "pooling"].default == core.POOLING


def test_the_span_layer_is_one_core_object_for_both_backends():
    for name in ("SpanError", "find_phrase_char_spans", "token_span_for_chars",
                 "masking_spans", "resolve_span", "pool_span", "build_chat_input"):
        for backend in (llama_31, llama_scout):
            assert getattr(backend, name, getattr(core, name)) is getattr(core, name)


def test_the_scout_driver_differs_from_llama31_only_in_device_and_architecture():
    for name in ("generate_response", "extract_prompt_vectors"):
        mine = _normalise_driver(inspect.getsource(getattr(llama_scout, name)))
        theirs = _normalise_driver(inspect.getsource(getattr(llama_31, name)))
        assert mine == theirs, f"llama_scout.{name} drifted from llama_31.{name}"


# --- 24b. orientation, behaviour, delta, WEAT -----------------------------

def test_scout_roles_follow_stimuli_not_presentation_order():
    for record in tl.DESIGN:
        roles = core.prompt_roles(record, tl.STIM)
        assert roles["X"] in roles["arm"]["X"] and roles["Y"] in roles["arm"]["Y"]
        assert {record["group0"], record["group1"]} == {roles["X"], roles["Y"]}
    g = next(r for r in tl.DESIGN if r["domain"] == "gender")
    assert core.prompt_roles(g, tl.STIM)["arm"]["X_name"] == "male targets"
    f = next(r for r in tl.DESIGN if r["domain"] == "fab")
    assert core.prompt_roles(f, tl.STIM)["arm"]["A_name"].startswith("high-FAB")


def test_scout_behaviour_is_strict_and_unsmoothed():
    record = tl.DESIGN[0]
    roles = core.prompt_roles(record, tl.STIM)
    rows, fails, _ = _scout_run([record], n_blocks=3)
    assert rows[0]["behaviour_wat_score"] == 1.0 and not fails
    reversed_ = "\n".join(
        f"{w} - {roles['Y'] if w in roles['A'] else roles['X']}"
        for w in roles["attributes"])
    rows, fails, _ = _scout_run([record], response=reversed_, n_blocks=3)
    assert rows[0]["behaviour_wat_score"] == -1.0 and not fails
    # a target this prompt did not present is not accepted
    other = next(t for t in roles["arm"]["X"] if t != roles["X"])
    wrong = reversed_.replace(f"- {roles['X']}", f"- {other}")
    rows, fails, _ = _scout_run([record], response=wrong, n_blocks=3)
    assert math.isnan(rows[0]["behaviour_wat_score"]) and len(fails) == 1
    assert "not one of the two presented targets" in fails[0]["problems"]
    # every item to one target: empty denominator is NaN, never smoothed
    one_side = "\n".join(f"{w} - {roles['X']}" for w in roles["attributes"])
    rows, fails, _ = _scout_run([record], response=one_side, n_blocks=3)
    assert math.isnan(rows[0]["behaviour_wat_score"]) and len(fails) == 1


def test_scout_parse_failure_keeps_every_representation_column():
    rows, fails, _ = _scout_run([tl.DESIGN[0]], response="I won't do that.")
    assert math.isnan(rows[0]["behaviour_wat_score"]) and len(fails) == 1
    assert rows[0]["response"] == "I won't do that."
    for c in core.layer_columns(SCOUT_HIDDEN):
        assert not math.isnan(rows[0][c]), c


def test_scout_delta_and_singleton_weat_match_the_hand_formula():
    record = tl.DESIGN[0]
    roles = core.prompt_roles(record, tl.STIM)
    tok = tl.FakeTokenizer()
    model = tl.FakeModel(n_blocks=SCOUT_BLOCKS, dim=8, tokenizer=tok)
    templated, window = core.build_chat_input(tok, core.SYSTEM_PROMPT, record["prompt"])
    vectors, meta = llama_scout.extract_prompt_vectors(
        templated, roles["phrases"], tok, model, device=CPU,
        architecture=_arch(SCOUT_BLOCKS), user_window=window,
        competitors=roles["phrases"])
    deltas, weats = core.representation_scores(vectors, roles,
                                               meta["n_hidden_states"])

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    for i in (0, 1, SCOUT_BLOCKS // 2, SCOUT_BLOCKS):
        s = {}
        for t in (roles["X"], roles["Y"]):
            v = np.asarray(vectors[t][i], float)
            s[t] = (np.mean([cos(v, np.asarray(vectors[a][i], float))
                             for a in roles["A"]])
                    - np.mean([cos(v, np.asarray(vectors[b][i], float))
                               for b in roles["B"]]))
        d = s[roles["X"]] - s[roles["Y"]]
        assert math.isclose(deltas[i], d, rel_tol=1e-9, abs_tol=1e-12), i
        sd = np.std([s[roles["X"]], s[roles["Y"]]], ddof=0)
        assert math.isclose(weats[i], d / (sd + 0.001), rel_tol=1e-9), i


def test_scout_run_prompts_reproduces_llama31_run_prompts_exactly():
    """Same frozen prompts, same fake model: identical rows, failures, prototypes."""
    records = [tl.DESIGN[0], tl.DESIGN[100], tl.DESIGN[-1]]
    acc_s, acc_l = core.PrototypeAccumulator(), core.PrototypeAccumulator()
    mine, mf, _ = _scout_run(records, n_blocks=6, acc=acc_s)
    theirs, tf = _llama_run(records, n_blocks=6, acc=acc_l)
    assert len(mine) == len(theirs) == 3 and mf == tf
    for a, b in zip(mine, theirs):
        _rows_equal(a, b)
    assert set(acc_s.sums) == set(acc_l.sums)
    for key in acc_s.sums:
        assert np.array_equal(acc_s.prototype(*key), acc_l.prototype(*key))
        assert acc_s.n(*key) == acc_l.n(*key)
    bad = "nonsense"
    mine, mf, _ = _scout_run(records[:1], response=bad, n_blocks=6)
    theirs, tf, _, _ = tl.run_one(records[0], response=bad, n_blocks=6)
    _rows_equal(mine[0], theirs[0])
    assert mf == tf


def test_scout_hidden_state_and_span_failures_still_abort_the_run():
    record = tl.DESIGN[0]
    tok = tl.FakeTokenizer()
    roles = core.prompt_roles(record, tl.STIM)
    model = tl.FakeModel(n_blocks=4, bad=3, response=tl.perfect_response(record, roles),
                         tokenizer=tok)
    try:
        wat_hidden.run_prompts([record], tl.STIM, _scout_backend(model, tok, _arch(4)),
                               args=tl.FakeArgs(), verbose=False)
    except core.NonFiniteHiddenState as exc:
        assert "index 3" in str(exc)
    else:
        raise AssertionError("a NaN hidden state did not abort the run")
    templated, window = core.build_chat_input(tok, core.SYSTEM_PROMPT, record["prompt"])
    try:
        llama_scout.extract_prompt_vectors(templated, ["not presented"], tok,
                                           tl.FakeModel(n_blocks=4, tokenizer=tok),
                                           device=CPU, architecture=_arch(4),
                                           user_window=window, competitors=[])
    except core.SpanError:
        pass
    else:
        raise AssertionError("an unresolvable span did not abort the run")


# --- 24c. dynamic schema and architecture checks ---------------------------

def test_a_48_block_fake_gives_the_49_level_primary_schema():
    rows, _, meta = _scout_run([tl.DESIGN[0]])
    assert meta["n_hidden_states"] == SCOUT_HIDDEN
    assert list(rows[0]) == core.primary_columns(SCOUT_HIDDEN)
    assert "weat_layer48" in rows[0] and "delta_layer48" in rows[0]
    assert "weat_layer49" not in rows[0]


def test_the_scout_backend_and_runner_hard_code_no_layer_count():
    for label, code in (("llama_scout", SC_CODE), ("wat_hidden", RUNNER_CODE)):
        for n in ("48", "49", "32", "33", "5120", "4096"):
            assert not re.search(rf"\b{n}\b", code), f"literal {n} in {label}"


def test_architecture_validation_rejects_a_wrong_hidden_state_count():
    states = [torch.zeros(1, 3, 8) for _ in range(SCOUT_HIDDEN)]
    llama_scout.check_architecture(states, _arch(SCOUT_BLOCKS))
    _expect_exit(lambda: llama_scout.check_architecture(states[:-1], _arch(SCOUT_BLOCKS)),
                 "48 hidden states", "implies 49")
    _expect_exit(lambda: _scout_run([tl.DESIGN[0]], n_blocks=47,
                                    architecture=_arch(SCOUT_BLOCKS)),
                 "implies 49")


def test_architecture_validation_rejects_a_wrong_hidden_size():
    states = [torch.zeros(1, 3, 8) for _ in range(SCOUT_HIDDEN)]
    states[17] = torch.zeros(1, 3, 6)
    _expect_exit(lambda: llama_scout.check_architecture(states, _arch(SCOUT_BLOCKS)),
                 "hidden state 17", "width 6")
    _expect_exit(lambda: _scout_run([tl.DESIGN[0]], dim=6,
                                    architecture=_arch(SCOUT_BLOCKS, 8)),
                 "hidden_size is 8")


def test_the_secondary_rows_cover_every_level_and_carry_the_scout_version():
    acc = core.PrototypeAccumulator()
    records = tl.by_category(tl.DESIGN, "creativity")[:3] + \
        [r for r in tl.DESIGN if r["domain"] == "fab"][:2]
    _scout_run(records, n_blocks=SCOUT_BLOCKS, dim=6, acc=acc)
    set_rows, stim_rows, _ = wat_hidden.secondary_rows(
        acc, records, tl.STIM, SCOUT_HIDDEN, llama_scout.WAT_ANALYSIS_VERSION)
    assert set_rows and stim_rows
    assert len(set_rows) == 2 * SCOUT_HIDDEN
    for r in set_rows + stim_rows:
        assert r["analysis_version"] == llama_scout.WAT_ANALYSIS_VERSION
        assert r["analysis_version"] != llama_31.WAT_ANALYSIS_VERSION
    assert max(r["hidden_state_index"] for r in set_rows) == SCOUT_BLOCKS
    assert all(r["effect_size_sd"] == "population (ddof=0)" for r in set_rows)


# --- 24d. Scout config, devices, loader -----------------------------------

def test_text_config_of_prefers_the_nested_text_config():
    text = types.SimpleNamespace(num_hidden_layers=48, hidden_size=5120)
    assert llama_scout.text_config_of(types.SimpleNamespace(text_config=text)) is text
    flat = types.SimpleNamespace(num_hidden_layers=32, hidden_size=4096)
    assert llama_scout.text_config_of(flat) is flat
    src = tl._code_only(inspect.getsource(llama_scout.load_scout))
    assert "tcfg = text_config_of(config)" in src
    assert "int(tcfg.num_hidden_layers)" in src and "int(tcfg.hidden_size)" in src
    assert "config.num_hidden_layers" not in src.replace("tcfg.", "")


def test_the_real_scout_config_resolves_to_48_blocks_of_5120():
    """Config JSON only — no weights."""
    if not os.path.isdir(llama_scout.CHECKPOINT):
        print("    (skipped: no Scout checkpoint on this machine)")
        return
    from transformers import AutoConfig
    tcfg = llama_scout.text_config_of(AutoConfig.from_pretrained(llama_scout.CHECKPOINT))
    assert (tcfg.num_hidden_layers, tcfg.hidden_size) == (48, 5120)


def test_the_input_device_is_the_embedding_entry_of_a_multi_gpu_map():
    dmap = dict(SCOUT_DEVICE_MAP, **{"language_model.model.embed_tokens": 1})
    assert llama_scout.resolve_mapped_device(dmap, "language_model.model.embed_tokens") \
        == torch.device("cuda:1")
    assert llama_scout.resolve_mapped_device(SCOUT_DEVICE_MAP,
                                             "language_model.model.layers.40.mlp") \
        == torch.device("cuda:2")
    parent = {"vision_model": 0, "language_model.model": 2,
              "language_model.lm_head": 1}
    assert llama_scout.resolve_mapped_device(parent, "language_model.model.embed_tokens") \
        == torch.device("cuda:2")
    assert llama_scout.resolve_mapped_device({"": "cuda:1"}, "a.b") == torch.device("cuda:1")
    _expect_exit(lambda: llama_scout.resolve_mapped_device(
        {"vision_model": 0}, "language_model.model.embed_tokens"), "not covered")


def test_the_embedding_module_is_found_by_identity_not_first_parameter():
    model = _TinyLlama4(SCOUT_DEVICE_MAP)
    assert llama_scout.embedding_module_name(model) == "language_model.model.embed_tokens"
    first = next(n for n, _ in model.named_parameters())
    assert first.startswith("vision_model"), "fixture must put vision first"
    for mod, name in ((llama_scout, "generate_response"),
                      (llama_scout, "extract_prompt_vectors"),
                      (llama_scout, "load_scout"), (llama_scout, "Backend"),
                      (wat_hidden, "run_prompts"), (wat_hidden, "smoke_test"),
                      (wat_hidden, "main")):
        assert "next(model.parameters())" not in tl._code_only(
            inspect.getsource(getattr(mod, name))), name


def test_a_mapped_device_that_disagrees_with_the_weight_is_refused():
    _expect_exit(lambda: llama_scout.input_embedding_device(_TinyLlama4({"": 0})),
                 "refusing to guess")


def test_cpu_disk_and_meta_placements_are_rejected():
    assert llama_scout.validate_device_map(SCOUT_DEVICE_MAP) == \
        ["cuda:0", "cuda:1", "cuda:2"]
    for bad in ("cpu", "disk", "meta"):
        dmap = dict(SCOUT_DEVICE_MAP, **{"language_model.model.layers.47": bad})
        _expect_exit(lambda: llama_scout.validate_device_map(dmap), "off-GPU", bad)
        _expect_exit(lambda: llama_scout.resolve_mapped_device(
            {"language_model.model": bad}, "language_model.model.embed_tokens"),
            bad)


def test_an_empty_or_missing_device_map_is_rejected():
    _expect_exit(lambda: llama_scout.validate_device_map({}), "no hf_device_map")
    _expect_exit(lambda: llama_scout.validate_device_map(None), "no hf_device_map")
    model = _TinyLlama4(None)
    _expect_exit(lambda: llama_scout.input_embedding_device(model), "not covered")


def test_actual_parameter_devices_are_validated_too():
    _expect_exit(lambda: llama_scout.validate_parameter_devices(torch.nn.Linear(2, 2)),
                 "cpu")
    _expect_exit(lambda: llama_scout.validate_parameter_devices(
        torch.nn.Linear(2, 2, device="meta")), "meta")
    src = tl._code_only(inspect.getsource(llama_scout.load_scout))
    assert "validate_parameter_devices(model)" in src


def test_the_loader_is_the_verified_bf16_sdpa_auto_load():
    src = tl._code_only(inspect.getsource(llama_scout.load_scout))
    assert "Llama4ForConditionalGeneration.from_pretrained(" in src
    assert 'device_map="auto"' in src and "low_cpu_mem_usage=True" in src
    assert "attn_implementation=ATTN_IMPLEMENTATION" in src
    assert llama_scout.ATTN_IMPLEMENTATION == "sdpa"
    assert 'dtype="bfloat16"' in inspect.getsource(llama_scout.load_scout).split(")")[0]
    assert "validate_device_map(device_map)" in src and "model.eval()" in src
    assert not re.search(r"max_memory\s*=", SC_CODE)
    for banned in ("offload_folder", "offload_state_dict", "load_in_4bit",
                   "load_in_8bit", "BitsAndBytesConfig", "quantization_config",
                   "bitsandbytes"):
        assert banned not in SC_CODE, banned


def test_the_scout_backend_is_text_only():
    for banned in ("pixel_values=", "AutoProcessor", "processor(", "<|image|>",
                   "images="):
        assert banned not in SC_CODE, banned


# --- 24e. no legacy path --------------------------------------------------

def test_no_legacy_scout_execution_path_remains():
    for banned in ("load_sessions", "session_D", "session_arms", "ext.DATASETS",
                   "ext.scoped", "jsonl", "behavioural_model",
                   "SCOUT_BEHAVIOURAL_MODELS", "_assigned_items",
                   "system_prompt_for", "preflight", "ollama", "Ollama",
                   "load_backend", "parameter_footprint", "execution_device",
                   "_gpu_budget", "_cpu_budget", "QUANTISATIONS",
                   "nullify", "top_components", "max_valid_k", "pca", "PCA",
                   "trial_layer_rows", "aggregate_pairs", "layer_correlations",
                   "layer_association", "extract_prompt_hidden_states"):
        assert banned not in SC_CODE, f"{banned!r} is still in the Scout backend"
        assert not hasattr(llama_scout, banned), f"llama_scout.{banned} survived"


def test_scout_writes_the_wat_output_set_to_its_own_directory():
    main_src = tl._code_only(inspect.getsource(wat_hidden.main))
    assert "PRIMARY_CSV" in main_src
    for fname in wat_hidden.OUTPUT_FILES:
        if fname != wat_hidden.PRIMARY_CSV:
            assert f'"{fname}"' in main_src, fname
    assert os.path.basename(llama_scout.WAT_OUT_DIR) == "embed_scout"
    assert llama_scout.WAT_OUT_DIR != llama_31.WAT_OUT_DIR
    legacy = {"hidden_trial_layer.csv", "hidden_pair_layer_aggregated.csv",
              "hidden_layer_correlations.csv", "hidden_nullified_layer.csv",
              "hidden_manifest.json"}
    assert legacy.isdisjoint(wat_hidden.OUTPUT_FILES)


def test_dry_run_loads_no_tokenizer_or_model():
    src = tl._code_only(inspect.getsource(wat_hidden.main))
    dry = src.index("if args.dry_run:")
    load = src.index("backend_module.load(")
    assert dry < load
    assert "AutoTokenizer" not in src[:load]


# --- 24f. manifest and BOS -------------------------------------------------

def test_the_scout_manifest_records_what_the_spec_requires():
    m = _fake_manifest()
    for key in ("analysis_version", "checkpoint", "model_class", "tokenizer",
                "transformers_version", "torch_version", "dtype", "hf_device_map",
                "devices", "input_embedding_module", "input_device",
                "attn_implementation", "quantisation", "cpu_offload",
                "disk_offload", "hidden_size", "n_transformer_blocks",
                "n_hidden_states", "layer_convention", "modality", "pooling_rule",
                "design", "generation", "generation_seed_rule", "bos",
                "orientation", "behavioural_metric", "fail_fast",
                "analysis_logic_source"):
        assert key in m, key
    assert m["analysis_version"] == llama_scout.WAT_ANALYSIS_VERSION
    assert m["hf_device_map"] == _fake_backend()["hf_device_map"]
    assert m["attn_implementation"] == "sdpa" and m["quantisation"] == "none"
    assert m["cpu_offload"] is False and m["disk_offload"] is False
    assert m["n_hidden_states"] == SCOUT_HIDDEN
    assert m["layer_convention"]["n_transformer_blocks"] == SCOUT_BLOCKS
    assert m["generation"] == core.GENERATION
    assert m["design"]["prompts_sha256"] == design.file_digest(design.PROMPTS_CSV)
    assert "architecture_mismatch" in m["fail_fast"]
    src = m["analysis_logic_source"]
    assert src["file"] == "embedding/core.py"
    assert src["sha256"] == design.file_digest(core.__file__)


def test_the_scout_manifest_carries_no_stale_llama31_description():
    m = _fake_manifest()
    m.pop("analysis_logic_source")                    # explicit provenance only
    text = json.dumps(m, default=str)
    for stale in ("Llama-3.1", "Llama 3.1", "llama3.1", "llama31", "Llama3",
                  "/home/syryu", llama_31.WAT_ANALYSIS_VERSION, llama_31.CHECKPOINT,
                  "3420-prompt"):
        assert stale not in text, stale


def test_the_bos_record_counts_template_and_tokenizer_bos_separately():
    class _BosTok:
        bos_token, bos_token_id = "<s>", 1

        def __call__(self, text, **kw):
            body = text[len("<s>"):] if text.startswith("<s>") else text
            return {"input_ids": [1, 1] + [5] * len(body.split())}

    rec = core.bos_record(_BosTok(), "<s>hello there")
    assert rec["bos_in_templated_text"] == 1 and rec["bos_in_model_input"] == 2
    assert "kept deliberately" in rec["decision"]
    for backend in (llama_31, llama_scout):
        for name in ("generate_response", "extract_prompt_vectors"):
            assert "add_special_tokens" not in inspect.getsource(getattr(backend, name))


def _scout_tokenizer():
    if not os.path.isfile(os.path.join(llama_scout.CHECKPOINT, "tokenizer.json")):
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(llama_scout.CHECKPOINT)


def test_the_real_scout_tokenizer_feeds_the_model_two_bos_tokens():
    """Tokenizer only.  Same double BOS as the completed Llama 3.1 run."""
    tok = _scout_tokenizer()
    if tok is None:
        print("    (skipped: no Scout tokenizer on this machine)")
        return
    templated, _ = core.build_chat_input(tok, core.SYSTEM_PROMPT, tl.DESIGN[0]["prompt"])
    rec = core.bos_record(tok, templated)
    assert rec["bos_token"] == "<|begin_of_text|>"
    assert rec["bos_in_templated_text"] == 1
    assert rec["bos_in_model_input"] == 2
    plain = tok(templated, add_special_tokens=False)["input_ids"]
    assert sum(1 for i in plain[:3] if i == tok.bos_token_id) == 1
    assert tok.is_fast


def test_every_frozen_stimulus_resolves_under_the_real_scout_tokenizer():
    tok = _scout_tokenizer()
    if tok is None:
        print("    (skipped: no Scout tokenizer on this machine)")
        return
    lo, hi, n_bos = core.validate_all_spans(tok, tl.ON_DISK, tl.STIM)
    assert n_bos == [2]
    assert 0 < lo <= hi


# --- 24g. package architecture --------------------------------------------

def _imported_modules(path):
    """Every module name a file imports (absolute imports, any depth)."""
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def test_no_backend_imports_another_backend():
    names = {"llama_31", "llama_scout", "openai"}
    for mod in (llama_31, llama_scout, backend_openai):
        own = mod.__name__.rsplit(".", 1)[1]
        imported = _imported_modules(mod.__file__)
        for other in names - {own}:
            assert not any(i.endswith(f"backends.{other}") for i in imported), \
                (own, other)


def test_core_and_design_import_no_backend_and_runners_only_the_registry():
    for mod in (core, design):
        assert not any("backends" in i for i in _imported_modules(mod.__file__)), \
            mod.__name__
    from embedding import chained_hidden
    for mod in (wat_hidden, chained_hidden):
        imported = _imported_modules(mod.__file__)
        assert not any(i.endswith(("llama_31", "llama_scout"))
                       for i in imported), mod.__name__


def test_importing_core_or_a_runner_loads_no_transformers_or_weights():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for module in ("embedding.core", "embedding.design", "embedding.wat_hidden",
                   "embedding.chained_hidden", "embedding.backends.llama_scout",
                   "embedding.backends.llama_31"):
        code = (f"import sys; sys.path.insert(0, {root!r}); import {module}; "
                f"assert 'transformers' not in sys.modules, 'transformers'; "
                f"assert 'torch' not in sys.modules, 'torch'; "
                f"assert 'openai' not in sys.modules, 'openai SDK'")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, (module, out.stderr[-600:])


def test_the_openai_backend_is_a_clear_placeholder():
    from embedding import backends
    assert "openai" in backends.names() and "openai" not in backends.implemented()
    try:
        backends.get_backend("openai")
    except NotImplementedError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("the openai placeholder was selectable")
    try:
        backend_openai.load()
    except NotImplementedError:
        return
    raise AssertionError("the openai placeholder loaded")


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
