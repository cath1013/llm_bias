"""
Tests for the embedding-level analyses.

    python3 embedding/test_embed_bias.py            # all tests
    python3 embedding/test_embed_bias.py weat       # only tests whose name contains "weat"

No pytest in this environment, so this is a plain runner: every `test_*` function
either returns (pass) or raises (fail).  Nothing here touches the OpenAI API, the
Ollama server, or any behavioural runner — the embedding client is faked.
"""

import os
import re
import sys
import json
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import embed_bias as eb
import embed_bias_scout_hidden as sc
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
    rho, p, n = eb.spearman(x, y)
    assert abs(rho - 1.0) < 1e-12, rho
    assert n == 5
    r, _, _ = eb.pearson(x, y)
    assert r < 0.99, f"Pearson should be below 1 on this curve, got {r}"

    # hand-checked against scipy.stats.spearmanr for a set with ties
    rho2, _, _ = eb.spearman([1, 2, 2, 3], [10, 20, 30, 40])
    assert abs(rho2 - 0.9486832980505138) < 1e-9, rho2

    ranks = eb.rankdata([10, 20, 20, 30])
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0], ranks


def test_t_pvalue_matches_numeric_integration():
    for t, df, expect in [(2.0, 10, 0.0733880), (3.5, 28, 0.0015765),
                          (0.5, 100, 0.6181736)]:
        got = eb.t_sf2(t, df)
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
    assert eb.weat_test_stat(X, Y, A, B) > 0
    assert eb.weat_effect_size(X, Y, A, B) > 0
    # and the incongruent configuration flips
    assert eb.weat_effect_size(Y, X, A, B) < 0

    p, n, exact = eb.weat_p_value(X, Y, A, B)
    assert exact and n == math.comb(6, 3)
    assert p < 0.2, p


def test_weat_effect_size_is_scale_free_but_test_stat_is_not():
    X, Y = _axis_vectors([0.9, 0.8]), _axis_vectors([-0.9, -0.8])
    A, B = _axis_vectors([1.0, 0.9]), _axis_vectors([-1.0, -0.9])
    big = eb.weat_test_stat(X, Y, A, B)
    small = eb.weat_test_stat(_axis_vectors([0.2, 0.1]), _axis_vectors([-0.2, -0.1]),
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

    hi = eb.wefat_score(near_brilliance, A, B)
    mid = eb.wefat_score(neutral, A, B)
    lo = eb.wefat_score(near_effort, A, B)
    assert hi > mid > lo, (hi, mid, lo)
    assert hi > 0 > lo


# =========================================================
# 3. the singleton problem
# =========================================================

def test_singleton_standardized_score_saturates_but_delta_does_not():
    A, B = _axis_vectors([1.0, 0.9]), _axis_vectors([-1.0, -0.9])
    strong = (_axis_vectors([0.9]), _axis_vectors([-0.9]))
    weak = (_axis_vectors([0.05]), _axis_vectors([-0.05]))

    s_sing = eb.bai_singleton_weat(*strong, A, B)
    w_sing = eb.bai_singleton_weat(*weak, A, B)
    s_delta = eb.weat_test_stat(*strong, A, B)
    w_delta = eb.weat_test_stat(*weak, A, B)

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
    spec = eb.DATASETS["fab"]
    arms = {k: v for k, v in eb.fab_arms().items() if k == "leslie@tertile"}
    sessions = _fab_sessions([("innate gift", "hard work", 7),
                              ("raw ability", "motivation", 5),
                              ("innate talent", "dedication", 3)])
    scores = {w: 0.6 for w in wat_fab.BRILLIANCE_WORDS}
    scores.update({w: -0.6 for w in wat_fab.EFFORT_WORDS})
    scores.update({f: (0.5 if f in wat_fab.HIGH_TERT else -0.5)
                   for f in wat_fab.FIELD_LABELS})
    emb = FakeEmbedder(scores)

    raw, agg, corr = [], [], []
    eb.run_pair_level("fab", spec, arms, emb, sessions, raw, agg, corr)

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
    spec = eb.DATASETS["fab"]
    arms = {k: v for k, v in eb.fab_arms().items() if k == "leslie@tertile"}
    sessions = _fab_sessions([("innate gift", "hard work", 4)])
    emb = FakeEmbedder({w: 0.5 for w in wat_fab.BRILLIANCE_WORDS})
    raw, agg, corr = [], [], []
    eb.run_pair_level("fab", spec, arms, emb, sessions, raw, agg, corr)
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
    spec = eb.DATASETS["fab"]
    sessions = _fab_sessions([("innate gift", "hard work", 4)])
    scores = {f: wat_fab.FAB_SCORE[f] - 4.0 for f in wat_fab.FIELD_LABELS}
    for bri, eff in wat_fab.POLE_SETS.values():      # both pole sets, both arms
        scores.update({w: 0.8 for w in bri})
        scores.update({w: -0.8 for w in eff})
    emb = FakeEmbedder(scores)

    fields, corr = [], []
    eb.run_fab_wefat(spec, emb, sessions, fields, corr)

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
    spec = eb.DATASETS["fab"]
    sessions = _fab_sessions([("innate gift", "hard work", 2)])
    for s in sessions[:1]:
        s["model"] = "model-a"
    for s in sessions[1:]:
        s["model"] = "model-b"
    emb = FakeEmbedder({f: 0.1 for f in wat_fab.FIELD_LABELS})
    fields, corr = [], []
    eb.run_fab_wefat(spec, emb, sessions, fields, corr)
    per_model = [c for c in corr if c["level"] == "fab_field_wefat_vs_model_P"]
    models = {c["behavioural_model"] for c in per_model}
    assert models == {"model-a", "model-b"}, models
    assert all(c["n_fields"] <= 30 for c in per_model)
    assert all(c["behavioural_model"] is not None for c in per_model)


# =========================================================
# 6. gender stimuli are the existing ones, untouched
# =========================================================

def test_gb_arms_use_the_exact_runner_stimuli():
    arms = eb.gb_arms()
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
    arms = eb.fab_arms()
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
    spec = eb.DATASETS["fab"]
    arm = eb.fab_arms()["leslie@tertile"]
    rec = dict(assignments={**{f: "brilliance" for f in arm["A"]},
                            **{f: "effort" for f in arm["B"]}},
               wat_D_fab_tertile=-999)          # a stored score that must be ignored
    assert eb.session_D(rec, spec, arm) == 1.0
    rec2 = dict(assignments={**{f: "effort" for f in arm["A"]},
                             **{f: "brilliance" for f in arm["B"]}})
    assert eb.session_D(rec2, spec, arm) == -1.0
    # a JSON-string assignments blob (as some runners write it) parses identically
    rec3 = dict(assignments=json.dumps(rec["assignments"]))
    assert eb.session_D(rec3, spec, arm) == 1.0


# =========================================================
# 7. no race code path remains
# =========================================================

def test_no_race_dataset_or_code_path():
    assert set(eb.DATASETS) == {"fab", "gb"}
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("embed_bias.py", "embed_bias_scout_hidden.py"):
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

    span = sc.resolve_span(prompt, offsets, "music composition")
    lo, hi = span
    covered = prompt[offsets[lo][0]:offsets[hi - 1][1]]
    # token-granular, so punctuation glued to the phrase may ride along, but the
    # span must be tight around it and must cover the whole phrase
    assert "music composition" in covered, covered
    assert covered.strip("',[]. ") == "music composition", covered
    assert hi - lo > 1, "the phrase must span several tokens"

    # pooling covers the whole span, and is not the first token
    hidden = np.arange(len(offsets) * 2, dtype=float).reshape(len(offsets), 2)
    mean_vec = sc.pool_span(hidden, span, "mean")
    last_vec = sc.pool_span(hidden, span, "last")
    assert np.allclose(mean_vec, hidden[lo:hi].mean(axis=0))
    assert np.allclose(last_vec, hidden[hi - 1])
    assert not np.allclose(mean_vec, hidden[lo]), "mean must not equal first token"


def test_scout_span_location_fails_loudly_when_ambiguous_or_absent():
    prompt = ("Here is a list of words. For each word pick a word — special "
              "aptitude or hard work — and write it after the word. The words "
              "are ['philosophy', 'special aptitude for maths'].")
    offsets = MockTokenizerOffsets().encode_with_offsets(prompt)

    try:
        sc.resolve_span(prompt, offsets, "special aptitude")
    except sc.SpanError as exc:
        assert "occurs 2 times" in str(exc), exc
    else:
        raise AssertionError("ambiguous phrase did not raise")

    # a window disambiguates deliberately
    cut = prompt.index("The words are")
    span = sc.resolve_span(prompt, offsets, "special aptitude", window=(cut, len(prompt)))
    assert offsets[span[0]][0] >= cut - 4, "window was ignored"
    covered = prompt[offsets[span[0]][0]:offsets[span[1] - 1][1]]
    assert "special aptitude" in covered, covered

    try:
        sc.resolve_span(prompt, offsets, "quantum basket weaving")
    except sc.SpanError as exc:
        assert "does not occur" in str(exc)
    else:
        raise AssertionError("missing phrase did not raise")


# =========================================================
# 9. one row per returned hidden state
# =========================================================

def test_hidden_rows_per_trial_equal_the_number_of_hidden_states():
    arm = dict(A=["genius", "brainiac"], B=["kindly", "chatty"],
               X=["man"], Y=["woman"])
    rng = np.random.default_rng(0)
    for n_layers in (5, 17, 49):          # never a hardcoded 33
        vectors = {p: [rng.normal(size=8) for _ in range(n_layers)]
                   for p in ["man", "woman"] + arm["A"] + arm["B"]}
        rows = sc.trial_layer_rows(vectors, arm, "man", "woman", 0.5,
                                   n_layers=n_layers, ks=(0,))
        assert len(rows) == n_layers, (n_layers, len(rows))
        assert [r["layer"] for r in rows] == list(range(n_layers))
        assert all(r["pca_k"] == 0 and r["pooling"] == "mean" for r in rows)

        rows_k = sc.trial_layer_rows(vectors, arm, "man", "woman", 0.5,
                                     n_layers=n_layers, ks=(0, 2))
        assert len(rows_k) == 2 * n_layers


def test_hidden_layer_correlations_aggregate_pairs_first():
    rows = []
    for layer in range(3):
        for pair, delta, ds in [("a || b", 0.3, [0.4, 0.6]),
                                ("c || d", -0.2, [-0.3, -0.1]),
                                ("e || f", 0.0, [0.05])]:
            x, y = pair.split(" || ")
            for i, d in enumerate(ds):
                rows.append(dict(dataset="gb", arm="warmth",
                                 behavioural_model="scout-ctx", layer=layer,
                                 pair=pair, X_stimulus=x, Y_stimulus=y,
                                 delta=delta, bai_singleton_weat=2.0,
                                 behavioural_D=d))
    agg = sc.aggregate_pairs(rows)
    assert len(agg) == 9                       # 3 layers x 3 unique pairs
    assert all(a["n_sessions"] in (1, 2) for a in agg)
    corr = sc.layer_correlations(agg, checkpoint="fake-scout", pooling="mean")
    assert len(corr) == 3                      # one row per layer
    assert all(c["n_unique_pairs"] == 3 for c in corr)
    assert all(c["level"] == "scout_hidden_delta_vs_behavioural_D" for c in corr)
    assert all(c["pca_k"] == 0 for c in corr)


# =========================================================
# 10. PCA nullification
# =========================================================

def test_pca_k0_is_an_exact_identity_and_k2_removes_two_directions():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(12, 6))
    assert np.array_equal(sc.nullify(V, 0), V)

    out = sc.nullify(V, 2)
    assert out.shape == V.shape
    # the centred residual must be orthogonal to the two removed directions
    centred = V - V.mean(axis=0)
    comps = sc.top_components(centred, 2)
    assert np.allclose(out @ comps.T, 0, atol=1e-8), "top-2 directions remain"
    assert not np.allclose(out, V)


def test_pca_nullification_does_not_change_the_primary_statistic():
    """k=0 must reproduce the unaltered association exactly."""
    rng = np.random.default_rng(2)
    stim = rng.normal(size=(6, 5))
    A, B = stim[2:4], stim[4:6]
    raw = sc.layer_association(stim[0], stim[1], A, B)
    nulled = sc.nullify(stim, 0)
    same = sc.layer_association(nulled[0], nulled[1], nulled[2:4], nulled[4:6])
    assert raw == same, (raw, same)


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
    spec = eb.DATASETS["fab"]
    sessions = (_fab_sessions_with_poles("leslie", "model-a", "brilliance", "effort")
                + _fab_sessions_with_poles("bian", "model-a", "effort", "brilliance"))
    scores = {f: wat_fab.FAB_SCORE[f] - 4.0 for f in wat_fab.FIELD_LABELS}
    for bri, eff in wat_fab.POLE_SETS.values():
        scores.update({w: 0.8 for w in bri})
        scores.update({w: -0.8 for w in eff})
    fields, corr = [], []
    eb.run_fab_wefat(spec, FakeEmbedder(scores), sessions, fields, corr)

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
    spec = eb.DATASETS["gb"]
    arms = {k: v for k, v in eb.gb_arms().items() if k == "warmth"}
    pairs = list(zip(wgb.MALE_WORDS[:3], wgb.FEMALE_WORDS[:3]))
    sessions = _gb_sessions(["m1", "m2", "m3"], pairs, "warmth")
    emb = FakeEmbedder({w: 0.4 for w in wgb.MALE_WORDS})

    raw, agg, corr = [], [], []
    eb.run_pair_level("gb", spec, arms, emb, sessions, raw, agg, corr)

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
    spec = eb.DATASETS["gb"]
    arms = {k: v for k, v in eb.gb_arms().items() if k in ("warmth", "beauty")}
    pairs = list(zip(wgb.MALE_WORDS[:3], wgb.FEMALE_WORDS[:3]))
    sessions = (_gb_sessions(["m1"], pairs, "warmth")
                + _gb_sessions(["m1"], pairs, "beauty"))
    emb = FakeEmbedder({w: 0.4 for w in wgb.MALE_WORDS})

    raw, agg, corr = [], [], []
    eb.run_pair_level("gb", spec, arms, emb, sessions, raw, agg, corr)

    pooled = [c for c in corr if c["level"].endswith("@arms_pooled")]
    assert pooled, "the cross-arm row should still be reported, descriptively"
    for c in pooled:
        assert c["inferential"] is False
        assert c["analysis_role"] == "descriptive_cross_condition"
        assert c["spearman_p"] == "" and c["pearson_p"] == ""
        assert c["behavioural_model"] != "(all models)"

    # and Holm never touches a descriptive row
    eb.apply_holm(corr, key_of=lambda r: (None if not r["inferential"]
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
    got = eb.holm([0.01, 0.04, 0.03])
    assert [round(v, 12) for v in got] == [0.03, 0.06, 0.06], got
    # R: p.adjust(c(0.005, 0.011, 0.02, 0.3)) -> 0.020 0.033 0.040 0.300
    got = eb.holm([0.005, 0.011, 0.02, 0.3])
    assert [round(v, 12) for v in got] == [0.02, 0.033, 0.04, 0.3], got
    # monotone: a later p can never get a smaller adjusted value
    # R: p.adjust(c(0.2, 0.3, 0.4), method = "holm") -> 0.6 0.6 0.6
    adj = eb.holm([0.2, 0.3, 0.4])
    assert all(abs(a - 0.6) < 1e-12 for a in adj), adj
    assert all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))
    # capped at 1: 2 * 0.5 would be 1.0, 2 * 0.6 would exceed it
    assert eb.holm([0.5, 0.6]) == [1.0, 1.0]
    # NaNs stay missing and do not count towards the family size
    got = eb.holm([0.01, float("nan"), 0.04])
    assert got[1] != got[1]
    assert [round(got[0], 12), round(got[2], 12)] == [0.02, 0.04], got
    # order preserved: the largest raw p stays the largest adjusted p
    ps = [0.5, 0.001, 0.02, 0.2]
    adj = eb.holm(ps)
    assert sorted(range(4), key=lambda i: ps[i]) == sorted(range(4), key=lambda i: adj[i])
    # a single finite p is unchanged
    assert eb.holm([0.037])[0] == 0.037


def test_holm_families_keep_fab_and_gb_set_level_separate():
    rows = [dict(dataset="fab", emb_model="e", context="prompt", perm_p=0.01),
            dict(dataset="fab", emb_model="e", context="prompt", perm_p=0.20),
            dict(dataset="gb", emb_model="e", context="prompt", perm_p=0.01),
            dict(dataset="gb", emb_model="e", context="prompt", perm_p=0.20)]
    eb.apply_holm(rows,
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
# 14. Scout delta_sd is a true SD
# =========================================================

def test_scout_delta_sd_is_a_sample_sd_not_a_range():
    rows = [dict(dataset="fab", arm="a", behavioural_model="m", layer=0,
                 pair="x || y", X_stimulus="x", Y_stimulus="y", delta=d,
                 bai_singleton_weat=0.0, behavioural_D=0.1)
            for d in (0.1, 0.3, 0.5)]
    agg = sc.aggregate_pairs(rows)
    assert len(agg) == 1
    a = agg[0]
    assert abs(a["delta_mean"] - 0.3) < 1e-12
    assert abs(a["delta_sd"] - 0.2) < 1e-12          # sample SD (ddof=1)
    assert abs(a["delta_range"] - 0.4) < 1e-12       # max - min
    assert a["delta_sd"] != a["delta_range"]
    assert a["delta"] == a["delta_mean"]             # correlated value is the mean
    assert a["n_sessions"] == 3
    # a single session has no sample SD
    one = sc.aggregate_pairs(rows[:1])[0]
    assert one["delta_sd"] != one["delta_sd"]        # NaN
    assert one["delta_range"] == 0.0


# =========================================================
# 15. centred-PCA maximum k
# =========================================================

def test_centred_pca_rank_allows_n_minus_one_and_rejects_n():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(10, 64))
    assert sc.max_valid_k(10, 64) == 9
    out = sc.nullify(V, 9)
    assert out.shape == V.shape
    # removing every available direction annihilates the centred rows
    assert float(np.abs(out).max()) < 1e-8
    try:
        sc.nullify(V, 10)
    except ValueError as exc:
        msg = str(exc)
        assert "10" in msg and "9" in msg and "64" in msg, msg
    else:
        raise AssertionError("k = n_vectors must be rejected, not truncated")
    # k is bounded by the embedding dimension too
    assert sc.max_valid_k(10, 4) == 4
    # k = 0 remains an exact identity
    assert np.array_equal(sc.nullify(V, 0), V)


# =========================================================
# 16. reported analysis scope
# =========================================================

def test_reported_scope_is_four_gb_controls_and_the_fab_tertile_split():
    gb = eb.scoped("gb", eb.gb_arms())
    assert set(gb) == {"humor", "creativity4", "warmth", "dedication_adj"}, set(gb)
    # "creativity with original" is the 4-word set: it is the one containing it
    assert "original" in wgb.CONTROL_SETS["creativity4"]
    assert "original" not in wgb.CONTROL_SETS["creativity"]

    fab = eb.scoped("fab", eb.fab_arms())
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
    assert set(eb.gb_arms()) == set(wgb.CONTROL_SETS)
    assert len(eb.fab_arms()) == 4


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
    system = sc.system_prompt_for("gb")
    # taken from the runner, never hardcoded in the analysis module
    assert system == wgb.SYSTEM_PROMPT == "You are a helpful assistant."
    assert sc.system_prompt_for("fab") == wat_fab.SYSTEM_PROMPT

    text, window = sc.build_chat_input(tok, system, user)
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
    text, window = sc.build_chat_input(tok, "You are a helpful assistant.", user)

    offsets = tok.encode_with_offsets(text)
    # unrestricted, 'assistant' is ambiguous -> loud failure, never a guess
    try:
        sc.resolve_span(text, offsets, "assistant")
    except sc.SpanError:
        pass
    else:
        raise AssertionError("ambiguous phrase must raise without a window")

    # confined to the user window it resolves to the list occurrence
    lo, hi = sc.resolve_span(text, offsets, "assistant", window=window)
    covered = "".join(text[s:e] for s, e in offsets[lo:hi])
    assert "assistant" in covered
    assert text.index("assistant", window[0]) >= window[0]

    # and that is the window extract_prompt_hidden_states passes by default
    import inspect
    sig = inspect.signature(sc.extract_prompt_hidden_states)
    assert "default_window" in sig.parameters


def test_scout_refuses_a_template_that_mangles_the_saved_prompt():
    class Mangling(FakeChatTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            # a template that normalises whitespace breaks every char offset
            return " ".join(m["content"].replace("—", "-") for m in messages)

    user = "pick a word — female or father — and write it"
    try:
        sc.build_chat_input(Mangling(), "You are a helpful assistant.", user)
    except sc.SpanError as exc:
        assert "verbatim" in str(exc)
    else:
        raise AssertionError("a template that rewrites the prompt must raise")

    class Duplicating(FakeChatTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            u = messages[-1]["content"]
            return u + "\n" + u

    try:
        sc.build_chat_input(Duplicating(), "sys", user)
    except sc.SpanError as exc:
        assert "twice" in str(exc)
    else:
        raise AssertionError("a duplicated user message must raise")


def test_scout_quantisation_choices_are_declared():
    assert sc.QUANTISATIONS == ("bf16", "4bit", "8bit")
    assert sc.QUANTISATIONS[0] == "bf16", "bf16 must be the default/first choice"
    # the loader must refuse an unknown quantisation before touching the weights
    try:
        sc.load_backend("/nonexistent/checkpoint", quant="int3")
    except (ValueError, ImportError, ModuleNotFoundError) as exc:
        if isinstance(exc, ValueError):
            assert "int3" in str(exc)
    else:
        raise AssertionError("unknown quantisation must not be accepted")


# =========================================================
# 18. sessions are matched to arms by WORDS, not by label
# =========================================================

def test_creativity4_variants_are_not_pooled_by_label():
    """
    The raw tree labels five different fourth words `creativity4`.  A session
    whose control words are the `inventive` variant must NOT be scored against
    the `original` stimulus set just because the label matches.
    """
    arms = eb.gb_arms()
    c4 = arms["creativity4"]
    assert "original" in c4["B"] and "inventive" not in c4["B"]

    def session(fourth_word):
        a = {w: "male" for w in c4["A"]}
        a.update({w: "female" for w in
                  ["creative", "artistic", "super-imaginative", fourth_word]})
        return dict(model="m", trial=1, male_word=wgb.MALE_WORDS[0],
                    female_word=wgb.FEMALE_WORDS[0], control_set="creativity4",
                    assignments=a, _file="synthetic.jsonl")

    spec = eb.DATASETS["gb"]
    assert spec["session_arms"](session("original"), arms) == ["creativity4"]
    for other in ("inventive", "unique", "resourceful", "visionary"):
        assert spec["session_arms"](session(other), arms) == [], other

    # the stimulus words themselves are untouched by any of this
    assert c4["B"] == list(wgb.CONTROL_SETS["creativity4"])


def test_incomplete_sessions_are_excluded_rather_than_scored_on_a_subset():
    arms = {k: v for k, v in eb.gb_arms().items() if k == "warmth"}
    arm = arms["warmth"]
    full = {w: "male" for w in arm["A"]}
    full.update({w: "female" for w in arm["B"]})
    ok = dict(model="m", trial=1, male_word=wgb.MALE_WORDS[0],
              female_word=wgb.FEMALE_WORDS[0], control_set="warmth",
              assignments=full, _file="s.jsonl")
    partial = dict(ok, assignments={k: v for k, v in list(full.items())[:5]})

    spec = eb.DATASETS["gb"]
    assert spec["session_arms"](ok, arms) == ["warmth"]
    assert spec["session_arms"](partial, arms) == []
    assert eb.arm_is_present(ok, arm) and not eb.arm_is_present(partial, arm)


def test_the_three_word_creativity_subset_does_not_swallow_creativity4():
    arms = eb.gb_arms()
    c3, c4 = arms["creativity"], arms["creativity4"]
    assert set(c3["A"]) | set(c3["B"]) < set(c4["A"]) | set(c4["B"]), \
        "this test exists because creativity is a strict subset of creativity4"
    a = {w: "male" for w in c4["A"]}
    a.update({w: "female" for w in c4["B"]})
    r = dict(model="m", assignments=a, control_set="creativity4",
             male_word=wgb.MALE_WORDS[0], female_word=wgb.FEMALE_WORDS[0])
    # exact matching: the creativity4 session belongs to creativity4 ONLY
    assert eb.DATASETS["gb"]["session_arms"](r, arms) == ["creativity4"]
    assert not eb.arm_is_present(r, c3, "exact")
    assert eb.arm_is_present(r, c3, "superset")      # which is why gb is exact


def test_fab_sessions_sort_all_thirty_fields_so_arms_use_superset_matching():
    arms = eb.scoped("fab", eb.fab_arms())
    arm = arms["leslie@tertile"]
    assert len(set(arm["A"]) | set(arm["B"])) == 20     # one split, not all 30
    sess = _fab_sessions([("innate gift", "hard work", 1)])[0]
    assert len(eb._assigned_items(sess)) == 30
    assert eb.DATASETS["fab"]["session_arms"](sess, arms) == ["leslie@tertile"]
    assert not eb.arm_is_present(sess, arm, "exact")    # 30 != 20
    assert eb.arm_is_present(sess, arm, "superset")


def test_workbooks_are_the_declared_source_of_record():
    for ds, book in (("fab", "llm bias fab.xlsx"), ("gb", "llm bias gender.xlsx")):
        path = eb.DATASETS[ds]["workbook"]
        assert os.path.basename(path) == book
        assert os.path.exists(path), path


# =========================================================
# runner
# =========================================================

# =========================================================
# 22. 4-bit does not shrink a Llama-4 mixture of experts
# =========================================================

def _tiny_llama4_config(num_local_experts=4):
    """A Llama-4 the size of a postage stamp, same module TYPES as Scout."""
    from transformers import Llama4Config, Llama4TextConfig, Llama4VisionConfig
    text = Llama4TextConfig(
        hidden_size=64, intermediate_size=64, intermediate_size_mlp=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=256, num_local_experts=num_local_experts,
        num_experts_per_tok=1, interleave_moe_layer_step=1)
    vision = Llama4VisionConfig(hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=2,
                                image_size=16, patch_size=8)
    return Llama4Config(text_config=text, vision_config=vision)


def test_four_bit_leaves_the_llama4_experts_at_full_precision():
    """
    bitsandbytes converts nn.Linear MODULES.  Llama-4 keeps its experts as 3-D
    nn.Parameter stacks, so `load_in_4bit` cannot touch them -- which is why a
    4-bit load of Scout asks for ~187 GiB, not ~55 GiB.  Measure it rather than
    asserting it: more experts must mean a smaller shrinkable fraction.
    """
    few = sc.parameter_footprint(_tiny_llama4_config(2), quant="4bit")
    many = sc.parameter_footprint(_tiny_llama4_config(16), quant="4bit")

    grew_outside = many["params_outside_linear"] - few["params_outside_linear"]
    grew_inside = many["params_in_linear"] - few["params_in_linear"]
    assert grew_outside > 0, "adding experts must add parameters outside nn.Linear"
    # the router IS an nn.Linear whose output width is the expert count, so
    # nn.Linear grows a little too; the expert stacks must dominate it utterly
    assert grew_outside > 100 * grew_inside, (grew_outside, grew_inside)
    assert many["shrinkable_fraction"] < few["shrinkable_fraction"]
    assert many["shrinkable_fraction"] < 0.5, (
        "most of a 16-expert Llama-4 must be unquantisable by bitsandbytes; "
        f"got {many['shrinkable_fraction']:.3f}")
    # and the footprint must reflect that, not the nominal 1/4 of bf16
    assert many["footprint_gib"] > 0.5 * many["full_precision_gib"]


def test_a_dense_model_is_almost_entirely_shrinkable():
    """The contrast case: the measurement is not a Llama-4-shaped hardcode."""
    from transformers import LlamaConfig
    dense = sc.parameter_footprint(
        LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2,
                    vocab_size=256), quant="4bit")
    assert dense["shrinkable_fraction"] > 0.8, dense["shrinkable_fraction"]
    assert dense["footprint_gib"] < 0.5 * dense["full_precision_gib"]


def test_four_bit_is_refused_with_the_measured_numbers_not_a_dispatch_error():
    """
    A 4/8-bit module cannot be placed on CPU or disk, so the loader must say so
    up front -- naming the unquantisable parameters -- instead of letting
    from_pretrained raise its generic 'dispatched on the CPU or the disk' error
    after minutes of reading weights.
    """
    import transformers

    class _Tok:
        is_fast = True
        chat_template = "{{ messages }}"

    class _AutoTok:
        @staticmethod
        def from_pretrained(*a, **k):
            return _Tok()

    cfg = _tiny_llama4_config(16)

    real_footprint = sc.parameter_footprint

    def fake_footprint(checkpoint, quant="bf16", **kw):
        fp = real_footprint(cfg, quant=quant)
        # scale the toy model up to Scout's order of magnitude
        fp["footprint_gib"] *= 1e6
        fp["full_precision_gib"] *= 1e6
        fp["params_outside_linear"] *= 10 ** 7
        fp["total_params"] *= 10 ** 7
        return fp

    saved = (sc.parameter_footprint, sc._gpu_budget, transformers.AutoTokenizer)
    sc.parameter_footprint = fake_footprint
    sc._gpu_budget = lambda reserve_gib=4.0: {0: 24.0}
    transformers.AutoTokenizer = _AutoTok
    try:
        sc.load_backend("/nonexistent/checkpoint", quant="4bit")
    except SystemExit as exc:
        msg = str(exc)
        assert "nn.Linear" in msg, msg
        assert "bf16" in msg, "the refusal must name the route that does work"
        assert "offload" in msg, msg
    else:
        raise AssertionError("a 4-bit load that cannot fit must be refused")
    finally:
        sc.parameter_footprint, sc._gpu_budget, transformers.AutoTokenizer = saved


def test_a_fitting_four_bit_load_is_not_refused():
    """The refusal must be driven by the budget, not by the word '4bit'."""
    import transformers

    class _Tok:
        is_fast = True
        chat_template = "{{ messages }}"

    saved = (sc.parameter_footprint, sc._gpu_budget, transformers.AutoTokenizer)
    sc.parameter_footprint = lambda ckpt, quant="bf16", **kw: dict(
        quant=quant, total_params=10 ** 9, params_in_linear=10 ** 9,
        params_outside_linear=0, full_precision_gib=2.0, footprint_gib=0.5,
        shrinkable_fraction=1.0)
    sc._gpu_budget = lambda reserve_gib=4.0: {0: 24.0}
    transformers.AutoTokenizer = type("A", (), {
        "from_pretrained": staticmethod(lambda *a, **k: _Tok())})
    try:
        sc.load_backend("/nonexistent/checkpoint", quant="4bit")
    except SystemExit as exc:
        raise AssertionError(f"a 4-bit load that fits must not be refused: {exc}")
    except Exception:
        pass          # it got past the budget check and failed on the weights
    finally:
        sc.parameter_footprint, sc._gpu_budget, transformers.AutoTokenizer = saved


def test_budgets_hold_memory_back_rather_than_claiming_all_of_it():
    cpu_small = sc._cpu_budget(reserve_gib=0.0)
    cpu_large = sc._cpu_budget(reserve_gib=8.0)
    assert cpu_small >= cpu_large
    assert cpu_small - cpu_large in (8.0, cpu_small), \
        "the reserve must come off the top of MemAvailable"
    assert sc._cpu_budget(reserve_gib=10 ** 9) == 0.0, "never report negative RAM"


def test_execution_device_ignores_offloaded_placements():
    """
    For an offloaded model `model.device` is `meta` or `cpu`; the forward still
    runs on an accelerator, and the inputs must go there.
    """
    import torch

    class _M:
        device = torch.device("meta")
        hf_device_map = {"model.embed_tokens": "disk",
                         "model.layers.0": "cpu",
                         "model.layers.1": 2,
                         "lm_head": "cpu"}

    dev = sc.execution_device(_M())
    assert dev.type == "cuda" and dev.index == 2, dev

    class _CpuOnly:
        device = torch.device("cpu")
        hf_device_map = {"model.layers.0": "cpu"}

    assert sc.execution_device(_CpuOnly()).type in ("cuda", "cpu")


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
        sc.resolve_span(text, offs, "history")
    except sc.SpanError as exc:
        assert "2 times" in str(exc)
    else:
        raise AssertionError("two raw matches must not resolve silently")

    span = sc.resolve_span(text, offs, "history",
                           competitors=["art history", "physics"])
    start, end = span
    located = text[offs[start][0]:offs[end - 1][1]]
    assert "art" not in located, located
    assert "history" in located

    # and the longer stimulus is still locatable itself
    sc.resolve_span(text, offs, "art history",
                    competitors=["art history", "history", "physics"])


def test_male_is_not_located_inside_female():
    text = "pair 'female' and 'male' with the words below"
    offs = _offsets_for(text)
    span = sc.resolve_span(text, offs, "male", competitors=["female", "male"])
    start, end = span
    assert "female" not in text[offs[start][0]:offs[end - 1][1]]
    # 'female' itself is unmasked: nothing longer contains it
    sc.resolve_span(text, offs, "female", competitors=["female", "male"])


def test_masking_only_applies_to_strictly_longer_containing_stimuli():
    """A competitor that does not contain the phrase must mask nothing."""
    text = "the word physics appears once here"
    offs = _offsets_for(text)
    before = sc.resolve_span(text, offs, "physics")
    after = sc.resolve_span(text, offs, "physics",
                            competitors=["chemistry", "physics", "art history"])
    assert before == after
    assert sc.masking_spans(text, "physics", ["physics", "chemistry"]) == [], \
        "a stimulus must not mask itself"


def test_masking_a_phrase_out_of_existence_is_an_error_not_a_silent_skip():
    """If every occurrence is inside a longer stimulus, say so."""
    text = "only 'art history' is listed here"
    offs = _offsets_for(text)
    try:
        sc.resolve_span(text, offs, "history", competitors=["art history"])
    except sc.SpanError as exc:
        assert "does not occur" in str(exc)
    else:
        raise AssertionError("a fully masked phrase must raise")


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
