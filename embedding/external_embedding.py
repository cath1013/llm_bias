"""
PART A — EXTERNAL SEMANTIC EMBEDDING ANALYSIS  (FAB + gender-brilliance)

WHAT THIS IS, AND WHAT IT IS NOT
    OpenAI text embeddings are used here as a COMMON, INDEPENDENT semantic
    reference space — a convergent representational check on our behavioural
    WATs.  They are NOT the internal representation of GPT, Claude, Gemini or
    Scout, and nothing here explains why any proprietary model behaved as it
    did.  Model-internal analysis is a separate concern and separate modules
    (`embedding/wat_hidden.py`, `embedding/chained_hidden.py`); the two share
    only the WEAT / statistics helpers in `embedding/core.py`.

METHODOLOGICAL REFERENCE
    Bai et al. (2025), SI section O — `ref/llm-implicit-bias/additional/`,
    audited in `ref/bai_method.md:682`.  We follow their contextualisation
    strategy and their WEAT formulation, and deliberately depart where their
    implementation is degenerate for our design:

    * Bai's every WEAT has ONE target word per side, so Caliskan's d collapses
      to  d/(|d|/2 + 0.001)  — bounded in (-2, 2) with the epsilon on the order
      of the signal.  Our stimulus SETS are multi-word, so set-level WEAT is run
      properly (real pooled SD, real effect size, exact permutation p).
    * At pair level a singleton really is 1-vs-1, so the PRIMARY statistic is
      the UNSTANDARDISED delta;  Bai's standardized singleton score is carried
      as a supplemental column only (`bai_singleton_weat`), never analysed.
    * Repeated sessions drawing the same target pair are NOT independent
      embedding observations.  Correlations are computed across UNIQUE PAIRS,
      so n = number of pairs, not number of sessions.
    * Their `weat_p_value` indexes an int32 0/1 array as if it were a boolean
      mask (`pool[i]`), so it re-selects rows 0 and 1; reimplemented here over
      `itertools.combinations`.
    * Their hidden-state script takes a phrase's FIRST token; irrelevant to an
      API embedder, and handled properly in the Scout module.
    * No PCA nullification of the external embeddings (exploratory, Scout only).

CONTEXTUALISATION  (Bai's fragments, verbatim)
    target    : "Here is a list of words. For each word pick a word — {word}"
    attribute : " — and write it after the word. The words are {word}"
    `wat_fab.build_prompt()` is character-identical to Bai's template, so both
    fragments are literal substrings of OUR prompts too — asserted at import.
    `--context bare` embeds the raw word instead, as a sensitivity check.

SOURCE OF RECORD
    `responses/llm bias fab.xlsx` and `responses/llm bias gender.xlsx` (every
    sheet named "wat *").  --source jsonl reads the full raw tree instead.
    A session is matched to an arm by the STIMULUS WORDS IT ACTUALLY SORTED, not
    by its control_set label: the label `creativity4` covers five different
    fourth words in the raw tree.

DATASETS  (race-brilliance is deliberately OUT OF SCOPE here)
    fab : X/Y = brilliance / effort poles    A/B = high-FAB / low-FAB fields
          arms are `{pole_set}@{split}` over pole sets x {tertile, median}
    gb  : X/Y = male / female words          A/B = brilliance / control words
          one arm per control set defined in wat_gender_brilliance.py
    Positive always means STEREOTYPE-CONGRUENT:
          fab  brilliance -> high-FAB field
          gb   male -> brilliance, female -> control

ANALYSES
    A1  set-level WEAT per arm: S, Caliskan d, exact two-sided permutation p.
    A2  FAB item-level WEFAT over all 30 fields, UNDICHOTOMISED, vs Leslie's
        continuous FAB score and vs each behavioural model's P(brilliance|field).
        Spearman is primary; Pearson secondary.  Model-specific correlations are
        reported separately — the WEFAT value is identical across behavioural
        models, so pooling them would fabricate n.
    A3  gb set-level WEAT per control condition (same machinery as A1).
    A4  pair level: per session delta = s(x,A,B) - s(y,A,B), then aggregated to
        unique (dataset, arm, model, pair); Spearman across unique pairs.

ANALYSIS SCOPE (see IN_SCOPE / --all-arms)
    gb  : humor, creativity4, warmth, dedication_adj.  beauty and physical are
          not part of the reported design; the noun-phrase `dedication` arm has
          one session in the corpus, so the adjectival form is used.
    fab : the tertile split only.
    Stimulus definitions are left complete and unchanged; this only selects
    which defined arms are analysed.

POLE-SET MATCHING (A2)
    Leslie WEFAT is compared with behaviour from Leslie-pole sessions only, and
    Bian WEFAT with Bian-pole sessions only.  The human FAB score is neither
    model- nor pole-set-specific, so that comparison needs no matching.

MULTIPLE COMPARISONS
    Holm-Bonferroni within pre-declared families (fab_set_level, gb_set_level,
    fab_wefat_vs_human, fab_wefat_vs_model_P::<model>, pair_level::<ds>::<model>).
    Raw and adjusted p-values are both kept; nothing is silently replaced.

BEHAVIOUR IS RECOMPUTED FROM RAW
    Session D and per-field P(brilliance) come from each session's raw
    `assignments`, never from a stored summary column.

OUTPUTS  (responses/embed/, tidy CSVs + manifest; existing files are never
    overwritten without --force)
    set_level.csv  fab_field_wefat.csv  fab_field_correlations.csv
    pair_level_raw.csv  pair_level_aggregated.csv  pair_level_correlations.csv
    embedding_manifest.json

USAGE
    python -m embedding.external_embedding --dry-run                 # inventory, no API calls
    python -m embedding.external_embedding --offline                 # cache only, no API calls
    python -m embedding.external_embedding --emb-model text-embedding-3-large --context both
"""

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
except ImportError:                 # python-dotenv is not installed everywhere
    def load_dotenv(path, override=False):
        """Minimal KEY=VALUE reader, so the module runs without python-dotenv."""
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if override or k.strip() not in os.environ:
                os.environ[k.strip()] = v
        return True

# This module lives in <project>/embedding/ but reads the behavioural runners and
# the response trees from the project root, so anchor everything there rather
# than on the current working directory.
from embedding import ROOT
import wat_fab
import wat_gender_brilliance as wgb
# The WEAT / WEFAT and statistics helpers are shared with the hidden-state
# experiments and live in embedding.core (moved there verbatim).
from embedding.core import (_scipy_stats, weat_test_stat, weat_effect_size,
                            bai_singleton_weat, weat_p_value, wefat_score,
                            pearson, spearman, apply_holm, write_csv)

load_dotenv(os.path.join(ROOT, ".env"), override=True)

ANALYSIS_VERSION = "2.3"          # 2.2 = reported arm scope (gb 4 controls, fab
                                  #       tertile); 2.1 = pole-set-matched WEFAT, no pooled
                                  #       pair rows, Holm families
                                  # 1.x = pre-spec version (rb + pooled item r)
OUT_DIR = os.path.join(ROOT, "responses", "embed")
DEFAULT_EMB_MODEL = "text-embedding-3-small"

OUTPUT_FILES = [
    "set_level.csv", "fab_field_wefat.csv", "fab_field_correlations.csv",
    "pair_level_raw.csv", "pair_level_aggregated.csv",
    "pair_level_correlations.csv", "embedding_manifest.json",
]

# =========================================================
# Contextualisation
# =========================================================

TARGET_CTX    = "Here is a list of words. For each word pick a word — {w}"
ATTRIBUTE_CTX = " — and write it after the word. The words are {w}"


def _verify_fragments():
    """Our prompt must literally contain Bai's two fragments, or the port is a lie."""
    p = wat_fab.build_prompt(["innate gift", "hard work"], ["philosophy"])
    assert TARGET_CTX.format(w="innate gift") in p, "target fragment drifted"
    assert ATTRIBUTE_CTX.format(w="['philosophy']").rstrip(".") in p + ".", \
        "attribute fragment drifted"


_verify_fragments()


def contextualise(word, kind, context):
    """kind: 'target' (a pick-one word) or 'attribute' (a sorted word)."""
    if context == "bare":
        return word
    return (TARGET_CTX if kind == "target" else ATTRIBUTE_CTX).format(w=word)


def corr_record(level, dataset, arm, model, xs, ys, *, emb_model, context,
                n_label="n", pooling=None, inferential=True,
                analysis_role="primary", **extra):
    """
    One tidy correlation row: Spearman primary, Pearson secondary, full metadata.

    `inferential=False` marks a row that is reported for description only; its
    p-values are blanked rather than written, because the observations it pools
    are not exchangeable.
    """
    rho, p_rho, n = spearman(xs, ys)
    r, p_r, _ = pearson(xs, ys)
    row = dict(level=level, dataset=dataset, arm=arm, behavioural_model=model,
               spearman_rho=rho, spearman_p=(p_rho if inferential else ""),
               pearson_r=r, pearson_p=(p_r if inferential else ""),
               inferential=inferential, analysis_role=analysis_role,
               emb_model=emb_model, context=context, pooling=pooling,
               analysis_version=ANALYSIS_VERSION)
    row[n_label] = n
    row.update(extra)
    return row


# =========================================================
# Embedding client + on-disk cache
# =========================================================

class Embedder:
    """Caches one vector per (embedding model, context, string) on disk."""

    def __init__(self, model, context, out_dir=OUT_DIR, offline=False):
        self.model, self.context, self.offline = model, context, offline
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"cache__{model}__{context}.json")
        self.cache = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.cache = json.load(f)
        self._client = None
        self.n_new = 0

    def _api(self):
        if self._client is None:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._client = OpenAI(api_key=key)
        return self._client

    def prime(self, words, kind):
        """Embed every not-yet-cached string, batched."""
        todo = sorted({contextualise(w, kind, self.context) for w in words}
                      - set(self.cache))
        if not todo:
            return
        if self.offline:
            raise RuntimeError(
                f"--offline, but {len(todo)} string(s) are not in "
                f"{self.path} (e.g. {todo[0]!r}). Rerun without --offline to embed them.")
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            resp = self._api().embeddings.create(input=chunk, model=self.model)
            for s, d in zip(chunk, resp.data):
                self.cache[s] = d.embedding
            self.n_new += len(chunk)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    def get(self, word, kind):
        return self.cache[contextualise(word, kind, self.context)]

    def mat(self, words, kind):
        import numpy as np
        return np.array([self.get(w, kind) for w in words], dtype=float)


# =========================================================
# Datasets — stimulus wiring.  Stimuli come from the behavioural runners
# unchanged; nothing is redefined here.
# =========================================================

FAB_SPLITS = {
    "tertile": (wat_fab.HIGH_TERT, wat_fab.LOW_TERT, "top 10", "bottom 10"),
    "median":  (wat_fab.HIGH_MEDIAN, wat_fab.LOW_MEDIAN, "top 15", "bottom 15"),
}


def fab_arms():
    """`{pole_set}@{split}` -> arm. Targets = poles, attributes = FAB field split."""
    arms = {}
    for pole_set, (bri, eff) in wat_fab.POLE_SETS.items():
        for split, (hi, lo, hname, lname) in FAB_SPLITS.items():
            arms[f"{pole_set}@{split}"] = dict(
                X=list(bri), Y=list(eff),
                A=[f for f in wat_fab.FIELD_LABELS if f in hi],
                B=[f for f in wat_fab.FIELD_LABELS if f in lo],
                pole_set=pole_set, split=split,
                X_name=f"brilliance poles ({pole_set})",
                Y_name=f"effort poles ({pole_set})",
                A_name=f"high-FAB fields ({hname})",
                B_name=f"low-FAB fields ({lname})")
    return arms


# ---------------------------------------------------------------------------
# ANALYSIS SCOPE.  The stimulus definitions above are left complete and
# untouched; this is the subset the study actually reports.  --all-arms runs
# every defined arm for exploration.
#
#   gb : humor, creativity4 ("creativity with original"), warmth, dedication_adj
#        -- beauty and physical are out of scope.  dedication_adj rather than
#        dedication: the noun-phrase arm has a single session in the corpus, and
#        the adjectival form matches the other three retained controls.
#   fab: the tertile split only (top 10 vs bottom 10 fields).
# ---------------------------------------------------------------------------
IN_SCOPE = {
    "fab": lambda name, arm: arm["split"] == "tertile",
    "gb": lambda name, arm: name in ("humor", "creativity4", "warmth",
                                     "dedication_adj"),
}


def scoped(dataset, arms):
    """The reported subset of `arms` for this dataset (see IN_SCOPE)."""
    keep = IN_SCOPE.get(dataset)
    return arms if keep is None else {k: v for k, v in arms.items() if keep(k, v)}


def gb_arms():
    """One arm per control set, each with the brilliance set Storage paired it with."""
    arms = {}
    for ctrl, words in wgb.CONTROL_SETS.items():
        bset, bwords = wgb.resolve_brilliance(ctrl)
        arms[ctrl] = dict(X=list(wgb.MALE_WORDS), Y=list(wgb.FEMALE_WORDS),
                          A=list(bwords), B=list(words),
                          pole_set=bset, split=None,
                          X_name="male words", Y_name="female words",
                          A_name=f"brilliance ({bset})", B_name=f"control ({ctrl})")
    return arms


def _assignments(record):
    """`assignments` is a JSON string in some runners' jsonl, a dict in others."""
    a = record["assignments"]
    if isinstance(a, str):
        a = json.loads(a)
    return list(a.items())


def fab_pole_set(record):
    """
    Which pole set a fab session was run with.  Single source of truth: the arm
    router and the WEFAT behavioural accumulator must agree, or Leslie WEFAT
    would end up correlated against Bian behaviour.
    """
    return ("bian" if record["brilliance_word"] in wat_fab.BIAN_BRILLIANCE_WORDS
            else "leslie")


def _assigned_items(record):
    """The set of items this session actually sorted."""
    return {item for item, _ in _assignments(record)}


def arm_is_present(record, arm, mode="exact"):
    """
    Decide from the WORDS THE SESSION ACTUALLY SORTED whether it belongs to this
    arm.

      mode="exact"  (gb)  : the session sorted exactly this arm's A u B.
      mode="superset" (fab): the arm's stimuli are all present.  A fab session
                             sorts all 30 fields while an arm uses the 20 of one
                             split, so equality cannot be required there.

    Exact matching is what gb needs: the 3-word `creativity` set is a strict
    SUBSET of `creativity4`, so a superset test would score every creativity4
    session as a creativity session as well.

    This is not a formality.  In `responses/gb baseline/` the label
    `control_set == "creativity4"` covers FIVE different fourth words -- original,
    inventive, unique, resourceful, visionary -- distinguishable only by file
    name, and the curated workbook splits only two of them out by label.  Matching
    on the words themselves is what keeps "creativity4" meaning the one set
    `wat_gender_brilliance.CONTROL_SETS["creativity4"]` defines, instead of
    silently pooling five attribute sets and scoring four of them against a word
    their sessions never saw.  It also drops the handful of unparsed trials.
    """
    items = _assigned_items(record)
    stimuli = set(arm["A"]) | set(arm["B"])
    if mode == "exact":
        return items == stimuli
    return stimuli <= items


def _fab_session_arms(record, arms):
    """A fab session belongs to its pole set, hence to both of that set's splits."""
    pole = fab_pole_set(record) + "@"
    return [a for a, arm in arms.items()
            if a.startswith(pole) and arm_is_present(record, arm, "superset")]


DATASETS = {
    "fab": dict(
        dir=os.path.join(ROOT, "responses", "fab baseline"),
        glob="*_wat_fab.jsonl",
        workbook=os.path.join(ROOT, "responses", "llm bias fab.xlsx"),
        arms=fab_arms,
        session_arms=_fab_session_arms,
        session_X=lambda r: r["brilliance_word"],
        session_Y=lambda r: r["effort_word"],
        pos_label="brilliance",           # assignments value meaning "set A side"
        item_of=_assignments,
    ),
    "gb": dict(
        dir=os.path.join(ROOT, "responses", "gb baseline"),
        glob="*_wat_gb.jsonl",
        workbook=os.path.join(ROOT, "responses", "llm bias gender.xlsx"),
        arms=gb_arms,
        session_arms=lambda r, arms: [a for a, arm in arms.items()
                                      if arm_is_present(r, arm, "exact")],
        session_X=lambda r: r["male_word"],
        session_Y=lambda r: r["female_word"],
        pos_label="male",
        item_of=_assignments,
    ),
}
# NOTE: race-brilliance (`responses/rb baseline/`) is deliberately absent — out of
# scope for this analysis.  Its raw data is untouched; no code path references it.


# =========================================================
# Behaviour, recomputed from raw `assignments`
# =========================================================

def session_D(record, spec, arm):
    """
    D = P(A-side | A-set) + P(B-side | B-set) - 1, from this session's raw
    assignments.  Positive = stereotype-congruent.  None if a cell is empty.
    """
    A, B = set(arm["A"]), set(arm["B"])
    pos = spec["pos_label"]
    a_hit = a_n = b_hit = b_n = 0
    for item, label in spec["item_of"](record):
        if item in A:
            a_n += 1; a_hit += (label == pos)
        elif item in B:
            b_n += 1; b_hit += (label != pos)
    if not a_n or not b_n:
        return None
    return a_hit / a_n + b_hit / b_n - 1


def load_sessions_xlsx(path, sheet_prefix="wat", strict=True):
    """
    Read the curated workbook: every sheet whose name starts with `sheet_prefix`
    is a block of WAT sessions with the same schema as the jsonl (`assignments`
    is a JSON string).  Rows without a model id, or whose `assignments` will not
    parse, are dropped and counted -- the workbook carries blank separator rows
    and a few unparsed trials.

    The BEHAVIOURAL MODEL IS TAKEN FROM THE `model` COLUMN, never from the sheet
    name: at least one sheet ("wat sonnet 4.6" in the gender workbook) contains
    a block of another model's trials.
    """
    import pandas as pd
    out, dropped = [], defaultdict(int)
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        if not sheet.startswith(sheet_prefix):
            continue
        df = xl.parse(sheet)
        for rec in df.to_dict("records"):
            model = rec.get("model")
            if not isinstance(model, str) or not model.strip():
                dropped["no model id"] += 1
                continue
            raw = rec.get("assignments")
            try:
                a = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(a, dict) or not a:
                    raise ValueError("empty")
            except Exception:
                dropped["unparsable assignments"] += 1
                continue
            rec["assignments"] = a
            rec["_file"] = f"{os.path.basename(path)}#{sheet}"
            out.append(rec)
    if dropped:
        print(f"  [{os.path.basename(path)}] dropped "
              + ", ".join(f"{n} rows ({why})" for why, n in sorted(dropped.items())))
    return out


def load_sessions(spec, models=None, source="xlsx"):
    if source == "xlsx":
        out = load_sessions_xlsx(spec["workbook"])
        return [r for r in out if not models or r.get("model") in models]

    import glob as _glob
    out = []
    for path in sorted(_glob.glob(os.path.join(spec["dir"], spec["glob"]))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if models and r.get("model") not in models:
                    continue
                r["_file"] = path
                out.append(r)
    return out


# =========================================================
# A1 / A3 — set-level WEAT
# =========================================================

def run_set_level(name, arms, emb, rows):
    for arm_name, arm in sorted(arms.items()):
        emb.prime(arm["X"] + arm["Y"], "target")
        emb.prime(arm["A"] + arm["B"], "attribute")
        X, Y = emb.mat(arm["X"], "target"), emb.mat(arm["Y"], "target")
        A, B = emb.mat(arm["A"], "attribute"), emb.mat(arm["B"], "attribute")
        p, nperm, exact = weat_p_value(X, Y, A, B)
        rows.append(dict(
            dataset=name, arm=arm_name, pole_set=arm["pole_set"], split=arm["split"],
            emb_model=emb.model, context=emb.context,
            n_X=len(X), n_Y=len(Y), n_A=len(A), n_B=len(B),
            weat_S=weat_test_stat(X, Y, A, B),
            weat_d=weat_effect_size(X, Y, A, B),
            perm_p=p, n_perm=nperm, perm_exact=exact,
            X_name=arm["X_name"], Y_name=arm["Y_name"],
            A_name=arm["A_name"], B_name=arm["B_name"],
            analysis_version=ANALYSIS_VERSION))


# =========================================================
# A2 — FAB item-level WEFAT (continuous, all 30 fields, never dichotomised)
# =========================================================

def run_fab_wefat(spec, emb, sessions, field_rows, corr_rows):
    """
    WEFAT(field) on the brilliance <-> effort dimension: the field is the single
    item (attribute context, as in the prompt), the two POLE SETS are the
    comparison sets (target context).  Positive = closer to brilliance.
    The 30 fields stay continuous; the FAB split is used nowhere in this analysis.
    """
    fields = list(wat_fab.FIELD_LABELS)
    emb.prime(fields, "attribute")

    # Behaviour, POLE-SET MATCHED: a session contributes only to the pole set it
    # was actually run with, so Leslie WEFAT is compared against Leslie-pole
    # behaviour and Bian WEFAT against Bian-pole behaviour.  Pooling the two
    # would correlate one embedding dimension with behaviour produced by a
    # different pair of poles.
    beh = defaultdict(lambda: [0, 0])          # (pole_set, model, field) -> [hits, n]
    models_of = defaultdict(set)               # pole_set -> behavioural models
    for r in sessions:
        ps = fab_pole_set(r)
        models_of[ps].add(r.get("model"))
        for field, label in spec["item_of"](r):
            key = (ps, r.get("model"), field)
            beh[key][1] += 1
            beh[key][0] += (label == spec["pos_label"])

    for pole_set, (bri, eff) in wat_fab.POLE_SETS.items():
        emb.prime(list(bri) + list(eff), "target")
        Abri, Beff = emb.mat(list(bri), "target"), emb.mat(list(eff), "target")
        wefat = {f: wefat_score(emb.get(f, "attribute"), Abri, Beff) for f in fields}
        models = sorted(models_of.get(pole_set, ()), key=str)

        # one row per field per behavioural model (plus a model-free stimulus row)
        for f in fields:
            field_rows.append(dict(
                dataset="fab", pole_set=pole_set, emb_model=emb.model,
                context=emb.context, field=f,
                human_FAB_score=wat_fab.FAB_SCORE[f], embedding_WEFAT=wefat[f],
                behavioural_model="(stimulus only)", n_behavioral_observations=0,
                P_brilliance="", behavioural_sessions=f"{pole_set}-pole sessions only",
                analysis_version=ANALYSIS_VERSION))
            for m in models:
                hits, n = beh.get((pole_set, m, f), [0, 0])
                field_rows.append(dict(
                    dataset="fab", pole_set=pole_set, emb_model=emb.model,
                    context=emb.context, field=f,
                    human_FAB_score=wat_fab.FAB_SCORE[f], embedding_WEFAT=wefat[f],
                    behavioural_model=m, n_behavioral_observations=n,
                    P_brilliance=(hits / n) if n else "",
                    behavioural_sessions=f"{pole_set}-pole sessions only",
                    analysis_version=ANALYSIS_VERSION))

        # PRIMARY: WEFAT vs the human FAB score.  The human score is neither
        # model- nor pole-set-specific, so this comparison needs no matching.
        corr_rows.append(corr_record(
            "fab_field_wefat_vs_human_FAB", "fab", pole_set, None,
            [wefat[f] for f in fields], [wat_fab.FAB_SCORE[f] for f in fields],
            emb_model=emb.model, context=emb.context, n_label="n_fields",
            pooling="none (30 fields)"))

        # PRIMARY: WEFAT vs each model's P(brilliance|field) from THIS pole set
        for m in models:
            got = [(wefat[f], beh[(pole_set, m, f)][0] / beh[(pole_set, m, f)][1])
                   for f in fields if beh.get((pole_set, m, f), [0, 0])[1]]
            if len(got) < 3:
                continue
            corr_rows.append(corr_record(
                "fab_field_wefat_vs_model_P", "fab", pole_set, m,
                [a for a, _ in got], [b for _, b in got],
                emb_model=emb.model, context=emb.context, n_label="n_fields",
                pooling=f"{pole_set}-pole sessions only "
                        f"(WEFAT and behaviour share the pole set)",
                n_sessions=sum(1 for r in sessions
                               if fab_pole_set(r) == pole_set and r.get("model") == m)))


# =========================================================
# A4 — pair level: raw sessions -> unique pairs -> correlation
# =========================================================

def run_pair_level(name, spec, arms, emb, sessions, raw_rows, agg_rows, corr_rows):
    for r in sessions:
        x, y = spec["session_X"](r), spec["session_Y"](r)
        if not x or not y:
            continue
        for arm_name in spec["session_arms"](r, arms):
            arm = arms[arm_name]
            emb.prime([x, y], "target")
            emb.prime(arm["A"] + arm["B"], "attribute")
            X, Y = emb.mat([x], "target"), emb.mat([y], "target")
            A, B = emb.mat(arm["A"], "attribute"), emb.mat(arm["B"], "attribute")
            d = session_D(r, spec, arm)
            if d is None:
                continue
            raw_rows.append(dict(
                dataset=name, arm=arm_name, emb_model=emb.model, context=emb.context,
                behavioural_model=r.get("model"), trial=r.get("trial"),
                X_word=x, Y_word=y, pair=f"{x} || {y}",
                delta=weat_test_stat(X, Y, A, B),                 # PRIMARY
                bai_singleton_weat=bai_singleton_weat(X, Y, A, B),  # supplemental
                behavioural_D=d, source_file=r.get("_file"),
                analysis_version=ANALYSIS_VERSION))

    # aggregate: one embedding observation per unique (dataset, arm, model, pair)
    groups = defaultdict(list)
    for row in raw_rows:
        if row["emb_model"] != emb.model or row["context"] != emb.context:
            continue
        groups[(row["dataset"], row["arm"], row["behavioural_model"],
                row["pair"])].append(row)
    this_agg = []
    for (ds, arm_name, model, pair), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if ds != name:
            continue
        deltas = {round(x["delta"], 12) for x in rs}
        assert len(deltas) == 1, f"pair {pair} has >1 embedding delta: {deltas}"
        agg = dict(
            dataset=ds, arm=arm_name, emb_model=emb.model, context=emb.context,
            behavioural_model=model, pair=pair,
            X_word=rs[0]["X_word"], Y_word=rs[0]["Y_word"],
            delta=rs[0]["delta"], bai_singleton_weat=rs[0]["bai_singleton_weat"],
            mean_behavioural_D=sum(x["behavioural_D"] for x in rs) / len(rs),
            n_sessions=len(rs), analysis_version=ANALYSIS_VERSION)
        agg_rows.append(agg)
        this_agg.append(agg)

    # PRIMARY: correlate ACROSS UNIQUE PAIRS, within one arm and one behavioural
    # model.  No "(all models)" row: the same pair carries the SAME embedding
    # delta for every model, so stacking models would repeat each embedding
    # observation k times and inflate n without adding information.
    by = defaultdict(list)
    for a in this_agg:
        by[(a["arm"], a["behavioural_model"])].append(a)
    for (arm_name, model), rs in sorted(by.items(), key=str):
        corr_rows.append(corr_record(
            "pair_level_delta_vs_behavioural_D", name, arm_name, model,
            [a["delta"] for a in rs], [a["mean_behavioural_D"] for a in rs],
            emb_model=emb.model, context=emb.context, n_label="n_unique_pairs",
            pooling="unique target pairs within one arm and one behavioural model",
            analysis_role="primary_within_arm",
            n_sessions=sum(a["n_sessions"] for a in rs)))

    # DESCRIPTIVE ONLY: the same unique pairs pooled across arms.  Different arms
    # use different A/B attribute sets, so their deltas are not exchangeable and
    # no inferential p-value is reported.  It is kept because the within-arm and
    # cross-arm pictures differ sharply for gb, and the difference is itself the
    # finding: a cross-arm correlation reflects WHICH CONTROL CONDITION a session
    # used, not which target pair it drew.
    pooled = defaultdict(list)
    for a in this_agg:
        pooled[a["behavioural_model"]].append(a)
    for model, rs in sorted(pooled.items(), key=str):
        if len({a["arm"] for a in rs}) < 2:
            continue
        corr_rows.append(corr_record(
            "pair_level_delta_vs_behavioural_D@arms_pooled", name,
            "(all arms)", model,
            [a["delta"] for a in rs], [a["mean_behavioural_D"] for a in rs],
            emb_model=emb.model, context=emb.context, n_label="n_unique_pairs",
            inferential=False, analysis_role="descriptive_cross_condition",
            pooling="unique pairs pooled across arms — mixes attribute sets, so "
                    "the deltas are not exchangeable; no p-value is reported",
            n_arms=len({a["arm"] for a in rs}),
            n_sessions=sum(a["n_sessions"] for a in rs)))


# =========================================================
# Output
# =========================================================


def star(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p != p:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def _fmt_p(p):
    """Blank for a descriptive row (no p-value) or a missing correction."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "   --"
    return "   --" if p != p else f"{p:8.4f}"


def report(set_rows, corr_rows, agg_rows):
    print()
    print("=" * 96)
    print("A1/A3  SET-LEVEL WEAT  (non-degenerate: multi-word X and Y)")
    print("=" * 96)
    print(f"{'ds':4} {'arm':18} {'ctx':7} {'S':>9} {'d':>7} {'p_perm':>8}   "
          f"{'p_holm':>8}   sets")
    for r in set_rows:
        print(f"{r['dataset']:4} {r['arm']:18} {r['context']:7} {r['weat_S']:+9.4f} "
              f"{r['weat_d']:+7.3f} {r['perm_p']:8.4f}{star(r['perm_p']):<3} "
              f"{_fmt_p(r.get('perm_p_holm')):>8}{star(r.get('perm_p_holm')):<3} "
              f"{r['A_name']} vs {r['B_name']}")

    for level, title in [
        ("fab_field_wefat_vs_human_FAB",
         "A2  FAB WEFAT vs human FAB score (30 fields, continuous)"),
        ("fab_field_wefat_vs_model_P",
         "A2  FAB WEFAT vs model P(brilliance|field), per behavioural model"),
        ("pair_level_delta_vs_behavioural_D",
         "A4  pair-level delta vs behavioural D, WITHIN arm (n = UNIQUE PAIRS)"),
        ("pair_level_delta_vs_behavioural_D@arms_pooled",
         "A4b pair-level, arms POOLED — DESCRIPTIVE ONLY, no p-value "
         "(mixes attribute sets)"),
    ]:
        rs = [r for r in corr_rows if r["level"] == level]
        if not rs:
            continue
        print()
        print("=" * 96)
        print(title + "   [Spearman primary, Pearson secondary]")
        print("=" * 96)
        nkey = "n_fields" if "n_fields" in rs[0] else "n_unique_pairs"
        print(f"{'ds':4} {'arm':18} {'ctx':7} {'model':26} {nkey:>7} "
              f"{'rho':>7} {'p_raw':>9} {'p_holm':>9} {'r':>7}")
        for r in rs:
            print(f"{r['dataset']:4} {r['arm']:18} {r['context']:7} "
                  f"{str(r['behavioural_model'] or '-'):26} {r[nkey]:7d} "
                  f"{r['spearman_rho']:+7.3f} "
                  f"{_fmt_p(r['spearman_p']):>8}{star(r['spearman_p']):<3} "
                  f"{_fmt_p(r.get('spearman_p_holm')):>8}{star(r.get('spearman_p_holm')):<3} "
                  f"{r['pearson_r']:+7.3f}")


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=list(DATASETS),
                    help="repeatable; default = fab and gb")
    ap.add_argument("--emb-model", default=DEFAULT_EMB_MODEL,
                    help="any OpenAI embedding model id")
    ap.add_argument("--context", default="prompt", choices=["prompt", "bare", "both"],
                    help="prompt = Bai's WAT-fragment contextualisation (default)")
    ap.add_argument("--source", default="xlsx", choices=["xlsx", "jsonl"],
                    help="xlsx (default) = the curated workbooks of record; "
                         "jsonl = the full raw response tree")
    ap.add_argument("--model", action="append",
                    help="restrict behavioural sessions to these LLMs (repeatable)")
    ap.add_argument("--arm", action="append", help="restrict to these arms (repeatable)")
    ap.add_argument("--all-arms", action="store_true",
                    help="run every DEFINED arm, not just the reported scope "
                         "(gb: humor/creativity4/warmth/dedication_adj; fab: tertile)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--offline", action="store_true",
                    help="fail instead of calling the API if anything is uncached")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting existing output files")
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory only: no embeddings, no API calls, no files")
    args = ap.parse_args()

    datasets = args.dataset or list(DATASETS)
    contexts = ["prompt", "bare"] if args.context == "both" else [args.context]
    os.makedirs(args.out, exist_ok=True)

    set_rows, field_rows, corr_rows = [], [], []
    pair_raw, pair_agg = [], []
    manifest = dict(
        analysis_version=ANALYSIS_VERSION,
        analysis="external semantic embedding space (NOT model-internal)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        emb_model=args.emb_model, contexts=contexts,
        context_templates={"target": TARGET_CTX, "attribute": ATTRIBUTE_CTX},
        token_pooling="n/a (API embedder returns one vector per string)",
        pca_k_values="n/a (no nullification in the external analysis)",
        statistics=dict(primary="spearman", secondary="pearson",
                        spearman_backend="scipy" if _scipy_stats else "ranks+t-approx",
                        permutation="exact two-sided",
                        multiple_comparisons="holm-bonferroni, see "
                                             "multiple_comparisons below"),
        datasets={}, stimuli={}, inputs={}, counts={})

    for name in datasets:
        spec = DATASETS[name]
        all_arms = spec["arms"]()
        arms = all_arms if args.all_arms else scoped(name, all_arms)
        out_of_scope = sorted(set(all_arms) - set(arms))
        if out_of_scope:
            print(f"[{name}] out of scope, not analysed: {', '.join(out_of_scope)}")
        if args.arm:
            arms = {k: v for k, v in arms.items() if k in args.arm}
            if not arms:
                print(f"[{name}] no arm matches {args.arm} — skipped")
                continue
        sessions = load_sessions(spec, set(args.model) if args.model else None,
                                 source=args.source)
        models = sorted({r.get("model") for r in sessions}, key=str)
        used = sorted({a for r in sessions for a in spec["session_arms"](r, arms)})
        print(f"[{name}] {len(sessions)} sessions, {len(models)} behavioural models, "
              f"{len(arms)} arms defined, {len(used)} with sessions")
        print(f"[{name}] models: {', '.join(models)}")

        manifest["datasets"][name] = dict(
            arms=sorted(arms), arms_with_sessions=used,
            arms_defined=sorted(all_arms), arms_out_of_scope=out_of_scope,
            scope=("all defined arms (--all-arms)" if args.all_arms else
                   "reported scope: gb = humor/creativity4/warmth/dedication_adj, "
                   "fab = tertile split"),
            behavioural_models=models, n_sessions=len(sessions))
        manifest["stimuli"][name] = {k: {s: v[s] for s in ("X", "Y", "A", "B")}
                                     for k, v in arms.items()}
        manifest["inputs"][name] = sorted({r["_file"] for r in sessions})
        manifest["datasets"][name]["source"] = args.source
        manifest["datasets"][name]["source_file"] = (
            spec["workbook"] if args.source == "xlsx" else spec["dir"])
        unmatched = [r for r in sessions if not spec["session_arms"](r, arms)]
        manifest["datasets"][name]["sessions_matching_no_arm"] = len(unmatched)
        if unmatched:
            byset = defaultdict(int)
            for r in unmatched:
                byset[r.get("control_set") or r.get("brilliance_word")] += 1
            print(f"[{name}] {len(unmatched)} sessions match no in-scope arm: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(byset.items(), key=str)))

        if args.dry_run:
            for a, arm in sorted(arms.items()):
                print(f"   arm {a:18} X={len(arm['X'])} Y={len(arm['Y'])} "
                      f"A={len(arm['A'])} B={len(arm['B'])}")
            continue

        for ctx in contexts:
            emb = Embedder(args.emb_model, ctx, args.out, offline=args.offline)
            run_set_level(name, arms, emb, set_rows)
            run_pair_level(name, spec, arms, emb, sessions,
                           pair_raw, pair_agg, corr_rows)
            if name == "fab":
                run_fab_wefat(spec, emb, sessions, field_rows, corr_rows)
            if emb.n_new:
                print(f"[{name}] {ctx}: embedded {emb.n_new} new strings "
                      f"({args.emb_model})")

    if args.dry_run:
        print("\ndry run — no embeddings fetched, no files written")
        return

    # ---- multiple comparisons: Holm within pre-declared families -------------
    # Families are formed within one embedding model and one contextualisation.
    apply_holm(
        set_rows,
        key_of=lambda r: (r["dataset"], r["emb_model"], r["context"]),
        label_of=lambda r: f"{r['dataset']}_set_level",
        p_key="perm_p", out_key="perm_p_holm")

    def _corr_family(r):
        if not r.get("inferential", True):
            return None                       # descriptive rows are not corrected
        if r["level"] == "fab_field_wefat_vs_human_FAB":
            return ("fab_wefat_vs_human", r["emb_model"], r["context"])
        if r["level"] == "fab_field_wefat_vs_model_P":
            return ("fab_wefat_vs_model_P", r["behavioural_model"],
                    r["emb_model"], r["context"])
        if r["level"] == "pair_level_delta_vs_behavioural_D":
            return ("pair_level", r["dataset"], r["behavioural_model"],
                    r["emb_model"], r["context"])
        return None

    def _corr_label(r):
        if r["level"] == "fab_field_wefat_vs_human_FAB":
            return "fab_wefat_vs_human"
        if r["level"] == "fab_field_wefat_vs_model_P":
            return f"fab_wefat_vs_model_P::{r['behavioural_model']}"
        return f"pair_level::{r['dataset']}::{r['behavioural_model']}"

    apply_holm(corr_rows, key_of=_corr_family, label_of=_corr_label,
               p_key="spearman_p", out_key="spearman_p_holm")

    manifest["multiple_comparisons"] = dict(
        method="holm-bonferroni (step-down), applied to the PRIMARY p-value only",
        raw_p_retained=True,
        families=sorted({r["holm_family"] for r in set_rows + corr_rows
                         if r.get("holm_family")}),
        note="families are formed within one embedding model and one "
             "contextualisation; descriptive cross-condition rows are not "
             "corrected because they carry no inferential p-value")

    manifest["counts"] = dict(set_level=len(set_rows), fab_field_wefat=len(field_rows),
                              pair_level_raw=len(pair_raw),
                              pair_level_aggregated=len(pair_agg),
                              correlations=len(corr_rows))

    existing = [f for f in OUTPUT_FILES if os.path.exists(os.path.join(args.out, f))]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite {len(existing)} existing file(s) in "
                         f"{args.out}: {', '.join(existing)}\n"
                         f"rerun with --force (nothing was written)")

    print()
    write_csv(os.path.join(args.out, "set_level.csv"), set_rows, args.force)
    write_csv(os.path.join(args.out, "fab_field_wefat.csv"), field_rows, args.force)
    write_csv(os.path.join(args.out, "fab_field_correlations.csv"),
              [r for r in corr_rows if r["level"].startswith("fab_field")], args.force)
    write_csv(os.path.join(args.out, "pair_level_raw.csv"), pair_raw, args.force)
    write_csv(os.path.join(args.out, "pair_level_aggregated.csv"), pair_agg, args.force)
    write_csv(os.path.join(args.out, "pair_level_correlations.csv"),
              [r for r in corr_rows if r["level"].startswith("pair_level")], args.force)
    with open(os.path.join(args.out, "embedding_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {os.path.join(args.out, 'embedding_manifest.json')}")

    report(set_rows, corr_rows, pair_agg)


if __name__ == "__main__":
    main()
