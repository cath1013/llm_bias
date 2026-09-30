"""
EMBEDDING-LEVEL BIAS  —  Bai et al. (2025) SI section O, retargeted at our stimuli

    Our WAT runners measure what a model *does* (which pole it assigns a word to).
    This measures whether the same association sits in the **geometry** of an
    embedding space, with no model behaviour involved at all.  It is Bai's
    "embedding bias" validity check, re-pointed at brilliance/FAB/gender/race.

WHAT BAI DID  (`ref/llm-implicit-bias/additional/`, see ref/bai_method.md:682)
    Two arms.  Arm 1 (main text, Fig. 4) = OpenAI API embeddings: each stimulus
    word embedded *inside a fragment of the WAT prompt*, then WEAT, then
    correlated with the session's decision bias.  Arm 2 (SI only) = Llama-3-8B
    hidden states, 33 layers, from the real prompt.
    THIS FILE IS ARM 1.  Arm 2 needs local weights and is a separate build.

WHY ARM 1 PORTS EXACTLY
    Bai's contextualisation fragments are slices of his WAT prompt, and our
    `build_prompt()` is character-identical to his template, so the same two
    fragments are literal substrings of OUR prompts too (asserted at import):
        target    : "Here is a list of words. For each word pick a word — {w}"
        attribute : " — and write it after the word. The words are {w}"
    Nothing about the fragment had to be invented for us.

WHAT WE CHANGED, AND WHY
  1. **Non-degenerate WEAT (`set` level).**  Every WEAT in Bai's paper has ONE
     target word per side, which collapses `weat_score` to Δ/(|Δ|/2 + 0.001) —
     bounded in (−2, 2) with ε on the order of the signal.  Bai noticed and fell
     back to the raw `test_stat`.  Our stimuli come in *sets* (4 brilliance
     words, 4 effort words, 8 male words, ...), so we can run WEAT the way
     Caliskan defined it — real pooled SD, real effect size, exact permutation
     p-value.  This is strictly better evidence than Bai could produce.
  2. **Roles inverted for `fab`**, exactly as `wat_fab.py` inverts them: the
     poles are the targets, the 30 Leslie fields are the attributes.
  3. **WEFAT (`item` level) — the addition that matters most.**  Bai computed
     WEFAT in `utils.py` and never used it.  For FAB it is the direct embedding
     analogue of our PRIMARY behavioural measure: a per-field continuous
     brilliance association, correlated across the 30 fields with Leslie's
     continuous FAB score.  That is Leslie (2015) Fig. 1 redrawn in embedding
     space — and then against our own per-field P(brilliance), which asks
     whether behaviour and geometry rank the fields the same way.
  4. **Multi-word stimuli are the norm here** ("special aptitude", "music
     composition", "very beautiful").  Bai's arm-2 code takes a phrase's FIRST
     token only; the API embedder has no such problem, which is another reason
     arm 1 comes first.
  5. **Correct permutation test.**  Bai's vendored `weat_p_value` builds an
     int32 0/1 array and then does `target_words[i]` — integer fancy-indexing,
     not a boolean mask, so it repeatedly selects rows 0 and 1.  It is broken,
     and unused by every notebook.  Reimplemented here over `combinations`.

THREE LEVELS OF ANALYSIS  (all three run for every dataset that supports them)
    set     : WEAT over the FULL stimulus sets.  One number per arm.  The
              headline "is it in the geometry at all" test.  Non-degenerate.
    session : Bai-faithful.  Per session, use the exact word pair THAT session
              drew (1 vs 1, so `test_stat` only) and correlate with that
              session's behavioural D.  This is Fig. 4's bottom row.
              Only meaningful where the drawn pair varies:
                fab 4x4=16 pairs, gb 8x8=64 pairs, rb FIXED ("White"/"Black")
              -> rb has no session-level variance and is skipped there, by
              design, not by omission.
    item    : WEFAT per sorted word, vs FAB score (fab) and vs behavioural
              P(pole) (all datasets).

DATASETS
    fab : X/Y = brilliance / effort poles     A/B = high-FAB / low-FAB fields
          (tertile split, top-10 vs bottom-10 — `wat_fab.HIGH_TERT/LOW_TERT`)
    gb  : X/Y = male / female group words     A/B = brilliance / control words
    rb  : X/Y = "White" / "Black"             A/B = brilliance / control words
    Sign convention matches each runner's D: positive = the human stereotype
    (brilliance->high-FAB field, brilliance->male, brilliance->White).

BEHAVIOUR IS RECOMPUTED FROM RAW
    Session D and per-item P are recomputed from the `assignments` dict in the
    jsonl, never read from the stored score columns — the project rule.

USAGE
    python embed_bias.py                            # all datasets, 3-small, prompt-context
    python embed_bias.py --dataset fab --context both
    python embed_bias.py --emb-model text-embedding-3-large
    python embed_bias.py --dry-run                  # stimulus/session inventory, no API calls

COST
    Embeddings are cached per (model, context) in responses/embed/cache__*.json,
    so a re-run is free.  A cold run over all three datasets is a few hundred
    short strings — well under a cent on text-embedding-3-small.
"""

import os
import re
import csv
import json
import math
import argparse
import itertools
from collections import defaultdict

from dotenv import load_dotenv

import wat_fab
import wat_gender_brilliance as wgb
import wat_race_brilliance as wrb

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
            override=True)

OUT_DIR = os.path.join("responses", "embed")
DEFAULT_EMB_MODEL = "text-embedding-3-small"
PERM_CAP = 200_000          # exact permutation test below this many splits

# =========================================================
# Contextualisation — Bai's fragments, verified to be substrings of our prompts
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
    if context == "bare":
        return word
    return (TARGET_CTX if kind == "target" else ATTRIBUTE_CTX).format(w=word)


# =========================================================
# WEAT / WEFAT  (math from Caliskan et al. 2017; layout follows Bai's weat.py,
# permutation test reimplemented — see docstring note 5)
# =========================================================

def _unit(v):
    import numpy as np
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cos_sim(v1, v2):
    import numpy as np
    return np.matmul(_unit(np.asarray(v1, dtype=float)),
                     _unit(np.asarray(v2, dtype=float)).T)


def weat_association(W, A, B):
    """s(w, A, B) for every w in W."""
    import numpy as np
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)


def weat_test_stat(X, Y, A, B):
    """S(X,Y,A,B) — the raw differential association Bai actually reports."""
    import numpy as np
    return float(np.sum(weat_association(X, A, B)) - np.sum(weat_association(Y, A, B)))


def weat_effect_size(X, Y, A, B):
    """Caliskan's d.  Meaningful only when len(X), len(Y) > 1 — see docstring 1."""
    import numpy as np
    sx = weat_association(X, A, B)
    sy = weat_association(Y, A, B)
    sd = np.std(np.concatenate((sx, sy), axis=0), ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float((np.mean(sx) - np.mean(sy)) / sd)


def weat_p_value(X, Y, A, B, cap=PERM_CAP, seed=0):
    """
    Two-sided exact permutation p over equal-size splits of X u Y.
    Falls back to a seeded random subsample past `cap` splits.
    Returns (p, n_perms, exact?).  nan when len(X) != len(Y).
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
    Per-ITEM association: s(w,A,B) normalised by the SD of its own cosines.
    Caliskan's WEFAT.  `w` is one vector; returns a scalar.
    """
    import numpy as np
    w = np.asarray(w, dtype=float)[None, :]
    num = weat_association(w, A, B)[0]
    sd = np.std(np.concatenate((cos_sim(w, A).ravel(), cos_sim(w, B).ravel())), ddof=1)
    return float(num / sd) if sd else float("nan")


# =========================================================
# Embedding client + on-disk cache
# =========================================================

class Embedder:
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
            raise RuntimeError(f"--offline but {len(todo)} strings are uncached "
                               f"(e.g. {todo[0]!r})")
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
# Dataset definitions — stimulus sets + how to read a session record
# =========================================================

def _fab_arms():
    """
    One arm per pole set (leslie / bian) x FAB split.  The tertile arm keeps the
    runner's name so `session_arm` maps onto it; the median arm is set-level only
    (at item level FAB is continuous over all 30 fields, so no split applies).
    """
    splits = {
        "": (wat_fab.HIGH_TERT, wat_fab.LOW_TERT, "top 10", "bottom 10"),
        "_median": (wat_fab.HIGH_MEDIAN, wat_fab.LOW_MEDIAN, "top 15", "bottom 15"),
    }
    arms = {}
    for name, (bri, eff) in wat_fab.POLE_SETS.items():
        for suffix, (hi, lo, hname, lname) in splits.items():
            arms[name + suffix] = dict(
                X=list(bri), Y=list(eff),
                A=[f for f in wat_fab.FIELD_LABELS if f in hi],
                B=[f for f in wat_fab.FIELD_LABELS if f in lo],
                items=list(wat_fab.FIELD_LABELS),   # item level: all 30, continuous
                set_only=bool(suffix),
                X_name="brilliance poles", Y_name="effort poles",
                A_name=f"high-FAB fields ({hname})",
                B_name=f"low-FAB fields ({lname})")
    return arms


def _gb_arms():
    arms = {}
    for ctrl, words in wgb.CONTROL_SETS.items():
        bset, bwords = wgb.resolve_brilliance(ctrl)
        arms[ctrl] = dict(X=list(wgb.MALE_WORDS), Y=list(wgb.FEMALE_WORDS),
                          A=list(bwords), B=list(words),
                          X_name="male words", Y_name="female words",
                          A_name=f"brilliance ({bset})", B_name=f"control ({ctrl})")
    return arms


def _rb_arms():
    arms = {}
    for ctrl, words in wrb.CONTROL_SETS.items():
        bset, bwords = wgb.resolve_brilliance(ctrl)
        arms[ctrl] = dict(X=[wrb.WHITE_LABEL], Y=[wrb.BLACK_LABEL],
                          A=list(bwords), B=list(words),
                          X_name="White", Y_name="Black",
                          A_name=f"brilliance ({bset})", B_name=f"control ({ctrl})")
    return arms


def _assignments(record):
    """`assignments` is written as a JSON string by some runners, a dict by others."""
    a = record["assignments"]
    if isinstance(a, str):
        a = json.loads(a)
    return list(a.items())


DATASETS = {
    "fab": dict(
        dir=os.path.join("responses", "fab baseline"),
        glob="*_wat_fab.jsonl",
        arms=_fab_arms,
        # a session's own X / Y, and which arm it belongs to
        session_X=lambda r: [r["brilliance_word"]],
        session_Y=lambda r: [r["effort_word"]],
        session_arm=lambda r: ("bian" if r["brilliance_word"] in
                               wat_fab.BIAN_BRILLIANCE_WORDS else "leslie"),
        # behaviour: assignments map field -> "brilliance"/"effort"
        pos_label="brilliance",
        item_of=_assignments,
        varies=True,
    ),
    "gb": dict(
        dir=os.path.join("responses", "gb baseline"),
        glob="*_wat_gb.jsonl",
        arms=_gb_arms,
        session_X=lambda r: [r["male_word"]],
        session_Y=lambda r: [r["female_word"]],
        session_arm=lambda r: r["control_set"],
        pos_label="male",
        item_of=_assignments,
        varies=True,
    ),
    "rb": dict(
        dir=os.path.join("responses", "rb baseline"),
        glob="*_wat_rb.jsonl",
        arms=_rb_arms,
        session_X=lambda r: [r["white_label"]],
        session_Y=lambda r: [r["black_label"]],
        session_arm=lambda r: r["control_set"],
        pos_label="white",
        item_of=_assignments,
        varies=False,          # fixed labels -> no session-level variance
    ),
}


# =========================================================
# Behaviour, recomputed from raw `assignments`
# =========================================================

def session_D(record, spec, arm):
    """
    Bai's D on this session: P(pole+ | A-set) + P(pole- | B-set) - 1.
    Recomputed from `assignments`, never read off the stored column.
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


def load_sessions(spec, models=None):
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
                r["_file"] = os.path.basename(path)
                out.append(r)
    return out


# =========================================================
# The three levels
# =========================================================

def run_set_level(name, spec, arms, emb, rows):
    for arm_name, arm in arms.items():
        emb.prime(arm["X"] + arm["Y"], "target")
        emb.prime(arm["A"] + arm["B"], "attribute")
        X, Y = emb.mat(arm["X"], "target"), emb.mat(arm["Y"], "target")
        A, B = emb.mat(arm["A"], "attribute"), emb.mat(arm["B"], "attribute")
        p, nperm, exact = weat_p_value(X, Y, A, B)
        rows.append(dict(
            dataset=name, arm=arm_name, emb_model=emb.model, context=emb.context,
            n_X=len(X), n_Y=len(Y), n_A=len(A), n_B=len(B),
            test_stat=weat_test_stat(X, Y, A, B),
            effect_size=weat_effect_size(X, Y, A, B),
            p_perm=p, n_perm=nperm, exact=exact,
            X_name=arm["X_name"], Y_name=arm["Y_name"],
            A_name=arm["A_name"], B_name=arm["B_name"],
            degenerate=(len(X) < 2 or len(Y) < 2),
        ))


def run_session_level(name, spec, arms, emb, sessions, rows):
    """Bai-faithful 1-vs-1 SEAT per session, paired with that session's D."""
    for r in sessions:
        arm_name = spec["session_arm"](r)
        arm = arms.get(arm_name)
        if arm is None:
            continue
        xs, ys = spec["session_X"](r), spec["session_Y"](r)
        if not all(xs) or not all(ys):
            continue
        emb.prime(xs + ys, "target")
        emb.prime(arm["A"] + arm["B"], "attribute")
        X, Y = emb.mat(xs, "target"), emb.mat(ys, "target")
        A, B = emb.mat(arm["A"], "attribute"), emb.mat(arm["B"], "attribute")
        d = session_D(r, spec, arm)
        if d is None:
            continue
        rows.append(dict(
            dataset=name, arm=arm_name, emb_model=emb.model, context=emb.context,
            model=r.get("model"), trial=r.get("trial"),
            X_word=xs[0], Y_word=ys[0], pair=f"{xs[0]} / {ys[0]}",
            seat_test_stat=weat_test_stat(X, Y, A, B),
            behavioural_D=d,
        ))


def run_item_level(name, spec, arms, emb, sessions, rows):
    """WEFAT per sorted item + that item's behavioural P(pos), per model."""
    # behaviour: (arm, model, item) -> hits / n
    beh = defaultdict(lambda: [0, 0])
    for r in sessions:
        arm_name = spec["session_arm"](r)
        if arm_name not in arms:
            continue
        for item, label in spec["item_of"](r):
            key = (arm_name, r.get("model"), item)
            beh[key][1] += 1
            beh[key][0] += (label == spec["pos_label"])

    for arm_name, arm in arms.items():
        if arm.get("set_only"):
            continue
        items = arm.get("items") or (arm["A"] + arm["B"])
        emb.prime(items, "attribute")
        emb.prime(arm["X"] + arm["Y"], "target")
        Xm, Ym = emb.mat(arm["X"], "target"), emb.mat(arm["Y"], "target")
        models = sorted({m for (a, m, _) in beh if a == arm_name}, key=str)
        # arms with no sessions on disk still get their WEFAT row: the
        # wefat-vs-FAB-score correlation is stimulus-only and needs no behaviour.
        models = models or ["(no sessions)"]
        for item in items:
            # NOTE the role swap: at item level the SORTED word is the single
            # target and the two POLES are the attribute sets, so WEFAT reads
            # "how far toward pole X does this word sit".
            score = wefat_score(emb.get(item, "attribute"), Xm, Ym)
            for m in models:
                hits, n = beh.get((arm_name, m, item), [0, 0])
                rows.append(dict(
                    dataset=name, arm=arm_name, emb_model=emb.model,
                    context=emb.context, model=m, item=item,
                    in_set=("A" if item in set(arm["A"])
                            else "B" if item in set(arm["B"]) else "mid"),
                    wefat=score,
                    p_pos=(hits / n) if n else float("nan"), n_sessions=n,
                    fab_score=wat_fab.FAB_SCORE.get(item),
                ))


# =========================================================
# Reporting
# =========================================================

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
    """Regularised incomplete beta I_x(a,b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf2(t, df):
    """Two-sided Student-t p.  scipy is not installed in this project's env."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def pearson(xs, ys):
    import numpy as np
    x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0, len(x)
    t = r * math.sqrt((len(x) - 2) / (1 - r * r))
    return r, t_sf2(t, len(x) - 2), len(x)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def star(p):
    if not (p == p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def report(set_rows, sess_rows, item_rows):
    print()
    print("=" * 78)
    print("SET LEVEL — WEAT over the full stimulus sets")
    print("=" * 78)
    print(f"{'dataset':5} {'arm':12} {'ctx':7} {'S':>9} {'d':>7} {'p':>9}  sets")
    for r in set_rows:
        flag = " [1v1: d not interpretable]" if r["degenerate"] else ""
        print(f"{r['dataset']:5} {r['arm']:12} {r['context']:7} "
              f"{r['test_stat']:+9.4f} {r['effect_size']:+7.3f} "
              f"{r['p_perm']:9.4f}{star(r['p_perm']):<3} "
              f"{r['X_name']} vs {r['Y_name']} | {r['A_name']} vs {r['B_name']}{flag}")

    if sess_rows:
        print()
        print("=" * 78)
        print("SESSION LEVEL — Bai Fig.4: does embedding bias track behaviour?")
        print("=" * 78)
        by = defaultdict(list)
        for r in sess_rows:
            by[(r["dataset"], r["context"], r["model"])].append(r)
        print(f"{'dataset':5} {'ctx':7} {'model':26} {'n':>5} {'r(emb,D)':>9} {'p':>9} "
              f"{'#pairs':>7}")
        for k in sorted(by, key=lambda t: (t[0], t[1], str(t[2]))):
            rs = by[k]
            r_, p_, n = pearson([x["seat_test_stat"] for x in rs],
                                [x["behavioural_D"] for x in rs])
            npair = len({x["pair"] for x in rs})
            print(f"{k[0]:5} {k[1]:7} {str(k[2]):26} {n:5d} {r_:+9.3f} "
                  f"{p_:9.4f}{star(p_):<3} {npair:7d}")

    if item_rows:
        print()
        print("=" * 78)
        print("ITEM LEVEL — WEFAT per sorted word")
        print("=" * 78)
        by = defaultdict(list)
        for r in item_rows:
            by[(r["dataset"], r["arm"], r["context"], r["model"])].append(r)
        print(f"{'dataset':5} {'arm':12} {'ctx':7} {'model':26} {'n':>4} "
              f"{'r(wefat,P)':>10} {'p':>8} {'r(wefat,FAB)':>12}")
        for k in sorted(by, key=lambda t: (t[0], t[1], t[2], str(t[3]))):
            rs = by[k]
            r1, p1, n = pearson([x["wefat"] for x in rs], [x["p_pos"] for x in rs])
            have_fab = [x for x in rs if x["fab_score"] is not None]
            r2 = pearson([x["wefat"] for x in have_fab],
                         [x["fab_score"] for x in have_fab])[0] if have_fab else float("nan")
            print(f"{k[0]:5} {k[1]:12} {k[2]:7} {str(k[3]):26} {n:4d} "
                  f"{r1:+10.3f} {p1:8.4f}{star(p1):<3} {r2:+12.3f}")

        # A raw item-level r can be carried entirely by SET MEMBERSHIP: brilliance
        # words sit on one side of both measures and control words on the other,
        # so r is high without the two measures agreeing on any within-set ranking.
        # Re-centring wefat and P within each set removes that and asks the harder
        # question -- among the 4 brilliance words alone, does geometry predict
        # which one the model sends male?  (Directly relevant to the
        # "brilliant"=jewellery word-sense check.)
        print()
        print("pooled over arms x models; n counts item x model CELLS, so the same "
              "wefat value\nrecurs across models -- these p-values are "
              "anti-conservative, read the r.")
        print(f"{'dataset':5} {'ctx':7} {'pooled raw r':>13} {'p':>9} "
              f"{'within-set r':>13} {'p':>9} {'n':>5}")
        pooled = defaultdict(lambda: ([], []))
        for (ds, arm, ctx, m), rs in by.items():
            raw, cen = pooled[(ds, ctx)]
            usable = [x for x in rs if x["n_sessions"]]
            raw.extend((x["wefat"], x["p_pos"]) for x in usable)
            for grp in {x["in_set"] for x in usable}:
                sub = [x for x in usable if x["in_set"] == grp]
                if len(sub) < 2:
                    continue
                mw = sum(x["wefat"] for x in sub) / len(sub)
                mp = sum(x["p_pos"] for x in sub) / len(sub)
                cen.extend((x["wefat"] - mw, x["p_pos"] - mp) for x in sub)
        for k in sorted(pooled):
            raw, cen = pooled[k]
            r1, p1, n = pearson([a for a, _ in raw], [b for _, b in raw])
            r2, p2, _ = pearson([a for a, _ in cen], [b for _, b in cen])
            print(f"{k[0]:5} {k[1]:7} {r1:+13.3f} {p1:9.4f}{star(p1):<3} "
                  f"{r2:+13.3f} {p2:9.4f}{star(p2):<3} {n:5d}")


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=list(DATASETS),
                    help="repeatable; default = all")
    ap.add_argument("--emb-model", default=DEFAULT_EMB_MODEL)
    ap.add_argument("--context", default="prompt", choices=["prompt", "bare", "both"],
                    help="prompt = Bai's WAT-fragment contextualisation (his choice)")
    ap.add_argument("--model", action="append",
                    help="restrict behaviour to these LLMs (repeatable)")
    ap.add_argument("--arm", action="append", help="restrict to these arms (repeatable)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--offline", action="store_true",
                    help="fail instead of calling the API if anything is uncached")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the inventory and exit without embedding anything")
    args = ap.parse_args()

    datasets = args.dataset or list(DATASETS)
    contexts = ["prompt", "bare"] if args.context == "both" else [args.context]
    os.makedirs(args.out, exist_ok=True)

    set_rows, sess_rows, item_rows = [], [], []

    for name in datasets:
        spec = DATASETS[name]
        arms = spec["arms"]()
        if args.arm:
            arms = {k: v for k, v in arms.items() if k in args.arm}
            if not arms:
                print(f"[{name}] no arm matches {args.arm} — skipped"); continue
        sessions = load_sessions(spec, set(args.model) if args.model else None)
        used = {spec["session_arm"](r) for r in sessions} & set(arms)
        nmod = len({r.get("model") for r in sessions})
        print(f"[{name}] {len(sessions)} sessions, {nmod} models, "
              f"{len(arms)} arms defined, {len(used)} with data: {sorted(used)}")
        if not spec["varies"]:
            print(f"[{name}] targets are fixed labels -> session level skipped "
                  f"(no between-session variance in the embedding bias)")
        if args.dry_run:
            for a, arm in arms.items():
                print(f"   arm {a:12} X={len(arm['X'])} Y={len(arm['Y'])} "
                      f"A={len(arm['A'])} B={len(arm['B'])}")
            continue

        for ctx in contexts:
            emb = Embedder(args.emb_model, ctx, args.out, offline=args.offline)
            run_set_level(name, spec, arms, emb, set_rows)
            if spec["varies"]:
                run_session_level(name, spec, arms, emb, sessions, sess_rows)
            run_item_level(name, spec, arms, emb, sessions, item_rows)
            if emb.n_new:
                print(f"[{name}] {ctx}: embedded {emb.n_new} new strings "
                      f"({args.emb_model})")

    if args.dry_run:
        return

    tag = args.emb_model.replace("text-embedding-", "")
    write_csv(os.path.join(args.out, f"set_level__{tag}.csv"), set_rows)
    write_csv(os.path.join(args.out, f"session_level__{tag}.csv"), sess_rows)
    write_csv(os.path.join(args.out, f"item_wefat__{tag}.csv"), item_rows)
    report(set_rows, sess_rows, item_rows)
    print(f"\nwrote -> {args.out}/{{set_level,session_level,item_wefat}}__{tag}.csv")


if __name__ == "__main__":
    main()
