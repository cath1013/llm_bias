#!/usr/bin/env python3
"""
FROZEN EXPERIMENTAL DESIGN — generate `embedding/prompts.csv` from `stimuli.csv`.

WHAT THIS IS
    The design manifest for the fixed-prompt Bai-style WAT experiment.  Every
    prompt Llama 3.1 will ever see in that experiment is written here, once,
    deterministically, and then frozen.  The hidden-state runner reads this file
    and randomises nothing of its own.

    Separating generation from execution is the point: with the design on disk,
    the stimulus pairing, the target order and the attribute order are all
    auditable before a single GPU second is spent, and a re-run cannot quietly
    draw a different sample.

SOURCE OF TRUTH
    `embedding/stimuli.csv` supplies every word.  Nothing is hardcoded here
    except the ASSERTIONS that cross-check that file against the behavioural
    runners (`wat_gender_brilliance`, `wat_fab`) it was transcribed from.  If
    the two ever disagree, this script stops.

PROMPT WORDING
    Taken verbatim from the behavioural runners — `wgb.build_prompt()` and
    `wat_fab.build_prompt(..., strict=False)`.  No new template is invented, so
    the fixed-prompt experiment asks exactly the question the API runs asked.

DESIGN
    Gender-brilliance, four conditions (creativity / warmth / humor /
    dedication), each with the 4-word Storage brilliance set:
        8 male x 8 female = 64 target pairs, EVERY pair exactly once
        4 conditions x 64 = 256 prompts
    FAB, Leslie poles, tertile fields only:
        4 brilliance x 4 effort = 16 pole pairs, EVERY pair exactly once
        20 fields = top-10 high-FAB + bottom-10 low-FAB; the middle tertile is
        not presented and no median-split prompt is generated
    Total 272 prompts.

    N is fixed by exhaustive stimulus coverage, not by an arbitrary draw.  There
    is no "n=50 trials" here to power-analyse: the design IS the population of
    pairs.

COUNTERBALANCING
    Target order — parity over the pair grid.  Pair (i, j) presents the
    stereotype-congruent target first iff (i + j) is even.  Over a square
    crossing that gives an exact split (gender 32/32, FAB 8/8) and makes every
    individual word appear first and second equally often (gender 4/4, FAB 2/2).

    Attribute order — seeded cyclic Latin squares.  A shuffled base ordering and
    its k rotations put each word in each serial position exactly once per
    square; for gender that is 8 squares over 64 prompts, so every word sits in
    every position exactly 8 times.  For FAB, 16 prompts cannot evenly fill 20
    positions, so 16 of the 20 rotations of one base are used: each field then
    occupies 16 distinct positions once each, which is as even as 16 x 20 can
    be.  The resulting orderings are shuffled before they are attached to pairs,
    so attribute order carries no information about which pair it landed on.

    `random.shuffle()` called independently per prompt is NOT used: it balances
    nothing and cannot be checked.  Every balance claim above is asserted.

DETERMINISM
    One `random.Random(seed)` consumed in a fixed order.  Same seed, byte-
    identical prompts.csv.

USAGE
    python3 embedding/make_bai_style_prompts.py
    python3 embedding/make_bai_style_prompts.py --seed 0 --out embedding/prompts.csv
    python3 embedding/make_bai_style_prompts.py --check     # verify, write nothing
"""

import os
import csv
import json
import random
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import wat_fab
import wat_gender_brilliance as wgb

STIMULI_CSV = os.path.join(HERE, "stimuli.csv")
PROMPTS_CSV = os.path.join(HERE, "prompts.csv")

COLUMNS = ["prompt_id", "domain", "category", "trial",
           "group0", "group1", "attributes", "prompt"]

# Clean labels for the frozen design.  The behavioural runners call these sets
# `creativity4` and `dedication_adj` internally; prompts.csv uses the short
# names and `CONDITION_SOURCE` records which internal set each one is.
GENDER_CONDITIONS = ("creativity", "warmth", "humor", "dedication")
CONDITION_SOURCE = {"creativity": "creativity4", "warmth": "warmth",
                    "humor": "humor", "dedication": "dedication_adj"}

N_GENDER_PER_CONDITION = 64
N_FAB = 16
N_TOTAL = N_GENDER_PER_CONDITION * len(GENDER_CONDITIONS) + N_FAB


# =========================================================
# stimuli.csv
# =========================================================

def load_stimuli(path=STIMULI_CSV):
    """
    Parse `stimuli.csv` into the role lists the design needs, IN FILE ORDER.

    File order matters: it is what makes the pair grid, and therefore the whole
    design, reproducible from the CSV alone.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"dataset", "condition", "stimulus_type", "role", "stimulus"}
    missing = need - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")

    def pick(dataset, condition, stype, role):
        return [r["stimulus"].strip() for r in rows
                if r["dataset"] == dataset and r["condition"] == condition
                and r["stimulus_type"] == stype and r["role"] == role]

    stim = {
        "gender": {
            "male": pick("gender_brilliance", "all", "target", "male"),
            "female": pick("gender_brilliance", "all", "target", "female"),
            "brilliance": pick("gender_brilliance", "all", "attribute", "brilliance"),
            "control": {c: pick("gender_brilliance", c, "attribute", "control")
                        for c in GENDER_CONDITIONS},
        },
        "fab": {
            "brilliance": pick("fab", "all", "pole", "brilliance"),
            "effort": pick("fab", "all", "pole", "effort"),
            "high_fab": pick("fab", "tertile", "field", "high_fab"),
            "low_fab": pick("fab", "tertile", "field", "low_fab"),
        },
    }
    validate_stimuli(stim, path)
    return stim


def validate_stimuli(stim, path=STIMULI_CSV):
    """
    Cross-check stimuli.csv against the behavioural runners it was transcribed
    from.  This is the only place the word lists are named twice, and it exists
    so that a silent edit to either side is caught here rather than in a result.
    """
    g, f = stim["gender"], stim["fab"]

    def same(got, want, label):
        if set(got) != set(want):
            raise SystemExit(
                f"{path}: {label} disagrees with the behavioural runner.\n"
                f"  csv    : {sorted(got)}\n  runner : {sorted(want)}")

    same(g["male"], wgb.MALE_WORDS, "gender male targets")
    same(g["female"], wgb.FEMALE_WORDS, "gender female targets")
    same(g["brilliance"], wgb.BRILLIANCE_SETS["storage4"], "brilliance attributes")
    for cond in GENDER_CONDITIONS:
        same(g["control"][cond], wgb.CONTROL_SETS[CONDITION_SOURCE[cond]],
             f"{cond} control attributes")
    if "original" not in g["control"]["creativity"]:
        raise SystemExit("the creativity condition must be the 4-word set "
                         "containing 'original' (creativity4), not creativity3 "
                         "and not a synonym variant")

    leslie_bri, leslie_eff = wat_fab.POLE_SETS["leslie"]
    same(f["brilliance"], leslie_bri, "FAB brilliance poles")
    same(f["effort"], leslie_eff, "FAB effort poles")
    bian_bri, bian_eff = wat_fab.POLE_SETS["bian"]
    bian = {w.lower() for w in list(bian_bri) + list(bian_eff)}
    stray = bian & {w.lower() for w in f["brilliance"] + f["effort"]}
    if stray:
        raise SystemExit(f"{path}: Bian pole words present ({sorted(stray)}); "
                         f"this experiment is Leslie-poles only")
    same(f["high_fab"], wat_fab.HIGH_TERT, "high-FAB tertile fields")
    same(f["low_fab"], wat_fab.LOW_TERT, "low-FAB tertile fields")

    middle = set(wat_fab.FIELD_LABELS) - set(wat_fab.HIGH_TERT) - set(wat_fab.LOW_TERT)
    overlap = middle & set(f["high_fab"] + f["low_fab"])
    if overlap:
        raise SystemExit(f"{path}: middle-tertile field(s) {sorted(overlap)} "
                         f"present; the tertile design excludes them")

    counts = {"male": (g["male"], 8), "female": (g["female"], 8),
              "brilliance": (g["brilliance"], 4),
              "fab brilliance poles": (f["brilliance"], 4),
              "fab effort poles": (f["effort"], 4),
              "high-FAB fields": (f["high_fab"], 10),
              "low-FAB fields": (f["low_fab"], 10)}
    for cond in GENDER_CONDITIONS:
        counts[f"{cond} control"] = (g["control"][cond], 4)
    for label, (words, n) in counts.items():
        if len(words) != n or len(set(words)) != n:
            raise SystemExit(f"{path}: {label} must be {n} distinct stimuli, "
                             f"got {len(words)} ({len(set(words))} distinct)")
    return stim


# =========================================================
# Counterbalancing
# =========================================================

def balanced_attribute_orders(words, n_prompts, rng):
    """
    `n_prompts` orderings of `words` whose serial-position occupancy is as even
    as the arithmetic allows.

    A shuffled base and its k cyclic rotations form a Latin square: within one
    square every word occupies every position exactly once.  Whole squares are
    emitted until `n_prompts` is reached; a final partial square (FAB: 16 of 20
    rotations) leaves every word in 16 DISTINCT positions once each, so no cell
    of the word x position table differs from another by more than one.

    The returned list is shuffled, so an ordering carries no information about
    which stimulus pair it will be attached to.
    """
    k = len(words)
    orders = []
    while len(orders) < n_prompts:
        base = list(words)
        rng.shuffle(base)
        for r in range(k):
            if len(orders) == n_prompts:
                break
            orders.append(tuple(base[r:] + base[:r]))
    rng.shuffle(orders)
    return orders


def position_table(orders, words):
    """{word: [count at position 0, 1, ...]} over a list of orderings."""
    table = {w: [0] * len(words) for w in words}
    for order in orders:
        for pos, w in enumerate(order):
            table[w][pos] += 1
    return table


def assert_position_balance(orders, words, label):
    """Every word x position cell within one of every other.  Hard failure."""
    table = position_table(orders, words)
    cells = [c for counts in table.values() for c in counts]
    if max(cells) - min(cells) > 1:
        raise AssertionError(
            f"{label}: serial-position occupancy is unbalanced "
            f"(min {min(cells)}, max {max(cells)}); "
            f"the cyclic-rotation design did not hold")
    for w, counts in table.items():
        if sum(counts) != len(orders):
            raise AssertionError(f"{label}: {w!r} appears {sum(counts)} times "
                                 f"in {len(orders)} prompts")
    return table


def pair_grid(first_pool, second_pool):
    """
    Every (a, b) pair exactly once, with presentation order counterbalanced by
    index parity.

    Yields (a, b, a_first).  `a_first` alternates along both axes, so over a
    square grid each individual word leads in exactly half of its own pairs and
    the overall split is exact.  The parity rule is deterministic and checkable;
    a coin flip per trial is neither.
    """
    for i, a in enumerate(first_pool):
        for j, b in enumerate(second_pool):
            yield a, b, (i + j) % 2 == 0


# =========================================================
# Prompt rows
# =========================================================

def gender_rows(condition, stim, rng):
    """64 prompts for one gender condition: every male x female pair once."""
    male, female = stim["gender"]["male"], stim["gender"]["female"]
    attrs_pool = list(stim["gender"]["brilliance"]) + list(stim["gender"]["control"][condition])
    orders = balanced_attribute_orders(attrs_pool, N_GENDER_PER_CONDITION, rng)
    assert_position_balance(orders, attrs_pool, f"gender/{condition} attributes")

    rows = []
    for trial, (m, f, male_first) in enumerate(pair_grid(male, female), start=1):
        g0, g1 = (m, f) if male_first else (f, m)
        attrs = list(orders[trial - 1])
        rows.append({
            "prompt_id": f"gb_{condition}_{trial}",
            "domain": "gender",
            "category": condition,
            "trial": trial,
            "group0": g0,
            "group1": g1,
            "attributes": json.dumps(attrs, ensure_ascii=False),
            # verbatim from the behavioural runner
            "prompt": wgb.build_prompt([g0, g1], attrs),
        })
    return rows


def fab_rows(stim, rng):
    """16 prompts: every brilliance x effort pole pair once, 20 tertile fields."""
    bri, eff = stim["fab"]["brilliance"], stim["fab"]["effort"]
    fields_pool = list(stim["fab"]["high_fab"]) + list(stim["fab"]["low_fab"])
    orders = balanced_attribute_orders(fields_pool, N_FAB, rng)
    assert_position_balance(orders, fields_pool, "fab fields")

    rows = []
    for trial, (b, e, bri_first) in enumerate(pair_grid(bri, eff), start=1):
        g0, g1 = (b, e) if bri_first else (e, b)
        fields = list(orders[trial - 1])
        rows.append({
            "prompt_id": f"fab_{trial}",
            "domain": "fab",
            "category": "fab",
            "trial": trial,
            "group0": g0,
            "group1": g1,
            "attributes": json.dumps(fields, ensure_ascii=False),
            # verbatim from the behavioural runner; strict=False is the wording
            # the reported FAB sessions used
            "prompt": wat_fab.build_prompt([g0, g1], fields, strict=False),
        })
    return rows


def build_prompts(stim, seed=0):
    """The whole frozen design.  One RNG, one fixed consumption order."""
    rng = random.Random(seed)
    rows = []
    for condition in GENDER_CONDITIONS:          # fixed order == reproducible
        rows.extend(gender_rows(condition, stim, rng))
    rows.extend(fab_rows(stim, rng))
    validate_design(rows, stim)
    return rows


# =========================================================
# Design validation — hard assertions, not a printed summary
# =========================================================

def validate_design(rows, stim):
    if len(rows) != N_TOTAL:
        raise AssertionError(f"expected {N_TOTAL} prompts, got {len(rows)}")
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise AssertionError(f"duplicate prompt_id(s): {dupes}")

    gender = [r for r in rows if r["domain"] == "gender"]
    fab = [r for r in rows if r["domain"] == "fab"]
    if len(gender) != 256 or len(fab) != 16:
        raise AssertionError(f"expected 256 gender + 16 fab, got "
                             f"{len(gender)} + {len(fab)}")

    male, female = set(stim["gender"]["male"]), set(stim["gender"]["female"])
    for condition in GENDER_CONDITIONS:
        sub = [r for r in rows if r["category"] == condition]
        if len(sub) != N_GENDER_PER_CONDITION:
            raise AssertionError(f"{condition}: {len(sub)} prompts, "
                                 f"expected {N_GENDER_PER_CONDITION}")
        if [r["trial"] for r in sub] != list(range(1, N_GENDER_PER_CONDITION + 1)):
            raise AssertionError(f"{condition}: trials are not 1..64 in order")
        if [r["prompt_id"] for r in sub] != [f"gb_{condition}_{t}"
                                             for t in range(1, 65)]:
            raise AssertionError(f"{condition}: prompt_id pattern broken "
                                 f"(expected gb_{condition}_1 .. _64, unpadded)")

        pairs = Counter()
        first = Counter()
        for r in sub:
            m = r["group0"] if r["group0"] in male else r["group1"]
            f = r["group1"] if r["group0"] in male else r["group0"]
            if m not in male or f not in female:
                raise AssertionError(f"{r['prompt_id']}: targets {r['group0']!r}/"
                                     f"{r['group1']!r} are not one male + one female")
            pairs[(m, f)] += 1
            first[r["group0"]] += 1
        if set(pairs) != {(m, f) for m in male for f in female}:
            raise AssertionError(f"{condition}: the 8x8 male x female crossing "
                                 f"is incomplete")
        if set(pairs.values()) != {1}:
            raise AssertionError(f"{condition}: some target pair occurs more than once")
        male_first = sum(n for w, n in first.items() if w in male)
        if male_first != 32 or len(sub) - male_first != 32:
            raise AssertionError(f"{condition}: target order is {male_first}/"
                                 f"{len(sub) - male_first}, expected 32/32")
        for w in male | female:
            if first[w] != 4:
                raise AssertionError(f"{condition}: {w!r} leads {first[w]} times, "
                                     f"expected 4 (and 4 as group1)")

        attrs_pool = set(stim["gender"]["brilliance"]) | set(stim["gender"]["control"][condition])
        for r in sub:
            got = json.loads(r["attributes"])
            if len(got) != 8 or set(got) != attrs_pool:
                raise AssertionError(f"{r['prompt_id']}: attributes are not the "
                                     f"4 brilliance + 4 {condition} words")
        assert_position_balance([tuple(json.loads(r["attributes"])) for r in sub],
                                sorted(attrs_pool), f"{condition} attributes (csv)")

    bri, eff = set(stim["fab"]["brilliance"]), set(stim["fab"]["effort"])
    if [r["prompt_id"] for r in fab] != [f"fab_{t}" for t in range(1, 17)]:
        raise AssertionError("fab prompt_id pattern broken (expected fab_1 .. "
                             "fab_16, unpadded, with no 'tertile' in the id)")
    pairs, first = Counter(), Counter()
    for r in fab:
        b = r["group0"] if r["group0"] in bri else r["group1"]
        e = r["group1"] if r["group0"] in bri else r["group0"]
        if b not in bri or e not in eff:
            raise AssertionError(f"{r['prompt_id']}: poles {r['group0']!r}/"
                                 f"{r['group1']!r} are not one brilliance + one effort")
        pairs[(b, e)] += 1
        first[r["group0"]] += 1
    if set(pairs) != {(b, e) for b in bri for e in eff} or set(pairs.values()) != {1}:
        raise AssertionError("the 4x4 brilliance x effort crossing is incomplete "
                             "or repeats a pair")
    bri_first = sum(n for w, n in first.items() if w in bri)
    if bri_first != 8 or len(fab) - bri_first != 8:
        raise AssertionError(f"fab target order is {bri_first}/{len(fab)-bri_first}, "
                             f"expected 8/8")
    for w in bri | eff:
        if first[w] != 2:
            raise AssertionError(f"fab: {w!r} leads {first[w]} times, expected 2")

    high, low = set(stim["fab"]["high_fab"]), set(stim["fab"]["low_fab"])
    middle = set(wat_fab.FIELD_LABELS) - high - low
    for r in fab:
        got = json.loads(r["attributes"])
        if len(got) != 20 or len(set(got)) != 20:
            raise AssertionError(f"{r['prompt_id']}: expected 20 distinct fields")
        if len(set(got) & high) != 10 or len(set(got) & low) != 10:
            raise AssertionError(f"{r['prompt_id']}: expected 10 high-FAB + "
                                 f"10 low-FAB fields")
        if set(got) & middle:
            raise AssertionError(f"{r['prompt_id']}: middle-tertile field(s) "
                                 f"{sorted(set(got) & middle)} presented")
    assert_position_balance([tuple(json.loads(r["attributes"])) for r in fab],
                            sorted(high | low), "fab fields (csv)")

    # the prompt string must actually contain what the row claims
    for r in rows:
        for w in (r["group0"], r["group1"], *json.loads(r["attributes"])):
            if w.lower() not in r["prompt"].lower():
                raise AssertionError(f"{r['prompt_id']}: {w!r} is not in the prompt")
    return rows


# =========================================================
# I/O
# =========================================================

def write_prompts(rows, path=PROMPTS_CSV):
    """Byte-stable: fixed column order, LF line endings, no float formatting."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})
    return path


def read_prompts(path=PROMPTS_CSV):
    """The frozen design, with the schema enforced on the way in."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path}: columns are {reader.fieldnames}, "
                             f"expected exactly {COLUMNS}")
        rows = [dict(r) for r in reader]
    for r in rows:
        r["trial"] = int(r["trial"])
    return rows


def summarise(rows):
    out = [f"{len(rows)} prompts"]
    by_cat = Counter((r["domain"], r["category"]) for r in rows)
    for (dom, cat), n in sorted(by_cat.items()):
        sub = [r for r in rows if r["domain"] == dom and r["category"] == cat]
        pairs = len({(r["group0"], r["group1"]) for r in sub})
        out.append(f"  {dom}/{cat:<11} n={n:<4} distinct ordered pairs={pairs}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stimuli", default=STIMULI_CSV)
    ap.add_argument("--out", default=PROMPTS_CSV)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="build and validate the design, write nothing")
    args = ap.parse_args()

    stim = load_stimuli(args.stimuli)
    rows = build_prompts(stim, seed=args.seed)
    print(f"stimuli : {args.stimuli}")
    print(f"seed    : {args.seed}")
    print(summarise(rows))
    if args.check:
        print("--check: design validated, nothing written.")
        return
    write_prompts(rows, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
