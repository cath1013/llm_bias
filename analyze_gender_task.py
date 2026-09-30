"""
ANALYZE GENDER TASK — stage 4: paired, item-clustered analysis of a gender_task.py run.

Separate from the runner on purpose (matching analyze_fab_gender.py /
analyze_brilliance_word.py) so results can be re-derived without spending calls.
Everything parseable is RECOMPUTED FROM `raw_response` + the recorded `q_order` and
`topic_text` (CLAUDE.md: recompute-from-raw); only the freetext coder's ratings, which
required their own API call, are read from the stored columns.

THE ESTIMAND
    Δ = rating(male name) − rating(female name) for the SAME cell, where a cell is one
    (item, correctness, style, framing, mode, run) and the two calls differ ONLY in the
    name. Then:
        per-cell Δ  ->  per-ITEM mean Δ  ->  t and a 95% clustered bootstrap CI
    Clustering by item is not optional. Item variance dwarfs the gender effect, so a
    test over cells treats correlated observations as independent and understates the
    standard error. Both are printed; the ITEM-level one is the evidence.

    male − control and female − control are also reported: a Δ alone cannot tell you
    whether the effect is a female penalty, a male boost, or both.

MODERATORS come in two kinds and the code picks automatically:
    within-item   framing / variant / style — the same item appears at both levels, so
                  the interaction is a paired difference of that item's two Δs.
    between-item  fab_level / category — items are NESTED in the level, so there is no
                  within-item pairing and the contrast is a two-sample comparison of the
                  two item-level Δ distributions. (An item-wise join here silently
                  returns nothing, which is how this was found.)

ALSO REPORTED
    lexicon coverage   per ref/wan_method.md: how much of the corpus the lexicon
                       actually touches. Near-zero coverage means the lexicon DVs are
                       noise and the coder DVs are carrying the freetext arm.
    error detection    on flawed items, hit rate by condition (is a woman's error caught
                       more often?); on correct items, the false-alarm rate.
    refusals           by condition, as in Poole-Dayan et al.

USAGE
    python analyze_gender_task.py --file "responses/gender task/<...>_gender_task.xlsx"
    python analyze_gender_task.py --file "responses/gender task/<...>_raw.jsonl"
"""

import os
import csv
import json
import math
import random
import argparse
import statistics

import pandas as pd

from gender_task import (OUTPUT_DIR, parse_likert, mask_identity,
                         second_person_text, lex_count)
from fab_decision import _prop_z
from bai_core import _slug

# r_attribution = Q1, whichever construct was asked (the construct is its own column).
LIKERT_KEYS = ["attribution", "potential"]
LIKERT_DVS = ["r_attribution", "r_potential"]
# `_adj` = topic-filtered + restricted to second-person sentences -> PRIMARY.
# The unfiltered `_raw` versions are kept alongside so the filtering is auditable.
FREETEXT_DVS = ["lex_innate_adj_p100", "lex_ability_adj_p100", "lex_effort_adj_p100",
                "lex_index_adj", "j_ability", "j_effort", "j_positivity",
                "j_brill_minus_effort",
                "lex_innate_raw_p100", "lex_effort_raw_p100", "lex_index_raw"]

# =========================================================
# Recompute every parsed column from the raw responses
# =========================================================

def derive(df):
    df = df.copy()
    for c in ("verdict_yes", "n_parsed", "refused") + tuple(f"r_{k}" for k in LIKERT_KEYS):
        df[c] = None
    for i, row in df.iterrows():
        raw = row.get("raw_response") or ""
        if row.get("mode") == "likert":
            q_map = str(row.get("q_order") or "").split(",")
            vals, n = parse_likert(raw, q_map)
            for k, v in vals.items():
                df.at[i, k] = v
            df.at[i, "n_parsed"] = n
            df.at[i, "refused"] = int(n == 0)
        else:
            masked, _ = mask_identity(raw)
            sp = second_person_text(masked)
            topic = row.get("topic_text") or ""
            for which in ("innate", "ability", "effort"):
                df.at[i, f"lex_{which}"] = lex_count(masked, which)[0]
                df.at[i, f"lex_{which}_adj"] = lex_count(sp, which, topic)[0]
            df.at[i, "n_words_response"] = len(raw.split())
            df.at[i, "n_words_2p"] = len(sp.split())
            df.at[i, "refused"] = int(len(raw.split()) < 15)

    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(
        [None] * len(df), index=df.index, dtype="float64")
    w_all = num("n_words_response").replace(0, pd.NA)
    w_2p = num("n_words_2p").replace(0, pd.NA)
    for which in ("innate", "ability", "effort"):
        df[f"lex_{which}_raw_p100"] = 100 * num(f"lex_{which}") / w_all
        df[f"lex_{which}_adj_p100"] = 100 * num(f"lex_{which}_adj") / w_2p
    for suffix, a, b in (("raw", "lex_innate", "lex_effort"),
                         ("adj", "lex_innate_adj", "lex_effort_adj")):
        tot = num(a) + num(b)
        df[f"lex_index_{suffix}"] = (num(a) - num(b)) / tot.replace(0, pd.NA)
    df["j_brill_minus_effort"] = num("j_ability") - num("j_effort")
    for c in LIKERT_DVS + FREETEXT_DVS + ["verdict_yes", "refused"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================================================
# Stats. p-values use a normal approximation, as bai_core does (scipy is not installed
# in this project's environment).
# =========================================================

def _t_p(vals):
    """(n, mean, t, p) — one-sample vs 0."""
    v = [x for x in vals if x is not None and not pd.isna(x)]
    n = len(v)
    if n < 2:
        return n, (v[0] if n else float("nan")), float("nan"), float("nan")
    m = statistics.fmean(v)
    sd = statistics.stdev(v)
    if sd == 0:
        return n, m, float("inf") if m else 0.0, 0.0 if m else 1.0
    t = m / (sd / math.sqrt(n))
    return n, m, t, math.erfc(abs(t) / math.sqrt(2))

def _boot_ci(item_means, B=5000, seed=7):
    """95% CI by resampling ITEMS with replacement (the cluster)."""
    v = [x for x in item_means if not pd.isna(x)]
    if len(v) < 3:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    ms = sorted(statistics.fmean(rng.choices(v, k=len(v))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]

def _two_sample(a, b, B=5000, seed=11):
    """Welch two-sample on two independent item-level delta sets + bootstrap CI of the
    difference. For moderators where items are NESTED in the level (fab_level)."""
    a = [x for x in a if not pd.isna(x)]
    b = [x for x in b if not pd.isna(x)]
    if len(a) < 2 or len(b) < 2:
        return len(a), len(b), float("nan"), float("nan"), float("nan"), \
            float("nan"), float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    diff = ma - mb
    se = math.sqrt(va + vb)
    t = diff / se if se > 0 else (float("inf") if diff else 0.0)
    p = math.erfc(abs(t) / math.sqrt(2)) if se > 0 else (0.0 if diff else 1.0)
    rng = random.Random(seed)
    ds = sorted(statistics.fmean(rng.choices(a, k=len(a)))
                - statistics.fmean(rng.choices(b, k=len(b))) for _ in range(B))
    return len(a), len(b), diff, t, p, ds[int(0.025 * B)], ds[int(0.975 * B)]

def _paired(df, dv, hi="male", lo="female"):
    """Cell-level delta (hi - lo) for dv, tagged with its item. One row per cell."""
    sub = df[df["condition"].isin([hi, lo])][["cell_id", "item_id", "condition", dv]]
    sub = sub.dropna(subset=[dv])
    p = sub.pivot_table(index=["cell_id", "item_id"], columns="condition",
                        values=dv, aggfunc="mean")
    if hi not in p or lo not in p:
        return pd.DataFrame(columns=["item_id", "delta"])
    p = p.dropna(subset=[hi, lo]).reset_index()
    p["delta"] = p[hi] - p[lo]
    return p[["item_id", "delta"]]

def _report_paired(df, dv, label, hi="male", lo="female"):
    d = _paired(df, dv, hi, lo)
    if d.empty:
        return None
    im = d.groupby("item_id")["delta"].mean()
    n_i, m_i, t_i, p_i = _t_p(im.tolist())
    n_c, m_c, t_c, _ = _t_p(d["delta"].tolist())
    lo_ci, hi_ci = _boot_ci(im.tolist())
    star = "***" if p_i < .001 else "**" if p_i < .01 else "*" if p_i < .05 else \
           "." if p_i < .1 else " "
    print(f"  {label:24s} d={m_i:+.3f} [{lo_ci:+.3f},{hi_ci:+.3f}] "
          f"t({n_i - 1})={t_i:+.2f} p={p_i:.3f}{star:3s} "
          f"(cells n={n_c} d={m_c:+.3f} t={t_c:+.2f})")
    return dict(dv=dv, contrast=f"{hi}-{lo}", subset=label, n_items=n_i, n_cells=n_c,
                delta_item=round(m_i, 4), ci_lo=round(lo_ci, 4), ci_hi=round(hi_ci, 4),
                t_item=round(t_i, 2), p_item=round(p_i, 4),
                delta_cell=round(m_c, 4), t_cell=round(t_c, 2))

# =========================================================
# The report
# =========================================================

def gender_x_construct(df, dv, hi="male", lo="female"):
    """THE HEADLINE. (male-female | brilliance) - (male-female | dedication).

    Paired twice over: the male-female delta is within a cell, and the two constructs
    are matched within a PAIR (same item, same reasoning, same drawn name pair), so the
    contrast differences out both the item and any general leniency toward one name.
    Positive = the male advantage is specific to brilliance, i.e. gender-brilliance
    bias rather than a male-competence halo."""
    d = df[df["condition"].isin([hi, lo])]
    if "construct" not in d or dv not in d:
        return None
    p = d.pivot_table(index=["pair_id", "item_id", "construct"], columns="condition",
                      values=dv, aggfunc="mean")
    if hi not in p or lo not in p:
        return None
    p = p.dropna(subset=[hi, lo]).reset_index()
    p["delta"] = p[hi] - p[lo]
    w = p.pivot_table(index=["pair_id", "item_id"], columns="construct",
                      values="delta", aggfunc="mean")
    if "brilliance" not in w or "dedication" not in w:
        return None
    w = w.dropna(subset=["brilliance", "dedication"]).reset_index()
    if w.empty:
        return None
    w["interaction"] = w["brilliance"] - w["dedication"]
    im = w.groupby("item_id")["interaction"].mean()
    n_i, m_i, t_i, p_i = _t_p(im.tolist())
    n_p, m_p, t_p_, _ = _t_p(w["interaction"].tolist())
    cl, ch = _boot_ci(im.tolist())
    d_br = w["brilliance"].mean()
    d_de = w["dedication"].mean()
    star = "***" if p_i < .001 else "**" if p_i < .01 else "*" if p_i < .05 else \
           "." if p_i < .1 else " "
    print(f"  {dv:16s} male-female | brilliance = {d_br:+.3f}")
    print(f"  {'':16s} male-female | dedication = {d_de:+.3f}")
    print(f"  {'':16s} INTERACTION              = {m_i:+.3f} "
          f"[{cl:+.3f},{ch:+.3f}] t({n_i - 1})={t_i:+.2f} p={p_i:.3f}{star}")
    print(f"  {'':16s} (items n={n_i}, matched pairs n={n_p}, pair-level "
          f"d={m_p:+.3f} t={t_p_:+.2f})")
    return dict(dv=dv, contrast="gender_x_construct", subset="headline",
                delta_brilliance=round(d_br, 4), delta_dedication=round(d_de, 4),
                delta_item=round(m_i, 4), ci_lo=round(cl, 4), ci_hi=round(ch, 4),
                t_item=round(t_i, 2), p_item=round(p_i, 4),
                n_items=n_i, n_cells=n_p, mode="likert")

def analyze(df, tag="analysis"):
    df = derive(df)
    rows = []
    print("\n" + "=" * 72)
    print("GENDER TASK — brilliance attribution for identical reasoning")
    print("=" * 72)
    print(f"rows={len(df)}  items={df['item_id'].nunique()}  "
          f"cells={df['cell_id'].nunique()}")
    if "error" in df:
        n_err = int((df["error"].fillna("") != "").sum())
        if n_err:
            print(f"API errors: {n_err} rows (excluded from every measure)")
            df = df[df["error"].fillna("") == ""]

    print("\n-- refusal / unparseable rate by condition " + "-" * 29)
    for mode, g in df.groupby("mode"):
        for cond, gg in g.groupby("condition"):
            r = pd.to_numeric(gg["refused"], errors="coerce")
            print(f"  {mode:8s} {cond:7s} n={len(gg):4d}  refused={r.mean():.3f}")

    # ---- THE HEADLINE: gender x construct ------------------------------------------
    lik = df[df["mode"] == "likert"] if "mode" in df else df
    if "construct" in lik and lik["construct"].nunique() > 1:
        print("\n" + "=" * 72)
        print("HEADLINE — gender x construct interaction   (2x2)")
        print("=" * 72)
        print("  > 0 means the male advantage is SPECIFIC to brilliance, i.e. brilliance")
        print("  bias rather than a general male-competence halo. Doubly paired: the")
        print("  male-female delta is within a cell, the two constructs within a pair.")
        for dv in LIKERT_DVS:
            r = gender_x_construct(lik, dv)
            if r:
                rows.append(r)
                print()
        for by in ("fab_level", "domain", "variant", "framing"):
            if by not in lik or lik[by].nunique() < 2:
                continue
            print(f"  -- by {by} " + "-" * (60 - len(by)))
            for lev in sorted(lik[by].dropna().unique(), key=str):
                print(f"  [{by}={lev}]")
                r = gender_x_construct(lik[lik[by] == lev], "r_attribution")
                if r:
                    r.update(subset=f"{by}={lev}", moderator=by, level=lev)
                    rows.append(r)
                print()
    elif "construct" in lik:
        only = lik["construct"].dropna().unique()
        print(f"\nNOTE only one construct present ({list(only)}) -- the 2x2 interaction "
              "cannot be\ncomputed. Run with --constructs brilliance,dedication.")

    for mode, g in df.groupby("mode"):
        dvs = LIKERT_DVS if mode == "likert" else FREETEXT_DVS
        dvs = [d for d in dvs
               if d in g and pd.to_numeric(g[d], errors="coerce").notna().any()]
        if not dvs:
            continue
        print("\n" + "=" * 72)
        print(f"MODE = {mode}")
        print("=" * 72)

        print("\n-- levels by condition " + "-" * 48)
        print(g.groupby("condition")[dvs].mean(numeric_only=True).round(3).to_string())

        if mode == "freetext":
            print("\n-- lexicon coverage (share of responses with >=1 hit) " + "-" * 17)
            for which in ("innate", "ability", "effort"):
                for suf in ("", "_adj"):
                    c = f"lex_{which}{suf}"
                    if c in g:
                        v = pd.to_numeric(g[c], errors="coerce")
                        print(f"  {c:20s} coverage={(v > 0).mean():.2f}  "
                              f"mean hits/response={v.mean():.2f}")

        print("\n-- PRIMARY: paired delta, male - female (item-clustered) " + "-" * 14)
        print("   d = mean of per-item mean deltas; [.] = 95% clustered bootstrap CI")
        for dv in dvs:
            r = _report_paired(g, dv, dv)
            if r:
                r["mode"] = mode
                rows.append(r)

        for hi, lo in (("male", "control"), ("female", "control")):
            if not {hi, lo} <= set(g["condition"]):
                continue
            print(f"\n-- {hi} - {lo} (is the effect a penalty or a boost?) " + "-" * 14)
            for dv in dvs:
                r = _report_paired(g, dv, dv, hi, lo)
                if r:
                    r["mode"] = mode
                    rows.append(r)

        for by in ("construct", "framing", "fab_level", "variant", "style", "domain"):
            if by not in g:
                continue
            levels = list(g[by].dropna().unique())
            if len(levels) < 2:
                continue
            print(f"\n-- male - female by {by} " + "-" * (49 - len(by)))
            for dv in dvs:
                per = {}
                for lev in sorted(levels, key=str):
                    sub = g[g[by] == lev]
                    r = _report_paired(sub, dv, f"{dv} [{by}={lev}]")
                    if r:
                        r.update(mode=mode, moderator=by, level=lev)
                        rows.append(r)
                        per[lev] = _paired(sub, dv).groupby("item_id")["delta"].mean()
                if len(per) == 2:
                    (l1, s1), (l2, s2) = list(per.items())
                    j = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
                    if len(j) >= 3:                     # same items at both levels
                        diff = (j["a"] - j["b"]).tolist()
                        n, m, t, p = _t_p(diff)
                        cl, ch = _boot_ci(diff)
                        kind, n_txt = "within-item", f"n={n}"
                    else:                               # items nested in the level
                        na, nb, m, t, p, cl, ch = _two_sample(s1.tolist(), s2.tolist())
                        n, kind, n_txt = na + nb, "between-item", f"n={na}v{nb}"
                    if not pd.isna(m):
                        print(f"    -> interaction  ({l1} - {l2}) = {m:+.3f} "
                              f"[{cl:+.3f},{ch:+.3f}] t={t:+.2f} p={p:.3f}  "
                              f"[{kind}, {n_txt}]")
                        rows.append(dict(dv=dv, mode=mode, contrast="interaction",
                                         subset=f"{by}: {l1} vs {l2}", n_items=n,
                                         delta_item=round(m, 4), ci_lo=round(cl, 4),
                                         ci_hi=round(ch, 4), t_item=round(t, 2),
                                         p_item=round(p, 4), moderator=by,
                                         interaction_kind=kind))

    lk = df[(df["mode"] == "likert") & df["verdict_yes"].notna()]
    if len(lk) and "variant" in lk:
        print("\n" + "=" * 72)
        print("ERROR DETECTION — was the wrong answer flagged?  (verdict = No)")
        print("=" * 72)
        for variant, g in lk.groupby("variant"):
            what = "hit rate (correctly flagged)" if variant == "flawed" \
                else "false-alarm rate (good work called wrong)"
            print(f"\n  {variant} items — {what}")
            per = {}
            for cond, gg in g.groupby("condition"):
                k = int((gg["verdict_yes"] == 0).sum())
                per[cond] = (k, len(gg))
                print(f"    {cond:7s} {k:4d}/{len(gg):4d} = {k / len(gg):.3f}")
            if {"male", "female"} <= set(per):
                (km, nm), (kf, nf) = per["male"], per["female"]
                z, p = _prop_z(kf, nf, km, nm)      # female - male
                gloss = ("a woman's error is caught more often" if variant == "flawed"
                         else "a woman's sound work is wrongly called wrong more often")
                print(f"    female - male: {kf / nf - km / nm:+.3f}  z={z:+.2f} "
                      f"p={p:.3f}   (>0 = {gloss})")
                rows.append(dict(dv="verdict_flagged_wrong", mode="likert",
                                 contrast="female-male", subset=f"variant={variant}",
                                 delta_item=round(kf / nf - km / nm, 4),
                                 t_item=round(z, 2), p_item=round(p, 4),
                                 n_cells=nm + nf))

    if rows:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"{tag}_summary.csv")
        pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\nsaved summary -> {path}")
    print("\nNOTE p-values use a normal approximation (as in bai_core) and are NOT "
          "corrected\nfor the many DVs/moderators above. Item-clustered CIs are the "
          "primary evidence;\nread the moderator rows as exploratory.")
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(
        description="Paired, item-clustered analysis of a gender_task.py run.")
    ap.add_argument("--file", required=True, help="the run's .xlsx or _raw.jsonl")
    args = ap.parse_args()
    df = (pd.read_excel(args.file) if args.file.endswith(".xlsx")
          else pd.DataFrame([json.loads(l) for l in open(args.file)]))
    analyze(df, _slug(os.path.splitext(os.path.basename(args.file))[0]))

if __name__ == "__main__":
    main()
