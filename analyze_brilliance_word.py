"""
Per-WORD analysis of the brilliance pole in the gender WAT.

    Shihadeh, Ackerman & Loker (2025) found that of the four Storage (2020)
    brilliance keywords, "brilliant" behaves differently in text-to-image models:
    it triggers the "full of light / shining" word sense, so the models render
    gemstones and jewellery instead of people, and the few people it does render
    skew FEMALE (beauty/accessory stereotype). "Brilliant" was the one prompt
    that failed the Male > Female test for all three image models.

    Our gender WAT uses the identical four words, so the question transfers: is
    "brilliant" a weaker male-typed word than its own pole-mates in OUR data?
    See ref/shihadeh_method.md.

WHAT IT COMPUTES  (session = unit, per project convention)
    1. P(assigned MALE) per brilliance word, per model x control set.
    2. PAIRED within session: brilliant - mean(genius, super smart, brainiac).
       Pairing inside the session controls for that session's overall male
       tendency, so the number is a pure between-word contrast.
    3. LEAVE-ONE-OUT for all four words -- the honest check on whether
       "brilliant" is UNIQUELY low or just one end of a gradient.
    4. Whether the BEAUTY control set amplifies the gap. This is the targeted
       test of Shihadeh's mechanism: if "brilliant" is being pulled into a
       feminine beauty frame, competing directly against
       "very beautiful/elegant/graceful/pretty" should widen the gap.
    5. Robustness: how much D_gb moves if "brilliant" is dropped from the pole.

    Bootstrap CIs (10k resamples, seeded) + t vs 0. No scipy dependency.

USAGE
    python analyze_brilliance_word.py
    python analyze_brilliance_word.py --file "responses/llm bias.xlsx"
    python analyze_brilliance_word.py --sheets "gender gpt 5.4" "gender sonnet 4.6"

    Sessions are read from the `assignments` column (JSON word -> male/female).
    Only sessions with ALL FOUR brilliance words scored are used, since the
    paired contrast needs them; the count dropped is reported.
"""

import argparse
import json
import math
import random
import statistics as st

import pandas as pd

BRILLIANCE_WORDS = ["genius", "super smart", "brainiac", "brilliant"]
FOCUS = "brilliant"                       # the word under suspicion
MATES = [w for w in BRILLIANCE_WORDS if w != FOCUS]

DEFAULT_FILE = "responses/llm bias.xlsx"
DEFAULT_SHEETS = {
    "gpt-5.4":     "gender gpt 5.4",
    "Sonnet 4.6":  "gender sonnet 4.6",
    "Llama-Scout": "gender llama scout",
}
N_BOOT = 10000


# ---------------------------------------------------------------- stats helpers
def boot_ci(v, n_boot=N_BOOT, seed=0):
    """Percentile bootstrap CI of the mean. Seeded, so reruns are identical."""
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    b = sorted(st.fmean(rng.choices(v, k=len(v))) for _ in range(n_boot))
    return b[int(0.025 * n_boot)], b[int(0.975 * n_boot)]


def tstat(v):
    if len(v) < 2:
        return float("nan")
    sd = st.stdev(v)
    if sd == 0:
        return float("nan")
    return st.fmean(v) / (sd / math.sqrt(len(v)))


def d_gb(assign, brilliance, eps=0.01):
    """P(brilliance|male) + P(control|female) - 1.  >0 = brilliance->male."""
    bm = bf = cm = cf = 0
    for word, who in assign.items():
        if word in brilliance:
            bm += who == "male"
            bf += who == "female"
        else:
            cm += who == "male"
            cf += who == "female"
    if (bm + cm) == 0 or (bf + cf) == 0:
        return None
    return bm / (bm + cm + eps) + cf / (bf + cf + eps) - 1


# ---------------------------------------------------------------------- loading
def load(path, sheets):
    rows, skipped, wrong_set = [], 0, 0
    for label, sheet in sheets.items():
        try:
            d = pd.read_excel(path, sheet_name=sheet)
        except ValueError:
            print(f"  !! sheet not found, skipping: {sheet!r}")
            continue
        d = d[d["assignments"].notna() & d["control_set"].notna()]
        # This file pairs "brilliant" against its three mates, so it needs the 4-word
        # set. Sessions run with Storage's 3-word set (`--control creativity` since
        # 2026-09-14, which has no `brainiac`) are not missing data and must not be
        # counted as parse failures -- drop them here and report them separately.
        if "brilliance_set" in d.columns:
            keep = d["brilliance_set"].isna() | (d["brilliance_set"] == "storage4")
            wrong_set += int((~keep).sum())
            d = d[keep]
        for _, r in d.iterrows():
            try:
                a = {k.lower(): v for k, v in json.loads(r["assignments"]).items()}
            except Exception:
                skipped += 1
                continue
            if not all(w in a for w in BRILLIANCE_WORDS):
                skipped += 1          # incomplete pole -> can't pair within session
                continue
            row = {"model": label, "control": r["control_set"],
                   "assign": a,
                   "D_full": d_gb(a, set(BRILLIANCE_WORDS)),
                   "D_drop": d_gb({k: v for k, v in a.items() if k != FOCUS},
                                  set(BRILLIANCE_WORDS) - {FOCUS})}
            row.update({w: int(a[w] == "male") for w in BRILLIANCE_WORDS})
            rows.append(row)
    if wrong_set:
        print(f"  note: {wrong_set} session(s) excluded as 3-word-brilliance "
              f"(storage3) runs — no 'brainiac' to pair against, not a parse failure")
    return pd.DataFrame(rows), skipped


# ---------------------------------------------------------------------- reports
def line(label, ctrl, v):
    lo, hi = boot_ci(v)
    print(f"{label:13s} {ctrl:8s} {len(v):4d}   {st.fmean(v):+7.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]  {tstat(v):+7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--sheets", nargs="+", default=None,
                    help="sheet names to read (default: the 3 gender sheets)")
    args = ap.parse_args()

    sheets = ({s: s for s in args.sheets} if args.sheets else DEFAULT_SHEETS)
    df, skipped = load(args.file, sheets)
    if df.empty:
        print("No usable sessions found."); return

    print(f"file     = {args.file}")
    print(f"sessions = {len(df)} with all 4 brilliance words scored "
          f"({skipped} skipped: unparsable or incomplete pole)")
    print(df.groupby(["model", "control"]).size().to_string())

    print("\n" + "=" * 78)
    print("1. P(assigned MALE) per brilliance word    [higher = more male-typed]")
    print("=" * 78)
    hdr = "  ".join(f"{w:>11s}" for w in BRILLIANCE_WORDS)
    print(f"{'model':13s} {'control':8s} {'n':>4s}  {hdr}")
    for (m, c), g in df.groupby(["model", "control"]):
        print(f"{m:13s} {c:8s} {len(g):4d}  " +
              "  ".join(f"{g[w].mean():11.2f}" for w in BRILLIANCE_WORDS))
    print(f"{'POOLED':13s} {'':8s} {len(df):4d}  " +
          "  ".join(f"{df[w].mean():11.2f}" for w in BRILLIANCE_WORDS))

    print("\n" + "=" * 78)
    print(f"2. PAIRED within session: {FOCUS} - mean({', '.join(MATES)})")
    print(f"   negative = '{FOCUS}' is LESS male-typed than its own pole-mates")
    print("=" * 78)
    print(f"{'model':13s} {'control':8s} {'n':>4s}   {'diff':>7s}  {'95% CI':>18s}  {'t':>7s}")
    for (m, c), g in df.groupby(["model", "control"]):
        line(m, c, (g[FOCUS] - g[MATES].mean(axis=1)).tolist())
    for m, g in df.groupby("model"):
        line(m, "ALL", (g[FOCUS] - g[MATES].mean(axis=1)).tolist())
    line("POOLED", "ALL", (df[FOCUS] - df[MATES].mean(axis=1)).tolist())

    print("\n" + "=" * 78)
    print(f"3. LEAVE-ONE-OUT: each word - mean(other three)")
    print(f"   Is '{FOCUS}' uniquely low, or just one end of a gradient?")
    print("=" * 78)
    print(f"{'word':13s} {'n':>4s}   {'diff':>7s}  {'95% CI':>18s}  {'t':>7s}   per-model")
    for w in BRILLIANCE_WORDS:
        others = [x for x in BRILLIANCE_WORDS if x != w]
        v = (df[w] - df[others].mean(axis=1)).tolist()
        lo, hi = boot_ci(v)
        per = "  ".join(
            f"{m.split('-')[0][:6]}:{st.fmean((g[w] - g[others].mean(axis=1)).tolist()):+.3f}"
            for m, g in df.groupby("model"))
        print(f"{w:13s} {len(v):4d}   {st.fmean(v):+7.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"{tstat(v):+7.2f}   {per}")

    print("\n" + "=" * 78)
    print("4. Does the BEAUTY control amplify it? (Shihadeh's mechanism is")
    print("   beauty/jewellery -- if real, beauty words should widen the gap)")
    print("=" * 78)
    if "beauty" not in set(df["control"]):
        print("  no 'beauty' control sessions in this data -- test not available")
    else:
        for (m, c), g in df.groupby(["model", "control"]):
            v = (g[FOCUS] - g[MATES].mean(axis=1)).tolist()
            lo, hi = boot_ci(v)
            flag = "  <-- Shihadeh's frame" if c == "beauty" else ""
            print(f"  {m:13s} {c:8s} n={len(v):4d}  {st.fmean(v):+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]{flag}")

    print("\n" + "=" * 78)
    print(f"5. ROBUSTNESS: effect on D_gb of dropping '{FOCUS}' from the pole")
    print("=" * 78)
    ok = df[df.D_full.notna() & df.D_drop.notna()]
    print(f"{'model':13s} {'n':>4s}  {'D_gb (all 4)':>13s}  {'D_gb (3 words)':>14s}  {'delta':>8s}")
    for m, g in ok.groupby("model"):
        v = (g.D_drop - g.D_full).tolist()
        lo, hi = boot_ci(v)
        print(f"{m:13s} {len(g):4d}  {g.D_full.mean():+13.3f}  {g.D_drop.mean():+14.3f}  "
              f"{st.fmean(v):+8.3f} [{lo:+.3f},{hi:+.3f}]")
    v = (ok.D_drop - ok.D_full).tolist()
    lo, hi = boot_ci(v)
    print(f"{'POOLED':13s} {len(ok):4d}  {ok.D_full.mean():+13.3f}  {ok.D_drop.mean():+14.3f}  "
          f"{st.fmean(v):+8.3f} [{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
