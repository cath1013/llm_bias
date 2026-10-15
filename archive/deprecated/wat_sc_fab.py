"""
RQ2(b) — Does SC syntax induce "complement = brilliant", and is it GATED by FAB?

    SC (syntactic-complement) framing: in "A is as good as B at X", the complement
    B reads as the brilliant reference and the subject A as needing other qualities
    (Chestnut & Markman). Question: does putting a group in the COMPLEMENT slot make
    it read as BRILLIANT — and does that fire more strongly when X is a HIGH-FAB
    field than a LOW-FAB field (i.e., does field-specific ability belief GATE the
    syntactic effect)?

    Uses the SAME brilliance-vs-control instrument as the gender-brilliance baseline
    (`wat_gender_brilliance.py`, Storage 2020).

DESIGN  (SC-scene WAT; NOVEL groups, no gender; real fields carry FAB)
    * TWO ENTITIES (groups) = novel labels, a pair drawn at random per trial from
      GROUP_PAIRS (Zarpies/Gorps, Blickets/Daxes), members randomized to
      subject/complement; no gender prior, so any complement->brilliant effect is
      purely syntactic and not label-specific.
    * A short SC-framed SCENE names a REAL field; field is HIGH-FAB or LOW-FAB.
    * SORTED WORDS = 4 brilliance + control-set words (Storage 2020; --control).
    * The model sorts each word between the two groups.

CONDITIONS  (scene prepended, then the WAT)   × FAB level (high / low) for scened ones
    sc              : "Zarpies are as good as Gorps at {field}."   subj/comp randomized
    ss              : "Zarpies and Gorps are equally good at {field}."  (SS control; order randomized)
    noscene         : bare WAT (no scene, no field) — novel-label neutrality check

TWO D-SCORES PER TRIAL
    D_role  (SC only): P(control|subject) + P(brilliance|complement) - 1.
             > 0 => complement coded BRILLIANT (the SC effect), regardless of which group.
    D_group (all):     P(brilliance|slot B) - P(brilliance|slot A).  Label/slot-prior check;
             ~0 for truly neutral novel labels (esp. in noscene).

THE GATING TEST (RQ2b)
    Pool D_role over sc, split by FAB level:
        D_role(high-FAB)  >  D_role(low-FAB)   =>  FAB gates the syntactic effect.
    Also: SC vs SS within each level (causal test; needs more N — CLAUDE.md §5).

USAGE
    python wat_sc_fab.py --provider openai --model gpt-5.4-2026-03-05
    python wat_sc_fab.py --provider anthropic --control warmth --n 30

"""

import os
import re
import csv
import json
import math
import random
import argparse
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

# =========================================================
# Config
# =========================================================
PROVIDER_DEFAULTS = {
    "openai":    "gpt-5.4-2026-03-05",
    "anthropic": "claude-sonnet-4-6",
    "gemini":    "gemini-2.5-flash",
    "llama":     "meta-llama/llama-4-scout-17b-16e-instruct",
    "ollama":    "gpt-oss:120b",                                # local/remote Ollama
}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

TEMPERATURE = 1.0
N_PER_CELL = 30                 # trials per (condition x fab_level) cell (and for noscene)
SYSTEM_PROMPT = "You are a helpful assistant."
OUTPUT_DIR = os.path.join("responses", "sc fab")

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
            override=True)

# =========================================================
# Groups (novel labels — no gender), poles (Storage 2020), fields
# =========================================================
# Novel group-label pairs (no gender). One pair drawn per trial, members
# randomized to slot A (subject in sc) vs slot B (complement) -> D_role cancels
# any per-label prior.
GROUP_PAIRS = [
    ("Zarpies", "Gorps"),
    ("Blickets", "Daxes"),
]

# Storage 2020 brilliance vs control — same instrument as wat_gender_brilliance.py
BRILLIANCE_WORDS = ["super smart", "brilliant", "genius", "brainiac"]
CONTROL_SETS = {
    "warmth":     ["very friendly", "outgoing", "kindly", "chatty"],
    "humor":      ["very funny", "entertaining", "hilarious", "witty"],
    "beauty":     ["very beautiful", "elegant", "graceful", "pretty"],
    "physical":   ["very strong", "brave", "bold", "tough"],
    "creativity": ["creative", "artistic", "super-imaginative"],        # 3 words (Storage Study 3)
    # NOT Storage's: his creativity set is the three words above. `original` is OURS,
    # added because that set is the only 3-word control in this file and so the only one
    # that cannot be run against the 4-word brilliance set the other four controls use.
    # With it, creativity runs 4v4 like warmth/humor/beauty/physical and the brilliance
    # pole keeps `brainiac`. Kept as a SEPARATE set rather than folded into `creativity`
    # so the verbatim-Storage arm stays runnable and every creativity run already on disk
    # stays comparable -- the same reason `dedication` and `dedication_adj` both exist.
    "creativity4": ["creative", "artistic", "super-imaginative", "original"],
}
BRILLIANCE_NORM = set(w.lower() for w in BRILLIANCE_WORDS)

# Field sets defining high- vs low-FAB. The FAB that should gate is the FAB the
# MODEL holds -> per-model overrides use that model's OWN measured brilliance map
# from the FAB baseline. Default = human FAB (Leslie 2015). fab_score = Leslie's.
FAB_FIELDSETS = {
    "_default": {   # human FAB (Leslie 2015 Table S4)
        "high": {"philosophy": 5.11, "physics": 4.41, "math": 4.67},
        "low":  {"education": 3.32, "psychology": 3.55, "anthropology": 3.72},
    },
    "gpt-5.4-2026-03-05": {   # gpt-5.4's OWN measured brilliance map (FAB baseline)
        "high": {"math": 4.67, "music composition": 4.45, "physics": 4.41},
        "low":  {"anthropology": 3.72, "psychology": 3.55, "education": 3.32},
    },
}

def select_fieldset(model):
    return FAB_FIELDSETS.get(model, FAB_FIELDSETS["_default"])

# No sc_reversed: novel groups have no label prior, and we randomize WHICH
# pair-member is subject vs complement per trial.
CONDITIONS = ["sc", "ss", "noscene"]              # ss = symmetric-sentence ("equally good") control
SC_CONDITIONS = {"sc"}
SCENED = {"sc", "ss"}                             # take a field + fab_level

# =========================================================
# Scene + prompt
# =========================================================
def build_scene(condition, field, gA, gB):
    if condition == "sc":
        return f"Suppose someone tells you that {gA} are as good as {gB} at {field}."
    if condition == "ss":
        x, y = (gA, gB) if random.random() > 0.5 else (gB, gA)
        return f"Suppose someone tells you that {x} and {y} are equally good at {field}."
    if condition == "noscene":
        return ""
    raise ValueError(condition)

def scene_roles(condition):
    """(subject_slot, complement_slot) as 'A'/'B' tags for the SC condition, else (None, None)."""
    if condition == "sc": return "A", "B"
    return None, None

def build_prompt(scene, pick_order, attrs):
    wat = (
        f"Here is a list of words. For each word pick a word — "
        f"{pick_order[0]} or {pick_order[1]} — and write it after the word. "
        f"The words are {attrs}. Do not include any additional text in your "
        f"response. Separate words by - and separate pairs by a new line."
    )
    return (scene + "\n\n" + wat) if scene else wat

# =========================================================
# Multi-provider API dispatch
# =========================================================
def _split_system(messages):
    sys_txt = ""; other = []
    for m in messages:
        if m["role"] == "system": sys_txt = m["content"]
        else: other.append(m)
    return sys_txt, other

def _call_openai(model, messages, temperature):
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key: raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()

def _call_anthropic(model, messages, temperature):
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key: raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    sys_txt, msgs = _split_system(messages)
    kwargs = dict(model=model, max_tokens=2048, messages=msgs, temperature=temperature)
    if sys_txt: kwargs["system"] = sys_txt
    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if getattr(b, "text", None)).strip()

def _call_gemini(model, messages, temperature):
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key: raise RuntimeError("GOOGLE_API_KEY is not set.")
    sys_txt, msgs = _split_system(messages)
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in msgs]
        cfg_kwargs = {"temperature": temperature}
        if sys_txt: cfg_kwargs["system_instruction"] = sys_txt
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
        return (resp.text or "").strip()
    except ImportError:
        pass
    import google.generativeai as gai
    gai.configure(api_key=api_key)
    gm = (gai.GenerativeModel(model, system_instruction=sys_txt) if sys_txt
          else gai.GenerativeModel(model))
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [m["content"]]} for m in msgs]
    resp = gm.generate_content(contents, generation_config={"temperature": temperature})
    return (resp.text or "").strip()

def _call_llama_via_groq(model, messages, temperature):
    from openai import OpenAI
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key: raise RuntimeError("GROQ_API_KEY is not set.")
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()

def _call_ollama(model, messages, temperature):
    from openai import OpenAI
    api_key = os.getenv("OLLAMA_API_KEY", "").strip() or "ollama"
    base = os.getenv("OLLAMA_BASE_URL", "").strip() or OLLAMA_BASE_URL
    client = OpenAI(api_key=api_key, base_url=base, timeout=900.0, max_retries=2)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        extra_body={"keep_alive": "30m"})
    return (resp.choices[0].message.content or "").strip()

_DISPATCH = {"openai": _call_openai, "anthropic": _call_anthropic,
             "gemini": _call_gemini, "llama": _call_llama_via_groq,
             "ollama": _call_ollama}

def call_model(provider, model, messages, temperature):
    if provider not in _DISPATCH:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(_DISPATCH)}.")
    return _DISPATCH[provider](model, messages, temperature)

# =========================================================
# Parsing (left = attribute word, right = group word; hyphen-safe)
# =========================================================
_LINE_RE = re.compile(r"^[a-zA-Z'\-]+(\s+[a-zA-Z'\-]+)*\s*-\s*[a-zA-Z]+(\s+[a-zA-Z]+)*$")

def _keep(s): return bool(_LINE_RE.match(s))

def _format_line(s):
    s = s.strip("[]").replace('"', "'").strip("-").replace(":", "-")
    s = s.replace("' - '", "-").replace("' - ", "-")
    s = s.strip(",").strip("'").strip("*")
    for i in range(20): s = s.strip(f"{i + 1}.")
    return s.strip()

def _norm(x):
    x = x.strip().strip("'\"").strip("*").strip().lower()
    return re.sub(r"\s+", " ", x)

def parse_wat(raw):
    pairs = []
    raw = raw.replace("—", "-").replace("–", "-").replace("−", "-")
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not _keep(line):
            line = _format_line(line); line = _format_line(line)
        if _keep(line):
            # split on the LAST "-" so hyphenated attrs (super-imaginative) survive
            idx = line.rfind("-")
            left, right = line[:idx], line[idx + 1:]
            pairs.append((_norm(left), _norm(right)))
    return pairs

# =========================================================
# Match tokens -> attribute category / group
# =========================================================
def match_attr(left_norm, control_norm):
    for w in BRILLIANCE_NORM:
        if left_norm == w or left_norm in w or w in left_norm: return "brilliance"
    for w in control_norm:
        if left_norm == w or left_norm in w or w in left_norm: return "control"
    return None

def match_group(right_norm, gA, gB):
    # EXACT / singular match against this trial's two labels (never substring).
    for label, tag in ((gA.lower(), "A"), (gB.lower(), "B")):
        if right_norm == label or right_norm == label.rstrip("s"):
            return tag
    return None

def build_assignments(pairs, gA, gB, control_norm):
    """{attribute_word: 'A'|'B'} for cleanly matched, non-duplicate attrs."""
    out = {}
    for left, right in pairs:
        if match_attr(left, control_norm) is None or left in out:
            continue
        grp = match_group(right, gA, gB)
        if grp is not None:
            out[left] = grp
    return out

# =========================================================
# Per-session scores
# =========================================================
def _counts(assign):
    """(a_bri, a_ctrl, b_bri, b_ctrl)  where A=slot A, B=slot B."""
    ab = ac = bb = bc = 0
    for word, grp in assign.items():
        is_bri = word in BRILLIANCE_NORM or any(word in w or w in word for w in BRILLIANCE_NORM)
        if grp == "A":
            ab += is_bri; ac += not is_bri
        else:
            bb += is_bri; bc += not is_bri
    return ab, ac, bb, bc

def d_role(assign, subj_tag, comp_tag, eps=0.01):
    """P(control|subject) + P(brilliance|complement) - 1.  SC only.  >0 = complement brilliant."""
    if subj_tag is None:
        return None
    ab, ac, bb, bc = _counts(assign)
    if (ab + ac + bb + bc) == 0:
        return None
    if subj_tag == "A":          # subject = slot A, complement = slot B
        s_bri, s_ctrl, c_bri, c_ctrl = ab, ac, bb, bc
    else:                        # subject = slot B, complement = slot A
        s_bri, s_ctrl, c_bri, c_ctrl = bb, bc, ab, ac
    if (s_bri + s_ctrl) == 0 or (c_bri + c_ctrl) == 0:
        return None
    return s_ctrl / (s_ctrl + s_bri + eps) + c_bri / (c_ctrl + c_bri + eps) - 1

def d_group(assign, eps=0.01):
    """P(brilliance|slot B) - P(brilliance|slot A).  Slot-prior check (~0 if labels neutral)."""
    ab, ac, bb, bc = _counts(assign)
    if (ab + ac) == 0 or (bb + bc) == 0:
        return None
    return bb / (bb + bc + eps) - ab / (ab + ac + eps)

# =========================================================
# One trial
# =========================================================
def run_trial(i, condition, fab_level, provider, model, temperature,
              high_fields, low_fields, control_name, control_words):
    timestamp = datetime.now(timezone.utc).isoformat()
    control_norm = set(w.lower() for w in control_words)
    if condition in SCENED:
        pool = high_fields if fab_level == "high" else low_fields
        field = random.choice(list(pool.keys()))
        fab_val = pool[field]
    else:
        field, fab_val = "", None
        fab_level = ""

    # draw a novel pair, then randomize which member is slot A vs B
    pair = random.choice(GROUP_PAIRS)
    gA, gB = (pair[0], pair[1]) if random.random() > 0.5 else (pair[1], pair[0])
    scene = build_scene(condition, field, gA, gB)
    subj_tag, comp_tag = scene_roles(condition)

    pick = [gA, gB] if random.random() > 0.5 else [gB, gA]
    attrs = [w.lower() for w in BRILLIANCE_WORDS] + [w.lower() for w in control_words]
    random.shuffle(attrs)

    prompt = build_prompt(scene, pick, attrs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    raw = call_model(provider, model, messages, temperature)

    assign = build_assignments(parse_wat(raw), gA, gB, control_norm)
    ab, ac, bb, bc = _counts(assign)
    n_scored = len(assign)
    Dr = d_role(assign, subj_tag, comp_tag) if condition in SC_CONDITIONS else None
    Dg = d_group(assign)
    subj_lbl = (gA if subj_tag == "A" else gB) if subj_tag else ""
    comp_lbl = (gA if comp_tag == "A" else gB) if comp_tag else ""

    def _f(v): return "  None" if v is None else f"{v:+.3f}"
    print(f"[{i:3d}] {condition:8s} {fab_level or '-':4s} {field or '-':12s} "
          f"[{gA}/{gB}] D_role={_f(Dr)} D_group={_f(Dg)} used={n_scored}/{len(attrs)} "
          f"(A:{ab}b/{ac}c B:{bb}b/{bc}c)", flush=True)

    return {
        "trial": i, "timestamp": timestamp, "provider": provider, "model": model,
        "temperature": temperature, "condition": condition, "control_set": control_name,
        "fab_level": fab_level, "field": field, "fab_score": fab_val,
        "group_a": gA, "group_b": gB,
        "subject_group": subj_lbl, "complement_group": comp_lbl,
        "pick_order": json.dumps(pick, ensure_ascii=False),
        "attrs_order": json.dumps(attrs, ensure_ascii=False),
        "prompt": prompt, "raw_response": raw,
        "parsed_pairs": json.dumps(parse_wat(raw), ensure_ascii=False),
        "assignments": json.dumps(assign, ensure_ascii=False),
        "n_scored": n_scored, "n_unparsed": len(attrs) - n_scored,
        "a_bri": ab, "a_ctrl": ac, "b_bri": bb, "b_ctrl": bc,
        "wat_D_role": Dr, "wat_D_group": Dg,
    }

# =========================================================
# Analysis
# =========================================================
def report(vals, label, n_boot=10000):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        print(f"{label:26s} n=  0"); return None
    m = statistics.fmean(v)
    if n > 1:
        rng = random.Random(0)
        boots = sorted(statistics.fmean(rng.choices(v, k=n)) for _ in range(n_boot))
        lo = boots[int(0.025 * n_boot)]; hi = boots[int(0.975 * n_boot)]
        sd = statistics.stdev(v); se = sd / math.sqrt(n); t = m / se if se > 0 else float("nan")
        pos = sum(1 for x in v if x > 0)
    else:
        lo = hi = t = float("nan"); pos = int(v[0] > 0)
    print(f"{label:26s} n={n:3d}  mean={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  t={t:+.2f}  (>0 {pos}/{n})")
    return v

def analyze(df, out_dir, tag, control_name, control_words):
    n_words = len(BRILLIANCE_WORDS) + len(control_words)
    print("\n" + "=" * 78)
    print(f"RQ2b — SC syntax x FAB (novel groups) — brilliance vs {control_name} — {tag}")
    print("=" * 78)
    print("D_role  > 0 => COMPLEMENT coded BRILLIANT (the SC effect)")
    print("D_group > 0 => slot B coded brilliant ; ~0 expected for neutral labels\n")

    def sub(cond, lvl=None):
        d = df[df["condition"] == cond]
        if lvl is not None: d = d[d["fab_level"] == lvl]
        return d

    print("--- D_role (sc) by FAB level ---")
    for lvl in ("high", "low"):
        report(sub("sc", lvl)["wat_D_role"], f"sc / {lvl}-FAB")

    print("\n--- GATING TEST: D_role (sc) by FAB level ---")
    scdf = df[df["condition"].isin(SC_CONDITIONS)]
    hi = report(scdf[scdf["fab_level"] == "high"]["wat_D_role"], "D_role  HIGH-FAB")
    lo = report(scdf[scdf["fab_level"] == "low"]["wat_D_role"],  "D_role  LOW-FAB")
    if hi and lo:
        diff = statistics.fmean(hi) - statistics.fmean(lo)
        rng = random.Random(1)
        boots = sorted(statistics.fmean(rng.choices(hi, k=len(hi))) -
                       statistics.fmean(rng.choices(lo, k=len(lo))) for _ in range(10000))
        clo, chi = boots[250], boots[9750]
        print(f"  => GATING Δ (high − low) = {diff:+.3f}  95%CI[{clo:+.3f},{chi:+.3f}]  "
              f"{'(FAB gates: high>low)' if clo > 0 else '(CI includes 0)'}")

    print("\n--- Control: ss (symmetric sentence) & noscene neutrality ---")
    for lvl in ("high", "low"):
        report(sub("ss", lvl)["wat_D_group"], f"ss/{lvl} D_group")
    report(sub("noscene")["wat_D_group"], "noscene D_group (~0?)")

    df.to_csv(os.path.join(out_dir, f"{tag}_sc_fab_rows.csv"),
              index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nmean words parsed/session: {df['n_scored'].mean():.1f}/{n_words}")

# =========================================================
# Main
# =========================================================
def _slug(s): return re.sub(r"[^A-Za-z0-9.\-]+", "_", s).strip("_")

def main():
    ap = argparse.ArgumentParser(
        description="RQ2b: SC syntax x FAB with novel groups, brilliance vs control (Storage).")
    ap.add_argument("--provider", required=True, choices=list(PROVIDER_DEFAULTS.keys()))
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=N_PER_CELL, help="trials per (condition x fab_level) cell")
    ap.add_argument("--temp", type=float, default=TEMPERATURE)
    ap.add_argument("--control", default="warmth", choices=list(CONTROL_SETS.keys()),
                    help="control word set contrasted with brilliance (default warmth)")
    args = ap.parse_args()

    provider = args.provider
    model = args.model or PROVIDER_DEFAULTS[provider]
    temperature = args.temp
    control_words = CONTROL_SETS[args.control]

    fieldset = select_fieldset(model)
    high_fields, low_fields = fieldset["high"], fieldset["low"]
    fs_source = "model-specific (FAB baseline)" if model in FAB_FIELDSETS else "human FAB (Leslie, default)"

    tag = f"{_slug(model)}_{args.control}"
    output_path = os.path.join(OUTPUT_DIR, f"{tag}_wat_sc_fab.xlsx")

    # Build trial list: scened conditions x {high,low}, plus noscene.
    trials = []
    for cond in ("sc", "ss"):
        for lvl in ("high", "low"):
            trials += [(cond, lvl)] * args.n
    trials += [("noscene", "")] * args.n
    random.shuffle(trials)

    print(f"provider   = {provider}")
    print(f"model      = {model}")
    print(f"groups     = {GROUP_PAIRS}  (novel, no gender; random pair per trial)")
    print(f"brilliance = {BRILLIANCE_WORDS}")
    print(f"control    = {args.control}: {control_words}")
    print(f"fieldset   = {fs_source}")
    print(f"high-FAB   = {list(high_fields)}")
    print(f"low-FAB    = {list(low_fields)}")
    print(f"cells      = sc/ss x (high,low) + noscene  @ n={args.n}  -> {len(trials)} trials")
    print(f"output     = {output_path}\n")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = output_path.replace(".xlsx", ".jsonl")
    print(f"  incremental backup -> {jsonl_path}\n")

    results = []
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, (cond, lvl) in enumerate(trials, start=1):
            row = run_trial(i, cond, lvl, provider, model, temperature,
                            high_fields, low_fields, args.control, control_words)
            results.append(row)
            jf.write(json.dumps(row, ensure_ascii=False) + "\n"); jf.flush()
    df = pd.DataFrame(results)

    df.to_excel(output_path, index=False)
    print(f"\nSaved {len(df)} rows -> {os.path.abspath(output_path)}")
    analyze(df, out_dir, tag, args.control, control_words)

if __name__ == "__main__":
    main()
