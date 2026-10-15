"""
RQ2(a) — Does SC syntax induce "complement = more skilled"?  (Chestnut & Markman
replication: gender labels, NOVEL activity, brilliance vs effort words)

    SC (syntactic-complement) framing: in "A is as good as B at X", the complement
    B reads as the more-skilled/brilliant reference and the subject A as working
    hard to catch up (Chestnut & Markman). Question: does putting a gender in the
    COMPLEMENT slot make it read as MORE SKILLED (brilliant) — a PURELY SYNTACTIC
    effect that flips with which gender is the complement, on a made-up activity?

DESIGN  (SC-scene WAT; GENDER labels; NOVEL activity; NO field/FAB)
    * TWO ENTITIES = a gender pair drawn per trial from GENDER_PAIRS
      (boys/girls, men/women); each label maps to a SEX tag (male/female), so
      boys+men pool as "male", girls+women as "female".
    * A short SC-framed SCENE names a NONSENSE activity (e.g. "freeching").
    * SORTED WORDS = 4 brilliance + 4 effort words (Leslie 2015 FAB items).

CONDITIONS  (scene prepended, then the WAT)
    sc          : "A are as good as B at {skill}."  subject/complement SEX
                  randomized per trial (replaces sc_reversed); gets 2x trials.
                  The sign-flip is recovered by splitting sc on complement_sex.
    ss          : "<male> and <female> are equally good at {skill}."  (order randomized)
    noscene     : bare WAT (no scene) — the model's DEFAULT gender prior on the nonsense skill

TWO D-SCORES PER TRIAL
    D_role  (SC only): P(effort|subject) + P(brilliance|complement) - 1.
             > 0 => complement coded MORE SKILLED (the SC effect), regardless of sex.
    D_gender (all):    P(brilliance|female) - P(brilliance|male).
             CONVENTION: + = FEMALE (girls/women) brilliant ; - = MALE brilliant.

THE DIAGNOSTIC (is the effect SYNTACTIC or a fixed GENDER PRIOR?)
    If SC-ROLE driven: pooled D_role > 0, and D_gender FLIPS sign when sc is split
        by complement sex: comp=female -> +  ;  comp=male -> -.
    If a FIXED gender prior: D_gender keeps the SAME sign across sc(both splits) /
        ss / noscene (one sex is "brilliant" no matter the role).

USAGE
    python wat_sc_gender.py --provider openai --model gpt-5.4-2026-03-05
    python wat_sc_gender.py --provider anthropic --n 50

ENV (.env): OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY / GROQ_API_KEY
            OLLAMA_BASE_URL / OLLAMA_API_KEY
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
N_PER_CONDITION = 50            # trials per condition (sc gets 2x)
SYSTEM_PROMPT = "You are a helpful assistant."
OUTPUT_DIR = os.path.join("responses", "sc brilliance")

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
            override=True)

# =========================================================
# Poles (brilliance vs effort), gender pairs, nonsense activities
# =========================================================
BRILLIANCE_WORDS = ["special aptitude", "innate gift", "innate talent", "raw ability"]
EFFORT_WORDS     = ["hard working", "dedicated", "motivated", "sustained effort"]
BRILLIANCE_NORM = set(w.lower() for w in BRILLIANCE_WORDS)
EFFORT_NORM     = set(w.lower() for w in EFFORT_WORDS)
ALL_WORDS = [w.lower() for w in BRILLIANCE_WORDS] + [w.lower() for w in EFFORT_WORDS]

# Gender pairs (male_label, female_label); a pair drawn per trial, sex-pooled.
GENDER_PAIRS = [("boys", "girls"), ("men", "women")]

# Nonsense gerunds (pronounceable, no real association). One drawn per trial.
NONSENSE_SKILLS = ["freeching", "blicking", "florping", "quibbing", "sprocking"]

CONDITIONS = ["sc", "ss", "noscene"]
SC_CONDITIONS = {"sc"}
SCENED = {"sc", "ss"}                             # take a nonsense skill

# =========================================================
# Scene + prompt
# =========================================================
def build_scene(condition, skill, subj_lbl, comp_lbl, male_lbl, female_lbl):
    if condition == "sc":       # subj/comp already chosen (sex randomized per trial)
        return f"Suppose someone tells you that {subj_lbl} are as good as {comp_lbl} at {skill}."
    if condition == "ss":
        x, y = (male_lbl, female_lbl) if random.random() > 0.5 else (female_lbl, male_lbl)
        return f"Suppose someone tells you that {x} and {y} are equally good at {skill}."
    if condition == "noscene":
        return ""
    raise ValueError(condition)

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
# Parsing (left = attribute word, right = gender word)
# =========================================================
_LINE_RE = re.compile(r"^[a-zA-Z']+(\s+[a-zA-Z']+)*\s*-\s*[a-zA-Z]+(\s+[a-zA-Z]+)*$")

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
            left, right = line.split("-", 1)
            pairs.append((_norm(left), _norm(right)))
    return pairs

# =========================================================
# Match tokens -> attribute category / sex
# =========================================================
def match_attr(left_norm):
    for w in BRILLIANCE_NORM:
        if left_norm == w or left_norm in w or w in left_norm: return "brilliance"
    for w in EFFORT_NORM:
        if left_norm == w or left_norm in w or w in left_norm: return "effort"
    return None

def match_sex(right_norm, male_lbl, female_lbl):
    # EXACT / singular match against this trial's two labels (never substring —
    # "men" is inside "women", "man" inside "woman").
    for lbl, sex in ((male_lbl.lower(), "male"), (female_lbl.lower(), "female")):
        if right_norm == lbl or right_norm == lbl.rstrip("s"):
            return sex
    return None

def build_assignments(pairs, male_lbl, female_lbl):
    """{attribute_word: 'male'|'female'} for cleanly matched, non-duplicate attrs."""
    out = {}
    for left, right in pairs:
        if match_attr(left) is None or left in out:
            continue
        sex = match_sex(right, male_lbl, female_lbl)
        if sex is not None:
            out[left] = sex
    return out

# =========================================================
# Per-session scores
# =========================================================
def _counts(assign):
    """(male_bri, male_eff, female_bri, female_eff)."""
    mb = me = fb = fe = 0
    for word, sex in assign.items():
        is_bri = word in BRILLIANCE_NORM or any(word in w or w in word for w in BRILLIANCE_NORM)
        if sex == "male":
            mb += is_bri; me += not is_bri
        else:
            fb += is_bri; fe += not is_bri
    return mb, me, fb, fe

def d_role(assign, subj_sex, comp_sex, eps=0.01):
    """P(effort|subject) + P(brilliance|complement) - 1.  SC only.  >0 = complement skilled."""
    if subj_sex is None:
        return None
    mb, me, fb, fe = _counts(assign)
    if (mb + me + fb + fe) == 0:
        return None
    if subj_sex == "male":       # subject=male, complement=female
        s_bri, s_eff, c_bri, c_eff = mb, me, fb, fe
    else:                        # subject=female, complement=male
        s_bri, s_eff, c_bri, c_eff = fb, fe, mb, me
    if (s_bri + s_eff) == 0 or (c_bri + c_eff) == 0:
        return None
    return s_eff / (s_eff + s_bri + eps) + c_bri / (c_eff + c_bri + eps) - 1

def d_gender(assign, eps=0.01):
    """P(brilliance|female) - P(brilliance|male).  + = female brilliant, - = male brilliant."""
    mb, me, fb, fe = _counts(assign)
    if (mb + me) == 0 or (fb + fe) == 0:
        return None
    return fb / (fb + fe + eps) - mb / (mb + me + eps)

# =========================================================
# One trial
# =========================================================
def run_trial(i, condition, provider, model, temperature):
    timestamp = datetime.now(timezone.utc).isoformat()
    skill = random.choice(NONSENSE_SKILLS) if condition in SCENED else ""
    male_lbl, female_lbl = random.choice(GENDER_PAIRS)

    # SC: randomize which SEX is subject vs complement (replaces sc_reversed)
    if condition == "sc":
        if random.random() > 0.5:
            subj_lbl, comp_lbl, subj_sex, comp_sex = male_lbl, female_lbl, "male", "female"
        else:
            subj_lbl, comp_lbl, subj_sex, comp_sex = female_lbl, male_lbl, "female", "male"
    else:
        subj_lbl = comp_lbl = None; subj_sex = comp_sex = None

    scene = build_scene(condition, skill, subj_lbl, comp_lbl, male_lbl, female_lbl)
    pick = [male_lbl, female_lbl] if random.random() > 0.5 else [female_lbl, male_lbl]
    attrs = ALL_WORDS[:]; random.shuffle(attrs)

    prompt = build_prompt(scene, pick, attrs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    raw = call_model(provider, model, messages, temperature)

    assign = build_assignments(parse_wat(raw), male_lbl, female_lbl)
    mb, me, fb, fe = _counts(assign)
    n_scored = len(assign)
    Dr = d_role(assign, subj_sex, comp_sex) if condition in SC_CONDITIONS else None
    Dg = d_gender(assign)

    def _f(v): return "  None" if v is None else f"{v:+.3f}"
    print(f"[{i:3d}] {condition:8s} {skill or '-':10s} [{male_lbl}/{female_lbl}] "
          f"comp={comp_sex or '-':6s} D_role={_f(Dr)} D_gender={_f(Dg)} "
          f"used={n_scored}/{len(ALL_WORDS)} (m:{mb}br/{me}ef f:{fb}br/{fe}ef)", flush=True)

    return {
        "trial": i, "timestamp": timestamp, "provider": provider, "model": model,
        "temperature": temperature, "condition": condition, "skill": skill,
        "gender_pair": f"{male_lbl}/{female_lbl}", "male_word": male_lbl, "female_word": female_lbl,
        "subject_gender": subj_lbl or "", "complement_gender": comp_lbl or "",
        "subject_sex": subj_sex or "", "complement_sex": comp_sex or "",
        "pick_order": json.dumps(pick, ensure_ascii=False),
        "attrs_order": json.dumps(attrs, ensure_ascii=False),
        "prompt": prompt, "raw_response": raw,
        "parsed_pairs": json.dumps(parse_wat(raw), ensure_ascii=False),
        "assignments": json.dumps(assign, ensure_ascii=False),
        "n_scored": n_scored, "n_unparsed": len(ALL_WORDS) - n_scored,
        "male_bri": mb, "male_eff": me, "female_bri": fb, "female_eff": fe,
        "wat_D_role": Dr, "wat_D_gender": Dg,
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

def analyze(df, out_dir, tag):
    print("\n" + "=" * 78)
    print(f"RQ2a — SC syntax on a NONSENSE skill (gender labels) — {tag}")
    print("=" * 78)
    print("D_role   > 0 => COMPLEMENT coded MORE SKILLED (the SC effect)")
    print("D_gender + = female (girls/women) brilliant ; - = male brilliant\n")

    def sub(cond): return df[df["condition"] == cond]
    scdf = df[df["condition"] == "sc"]
    sc_f = scdf[scdf["complement_sex"] == "female"]
    sc_m = scdf[scdf["complement_sex"] == "male"]

    # degeneracy (model put all words on one sex) per condition
    print("--- refusal/degeneracy (all words to ONE sex) per condition ---")
    for c in CONDITIONS:
        d = sub(c)
        mt = d["male_bri"] + d["male_eff"]; ft = d["female_bri"] + d["female_eff"]
        deg = int(((mt == 0) | (ft == 0)).sum())
        print(f"  {c:8s} n={len(d):3d}  one-sex sessions={deg}")

    print(f"\n--- SC effect: pooled D_role (sc; {len(sc_f)} comp=female, {len(sc_m)} comp=male) ---")
    report(scdf["wat_D_role"], "D_role pooled")
    def cflag(r):
        if r["complement_sex"] == "female":  c, s = r["female_bri"], r["male_bri"]
        elif r["complement_sex"] == "male":  c, s = r["male_bri"], r["female_bri"]
        else: return None
        return None if c == s else int(c > s)
    fl = scdf.apply(cflag, axis=1).dropna()
    if len(fl): print(f"  binomial: complement>subject in brilliance {int(fl.sum())}/{len(fl)} ({fl.mean()*100:.0f}%, ties excl)")

    print("\n--- KEY DIAGNOSTIC: does D_gender FLIP with the complement sex? ---")
    def gm(d, label):
        s = report(d["wat_D_gender"], label)
        return statistics.fmean(s) if s else float("nan")
    m_f = gm(sc_f, "sc comp=female D_gender")
    m_m = gm(sc_m, "sc comp=male   D_gender")
    m_ss = gm(sub("ss"), "ss D_gender")
    m_ns = gm(sub("noscene"), "noscene D_gender")
    print(f"\n  sc comp=female D_gender={m_f:+.3f}  [SC predicts > 0: female brilliant]")
    print(f"  sc comp=male   D_gender={m_m:+.3f}  [SC predicts < 0: male brilliant]")
    print(f"  ss  D_gender={m_ss:+.3f}   noscene D_gender={m_ns:+.3f}  [baseline gender prior]")
    if m_f > 0 and m_m < 0:
        print("  => D_gender FLIPS sign with complement -> SYNTACTIC (SC-role) effect.")
    elif (m_f > 0) == (m_m > 0):
        print("  => D_gender keeps the SAME sign -> looks like a FIXED gender prior, not syntax.")

    df.to_csv(os.path.join(out_dir, f"{tag}_sc_brilliance_rows.csv"),
              index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nmean words parsed/session: {df['n_scored'].mean():.1f}/{len(ALL_WORDS)}")

# =========================================================
# Main
# =========================================================
def _slug(s): return re.sub(r"[^A-Za-z0-9.\-]+", "_", s).strip("_")

def main():
    ap = argparse.ArgumentParser(
        description="RQ2a: SC syntax on a nonsense skill with gender labels, brilliance vs effort.")
    ap.add_argument("--provider", required=True, choices=list(PROVIDER_DEFAULTS.keys()))
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=N_PER_CONDITION, help="trials per condition (sc gets 2x)")
    ap.add_argument("--temp", type=float, default=TEMPERATURE)
    args = ap.parse_args()

    provider = args.provider
    model = args.model or PROVIDER_DEFAULTS[provider]
    temperature = args.temp

    tag = f"{_slug(model)}"
    output_path = os.path.join(OUTPUT_DIR, f"{tag}_wat_sc_brilliance.xlsx")

    # sc spans both complement directions -> 2x trials (~n per direction)
    trials = ["sc"] * (2 * args.n) + ["ss"] * args.n + ["noscene"] * args.n
    random.shuffle(trials)

    print(f"provider   = {provider}")
    print(f"model      = {model}")
    print(f"gender     = {GENDER_PAIRS}  (random pair per trial; sex-pooled)")
    print(f"poles      = {BRILLIANCE_WORDS}  vs  {EFFORT_WORDS}")
    print(f"skills     = {NONSENSE_SKILLS}")
    print(f"conditions = sc(2n, subj/comp sex randomized), ss, noscene  @ n={args.n}  -> {len(trials)} trials")
    print(f"output     = {output_path}\n")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = output_path.replace(".xlsx", ".jsonl")
    print(f"  incremental backup -> {jsonl_path}\n")

    results = []
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, cond in enumerate(trials, start=1):
            row = run_trial(i, cond, provider, model, temperature)
            results.append(row)
            jf.write(json.dumps(row, ensure_ascii=False) + "\n"); jf.flush()
    df = pd.DataFrame(results)

    df.to_excel(output_path, index=False)
    print(f"\nSaved {len(df)} rows -> {os.path.abspath(output_path)}")
    analyze(df, out_dir, tag)

if __name__ == "__main__":
    main()
