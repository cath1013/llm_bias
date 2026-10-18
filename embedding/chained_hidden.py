#!/usr/bin/env python3
"""
CHAINED BAI-STYLE EXPERIMENT — Task 1 (WAT) -> Task 2 (profiles) -> Task 3
(relative decision), paired with Task-1 hidden states, for Llama 3.1 and Scout.

WHAT ONE ROW GIVES
    From ONE chained prompt (`embedding/chained_prompts.csv`) and ONE generated
    continuation:
        1. Task-1 hidden-state representation bias   delta_layer*, weat_score_layer*
        2. Task-1 behavioural WAT score              behaviour_wat_score
        3. Task-2 generated profiles                 response_2
        4. Task-3 downstream decision                decision_bias
    The WAT-only experiment (`embed_bias_llama3_hidden.py`,
    `embed_bias_scout_hidden.py`, `prompts.csv`) is untouched; this is a
    separate design file, runner and output directory.

REUSED, NOT RE-IMPLEMENTED
    Model loading, generation and the hidden-state forward pass are the selected
    backend's (`embedding.backends`, chosen once with --model).  Span
    resolution, competitor masking, mean pooling, role resolution, the strict
    WAT parser, the unsmoothed Bai score, delta / standardized WEAT and the
    prototype accumulator are `embedding.core`; the design is
    `embedding.design`.  Task 2/3 wording and the Task-3 judges are
    `decision_text.py`, the code the decision tasks render with.

TASK-1 ONLY REPRESENTATIONS
    The chained prompt BEGINS with the frozen WAT prompt, byte for byte, so the
    user message up to the end of Task 1 — and its token IDs — is identical to
    the WAT-only input.  Spans are resolved inside that exact prefix window
    (characters 0 .. len(WAT prompt) of the user message), never in Task 2 or 3.  Causal attention
    already keeps later text out of Task-1 token states; the window keeps a
    target or field REPEATED later from ever being the occurrence selected.

NOTATION (fixed throughout this file and its outputs)
    prompt level   delta_layer*       = s(X,A,B) - s(Y,A,B)                (Δ)
                   weat_score_layer*  = Δ / (population SD of {sX, sY} + 0.001)
                                      = Δ / (|Δ|/2 + 0.001) for singleton X, Y
    set level      weat_score         = [mean s(X) - mean s(Y)]
                                        / population SD of s over X ∪ Y
    with s(w,A,B) = mean cos(w,A) - mean cos(w,B).  The formulas are the WAT-only
    experiment's; only the names are new.

ORIENTATION (positive = stereotype-congruent everywhere)
    gender  X = male, Y = female, A = brilliance, B = condition control;
            decision_bias = 1 when the male person gets the brilliance role
    fab     X = brilliance pole, Y = effort pole, A = high-FAB, B = low-FAB;
            decision_bias = 1 when the brilliance person gets the high-FAB class

RESPONSE SECTIONS
    The model writes one continuation; `response_full` is always stored.  A
    strict parser splits it at line-initial "Task 1" / "Task 2" / "Task 3"
    labels (optionally markdown-decorated, followed by ':' '.' ')' '-' or end of
    line).  Each label must occur exactly once, in order, with a non-empty body;
    otherwise response_1..3 stay empty, `section_parse_error` says why, and
    nothing is guessed.  The WAT is scored from response_1 only (strict parser
    unchanged; a malformed Task 1 gives NaN and keeps the representation).  The
    decision is judged from response_2 + response_3 only.

TASK-3 DECISION — A SEPARATE JUDGE STAGE
    The decision tasks code Task 3 with an LLM judge (default:
    bai_core.JUDGE_PROVIDER_DEFAULT).  That needs network access and an API
    key, which the GPU run should not depend on, so generation writes
    decision_status = pending_judge and `--judge` fills the decision columns
    later from the saved responses (representation columns are carried through
    byte for byte).  The judge sees exactly the two-task text it was written
    for: "Task 1: <response_2>\\n\\nTask 2: <response_3>" — the WAT section is
    never shown to it.  Its full prompt and raw output are stored.

OUTPUT  (responses/embed_llama31_chained/ or responses/embed_scout_chained/)
    prompt_weat.csv               one row per chained prompt (primary, wide)
    set_weat.csv                  full-set WEAT score per condition x layer
    stimulus_layer.csv            stimulus x layer prototype audit table
    stimulus_prototypes.npz       the prototype vectors it refers to
    behaviour_parse_failures.csv  Task-1 rows whose WAT could not be scored
    decision_parse_failures.csv   rows whose decision could not be scored
    manifest.json

USAGE
    python -m embedding.chained_hidden --model scout --dry-run
    python -m embedding.chained_hidden --model scout --smoke-test
    python -m embedding.chained_hidden --model scout --limit 5 \\
        --out-dir responses/embed_scout_chained_diag5
    python -m embedding.chained_hidden --model scout
    python -m embedding.chained_hidden --model scout --judge
    python -m embedding.chained_hidden --model llama31 \\
        --checkpoint <path to Llama-3.1-8B-Instruct>
"""

import os
import re
import sys
import csv
import json
import math
import time
import argparse
from collections import defaultdict
from datetime import datetime, timezone

from embedding import ROOT, backends, core, design
import decision_text as dt

# A backend's chained analysis version and output directory are its own
# (CHAINED_ANALYSIS_VERSION, CHAINED_OUT_DIR); the WAT-only result directories
# of every backend are off limits to this runner.
WAT_ONLY_DIRS = backends.wat_out_dirs()

# Same sampling as the WAT-only run (temperature 0.3, do_sample, top_p 1, top_k 0);
# only the length budget is larger, because one continuation now holds a WAT
# answer, two profiles and a recommendation.  Truncation is flagged per row.
CHAINED_MAX_NEW_TOKENS = 1024

PROMPT_WEAT_CSV = "prompt_weat.csv"
SET_WEAT_CSV = "set_weat.csv"
STIMULUS_LAYER_CSV = "stimulus_layer.csv"
PROTOTYPES_NPZ = "stimulus_prototypes.npz"
BEHAVIOUR_FAILURES_CSV = "behaviour_parse_failures.csv"
DECISION_FAILURES_CSV = "decision_parse_failures.csv"
OUTPUT_FILES = [PROMPT_WEAT_CSV, SET_WEAT_CSV, STIMULUS_LAYER_CSV, PROTOTYPES_NPZ,
                BEHAVIOUR_FAILURES_CSV, DECISION_FAILURES_CSV, "manifest.json"]

DESIGN_COLUMNS = ["group0", "group1", "attributes", "t3_x_stimulus",
                  "t3_y_stimulus", "t3_x", "t3_y", "t3_a", "t3_b", "task1_order",
                  "person_order", "option_order", "t3_frame", "t3_relation"]
DECISION_COLUMNS = ["decision_bias", "decision_status", "decision_option_a_to",
                    "judge_provider", "judge_model", "judge_prompt",
                    "judge_response", "judge_answer", "judge_resolved_by",
                    "judge_name", "judge_label_fixed", "decision_error"]


def layer_columns(n_hidden_states):
    return ([f"delta_layer{i}" for i in range(n_hidden_states)]
            + [f"weat_score_layer{i}" for i in range(n_hidden_states)])


def prompt_weat_columns(n_hidden_states):
    return (["model", "prompt_id", "domain", "category", "trial"] + DESIGN_COLUMNS
            + ["prompt", "response_full", "response_1", "response_2", "response_3",
               "section_parse_error", "behaviour_wat_score",
               "behaviour_parse_error"]
            + DECISION_COLUMNS
            + ["seed", "temperature", "max_new_tokens", "n_new_tokens",
               "truncated", "generate_seconds", "n_prompt_tokens"]
            + layer_columns(n_hidden_states))


# =========================================================
# Response sections
# =========================================================

_SECTION_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_]{1,3}[ \t]*)?"
    r"Task[ \t]*([123])(?![0-9])"
    r"[ \t]*(?:[*_]{1,3}[ \t]*)?"
    r"(?:[:.)\-–—](?:[ \t]*[*_]{1,3})?|(?=[ \t]*$))",
    re.I | re.M)


def split_sections(text):
    """
    ((response_1, response_2, response_3), []) or (None, [problems]).

    A label counts only at the start of a line and only when followed by a
    delimiter or the end of that line, so prose such as "Task 3 asks..." is
    not a label.  Exactly one label per task, in order, each with a body —
    anything else is reported, never repaired.
    """
    text = text or ""
    hits = list(_SECTION_RE.finditer(text))
    nums = [int(m.group(1)) for m in hits]
    problems = []
    for n in (1, 2, 3):
        c = nums.count(n)
        if c == 0:
            problems.append(f"no 'Task {n}' label")
        elif c > 1:
            problems.append(f"'Task {n}' label appears {c} times")
    if not problems and nums != [1, 2, 3]:
        problems.append(f"task labels out of order {nums}")
    if problems:
        return None, problems
    bodies = []
    for k, m in enumerate(hits):
        end = hits[k + 1].start() if k + 1 < len(hits) else len(text)
        bodies.append(text[m.end():end].strip())
    for n, body in zip((1, 2, 3), bodies):
        if not body:
            problems.append(f"'Task {n}' section is empty")
    if problems:
        return None, problems
    return tuple(bodies), []


# =========================================================
# Task-1 window
# =========================================================

def task1_window(templated, user_window, row):
    """Character range of the Task-1 WAT text inside the templated input."""
    start = user_window[0] + int(row["task1_start"])
    end = user_window[0] + int(row["task1_end"])
    if templated[start:end] != row["task1_prompt"]:
        raise core.SpanError(f"{row['prompt_id']}: the Task-1 window does not hold "
                           f"the Task-1 prompt; refusing to locate spans")
    return start, end


# =========================================================
# Backends — thin adapters over the two WAT-only runners
# =========================================================

def generation_config(args):
    return dict(core.GENERATION, max_new_tokens=args.max_new_tokens)


# =========================================================
# One chained prompt
# =========================================================

def score_behaviour(response_1, row, roles):
    """(score, problem) — the WAT-only strict parser and unsmoothed score."""
    assign, problems = core.parse_behaviour(response_1, row, roles)
    if problems:
        return float("nan"), " | ".join(problems), len(assign)
    score, denom = core.behaviour_wat_score(assign, row, roles)
    return score, (denom or ""), len(assign)


def run_one(row, stim, backend, *, args, acc=None):
    """(prompt_weat row, behaviour failure or None, decision failure or None, meta)."""
    tok = backend.tokenizer
    roles = core.prompt_roles(row, stim)
    templated, user_window = core.build_chat_input(tok, core.SYSTEM_PROMPT, row["prompt"])
    window = task1_window(templated, user_window, row)
    seed = core.prompt_seed(row["prompt_id"], args.seed)
    cfg = generation_config(args)

    response, gen_meta = backend.generate(templated, seed=seed, generation=cfg)
    vectors, meta = backend.extract_hidden(templated, roles["phrases"],
                                           user_window=window,
                                           competitors=roles["phrases"],
                                           context=row["prompt_id"])
    want = backend.meta["n_transformer_blocks"] + 1
    if meta["n_hidden_states"] != want:
        raise core.ArchitectureMismatch(
            f"{row['prompt_id']}: {meta['n_hidden_states']} hidden states, "
            f"config implies {want} — aborting the run")
    offsets = token_offsets(tok, templated)
    for phrase, (lo, hi) in meta["spans"].items():
        if not (window[0] <= offsets[lo][0] and offsets[hi - 1][1] <= window[1]):
            raise core.SpanError(f"{row['prompt_id']}: {phrase!r} resolved outside "
                               f"Task 1")

    sections, sec_problems = split_sections(response)
    r1, r2, r3 = sections if sections else ("", "", "")
    beh_fail = dec_fail = None
    if sections is None:
        score, beh_problem, n_assigned = (float("nan"),
                                          "section parse failed: "
                                          + " | ".join(sec_problems), 0)
    else:
        score, beh_problem, n_assigned = score_behaviour(r1, row, roles)
    if beh_problem:
        beh_fail = dict(prompt_id=row["prompt_id"], domain=row["domain"],
                        category=row["category"], trial=row["trial"],
                        n_items_assigned=n_assigned,
                        n_items_presented=len(roles["attributes"]),
                        problems=beh_problem,
                        response=r1 if sections else response)

    deltas, weats = core.representation_scores(vectors, roles, meta["n_hidden_states"])
    out = {"model": backend.name, "prompt_id": row["prompt_id"],
           "domain": row["domain"], "category": row["category"],
           "trial": int(row["trial"])}
    out.update({c: row[c] for c in DESIGN_COLUMNS})
    out.update(prompt=row["prompt"], response_full=response, response_1=r1,
               response_2=r2, response_3=r3,
               section_parse_error=" | ".join(sec_problems),
               behaviour_wat_score=score, behaviour_parse_error=beh_problem)
    out.update({c: "" for c in DECISION_COLUMNS})
    if sections is None:
        out.update(decision_status="section_parse_failed",
                   decision_error="; ".join(sec_problems))
        dec_fail = decision_failure(out)
    else:
        out["decision_status"] = "pending_judge"
    out.update(seed=seed, temperature=cfg["temperature"],
               max_new_tokens=cfg["max_new_tokens"],
               n_new_tokens=gen_meta["n_new_tokens"],
               truncated=int(gen_meta["n_new_tokens"] >= cfg["max_new_tokens"]),
               generate_seconds=round(gen_meta["generate_seconds"], 3),
               n_prompt_tokens=meta["n_prompt_tokens"])
    for i, v in enumerate(deltas):
        out[f"delta_layer{i}"] = v
    for i, v in enumerate(weats):
        out[f"weat_score_layer{i}"] = v

    if acc is not None:
        core.accumulate_prompt(acc, row["domain"], row["category"], vectors,
                             roles["phrases"], row["prompt_id"])
    return out, beh_fail, dec_fail, dict(meta, **gen_meta)


def token_offsets(tok, templated):
    """[(char_start, char_end)] per token of the templated input."""
    offsets = tok(templated, return_offsets_mapping=True)["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], (list, tuple)) and offsets[0] \
            and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    return [(int(a), int(b)) for a, b in offsets]


def run_prompts(rows, stim, backend, *, args, acc=None, verbose=True):
    out, beh, dec = [], [], []
    n_hidden = None
    t0 = time.time()
    for k, row in enumerate(rows, start=1):
        r, bf, df, meta = run_one(row, stim, backend, args=args, acc=acc)
        n_hidden = meta["n_hidden_states"]
        out.append(r)
        if bf:
            beh.append(bf)
        if df:
            dec.append(df)
        if verbose:
            el = time.time() - t0
            print(f"  [{k:3d}/{len(rows)}] {row['prompt_id']:<20} "
                  f"wat={r['behaviour_wat_score']:+.3f} "
                  f"sections={'ok' if not r['section_parse_error'] else 'FAIL'} "
                  f"Δ_L{n_hidden - 1}={r[f'delta_layer{n_hidden - 1}']:+.5f} "
                  f"tok={r['n_new_tokens']}{' TRUNC' if r['truncated'] else ''} "
                  f"({el / k:.1f}s/prompt)", flush=True)
    return out, beh, dec, dict(n_hidden_states=n_hidden,
                               total_seconds=time.time() - t0)


# =========================================================
# Task-3 judge stage
# =========================================================

def judge_text(response_2, response_3):
    """The two-task text the decision judges were written for."""
    return f"Task 1: {response_2}\n\nTask 2: {response_3}"


def decision_failure(r):
    return dict(prompt_id=r["prompt_id"], domain=r["domain"],
                category=r["category"], trial=r["trial"],
                decision_status=r["decision_status"],
                decision_error=r["decision_error"],
                judge_answer=r.get("judge_answer", ""),
                judge_response=r.get("judge_response", ""),
                response_3=r.get("response_3", ""),
                response_full=r.get("response_full", ""))


def judge_row(r, call, jp, jm):
    """Fill one row's decision columns.  `call(messages) -> raw judge text`."""
    if r.get("section_parse_error"):
        r.update(decision_bias="", decision_status="section_parse_failed",
                 decision_option_a_to="", judge_provider="", judge_model="",
                 judge_prompt="", judge_response="", judge_answer="",
                 judge_resolved_by="", judge_name="", judge_label_fixed="",
                 decision_error=r["section_parse_error"])
        return r
    text = judge_text(r["response_2"], r["response_3"])
    if r["domain"] == "gender":
        msgs = dt.build_adj_judge_messages(r["t3_x"], r["t3_y"], r["t3_a"],
                                           r["t3_frame"], text, "words")
    else:
        msgs = dt.build_fab_judge_messages(r["t3_x"], r["t3_y"], r["t3_a"], text)
    r.update(judge_provider=jp, judge_model=jm,
             judge_prompt=json.dumps(msgs, ensure_ascii=False))
    try:
        raw = call(msgs)
    except Exception as exc:                                     # noqa: BLE001
        r.update(decision_bias="", decision_status="judge_error",
                 decision_option_a_to="", judge_response="", judge_answer="",
                 judge_resolved_by="", judge_name="", judge_label_fixed="",
                 decision_error=f"{type(exc).__name__}: {exc}")
        return r
    if r["domain"] == "gender":
        assigned, how, name, fixed = dt.resolve_judge(raw, r["t3_x"], r["t3_y"])
    else:
        assigned, how, name, fixed = dt.parse_robust_judge(raw), "answer", "", 0
    r.update(judge_response=raw, judge_answer=assigned or "UNCLEAR",
             judge_resolved_by=how, judge_name=name, judge_label_fixed=fixed)
    if assigned is None:
        r.update(decision_bias="", decision_status="judge_unclear",
                 decision_option_a_to="",
                 decision_error="judge could not tell who received option A")
    else:
        # judge person A is ALWAYS t3_x (male / brilliance person), and the role
        # it is asked about is ALWAYS t3_a (brilliance role / high-FAB class)
        r.update(decision_bias=int(assigned == "A"), decision_status="scored",
                 decision_option_a_to="X" if assigned == "A" else "Y",
                 decision_error="")
    return r


def judge_rows(rows, call, jp, jm, *, only_pending=True):
    failures = []
    for r in rows:
        if only_pending and r.get("decision_status") not in ("pending_judge",
                                                              "judge_error"):
            if r.get("decision_status") != "scored":
                failures.append(decision_failure(r))
            continue
        judge_row(r, call, jp, jm)
        if r["decision_status"] != "scored":
            failures.append(decision_failure(r))
    return rows, failures


def default_judge_call(jp, jm):
    sys.path.insert(0, os.path.join(ROOT, "replicate"))
    from bai_core import call_with_retry, JUDGE_TEMPERATURE
    return lambda msgs: call_with_retry(jp, jm, msgs, JUDGE_TEMPERATURE)


# =========================================================
# Set level
# =========================================================

def stimulus_rows(acc, rows_meta, stim, model_key, analysis_version):
    out = []
    for domain, category in sorted({(r["domain"], r["category"]) for r in rows_meta}):
        if not acc.stimuli(domain, category):
            continue
        arm = core.arm_for(domain, category, stim)
        for r in core.stimulus_layer_rows(acc, domain, category, arm,
                                        n_hidden_states=_n_levels(acc),
                                        analysis_version=analysis_version):
            r = dict(model=model_key, **r)
            r.update(prototype_file=PROTOTYPES_NPZ,
                     analysis_version=analysis_version)
            r.pop("analysis_role", None)
            out.append(r)
    return out


def _n_levels(acc):
    return next(iter(acc.sums.values())).shape[0]


def set_weat_rows(stim_rows, stim, model_key, analysis_version):
    """
    The conventional full-set WEAT score, from the stimulus-level associations:
        weat_score = [mean s(X) - mean s(Y)] / population SD of s over X ∪ Y
    """
    import numpy as np
    groups = defaultdict(list)
    for r in stim_rows:
        groups[(r["domain"], r["category"], r["hidden_state_index"])].append(r)
    out = []
    for (domain, category, i), rs in sorted(groups.items()):
        arm = core.arm_for(domain, category, stim)
        sX = [r["s_w_A_B"] for r in rs if r["role"] == "X"]
        sY = [r["s_w_A_B"] for r in rs if r["role"] == "Y"]
        if not sX or not sY:
            continue
        sd = float(np.std(np.asarray(sX + sY, dtype=float), ddof=0))
        score = (float(np.mean(sX) - np.mean(sY)) / sd
                 if sd > 0 and math.isfinite(sd) else float("nan"))
        lab = core.layer_label(i)
        out.append(dict(
            model=model_key, domain=domain, category=category,
            hidden_state_index=lab["hidden_state_index"],
            transformer_layer=lab["transformer_layer"],
            representation_type=lab["representation_type"],
            weat_score=score, mean_s_X=float(np.mean(sX)),
            mean_s_Y=float(np.mean(sY)), sd_s_XY=sd,
            n_X=len(sX), n_Y=len(sY),
            n_A=sum(1 for r in rs if r["role"] == "A"),
            n_B=sum(1 for r in rs if r["role"] == "B"),
            X_name=arm["X_name"], Y_name=arm["Y_name"],
            A_name=arm["A_name"], B_name=arm["B_name"],
            sd_convention="population (ddof=0) over X ∪ Y",
            pooling=core.POOLING, analysis_version=analysis_version))
    return out


# =========================================================
# Writing
# =========================================================

def write_table(path, rows, columns, force):
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite {path} — rerun with --force")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    print(f"  wrote {path}  ({len(rows)} rows)")


def failure_columns(kind):
    if kind == "behaviour":
        return ["prompt_id", "domain", "category", "trial", "n_items_assigned",
                "n_items_presented", "problems", "response"]
    return ["prompt_id", "domain", "category", "trial", "decision_status",
            "decision_error", "judge_answer", "judge_response", "response_3",
            "response_full"]


def check_out_dir(out_dir):
    real = os.path.realpath(out_dir)
    for d in WAT_ONLY_DIRS:
        if real == os.path.realpath(d):
            raise SystemExit(f"{out_dir} is a WAT-only result directory; the "
                             f"chained runner never writes there")


def build_manifest(*, model_key, analysis_version, backend_file, backend_meta,
                   n_hidden_states, args, rows, n_behaviour_failures, bos):
    import torch, transformers
    files = {"chained_prompts_csv": design.CHAINED_CSV, "prompts_csv": design.PROMPTS_CSV,
             "stimuli_csv": design.STIMULI_CSV, "decision_text": dt.__file__,
             "design": design.__file__, "core": core.__file__,
             "chained_runner": __file__, "backend": backend_file}
    return {
        "analysis": "Chained Bai-style experiment: Task 1 WAT -> Task 2 profiles "
                    "-> Task 3 relative decision, with Task-1 hidden states",
        "analysis_version": analysis_version,
        "model": model_key,
        "generated": datetime.now(timezone.utc).isoformat(),
        "inputs": {k: {"path": os.path.relpath(v, ROOT),
                       "sha256": design.file_digest(v)} for k, v in files.items()},
        "n_prompts": len(rows),
        "backend": backend_meta,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "n_transformer_blocks": backend_meta["n_transformer_blocks"],
        "n_hidden_states": n_hidden_states,
        "layer_convention": {
            "layer0": "hidden_states[0], the input embedding",
            "layer1..layerN": "output of transformer block 1..N (layerN after the "
                              "final norm)"},
        "system_prompt": core.SYSTEM_PROMPT,
        "chat_template": "apply_chat_template([system, user], tokenize=False, "
                         "add_generation_prompt=True), then tokenizer(templated) — "
                         "the WAT-only pipeline",
        "bos": bos,
        "generation": generation_config(args),
        "generation_note": "temperature, do_sample, top_p and top_k are the WAT-only "
                           "run's; `truncated` flags rows that hit max_new_tokens",
        "max_new_tokens": {
            "chained": args.max_new_tokens,
            "wat_only": core.GENERATION["max_new_tokens"],
            "why_different": "the chained continuation holds three task answers "
                             "(the WAT assignment, two profiles and the decision) "
                             "where the WAT-only run generated the WAT answer only"},
        "generation_seed_rule": core.SEED_RULE,
        "generation_seed_base": args.seed,
        "hidden_states_from": "the input prompt only, Task-1 window only",
        "task1_window": "character range of the WAT text inside the user message "
                        "(chained_prompts.csv task1_start/task1_end); spans "
                        "outside it are never resolved",
        "pooling": core.POOLING,
        "pooling_rule": "mean over every sub-token of the stimulus phrase",
        "orientation": {
            "gender": "X = male, Y = female, A = brilliance, B = condition control",
            "fab": "X = brilliance pole, Y = effort pole, A = high-FAB, B = low-FAB",
            "decision_bias": "1 = X person received option A (brilliance role / "
                             "high-FAB class); 0 = Y did",
            "meaning": "positive = stereotype-congruent for every score"},
        "notation": {
            "delta_layer*": "s(X,A,B) - s(Y,A,B)",
            "weat_score_layer*": "delta / (population SD of {sX, sY} + 0.001)",
            "set weat_score": "[mean s(X) - mean s(Y)] / population SD of s over "
                              "X ∪ Y (prototypes over this experiment's prompts)"},
        "sections": {
            "labels": "line-initial 'Task 1' / 'Task 2' / 'Task 3', optional "
                      "markdown, then ':' '.' ')' '-' or end of line",
            "rule": "each exactly once, in order, non-empty; otherwise "
                    "response_1..3 empty and section_parse_error set; "
                    "response_full is always kept",
            "behaviour_from": "response_1 only",
            "decision_from": "response_2 + response_3 only"},
        "behaviour": {"metric": "behaviour_wat_score, Bai assignment score, "
                                "unsmoothed, strict parser unchanged",
                      "n_failures": n_behaviour_failures},
        "judge": {"status": "pending — run --judge"},
        "fail_fast": ["non-finite hidden state", "non-finite pooled vector",
                      "unresolvable/ambiguous Task-1 span",
                      "hidden-state count != num_hidden_layers + 1"],
        "not_run": {"pca_nullification": False, "legacy_response_replay": False},
        "outputs": OUTPUT_FILES,
        "cli": vars(args),
    }


# =========================================================
# Main
# =========================================================

def load_design(path=None):
    stim = design.load_stimuli()
    rows = design.read_chained(path or design.CHAINED_CSV)
    fresh = design.build_chained(stim)
    if rows != [{c: (int(v) if c in ("trial", "task1_start", "task1_end") else v)
                 for c, v in r.items()} for r in fresh]:
        raise SystemExit(f"{path or design.CHAINED_CSV} differs from the regenerated "
                         f"design; refusing to run on it")
    return stim, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True, choices=backends.names())
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gpu", type=int, default=0, help="llama31 only")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=core.SEED_BASE)
    ap.add_argument("--max-new-tokens", type=int, default=CHAINED_MAX_NEW_TOKENS)
    ap.add_argument("--domain", choices=("gender", "fab", "both"), default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the chained design; load no model")
    ap.add_argument("--smoke-test", action="store_true",
                    help="one gender + one FAB prompt, full diagnostics, writes nothing")
    ap.add_argument("--judge", action="store_true",
                    help="judge stage: fill Task-3 decisions in an existing "
                         "prompt_weat.csv (no model load)")
    ap.add_argument("--rejudge-all", action="store_true",
                    help="with --judge: re-judge every row, not just pending ones")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()
    backend_module = backends.get_backend(args.model)     # the only model choice
    out_dir = args.out_dir or backend_module.CHAINED_OUT_DIR
    check_out_dir(out_dir)

    if args.judge:
        return judge_stage(args, out_dir)

    stim, rows = load_design()
    print(f"Chained Bai-style experiment — {args.model}  "
          f"(analysis {backend_module.CHAINED_ANALYSIS_VERSION})")
    print(f"  design     : {design.CHAINED_CSV}  ({len(rows)} chained prompts)")
    if args.domain != "both":
        rows = [r for r in rows if r["domain"] == args.domain]
    if args.smoke_test:
        rows = [next(r for r in rows if r["domain"] == d)
                for d in ("gender", "fab") if any(x["domain"] == d for x in rows)]
    elif args.limit:
        rows = rows[:args.limit]
    for r in rows:
        core.prompt_roles(r, stim)
        if r["prompt"][r["task1_start"]:r["task1_end"]] != r["task1_prompt"]:
            raise SystemExit(f"{r['prompt_id']}: Task-1 window broken")
    print(f"  roles and Task-1 windows checked for {len(rows)} prompts")
    print(f"  generation : {generation_config(args)}")
    if args.dry_run:
        print("\n--dry-run: design validated, no model loaded, nothing written.")
        return

    backend = backend_module.load(args.checkpoint, args.gpu)
    print(f"  loaded {backend.meta['model_class']} "
          f"({backend.meta['n_transformer_blocks']} blocks)")
    if args.smoke_test:
        return smoke_test(rows, stim, backend, args)

    acc = core.PrototypeAccumulator()
    out, beh, dec, meta = run_prompts(rows, stim, backend, args=args, acc=acc)
    n_hidden = meta["n_hidden_states"]
    version = backend_module.CHAINED_ANALYSIS_VERSION
    stim_rows = stimulus_rows(acc, rows, stim, args.model, version)
    set_rows = set_weat_rows(stim_rows, stim, args.model, version)

    os.makedirs(out_dir, exist_ok=True)
    write_table(os.path.join(out_dir, PROMPT_WEAT_CSV), out,
                prompt_weat_columns(n_hidden), args.force)
    write_table(os.path.join(out_dir, SET_WEAT_CSV), set_rows,
                list(set_rows[0]) if set_rows else [], args.force)
    write_table(os.path.join(out_dir, STIMULUS_LAYER_CSV), stim_rows,
                list(stim_rows[0]) if stim_rows else [], args.force)
    write_table(os.path.join(out_dir, BEHAVIOUR_FAILURES_CSV), beh,
                failure_columns("behaviour"), args.force)
    write_table(os.path.join(out_dir, DECISION_FAILURES_CSV), dec,
                failure_columns("decision"), args.force)
    import numpy as np
    np.savez_compressed(os.path.join(out_dir, PROTOTYPES_NPZ),
                        **{f"{d}|{a}|{s}": acc.prototype(d, a, s)
                           for (d, a, s) in acc.sums})
    tok = backend.tokenizer
    first, _ = core.build_chat_input(tok, core.SYSTEM_PROMPT, rows[0]["prompt"])
    manifest = build_manifest(model_key=args.model, analysis_version=version,
                              backend_file=backend_module.__file__,
                              backend_meta=backend.meta,
                              n_hidden_states=n_hidden, args=args, rows=out,
                              n_behaviour_failures=len(beh),
                              bos=core.bos_record(tok, first))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  wrote {os.path.join(out_dir, 'manifest.json')}")
    print(f"  next: --model {args.model} --judge  (fills decision_bias)")


def judge_stage(args, out_dir):
    path = os.path.join(out_dir, PROMPT_WEAT_CSV)
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; run the generation stage first")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    sys.path.insert(0, os.path.join(ROOT, "replicate"))
    from bai_core import PROVIDER_DEFAULTS, JUDGE_PROVIDER_DEFAULT, JUDGE_TEMPERATURE
    jp = args.judge_provider or JUDGE_PROVIDER_DEFAULT
    jm = args.judge_model or PROVIDER_DEFAULTS[jp]
    print(f"judging {len(rows)} rows in {path} with {jp} / {jm}")
    rows, failures = judge_rows(rows, default_judge_call(jp, jm), jp, jm,
                                only_pending=not args.rejudge_all)
    write_table(path, rows, columns, force=True)
    write_table(os.path.join(out_dir, DECISION_FAILURES_CSV), failures,
                failure_columns("decision"), force=True)
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
        status = defaultdict(int)
        for r in rows:
            status[r["decision_status"]] += 1
        manifest["judge"] = {
            "provider": jp, "model": jm, "temperature": JUDGE_TEMPERATURE,
            "judged": datetime.now(timezone.utc).isoformat(),
            "gender_judge": "decision_text.build_adj_judge_messages (people_mode="
                            "'words') + resolve_judge; A = t3_x (male)",
            "fab_judge": "decision_text.build_fab_judge_messages + "
                         "parse_robust_judge; A = t3_x (brilliance person)",
            "judge_input": "'Task 1: <response_2>\\n\\nTask 2: <response_3>' — the "
                           "WAT section is never shown",
            "status_counts": dict(status)}
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
    print(f"  {sum(1 for r in rows if r['decision_status'] == 'scored')} scored, "
          f"{len(failures)} not scoreable")


def smoke_test(rows, stim, backend, args):
    import torch
    tok = backend.tokenizer
    for row in rows:
        print(f"\n{'=' * 74}\nSMOKE — {row['prompt_id']}\n{'=' * 74}")
        print(row["prompt"])
        out, bf, df, meta = run_one(row, stim, backend, args=args)
        print(f"\n  response ({out['n_new_tokens']} tokens"
              f"{', TRUNCATED' if out['truncated'] else ''}):")
        for line in out["response_full"].splitlines():
            print(f"    {line}")
        print(f"\n  sections             : "
              f"{'ok' if not out['section_parse_error'] else out['section_parse_error']}")
        print(f"  behaviour_wat_score  : {out['behaviour_wat_score']}"
              f"  {out['behaviour_parse_error']}")
        print(f"  hidden states        : {meta['n_hidden_states']}")
        templated, uw = core.build_chat_input(tok, core.SYSTEM_PROMPT, row["prompt"])
        win = task1_window(templated, uw, row)
        for phrase, (lo, hi) in meta["spans"].items():
            print(f"    {phrase:<26} tokens [{lo}, {hi})  {meta['subtokens'][phrase]}"
                  f"  (Task-1 window chars {win})")
        last = meta["n_hidden_states"] - 1
        print(f"  delta layer0 / layer{last}: {out['delta_layer0']:+.5f} / "
              f"{out[f'delta_layer{last}']:+.5f}")
        print(f"  decision_status      : {out['decision_status']}")
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                print(f"  cuda:{d} peak allocated : "
                      f"{torch.cuda.max_memory_allocated(d) / (1 << 30):.2f} GiB")
    print(f"\n{'=' * 74}\nSMOKE TEST COMPLETE — nothing written\n{'=' * 74}")


if __name__ == "__main__":
    main()
