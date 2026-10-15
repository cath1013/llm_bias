#!/usr/bin/env python3
"""
FIXED-PROMPT BAI-STYLE WAT — paired behaviour and hidden states, Llama-3.1-8B-Instruct.

WHAT THIS IS
    One frozen set of 272 WAT prompts (`embedding/prompts.csv`) is sent to
    Llama 3.1 twice over:

        the same prompt -> generate()  -> the model's own WAT response
                        -> forward()   -> contextualised hidden states

    so the behavioural score and the representational score on every row come
    from the SAME model reading the SAME words in the SAME order.  "Paired" here
    means exactly that and nothing more: there is no profile stage and no
    decision stage, only the WAT prompt.

WHAT CHANGED FROM THE PREVIOUS VERSION
    The earlier runner replayed saved behavioural JSONL from GPT, Claude,
    Gemini, Scout and gpt-oss — 3420 prompts whose behaviour belonged to other
    models, so the representation->behaviour link could never be drawn.  That
    design is retired here.  Nothing in this file reads a response JSONL, a
    behavioural model id, or `eb.load_sessions`; the old corpus is legacy/pilot
    material and is never pooled into this experiment.

    The primary table is also no longer long.  It is Bai-like and WIDE: one row
    per prompt, with `weat_layer0..32` and `delta_layer0..32` across the row.

DESIGN SOURCE
    `embedding/stimuli.csv`   — every stimulus word, and the ROLE of each word
    `embedding/prompts.csv`   — the frozen 272-prompt design, from
                                `make_bai_style_prompts.py`
    This runner randomises nothing.  Target pairing, target order and attribute
    order are all fixed in prompts.csv and merely read back here.

ORIENTATION — THE SIGN NEVER DEPENDS ON PRESENTATION ORDER
    Roles come from stimuli.csv, never from `group0`/`group1`:
        gender : X = male target,            Y = female target
                 A = brilliance attributes,  B = that condition's control
        fab    : X = brilliance pole,        Y = effort pole
                 A = high-FAB fields,        B = low-FAB fields
    So positive `delta_layer*`, positive `weat_layer*` and positive
    `behaviour_wat_score` all mean the same thing — stereotype-congruent — whether
    the congruent target happened to be shown first or second.

TWO REPRESENTATION SCORES PER LAYER
        s(w,A,B) = mean cos(w,A) - mean cos(w,B)
        delta    = s(X,A,B) - s(Y,A,B)                      [unstandardised]
        weat     = [mean s(X) - mean s(Y)] / [pop_SD(s(X),s(Y)) + 0.001]
    Both are kept because each loses something the other has.  With one target
    per side the standardized score reduces to delta/(|delta|/2 + 0.001), which
    saturates near +-2 for an association difference of any size; delta keeps
    the magnitude but is not comparable across layers of different scale.
    `s_group0_layer*` / `s_group1_layer*` are deliberately not written: they are
    presentation-order-indexed, which is precisely what must not drive a sign.

BEHAVIOURAL SCORE  (`behaviour_wat_score`, function and column alike)
    WEAT is reserved for the representation.  The behavioural metric is Bai's
    assignment score, EXACT and unsmoothed:
        gender : P(brilliance | male-assigned) + P(control | female-assigned) - 1
        fab    : P(brilliance | high-FAB)      + P(effort  | low-FAB)         - 1
    The +0.01 denominator smoothing used by `wat_gender_brilliance.d_gb` is not
    reproduced; denominators are checked instead.
    A response counts only if EVERY presented item is parsed exactly once and
    assigned to one of the TWO TARGETS THAT PROMPT PRESENTED.  A different male
    word, or a different brilliance pole, is not silently accepted just because
    it belongs to the same category.

PARSE FAILURE IS NOT DATA LOSS
    A malformed response sets `behaviour_wat_score` to NaN and is logged.  The
    prompt, the raw response and all 66 representation columns stay in the
    table: the forward pass was fine, and dropping the row would punch a hole in
    the exhaustive pair coverage the design exists to provide.

FAIL FAST ON EVERYTHING ELSE
    A non-finite hidden state, a non-finite pooled vector, or a stimulus whose
    span cannot be located unambiguously ABORTS THE RUN.  The old behaviour —
    skip the prompt and carry on — would silently destroy full-coverage
    balance, and a reduced-set score computed from a missing stimulus is not a
    score.

LAYER CONVENTION
    layer0  = hidden_states[0], the input embedding (before any block)
    layer1  = output of transformer block 1
    ...
    layer32 = output of transformer block 32
    Recorded in the manifest.  Hidden states come from the INPUT PROMPT ONLY;
    generated assistant tokens are never embedded or pooled.

MODEL LOADING  (unchanged, deliberately plain)
    AutoModelForCausalLM, bfloat16, ONE GPU, `device_map={"": gpu}`.
    No quantisation, no CPU offload, no disk offload, no `max_memory`.
    `.eval()`, and the hidden-state pass runs under `torch.inference_mode()`.
    If it does not fit, the run stops rather than quietly changing strategy.

USAGE
    python3 embedding/embed_bias_llama3_hidden.py --dry-run
    python3 embedding/embed_bias_llama3_hidden.py --smoke-test
    python3 embedding/embed_bias_llama3_hidden.py --out-dir responses/embed_llama31
"""

import os
import re
import sys
import csv
import json
import time
import hashlib
import argparse
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import embed_bias as eb              # cosine / WEAT utilities, csv writer
import wat_fab
import wat_gender_brilliance as wgb
# Span/pooling layer only.  The Scout BACKEND (quantisation, offload, MoE
# footprint) is deliberately NOT imported and must never be.
from embed_bias_scout_hidden import (
    SpanError, find_phrase_char_spans, token_span_for_chars, masking_spans,
    resolve_span, pool_span, build_chat_input,
)
from make_bai_style_prompts import (
    GENDER_CONDITIONS, N_TOTAL, STIMULI_CSV, PROMPTS_CSV,
    load_stimuli, read_prompts,
)

ANALYSIS_VERSION = "4.0-llama31-fixed-prompt-bai"
CHECKPOINT = "/home/syryu/models/llama3.1-8b-instruct"
OUT_DIR = os.path.join(ROOT, "responses", "embed_llama31")
POOLING = "mean"
SYSTEM_PROMPT = "You are a helpful assistant."

# The chosen low-nonzero sampling setting.  Recorded verbatim in the manifest.
GENERATION = {
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 512,
}
SEED_BASE = 0
SEED_RULE = ("per_prompt_seed = int(sha256(f'{seed_base}|{prompt_id}')"
             ".hexdigest()[:8], 16); torch.manual_seed(per_prompt_seed) "
             "immediately before generate(). Derived from the prompt id, not "
             "from execution order, so a resumed or reordered run reproduces "
             "the same responses.")

PRIMARY_CSV = "prompt_response_hidden.csv"
OUTPUT_FILES = [
    PRIMARY_CSV,
    "behaviour_parse_failures.csv",
    "secondary_set_level_weat.csv",
    "secondary_stimulus_layer.csv",
    "secondary_stimulus_prototypes.npz",
    "manifest.json",
]


class NonFiniteHiddenState(RuntimeError):
    """A forward pass returned NaN/Inf.  No statistic may be derived from it."""


# =========================================================
# Layer indexing
# =========================================================

def layer_label(hidden_state_index):
    """
    (hidden_state_index, transformer_layer, representation_type) for the long
    secondary files.

    Index 0 is the input/token embedding, NOT "transformer layer 0": it has not
    been through a block, and numbering it as one would invite averaging a
    lookup table into a sequence of residual-stream states.  The wide primary
    table calls it `layer0` and the manifest says what that means.
    """
    i = int(hidden_state_index)
    if i < 0:
        raise ValueError(f"hidden_state_index must be >= 0, got {i}")
    if i == 0:
        return dict(hidden_state_index=0, transformer_layer="",
                    representation_type="input_embedding")
    return dict(hidden_state_index=i, transformer_layer=i,
                representation_type="transformer_block")


def transformer_layers_only(rows):
    """The exploratory trajectory: block outputs, with the embedding dropped."""
    return [r for r in rows if r["representation_type"] == "transformer_block"]


def layer_columns(n_hidden_states):
    """The wide table's score columns, in their fixed order."""
    return ([f"weat_layer{i}" for i in range(n_hidden_states)]
            + [f"delta_layer{i}" for i in range(n_hidden_states)])


def primary_columns(n_hidden_states):
    return (["prompt_id", "domain", "category", "trial", "prompt", "response",
             "behaviour_wat_score"] + layer_columns(n_hidden_states))


# =========================================================
# Effect size — hidden-analysis specific, population SD
# =========================================================

def weat_effect_size_hidden(X, Y, A, B, ddof=0):
    """
    Caliskan's d with the CANONICAL POPULATION SD (ddof=0), for the secondary
    set-level file.

    Deliberately NOT `eb.weat_effect_size`, which uses the sample SD (ddof=1).
    Changing that function would silently move every historical external-
    embedding number in `responses/embed/`; this module keeps its own.
    """
    import numpy as np
    sx = eb.weat_association(X, A, B)
    sy = eb.weat_association(Y, A, B)
    sd = np.std(np.concatenate((sx, sy), axis=0), ddof=ddof)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float((np.mean(sx) - np.mean(sy)) / sd)


# =========================================================
# Roles — resolved from stimuli.csv, never from group0/group1
# =========================================================

def arm_for(domain, category, stim):
    """
    The full concept sets for one condition.  X/Y/A/B here define the sign, and
    they are read out of stimuli.csv by ROLE, so presentation order cannot
    reach them.
    """
    if domain == "gender":
        if category not in GENDER_CONDITIONS:
            raise SystemExit(f"unknown gender condition {category!r} "
                             f"(have {list(GENDER_CONDITIONS)})")
        g = stim["gender"]
        return dict(X=list(g["male"]), Y=list(g["female"]),
                    A=list(g["brilliance"]), B=list(g["control"][category]),
                    X_name="male targets", Y_name="female targets",
                    A_name="brilliance (storage4)",
                    B_name=f"control ({category})")
    if domain == "fab":
        f = stim["fab"]
        return dict(X=list(f["brilliance"]), Y=list(f["effort"]),
                    A=list(f["high_fab"]), B=list(f["low_fab"]),
                    X_name="brilliance poles (leslie)",
                    Y_name="effort poles (leslie)",
                    A_name="high-FAB fields (tertile)",
                    B_name="low-FAB fields (tertile)")
    raise SystemExit(f"unknown domain {domain!r}")


def prompt_roles(record, stim):
    """
    This prompt's X, Y, A, B.

    `group0`/`group1` say only which target was printed first.  Which one is X
    is decided by looking each up in the stimuli.csv role lists; a prompt whose
    two targets fall in the same pool, or in neither, is a corrupt design row
    and stops the run.
    """
    arm = arm_for(record["domain"], record["category"], stim)
    g0, g1 = record["group0"], record["group1"]
    in_X = [g for g in (g0, g1) if g in arm["X"]]
    in_Y = [g for g in (g0, g1) if g in arm["Y"]]
    if len(in_X) != 1 or len(in_Y) != 1:
        raise SystemExit(
            f"{record['prompt_id']}: targets ({g0!r}, {g1!r}) do not resolve to "
            f"exactly one {arm['X_name']} and one {arm['Y_name']}")

    attributes = json.loads(record["attributes"])
    A = [w for w in arm["A"] if w in attributes]
    B = [w for w in arm["B"] if w in attributes]
    if len(A) != len(arm["A"]) or len(B) != len(arm["B"]):
        raise SystemExit(
            f"{record['prompt_id']}: the prompt presents {len(A)}/{len(arm['A'])} "
            f"{arm['A_name']} and {len(B)}/{len(arm['B'])} {arm['B_name']}; "
            f"refusing to score a reduced set")
    if len(attributes) != len(A) + len(B):
        extra = [w for w in attributes if w not in A and w not in B]
        raise SystemExit(f"{record['prompt_id']}: unrecognised presented "
                         f"item(s) {extra}")

    return dict(X=in_X[0], Y=in_Y[0], A=A, B=B,
                attributes=list(attributes), arm=arm,
                phrases=[in_X[0], in_Y[0]] + list(attributes))


# =========================================================
# Behavioural WAT score  (NOT called WEAT — that is the representation metric)
# =========================================================

def _norm(x):
    """Same normalisation the behavioural parsers apply to both sides."""
    x = str(x).strip().strip("'\"").strip("*").strip().lower()
    return re.sub(r"\s+", " ", x)


def parse_behaviour(raw, record, roles):
    """
    ({item: target}, problems).  Strict: exact matches only, every presented
    item exactly once, and only the two targets THIS prompt presented.

    The line parsers are the behavioural runners' own — `wgb.parse_wat` splits
    on the LAST hyphen so `super-imaginative` survives, `wat_fab.parse_wat`
    splits on the first, which is right for field names that contain none.
    Matching after that is deliberately stricter than the runners': their
    substring fallbacks exist to rescue n from a paid API call, and here they
    would let a response that named a non-presented target still score.
    """
    parser = wgb.parse_wat if record["domain"] == "gender" else wat_fab.parse_wat
    pairs = parser(raw or "")

    items = {_norm(w): w for w in roles["attributes"]}
    targets = {_norm(t): t for t in (roles["X"], roles["Y"])}
    assign, problems = {}, []
    for left, right in pairs:
        item = items.get(_norm(left))
        if item is None:
            problems.append(f"unrecognised item {left!r}")
            continue
        if item in assign:
            problems.append(f"item {item!r} assigned more than once")
            continue
        target = targets.get(_norm(right))
        if target is None:
            problems.append(f"{item!r} assigned to {right!r}, which is not one "
                            f"of the two presented targets")
            continue
        assign[item] = target

    unassigned = [w for w in roles["attributes"] if w not in assign]
    if unassigned:
        problems.append(f"{len(unassigned)} item(s) never assigned: "
                        f"{', '.join(unassigned)}")
    return assign, problems


def behaviour_wat_score(assign, record, roles):
    """
    Bai's assignment score, EXACT — no +0.01 smoothing.  Positive is always
    stereotype-congruent, because the counts are keyed on ROLE (X/Y, A/B), not
    on which target was printed first.

        gender : P(brilliance | male-assigned) + P(control | female-assigned) - 1
        fab    : P(brilliance | high-FAB)      + P(effort  | low-FAB)         - 1

    Returns (score, problem).  An empty denominator — every item sent to one
    side — yields (nan, reason) rather than a smoothed number.
    """
    X, Y = roles["X"], roles["Y"]
    A = set(roles["A"])
    if record["domain"] == "gender":
        # left of the + : items given to the male target
        aX = sum(1 for i, t in assign.items() if t == X and i in A)
        bX = sum(1 for i, t in assign.items() if t == X and i not in A)
        aY = sum(1 for i, t in assign.items() if t == Y and i in A)
        bY = sum(1 for i, t in assign.items() if t == Y and i not in A)
        if (aX + bX) == 0:
            return float("nan"), f"no item assigned to {X!r}"
        if (aY + bY) == 0:
            return float("nan"), f"no item assigned to {Y!r}"
        return aX / (aX + bX) + bY / (aY + bY) - 1, None

    B = set(roles["B"])
    # left of the + : high-FAB fields; right : low-FAB fields
    hX = sum(1 for i, t in assign.items() if i in A and t == X)
    hY = sum(1 for i, t in assign.items() if i in A and t == Y)
    lX = sum(1 for i, t in assign.items() if i in B and t == X)
    lY = sum(1 for i, t in assign.items() if i in B and t == Y)
    if (hX + hY) == 0:
        return float("nan"), "no high-FAB field assigned to either pole"
    if (lX + lY) == 0:
        return float("nan"), "no low-FAB field assigned to either pole"
    return hX / (hX + hY) + lY / (lX + lY) - 1, None


# =========================================================
# Model
# =========================================================

def load_llama3(checkpoint=CHECKPOINT, gpu=0, dtype="bfloat16"):
    """
    Plain single-GPU bf16 load.  No quantisation, no offload, no `max_memory`,
    no `device_map="auto"`.

    If the checkpoint does not fit on one card the exception propagates and the
    run ends.  That is the point: the Scout run spent ten minutes per forward
    pass streaming weights off a spinning disk, and then produced NaN, because a
    loader was allowed to "cope".  This one is not.
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    if not os.path.isdir(checkpoint):
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    shards = sorted(f for f in os.listdir(checkpoint) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors in {checkpoint} — this runner reads the "
                         f"HF repository root, not original/consolidated.*.pth")

    torch_dtype = getattr(torch, dtype)
    config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch_dtype, device_map={"": gpu}, low_cpu_mem_usage=True)
    load_seconds = time.time() - t0
    model.eval()

    placement = defaultdict(int)
    for _, p in model.named_parameters():
        placement[str(p.device)] += 1
    stray = [d for d in placement if d.startswith(("cpu", "meta", "disk"))]
    if stray:
        raise SystemExit(f"parameters landed on {stray} — this runner forbids "
                         f"CPU/disk offload; refusing to continue")

    meta = dict(
        checkpoint=checkpoint,
        safetensors_shards=shards,
        ignored_duplicates=sorted(
            os.path.join("original", f)
            for f in os.listdir(os.path.join(checkpoint, "original"))
            if f.endswith((".pth", ".pt"))) if os.path.isdir(
                os.path.join(checkpoint, "original")) else [],
        model_class=model.__class__.__name__,
        architectures=list(getattr(config, "architectures", []) or []),
        dtype=str(model.dtype),
        tokenizer_class=tokenizer.__class__.__name__,
        n_transformer_blocks=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        device=f"cuda:{gpu}",
        device_placement=dict(placement),
        hf_device_map=getattr(model, "hf_device_map", None),
        quantisation="none",
        cpu_offload=False,
        disk_offload=False,
        load_seconds=round(load_seconds, 2),
    )
    return model, tokenizer, meta


def prompt_seed(prompt_id, seed_base=SEED_BASE):
    """Deterministic per-row generation seed; see SEED_RULE."""
    digest = hashlib.sha256(f"{seed_base}|{prompt_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_response(templated, tokenizer, model, *, seed, generation=None):
    """
    The model's own WAT answer to the SAME templated chat input the hidden-state
    pass will see.  Only the newly generated tokens are decoded.

    Generation is for behaviour only.  Its tokens are never embedded: the
    representation of a stimulus is its contextual vector in the INPUT, and
    pooling the model's own continuation into it would mix what the model wrote
    into what the model read.
    """
    import torch
    cfg = dict(GENERATION if generation is None else generation)
    enc = tokenizer(templated, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, pad_token_id=pad, **cfg)
    seconds = time.time() - t0

    new_tokens = out[0][enc["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, dict(seed=seed, n_new_tokens=int(new_tokens.shape[0]),
                      generate_seconds=seconds, **cfg)


def assert_finite_hidden_states(hidden_states, context=""):
    """
    `torch.isfinite(h).all()` on EVERY returned hidden state, before anything is
    pooled.  Raises with the first offending index and its exact NaN/Inf counts.

    Scout went non-finite at hidden state 7 and the residual stream collapsed to
    absmax 0 from there; with no check, that run would have written a CSV full of
    numbers that looked like results.
    """
    import torch
    for i, h in enumerate(hidden_states):
        if torch.isfinite(h).all():
            continue
        n_nan = int(torch.isnan(h).sum().item())
        n_inf = int(torch.isinf(h).sum().item())
        raise NonFiniteHiddenState(
            f"non-finite hidden state at index {i} "
            f"({layer_label(i)['representation_type']}): "
            f"{n_nan} NaN, {n_inf} Inf of {h.numel()} values"
            + (f" [{context}]" if context else "")
            + " — aborting the run; no score may be derived from it")


def extract_prompt_vectors(templated, phrases, tokenizer, model, *,
                           user_window, competitors, pooling=POOLING, context=""):
    """
    ONE forward pass over the templated prompt; returns
    ({phrase: [vector per hidden state]}, meta).

    No generation here, ever.  `torch.inference_mode()`,
    `output_hidden_states=True`, nothing else.  Span lookup is confined to
    `user_window` so template boilerplate and the system message cannot supply a
    match, and `competitors` (the other presented stimuli) mask occurrences that
    are really inside a longer stimulus — `man` inside `woman`, `male` inside
    `female`, `history` inside `art history`.
    """
    import torch
    import numpy as np

    enc = tokenizer(templated, return_tensors="pt", return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    t0 = time.time()
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
    forward_seconds = time.time() - t0

    assert_finite_hidden_states(out.hidden_states, context)
    hidden = [h[0].to(dtype=torch.float32).cpu().numpy() for h in out.hidden_states]

    spans, vectors = {}, {}
    for phrase in phrases:
        span = resolve_span(templated, offsets, phrase, window=user_window,
                            competitors=[c for c in competitors if c != phrase])
        spans[phrase] = span
        pooled = [pool_span(layer, span, pooling) for layer in hidden]
        for i, v in enumerate(pooled):
            if not np.isfinite(v).all():
                raise NonFiniteHiddenState(
                    f"pooled vector for {phrase!r} is non-finite at hidden state "
                    f"{i} ({layer_label(i)['representation_type']})"
                    + (f" [{context}]" if context else "")
                    + " — aborting the run")
        vectors[phrase] = pooled

    meta = dict(
        n_hidden_states=len(hidden),
        n_transformer_blocks=len(hidden) - 1,
        hidden_size=int(hidden[0].shape[-1]),
        n_prompt_tokens=int(hidden[0].shape[0]),
        pooling=pooling,
        forward_seconds=forward_seconds,
        spans={p: list(s) for p, s in spans.items()},
        subtokens={p: tokenizer.convert_ids_to_tokens(
            enc["input_ids"][0][s[0]:s[1]].tolist()) for p, s in spans.items()},
        finite_check="pass",
    )
    return vectors, meta


# =========================================================
# Representation scores — two per layer
# =========================================================

def representation_scores(vectors, roles, n_hidden_states):
    """
    (deltas, weats), one value per hidden state.

        s(w,A,B) = mean cos(w,A) - mean cos(w,B)
        delta    = s(X,A,B) - s(Y,A,B)
        weat     = [mean s(X) - mean s(Y)] / [pop_SD(s(X) u s(Y)) + 0.001]

    With one X and one Y, `eb.bai_singleton_weat` IS that standardized formula
    (population SD, eps = 0.001) and is already tested, so it is reused rather
    than re-derived.  The output column is `weat_layer*`; the Bai-specific name
    stays inside this module.
    """
    import numpy as np
    A_words, B_words = list(roles["A"]), list(roles["B"])
    deltas, weats = [], []
    for i in range(n_hidden_states):
        xv = np.asarray(vectors[roles["X"]][i], dtype=float)[None, :]
        yv = np.asarray(vectors[roles["Y"]][i], dtype=float)[None, :]
        A = np.asarray([vectors[w][i] for w in A_words], dtype=float)
        B = np.asarray([vectors[w][i] for w in B_words], dtype=float)
        s_X = float(eb.weat_association(xv, A, B)[0])
        s_Y = float(eb.weat_association(yv, A, B)[0])
        deltas.append(s_X - s_Y)
        weats.append(float(eb.bai_singleton_weat(xv, yv, A, B)))
    return deltas, weats


def wide_row(record, response, score, deltas, weats):
    """
    One row of the primary table, in the fixed column order.

    `group0`, `group1` and `attributes` are NOT repeated here — they live in
    prompts.csv and join on `prompt_id`.  Keeping them out is what stops a
    presentation-order column from quietly becoming an analysis variable.
    """
    if len(deltas) != len(weats):
        raise ValueError(f"{len(deltas)} deltas vs {len(weats)} weats")
    row = {
        "prompt_id": record["prompt_id"],
        "domain": record["domain"],
        "category": record["category"],
        "trial": int(record["trial"]),
        "prompt": record["prompt"],
        "response": response,
        "behaviour_wat_score": score,
    }
    for i, v in enumerate(weats):
        row[f"weat_layer{i}"] = v
    for i, v in enumerate(deltas):
        row[f"delta_layer{i}"] = v
    return row


def write_primary_csv(path, rows, n_hidden_states, force=False):
    """Explicit fieldnames: the column order is part of the specification."""
    columns = primary_columns(n_hidden_states)
    if os.path.exists(path) and not force:
        raise SystemExit(f"refusing to overwrite {path} — rerun with --force "
                         f"(nothing written this run)")
    for r in rows:
        extra = set(r) - set(columns)
        if extra:
            raise ValueError(f"{r.get('prompt_id')}: unexpected column(s) "
                             f"{sorted(extra)} in the primary table")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows x {len(columns)} columns)")
    return columns


# =========================================================
# SECONDARY — contextual prototypes and set-level WEAT
# =========================================================

class PrototypeAccumulator:
    """
    Running mean of each stimulus's contextual vector, per (domain, category,
    stimulus, hidden state).

    A running sum rather than a stored list: 272 prompts x 33 hidden states x
    4096 dims is small, but the accumulator predates this design and there is no
    reason to make it worse.
    """

    def __init__(self):
        self.sums = {}
        self.counts = defaultdict(int)
        self.contexts = defaultdict(list)

    def add(self, dataset, arm, stimulus, vectors, context_id=None):
        """
        `vectors` is ONE stimulus's per-hidden-state vectors — the value of
        `extract_prompt_vectors()[0][stimulus]`, not the whole mapping.  The
        shape is checked because passing the mapping instead is an easy slip and
        numpy's own error ("float() argument must be ... not 'dict'") says
        nothing about which call site was wrong.
        """
        import numpy as np
        key = (dataset, arm, stimulus)
        if isinstance(vectors, dict):
            raise TypeError(
                f"PrototypeAccumulator.add({stimulus!r}) was given the whole "
                f"{{phrase: vectors}} mapping; pass vectors[{stimulus!r}]")
        stack = np.asarray(vectors, dtype=np.float64)     # (n_hidden_states, dim)
        if stack.ndim != 2:
            raise ValueError(
                f"expected one (n_hidden_states, dim) block for {stimulus!r}, "
                f"got shape {stack.shape}")
        if key in self.sums:
            if self.sums[key].shape != stack.shape:
                raise ValueError(f"shape change for {key}: "
                                 f"{self.sums[key].shape} vs {stack.shape}")
            self.sums[key] += stack
        else:
            self.sums[key] = stack.copy()
        self.counts[key] += 1
        if context_id is not None:
            self.contexts[key].append(context_id)

    def prototype(self, dataset, arm, stimulus):
        key = (dataset, arm, stimulus)
        if key not in self.sums:
            raise KeyError(f"no contextual observation of {stimulus!r} in "
                           f"{dataset}/{arm}")
        return self.sums[key] / self.counts[key]

    def n(self, dataset, arm, stimulus):
        return self.counts[(dataset, arm, stimulus)]

    def stimuli(self, dataset, arm):
        return sorted(s for (d, a, s) in self.sums if d == dataset and a == arm)


def accumulate_prompt(acc, dataset, arm_name, vectors, phrases, prompt_id):
    """
    Feed one prompt's contextual vectors into the prototype accumulator.

    A named seam rather than a loop inside `main`, so the wiring — one stimulus's
    vectors per call, not the mapping — is covered by a test instead of by the
    first prompt of a full run.
    """
    for w in phrases:
        acc.add(dataset, arm_name, w, vectors[w], context_id=prompt_id)
    return acc


def set_level_rows(acc, dataset, arm_name, arm, *, n_hidden_states):
    """
    SECONDARY.  The full-set WEAT over contextual prototypes, at every hidden
    state, within this 272-prompt experiment only.

    X/Y/A/B and therefore the sign come from `arm_for()`, i.e. from stimuli.csv,
    so stereotype-congruent stays positive.  Effect size uses the population SD
    (ddof=0) — `weat_effect_size_hidden`, not `eb.weat_effect_size` — so the
    external-embedding numbers never move.
    """
    import numpy as np
    roles = [("X", arm["X"]), ("Y", arm["Y"]), ("A", arm["A"]), ("B", arm["B"])]
    present, missing = {}, defaultdict(list)
    for role, words in roles:
        for w in words:
            if acc.n(dataset, arm_name, w):
                present.setdefault(role, []).append(w)
            else:
                missing[role].append(w)
    if any(len(present.get(r, [])) == 0 for r, _ in roles):
        return [], dict(missing)

    rows = []
    for i in range(n_hidden_states):
        mats = {r: np.asarray([acc.prototype(dataset, arm_name, w)[i]
                               for w in present[r]], dtype=float)
                for r, _ in roles}
        sx = eb.weat_association(mats["X"], mats["A"], mats["B"])
        sy = eb.weat_association(mats["Y"], mats["A"], mats["B"])
        row = dict(domain=dataset, category=arm_name,
                   X_name=arm["X_name"], Y_name=arm["Y_name"],
                   A_name=arm["A_name"], B_name=arm["B_name"])
        row.update(layer_label(i))
        row.update(
            weat_S=eb.weat_test_stat(mats["X"], mats["Y"], mats["A"], mats["B"]),
            weat_d=weat_effect_size_hidden(mats["X"], mats["Y"],
                                           mats["A"], mats["B"], ddof=0),
            effect_size_sd="population (ddof=0)",
            mean_s_X=float(np.mean(sx)), mean_s_Y=float(np.mean(sy)),
            n_X=len(present["X"]), n_Y=len(present["Y"]),
            n_A=len(present["A"]), n_B=len(present["B"]),
            target_s_values=json.dumps(
                {w: float(v) for w, v in
                 list(zip(present["X"], sx)) + list(zip(present["Y"], sy))}),
            n_contexts_per_target=json.dumps(
                {w: acc.n(dataset, arm_name, w)
                 for w in present["X"] + present["Y"]}),
            n_contexts_per_attribute=json.dumps(
                {w: acc.n(dataset, arm_name, w)
                 for w in present["A"] + present["B"]}),
            missing_stimuli=json.dumps({k: v for k, v in missing.items() if v}),
            prototype="mean over the prompts of THIS experiment in which the "
                      "stimulus appeared",
            pooling=POOLING,
            analysis_role="secondary", exploratory_across_layers=True,
            analysis_version=ANALYSIS_VERSION)
        rows.append(row)
    return rows, dict(missing)


def stimulus_layer_rows(acc, dataset, arm_name, arm, *, n_hidden_states):
    """
    SECONDARY.  stimulus x hidden state: enough metadata to audit how each
    prototype was built, and where it sits on the A<->B axis.  The 4096-D
    prototypes themselves go to the npz; a CSV is not the place for them.
    """
    import numpy as np
    role_of = {}
    for role, words in (("X", arm["X"]), ("Y", arm["Y"]),
                        ("A", arm["A"]), ("B", arm["B"])):
        for w in words:
            role_of[w] = role
    A_words = [w for w in arm["A"] if acc.n(dataset, arm_name, w)]
    B_words = [w for w in arm["B"] if acc.n(dataset, arm_name, w)]
    rows = []
    for w in acc.stimuli(dataset, arm_name):
        proto = acc.prototype(dataset, arm_name, w)
        for i in range(n_hidden_states):
            v = proto[i]
            A = np.asarray([acc.prototype(dataset, arm_name, a)[i]
                            for a in A_words], dtype=float)
            B = np.asarray([acc.prototype(dataset, arm_name, b)[i]
                            for b in B_words], dtype=float)
            cA = float(np.mean(eb.cos_sim(v[None, :], A)))
            cB = float(np.mean(eb.cos_sim(v[None, :], B)))
            row = dict(domain=dataset, category=arm_name, stimulus=w,
                       role=role_of.get(w, ""))
            row.update(layer_label(i))
            row.update(n_contexts=acc.n(dataset, arm_name, w),
                       prototype_l2_norm=float(np.linalg.norm(v)),
                       mean_cos_to_A=cA, mean_cos_to_B=cB,
                       s_w_A_B=cA - cB,
                       prototype_dim=int(v.shape[-1]),
                       prototype_file="secondary_stimulus_prototypes.npz",
                       pooling=POOLING, analysis_role="secondary",
                       analysis_version=ANALYSIS_VERSION)
            rows.append(row)
    return rows


# =========================================================
# Manifest
# =========================================================

def build_manifest(*, backend, arch, design, n_hidden_states, parse_failures,
                   generation, missing, args):
    return {
        "analysis": "Fixed-prompt Bai-style WAT: paired Llama-3.1 behaviour and "
                    "contextualised hidden states from the same prompt",
        "analysis_version": ANALYSIS_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(),
        "design": design,
        "checkpoint": backend["checkpoint"],
        "safetensors_shards": backend["safetensors_shards"],
        "ignored_duplicate_weights": backend["ignored_duplicates"],
        "model_class": backend["model_class"],
        "architectures": backend["architectures"],
        "dtype": backend["dtype"],
        "tokenizer": backend["tokenizer_class"],
        "transformers_version": arch["transformers_version"],
        "torch_version": arch["torch_version"],
        "device": backend["device"],
        "quantisation": backend["quantisation"],
        "cpu_offload": backend["cpu_offload"],
        "disk_offload": backend["disk_offload"],
        "hidden_size": backend["hidden_size"],
        "n_transformer_blocks": backend["n_transformer_blocks"],
        "n_hidden_states": n_hidden_states,
        "system_prompt": SYSTEM_PROMPT,
        "chat_template_reconstruction":
            "tokenizer.apply_chat_template([{system}, {user}], tokenize=False, "
            "add_generation_prompt=True); the prompt must appear verbatim "
            "exactly once or the run stops. The SAME templated string is used "
            "for generate() and for the hidden-state forward pass",
        "generation": generation,
        "generation_seed_rule": SEED_RULE,
        "generation_seed_base": args.seed if args is not None else SEED_BASE,
        "hidden_states_from": "the input prompt only; generated assistant "
                              "tokens are never embedded or pooled",
        "pooling": POOLING,
        "pooling_rule": "mean over every sub-token of the stimulus phrase "
                        "(never Bai's first token only)",
        "layer_convention": {
            "layer0": "hidden_states[0] — the input embedding, before any "
                      "transformer block",
            "layer1..layerN": "output of transformer block 1..N",
            "n_transformer_blocks": backend["n_transformer_blocks"],
        },
        "orientation": {
            "gender": "X = male target, Y = female target, A = brilliance, "
                      "B = condition control",
            "fab": "X = brilliance pole, Y = effort pole, A = high-FAB fields, "
                   "B = low-FAB fields",
            "resolved_from": "stimuli.csv roles, never group0/group1",
            "meaning": "positive behaviour_wat_score / delta_layer* / weat_layer* "
                       "= stereotype-congruent",
        },
        "representation_metrics": {
            "s": "s(w,A,B) = mean cos(w,A) - mean cos(w,B)",
            "delta_layer*": "s(X,A,B) - s(Y,A,B), unstandardised",
            "weat_layer*": "[mean s(X) - mean s(Y)] / "
                           "[population SD(s(X) u s(Y)) + 0.001], ddof=0",
            "why_both": "the singleton standardized score saturates near +-2 "
                        "for an association difference of any magnitude; delta "
                        "preserves the magnitude",
            "not_written": ["s_group0_layer*", "s_group1_layer*"],
        },
        "behavioural_metric": {
            "name": "behaviour_wat_score",
            "column": "behaviour_wat_score",
            "note": "WEAT is reserved for the representation metric; this is "
                    "Bai's assignment score and is never called behaviour WEAT",
            "gender": "P(brilliance | male-assigned) + "
                      "P(control | female-assigned) - 1",
            "fab": "P(brilliance | high-FAB) + P(effort | low-FAB) - 1",
            "smoothing": "none (the +0.01 of wat_gender_brilliance.d_gb is NOT "
                         "applied; denominators are checked instead)",
            "validity": "every presented item parsed exactly once and assigned "
                        "to one of the two targets THIS prompt presented",
            "on_failure": "behaviour_wat_score = NaN; prompt, raw response and all "
                          "representation columns are kept",
            "n_parse_failures": len(parse_failures),
            "parse_failure_ids": [p["prompt_id"] for p in parse_failures],
        },
        "fail_fast": {
            "non_finite_hidden_state": "aborts the run",
            "non_finite_pooled_vector": "aborts the run",
            "unresolvable_or_ambiguous_span": "aborts the run",
            "rationale": "skipping a prompt would destroy the exhaustive "
                         "stimulus-pair coverage the design is built on",
        },
        "secondary_outputs": {
            "set_level_weat": "contextual prototypes over THIS experiment's "
                              "prompts only; never pooled with the legacy "
                              "3420-prompt replay corpus",
            "effect_size_sd": "population SD, ddof=0 (weat_effect_size_hidden); "
                              "embed_bias.weat_effect_size keeps ddof=1 and is "
                              "untouched",
            "missing_stimuli": missing,
        },
        "not_run": {
            "legacy_response_replay": "this runner reads no saved behavioural "
                                      "response file and "
                                   "no behavioural-model provenance",
            "pca_nullification": False,
            "permutation_inference": False,
            "profile_or_decision_chain": "not part of this experiment; the WAT "
                                         "prompt is the only prompt",
        },
        "exploratory": "the full layer profile is exploratory; no layer is "
                       "selected, and none may be selected after seeing the data",
        "cli": vars(args) if args is not None else {},
    }


# =========================================================
# Driver
# =========================================================

def file_digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def design_summary(prompts, stimuli_path, prompts_path):
    by_cat = defaultdict(int)
    for r in prompts:
        by_cat[f"{r['domain']}/{r['category']}"] += 1
    return {
        "stimuli_csv": os.path.relpath(stimuli_path, ROOT),
        "stimuli_sha256": file_digest(stimuli_path),
        "prompts_csv": os.path.relpath(prompts_path, ROOT),
        "prompts_sha256": file_digest(prompts_path),
        "n_prompts": len(prompts),
        "n_by_condition": dict(sorted(by_cat.items())),
        "generator": "embedding/make_bai_style_prompts.py",
        "frozen": "the runner randomises nothing; target pairing, target order "
                  "and attribute order are read back from prompts.csv",
    }


def load_design(stimuli_path, prompts_path, *, expect_total=N_TOTAL):
    stim = load_stimuli(stimuli_path)
    prompts = read_prompts(prompts_path)
    if expect_total is not None and len(prompts) != expect_total:
        raise SystemExit(f"{prompts_path}: {len(prompts)} prompts, expected "
                         f"{expect_total}; regenerate with "
                         f"make_bai_style_prompts.py")
    ids = [r["prompt_id"] for r in prompts]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{prompts_path}: duplicate prompt_id")
    return stim, prompts


def run_prompts(prompts, stim, model, tokenizer, *, args, acc=None, verbose=True):
    """
    The whole experiment: one generate() and one forward() per frozen prompt.

    Returns (rows, parse_failures, meta).  Anything that is not a behavioural
    parse failure raises out of here, by design.
    """
    rows, failures = [], []
    n_hidden = None
    t_start = time.time()
    for k, record in enumerate(prompts, start=1):
        roles = prompt_roles(record, stim)
        templated, window = build_chat_input(tokenizer, SYSTEM_PROMPT,
                                             record["prompt"])
        context = record["prompt_id"]

        response, gen_meta = generate_response(
            templated, tokenizer, model,
            seed=prompt_seed(record["prompt_id"], args.seed),
            generation=dict(GENERATION, max_new_tokens=args.max_new_tokens))

        vectors, meta = extract_prompt_vectors(
            templated, roles["phrases"], tokenizer, model,
            user_window=window, competitors=roles["phrases"], context=context)
        n_hidden = meta["n_hidden_states"]

        assign, problems = parse_behaviour(response, record, roles)
        if problems:
            score = float("nan")
            failures.append(dict(prompt_id=record["prompt_id"],
                                 domain=record["domain"],
                                 category=record["category"],
                                 trial=record["trial"],
                                 n_items_assigned=len(assign),
                                 n_items_presented=len(roles["attributes"]),
                                 problems=" | ".join(problems),
                                 response=response))
        else:
            score, denom_problem = behaviour_wat_score(assign, record, roles)
            if denom_problem:
                failures.append(dict(prompt_id=record["prompt_id"],
                                     domain=record["domain"],
                                     category=record["category"],
                                     trial=record["trial"],
                                     n_items_assigned=len(assign),
                                     n_items_presented=len(roles["attributes"]),
                                     problems=denom_problem,
                                     response=response))

        deltas, weats = representation_scores(vectors, roles, n_hidden)
        rows.append(wide_row(record, response, score, deltas, weats))

        if acc is not None:
            accumulate_prompt(acc, record["domain"], record["category"],
                              vectors, roles["phrases"], record["prompt_id"])
        if verbose:
            elapsed = time.time() - t_start
            print(f"  [{k:3d}/{len(prompts)}] {record['prompt_id']:<22} "
                  f"bias={score:+.3f}  delta_L{n_hidden-1}={deltas[-1]:+.5f}  "
                  f"gen={gen_meta['generate_seconds']:.1f}s "
                  f"fwd={meta['forward_seconds']:.2f}s  "
                  f"({elapsed/k:.1f}s/prompt)", flush=True)

    return rows, failures, dict(n_hidden_states=n_hidden,
                                total_seconds=time.time() - t_start)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stimuli", default=STIMULI_CSV)
    ap.add_argument("--prompts", default=PROMPTS_CSV)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED_BASE,
                    help="base generation seed; per-prompt seeds derive from it")
    ap.add_argument("--max-new-tokens", type=int,
                    default=GENERATION["max_new_tokens"])
    ap.add_argument("--domain", choices=("gender", "fab", "both"), default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of prompts (diagnostics only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the frozen design; load no model")
    ap.add_argument("--smoke-test", action="store_true",
                    help="one gender + one FAB prompt, full diagnostics, "
                         "writes nothing")
    args = ap.parse_args()

    partial = bool(args.limit) or args.domain != "both" or args.smoke_test
    stim, prompts = load_design(args.stimuli, args.prompts,
                                expect_total=None if partial else N_TOTAL)
    design = design_summary(prompts, args.stimuli, args.prompts)

    print(f"Fixed-prompt Bai-style WAT  (analysis {ANALYSIS_VERSION})")
    print(f"  stimuli    : {args.stimuli}")
    print(f"  prompts    : {args.prompts}  ({len(prompts)} frozen prompts)")
    for cond, n in design["n_by_condition"].items():
        print(f"    {cond:<20} {n}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  generation : {dict(GENERATION, max_new_tokens=args.max_new_tokens)}")
    print(f"  seed base  : {args.seed}")

    if args.domain != "both":
        prompts = [r for r in prompts if r["domain"] == args.domain]
    if args.smoke_test:
        gender = next((r for r in prompts if r["domain"] == "gender"), None)
        fab = next((r for r in prompts if r["domain"] == "fab"), None)
        prompts = [r for r in (gender, fab) if r is not None]
    elif args.limit:
        prompts = prompts[:args.limit]

    # role resolution needs no GPU — do it for every prompt before loading 16 GiB
    for record in prompts:
        prompt_roles(record, stim)
    print(f"  roles resolved for all {len(prompts)} prompts "
          f"(X/Y/A/B from stimuli.csv, not group0/group1)")

    if args.dry_run:
        print("\n--dry-run: design validated, no model loaded, nothing written.")
        return

    import torch, transformers
    model, tok, backend = load_llama3(args.checkpoint, args.gpu)
    print(f"\n  loaded {backend['model_class']} in {backend['load_seconds']}s "
          f"on {backend['device']}  ({backend['dtype']}, {backend['quantisation']})")

    if args.smoke_test:
        return smoke_test(args, stim, prompts, model, tok, backend)

    acc = PrototypeAccumulator()
    rows, failures, meta = run_prompts(prompts, stim, model, tok,
                                       args=args, acc=acc)
    n_hidden = meta["n_hidden_states"]
    print(f"\n  {len(rows)} prompts in {meta['total_seconds']/60:.1f} min; "
          f"{len(failures)} behavioural parse failure(s)")

    set_rows, stim_rows, missing = [], [], {}
    for domain, category in sorted({(r["domain"], r["category"]) for r in prompts}):
        arm = arm_for(domain, category, stim)
        if not acc.stimuli(domain, category):
            continue
        srows, miss = set_level_rows(acc, domain, category, arm,
                                     n_hidden_states=n_hidden)
        set_rows.extend(srows)
        if any(miss.values()):
            missing[f"{domain}/{category}"] = {k: v for k, v in miss.items() if v}
        stim_rows.extend(stimulus_layer_rows(acc, domain, category, arm,
                                             n_hidden_states=n_hidden))

    os.makedirs(args.out_dir, exist_ok=True)
    write_primary_csv(os.path.join(args.out_dir, PRIMARY_CSV), rows, n_hidden,
                      force=args.force)
    eb.write_csv(os.path.join(args.out_dir, "behaviour_parse_failures.csv"),
                 failures, args.force)
    eb.write_csv(os.path.join(args.out_dir, "secondary_set_level_weat.csv"),
                 set_rows, args.force)
    eb.write_csv(os.path.join(args.out_dir, "secondary_stimulus_layer.csv"),
                 stim_rows, args.force)

    import numpy as np
    np.savez_compressed(
        os.path.join(args.out_dir, "secondary_stimulus_prototypes.npz"),
        **{f"{d}|{a}|{s}": acc.prototype(d, a, s) for (d, a, s) in acc.sums})

    manifest = build_manifest(
        backend=backend,
        arch=dict(transformers_version=transformers.__version__,
                  torch_version=torch.__version__),
        design=design, n_hidden_states=n_hidden, parse_failures=failures,
        generation=dict(GENERATION, max_new_tokens=args.max_new_tokens),
        missing=missing, args=args)
    with open(os.path.join(args.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  wrote {os.path.join(args.out_dir, 'manifest.json')}")


def smoke_test(args, stim, prompts, model, tokenizer, backend):
    """One gender + one FAB prompt, every diagnostic, nothing written."""
    import torch

    def rule(t):
        print(f"\n{'='*74}\n{t}\n{'='*74}", flush=True)

    GIB = float(1 << 30)
    free, total = torch.cuda.mem_get_info(args.gpu)
    print(f"  GPU {args.gpu} after load : {(total-free)/GIB:.2f} GiB used, "
          f"{free/GIB:.2f} GiB free")

    for record in prompts:
        roles = prompt_roles(record, stim)
        rule(f"SMOKE — {record['prompt_id']}  ({record['domain']}/"
             f"{record['category']}, trial {record['trial']})")
        print(f"  presented order      : group0={record['group0']!r}  "
              f"group1={record['group1']!r}")
        print(f"  roles from stimuli   : X={roles['X']!r} ({roles['arm']['X_name']})"
              f"  Y={roles['Y']!r} ({roles['arm']['Y_name']})")
        print(f"  A ({roles['arm']['A_name']}): {roles['A']}")
        print(f"  B ({roles['arm']['B_name']}): {roles['B']}")
        print(f"  presented order of items: {roles['attributes']}")
        print(f"\n  prompt:\n    {record['prompt']}")

        templated, window = build_chat_input(tokenizer, SYSTEM_PROMPT,
                                             record["prompt"])
        n_tok = len(tokenizer(templated)["input_ids"])
        print(f"\n  system prompt        : {SYSTEM_PROMPT!r}")
        print(f"  chat-template tokens : {n_tok}")

        seed = prompt_seed(record["prompt_id"], args.seed)
        response, gen_meta = generate_response(
            templated, tokenizer, model, seed=seed,
            generation=dict(GENERATION, max_new_tokens=args.max_new_tokens))
        print(f"\n  generation           : seed={seed} "
              f"temp={gen_meta['temperature']} do_sample={gen_meta['do_sample']} "
              f"new_tokens={gen_meta['n_new_tokens']} "
              f"({gen_meta['generate_seconds']:.1f}s)")
        print("  response:")
        for line in response.splitlines():
            print(f"    {line}")

        vectors, meta = extract_prompt_vectors(
            templated, roles["phrases"], tokenizer, model, user_window=window,
            competitors=roles["phrases"], context=record["prompt_id"])
        print(f"\n  hidden states        : {meta['n_hidden_states']} "
              f"(= 1 input embedding + {meta['n_transformer_blocks']} blocks), "
              f"dim {meta['hidden_size']}")
        print(f"  prompt tokens        : {meta['n_prompt_tokens']}   "
              f"forward {meta['forward_seconds']:.2f}s   "
              f"finite check {meta['finite_check']}")
        print(f"  pooling              : {meta['pooling']} over every sub-token")
        for phrase in roles["phrases"]:
            lo, hi = meta["spans"][phrase]
            print(f"    {phrase:<26} tokens [{lo}, {hi})  "
                  f"{meta['subtokens'][phrase]}")

        assign, problems = parse_behaviour(response, record, roles)
        if problems:
            score, denom = float("nan"), None
            print(f"\n  behavioural parse    : FAILED — {' | '.join(problems)}")
            print("  (representation columns are kept; behaviour_wat_score = NaN)")
        else:
            score, denom = behaviour_wat_score(assign, record, roles)
            print(f"\n  behavioural parse    : all "
                  f"{len(roles['attributes'])} items assigned")
            if denom is None:
                print(f"  behaviour_wat_score  : {score:+.4f}  "
                      f"(positive = stereotype-congruent)")
            else:
                print(f"  behaviour_wat_score  : NaN — {denom}")

        deltas, weats = representation_scores(vectors, roles,
                                              meta["n_hidden_states"])
        last = meta["n_hidden_states"] - 1
        mid = last // 2
        print(f"\n  layer0  (input embedding) delta={deltas[0]:+.5f}  "
              f"weat={weats[0]:+.4f}")
        print(f"  layer1  (block 1)         delta={deltas[1]:+.5f}  "
              f"weat={weats[1]:+.4f}")
        print(f"  layer{mid:<2} (block {mid})        delta={deltas[mid]:+.5f}  "
              f"weat={weats[mid]:+.4f}")
        print(f"  layer{last:<2} (block {last})       delta={deltas[last]:+.5f}  "
              f"weat={weats[last]:+.4f}")
        print(f"  non-finite values    : "
              f"{sum(1 for v in deltas + weats if v != v)}")
        print(f"  GPU peak allocated   : "
              f"{torch.cuda.max_memory_allocated(args.gpu)/GIB:.2f} GiB")

    rule("SMOKE TEST COMPLETE — nothing written")


if __name__ == "__main__":
    main()
