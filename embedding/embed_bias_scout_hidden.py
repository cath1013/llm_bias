#!/usr/bin/env python3
"""
FIXED-PROMPT BAI-STYLE WAT — paired behaviour and hidden states, Llama-4-Scout.

WHAT THIS IS
    The Llama-4-Scout port of `embed_bias_llama3_hidden.py`.  The SAME frozen
    272 prompts (`embedding/prompts.csv`), the same system prompt, the same
    sampling setting and prompt-id seed rule, the same strict behavioural parser
    and unsmoothed Bai score, the same span resolution, competitor masking and
    mean pooling, the same delta / singleton-WEAT equations and the same
    secondary set-level WEAT.  A Scout row and a Llama 3.1 row with the same
    `prompt_id` are directly comparable; the only intended difference is the
    model, its backend, and therefore the number of layers.

    Every model-independent function is CALLED from the Llama 3.1 runner
    (`lh.<name>`), never copied, and that file is not modified.  What lives
    here is only what the model forces:

        * the loader — Llama4ForConditionalGeneration, bf16, SDPA,
          device_map="auto" across the visible GPUs, then hard checks that no
          module or parameter sits on CPU, disk or meta;
        * the input device — the device of the text input-embedding module
          (`language_model.model.embed_tokens`) from hf_device_map, never
          `next(model.parameters()).device`, which on Llama 4 is the vision
          tower's;
        * generate / forward / driver, identical to the Llama 3.1 runner apart
          from that device and an architecture fail-fast check;
        * the manifest's model-specific fields.

THIS FILE ALSO HOSTS THE GENERIC SPAN LAYER
    `SpanError`, `find_phrase_char_spans`, `token_span_for_chars`,
    `masking_spans`, `resolve_span`, `build_chat_input` and `pool_span` are
    imported by the completed Llama 3.1 runner and are kept here unchanged.
    They are defined BEFORE `embed_bias_llama3_hidden` is imported, and that
    module is only dereferenced at call time, so the two files may be imported
    in either order.

RETIRED
    The previous contents of this file — Ollama behavioural sessions,
    saved-response replay, the old arms and scopes, PCA nullification, pair
    aggregation and the quantised/offloaded backend — are retired.  Git history
    is their provenance; none of it is reachable from here.

LAYER CONVENTION
    layer0   = hidden_states[0], the input embedding (before any block)
    layer1.. = output of transformer block 1..N
    N comes from config.text_config.num_hidden_layers (Scout: 48), and every
    forward is checked to return N + 1 states of width text_config.hidden_size.
    The primary table's layer columns are generated from that count.

INPUT AND BOS
    Exactly the Llama 3.1 pipeline: apply_chat_template(system + user,
    tokenize=False, add_generation_prompt=True), then tokenizer(templated).
    The template emits one BOS and the tokenizer prepends another, so the model
    reads TWO BOS tokens — as the completed Llama 3.1 run did.  This is kept on
    purpose and recorded in the manifest.

TEXT ONLY
    No processor, no image inputs, no image tokens.  The vision tower is
    loaded with the checkpoint and never run.

USAGE
    python3 embedding/embed_bias_scout_hidden.py --dry-run
    python3 embedding/embed_bias_scout_hidden.py --smoke-test
    python3 embedding/embed_bias_scout_hidden.py --limit 5 \\
        --out-dir responses/embed_scout_diag5
    python3 embedding/embed_bias_scout_hidden.py
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import embed_bias as eb           # csv writer
from embed_bias import ROOT       # project root, not this package directory


# =========================================================
# B2 — robust phrase -> token span location
# =========================================================

class SpanError(RuntimeError):
    """Raised when a stimulus phrase cannot be located unambiguously."""


def find_phrase_char_spans(prompt, phrase, window=None):
    """
    All character spans of `phrase` in `prompt`, case-insensitively.
    `window` = (start, end) restricts the search to one section of the prompt
    (e.g. only the sorted-word list), which is how an intended occurrence is
    disambiguated from an incidental one.
    """
    lo, hi = (0, len(prompt)) if window is None else window
    hay, needle = prompt.lower(), phrase.lower()
    spans, i = [], lo
    while True:
        j = hay.find(needle, i, hi)
        if j < 0:
            return spans
        spans.append((j, j + len(phrase)))
        i = j + 1


def token_span_for_chars(offsets, char_span):
    """
    Token indices whose character offsets overlap `char_span`.
    `offsets` is a HF fast-tokenizer `offset_mapping`: one (start, end) per token.
    Zero-width tokens (specials) are skipped.
    """
    start, end = char_span
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not idx:
        raise SpanError(f"no tokens overlap character span {char_span}")
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise SpanError(f"token span for {char_span} is not contiguous: {idx}")
    return idx[0], idx[-1] + 1          # half-open


def masking_spans(prompt, phrase, competitors, window=None):
    """
    Spans where `phrase` appears only because a LONGER stimulus contains it.

    The FAB prompt lists all 30 Leslie fields, among them both 'art history' and
    'history'.  A substring search for the stimulus 'history' therefore also
    matches inside 'art history', which is a different field and a different
    item in the sorting task.  Any stimulus in the same prompt that strictly
    contains the phrase masks the occurrences that fall inside it.
    """
    blocked = []
    needle = phrase.lower()
    for other in competitors:
        other = str(other)
        if len(other) > len(phrase) and needle in other.lower():
            blocked.extend(find_phrase_char_spans(prompt, other, window))
    return blocked


def resolve_span(prompt, offsets, phrase, window=None, occurrence="single",
                 competitors=()):
    """
    Locate ONE intended occurrence of `phrase` and return its half-open token span.

    occurrence:
      'single' (default) — exactly one match must exist, else SpanError.
      'first' / 'last'   — accept several and take that end deliberately.
    `competitors` are the other stimuli the same prompt presents; an occurrence
    inside a longer one of them is not a mention of `phrase` (see
    `masking_spans`) and is discarded before the count is checked.

    Fails loudly rather than guessing: Bai's first-token `.index()` silently
    returns the wrong position for a multi-token phrase, or for a phrase that
    also appears in the instruction sentence.
    """
    spans = find_phrase_char_spans(prompt, phrase, window)
    blocked = masking_spans(prompt, phrase, competitors, window)
    if blocked:
        spans = [sp for sp in spans
                 if not any(b0 <= sp[0] and sp[1] <= b1 for b0, b1 in blocked)]
    if not spans:
        raise SpanError(f"{phrase!r} does not occur in the prompt"
                        + (f" within {window}" if window else ""))
    if len(spans) > 1 and occurrence == "single":
        raise SpanError(f"{phrase!r} occurs {len(spans)} times "
                        f"{[s for s, _ in spans]} — pass a window or an explicit "
                        f"occurrence rule; refusing to guess")
    chosen = spans[0] if occurrence in ("single", "first") else spans[-1]
    return token_span_for_chars(offsets, chosen)


# =========================================================
# B1a — reconstruct the EXACT input the model saw
# =========================================================


def build_chat_input(tokenizer, system, user_prompt):
    """
    Apply the model's chat template to the same two messages the runner sent, so
    the forward pass sees the Llama-4 header tokens Scout actually read — not the
    bare user string, which is what the jsonl stores.

    Returns (templated_text, user_window) where `user_window` is the character
    span of the user message inside the templated text.  Span lookups are
    restricted to that window so template boilerplate and the system message can
    never supply a match.  Raises if the template does not reproduce the user
    message verbatim, since every char offset downstream depends on it.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    if not isinstance(text, str):
        raise SpanError("apply_chat_template did not return a string; pass "
                        "tokenize=False is required for offset-based spans")
    start = text.find(user_prompt)
    if start < 0:
        raise SpanError(
            "the chat template did not reproduce the saved user prompt "
            "verbatim, so character offsets cannot be trusted. Saved prompt "
            f"starts {user_prompt[:60]!r}; templated text starts {text[:120]!r}")
    if text.find(user_prompt, start + 1) >= 0:
        raise SpanError("the saved user prompt occurs twice in the templated "
                        "text; refusing to guess which copy is the message")
    return text, (start, start + len(user_prompt))


def pool_span(layer_matrix, span, rule="mean"):
    """
    Pool a phrase's token vectors.  `layer_matrix` is (n_tokens, dim) for ONE layer.
    'mean' (default) averages the span; 'last' takes its final token.
    Never silently uses the first token of a multi-token phrase.
    """
    import numpy as np
    lo, hi = span
    block = np.asarray(layer_matrix, dtype=float)[lo:hi]
    if block.size == 0:
        raise SpanError(f"empty span {span}")
    if rule == "mean":
        return block.mean(axis=0)
    if rule == "last":
        return block[-1]
    raise ValueError(f"unknown pooling rule {rule!r} (expected 'mean' or 'last')")


# =========================================================
# The shared scientific logic
# =========================================================

# Imported HERE, after the span layer, because `embed_bias_llama3_hidden`
# imports that layer from this file.  Only module attributes are read, and only
# inside functions, so this works whichever of the two is imported first.
import embed_bias_llama3_hidden as lh

ANALYSIS_VERSION = "4.0-scout-fixed-prompt-bai"
CHECKPOINT = "/home/kjin/models/llama4-scout"
OUT_DIR = os.path.join(ROOT, "responses", "embed_scout")
ATTN_IMPLEMENTATION = "sdpa"
FORBIDDEN_PLACEMENTS = ("cpu", "disk", "meta")


class ArchitectureMismatch(RuntimeError):
    """A forward pass returned a hidden-state stack the config does not describe."""


# =========================================================
# Architecture metadata and device placement
# =========================================================

def text_config_of(config):
    """
    The language model's config.  Llama 4 nests it under `text_config`, where
    num_hidden_layers and hidden_size live; a plain causal LM does not.
    """
    return getattr(config, "text_config", None) or config


def _as_device_str(value):
    """accelerate records GPU placements as ints; normalise to 'cuda:N'."""
    if isinstance(value, int):
        return f"cuda:{value}"
    return str(value)


def validate_device_map(device_map):
    """
    Every module on a GPU, or the run stops.  Returns the sorted devices used.

    device_map="auto" quietly spills to CPU and then disk when the GPUs are
    full.  The verified Scout load does not; a load that did would be a
    different execution path, so it is refused rather than recorded.
    """
    if not device_map:
        raise SystemExit("the model has no hf_device_map — expected a "
                         "device_map='auto' load across the visible GPUs")
    stray = {name: _as_device_str(dev) for name, dev in device_map.items()
             if _as_device_str(dev).startswith(FORBIDDEN_PLACEMENTS)}
    if stray:
        raise SystemExit(f"{len(stray)} module(s) placed off-GPU "
                         f"{dict(list(stray.items())[:5])} — this runner forbids "
                         f"CPU/disk offload; refusing to continue")
    return sorted({_as_device_str(d) for d in device_map.values()})


def validate_parameter_devices(model):
    """
    {device: n_parameters}, after checking where every parameter ACTUALLY is.
    The map can say cuda while an offload hook keeps the weight on meta/cpu.
    """
    placement = defaultdict(int)
    for _, p in model.named_parameters():
        placement[str(p.device)] += 1
    stray = [d for d in placement if d.startswith(FORBIDDEN_PLACEMENTS)]
    if stray:
        raise SystemExit(f"parameters landed on {stray} — this runner forbids "
                         f"CPU/disk offload; refusing to continue")
    return dict(placement)


def resolve_mapped_device(device_map, module_name):
    """
    The device hf_device_map assigns to `module_name`: the entry with the
    longest key that is that module or one of its ancestors ('' is the root).
    """
    import torch
    best = None
    for key in device_map:
        if key == "" or module_name == key or module_name.startswith(key + "."):
            if best is None or len(key) > len(best):
                best = key
    if best is None:
        raise SystemExit(f"{module_name!r} is not covered by hf_device_map")
    dev = _as_device_str(device_map[best])
    if dev.startswith(FORBIDDEN_PLACEMENTS):
        raise SystemExit(f"{module_name!r} is mapped to {dev!r} via {best!r}; "
                         f"inputs must start on a GPU")
    return torch.device(dev)


def embedding_module_name(model):
    """Qualified name of `model.get_input_embeddings()` inside `model`."""
    emb = model.get_input_embeddings()
    for name, module in model.named_modules():
        if module is emb:
            return name
    raise SystemExit("get_input_embeddings() is not a submodule of the model")


def input_embedding_device(model):
    """
    (device, module_name) for the inputs of BOTH generate() and forward().

    On Llama4ForConditionalGeneration the text embedding is
    `language_model.model.embed_tokens`.  The device is read from
    hf_device_map and must agree with where the embedding weight really is.
    """
    name = embedding_module_name(model)
    dev = resolve_mapped_device(getattr(model, "hf_device_map", None) or {}, name)
    actual = model.get_input_embeddings().weight.device
    if actual != dev:
        raise SystemExit(f"hf_device_map puts {name!r} on {dev} but its weight "
                         f"is on {actual}; refusing to guess the input device")
    return dev, name


def check_architecture(hidden_states, architecture, context=""):
    """
    len(hidden_states) == num_hidden_layers + 1, and every state has width
    hidden_size.  A mismatch means the layer columns would not mean what the
    manifest says, so the run stops.
    """
    want = architecture["n_transformer_blocks"] + 1
    if len(hidden_states) != want:
        raise ArchitectureMismatch(
            f"forward returned {len(hidden_states)} hidden states; text_config "
            f"implies {want} (1 input embedding + "
            f"{architecture['n_transformer_blocks']} blocks)"
            + (f" [{context}]" if context else "") + " — aborting the run")
    for i, h in enumerate(hidden_states):
        if int(h.shape[-1]) != architecture["hidden_size"]:
            raise ArchitectureMismatch(
                f"hidden state {i} has width {int(h.shape[-1])}; text_config "
                f"hidden_size is {architecture['hidden_size']}"
                + (f" [{context}]" if context else "") + " — aborting the run")


def leading_bos_count(tokenizer, templated):
    """BOS ids opening the tokenised input (template BOS + tokenizer BOS)."""
    ids = tokenizer(templated)["input_ids"]
    bos = tokenizer.bos_token_id
    n = 0
    while n < len(ids) and ids[n] == bos:
        n += 1
    return n


def bos_record(tokenizer, templated):
    """What the manifest says about BOS, measured rather than asserted."""
    bos = tokenizer.bos_token
    in_template = 0
    while templated.startswith(bos * (in_template + 1)):
        in_template += 1
    return {
        "bos_token": bos,
        "bos_token_id": tokenizer.bos_token_id,
        "bos_in_templated_text": in_template,
        "bos_in_model_input": leading_bos_count(tokenizer, templated),
        "how": "apply_chat_template(..., tokenize=False) emits one BOS; "
               "tokenizer(templated) prepends another special BOS",
        "decision": "kept deliberately: the completed Llama 3.1 run fed the "
                    "model the same double BOS, so preprocessing is comparable "
                    "across the two models",
    }


# =========================================================
# Model
# =========================================================

def load_scout(checkpoint=CHECKPOINT, dtype="bfloat16"):
    """
    The verified multi-GPU bf16 load.  No quantisation, no `max_memory`, no
    offload folder; any CPU/disk/meta placement aborts.
    """
    import torch
    from transformers import (AutoConfig, AutoTokenizer,
                              Llama4ForConditionalGeneration)

    if not os.path.isdir(checkpoint):
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    shards = sorted(f for f in os.listdir(checkpoint) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors in {checkpoint}")

    torch_dtype = getattr(torch, dtype)
    config = AutoConfig.from_pretrained(checkpoint)
    tcfg = text_config_of(config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("a fast tokenizer is required for offset-based spans")

    t0 = time.time()
    model = Llama4ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch_dtype, device_map="auto",
        attn_implementation=ATTN_IMPLEMENTATION, low_cpu_mem_usage=True)
    load_seconds = time.time() - t0
    model.eval()

    device_map = dict(getattr(model, "hf_device_map", None) or {})
    devices = validate_device_map(device_map)
    placement = validate_parameter_devices(model)
    input_device, embed_name = input_embedding_device(model)
    param_dtypes = defaultdict(int)
    for _, p in model.named_parameters():
        param_dtypes[str(p.dtype)] += 1

    gen_cfg = getattr(model, "generation_config", None)
    meta = dict(
        checkpoint=checkpoint,
        safetensors_shards=shards,
        ignored_duplicates=[],
        model_class=model.__class__.__name__,
        architectures=list(getattr(config, "architectures", []) or []),
        dtype=str(model.dtype),
        parameter_dtypes=dict(param_dtypes),
        tokenizer_class=tokenizer.__class__.__name__,
        n_transformer_blocks=int(tcfg.num_hidden_layers),
        hidden_size=int(tcfg.hidden_size),
        num_local_experts=getattr(tcfg, "num_local_experts", None),
        num_experts_per_tok=getattr(tcfg, "num_experts_per_tok", None),
        attn_implementation=getattr(model.config, "_attn_implementation",
                                    ATTN_IMPLEMENTATION),
        device=", ".join(devices),
        devices=devices,
        input_device=str(input_device),
        input_embedding_module=embed_name,
        device_placement=placement,
        hf_device_map={k: _as_device_str(v) for k, v in device_map.items()},
        checkpoint_generation_config=(gen_cfg.to_diff_dict()
                                      if gen_cfg is not None else None),
        quantisation="none",
        cpu_offload=False,
        disk_offload=False,
        load_seconds=round(load_seconds, 2),
    )
    return model, tokenizer, meta


def generate_response(templated, tokenizer, model, *, seed, device,
                      generation=None):
    """
    `lh.generate_response`, with the input device passed in.  Generation is
    for behaviour only; its tokens are never embedded or pooled.
    """
    import torch
    cfg = dict(lh.GENERATION if generation is None else generation)
    enc = tokenizer(templated, return_tensors="pt")
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


def extract_prompt_vectors(templated, phrases, tokenizer, model, *, device,
                           architecture, user_window, competitors,
                           pooling="mean", context=""):
    """
    `lh.extract_prompt_vectors`, with the input device passed in and the
    hidden-state stack checked against text_config before anything is pooled.
    """
    import torch
    import numpy as np

    enc = tokenizer(templated, return_tensors="pt", return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    enc = {k: v.to(device) for k, v in enc.items()}

    t0 = time.time()
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
    forward_seconds = time.time() - t0

    lh.assert_finite_hidden_states(out.hidden_states, context)
    check_architecture(out.hidden_states, architecture, context)
    hidden = [h[0].to(dtype=torch.float32).cpu().numpy() for h in out.hidden_states]

    spans, vectors = {}, {}
    for phrase in phrases:
        span = resolve_span(templated, offsets, phrase, window=user_window,
                            competitors=[c for c in competitors if c != phrase])
        spans[phrase] = span
        pooled = [pool_span(layer, span, pooling) for layer in hidden]
        for i, v in enumerate(pooled):
            if not np.isfinite(v).all():
                raise lh.NonFiniteHiddenState(
                    f"pooled vector for {phrase!r} is non-finite at hidden state "
                    f"{i} ({lh.layer_label(i)['representation_type']})"
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
# Driver
# =========================================================

def run_prompts(prompts, stim, model, tokenizer, *, args, device, architecture,
                acc=None, verbose=True):
    """
    `lh.run_prompts`, with the input device and architecture passed through:
    one generate() and one forward() per frozen prompt.
    """
    rows, failures = [], []
    n_hidden = None
    t_start = time.time()
    for k, record in enumerate(prompts, start=1):
        roles = lh.prompt_roles(record, stim)
        templated, window = build_chat_input(tokenizer, lh.SYSTEM_PROMPT,
                                             record["prompt"])
        context = record["prompt_id"]

        response, gen_meta = generate_response(
            templated, tokenizer, model, device=device,
            seed=lh.prompt_seed(record["prompt_id"], args.seed),
            generation=dict(lh.GENERATION, max_new_tokens=args.max_new_tokens))

        vectors, meta = extract_prompt_vectors(
            templated, roles["phrases"], tokenizer, model, device=device,
            architecture=architecture,
            user_window=window, competitors=roles["phrases"], context=context)
        n_hidden = meta["n_hidden_states"]

        assign, problems = lh.parse_behaviour(response, record, roles)
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
            score, denom_problem = lh.behaviour_wat_score(assign, record, roles)
            if denom_problem:
                failures.append(dict(prompt_id=record["prompt_id"],
                                     domain=record["domain"],
                                     category=record["category"],
                                     trial=record["trial"],
                                     n_items_assigned=len(assign),
                                     n_items_presented=len(roles["attributes"]),
                                     problems=denom_problem,
                                     response=response))

        deltas, weats = lh.representation_scores(vectors, roles, n_hidden)
        rows.append(lh.wide_row(record, response, score, deltas, weats))

        if acc is not None:
            lh.accumulate_prompt(acc, record["domain"], record["category"],
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


def restamp(rows):
    """
    The secondary builders live in lh and stamp lh's version; these rows are
    Scout's.  Nothing else in a row is touched.
    """
    for r in rows:
        r["analysis_version"] = ANALYSIS_VERSION
    return rows


def secondary_rows(acc, prompts, stim, n_hidden):
    """(set-level rows, stimulus-layer rows, missing), Scout-stamped."""
    set_rows, stim_rows, missing = [], [], {}
    for domain, category in sorted({(r["domain"], r["category"]) for r in prompts}):
        arm = lh.arm_for(domain, category, stim)
        if not acc.stimuli(domain, category):
            continue
        srows, miss = lh.set_level_rows(acc, domain, category, arm,
                                        n_hidden_states=n_hidden)
        set_rows.extend(restamp(srows))
        if any(miss.values()):
            missing[f"{domain}/{category}"] = {k: v for k, v in miss.items() if v}
        stim_rows.extend(restamp(lh.stimulus_layer_rows(
            acc, domain, category, arm, n_hidden_states=n_hidden)))
    return set_rows, stim_rows, missing


# =========================================================
# Manifest
# =========================================================

def build_manifest(*, backend, arch, design, n_hidden_states, parse_failures,
                   generation, missing, args, bos):
    """
    The Llama 3.1 manifest, key for key, with every model-specific field
    replaced so the two can be diffed — and nothing left that describes the
    other model, except the explicit `analysis_logic_source` provenance.
    """
    m = lh.build_manifest(backend=backend, arch=arch, design=design,
                          n_hidden_states=n_hidden_states,
                          parse_failures=parse_failures, generation=generation,
                          missing=missing, args=args)
    m.pop("ignored_duplicate_weights", None)
    m["analysis"] = ("Fixed-prompt Bai-style WAT: paired Llama-4-Scout "
                     "behaviour and contextualised hidden states from the same "
                     "prompt")
    m["analysis_version"] = ANALYSIS_VERSION
    m["analysis_logic_source"] = {
        "file": os.path.relpath(lh.__file__, ROOT),
        "sha256": lh.file_digest(lh.__file__),
        "shared": "design, roles, behavioural parsing and score, span "
                  "resolution, pooling, delta / singleton WEAT, prototype "
                  "accumulation and set-level WEAT are called from this file "
                  "unchanged",
    }
    m["layer_convention"]["layerN"] = (
        f"layer{n_hidden_states - 1}: output of block {n_hidden_states - 1} "
        f"after the final RMSNorm (transformers ties the last hidden state to "
        f"last_hidden_state)")
    m["architecture_source"] = ("config.text_config.num_hidden_layers / "
                                "hidden_size")
    m["fail_fast"]["architecture_mismatch"] = (
        "aborts the run: every forward must return num_hidden_layers + 1 "
        "states, each of width hidden_size")
    m["secondary_outputs"]["set_level_weat"] = (
        "contextual prototypes over THIS experiment's prompts only")
    m["secondary_outputs"]["analysis_version"] = (
        "every secondary row is stamped " + ANALYSIS_VERSION)
    m.update({
        "devices": backend["devices"],
        "hf_device_map": backend["hf_device_map"],
        "device_placement": backend["device_placement"],
        "input_embedding_module": backend["input_embedding_module"],
        "input_device": backend["input_device"],
        "input_device_rule": "device of get_input_embeddings() from "
                             "hf_device_map (longest matching ancestor), "
                             "checked against its weight; used for generate() "
                             "and forward()",
        "attn_implementation": backend["attn_implementation"],
        "parameter_dtypes": backend["parameter_dtypes"],
        "num_local_experts": backend["num_local_experts"],
        "num_experts_per_tok": backend["num_experts_per_tok"],
        "modality": "text only — no processor, no image inputs, no image "
                    "tokens; the vision tower is loaded but never run",
        "checkpoint_generation_config": backend["checkpoint_generation_config"],
        "generation_override": "the explicit `generation` settings override "
                               "the checkpoint's generation_config defaults",
        "bos": bos,
    })
    return m


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--stimuli", default=lh.STIMULI_CSV)
    ap.add_argument("--prompts", default=lh.PROMPTS_CSV)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=lh.SEED_BASE,
                    help="base generation seed; per-prompt seeds derive from it")
    ap.add_argument("--max-new-tokens", type=int,
                    default=lh.GENERATION["max_new_tokens"])
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
    stim, all_prompts = lh.load_design(
        args.stimuli, args.prompts,
        expect_total=None if partial else lh.N_TOTAL)
    design = lh.design_summary(all_prompts, args.stimuli, args.prompts)

    print(f"Fixed-prompt Bai-style WAT — Llama-4-Scout  "
          f"(analysis {ANALYSIS_VERSION})")
    print(f"  stimuli    : {args.stimuli}")
    print(f"  prompts    : {args.prompts}  ({len(all_prompts)} frozen prompts)")
    for cond, n in design["n_by_condition"].items():
        print(f"    {cond:<20} {n}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  generation : "
          f"{dict(lh.GENERATION, max_new_tokens=args.max_new_tokens)}")
    print(f"  seed base  : {args.seed}")

    prompts = all_prompts
    if args.domain != "both":
        prompts = [r for r in prompts if r["domain"] == args.domain]
    if args.smoke_test:
        gender = next((r for r in prompts if r["domain"] == "gender"), None)
        fab = next((r for r in prompts if r["domain"] == "fab"), None)
        prompts = [r for r in (gender, fab) if r is not None]
    elif args.limit:
        prompts = prompts[:args.limit]

    # role resolution needs no GPU — do it for every prompt before loading Scout
    for record in prompts:
        lh.prompt_roles(record, stim)
    print(f"  roles resolved for all {len(prompts)} prompts "
          f"(X/Y/A/B from stimuli.csv, not group0/group1)")

    if args.dry_run:
        print("\n--dry-run: design validated, no model loaded, nothing written.")
        return

    import torch, transformers
    model, tok, backend = load_scout(args.checkpoint)
    device = torch.device(backend["input_device"])
    architecture = dict(n_transformer_blocks=backend["n_transformer_blocks"],
                        hidden_size=backend["hidden_size"])
    print(f"\n  loaded {backend['model_class']} in {backend['load_seconds']}s "
          f"on {backend['device']}  ({backend['dtype']}, "
          f"{backend['quantisation']}, {backend['attn_implementation']})")
    print(f"  inputs -> {device} ({backend['input_embedding_module']})")

    if args.smoke_test:
        return smoke_test(args, stim, prompts, all_prompts, model, tok,
                          backend, device, architecture)

    acc = lh.PrototypeAccumulator()
    rows, failures, meta = run_prompts(prompts, stim, model, tok, args=args,
                                       device=device, architecture=architecture,
                                       acc=acc)
    n_hidden = meta["n_hidden_states"]
    print(f"\n  {len(rows)} prompts in {meta['total_seconds']/60:.1f} min; "
          f"{len(failures)} behavioural parse failure(s)")

    set_rows, stim_rows, missing = secondary_rows(acc, prompts, stim, n_hidden)

    os.makedirs(args.out_dir, exist_ok=True)
    lh.write_primary_csv(os.path.join(args.out_dir, lh.PRIMARY_CSV), rows,
                         n_hidden, force=args.force)
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

    first, _ = build_chat_input(tok, lh.SYSTEM_PROMPT, prompts[0]["prompt"])
    manifest = build_manifest(
        backend=backend,
        arch=dict(transformers_version=transformers.__version__,
                  torch_version=torch.__version__),
        design=design, n_hidden_states=n_hidden, parse_failures=failures,
        generation=dict(lh.GENERATION, max_new_tokens=args.max_new_tokens),
        missing=missing, args=args, bos=bos_record(tok, first))
    with open(os.path.join(args.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  wrote {os.path.join(args.out_dir, 'manifest.json')}")


def validate_all_spans(tokenizer, prompts, stim):
    """
    Tokenizer-only: every stimulus of every prompt resolves to exactly one span
    under THIS tokenizer.  Returns (min tokens, max tokens, {leading BOS}).
    """
    n_tok, n_bos = [], set()
    for record in prompts:
        roles = lh.prompt_roles(record, stim)
        templated, window = build_chat_input(tokenizer, lh.SYSTEM_PROMPT,
                                             record["prompt"])
        enc = tokenizer(templated, return_offsets_mapping=True)
        for phrase in roles["phrases"]:
            resolve_span(templated, enc["offset_mapping"], phrase, window=window,
                         competitors=[c for c in roles["phrases"] if c != phrase])
        n_tok.append(len(enc["input_ids"]))
        n_bos.add(leading_bos_count(tokenizer, templated))
    return min(n_tok), max(n_tok), sorted(n_bos)


def smoke_test(args, stim, prompts, all_prompts, model, tokenizer, backend,
               device, architecture):
    """One gender + one FAB prompt, every diagnostic, nothing written."""
    import torch

    def rule(t):
        print(f"\n{'='*74}\n{t}\n{'='*74}", flush=True)

    GIB = float(1 << 30)
    gpus = [torch.device(d) for d in backend["devices"]]
    for g in gpus:
        free, total = torch.cuda.mem_get_info(g)
        print(f"  {g} after load : {(total-free)/GIB:.2f} GiB used, "
              f"{free/GIB:.2f} GiB free")
    print(f"  text_config        : {architecture['n_transformer_blocks']} "
          f"blocks, hidden {architecture['hidden_size']}, "
          f"{backend['num_local_experts']} experts "
          f"(top-{backend['num_experts_per_tok']})")
    lo, hi, n_bos = validate_all_spans(tokenizer, all_prompts, stim)
    print(f"  spans              : every stimulus of all {len(all_prompts)} "
          f"prompts resolves once; {lo}..{hi} tokens; leading BOS {n_bos}")

    for record in prompts:
        roles = lh.prompt_roles(record, stim)
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

        templated, window = build_chat_input(tokenizer, lh.SYSTEM_PROMPT,
                                             record["prompt"])
        n_tok = len(tokenizer(templated)["input_ids"])
        bos = bos_record(tokenizer, templated)
        print(f"\n  system prompt        : {lh.SYSTEM_PROMPT!r}")
        print(f"  chat-template tokens : {n_tok}   BOS in template "
              f"{bos['bos_in_templated_text']}, in model input "
              f"{bos['bos_in_model_input']}")

        seed = lh.prompt_seed(record["prompt_id"], args.seed)
        response, gen_meta = generate_response(
            templated, tokenizer, model, seed=seed, device=device,
            generation=dict(lh.GENERATION, max_new_tokens=args.max_new_tokens))
        print(f"\n  generation           : seed={seed} "
              f"temp={gen_meta['temperature']} do_sample={gen_meta['do_sample']} "
              f"new_tokens={gen_meta['n_new_tokens']} "
              f"({gen_meta['generate_seconds']:.1f}s)")
        print("  response:")
        for line in response.splitlines():
            print(f"    {line}")

        vectors, meta = extract_prompt_vectors(
            templated, roles["phrases"], tokenizer, model, device=device,
            architecture=architecture, user_window=window,
            competitors=roles["phrases"], context=record["prompt_id"])
        print(f"\n  hidden states        : {meta['n_hidden_states']} "
              f"(= 1 input embedding + {meta['n_transformer_blocks']} blocks), "
              f"dim {meta['hidden_size']}  — matches text_config")
        print(f"  prompt tokens        : {meta['n_prompt_tokens']}   "
              f"forward {meta['forward_seconds']:.2f}s   "
              f"finite check {meta['finite_check']}")
        print(f"  pooling              : {meta['pooling']} over every sub-token")
        for phrase in roles["phrases"]:
            lo, hi = meta["spans"][phrase]
            print(f"    {phrase:<26} tokens [{lo}, {hi})  "
                  f"{meta['subtokens'][phrase]}")

        assign, problems = lh.parse_behaviour(response, record, roles)
        if problems:
            score, denom = float("nan"), None
            print(f"\n  behavioural parse    : FAILED — {' | '.join(problems)}")
            print("  (representation columns are kept; behaviour_wat_score = NaN)")
        else:
            score, denom = lh.behaviour_wat_score(assign, record, roles)
            print(f"\n  behavioural parse    : all "
                  f"{len(roles['attributes'])} items assigned")
            if denom is None:
                print(f"  behaviour_wat_score  : {score:+.4f}  "
                      f"(positive = stereotype-congruent)")
            else:
                print(f"  behaviour_wat_score  : NaN — {denom}")

        deltas, weats = lh.representation_scores(vectors, roles,
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
        for g in gpus:
            print(f"  {g} peak allocated : "
                  f"{torch.cuda.max_memory_allocated(g)/GIB:.2f} GiB")

    rule("SMOKE TEST COMPLETE — nothing written")


if __name__ == "__main__":
    main()
