"""
LLAMA 3.1 BACKEND — Llama-3.1-8B-Instruct, bf16, ONE GPU.

Model-specific code only, moved verbatim from `embed_bias_llama3_hidden.py`:
the loader (`device_map={"": gpu}`, no quantisation, no offload), generation,
and the hidden-state forward pass.  Everything model-independent (spans,
pooling, roles, scores) comes from `embedding.core`.

LAYER CONVENTION
    layer0 = hidden_states[0] (input embedding); layer1..layer32 = block outputs.
    33 hidden-state levels, read from the model, never hard-coded.
"""

import os
import time
from collections import defaultdict

from embedding import ROOT
from embedding.core import (GENERATION, POOLING, NonFiniteHiddenState,
                            assert_finite_hidden_states, layer_label,
                            resolve_span, pool_span)

NAME = "llama31"
CHECKPOINT = "/home/syryu/models/llama3.1-8b-instruct"

WAT_ANALYSIS_VERSION = "4.0-llama31-fixed-prompt-bai"
WAT_OUT_DIR = os.path.join(ROOT, "responses", "embed_llama31")
WAT_MANIFEST_ANALYSIS = ("Fixed-prompt Bai-style WAT: paired Llama-3.1 behaviour and "
                         "contextualised hidden states from the same prompt")
CHAINED_ANALYSIS_VERSION = "chained-1.0-llama31"
CHAINED_OUT_DIR = os.path.join(ROOT, "responses", "embed_llama31_chained")


# =========================================================
# Model — from embed_bias_llama3_hidden.py, verbatim
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
# Backend interface
# =========================================================

class Backend:
    """A loaded Llama 3.1: the interface the runners use."""

    name = NAME

    def __init__(self, model, tokenizer, meta, gpu=0):
        self.model, self.tokenizer, self.meta, self.gpu = model, tokenizer, meta, gpu

    def generate(self, templated, *, seed, generation=None):
        return generate_response(templated, self.tokenizer, self.model, seed=seed,
                                 generation=generation)

    def extract_hidden(self, templated, phrases, *, user_window, competitors,
                       context=""):
        return extract_prompt_vectors(templated, phrases, self.tokenizer, self.model,
                                      user_window=user_window,
                                      competitors=competitors, context=context)

    def describe(self):
        return []

    def memory_report(self, peak=False):
        import torch
        GIB = float(1 << 30)
        if peak:
            return [f"  GPU peak allocated   : "
                    f"{torch.cuda.max_memory_allocated(self.gpu)/GIB:.2f} GiB"]
        free, total = torch.cuda.mem_get_info(self.gpu)
        return [f"  GPU {self.gpu} after load : {(total-free)/GIB:.2f} GiB used, "
                f"{free/GIB:.2f} GiB free"]


def load(checkpoint=None, gpu=0):
    model, tokenizer, meta = load_llama3(checkpoint or CHECKPOINT, gpu)
    return Backend(model, tokenizer, meta, gpu)


def finalize_wat_manifest(m, *, backend_meta, n_hidden_states, bos):
    """The WAT-only manifest is already Llama 3.1's; nothing to add."""
    return m
