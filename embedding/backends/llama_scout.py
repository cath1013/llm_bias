"""
LLAMA 4 SCOUT BACKEND — Llama-4-Scout-17B-16E-Instruct, bf16, multi-GPU.

Model-specific code only, moved verbatim from `embed_bias_scout_hidden.py`:

    * Llama4ForConditionalGeneration, bf16, attn_implementation="sdpa",
      device_map="auto", low_cpu_mem_usage=True; no quantisation, no
      max_memory, no offload folder;
    * hard checks that no module (hf_device_map) and no parameter sits on CPU,
      disk or meta;
    * text_config metadata (num_hidden_layers, hidden_size);
    * the input device = the device of `language_model.model.embed_tokens`
      from hf_device_map, checked against its weight — never
      `next(model.parameters()).device`, which is the vision tower's;
    * generation and the hidden-state forward pass on that device, with the
      architecture check (num_hidden_layers + 1 states of width hidden_size)
      before anything is pooled;
    * the Scout-specific WAT-only manifest fields.

Text only: no processor, no image inputs.  Same chat-template input as Llama
3.1, including its double BOS (template BOS + tokenizer BOS).  Everything
model-independent comes from `embedding.core`.
"""

import os
import time
from collections import defaultdict

from embedding import ROOT
from embedding import core
from embedding.core import (GENERATION, NonFiniteHiddenState, ArchitectureMismatch,
                            assert_finite_hidden_states, layer_label, resolve_span,
                            pool_span, file_digest)

NAME = "scout"
CHECKPOINT = "/home/kjin/models/llama4-scout"

WAT_ANALYSIS_VERSION = "4.0-scout-fixed-prompt-bai"
WAT_OUT_DIR = os.path.join(ROOT, "responses", "embed_scout")
WAT_MANIFEST_ANALYSIS = ("Fixed-prompt Bai-style WAT: paired Llama-4-Scout "
                         "behaviour and contextualised hidden states from the same "
                         "prompt")
CHAINED_ANALYSIS_VERSION = "chained-1.0-scout"
CHAINED_OUT_DIR = os.path.join(ROOT, "responses", "embed_scout_chained")
ATTN_IMPLEMENTATION = "sdpa"

FORBIDDEN_PLACEMENTS = ("cpu", "disk", "meta")



# =========================================================
# Architecture metadata and device placement — from embed_bias_scout_hidden.py, verbatim
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



# =========================================================
# Model — from embed_bias_scout_hidden.py, verbatim
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
    `generate_response`, with the input device passed in.  Generation is
    for behaviour only; its tokens are never embedded or pooled.
    """
    import torch
    cfg = dict(GENERATION if generation is None else generation)
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
    `extract_prompt_vectors`, with the input device passed in and the
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

    assert_finite_hidden_states(out.hidden_states, context)
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
    """A loaded Scout: the interface the runners use."""

    name = NAME

    def __init__(self, model, tokenizer, meta):
        import torch
        self.model, self.tokenizer, self.meta = model, tokenizer, meta
        self.device = torch.device(meta["input_device"])
        self.architecture = dict(n_transformer_blocks=meta["n_transformer_blocks"],
                                 hidden_size=meta["hidden_size"])

    def generate(self, templated, *, seed, generation=None):
        return generate_response(templated, self.tokenizer, self.model, seed=seed,
                                 device=self.device, generation=generation)

    def extract_hidden(self, templated, phrases, *, user_window, competitors,
                       context=""):
        return extract_prompt_vectors(templated, phrases, self.tokenizer, self.model,
                                      device=self.device,
                                      architecture=self.architecture,
                                      user_window=user_window,
                                      competitors=competitors, context=context)

    def describe(self):
        m = self.meta
        return [f"  text_config        : {m['n_transformer_blocks']} "
                f"blocks, hidden {m['hidden_size']}, "
                f"{m['num_local_experts']} experts "
                f"(top-{m['num_experts_per_tok']})",
                f"  inputs -> {self.device} ({m['input_embedding_module']})"]

    def memory_report(self, peak=False):
        import torch
        GIB = float(1 << 30)
        out = []
        for g in [torch.device(d) for d in self.meta["devices"]]:
            if peak:
                out.append(f"  {g} peak allocated : "
                           f"{torch.cuda.max_memory_allocated(g)/GIB:.2f} GiB")
            else:
                free, total = torch.cuda.mem_get_info(g)
                out.append(f"  {g} after load : {(total-free)/GIB:.2f} GiB used, "
                           f"{free/GIB:.2f} GiB free")
        return out


def load(checkpoint=None, gpu=0):
    """`gpu` is ignored: Scout is placed by device_map="auto"."""
    model, tokenizer, meta = load_scout(checkpoint or CHECKPOINT)
    return Backend(model, tokenizer, meta)


# =========================================================
# Scout-specific WAT-only manifest fields
# =========================================================

def finalize_wat_manifest(m, *, backend_meta, n_hidden_states, bos):
    """
    Scout's edits to the shared WAT-only manifest (was the body of
    embed_bias_scout_hidden.build_manifest after its call to the Llama 3.1
    manifest): every model-specific field replaced, nothing left that describes
    the other model, except the explicit `analysis_logic_source` provenance.
    """
    backend = backend_meta
    m.pop("ignored_duplicate_weights", None)
    m["analysis"] = WAT_MANIFEST_ANALYSIS
    m["analysis_version"] = WAT_ANALYSIS_VERSION
    m["analysis_logic_source"] = {
        "file": os.path.relpath(core.__file__, ROOT),
        "sha256": file_digest(core.__file__),
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
        "every secondary row is stamped " + WAT_ANALYSIS_VERSION)
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

