"""
PART B/C — MODEL-INTERNAL HIDDEN-STATE ANALYSIS  (Llama 4 Scout only)

HOW THIS DIFFERS FROM `embed_bias.py`
    `embed_bias.py` uses an OpenAI embedding model as an INDEPENDENT semantic
    reference space.  It says nothing about any generating model's internals.
    This file is the other thing: representations taken from the SAME model whose
    behavioural WAT responses we collected (Scout).  The two analyses are kept
    conceptually and programmatically separate; nothing here reads an API
    embedding, and nothing in `embed_bias.py` reads a hidden state.

STATUS IN THIS REPOSITORY  —  the forward pass cannot run here
    Every `scout-ctx` / `scout-ctx:latest` session in `responses/` was produced
    through **Ollama** (`provider: "ollama"`, OpenAI-compatible
    /v1/chat/completions; see `wat_fab.py` / `wat_gender_brilliance.py`).  That
    interface returns text only.  Ollama's HTTP API has no per-layer hidden-state
    output at all: `/api/embed` returns a single final-layer vector, which is NOT
    a layer profile and is NOT what Bai's SI analysis used.
    `preflight()` checks this at runtime and `main()` stops with that message
    rather than substituting anything.  Substituting an API embedding, or a
    different open model, for Scout's internals would make the analysis a
    different claim, so this module refuses to do it.

    What would be needed: Scout weights behind an interface that returns
    `output_hidden_states=True` (HF transformers, vLLM with hidden-state return,
    or llama.cpp patched to dump per-layer activations).

WHAT IS IMPLEMENTED AND TESTED HERE
    Everything that does not need the weights, so the analysis is ready the day a
    hidden-state backend exists:
      * robust exact-span location for multi-token phrases (B2), with explicit
        failure on zero or ambiguous matches — not Bai's `.index()` on the first
        token id,
      * configurable pooling over a phrase's token span (mean default, last
        optional), recorded in the manifest,
      * Bai-inspired exploratory all-but-the-top-style PC nullification
        (Part C), with k=0 an exact identity and the maximum k bounded by the
        centred rank min(n_stimulus_vectors - 1, hidden_dim),
      * layer-wise pair-aggregated Spearman/Pearson (B4), no pseudoreplication.
    These are covered by `test_embed_bias.py`.  `extract_prompt_hidden_states()`
    is written against the HF transformers API and is the only part this
    environment cannot exercise.

INTERPRETATION BOUNDARIES
    * A layer profile describes Scout's internal geometry while it READS the
      prompt.  The generated response is never embedded.
    * PC nullification is NOT debiasing.  It asks whether dominant shared
      directions obscure the association of interest.  k=0 (unaltered) is the
      primary analysis; nullified results are exploratory and stored separately.
      It is Bai-inspired, not a reproduction of his `pca.py` (which re-adds the
      mean; available here only as the explicit `add_mean_back=True` mode).

USAGE
    # 1. what is and is not available (no weights touched)
    python3 embedding/embed_bias_scout_hidden.py --preflight

    # 2. smoke run: 20 sessions, writes nothing, proves the plumbing
    .venv-hf/bin/python embedding/embed_bias_scout_hidden.py \
        --hf-checkpoint <path to Llama-4-Scout> --limit 20

    # 3. full run under the reported scope (4-bit NF4 by default)
    .venv-hf/bin/python embedding/embed_bias_scout_hidden.py \
        --hf-checkpoint <path> --quant 4bit --pca-k 2

MODEL INPUT
    The jsonl stores the user message only.  Scout actually read the runner's
    system message plus that user message through the Llama-4 chat template, so
    the forward pass rebuilds exactly that with `apply_chat_template`, and every
    span lookup is confined to the user-message window inside the result.  The
    generated response is still never embedded.
"""

import os
import sys
import re
import math
import csv
import json
import argparse
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import embed_bias as eb           # stats + dataset wiring + behaviour recomputation
from embed_bias import ROOT       # project root, not this package directory

ANALYSIS_VERSION = "2.0"
OUT_DIR = os.path.join(ROOT, "responses", "embed_scout")

# The behavioural model ids this analysis is about, and the Scout repo ids that
# would be a legitimate source of its internals.
SCOUT_BEHAVIOURAL_MODELS = ("scout-ctx", "scout-ctx:latest", "llama4:scout",
                            "meta-llama/llama-4-scout-17b-16e-instruct")
SCOUT_CHECKPOINT_PATTERN = re.compile(r"llama[-_ ]?4.*scout", re.I)

OUTPUT_FILES = ["hidden_trial_layer.csv", "hidden_pair_layer_aggregated.csv",
                "hidden_layer_correlations.csv", "hidden_nullified_layer.csv",
                "hidden_manifest.json"]


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

def system_prompt_for(dataset):
    """
    The system message the behavioural runner sent, taken from the runner itself
    (never hardcoded here).  Ollama applies the Modelfile's own SYSTEM only when
    the request carries none; both runners send their own, so this is what Scout
    actually read.
    """
    mod = {"fab": eb.wat_fab, "gb": eb.wgb}[dataset]
    return mod.SYSTEM_PROMPT


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
# Part C — all-but-the-top-k PCA nullification  (EXPLORATORY, Scout only)
# =========================================================

def max_valid_k(n_vectors, dim):
    """
    Largest number of principal directions that exist for n mean-centred rows.
    Centring costs one degree of freedom, so the centred matrix has rank at most
    n - 1; e.g. 10 stimulus vectors in 4096-D admit k <= 9, never k = 10.
    """
    return max(0, min(int(n_vectors) - 1, int(dim)))


def nullify(vectors, k, add_mean_back=False):
    """
    All-but-the-top-style removal of the top-k principal directions of the
    SUPPLIED stimulus-vector set (after Mu, Bhat & Viswanath 2018).

    k = 0 returns the input unchanged, exactly — the primary (unaltered) analysis.
    The PCA is fit ONLY on the vectors passed in; the caller records which ones
    those were in the manifest.

    This is Bai-INSPIRED exploratory PC nullification, not a reproduction of his
    `pca.py`: his re-adds the mean after subtracting the projections, which is a
    deviation from Mu et al. (cosine is scale- but not translation-invariant).
    That behaviour is available as the explicit compatibility flag
    `add_mean_back=True`, and is off by default.

    This is NOT debiasing: the removed directions are the highest-variance ones,
    defined without reference to the association being measured.
    """
    import numpy as np
    V = np.asarray(vectors, dtype=float)
    if k == 0:
        return V.copy()
    n_vectors, dim = V.shape
    kmax = max_valid_k(n_vectors, dim)
    if k > kmax:
        raise ValueError(
            f"k={k} exceeds the number of principal directions a mean-centred "
            f"stimulus set can have: {n_vectors} stimulus vectors of dimension "
            f"{dim} give a centred rank of at most min(n-1, dim) = {kmax}. "
            f"Requested k={k}; use k <= {kmax} (k is never silently truncated).")
    mean = V.mean(axis=0)
    centred = V - mean
    comps = top_components(centred, k)
    out = centred - (centred @ comps.T) @ comps
    return out + mean if add_mean_back else out


def top_components(centred, k):
    """
    Top-k principal directions of already-centred rows, via SVD.
    Equivalent to sklearn's PCA(n_components=k).components_ up to sign, and keeps
    this module dependency-free (sklearn is not installed in this environment).
    """
    import numpy as np
    _, _, vt = np.linalg.svd(np.asarray(centred, dtype=float), full_matrices=False)
    return vt[:k]


# =========================================================
# B3/B4 — per-layer association and layer-wise correlation
# =========================================================

def layer_association(x_vec, y_vec, A, B):
    """
    One layer, one target pair: (delta, bai_singleton_weat).
    `delta` is the unstandardised differential association — the PRIMARY measure,
    for the same reason as in the external analysis: with one X and one Y, the
    standardized score collapses toward +-2 and carries no magnitude.
    """
    import numpy as np
    X = np.asarray(x_vec, dtype=float)[None, :]
    Y = np.asarray(y_vec, dtype=float)[None, :]
    return (eb.weat_test_stat(X, Y, A, B), eb.bai_singleton_weat(X, Y, A, B))


def trial_layer_rows(vectors, arm, x, y, behavioural_D, *, n_layers, ks=(0,),
                     pooling="mean", base=None):
    """
    One trial -> ONE ROW PER (hidden state, k).  `vectors` maps each stimulus
    phrase to a list of per-layer vectors, as returned by
    `extract_prompt_hidden_states`; `n_layers` is that model's actual hidden-state
    count, never a hardcoded 33.
    """
    phrases = [x, y] + list(arm["A"]) + list(arm["B"])
    rows = []
    for layer in range(n_layers):
        stim = [vectors[p][layer] for p in phrases]
        for k in ks:
            lookup = dict(zip(phrases, nullify(stim, k)))
            delta, singleton = layer_association(
                lookup[x], lookup[y],
                [lookup[p] for p in arm["A"]], [lookup[p] for p in arm["B"]])
            row = dict(base or {})
            row.update(X_stimulus=x, Y_stimulus=y, pair=f"{x} || {y}",
                       layer=layer, pca_k=k, pooling=pooling,
                       delta=delta, bai_singleton_weat=singleton,
                       behavioural_D=behavioural_D,
                       analysis_version=ANALYSIS_VERSION)
            rows.append(row)
    return rows


def aggregate_pairs(trial_rows):
    """
    Collapse repeated sessions to unique (dataset, arm, model, layer, pair):
    `delta_mean` (the statistic used downstream), its sample SD and its range as
    separate columns, the mean behavioural D, and the session count.
    Same anti-pseudoreplication rule as the external pair-level analysis.
    """
    groups = defaultdict(list)
    for r in trial_rows:
        groups[(r["dataset"], r["arm"], r["behavioural_model"], r["layer"],
                r["pair"])].append(r)
    out = []
    for (ds, arm, model, layer, pair), rs in sorted(groups.items(), key=str):
        deltas = [r["delta"] for r in rs]
        n = len(deltas)
        mean = sum(deltas) / n
        # sample SD (ddof=1); undefined for a single session, never a range
        sd = (math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
              if n >= 2 else float("nan"))
        out.append(dict(
            dataset=ds, arm=arm, behavioural_model=model, layer=layer, pair=pair,
            X_stimulus=rs[0]["X_stimulus"], Y_stimulus=rs[0]["Y_stimulus"],
            delta_mean=mean,                 # the value correlated downstream
            delta=mean,                      # alias kept for readers of older files
            delta_sd=sd,                     # sample SD, ddof=1, NaN if n == 1
            delta_range=max(deltas) - min(deltas),
            bai_singleton_weat=sum(r["bai_singleton_weat"] for r in rs) / n,
            mean_behavioural_D=sum(r["behavioural_D"] for r in rs) / n,
            n_sessions=n, analysis_version=ANALYSIS_VERSION))
    return out


def layer_correlations(agg_rows, *, checkpoint, pooling, k=0, context="original prompt"):
    """
    Per (dataset, arm, behavioural model, layer): Spearman (primary) and Pearson
    across UNIQUE PAIRS.  The full profile is returned; no layer is selected.
    """
    by = defaultdict(list)
    for r in agg_rows:
        by[(r["dataset"], r["arm"], r["behavioural_model"], r["layer"])].append(r)
    rows = []
    for (ds, arm, model, layer), rs in sorted(by.items(), key=str):
        rec = eb.corr_record(
            "scout_hidden_delta_vs_behavioural_D", ds, arm, model,
            [r["delta_mean"] for r in rs], [r["mean_behavioural_D"] for r in rs],
            emb_model=checkpoint, context=context, n_label="n_unique_pairs",
            pooling=pooling,
            n_sessions=sum(r["n_sessions"] for r in rs))
        rec.update(layer=layer, pca_k=k)
        rows.append(rec)
    return rows


# =========================================================
# Preflight — is a hidden-state backend actually available?
# =========================================================

def _ollama_models():
    base = (os.getenv("OLLAMA_BASE_URL", "").strip()
            or "http://localhost:11434/v1").rsplit("/v1", 1)[0]
    try:
        import urllib.request
        with urllib.request.urlopen(base + "/api/tags", timeout=5) as r:
            return base, [m["name"] for m in json.load(r).get("models", [])]
    except Exception as exc:
        return base, f"unreachable ({exc.__class__.__name__})"


def preflight(checkpoint=None):
    """What exists, what does not, and whether the forward pass may run."""
    info = {"scout_behavioural_sessions": {}, "blockers": []}

    for ds, spec in eb.DATASETS.items():
        counts = defaultdict(int)
        for r in eb.load_sessions(spec):
            if r.get("model") in SCOUT_BEHAVIOURAL_MODELS:
                counts[(r.get("provider"), r.get("model"))] += 1
        info["scout_behavioural_sessions"][ds] = {f"{p}/{m}": n
                                                  for (p, m), n in sorted(counts.items())}

    base, models = _ollama_models()
    info["ollama_endpoint"] = base
    info["ollama_models"] = models
    info["ollama_exposes_hidden_states"] = False   # chat + /api/embed only, by design

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_devices"] = [
            {"name": torch.cuda.get_device_name(i),
             "total_GiB": round(torch.cuda.get_device_properties(i).total_memory
                                / 1024 ** 3, 1)}
            for i in range(torch.cuda.device_count())]
        info["total_VRAM_GiB"] = round(sum(d["total_GiB"]
                                           for d in info["cuda_devices"]), 1)
    except Exception:
        info["torch"] = None
        info["blockers"].append("torch is not installed in this environment")

    try:
        import transformers
        from transformers.models.auto import modeling_auto  # noqa: F401
        info["transformers"] = transformers.__version__
        try:
            from transformers import Llama4ForConditionalGeneration  # noqa: F401
            info["llama4_supported"] = True
        except ImportError:
            info["llama4_supported"] = False
            info["blockers"].append(
                f"transformers {transformers.__version__} has no Llama-4 support; "
                f"4.51+ is required for Llama4ForConditionalGeneration")
    except ImportError:
        info["transformers"] = None
        info["blockers"].append("transformers is not installed in this environment")

    try:
        import bitsandbytes  # noqa: F401
        info["bitsandbytes"] = True
    except ImportError:
        info["bitsandbytes"] = False     # only a blocker for a quantised load

    # how much work a full run is, under the reported scope
    work = {}
    for ds, spec in eb.DATASETS.items():
        arms = eb.scoped(ds, spec["arms"]())
        prompts, trials = set(), 0
        for r in eb.load_sessions(spec):
            if r.get("model") not in SCOUT_BEHAVIOURAL_MODELS:
                continue
            for _ in spec["session_arms"](r, arms):
                trials += 1
                prompts.add(r["prompt"])
        work[ds] = {"in_scope_trials": trials, "unique_prompts": len(prompts),
                    "arms": sorted(arms)}
    info["work_in_scope"] = work

    info["hf_checkpoint"] = checkpoint
    if checkpoint is None:
        info["blockers"].append(
            "no --hf-checkpoint given; the behavioural Scout runs used Ollama, "
            "whose API returns text only (and /api/embed returns a single "
            "final-layer vector, not per-layer hidden states)")
    elif not SCOUT_CHECKPOINT_PATTERN.search(str(checkpoint)):
        info["blockers"].append(
            f"checkpoint {checkpoint!r} does not look like Llama-4-Scout; this "
            f"module will not label another model's internals as Scout's")
    elif not os.path.exists(str(checkpoint)):
        info["blockers"].append(f"checkpoint path {checkpoint!r} does not exist")

    info["can_run_forward_pass"] = not info["blockers"]
    return info


# =========================================================
# B1 — forward pass over the ORIGINAL saved prompts
# =========================================================

def execution_device(model):
    """
    Where the input tensors must live.

    `model.device` is the device of the first parameter, which for an offloaded
    model is `meta` or `cpu` -- neither of which is where the forward actually
    runs.  accelerate records the real answer in `hf_device_map`: the first
    accelerator in it is the execution device that offloaded modules are copied
    to for each forward pass.
    """
    import torch
    for dev in (getattr(model, "hf_device_map", None) or {}).values():
        if str(dev) not in ("cpu", "disk", "meta"):
            return torch.device(dev if not isinstance(dev, int) else f"cuda:{dev}")
    dev = getattr(model, "device", None)
    if dev is None or str(dev) == "meta":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev


def extract_prompt_hidden_states(prompt, phrases, tokenizer, model,
                                 pooling="mean", windows=None,
                                 default_window=None, competitors=()):
    """
    One forward pass over the EXACT input the model saw; returns
    {phrase: [vector per hidden state]} plus architecture metadata.

    `prompt` is the templated text from `build_chat_input`, and
    `default_window` is its user-message span: every phrase is looked up inside
    that window unless `windows` gives it a narrower one.

    The generated response is never embedded.  The number of hidden states comes
    from the model (len(outputs.hidden_states)); nothing is hardcoded.  Index 0 is
    the embedding/input representation, 1..n the transformer block outputs.

    Only reachable once `preflight()` reports a hidden-state-capable backend; this
    environment has none, so this function is unexercised here by design.
    """
    import torch
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    enc = {k: v.to(execution_device(model)) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden = [h[0].to(dtype=torch.float32).cpu().numpy() for h in out.hidden_states]

    vectors = {}
    for phrase in phrases:
        span = resolve_span(prompt, offsets, phrase,
                            window=(windows or {}).get(phrase, default_window),
                            competitors=competitors)
        vectors[phrase] = [pool_span(layer, span, pooling) for layer in hidden]
    meta = dict(n_hidden_states=len(hidden),
                layer_0="embedding/input representation",
                layers_1_to_n="transformer block outputs",
                n_transformer_blocks=len(hidden) - 1,
                hidden_size=int(hidden[0].shape[-1]),
                n_prompt_tokens=int(hidden[0].shape[0]),
                pooling=pooling)
    return vectors, meta


# =========================================================
# Backend — load Scout so that it returns hidden states
# =========================================================

QUANTISATIONS = ("bf16", "4bit", "8bit")

GIB = float(1 << 30)


def _auto_model_classes():
    """
    The candidate auto-classes, in the order this module tries them.

    Llama-4 ships as a multimodal wrapper (`Llama4ForConditionalGeneration`), so
    the causal-LM mapping can refuse it; text-only input still returns the
    language model's hidden states through the image-text class.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return (AutoModelForCausalLM, AutoModelForImageTextToText)


def parameter_footprint(checkpoint, quant="bf16", bytes_per_param=2.0):
    """
    Measure what `quant` would actually shrink, on meta tensors: no weights are
    read, so this is cheap enough to run before every load.

    bitsandbytes replaces `nn.Linear` MODULES.  A weight held directly on a
    module as an `nn.Parameter` is therefore left at full precision -- and
    Llama-4's mixture-of-experts stacks (`Llama4TextExperts.gate_up_proj`,
    `.down_proj`) are exactly that: 3-D `nn.Parameter`s, not `nn.Linear`.  On
    Scout that is 96.6 of 108.6 billion parameters, so `load_in_4bit` shrinks
    only a ninth of the model.

    Returns a dict of parameter counts and the resulting GiB, so a caller can
    say WHY a load does or does not fit rather than asserting it.
    """
    import torch.nn as nn
    from accelerate import init_empty_weights
    from transformers import AutoConfig, PretrainedConfig

    # A config may be passed directly, which lets the measurement be exercised
    # on a small architecture instead of a 200 GiB checkpoint.
    cfg = (checkpoint if isinstance(checkpoint, PretrainedConfig)
           else AutoConfig.from_pretrained(checkpoint))
    model, errors = None, []
    with init_empty_weights():
        for cls in _auto_model_classes():
            try:
                model = cls.from_config(cfg)
                break
            except (ValueError, KeyError) as exc:
                errors.append(f"{cls.__name__}: {exc.__class__.__name__}")
    if model is None:
        raise RuntimeError("no auto-class accepted this config: "
                           + "; ".join(errors))

    quantisable = unquantisable = 0
    seen = set()
    for mod_name, mod in model.named_modules():
        is_linear = isinstance(mod, nn.Linear)
        for par_name, par in mod.named_parameters(recurse=False):
            full = f"{mod_name}.{par_name}"
            if full in seen:
                continue
            seen.add(full)
            if is_linear and par_name == "weight":
                quantisable += par.numel()
            else:
                unquantisable += par.numel()

    per_param = {"4bit": 0.5, "8bit": 1.0, "bf16": bytes_per_param}[quant]
    quantised_bytes = quantisable * (per_param if quant != "bf16"
                                     else bytes_per_param)
    total_gib = (quantised_bytes + unquantisable * bytes_per_param) / GIB
    return dict(
        quant=quant,
        total_params=quantisable + unquantisable,
        params_in_linear=quantisable,
        params_outside_linear=unquantisable,
        full_precision_gib=(quantisable + unquantisable) * bytes_per_param / GIB,
        footprint_gib=total_gib,
        shrinkable_fraction=quantisable / max(1, quantisable + unquantisable))


def _gpu_budget(reserve_gib=4.0):
    """Free VRAM per visible device, less a per-device reserve for activations."""
    import torch
    if not torch.cuda.is_available():
        return {}
    budget = {}
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        budget[i] = max(0.0, free / GIB - reserve_gib)
    return budget


def _cpu_budget(reserve_gib=8.0):
    """MemAvailable, less a reserve for the loading process and the OS."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return max(0.0, int(line.split()[1]) / (1 << 20) - reserve_gib)
    except OSError:
        pass
    return 0.0


def load_backend(checkpoint, quant="bf16", device_map="auto",
                 offload_folder=None, gpu_reserve_gib=4.0, cpu_reserve_gib=8.0):
    """
    Load the checkpoint for PREFILL ONLY (no generation) with hidden states on.

    The default is bf16 -- the checkpoint's own dtype -- spilled across GPUs,
    host RAM and, if needed, `offload_folder`, because 4-bit does NOT work on
    this architecture: see `parameter_footprint`.  bitsandbytes quantises
    `nn.Linear` only, and Scout keeps 89% of its parameters in `nn.Parameter`
    expert stacks, so `load_in_4bit` leaves ~180 GiB untouched and then refuses
    the CPU dispatch it has itself made unavoidable.  Loading at bf16 also
    removes a confound: internals and behaviour then differ in SERVING STACK
    (Ollama Q4_K_M generated the sessions) but not in the weights' precision on
    the internals side, and the manifest records both.

    Returns (tokenizer, model, meta).
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if quant not in QUANTISATIONS:
        raise ValueError(f"unknown quantisation {quant!r}; expected one of "
                         f"{QUANTISATIONS}")

    tok = AutoTokenizer.from_pretrained(checkpoint)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("a FAST tokenizer is required: span location uses "
                           "offset mappings, which the slow tokenizer lacks")
    if getattr(tok, "chat_template", None) in (None, ""):
        raise RuntimeError("the tokenizer carries no chat template, so the "
                           "behavioural input cannot be reconstructed")

    gpu = _gpu_budget(gpu_reserve_gib)
    cpu = _cpu_budget(cpu_reserve_gib)
    fp = parameter_footprint(checkpoint, quant=quant)
    gpu_total = sum(gpu.values())

    print(f"  footprint at {quant}: {fp['footprint_gib']:.1f} GiB "
          f"({fp['total_params']/1e9:.1f}B params, "
          f"{fp['shrinkable_fraction']*100:.0f}% of them inside nn.Linear)")
    print(f"  budget: {gpu_total:.1f} GiB across {len(gpu)} GPU(s) "
          f"+ {cpu:.1f} GiB host RAM"
          + (f" + {offload_folder}" if offload_folder else " (no disk offload)"))

    # bitsandbytes cannot place a quantised module on CPU or disk, so a 4/8-bit
    # load must fit in VRAM outright.  Say so with the measured numbers instead
    # of letting from_pretrained raise its generic dispatch error.
    if quant in ("4bit", "8bit") and fp["footprint_gib"] > gpu_total:
        raise SystemExit(
            f"\n{quant} CANNOT LOAD THIS CHECKPOINT\n"
            f"  bitsandbytes converts nn.Linear modules only.  "
            f"{fp['params_outside_linear']/1e9:.1f}B of this model's "
            f"{fp['total_params']/1e9:.1f}B parameters sit outside nn.Linear "
            f"(Llama-4 holds its MoE experts as 3-D nn.Parameter stacks), so "
            f"they stay at full precision.\n"
            f"  {quant} footprint {fp['footprint_gib']:.1f} GiB > "
            f"{gpu_total:.1f} GiB of free VRAM, and a quantised module cannot "
            f"be offloaded to CPU or disk.\n"
            f"  Use --quant bf16, which may be offloaded "
            f"({fp['full_precision_gib']:.1f} GiB across VRAM + RAM + disk).")

    kwargs = dict(device_map=device_map, dtype=torch.bfloat16,
                  low_cpu_mem_usage=True)
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quant == "4bit"), load_in_8bit=(quant == "8bit"),
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        # An explicit budget: accelerate's default leaves most of host RAM
        # unused, which on this model is the difference between spilling a few
        # GiB to disk and spilling tens of GiB to it.
        max_memory = {i: f"{g:.1f}GiB" for i, g in gpu.items()}
        if cpu > 0:
            max_memory["cpu"] = f"{cpu:.1f}GiB"
        if max_memory:
            kwargs["max_memory"] = max_memory
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)
            kwargs["offload_folder"] = offload_folder
            kwargs["offload_state_dict"] = True

    cfg = AutoConfig.from_pretrained(checkpoint)
    arch = (getattr(cfg, "architectures", None) or [""])[0]

    def _load(cls):
        try:
            return cls.from_pretrained(checkpoint, **kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            bc = dict(kwargs)                      # transformers < 4.56
            bc["torch_dtype"] = bc.pop("dtype")
            return cls.from_pretrained(checkpoint, **bc)

    model, loaded_with, errors = None, None, []
    for cls in _auto_model_classes():
        try:
            model = _load(cls)
            loaded_with = cls.__name__
            break
        except (ValueError, KeyError) as exc:
            errors.append(f"{cls.__name__} refused ({exc.__class__.__name__}: {exc})")
            print(f"  {errors[-1].splitlines()[0]}")
    if model is None:
        raise RuntimeError("no auto-class could load this checkpoint:\n  "
                           + "\n  ".join(errors))
    model.eval()

    placement = {}
    for dev in (getattr(model, "hf_device_map", None) or {}).values():
        placement[str(dev)] = placement.get(str(dev), 0) + 1
    if placement:
        print(f"  placement: {placement}")

    import transformers
    meta = dict(checkpoint=str(checkpoint), quantisation=quant,
                config_architecture=arch, loaded_with=loaded_with,
                transformers=transformers.__version__, torch=torch.__version__,
                device_map=device_map, module_placement=placement,
                footprint=fp,
                quantisation_note=("behaviour came from the Q4_K_M GGUF served "
                                   "by Ollama; these internals come from a "
                                   f"{quant} load of the same checkpoint")
                if quant != "bf16" else
                ("unquantised (checkpoint dtype); behaviour came from the "
                 "Q4_K_M GGUF served by Ollama, so the two differ in serving "
                 "stack, not in the precision of these internals"))
    return tok, model, meta


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=list(eb.DATASETS))
    ap.add_argument("--hf-checkpoint", default=None,
                    help="local path to Llama-4-Scout weights exposing hidden states")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--pca-k", action="append", type=int,
                    help="repeatable; 0 = unaltered (always included, primary). "
                         "k is bounded by the centred rank of the stimulus set, "
                         "min(n_vectors - 1, hidden_dim), and never truncated")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--quant", default="bf16", choices=list(QUANTISATIONS),
                    help="bf16 (default; the checkpoint's own dtype, offloaded "
                         "across VRAM + RAM + disk). 4bit/8bit shrink nn.Linear "
                         "ONLY, which on a Llama-4 MoE is ~11%% of the weights, "
                         "and cannot be offloaded; see parameter_footprint()")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--offload-folder", default=None,
                    help="directory for weights that fit in neither VRAM nor "
                         "RAM; put it on the FASTEST disk available, since it "
                         "is re-read on every forward pass")
    ap.add_argument("--gpu-reserve-gib", type=float, default=4.0,
                    help="VRAM held back per GPU for activations (default 4)")
    ap.add_argument("--cpu-reserve-gib", type=float, default=8.0,
                    help="host RAM held back for the process and OS (default 8)")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE MODE: process at most N sessions per dataset, "
                         "write nothing unless --force-write-smoke")
    ap.add_argument("--force-write-smoke", action="store_true",
                    help="allow a --limit run to write output files")
    ap.add_argument("--all-arms", action="store_true",
                    help="run every defined arm, not just the reported scope")
    ap.add_argument("--preflight", action="store_true",
                    help="report backend availability and exit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    info = preflight(args.hf_checkpoint)
    print(json.dumps(info, indent=2))

    if not info["can_run_forward_pass"]:
        print()
        print("=" * 78)
        print("MODEL-INTERNAL ANALYSIS NOT POSSIBLE WITH THE CURRENT SCOUT INTERFACE")
        print("=" * 78)
        for b in info["blockers"]:
            print(f"  - {b}")
        print()
        print("The Scout behavioural sessions were collected through Ollama, which\n"
              "returns generated text only; it has no per-layer hidden-state output.\n"
              "This module will not substitute an API embedding or a different model\n"
              "for Scout's internal representations, so it stops here.\n"
              "To proceed, serve Llama-4-Scout through an interface that returns\n"
              "output_hidden_states=True and pass --hf-checkpoint.")
        raise SystemExit(2)

    if args.preflight:
        return

    # --- forward-pass path (requires a hidden-state backend; see docstring) ---
    tok, model, backend = load_backend(args.hf_checkpoint, quant=args.quant,
                                       device_map=args.device_map,
                                       offload_folder=args.offload_folder,
                                       gpu_reserve_gib=args.gpu_reserve_gib,
                                       cpu_reserve_gib=args.cpu_reserve_gib)
    print(f"  loaded {backend['loaded_with']} ({backend['quantisation']}), "
          f"transformers {backend['transformers']}")

    ks = sorted(set([0] + list(args.pca_k or [])))
    trial_rows, null_rows, arch = [], [], None
    systems, n_span_errors = {}, 0
    os.makedirs(args.out, exist_ok=True)

    for name in (args.dataset or list(eb.DATASETS)):
        spec = eb.DATASETS[name]
        all_arms = spec["arms"]()
        arms = all_arms if args.all_arms else eb.scoped(name, all_arms)
        sessions = [r for r in eb.load_sessions(spec)
                    if r.get("model") in SCOUT_BEHAVIOURAL_MODELS]
        if args.limit:
            sessions = sessions[:args.limit]
        system = system_prompt_for(name)
        systems[name] = system
        print(f"[{name}] {len(sessions)} Scout sessions, arms: {sorted(arms)}"
              + (f"  [SMOKE: limited to {args.limit}]" if args.limit else ""))
        for i, r in enumerate(sessions, 1):
            # the EXACT input the model saw: runner system message + user prompt,
            # through the model's own chat template
            prompt, user_window = build_chat_input(tok, system, r["prompt"])
            x, y = spec["session_X"](r), spec["session_Y"](r)
            for arm_name in spec["session_arms"](r, arms):
                arm = arms[arm_name]
                d = eb.session_D(r, spec, arm)
                if d is None:
                    continue
                phrases = [x, y] + list(arm["A"]) + list(arm["B"])
                try:
                    vecs, arch = extract_prompt_hidden_states(
                        prompt, phrases, tok, model, pooling=args.pooling,
                        default_window=user_window,
                        # every stimulus THIS prompt lists -- sorted items AND
                        # the two target words -- so that a stimulus which is a
                        # substring of another ('history' inside 'art history',
                        # 'male' inside 'female', 'man' inside 'woman') is not
                        # located inside it
                        competitors=set(eb._assigned_items(r)) | {x, y})
                except SpanError as exc:
                    n_span_errors += 1
                    raise SpanError(
                        f"[{name} trial {r.get('trial')} arm {arm_name}] {exc}"
                    ) from exc
                if i == 1:
                    print(f"  first trial: {arch['n_prompt_tokens']} tokens, "
                          f"{arch['n_hidden_states']} hidden states, "
                          f"hidden_size {arch['hidden_size']}")
                for row in trial_layer_rows(
                        vecs, arm, x, y, d, n_layers=arch["n_hidden_states"],
                        ks=ks, pooling=args.pooling,
                        base=dict(dataset=name, arm=arm_name,
                                  behavioural_model=r.get("model"),
                                  trial=r.get("trial"),
                                  session_id=f"{os.path.basename(r['_file'])}"
                                             f"#{r.get('trial')}",
                                  checkpoint=args.hf_checkpoint)):
                    (trial_rows if row["pca_k"] == 0 else null_rows).append(row)
            if i % 25 == 0:
                print(f"  [{name}] {i}/{len(sessions)} sessions")

    agg = aggregate_pairs(trial_rows)
    corr = layer_correlations(agg, checkpoint=args.hf_checkpoint,
                              pooling=args.pooling, k=0)
    for k in [k for k in ks if k]:
        corr += layer_correlations(aggregate_pairs([r for r in null_rows
                                                    if r["pca_k"] == k]),
                                   checkpoint=args.hf_checkpoint,
                                   pooling=args.pooling, k=k)

    if args.limit and not args.force_write_smoke:
        print(f"\nSMOKE RUN (--limit {args.limit}) — nothing written. "
              f"{len(trial_rows)} trial-layer rows, {len(agg)} aggregated, "
              f"{len(corr)} correlations would be produced. "
              f"Pass --force-write-smoke to write them anyway.")
        return

    def _write(fname, rows):
        path = os.path.join(args.out, fname)
        if os.path.exists(path) and not args.force:
            raise SystemExit(f"refusing to overwrite {path} — rerun with --force")
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {path} ({len(rows)} rows)")

    _write("hidden_trial_layer.csv", trial_rows)
    _write("hidden_pair_layer_aggregated.csv", agg)
    _write("hidden_layer_correlations.csv", corr)
    _write("hidden_nullified_layer.csv", null_rows)
    with open(os.path.join(args.out, "hidden_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(
            analysis="model-internal hidden states (Scout only)",
            analysis_version=ANALYSIS_VERSION,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=args.hf_checkpoint, architecture=arch,
            backend=backend,
            model_input=dict(
                source="the saved behavioural prompt, re-templated",
                chat_template="tokenizer.apply_chat_template(system + user, "
                              "add_generation_prompt=True)",
                system_prompt=systems,
                span_lookup="restricted to the user-message window inside the "
                            "templated text",
                response_embedded=False),
            scope=("all defined arms (--all-arms)" if args.all_arms
                   else "reported scope (see embed_bias.IN_SCOPE)"),
            limit=args.limit,
            token_pooling=args.pooling, pca_k_values=ks,
            pca_method="Bai-inspired exploratory all-but-the-top-style PC "
                       "removal (Mu, Bhat & Viswanath 2018); NOT a reproduction "
                       "of Bai's pca.py, which re-adds the mean "
                       "(add_mean_back=True is the optional compatibility mode). "
                       "Nullification is not debiasing.",
            pca_fit_on="the stimulus vectors of that trial and layer: the two "
                       "target phrases plus every attribute phrase of the arm",
            pca_max_k="min(n_stimulus_vectors - 1, hidden_dim) — the centred rank",
            primary="pca_k=0 (unaltered)",
            statistics=dict(primary="spearman", secondary="pearson",
                            level="unique target pairs per layer"),
            datasets=args.dataset or list(eb.DATASETS),
            counts=dict(trial_layer=len(trial_rows), aggregated=len(agg),
                        correlations=len(corr), nullified=len(null_rows)),
        ), f, indent=2)


if __name__ == "__main__":
    main()
