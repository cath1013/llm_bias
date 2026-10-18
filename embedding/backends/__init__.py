"""
Model backends for the paired behaviour + hidden-state experiments.

One module per model.  A runner selects ONE by name, once, and afterwards uses
only this interface:

    module constants
        NAME                       registry key ("llama31", "scout")
        CHECKPOINT                 default checkpoint path
        WAT_ANALYSIS_VERSION       stamped on WAT-only outputs
        WAT_OUT_DIR                WAT-only output directory
        WAT_MANIFEST_ANALYSIS      the manifest's one-line description
        CHAINED_ANALYSIS_VERSION   stamped on chained outputs
        CHAINED_OUT_DIR            chained output directory
    load(checkpoint=None, gpu=0) -> Backend
    finalize_wat_manifest(manifest, *, backend_meta, n_hidden_states, bos) -> manifest

    Backend
        .name, .model, .tokenizer, .meta
        .generate(templated, *, seed, generation=None)        -> (text, gen_meta)
        .extract_hidden(templated, phrases, *, user_window,
                        competitors, context="")               -> (vectors, meta)
        .describe()                 -> [str]   model-specific smoke-test lines
        .memory_report(peak=False)  -> [str]

Backends import `embedding.core` and never each other.  Importing this package
or a backend module loads no weights and no transformers; `load()` does.
"""

import importlib

BACKENDS = {
    "llama31": "embedding.backends.llama_31",
    "scout": "embedding.backends.llama_scout",
    "openai": "embedding.backends.openai",
}


def names():
    """Every registered backend name (CLI choices)."""
    return list(BACKENDS)


def get_backend(name):
    """The backend module for `name`; an unimplemented one fails here, clearly."""
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r} (have {names()})")
    module = importlib.import_module(BACKENDS[name])
    if not getattr(module, "IMPLEMENTED", True):
        raise NotImplementedError(module.NOT_IMPLEMENTED_MESSAGE)
    return module


def implemented():
    """Names of the backends that can actually run."""
    return [n for n in BACKENDS
            if getattr(importlib.import_module(BACKENDS[n]), "IMPLEMENTED", True)]


def wat_out_dirs():
    """Every implemented backend's WAT-only result directory (never written by
    the chained runner)."""
    return tuple(get_backend(n).WAT_OUT_DIR for n in implemented())
