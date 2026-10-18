"""
OpenAI backend — PLACEHOLDER.

Reserved for running the paired experiment on an OpenAI model.  Nothing is
implemented yet, and in particular no hidden-state method is: the OpenAI API
does not return per-layer hidden states, and how (or whether) a representation
measure is obtained for this backend is still to be decided.

Selecting it fails immediately with NotImplementedError
(`embedding.backends.get_backend("openai")`).  This module does not import the
OpenAI SDK, so importing `embedding` or this package never requires it.

Not to be confused with `embedding/external_embedding.py`, the independent
text-embedding reference analysis.
"""

NAME = "openai"
IMPLEMENTED = False
NOT_IMPLEMENTED_MESSAGE = (
    "the 'openai' backend is a placeholder: no generation or representation "
    "method is implemented yet. Use --model llama31 or --model scout.")


def load(checkpoint=None, gpu=0):
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
