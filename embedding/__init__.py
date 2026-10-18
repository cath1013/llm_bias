"""
Embedding / hidden-state experiments.

    embedding.design              frozen WAT and chained designs (stimuli, schedules)
    embedding.core                model-independent scoring, spans, statistics
    embedding.backends            one module per model; selected by name
    embedding.wat_hidden          fixed-prompt WAT runner (model-agnostic)
    embedding.chained_hidden      chained Task 1 -> 2 -> 3 runner (model-agnostic)
    embedding.external_embedding  OpenAI text-embedding reference analysis

Run the entry points as modules from the project root, e.g.

    python -m embedding.wat_hidden --model scout --dry-run

The behavioural runners these modules read their stimuli and wording from
(`wat_fab.py`, `wat_gender_brilliance.py`, `decision_text.py`) live at the
project root and are not a package, so the root is put on sys.path here, once,
for every submodule.
"""

import os as _os
import sys as _sys

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
