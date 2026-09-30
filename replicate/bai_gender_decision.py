"""Measure 2 (decision) — Bai 2025 GENDER replication (career/power/science/sexuality).

Per-dataset boolean flags select which to run; none given = all gender datasets.
    python bai_gender_decision.py --provider openai                 # all four
    python bai_gender_decision.py --provider openai --science
    python bai_gender_decision.py --provider anthropic --judge-provider openai
Task 1 profiles are unscored; Task 2 recommendation coded by a fixed judge
(--judge-provider/--judge-model, default openai) -> mean congruent rate vs 0.5.
Outputs: replicate/responses/<provider>_<model>_<dataset>_decision_replicate.xlsx
See bai_core.py (logic) and bai_specs.py (verbatim per-dataset prompts + judge)."""
from bai_core import run_category
if __name__ == "__main__":
    run_category("gender", "decision")
