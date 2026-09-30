"""Measure 1 (implicit WAT) — Bai 2025 GENDER replication (career/power/science/sexuality).

Per-dataset boolean flags select which to run; none given = all gender datasets.
    python bai_gender_wat.py --provider openai              # all four
    python bai_gender_wat.py --provider openai --science    # science only
    python bai_gender_wat.py --provider openai --career --power
Outputs: replicate/responses/<provider>_<model>_<dataset>_wat_replicate.xlsx
See bai_core.py (logic) and bai_specs.py (verbatim stimuli keys)."""
from bai_core import run_category
if __name__ == "__main__":
    run_category("gender", "wat")
