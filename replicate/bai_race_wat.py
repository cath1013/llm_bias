"""Measure 1 (implicit WAT) — Bai 2025 RACE replication
(racism/weapon/guilt/skintone/arab/asian/black/hispanic/english).

Per-dataset boolean flags select which to run; none given = all race datasets.
    python bai_race_wat.py --provider openai              # all nine
    python bai_race_wat.py --provider openai --weapon --racism
Note: --arab maps to the CSV 'arab/muslim' stimuli row (see bai_specs.CSV_KEY_OVERRIDE).
Outputs: replicate/responses/<provider>_<model>_<dataset>_wat_replicate.xlsx"""
from bai_core import run_category
if __name__ == "__main__":
    run_category("race", "wat")
