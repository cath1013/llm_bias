"""Measure 2 (decision) — Bai 2025 RACE replication
(racism/weapon/guilt/skintone/arab/asian/black/hispanic/english).

Two group modes (both verbatim from Bai):
  - drawn groups + 2-task scenario: racism, weapon, guilt, skintone, english
  - fixed 'White vs X candidate' job-fair prompt: arab, asian, black, hispanic
Per-dataset boolean flags select which to run; none given = all race datasets.
    python bai_race_decision.py --provider openai              # all nine
    python bai_race_decision.py --provider openai --weapon
    python bai_race_decision.py --provider anthropic --judge-provider openai
Outputs: replicate/responses/<provider>_<model>_<dataset>_decision_replicate.xlsx"""
from bai_core import run_category
if __name__ == "__main__":
    run_category("race", "decision")
