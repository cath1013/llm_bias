"""
FIXED-PROMPT BAI-STYLE WAT — paired behaviour and hidden states, any backend.

The model-agnostic runner for the WAT-only experiment: the frozen 272 prompts
(`embedding/prompts.csv`), one generate() and one hidden-state forward pass per
prompt through the backend selected with --model, the strict behavioural parser
and unsmoothed Bai score, delta / singleton WEAT per hidden state, and the
secondary set-level WEAT over contextual prototypes.

This is the orchestration that was duplicated in `embed_bias_llama3_hidden.py`
and `embed_bias_scout_hidden.py`, moved here verbatim except that the model is
reached through the backend interface (see `embedding/backends/__init__.py`).
The backend is chosen ONCE; nothing below branches on the model.

OUTPUT  (the backend's WAT_OUT_DIR: responses/embed_llama31/ or
responses/embed_scout/; schemas unchanged)
    prompt_response_hidden.csv  behaviour_parse_failures.csv
    secondary_set_level_weat.csv  secondary_stimulus_layer.csv
    secondary_stimulus_prototypes.npz  manifest.json

LAYER CONVENTION
    layer0 = hidden_states[0], the input embedding; layer1..layerN = block
    outputs.  N comes from the model (Llama 3.1: 32, Scout: 48).

USAGE
    python -m embedding.wat_hidden --model scout --dry-run
    python -m embedding.wat_hidden --model scout --smoke-test
    python -m embedding.wat_hidden --model scout --limit 5 --out-dir responses/embed_scout_diag5
    python -m embedding.wat_hidden --model llama31 --checkpoint <path> --gpu 0
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone

from embedding import backends
from embedding.design import N_TOTAL, STIMULI_CSV, PROMPTS_CSV, load_design, design_summary
from embedding.core import (SYSTEM_PROMPT, GENERATION, SEED_BASE, SEED_RULE, POOLING,
                            arm_for, prompt_roles, parse_behaviour,
                            behaviour_wat_score, prompt_seed, build_chat_input,
                            representation_scores, wide_row, write_primary_csv,
                            PrototypeAccumulator, accumulate_prompt, set_level_rows,
                            stimulus_layer_rows, bos_record, validate_all_spans,
                            write_csv)

PRIMARY_CSV = "prompt_response_hidden.csv"

OUTPUT_FILES = [
    PRIMARY_CSV,
    "behaviour_parse_failures.csv",
    "secondary_set_level_weat.csv",
    "secondary_stimulus_layer.csv",
    "secondary_stimulus_prototypes.npz",
    "manifest.json",
]



# =========================================================
# Driver
# =========================================================

def run_prompts(prompts, stim, backend, *, args, acc=None, verbose=True):
    """
    The whole experiment: one generate() and one forward() per frozen prompt,
    through whichever backend was selected.

    Returns (rows, parse_failures, meta).  Anything that is not a behavioural
    parse failure raises out of here, by design.
    """
    tokenizer = backend.tokenizer
    rows, failures = [], []
    n_hidden = None
    t_start = time.time()
    for k, record in enumerate(prompts, start=1):
        roles = prompt_roles(record, stim)
        templated, window = build_chat_input(tokenizer, SYSTEM_PROMPT,
                                             record["prompt"])
        context = record["prompt_id"]

        response, gen_meta = backend.generate(
            templated,
            seed=prompt_seed(record["prompt_id"], args.seed),
            generation=dict(GENERATION, max_new_tokens=args.max_new_tokens))

        vectors, meta = backend.extract_hidden(
            templated, roles["phrases"],
            user_window=window, competitors=roles["phrases"], context=context)
        n_hidden = meta["n_hidden_states"]

        assign, problems = parse_behaviour(response, record, roles)
        if problems:
            score = float("nan")
            failures.append(dict(prompt_id=record["prompt_id"],
                                 domain=record["domain"],
                                 category=record["category"],
                                 trial=record["trial"],
                                 n_items_assigned=len(assign),
                                 n_items_presented=len(roles["attributes"]),
                                 problems=" | ".join(problems),
                                 response=response))
        else:
            score, denom_problem = behaviour_wat_score(assign, record, roles)
            if denom_problem:
                failures.append(dict(prompt_id=record["prompt_id"],
                                     domain=record["domain"],
                                     category=record["category"],
                                     trial=record["trial"],
                                     n_items_assigned=len(assign),
                                     n_items_presented=len(roles["attributes"]),
                                     problems=denom_problem,
                                     response=response))

        deltas, weats = representation_scores(vectors, roles, n_hidden)
        rows.append(wide_row(record, response, score, deltas, weats))

        if acc is not None:
            accumulate_prompt(acc, record["domain"], record["category"],
                              vectors, roles["phrases"], record["prompt_id"])
        if verbose:
            elapsed = time.time() - t_start
            print(f"  [{k:3d}/{len(prompts)}] {record['prompt_id']:<22} "
                  f"bias={score:+.3f}  delta_L{n_hidden-1}={deltas[-1]:+.5f}  "
                  f"gen={gen_meta['generate_seconds']:.1f}s "
                  f"fwd={meta['forward_seconds']:.2f}s  "
                  f"({elapsed/k:.1f}s/prompt)", flush=True)

    return rows, failures, dict(n_hidden_states=n_hidden,
                                total_seconds=time.time() - t_start)



def secondary_rows(acc, prompts, stim, n_hidden, analysis_version):
    """(set-level rows, stimulus-layer rows, missing), stamped `analysis_version`."""
    set_rows, stim_rows, missing = [], [], {}
    for domain, category in sorted({(r["domain"], r["category"]) for r in prompts}):
        arm = arm_for(domain, category, stim)
        if not acc.stimuli(domain, category):
            continue
        srows, miss = set_level_rows(acc, domain, category, arm,
                                     n_hidden_states=n_hidden,
                                     analysis_version=analysis_version)
        set_rows.extend(srows)
        if any(miss.values()):
            missing[f"{domain}/{category}"] = {k: v for k, v in miss.items() if v}
        stim_rows.extend(stimulus_layer_rows(
            acc, domain, category, arm, n_hidden_states=n_hidden,
            analysis_version=analysis_version))
    return set_rows, stim_rows, missing



# =========================================================
# Manifest
# =========================================================

def build_manifest(*, backend_module, backend, arch, design, n_hidden_states,
                   parse_failures, generation, missing, args, bos):
    """
    The WAT-only manifest (was embed_bias_llama3_hidden.build_manifest), with the
    model's one-line description and analysis version taken from the backend,
    then handed to the backend for its model-specific fields.
    """
    m = {
        "analysis": backend_module.WAT_MANIFEST_ANALYSIS,
        "analysis_version": backend_module.WAT_ANALYSIS_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(),
        "design": design,
        "checkpoint": backend["checkpoint"],
        "safetensors_shards": backend["safetensors_shards"],
        "ignored_duplicate_weights": backend["ignored_duplicates"],
        "model_class": backend["model_class"],
        "architectures": backend["architectures"],
        "dtype": backend["dtype"],
        "tokenizer": backend["tokenizer_class"],
        "transformers_version": arch["transformers_version"],
        "torch_version": arch["torch_version"],
        "device": backend["device"],
        "quantisation": backend["quantisation"],
        "cpu_offload": backend["cpu_offload"],
        "disk_offload": backend["disk_offload"],
        "hidden_size": backend["hidden_size"],
        "n_transformer_blocks": backend["n_transformer_blocks"],
        "n_hidden_states": n_hidden_states,
        "system_prompt": SYSTEM_PROMPT,
        "chat_template_reconstruction":
            "tokenizer.apply_chat_template([{system}, {user}], tokenize=False, "
            "add_generation_prompt=True); the prompt must appear verbatim "
            "exactly once or the run stops. The SAME templated string is used "
            "for generate() and for the hidden-state forward pass",
        "generation": generation,
        "generation_seed_rule": SEED_RULE,
        "generation_seed_base": args.seed if args is not None else SEED_BASE,
        "hidden_states_from": "the input prompt only; generated assistant "
                              "tokens are never embedded or pooled",
        "pooling": POOLING,
        "pooling_rule": "mean over every sub-token of the stimulus phrase "
                        "(never Bai's first token only)",
        "layer_convention": {
            "layer0": "hidden_states[0] — the input embedding, before any "
                      "transformer block",
            "layer1..layerN": "output of transformer block 1..N",
            "n_transformer_blocks": backend["n_transformer_blocks"],
        },
        "orientation": {
            "gender": "X = male target, Y = female target, A = brilliance, "
                      "B = condition control",
            "fab": "X = brilliance pole, Y = effort pole, A = high-FAB fields, "
                   "B = low-FAB fields",
            "resolved_from": "stimuli.csv roles, never group0/group1",
            "meaning": "positive behaviour_wat_score / delta_layer* / weat_layer* "
                       "= stereotype-congruent",
        },
        "representation_metrics": {
            "s": "s(w,A,B) = mean cos(w,A) - mean cos(w,B)",
            "delta_layer*": "s(X,A,B) - s(Y,A,B), unstandardised",
            "weat_layer*": "[mean s(X) - mean s(Y)] / "
                           "[population SD(s(X) u s(Y)) + 0.001], ddof=0",
            "why_both": "the singleton standardized score saturates near +-2 "
                        "for an association difference of any magnitude; delta "
                        "preserves the magnitude",
            "not_written": ["s_group0_layer*", "s_group1_layer*"],
        },
        "behavioural_metric": {
            "name": "behaviour_wat_score",
            "column": "behaviour_wat_score",
            "note": "WEAT is reserved for the representation metric; this is "
                    "Bai's assignment score and is never called behaviour WEAT",
            "gender": "P(brilliance | male-assigned) + "
                      "P(control | female-assigned) - 1",
            "fab": "P(brilliance | high-FAB) + P(effort | low-FAB) - 1",
            "smoothing": "none (the +0.01 of wat_gender_brilliance.d_gb is NOT "
                         "applied; denominators are checked instead)",
            "validity": "every presented item parsed exactly once and assigned "
                        "to one of the two targets THIS prompt presented",
            "on_failure": "behaviour_wat_score = NaN; prompt, raw response and all "
                          "representation columns are kept",
            "n_parse_failures": len(parse_failures),
            "parse_failure_ids": [p["prompt_id"] for p in parse_failures],
        },
        "fail_fast": {
            "non_finite_hidden_state": "aborts the run",
            "non_finite_pooled_vector": "aborts the run",
            "unresolvable_or_ambiguous_span": "aborts the run",
            "rationale": "skipping a prompt would destroy the exhaustive "
                         "stimulus-pair coverage the design is built on",
        },
        "secondary_outputs": {
            "set_level_weat": "contextual prototypes over THIS experiment's "
                              "prompts only; never pooled with the legacy "
                              "3420-prompt replay corpus",
            "effect_size_sd": "population SD, ddof=0 (weat_effect_size_hidden); "
                              "embed_bias.weat_effect_size keeps ddof=1 and is "
                              "untouched",
            "missing_stimuli": missing,
        },
        "not_run": {
            "legacy_response_replay": "this runner reads no saved behavioural "
                                      "response file and "
                                   "no behavioural-model provenance",
            "pca_nullification": False,
            "permutation_inference": False,
            "profile_or_decision_chain": "not part of this experiment; the WAT "
                                         "prompt is the only prompt",
        },
        "exploratory": "the full layer profile is exploratory; no layer is "
                       "selected, and none may be selected after seeing the data",
        "cli": vars(args) if args is not None else {},
    }
    return backend_module.finalize_wat_manifest(
        m, backend_meta=backend, n_hidden_states=n_hidden_states, bos=bos)


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True, choices=backends.names(),
                    help="which backend runs the experiment")
    ap.add_argument("--checkpoint", default=None,
                    help="default: the backend's CHECKPOINT")
    ap.add_argument("--gpu", type=int, default=0,
                    help="llama31 only (single-GPU placement); scout uses "
                         "device_map='auto'")
    ap.add_argument("--stimuli", default=STIMULI_CSV)
    ap.add_argument("--prompts", default=PROMPTS_CSV)
    ap.add_argument("--out-dir", default=None,
                    help="default: the backend's WAT_OUT_DIR")
    ap.add_argument("--seed", type=int, default=SEED_BASE,
                    help="base generation seed; per-prompt seeds derive from it")
    ap.add_argument("--max-new-tokens", type=int,
                    default=GENERATION["max_new_tokens"])
    ap.add_argument("--domain", choices=("gender", "fab", "both"), default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of prompts (diagnostics only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the frozen design; load no model")
    ap.add_argument("--smoke-test", action="store_true",
                    help="one gender + one FAB prompt, full diagnostics, "
                         "writes nothing")
    args = ap.parse_args()

    backend_module = backends.get_backend(args.model)     # the only model choice
    args.checkpoint = args.checkpoint or backend_module.CHECKPOINT
    args.out_dir = args.out_dir or backend_module.WAT_OUT_DIR

    partial = bool(args.limit) or args.domain != "both" or args.smoke_test
    stim, all_prompts = load_design(args.stimuli, args.prompts,
                                    expect_total=None if partial else N_TOTAL)
    design = design_summary(all_prompts, args.stimuli, args.prompts)

    print(f"Fixed-prompt Bai-style WAT — {backend_module.NAME}  "
          f"(analysis {backend_module.WAT_ANALYSIS_VERSION})")
    print(f"  stimuli    : {args.stimuli}")
    print(f"  prompts    : {args.prompts}  ({len(all_prompts)} frozen prompts)")
    for cond, n in design["n_by_condition"].items():
        print(f"    {cond:<20} {n}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  generation : {dict(GENERATION, max_new_tokens=args.max_new_tokens)}")
    print(f"  seed base  : {args.seed}")

    prompts = all_prompts
    if args.domain != "both":
        prompts = [r for r in prompts if r["domain"] == args.domain]
    if args.smoke_test:
        gender = next((r for r in prompts if r["domain"] == "gender"), None)
        fab = next((r for r in prompts if r["domain"] == "fab"), None)
        prompts = [r for r in (gender, fab) if r is not None]
    elif args.limit:
        prompts = prompts[:args.limit]

    # role resolution needs no GPU — do it for every prompt before loading a model
    for record in prompts:
        prompt_roles(record, stim)
    print(f"  roles resolved for all {len(prompts)} prompts "
          f"(X/Y/A/B from stimuli.csv, not group0/group1)")

    if args.dry_run:
        print("\n--dry-run: design validated, no model loaded, nothing written.")
        return

    import torch, transformers
    backend = backend_module.load(args.checkpoint, args.gpu)
    meta = backend.meta
    print(f"\n  loaded {meta['model_class']} in {meta['load_seconds']}s "
          f"on {meta['device']}  ({meta['dtype']}, {meta['quantisation']})")

    if args.smoke_test:
        return smoke_test(args, stim, prompts, all_prompts, backend)

    acc = PrototypeAccumulator()
    rows, failures, run_meta = run_prompts(prompts, stim, backend, args=args, acc=acc)
    n_hidden = run_meta["n_hidden_states"]
    print(f"\n  {len(rows)} prompts in {run_meta['total_seconds']/60:.1f} min; "
          f"{len(failures)} behavioural parse failure(s)")

    set_rows, stim_rows, missing = secondary_rows(
        acc, prompts, stim, n_hidden, backend_module.WAT_ANALYSIS_VERSION)

    os.makedirs(args.out_dir, exist_ok=True)
    write_primary_csv(os.path.join(args.out_dir, PRIMARY_CSV), rows, n_hidden,
                      force=args.force)
    write_csv(os.path.join(args.out_dir, "behaviour_parse_failures.csv"),
              failures, args.force)
    write_csv(os.path.join(args.out_dir, "secondary_set_level_weat.csv"),
              set_rows, args.force)
    write_csv(os.path.join(args.out_dir, "secondary_stimulus_layer.csv"),
              stim_rows, args.force)

    import numpy as np
    np.savez_compressed(
        os.path.join(args.out_dir, "secondary_stimulus_prototypes.npz"),
        **{f"{d}|{a}|{s}": acc.prototype(d, a, s) for (d, a, s) in acc.sums})

    first, _ = build_chat_input(backend.tokenizer, SYSTEM_PROMPT, prompts[0]["prompt"])
    manifest = build_manifest(
        backend_module=backend_module, backend=meta,
        arch=dict(transformers_version=transformers.__version__,
                  torch_version=torch.__version__),
        design=design, n_hidden_states=n_hidden, parse_failures=failures,
        generation=dict(GENERATION, max_new_tokens=args.max_new_tokens),
        missing=missing, args=args, bos=bos_record(backend.tokenizer, first))
    with open(os.path.join(args.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  wrote {os.path.join(args.out_dir, 'manifest.json')}")


def smoke_test(args, stim, prompts, all_prompts, backend):
    """One gender + one FAB prompt, every diagnostic, nothing written."""

    def rule(t):
        print(f"\n{'='*74}\n{t}\n{'='*74}", flush=True)

    tokenizer = backend.tokenizer
    for line in backend.memory_report() + backend.describe():
        print(line)
    lo, hi, n_bos = validate_all_spans(tokenizer, all_prompts, stim)
    print(f"  spans              : every stimulus of all {len(all_prompts)} "
          f"prompts resolves once; {lo}..{hi} tokens; leading BOS {n_bos}")

    for record in prompts:
        roles = prompt_roles(record, stim)
        rule(f"SMOKE — {record['prompt_id']}  ({record['domain']}/"
             f"{record['category']}, trial {record['trial']})")
        print(f"  presented order      : group0={record['group0']!r}  "
              f"group1={record['group1']!r}")
        print(f"  roles from stimuli   : X={roles['X']!r} ({roles['arm']['X_name']})"
              f"  Y={roles['Y']!r} ({roles['arm']['Y_name']})")
        print(f"  A ({roles['arm']['A_name']}): {roles['A']}")
        print(f"  B ({roles['arm']['B_name']}): {roles['B']}")
        print(f"  presented order of items: {roles['attributes']}")
        print(f"\n  prompt:\n    {record['prompt']}")

        templated, window = build_chat_input(tokenizer, SYSTEM_PROMPT,
                                             record["prompt"])
        n_tok = len(tokenizer(templated)["input_ids"])
        bos = bos_record(tokenizer, templated)
        print(f"\n  system prompt        : {SYSTEM_PROMPT!r}")
        print(f"  chat-template tokens : {n_tok}   BOS in template "
              f"{bos['bos_in_templated_text']}, in model input "
              f"{bos['bos_in_model_input']}")

        seed = prompt_seed(record["prompt_id"], args.seed)
        response, gen_meta = backend.generate(
            templated, seed=seed,
            generation=dict(GENERATION, max_new_tokens=args.max_new_tokens))
        print(f"\n  generation           : seed={seed} "
              f"temp={gen_meta['temperature']} do_sample={gen_meta['do_sample']} "
              f"new_tokens={gen_meta['n_new_tokens']} "
              f"({gen_meta['generate_seconds']:.1f}s)")
        print("  response:")
        for line in response.splitlines():
            print(f"    {line}")

        vectors, meta = backend.extract_hidden(
            templated, roles["phrases"], user_window=window,
            competitors=roles["phrases"], context=record["prompt_id"])
        print(f"\n  hidden states        : {meta['n_hidden_states']} "
              f"(= 1 input embedding + {meta['n_transformer_blocks']} blocks), "
              f"dim {meta['hidden_size']}")
        print(f"  prompt tokens        : {meta['n_prompt_tokens']}   "
              f"forward {meta['forward_seconds']:.2f}s   "
              f"finite check {meta['finite_check']}")
        print(f"  pooling              : {meta['pooling']} over every sub-token")
        for phrase in roles["phrases"]:
            lo, hi = meta["spans"][phrase]
            print(f"    {phrase:<26} tokens [{lo}, {hi})  "
                  f"{meta['subtokens'][phrase]}")

        assign, problems = parse_behaviour(response, record, roles)
        if problems:
            score, denom = float("nan"), None
            print(f"\n  behavioural parse    : FAILED — {' | '.join(problems)}")
            print("  (representation columns are kept; behaviour_wat_score = NaN)")
        else:
            score, denom = behaviour_wat_score(assign, record, roles)
            print(f"\n  behavioural parse    : all "
                  f"{len(roles['attributes'])} items assigned")
            if denom is None:
                print(f"  behaviour_wat_score  : {score:+.4f}  "
                      f"(positive = stereotype-congruent)")
            else:
                print(f"  behaviour_wat_score  : NaN — {denom}")

        deltas, weats = representation_scores(vectors, roles,
                                              meta["n_hidden_states"])
        last = meta["n_hidden_states"] - 1
        mid = last // 2
        print(f"\n  layer0  (input embedding) delta={deltas[0]:+.5f}  "
              f"weat={weats[0]:+.4f}")
        print(f"  layer1  (block 1)         delta={deltas[1]:+.5f}  "
              f"weat={weats[1]:+.4f}")
        print(f"  layer{mid:<2} (block {mid})        delta={deltas[mid]:+.5f}  "
              f"weat={weats[mid]:+.4f}")
        print(f"  layer{last:<2} (block {last})       delta={deltas[last]:+.5f}  "
              f"weat={weats[last]:+.4f}")
        print(f"  non-finite values    : "
              f"{sum(1 for v in deltas + weats if v != v)}")
        for line in backend.memory_report(peak=True):
            print(line)

    rule("SMOKE TEST COMPLETE — nothing written")


if __name__ == "__main__":
    main()
