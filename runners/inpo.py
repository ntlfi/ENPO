#!/usr/bin/env python3
"""
INPO — Iterative Nash Policy Optimization (algos/specs/inpo.md).

Each outer iteration t = 1..T:
  1. Generate n completions per prompt from π_t (vLLM, K-completion sampling).
  2. Score with the strong judge → preference pairs D_t.
  3. Pre-compute log-probs for the FROZEN anchors:
       Δ_ref = log π_ref(y_w|x) − log π_ref(y_l|x)
       Δ_t   = log π_t  (y_w|x) − log π_t  (y_l|x)
     stored as four scalar columns in the dataset.
     In iter 1, π_t = π_ref so we precompute once and copy.
  4. INPO train: minimise L_t(π) = E[(Δ_π − (τ/η)Δ_ref − ((η−τ)/η)Δ_t − 1/(2η))²]
     starting from π_t. Save the result as π_{t+1}.

π_t is FROZEN during iter t — it appears only via the cached log-probs, never
as a forward-pass model during training. (This is the property INPO needs that
went wrong in the earlier "online DPO" implementation we removed.)
"""
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

# Make `import src.*` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts     import load_prompts, truncate_prompts
from src.generation  import run_vllm_generate
from src.score       import run_score
from src.precompute  import run_precompute_logp
from src.train_inpo  import run_train_inpo


def parse_args():
    p = argparse.ArgumentParser()
    # Models
    p.add_argument("--base_model",    default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Initial policy AND frozen reference π_ref")
    p.add_argument("--judge_model",   default="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M")
    # Data
    p.add_argument("--dataset_name",  default="HuggingFaceH4/ultrafeedback_binarized")
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--num_prompts",    type=int, default=20000,
                   help="Prompts per iteration (disjoint slices)")
    # Generation
    p.add_argument("--max_new_tokens",    type=int, default=512)
    p.add_argument("--max_input_tokens",  type=int, default=1024)
    p.add_argument("--n_completions",     type=int, default=8,
                   help="Paper uses K=8; best-of-K vs worst-of-K via the judge.")
    p.add_argument("--temperature",       type=float, default=0.7)
    p.add_argument("--top_p",             type=float, default=0.9)
    p.add_argument("--vllm_tp_size",      type=int, default=8)
    p.add_argument("--vllm_gpu_mem_util", type=float, default=0.90)
    # Judge
    p.add_argument("--judge_batch_size",  type=int, default=8)
    p.add_argument("--judge_max_length",  type=int, default=4096)
    p.add_argument("--margin_keep_pct",   type=float, default=0.7)
    # INPO loss
    p.add_argument("--eta",               type=float, default=7.5e-3,
                   help="OMD inverse-LR (paper default 7.5e-3)")
    p.add_argument("--tau",               type=float, default=2.5e-3,
                   help="KL strength; paper default eta/3 = 2.5e-3. Must satisfy 0 < tau < eta.")
    # Trainer
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate",     type=float, default=1e-6)
    p.add_argument("--num_train_epochs",  type=int,   default=1)
    p.add_argument("--max_length",        type=int,   default=1024)
    # Output
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--post_iter_hook",    default=None,
                   help="Optional shell command run (blocking) right after each "
                        "iteration's checkpoint is saved, before the next "
                        "iteration's generation starts. {iter_dir} and {iter} "
                        "are substituted.")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0 < args.tau < args.eta):
        raise ValueError(
            f"INPO requires 0 < tau < eta; got tau={args.tau}, eta={args.eta}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pi_ref = args.base_model            # FROZEN reference (the SFT/base anchor)
    pi_t   = args.base_model            # iter 1: π_1 = π_ref (algorithm line 1)

    tok = AutoTokenizer.from_pretrained(args.base_model)

    for t in range(1, args.num_iterations + 1):
        iter_dir = os.path.join(args.output_dir, f"iter{t}")
        os.makedirs(iter_dir, exist_ok=True)

        start_idx = (t - 1) * args.num_prompts
        prompts, instructions, msg_lists = load_prompts(
            args.dataset_name, args.num_prompts, tok, start_idx)
        prompts_for_gen, n_trunc = truncate_prompts(prompts, tok, args.max_input_tokens)
        if n_trunc:
            print(f"[Truncate] {n_trunc}/{len(prompts)} prompts > {args.max_input_tokens} "
                  f"tokens; left-truncated to last {args.max_input_tokens}", flush=True)

        print(f"\n{'='*60}")
        print(f"INPO — iteration {t}/{args.num_iterations}")
        print(f"  policy π_t   : {pi_t}                    (= π_{t-1}+update)")
        print(f"  ref    π_ref : {pi_ref}                  (frozen)")
        print(f"  η = {args.eta}, τ = {args.tau}, "
              f"target = 1/(2η) = {1.0/(2*args.eta):.4f}")
        print(f"  prompts      : [{start_idx}, {start_idx + args.num_prompts})")
        print(f"  output       : {iter_dir}")
        print(f"{'='*60}", flush=True)

        # ──────────────────────────────────────────────────────────────────
        # Phase 1 — generation from π_t
        # ──────────────────────────────────────────────────────────────────
        completions_file = os.path.join(iter_dir, "completions.json")
        if os.path.exists(completions_file):
            print(f"[Phase 1] cached completions found → skipping vLLM")
        else:
            run_vllm_generate(
                model=pi_t, prompts=prompts_for_gen, iter_dir=iter_dir,
                n=args.n_completions,
                temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                tp_size=args.vllm_tp_size,
                gpu_mem_util=args.vllm_gpu_mem_util,
                tokenizer=args.base_model,
                seed=args.seed + 100 * t,
            )

        # ──────────────────────────────────────────────────────────────────
        # Phase 2a — score → D_t (HF dataset with prompt/chosen/rejected)
        # ──────────────────────────────────────────────────────────────────
        d_t = run_score(
            prompts=prompts, instructions=instructions, message_lists=msg_lists,
            completions_file=completions_file, iter_dir=iter_dir,
            judge_model=args.judge_model,
            margin_keep_pct=args.margin_keep_pct,
            judge_batch_size=args.judge_batch_size,
            judge_max_length=args.judge_max_length,
        )

        # ──────────────────────────────────────────────────────────────────
        # Phase 2b — precompute frozen-anchor log-probs
        #
        # We always need both Δ_ref AND Δ_t. In iter 1 they are equal because
        # π_1 = π_ref, so we run precompute once and copy the columns.
        # ──────────────────────────────────────────────────────────────────
        d_t_with_ref = os.path.join(iter_dir, "preference_data_with_ref")
        if os.path.exists(d_t_with_ref):
            print(f"[Phase 2b] cached ref-logp dataset found → skipping precompute")
        else:
            run_precompute_logp(
                model_path=pi_ref,
                dataset_dir=d_t,
                output_dataset_dir=d_t_with_ref,
                logp_chosen_field="logp_ref_chosen",
                logp_rejected_field="logp_ref_rejected",
                max_length=args.max_length,
            )

        d_t_full = os.path.join(iter_dir, "preference_data_inpo")
        if os.path.exists(d_t_full):
            print(f"[Phase 2b] cached full-logp dataset found → skipping precompute")
        elif t == 1:
            # π_t = π_ref ⇒ Δ_t = Δ_ref. Just duplicate the columns under the t-side names.
            print(f"[Phase 2b] iter 1: π_t = π_ref → copying logp_ref_* into logp_t_*")
            ds = load_from_disk(d_t_with_ref)
            ds = ds.add_column("logp_t_chosen",   list(ds["logp_ref_chosen"]))
            ds = ds.add_column("logp_t_rejected", list(ds["logp_ref_rejected"]))
            ds.save_to_disk(d_t_full)
        else:
            run_precompute_logp(
                model_path=pi_t,
                dataset_dir=d_t_with_ref,
                output_dataset_dir=d_t_full,
                logp_chosen_field="logp_t_chosen",
                logp_rejected_field="logp_t_rejected",
                max_length=args.max_length,
            )

        # ──────────────────────────────────────────────────────────────────
        # Phase 2c — INPO training
        # Trainable policy starts from π_t. Anchors enter only via cached
        # logp columns; the trainable forward pass only computes Δ_π.
        # ──────────────────────────────────────────────────────────────────
        run_train_inpo(
            policy_path=pi_t,
            preference_data_dir=d_t_full,
            output_dir=iter_dir,
            eta=args.eta,
            tau=args.tau,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            max_length=args.max_length,
            wandb_run_name=f"inpo-iter{t}-{Path(args.output_dir).name}",
            seed=args.seed,
        )

        pi_t = iter_dir
        print(f"\nIteration {t} done → {iter_dir}", flush=True)

        if args.post_iter_hook:
            cmd = args.post_iter_hook.format(iter_dir=iter_dir, iter=t)
            print(f"\n[post_iter_hook] running: {cmd}", flush=True)
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"[post_iter_hook] WARNING: hook exited {result.returncode} "
                      f"(e.g. eval API credits/errors) -- continuing training "
                      f"regardless; rerun the eval separately once resolved.",
                      flush=True)

    print(f"\nINPO complete. Final policy: {pi_t}")


if __name__ == "__main__":
    main()
