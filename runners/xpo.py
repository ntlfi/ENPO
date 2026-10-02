#!/usr/bin/env python3
"""
Batch XPO — Exploratory Preference Optimization (algos/specs/xpo.md, Xie et al. 2024).

Each outer iteration t = 1..T:
  1a. Generate K=8 completions per prompt from π^(t).
  1b. Generate K=8 completions per prompt from π_ref (the FROZEN base).
      Iter 1 short-circuit: π^(1) = π_ref, so we run 1b only and reuse for both.
  2.  Score all 16 with the strong judge; per prompt, pick the cross-side
      pair (one from π^(t), one from π_ref) with the largest |score margin|;
      tag whichever side is from π_ref as τ̃ (`ref_is_chosen` flag).
  3.  Pre-compute log π_ref(chosen) and log π_ref(rejected) once per pair
      (used by the DPO-sigmoid term during training).
  4.  Accumulate cumulative buffer D^(t) = D^(t-1) ∪ B_t.
  5.  Train: minimise per-pair
        L_DPO(π) = softplus( -β·(Δ_π - Δ_ref) )
        L_OPT(π) = -α·log π(τ̃)                (sign chosen for grad-descent)
      starting from π^(t). Save as π^(t+1).

Per-iter α follows the paper schedule (default {1e-5, 5e-6, 0} for T=3).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts     import load_prompts, truncate_prompts
from src.generation  import run_vllm_generate
from src.score_xpo   import run_score_xpo
from src.precompute  import run_precompute_logp
from src.train_xpo   import run_train_xpo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",    default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--judge_model",   default="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M")
    p.add_argument("--dataset_name",  default="HuggingFaceH4/ultrafeedback_binarized")
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--num_prompts",    type=int, default=20000)
    p.add_argument("--max_new_tokens",    type=int, default=512)
    p.add_argument("--max_input_tokens",  type=int, default=1024)
    p.add_argument("--n_completions",     type=int, default=8,
                   help="K completions per side (π^(t) and π_ref each)")
    p.add_argument("--temperature",       type=float, default=0.7)
    p.add_argument("--top_p",             type=float, default=0.9)
    p.add_argument("--vllm_tp_size",      type=int, default=8)
    p.add_argument("--vllm_gpu_mem_util", type=float, default=0.90)
    p.add_argument("--judge_batch_size",  type=int, default=8)
    p.add_argument("--judge_max_length",  type=int, default=4096)
    p.add_argument("--margin_keep_pct",   type=float, default=0.7)
    # XPO loss
    p.add_argument("--beta",            type=float, default=0.05)
    p.add_argument("--alpha_schedule",  type=str,   default="1e-5,5e-6,0",
                   help="Comma-separated α per iter (defaults match paper for T=3)")
    # Trainer
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate",     type=float, default=1e-6)
    p.add_argument("--num_train_epochs",  type=int,   default=1)
    p.add_argument("--max_length",        type=int,   default=1024)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--post_iter_hook",    default=None,
                   help="Optional shell command run (blocking) right after each "
                        "iteration's checkpoint is saved, before the next "
                        "iteration's generation starts. {iter_dir} and {iter} "
                        "are substituted.")
    return p.parse_args()


def main():
    args = parse_args()
    alphas = [float(s) for s in args.alpha_schedule.split(",")]
    if len(alphas) != args.num_iterations:
        raise ValueError(
            f"alpha_schedule has {len(alphas)} entries but num_iterations={args.num_iterations}")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pi_ref = args.base_model
    pi_t   = args.base_model            # π^(1) = π_ref

    tok = AutoTokenizer.from_pretrained(args.base_model)
    accumulated_data = []               # cumulative buffer

    for t in range(1, args.num_iterations + 1):
        iter_dir = os.path.join(args.output_dir, f"iter{t}")
        os.makedirs(iter_dir, exist_ok=True)
        alpha_t = alphas[t - 1]

        start_idx = (t - 1) * args.num_prompts
        prompts, instructions, msg_lists = load_prompts(
            args.dataset_name, args.num_prompts, tok, start_idx)
        prompts_for_gen, n_trunc = truncate_prompts(prompts, tok, args.max_input_tokens)
        if n_trunc:
            print(f"[Truncate] {n_trunc}/{len(prompts)} prompts > {args.max_input_tokens} "
                  f"tokens; left-truncated to last {args.max_input_tokens}", flush=True)

        print(f"\n{'='*60}")
        print(f"XPO — iteration {t}/{args.num_iterations}")
        print(f"  policy π^(t) : {pi_t}")
        print(f"  ref    π_ref : {pi_ref}                  (frozen)")
        print(f"  β = {args.beta}, α_t = {alpha_t}")
        print(f"  K per side   : {args.n_completions} (max-margin cross-pair)")
        print(f"  prompts      : [{start_idx}, {start_idx + args.num_prompts})")
        print(f"  buffer so far: {len(accumulated_data)} prior dirs (this iter adds D_{t})")
        print(f"  output       : {iter_dir}")
        print(f"{'='*60}", flush=True)

        # ──────────────────────────────────────────────────────────────────
        # Phase 1a: gen from π^(t)
        # Phase 1b: gen from π_ref
        # Iter 1 short-circuit: π_t = π_ref → gen ONCE, link both files to it.
        # ──────────────────────────────────────────────────────────────────
        completions_pi  = os.path.join(iter_dir, "completions_pi.json")
        completions_ref = os.path.join(iter_dir, "completions_ref.json")

        if t == 1:
            if not os.path.exists(completions_pi):
                run_vllm_generate(
                    model=pi_ref, prompts=prompts_for_gen, iter_dir=iter_dir,
                    n=args.n_completions,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                    tp_size=args.vllm_tp_size,
                    gpu_mem_util=args.vllm_gpu_mem_util,
                    output_filename="completions_pi.json",
                    input_filename="vllm_input_pi.json",
                    tokenizer=args.base_model,
                    seed=args.seed + 100 * t,
                )
            if not os.path.exists(completions_ref):
                # Iter 1: π_t = π_ref. Run a SECOND independent gen call so the
                # two sets are stochastically distinct (different seeds), matching
                # the algorithm spec. Otherwise the score worker would always see
                # identical completions and pair construction would degenerate.
                run_vllm_generate(
                    model=pi_ref, prompts=prompts_for_gen, iter_dir=iter_dir,
                    n=args.n_completions,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                    tp_size=args.vllm_tp_size,
                    gpu_mem_util=args.vllm_gpu_mem_util,
                    output_filename="completions_ref.json",
                    input_filename="vllm_input_ref.json",
                    tokenizer=args.base_model,
                    seed=args.seed + 100 * t + 1,
                )
        else:
            if not os.path.exists(completions_pi):
                run_vllm_generate(
                    model=pi_t, prompts=prompts_for_gen, iter_dir=iter_dir,
                    n=args.n_completions,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                    tp_size=args.vllm_tp_size,
                    gpu_mem_util=args.vllm_gpu_mem_util,
                    output_filename="completions_pi.json",
                    input_filename="vllm_input_pi.json",
                    tokenizer=args.base_model,
                    seed=args.seed + 100 * t,
                )
            if not os.path.exists(completions_ref):
                run_vllm_generate(
                    model=pi_ref, prompts=prompts_for_gen, iter_dir=iter_dir,
                    n=args.n_completions,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                    tp_size=args.vllm_tp_size,
                    gpu_mem_util=args.vllm_gpu_mem_util,
                    output_filename="completions_ref.json",
                    input_filename="vllm_input_ref.json",
                    tokenizer=args.base_model,
                    seed=args.seed + 100 * t + 1,
                )

        # ──────────────────────────────────────────────────────────────────
        # Phase 2: score → cross-side max-margin pairs with τ̃ tag → D_t
        # ──────────────────────────────────────────────────────────────────
        d_t = os.path.join(iter_dir, "preference_data")
        if not os.path.exists(d_t):
            d_t = run_score_xpo(
                prompts=prompts, instructions=instructions, message_lists=msg_lists,
                completions_pi_file=completions_pi,
                completions_ref_file=completions_ref,
                iter_dir=iter_dir,
                judge_model=args.judge_model,
                margin_keep_pct=args.margin_keep_pct,
                judge_batch_size=args.judge_batch_size,
                judge_max_length=args.judge_max_length,
            )

        # ──────────────────────────────────────────────────────────────────
        # Phase 3: precompute Δ_ref under π_ref (for the DPO sigmoid term)
        # ──────────────────────────────────────────────────────────────────
        d_t_with_ref = os.path.join(iter_dir, "preference_data_with_ref")
        if not os.path.exists(d_t_with_ref):
            run_precompute_logp(
                model_path=pi_ref,
                dataset_dir=d_t,
                output_dataset_dir=d_t_with_ref,
                logp_chosen_field="logp_ref_chosen",
                logp_rejected_field="logp_ref_rejected",
                max_length=args.max_length,
            )

        accumulated_data.append(d_t_with_ref)

        # ──────────────────────────────────────────────────────────────────
        # Phase 4: train XPO on the cumulative buffer
        # ──────────────────────────────────────────────────────────────────
        run_train_xpo(
            policy_path=pi_t,
            preference_data_dirs=accumulated_data,
            output_dir=iter_dir,
            beta=args.beta,
            alpha=alpha_t,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            max_length=args.max_length,
            wandb_run_name=f"xpo-iter{t}-{Path(args.output_dir).name}",
            seed=args.seed,
        )

        pi_t = iter_dir
        print(f"\nIteration {t} done → {iter_dir}  "
              f"(buffer: D_1..D_{t}, {len(accumulated_data)} dirs)", flush=True)

        if args.post_iter_hook:
            cmd = args.post_iter_hook.format(iter_dir=iter_dir, iter=t)
            print(f"\n[post_iter_hook] running: {cmd}", flush=True)
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"[post_iter_hook] WARNING: hook exited {result.returncode} "
                      f"(e.g. eval API credits/errors) -- continuing training "
                      f"regardless; rerun the eval separately once resolved.",
                      flush=True)

    print(f"\nXPO complete. Final policy: {pi_t}")


if __name__ == "__main__":
    main()
