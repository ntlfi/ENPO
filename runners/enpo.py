#!/usr/bin/env python3
"""
ENPO — Single-pair policy optimization with proximal regularization and
adversarial sampling. Implements the Step A + Step B per-iteration pipeline:

  - Step A (exploit): OMD update with an on-policy proximal regularizer.
                      Trains π_{t+1} from π_t.
  - Step B (explore): synthetic-DPO update on an adversarial policy. Ranks K
                      fresh on-policy candidates from π_t under the scalar
                      reward r(x,z) = s·(log π_{t+1}(z|x) − log π_t(z|x)),
                      forms (top vs bottom) synthetic pref pairs, runs DPO
                      with reference π_t and β anchor. Trains π̂_{t+1} from π̂_t.

Two checkpoints maintained per iter:
  - π_t   (policy)    — saved to iter_dir/policy/       — returned at end
  - π̂_t (adversary) — saved to iter_dir/adversary/    — used only for sampling

SINGLE-PAIR (spec-faithful K=2) variant. Pair role is fixed by sample origin:
y_i = 1 random sample from π_t pool, y'_i = 1 random sample from π̂_t pool
(no Skywork max-margin selection at the pair-build stage). Phase 7's triple-
sample judge sets the signed target via I^y vs I^y'; sign can be either.

At iter 1, π̂_1 = π_t = π_ref so y, y' are iid from π_ref (different RNG).

D_t' (proximal regularizer dataset) uses ONLY y_i ~ π_t (the chosen column of
this iter's buffer shard); |D_t'| = n. Matches spec line 11.

Step B candidate pool: FRESH K samples from π_t (separate vLLM gen pass — NOT
reused from Phase 1a). Ranked by r, chosen/rejected synth pair built, then
DPO trainer with π_t reference produces π̂_{t+1}.

Per-iter pipeline at iter t:

  Phase 1a: vLLM gen K_pair from π_t        → completions_pi.json    (n × K_pair;
                                               default K_pair=1 — only 1 sample/pool
                                               is ever used by Phase 2, see --n_pair)
  Phase 1b: vLLM gen K_pair from π̂_t      → completions_advs.json  (n × K_pair; runs
                                               unconditionally including iter 1)
  Phase 1c: vLLM gen K from π_t fresh  → completions_pi_fresh.json (n × K; for Step B,
                                          must be even and >=2, see --n_completions)
  Phase 2:  Single-pair build (NO Skywork pass at this stage):
              chosen   = 1 random sample from π_t pool   (= y  ~ π_t)
              rejected = 1 random sample from π̂_t pool  (= y' ~ π̂_t)
            1 row per prompt. No tie drops.
  Phase 3:  precompute logp_ref on shard.
  Phase 4:  concatenate shards → cum_buffer (or current shard only if --no_buffer).
  Phase 5:  sample m rows from cum_buffer → D_t_pairs.
  Phase 6:  vLLM gen n_z reference z's from π_t.
  Phase 7:  triple-sample judge → label_chosen, label_rejected.
  Phase 8:  precompute logp_t on the m sampled rows.
  Phase 8.5: D_t' singletons = chosen column of buffer shard (= y_i ~ π_t), |D_t'| = n.
  Phase 9a (Step A): ENPO Step A training → iter_dir/policy/.
  Phase 9b (Step B):
    9b-i:   Build pair-adjacent dataset from completions_pi_fresh.json
            (K samples per prompt → K/2 pairs per prompt as adjacent (k=0,1),
             (k=2,3), ...). |dataset| = n*K/2.
    9b-ii:  precompute logp_pi_t  on this dataset (iter 1: copy logp_ref).
    9b-iii: precompute logp_pi_t1 on this dataset (uses Step A output).
    9b-iv:  Reshape: per prompt, pull K logps from each model. Compute
            r(x, z_k) = s · (logp_pi_t1[k] − logp_pi_t[k]). Per prompt:
            chosen = argmax_k r, rejected = argmin_k r. Build synth pref
            dataset with logp_t (= logp_pi_t for chosen/rejected) cached.
    9b-v:   Step B trainer: DPO with β, ref=π_t (cached), init=π̂_t.
            → iter_dir/adversary/.

Output policy: iter_dir/policy/ at the end. (π_{T+1} last-iterate; the spec
returns π̄_T but we keep last-iterate to match the rest of the codebase. The
runner saves all T policy AND adversary checkpoints for inspection.)
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts             import load_prompts, truncate_prompts
from src.generation          import run_vllm_generate
from src.precompute          import run_precompute_logp
from src.score_enpo      import run_score_enpo
from src.train_enpo_step_a import run_train_enpo_step_a
from src.train_enpo_step_b   import run_train_enpo_step_b


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",    default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Initial policy AND frozen reference π_ref AND initial π̂")
    p.add_argument("--judge_model",   default="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M")
    p.add_argument("--dataset_name",  default="HuggingFaceH4/ultrafeedback_binarized")
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--num_prompts",    type=int, default=20000,
                   help="New prompts per iteration (disjoint slices)")
    p.add_argument("--m_train_per_iter", type=int, default=None,
                   help="Size of D_t. Defaults to --num_prompts.")
    p.add_argument("--max_new_tokens",    type=int, default=512)
    p.add_argument("--max_input_tokens",  type=int, default=1024)
    p.add_argument("--n_completions",     type=int, default=8,
                   help="K candidates per prompt for the Step B fresh-π_t pool "
                        "(Phase 1c). Must be even and >= 2 (paired adjacently "
                        "into K/2 rows). Historically also drove Phase 1a/1b "
                        "(the π_t/π̂_t pair pools) — see --n_pair.")
    p.add_argument("--n_pair",            type=int, default=1,
                   help="K candidates per prompt for the Step A pair pools "
                        "(π_t and π̂_t, Phase 1a/1b). Each prompt uses exactly "
                        "1 random sample from each pool regardless of K (Phase "
                        "2's single-pair build), so K>1 here only generates "
                        "samples that are never scored or used. Default 1 "
                        "avoids that waste; must be >= 1.")
    p.add_argument("--n_z",               type=int, default=1)
    p.add_argument("--temperature",       type=float, default=0.7)
    p.add_argument("--top_p",             type=float, default=0.9)
    p.add_argument("--vllm_tp_size",      type=int, default=8)
    p.add_argument("--vllm_gpu_mem_util", type=float, default=0.90)
    p.add_argument("--judge_batch_size",  type=int, default=8)
    p.add_argument("--judge_max_length",  type=int, default=4096)
    # Step A hyperparameters
    p.add_argument("--eta",               type=float, default=3.75e-3,
                   help="OMD parameter (Step A). Default 3.75e-3.")
    p.add_argument("--tau",               type=float, default=2.5e-3,
                   help="KL strength to π_ref in Step A loss.")
    p.add_argument("--alpha",             type=float, default=0.001,
                   help="Proximal regularization strength α (Step A). Constant across iters.")
    # Step B hyperparameters
    p.add_argument("--beta",              type=float, default=0.05,
                   help="Step B KL anchor strength to π_t (the spec's β). "
                        "Closed-form Step B optimum: π̂* ∝ π_t · exp(r/β). "
                        "Smaller β = more aggressive tilt away from π_t (collapse-prone). "
                        "Range 0.05–0.5 keeps π̂ in support of π_t.")
    p.add_argument("--s_sign",            type=float, default=+1.0,
                   help="Step B reward sign s ∈ {+1, −1}. +1: amplify π's "
                        "recent-movement direction. −1: conservative.")
    # Sampling reproducibility
    p.add_argument("--sample_seed",       type=int, default=0)
    # Trainer
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate",     type=float, default=1e-6)
    p.add_argument("--num_train_epochs",  type=int,   default=1)
    p.add_argument("--max_length",        type=int,   default=1024)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--no_buffer",         action="store_true",
                   help="Skip cum_buffer concat in Phase 4. Step A trains on the current iter's shard only (fresh-data ENPO).")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--post_iter_hook",    default=None,
                   help="Optional shell command run (blocking) right after each "
                        "iteration's policy checkpoint is saved, before the next "
                        "iteration's generation starts. {iter_dir} and {iter} "
                        "are substituted.")
    return p.parse_args()


def attach_prox_singletons(d_t_full_dir, prox_prompts_in, prox_responses_in,
                           seed, output_dir):
    """Attach on-policy proximal samples to d_t_full; |D_t'| = n_kept."""
    ds = load_from_disk(d_t_full_dir)
    M = len(ds)
    singletons = [(p, r) for p, r in zip(prox_prompts_in, prox_responses_in) if r]
    n_kept = len(singletons)
    if n_kept == 0:
        raise ValueError("[Phase 8.5] no on-policy singletons available for D_t'")

    rng = random.Random(seed)
    if n_kept >= M:
        rng.shuffle(singletons)
        chosen = singletons[:M]
        replacement = False
    else:
        chosen = [singletons[rng.randrange(n_kept)] for _ in range(M)]
        replacement = True

    prox_prompts   = [c[0] for c in chosen]
    prox_responses = [c[1] for c in chosen]
    ds_out = ds.add_column("prox_prompt",   prox_prompts)
    ds_out = ds_out.add_column("prox_response", prox_responses)
    ds_out.save_to_disk(output_dir)
    print(f"[Phase 8.5] D_t' singletons attached: |D_t'|={n_kept} (one per prompt), "
          f"paired with {M} d_t_full rows (replacement={'yes' if replacement else 'no'}), "
          f"seed={seed}", flush=True)


def build_step_b_pair_dataset(prompts, completions, K, output_dir):
    """Pair adjacent K candidates into (K/2) preference rows per prompt.

    Each row: {prompt, chosen=z_2k, rejected=z_2k+1}. Used as input to
    precompute_logp (which expects chosen/rejected columns). After precompute
    we have logp values for all K candidates per prompt across the K/2 rows.

    Returns total row count (= n_kept * K/2).
    """
    if K % 2 != 0:
        raise ValueError(f"K must be even for pair-adjacent batching; got K={K}")
    rows = []
    n_dropped = 0
    for prompt, comp_list in zip(prompts, completions):
        # Filter empties (vLLM may produce empty strings on truncation).
        comps = [c for c in comp_list if c]
        if len(comps) < K:
            n_dropped += 1
            continue
        for k in range(0, K, 2):
            rows.append({"prompt": prompt, "chosen": comps[k], "rejected": comps[k + 1]})
    if n_dropped:
        print(f"[Step B 9b-i] dropped {n_dropped} prompts with < {K} non-empty completions",
              flush=True)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(output_dir)
    return ds


def build_step_b_synth_pref(stepb_with_logp_dir, K, s_sign, output_dir):
    """Reshape pair-adjacent rows into per-prompt K-vectors, rank by r,
    pick (chosen=argmax, rejected=argmin), output a synth pref dataset.

    Input dataset has columns:
      prompt, chosen, rejected,
      logp_pi_t_chosen,  logp_pi_t_rejected,
      logp_pi_t1_chosen, logp_pi_t1_rejected.

    Each unique `prompt` has K/2 rows × 2 columns = K candidate logp values
    under each model. We group by prompt, recover the K candidate responses
    and their logp values, compute r = s·(logp_t1 − logp_t) per candidate,
    select chosen=argmax_k r and rejected=argmin_k r, and emit the synth
    preference row with logp_t_chosen / logp_t_rejected attached for the
    Step B trainer.

    Returns the saved Dataset object.
    """
    ds = load_from_disk(stepb_with_logp_dir)
    rows_per_pair = K // 2

    # Group rows by prompt (preserving order). Assume rows are in pair-adjacent
    # order from build_step_b_pair_dataset (i.e., consecutive K/2 rows belong to
    # the same prompt). Verify.
    n_total = len(ds)
    n_groups = n_total // rows_per_pair
    if n_total != n_groups * rows_per_pair:
        raise ValueError(
            f"Step B logp dataset has {n_total} rows, not a multiple of K/2={rows_per_pair}.")

    out_rows = []
    n_skipped_tied = 0
    n_skipped_empty = 0
    for g in range(n_groups):
        base = g * rows_per_pair
        group_rows = [ds[base + i] for i in range(rows_per_pair)]
        prompt = group_rows[0]["prompt"]
        if not all(r["prompt"] == prompt for r in group_rows):
            raise ValueError(
                f"prompts within group {g} not consistent — pair-adjacent assumption broken")

        # Reassemble K responses + K logps per model.
        responses  = []
        logp_t     = []
        logp_t1    = []
        for r in group_rows:
            responses.append(r["chosen"])
            responses.append(r["rejected"])
            logp_t.append(r["logp_pi_t_chosen"])
            logp_t.append(r["logp_pi_t_rejected"])
            logp_t1.append(r["logp_pi_t1_chosen"])
            logp_t1.append(r["logp_pi_t1_rejected"])

        if any(not z for z in responses):
            n_skipped_empty += 1
            continue

        logp_t_arr  = np.asarray(logp_t,  dtype=np.float64)
        logp_t1_arr = np.asarray(logp_t1, dtype=np.float64)
        r_vec = s_sign * (logp_t1_arr - logp_t_arr)

        # Skip if all r values tied (essentially zero probability with float logps).
        if r_vec.max() == r_vec.min():
            n_skipped_tied += 1
            continue

        i_max = int(np.argmax(r_vec))
        i_min = int(np.argmin(r_vec))
        if i_max == i_min:
            n_skipped_tied += 1
            continue

        out_rows.append({
            "prompt":          prompt,
            "chosen":          responses[i_max],
            "rejected":        responses[i_min],
            "logp_t_chosen":   float(logp_t_arr[i_max]),
            "logp_t_rejected": float(logp_t_arr[i_min]),
            "r_chosen":        float(r_vec[i_max]),
            "r_rejected":      float(r_vec[i_min]),
        })

    if n_skipped_tied or n_skipped_empty:
        print(f"[Step B 9b-iv] skipped {n_skipped_tied} tied, {n_skipped_empty} empty groups",
              flush=True)

    ds_out = Dataset.from_list(out_rows)
    ds_out.save_to_disk(output_dir)
    print(f"[Step B 9b-iv] synth pref data: {len(ds_out)} rows "
          f"(s={s_sign:+g}, chosen=argmax_k r, rejected=argmin_k r)", flush=True)
    return ds_out


def main():
    args = parse_args()
    if not (0 < args.tau < args.eta):
        raise ValueError(f"ENPO requires 0 < tau < eta; got tau={args.tau}, eta={args.eta}")
    if args.n_completions < 2 or args.n_completions % 2 != 0:
        raise ValueError(f"--n_completions must be even and ≥ 2; got {args.n_completions}")
    if args.n_pair < 1:
        raise ValueError(f"--n_pair must be ≥ 1; got {args.n_pair}")
    if args.n_z < 1:
        raise ValueError(f"--n_z must be ≥ 1; got {args.n_z}")
    if args.alpha < 0:
        raise ValueError(f"--alpha must be ≥ 0; got {args.alpha}")
    if args.beta <= 0:
        raise ValueError(f"--beta must be > 0; got {args.beta}")
    if args.s_sign not in (+1.0, -1.0):
        raise ValueError(f"--s_sign must be +1 or -1; got {args.s_sign}")
    m_train = args.m_train_per_iter if args.m_train_per_iter is not None else args.num_prompts
    if m_train <= 0:
        raise ValueError(f"--m_train_per_iter must be > 0; got {m_train}")

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pi_ref     = args.base_model
    pi_t       = args.base_model           # iter 1: π_1 = π_ref
    pi_t_hat   = args.base_model           # iter 1: π̂_1 = π_ref

    accumulated_shards = []
    tok = AutoTokenizer.from_pretrained(args.base_model)
    K = args.n_completions   # Step B fresh-π_t pool (Phase 1c) — must be even, >=2
    K_pair = args.n_pair     # Step A pair pools, π_t/π̂_t (Phase 1a/1b) — only 1 sample/pool ever used

    for t in range(1, args.num_iterations + 1):
        iter_dir = os.path.join(args.output_dir, f"iter{t}")
        os.makedirs(iter_dir, exist_ok=True)
        policy_dir    = os.path.join(iter_dir, "policy")
        adversary_dir = os.path.join(iter_dir, "adversary")

        start_idx = (t - 1) * args.num_prompts
        prompts, instructions, msg_lists = load_prompts(
            args.dataset_name, args.num_prompts, tok, start_idx)
        prompts_for_gen, n_trunc = truncate_prompts(prompts, tok, args.max_input_tokens)
        if n_trunc:
            print(f"[Truncate] {n_trunc}/{len(prompts)} prompts > {args.max_input_tokens} "
                  f"tokens; left-truncated to last {args.max_input_tokens}", flush=True)

        print(f"\n{'='*60}")
        print(f"ENPO — iteration {t}/{args.num_iterations}")
        print(f"  policy   π_t   : {pi_t}")
        print(f"  adversary π̂_t : {pi_t_hat}")
        print(f"  ref      π_ref : {pi_ref}     (frozen)")
        print(f"  η = {args.eta}, τ = {args.tau}, α = {args.alpha}  (Step A)")
        print(f"  β = {args.beta}, s = {args.s_sign:+g}             (Step B)")
        print(f"  prompts        : [{start_idx}, {start_idx + args.num_prompts})")
        print(f"  K_pair         : {K_pair}  (π_t/π̂_t pair pools, Phase 1a/1b — 1 random sample used from each)")
        print(f"  K_stepb        : {K}  (Step B fresh-π_t pool, Phase 1c)")
        print(f"  m_train        : {m_train}")
        print(f"  buffer so far  : {len(accumulated_shards)} prior shards")
        print(f"  output         : {iter_dir} (policy/, adversary/)")
        print(f"{'='*60}", flush=True)

        # ──────────────────────────────────────────────────────────────────
        # Phase 1a — gen K_pair from π_t (for pair y_i + D_t' source)
        # ──────────────────────────────────────────────────────────────────
        completions_pi_file = os.path.join(iter_dir, "completions_pi.json")
        if os.path.exists(completions_pi_file):
            print(f"[Phase 1a] cached completions_pi.json found → skipping vLLM")
        else:
            run_vllm_generate(
                model=pi_t, prompts=prompts_for_gen, iter_dir=iter_dir,
                n=K_pair,
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
        with open(completions_pi_file) as f:
            completions_pi = json.load(f)

        # ──────────────────────────────────────────────────────────────────
        # Phase 1b — gen K_pair from π̂_t (for adversary pool). Always runs in
        # single-pair mode, including at iter 1 where π̂_1 = π_t = π_ref
        # (gen produces independent iid samples — same distribution, different
        # RNG). The Phase 2 pair will be (1 random sample from π_t pool,
        # 1 random sample from π̂_t pool); roles are fixed by origin
        # (chosen = y ~ π_t, rejected = y' ~ π̂_t per spec lines 8-9).
        # ──────────────────────────────────────────────────────────────────
        completions_advs_file = os.path.join(iter_dir, "completions_advs.json")
        if os.path.exists(completions_advs_file):
            print(f"[Phase 1b] cached completions_advs.json found → skipping vLLM")
        else:
            run_vllm_generate(
                model=pi_t_hat, prompts=prompts_for_gen, iter_dir=iter_dir,
                n=K_pair,
                temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                tp_size=args.vllm_tp_size,
                gpu_mem_util=args.vllm_gpu_mem_util,
                output_filename="completions_advs.json",
                input_filename="vllm_input_advs.json",
                tokenizer=args.base_model,
                seed=args.seed + 100 * t + 1,
            )

        # ──────────────────────────────────────────────────────────────────
        # Phase 1c — fresh K from π_t (for Step B candidate pool)
        # ──────────────────────────────────────────────────────────────────
        completions_pi_fresh_file = os.path.join(iter_dir, "completions_pi_fresh.json")
        if os.path.exists(completions_pi_fresh_file):
            print(f"[Phase 1c] cached completions_pi_fresh.json found → skipping vLLM")
        else:
            run_vllm_generate(
                model=pi_t, prompts=prompts_for_gen, iter_dir=iter_dir,
                n=K,
                temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                tp_size=args.vllm_tp_size,
                gpu_mem_util=args.vllm_gpu_mem_util,
                output_filename="completions_pi_fresh.json",
                input_filename="vllm_input_pi_fresh.json",
                tokenizer=args.base_model,
                seed=args.seed + 100 * t + 2,
            )
        with open(completions_pi_fresh_file) as f:
            completions_pi_fresh = json.load(f)

        # ──────────────────────────────────────────────────────────────────
        # Phase 2 — SINGLE-PAIR MODE (spec-faithful, algos/specs/enpo.md lines 8-9):
        # Pair role is fixed by sample origin, NOT by Skywork comparison:
        #   chosen   = y_i  ~ π_t   (1 random sample from π_t pool)
        #   rejected = y'_i ~ π̂_t   (1 random sample from π̂_t pool)
        # No Skywork pass at this stage. Phase 7 triple-sample judge sets the
        # signed target (I^y − I^y')/η, which can be either sign depending on
        # which policy made the better sample for each prompt.
        # ──────────────────────────────────────────────────────────────────
        shard_raw_dir = os.path.join(iter_dir, "buffer_shard_raw")
        if os.path.exists(shard_raw_dir):
            print(f"[Phase 2] cached buffer shard found → skipping")
        else:
            with open(completions_pi_file) as f:
                comps_pi = json.load(f)
            with open(completions_advs_file) as f:
                comps_advs = json.load(f)
            assert len(comps_pi) == len(comps_advs) == len(prompts), \
                f"len mismatch: pi={len(comps_pi)} advs={len(comps_advs)} prompts={len(prompts)}"
            rng_merge = random.Random(args.sample_seed * 1000 + t)
            rows = []
            for samples_pi, samples_advs, prompt, msgs in zip(
                comps_pi, comps_advs, prompts, msg_lists
            ):
                idx_pi = rng_merge.randrange(len(samples_pi))
                idx_advs = rng_merge.randrange(len(samples_advs))
                rows.append({
                    "prompt":        prompt,
                    "chosen":        samples_pi[idx_pi],   # = y  (from π_t)
                    "rejected":      samples_advs[idx_advs],  # = y' (from π̂_t)
                    "messages_json": json.dumps(msgs),
                })
            shard = Dataset.from_list(rows)
            shard.save_to_disk(shard_raw_dir)
            print(f"[Phase 2] iter {t} buffer shard built (single-pair, spec-faithful): "
                  f"{len(shard)} rows (= {len(prompts)} prompts × 1 pair; "
                  f"chosen=random π_t sample, rejected=random π̂_t sample)", flush=True)

        # ──────────────────────────────────────────────────────────────────
        # Phase 3 — precompute logp_ref on shard
        # ──────────────────────────────────────────────────────────────────
        shard_with_ref_dir = os.path.join(iter_dir, "buffer_shard_with_ref")
        if os.path.exists(shard_with_ref_dir):
            print(f"[Phase 3] cached logp_ref shard found → skipping precompute")
        else:
            run_precompute_logp(
                model_path=pi_ref,
                dataset_dir=shard_raw_dir,
                output_dataset_dir=shard_with_ref_dir,
                logp_chosen_field="logp_ref_chosen",
                logp_rejected_field="logp_ref_rejected",
                max_length=args.max_length,
            )
        accumulated_shards.append(shard_with_ref_dir)

        # ──────────────────────────────────────────────────────────────────
        # Phase 4 — concat shards → cum_buffer (or use current shard only if --no_buffer)
        # ──────────────────────────────────────────────────────────────────
        if args.no_buffer:
            cum_buffer = load_from_disk(accumulated_shards[-1])
            print(f"[Phase 4] no_buffer mode: cum_buffer = current shard only "
                  f"({len(cum_buffer)} pair rows)", flush=True)
        elif len(accumulated_shards) == 1:
            cum_buffer = load_from_disk(accumulated_shards[0])
            print(f"[Phase 4] cum_buffer: {len(cum_buffer)} pair rows "
                  f"({len(accumulated_shards)} shard(s) concatenated)", flush=True)
        else:
            cum_buffer = concatenate_datasets(
                [load_from_disk(p) for p in accumulated_shards])
            print(f"[Phase 4] cum_buffer: {len(cum_buffer)} pair rows "
                  f"({len(accumulated_shards)} shard(s) concatenated)", flush=True)

        # ──────────────────────────────────────────────────────────────────
        # Phase 5 — sample m_train rows from cum_buffer → D_t_pairs
        # ──────────────────────────────────────────────────────────────────
        d_t_pairs_dir = os.path.join(iter_dir, "d_t_pairs_sampled")
        if os.path.exists(d_t_pairs_dir):
            print(f"[Phase 5] cached sampled D_t pairs found → skipping sample")
            d_t_pairs = load_from_disk(d_t_pairs_dir)
        else:
            seed_t = args.sample_seed + t
            shuffled = cum_buffer.shuffle(seed=seed_t)
            if len(shuffled) >= m_train:
                d_t_pairs = shuffled.select(range(m_train))
            else:
                rng = random.Random(seed_t)
                idx = [rng.randrange(len(shuffled)) for _ in range(m_train)]
                d_t_pairs = shuffled.select(idx)
            d_t_pairs.save_to_disk(d_t_pairs_dir)
            print(f"[Phase 5] sampled {len(d_t_pairs)} rows from cum_buffer "
                  f"(seed={seed_t}, replacement={'no' if len(shuffled) >= m_train else 'yes'})",
                  flush=True)

        # ──────────────────────────────────────────────────────────────────
        # Phase 6 — generate n_z reference samples z from π_t
        # ──────────────────────────────────────────────────────────────────
        z_completions_file = os.path.join(iter_dir, "completions_z.json")
        d_t_with_z_dir = os.path.join(iter_dir, "d_t_with_z")
        if os.path.exists(d_t_with_z_dir):
            print(f"[Phase 6] cached D_t-with-z found → skipping z generation")
        else:
            if not os.path.exists(z_completions_file):
                raw_z_prompts = list(d_t_pairs["prompt"])
                z_prompts, n_trunc_z = truncate_prompts(
                    raw_z_prompts, tok, args.max_input_tokens)
                if n_trunc_z:
                    print(f"[Phase 6] {n_trunc_z}/{len(raw_z_prompts)} z-prompts truncated",
                          flush=True)
                run_vllm_generate(
                    model=pi_t, prompts=z_prompts, iter_dir=iter_dir,
                    n=args.n_z,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                    tp_size=args.vllm_tp_size,
                    gpu_mem_util=args.vllm_gpu_mem_util,
                    output_filename="completions_z.json",
                    input_filename="vllm_input_z.json",
                    tokenizer=args.base_model,
                    seed=args.seed + 100 * t + 3,
                )
            with open(z_completions_file) as f:
                z_completions = json.load(f)
            zs = []
            for comp_list in z_completions:
                comp_list = list(comp_list) if comp_list else []
                if len(comp_list) < args.n_z:
                    comp_list = comp_list + [""] * (args.n_z - len(comp_list))
                zs.append(comp_list[:args.n_z])
            if len(zs) != len(d_t_pairs):
                raise RuntimeError(
                    f"z gen produced {len(zs)} rows but D_t has {len(d_t_pairs)}")
            d_t_with_z = d_t_pairs.add_column("z", zs)
            d_t_with_z.save_to_disk(d_t_with_z_dir)
            print(f"[Phase 6] {args.n_z} z's generated for {len(zs)} rows", flush=True)

        # ──────────────────────────────────────────────────────────────────
        # Phase 7 — triple-sample judge
        # ──────────────────────────────────────────────────────────────────
        d_t_scored_dir = os.path.join(iter_dir, "d_t_scored")
        if os.path.exists(d_t_scored_dir):
            print(f"[Phase 7] cached scored D_t found → skipping judge")
        else:
            run_score_enpo(
                triples_dataset_dir=d_t_with_z_dir,
                output_dataset_dir=d_t_scored_dir,
                judge_model=args.judge_model,
                judge_batch_size=args.judge_batch_size,
                judge_max_length=args.judge_max_length,
            )

        # ──────────────────────────────────────────────────────────────────
        # Phase 8 — precompute logp_t on the m sampled rows
        # ──────────────────────────────────────────────────────────────────
        d_t_full_dir = os.path.join(iter_dir, "d_t_full")
        if os.path.exists(d_t_full_dir):
            print(f"[Phase 8] cached full-logp D_t found → skipping")
        elif t == 1:
            print(f"[Phase 8] iter 1: π_t = π_ref → copying logp_ref_* into logp_t_*",
                  flush=True)
            ds = load_from_disk(d_t_scored_dir)
            ds = ds.add_column("logp_t_chosen",   list(ds["logp_ref_chosen"]))
            ds = ds.add_column("logp_t_rejected", list(ds["logp_ref_rejected"]))
            ds.save_to_disk(d_t_full_dir)
        else:
            run_precompute_logp(
                model_path=pi_t,
                dataset_dir=d_t_scored_dir,
                output_dataset_dir=d_t_full_dir,
                logp_chosen_field="logp_t_chosen",
                logp_rejected_field="logp_t_rejected",
                max_length=args.max_length,
            )

        # ──────────────────────────────────────────────────────────────────
        # Phase 8.5 — D_t' = {(x_i, y_i)} where y_i ~ π_t (spec line 11).
        # Single-pair mode: y_i = the random π_t sample for each prompt
        # already saved in this iter's buffer_shard.chosen column.
        # ──────────────────────────────────────────────────────────────────
        d_t_full_prox_dir = os.path.join(iter_dir, "d_t_full_prox")
        if alpha_active := (args.alpha > 0):
            if os.path.exists(d_t_full_prox_dir):
                print(f"[Phase 8.5] cached d_t_full_prox found → skipping")
            else:
                cur_shard = load_from_disk(shard_raw_dir)
                prox_prompts_in   = list(cur_shard["prompt"])
                prox_responses_in = list(cur_shard["chosen"])    # = y_i (random π_t sample)
                attach_prox_singletons(
                    d_t_full_dir=d_t_full_dir,
                    prox_prompts_in=prox_prompts_in,
                    prox_responses_in=prox_responses_in,
                    seed=args.sample_seed + 1000 + t,
                    output_dir=d_t_full_prox_dir,
                )
            train_data_dir = d_t_full_prox_dir
        else:
            print(f"[Phase 8.5] α=0 → skipping D_t' build; trainer uses d_t_full directly")
            train_data_dir = d_t_full_dir

        # ──────────────────────────────────────────────────────────────────
        # Phase 9a — Step A trainer (OMD + proximal loss) → policy_dir
        # ──────────────────────────────────────────────────────────────────
        if os.path.exists(os.path.join(policy_dir, "model.safetensors")) or \
           os.path.exists(os.path.join(policy_dir, "model.safetensors.index.json")):
            print(f"[Phase 9a] cached policy checkpoint found at {policy_dir} → skipping Step A trainer")
        else:
            run_train_enpo_step_a(
                policy_path=pi_t,
                preference_data_dir=train_data_dir,
                output_dir=policy_dir,
                eta=args.eta,
                tau=args.tau,
                alpha=args.alpha,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_train_epochs,
                max_length=args.max_length,
                wandb_run_name=f"enpo-stepA-iter{t}-{Path(args.output_dir).name}",
                seed=args.seed,
            )

        # ──────────────────────────────────────────────────────────────────
        # Phase 9b — Step B
        # ──────────────────────────────────────────────────────────────────
        # 9b-i: pair-adjacent dataset from completions_pi_fresh
        stepb_pair_dir = os.path.join(iter_dir, "stepb_pair_raw")
        if os.path.exists(stepb_pair_dir):
            print(f"[Phase 9b-i] cached stepb_pair_raw found → skipping")
        else:
            ds_pair = build_step_b_pair_dataset(
                prompts=prompts, completions=completions_pi_fresh,
                K=K, output_dir=stepb_pair_dir,
            )
            print(f"[Phase 9b-i] step-B pair-adjacent dataset: {len(ds_pair)} rows "
                  f"(= n_kept × K/2)", flush=True)

        # 9b-ii: precompute logp under π_t (or copy logp_ref at iter 1)
        stepb_with_logp_t_dir = os.path.join(iter_dir, "stepb_with_logp_t")
        if os.path.exists(stepb_with_logp_t_dir):
            print(f"[Phase 9b-ii] cached stepb_with_logp_t found → skipping")
        elif t == 1:
            print(f"[Phase 9b-ii] iter 1: π_t = π_ref → precompute under π_ref",
                  flush=True)
            run_precompute_logp(
                model_path=pi_ref,
                dataset_dir=stepb_pair_dir,
                output_dataset_dir=stepb_with_logp_t_dir,
                logp_chosen_field="logp_pi_t_chosen",
                logp_rejected_field="logp_pi_t_rejected",
                max_length=args.max_length,
            )
        else:
            run_precompute_logp(
                model_path=pi_t,
                dataset_dir=stepb_pair_dir,
                output_dataset_dir=stepb_with_logp_t_dir,
                logp_chosen_field="logp_pi_t_chosen",
                logp_rejected_field="logp_pi_t_rejected",
                max_length=args.max_length,
            )

        # 9b-iii: precompute logp under π_{t+1} (= policy_dir from Step A)
        stepb_with_logp_t1_dir = os.path.join(iter_dir, "stepb_with_logp_t1")
        if os.path.exists(stepb_with_logp_t1_dir):
            print(f"[Phase 9b-iii] cached stepb_with_logp_t1 found → skipping")
        else:
            run_precompute_logp(
                model_path=policy_dir,
                dataset_dir=stepb_with_logp_t_dir,
                output_dataset_dir=stepb_with_logp_t1_dir,
                logp_chosen_field="logp_pi_t1_chosen",
                logp_rejected_field="logp_pi_t1_rejected",
                max_length=args.max_length,
            )

        # 9b-iv: rank, build synth pref dataset
        stepb_synth_dir = os.path.join(iter_dir, "stepb_synth_pref")
        if os.path.exists(stepb_synth_dir):
            print(f"[Phase 9b-iv] cached stepb_synth_pref found → skipping")
        else:
            build_step_b_synth_pref(
                stepb_with_logp_dir=stepb_with_logp_t1_dir,
                K=K, s_sign=args.s_sign,
                output_dir=stepb_synth_dir,
            )

        # 9b-v: Step B trainer
        if os.path.exists(os.path.join(adversary_dir, "model.safetensors")) or \
           os.path.exists(os.path.join(adversary_dir, "model.safetensors.index.json")):
            print(f"[Phase 9b-v] cached adversary checkpoint found at {adversary_dir} → skipping Step B trainer")
        else:
            run_train_enpo_step_b(
                policy_path=pi_t_hat,
                preference_data_dir=stepb_synth_dir,
                output_dir=adversary_dir,
                beta=args.beta,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_train_epochs,
                max_length=args.max_length,
                wandb_run_name=f"enpo-stepB-iter{t}-{Path(args.output_dir).name}",
                seed=args.seed,
            )

        # Update both checkpoints for next iter
        pi_t     = policy_dir
        pi_t_hat = adversary_dir
        print(f"\nIteration {t} done → policy={policy_dir}, adversary={adversary_dir}  "
              f"(buffer: {len(accumulated_shards)} shards)", flush=True)

        if args.post_iter_hook:
            cmd = args.post_iter_hook.format(iter_dir=policy_dir, iter=t)
            print(f"\n[post_iter_hook] running: {cmd}", flush=True)
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"[post_iter_hook] WARNING: hook exited {result.returncode} "
                      f"(e.g. eval API credits/errors) -- continuing training "
                      f"regardless; rerun the eval separately once resolved.",
                      flush=True)

    print(f"\nENPO complete. Final policy: {pi_t}, final adversary: {pi_t_hat}")


if __name__ == "__main__":
    main()
