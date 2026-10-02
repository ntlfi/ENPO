#!/usr/bin/env python3
"""
Scalar-reward judge worker. Run via:

    accelerate launch --num_processes 8 workers/score.py \
        --judge_model Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M \
        --prompts_file iter_dir/prompts.json \
        --completions_file iter_dir/completions.json \
        --output_dir iter_dir \
        --margin_keep_pct 0.7

Produces:
    iter_dir/preference_data/   HF dataset with {prompt, chosen, rejected}
    iter_dir/judge_stats.json   counts + margin distribution

This worker constructs the preference pairs used by the INPO runner.
"""
import argparse
import gc
import json
import os
import sys
from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make `import src.*` work when invoked as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup_gpu(*objects):
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_model",        required=True)
    p.add_argument("--prompts_file",       required=True)
    p.add_argument("--completions_file",   required=True)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--judge_batch_size",   type=int,   default=8)
    p.add_argument("--judge_max_length",   type=int,   default=4096)
    p.add_argument("--margin_keep_pct",    type=float, default=0.7)
    return p.parse_args()


def score_with_reward_model(message_lists, completions, args, accelerator):
    """Sharded scalar-reward scoring of K completions per prompt.

    Returns scores: list[len(message_lists)][K] of floats in original order.
    """
    if accelerator.is_main_process:
        print(f"\n[Judge] Loading {args.judge_model} on {accelerator.num_processes} ranks")

    rank  = accelerator.process_index
    world = accelerator.num_processes
    my_indices = list(range(rank, len(message_lists), world))

    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    judge_model = AutoModelForSequenceClassification.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16,
        num_labels=1,
        attn_implementation="sdpa",
    ).to(accelerator.device)
    judge_model.eval()

    flat = []  # (i, k, formatted_text)
    for i in my_indices:
        msgs = message_lists[i]
        for k, comp in enumerate(completions[i]):
            full = msgs + [{"role": "assistant", "content": comp}]
            text = judge_tokenizer.apply_chat_template(full, tokenize=False)
            flat.append((i, k, text))

    flat_scores = [0.0] * len(flat)
    for batch_start in range(0, len(flat), args.judge_batch_size):
        batch = flat[batch_start : batch_start + args.judge_batch_size]
        texts = [t for _, _, t in batch]
        enc = judge_tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.judge_max_length,
        ).to(accelerator.device)
        with torch.no_grad():
            logits = judge_model(**enc).logits.squeeze(-1)
        for j, score in enumerate(logits.float().cpu().tolist()):
            flat_scores[batch_start + j] = score

    cleanup_gpu(judge_model, judge_tokenizer)

    K = len(completions[0]) if completions else 0
    local = []
    for offset, i in enumerate(my_indices):
        scores_K = flat_scores[offset * K : (offset + 1) * K]
        local.append((i, scores_K))

    all_paired = sorted(gather_object(local), key=lambda x: x[0])
    return [s for _, s in all_paired]


def build_dataset_from_scores(prompts, completions, scores, keep_pct):
    """Build (prompt, chosen, rejected) pairs from K-score lists.
    Drop bottom (1 - keep_pct) by margin."""
    pairs = []
    skipped_tied = 0
    for prompt, comps, scs in zip(prompts, completions, scores):
        if not scs or len(set(scs)) < 2:
            skipped_tied += 1
            continue
        ci = int(np.argmax(scs))
        ri = int(np.argmin(scs))
        if ci == ri:
            skipped_tied += 1
            continue
        margin = float(scs[ci] - scs[ri])
        pairs.append({
            "prompt":   prompt,
            "chosen":   comps[ci],
            "rejected": comps[ri],
            "margin":   margin,
        })

    n_before = len(pairs)
    if 0.0 < keep_pct < 1.0 and pairs:
        pairs.sort(key=lambda x: x["margin"], reverse=True)
        pairs = pairs[:max(1, int(round(len(pairs) * keep_pct)))]
    n_after = len(pairs)

    margins = [p["margin"] for p in pairs]
    stats = {
        "n_input":          len(prompts),
        "n_skipped_tied":   skipped_tied,
        "n_before_filter":  n_before,
        "n_after_filter":   n_after,
        "margin_min":       float(min(margins)) if margins else 0.0,
        "margin_max":       float(max(margins)) if margins else 0.0,
        "margin_mean":      float(np.mean(margins)) if margins else 0.0,
        "margin_threshold": float(min(margins)) if margins else 0.0,
    }
    records = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
               for p in pairs]
    return Dataset.from_list(records), stats


def main():
    args = parse_args()
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs])

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    with open(args.prompts_file) as f:
        pdata = json.load(f)
    prompts       = pdata["prompts"]
    message_lists = pdata["message_lists"]
    with open(args.completions_file) as f:
        completions = json.load(f)

    K = len(completions[0]) if completions else 0
    if accelerator.is_main_process:
        print(f"[Loaded] {len(prompts)} prompts × K={K} completions")

    scores = score_with_reward_model(message_lists, completions, args, accelerator)
    dataset, stats = build_dataset_from_scores(
        prompts, completions, scores, args.margin_keep_pct)

    if accelerator.is_main_process:
        print(f"[Dataset] {stats['n_input']} prompts → "
              f"{stats['n_before_filter']} pairs after dedup ({stats['n_skipped_tied']} tied) → "
              f"{stats['n_after_filter']} pairs after margin filter "
              f"(keep top {args.margin_keep_pct:.0%})")
        print(f"[Margin]  min={stats['margin_min']:.3f}  "
              f"mean={stats['margin_mean']:.3f}  "
              f"max={stats['margin_max']:.3f}  "
              f"kept-threshold={stats['margin_threshold']:.3f}")
        dataset.save_to_disk(os.path.join(args.output_dir, "preference_data"))
        with open(os.path.join(args.output_dir, "judge_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
