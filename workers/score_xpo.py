#!/usr/bin/env python3
"""
XPO score worker. Scores 8 completions from π^(t) AND 8 completions from π_ref
per prompt, then picks the **max-margin cross-side pair** per prompt — i.e.,
the (a, b) pair where a is from π^(t), b is from π_ref, and |score(a) − score(b)|
is largest. The π_ref-sampled element of that pair is τ̃ (kept tagged).

Run via:

    accelerate launch --num_processes 8 workers/score_xpo.py \
        --judge_model    Skywork/...                      \
        --prompts_file   iter_dir/prompts.json            \
        --completions_pi_file  iter_dir/completions_pi.json   \
        --completions_ref_file iter_dir/completions_ref.json  \
        --output_dir     iter_dir                         \
        --judge_batch_size 8 --judge_max_length 4096      \
        --margin_keep_pct 0.7

Output (under --output_dir):
    preference_data/    HF dataset with columns:
        prompt, chosen, rejected, ref_is_chosen, margin
    judge_stats.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup_gpu(*objects):
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_model",            required=True)
    p.add_argument("--prompts_file",           required=True)
    p.add_argument("--completions_pi_file",    required=True,
                   help="Completions from π^(t)")
    p.add_argument("--completions_ref_file",   required=True,
                   help="Completions from π_ref")
    p.add_argument("--output_dir",             required=True)
    p.add_argument("--judge_batch_size",       type=int,   default=8)
    p.add_argument("--judge_max_length",       type=int,   default=4096)
    p.add_argument("--margin_keep_pct",        type=float, default=0.7)
    return p.parse_args()


def score_all_completions(args, message_lists, completions_pi, completions_ref, accelerator):
    """Returns scores_pi and scores_ref, each list[N][K] in original order."""
    if accelerator.is_main_process:
        print(f"\n[Judge] Loading {args.judge_model} on {accelerator.num_processes} ranks", flush=True)

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

    # Flatten: for each assigned prompt, score 8 from π and 8 from π_ref.
    flat = []  # (i, source, k, formatted_text); source ∈ {"pi", "ref"}
    for i in my_indices:
        msgs = message_lists[i]
        for k, comp in enumerate(completions_pi[i]):
            full = msgs + [{"role": "assistant", "content": comp}]
            text = judge_tokenizer.apply_chat_template(full, tokenize=False)
            flat.append((i, "pi", k, text))
        for k, comp in enumerate(completions_ref[i]):
            full = msgs + [{"role": "assistant", "content": comp}]
            text = judge_tokenizer.apply_chat_template(full, tokenize=False)
            flat.append((i, "ref", k, text))

    flat_scores = [0.0] * len(flat)
    for batch_start in range(0, len(flat), args.judge_batch_size):
        batch = flat[batch_start : batch_start + args.judge_batch_size]
        texts = [t for _, _, _, t in batch]
        enc = judge_tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.judge_max_length,
        ).to(accelerator.device)
        with torch.no_grad():
            logits = judge_model(**enc).logits.squeeze(-1)
        for j, score in enumerate(logits.float().cpu().tolist()):
            flat_scores[batch_start + j] = score

    cleanup_gpu(judge_model, judge_tokenizer)

    Kpi  = len(completions_pi[0])  if completions_pi  else 0
    Kref = len(completions_ref[0]) if completions_ref else 0
    Ktotal = Kpi + Kref

    local = []
    for offset, i in enumerate(my_indices):
        block = flat_scores[offset * Ktotal : (offset + 1) * Ktotal]
        local.append((i, block[:Kpi], block[Kpi:]))

    all_paired = sorted(gather_object(local), key=lambda x: x[0])
    scores_pi  = [b for _, b, _ in all_paired]
    scores_ref = [b for _, _, b in all_paired]
    return scores_pi, scores_ref


def build_xpo_pairs(prompts, comps_pi, comps_ref, scs_pi, scs_ref, keep_pct):
    """For each prompt, find the cross-side pair (one π element + one π_ref element)
    with the largest |score margin|. Tag whichever is from π_ref as τ̃."""
    pairs = []
    skipped = 0
    for prompt, cp, cr, sp, sr in zip(prompts, comps_pi, comps_ref, scs_pi, scs_ref):
        if not sp or not sr:
            skipped += 1
            continue
        # Find max-|diff| over Kpi × Kref cross-pairs.
        sp_arr = np.asarray(sp, dtype=np.float32)
        sr_arr = np.asarray(sr, dtype=np.float32)
        diff = sp_arr[:, None] - sr_arr[None, :]            # (Kpi, Kref)
        idx = np.argmax(np.abs(diff))
        i_pi, i_ref = np.unravel_index(idx, diff.shape)
        d = float(diff[i_pi, i_ref])
        if d == 0.0:                                         # ties → skip
            skipped += 1
            continue
        if d > 0:
            chosen, rejected = cp[i_pi], cr[i_ref]
            ref_is_chosen = False                            # τ̃ is the rejected (π_ref) one
            margin = d
        else:
            chosen, rejected = cr[i_ref], cp[i_pi]
            ref_is_chosen = True                             # τ̃ is the chosen (π_ref) one
            margin = -d
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "ref_is_chosen": bool(ref_is_chosen),
            "margin": float(margin),
        })

    n_before = len(pairs)
    if 0.0 < keep_pct < 1.0 and pairs:
        pairs.sort(key=lambda x: x["margin"], reverse=True)
        pairs = pairs[: max(1, int(round(len(pairs) * keep_pct)))]
    n_after = len(pairs)

    margins = [p["margin"] for p in pairs]
    refs_chosen = sum(1 for p in pairs if p["ref_is_chosen"])
    stats = {
        "n_input":          len(prompts),
        "n_skipped_tied":   skipped,
        "n_before_filter":  n_before,
        "n_after_filter":   n_after,
        "ref_is_chosen_count": refs_chosen,
        "ref_is_rejected_count": n_after - refs_chosen,
        "margin_min":       float(min(margins)) if margins else 0.0,
        "margin_max":       float(max(margins)) if margins else 0.0,
        "margin_mean":      float(np.mean(margins)) if margins else 0.0,
        "margin_threshold": float(min(margins)) if margins else 0.0,
    }
    records = [
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"],
         "ref_is_chosen": p["ref_is_chosen"]}
        for p in pairs
    ]
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
    prompts = pdata["prompts"]
    message_lists = pdata["message_lists"]

    # vLLM worker writes a bare list-of-lists: [[k0,k1,..], [..], ...]
    with open(args.completions_pi_file) as f:
        comps_pi = json.load(f)
    with open(args.completions_ref_file) as f:
        comps_ref = json.load(f)

    if accelerator.is_main_process:
        Kpi  = len(comps_pi[0])  if comps_pi  else 0
        Kref = len(comps_ref[0]) if comps_ref else 0
        print(f"[Loaded] {len(prompts)} prompts × Kπ={Kpi} + Kref={Kref}", flush=True)

    scs_pi, scs_ref = score_all_completions(
        args, message_lists, comps_pi, comps_ref, accelerator)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        ds, stats = build_xpo_pairs(
            prompts, comps_pi, comps_ref, scs_pi, scs_ref, args.margin_keep_pct)
        out_dataset_dir = os.path.join(args.output_dir, "preference_data")
        ds.save_to_disk(out_dataset_dir)
        with open(os.path.join(args.output_dir, "judge_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(
            f"[Done] {stats['n_input']} prompts → {stats['n_before_filter']} pairs "
            f"→ {stats['n_after_filter']} after margin filter "
            f"(τ̃ chosen={stats['ref_is_chosen_count']}, "
            f"τ̃ rejected={stats['ref_is_rejected_count']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
