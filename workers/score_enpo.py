#!/usr/bin/env python3
"""
ENPO triple-sample preference judge worker.

For each training row (prompt x, response y, response y', reference list z[0..n_z-1]),
we produce two preference labels averaged over the n_z reference samples:
    label_chosen   = mean_j 1[ r(y)  > r(z_j) | x ]   ∈ [0, 1]
    label_rejected = mean_j 1[ r(y') > r(z_j) | x ]   ∈ [0, 1]

With n_z=1 the labels are {0,1}-valued (the original spec). With n_z>1, each
label is a Monte-Carlo average and the per-pair signed target
(label_chosen - label_rejected)/η is a variance-reduced estimator of
(p*(y≻π_t) - p*(y'≻π_t))/η. The squared-loss decomposition still gives an
unbiased population objective because h_t doesn't depend on z.

r(·|x) is the scalar reward from the Skywork-Reward-V2 judge applied to the
chat-templated dialog. (Names "chosen"/"rejected" carry no preference ordering
— they retain the pair roles y from the policy and y' from the adversary.)

This is structurally different from workers/score.py (best-of-K vs worst-of-K
ranking) and workers/score_xpo.py (max-margin cross-side pair selection).
Here we do NOT pick a winner among K samples; we score (2 + n_z) specific
responses per training row and threshold differences against each reference.

Run via:

    accelerate launch --num_processes 8 workers/score_enpo.py \
        --judge_model       Skywork/...                          \
        --triples_dataset_dir   iter_dir/d_t_with_z              \
        --output_dataset_dir    iter_dir/d_t_scored              \
        --judge_batch_size 8 --judge_max_length 4096

Input dataset columns (built by the runner):
    prompt          (str)         — chat-templated, for inspection
    chosen          (str)         — y (a policy sample)
    rejected        (str)         — y' (an adversary sample)
    z               (List[str])   — fresh reference responses from π_t this iter,
                                    length = n_z (≥ 1)
    messages_json   (str)         — JSON-serialized message_list, used by judge
                                    tokenizer to apply the reward model's template

Output dataset (saved to --output_dataset_dir): same columns plus
    label_chosen     (float32, in [0, 1])
    label_rejected   (float32, in [0, 1])
    r_chosen         (float32 — reward of y)
    r_rejected       (float32 — reward of y')
    r_z              (List[float32] — rewards of each z_j; length n_z)

Also writes <output_dataset_dir>/judge_stats.json with label distributions and
per-response reward stats.
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
from datasets import load_from_disk
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
    p.add_argument("--judge_model",          required=True)
    p.add_argument("--triples_dataset_dir",  required=True,
                   help="HF dataset with prompt, chosen (=y), rejected (=y'), z, messages_json")
    p.add_argument("--output_dataset_dir",   required=True)
    p.add_argument("--judge_batch_size",     type=int, default=8)
    p.add_argument("--judge_max_length",     type=int, default=4096)
    return p.parse_args()


def score_triples(args, ds, accelerator, n_z):
    """Return r_y, r_yp lists (length N) and r_z lists-of-lists (N × n_z).

    Each row contributes (2 + n_z) responses to score: y, y', and z[0..n_z-1].
    Sharded across ranks; each rank's forward passes touch ~N(2+n_z)/world rows.
    """
    if accelerator.is_main_process:
        print(f"\n[Judge] Loading {args.judge_model} on {accelerator.num_processes} ranks "
              f"(n_z={n_z} → {2 + n_z} scores/row)", flush=True)

    rank  = accelerator.process_index
    world = accelerator.num_processes
    N     = len(ds)
    my_indices = list(range(rank, N, world))

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

    # Flatten (2 + n_z) responses per row in stable (i, slot) order.
    # slot ∈ {"y", "yp", "z0", ..., "z{n_z-1}"}; re-assembled after scoring.
    flat = []
    for i in my_indices:
        row = ds[i]
        msgs = json.loads(row["messages_json"])
        z_list = row["z"]
        if not isinstance(z_list, (list, tuple)):
            # Back-compat: if upstream stored z as a string (legacy n_z=1).
            z_list = [z_list]
        if len(z_list) != n_z:
            raise ValueError(
                f"row {i}: z has length {len(z_list)} but --n_z={n_z}")
        slots = [("y", row["chosen"]), ("yp", row["rejected"])]
        slots.extend((f"z{j}", z_list[j]) for j in range(n_z))
        for slot, resp in slots:
            full = msgs + [{"role": "assistant", "content": resp}]
            text = judge_tokenizer.apply_chat_template(full, tokenize=False)
            flat.append((i, slot, text))

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

    # Re-assemble per-row tuples. (2 + n_z) slots in fixed order y, yp, z0..z_{n_z-1}.
    stride = 2 + n_z
    local = []
    for offset, i in enumerate(my_indices):
        chunk = flat_scores[offset * stride : (offset + 1) * stride]
        r_y, r_yp = chunk[0], chunk[1]
        r_z_vec  = chunk[2:]
        local.append((i, r_y, r_yp, r_z_vec))

    all_paired = sorted(gather_object(local), key=lambda x: x[0])
    r_y_list   = [t[1] for t in all_paired]
    r_yp_list  = [t[2] for t in all_paired]
    r_z_lists  = [t[3] for t in all_paired]   # list of length-n_z lists
    return r_y_list, r_yp_list, r_z_lists


def main():
    args = parse_args()
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs])

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output_dataset_dir.rstrip("/")) or ".",
                    exist_ok=True)
    accelerator.wait_for_everyone()

    ds = load_from_disk(args.triples_dataset_dir)
    required = {"prompt", "chosen", "rejected", "z", "messages_json"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset {args.triples_dataset_dir} missing columns: {missing}")

    # Infer n_z from the first row's z (List[str]) length.
    first_z = ds[0]["z"]
    if isinstance(first_z, (list, tuple)):
        n_z = len(first_z)
    else:
        n_z = 1   # legacy: scalar string
    if n_z < 1:
        raise ValueError(f"Empty z list at row 0; n_z must be ≥ 1")

    if accelerator.is_main_process:
        print(f"[score_enpo] {len(ds)} rows × (2 + n_z={n_z}) scores, "
              f"judge={args.judge_model}", flush=True)

    r_y, r_yp, r_z_lists = score_triples(args, ds, accelerator, n_z)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        r_y_arr  = np.asarray(r_y,  dtype=np.float32)             # (N,)
        r_yp_arr = np.asarray(r_yp, dtype=np.float32)             # (N,)
        r_z_arr  = np.asarray(r_z_lists, dtype=np.float32)        # (N, n_z)

        # Strict > thresholding per (row, j); ties → 0. Average across z's gives
        # label_chosen, label_rejected ∈ [0, 1] (variance-reduced when n_z>1).
        wins_y  = (r_y_arr[:, None]  > r_z_arr).astype(np.float32)  # (N, n_z)
        wins_yp = (r_yp_arr[:, None] > r_z_arr).astype(np.float32)
        label_y  = wins_y.mean(axis=1)    # (N,) ∈ [0, 1]
        label_yp = wins_yp.mean(axis=1)

        ds_out = ds.add_column("label_chosen",    [float(v) for v in label_y])
        ds_out = ds_out.add_column("label_rejected", [float(v) for v in label_yp])
        ds_out = ds_out.add_column("r_chosen",   [float(s) for s in r_y])
        ds_out = ds_out.add_column("r_rejected", [float(s) for s in r_yp])
        ds_out = ds_out.add_column("r_z",        [list(map(float, row)) for row in r_z_lists])
        ds_out.save_to_disk(args.output_dataset_dir)

        N = len(ds_out)
        signed = label_y - label_yp                               # (N,) ∈ [-1, 1]

        stats = {
            "n_rows":             N,
            "n_z":                n_z,
            "label_chosen_mean":   float(label_y.mean()),
            "label_rejected_mean": float(label_yp.mean()),
            "frac_signed_target_pos":   float(np.mean(signed >  0)),
            "frac_signed_target_neg":   float(np.mean(signed <  0)),
            "frac_signed_target_zero":  float(np.mean(signed == 0)),
            "abs_signed_target_mean":   float(np.mean(np.abs(signed))),
            "r_chosen_mean":            float(r_y_arr.mean()),
            "r_rejected_mean":          float(r_yp_arr.mean()),
            "r_z_mean":                 float(r_z_arr.mean()),
            "r_chosen_minus_r_z_mean":   float((r_y_arr  - r_z_arr.mean(axis=1)).mean()),
            "r_rejected_minus_r_z_mean": float((r_yp_arr - r_z_arr.mean(axis=1)).mean()),
        }
        if n_z == 1:
            # Legacy 4-cell histogram only meaningful for {0,1}-valued labels.
            l_y_int  = wins_y[:, 0].astype(np.int8)
            l_yp_int = wins_yp[:, 0].astype(np.int8)
            stats.update({
                "cell_y_beats_z_only":      int(np.sum((l_y_int == 1) & (l_yp_int == 0))),
                "cell_yprime_beats_z_only": int(np.sum((l_y_int == 0) & (l_yp_int == 1))),
                "cell_both_beat_z":         int(np.sum((l_y_int == 1) & (l_yp_int == 1))),
                "cell_neither_beats_z":     int(np.sum((l_y_int == 0) & (l_yp_int == 0))),
            })
        with open(os.path.join(args.output_dataset_dir, "judge_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(
            f"[score_enpo] N={N}  n_z={n_z}  "
            f"label_chosen_mean={stats['label_chosen_mean']:.3f}  "
            f"label_rejected_mean={stats['label_rejected_mean']:.3f}  "
            f"|signed_target|_mean={stats['abs_signed_target_mean']:.3f}",
            flush=True,
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
