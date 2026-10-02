#!/usr/bin/env python3
"""
Pre-compute per-pair log-probability differences (chosen vs rejected) under a
frozen model, and add them as float columns to a preference HF dataset.

Used by INPO to cache Δ_ref = log π_ref(y_w|x) − log π_ref(y_l|x) once per
dataset (π_ref never moves) and Δ_t for each iteration's frozen π_t.

Run via:

    accelerate launch --num_processes 8 workers/precompute_logp.py \
        --model_path  /path/to/model               \
        --dataset_dir iter_dir/preference_data     \
        --output_dataset_dir iter_dir/preference_data_with_ref_logp \
        --logp_chosen_field   logp_ref_chosen      \
        --logp_rejected_field logp_ref_rejected    \
        --max_length 1024

Reads the dataset's {prompt, chosen, rejected} string columns, tokenizes
prompt+response, runs a forward pass, and writes per-row scalar log-probs
(sum of token log-probs over the response only — prompt tokens are masked).

The per-pair difference Δ = logp_chosen − logp_rejected is what INPO's loss
needs; we store the two log-probs separately for transparency / reusability.
"""
import argparse
import gc
import os
import sys
from datetime import timedelta

import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup_gpu(*objects):
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",            required=True,
                   help="Frozen model whose log-probs we compute (e.g. π_ref or π_t)")
    p.add_argument("--dataset_dir",           required=True,
                   help="HF dataset on disk with {prompt, chosen, rejected} columns")
    p.add_argument("--output_dataset_dir",    required=True)
    p.add_argument("--logp_chosen_field",     required=True,
                   help="Column name to write log p(chosen|prompt), e.g. logp_ref_chosen")
    p.add_argument("--logp_rejected_field",   required=True)
    p.add_argument("--max_length",            type=int, default=1024)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    return p.parse_args()


def encode_pair(tokenizer, prompt: str, response: str, max_length: int):
    """Return (input_ids, attention_mask, response_mask) for prompt + response.

    response_mask is 1 on response tokens (the only ones that contribute to
    log p(y|x)), 0 on prompt + padding tokens.
    """
    prompt_ids   = tokenizer(prompt,   add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    # Append EOS so log-prob accounts for stopping (matches DPOTrainer behavior).
    if tokenizer.eos_token_id is not None and (
        not response_ids or response_ids[-1] != tokenizer.eos_token_id
    ):
        response_ids = response_ids + [tokenizer.eos_token_id]

    # Truncate from the left of the prompt if the combined sequence is too long
    # (keeps the response intact — we need its tokens for the log-prob sum).
    overflow = len(prompt_ids) + len(response_ids) - max_length
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]

    input_ids   = prompt_ids + response_ids
    attn_mask   = [1] * len(input_ids)
    resp_mask   = [0] * len(prompt_ids) + [1] * len(response_ids)
    return input_ids, attn_mask, resp_mask


def collate(batch, pad_id, max_length):
    """Right-pad to the longest example in the batch (capped by max_length)."""
    L = min(max(len(x["input_ids"]) for x in batch), max_length)
    out = {"input_ids": [], "attention_mask": [], "response_mask": []}
    for ex in batch:
        ids, am, rm = ex["input_ids"], ex["attention_mask"], ex["response_mask"]
        ids, am, rm = ids[:L], am[:L], rm[:L]
        pad = L - len(ids)
        out["input_ids"].append(ids + [pad_id] * pad)
        out["attention_mask"].append(am + [0] * pad)
        out["response_mask"].append(rm + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


@torch.no_grad()
def sum_logp(model, batch, device):
    """Return (B,) tensor of summed log-probs over response tokens only."""
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    resp_mask = batch["response_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits.float()                                    # (B, T, V) fp32

    # Causal shift: token t's prediction is at logits[t-1].
    # log p(token_t | prefix_<t) = log_softmax(logits[t-1])[token_t]
    shift_logits = logits[:, :-1, :]                               # (B, T-1, V)
    shift_labels = input_ids[:, 1:]                                # (B, T-1)
    shift_resp   = resp_mask[:, 1:].float()                        # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    seq_lp    = (token_lp * shift_resp).sum(dim=1)                 # (B,)
    return seq_lp.cpu().tolist()


def main():
    args = parse_args()
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs])

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output_dataset_dir.rstrip("/")) or ".", exist_ok=True)
    accelerator.wait_for_everyone()

    # Load dataset on every rank (small — list of dicts in memory).
    ds = load_from_disk(args.dataset_dir)
    n  = len(ds)
    if accelerator.is_main_process:
        print(f"[precompute_logp] model={args.model_path}", flush=True)
        print(f"[precompute_logp] dataset={args.dataset_dir} ({n} pairs)", flush=True)
        print(f"[precompute_logp] writing fields: "
              f"{args.logp_chosen_field}, {args.logp_rejected_field}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)

    # Shard rows across ranks. Each rank handles indices [rank, rank+W, rank+2W, ...].
    W    = accelerator.num_processes
    rank = accelerator.process_index

    local_logp_chosen   = []
    local_logp_rejected = []
    local_indices       = []

    bs = args.per_device_batch_size
    indices = list(range(rank, n, W))

    for i_start in range(0, len(indices), bs):
        chunk_idx = indices[i_start:i_start + bs]
        examples  = [ds[j] for j in chunk_idx]

        chosen_pack   = [encode_pair(tokenizer, ex["prompt"], ex["chosen"],   args.max_length) for ex in examples]
        rejected_pack = [encode_pair(tokenizer, ex["prompt"], ex["rejected"], args.max_length) for ex in examples]

        chosen_batch = collate(
            [{"input_ids": x[0], "attention_mask": x[1], "response_mask": x[2]} for x in chosen_pack],
            pad_id, args.max_length,
        )
        rejected_batch = collate(
            [{"input_ids": x[0], "attention_mask": x[1], "response_mask": x[2]} for x in rejected_pack],
            pad_id, args.max_length,
        )

        lp_c = sum_logp(model, chosen_batch,   accelerator.device)
        lp_r = sum_logp(model, rejected_batch, accelerator.device)

        local_logp_chosen.extend(lp_c)
        local_logp_rejected.extend(lp_r)
        local_indices.extend(chunk_idx)

        if accelerator.is_main_process and (i_start // bs) % 10 == 0:
            done = i_start + len(chunk_idx)
            print(f"[precompute_logp] rank0 progress {done}/{len(indices)} "
                  f"(global ~{done * W}/{n})", flush=True)

    accelerator.wait_for_everyone()

    # Gather (idx, lp_chosen, lp_rejected) triples across ranks and assemble in order.
    all_records = gather_object([
        (idx, lc, lr) for idx, lc, lr in zip(local_indices, local_logp_chosen, local_logp_rejected)
    ])

    if accelerator.is_main_process:
        # Build per-row scalars in dataset-order.
        logp_c = [None] * n
        logp_r = [None] * n
        for idx, lc, lr in all_records:
            logp_c[idx] = float(lc)
            logp_r[idx] = float(lr)
        assert all(v is not None for v in logp_c), "missing rows in chosen log-probs"
        assert all(v is not None for v in logp_r), "missing rows in rejected log-probs"

        # Add columns; preserves all existing columns (prompt/chosen/rejected/etc.)
        ds_out = ds.add_column(args.logp_chosen_field,   logp_c)
        ds_out = ds_out.add_column(args.logp_rejected_field, logp_r)

        # Sanity: print a few sample rows
        sample_diffs = [logp_c[i] - logp_r[i] for i in range(min(5, n))]
        print(f"[precompute_logp] sample Δ (chosen−rejected) for first 5: "
              f"{[f'{d:+.3f}' for d in sample_diffs]}", flush=True)
        print(f"[precompute_logp] saving → {args.output_dataset_dir}", flush=True)
        ds_out.save_to_disk(args.output_dataset_dir)

    cleanup_gpu(model)


if __name__ == "__main__":
    main()
