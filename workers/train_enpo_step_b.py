#!/usr/bin/env python3
"""
ENPO Step B trainer. Runs a sigmoid-DPO surrogate update on a synthetic
preference dataset where the synthetic preferences are derived by ranking K
on-policy candidates from π_t under the scalar reward

    r(x, z) = s · ( log π_{t+1}(z|x) − log π_t(z|x) )

with chosen = argmax_k r and rejected = argmin_k r per prompt. The DPO loss
uses β as the inverse-temperature coefficient and π_t as the reference policy
(its logp's are cached in the dataset's logp_t_chosen / logp_t_rejected
columns). The trainable policy is initialised from π̂_t and the trained
checkpoint becomes π̂_{t+1}.

This hard-ranked pair objective is a practical surrogate for the KL-regularized
Step B update; it does not establish the exact Gibbs optimum of that objective.

Loss (per row):
    Δ_π   = log π̃(chosen|x) − log π̃(rejected|x)        ← trainable
    Δ_ref = log π_t(chosen|x) − log π_t(rejected|x)      ← cached
    z     = β · (Δ_π − Δ_ref)
    L     = softplus(−z)   (= −log σ(z))

Run via:

    accelerate launch --num_processes 8 workers/train_enpo_step_b.py \\
        --policy_path  /path/to/pi_hat_t              \\
        --output_dir   iter_dir/adversary             \\
        --preference_data_dir  iter_dir/stepb_synth_pref  \\
        --beta 0.05 --num_train_epochs 1
"""
import argparse
import gc
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def encode_pair(tokenizer, prompt: str, response: str, max_length: int):
    prompt_ids   = tokenizer(prompt,   add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None and (
        not response_ids or response_ids[-1] != tokenizer.eos_token_id
    ):
        response_ids = response_ids + [tokenizer.eos_token_id]

    overflow = len(prompt_ids) + len(response_ids) - max_length
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]

    input_ids = prompt_ids + response_ids
    attn_mask = [1] * len(input_ids)
    resp_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return input_ids, attn_mask, resp_mask


@dataclass
class StepBCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_pack   = [encode_pair(self.tokenizer, f["prompt"], f["chosen"],   self.max_length) for f in features]
        rejected_pack = [encode_pair(self.tokenizer, f["prompt"], f["rejected"], self.max_length) for f in features]

        def pad(pack):
            L = max(len(x[0]) for x in pack)
            ids, am, rm = [], [], []
            pad_id = self.tokenizer.pad_token_id
            for x in pack:
                pad_n = L - len(x[0])
                ids.append(list(x[0]) + [pad_id] * pad_n)
                am.append(list(x[1]) + [0] * pad_n)
                rm.append(list(x[2]) + [0] * pad_n)
            return (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(am,  dtype=torch.long),
                torch.tensor(rm,  dtype=torch.long),
            )

        c_ids, c_am, c_rm = pad(chosen_pack)
        r_ids, r_am, r_rm = pad(rejected_pack)

        return {
            "chosen_input_ids":      c_ids,
            "chosen_attention_mask": c_am,
            "chosen_response_mask":  c_rm,
            "rejected_input_ids":      r_ids,
            "rejected_attention_mask": r_am,
            "rejected_response_mask":  r_rm,
            "logp_t_chosen":   torch.tensor([f["logp_t_chosen"]   for f in features], dtype=torch.float32),
            "logp_t_rejected": torch.tensor([f["logp_t_rejected"] for f in features], dtype=torch.float32),
        }


def _sum_logp(model, input_ids, attention_mask, response_mask):
    """Return (B,) tensor of summed log-probs over response tokens. With gradients."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_resp   = response_mask[:, 1:].float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * shift_resp).sum(dim=1)


class StepBTrainer(Trainer):
    """Sigmoid-DPO loss with reference π_t (cached logps in dataset). β is the
    spec's KL coefficient — same role as DPO's β: π̃*(z|x) ∝ π_t(z|x)·exp(r/β)."""

    def __init__(self, *, beta: float, **kwargs):
        super().__init__(**kwargs)
        if beta <= 0:
            raise ValueError(f"beta must be > 0; got {beta}")
        self.beta = float(beta)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logp_pi_c = _sum_logp(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_response_mask"],
        )
        logp_pi_r = _sum_logp(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_response_mask"],
        )

        delta_pi  = logp_pi_c - logp_pi_r                                     # (B,)
        delta_t   = (inputs["logp_t_chosen"] - inputs["logp_t_rejected"]).to(delta_pi)
        z = self.beta * (delta_pi - delta_t)                                  # (B,)
        loss = F.softplus(-z).mean()

        with torch.no_grad():
            self.log_metrics_buffer = {
                "delta_pi_mean":   delta_pi.mean().item(),
                "delta_t_mean":    delta_t.mean().item(),
                "z_mean":          z.mean().item(),
                "beta":            self.beta,
                "logp_pi_chosen":  logp_pi_c.mean().item(),
                "logp_pi_rejected": logp_pi_r.mean().item(),
            }

        if return_outputs:
            return loss, {"delta_pi": delta_pi, "z": z}
        return loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "log_metrics_buffer"):
            logs = {**logs, **self.log_metrics_buffer}
        super().log(logs, *args, **kwargs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path",         required=True,
                   help="Initial adversarial policy checkpoint (= π̂_t). Training starts here.")
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--preference_data_dir", required=True,
                   help="HF dataset with prompt, chosen, rejected, logp_t_chosen, logp_t_rejected.")
    p.add_argument("--beta",                type=float, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate",       type=float, default=1e-6)
    p.add_argument("--num_train_epochs",    type=int,   default=1)
    p.add_argument("--max_length",          type=int,   default=1024)
    p.add_argument("--warmup_ratio",        type=float, default=0.1)
    p.add_argument("--lr_scheduler_type",   default="cosine")
    p.add_argument("--logging_steps",       type=int,   default=10)
    p.add_argument("--wandb_run_name",      default=None)
    p.add_argument("--seed",                type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs])

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    dataset = load_from_disk(args.preference_data_dir)
    required = {"prompt", "chosen", "rejected", "logp_t_chosen", "logp_t_rejected"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {args.preference_data_dir} missing columns: {missing}")

    if accelerator.is_main_process:
        print(f"\n[ENPO Step B train] dataset = {len(dataset)} synth-pref rows from "
              f"{args.preference_data_dir}", flush=True)
        print(f"[ENPO Step B train] policy   = {args.policy_path}", flush=True)
        print(f"[ENPO Step B train] β = {args.beta} (KL anchor strength to π_t; "
              f"closed form: π̂* ∝ π_t · exp(r/β))", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.policy_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.policy_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False

    deepspeed_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "deepspeed_z3.json",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="wandb",
        run_name=args.wandb_run_name or f"enpo-stepb-{Path(args.output_dir).name}",
        deepspeed=deepspeed_cfg,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    collator = StepBCollator(tokenizer=tokenizer, max_length=args.max_length)
    trainer  = StepBTrainer(
        beta=args.beta,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
    accelerator.wait_for_everyone()

    del model, trainer
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    unset_hf_deepspeed_config()


if __name__ == "__main__":
    main()
