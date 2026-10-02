#!/usr/bin/env python3
"""
ENPO Step A training worker: OMD loss with on-policy proximal regularization.

The on-policy proximal term

    L_prox(π) = -α · E_{(x̃, z̃) ∼ D_t'}[ log π(z̃ | x̃) ]

is added to the OMD squared loss. Up to a π-independent constant, L_prox =
α · KL(π_t || π) (forward KL anchor toward π_t). Total:

    L = mean( (h_t − target)² )   −   α · mean( log π(z̃ | x̃) )

where (x̃, z̃) ~ D_t' are fresh on-policy samples from THIS iteration's
generations. Each row of the input dataset carries the pair, cached reference
and current-policy log-probabilities, preference labels, and these columns:

    prox_prompt    (str)  — chat-templated x̃ ~ d_0 (this iter's prompts)
    prox_response  (str)  — z̃ ~ π_t(·|x̃)        (this iter's on-policy gen)

The runner is responsible for sampling these into D_t (one prox singleton per
preference row, drawn from this iteration's policy samples).

When α = 0 the proximal forward pass is SKIPPED (zero compute overhead) and
only the OMD squared loss remains.

Run via:

    accelerate launch --num_processes 8 workers/train_enpo_step_a.py \\
        --policy_path  /path/to/pi_t                       \\
        --output_dir   iter_dir/policy                            \\
        --preference_data_dir  iter_dir/d_t_full_prox      \\
        --eta 7.5e-3 --tau 2.5e-3 --alpha 1e-5             \\
        --num_train_epochs 1
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


# ──────────────────────────────────────────────────────────────────────────────
# Encoding (matches workers/precompute_logp.py)
# ──────────────────────────────────────────────────────────────────────────────
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
class ENPOStepACollator:
    tokenizer: Any
    max_length: int = 1024
    include_prox: bool = True   # set False when α=0 to skip prox encoding entirely

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

        out = {
            "chosen_input_ids":      c_ids,
            "chosen_attention_mask": c_am,
            "chosen_response_mask":  c_rm,
            "rejected_input_ids":      r_ids,
            "rejected_attention_mask": r_am,
            "rejected_response_mask":  r_rm,
            "logp_ref_chosen":   torch.tensor([f["logp_ref_chosen"]   for f in features], dtype=torch.float32),
            "logp_ref_rejected": torch.tensor([f["logp_ref_rejected"] for f in features], dtype=torch.float32),
            "logp_t_chosen":     torch.tensor([f["logp_t_chosen"]     for f in features], dtype=torch.float32),
            "logp_t_rejected":   torch.tensor([f["logp_t_rejected"]   for f in features], dtype=torch.float32),
            "label_chosen":      torch.tensor([f["label_chosen"]      for f in features], dtype=torch.float32),
            "label_rejected":    torch.tensor([f["label_rejected"]    for f in features], dtype=torch.float32),
        }

        if self.include_prox:
            prox_pack = [encode_pair(self.tokenizer, f["prox_prompt"], f["prox_response"], self.max_length) for f in features]
            p_ids, p_am, p_rm = pad(prox_pack)
            out["prox_input_ids"]      = p_ids
            out["prox_attention_mask"] = p_am
            out["prox_response_mask"]  = p_rm

        return out


def _sum_logp(model, input_ids, attention_mask, response_mask):
    """Return (B,) tensor of summed log-probs over response tokens only.
    With gradients (the policy is trainable here)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_resp   = response_mask[:, 1:].float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * shift_resp).sum(dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Custom trainer
# ──────────────────────────────────────────────────────────────────────────────
class ENPOStepATrainer(Trainer):
    """OMD squared loss + on-policy forward-KL proximal term."""

    def __init__(self, *, eta: float, tau: float, alpha: float, **kwargs):
        super().__init__(**kwargs)
        if not (0 < tau < eta):
            raise ValueError(f"ENPO Step A requires 0 < tau < eta; got tau={tau}, eta={eta}")
        if alpha < 0:
            raise ValueError(f"alpha must be ≥ 0; got {alpha}")
        self.eta = float(eta)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self._w_ref = self.tau / self.eta
        self._w_t   = (self.eta - self.tau) / self.eta
        self._inv_eta = 1.0 / self.eta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ── OMD squared term ──
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
        delta_ref = (inputs["logp_ref_chosen"] - inputs["logp_ref_rejected"]).to(delta_pi)
        delta_t   = (inputs["logp_t_chosen"]   - inputs["logp_t_rejected"]  ).to(delta_pi)

        h = delta_pi - self._w_ref * delta_ref - self._w_t * delta_t           # (B,)

        label_c = inputs["label_chosen"].to(delta_pi)
        label_r = inputs["label_rejected"].to(delta_pi)
        target  = (label_c - label_r) * self._inv_eta                          # (B,)

        loss_pref = (h - target).pow(2).mean()

        # ── Proximal term (skipped when α=0 or prox columns absent) ──
        if self.alpha > 0 and "prox_input_ids" in inputs:
            logp_pi_prox = _sum_logp(
                model,
                inputs["prox_input_ids"],
                inputs["prox_attention_mask"],
                inputs["prox_response_mask"],
            )                                                                  # (B,)
            loss_prox = -self.alpha * logp_pi_prox.mean()
            logp_pi_prox_mean = logp_pi_prox.mean().item()
        else:
            loss_prox = torch.zeros((), device=loss_pref.device, dtype=loss_pref.dtype)
            logp_pi_prox_mean = 0.0

        loss = loss_pref + loss_prox

        with torch.no_grad():
            self.log_metrics_buffer = {
                "loss_pref":              loss_pref.item(),
                "loss_prox":              loss_prox.item(),
                "logp_pi_prox_mean":      logp_pi_prox_mean,
                "alpha":                  self.alpha,
                "delta_pi_mean":          delta_pi.mean().item(),
                "delta_ref_mean":         delta_ref.mean().item(),
                "delta_t_mean":           delta_t.mean().item(),
                "h_mean":                 h.mean().item(),
                "target_mean":            target.mean().item(),
                "signed_target_pos_frac":  ((label_c - label_r) >  0).float().mean().item(),
                "signed_target_neg_frac":  ((label_c - label_r) <  0).float().mean().item(),
                "signed_target_zero_frac": ((label_c - label_r) == 0).float().mean().item(),
                "h_minus_target_mean":     (h - target).mean().item(),
            }

        if return_outputs:
            return loss, {"delta_pi": delta_pi, "h": h, "target": target}
        return loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "log_metrics_buffer"):
            logs = {**logs, **self.log_metrics_buffer}
        super().log(logs, *args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Argparse + main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path",          required=True)
    p.add_argument("--output_dir",           required=True)
    p.add_argument("--preference_data_dir",  required=True,
                   help="HF dataset with pair/log-probability/label columns plus prox_prompt, prox_response.")
    p.add_argument("--eta",                  type=float, required=True)
    p.add_argument("--tau",                  type=float, required=True)
    p.add_argument("--alpha",                type=float, required=True,
                   help="Proximal regularization strength for THIS iter. α=0 disables.")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate",        type=float, default=1e-6)
    p.add_argument("--num_train_epochs",     type=int,   default=1)
    p.add_argument("--max_length",           type=int,   default=1024)
    p.add_argument("--warmup_ratio",         type=float, default=0.1)
    p.add_argument("--lr_scheduler_type",    default="cosine")
    p.add_argument("--logging_steps",        type=int,   default=10)
    p.add_argument("--wandb_run_name",       default=None)
    p.add_argument("--seed",                 type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs])

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    dataset = load_from_disk(args.preference_data_dir)
    required_pref = {
        "prompt", "chosen", "rejected",
        "label_chosen", "label_rejected",
        "logp_ref_chosen", "logp_ref_rejected",
        "logp_t_chosen",   "logp_t_rejected",
    }
    missing = required_pref - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {args.preference_data_dir} missing required columns: {missing}.")

    has_prox = {"prox_prompt", "prox_response"}.issubset(set(dataset.column_names))
    use_prox = has_prox and args.alpha > 0
    if args.alpha > 0 and not has_prox:
        raise ValueError(
            f"--alpha={args.alpha} > 0 but dataset has no prox_prompt/prox_response columns. "
            "Runner Phase must build them before training.")

    if accelerator.is_main_process:
        print(f"\n[ENPO Step A train] dataset = {len(dataset)} rows from {args.preference_data_dir}",
              flush=True)
        print(f"[ENPO Step A train] policy   = {args.policy_path}", flush=True)
        print(f"[ENPO Step A train] η = {args.eta}, τ = {args.tau}, α = {args.alpha}, "
              f"weights w_ref={args.tau/args.eta:.4f}, w_t={(args.eta-args.tau)/args.eta:.4f}, "
              f"target ∈ {{-1/η, 0, +1/η}} = {{{-1.0/args.eta:.3f}, 0, {1.0/args.eta:.3f}}}, "
              f"prox_active={use_prox}",
              flush=True)

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
        run_name=args.wandb_run_name or f"enpo-stepa-{Path(args.output_dir).name}",
        deepspeed=deepspeed_cfg,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    collator = ENPOStepACollator(tokenizer=tokenizer, max_length=args.max_length, include_prox=use_prox)
    trainer  = ENPOStepATrainer(
        eta=args.eta,
        tau=args.tau,
        alpha=args.alpha,
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
