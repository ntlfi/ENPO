"""Subprocess wrapper around workers/vllm_generate.py."""
import json
import os
import subprocess

from src.env import (
    LLM_VLLM_PYTHON, WORKER_VLLM_GEN, hf_offline_env, validate_cuda_visibility,
)


def run_vllm_generate(model, prompts, iter_dir, *,
                      n=4, temperature=0.7, top_p=0.9,
                      max_new_tokens=512, max_input_tokens=1024,
                      tp_size=8, gpu_mem_util=0.90,
                      output_filename="completions.json",
                      input_filename="vllm_input.json",
                      tokenizer=None,
                      seed=None):
    """Spawn a vLLM subprocess using `tp_size` GPUs from the inherited allocation.
    Writes completions to `<iter_dir>/<output_filename>` (default
    `completions.json`). Returns that path. The XPO runner sets
    output_filename per source policy (completions_pi.json / completions_ref.json)."""
    env = hf_offline_env()
    validate_cuda_visibility(tp_size, env)

    in_file  = os.path.join(iter_dir, input_filename)
    out_file = os.path.join(iter_dir, output_filename)

    cfg = {
        "model":          model,
        "prompts":        prompts,
        "n":              n,
        "temperature":    temperature,
        "top_p":          top_p,
        "max_new_tokens": max_new_tokens,
        "max_model_len":  max_input_tokens + max_new_tokens,
        "tokenizer":      tokenizer,
        "seed":           seed,
    }
    with open(in_file, "w") as f:
        json.dump(cfg, f)

    print(f"\n[Phase 1] vLLM generate — {len(prompts)} prompts × n={n} (TP={tp_size})", flush=True)
    subprocess.run([
        LLM_VLLM_PYTHON, WORKER_VLLM_GEN,
        "--input",                  in_file,
        "--output",                 out_file,
        "--tensor_parallel_size",   str(tp_size),
        "--gpu_memory_utilization", str(gpu_mem_util),
    ], env=env, check=True)
    return out_file
