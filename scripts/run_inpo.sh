#!/usr/bin/env bash
# INPO: 20,000 prompts per iteration, three iterations, K=8.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run.sh" inpo \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --judge_model Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --output_dir ./outputs/inpo \
    --num_iterations 3 \
    --num_prompts 20000 \
    --max_new_tokens 512 \
    --max_input_tokens 1024 \
    --n_completions 8 \
    --vllm_tp_size 8 \
    --vllm_gpu_mem_util 0.90 \
    --judge_max_length 4096 \
    --judge_batch_size 8 \
    --margin_keep_pct 0.7 \
    --eta 7.5e-3 \
    --tau 2.5e-3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --max_length 1024 \
    "$@"
