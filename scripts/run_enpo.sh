#!/usr/bin/env bash
# ENPO paper recipe: K=2, n_pair=1, n_z=4, no replay buffer, eta=7.5e-3.
# Use run.sh enpo for the buffered variant.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run.sh" enpo \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --judge_model Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --output_dir ./outputs/enpo \
    --num_iterations 3 \
    --num_prompts 20000 \
    --m_train_per_iter 20000 \
    --max_new_tokens 512 \
    --max_input_tokens 1024 \
    --n_completions 2 \
    --n_pair 1 \
    --n_z 4 \
    --vllm_tp_size 8 \
    --vllm_gpu_mem_util 0.90 \
    --judge_max_length 4096 \
    --judge_batch_size 8 \
    --eta 7.5e-3 \
    --tau 2.5e-3 \
    --alpha 0.001 \
    --beta 0.05 \
    --s_sign 1 \
    --no_buffer \
    --sample_seed 0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --max_length 1024 \
    "$@"
