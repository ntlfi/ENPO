#!/usr/bin/env bash
# Shared launcher setup. Source this file; use run.sh or a recipe to launch.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

training_python() {
    if [[ -n "${LLM_TRAIN_PYTHON:-}" ]]; then
        printf '%s\n' "$LLM_TRAIN_PYTHON"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        command -v python3
    fi
}

run_command() {
    cd -- "$REPO_ROOT"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'cd %q\n' "$REPO_ROOT"
        printf '%q ' env "WANDB_MODE=$WANDB_MODE" "$@"
        printf '\n'
        return
    fi
    exec "$@"
}
