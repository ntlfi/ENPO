#!/usr/bin/env bash
# Portable entry point for every retained training runner.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

show_help() {
    cat <<'HELP'
Usage: bash scripts/run.sh RUNNER [runner arguments]

Available runners:
  enpo   ENPO with uniformly sampled pairs
  inpo   Iterative Nash policy optimization
  xpo    Exploratory preference optimization

Supply --output_dir for each runner. Arguments are passed through unchanged.
The canonical recipes are run_enpo.sh, run_inpo.sh, and run_xpo.sh.
Their final command-line options override recipe defaults.

Environment:
  LLM_TRAIN_PYTHON     Training Python executable (default: active Python)
  LLM_TRAIN_ACCEL      Training accelerate executable
  LLM_VLLM_PYTHON      Python executable in the vLLM environment
  LLM_NUM_PROCESSES    Scoring/training worker count (default: 8)
  WANDB_MODE          Weights & Biases mode (default: disabled)
  DRY_RUN=1           Print the command without starting training

Generation uses --vllm_tp_size independently of LLM_NUM_PROCESSES.
Relative paths are resolved from the repository root.
This launcher help does not import training dependencies. For the full
runner-specific parser help after installation, run:
  python -m runners.RUNNER --help

Examples:
  DRY_RUN=1 bash scripts/run_inpo.sh --output_dir ./outputs/my_inpo
  bash scripts/run.sh enpo --output_dir ./outputs/enpo_custom --n_pair 1
HELP
}

if [[ $# -eq 0 || "$1" == -h || "$1" == --help ]]; then
    show_help
    exit 0
fi
runner="$1"
shift
case "$runner" in
    enpo|inpo|xpo) ;;
    *) printf 'Unknown runner: %s\nUse --help to list available runners.\n' "$runner" >&2; exit 2 ;;
esac
for argument in "$@"; do
    if [[ "$argument" == -h || "$argument" == --help ]]; then
        show_help
        exit 0
    fi
done
run_command "$(training_python)" -m "runners.$runner" "$@"
