"""Subprocess wrapper around workers/precompute_logp.py (accelerate launch)."""
import os
import subprocess

from src.env import LLM_TRAIN_ACCEL, resolve_num_processes, hf_offline_env, REPO_ROOT


WORKER_PRECOMPUTE = os.path.join(REPO_ROOT, "workers", "precompute_logp.py")


def run_precompute_logp(
    *,
    model_path: str,
    dataset_dir: str,
    output_dataset_dir: str,
    logp_chosen_field: str,
    logp_rejected_field: str,
    max_length: int = 1024,
    per_device_batch_size: int = 4,
    num_processes: int | None = None,
):
    """Forward `dataset_dir` through `model_path`, add per-row log-probs as columns.

    Both `logp_<chosen|rejected>_field` columns are written to a NEW dataset at
    `output_dataset_dir`. The input is not modified.
    """
    env = hf_offline_env()
    num_processes = resolve_num_processes(num_processes, env)

    print(f"\n[Precompute] {logp_chosen_field}/{logp_rejected_field} "
          f"under {model_path}", flush=True)

    cmd = [
        LLM_TRAIN_ACCEL, "launch",
        "--num_processes", str(num_processes),
        "--num_machines",  "1",
        "--mixed_precision", "bf16",
        WORKER_PRECOMPUTE,
        "--model_path",            model_path,
        "--dataset_dir",           dataset_dir,
        "--output_dataset_dir",    output_dataset_dir,
        "--logp_chosen_field",     logp_chosen_field,
        "--logp_rejected_field",   logp_rejected_field,
        "--max_length",            str(max_length),
        "--per_device_batch_size", str(per_device_batch_size),
    ]
    subprocess.run(cmd, env=env, check=True)
