"""Subprocess wrapper around workers/score_enpo.py (accelerate launch).

Used by runners/enpo.py at Phase 7 (after the runner has built an
HF dataset of (prompt, chosen=y, rejected=y', z, messages_json) triples). The
worker compares both pair responses to n_z reference responses and writes
mean preference labels (binary when n_z=1).
"""
import os
import subprocess

from src.env import LLM_TRAIN_ACCEL, resolve_num_processes, hf_offline_env, REPO_ROOT


WORKER_SCORE_ENPO = os.path.join(REPO_ROOT, "workers", "score_enpo.py")


def run_score_enpo(
    *,
    triples_dataset_dir: str,
    output_dataset_dir: str,
    judge_model: str,
    judge_batch_size: int = 8,
    judge_max_length: int = 4096,
    num_processes: int | None = None,
):
    """Score (y, y', z) triples → write per-row mean preference labels.

    Returns `output_dataset_dir` (the HF dataset path, which now has
    `label_chosen`, `label_rejected` columns added).
    """
    env = hf_offline_env()
    num_processes = resolve_num_processes(num_processes, env)

    print(f"\n[Phase 7] ENPO triple-sample judge ({judge_model.split('/')[-1]}) — "
          f"accelerate launch ({num_processes} GPUs)", flush=True)
    subprocess.run([
        LLM_TRAIN_ACCEL, "launch",
        "--num_processes", str(num_processes),
        "--num_machines",  "1",
        "--mixed_precision", "bf16",
        WORKER_SCORE_ENPO,
        "--judge_model",          judge_model,
        "--triples_dataset_dir",  triples_dataset_dir,
        "--output_dataset_dir",   output_dataset_dir,
        "--judge_batch_size",     str(judge_batch_size),
        "--judge_max_length",     str(judge_max_length),
    ], env=env, check=True)
    return output_dataset_dir
