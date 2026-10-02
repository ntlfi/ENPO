"""Subprocess wrapper around workers/train_enpo_step_b.py (accelerate launch)."""
import os
import subprocess

from src.env import LLM_TRAIN_ACCEL, resolve_num_processes, hf_offline_env, REPO_ROOT


WORKER_TRAIN_ENPO_STEP_B = os.path.join(REPO_ROOT, "workers", "train_enpo_step_b.py")


def run_train_enpo_step_b(
    *,
    policy_path: str,
    preference_data_dir: str,
    output_dir: str,
    beta: float,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-6,
    num_train_epochs: int = 1,
    max_length: int = 1024,
    wandb_run_name: str | None = None,
    num_processes: int | None = None,
    seed: int = 42,
):
    """Train ENPO Step B (synthetic-DPO update on the adversarial policy π̂_t).

    `policy_path` = π̂_t starting checkpoint.
    `preference_data_dir` must have {prompt, chosen, rejected, logp_t_chosen,
    logp_t_rejected} where logp_t is π_t's reference log-probabilities.
    """
    env = hf_offline_env()
    num_processes = resolve_num_processes(num_processes, env)

    print(f"\n[Phase 9b-iv] Step B trainer — policy={policy_path}, β={beta}",
          flush=True)

    cmd = [
        LLM_TRAIN_ACCEL, "launch",
        "--num_processes", str(num_processes),
        "--num_machines",  "1",
        "--mixed_precision", "bf16",
        WORKER_TRAIN_ENPO_STEP_B,
        "--policy_path",                 policy_path,
        "--preference_data_dir",         preference_data_dir,
        "--output_dir",                  output_dir,
        "--beta",                        str(beta),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate",               str(learning_rate),
        "--num_train_epochs",            str(num_train_epochs),
        "--max_length",                  str(max_length),
        "--seed",                        str(seed),
    ]
    if wandb_run_name:
        cmd += ["--wandb_run_name", wandb_run_name]

    subprocess.run(cmd, env=env, check=True)
