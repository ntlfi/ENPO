"""Subprocess wrapper around workers/train_inpo.py (accelerate launch)."""
import os
import subprocess

from src.env import LLM_TRAIN_ACCEL, resolve_num_processes, hf_offline_env, REPO_ROOT


WORKER_TRAIN_INPO = os.path.join(REPO_ROOT, "workers", "train_inpo.py")


def run_train_inpo(
    *,
    policy_path: str,
    preference_data_dir: str,
    output_dir: str,
    eta: float,
    tau: float,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-6,
    num_train_epochs: int = 1,
    max_length: int = 1024,
    wandb_run_name: str | None = None,
    num_processes: int | None = None,
    seed: int = 42,
):
    """Train one INPO iteration on the precomputed-logp dataset.

    `policy_path` must equal π_t (the start-of-iter policy, frozen during
    training). The dataset at `preference_data_dir` must have logp_ref_*
    AND logp_t_* columns added by run_precompute_logp.
    """
    env = hf_offline_env()
    num_processes = resolve_num_processes(num_processes, env)

    print(f"\n[Train INPO] policy={policy_path}, η={eta}, τ={tau}", flush=True)

    cmd = [
        LLM_TRAIN_ACCEL, "launch",
        "--num_processes", str(num_processes),
        "--num_machines",  "1",
        "--mixed_precision", "bf16",
        WORKER_TRAIN_INPO,
        "--policy_path",                 policy_path,
        "--preference_data_dir",         preference_data_dir,
        "--output_dir",                  output_dir,
        "--eta",                         str(eta),
        "--tau",                         str(tau),
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
