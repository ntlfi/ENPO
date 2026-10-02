"""Subprocess wrapper around workers/train_enpo_step_a.py (accelerate launch)."""
import os
import subprocess

from src.env import LLM_TRAIN_ACCEL, resolve_num_processes, hf_offline_env, REPO_ROOT


WORKER_TRAIN_ENPO_STEP_A = os.path.join(REPO_ROOT, "workers", "train_enpo_step_a.py")


def run_train_enpo_step_a(
    *,
    policy_path: str,
    preference_data_dir: str,
    output_dir: str,
    eta: float,
    tau: float,
    alpha: float,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-6,
    num_train_epochs: int = 1,
    max_length: int = 1024,
    wandb_run_name: str | None = None,
    num_processes: int | None = None,
    seed: int = 42,
):
    """Train ENPO Step A with its OMD loss and on-policy proximal regularizer.

    The dataset includes prox_prompt/prox_response when α > 0, drawn from this
    iteration's policy samples. With α = 0 the proximal term is omitted."""
    env = hf_offline_env()
    num_processes = resolve_num_processes(num_processes, env)

    print(f"\n[Phase 9a] Train ENPO Step A — policy={policy_path}, η={eta}, τ={tau}, α={alpha}",
          flush=True)

    cmd = [
        LLM_TRAIN_ACCEL, "launch",
        "--num_processes", str(num_processes),
        "--num_machines",  "1",
        "--mixed_precision", "bf16",
        WORKER_TRAIN_ENPO_STEP_A,
        "--policy_path",                 policy_path,
        "--preference_data_dir",         preference_data_dir,
        "--output_dir",                  output_dir,
        "--eta",                         str(eta),
        "--tau",                         str(tau),
        "--alpha",                       str(alpha),
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
