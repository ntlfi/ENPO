"""Subprocess wrapper around workers/score_xpo.py (accelerate launch)."""
import json
import os
import subprocess

from src.env import LLM_TRAIN_ACCEL, resolve_num_processes, hf_offline_env, REPO_ROOT


WORKER_SCORE_XPO = os.path.join(REPO_ROOT, "workers", "score_xpo.py")


def run_score_xpo(prompts, instructions, message_lists,
                  completions_pi_file, completions_ref_file, iter_dir, *,
                  judge_model, margin_keep_pct=0.7,
                  judge_batch_size=8, judge_max_length=4096,
                  num_processes=None):
    """Score K=8 π-completions and K=8 ref-completions per prompt with the judge,
    pick the max-margin cross-side pair per prompt, and tag which side τ̃ is from.
    Returns the path to the resulting preference_data dir."""
    prompts_file = os.path.join(iter_dir, "prompts.json")
    with open(prompts_file, "w") as f:
        json.dump({
            "prompts":       prompts,
            "instructions":  instructions,
            "message_lists": message_lists,
        }, f)

    env = hf_offline_env()
    num_processes = resolve_num_processes(num_processes, env)

    print(f"\n[Phase 2a] XPO score ({judge_model.split('/')[-1]}) — "
          f"accelerate launch ({num_processes} GPUs)", flush=True)
    subprocess.run([
        LLM_TRAIN_ACCEL, "launch",
        "--num_processes", str(num_processes),
        "--num_machines",  "1",
        "--mixed_precision", "bf16",
        WORKER_SCORE_XPO,
        "--judge_model",            judge_model,
        "--prompts_file",           prompts_file,
        "--completions_pi_file",    completions_pi_file,
        "--completions_ref_file",   completions_ref_file,
        "--output_dir",             iter_dir,
        "--judge_batch_size",       str(judge_batch_size),
        "--judge_max_length",       str(judge_max_length),
        "--margin_keep_pct",        str(margin_keep_pct),
    ], env=env, check=True)
    return os.path.join(iter_dir, "preference_data")
