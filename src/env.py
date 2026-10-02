"""Portable interpreter paths and inherited environments for subprocesses."""
import os
from pathlib import Path
import sys


def runtime_paths(env=None, executable=None):
    """Resolve training Python, Accelerate, and generation Python.

    Explicit LLM_* overrides win. Otherwise training uses the current Python
    and its sibling Accelerate executable. Generation looks for an executable
    in a sibling conda environment named llm_vllm, first beside training Python,
    then beside the current Python. A conda base environment is also supported.
    If no llm_vllm environment is found, generation uses the current Python;
    that environment must then contain vLLM and compatible dependencies.
    """
    env = os.environ if env is None else env
    executable = sys.executable if executable is None else executable
    train_python = env.get("LLM_TRAIN_PYTHON", executable)
    train_accel = env.get(
        "LLM_TRAIN_ACCEL", str(Path(train_python).parent / "accelerate")
    )
    vllm_python = env.get("LLM_VLLM_PYTHON")
    if vllm_python is None:
        for python in (train_python, executable):
            prefix = Path(python).expanduser().parent.parent
            if prefix.parent.name == "envs":
                envs_dir = prefix.parent
            elif (prefix / "conda-meta").is_dir():
                envs_dir = prefix / "envs"
            else:
                continue
            candidate = envs_dir / "llm_vllm" / "bin" / "python"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                vllm_python = str(candidate)
                break
        else:
            vllm_python = executable
    return train_python, train_accel, vllm_python


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_TRAIN_PYTHON, LLM_TRAIN_ACCEL, LLM_VLLM_PYTHON = runtime_paths()

WORKER_VLLM_GEN    = os.path.join(REPO_ROOT, "workers", "vllm_generate.py")
WORKER_SCORE       = os.path.join(REPO_ROOT, "workers", "score.py")

DEEPSPEED_Z3_CONFIG = os.path.join(REPO_ROOT, "configs", "deepspeed_z3.json")


def hf_offline_env(env=None):
    """Copy the caller's environment, preserving HF login and offline choices.

    The historical name is retained for callers. Set HF_HUB_OFFLINE,
    HF_DATASETS_OFFLINE, and/or TRANSFORMERS_OFFLINE explicitly for cached-only
    runs. Hugging Face handles its own stored credentials. Experiment tracking
    defaults to disabled unless the caller has already selected WANDB_MODE.
    """
    env = (os.environ if env is None else env).copy()
    env.setdefault("WANDB_MODE", "disabled")
    return env


def _positive_int(value, name):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError
        result = int(value)
        if result < 1:
            raise ValueError
    except ValueError:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from None
    return result


def resolve_num_processes(num_processes=None, env=None):
    """Use an explicit process count, LLM_NUM_PROCESSES, or the default of 8."""
    env = os.environ if env is None else env
    if num_processes is not None:
        return _positive_int(num_processes, "num_processes")
    return _positive_int(env.get("LLM_NUM_PROCESSES", "8"), "LLM_NUM_PROCESSES")


def validate_cuda_visibility(tp_size, env=None):
    """Check TP against a supplied CUDA mask without changing GPU visibility.

    Numeric device IDs and GPU/MIG UUIDs all count as one visible device. With
    no mask, vLLM validates the hardware at startup; no GPU library is imported
    in the orchestration environment.
    """
    tp_size = _positive_int(tp_size, "tp_size")
    env = os.environ if env is None else env
    if "CUDA_VISIBLE_DEVICES" not in env:
        return
    mask = env["CUDA_VISIBLE_DEVICES"]
    devices = [device.strip() for device in mask.split(",")] if mask.strip() else []
    # CUDA stops enumerating devices at the first invalid/negative device ID.
    for index, device in enumerate(devices):
        if not device or device.startswith("-"):
            devices = devices[:index]
            break
    if len(set(devices)) < tp_size:
        raise ValueError(
            f"tp_size={tp_size} requires at least {tp_size} visible GPUs, but "
            f"CUDA_VISIBLE_DEVICES={mask!r} exposes at most {len(set(devices))}. "
            "Reduce tp_size or request a larger GPU allocation."
        )
