# Dependencies

Only two environments are required:

- `llm_train`: runners, reward scoring, log-probability precomputation, and training.
- `llm_vllm`: response generation through `workers/vllm_generate.py`.

The requirements pin directly needed packages to versions recorded in the
source repository. Unrelated workstation/evaluation packages and transitive
pins have been removed; pip resolves transitive dependencies. No TRL or
external evaluation environment is needed by the retained code.

The training environment uses PyTorch's [CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/torch/);
generation uses its [CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/).
Those indexes are configured in the requirement files. Keep the environments
separate and use an NVIDIA driver compatible with both stacks.

Follow the installation commands in the [root README](../README.md).
The CPU checks do not validate dependency installation, CUDA compatibility,
or a full training run. These reduced dependency lists still need a fresh
Linux GPU installation test before claiming a validated environment.
