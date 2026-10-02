# ENPO, INPO, and XPO

Minimal training code for the three algorithms used in the paper. ENPO uses
the single-pair implementation called **Direct ENPO (DENPO)** in the manuscript.
This release contains training code, configuration, documentation, and CPU
tests. It excludes evaluation code/results, experiment logs, and checkpoints.

## 1. Set up the environments

Run these commands from this repository's root. Use Linux, Python 3.11,
Conda, and NVIDIA GPUs. The paper recipes use eight H100 GPUs. Training and
vLLM generation require separate environments because their CUDA and
Transformers versions differ.

```bash
conda create -n llm_train python=3.11 -y
conda run -n llm_train python -m pip install 'torch==2.11.0+cu126' \
  --index-url https://download.pytorch.org/whl/cu126
conda run -n llm_train python -m pip install -r requirements/requirements-llm_train.txt

conda create -n llm_vllm python=3.11 -y
conda run -n llm_vllm python -m pip install -r requirements/requirements-llm_vllm.txt

conda activate llm_train
export LLM_VLLM_PYTHON="$(conda run -n llm_vllm python -c 'import sys; print(sys.executable)')"
hf auth login
```

Obtain access to `meta-llama/Meta-Llama-3-8B-Instruct` before running.
The scripts also download `Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M`
and `HuggingFaceH4/ultrafeedback_binarized`. Downloads use normal Hugging Face
authentication, including `HF_TOKEN` when supplied. Use an NVIDIA driver
compatible with the CUDA 12.6 training and CUDA 13.0 generation stacks.

The [requirements](requirements/README.md) pin core package versions from
the source environment. A fresh installation and GPU training run have not
yet been validated for this reduced release.

## 2. Run an algorithm

Keep `llm_train` active and launch one recipe:

```bash
bash scripts/run_enpo.sh
bash scripts/run_inpo.sh
bash scripts/run_xpo.sh
```

Each command starts a separate training run: three iterations with 20,000
prompts per iteration. To inspect the exact command without running it:

```bash
DRY_RUN=1 bash scripts/run_enpo.sh
```

Append arguments to override recipe settings. Always use a new output
directory when changing settings, because runners reuse cached intermediate files.

```bash
bash scripts/run_enpo.sh --seed 1 --output_dir outputs/enpo_seed1
bash scripts/run_inpo.sh --num_iterations 1 --num_prompts 128 --output_dir outputs/inpo_small
```

The smaller example still requires the configured GPU allocation. To change
GPU count, set both the training process count and generation parallelism:

```bash
LLM_NUM_PROCESSES=2 CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/run_inpo.sh --vllm_tp_size 2 --output_dir outputs/inpo_2gpu
```

Choose model and batch sizes that fit your devices; this is a configuration
example, not a validated two-GPU memory budget.

| Recipe | Final policy checkpoint |
| --- | --- |
| ENPO | `outputs/enpo/iter3/policy/` |
| INPO | `outputs/inpo/iter3/` |
| XPO | `outputs/xpo/iter3/` |

ENPO also saves `iterN/adversary/`. All runners save intermediate generations,
datasets, and iteration checkpoints under their output directory. W&B logging
is disabled by default; set `WANDB_MODE=online` to enable it.

## 3. Configure the paper settings

ENPO's recipe uses one Step A response from each policy, two fresh Step B
candidates, four comparator responses, and no replay buffer. INPO samples
8 candidates per prompt; XPO samples 8 from each policy.

The paper's inverse stepsize `1/η` is `--eta` in the code, and its KL
coefficient is `--tau`. The ENPO/INPO sweep is:

| Setting | `--eta` | `--tau` |
| --- | --- | --- |
| Default | `7.5e-3` | `2.5e-3` |
| Half | `3.75e-3` | `2.5e-3` |
| Quarter | `1.875e-3` | `1.25e-3` |
| Eighth | `0.9375e-3` | `0.625e-3` |

Override both together. ENPO's `--alpha 0.001` and `--beta 0.05` control
separate terms. Use `python -m runners.enpo --help` (or `inpo`, `xpo`) for
all options after installing dependencies. The recipe scripts select paper
settings; direct Python invocation retains the source runner defaults.

Interpreter paths can be overridden with `LLM_TRAIN_PYTHON`,
`LLM_TRAIN_ACCEL`, and `LLM_VLLM_PYTHON`. GPU allocation and Hugging Face
offline flags are inherited. See [`.env.example`](.env.example) for optional
exports; it is not loaded automatically.

## Code layout and checks

- `runners/`: ENPO, INPO, and XPO orchestration.
- `src/`: shared subprocess helpers.
- `workers/`: generation, scoring, precomputation, and training losses.
- `configs/`: DeepSpeed configuration.
- `algos/`: specifications and the [ENPO implementation map](algos/enpo_implementation.md).

CPU-only checks (no models, network, or training dependencies required):

```bash
python -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
```

## Release status

Prepared from source commit `af878d250ebf1d61b883aa36c1b96ea6abc49bb8`.
The training computations are preserved. One existing discrepancy requires
attention before paper-reproduction claims: the XPO worker minimizes a term
`-alpha * log pi`, while the paper and reference specification use
`+alpha * log pi`. See the [implementation note](algos/specs/xpo.md).

No project license has been selected. Model and dataset licenses remain
separate. CPU checks do not establish successful GPU training or reproduce
the paper's results.

## Citation

If you use this code, please cite the [paper](https://arxiv.org/abs/2606.01382):

```bibtex
@misc{nan2026efficient,
  title         = {Efficient Exploration for Iterative Nash Preference Optimization},
  author        = {Tianlong Nan and Xiaopeng Li and Christian Kroer and Tianyi Lin},
  year          = {2026},
  eprint        = {2606.01382},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.01382}
}
```
