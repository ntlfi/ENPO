# ENPO implementation map

The public release contains ENPO, INPO, and XPO. The paper calls the practical
ENPO variant **Direct ENPO (DENPO)**; its public entry point is
[`runners/enpo.py`](../runners/enpo.py). The [ENPO specification](specs/enpo.md)
provides the algorithmic motivation.

ENPO maintains a policy `π_t`, an exploratory policy `π̂_t`, and a frozen
reference `π_ref`; all three start from `--base_model`. It builds one pair per
prompt, with one random response from each policy's pool. The `chosen` column
comes from `π_t` and `rejected` comes from `π̂_t`: these names identify sample
origin, not a preference ranking. Both pools are generated at iteration 1.

`--n_pair` controls each Step A pool's size (default `1`), with exactly one
response selected from each pool. `--n_completions` separately controls the
fresh Step B candidate pool.

## One training iteration

1. Load the next disjoint slice of prompts. Generate the Step A pools and a
   separate fresh pool from `π_t` for Step B, through
   [`vllm_generate.py`](../workers/vllm_generate.py).
2. Build the current pair shard and cache response log probabilities under
   `π_ref` using [`precompute_logp.py`](../workers/precompute_logp.py).
3. Concatenate all pair shards, or retain only the current shard with
   `--no_buffer`. Sample `--m_train_per_iter` rows, defaulting to
   `--num_prompts`; sample with replacement if the available buffer is smaller.
4. For each sampled pair, generate `--n_z` comparator responses from `π_t`.
   [`score_enpo.py`](../workers/score_enpo.py) compares each pair member's reward
   to each comparator reward. Its labels are the fractions of strict wins;
   reward ties count as zero. Cache the pair's log probabilities under `π_t`
   (reuse the reference values at iteration 1).
5. When `--alpha > 0`, attach one proximal prompt/response to each training row,
   sampling from the current iteration's `chosen` responses. Train Step A from
   `π_t` and save `iterN/policy/`, which becomes `π_{t+1}`.
6. Rank the fresh Step B candidates using their cached log probabilities under
   `π_t` and `π_{t+1}`. Train the exploratory policy from `π̂_t`, saving
   `iterN/adversary/`, which becomes `π̂_{t+1}`.

## Implemented objectives

Log probabilities below are sums over response tokens, including an appended
EOS when needed. Prompt tokens do not contribute to these sums.

For a Step A pair `(y, y')`, define
`Δ_q = log q(y | x) − log q(y' | x)`. The trainer
[`train_enpo_step_a.py`](../workers/train_enpo_step_a.py) minimizes

```text
h      = Δ_π − (τ/η) Δ_ref − ((η − τ)/η) Δ_t
target = (label_chosen − label_rejected) / η
L_A    = mean((h − target)²) − α mean(log π(z_prox | x_prox))
```

The signed target can be negative. The proximal pool contains samples from
`π_t`. Setting `--alpha 0` skips the proximal forward pass.

Step B ranks each fresh response by
`r(x, z) = s × (log π_{t+1}(z | x) − log π_t(z | x))`, with
`s = --s_sign`. It selects the largest and smallest score per prompt and
omits tied groups. Prompts with too few nonempty candidates are also omitted.
Step B uses model log probabilities and does not call the reward judge.

[`train_enpo_step_b.py`](../workers/train_enpo_step_b.py) applies a sigmoid-DPO
surrogate to those synthetic pairs:

```text
L_B = mean(softplus(−β × (Δ_exploratory − Δ_t)))
```

The reference is the previous policy `π_t`, with log probabilities cached in
the dataset. The trainable model starts from the previous exploratory
checkpoint. The reward magnitudes determine pair ordering; the trainer does
not regress directly against their magnitudes.

## Configuration and outputs

The public launch script [`run_enpo.sh`](../scripts/run_enpo.sh) uses
`--eta 0.0075`, `--n_pair 1`, `--n_completions 2`, `--n_z 4`, and
`--no_buffer`. This selects the practical single-pair configuration with four
comparator responses and training on the current iteration's data.

Invoking the Python runner directly retains its defaults: `--eta 0.00375`,
`--n_pair 1`, `--n_completions 8`, `--n_z 1`, and an accumulated buffer. Both
interfaces default to three iterations, 20,000 new prompts per iteration,
`τ = 0.0025`, `α = 0.001`, `β = 0.05`, and `s = +1`. Step B requires an even
candidate count of at least two; the trainers require `0 < τ < η`, `α ≥ 0`,
and `β > 0`. Use the runner's `--help` for the complete interface and the
repository README for environment setup.

Intermediate generations and datasets are saved alongside each iteration's
policy and adversary checkpoints. Existing artifacts are reused according to
path-existence checks; use a new output directory when changing configuration
or data. The runner reports the final policy checkpoint and saves individual
iterates. It does not construct the averaged policy described in the
algorithm specification.
