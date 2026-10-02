# Batch Exploratory Preference Optimization

## Implementation note

The algorithm description below is retained as a reference specification.
The released implementation in `workers/train_xpo.py` minimizes the DPO loss
**minus** `alpha * log pi(response_from_reference | prompt)`. The
`runners/xpo.py` entry point saves every iteration and returns the final
checkpoint; it does not perform the validation-based selection described
below. Use the executable loss and saved checkpoints when reproducing this
release.

## Reference specification

A batch-sampling instantiation of XPO (Xie et al., 2024, Algorithm 1). Each
outer iteration generates $N$ preference pairs from the **current** policy
paired against the **frozen reference**, labels them with a preference oracle,
and trains on the accumulated buffer with the optimism-augmented DPO
objective. The α-weighted log-likelihood term over the $\pi_\text{ref}$-sampled
trajectories — the "one-line change" from iterative DPO — drives **deliberate
exploration** beyond the support of $\pi_\text{ref}$.

The two structural features that distinguish this from iterative DPO:
1. **Asymmetric sampling.** For each prompt, one response is drawn from the
   current policy $\pi^{(t)}$ and the other from $\pi_\text{ref}$ — *not* both
   from $\pi^{(t)}$.
2. **Optimism bonus.** The loss adds an α-weighted $\log \pi(\tilde\tau)$ term
   summed over every $\pi_\text{ref}$-sampled trajectory ever collected.

## Algorithm 1 — Batch XPO

```text
─────────────────────────────────────────────────────────────────────
Input :  reference policy π_ref,    prompt distribution ρ,
         preference oracle P,       KL parameter β,
         optimism coefficient α,    iterations T,
         pairs per iteration N,     inner epochs E,    batch size B
Output:  best policy π̂ from {π^(1), …, π^(T+1)} on validation
─────────────────────────────────────────────────────────────────────

Define, per training example (τ_+, τ_-, τ̃) under reference π_ref :

    ℓ_XPO(π; τ_+, τ_-, τ̃)  =  α · log π(τ̃)                       ◀ optimism
                              − log σ ( β log π(τ_+) / π_ref(τ_+)
                                      − β log π(τ_-) / π_ref(τ_-) )  ◀ DPO

─────────────────────────────────────────────────────────────────────
 1:  π^(1) ← π_ref ;   D^(0) ← ∅
 2:  for  t = 1, 2, …, T  do
 3:      ▷ ── Asymmetric batch sampling ──
 4:      B_t ← ∅
 5:      for  i = 1, …, N  do
 6:          s_i^(t)   ~  ρ
 7:          τ_i^(t)   ~  π^(t)   | s_i^(t)
 8:          τ̃_i^(t)  ~  π_ref   | s_i^(t)              ◀ second from π_ref
 9:          (τ_+, τ_-)  ←  P( s_i^(t), τ_i^(t), τ̃_i^(t) )
10:          B_t ← B_t ∪ { (τ_+, τ_-, τ̃_i^(t)) }       ◀ tag the π_ref-sample
11:      end for
12:      D^(t) ← D^(t-1) ∪ B_t                          ◀ accumulate buffer
13:      ▷ ── Inner training: argmin of the XPO objective ──
14:      π^(t+1) ← π^(t)                                ◀ warm-start
15:      for  e = 1, …, E  do
16:          for each minibatch  M ⊂ D^(t)  of size B  do
17:              L(θ)  ←  (1/|M|) Σ_{(τ_+, τ_-, τ̃) ∈ M}  ℓ_XPO(π_θ; τ_+, τ_-, τ̃)
18:              θ  ←  θ − η ∇_θ L(θ)
19:          end for
20:      end for
21:  end for
22:  return  π̂  ∈  argmax_{π ∈ {π^(1),…,π^(T+1)}}  J_β(π)    ◀ via validation
─────────────────────────────────────────────────────────────────────
```

## Loss

$$
\mathcal{L}^{(t)}_{\text{XPO}}(\pi)
\;=\;
\underbrace{\alpha\!\!\!\sum_{(\tau_+,\,\tau_-,\,\tilde\tau)\,\in\,\mathcal{D}^{(t)}}\!\!\!\log \pi(\tilde\tau)}_{\text{exploration bonus (one-line change)}}
\;-\;
\underbrace{\sum_{(\tau_+,\,\tau_-,\,\tilde\tau)\,\in\,\mathcal{D}^{(t)}}\!\log\sigma\!\left(
\beta\log\frac{\pi(\tau_+)}{\pi_\text{ref}(\tau_+)}
-\beta\log\frac{\pi(\tau_-)}{\pi_\text{ref}(\tau_-)}
\right)}_{\text{standard DPO loss on the same buffer}}
$$

The DPO term is unchanged from offline DPO and acts on every preference pair
in the buffer. The exploration term adds an $\alpha$-weighted log-likelihood
of the trajectory drawn from $\pi_\text{ref}$. Over $T$ iterations of $N$ pairs
each, both sums grow to $|\mathcal{D}^{(T)}| = NT$ terms.

## Symbols

| Symbol | Meaning |
|---|---|
| $\pi^{(t)}$ | policy at iteration $t$ ($\pi^{(1)} = \pi_\text{ref}$) |
| $\pi_\text{ref}$ | frozen reference policy (typically the SFT checkpoint) |
| $\mathcal{P}$ | preference oracle, $\mathcal{P}(\tau \succ \tilde\tau \mid s_1) \in [0,1]$ |
| $\mathcal{D}^{(t)}$ | accumulated buffer of $(\tau_+, \tau_-, \tilde\tau)$ triples through iteration $t$, $\lvert\mathcal{D}^{(t)}\rvert = Nt$ |
| $T$ | outer iterations (paper uses $T = 3$) |
| $N$ | preference pairs collected per iteration |
| $E, B$ | inner epochs, minibatch size |
| $\beta$ | KL regularisation strength (paper uses $0.1$) |
| $\alpha$ | optimism coefficient (paper uses schedule $\{10^{-5}, 5\!\times\!10^{-6}, 0\}$ across iterations) |
| $\sigma$ | logistic sigmoid |

Practical stabilisation from the paper (Appendix E): clip
$\log \pi(\tau) / \pi_\text{ref}(\tau) \in [-500, 500]$ **only** within the
exploration term, motivated by the bounded-density-ratio assumption.

## Notes for implementation

- **Tag $\tilde\tau$ at collection time.** Line 11 stores the $\pi_\text{ref}$-sampled trajectory alongside the labelled pair so the optimism term can be evaluated per-example without re-identifying which response came from $\pi_\text{ref}$.
- **Setting $\alpha = 0$ recovers a variant of iterative DPO.** Specifically, iterative DPO with the asymmetric sampling scheme ($\tau \sim \pi^{(t)}$, $\tilde\tau \sim \pi_\text{ref}$). Standard iterative DPO instead samples both responses from $\pi^{(t)}$.
- **Warm-start vs. reset.** Line 15 warm-starts $\pi^{(t+1)}$ from $\pi^{(t)}$, matching the paper's implementation. Resetting from $\pi_\text{ref}$ each iteration would discard learning and is rarely used in practice.
- **Validation selection (Line 23).** The paper returns the iterate with highest $J_\beta$ on a held-out set, not the last iterate.
- **Generalisations.** The paper's Algorithm 2 (Appendix C.1) lets $\tilde\tau \sim \tilde\pi^{(t)}$ for any sampling policy; one practical choice is $\tilde\pi^{(t)} = \pi^{(t)}$ with $\mathcal{D}_\text{opt}^{(t)} = \mathcal{D}_\text{pref}^{(t)}$, which is what the paper's experiments actually use.

## References

- Xie, Foster, Krishnamurthy, Rosset, Awadallah, Rakhlin, *Exploratory Preference Optimization: Harnessing Implicit Q\*-Approximation for Sample-Efficient RLHF*, 2024.
- Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023.
- Guo et al., *Direct Language Model Alignment from Online AI Feedback*, 2024.
- Dong et al., *RLHF Workflow: From Reward Modeling to Online RLHF*, 2024.
