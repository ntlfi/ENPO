# Iterative Nash Policy Optimization

INPO frames LLM alignment as a symmetric two-player game under a **general
preference oracle** $\mathcal{P}$ (no Bradley–Terry assumption) and learns the
Nash policy via no-regret learning. At each iteration the current policy plays
against itself: pairs are sampled from $\pi_t$, labelled by $\mathcal{P}$, and
the next policy $\pi_{t+1}$ is obtained as the closed-form online mirror
descent (OMD) update — recovered by minimising a squared-error loss over the
collected preference dataset, with no need to estimate per-response win rates.

## Game objective

$$
J(\pi_1, \pi_2) \;=\; \mathbb{E}_{x \sim d_0}\!\left[
\mathbb{E}_{y_1 \sim \pi_1,\, y_2 \sim \pi_2}\!\left[\mathcal{P}(y_1 \succ y_2 \mid x)\right]
- \tau\,\mathrm{KL}(\pi_1(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x))
+ \tau\,\mathrm{KL}(\pi_2(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x))
\right]
$$

The Nash policy $\pi^* = \arg\max_{\pi_1} \min_{\pi_2} J(\pi_1, \pi_2)$ is unique
(by symmetry $\pi_1^* = \pi_2^* = \pi^*$) and satisfies $J(\pi^*, \pi) \ge \tfrac{1}{2}$
for all $\pi$ (modulo KL terms) — so $\pi^*$ wins against any opponent at least
half the time.

## Algorithm 1 — INPO

```text
─────────────────────────────────────────────────────────────────────
Input :  reference policy π_ref,  prompt distribution d_0,
         preference oracle P,     KL parameter τ,
         OMD parameter η,         iterations T,
         pairs per iteration n
Output:  trained policy π_{T+1}
─────────────────────────────────────────────────────────────────────

Define, for any policy π and tokens y, y' under prompt x :

    h_t(π, x, y, y') = log [ π(y|x) / π(y'|x) ]
                     − (τ / η)       · log [ π_ref(y|x) / π_ref(y'|x) ]
                     − ((η − τ) / η) · log [ π_t(y|x)   / π_t(y'|x)   ]

─────────────────────────────────────────────────────────────────────
 1:  π_1 ← π_ref
 2:  for  t = 1, 2, …, T  do
 3:      ▷ ── Self-play data collection from π_t ──
 4:      D_t ← ∅
 5:      for  i = 1, …, n  do
 6:          x_i  ~  d_0
 7:          y_1^(i), y_2^(i)  ~  π_t(· | x_i)
 8:          (y_w^(i), y_l^(i))  ←  P(x_i, y_1^(i), y_2^(i))
 9:          D_t ← D_t ∪ { (x_i, y_w^(i), y_l^(i)) }
10:      end for
11:      ▷ ── OMD update via squared-error loss ──
12:      L_t(π) ←  E_{(x, y_w, y_l) ∼ D_t} [ ( h_t(π, x, y_w, y_l) − 1/(2η) )^2 ]
13:      π_{t+1} ← argmin_{π ∈ Π}  L_t(π)
14:  end for
15:  return π_{T+1}
─────────────────────────────────────────────────────────────────────
```

## Why this loss

The unique minimiser of $\mathcal{L}_t$ in $\Pi$ is exactly the OMD iterate

$$
\pi_{t+1}(y\mid x) \;\propto\;
\exp\!\left(\tfrac{1}{\eta}\,\mathcal{P}(y \succ \pi_t \mid x)\right)\,
\pi_\text{ref}(y\mid x)^{\tau/\eta}\,
\pi_t(y\mid x)^{\,1-\tau/\eta},
$$

so minimising $\mathcal{L}_t$ on $\mathcal{D}_t$ recovers $\pi_{t+1}$ without
ever estimating the win-rate term $\mathcal{P}(y \succ \pi_t \mid x)$. Compared
to DPO-style losses, $h_t$ carries an extra $\log \pi_t / \log \pi_\text{ref}$
term reflecting that the OMD update anchors to **both** the reference policy
$\pi_\text{ref}$ and the previous iterate $\pi_t$.

## Symbols

| Symbol | Meaning |
|---|---|
| $\pi_t$ | policy at iteration $t$ ($\pi_1 = \pi_\text{ref}$) |
| $\pi_\text{ref}$ | frozen reference policy (typically the SFT checkpoint) |
| $\mathcal{P}$ | general preference oracle, $\mathcal{P}(y \succ y' \mid x) \in [0,1]$ |
| $\mathcal{D}_t$ | preference pairs sampled from $\pi_t$ at iteration $t$, $\lvert\mathcal{D}_t\rvert = n$ |
| $T$ | number of outer iterations |
| $\tau$ | KL regularisation strength in the game objective |
| $\eta$ | OMD inverse-learning-rate (larger $\eta$ → smaller step) |
| $\Pi$ | policy class with same support as $\pi_\text{ref}$ |

Practical setting in the paper: $\eta = 7.5\times 10^{-3}$, $\tau = \eta/3$,
$T = 3$, with rejection sampling ($K = 8$ responses per prompt, best-of-$K$ as
$y_w$ and worst-of-$K$ as $y_l$).

## References

- Zhang et al., *Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning*, 2024.
- Munos et al., *Nash Learning from Human Feedback*, 2023.
- Calandriello et al., *Human Alignment of LLMs through Online Preference Optimisation*, 2024.
- Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023.
