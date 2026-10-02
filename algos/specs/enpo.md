# ENPO

A further variant of cumulative INPO with proximal regularization that
incorporates a **separate adversarial policy checkpoint** $\hat\pi_t$ for
on-policy data collection, in the spirit of ENPO's exploration oracle but
with the inner $\max_\pi$ dropped so the per-iteration cost stays at the
scale of a single extra training pass.

Two policy checkpoints are now maintained throughout training:

- $\pi_t$ — the **trained policy**. Updated by the OMD step with proximal
  regularizer (as before). This is the policy returned at the end of
  training, and the proximal regularizer continues to anchor the next
  iterate to $\pi_t$.
- $\hat\pi_t$ — the **adversarial / exploratory policy**. Updated by a
  separate single-max oracle (Step B below). Used **only** for sampling
  one of the two responses in each on-policy pair; never returned as the
  final model.

At iteration $t$, on-policy pairs are now generated as
$y_i \sim \pi_t(\cdot \mid x_i)$ and $y_i' \sim \hat\pi_t(\cdot \mid x_i)$
(rather than both from $\pi_t$). The intuition: $\hat\pi_t$ amplifies the
direction of recent policy movement, so $y_i'$ is a "harder" competitor
for $y_i$ than a second i.i.d.\ self-play sample would be. The preference
oracle's labels on these mixed pairs carry more information per query,
provided $\hat\pi_t$ does not drift too far from the support of $\pi_t$.

The per-iteration structure is now a clean two-step:

- **Step A** (exploit): solve the OMD + proximal-regularizer objective
  to obtain $\pi_{t+1}$ from $\pi_t$. Unchanged from the previous version.
- **Step B** (explore): solve a single max over $\tilde\pi$ to obtain
  $\hat\pi_{t+1}$ from $(\pi_{t+1}, \pi_t)$. New.

Step B is a simplified form of ENPO's exploration oracle (Eq. 3 in ENPO)
in which the inner $\max_\pi$ has been dropped by fixing the inner policy
at $\pi_t$ (the iterate before the just-completed update). With that
simplification the variance-normalization denominator becomes a constant
in the optimization variable and is dropped, leaving a single max over a
single policy.

## Algorithm 2 — Cumulative INPO with Proximal Regularization and Adversarial Sampling

```text
─────────────────────────────────────────────────────────────────────
Input :  reference policy π_ref,  prompt distribution d_0,
         preference oracle P,     KL parameter τ,
         OMD parameter η,         iterations T,
         pairs per iteration n,   training batch size m,
         proximal strength α ≥ 0,
         adversarial KL strength β > 0,  adversary sign s ∈ {+1, −1}
Output:  trained policy π̄_T  (average iterate)
─────────────────────────────────────────────────────────────────────
 1:  π_1 ← π_ref ;   π̂_1 ← π_ref ;   B ← ∅           ▷ two checkpoints, one buffer
 2:  for  t = 1, 2, …, T  do
 3:      ▷ ── Augment buffer with on-policy pairs ;
 4:        ── y_i from current policy, y'_i from adversary ──
 5:      D_t' ← ∅
 6:      for  i = 1, …, n  do
 7:          x_i  ~  d_0
 8:          y_i   ~  π_t(· | x_i)                    ▷ from current policy
 9:          y'_i  ~  π̂_t(· | x_i)                    ▷ from adversary
10:          B    ← B    ∪ { (x_i, y_i, y'_i) }
11:          D_t' ← D_t' ∪ { (x_i, y_i) }              ▷ only y_i ~ π_t goes here
12:      end for
13:      ▷ ── Build D_t : pairs from B, refs from π_t (unchanged) ──
14:      D_t ← ∅
15:      for  j = 1, …, m  do
16:          (x_j, y_j, y'_j)  ~  Uniform(B)
17:          z_j  ~  π_t(· | x_j)
18:          I^(y)_j   ←  P(y_j ≻ z_j | x_j)
19:          I^(y')_j  ←  P(y'_j ≻ z_j | x_j)
20:          D_t ← D_t ∪ { (x_j, y_j, y'_j, I^(y)_j, I^(y')_j) }
21:      end for
22:      ▷ ── Step A : OMD update + on-policy proximal regularizer ──
23:      π_{t+1} ← argmin_{π ∈ Π}
                     E_{(x, y, y', I^(y), I^(y')) ∼ D_t}
                       [ ( h_t(π, x, y, y') − η⁻¹(I^(y) − I^(y')) )² ]
                   − α · E_{(x̃, z̃) ∼ D_t'} [ log π(z̃ | x̃) ]
24:      ▷ ── Step B : single-max exploration oracle for adversary ──
25:      π̂_{t+1} ← argmax_{π̃ ∈ Π}
                     s · E_{x ∼ d_0,  z ∼ π̃(· | x)}
                          [ log π_{t+1}(z | x) − log π_t(z | x) ]
                   − β · E_{x ∼ d_0} [ KL( π̃(· | x) ‖ π_t(· | x) ) ]
26:  end for
27:  return  π̄_T = (1/T) Σ_{s=1}^T π_s
─────────────────────────────────────────────────────────────────────
```

The function $h_t$ at line 23 is unchanged:

$$
h_t(\pi, x, y, y')
\;=\;
\log\!\frac{\pi(y \mid x)}{\pi(y' \mid x)}
\;-\;\frac{\tau}{\eta}\log\!\frac{\pi_\text{ref}(y \mid x)}{\pi_\text{ref}(y' \mid x)}
\;-\;\frac{\eta - \tau}{\eta}\log\!\frac{\pi_t(y \mid x)}{\pi_t(y' \mid x)}.
$$

## Notes

- **Two checkpoints, one returned.** $\hat\pi_t$ exists purely as a
  sampling distribution for one of the two responses in each on-policy
  pair. It is never averaged into $\bar\pi_T$ and is never used as a
  reference inside $h_t$. The output and its theoretical interpretation
  (average iterate of OMD on the regularized Nash objective) are
  unchanged.

- **Indexing.** At the start of iteration $t$, both $\pi_t$ and
  $\hat\pi_t$ are available — $\pi_t$ from Step A of iteration $t-1$ (or
  $\pi_\text{ref}$ for $t=1$), $\hat\pi_t$ from Step B of iteration $t-1$
  (or $\pi_\text{ref}$ for $t=1$). The first iteration is therefore
  pure self-play against the reference, which is the right behavior:
  there is no "direction of recent movement" to amplify yet.

- **Origin of Step B.** Starting from ENPO Eq. (3), fix the inner
  variable $\pi$ at $\pi_t$ (the previous iterate). The denominator
  $\lambda + \tfrac{1}{n_t}\sum (\log\tfrac{\pi_{t+1}}{\pi_t}(y) -
  \log\tfrac{\pi_{t+1}}{\pi_t}(y'))^2$ becomes a constant in the
  optimization variable $\tilde\pi$ and drops out. The numerator
  becomes
  $\bigl(\tfrac{1}{|\mathcal{D}_t'|}\!\sum \log\tfrac{\pi_{t+1}}{\pi_t}(\tilde z)
  - \mathbb{E}_{z \sim \tilde\pi}[\log\tfrac{\pi_{t+1}}{\pi_t}(z)]\bigr)^2$.
  The first term inside is also a constant in $\tilde\pi$, so the squared
  objective reduces to "push $\mathbb{E}_{z \sim \tilde\pi}[\log(\pi_{t+1}/\pi_t)(z)]$
  away from a fixed scalar." We replace the squared "push away" with a
  signed "push in chosen direction" (controlled by $s$), and add an
  explicit KL anchor at $\pi_t$ to prevent collapse to a Dirac. This
  yields Step B as written.

- **Sign $s$.** With $s = +1$ (default), $\hat\pi_{t+1}$ samples
  responses on which $\pi_{t+1}$ has *increased* probability mass
  relative to $\pi_t$ — i.e., responses the policy is moving toward.
  These act as adversarial competitors for $\pi_{t+1}$'s own samples in
  the next iteration, which is the more useful signal for self-play
  improvement. With $s = -1$, $\hat\pi_{t+1}$ samples responses the
  policy is moving away from, which is more conservative and provides
  data on whether the recent move was justified. $s = +1$ is the natural
  default; $s = -1$ may be preferable if early iterations show
  $\hat\pi_t$ degenerating into reward-hacked outputs.

- **Closed-form interpretation.** The exact maximizer of Step B is the
  tilted policy
  $\hat\pi_{t+1}(z \mid x) \propto \pi_t(z \mid x) \cdot
  \bigl(\pi_{t+1}(z \mid x) / \pi_t(z \mid x)\bigr)^{s/\beta}$.
  This is useful for two things: (i) it confirms $\hat\pi_{t+1}$ stays
  in the support of $\pi_t$ when $\beta > 0$; (ii) for prototyping it
  can be simulated without training $\hat\pi_{t+1}$ at all — just draw
  $K$ candidates from $\pi_t$ and resample with weights
  $(\pi_{t+1}/\pi_t)^{s/\beta}$. This is a useful sanity check before
  committing to a full Step B training run.

- **Practical Step B implementation.** Treat Step B as a one-step
  policy-gradient / DPO-style update on a scalar reward
  $r(x, z) = s \cdot (\log \pi_{t+1}(z \mid x) - \log \pi_t(z \mid x))$
  with KL anchor $\pi_t$. Two paths:
  *(a)* REINFORCE: sample $z \sim \tilde\pi$, treat $r(x, z)$ as the
  return, subtract a batch-mean baseline, add KL penalty.
  *(b)* Synthetic DPO/IPO: sample $K$ candidates from $\pi_t$ per prompt,
  rank by $r$, take top vs bottom as a synthetic preference pair, run
  one DPO/IPO epoch with $\pi_t$ as the reference. This reuses the same
  training kernel as Step A and is usually the path of least
  engineering resistance in an existing RLHF stack.

- **$\mathcal{D}_t'$ is now half the size.** Because $y_i'$ is no longer
  from $\pi_t$, only $(x_i, y_i)$ is a valid sample from $d_0 \otimes
  \pi_t$, and $\mathcal{D}_t'$ shrinks from $2n$ to $n$. The proximal
  regularizer's variance is correspondingly higher; if this matters,
  $\alpha$ can be reduced or a small number of fresh on-policy
  completions can be drawn separately for $\mathcal{D}_t'$.

- **Step B does not query the preference oracle.** Step B uses only
  log-ratios under $\pi_{t+1}$ and $\pi_t$, both of which are computable
  from forward passes. No new preference labels are needed beyond the
  $2m$ already queried in lines 18–19. The only added cost per
  iteration is one extra training pass for $\hat\pi_{t+1}$.

- **Interaction with $\alpha$.** The proximal regularizer pulls
  $\pi_{t+1}$ toward $\pi_t$; the adversarial sampler pushes data
  collection away from $\pi_t$. These effects partially cancel. When
  enabling Step B, expect to retune $\alpha$ downward — a starting point
  is to halve whatever value worked for Algorithm 1.

- **Interaction with $\beta$.** $\beta \to \infty$ recovers the original
  algorithm exactly ($\hat\pi_{t+1} \to \pi_t$, so $y_i'$ is again drawn
  from a self-play distribution). $\beta \to 0$ collapses
  $\hat\pi_{t+1}$ onto the argmax response and kills exploration
  entirely. Useful range is comparable to the OMD parameter $\eta$;
  starting at $\beta = \eta$ and tuning by a factor of 2 in each
  direction is a reasonable first sweep.

- **What was lost.** Dropping the inner $\max_\pi$ and the
  variance-normalization denominator costs the information-ratio /
  elliptical-bonus structure that gave ENPO its $\sqrt{d_\text{eluder}}$
  improvement over generic optimism. The remaining exploration is
  "generic optimism" in the sense that $\hat\pi_t$ is steered by the
  direction of recent policy movement rather than by a function-class
  uncertainty estimate. For RLHF on natural-language preference data
  this almost certainly does not matter empirically; for a paper this
  is the honest framing of the simplification.
