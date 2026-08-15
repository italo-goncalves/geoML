# MAP priors on the point-estimated parameters

Status: **executed 2026-08-15** (planned the same day). The "constrain the
non-Bayesian parameters" request, turned into a mechanism, four canonical
priors, and the measurements that decided which of them ship on by default.
This file records the design decisions and the numbers.

## The problem

Only the variational state is priced. `BasicGP`, `MultiStructureGP` and
`GradientConstrainedInput` register `alpha_white_{i}`/`delta_{i}` per expert,
and those are the only parameters any `kl_divergence` sees. Everything else
the network trains — ranges, mixing weights, scales, biases — is a point
estimate optimized straight against the ELBO, with box bounds but no prior
and no penalty. Type-II maximum likelihood on a handful of kernel
hyperparameters is standard and usually harmless; the concern was the
parameters whose count grows with the network, and the ones seen misbehaving.

A census put the concern in proportion. On the non-stationary Jura network
(the largest in routine use) the unpriced parameters number **45 against
21 628 priced ones** — so the earlier finding that this model degrades with
inducing-point density was never about them; that is the variational state
itself. What remained worth acting on was specific: `Linear.weights` (the one
that scales, `size_in x size_out`), `MultiStructureGP` (the one place
overfitting was actually seen), and the ranges (well-behaved "most of the
time but not always").

## The mechanism: MAP, not variational inference

No posterior is integrated. A parameter that declares a `prior` (any object
with a `log_prob`; in practice a float64 TFP distribution) stays exactly the
point estimate it was, and the prior's log-density joins the training
objective next to the KL:

    objective = ELBO + sum(log p(theta))

which is a valid lower bound on `log p(y, theta) = log p(y | theta) +
log p(theta)`. One differentiable term per parameter, no new inference
machinery, and the gradient does the rest. The term follows the KL's
batch-scaling convention by construction, since it is added where the KL is.

Implementation: `RealParameter(prior=...)` and `RealParameter.log_prior()`
in `parameter.py`; `Parametric.log_prior()` sums over unfixed parameters
(a fixed parameter's term would be a gradient-free constant polluting the
reported objective); `VGPNetwork._training_elbo` adds it. A model with no
priors trains on bit-for-bit the objective it always did. Priors are read
when the training graph traces, so they must be in place before the first
call to train — the node constructors, which is where all of them live.

Two subtleties, chosen deliberately:

- **MAP pulls toward the mode, not the mean.** "Gamma with mean 1" would
  drag ranges toward its mode at 0.5. The canonical range prior is
  therefore `Gamma(c, c - 1)`, whose *mode* is 1 — the whitened scale — with
  log-density falling to minus infinity as a range collapses toward zero
  (the failure worth guarding) and linearly on the long side (a large range
  merely says the field is smooth).
- **The log-density is evaluated on the natural scale.** `PositiveParameter`
  optimizes in log space, and adding `log p(theta)` without the Jacobian is
  MAP in theta-space rather than log-space. For a regularizer the
  distinction is cosmetic; it is stated here so nobody rediscovers it.

## The canonical defaults

Users are not expected to set priors. The network works in whitened,
normalized space, which is what makes fixed defaults defensible; the numeric
knobs exist for experienced users, and `None` disables each.

| where | prior | default |
|---|---|---|
| `BasicGP(range_prior=2.0)` | `Gamma(2, 1)`, mode 1, per element of `ranges` | **on** |
| `MultiStructureGP(range_prior=2.0)` | `Gamma(2, n+1)` per structure — each peaks at its own staircase range `1/(n+1)`; a common peak would fight the staircase | **on** |
| `MultiStructureGP(weight_concentration="staircase")` | Dirichlet with concentrations `1 + 2/(n+1)` — peak shares proportional to each structure's range; **uniform start** | **on** |
| `Linear(weight_prior=1.0)`, `unit_norm=False` only | `N(0, 1)` per weight; the hard [-1, 1] walls step back to ±10 | **on** |
| `LinearCombination(per_component=True, weight_concentration=2.0)` | symmetric Dirichlet per output component | feature **off** by default; prior comes with it |

`unit_norm=True` weights, the shared `LinearCombination` weights, warpings
(the log-determinant prices them), transforms (deliberate prior assumptions),
and the small scalars (`bias_{i}`, `Scale`, `Bias`, `Exponentiation`,
`GPWalk`, `RadialTrend`) are marked safe and untouched. `bias_{i}` is
confirmed as a point estimate; `cross_validate` still re-initializes it with
the variational state because it encodes the data's level.

## The measurements

Identical seeds (1234, 7, 99) per arm; run-to-run spread is ±0.03 on these
scores, so only differences consistent across seeds count.

**Walker Lake, `MultiStructureGP` (3 structures), against the exhaustive
truth** — the one dataset where overfitting is measurable against the field
rather than inferred. 300 iterations, held-out points plus the full grid.

| arm | rmse vs truth | variogram score vs truth | goodness (held out) |
|---|---|---|---|
| old (no priors, uniform start) | 200.10 | 42.38 | 0.642 |
| **priors, uniform start** | **199.07** | **42.11** | **0.652** |
| priors, staircase start | 204.87 | 43.22 | 0.653 |
| staircase start alone | 206.17 | 43.61 | 0.646 |

The prior wins and the staircase **initialization** loses, pairwise on every
seed. The final weights say why: from a uniform start training finds
[0.72, 0.13, 0.15] with or without the prior — the data already knows the
long structure carries the field — and the prior firms up what it finds.
From the staircase start it converges to [0.83, 0.10, 0.07] and never
leaves: a basin, not a belief. Hence the shipped default: staircase *prior*,
uniform *start*.

**Chapter-16 Jura network (non-stationary, multivariate), 100 held-out
sites, 250 iterations** — the A/B testbed, since its trend is
`Linear(cat, size=7, unit_norm=False)`.

| arm | rmse/sd | crps | goodness |
|---|---|---|---|
| old (no priors) | 0.8978 | 8.1974 | 0.5138 |
| full defaults | 0.8976 | 8.1876 | 0.5130 |
| linear prior only | 0.8968 | 8.1656 | 0.5151 |
| range priors only | 0.8992 | 8.1881 | 0.5120 |
| full + per-component combination | 0.8981 | 8.1787 | 0.5148 |

Everything sits inside the seed noise except one directional signal: CRPS
under the Linear prior is better than `old` **pairwise on every seed**
(by 0.1-0.8%, never worse). The range priors are neutral on these healthy
fits — their value is the collapse they make expensive, which a healthy fit
never shows. The per-component combination gains nothing on Jura and stays
a non-default feature: on this dataset the seven elements evidently agree
about how much trend they want.

## What did not survive

- The staircase initialization (above): measured against truth, rejected,
  and said so in the code comment where someone would go to add it back.
- Hard walls as the regularizer for free `Linear` weights: a wall stops
  nothing until a weight hits it, then distorts the gradient; the prior
  pushes everywhere and can be out-argued by the data. The walls stay at
  ±10 as a safety net.

## Consequences

- The reported objective (`training_log`, the stopping criterion) includes
  the prior term for models whose nodes carry priors — which is now the
  default construction. Comparing ELBOs across code versions therefore has
  a constant-ish offset; comparing across warpings was already repaired by
  the log-determinant work in the same release.
- Persistence replays constructors, so an old save reloaded and refit gains
  the default priors. Its parameter *values* load untouched.
- `test_priors.py` covers the mechanism (off = identical, the pull, fixed
  excluded), the canonical defaults on all four nodes, the per-component
  broadcasting end to end, and the round trip.
