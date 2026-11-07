# Expert weights

How a GP node weights its experts where they overlap, and what the fixed
weights of 0.10.6 changed. Built from recommendations 1 and 2 of the
research note *Improving geoML's local experts* (3 November 2026, section
10); measured in `docs/benchmarks/fixed_expert_weights.py`.

## The rules

**Variance weights** (`GPOptions(expert_weights="variance")`, every model
before 0.10.6): at a location, expert `e` gets `(1 - v_e) / v_e`, its
explained variance over what it leaves, normalized over the experts. The
weights read the expert's own posterior, so they move with training (the
note: one expert's `delta` from 0.05 to 5 moved a point's weights from
0.50/0.50 to 0.02/0.98), the objective bounds the evidence of a model that
changes with the variational state, and every expert must be computed at
every location to know them.

**Fixed weights** (`expert_weights="coverage"`): the same formula at a
reference posterior. `v_e` is one less the coverage `k (K + delta I)^-1 k`
of a Gaussian reference kernel over the expert's inducing points: its
length four times the points' median spacing, in the input's own
coordinates (where `inducing.experts` divides them), `delta = 1e-2`. A raw
weight under a hundredth of the location's largest is zero. The weights
are a function of the location alone, the same for every GP node of the
input, and an expert of zero weight is never computed there. Because the
floor is relative to the largest weight and that expert always carries
weight, a location's weights are the same under any set of experts that
holds its own; that is what lets `predict` compute a group of locations
with its own experts alone and give the answer every expert computed
everywhere gives (to 1e-9, realizations included). Where every coverage
has underflowed, beyond some 38 reference lengths of every expert, the
nearest expert takes the whole weight.

**Temperature** (`expert_temperature=T`, either rule): the raw weights
raised to `T` before they are normalized, the note's recommendation 1.

**Balanced blend** (`expert_blend="balanced"`, fixed weights only): the
weights normalized to a unit sum of squares and the variances blended by
their squares, the moments of `sum w_e f_e` for independent experts, a
model whose prior variance is one everywhere and whose objective is its
exact bound. Each expert is then a part of the field, not a prediction of
it, and a GP node reading another node's per-expert chain reads it as a
prediction, so the balanced blend is refused wherever a GP node reads an
uncertain input (a GP on a GP, a walk, a `GaussianInput`).

## What reads them

The blend of a GP node (`BasicGP._blend`, its slots version), a walk's
blend of its experts (`GPWalk._joint_propagate`), the realizations'
rescaling (`_blend_scale`, which is one under the balanced blend) and the
gaps and jitters blended as variances. The weight table that training and
prediction by expert read (`VGPNetwork._weight_table`) is the fixed weights
themselves, with no sweep over the network and nothing carried from the
data to the prediction locations. Under fixed weights `predict` groups the
locations by their experts where a GP node reads an uncertain input
(`_groups_by_expert`); a GP on the coordinates computes every expert, which
costs less than a group's overhead.

Refused: the marginal propagation rule (its walk queries the field at
walked coordinates, which carry no location of their own),
`UncertainInputGP` (its nodes are locations of its own) and
`GradientConstrainedInput` (its own blend).

A prediction by expert pads each group's batches with copies of their last
row to a power of two and keeps the batch size the first group measured:
each group had probed and measured its own, and each new batch shape is a
trace and, under XLA, a compilation (Tom East, 33 groups: a first
prediction 1384 s before, 376 s after).

## Measured

Means of two seeds, the expert split fixed by its own seed (the two seeds
agree to four digits on the walk). V1 is today's rule, V2 it at `T = 2`,
C1 fixed weights at the defaults (reference 4 spacings, floor 1e-2; one
seed), P1 balanced (reference 4, floor 1e-3).

| | Walker rmse / CRPS | Jura Co rmse / CRPS | Tom East AUC / Brier / log | folded vein Brier / log / seams |
|---|---|---|---|---|
| V1 | **161** / **95.1** | **2.58** / **1.47** | 0.798 / 0.092 / 0.304 | 0.064 / 0.221 / 0.082 |
| V2 | 159 / 94.8 | 2.67 / 1.52 | 0.819 / 0.103 / 0.350 | 0.067 / 0.241 / 0.088 |
| C1 | 173 / 106 | 2.69 / 1.54 | **0.849** / **0.090** / **0.286** | **0.049** / **0.181** / **0.051** |
| P1 | 167 / 101 | 2.65 / 1.50 | 0.785 / 0.094 / 0.410 (ref 1) | refused |

Walker: 16 `grid_experts` blocks under a 50-unit spherical range, scored
against the exhaustive grid; Jura: 4 experts, the 100 validation samples;
Tom East: 20 experts, a fifth of the holes held out; the folded vein: the
walk model of chapter 17, 3 experts, against the true vein ("seams" is the
Brier where today's rule gives no expert 0.9 of the weight). On Walker
today's rule gives no expert 0.9 anywhere: the range is long against the
blocks, every expert explains most of the variance everywhere, and the
variance rule averages many experts at every location, which the fixed
weights do not.

Reference length (C1, floor 1e-3): 1, 2 and 4 spacings gave Tom East AUC
0.799, 0.841, 0.849 and the folded vein Brier 0.053, 0.051, 0.049; Walker
and Jura moved by under a percent. Floor: 1e-2 scored as 1e-3 did; 1e-1
began to cost (Tom East log score 0.305, the vein's Brier 0.049).

Prediction, the GeoScape walk network of 2026-10-22 (32 experts, three
readers), trained under the variance rule and predicted at 4429 rows:

| | first call | second call | peak RSS |
|---|---|---|---|
| every expert everywhere (variance) | 655 s | 206 s | 15.9 GB |
| fixed weights, reference 1, floor 1e-3 | 127 s | 33 s | 7.3 GB |
| fixed weights, reference 4, floor 1e-3 | 145 s | 67 s | 7.5 GB |
| fixed weights, reference 4, floor 1e-2 | 128 s | 51 s | 7.5 GB |

The folded vein's GPU peak fell from 0.55 to 0.36 GB at three experts,
its time unchanged (7.3 against 7.9 s). On Tom East, where predict
computes every expert under either rule, the CPU took 2.2 and 2.6 s for
the 20 m grid's second prediction; the GPU took 9.0 s under the variance
rule with XLA, 1.3 s with it off, and 0.93 s under fixed weights with XLA
-- the variance rule's graph is what XLA handles badly there, a matter of
the old rule (roadmap).

## Verdict

Fixed weights win where experts are many or a GP reads an uncertain input
-- Tom East and the walk, the case they were built for, and the GeoScape
network's time and memory -- and lose 4-7% of rmse on the two small
shallow models, where the variance rule's averaging over many experts
helps. The user chose fixed weights as the default for every new model
(2026-11-06), `expert_weights=None` taking them wherever the network can
and the variance rule where it cannot; the temperature helps nowhere
consistently and stays at one.
