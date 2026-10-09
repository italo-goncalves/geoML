# The expected kernel

*Built 2026-10-08 for 0.9.0, branch `expected-kernel`. `geoml/latent/network.py`
and its contacts in `geoml/models.py`; tests in
`geoml/test/test_expected_kernel.py`; the gates in
`docs/benchmarks/expected_kernel.py`. The research behind it is the project
*Spatial cross-validation* (`report/new_covariance.md`,
`report/dgp_propagation.tex`, `report/moment_derivation.tex`).*

A GP node whose input is another node's uncertain output takes the
covariance `E[k(h(x), h(y))]` over that input's distribution. What decides
the average is the variance of the *difference* `h(x) - h(y)`, so a node
hands its children its covariance between locations as well as its
variances. Until 0.9.0 it handed on each location's variance alone, and the
child widened its range by half of it.

## Why

On the research's folded section (16 drillholes of 12 samples, a two-layer
network), the old rule's predictive variance came out at 0.002 of the
model's own Monte Carlo predictive. The leaf reads the hidden mean at ranges
of 0.1-0.3, and an input variance added to the squared range with weight
one half -- the Gaussian kernel's own weight is 6, and Paciorek's
normalization cancels the rest -- barely registers. The bound then buys a
tiny noise variance with that confidence. Measured here (gate 2), on 200
new drillholes:

| Model | Rule | Log score a new hole | Calibration (mean z²) |
|---|---|---|---|
| VGP | - | -12.17 | 0.85 |
| GP on a GP | marginal | -25.39 | 5.34 |
| GP on a GP | joint | -8.99 | 1.19 |
| GP on the GP beside the coordinates | marginal | -18.92 | 3.17 |
| GP on the GP beside the coordinates | joint | -11.12 | 1.31 |
| GP on a `GPWalk` | marginal | -31.65 | 6.04 |
| GP on a `GPWalk` | joint | -10.46 | 1.26 |

The research measured -155 and -31 for the marginal networks after a
trust-region polish to convergence; at 2000 Adam iterations they are less
far gone, and the order is the same.

## The kernel

For the Gaussian kernel, with each input dimension independent and
`v = Var(h(x) - h(y)) / r²`, `d² = (m(x) - m(y))² / r²`:

    E k = prod_s (1 + 6 v_s)^(-1/2) exp(-3 sum_s d²_s / (1 + 6 v_s))

`Exponential`, `Matern32`, `Matern52` and `RationalQuadratic` are scale
mixtures of Gaussians, `k(d) = E_w[exp(-w d²)]`, and take the same form
component by component: `prod (1 + 2 w v)^(-1/2) exp(-w sum d² / (1 + 2 w
v))`. The Matérn family at smoothness `nu` and rate `c` is `w = c² / (4 g)`,
`g ~ Gamma(nu, 1)`; the rational quadratic at `scale` alpha is
`w = 3 g / alpha`, `g ~ Gamma(alpha, 1)`.

**The components are fixed and their weights positive**, so the result is a
covariance whatever they are, and the inducing points' matrix stays
positive definite: a positive sum of expected Gaussian kernels is one.
Components placed pair by pair were tried first -- Laplace-Hermite about
each pair's tilted measure -- and erred by 7e-3 on the exponential, with no
such guarantee. The Matérn kernels are read through **eight Gaussians
each**, fitted once and kept as constants (`_KERNEL_MIXTURES`, refitted by
`docs/benchmarks/kernel_mixtures.py`): the positive mixture closest to the
kernel in the largest error over [0, 6] ranges, its weights summing to one
so that a covariance's diagonal stays one. They miss the kernel by 5.1e-4
(exponential), 9.8e-6 (Matern32) and 9.1e-6 (Matern52), and the expected
kernel by no more at any range and any uncertainty, the expectation being
linear in the mixture. They replaced a trapezoid over each kernel's own
mixing measure, accurate to 1e-7 at 52, 32 and 28 components: 3.5 to 6.5
times the evaluations for an accuracy no gate asks for. The rational
quadratic keeps a trapezoid of 48 placed about its mode, which moves with
the trained `scale`; the mass outside its grid is a constant (`w` toward
zero) and a nugget (`w` toward infinity), the nugget reaching only a pair
at one place with nothing uncertain between them, which reads one exactly.
**Rejected on the way**: Paciorek's form with the kernel substituted for the
Gaussian, its inflation fitted per kernel -- the expectation only for the
Gaussian, and 0.03 to 0.13 off for the others.

Each component's derivative in the mean is closed too, `-2 w (m(x) -
m(y)) / (r² (1 + 2 w v))` times the component, which is how a `GPWalk`
reads the slope of its field: one pass, where it took one forward-mode pass
per dimension.

Measured against adaptive quadrature over unit ranges, distances to 1.5
and variances to 5, in one to three dimensions, the rational quadratic
is within 4e-5 at scales from 1e-3 to 100. Against the average of the kernel over 4e5 draws of a jointly
Gaussian input at twelve locations (gate 1, the function):

| Kernel | Error | In standard errors | Correlation dropped | Marginal rule |
|---|---|---|---|---|
| Gaussian | 0.0008 | 2.1 | 0.195 | 0.796 |
| Exponential | 0.0005 | 3.0 | 0.107 | 0.616 |
| Matern32 | 0.0005 | 2.3 | 0.138 | 0.773 |
| Matern52 | 0.0006 | 2.2 | 0.159 | 0.794 |
| RationalQuadratic | 0.0007 | 2.1 | 0.201 | 0.618 |

`Spherical` and `Cubic` are not scale mixtures of Gaussians -- measured
2026-10-08: a spherical Gram matrix in six dimensions has eigenvalue
-0.026, and the best non-negative Gaussian mixture misses the spherical by
0.020 and the cubic by 0.015, against 6e-7 for the Matern32 -- and are
refused on an uncertain input, and in a `GPWalk`'s field, which the walk
reads where it has carried the walkers. `Cosine` is refused in every GP
node of a network. On a certain input every kernel is what it was.

## What travels

`propagate` returns a `_Moments`: a tuple that unpacks as the
`(mean, variance)` pair every caller has always read -- `[n, size]` each,
blended over the experts -- with an `experts` attribute holding, where a GP
node under the expected kernel reads the output, one `_Joint(mean,
variance, covariance)` per active expert: the output as that expert alone
gives it at the data, and its covariance with the expert's outputs at the
expert's own inducing points, `[n, m, size]`. A `refresh` keeps
`inducing_points_covariance`, `[m, m, size]` per expert, beside the
variance, which is its diagonal. Outputs are the last axis everywhere, so
an operation acts on a covariance exactly as on a variance; `None` stands
for zero (a certain node, an input). Under slots a leading slot axis comes
first, of length one where every slot holds the same, and the inducing
covariances are flattened over the slots like the points.

A GP node's own posterior covariance with its inducing points is
`k_x (K + D)^-1 D`, and among them `K (K + D)^-1 D`, made symmetric, whose
diagonal is the variance exactly (so the difference between a point and
itself has variance zero, and the next node's matrix has a unit diagonal).
`(K + D)^-1 D` is kept from the refresh as `joint_gain`. The operations:
`Linear` maps a covariance by the squared weights, as a variance;
`SelectInput` picks; `Bias` leaves it; `Scale` multiplies it; `Add` and
`LinearCombination` add their parents' (with squared weights), the parents
taken as independent as their variances are; `Concatenate` joins them,
zero where a parent has none. Each output is taken as independent of the
others: a `Linear` mixing columns creates covariance between its outputs
that is dropped, as before (roadmap, "Covariance between a node's
outputs"). Nodes that never feed a GP (`Multiply`, `Exponentiation`,
`GaussianMixture`, `ProductOfExperts`, `Stack`) carry no chain.

**Experts chain per expert.** A child expert reads its parent's same
expert at the data and at its inducing points, and moments are blended only
where a value leaves the tree. The expected kernel therefore takes the
independent rule; the consensus is refused under it.

The leaf of a deep network stays an ordinary sparse GP on the root's
inducing points: its KL, its bound and the likelihood door are unchanged,
and its realizations are drawn as any GP node's, from its expected-kernel
covariance -- not conditioned on the parent's realizations, which is the
roadmap's "Propagate individual realizations".

Checked through a network (`test_expected_kernel.py`): a leaf's covariances
with each expert's inducing points against the average over 2e5 draws of
the hidden GP's own posterior, one and three experts, Gaussian and
Matern52, and through `Linear`, `Scale`, `Bias`, `Concatenate`,
`SelectInput`, `Add` and `LinearCombination`, all within 0.005 (the
marginal rule misses by more than 0.05); a GP on a GP on a GP against the
middle posterior rebuilt by hand, to 2e-6.

## Saves and options

`GPOptions(propagation=)`: `"joint"` the `__init__` default, `"marginal"`
the class attribute an older save falls back to (persistence rebuilds
options without calling `__init__`); `expert_propagation` likewise
`"independent"` for a new model and `"consensus"` for an old one. Every
refusal lives in `VGPNetwork`'s constructor and in the propagation context
each training and prediction call opens, never in a node's, so a save that
names a refused kernel still opens under the rule it was trained with.
Under `"marginal"` the code is the old code: a deep network, three experts,
training by expert and a tree of `Add`, `Linear`, `SelectInput` and `Scale`
gave the same digests as 0.8.8, bit for bit. A single-layer model is the
same under both rules to the bit (an input is certain).

## The walk

`GPWalk` under the expected kernel walks along one random field, the same
at every step. A GP's realization is `k(u, U) (alpha + R eta) + b`, `R` its
`chol_r` and `eta` standard normals, so the part of the field's uncertainty
its inducing points explain is a finite vector shared by every point and
step. Each point carries, linearized, its sensitivity to its own start
(`a`, `[n, d, d]`) and to `eta` (`h`, `[n, d, d, m]`); its variance and its
covariance with any other point -- the walked inducing points included --
follow in closed form. **The rest of the field's variance, `1 - k K^-1 k`**,
small near the inducing points and the whole prior far from them, is
carried as each point's sensitivity to normals of its own (`r`, `[n, d,
d]`), the same along its path and shared with no other point: it raises a
point's variance and leaves the covariance between points alone. At each
step the field is read under the expected kernel with the variance
accumulated so far, so an uncertain walker reads a weaker field and slows
down; its slope is the expected gradient (Stein's lemma), and the
correlation between a walker the field has pushed and the field it then
meets enters the mean the same way. A realization walks a realization of
the field -- the field's own normals, so realization `s` of the walk rides
realization `s` of the field. The walk's KL prices the inducing points'
displacement against the walk's spread where they land; `precision` is
ignored (deprecated).

Three findings settled it, in order.

**The unexplained variance first.** The walk was first built on the
explained part alone, the part a realization carries. On chapter 16's Jura
model it put a confident Argovian region over unsampled ground, where 0.8.8
had high entropy: far from every inducing point the field has almost no
explained variance, so a walker there moved with almost no uncertainty and
the GP above read it confidently. Carrying the unexplained part removed the
region and brought the metals' calibration back to 0.8.8's (goodness Cd
0.86, Co 0.87, Cr 0.91, Cu 0.82, Ni 0.81, Pb 0.95, Zn 0.85 against 0.85,
0.89, 0.91, 0.82, 0.81, 0.93, 0.85; without it 0.84, 0.86, 0.90, 0.80,
0.78, 0.93, 0.83). With the displacement KL as well (below), the final
walk: 0.85, 0.88, 0.92, 0.82, 0.81, 0.93, 0.86, and rock maps much like
0.8.8's.

**Fixed draws of the field, walked exactly, were measured and dropped.**
32 or 64 draws scored like the linearized walk on the folded section
(-10.95 and -11.57 a new hole against -10.75, calibration 1.95 and 2.07
against 1.93), left the Jura region in place, and cost 3 to 6 times as
much; with that few draws the moments are noisy (variance ratio 0.4-2.5).
The linearization was never what went wrong.

**The displacement KL stays.** With the field's KL alone the walk network
trained a near-certain deformation: calibration about 1.95 on the folded
section under every variant above. Priced through the amplitude, nothing
moved; through the displacement, the calibration came right:

| Price | Score a new hole | Median | Calibration | Reach |
|---|---|---|---|---|
| none | -11.28 | -1.46 | 1.97 | 0.26 |
| displacement (the walk's KL) | -10.46 | -5.59 | 1.26 | 0.29 |
| Gamma prior on `amp`, mode 1, c = 2 | -11.24 | -1.49 | 1.97 | 0.26 |
| Gamma prior on `amp`, mode 1, c = 5 | -11.15 | -1.55 | 1.96 | 0.24 |
| exponential prior on `amp`, rate 1 | -11.21 | -1.51 | 1.96 | 0.25 |
| exponential prior on `amp`, rate 3 | -11.08 | -1.61 | 1.94 | 0.23 |

The median pays for honest intervals.

Against walks along sampled fields -- each walker with normals of its own
for the unexplained part, criteria fixed before measuring (mean within 0.2
of the draws' standard deviation, variance ratio within [0.8, 1.25],
covariance within 0.05 on a correlation scale), one expert:

| amp (reach 0.1 amp) | Mean error (sd) | Variance ratio | Covariance error |
|---|---|---|---|
| 1 | 0.04 | 0.98-1.07 | 0.06 (0.04 at 4e4 draws) |
| 2 | 0.07 | 0.99-1.13 | 0.13 |
| 4 | 0.12 | 0.86-1.54 | 0.53 |
| 8 | 0.31 | 0.90-3.19 | 2.19 |

Accurate to a reach of a fifth of the field's range; beyond it the
linearized spread errs both ways (without the unexplained part it fell
short only, to a ratio of 0.14 at amp 8). The folded section trains to a
reach of 0.3.

## Gate 3: chapter 5

Chapter 5's two-layer Walker Lake model (its outer kernel a Matern32 where
the chapter had a spherical, which the expected kernel refuses on an
uncertain input), scored against the exhaustive grid at every 20th node:

| Model | Rule | Iterations | rmse | CRPS | 90% coverage | Time s |
|---|---|---|---|---|---|---|
| flat | - | 100 | 170.4 | 100.7 | 0.49 | 18 |
| deep | marginal | 100 | 163.9 | 94.5 | 0.54 | 106 |
| deep | joint | 100 | 163.0 | 91.8 | 0.60 | 74 |
| flat | - | 400 | 162.0 | 102.2 | 0.32 | 31 |
| deep | marginal | 400 | 160.9 | 100.1 | 0.32 | 169 |
| deep | joint | 400 | 158.5 | 95.8 | 0.39 | 707 |

The expected kernel scores best at both lengths, by a little: Walker Lake's
geometry is only mildly curved (chapter 5). The coverage is low for every
model, the realizations being of the ground and the exhaustive values
carrying the variability below it. The times at 400 iterations are with
the trapezoid's 52 components for the Matern32; at 100, re-timed with the
table of eight, the expected kernel went from 262 s to 74 s with the
same scores to the last figure shown, faster than the marginal rule's
106 s.

## Cost

The Gaussian kernel's expectation costs what the old covariance did. A
scale mixture costs one Gaussian evaluation per component and pair (8 for
the Matérn family, 48 for the rational quadratic) wherever its input is
uncertain, in memory as well under a gradient: on chapter 5's model, about
sixteen experts and a Matern32 outer node, a training iteration cost 2.5
to 4.2 times the old rule's at the trapezoid's 52 components, and 0.7
times at the table's eight. A GP node feeding
another adds `k_x (K + D)^-1 D`, `n m² size` per expert, the order of its
explained variance.
