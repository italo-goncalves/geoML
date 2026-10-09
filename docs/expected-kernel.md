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
reach of 0.3. With the field's slope read in closed form the folded
section's walk network trains its 2000 iterations in 48 s, against 53 to
56 s through forward-mode passes, to the same scores.

## The second moment

Where the inducing points are certain -- a `GaussianInput` root, through
nodes acting row by row -- and the input is uncertain, the expected kernel
alone gets the moments wrong: averaging the kernel before the posterior is
formed leaves out that the posterior's mean moves with the input. A GP
node there completes them with Girard's second moment, `L = E[k(x, z)
k(x, z)ᵀ]` beside `l = E[k(x, z)]`:

    var = 1 - tr((K + D)^-1 L) + alphaᵀ L alpha - (l alpha)²

the posterior's variance averaged over the input plus the variance of its
mean, which is the mixture's exactly; the explained variance is the trace,
`tr((K + D)^-1 L)`, and an expert is weighted by what that leaves, since
the spread of the mean says nothing of how well the expert knows the
ground.

**For the Gaussian kernel `L` is closed.** Each pair of Gaussians `(a, b)`
gives, per input dimension, `exp(-ab/(a+b) (z_i - z_j)²)` times the
expectation of `exp(-(a+b)(x - z_ij)²)` about their weighted midpoint --
one pair for the Gaussian kernel, and for a `MultiStructureGP` every pair
of its structures. The part tying `i`, `j` and the location together is a
bilinear form in the offsets `x - z`, so the exponent of a pair is one
product of matrices, `[n, m, m]`, never an array with the dimensions on it
as well. An `AdditiveGP` reads its dimensions as independent: `L = (s sᵀ -
sum_d l_d l_dᵀ + sum_d L_d) / D²`, `s = sum_d l_d`.

**For the scale mixtures it is a quadrature over the input.** Closed, a
table of eight Gaussians pairs into 36 terms, each an `[n, m, m]` array,
and a training iteration on 1000 uncertain locations with a `Matern32`
cost 100 times the first moment's at 100 inducing points and ran out of
45 GB at 300 (`second_moment.py cost`). So the node's own kernel is read
at 64 points of each location's input -- 32 scrambled Sobol points and
their negatives, whitened to unit covariance, so the rule is exact for a
quadratic -- and `tr((K + D)^-1 L)` is `l (K + D)^-1 lᵀ` from the expected
kernel plus the quadrature's covariance of `k`, the spread of the mean and
the jitter variances over the points, never negative. The rational
quadratic takes it the same way. Deep networks, whose inducing points are
themselves uncertain, keep the first moment; that is its own roadmap item.
`UncertainInputGP`, which took the mixture by quadrature over every
expert's whole posterior, is deprecated with a `FutureWarning` and goes in
the breaking version.

| m | Kernel | First moment | Second moment | `UncertainInputGP`, 32 nodes |
|---|---|---|---|---|
| 100 | Gaussian | 0.010 s, 1.1 GB | 0.071 s, 1.4 GB (closed) | -- |
| 300 | Gaussian | 0.024 s, 1.3 GB | 0.70 s, 3.4 GB (closed) | -- |
| 100 | Matern32 | 0.021 s, 1.2 GB | 2.1 s, 8.9 GB (closed); 0.40 s, 2.9 GB (64 nodes) | 0.18 s, 2.1 GB |
| 300 | Matern32 | 0.071 s, 1.5 GB | out of 45 GB (closed); 1.23 s, 6.4 GB (64 nodes) | 0.58 s, 3.5 GB |

Seconds a training iteration and the process's peak, 1000 uncertain
locations.

Measured on Walker Lake (`docs/benchmarks/second_moment.py`, one
`BasicGP`, 100 inducing points, 200 locations, input variance a multiple
of the squared range), the error against the mixture by Monte Carlo (3000
draws a location), relative to its mean -- for the exponential the
quadrature at 64 nodes:

| Kernel | var / r² | Variance, second moment | Variance, first alone | Paciorek | `UncertainInputGP`, 32 nodes | Mean, second moment | Mean, Paciorek |
|---|---|---|---|---|---|---|---|
| Gaussian | 0.01 | 0.007 | 0.627 | 0.281 | 0.009 | 0.007 | 0.112 |
| Gaussian | 0.1 | 0.013 | 1.317 | 0.727 | 0.031 | 0.018 | 0.613 |
| Gaussian | 1 | 0.009 | 0.468 | 0.899 | 0.019 | 0.015 | 0.602 |
| Gaussian | 3 | 0.005 | 0.176 | 0.855 | 0.029 | 0.012 | 0.712 |
| Exponential | 0.01 | 0.013 | 0.408 | 0.380 | 0.023 | 0.013 | 0.613 |
| Exponential | 0.1 | 0.015 | 0.440 | 0.664 | 0.025 | 0.012 | 0.680 |
| Exponential | 1 | 0.009 | 0.098 | 0.764 | 0.012 | 0.007 | 0.681 |
| Exponential | 3 | 0.011 | 0.037 | 0.640 | 0.008 | 0.005 | 0.620 |

The largest error of the second moment, mean or variance, is 0.018 for the
Gaussian kernel in closed form, and at 64 nodes 0.015 for the exponential
and 0.018 for the Matern32, the Monte Carlo's own included -- within the 2%
gate everywhere. At 32 nodes the Matern32 reached 0.042; in closed form
the tables gave 0.013 to 0.015. The first moment alone
overstates the variance by up to 130%: `tr((K + D)^-1 L)` exceeds `l (K +
D)^-1 lᵀ`, and the spread of the mean is missing. For the Gaussian kernel
the node is Girard's closed form to 1e-10 (`test_expected_kernel.py`).

**The realizations stay the expected kernel's**, `l (alpha + R eps) + b`,
and carry `l R Rᵀ lᵀ` where the mixture's -- each drawn at an input of its
own -- carry `tr(R Rᵀ L)` and the spread of the mean. The difference,

    jitter = tr((R Rᵀ + alpha alphaᵀ)(L - l lᵀ)) >= 0

rides beside them as a per-location latent variance: a GP node stamps it
(`_input_jitter`, blended over the experts as the variances are; nothing
reads it in training, so the graph prunes its contraction), the nodes
acting linearly carry it as a variance (`Linear`, `SelectInput`,
`LinearCombination`, `Add`, `Concatenate`, `Bias`, `Scale`) and the others
drop it, `predict` returns it on its tuple, and `_predict_raw` and
`measurement_batches` hand it to the likelihood. A continuous likelihood
integrates it beside its noise: a value is `E[g(z + sqrt(jitter) eta +
eps)]` over eight Gauss-Hermite nodes of `eta` for each noise node, and a
measurement sample draws it, paired node by node with the noise through a
rank-1 lattice and rotated per location and realization from a stream of
its own. One draw serves every component of a location, their jitters
arising from one uncertain input. A categorical likelihood reads its
probabilities off the moments, which already hold it. Chosen over
realizations at drawn inputs, which turn spiky
(`docs/benchmarks/uncertain_input_realizations.py`).

Measured against realizations at drawn inputs with the same normals, on a
synthetic field through a sinh-arcsinh warping that bends (skewness 0.8,
tail weight 0.6), 15 locations, 4000 realizations: the jitter is the
quadrature's `tr((R Rᵀ + alpha alphaᵀ)(L - l lᵀ))` over 4000 points of
the input within 2% for the Gaussian kernel and 9% for the Matern52 at 64
nodes, and the latent variance it restores is the mixture's to 0.3%
(Gaussian).

| Kernel | var / r² | Prediction, with | without | Measurement quantiles, with | without |
|---|---|---|---|---|---|
| Gaussian | 0.1 | 0.011 | 0.206 | 0.045 | 0.277 |
| Gaussian | 0.5 | 0.011 | 0.265 | 0.045 | 0.319 |
| Gaussian | 1 | 0.009 | 0.258 | 0.043 | 0.318 |
| Matern52 (64 nodes) | 0.1 | 0.009 | 0.191 | 0.035 | 0.271 |
| Matern52 (64 nodes) | 0.5 | 0.014 | 0.236 | 0.047 | 0.307 |
| Matern52 (64 nodes) | 1 | 0.025 | 0.227 | 0.054 | 0.301 |

The prediction's error is relative to its mean, the quantiles' (5% and
95%) to the interval's width. What is left in the quantiles is the mixture
not being Gaussian: at the latent scale, with no warping and no noise,
4 to 6% as well. A model whose inputs are certain passes no jitter and
takes the code it always took.

## Training on uncertain inputs

The `GaussianInput` gate's two cases (`docs/benchmarks/gaussian_input.py`,
run by `second_moment.py train`), the Gaussian kernel, three seeds, 250
iterations; rmse, coverage of the central 90% of a measurement and CRPS on
held-out data, and the seconds a model took:

| Case | Told nothing | First moment | Second moment | `UncertainInputGP` |
|---|---|---|---|---|
| A: eight inputs, 30% of entries missing, given their conditional moments; test rows uncertain too; 100 inducing points | 1.701 / 0.877 / 0.894, 5 s (imputed) | 1.590 / 0.890 / 0.816, 4 s | 1.591 / 0.907 / 0.820, 11 s | 1.565 / 0.903 / 0.812, 49 s |
| B: Walker Lake, reported locations off by sd 5, test locations exact; 255 inducing points | 176.9 / 0.927 / 99.2, 6 s | 184.1 / 0.928 / 103.5, 5 s | 180.3 / 0.933 / 101.5, 61 s | 178.0 / 0.938 / 100.3, 55 s |
| B, sd 15 | 205.0 / 0.928 / 115.8, 6 s | 231.4 / 0.919 / 130.5, 5 s | 220.2 / 0.930 / 124.4, 61 s | 220.6 / 0.927 / 124.9, 55 s |

The second moment improves on the first in every case -- the coverage in
A to nominal, the rmse and CRPS in B -- and ties `UncertainInputGP` at a
fraction of its time on eight inputs. A location error is still better
left untold, as the `GaussianInput` gate found: a noise term absorbs it.

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
