# Roadmap: open work, and what has already been decided

What is being considered, what was tried and dropped, and what was refused
without trying. The second and third parts matter as much as the first: a
rejection here carries the numbers that killed the idea, so it does not come
back around.

Sizes are of the work, not the value: **S** a session or less, **M** a few
sessions, **L** a workstream with its own design record, **XL** research
with an uncertain end. Items marked *[geostat]* came from a 2026-08-16
survey of what the geostatistical toolbox has that this package does not.

Code-anchored tasks live as `# CLAUDE:` notes in the source instead; this
file is for the cross-cutting ones. Design records for finished work are the
`docs/*.md` files, published under "internals" on the documentation site.

---

## 1. Modelling and inference

(The tree's leaves as the points of contact with the likelihoods, and
independent trees: **done 2026-09-05, 0.6.10** — see "Settled by
measurement" below for the numbers. `VGPNetwork` takes a list of leaves,
one per likelihood, or a mapping from each variable to its likelihood; the
single node is still accepted and still split, bit-identically. Leaves on
roots of their own work with no join at all, which is what a tree of
inducing points near the drillholes beside a gridded tree for geophysics
needs.)

**M–L — Propagate individual realizations through the tree** (requested
2026-09-08). What happens today: moments at every node. `_GPNode.propagate`
takes the parent's mean and variance and evaluates the expected kernel over
that variance (`covariance_matrix(x, ip, x_var, ip_var)`), so a parent's
uncertainty is integrated analytically, and the node's `simulate` draws
fresh normals about the mean at the parent's *mean* — a parent's
realizations never enter a GP node. The pathwise ride the 0.6.9 protocol
built is over the *operation* nodes only (`Linear`, `LinearCombination`,
`Stack`, ...); `GPWalk` steps by the walker's mean. Across leaves, the
realizations are paired only through a shared parent's own draw (stateless,
so realization *k* of a shared parent is the same in every leaf) and
otherwise unrelated: no leaf reads another's realization. Two wants, worth
separating. **(1) Within a tree, through a GP node**: the child evaluated
at each parent realization `x^(k)` with the plain kernel (`x_var=None`),
one draw each — the doubly stochastic route (Salimbeni & Deisenroth 2017)
beside the moment-matching one the bound is trained with (Damianou &
Lawrence 2013; the expected kernel of Titsias & Lawrence 2010). Cost:
`n_sim` kernel evaluations of `[n, m]` where there is one, looped over
realizations so memory stays at one; the whitened `chol_r @ rnd` is shared.
Prediction-only first — training keeps the moment bound, so the ensemble's
spread then differs from the variance the bound optimized; measure coverage
and CRPS both ways on a deep model, held-out. Training by sampling is a
different model with a different bound. **(2) Across leaves, the case
asked for**: assays controlled by lithology should respect each
realization's boundary — realization *k* of the grade conditioned on
realization *k* of the rock. Needs (a) an edge from the rock's leaf into
the grade's tree, which the network never has today — the same missing edge
and per-realization gate the "Mixtures of GP nodes" item names as its
second design; (b) a per-realization gate: the rock realization's label
(the likelihood's rule, `ind_skew`'s best category, applied today to the
leaf's sims inside `_CategoricalLikelihood`) selecting, per realization,
which regime's field the grade draws from; (c) the sweep carrying the
rock's realizations to the grade leaf — `_predict_raw` predicts the leaves
one by one on the same stateless seed, so a shared parent already pairs
exactly, and a gate node would pair the same way once the rock's sims are
stamped as sweep state. This is the geostatistical cascade — domains
simulated, then grades within each realization's domains — made joint.
Gate: a synthetic two-regime field with a boundary; the paired ensemble's
tonnage-above-cut-off distribution against the truth, versus today's
unpaired ensemble and the hard-domained workflow. Both (1) and (2)
presupposed the sibling-normals fix, done 2026-09-10 (settled below).

**M–L — Merging subcompositions** (requested 2026-09-10). Some
compositional data sets are missing data in a structured way: the same
elements unassayed in a large share of the samples, as a campaign that ran
a shorter suite leaves them. `_prepare_composition` marks a row missing
entirely when any part is missing, so one composition over every element
throws those samples away, and modelling the sparse elements as a separate
variable gives up the closure. The proposal: two compositions. The
*master* holds the elements every sample has, the partially missing ones
aggregated into its rest (`rest=True` already does that for whatever is
not listed). The *subcomposition* holds the partially missing elements as
shares of the master's rest, trained only on the samples that carry them.
Correlations between the two sets of latent variables may link them. After
prediction the two are joined realization by realization: the master's
rest `r` in each realization is split into that realization's
subcomposition proportions `q`, part *i* becoming `r·q_i` of the whole,
and the prediction, the quantiles and both variances are taken over the
joined realizations -- never the product of the two predictions, which
drops whatever correlation links `r` and `q`.

What exists: two compositional variables on one container, each with its
own likelihood and warping, on two leaves of one tree (0.6.10). A shared
parent is what correlates them, and it is also what pairs their
realizations: realization *k* of a shared parent is the same in every leaf
(the item above), while two separate GP leaves draw independently since
the node-keyed draws. A sample without the partially missing elements is a
missing row for the subcomposition's likelihood, which training already
skips. `container.derive` already applies a function once per realization,
walking the realizations in bands.

What it needs: (1) the subcomposition's own rest -- what the master's rest
holds beyond the partially missing elements -- without which the split
would hand the whole rest to them. Building it means dividing those
elements by the master's rest in fractions of the whole, with the crowding
problem `_prepare_composition` already settles for a rest (samples where
they account for all of it or more). One constructor taking both groups
from one table is the natural door, units carried as 0.6.10 carries them.
(2) The join, into one `CompositionalVariable` holding every part in its
own unit -- `derive` would give loose continuous variables, one per part.
(3) Block support: a block's stored realizations are averages over its
sub-blocks, and the average of a product is not the product of the
averages, so on a block model the join belongs inside the prediction, per
sub-block, before `_aggregate`; on points the stored realizations suffice.

Gate: Jura's seven metals with two hidden over a spatially contiguous share
of the samples, as a shorter campaign would leave them. The joined model
against one composition over the complete rows only, with the full-data
model as the ceiling: rmse, CRPS and coverage on the hidden elements, the
common elements no worse, and every joined realization summing to the
whole.

**M–L — *[geostat]* Censored observations (a Tobit likelihood).** Plan
proposed and parked. An assay reported as `<0.01` is substituted with half
the detection limit by universal practice, and that biases exactly the low
tail every cut-off calculation reads. The contribution is
`log Φ((warped(dl) − μ)/σ)` — the latent quadrature `log_lik` already takes,
integrating a CDF instead of a density; interval-censored is the two-term
version. Structurally it is partial missingness, so `warping.elementwise`
already says when it is admissible. What it needs: a censoring mask that
reaches the likelihood, a `censored` role in `drillhole.py`, and a decision
on what `predict_measurements` reports below the limit. The open design
question is the user's fan-out alternative — K rows per censored
observation combined by log-sum-exp — against the direct term.

*The mechanism already exists, half-built.* `_Variable.training_input(idx)`
is the one channel by which a variable hands the likelihood something per
row at training time. Its payload today is dead: the only producer is
`RockTypeVariable`, whose `is_boundary` both categorical `log_lik`s take in
their signature and have never read. Its *door* is live —
`DerivedVariable` overrides it to refuse a derived variable at the training
door. So a censoring mask is not a new mechanism but the first real payload
this one carries. Two things to fix when it becomes load-bearing:
`train_svi` passes empty dicts instead of `training_input(idx)` (a
commented-out attempt sits beside it), and `training_input()` is called once
for the whole data set in `train_full`, so the per-batch indexing has never
been exercised.

**M — Survey error as a location variance from the drillhole.** The use
case for `GaussianInput` on coordinates: `as_point_data` returning a
`GaussianData` whose variance grows down the hole from a declared survey
accuracy, by minimum-curvature error propagation of dip and azimuth. Without
it there is no honest source of coordinate variance. The primary stated use
— high-dimensional inputs with missing entries — also has no container
helper; the benchmark builds both by hand.

**L — *[geostat]* Locally varying anisotropy, as a transform.** Every
transform is a globally constant ellipsoid, so the package cannot say that
the structure turns. The literature does LVA with graph distances because
kriging has nowhere else to put the orientation; that route is
non-differentiable and its metric is not Euclidean, so the induced
covariance is not positive definite in general. geoML has a better route it
is not using: `StructuralField` already fits an orientation field, so an
`AnisotropyField` transform reading local orientation from it is LVA
end-to-end differentiable, trained by the same ELBO as everything else.
Folded stratigraphy and vein swarms are where kriging visibly fails. Open:
fit the field and freeze it, or train it jointly and risk orientation and
range trading against each other unidentifiably. Measure the frozen version
first — it is also the fallback.

**L — *[geostat]* Truncated plurigaussian.** The lithotype rule encodes
which domains may touch which, which is real geological knowledge
`CategoricalGaussianIndicator` cannot express — it treats categories
symmetrically. The maths fits: the likelihood is an orthant probability over
two correlated latent Gaussians, and the Sobol machinery already integrates
things of that shape. The hard part is the interface for the rule diagram,
not the integral. Weigh against what the indicator formulation already buys
and PGS has no equivalent of: contacts as the zero level set of `ind_skew`,
which the block refinement and the surface extraction both read. Reopens the
shelved transition-counts diagnostic in §3, which shares the rule.

**L — Mixtures of GP nodes for multimodal distributions.** Nothing can say
"this field is drawn from one of K regimes": every composition stays
unimodal-Gaussian in the latent, and multimodality can only come from the
warping, which is a global marginal reshaping rather than a spatial
selection. Two gating designs, and the second is the interesting one: a free
spatial gate, or **the gate read from a modelled rock type**, which is the
classic domain-then-grade workflow made joint and *soft*, so domain
uncertainty finally propagates into the grade instead of being frozen by a
hard contour. To settle first: the ELBO of a mixture latent, the simulation
path (draw the gate per realization so realizations are single-regime and
the multimodality lives across the ensemble), and the new kind of edge the
second design needs — a gate reading another variable's latent couples two
likelihood heads, which the network never does today. Gate: a deposit with
genuinely regime-split grades, scored held-out against the hard-domained
workflow; the soft gate must beat the hard cut to earn its complexity.

**XL — Change of support from the theory of sampling.** From the author's
paper draft, which is the specification. A window of volume `v` has grade
`g ~ Beta(μ s(v), (1−μ) s(v))`: the mean pinned to the block grade at every
support, all support dependence in the concentration. What the package
needs: the support must become model-visible (today the sample length is
deliberately metadata); a `BetaSupport` likelihood, genuinely non-Gaussian
with a link, whose noise is not additive and not integrated out by the
existing machinery; and a decision on how it composes with the sub-block
aggregation. It *is* an analytical change of support — predicting at block
support means evaluating at `s(V_block)` rather than fanning out
sub-blocks. A doctrine inversion worth flagging: compositing to equal length
homogenizes support, which is exactly the signal the texture parameters
need, so heterogeneous support stops being a nuisance and becomes data.

Support and censoring want the same channel. **Decide once, for both**,
whether they ride the data object (the `GaussianData` precedent, which also
reaches prediction) or the variable channel (training-only, needs nothing
new).

---

## 2. Fitting and initialization

The theme is measured rather than assumed: **training does not leave the
basin it starts in**, so where a start can be computed rather than guessed,
it should be.

**M — Gradient-free training experiments.** Closed against three conditions
in 2026-09-03 (see below) and kept here only for the part that survived: the
memory saving is real and available today, since taking the ranges off the
tape cuts peak GPU memory 3–4× and the step 1.5×.

---

## 3. Diagnostics and validation

**M — *[geostat]* Transition/adjacency counts for categoricals.** Shelved:
return to it only after the truncated-plurigaussian item, since the two
share the lithotype rule and that decision settles what this figure must
read. Which categories touch which, from the drillholes against the same
counts read from the model's realizations. It answers what the confusion
matrix cannot see — *does the model create contacts the geology forbids?* —
and it is the empirical half of the plurigaussian question. Design settled:
both tables come from the same point sequence, ordered by the `HOLEID` and
`DEPTH` metadata; and because a model that reproduces every sequence at the
holes can still break the rules away from data, a second reading through the
volume is owed. Report three tables — data, model at the holes, model in the
volume — as row-normalized frequencies, with raw counts kept for the
forbidden cells.

**M — *[geostat]* Resource classification from the ensemble.** Shelved
until there is a way to define mining volumes. The criterion practitioners
want is the relative error of a production-volume aggregate, read off the
simulations, and every input exists. Design settled: the function reads one
thing, a partition of blocks into aggregates as a metadata column with
weights, produced three ways — a panel size, a metadata column the user
already has (a schedule naming what a quarter actually mines), or a list of
solids turned into that column by the fraction each block sits inside.
Read it after calibration: the ladder measured 0.86 coverage at nominal
0.90 on Jura, and a relative error off overconfident intervals flatters the
deposit by exactly that.

**S — A `Mixture`'s measurement samples bisect over the whole batch**
(measured 2026-09-09). Rotating the noise nodes per location made the
mixture quantile's sixty bisections run over `(n_nodes, n, size, n_sim)`
rather than over `n_nodes` values: 18 s against 0.1 for a 20 000-row batch
at the defaults on the CPU, 2.6 against 1.7 on the GPU. Only the
`Mixture` likelihood pays it, only through `predict_measurements`. If it
matters: bisect once on the unrotated node grid per component and
interpolate the rotated `u` through the monotone quantile, or halve the
iteration count (a bracket the width of the widest component reaches
1e-14 in about fifty).

**S–M — `noise_variance` off an eight-node quadrature for an exponential-
tailed noise through a convex link** (measured 2026-09-10). The second
moment `integrated_backward` writes comes off the same eight Gauss–Hermite
nodes as the value, mapped through the noise law's quantile. For a
Gaussian law it is exact to 0.3% against a 200-node reference; for the
epsilon-insensitive (Laplace, once `epsilon` trains to zero) through Jura's
spline chain it reads +19%/−20% of the exact law (Cu at the median and
the 90th percentile latent) and 0.63× under a Box-Cox link, the outermost
node pair carrying 10–50% of the value. The quantity itself is the
problem: with a log link the Laplace tail's second moment has a pole at
`2·sigma_log / c_rate = 1`, and Cu's fit sat at 0.956 — ±5% of `c_rate`,
under a nat of bound, moves the data-unit variance between 2.6 and 9.6
times the sill. Two things to do: (a) a diagnostic — compute `2·sigma/c`
per column after training and warn above ~0.8 that the measurement
variance is dominated by extrapolated tail and `noise_variance`
unreliable; (b) more or better-placed nodes for the exponential-tailed
laws (the Sobol path already uses 64), gated on the 200-node reference.
Context: `docs/benchmarks/jura_noise_footing.py` and the record.

**M–L — Cheaper cross-validation** (requested 2026-09-08). Since
2026-09-08 the driver costs one refit per fold and nothing else, on the
full 20k-row Tom v6 model eleven minutes a fold. The requirement, the
author's (2026-09-10): the answer must be general, serving any likelihood
and any network configuration.

*Leave-expert-out (the author's, 2026-09-10).* Remove one
expert at a time from a trained multi-expert model, predict the data with
no retraining, and weigh each point's leave-one-expert-out predictions by
the trained expert weights; a single-expert model is small enough for
`cross_validate`. First look on Walker V, `docs/benchmarks/leave_expert_out.py`
(grid experts at step 26, range 50, `ZScore -> Spline`, 500 iterations, one
seed), rmse against the samples:

| experts | in-sample | leave-out, weighted | leave-out, home expert | refit on home-expert folds | spatial folds (today) | truth, exhaustive grid |
|---|---|---|---|---|---|---|
| 4 | 193 | 317 | 367 | 336 | 223 | 180 |
| 9 | 182 | 301 | 442 | 348 | 222 | 171 |
| 16 | 181 | 276 | 447 | 376 | 222 | 172 |

Leave-out took 1–3 s, against 50–350 s for the refit and 50–110 s for
today's route. Three findings. (1) The literal reset is not a removal: an
expert set to the fresh-init values kept 12–67% of its weight on its home
points, and with `delta` at its upper bound up to 15%; exact removal is a
mask on the weights before they are normalized. (2) Removing an expert
removes *capacity*, not data: the experts are trained jointly and
co-adapt, so the survivors were never taught to cover the removed
expert's ground. The home-expert score sits 9–27% above an honest refit on
the same folds, and at points three or more experts share it is 2.3–2.6
times the in-sample error, not near it -- the overlap leak one would fear
is not what happens. (3) The larger error is geometric: an expert's region
is a far larger hole than the prediction target has, so even the honest
refit on expert-shaped folds reads 1.9–2.2 times the true error, where
today's spatial folds read 24–30% over it. *Shelved 2026-09-10, by the
author's decision, for missing the requirement above:* it needs a
multi-expert model, and it shares the flaw measured below for the
inducing-point filters -- a datum's information sits in the kept
neighbours' trained state, which no removal reaches. Were it taken up
again, its gate stood at the same arms on Jura with experts (its 100
held-out points as the truth) and two more Walker seeds, adding CRPS and
the coverage a conformal cut from each arm's PITs achieves on the truth
set, killed if leave-out stayed more than 20% from the truth on most
layouts; and its build at a non-trainable mask on the multi-expert root,
read wherever the weights are normalized (both `get_expert_weights` sites
in `BasicGP` and in `MultiStructureGP`, the consensus cross-prediction
included, so one trace serves every expert), with `models.leave_expert_out`
returning what `cross_validate` returns.

*Then: neutralizing the fold's inducing points through `delta` (the
author's, 2026-09-10), measured in `docs/benchmarks/neutralized_sites.py`.*
As stated it cannot work: the posterior mean `k_x K^-1 L a + b` never
reads `delta`, so the held-out prediction stays at its in-sample value and
only the variance moves. Removing the points properly, as
pseudo-observations whose targets reproduce the trained mean, reads 17–23%
above the honest refit; dropping them and kriging from the kept means reads
in-sample; a one-step ring overshoots either way. A datum's information is
spread by its kriging weights over every inducing point within reach, so
no assignment of inducing points to folds removes it cleanly -- the flaw
leave-expert-out has too, which makes its gate above likely moot. Putting
the diagonal on the *prior* instead (`cov = covariance_matrix(ip, ip) +
jitter` in `BasicGP.refresh`, the trained state kept) is not well defined:
the whitened mean is a sequence of innovations in the Cholesky order, so
changing some points' prior changes what every later coordinate means.
The same trained model and folds read 243 and 238 (one expert, nine) in
the stored order, 238 and 217 with the order reversed, and 203 and 205
with the fold's points last -- which is exactly the kriging arm, since
that order keeps the kept points' posterior means. A score that moves when
the inducing points are relabelled cannot be trusted, even where one order
lands near the refit. Keeping the inducing means while the prior changes
(the author's three steps: `m = L a`, the diagonal added, `a' =
chol(K')^-1 m`) removes the order and lands on the kriging arm exactly,
since the mean at x becomes `k_x K'^-1 m`: 203 and 205, 4–7% under the
converged refit; with the one-step ring 283 and 272, 27–29% over it. Every
held-out sample sits within 0.35 kernel ranges of a kept inducing point
(median; 0.7 at most), whose posterior mean already carries what the
sample said. `docs/benchmarks/filtered_prior.py` writes two readings
into `BasicGP.refresh` explicitly, each checked against BasicGP at zero
filter and against its own closed-form limit at a filter of 1e6. The
author's definition -- after the three steps, K' replaces K in every
formula, `K' + D` included, nothing else adjusted -- reproduces the
wrapper's mean-kept arm (the same latent variances to three decimals, rmse
within 0.1), so that arm had computed it; its limit is the kriging of the
kept inducing means with the variance `1 - k_k (K_kk + D_kk)^-1 k_k`. The
other reading keeps the trained posterior q(u) = N(m, S) whole, S computed
under the trained prior, and lands at in-sample: 202–206 with one expert,
182–185 with nine. Across k = 4, 5 and 10 the two read 202–207 and
182–205 where the converged refit reads 219–235 and 213–247, and neither
responds to the fold size. Filtering inducing points, whichever way it is
written, predicts from a posterior fitted with the fold's measurements.
The author's projected prior (2026-09-10) is singular as written, both as
`K + Delta - K (K + Delta)^-1 K` and as `K + Delta - K_.k K_kk^-1 K_k.`: its
kept rows and columns are exactly zero, since `K - K (K + Delta)^-1 K =
K (K + Delta)^-1 Delta` and Delta is zero there (4e-16 against K's own 1 on
Walker's layout), so it cannot stand in for K. With K_kk kept on that block
(`projected` in `filtered_prior.py`) it reads 203.1–203.3 with one expert,
the kriging limit again, and with nine it falls with the filter from 423 at
Delta = 1 to the limit by 1000, crossing the refit at a filter that moves
with the fold size: 214.7, 210.4 and 203.3 at Delta = 100 for k = 4, 5 and
10, where the refit reads 246.8, 213.0 and 214.4. Short of the limit the
prior no longer agrees with the kernel's cross-covariance: the latent
variance at the held-out rows collapses to 0.000–0.003 at Delta = 10 and
below, and the expert weights follow it
(`filtered_prior_*_projected*.txt`).
Zeroing the filtered points' means as well changes nothing,
since the prior diagonal has already taken their weight; zeroing them
under the stored prior instead pins the fold's inducing values to the
prior mean, reads 265–389, and needs whitened values past the ±10 bound.
Across k = 4, 5 and 10 (`neutralized_sites_*_k*.txt`) the nearest-location
rule barely moves -- 202–207 with one expert, 201–205 with nine -- while
the converged refit moves with the fold size, 219–235 and 213–247: the kept
inducing points sit beside every held-out location whatever the fold, so
the rule is blind to the one thing cross-validation measures. The row
solve follows the refit at every k with one expert (within 0.4%) and sits
6–13% under it with nine (3–9% with the fold's `delta` at the bound), the
gap largest at k = 4. The same
trick on the *data's* noise diagonal does work: a held-out row at infinite
noise is a row left out, and for a Gaussian leaf with the hyperparameters,
warping and noise frozen the bound is exactly quadratic in the whitened
means and biases, so the fold optimum is one ridge solve over the training
rows (item (1) of the shelf below, reached from this side). rmse against
the samples, Walker V, one seed:

| layout | in-sample | `delta` only | sites | krige | rows | rows, fold's `delta` at bound | refit-200 | refit-1000 | truth |
|---|---|---|---|---|---|---|---|---|---|
| one expert | 202.3 | 203.3 | 268.4 | 203.9 | 218.6 | 218.9 | 225.2 | 219.0 | 187.4 |
| nine experts | 181.9 | 203.6 | 259.7 | 204.1 | 198.6 | 207.3 | 221.8 | 213.0 | 170.7 |

The row solve took 2–4 s for five folds against 83–305 s for the
1000-iteration refit. With one expert it matches the converged refit to
0.2%; with nine it sits 3–7% under it, the spread's role in the expert
weights being what the refit re-solves and the solve does not (the
refit's own score moved 4% between 200 and 1000 iterations, so part of
the gap may be the refit's). Adam's 500-iteration state was itself 2–6%
short of the exact optimum on all rows. *Shelved 2026-09-10, by the
author's decision, for missing the requirement above:* the solve is exact
only for Gaussian or multivariate Gaussian leaves under a frozen warping on
a single layer -- another likelihood needs Newton passes, each a solve like
this one, and a deep tree's interior keeps E2's memory -- and with several
experts `delta` has to be refit beside it. Were it taken up again, its gate
stood at CRPS and the coverage of a conformal cut from its PITs, Jura's
held-out set, more seeds, and a refit run to convergence for the experts.

*Shelved 2026-09-10, by the author's decision, for something simpler and
more general.* The candidates a four-angle review produced; the structural
claims were checked against the code, no saving was measured. (1)
Closed-form refit for Gaussian leaves: with the hyperparameters frozen the
bound is exactly quadratic in the whitened means and biases (the mean is
linear in them, the variance and the expert weights depend on `delta`
alone, the KL is quadratic, the 64-node quadrature of a Gaussian
log-density is exact), so the refit is one ridge solve per column and
every fold comes from one pass, the full-data statistics minus the
block's; `delta`'s optimum never sees the measured values, only where the
samples sit, so a frozen all-data `delta` leaks the held-out *locations*.
(2) Score only some folds of the partition. (3) An honest ridge start for
the fold's leaves. (4) Newton passes for non-Gaussian leaves -- the
rock-type likelihood's objective is the log of a probability under q, not
an expected log-density, so its site weights must be the second derivative
in the mean, not `-2 dl/dvar`, which goes negative on misclassified rows.
(5) Measure the refit's budget and stop on the gradient norm. (6) Refit
only the experts near the fold, with compact folds -- nothing to gain on
Walker, Jura or Tom v6, which are too few ranges across. (7) Newton-CG over
the whole network, interior included -- a small gradient certifies a
stationary point, not the basin a fresh fit would reach. (8) Hoist the
frozen root out of the training step. (9) All folds in one traced loop.
With them go the older candidates this item carried: importance-sampling
leave-out on blocks, the EP-style cavity, and the closed-form leaf under a
frozen interior (E2's verdict stands).

**S–M — Batched prediction from a latent node, into a container**
(requested 2026-09-05). A node's `predict(x, x_var, n_sim, seed)` returns
the raw four-tuple — mean, variance, simulations, explained variance — and
nothing else: no batching, no refresh (a caller must `refresh` the tree by
hand first, or the inducing points still point into the last training
graph), and no container to land in. Everything that makes a model's
prediction usable — `_over_batches`, the refresh-once-and-snapshot of
`refresh_cached`, writing into a variable — lives on `VGPNetwork`, so the
intermediate values of a tree are reachable only by hand: the benchmark
that measured chapter 17's walked coordinates read them through
`interpolate` in hand-cut chunks. Yet those intermediates are what a
reader asks for — where a `GPWalk` moved the coordinates, what a shared
parent says before two leaves diverge, what a `Linear` trend contributes —
for interpretability, or curiosity. Wanted: `node.predict_into(container,
n_sim=)` or `model.predict_node(node, container, n_sim=)`, batched and
refreshed like the model's own `predict`, at any node of the tree. **The
open design question is where the result lives.** Raw arrays are the cheap
answer and lose the tree addressing, Zarr and the plots. A new variable
kind is the better one — a *latent* variable of `size` columns holding
`latent_mean`, `latent_variance` and the simulations, which is exactly the
shape `ContinuousVariable` already keeps for a model's own output, minus
what a node has none of: measurements, a likelihood, a unit, a
back-transform. Declaring it through `_ZARR_ATTRS`/`_DICT_FAMILIES` makes
the frame, pyvista, Zarr, subsetting and carrying free, as they are for
every variable. To settle: whether it sits under a name of its own or
under the variable whose tree it belongs to (`Elements/_latent/walked`),
what `predict` on a block model does with it (sub-block fan-out and
`_aggregate` are the likelihood's, and a node has no likelihood — the
honest answer is point support only, refused on a `BlockSet3D`), and that
the model's own `predict` keeps writing `latent_mean`/`latent_variance` on
the measured variable as it does, this being a second door rather than a
replacement.

**M — Integrated gradients for explainability.** Attribution of a
prediction to its inputs, computed as a quadrature sum of gradients along a
straight path from a baseline — everything here is differentiable with
respect to the coordinates, so the machinery is nearly free. Most useful
where the network takes more than coordinates. To settle: **what to
attribute** (the latent mean, the back-transformed prediction, a category's
probability — each a different function, and the nonlinear warping means
latent- and value-space attributions differ), and **the baseline**, which is
the method's crux and is not obvious for spatial inputs. Attribution of the
*uncertainty* is a separate and possibly more interesting question for
drillhole planning.

---

## 4. Data, I/O and interchange

(Mesh operations through Manifold: **booleans done 2026-09-10, 0.6.10** —
see "Settled by measurement" below for the numbers. `Solid3D`'s union,
intersection and difference go to manifold3d itself and are exact; the
rest of what Manifold offers was measured and replaces nothing of
geoML's.)

**S — Surface I/O residue.** OBJ/PLY/STL both ways, the vendor formats, and
any attribute travelling with the geometry. Nothing has demanded them yet.

**S — `simplify` keeps half its promise.** Found measuring it against
Manifold's (2026-09-10, `docs/benchmarks/manifold_features.py`), on the
Assen shells. The budget is checked one way only, the simplified faces
against the original: at 0.5 m the simplified BIF shell sat within 0.46 m
of the original, and the original's vertices up to 0.72 m from it. And at
2 m both shells come back unchanged after 4–8 s -- the last resort, for a
mesh every cut breaks -- where 0.5 m took them down 22 and 16 times. A
likely cause, unverified: at 2 m the quadric pre-pass is accepted, being
within half the budget, and every gentler cut starts from it, so a body
the pre-pass broke stays broken however little is cut after. The reverse
check costs a locator on each candidate, where today's is built once on
the original.

**L — Import a `BlockSet3D` from CSV.** There is no way to read a block
model somebody else made. The hard part is not parsing: `BlockSet3D` is a
strict octree, and most vendor sub-blocked models satisfy none of its three
invariants, so a general sub-blocked model cannot be represented exactly.
The first decision is what to do with one that does not fit — refuse it,
snap it onto the lattice (changing volumes), or fall back to `PointData`
carrying block size as metadata. Inferring the lattice from centroids and
sizes is mechanical: the base cell is the elementwise minimum, and every
distinct size must be `base × discretization**k` per axis, which is a
stronger test than a common divisor and exactly what the lattice needs.
Fullness is the other snag, since exports usually hold only the blocks
inside a wireframe; filling the gaps with unpredicted blocks and a filter
matches the always-full-plus-filters design. Import-time checks for overlaps
(via octree addresses, no lattice painting) and gaps are both required.
**Test on a real Micromine export before building.** Only if real files
demand it, phase two is a lattice-free `FreeBlocks3D` with per-block sizes
as a real attribute — block-support prediction and tonnage still work,
refinement and contour-cutting do not.

---

## 5. Housekeeping, docs and release

**S — Verify the tag-triggered manual CI job.** The `full` job passed at
v0.6.9 for the first time since v0.6.5, but `manual` alone was still killed
at 22 minutes of a 90-minute budget with no pytest output — the OOM
signature of a 16 GB runner accumulating TensorFlow, matplotlib and pyvista.
`test_manual.py` now runs one subprocess per chapter. Owed: a dispatch, then
read the `manual` job. If it ever dies the same way again, the next lever is
not a longer timeout.

(Chapter 16's figures: **verified 2026-09-05** — rerun through the manual's
own runner after the leaves change, the seeded chapter reproduced every
committed figure byte for byte, and the §16.5 prose matches what the model
prints: Portlandian at a balanced accuracy of 0.5 and a Jaccard of zero.)

**S — Retire the 0.6.0 deprecation shims in 0.7.0.** Ten one-line modules.
The check that decides the item has been run: `persistence._resolve` replays
the dotted path recorded in a save, so any module named by a save must stay
importable forever — but none of the ten defines a `Parametric` subclass,
the only kind of thing a model store records, and container stores dispatch
on bare class names. The warning half is done and tested; what is left is
only the deletion. Re-run the persistence check before doing it.

**S — `latent/network.py` stays out of the pyright list.** About 28 of the
file's diagnostics read `inducing_points` as possibly-None, and they cannot
be fixed by declaring them: `None` is load-bearing in the finished state.
Declaring `tuple` took 33 diagnostics to 67 and was reverted. Do not repeat
that attempt.

---

## Settled by measurement

These were tried. The numbers are why they are, or are not, in the package.

**Solid booleans through manifold3d — done** (2026-09-10, 0.6.10).
`docs/benchmarks/manifold_booleans.py` and `manifold_features.py`, on the
six Assen rock shells (325k–681k triangles each, at mine coordinates). The
bodies go to manifold3d welded, in double precision and in the pair's
local frame, and a status other than NoError is raised rather than
returned empty: the 15 pair intersections, a union and two differences
took 15 s against the signed-distance grid's 141 s, with no crash in 54
isolated calls, every answer a consistent `Solid3D`, and the block shells
that crashed VTK's filter pass in-process. The rocks meet along films
thinner than the grid's 1.1–1.75 m step, and its 14 non-empty
intersections read 5 to 5231 times the exact volume, or next to nothing
(0.0002 m³ against 1.70); the union and the differences agreed to
0.01–0.44%. **Not the `pyvista-manifold` accessor** the item asked for:
it casts to float32, and gave inconsistent meshes in 4 of 18 jobs in the
local frame and 15 of 18 at mine coordinates, so its pyvista 0.48 floor is
not needed either. **Nothing else of Manifold's replaces geoML's own**:
`Manifold.simplify` is 20–60 times faster but strayed past its tolerance
on real shells (at 0.5 m, 0.88–0.97 m out and 1.1–2.7 m back; at 2 m up to
5.4 m out and 14.6 m back, and an inconsistent mesh on Hematite);
`decompose` finds `split`'s pieces but costs as much once they are handed
back, and takes solids only; `level_set` wants a function rather than
samples and ran 5–8 times slower than flying edges, for half the volume
error at twice the triangles. Measuring the films exposed `signed_volume`
summing about the world origin, fixed the same day (a millimetre film 23%
off at a northing of 7,000 km). What the comparison found open in
`simplify` is an item of its own in §4.

**A vector variable's components never received their latent moments —
fixed** (2026-09-10, 0.6.10). `VectorVariable.update` now forwards `mean`
and `variance` when the model says the latent columns are the components'
own (`elementwise=`, from the likelihood's warping); a rotation or a
projection leaves them NaN rather than mislabel a mixed column, and a
composition's parts stay None by design. `test_plots.py`, both ways.

**Sibling GP nodes drew the same whitened normals — fixed** (2026-09-10,
0.6.10). Measured 0.995 cross-leaf correlation of the latent realizations
on Jura, 0.000 with the seed offset; the node's name, numbered within the
tree and replayed by a save, is now folded into the seed at the one draw
function; under the Sobol rule it also permutes the realization order per
node, since two linear-matrix scrambles of one sequence stayed paired at
0.37; under 0.2 after, both rules, replay kept, `test_leaves.py`. The experts of one node still share their normals
across the overlap, deliberately.

**Jura's metals: the likelihood is the lever, not the link** (2026-09-10).
The parametric links were the first recommendation for copper's fan at
three times the data under the epsilon-insensitive likelihood; measured,
none of them fixes it (Box-Cox 3.2, Yeo-Johnson 2.2, robust ZScore 2.7),
and trained to 1200 iterations the excess spreads to six metals
(1.3–2.6×). A Laplace tail through a log-like link has a second moment
with a pole at `2·sigma_log/c_rate = 1`; copper sat at 0.956. The
multivariate Gaussian on the same chain puts every metal within 0.97–1.41,
goodness 0.86 → 0.96, rmse equal, CRPS within 1%, confirmed on the true
held-out set — chosen against a bound 200 nats lower. The two-scale
mixture's tight fan is under-dispersion (goodness below the baseline at
64 draws); not recommended. Three skeptics; record in
`docs/cross-validation.md`, `docs/benchmarks/jura_noise_footing.py`.

**Measurement samples on rotated nodes — done** (2026-09-09, 0.6.10).
The strata's midpoints carried 35-47% of `EpsilonInsensitive`'s noise
variance on Jura's Pb, Cu and Cr at the default 32 nodes and put one
noise value on every location of a column. Rotated modulo one by a
uniform per location, component and realization from the model's seed:
the samples' warped-space variance at 0.985-1.014 of the law's on every
metal and on Walker, their variogram on the lifted ground fan (Zn
0.97-1.00 by lag, Walker 0.999-1.001) where it sat 1.3-2.8x below,
locations decorrelated, batch invariance kept. `test_measurement_samples.py`.

**The variogram figure's noise lift is exact; the gap it shows is the
model's** (2026-09-09). The fan is the ground realizations (noise
integrated out) raised by the pair-averaged `noise_variance`; against an
honest Monte Carlo measurement fan (noise drawn independently per location
from the fitted likelihood, back-transformed) the lifted fan sits at
0.993-1.002 on Walker and 0.95-1.03 on every Jura metal under Gaussian and
epsilon-insensitive likelihoods with a spline warping, within Monte Carlo
noise at every lag, three measurers each audited and re-run by a skeptic
at another seed. Nor is in-sample conditioning the gap: the same fans on
the out-of-fold container move by 0.02-0.05. What remains is the model's
own total variance (Walker: noise 0.88 of the sill plus ground 0.48, a
fresh measurement scattered 1.36 times the data). Record in
`docs/cross-validation.md`.

**The leaf-only refit leaks, and more than the warm start** (2026-09-09).
`refit="leaves"` — re-initialize and refit the variational state of the
terminal GP nodes only, the interior kept as all the data taught it, the
author's proposal that the interior encodes the spatial pattern and the
conditioning to data happens at the leaves. On chapter 16's Jura tree
(the displacement field the interior), five spatial folds, three seeds:
rmse/sd 0.80 against the scratch gold's 0.99 and the warm start's 0.92, rock
out-of-fold accuracy 0.91 against 0.83 (in-sample 0.97). The same refit
from an interior trained on the fold's rows alone scores 1.00 — the gold.
The field remembers where the held-out holes put the contacts, and
freezing it keeps that memory intact where warm training lets it drift.
Kept as a diagnostic, not a score; E2 in `docs/cross-validation.md`.

**Independent trees work, and the bookkeeping join is gone** (2026-09-05).
Two leaves on roots of their own — a `BasicGP` on 40 k-means inducing
points for the rock type, another on 120 for the seven metals, no join
anywhere — against the same two leaves on one shared root of 120, on Jura
held out, three seeds, 600 iterations: the metals identical (rmse/sd 0.949
against 0.950, crps/sd 0.477 against 0.478), the rock five points of
accuracy worse (0.630 against 0.680) on its own coarser tree, the bound
lower by the same token (−6167 against −6122), and the two-tree model a
third faster to train (33 s against 44 s). So a tree costs what its own
inducing set costs and affects only the variable it carries, which is the
property the drillhole-tree-plus-geophysics-tree use needs. Two facts from
the way there: a terminal `Stack` already joined two trees (it propagates
no inducing points, so nothing can sit on it), while a terminal
`Concatenate` over two roots raises, its `root` being `None`; and the old
single-node path survives the refactor to the last bit — twelve iterations
of the manual's Jura model, same bound before and after.

**A full-covariance variational family is not the answer to interval
tightening — a richer family makes it worse** (2026-09-02). A `FullGP` with
a full whitened Cholesky per output reached a higher bound than `BasicGP` at
every size of the Jura ladder, and its held-out intervals came out *tighter*
still, coverage lower (0.87 → 0.80 at 625 in four experts), rmse worse once
divided into experts. It was also 4× faster to train at 1225 points. So a
better family is cheap and buys nothing here: the saturation is the bound's
own optimum on this data. Prototype deleted.

**The intervals keep tightening past 625 inducing points, but toward an
asymptote — and on Walker it is not a defect** (2026-08-19). On Jura,
predicted-sd/data-sd falls per doubling by 0.045, 0.044, 0.033, 0.020,
0.010, flattening toward 0.75 with coverage settling near 0.86. Over the
extended ladder Jura's accuracy is no longer flat — rmse/sd 0.937 at 169
inducing points to 0.967 at 2401, monotone and far beyond the seed spread —
so past roughly the data count this data set shows mild genuine overfitting.
None of it appears on Walker, where rmse/sd improves across the whole ladder
and coverage never crosses nominal.

**Gradient-free training, against three conditions** (2026-09-03). The
memory saving holds, for the ranges only: off the tape they cut peak GPU
memory 152 → 40 MiB and the step 302 → 202 ms on a 16-expert model. Better
optima do not: the bound's profile along the range is unimodal on Jura and a
flat plateau on Walker, with Adam sitting on the optimum in both — no hidden
basin for a search to find. The better *held-out* optima sit off the bound's
peak on both, which a bound-driven search cannot see. Ruled out along the
way: evolution strategies and SPSA over everything, whose gradient-estimate
variance grows with a dimension that here is thousands to millions.

**The `Isotropic` range floor does not bind** (2026-09-03). The pinning that
raised the question was an artefact of a profile that froze the kernel's own
ranges. With them free the range never sits on the floor, and every
held-out score is equal or a hair better at the current floor than at any
lower one.

**Variogram-based kernel initialization — rejected** (2026-08-19, the
author's call): "too much work for something this library exists to
replace." The ELBO is the fitting surface; the experimental variogram stays
a *check*, never a fitting target, not even as a start.

**MAF, min/max autocorrelation factors — shelved** (2026-09-03) "until I'm
proven wrong". A warping acts on the outputs alone and never sees the
coordinates, while MAF's lag-h side presumes a stationary cross-covariance;
the latent network need not be stationary and in this package's spirit is
not expected to be, so a spatial assumption baked into an output transform
is the wrong layer for it. Reopen only with a measurement showing an MAF
start beating the existing ones on a non-stationary case.

**Flow warpings are correct but not superior** (2026-09-01). Both the
continuous normalizing flow and the tensor-product variant are right on
every test; the CP flow is the best Jura arm on rmse, neither wins Macpass
against the marginal chain, and they cost 4–10×. The standing
recommendation stays one initialized marginal transform.

**OMF as the interchange route — dropped** (2026-08-12). The package on
PyPI is from 2019; v2, the only spec with sub-blocked models, has been at an
alpha since 2021. Forking it or launching a new format was raised and
rejected by analysis: a format's value is its readers, a fork moves no
vendor's roadmap, and the maintenance tax lands on the hours the modelling
needs. CSV is the only universal route for sub-blocked models.

---

## Refused without measuring

Each of these is refused on a structural mismatch, which is a weaker kind of
no than a number. Revisit if the premise changes.

- **MPS / training images.** No likelihood, no parameters, conditioning by
  pattern search. Importing it means carrying a second, incompatible
  inference story beside the ELBO. This package's answer to curvilinear
  structure is a deep GP or a flow, inside the objective it already has.
- **Graph-based LVA distances.** Non-differentiable, and the shortest-path
  metric is not Euclidean, so the induced covariance is not positive
  definite in general. Superseded by the `AnisotropyField` route above.
- **Hermite polynomial anamorphosis.** The spline warping is better
  conditioned and exactly invertible; Hermite adds tail oscillation and a
  truncation order to tune, for nothing needed here.
- **IRF-k / generalized covariances.** The formalism exists to avoid
  estimating a trend one cannot afford to estimate, which is not the
  situation here; a mean function or a `Linear` node is more direct.
- **QKNA.** A diagnostic for a kriging neighbourhood search that does not
  exist here. The one piece worth stealing, the slope of regression, is what
  `prediction_scatter` already draws.

### Already here under another name

Written down because the survey kept rediscovering them, and because the
mapping is the manual's vocabulary bridge. Kriging and cokriging → the
posterior, `LinearCombination`, `MultiStructureGP`. Normal score →
`ZScore`/`Spline`. Sequential Gaussian simulation, turning bands, FFT-MA →
posterior simulation. Indicator kriging → `CategoricalGaussianIndicator`.
Logratio → `CenteredLogRatio`/`ScaledSimplex`. Capping and top-cuts → the
`Mixture` likelihood, which *names* the bad sample instead of truncating
everyone. Spatial bootstrap → the ensemble, directly. Cell declustering →
`math.geometry.declustering_weights`. **Uniform conditioning and the
discrete Gaussian model** need a specific note: the sub-block simulation
route is strictly stronger, needing no permanence-of-distribution
hypothesis, so the change-of-support likelihood above is the remaining
piece and DGM is not a reason to import anything.
