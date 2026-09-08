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

**S — Surface I/O residue.** OBJ/PLY/STL both ways, the vendor formats, and
any attribute travelling with the geometry. Nothing has demanded them yet.

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
