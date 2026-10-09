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

**L — The expected kernel: deep networks that stop overfitting** (agreed
2026-10-08, for 0.9.0; **built 2026-10-08** on branch `expected-kernel`,
design record `docs/expected-kernel.md`: gates 1-3 passed -- the kernel
within 0.0008 of 4e5 draws for every kernel, a leaf within 0.005 of draws
of its parent's posterior through every supported operation, the folded
section's deep networks at -9.0 and -11.1 a new hole against -25.4 and
-18.9 under the old rule and -12.2 for a VGP, calibrations 1.19 and 1.31
-- and the walk's first gate only at a short reach, see below). Found in the research
project *Spatial cross-validation* (`report/new_covariance.md`,
`report/dgp_propagation.tex`): a GP node hands its children only its
variance at each location, the child treats any two locations as
independent and inflates its range by half the input variance where the
Gaussian kernel's own weight is 6, so a deep model's leaf barely feels the
hidden layer's uncertainty. Its predictive variance read 0.002 of the
model's own Monte Carlo predictive, the bound bought a tiny noise variance
with that, and on a folded synthetic section scored per new drillhole:

| Model | Log score per new hole |
|---|---|
| geoML `dgp` (marginal propagation) | −155.2 |
| geoML `dgpc` (input-connected) | −31.2 |
| VGP | −12.15 |
| expected kernel, `dgp-cu` | −9.26 |
| oracle on the unfolded coordinates | +3.1 |

The expected kernel `E[k(h(x), h(x'))]` needs the variance of the
*difference* `h(x) − h(x')`, so a node passes its children its covariance
between locations; the child stays an ordinary sparse GP on the root's
inducing points at the same O(N·U²) cost, its KL and the likelihood door
unchanged. Decided:
- **An option, old saves untouched**: `GPOptions(propagation=)`, the class
  attribute `"marginal"` (what an old save falls back to) and the
  `__init__` default `"joint"`; under `"marginal"` the code is today's.
  `expert_propagation` likewise defaults to `"independent"` for new models.
- **What travels**: per output, an N×U covariance (data against inducing
  points, made in `propagate`) and a U×U one (inducing points against each
  other, made in `refresh`). Mixing operations apply their rule entry by
  entry, as to the variances today, so the covariance *between outputs* is
  dropped after a `Linear` as it is now. A GP node's own posterior
  covariance against its inducing points is `k_x (K+Λ)⁻¹ Λ`.
- **The signature**: `propagate` returns a private named tuple
  `_Moments(mean, variance, covariance)`, each a list over experts,
  `covariance` `[size, n, m_e]` or `None` (a deterministic node, or
  marginal mode); a refresh stores `inducing_points_covariance`
  `[size, m_e, m_e]` beside the variance, which stays its diagonal; slots
  carry the same with a leading slot axis.
- **Experts chain per expert**: child expert *e* reads its parent's expert
  *e* at the data and at its inducing points; moments stay per expert along
  the chain and are blended only where a value leaves the tree (a leaf,
  `predict_node`). `"joint"` with `"consensus"` is refused. Slots and
  subsets work under `"joint"` in 0.9.0, the subset gate ("every expert in
  the subset changes nothing") held on shallow and deep trees.
- **Kernels**: the Gaussian in closed form; `Exponential`, `Matern32`,
  `Matern52` and `RationalQuadratic`, scale mixtures of Gaussians, by a
  fixed-node 1-D quadrature over the mixing measure. `Spherical` and
  `Cubic` are **not** scale mixtures (measured 2026-10-08: a Spherical
  Gram matrix in R⁶ has eigenvalue −0.026, and the best non-negative
  Gaussian mixture misses Spherical by 0.020 and Cubic by 0.015 against
  6e-7 for Matern32), and are refused under a random input with Cosine
  refused in every GP node of a network; the remaining kernels cover the
  uses.
- **Nodes**: supported under `"joint"` are the inputs (`GaussianInput`'s
  location variance entering the difference with no covariance against the
  inducing points), `SelectInput`, `Linear`,
  `Scale`, `Bias`, `Add`, `LinearCombination`, `Concatenate`, `BasicGP`,
  `MultiStructureGP`, `AdditiveGP` and `GPWalk` (below); refused under a
  random input are `UncertainInputGP` and `RadialTrend`, and a GP on a
  `GradientConstrainedInput` (item below); a `GPWalk`'s field takes a
  mixture kernel too, the walk reading it at uncertain positions;
  `Multiply`, `Exponentiation`, `GaussianMixture`, `ProductOfExperts` and
  `Stack` never feed a GP and keep their moments.
- **Every refusal lives in `VGPNetwork`'s constructor under `"joint"`**,
  never in a node's, so a save under `"marginal"` keeps opening.
- **Realizations**: a GP node draws as now, `K_eff(x,Z)·chol_r·normals +
  mean` -- the right marginal and the explained covariance, not conditioned
  on the parent's realizations (that is the item "Propagate individual
  realizations" below, a different model).
- **Changed while building**: the walk's own KL stays (measured
  necessary, see below), and the walk carries the field's unexplained
  variance.
- **Left out**: centring (identity plus amplitude) and the node-role rule;
  the input connection is the recommended construction, which the manual
  says, and nothing enforces it.
- **Gates**: (1) the expected kernel against 2×10⁴ joint Monte Carlo draws
  of the parent, max error under 0.005, on one expert, several, and three
  layers; (2) on the folded section, 200 new holes, `"joint"` beats
  `"marginal"` and the VGP on mean log score with calibration 1–1.5;
  (3) chapter 5's deep GP rerun under `"joint"` and recorded. Nothing ships
  unless (1) and (2) pass. `docs/benchmarks/expected_kernel.py`.
- **`GPWalk` before the release** (added 2026-10-08): the walk carries the
  covariances its children need, the field read at each step under the
  expected kernel with the uncertainty the walk has accumulated, so an
  uncertain walker reads a weaker field and slows down (and far from the
  data drifts at the field's bias). The walk is an Euler solve of
  `dw/dt = amp·f(w)` along **one** uncertain field, not fresh noise each
  step: the field's residual is taken fully correlated along a point's
  path, so its standard deviations add (spread `~K·s·sd_f`, where an SDE's
  would be `~sqrt(K)·s·sd_f`), `n_steps` refines the solve rather than
  adding noise, and `step·n_steps·amp` is the reach. The covariance against
  the start positions (where the field's inducing points sit) is updated
  exactly by Stein's lemma with the expected gradient of the expected
  kernel; the covariances between two moved positions, which the children
  need, by the expected Jacobian on both sides. Departing from 0.8's
  numbers is accepted: the node's goal is analytical non-stationary kernel
  learning. The fallback, if the gate refuses the analytic walk, is Q fixed
  field samples walked exactly. Under `"joint"`
  the `precision` parameter (the variance divided by `1 + precision` each
  step) is ignored and on the deprecation list (section 5); the walk's own
  KL was planned to go with it and was kept, measured (below). Realization *s* walks a field draw of its
  own from the simulation stream. Gates: the walk's moments and
  covariances against 10⁴ brute-force trajectories on one expert and on
  several; a `GPWalk` network on the folded section under `"joint"`
  against `"marginal"` on the 200 new holes. **Measured 2026-10-08**: the
  linearized walk matches walks along sampled fields within their noise at
  a reach up to a tenth of the field's range (mean 0.03 sd, variance ratio
  0.96-1.05, covariance 0.05), and falls short beyond it -- variance ratio
  down to 0.53 at amp 4 and 0.14 at amp 8, covariance off by 0.42 and
  0.83. The folded section's walk network trains to amp 3 and scores -10.75
  a new hole against -31.65 under the old rule (VGP -12.17), median -1.67,
  but calibration 1.93. **Settled 2026-10-08** (the user's decision to
  measure the fallback first, then to price the walk): fixed field draws
  walked exactly scored the same (-10.95 and -11.57, calibration 1.95 and
  2.07) at 3-6x the cost, and left chapter 16's confident region over
  unsampled ground, so the linearization was never the problem -- dropped.
  The cause of the region was the field's unexplained variance, which the
  walk now carries per point (chapter 16's geometry and metals'
  calibration back to 0.8.8's); the calibration's was an unpriced
  deformation, which the walk's displacement KL fixes (1.97 -> 1.26, score
  -10.46) where priors on `amp` changed nothing -- so that KL stays, against
  the plan, and only `precision` is deprecated. The walk is accurate to a
  reach of a fifth of the field's range and errs both ways beyond it.
  Record: `docs/expected-kernel.md`, "The walk".
- **Phases**: (1) the expected kernel and its quadrature as a function,
  gate 1 on one expert; (2) `_Moments` through every node under
  `"marginal"`, covariance always `None`, the covering tests to the bit;
  (3) `"joint"` in the network -- the two matrices, per-expert chains, the
  refusals, `GPOptions` -- and gate 1 on several experts and three layers;
  (4) slots and subsets, the subset gate; (5) gates 2 and 3; (6) `GPWalk`
  and its gates; then the changelog, the manual (chapter 5, concatenation
  as the recommended construction), the skill and
  `docs/expected-kernel.md`. The release waits for the user's word. The
  range bound stays at its default unless gate 2 shows it decides the
  score; the covariance path takes the double-`where` square root, whose
  second derivatives are finite at zero distance.

**M — Eight Gaussians and the second moment in `BasicGP`** (agreed
2026-10-08, the two items below; for 0.9.0; **steps 1 and 1b built
2026-10-08**: kernel errors 5.1e-4, 9.8e-6 and 9.1e-6, gate 1 unchanged,
chapter 5's deep model 262 s to 74 s at 100 iterations with the same
scores). Plan of record:
1. **Tables.** `Exponential`, `Matern32` and `Matern52` read an uncertain
   input through 8 fitted Gaussians each (rates and weights stored as
   constants in `network.py`, weights summing to one; refitted by
   `docs/benchmarks/kernel_mixtures.py`), independent of range and
   uncertainty; the rational quadratic keeps its `scale`-adaptive rule and a
   certain input the exact kernel. Gate: kernel error <= 1e-3 each, the
   expected-kernel tests (the no-uncertainty tolerance relaxed to 1e-3),
   positive definite, speed on chapter 5 and chapter 16. **1b**: the walk's
   slopes in closed form per component, replacing a forward-mode pass per
   dimension.
2. **Second moment.** Where the inducing points are certain (a
   `GaussianInput` root, through `Linear`/`SelectInput`), under `"joint"`
   only: `v = 1 - tr(C L) + alphaᵀ L alpha - m²`, `L_ij = E[k(x, z_i)
   k(x, z_j)]` in closed form per pair of components (one for the Gaussian,
   36 for a table), `MultiStructureGP` over pairs of structures,
   `AdditiveGP` within and across dimensions; the explained variance
   `tr(C L)`. Deep networks (uncertain inducing inputs, the research's ~11%)
   stay out -- their own item, later. Gate: against `girard()` to 1e-10
   (Gaussian), against the exact mixture on Walker Lake mean and variance
   within 2% at every input variance for all four kernels; bit-identical
   with no input variance; slots as sets.
3. **The input's spread integrated like noise** (the user's choice over
   realizations at drawn inputs, after
   `docs/benchmarks/uncertain_input_realizations.py`: on a 1-D field at an
   input sd of 0.4 the 5%/95% quantiles miss the exact mixture by 0.224
   with the expected kernel alone, 0.092 integrated like noise, 0.038 at
   drawn inputs whose realizations turn spiky). The realizations stay the
   expected kernel's; what they leave out because the input is uncertain,
   `Delta = tr((R Rᵀ + alpha alphaᵀ) Cov k)`, `Cov k = L - l lᵀ` -- never
   negative -- rides beside them as a per-location latent jitter: a GP node
   returns it beside its four outputs (an attribute on the tuple, as
   `_Moments`), `_predict_raw` hands it to the likelihood, and
   `integrated_backward` and `measurement_samples` take `sims + sqrt(Delta)
   eta` over a few Gauss-Hermite nodes `eta` on top of the noise nodes
   (about 8x the back-transform, at uncertain inputs only); the categorical
   likelihoods the same in their probabilities. Training needs nothing: the
   quadrature over the latent variance already integrates it. Gate:
   `prediction` and the quantiles in data units against the exact mixture
   through a nonlinear warping; with no input variance, bit-identical.
4. **Training gate** on the `GaussianInput` plan's jittered locations:
   held-out coverage and CRPS against the expected kernel alone and against
   `UncertainInputGP`; time and memory at m = 100 and 300 (`L` is `n m²` per
   expert under a gradient).
5. `UncertainInputGP` deprecated with a warning naming `BasicGP`, removed in
   the breaking version; changelog, design record, skill.

**S — Fewer components for the Matérn kernels' expected kernel** (measured
2026-10-08; **built** as step 1 of the item above). The fixed trapezoid (52, 32 and 28 components for the
exponential, Matern32 and Matern52) is accurate to 1e-7, far below any
gate; the best positive mixture of 8 Gaussians, fitted once and stored as
constants (weights summing to one, so the diagonal stays exact), misses the
kernel by 5.5e-4, 1.5e-4 and 9.4e-5 on [0, 3] ranges, which bounds the
expected kernel's error too (the expectation is linear in the mixture) --
3.5 to 6.5 times fewer evaluations, still positive definite by
construction. With it, the walk's slopes in closed form (each component's
derivative in the mean is `-2 w mu / (r² (1 + 2 w v))` times itself) rather
than one forward-mode pass per dimension. **Rejected on the way**:
Paciorek's form with the kernel substituted, `prod (1 + kappa v)^(-1/2)
R(sqrt(sum mu² / (1 + kappa v)))`, its inflation `kappa` fitted per kernel
(23.8, 10.4, 8.1): it is the expectation only for the Gaussian, and misses
it by 0.10-0.13 (exponential), 0.04-0.06 (Matern32) and 0.03-0.04
(Matern52) in one to three dimensions; positive definite for independent
inputs (Paciorek's theorem, by congruence) and not guaranteed with
correlated ones, though no failure turned up on near-singular lattices.

**M — A principled price for the walk** (from the expected kernel,
2026-10-08). The walk's KL under the expected kernel is the marginal rule's
term, `1/2 sum (m_walked - m_start)² / v_walked` over the inducing points:
not a KL (it treats the start as a prior and divides by the posterior's
variance, with no trace or log-determinant), and since the displacement and
its spread both scale with `amp`, in effect a penalty on the field's
signal-to-noise ratio where it moves points. It works -- the folded
section's calibration 1.97 -> 1.26, where MAP priors on `amp` (Gamma at mode
1, exponential) changed nothing -- because it keeps the walk uncertain. In
a proper bound the walk adds no random variable of its own, so no KL; what
the point estimates miss is the uncertainty in the deformation's scale.
The principled version: `amp` a random variable, a log-normal variational
posterior `q(log amp)` against a prior whose base model is no deformation
(an exponential on `amp`, the PC prior), `KL(q || p)` in the bound, and the
walk's moments averaged over `q` -- a few Gauss-Hermite nodes in `log amp`,
each a walk, the mean and covariance by the law of total variance over
them; the realizations each at a node. Gate: on the folded section without
the displacement term, calibration within 1-1.5 and a score at least the
heuristic's (-10.46); chapter 16's rock maps no worse; the cost (one walk
per node) against the term it replaces.

**M — The second moment of a GP at an uncertain input** (Girard's term,
measured 2026-10-08 on Walker Lake, `GaussianInput` root). Under the
expected kernel `BasicGP`'s mean at an uncertain input is within 1-2% of
the exact mixture's (better than `UncertainInputGP`'s 32 Sobol nodes, 3-9%),
but its variance misses the variance of the mean over the input -- the
matched GP's variance is `1 - E[k] (K + D)^-1 E[k]` -- by 18-133% for the
Gaussian and 8-89% for the Matern32 at input variances of 0.01-3 squared
ranges, where `UncertainInputGP` is within 1-3%. So `UncertainInputGP` is
not obsolete for a single node on an uncertain input. The fix that would
retire it: `E[k kᵀ]` (closed for the Gaussian, Girard; through the mixture
for the others) for the variance, `tr((K+D)^-1 Cov k) - alphaᵀ Cov(k)
alpha`; in a deep network the same term is what the research measured at
~11% of the leaf's variance.

**M — Covariance between a node's outputs** (from the expected kernel,
2026-10-08). Mixing operations drop it, as they drop it for the variances
today; the closed form takes a full S×S block per pair of locations
(`det(I + 6Σ/r²)^(-1/2) exp(-3 mᵀ(r²I + 6Σ)⁻¹ m)`) at `[n, m, S, S]` and a
determinant a pair. Worth it only if a tree mixing columns under a GP
measures worse than the same tree unmixed.

**M — `UncertainInputGP`, `RadialTrend` and `GradientConstrainedInput`
under the expected kernel** (from the expected kernel, 2026-10-08). Refused
under `"joint"`: a GP node on a `GradientConstrainedInput`, a random root
whose covariance between locations is not carried (its refresh also hands
on predictions at the directional rows, which no child can read);
`UncertainInputGP`'s Sobol mixture of marginals is what the
expected kernel replaces, so it may simply retire (see the simplification
item in section 5); `RadialTrend` is a nonlinear function of its parent's
mean whose variance is dropped.

**L–XL — Batched experts: memory independent of the number of experts**
(**top priority**, raised 2026-10-05). Train the product of experts on a
few experts per step, down to one, so that device memory does not grow
with the number of experts J. The research is a project of its own,
`OneDrive\Claude\Research\Batched experts\`: a literature review
(`report/literature_review.tex`, section 2), a notation table, and
Proposition 1. Proposition 1 says that sampling s experts per step, scaled
by J/s, gives an unbiased estimate of the bound when the bound splits into
independent blocks, one per expert, with only the hyperparameters shared;
it is checked by exact enumeration in
`verification/expert_subsampling_unbiased.py`. No published method was
found that samples experts per step together with data minibatches; the
closest is ProSpar-GP (Li & Mak, JCGS 2025), which minibatches data only.
That search is not yet complete.

**What geoML does today** (read 2026-10-05; the research assumed it
without reading the code):
- **The bound does not split per expert.** `BasicGP._moments` predicts
  every data point from *every* expert's inducing set and blends them with
  `_GPNode.get_expert_weights`, soft weights normalized over all experts.
  A data point therefore reaches every expert, and Proposition 1 does not
  apply as stated. Only the KL term, `kl_divergence`, is a sum over
  experts.
- **Per step, `refresh` builds every expert:** its covariance, Cholesky
  and inverse, and its smoothed covariance. That is the U_j² and U_j³
  work, the memory to save. Under the default `expert_propagation=
  "consensus"`, a node with children also predicts every expert's
  inducing set from every other, which is O(J²).
- **The stored parameters are O(U_j) per expert** (`alpha_white_i`,
  `delta_i`, `bias_i`), so all of them can stay on the device and only the
  sampled experts' computation need run. Data minibatching exists already
  (`train_svi`).

**The plan of record (the user's, 2026-10-06).**

*Training.*
- **Cache each data point's weight for each expert**, in one sweep over
  the data: the expert weights the model already computes
  (`_GPNode.get_expert_weights`), averaged over the latent outputs where a
  node has several (and over the GP nodes, where a tree has several on one
  root). An N by J table, refreshed once per epoch or less often.
- **An epoch visits the experts in a random order.** For expert j, a batch
  of data points is drawn with probability proportional to their weights
  for j. The batches are random, and they fall where j matters.
- **The batch updates the experts that are active on it.** The points of
  expert j's batch sit mostly in j's ground, so the experts whose cached
  weight is above a small threshold there are j and a few neighbours. Only
  they are computed, and their local parameters (`alpha_white_i`,
  `delta_i`, `bias_i`) are updated with each batch; the rest are truncated
  out of the blend on that batch.
- **One property worth keeping.** At a point the weights sum to one over
  the experts, so if expert j's batch is sized by its total weight, each
  data point is drawn equally often over an epoch, in expectation: the
  epoch is a fair pass over the data. With equal batch sizes instead,
  each drawn point's term is scaled by the inverse of its probability.
  Both to be tried. The data term is scaled to the whole data set as
  `train_svi` scales it; how each batch counts the KL of the experts it
  updates is to be tested too.
- **The global parameters** (ranges, the input transform, the likelihood
  and its warping): **test whether they can be updated at every step**, or
  whether their gradients must be accumulated over the epoch and applied
  once. The accumulated gradient over a fair epoch is the full gradient's
  estimate; a step per batch moves them on a gradient that sees one
  expert's ground at a time.

*Prediction.*
- **The data's cached weights are interpolated to the targets** (grid
  nodes, blocks, points) by a simple method: the nearest data point, or an
  inverse-distance average of the k nearest, in the input's transformed
  space. The result says which experts are active at each target.
- **Targets are grouped by their set of active experts**, and each group
  predicted with only those experts. A small truncation of the weights is
  acceptable; its size is measured against predicting with every expert.
- What any method must keep: a target's set depends on the target alone,
  never on its batch, as the reproducibility contract promises; a rule
  for targets far from every expert, where today every weight sits at the
  floor and the prediction is the plain average of every expert's prior,
  the k nearest say; and a block's set is the union over its sub-blocks.

*The code.*
- **Ideally new methods on `VGPNetwork` and little else**: one
  `predict_*` (the user's words), and a `train_*` beside it, since
  training needs an entry point too. The adaptations at the points of
  contact are kept surgical. The main one: the expert loops in `BasicGP`
  (`refresh`, `_moments`, `kl_divergence`) run over a given subset of the
  experts rather than all of them, set as a context flag the way
  `propagation_rule` is (so no signature changes). A traced function sees
  the subset as a Python tuple, which retraces once per distinct subset;
  if the subsets are few that is acceptable, and if not, the prototype
  measures what a tensor index costs.
- **All experiments outside the package**: a separate git worktree on its
  own branch (`batched-experts`, from `claude`), with the scripts in its
  `experiments/` and nothing merged until the gates pass. The
  verification scripts stay in the research project.

**Kept in reserve** (2026-10-05, not the plan): freezing the other
experts' contributions as stored moments, which gives the exact gradient
for one expert at a time; summing the global parameters' contributions
over a sweep for their full gradient; making `expert_propagation=
"independent"` the default, which leaves each expert's inducing set to
itself at every node (measured 1.6x to 6.3x faster training as J grows
from 5 to 40); and, for prediction, the expert's footprint in the
kernel's space (exact for a compactly supported kernel), neighbouring
experts, coarse to fine through `refine`, accumulating the weighted sums
expert by expert, and a learned multi-label classifier, which is
published (Jalali & Kasneci, NeurIPS 2022 workshop, arXiv 2211.09940).
Also from the literature: selection through a sparse precision matrix
between experts (Jalali, Pawelczyk & Kasneci 2021, arXiv 2102.01496); a
gating network over sparse GP experts fitted by Cluster-Classify-Regress
(Etienam et al., Machine Learning 2024, arXiv 2006.13309); patchwork
kriging (Park & Apley, JMLR 2018); GRBCM (Liu et al., ICML 2018).

**Steps, in order:**
1. **The worktree and a baseline.** A synthetic case with many experts
   and a real one (Tom v6 or Assen): peak device memory, time per epoch
   and held-out scores, trained with every expert at once.
2. **The weight table.** One sweep caching the N by J weights, averaged
   over the latent outputs; how concentrated they are (how many experts
   carry, say, 99% of a point's weight) decides how small the active sets
   and the truncation are.
3. **The subset of experts at the points of contact**, and its gate: with
   every expert in the subset, training and prediction identical to
   today's to the bit.
4. **The training loop**: experts in random order, batches drawn by
   weight, the active experts updated per batch; batch sizes by total
   weight against equal sizes with the inverse-probability scaling; the
   global parameters stepped per batch against accumulated per epoch.
5. **Training gates:** peak memory flat in J; convergence and held-out
   scores against step 1's baseline.
6. **Prediction**: the weights interpolated to the targets, the targets
   grouped by active set; against predicting with every expert, the error
   the truncation costs, measured, and memory per batch against J; then
   blocks and deep trees.
7. The novelty search finished: Spatial Statistics, JABES, Mathematical
   Geosciences, and the full texts of Zhang et al. (2023) and pFedGP.

**Measured 2026-10-06** (prototype on branch `batched-experts`, worktree
`geoML-batched`, not merged; experiments in its `experiments/
batched_experts/`, results in `results/*.jsonl`, figures copied to the
research project). Synthetic field whose ground and data grow with J (400
rows and 150 inducing points per expert, 50-unit tiles), batches of 400,
on the GPU; Tom's rock type (20 126 composites, a fifth of the holes held
out) as the real case.

- **The point of contact** is `latent.expert_subset(experts)`, a context
  flag `BasicGP`'s loops (`refresh`, `_moments`, `simulate`,
  `kl_divergence`, now `expert_kl_terms`) read, with the refresh trace and
  `predict_raw` keyed by it and `_log_lik` split into an unscaled
  `_data_log_lik`. Gate: with every expert in the subset the training log,
  predictions and realizations are the unchanged package's to the bit, on
  shallow and deep trees under both propagation rules -- single-threaded:
  with default threading the unchanged package itself drifts ~1e-14
  between processes on a model this size, which op determinism does not
  remove. The methods: `expert_weights`, `train_by_expert`,
  `predict_by_expert`; tests in `test_batched_experts.py` (9).
- **Weights** (step 2): a point needs 1.9 / 3.1 / 3.9 experts on average
  for 99% of its weight at J = 4 / 16 / 64, up to 21 at 64; 2.1-2.5 on
  Tom, up to 9.
- **Training memory falls as planned** -- peak device memory at J = 4 /
  16 / 64: every expert 73 / 276 / 1101 MB, by expert (per epoch, four
  visits) 94 MB at 16 and 208 at 64 (5.3x less); Tom J = 40, 907 against
  392. The cost moves to the host: each active set is a traced step with
  gradients, never given back, ~80 MB each (5.7 GB at J = 64 against
  2.0).
- **Global parameters must be accumulated over the epoch** (the step-4
  question): stepped per batch on one region at a time they wander, the
  weights spread (99% needs 6.9 experts at J = 16 against 2.6) and the
  fixed sets drop 7% of the weight; held-out rmse 0.299 against 0.137.
- **One visit per expert per epoch converges far too slowly**: an expert
  takes one full step an epoch where `train_svi` gives it J noisy ones (J =
  64: rmse 0.295 at 20 epochs against 0.088). Four visits (batches of
  100) fix most of it -- J = 16: rmse 0.091, CRPS 0.129 against `train_svi`'s
  0.076 / 0.139; J = 64: 0.121 / 0.145 against 0.088 / 0.135 at 783 s
  against 922 -- sixteen are unstable (rmse back up from 0.094 to 0.113).
  At equal time `train_svi` is ahead on rmse at every J measured.
- **Tom**: four visits are better calibrated and call less ore -- J = 10,
  Brier 0.072 against 0.098, balanced accuracy 0.690 against 0.769; J =
  40, 0.101 against 0.120 and 0.750 against 0.770. Active sets are larger
  on drillholes (16.6 of 40 experts).
- **The active sets must be formed once and kept**: re-formed with each
  weight table, a 16-expert run traced 76 steps and grew 6.6 GB.
- **Prediction memory falls ten-fold**: J = 64, 180 MB against 1953 on a
  point grid, 145 against 1885 on blocks; every location keeps at least
  99% of its weight, the prediction moving by 0.020 at most (field sd ~1),
  no Tom block's call changed above 0.015%. **Grouping is the cost**: by
  exact active set 164 groups (a trace each) at J = 16 and 92 s against 3;
  by the home expert's training set, falling back to the union of the
  leading experts' sets, 16 groups at 16, 108 at 64 (149 s against 12),
  and 317-707 on Tom's blocks between drillholes (556 s against 8). The
  first home rule -- the leading expert's set alone -- dropped up to 40% of
  a location's weight where it did not reach.
- **A GP reading only a GP is out of reach of spatial experts**: the
  second GP reads the first's output, where its experts are not local
  (99% of a point's weight needs 12 of 16 experts, against 2.2 at the
  first layer), so by expert it moves the prediction by 0.19 on average.
- **Concatenating the coordinates into the second GP's input makes it
  local again** (the user's proposal; `probe_deep_concat.py`, J = 16, 10
  epochs of `train_svi`, independent propagation): 99% of a point's
  weight needs 3.2 experts at the second layer (max 8) against 12.1 (max
  15) for the GP reading only the GP, 2.7 at the first. The second GP's
  ranges trained to 1.8 on each coordinate and 0.69 on the latent, an
  expert's tile being 3.3 units wide in the transformed space -- the
  locality rests on the coordinate ranges staying under the expert
  spacing, and a longer training that stretches them would bring the
  collapse back, so the gate reads them. Duvenaud et al. (AISTATS 2014)'s
  input-connected networks are the same remedy for the same pathology.
  The prototype needs four changes for it: accept a `Concatenate` of the
  root and `BasicGP` nodes (`_expert_gp_nodes` refuses it), slice the
  concatenated inducing points by the subset (`_parent_points`), form the
  sets from both layers' weights (the layers share one partition, so one
  set serves both), and gate consensus propagation, which under a subset
  blends the first layer's experts at the second layer's inducing points
  from the active ones only.
- **Novelty** (`report/novelty_search_2026-10-06.md` in the research
  project): possibly novel as a combination; the closest are PSVGP §4.2,
  Hoang et al. (AAAI 2017), Yu et al. (IJCNN 2019) and ProSpar-GP, and
  expert-choice routing (Zhou et al. 2022) as an analogue.

**Measured 2026-10-06, second round** -- the eight open items above,
built and measured on the same branch (`train2.py`, `tom2.py`,
`predict2.py`, `probe_consensus.py`, tables by `summarize2.py`; results in
`results/v2.jsonl`, `tom2.jsonl`, `prediction2.jsonl`,
`consensus.jsonl`). `test_batched_experts.py` holds 24 tests.

1. **One trace serves every set** (`latent.network.expert_slots`): the
   active experts are computed in a fixed number of slots -- the largest
   active set -- with their indices a tensor, an empty slot holding a
   padding expert masked out of every blend, an expert's missing inducing
   points padded with the identity's rows and columns. `BasicGP` has a
   batched twin of `refresh`, `_moments`, `simulate` and the KL under it;
   `covariance_matrix` takes leading batch axes (`x[..., :, None, :]`, the
   same on matrices). Training steps a stacked copy of every expert's
   `alpha_white`, `delta` and `bias` with a masked AMSGrad on the slots'
   rows, written back to the parameters before each weight sweep and at the
   end, a cancel included. Gate: against the trace-per-set path the bound
   agrees to 1e-8 and the parameters to 1e-6 on the shallow and the
   concatenated tree -- once the rate and the betas' powers are taken in
   single precision as Keras takes them (in double precision every step
   came out 6.7e-6 longer). J = 16: one trace against 16, 132 s against
   307, host growth 305 MB against 1199, curves identical; J = 64 (epoch
   update): 311 s against 783, host growth 341 MB against 5656, same
   scores; device peak 278 MB against 208, every step padding to 19 slots.
   With a trace no longer per set, the sets are re-formed at every weight
   sweep (the slots only widen; two or three traces a run), which keeps
   the dropped weight under 1%.
2. **More shared steps were not the gap.** A round update (a step after
   each round of the experts) brings the ranges where `train_svi` takes
   them (2.05 against 2.0, J = 16) and scores worse than one step an
   epoch: rmse 0.111 against 0.092, CRPS 0.138 against 0.130; at J = 64
   0.123 / 0.135 against 0.121 / 0.144. The default is `"epoch"`.
3. **The instability of many visits is the learning rates' clocks.** Each
   expert's optimizer decays its rate 0.999 a step of its own, and an
   expert steps on every batch whose set it is in -- with 16 visits about
   94 steps an epoch, so its rate is at 2% by epoch 40, while the shared
   parameters, one step an epoch, have hardly decayed: they keep moving
   and the frozen experts cannot follow. Neither the batch size (25 and
   100 rows both rise from epoch 40) nor stale sets explain it. Decaying
   the experts on `train_svi`'s clock does (16 visits: rmse 0.081 at epoch
   60, still falling, against 0.111) -- but at J = 64, four visits, that
   clock let a seed rise late (0.114 at 40, 0.126 at 60), and putting the
   shared parameters on it too freezes them before they converge, by
   expert needing about three times `train_svi`'s epochs (ranges stop at
   1.14 against 2.0). Rmse / CRPS at epoch 60, four visits:

   | schedule | J = 16 (3 seeds) | J = 64 (3 seeds) |
   |---|---|---|
   | steps counted (the default, `decay="steps"`) | 0.095 +- 0.005 / 0.125 | 0.095 +- 0.004 / 0.126 +- 0.001 |
   | experts on `train_svi`'s clock | 0.092 +- 0.005 / 0.125 | 0.126 / 0.135 (one seed) |
   | both on it (`decay="epochs"`) | 0.096 +- 0.005 / 0.127 | 0.119 +- 0.005 / 0.143 +- 0.001 |

   Four visits counting steps is the default; `decay="epochs"` is kept for
   many visits, where it ends the late rise (16 visits, J = 16: 0.090 /
   0.128, still falling).
4. **Tom's ore calls were the threshold, not the model.** By expert ranks
   ore better and is better calibrated; balanced accuracy at 0.5 rewards
   `train_svi` for overstating ore (mean probability 0.24-0.28 against a
   held-out share of 0.10), which puts more composites over the cut. At a
   cut at the training share of ore by expert wins. Three seeds (the
   schedule moves none of it by more than 0.005), training seconds from the
   runs made alone on the GPU:

   | Tom | AUC | Brier | balanced at 0.5 | at the ore share | mean p | train s |
   |---|---|---|---|---|---|---|
   | J = 10, every expert | 0.869 | 0.098 | 0.768 | 0.695 | 0.240 | 119 |
   | J = 10, by expert | 0.911 | 0.073 | 0.690 | 0.774 | 0.172 | 55 |
   | J = 40, every expert | 0.817 | 0.120 | 0.770 | 0.649 | 0.281 | 506 |
   | J = 40, by expert | 0.854 | 0.101 | 0.742 | 0.680 | 0.242 | 314 |

5. **Prediction in slots needs no grouping rule.** Each location takes its
   own experts (those holding 99% of its weight), and the groups are
   packed into as many slots as training's largest active set -- a
   location needing more keeps its leading ones, `left_out` saying what
   that drops; `slots=` takes a number instead, `pack=False` leaves a
   location's answer depending on the location alone. Memory goes with
   slots times rows a batch:

   | every-expert model | groups | slots | seconds | peak MB | changed max / mean |
   |---|---|---|---|---|---|
   | J = 16 points, every expert | 1 | | 1.2 | 493 | |
   | J = 16 points, slots | 11 | 10 | 1.7 | 231 | 0.020 / 0.0017 |
   | J = 64 points, every expert | 1 | | 11.3 | 1952 | |
   | J = 64 points, slots packed | 38 | 22 | 11.3 | 894 | 0.023 / 0.0020 |
   | J = 64 points, slots unpacked | 1052 | 22 | 34.8 | 148 | 0.029 / 0.0047 |
   | J = 64 points, ten slots | 214 | 10 | 15.2 | 210 | 0.111 / 0.0037 |
   | J = 64 blocks, every expert | 1 | | 5.7 | 2134 | |
   | J = 64 blocks, slots packed | 30 | 22 | 13.8 | 533 | 0.013 / 0.0012 |
   | J = 64 blocks, ten slots | 121 | 10 | 19.5 | 366 | 0.079 / 0.0024 |
   | Tom J = 40 blocks, every expert | 1 | | 10.8 | 1753 | |
   | Tom J = 40 blocks, slots | 88 | 13 | 17.3 | 787 | 0.039, calls 0.02% |

   Ten slots at J = 64 leave 4-5% of the locations short of 99% of their
   weight, the worst at 17-21%. Tom's blocks went from 317-707 groups and
   556 s to 88 and 17 s. A model trained by expert carries larger sets
   (Tom J = 40: 24 slots, 1306 MB against 1909) -- the cap is the lever.
6. **Deep trees through the coordinates work.** `Concatenate` refreshes
   under a subset and under slots, `_expert_gp_nodes` takes a GP reading a
   `Concatenate` of the input and `BasicGP` nodes, and the weight table
   reads every GP node, a node past the first reading what the one expert
   gives below it. Gates: by expert with every expert in the slots is the
   model to 1e-9 under both propagation rules; a GP reading only a GP is
   still taken, and still not local. J = 16, 60 epochs (by expert with the
   experts on `train_svi`'s clock, the default when it ran):

   | concatenated tree | rmse | CRPS | 2nd-layer coordinate range / tile | experts for 99% |
   |---|---|---|---|---|
   | every expert | 0.119 | 0.164 | 2.3 / 1.55 | 3.1 |
   | by expert | 0.085 | 0.128 | 1.43 / 2.35 | 2.1 |

   Every expert peaks at 0.112 by epoch 20-30 and falls back as its noise
   shrinks; the second layer's coordinate ranges grow past an expert's
   tile there (the input transform stretching too), and stay within it by
   expert. **Consensus propagation under a subset is an approximation**
   (`probe_consensus.py`): the second layer's inducing inputs move by
   0.014 on average from the full blend (the first layer's sd 0.37) and by
   0.57 at worst, at the sets' edges; the prediction by 0.023 on average,
   0.34 at worst, rmse 0.135 against 0.125. Deep trees by expert should
   propagate independently.
7. **Coverage.** Wired in: progress and cancel (one event a batch, the
   working copy written back on a cancel), `options.training_tolerance`,
   `cross_validate(method="by_expert", expert_options=)`,
   `refine(by_expert=True)`, and `predict_by_expert(where=)`, which
   resumes from `unpredicted()`. Mesh sets never call the model and needed
   nothing -- the list above was wrong to name them. Still refused:
   directional data, leaves on several roots, GP nodes other than
   `BasicGP` (each needs a slot path of its own). One bug found on the
   way: a trace of one kind left its symbolic state on the nodes and the
   next trace of the other kind collected it; `_graph_state` now takes only
   tensors of the graph being traced.
8. **Replication** (three seeds; training seconds from the runs made alone
   on the GPU):

   | synthetic, held-out | rmse | CRPS | train s |
   |---|---|---|---|
   | J = 16, every expert, 60 epochs | 0.080 +- 0.003 | 0.132 +- 0.006 | 168 |
   | J = 16, by expert, 60 epochs | 0.095 +- 0.005 | 0.125 +- 0.004 | 130 |
   | J = 64, every expert, 20 epochs | 0.088 +- 0.003 | 0.135 +- 0.001 | 922 |
   | J = 64, by expert, 60 epochs | 0.095 +- 0.004 | 0.126 +- 0.001 | 1013-1070 |

   At equal time (~1000 s at J = 64) by expert is behind on rmse and
   ahead on CRPS -- ahead on both until ~650 s (figure
   `round2_heldout_by_time_64.png`) -- with a quarter of the device memory
   (264-333 MB against 1075-1109); on Tom it is ahead on everything but
   balanced accuracy at 0.5.

**Measured 2026-10-06, third round** -- the network's nodes by expert, and
a fixed total of inducing points split among different numbers of experts
on Tom. `test_batched_experts.py` holds 69 tests. The split ran from a
frozen copy of the second round's commit, so the node work could go on
beside it.

- **Every node but five now trains and predicts by expert.** Refused, by
  the user's choice: `AdditiveGP`, `UncertainInputGP`,
  `GradientConstrainedInput` -- and with it directional data, which only
  it reaches, `BasicGP`'s directional prediction being commented out --
  `RadialTrend` and `GaussianMixture`. Taken: `MultiStructureGP`,
  `GaussianInput`, several inputs (a list of leaves, or a `Stack` joining
  trees), and below or above the GP nodes `Linear`, `SelectInput`,
  `GPWalk`, `Bias`, `Scale`, `Add`, `LinearCombination` and a
  `Concatenate` of any of them; `Multiply`, `ProductOfExperts` and
  `Exponentiation` hand no inducing points on, so they sit above the GP
  nodes only, as they always have.
- **How.** Under slots every node that hands inducing points on holds them
  as one tensor flattened over the slots, so the nodes working row by row
  (`Linear`, `SelectInput`, `Bias`, `Scale`) need nothing of their own;
  `Concatenate`, `LinearCombination` and `Add` align their parents' points
  expert by expert (`_aligned_points`), a node computed from the input
  alone holding every expert's and one downstream of a GP node the active
  ones; `GPWalk` walks the active experts' points and its KL is a term per
  expert, shared out as a GP node's is; `MultiStructureGP`'s covariance
  takes leading axes. Several inputs: subsets and slots per input
  (`expert_subset` and `expert_slots` take a mapping), the experts numbered
  across the inputs and visited together, each batch an active set per
  input, its data term divided by the number of inputs (a row's weights sum
  to one per input), the KL shares read off the overlap across inputs;
  prediction carries the weights in each input's own transformed space and
  packs the groups input by input.
- **Gates.** Each of 20 networks -- nine below a GP node, eight above,
  `GaussianInput`, two inputs, a `Stack` of them -- predicts by expert with
  every expert in the slots as the model does (prediction to 1e-7,
  realizations 1e-6), and trains by expert in at most two traces with every
  GP node's experts moving; on the walk, a sum below a GP node and two
  inputs, the slots train as a trace per set does (bound to 1e-8). The 19
  test files covering the nodes touched pass (767 s), the catalogue's
  every-node tests included.
- **One bug.** Two trees may number their nodes alike, and the working copy
  was keyed by name: two inputs collided. It is keyed by the nodes' ids.
- **A fixed total split** (Tom, 20 epochs, seed 1; training seconds and
  peak device MB; by expert four visits, steps counted):

  | total | experts | each | every expert: AUC / Brier / s / MB | by expert: AUC / Brier / s / MB |
  |---|---|---|---|---|
  | 1500 | 5 | 300 | 0.890 / 0.083 / 111 / 339 | 0.926 / 0.066 / 59 / 537 |
  | 1500 | 10 | 150 | 0.868 / 0.098 / 118 / 236 | 0.912 / 0.072 / 49 / 248 |
  | 1500 | 20 | 75 | 0.848 / 0.105 / 181 / 185 | 0.904 / 0.076 / 62 / 123 |
  | 1500 | 40 | 37 | 0.846 / 0.113 / 328 / 156 | 0.884 / 0.083 / 108 / 63 |
  | 6000 | 10 | 600 | 0.850 / 0.102 / 770 / 2220 | 0.889 / 0.091 / 531 / 2065 |
  | 6000 | 20 | 300 | 0.815 / 0.112 / 422 / 1329 | 0.883 / 0.092 / 388 / 1018 |
  | 6000 | 40 | 150 | 0.818 / 0.119 / 446 / 919 | 0.857 / 0.101 / 308 / 448 |

  Fewer, larger experts score better by both methods at both totals, and
  by expert is ahead on AUC and Brier at every split. Every expert's time
  grows with the number of experts at 1500 points (111 s at 5, 328 s at 40
  -- the 2026-08-06 finding again) and is lowest in the middle at 6000;
  by expert's memory falls with the number of experts, but at 5 experts it
  peaks above every expert's (537 MB against 339): a batch holds N / (J x
  visits) rows, 800 here. The larger total scores worse at equal experts
  (10 experts: 0.889 against 0.912 by expert) -- 20 epochs may not be
  enough for 6000 points, or Tom's composites do not support them.

**Measured 2026-10-06, fourth round** -- a partition of the rows, the
user's design: each epoch the experts, in a random order, draw their share
of the rows still unused, N / (J x visits), by their weight and without
replacement, so that every row is read once an epoch and a row a dense
expert's quota leaves behind falls to a later expert; only the batch's own
expert steps on it, its KL counted whole in its own batches, the data term
the batch's plain sum; each batch's active sets come from the rows it drew;
the shared parameters step once an epoch or per batch
(`train_by_expert(sampling="partition")`, `VGPNetwork._partition`). Gates:
every row once an epoch, on one input and on two; at fixed parameters an
epoch adds up to the bound to 1e-9; only the batch's own expert moves; the
slots train as a trace per set does. Seed 1, the earlier runs alongside:

| case | design | epochs | seconds | held-out |
|---|---|---|---|---|
| J = 16 | every expert | 60 | 168 | rmse 0.076, CRPS 0.139 |
| J = 16 | with replacement, 4 visits | 60 | 130 | 0.092, 0.130 |
| J = 16 | partition, 1 visit, shared once an epoch | 200 | 243 | 0.157, 0.150 |
| J = 16 | partition, 1 visit, shared per batch | 200 | 310 | 0.178, 0.164 |
| J = 16 | partition, 4 visits | 60 | 162 | 0.106, 0.136 |
| J = 16 | partition, 16 visits | 60 | 476 | 0.079, 0.127 |
| J = 64 | every expert | 20 | 922 | 0.088, 0.135 |
| J = 64 | with replacement, 4 visits | 60 | 1013 | 0.097, 0.126 |
| J = 64 | partition, 4 visits | 60 | 1349 | 0.112, 0.133 |
| J = 64 | partition, 16 visits | 15 | 793 | 0.129, 0.148 |
| Tom J = 10 | every expert (GPU) | 20 | 118 | AUC 0.868, Brier 0.098 |
| Tom J = 10 | with replacement, 4 visits (CPU) | 20 | 102 | 0.911, 0.073 |
| Tom J = 10 | partition, 4 visits (CPU) | 20 | 102 | 0.841, 0.084 |
| Tom J = 10 | partition, 16 visits (CPU) | 20 | 185 | 0.892, 0.078 |
| Tom J = 40 | every expert | 20 | 446 | 0.818, 0.119 |
| Tom J = 40 | with replacement, 4 visits | 20 | 308 | 0.857, 0.101 |
| Tom J = 40 | partition, 4 visits | 20 | 307 | 0.790, 0.114 |
| Tom J = 40 | partition, 16 visits | 10 | 478 | 0.810, 0.109 |

- **Per epoch the partition learns well, once an expert gets enough
  steps**: at J = 16 with 16 visits it is the best by expert on rmse, and
  still falling -- the instability of many visits with replacement gone,
  since an expert steps on its own batches only, 16 steps an epoch rather
  than about 94. One visit is far too slow, an expert stepping once an
  epoch.
- **Per second it loses everywhere measured.** The leftover rows spread the
  late batches over the field, so their active sets grow -- at most 37
  of 64 experts at 4 visits and 25 at 16, against 19 with replacement,
  Tom J = 40 a mean of 13 to 14 and at most 29 -- and every slot pays for
  the widest; with many visits the batches are small (25 rows) and a
  step's fixed cost dominates. The late batches belong less to their
  expert: in the last epoch their rows' weight for it 0.73 against 0.80
  for the first quarter at J = 16, 0.54 against 0.68 on Tom J = 40.
- **Shared parameters per batch are worse here too** (rmse 0.178 against
  0.157).
- Sampling with replacement stays the default; the partition is an option.
- **Quotas by weight change the batches and nothing else** (measured
  2026-10-07, `quotas="weight"`: each expert draws its share of the weight
  the rows carry rather than an equal share, the remainders to the largest
  fractions; the gates hold for both). Same seed, same epochs as above:

  | case | visits | equal: seconds, score | weight: seconds, score | rows a batch | widest set, equal / weight | late own weight, equal / weight |
  |---|---|---|---|---|---|---|
  | J = 16 | 4 | 162, 0.106 / 0.136 | 157, 0.106 / 0.136 | 92-108 | 14 / 13 | 0.73 / 0.73 |
  | J = 16 | 16 | 476, 0.079 / 0.127 | 473, 0.079 / 0.127 | 23-27 | 12 / 11 | 0.78 / 0.83 |
  | J = 64 | 4 | 1349, 0.112 / 0.133 | 1398, 0.112 / 0.133 | 88-111 | 37 / 36 | 0.66 / 0.69 |
  | J = 64 | 16 | 793, 0.129 / 0.148 | 735, 0.129 / 0.148 | 22-28 | 25 / 23 | 0.75 / 0.80 |
  | Tom J = 10, CPU | 4 | 102, 0.841 / 0.084 | 101, 0.841 / 0.084 | 350-453 | 10 / 10 | 0.54 / 0.56 |
  | Tom J = 10, CPU | 16 | 185, 0.892 / 0.078 | 203, 0.892 / 0.078 | 87-111 | 10 / 10 | 0.67 / 0.69 |
  | Tom J = 40 | 4 | 307, 0.790 / 0.114 | 341, 0.790 / 0.113 | 69-137 | 29 / 34 | 0.54 / 0.59 |
  | Tom J = 40 | 16 | 478, 0.810 / 0.109 | 517, 0.807 / 0.109 | 17-35 | 29 / 32 | 0.59 / 0.66 |

  Scores are rmse / CRPS on the synthetic field and AUC / Brier on Tom.
  On the synthetic field the experts carry nearly equal weight, so the
  quotas hardly move; on Tom they span a factor of two. The late batches
  belong more to their own expert, by up to 0.07, but the widest set, which
  the cost follows, does not narrow -- on Tom it widens -- and every score
  is the equal quotas' to the third decimal. Time moves by -7% to +11%,
  the varying batch sizes taking two more traces. The leftover rows are
  not a shortfall in the quotas: an expert drawing by weight takes rows
  from its tails and leaves part of its core to its neighbours, whatever
  its quota, and the last batches take what remains wherever it lies.
- **Rows choosing their expert** (measured 2026-10-07,
  `sampling="assignment"`): every row draws one expert from its own
  weights, so a row lands in an expert's batch with exactly the probability
  sampling with replacement gives it, and every row is read once; an
  expert's rows are split into round(rows / target) batches, at least one,
  the target N / (J x visits), so a crowded expert steps more often (3 to 6
  times an epoch at 4 visits, 10 to 21 at 16 on Tom J = 40); its KL is
  shared among its batches. The count of batches changes from epoch to
  epoch, so the share of the rest's KL and of the priors a batch carries
  is a step argument now, divided by as `/ per_epoch` was, the other paths
  unchanged to the bit. Gates: every row once, the batches within 1.5 x
  the target, each expert's KL shares adding to one, a row landing with its
  weight over 4000 draws, the epoch adding up to the bound, the own expert
  alone moving, slots as by set.

  | case | visits | assignment: s, score | partition, equal: s, score | widest set, assignment / partition | own weight, first / last quarter |
  |---|---|---|---|---|---|
  | J = 16 | 4 | 148, 0.106 / 0.136 | 162, 0.106 / 0.136 | 11 / 14 | 0.82 / 0.81 |
  | J = 16 | 16 | 430, 0.080 / 0.127 | 476, 0.079 / 0.127 | 10 / 12 | 0.85 / 0.85 |
  | J = 64 | 4 | 1031, 0.111 / 0.132 | 1349, 0.112 / 0.133 | 24 / 37 | 0.78 / 0.78 |
  | J = 64 | 16 | 704, 0.129 / 0.148 | 793, 0.129 / 0.148 | 23 / 25 | 0.84 / 0.84 |
  | Tom J = 10, CPU | 4 | 86, 0.843 / 0.083 | 102, 0.841 / 0.084 | 9 / 10 | 0.60 / 0.64 |
  | Tom J = 10, CPU | 16 | 205, 0.892 / 0.078 | 185, 0.892 / 0.078 | 10 / 10 | 0.69 / 0.70 |
  | Tom J = 40 | 4 | 295, 0.791 / 0.113 | 307, 0.790 / 0.114 | 28 / 29 | 0.68 / 0.68 |
  | Tom J = 40 | 16 | 483, 0.807 / 0.108 | 478, 0.810 / 0.109 | 29 / 29 | 0.69 / 0.68 |

  It does what it was built for: no leftovers, the last batches as much
  their expert's as the first, the widest set narrower on the synthetic
  field (24 against 37 at J = 64), and the fastest of the three splits
  there (1031 s against 1349). On Tom the widest set stays at 28-29, as
  with replacement (25): the field's own clusters straddle experts.
  **The scores do not move.** Three ways of splitting the rows, one score
  to the third decimal in every case, and still short of sampling with
  replacement at the same epochs (Tom J = 10 at 4 visits: 0.843 against
  0.911; J = 16: 0.106 against 0.092). Since the assignment's batches hold
  the rows replacement's draws do, what is left between them is who steps:
  with replacement every active expert steps on every batch, about 51
  steps an expert an epoch on Tom J = 40 at 4 visits (160 batches, 12.8
  active each), against 3 to 6 here. The rule that only the batch's own
  expert steps, not the row split, is what holds the partition back; the
  next test is the assignment with every active expert stepping on the
  batch's plain sum, each expert's KL shared by the weight it carries in
  each batch so an epoch still counts it once.
- **Every active expert stepping on the assigned rows** (measured
  2026-10-07, `sampling="assignment", stepping="active"`): the rows as
  above, every expert in a batch's active sets stepping on its plain sum,
  each expert's KL shared among the epoch's batches it is active on by the
  weight it carries in each, so an epoch still adds up to the bound (gate
  as before, and every active expert moving on the first batch). Seed 1,
  scores at the last epoch:

  | case | visits | active stepping: s, score | own only: s, score | with replacement, 4 visits: s, score |
  |---|---|---|---|---|
  | J = 16 | 4 | 144, 0.090 / 0.129 | 148, 0.106 / 0.136 | 130, 0.092 / 0.130 |
  | J = 16 | 16 | 446, 0.101 / 0.131 | 430, 0.080 / 0.127 | 422 at 16 visits, 0.111 / 0.134 |
  | J = 64 | 4 | 1031, 0.094 / 0.126 | 1031, 0.111 / 0.132 | 1013, 0.097 / 0.126 |
  | J = 64 | 16 | 708, 0.105 / 0.137 | 704, 0.129 / 0.148 | -- |
  | Tom J = 10, CPU | 4 | 85, 0.907 / 0.073 | 86, 0.843 / 0.083 | 102, 0.911 / 0.073 |
  | Tom J = 10, CPU | 16 | 188, 0.909 / 0.073 | 205, 0.892 / 0.078 | -- |
  | Tom J = 40 | 4 | 299, 0.856 / 0.100 | 295, 0.791 / 0.113 | 308, 0.857 / 0.101 |
  | Tom J = 40 | 16 | 479, 0.857 / 0.099 | 483, 0.807 / 0.108 | -- |

  At 4 visits it matches sampling with replacement in all four cases --
  ahead on the synthetic field, level on Tom -- in about the same time,
  confirming that the own-expert rule was what held the partition back.
  An expert now steps 12 to 87 times an epoch at 4 visits on Tom J = 40
  and J = 64, crowded ones the most. At 16 visits the many-visits drift of
  replacement returns (J = 16: 0.090 at epoch 40, 0.101 at 60; replacement
  0.111), which is item 1 below. The widest set is that of the assignment
  (23 at J = 64 against 19-21 with replacement), and peak device memory a
  little above replacement's (411 MB against 333 at J = 64, 460 against
  448 on Tom J = 40). What it adds over replacement: every row read once
  an epoch, an epoch's batches adding up to the bound, no inverse-
  probability scale, crowded experts taking more batches by construction.
  One seed: whether it should replace sampling with replacement as the
  default needs the replication the second round gave replacement.
- **Replicated over three seeds** (2026-10-07; seeds 1-3, 4 visits, both
  designs rerun on the GPU under the current code, scores of the
  prediction by expert):

  | case | with replacement: mean, s, MB | assignment, every active expert: mean, s, MB | assignment minus replacement, per seed |
  |---|---|---|---|
  | J = 16 (rmse / CRPS) | 0.094 +- 0.005 / 0.125 +- 0.004, 133, 118 | 0.093 +- 0.005 / 0.125 +- 0.005, 145, 135 | rmse -0.0022, -0.0014, -0.0021 |
  | J = 64 | 0.095 +- 0.004 / 0.126 +- 0.001, 927, 325 | 0.092 +- 0.004 / 0.126 +- 0.001, 987, 381 | rmse -0.0029, -0.0021, -0.0027 |
  | Tom J = 40 (AUC / Brier) | 0.854 +- 0.003 / 0.101 +- 0.001, 308, 447 | 0.856 +- 0.001 / 0.100 +- 0.001, 310, 463 | AUC -0.0014, +0.0007, +0.0056 |

  The assignment is ahead on rmse in all six synthetic pairs, by about
  0.002, and level or ahead on CRPS in all six; on Tom level or ahead on
  Brier in all three and on AUC in two. It costs 6-9% more time on the
  synthetic field and none on Tom, and 4-17% more device memory, its
  widest set being the larger. (One replacement run on Tom, seed 2, shared
  the GPU with another training job and took 496 s; its score is
  unaffected and the time above leaves it out.) The seed-1 reruns check
  the default path: sampling with replacement gives the second round's
  numbers under the current code -- 0.09191149854906935 against
  0.09191149854907349 at J = 16, 0.09664179726490743 against
  0.09664179726491132 at J = 64, Tom identical -- the difference being
  the known drift between processes under default threading. **The
  assignment with every active expert stepping is `train_by_expert`'s
  default since 2026-10-07** (`sampling="assignment", stepping="active"`);
  sampling with replacement and the partition remain options.

Open, in the order they matter:

1. **A learning-rate schedule tied to progress**, for the experts and the
   shared parameters alike (item 3): counted per step, many visits freeze
   the experts while the shared parameters move on; on `train_svi`'s
   clock, the shared parameters freeze before they converge.
2. **Slots sized to memory, not to the largest set**: prediction memory
   goes with slots times rows a batch, so choosing the batch from a memory
   budget would let the packed groups keep every location's 99%.
3. **The rmse gap on the synthetic field** (0.015 at J = 16, 0.007 at 64)
   while CRPS is better by expert: unexplained.
4. **Batches sized to the experts**: training by expert draws N / (J x
   visits) rows a batch, so few experts make large batches and a peak
   above `train_svi`'s (5 experts on Tom: 537 MB against 339). Batch
   sizes by each expert's total weight, against equal sizes with the
   inverse-probability scaling, were in the plan of record and never run
   with replacement; under the partition they were run and changed
   nothing (fourth round).
5. **The five refused nodes**: `AdditiveGP`, `UncertainInputGP`,
   `GradientConstrainedInput` (and directional data), `RadialTrend`,
   `GaussianMixture`.

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

*Started 2026-09-25 (0.8.0): the free gate is `latent.GaussianMixture`,
internal in the catalogue.* Settled in design: a soft blend per realization,
`y = Σ softmax(w)_k c_k`, no temperature and no hard mode; K weights; the
moments from fixed Sobol points over the weights, taken independent of the
components; a leaf that is not Gaussian trains its continuous likelihood on
the realizations (`gaussian` on every node), a likelihood with no samples
path moment-matching with a warning. Measured
(`docs/benchmarks/gaussian_mixture_gates.py`, 1-D sinusoids):
- *Split at a boundary*: passes, RMSE 0.161 / CRPS 0.043 against one GP's
  0.192 / 0.078, switch at 4.79 against the boundary at 5 — but only with
  components smoother than the jump. On the GP's own root one component
  fits everything (share 1.00). A `BasicGP` has unit prior variance, so
  the node carries its own `amplitude` on the weights (decided 2026-09-25;
  through `Scale` instead, 0.153 / 0.046). It trains to its ceiling of
  100: the switch wants to be a step. Drawn
  (`docs/benchmarks/figures/gaussian_mixture_gate1.png`), the components
  are not "the left regime and the right": one smooth curve follows the
  data left of the boundary and again past 7.5, the other only bridges
  [5, 7].
- *Against the construction in use* (chapter 5's deep GP: the input and
  an inner two-column GP concatenated under an outer GP, root range 2):
  RMSE 0.151 / CRPS 0.049 with the default kernels, 0.162 / 0.052 with the
  chapter's spherical outer kernel. It follows the jump as closely as the
  mixture, a better RMSE than its 0.161, but overshoots it (1.3 against
  1.0 just before, ringing after) inside a band too narrow to cover the
  miss, so the mixture's CRPS (0.043) is the better. Neither is bimodal on
  the crossing curves. The deep GP's bound was still oscillating at 1000
  iterations (step spread 0.34 against the mixture's 0.002). So on this
  gate the mixture's gain over the construction in use is calibration, not
  accuracy — too thin to make the node public on its own.
- *Curves crossing throughout*: fails. The mixture blends the two into
  their mean, no realization near either; the expected log-likelihood
  rewards agreeing with each sample, and the weights never learn to switch
  sample by sample.
- *The training draws are the same every iteration* (`seed=options.seed`
  on every step): the Monte Carlo bound is a fixed-sample average, biased
  rather than noisy (item "Fresh training draws" below).

Open: the real-deposit gate against hard domains (likely Jura) before the
node is public; Monte Carlo paths for the categorical likelihoods, so a rock-type
leaf that is not Gaussian can train on its realizations; and the gate read
from a rock-type leaf, where the indicator rule (the largest latent) and
the mixture's softmax read one latent through two different links.

**M — A mixture of likelihoods: overlapping mixtures of GPs** (agreed
2026-09-25, redesigned 2026-10-01 with a warping per component; **done
2026-10-01, public from 0.8.6**: every gate below passed, phase 2 included).
Lázaro-Gredilla, Van Vaerenbergh & Lawrence (2012), *Overlapping Mixtures of
Gaussian Processes for the data association problem*, Pattern Recognition
45(4). M global latent functions; each sample comes from exactly one,
chosen by its value rather than its place, so the components cross and
overlap anywhere and the prediction is a mixture -- multimodal where they
separate. This is what `GaussianMixture` cannot do: it blends *values* in
the latent, so a realization is bimodal only where its gate is sharp, which
is why it failed the crossing gate; this mixes *densities* in the
likelihood, sample by sample.

The design, settled:
- **`likelihood.LikelihoodMixture(components)`**, beside today's `Mixture`
  (noise scales around one latent value, one warping), which keeps its name
  and path; the docstrings say how the two differ. Experimental, and
  internal in the catalogue until the gates below pass.
- **A component is any continuous likelihood instance**, with its own
  family, noise and warping -- `[Gaussian(BoxCox(...)), Gaussian(ZScore(1))]`
  -- today's `Mixture` included. A shared warping is the same object handed
  to every component (persistence keeps it shared, and its Jacobian cancels
  out of the sum). Every component takes the variable's width P, any P; a
  mismatch is refused at construction.
- **Component k reads its own slice of the leaf**, in list order; the leaf
  is the sum of the components' latent widths (a `PCA` link can make one
  narrower than P). A user wanting independent components builds the leaf
  with `Stack`.
- **The bound**, per row: `log Σ_k π_k exp(E_q[log p_k(y | f_k)])`, each
  component's density in data space with its own warping's Jacobian -- the
  term a shared warping lets factor out. Quadrature per component where its
  warping is elementwise, Monte Carlo on realizations otherwise.
- **Shares**: fixed `π`, trained (phase 1). Spatially varying shares read
  from further latent columns through a softmax are phase 2, measured on
  Tom East with the share columns on the same tree as a rock-type
  likelihood on `Code_Simple`.
- **Realizations divided among the components in fixed numbers**, in
  proportion to `π` by largest remainder and interleaved, so any first k
  hold the components in about the right proportions; the labels stored as
  a node fact on the variable (one list, the same everywhere), a resumed
  prediction refused if they would differ. Each realization goes through
  its own component's noise and warping, so the plain ensemble *is* the
  mixture and quantiles, cut-off shares, block support, measurement
  samples and cross-validation need nothing new.
- **Columns**: the mixture's usual ones; `responsibilities/<k>` -- the
  posterior where measured, the prior share elsewhere -- written by the
  outer mixture only (an inner noise `Mixture`'s are not, in phase 1); and
  `component_prediction/<k>`, each component's prediction in data units.
  The latent moments per component are what `predict_node` answers.
- **No reaching for `lik.warping`**: the model, the diagram and the plots
  ask the likelihood; warped metadata and the transformed pairs are skipped
  for a mixture, which has no one warped space.
- **Symmetry broken by initialization**: the data clustered into M groups
  (seeded from the package generator), each component's warping
  initialized on its group, `π` started at the group sizes -- so latent
  zero means a different value in each component. Clustering on values
  splits crossing curves at the crossing; the crossing gate says whether
  training mends it, random responsibilities (the paper's) in reserve.

Gates, before it is made public (`docs/benchmarks/gaussian_mixture_gates.py`,
beside `GaussianMixture`'s, and a Tom East script reading a local copy the
repository never holds):

| gate | passes when |
|---|---|
| crossing sinusoids | realizations near both curves; ≥ 90% of samples' largest responsibility on their true curve |
| split sinusoids | held-out rmse and CRPS within 5% of one GP |
| population skew | separate warpings beat one shared warping on held-out CRPS; ≥ 80% largest responsibility on the true population |
| Tom East (Ag, Pb, Zn as a vector; folds by hole) | held-out CRPS no worse than `BoxCox → RobustPCA → ZScore → SinhArcsinh → ZScore` in one likelihood; out of fold, the expected shares agreeing with `Code_Simple` at least as often as the rock type's own prediction does (changed 2026-10-01 from "well above chance") |

A near miss is reported, not met by moving the threshold.

**Phase 1 measured, 2026-10-01** (the class and the model's hooks built;
the stored columns, labels fact, catalogue and diagram not yet):

| gate | result | verdict |
|---|---|---|
| crossing | every location bimodal where the curves are apart, 49% / 49% of the realizations near each; the largest responsibility on the sample's own curve 63% under one global naming, **97.5% (100% apart)** asked locally -- a component passes from one curve to the other at a crossing, where nothing tells them apart | passes, on the local reading |
| split | rmse 0.369, CRPS 0.114 against one GP's 0.192, 0.078; shares 0.08 / 0.92 | **fails**: fixed shares put the second component into every location -- what spatially varying shares (phase 2) are for |
| population skew | separate warpings CRPS 0.561, agreement 96%; one shared warping 0.575, 93% | passes, narrowly (2.4% on CRPS) |
| Tom East | CRPS Ag / Pb / Zn 49.4 / 3.67 / 3.10, goodness 0.87 / 0.82 / 0.81; one Gaussian through the recommended chain 68.3 / 3.96 / 4.17 (its rmse exploding, 5568 on Ag), through the components' own plain `BoxCox -> ZScore` 52.4 / 4.20 / 3.90 at goodness 0.73 / 0.60 / 0.67; shares 0.46 / 0.54, the largest responsibility agreeing with `Code_Simple` at 94% (in sample; chance 52.5%) | passes |

The first run of these gates started the components from each column's
normal scores, and the shared-warping arm came back at 54% and at 91% from
one seed: on one column normal scores are symmetric whatever the data, so
two groups always met at the median, the splits either side of it tied,
and the clustering broke the tie differently from run to run. The start is
a Yeo-Johnson power transform per column now, and every number above is
from that start, repeated exactly three times. Decided 2026-10-01: the
crossing gate is read locally (accepted), and the class stays internal
until phase 2 passes the split gate.

**Phase 2, agreed 2026-10-01: shares that change from place to place.**
- `LikelihoodMixture(components, shares="fixed" | "latent")`; under
  `"latent"` the leaf carries K share columns after the populations', their
  softmax the shares, scaled by a trained `amplitude` (a variance, as
  `GaussianMixture`'s) -- none at first, added when the split gate asked
  for it: a share GP of prior variance one could not pass from one
  population to the next sharply enough, and the realizations divided at the
  boundary -- and moved by a trained `bias` per population, what the shares
  return to away from the data (added 2026-10-01 at the user's request: Tom
  East's ore is not half the ground), started by `initialize` on the
  clustering's group sizes. A share
  GP node may feed a rock-type likelihood before it is concatenated into the
  mixture's leaf.
- **Realizations**: realization s keeps its one interleaved point `u_s`,
  and at each location takes the population whose interval of *that
  realization's own* cumulative shares holds it. It changes population
  where its share field crosses its point -- a smooth boundary, different
  in every realization, carrying the boundary's uncertainty -- and at each
  location the fraction of realizations in a population is the expected
  share, so the plain ensemble stays the mixture and no reader changes.
  Fixed shares are the special case (phase 1 unchanged). **This reverses
  the 2026-09-25 rule** (a whole realization in one population, the
  realizations split into equal groups, every reader weighting them by the
  local shares): under shares that vary, that rule turned every reader of
  realizations into a weighted one, and a tonnage read off one realization
  could not hold one orebody. Rejected also: the point read against the
  *mean* shares, which nests every realization's boundary inside one field.
- **Training**: per row, the average over share realizations of
  `log Σ_k softmax_k(g) exp(term_k)` -- on realizations, the path
  non-Gaussian leaves already take, the fixed training draws a known bias
  ("Fresh training draws" below).
- **Storage**: each realization's population at each location, small
  integers under `<variable>/population/<i>`, on the variable (a
  vector's mixture is over the row), absent where no mixture models the
  variable; coarsening leaves it missing. Built on a refactor first: the
  realization axis was handled by the name `simulations` in about eight
  places (carrying, coarsening, subsetting, Zarr both ways, the export, the
  path, the realization walk), so a declared list of realization stores
  replaces the name, held by a test before the population store exists.
- **Gates**: the split gate (rmse and CRPS within 5% of one GP); crossing
  and skew rerun under latent shares; Tom East with folds by hole, three
  arms -- fixed shares, latent shares from the metals alone, latent shares
  tied to `Code_Simple` -- scored by out-of-fold CRPS per metal and by the
  expected shares' out-of-fold agreement with `Code_Simple`. Internal until
  every gate passes; figures with the results.

**Phase 2 measured, 2026-10-01.** Tom East's folds are built against the
assayed intervals themselves (160 / 160 / 160 / 159 / 159). Built against
every logged interval, most of them in holes with no assay, they mimicked
distances so long that one fold held out 407 of the 798 and the rock type's
own out-of-fold prediction agreed with the log at 0.609.

| gate | no amplitude | with amplitude | populations and shares on separate GPs | verdict |
|---|---|---|---|---|
| split | rmse 0.221, CRPS 0.046 (one GP 0.192, 0.078) | rmse 0.166, CRPS 0.034, amplitude 26 | **rmse 0.149, CRPS 0.032**, amplitude 27 | passes |
| crossing | local agreement 97.5% (100% apart) | the same, amplitude 0.035 | -- | unchanged |
| skew | CRPS 0.567, agreement 96% | CRPS 0.558, 96%, amplitude 0.033 | -- | unchanged |

The amplitude trains large where the shares must jump and near zero where
they should be flat, which is what lets one construction serve both. On the
split, the populations and the shares are two `BasicGP`s of two outputs
each on the gate's grid of 45 inducing points, so each trains its own range.

Tom East, out of fold over the new folds. Every arm reads one root: 500
k-means centroids of the assayed intervals divided into five experts with
10% overlap (110 points each), through `Anisotropy3D(100, 0.75, 0.5, 345,
15, 70)`; the populations (six outputs) and the shares (two) on separate
`BasicGP`s. Cross-validation drops the held-out data only -- a fold model
rebuilt from the save holds the same 550 points to the bit, since only the
`data` argument is swapped.

| arm | CRPS Ag / Pb / Zn | goodness | expected shares agreeing with `Code_Simple` |
|---|---|---|---|
| one Gaussian, recommended chain | 84.6 / 4.25 / 3.48 (rmse on Ag 4e5) | 0.83 / 0.56 / 0.69 | -- |
| one Gaussian, `BoxCox -> ZScore` | **46.0** / 3.77 / 3.45 | 0.71 / 0.64 / 0.55 | -- |
| fixed shares | 46.7 / 3.32 / 2.89 | 0.78 / 0.81 / 0.79 | 0.525 (constant; chance 0.525) |
| latent, metals alone | 47.3 / 3.44 / 2.85, amplitude 12.5 | 0.63 / 0.66 / 0.72 | 0.595 |
| latent, tied to `Code_Simple` | 47.1 / **3.31 / 2.81**, amplitude 6.8 | 0.73 / 0.77 / 0.79 | **0.609**; the rock type's own 0.604 |

Every mixture beats the recommended chain on every metal, so the CRPS half
of the gate passes; the plain `BoxCox -> ZScore` likelihood is the best on
Ag, by 1.5% over the best mixture, and the worst on Pb and Zn. The
agreement half was "well above chance", and **decided 2026-10-01: the bar
is the rock type's own out-of-fold prediction** -- the logged rock type is
barely predictable between holes here, and a share field cannot know more
about it than a likelihood trained on it. Against it the shares tied to the
rock type pass, **0.609 against 0.604**; they follow that prediction at
0.742. In sample the shares agree at 0.94 in every arm. **Every phase 2
gate passes.**

**With a bias per share** (same configuration; the baselines and fixed
shares unchanged):

| arm | CRPS Ag / Pb / Zn | goodness | agreement | shares away from the data |
|---|---|---|---|---|
| latent, metals alone | **45.6 / 3.28 / 2.71**, amplitude 11.6 | 0.66 / 0.68 / 0.75 | **0.713** | 0.978 / 0.022 |
| latent, tied to `Code_Simple` | 46.8 / 3.29 / 2.81, amplitude 7.0 | 0.73 / 0.77 / 0.79 | 0.619; the rock type's own 0.605 | 0.537 / 0.463 |

and the split gate rmse 0.146, CRPS 0.031, its bias at 0.96 / 0.04, the
crossing and skew cases unchanged with their biases near equal shares. The
shares from the metals alone now return to the lower population away from
the data -- waste as the ground's background -- and are the best arm on every
metal, Ag included, and the best predictor of the logged rock type out of
fold, above the rock-type likelihood's own 0.605. Tied to the rock type the
bias barely moves: the share columns are the rock-type likelihood's latent,
and `CategoricalGaussianIndicator` has no bias of its own, so away from the
data it says even odds and pulls the shares there. Goodness is lower for
the metals-alone arm (0.66 to 0.75 against fixed shares' 0.78 to 0.81):
its intervals are narrower than the held-out data support.

**With a bias on the rock type too** (`CategoricalGaussianIndicator(2,
bias=True)`, added 2026-10-01 at the user's request, optional because a
save stores its parameters by position): CRPS 46.0 / 3.25 / 2.79, the
shares' agreement 0.607 against the rock type's own 0.594, the shares
returning to 0.677 / 0.323 and following the rock type at 0.932. The rock
type's own bias trains small, +0.12 / -0.13: the likelihood sees only the
798 assayed intervals, where Tom East is 52.5%, while across all 4585
logged intervals it is 11.0% and across the 3787 never assayed 2.2% --
the assays were taken where the ore is.

**The rock type trained on every logged interval** (4585, the metals
missing where unassayed; the 23 holes with no assay in the fold of the
nearest assayed hole, so the assayed folds are unchanged; the inducing
points still the assayed intervals'): the rock type's bias trains to
+1.17 / -1.17 and the shares return to 0.999 / 0.001 -- the waste
background. CRPS 45.8 / 3.30 / 2.83, goodness 0.69 / 0.71 / 0.74; the
shares' agreement on the assayed intervals 0.637 against the rock type's
own 0.551, which fell from 0.594: read on the assayed intervals, half of
them ore, a model that has learned the ground is mostly waste calls more
of them waste. The bar was set on a sample the assays chose, and is
fair only between arms that saw the same rock data. Training took 638 s
against 159.

Scored instead on every logged interval, out of fold, by balanced accuracy
(the mean of the two categories' recall -- calling everything waste is
right 89% of the time there, and scores 0.5):

| | accuracy | recall Waste | recall Tom East | balanced |
|---|---|---|---|---|
| the rock type's own call, all logged | 0.893 | 0.976 | 0.219 | 0.598 |
| the expected shares, all logged | 0.874 | 0.913 | 0.555 | **0.734** |
| the rock type's own call, assayed | 0.551 | 0.897 | 0.239 | 0.568 |
| the expected shares, assayed | 0.637 | 0.712 | 0.568 | **0.640** |

The shares find two and a half times the ore the rock type does between
holes, for a few points of waste recall: the metals at neighbouring holes
tell them where the ore runs, which the logged rock type alone cannot.
The rock type's own call is barely better than calling everything waste.
Mapped (`likelihood_mixture_figures.py latent`), the shares tied to it now
draw one continuous lens dipping steeply to the northwest inside a waste
background, where they drew separate pockets at even odds between holes.

The first run under these folds, on 150 k-means points in one expert
through `Isotropic(80)` with populations and shares on one GP, gave fixed
shares 44.1 / 3.26 / 2.81, latent 49.3 / 3.46 / 3.06 at 0.564 agreement,
tied to the rock type 46.4 / 3.36 / 2.94 at 0.570 against the rock type's
own 0.603 -- a near miss, which the anisotropy and the separate share GP
closed; the plain chain was 58.6 / 5.58 / 8.46 there, so the configuration
helped the single likelihood most.

**L — A likelihood subsystem** (raised 2026-10-01, deferred). Likelihoods
composed as warpings are chained: a small protocol every component
implements (the latent columns it reads, its log density in data space at
quadrature nodes or realizations, its back-transform with the noise
integrated out, a measurement draw, `initialize`) and combinators over it --
the mixture above, today's `Mixture` as a configuration of it, and
**physics-driven likelihoods** whose mean is a known function of several
latent columns (a forward model), trained on realizations through the path
non-Gaussian leaves use. To settle when taken up: the protocol, the naming
of the family (`Mixture` against `LikelihoodMixture`), and a first forward
model.

**S — Fresh training draws** (found 2026-09-25, kept on the list by the
user). Every training step passes `seed=options.seed`, so a likelihood that
trains by Monte Carlo — a warping that mixes, and since 0.8.0 any leaf that
is not Gaussian — optimizes one fixed set of `training_samples`
realizations: a sample-average bound, biased rather than noisy, which the
model can partly fit. On gate 1 of `GaussianMixture`, 20 draws reach a
bound of 36.2 and 50 draws 26.7, the scores against the truth better at 50
(RMSE 0.141 / CRPS 0.037 against 0.161 / 0.043). Fix: a seed keyed by the
iteration (the phase's count), so a run split into chunks still replays
one call bit for bit. Gate: the same two numbers, and the chunked-training
test in `test_early_stopping.py` still exact.

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

**S — Directional data ignored under a variable named as a string**
(found by reading, 2026-09-15, not run). `VGPNetwork.__init__` pairs the
directional measurements with the variables by iterating the raw
`variables` argument, so `variables="Rock"` walks the letters, looks up
`"R"`, finds nothing and trains on no directional measurement at all. A
list or the mapping spelling is unaffected. Iterate the normalized names.

**S — `ProjectionTo1D` projects onto positive directions only**
(found 2026-09-25, reading why the catalogue should not offer it). Its
`directions` is a `PositiveParameter` in [0.001, 1], so a projection can
point into the positive orthant and nowhere else -- in 2-D, north-east but
never north-west. A signed unit vector (a `UnitColumnNormParameter`, as the
dynamic anisotropies hold their weights) would let it point anywhere; the
catalogue marks it `experimental` until then, and `RandomProjections` with
it, nothing in it training.

**S — A GP node that fails to build is left among its parent's children**
(found by reading, 2026-09-15). `_FunctionalLatentVariable.__init__`
appends the node to `parent.children` before `_GPNode.__init__` can raise
`BrokenPropagationError`, so a caller that catches the error and goes on
holds a parent with a child that does not exist. Register the child last.

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

**S–M — Categorical scores in `cross_validate`** (suggested 2026-09-11).
The driver's score table holds continuous variables only: a categorical
one is predicted into the out-of-fold container like the rest and then
given no rows, its docstring telling the reader to subset the container by
fold and call the variable's own `compute_metrics`. That table reads the
labels and the probabilities the container already holds -- no measurement
samples, so none of the streaming the continuous scores need -- and since
2026-09-11 carries kappa, precision and recall, the quantity and allocation
split and the Brier and log scores beside balanced accuracy, Jaccard and
Matthews. Wanted: those rows per fold and pooled, from the driver. To
settle: the shape, the continuous table being one row per variable,
component and fold and the categorical one a score by category, and
whether `decluster=` passes through.

**S — Scores for ordered categories** (suggested 2026-09-11, with the
categorical scores). Every categorical score is of one category against
the rest, which suits unordered rocks and is blind to order: for an
`OrderedRockType`, calling the unit next door and calling one three units
away cost the same. Two scores see the order: Cohen's weighted kappa
(Cohen 1968), linear weights giving partial credit for a neighbour, and the
ranked probability score (Epstein 1969), the proper score over the
cumulative probabilities, which is to an ordered category what CRPS is to a
grade. Both read the whole classification at once, and the author chose one
column per category for the kappa (2026-09-11), so where an overall score
lives in the table is the open question.

**S–M — Batched prediction from a latent node, into a container**
(requested 2026-09-05; plan agreed and **built 2026-09-28, 0.8.5**;
`test_predict_node.py`. Found in the building: a prediction resumed from
`unpredicted()` matches a whole one to rounding, 5e-15 at the last rows of
a batch, not to the bit as the model's `predict` does -- the same draws,
the batch's shape moving the arithmetic).
A node's `predict(x, x_var, n_sim, seed)` returns the raw four-tuple and
nothing else: no batching, no refresh, no container to land in, so the
values inside a tree -- where a `GPWalk` moved the coordinates, what a
shared parent says before two leaves diverge, what a `Linear` trend
contributes -- were reachable only by hand (chapter 17's benchmark read the
walked coordinates through `interpolate` in hand-cut chunks). Decided:

- **The door is on the model**: `VGPNetwork.predict_node(node, container,
  n_sim=None, name=None, labels=None, where=None)`, run on `_over_batches`
  so the refresh, progress, cancel, `where=` (a mask or a stored filter)
  and `n_sim=None` are `predict`'s own. Any node above the input, the
  leaves included; an input node is refused (the transform answers that),
  as is a node outside the model's tree, and on a `ProjectedVGP` if it does
  not work there. The model's `predict` is unchanged.
- **The result is a `LatentVariable`** of `size` columns, named after the
  node unless `name=`, its parts labelled by `labels=` or numbered; each
  holds `latent_mean` and `latent_variance` (the propagated moments) and
  the simulations -- no prediction column, no measurements, likelihood,
  unit or back-transform. Declared like every variable, so paths, frame,
  pyvista, Zarr, subsetting and carrying come free; `latent_mean` is the
  predicted marker. The docstring says the realizations carry only the
  variance the inducing points explain, and that on a nonlinear node the
  moments are approximations. An existing variable of the same name and
  another class, size or simulation count is refused; a model refuses it
  as a training target, as it refuses a `DerivedVariable`.
- **Realization s is the draw the model used on the way to the first leaf
  that reaches the node**: the seed shifts of `Add`, `LinearCombination`,
  `Multiply` and `ProductOfExperts` (parent i gets `seed[0] + i`) replayed
  along that path, so realization 7 of a trend is the trend inside
  realization 7 of the output -- through operation nodes; a GP above
  another reads its moments, not its draws. The gate: the node's
  realizations, carried through the operations above, give the leaf's.
- **Point support only**: `Blocks3D`, `BlockSet3D`, `RotatedBlockSet3D`
  refused, a `Grid3D` over the same box suggested -- the sub-block average
  is a likelihood's, and a node has none.
- **Catalogue**: `predict_node` in the workflow, `LatentVariable` in
  `variable_types` with its columns' roles and scales. The store format
  stays 2; an older geoML meeting the class fails naming it.
- **Docs**: a section of chapter 17 on the walked coordinates; no figures.

**M — Latent variables as a model's input** (split from the item above,
2026-09-28). A `LatentVariable` may enter a new model as a `GaussianInput`
-- its `latent_mean` the coordinates, its `latent_variance` the location
variance -- but the door is not settled because the use is wider than one
variable: several latent variables joined into one input, and latent
variables from different models joined. To settle: the door (a container
method building a `GaussianData`, or `GaussianInput` taking variables),
how variables from different containers are matched by location, and
whether the full variance or the explained part goes in (the full one is
the honest uncertainty of where a location sits).

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

**S — `refit="leaves"` leaves a `GradientConstrainedInput` as it was**
(found by reading, 2026-09-15). `models._terminal_gp_nodes` treats it as a
stateless root, although it carries `alpha_white`, `delta` and `bias` of
its own, so a leaves refit over a gradient-constrained model keeps what
that node learned from the held-out rows. Count it among the GP nodes.

---

## 4. Data, I/O and interchange

(Mesh sets: **done 2026-09-11, 0.6.10** — see "Settled by measurement"
below for the numbers. `MeshSet` contours a block model or a grid at every
cut-off, or once per category, for the prediction and every realization,
and holds the bodies as one mapping with the reports a set can make.)

(Mesh operations through Manifold: **booleans done 2026-09-10, 0.6.10** —
see "Settled by measurement" below for the numbers. `Solid3D`'s union,
intersection and difference go to manifold3d itself and are exact; the
rest of what Manifold offers was measured and replaces nothing of
geoML's.)

**S — Investigate what `get_contacts` makes of an unlogged interval**
(asked 2026-09-25). `as_point_data(contacts=True)` makes no contact where
a class meets an unlogged stretch or a gap, which is what GeoScape's "To
points" does. `get_contacts` merges runs with two missing values counting
as one run, so a class meeting an interval logged with no category makes a
contact with one side empty -- and `as_classification_input` inherits it.
Whether that is ever wanted, and whether the two should agree, is the
question; nothing has been measured.

**S — Surface I/O residue.** OBJ/PLY/STL both ways, the vendor formats, and
any attribute travelling with the geometry. Nothing has demanded them yet.

**S — `RotatedBlocks3D` has no geoh5 door.** Built 2026-09-11 as what
`BlockSet3D.as_blocks3d` returns for a rotated set. A geoh5 BlockModel
carries one rotation, about the vertical, so a `to_geoh5` would write a
model turned in azimuth only, dip and rake refused as the octree writer
refuses them, with its origin at the turned lattice corner --
`write_grid_blocks` takes the corner from the centres' minimum, which is
right only unturned. `Blocks3D.from_geoh5` could then return a rotated
BlockModel as a `RotatedBlocks3D` rather than today's
`RotatedBlockSet3D(max_levels=0)`; that changes a return type, which is
why it waits for a user.

(`get_contour(close=)` meeting its own cap edge-on and rounding the box's
edges: **done 2026-09-14, 0.6.10**. Found building mesh sets on Assen
(2026-09-11): FeO_total at 0.7 kept a layer one cell thick against the top
of the lattice and met the cap along one edge four triangles shared, the
welded fallback came back open at 44 edges, and a body closed against the
box rounded its edges by half a boundary block (5% of a slab-filled 80 m
box left uncovered); a set retried such shells a hair off their level,
thirteen attempts and twenty minutes a shell on Tom v6. The split of
touching edges (2026-09-13) took the retried Assen shells from 16 to 8; the
rest were the cap folding flat onto itself, drawn as it was through lattice
points valued exactly at the level. Now the ghosts carry copies of the
blocks they mirror, the surface closes a cell past them, and Manifold cuts
the body at the box: the slabs tile their box, a ball against the box
reads within 1.40% of Monte Carlo at worst where it read 2.61%, and a
census of every Assen realization at its own level finds none of 250
shells needing a retry. Not rerun on the Tom v6 model.)

(`simplify` keeping half its promise: **done 2026-09-14, 0.6.10**,
`docs/benchmarks/simplify_both_ways.py`. The cause suspected was the one:
the quadric pre-pass came back open on both Assen shells, was taken at 1
and 2 m for being within half the budget, and every cut started from it,
so both came back whole; a pre-pass that is not the mesh's kind is dropped
now, and BIF and Hematite simplify at 0.5, 1 and 2 m to 12 522-38 856
triangles, within budget both ways -- the reverse is measured too -- in 3
to 7 s. The 0.72 m once read between BIF's vertices and its simplified
shell was 35 vertices no triangle uses, which its store carried.)

**M — A categorical realization's bodies leave a gap where three meet.**
Found building mesh sets on the Assen rocks (2026-09-11,
`docs/mesh-sets.md`): each category contoured on its own field, taken block
by block, three realizations' six bodies overlapped by about 1.1% of the
model and left 0.75-0.84% uncovered. **The overlap is settled
(2026-09-14)**: a realization's categories come from one cut and one paint
of its draws, each field read off the draws' corner means
(`BlockSet3D._contour_fields`), overlap 0.0000% of the model, 54 s a
realization instead of 72. **What is left is a gap of 0.09%**, where three
categories meet inside one cell and each field's marching-cubes piece cuts
off only the corners it wins; a winner's margin taken against its
neighbours' winners closed a tenth of it on a plain lattice, so the cure
is the cell's, not the edge crossings'. What would close it is a
multi-material contour -- each cell split by the argmax of its corner
values, each interface drawn once and handed to both sides -- which VTK
does not offer on scalar fields (its SurfaceNets works on labels, losing
the sub-cell placement). The prediction's own bodies leave 0.053% and
overlap by 0.008%, contoured on the likelihood's per-block fields one at a
time; they would take the same cure.

(One contour of a big block model costing 4 to 8 GB: **more than halved
2026-09-13, 0.6.10**, in the same time and with every mesh identical to the
bit -- see the changelog. The corner tables of the cut and the paint are
built a corner at a time with 32-bit ranks, the first level of the cut is
kept on the set, and the paint's fills go in chunks; on Assen the peaks
fell 55-70% (`docs/benchmarks/contour_stages.py`). A mesh set's pool is
sized by what its prediction's contours measured, so it now takes more
workers on its own. Not done, and not needed yet: keeping the table to the
blocks that can be marked, one ring past any whose corners straddle the
level, which would leave the first level's the only full one. Measure the
Tom v6 model before taking it up.)

(Mesh set workers scaling 2.2 times on eight: **explained 2026-09-14**,
`docs/benchmarks/mesh_set_workers.py`. None of the three candidates: the
parent's writes took 3-15 s of a 100-290 s pool, the 114 MB a realization
sends back nothing measurable, and VTK runs one thread here. A forked
worker runs Manifold on one thread, the parent having started its thread
pool (a boolean 3.8 s on one thread against 1.1 s on eighteen, which burn
five times the CPU), so a task is one core's work -- 93 s, where one
process takes 45 s on two. And the machine saturates: 8, 12 and 24 workers
give 12.0, 10.8 and 11.2 s a realization, each task's CPU time growing with
the workers for the same work (93, 125, 256 s), every stage alike -- a
16-core, two-channel Ryzen 9 7950X out of memory bandwidth, then out of
cores. Steps 3 and 4's leaner contours took eight workers from 27 s a
realization to 12.8 s (3.6 times one process); OpenBLAS held to one thread
a worker saves 38% of each task's CPU, 80 s of 173 having been spinning,
for 5% of the time. Past that the only lever is less memory traffic a
contour.)

(A mesh set's summary showing only what the limits left: **done
2026-09-14, 0.6.10** -- with limits present, printing a set shows each
shell's volume before them beside what is left, a set reopened from its
store too, where the Tom v6 sets an uncertainty limit had emptied read
"volume 0" at every cut-off.)

(A contour through blocks without a value getting NaN vertices: **done
2026-09-15, 0.6.12**. Every fourth block of a small ball model unpredicted
put 138 of 548 vertices at NaN in a `Solid3D` of NaN volume, and a region
left out whole did the same wherever the surface reached it -- the
contiguous one once reported clean had simply not been reached. A
valueless block is now read across from its corners where valued blocks
have them all, and left whole rather than cut; anything else is absent
ground, where an open contour stops and a closed one closes, within a
hundredth of a cell. `Mesh3D` refuses a point that is not a finite
number.)

**S–M — Refine on the average, not on every realization** (requested
2026-09-29; **built 2026-09-29, 0.8.5**: 2633 blocks against 582 on the
gate's case, the empty half left at 32 coarse blocks instead of 1439, the
volume on the wrong side of the cut-off 625 m3 against 500;
`docs/variable-block-models.md`). `needs_splitting` marks a block by `divided`
(`likelihood._divided`): the share of realizations whose sub-blocks fall
on both sides of a cut-off, each realization judged on its own. Where the
data do not constrain the model, every realization is rough on its own
account and straddles somewhere, so the unconstrained ground is refined as
hard as the contacts are -- the blocks multiply where there is least to
resolve. Wanted: judge the split on the average alone -- the sub-blocks of
the prediction (the mean over realizations, per sub-block), a block cut
where those fall on both sides of the cut-off, a category's on the mean
`ind_skew`. This reverses `_divided`'s own argument (that a block is
divided only if one realization holds two answers, so that model doubt
never licenses a cut): the mean is smooth where the realizations disagree,
which is the point. Only the continuous likelihoods change -- a category's
`ind_skew` is read off the probabilities, already expectations. **Plan
agreed 2026-09-29**: `_divided` judges the per-sub-block mean over
realizations (the prediction at sub-block support, the field a contour is
drawn on), so `divided` is redefined as a 0/1 flag, its catalogue role a
`value` on `flag`; `tolerance` on `refine` and `needs_splitting` is kept
and deprecated -- a warning when passed, removed later -- since any value
under one now gives the same answer. The gate, red first: a small 3-D set
with data in one half and inducing points in both, the unconstrained half
left nearly unrefined, the contact still refined, the refined contour
within a stated tolerance of a fine grid's; measured before and after, and
recorded in `docs/variable-block-models.md`.

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

**M — Save a container into the store it was opened from.** Refused since
0.6.13, because `to_zarr` emptied the store before copying the arrays it
was reading from it, and wrote back NaN. What a notebook wants after
`open`, a prediction and `to_zarr(same path)` is an in-place write: the
arrays already in the store stay, the new variables' arrays are added, the
removed ones deleted and the root attribute rewritten -- what
`MeshSet.to_zarr` already does for its description alone. The same for a
model loaded and saved over its own store.

---

## 5. Housekeeping, docs and release

(The tag's `full` CI job dying at 93%: **done 2026-09-15, 0.6.12, and
passed on the v0.6.12 tag in 37 minutes**, the first full job to pass on
a tag since v0.6.9's. On the v0.6.10 and v0.6.11 tags the suite ran clean
to 93-95% and was cancelled with no test failing; as one pytest process
it peaks at 23.1 GB, and the runner has 16. The job runs one process per
test file now: under a 14 GB cap every file passed, the heaviest peaking
at 6.1 GB.)

**L — Simplify after the expected kernel, breaking old saves** (agreed
2026-10-08, after 0.9.0; a major version). 0.9.0 keeps every save opening
by carrying two propagation rules side by side. Once the expected kernel
has been in use for a release, drop what it made obsolete, and say in the
changelog that saves from before cannot be read (or ship a converter that
refits):
- `GPOptions(propagation=)` and marginal propagation: the Paciorek
  inflation in `covariance_matrix`, `inducing_points_variance` as a
  separate diagonal, and the `None` covariance branch in every node.
- `GPOptions(expert_propagation=)` and the consensus rule (the O(E²)
  cross-prediction in `BasicGP.refresh` and its slot twin).
- `Spherical`, `Cubic` and `Cosine` as kernels of a GP node, and the
  refusals that guard them.
- Possibly the `kernel` argument itself, nodes described by their
  smoothness instead (the Matérn family's order, the Gaussian as its
  limit), which every remaining kernel is a point of.
- `UncertainInputGP`, once `BasicGP` takes the second moment at an
  uncertain input (the item in section 1): measured 2026-10-08, the expected
  kernel alone misses the variance there by up to 133%, where
  `UncertainInputGP` is within 3%.
- **Deprecated in 0.9.0** (ignored under `"joint"`, removed here):
  `GPWalk`'s `precision` parameter and the variance shrinking it drives.
  (The walk's own KL term was to go too, and was measured necessary: see
  the 0.9.0 item.)
Measure nothing new for it: it removes code whose replacement the 0.9.0
gates passed.

**M — Keep the package skill true at every release** (agreed 2026-09-26;
**built 2026-09-26**, 0.8.2: `--show`/`--list`/`--json`, `sync.py`, the
skill rewritten, `test_skill.py` and `test_skill_release.py`).
Found: two skills both named `geo-ml` — the research skill outside the
repository (research line, notation, LaTeX, style, and a section 5 on the
package) and the package skill in `plugins/geoml/skills/` — whose package
text had drifted apart, no script generating one from the other though this
file's Version note said one did, and every release bumping only the
version line: at 0.8.0 the package skill still named the shims removed in
0.7.0 and never mentioned the parametric warpings, leaves, `MeshSet`, the
catalogue, `progress` or `unpredicted`. Settled:
- **Names**: the package skill is `geoml`, the research skill
  `geoml-research`, so the two never share one in a session.
- **One source**: the package skill holds all package knowledge. The
  research skill's section 5 shrinks to a pointer and about ten lines (the
  object model in a paragraph, the install line, where the manual and the
  reference live), since it is also used where the plugin is not installed.
- **Prose only for what the catalogue cannot say**: the object model, the
  workflow, the recommendations, the gotchas, the geostatistical
  intuitions. The class lists and argument details go; a recommendation
  names its classes and sends the reader to the catalogue for arguments.
  Internal classes are named nowhere, experimental ones only as such.
- **The catalogue queried, not copied**: `python -m geoml.catalogue` gains
  `--show NAME` (short or dotted; an ambiguous short name lists the
  candidates), `--list CATEGORY` and `--json`, writing the whole file as
  now when given a path. The skill has the agent query the installed
  version for any signature or bound, and falls back to the site's
  reference pages where geoML is not installed. A snapshot was rejected:
  stale the moment the user installs another version, and the whole
  catalogue is 282 KB (about 70k tokens).
- **The manual replaces the notebooks**: the 17 chapters, run at every
  release by `test_manual.py`, copied into `references/` as Markdown with
  their figure links rewritten to the site (the figures are 5.7 MB); the
  eight notebooks, which predate the path notation and nothing tests, go.
  `notebook-style.md` stays.
- **Checks**: a name check in CI's structural job (every dotted name and
  every class or function the skill mentions resolves and is not
  internal) and a code check at release (every Python block in the skill
  runs).
- **The release step**, written into CLAUDE.md's Version note: bump the
  version; run `plugins/geoml/sync.py` (the chapters copied, links
  rewritten); review the skill's prose against the changelog's top
  section; both checks pass.

(The tag-triggered manual CI job: **verified 2026-09-14**. At v0.6.9 it was
killed at 22 minutes of a 90-minute budget with no pytest output, the OOM
signature of a 16 GB runner accumulating TensorFlow, matplotlib and
pyvista. With `test_manual.py` running one subprocess per chapter, it
passed on the v0.6.10 tag in 28 minutes. If it ever dies the same way
again, the next lever is not a longer timeout.)

(Chapter 13's variogram verdict, read against its figure: **done
2026-09-15, 0.6.12**, `docs/benchmarks/walker_zero_shift.py`. The model
did fit its noise high, and the zeros were why: Box-Cox's default shift of
a millionth put Walker's 22 zero samples so far down the logarithm that
they pulled the exponent from 0.58 to 0.42, and the steeper inverse
widened the noise where the grade is high, noise and ground together 1.19
of the data's declustered variance. Shifted by one, 0.98, and the fan
follows the true variogram within 7% from the second lag on; denser
inducing points, longer training and an exponential kernel mended none of
it. Every chapter that builds Walker's V shifts it now, 4, 5, 7, 11, 13
and 15, and 13 and 15 read their figures anew; the shortest lag's
comparison with the truth is measured now, 26 900 true against 46 600 in
the declustered data. Chapter 16's Jura holds no zeros, its smallest
value 0.135, and keeps the default. Chapter 5 described its chain as
chapter 4's "with a `Scale` in front" and credited "the softplus in the
chain" with its positivity, neither of which it held; corrected with it.)

**S — Chapters 16 and 17 no longer reproduce their figures byte for
byte** (found 2026-09-15, at the 0.6.12 release run; **chapter 16 fixed
2026-09-15, 0.7.0**). Chapter 16 changed from run to run: two runs of one
checkout printed rmse 0.765 / 2.645 / ... / 32.830 and 0.768 / 2.646 /
... / 32.831, its six figures moved by up to a fifth of their pixels,
invisibly, and one of three runs landed back on the committed figures.
Chapter 17 comes back the same in three runs, on 0.6.12's code and on
0.6.11's, but not as committed on 2026-08-19, at under 0.6% of its pixels.
Both reproduced exactly at the 0.6.11 release the day before, so this
release's code was not the cause.

The cause was **`RobustPCA`, whose FastMCD drew its starting subsets from
NumPy's global generator** -- which `set_seed` does not reach. The note
above says it was seeded from the package RNG; that was assumed from
`Rotation`'s FastICA, which is, and never checked. Chapter 16's chain
holds a `RobustPCA` and chapter 17's does not, which is exactly the split
between the chapter that moved every run and the chapter that does not.
Seeded (`warping.py`, `random_state` from `_rnd.rng()`), four processes
under different hash seeds build that model to the same bits and three
train it to the same bound. Two orders that changed with the process went
with it -- `_Operation.get_unique_parents` iterated a set, which is the
order `VGPNetwork._nodes` sums the KL in, and `get_unfixed_variables`
likewise -- neither having moved a number, both able to.

Chapter 16 was re-run on the fix and reproduced all six committed figures
byte for byte, printing the rmse of the higher of the two runs, 0.768 /
2.646 / 8.492 / 6.093 / 38.130 / 32.831, and §16.5's prose still holds.

Chapter 17's own 0.6% is **not** this: no `RobustPCA`, no `Rotation`, and
three runs agreed with each other before the fix. Two more agree after it,
identically, 0.554% of `17-elbo.png` and 0.119% of `17-vein-surface.png`
from the committed pair and nothing else touched -- a changed number, not a
nondeterminism, its committed figures predating 0.6.12 by three weeks. Its
prose quotes none of the numbers it prints. **Re-commit those two figures
and the item closes.** Committed with 0.7.0 on 2026-09-16, after a third
identical run inside the release suite; **closed.**

The contract they broke is written down now:
`docs/source/reference/reproducibility.md`, pinned by five tests in
`test_seed.py`, the last of them training and predicting in another
process.

(Chapter 16's figures: **verified 2026-09-05** — rerun through the manual's
own runner after the leaves change, the seeded chapter reproduced every
committed figure byte for byte, and the §16.5 prose matches what the model
prints: Portlandian at a balanced accuracy of 0.5 and a Jaccard of zero.)

(**S — Retire the 0.6.0 deprecation shims in 0.7.0: done 2026-09-16.**
The ten one-line modules and the lazy `__getattr__` that resolved them are
gone. The persistence check was re-run first, and against the code rather
than a reading of it: a store records the path of a `Parametric` or
`_ModelOptions` class only, anything else refusing to save, and walking
every subclass of both finds them in six modules -- `kernels`,
`latent.network`, `likelihood`, `models`, `transform`, `warping` -- none a
shim, while the ten shims' targets define none. So no save, old or new, can
name a removed path. `test_removed_paths.py` pins the removal and, more
usefully, that every recordable class resolves from the path it records.)

**S — `latent/network.py` stays out of the pyright list.** About 28 of the
file's diagnostics read `inducing_points` as possibly-None, and they cannot
be fixed by declaring them: `None` is load-bearing in the finished state.
Declaring `tuple` took 33 diagnostics to 67 and was reverted. Do not repeat
that attempt.

**S — One checked file is not clean: `warping.py`** (found 2026-09-11,
re-measured 2026-09-15). The 251 errors this item used to list are gone.
Pyright 1.1.411 over the `[tool.pyright]` list, run from the repository
root in the `geoml` conda env, now reports **one**:
`warping.py:1294` (1281 before 0.8.0's docstrings) `"NoReturn" is not
iterable`, on
`_ContinuousFlow.initialize`'s `x, _ = self.forward(x)` -- the base class's
`forward` raises, so the checker reads the call as returning nothing and
the unpacking as impossible. It is on HEAD's version of the file too, so it
predates the 0.7.0 work. Whatever moved (a library version, the checker's
own inference) took `data/drillhole.py`'s 107, `data/geoh5.py`'s 36 and the
rest with it, which is why the old census is not worth chasing. Declare the
return on the flow base, or annotate the call; either is minutes.

---

## 6. What GeoScape needs

GeoScape is a UI that calls geoML: it draws networks, writes the Scripts
that build and run them, and reads what they leave behind. Its repository
(`C:\Repos\geoscape`, same owner) keeps what it needs from geoML in
`docs/geoml-requirements.md`, revised 2026-09-15 against 0.6.11, and the
catalogue's format in `docs/geoml-catalogue.md`. "GeoScape's item N" below
is that list's numbering, and M3 and M4 its milestones. **Every item on it
is met as of 0.8.0**; what is left below is the paperwork outside the code.
Three were met before the list was worked through -- one leaf per
likelihood (item 2) and names replayed by a save (item 5), both in 0.6.10,
and `geoml.__version__` at run time (item 7) -- then seven in 0.6.13,
items 1, 3, 8, 13 and 15 in 0.7.0, and the eighteen the list gained on
2026-09-16 (16-33) in 0.8.0, item 26 turning out not to be a bug, and
the three added 2026-09-27 (34-36) in 0.8.3 -- item 34 a masked subset of
any container opened from Zarr, not only `cross_validate`'s. Of the two
added 2026-09-28, both in 0.8.4: item 38, read-only opens; and item 37,
which asked `GPOptions.expert_propagation`'s docstring to say the option
affects any network with experts. Measured, it does not -- a
`LinearCombination` or `Add` of GP nodes over 3 and 20 experts trains to
the same log and predicts the same means to the bit under both rules, in
the same time -- so the docstring now says an operation over GP nodes is
still one layer. The done-notes below hold what each cost and what it taught; GeoScape's
requirements document is marked item by item since 0.8.0.

(Items 16-33: **done 2026-09-25, 0.8.0.** The catalogue is format 2: a
`description` on every entry, read off the docstring between its summary
and its first section, and forty docstrings written to have one; every
bound a constructor clips to declared as a constraint, a test holding each
declaration to the built parameter and the constructor warning when it
clips; `network: false` on `Constant` and `Cosine`; the multivariate
likelihoods `internal`; `Spline` public, GeoScape passing `backbone="rq"`
itself since the default must stay `"cubic"` for old saves; `PCA`'s width
out a `param` rule with a `fallback`; `contour_rule` on the two
categorical likelihoods; a `containers` section, every public container
listed or left out with a reason, each built and predicted into by a test;
each variable type's columns with a role and a scale. In the code:
`predict(where=)` takes a stored filter's name, `n_sim=None` takes the
target's count and a different one is refused under `where` (one
realization used to be copied into every stored column without a word);
`combine` merges by distance; a categorical's measurement columns share its
order of labels; `as_point_data(contacts=True)`; and `predict` into
measured data writes each measurement's PIT and its warped value as
metadata. Two bugs came out of it: `BlockSet3D.unpredicted` never
cleared its split flags, so a fully predicted set read as unpredicted
throughout (0.7.0's union), and `cross_validate` scored a composition
declared in units against its assays as fractions of the whole, every PIT
0.)

What GeoScape builds on, where a change must be flagged to it rather than
made quietly: dotted class paths importable forever, already policy;
constructor arguments as the whole persisted story, since the editor
exposes nothing else; the Fourier-feature classes kept internal; **batch
invariance**, a location's realizations independent of the call that
computed them given the same fit, seed and count
(`test_predicting_only_the_new_blocks_gives_the_same_answer`), which a
per-location residual draw would break; and node names seeding the draws,
GeoScape never renaming a node once created.

(The catalogue, GeoScape's item 1: **done 2026-09-15, for 0.7.0**,
`geoml/catalogue.py`, design record `docs/catalogue.md`. 104 classes and
21 functions -- the last two being `geoml.progress` and `unpredicted`,
added with items 13 and 15 so that GeoScape can discover the calls it needs
for them -- every class declaring itself in a block where its module
ends, and 238 tests building each declaration against its class. GeoScape's
spec was revised with the answers read off the code, four amendments and a
`nullable` field. The fixes its tests demanded went with it: `Identity()`
and `Periodic()` saved at last, the shared `Identity()` defaults replaced by
`None`, `Concatenate` asking its parents before claiming to pass inducing
points on, operations refusing no parents, `GPWalk` a non-GP and
`MultiStructureGP` a single structure, and the `__all__` gaps closed.
`NormalizeWithBoundingBox` still cannot be saved, a `BoundingBox` not being
encodable, and is catalogued internal.)

(GeoScape's items 4, 6, 9, 10, 11, 12 and 14: **done 2026-09-15, for
0.6.13**; the changelog has the detail. A target whose cut-offs differ from
the model's takes the model's, in its own unit and with a warning -- the
user chose re-keying over raising -- and `set_cutoffs` drops the shares of
the cut-offs it removes. The stopping rule's trail and `train_svi`'s
shuffle belong to the phase, so chunks reproduce one call bit for bit, and
`converged` tells a Script when to stop calling. Every store writer keeps
the root attributes not starting with `geoml`, and none writes over a store
something still reads from, which had been silent data loss for an opened
container, a loaded model and a mesh-set realization. The cross-validation
container round-trips, tested, its score columns documented; chapters 8
and 10 say at or below, and a category's share is the share inside it, so
GeoScape flips grades only. `docs/source/reference/stores.md` documents
the stores, the mesh set's attribute gaining `groups`, the layout pinned
by `test_the_store_is_laid_out_as_its_reference_page_says`.)

(**S — The reproducibility contract, written down** (GeoScape's item 8):
**done 2026-09-15, 0.7.0.** `docs/source/reference/reproducibility.md` says
what comes back the same -- the parameters, the log, the predictions, the
realizations and the measurement samples, in any process; a location's
realizations whatever the batching; a node's draws, keyed by its name; a
training split into chunks; a mesh set contoured on workers; the catalogue
-- and what does not: another machine, another version, a GPU, another
release. The one knob is `set_seed` before anything is built. Finding out
which was which is what closed §5's chapter-16 item: `RobustPCA` left
FastMCD reading NumPy's global generator, and two orders came off a set.
Five tests in `test_seed.py`, the last training and predicting in a second
process under a different hash seed and comparing every number as hex.)

(**M — A progress hook** (GeoScape's item 13): **done 2026-09-15, 0.7.0.**
`geoml.progress(callback)`, a context manager, one `Progress(task, done,
total, unit, bound, within)` per unit *finished* -- `train_full` per
iteration, `train_svi` per batch, `predict` per batch, `refine` per pass,
`cross_validate` per fold, a mesh set per prediction body and per
realization. The user chose the context manager over a per-call argument
(D4), and cancel is the callback raising. A `ContextVar` rather than an
argument because the calls nest; `within` names the enclosing tasks, so a
refinement's predictions are told from a bare one. **Logging was rejected**:
`Handler.handleError` swallows a handler's exception, so a cancel could not
travel back. Two things the build taught. A generator must not set a
`ContextVar` -- its body runs in the caller's context, so the mark leaks at
every `yield` and outlives an abandoned generator -- which is why
`_over_batches` reports through `emit` and not `reporting`. And **a
refinement's passes are not bounded by `max_levels`**: a block still at
level 0 can be marked by any later pass as `unbalanced` reaches it, so
`refine` reports no total. A test caught the wrong cap.)

(**M — A mesh set openable mid-build, and `unpredicted()` everywhere**
(GeoScape's item 15): **done 2026-09-15, 0.7.0.** The user chose resume over
discard (D5). The set's description goes into the store before the first
realization and is rewritten after each one, `complete: false` and only the
rows actually filled (`_attrs(keep=)`, the measure tables being allocated
for every realization up front); `open` reads a partial store, warns, and
reports `complete`. Store format **2**, format 1 still read since it was
always a finished set. `unpredicted()` is on `_SpatialData` now, each
variable class declaring its marker column (`_PREDICTED_MARKER`) because a
rock type has an `entropy` and a vector variable an `uncertainty` where a
grade has a `prediction`; without a variable a location counts as
unpredicted where *any* of them misses it. `BlockSet3D` keeps its override
and unions the two notions. The gate: predicting what `unpredicted()` names
after a cancel gives, to the last bit, what predicting the lot gives.)

(**M — Chunks that split the realization axis** (GeoScape's item 3, M3,
performance): **done 2026-09-15, 0.7.0.** Past 32 realizations the trailing
axis is split too, ten columns a chunk. Measured cold -- page cache dropped
-- on a `(2 000 000, 100)` float64 store, 1.49 GB, at 100, 25 and 10
columns a chunk: reading one realization 0.76, 0.13 and 0.06 s, **and the
reductions no worse for it**, a quantile pass 1.34, 1.09 and 1.26 s and a
pass in row bands 2.53, 1.19 and 1.07 s, ten columns a chunk being ten
times the rows and so fewer, longer reads. Ten wins on every measure, so
there is no trade-off left to tune:
`docs/benchmarks/realization_chunks.py`. A band still holds whole rows,
reading each of its column chunks; `row_quantiles`/`row_cdf` gather the
realization axis first (`_whole_rows`), one pass over the same bytes, since
a block of a column-chunked dask array holds part of a row. Stores written
earlier keep their chunks.)

Outside the code, from the same list: a contributor licence agreement
before the first outside contribution is merged; and title in writing, with
no institutional claim, before any geoML code ships inside something
RockAnalytica distributes -- the desktop helper (M7), a self-hosted tier,
or a `geoscape-cli` that imports geoML. Version 1 and the cloud need
neither, Scripts being text and portal services not distribution.

---

## Settled by measurement

These were tried. The numbers are why they are, or are not, in the package.

**Experts that keep to their neighbours -- done** (raised 2026-10-02,
measured and replaced 2026-10-08, 0.8.8). `inducing.experts` lent an
expert points far from its own cluster. Both suspects were real: the cap
at n/k in `_balanced_labels`, placing points one at a time, filled the
nearby clusters first and sent the last points to whichever cluster had
room -- whole cores of 25 assembled from leftovers at 20 experts -- and
borrowing by the core's Mahalanobis distance from its centre, the
ellipsoid of a cluster along a drillhole, reached past the neighbours down
the hole's line. Measured (`docs/benchmarks/expert_overlap.py`, which keeps
the old algorithm) on 500 k-means points of drillholes with uneven density
and on Tom East's assayed holes, against three compact algorithms with a
+-10% size band and borrowing by Euclidean distance to the nearest core
member -- trading points after the cap, a size-bounded k-means assignment
solved as a transport problem, and recursive bisection along the
principal axis. Worst over the experts, at 20 on Tom East: a core member
6.22 median radii from its centre and an isolated one 21.8 core spacings
from its nearest fellow, against 2.64 / 6.51 trading, 2.17 / 2.86
transport and 3.58 / 18.0 bisection; borrowed points' median gap 2.13
spacings against about one for all three. The held-out score of a rock
model (three seeds, which barely differ since a layout is one per
algorithm) did not decide it: transport best at 5 experts (AUC 0.896
against 0.875), worst at 20 (0.799 against 0.825), trading and bisection
between. The figures showed transport's cores tiling the field and
following the holes, and the user chose on geometry. `experts` is now
k-means with the transport assignment (`balance=0.1`), each point's 8
nearest centres as its candidates -- 6000 points in 64 experts 45 s to
10 s, the same assignment wherever that is feasible, the full problem
where not. **Borrowing is spread over the neighbours** (the user's rule):
up to `ceil(overlap x own)` points, one from each neighbour a round, that
neighbour's nearest to the core, nearest neighbour first; a neighbour is a
cluster some member faces as its nearest point outside, or that faces it.
Borrowing by nearness alone left most touching experts sharing nothing --
on the drillhole case at 20 experts, 28 of 44 touching pairs at overlap
0.1, 17 at 0.2, 6 still at 0.5 -- since every point came from the one or
two nearest neighbours. Spread, 12 of 43 facing pairs at 0.1, where an
expert of 25 points borrows 3 against up to six neighbours, and none at
0.2 or more: the rule reaches every neighbour once an expert is large
enough for its overlap. Tests in `test_experts.py`: the band, even
borrowing, every facing neighbour shared where the budget covers them,
and compact cores along the drillhole case, which the old algorithm
fails. On the way, prediction by expert was found drawing other normals
for a GP of two outputs wherever experts differ in size -- a slot drew one
array at the padded size where `simulate` draws each expert's at its own,
and a draw fills its array in order -- hidden while experts were equal;
the slots now gather each expert's own draw.

**Mesh sets — done** (2026-09-11, 0.6.10). `geoml/data/meshsets.py`;
design record `docs/mesh-sets.md`, measurements
`docs/benchmarks/mesh_sets.py`. The gate the item set, on the Assen block
model (908 237 blocks, 25 realizations, eight workers): the prediction's
four FeO_total shells cross each other by 1.8e-15 m³ as contoured and no
realization's by more than 7e-15; simplified to 1 m and nested again, the
nesting took nothing back -- so `repair` stays off by default. A
realization costs 59 s in one process at four cut-offs, 27 s each on eight
workers; the Fe set took 800 s whole, the six rocks 1024 s, the largest
worker 6.8 and 9.1 GB. What the set measured: the prediction's shell at
0.85 holds 1.42 Mm³ where the realizations hold 1.65 / 2.18 / 3.00
(P10/P50/P90), smaller than 23 of the 25, while at 0.6, below the median
grade, it is larger than three in four -- the smoothing a mean field does
at either tail -- and every realization holds more BIF than the prediction,
by a third at the median. The rock set's prediction bodies overlap by 0.008%
of the model and leave 0.1% uncovered, mostly the box's rounded edges; a
realization's overlap by 1.1% and leave 0.8% (an open item in §4). Found
and settled on the way: a band between two shells closed against the same
face touched itself, which geoML's welding read as a `Mesh3D` -- fixed in
the booleans (`_separated`); and a contour meeting its own cap edge-on,
which a set now retries a hair off its level, recorded as `nudge` (7 of
the 104 Fe meshes needed it, one at 1e-4 of the span, and 8 of the 156 rock
bodies, of which one did not close even so). The item as filed asked for tonnage from the
realizations' own bands too: that is `realization_table`, opt-in, since it
asks the blocks near every band of every realization about their
sub-blocks.

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
