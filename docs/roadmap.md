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

**S — XLA refuses a walk network's prediction on the CPU** (found
2026-11-07): `Invalid TensorList shape: (f64[5,5,2,6,1021]) expected
(f64[5,25,2,6,1021])`, a realization count of 5 against 25 in a list the
walk builds, on the Assen GeoScape network at `n_sim=25`. geoML warns and
predicts without XLA, so it costs speed, not answers.

**M — The walk's memory beyond 0.9.2** (measured 2026-10-16 on a GeoScape
network: a size-3 field, a five-step walk, three readers, 32 experts of
~200 inducing points, 17 active on a batch, a 16 GB card). 0.9.2 took one
training epoch from ~26 GB (out of memory) to 3.7 GB, in 153 s an epoch:
each walk step recomputed in the backward pass (10.1 GB), the expected
kernel's sum over components given a gradient of its own (8.2 GB), and a
parent several nodes share refreshed and propagated once a pass (3.7 GB,
half the time) -- the walk sat under four paths and a step walked seven
times. What is left at the 3.65 GB peak, read off the allocator's trace
(`tf.profiler` events replayed, the walk's pieces in name scopes): 0.80 GB
held before the step; ~1.0 GB in the per-expert Choleskys and solves of
the refresh, forward and gradient (`[17, 3, 227, 227]` for the size-3
GPs, `[17, 1, 227, 227]` for the size-1 one) -- the model's own size,
where the peak sits; 0.6 GB in the walk's stored checkpoints, one `h`
(`[17, n, 3, 3, 227]`) per step for each of the two walks; ~0.2 GB in the
readers' expected kernel; the rest small or of unknown shape (~1 GB).
Tried and dropped: recomputing the walked covariance (no gain); the slope
given the expected kernel's own gradient (1 GB worse -- one Gaussian
component, four accumulators); recomputing the readers' kernel with
`tf.recompute_grad` instead (0.24 GB, the executor rebuilt every component
at once); handing the field's matrices from step to step so their
gradient comes back as one sum (no gain); TensorFlow's arithmetic
optimizer switched off (no gain). Ideas that would change the method
rather than squeeze it:
- **Fewer, smaller experts.** The expert matrices and `h` grow with the
  square of the points per expert, and the matrices are now the largest
  item: the same inducing points in more experts should cost less per
  batch. Measure memory and score together.
- **A local `h`.** A walker's sensitivity to the field's normals reaches
  all `m` inducing points of its expert, but the field it meets is the
  kernel's reach around its path: keeping only the nearest few dozen
  points per walker would cut `h`, its checkpoints and its products by
  `m / k`. Needs a measure of what the truncated covariance loses against
  the walk-along-draws test.
- **Fewer checkpoints.** Storing every second step's `h` and recomputing
  from it halves the 0.6 GB for one more step's work in the backward
  pass; worth it only if the walk grows longer.

**M — The shared prior draw's fixed part, once per call** (filed
2026-10-27). A prediction batch at a certain input rebuilds, in every GP
node, what `_shared_prior_simulate` computes at the inducing points alone:
the features there, whitened (`_whitened_prior`, `[slots, m, J + 1]`), and
`chol_r` times them (`[slots, size, m, J + 1]`). None of it depends on the
batch's rows -- only on the node's seed, its state and the experts in the
slots. Profiled on the GeoScape Run behind item 45 (four GP nodes, 19
slots, J = 1024, an RTX 4060 Ti): a batch costs about 0.76 s whatever its
rows plus 2 ms a row, 99% inside the graph, of it about 0.32 s of float64
matrix products a 56-row batch spends at the inducing points and the rest
launching some 3200 kernels. XLA halves the fixed part to 0.38 s, and the
batches 0.9.11 grows past the estimate spread it: at the steady sizes it is
about 5% of a batch under XLA (613 blocks, 7.2 s) and 26% on the plain
graph (132 blocks, 2.9 s) -- the plain graph being what a model saved
before 0.9.11 predicts with, and a GPU network reading an uncertain input.
- **What it takes.** The values depend on the node's seed -- the basis and
  the feature weights are drawn from it, and an operation above shifts its
  parents' seeds (`_SHIFTS_PARENT_SEEDS`, replayed by `_seed_offset`) -- and
  on the experts in the slots, which change per group of a prediction by
  expert. So a cache keyed by seed, held in Variables the traced graph
  reads, rebuilt at each call's refresh (`refresh_cached`) for every seed
  the call will draw at. A cache left stale gives other realizations and
  no error, which is the risk to gate on: realizations with the cache equal
  to those without, bit for bit, by expert and with every expert, through
  `Add`/`LinearCombination`, a `GPWalk` riding the field's normals
  (`_shared_normals`), `predict_node` and `measurement_batches` (which runs
  eagerly).
- **The other way round**, deeper: contract the prior with the realizations'
  normals first (`[slots, m, S]`, S the realizations, against `J + 1`
  features) so that a row costs `S` columns rather than `J + 1`. What stops
  it is the blend's rescale, which needs each row's norm over the features
  -- the same obstacle the experts-as-blocks item names -- and would have to
  come from the features' Gram matrices across experts.
- Go only if a plain-graph workload matters enough (old saves, GPU
  networks with the quadrature); under XLA there is little left to win.

**M — Refinement cuts unsampled ground wherever the background sits on a
cut-off** (found 2026-10-27, on a GeoScape Run). GeoScape's Run 12 -- a
`GPWalk` network, a `LikelihoodMixture` of two populations over Ag, Pb and
Zn with `shares="latent"` beside a rock type, 32 experts -- refined a 20 m
block set to three levels, every variable voting (`split_on` unset): the
rock contact, and Ag at 100/200/300 ppm, Pb and Zn at 5/10/15 %. Pass 1
cut 21405 of 22724 blocks, and the run was stopped seven hours in, its
third prediction's first group about a million blocks at 68 ms a block.
Read off the first prediction (`C:\Users\Public\segv\why94.py`):
- **What marks the blocks.** 93% of them by any decision, 98% beyond 100 m
  of a sample and 61% within 20 m. The rock contact marks 10.5% within
  20 m and none past 50 m, Ag next to nothing; **Zn @ 10 marks 79% of the
  blocks beyond 50 m, Zn @ 15 half of them, Pb @ 5 57%**. A block is
  divided where its sub-blocks' prediction straddles a cut-off, and far
  from the data the prediction sits on these three: Zn 11.2 % (5-95 %:
  7.6-15.6), Pb 3.9 % (2.9-5.4), against 3.5 % and 1.6 % within 20 m.
- **Why the background is high.** The populations are a barren one (Zn
  0.03 %) and a mineralized one (Zn 12.5 % near the data). Near the data
  the shares are the samples', 72 % barren; far from them they return to
  the mixture's trained `bias`, **59 % mineralized**, and the mineralized
  population's own mean rises with the latent variance (Zn 12.5 -> 19.2 %).
  Not the mean alone: the realizations' median Zn is 8.4 % far and 0.8 %
  near. The data's own median is 2.0 %, mean 4.6 %. Whether a bias that
  calls unsampled ground mineralized is what the fit should say (drilling
  aimed at ore, no declustering in the share's start, the bias weakly
  identified where the shares' GP explains the data) is open.
- **What to do**, in order of cost: GeoScape names `split_on` (the rock
  type alone here, or the metals at the cut-offs that matter), which is
  what `needs_splitting`'s docstring advises for a polymetallic deposit;
  `refine(where=)` with `assign_from_data` keeps unsampled ground out of
  the passes altogether (it is then not predicted either); in geoML, a
  criterion that leaves a block whose cut-off the model cannot place --
  the mean's contour through ground where the variance is the prior's
  buys no surface, the argument 0.8.5 made against judging realizations
  -- needs a measure of "cannot place" (the explained share of the
  variance, the proportion's distance from a half) and a gate on a model
  with a real contact. And the share bias above, as a question of its own.
  **Built 2026-10-29 (0.9.12)** -- the criterion is the realizations'
  coherence (`likelihood._coherence`, the share of the variation inside a
  block they hold in common), at least max(0.2, a Beta noise floor):
  Run 12's far blocks marked 0% at 25 and 200 realizations, every block
  the old rule marked on four known fields still marked
  (`docs/benchmarks/split_coherence.py`). The share bias stays open.

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
expected kernel replaces, and it is deprecated since 2026-10-09 (removed
with the simplification item in section 5); `RadialTrend` is a nonlinear function of its parent's
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
   clock, the shared parameters freeze before they converge. A
   natural-gradient step held constant rises at 16 visits too (2026-10-09,
   "Natural-gradient steps for the experts" under *Settled by
   measurement*), so the decaying clocks are not the whole cause.
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

**L–XL — Experts as blocks of one GP, conditioned in sequence** (requested
2026-10-26; theory only, nothing measured). The user's proposal: with the
Fourier-feature prior draw and pathwise conditioning, condition the
experts one at a time in a random order -- the first conditions the prior
draws, its conditioned realizations are the prior for the next -- as
sequential Gaussian simulation does, a whole expert at a time; no blend,
and experts that are not neighbours treated as independent, a Vecchia
approximation in blocks. The assessment:
- **The sequence is sound, but not over today's experts.** An expert's
  posterior is a GP conditioned on m pseudo-observations at its inducing
  points with noise D (`k (K + D)^-1`, the sites), so the sequence is the
  chain rule `p(f_1..f_J | sites) = prod_j p(f_j | f_<j, sites)`, and done
  in full it draws from the GP conditioned on every expert's sites
  jointly. The sites as trained explain the data through the *blend*, each
  expert seeing its data alone; conditioning on all of them counts the
  data twice where experts overlap and partly along every edge. The 0.9.4
  coupling measurement says so: sequential conditioning on an earlier
  expert's values kept 0.72 of the variance and pulled 74% of the means
  off (the expert-coupling list under "Propagate individual realizations"
  below). So it is a model change -- the experts become blocks of one GP,
  trained as such -- not a new way to draw the realizations of the
  present one.
- **Two corrections to the sequence as proposed.** (1) A block
  conditioned on the earlier draws and on its *own* sites only moves
  information forward: an earlier block never sees a later one's data a
  metre past its edge, and every edge is informed from whichever side the
  order put first -- a seam. Sequential simulation avoids it by
  conditioning each node on all the data in its neighbourhood, whatever
  the path; in blocks, `f_j` given the earlier neighbours' drawn values
  *and* every neighbour's sites. (2) The non-neighbours are conditionally
  independent given the blocks between them (the screening effect), not
  independent; where the range spans several expert widths the
  neighbourhood does not screen and the long lags come out too smooth, as
  sequential simulation does with too small a search -- a coarse-to-fine
  order, or the long structure in a single-expert node of its own.
- **An exact version, independent of the order.** The pathwise draw from
  the joint posterior is `f_0 + K_xZ (K_ZZ + D)^-1 (y_sites - f_0(Z) -
  eps)`, `Z` every expert's inducing points. Solving that system an expert
  block at a time -- each block's Cholesky already cached, the kernel to
  its neighbours computed on the fly -- is block Gauss-Seidel, and one
  sweep in a random order is exactly the proposed sequence with the prior
  kernel. More sweeps, or conjugate gradients with the experts as a block
  preconditioner, converge to the exact joint draw; what is stored is one
  weight vector per expert and realization (`[m, S]`), and any location
  is `phi(x) w + sum_near k(x, Z_j) v_j` -- batch-invariant, the features'
  residual included. One sweep then serves as the cheap approximation,
  several as the exact answer.
- **Memory is not the argument.** Item 45's realization memory is how the
  blend is written (`_shared_prior_simulate` builds `[rows, features]`
  per expert, 32 branches at once), not blending: the weight form above
  with the experts' terms accumulated one at a time -- already "kept in
  reserve" under the batched experts -- would save most of it with no
  change of model; the rescale's per-row norm over the features is what
  forces the large arrays and would have to be had another way.
- **What the model change would buy.** No blend: the 0.93 variance ratio
  the experts' disagreeing means leave goes, and with it the rescale. The
  bound splits by expert under a local (Vecchia) data term, each row
  depending on its block and the neighbours only, so Proposition 1 holds
  and `train_by_expert`'s expert sampling is unbiased, which the blend
  prevents today. And conjugacy returns: the conjugate trainer (section
  2) is single-expert because the blend is not conjugate, and block site
  updates become possible.
- **What it costs.** A new model: an option or a new submodule, every
  expert retrained, old saves keeping the blend -- **if the gains are
  significant, worth a breaking change or a submodule of its own** (the
  user, 2026-10-26). The inducing sets a partition rather than overlapping
  (`grid_experts`, `experts(overlap=)`). Training needs every data point's
  variance under the joint posterior, which must be local (the
  neighbourhood's sites) or it is the whole M-point problem. In 3-D,
  conditioning on whole neighbouring experts is about `(27 m)^3` a block,
  worse than the blend, unless the conditioning sets are cut to the
  nearest points. The moment path the deep networks train on needs the
  same local treatment, and the order (random, or maxmin) is a new source
  of approximation error in the one-sweep version.
- **Prior art** (cited from memory, to be checked): Vecchia (1988);
  Guinness (2018, Technometrics), grouping and orderings for Vecchia, a
  random or maxmin order beating a coordinate one; Katzfuss & Guinness
  (2021, Statistical Science), the general Vecchia framework; Dimitrakopoulos
  & Luo (2004, Mathematical Geology), generalized sequential Gaussian
  simulation of groups of nodes; Wu, Pleiss & Cunningham (2022), the
  variational nearest-neighbour GP, the same idea inside a sparse
  variational GP; Wilson et al. (2020, 2021), pathwise conditioning; Lin
  et al. (NeurIPS 2023), pathwise posterior samples by iterative solvers.
- **Gates, if pursued**, on Walker in 2-D first: the blend; one block
  sweep in a random order; k sweeps and CG; the exact joint GP on the
  same sites -- every arm but the blend retrained under the joint model.
  The realizations' variance against the reported one, seams at the block
  edges, the long-lag variogram against the exhaustive field, held-out
  rmse/CRPS/goodness, memory and time per realization. It goes on only if
  it beats the blend on scores or memory at equal time.

**S–M — Network presets** (requested 2026-10-14). Common patterns that
today take several nodes wired by hand, offered as one:
- `ReduceSum(parent)`: the sum of a node's outputs into one, which today
  is a `SelectInput` per column and an `Add` over them (the user's name for
  the selector was `SelectVariable`; the node is `SelectInput`).
- A linear combination with spatially varying weights: the outputs of
  one node weighted, location by location, by the outputs of a weights
  node and summed -- `ReduceSum(Multiply(values, weights))`, the GP-based
  analogue of geographically weighted regression, where
  `LinearCombination` has constant weights.
The GeoScape network that found 0.9.1's `Concatenate` bug built exactly
the second by hand (`Multiply`, eight `SelectInput`s, two `Add`s), so the
hand-wired form is error-prone as well as long. Decide first whether a
preset is a **function that returns ordinary nodes** -- nothing new in a
save, which replays the underlying nodes, and nothing new to keep
importable forever -- or a **class** (one node in the diagram and the
catalogue, a cheaper single sum instead of k selections, but a new dotted
path saves will record, a `_catalogue` declaration and a stand-in in
`test_catalogue.py`, and its own `propagates_inducing` and `_GAUSSIAN`:
a sum of Gaussian outputs is Gaussian, a product is not, so the weighted
form trains on realizations like any `Multiply` leaf). Either way the
catalogue must offer it, since GeoScape builds networks from it. Gate: the
preset gives the hand-wired network's bound and prediction to the bit,
and trains under `train_svi` and `train_by_expert`. The weighted form
may be better served by the Gaussian node of the next item, which would
make that half of this preset unnecessary.

**S — Independent trees left unnamed draw the same realizations** (found
2026-10-30, building `TaperedGP`; measured, and open). `_set_name` numbers a node
among the nodes connected to it, so leaves on separate roots both come out
`BasicGP_1`, and `_node_seed` keys the draws by name: the 0.6.10 fix for
identical normals between GP nodes does not reach across trees. Measured on two
untrained `BasicGP` leaves on two `BasicInput` roots with the same inducing
points, two variables, 200 realizations: the correlation between the variables'
realizations at a location is 1.000 at every location unnamed, and 0.043 (median)
named apart. Training the two trees apart is not measured. Moments and the bound
are untouched; anything reading realizations jointly across such variables is
wrong, a joint grade-tonnage or `derive` above all. GeoScape passes every node a
name, so its Runs are not affected as long as those differ. `TaperedGP` settles
its own names when it joins trees (`_settle_names`). The fix for
`VGPNetwork` is the same call across the leaves, but renaming changes a saved
model's realizations on reload, against the reproducibility contract, so it needs
a decision: new models only, through a class default as `propagation` did.

**S — Check `AdditiveGP`'s realizations** (noticed 2026-10-30, building
`TaperedGP`; found by reading the code, not measured). `BasicGP._feature_kernel`
returns True for it: it has no override and inherits
`_CONDITIONED_SECOND_MOMENT = True`, where `MultiStructureGP` sets the flag False,
and the comment on `_feature_kernel` says both keep the gap. But `_feature_basis`
draws frequencies `[J, d]` and `_features` projects all `d` dimensions at once,
random features of a product of the dimensions' kernels, where
`AdditiveGP.covariance_matrix` is their mean. So at a certain input its
realizations' residual should carry the product kernel's correlation, with the
right variance at each point. To do: compare the sample covariance of its
realizations at nearby pairs with `k12 - k1 (K+D)^-1 k2` (the same comparison
`test_pair_covariance.py` makes for a `BasicGP`), and check `_pathwise_simulate`
at an uncertain input. If it is wrong, either `_feature_kernel` returns False, so
it keeps the gap as `MultiStructureGP` does, or the features are drawn per
dimension. `TaperedGP` refuses it as a taper meanwhile (`_taper_refusal`), which
can go once this is settled.

**S — The conditioned quadrature takes a wide `Linear`'s columns as independent**
(found 2026-10-31 by the research project; the code read here, the error not
measured here). `_conditioned_sums` and `_at_input_nodes` place a location's
quadrature nodes at `mean + sd * nodes`, one standard deviation per parent column
and independent nodes (`_input_rule`, `_input_nodes`). A `Linear` wider than what
it reads has correlated columns, so the input's spread is put in directions it
does not have. The project reports a kernel error of 0.039 at width 4 against
0.003 at width 0. This is released code and not the taper's: it should reach any
GP node on a wide `Linear` over an uncertain input. To do: measure a plain GP,
wide `Linear`, GP network against Monte Carlo over the input's full covariance,
then decide whether the node takes the columns' covariance (the `Linear` knows
its weights and what it reads) or a wide `Linear` warns.

**L — Deep moments without the quadrature** (agreed 2026-11-02 from the
research project's report on the expected kernel's closure, `kq_closure.tex`;
**being built** on branch `deep-moments`). `GPOptions(deep_moments=)` chooses
how a GP node whose own inducing points are uncertain takes its variance:
`"quadrature"` (what a save made before keeps), `"hinge"` (the default for a
new model), `"blur"` and `"bound"`, the last three without nodes. The end is
fast cross-validation, which needs a smooth objective; this step is fast
training and prediction. `"bound"` was to be the default and was measured
out of it (2026-11-02, chapter 5's model, 16 experts of 100 points, a
training step on the GPU / on the CPU with XLA): the Gaussian kernel 1307 /
2042 ms under the quadrature, 526 ms under the hinge, 477 / 249 under the
blur, **2816 / 3946 under the bound**; the Matérn 3/2 3655 / 5899 under the
quadrature, 1454 and 1285 under the hinge and the blur, over 290 s / 51 s
under the bound. Its exact `alphaᵀ Psi alpha` is elementwise over every pair
of inducing points, for every pair of the kernel's components (36 for the
Matérn family), at every location of every expert; XLA's fusion on the CPU
does not save it, while the quadrature's `m²` work is matrix products. The
hinge's objective is the closest to the quadrature's (−3903 against −3913
after 21 steps; the blur −3933, the bound −3998). The realizations follow
the form: every draw takes the residual the form reports. Drawing the
hinge's with the quadrature's per-draw clamp was the plan and was measured
out of it -- on `test_full_variance.py`'s GP on a GP the ensemble carried
1.165 of the hinge's reported variance (the first-order kernel overstates
`k B kᵀ` for draws far out); on the residual it reports, 0.93. The blur's
ensemble carries 1.02 of its variance, the bound's 0.88 (0.95 with the
coordinates read beside the GP): the bound's spread holds the inducing
points' own spread given the input, `alphaᵀ R_n alpha`, which a draw at the
conditioned kernel leaves out. The manual's three deep chapters, run under
each form with the Gaussian kernel at their deep nodes (GPU, chapter time
quadrature / hinge / blur / bound): chapter 5 234 / 133 / 124 / 368 s,
positive everywhere under all four; chapter 16 480 / 381 / 371 / 654 s,
metals within 0.01 of each other, the bound's rock a little behind (mean
kappa 0.39 against 0.41-0.42); **chapter 17 reverses under the bound** --
its walk trained to a reach of 0.02 (0.14-0.16 under the quadrature and the
hinge, 0.11 under the blur), the deep map turned into the stationary one and
scored below it (IoU 0.742 against 0.769, two bodies), the deep objective
ending at −105.4 against the stationary −94.9. The bound charges for
uncertain inducing inputs, through its majorant divergence and the
`alphaᵀ R_n alpha` in its spread, and a walk is exactly what makes them
uncertain, so before the bound serves cross-validation on a walk this needs
an answer. Under the blur the deep objective also ends below the
stationary one (−103.8), its maps still ahead. The manual runs the hinge.
What it leaves open, each its own item when taken up:
- *A cheaper exact spread for the bound.* Dimensions certain everywhere (a
  `Concatenate`'s coordinates) factor out of the pairs as plain kernel
  values, halving the work on chapter 5's model; fewer Matérn components
  (the item "Fewer components for the Matérn kernels' expected kernel") cut
  the 36 pairs. Neither brings it under the hinge.
- *Minorants for the other likelihoods.* `"bound"` is a lower bound on the
  evidence only under a Gaussian likelihood, two layers and one expert;
  elsewhere the likelihood reads the moments through its Gaussian stand-in
  and the objective is a closure. The report's minorants keep it a bound: a
  logistic Bernoulli with curvature 1/4, a Student-t with `(nu + 1) / (nu
  s²)`, a Laplace through Jensen's `E|y - f| <= sqrt((y - mu)² + v)`; a probit
  with curvature 1 is unconfirmed, and a Poisson with a log link has none. A
  method on each likelihood beside its quadrature, so a model chooses.
- *Fast cross-validation*, the end goal: it needs second derivatives of the
  bound, which the compiled Ψ's custom gradient gives (checked against
  differences of the gradient).
- *`MultiStructureGP`* keeps its own rule (`_CONDITIONED_SECOND_MOMENT =
  False`, the plain `1 - k (K + D)^-1 kᵀ` at an uncertain input), so its
  variance takes no part of the spread over its inducing points at all.
- *`AdditiveGP`* averages its kernels over the dimensions, where the
  conditioned quadrature and the closed forms multiply them: under the
  quadrature it is read as the product kernel, which is not its kernel (the
  item "Check `AdditiveGP`'s realizations" above is the same mismatch).
  Refused under `"hinge"` and `"blur"`, the quadrature with a warning under
  `"bound"`. Not used in practice: internal in the catalogue since 0.10.0,
  a candidate for deprecation.
- *The rational quadratic under `"bound"`* takes the quadrature, with a
  warning: its 48 components and constant part make 1176 pairs.
- *Several outputs* share the input and the inducing points, so their
  spreads are correlated; the forms keep each output's alone, as the
  covariance between outputs is kept nowhere (see "Covariance between a
  node's outputs" above).

**L — Spatially varying weights as a Gaussian node (the offset taper)**
(requested 2026-10-26, planned 2026-10-30, from the research project
`OneDrive\Claude\Research\Deep taper\`, report `report/main.pdf`, run against
0.9.8 at `f83603e`).
**State, 2026-10-31**: stages 1-4 are built on `tapered-gp`, not committed and
in no release. The fold gate passes only at its minimum as first specified.
The research project's follow-up (under stage 4, from its `report/geoml_note.pdf`)
found better forms that have not been run on the node, and that contradict part
of the design below; the statements it overturned are corrected where they
stand. Open for the user: whether stage 5 is held, and whether stage 6's "no
worse" allows a margin.
Today a combination whose weights vary in space is a `Multiply` of a
weights node and the values -- a product, not Gaussian, so a leaf like it
trains on `training_samples` realizations, passes no inducing points on,
and nothing can sit on it.
- **The model**: for output m, `z_m = sum_j a_mj g_mj f_j`, with shared
  basis GPs `f_j` (covariance `k_j`), an LMC matrix `a`, and an
  *offset-normalized* weight `g_mj = (sqrt(kappa_j) xi_j + w_mj) / d_mj`
  -- a random global intercept `xi_j` shared by the outputs of basis j, the
  weight GP `w_mj` (mean `mu`, covariance `C`), `d_mj` making the weight's
  second moment one everywhere. Its covariance is closed:
  `sum_j a_mj a_m'j k_j T_j`, `T_j = (kappa_j + mu_mj mu_m'j + [m = m']
  C_mj) / (d_mj d_m'j)` (the report's Proposition 7, checked against 1e6
  Monte Carlo draws, PSD in 100 random configurations), and with `w = 0`
  or `kappa -> infinity` it is exactly the LMC. Proposition 8 builds a
  Gaussian process with that covariance from ordinary GPs, so the node is
  a GP node: inducing points pass through, nodes stack on it, and no
  Monte Carlo is needed. It matches the product model's first two moments
  jointly, so its mean is the product model's best linear predictor; it
  drops the product's heavy tails where the weights are uncertain
  (excess kurtosis up to 6). Uncertain weights shrink the outputs'
  cross-correlation toward `kappa / (kappa + 1)` instead of inventing it.
  **Measured 2026-10-31**: the heavy tails are the prior's (excess kurtosis
  median 3.3, up to 4.9); conditioned draws show almost none (median 0.01,
  largest 0.17), so there is little to drop once there are data. The floor is
  `E[k_s] (kappa + rho_w) / (kappa + 1)`, `rho_w` the taper's own far
  correlation, so `kappa / (kappa + 1)` needs `rho_w = 0`. The two-moment proof
  at a certain input holds.
- **Measured in the report**, on Jura's seven metals (one split, three
  seeds; the research's own exact models, so not yet under geoML's
  likelihoods and warpings):
  - **RMSE/sd** at J = 3, 5 and 7 bases: the offset taper 0.975, 0.966
    and 0.960; the exact LMC 0.991, 0.992 and 0.989; geoML's
    `Multiply` network 0.969 at J = 3 and 1.037 at J = 7 (56 GP nodes,
    worse than kriging).
  - **CRPS/sd** for the offset taper: 0.471, 0.462 and 0.460. At J = 7 it
    is the best exact model on all three scores.
  - **Against the LMC with the same bases**, both tapers win on squared
    error and CRPS, the bootstrap intervals excluding zero.
  - **Against seven independent GPs**, nothing wins decisively.
  - **The unnormalized (linear) weight** has the highest objective and the
    worst NLPD: it overfits the variance, which the normalization stops.
  - **On the folded-vein benchmark**, a bank of four fixed orientations
    selected by such weights, its range cap at 30 m, beat the stationary
    model with 5 holes (IoU 0.842 against 0.816). The offset taper scored
    0.849 and `GPWalk` 0.859. With 3 near-vertical holes only `GPWalk`
    beat the stationary model, narrowly.
- **What the report does not say.** With 3 near-vertical holes the offset
  taper is clearly worse than a stationary model under geoML's indicator
  (IoU 0.656, log score 0.458, 3.3 bodies, against 0.709, 0.270, 1), and
  worse than the selection bank (0.685, 0.388, 2), which also keeps one body
  with 5 holes. On the fold it works as a soft domain, anticorrelated across
  the boundary, not as anisotropy: free bases grew into a trend in every run.
  The selection taper stays the recorded alternative.
- **Decided 2026-10-30** (a grilling session; these replace the options
  listed here before):
  - **The node**: `TaperedGP(parent, tapers, kernel=None, ...)`, a subclass
    of `MultiStructureGP`, experimental. `tapers` is a list of nodes, one per
    output, so `size = len(tapers)`; every taper has one column per basis, so
    `J` is their common size, independent of the parent's. Each basis reads
    the whole parent, with ranges per parent dimension (the measured form).
    No per-basis rotation: a `Linear` wider than the hidden layer under it
    gives orientation through the ranges, up to their ceiling. **Measured
    2026-10-31, this reason is wrong**: the `Linear` gives reach, not
    orientation (stage 4).
  - **Outputs are independent** (the report's option A, one output at a
    time). Each output has its own taper and so its own `K`, `M` Choleskys a
    step where a `BasicGP` has one. It does not replace a `Multiply` network
    with bases shared across outputs; that needs its own evidence, since on
    Jura nothing beat independent GPs decisively.
  - **`offset`**: one per basis, shared by the outputs, trainable, starting
    at 1 within 1e-4 to 1e4. Large switches the taper off, leaving a
    `MultiStructureGP`.
  - **The taper's parents are independent of the bases**, by construction,
    so no covariance between them is needed, and none between a taper's
    columns. What a taper needs is each column's own `C(x, x')` at any two
    point sets, through a new method `_pair_covariance(x1, x2)`, recursive:
    exact for a GP on a certain input on any root (`k12 - k1 (K+D)^-1 k2`)
    and the linear operations; approximate for a GP on an uncertain input
    (`E[k(h1, h2)] - l1 (K+D)^-1 l2`, which drops the mean's spread between
    locations; on a hand-set GP on a GP it correlates 0.94 with the model's own
    realizations at nearby pairs, rms 0.030 on covariances of 0.09, Monte Carlo
    0.011). **Its diagonal is its own, never the node's variance**, which may
    read either side of it, so that `d` and `T` stay PSD with a unit diagonal
    and no jump at coincident points (the first design put the variance there;
    a test found it smaller at some locations, which could break positive
    definiteness). Refused for `Multiply`, `Exponentiation`, `GaussianMixture` and
    `GPWalk`, though the taper needs only `E[w w']`, so `Multiply` could come
    later. Experts blend as `sum_e sqrt(pi_e(x) pi_e(x')) C_e(x, x')`: PSD, and
    at a certain input its diagonal is the variance the blend reports. At an
    uncertain input its correlation against draws reads RMS 0.053-0.059, Monte
    Carlo 0.016 (2026-10-31).
  - **Certain and uncertain inputs are both supported.** The taper is a fixed
    `[n, m]` factor computed at the coordinates, whatever the input does, and
    adds no second moment of its own: at a certain input the variance is
    `1 - k (K+D)^-1 k^T` (the report's Proposition 8 and what it measured; the
    Isserlis line once written here is dropped). At an uncertain input the
    existing conditioned second moment takes `k_q = sum_j a_j^2 k_j(h_q, Z |
    h_q) o T_j(x, Z)` and `K = sum_j a_j^2 E[k_j(Z, Z)] o T_j(Z, Z)`, always by
    the 64-point quadrature, since the closed form's fixed `[m, m]` matrix
    would need one per row. `MultiStructureGP` itself keeps its flag and its
    bits. Expected about `J` times the quadrature's time, on networks where it
    is already 83% of a step; measured about `2J` (stage 5).
  - **Realizations are the full draw.** `g_mj,s = (sqrt(kappa_j) xi_js +
    w_mj,s) / d_mj`, `z~_m,s = sum_j a_j g_mj,s f_mj,s` by random features, and
    `z_m,s = z~_m,s + k K^-1 (K (alpha + R eta_s) - z~_m,s(Z))`. Exact in two
    moments at a certain input, non-Gaussian where the weights are unknown
    (the product's own tail). At an uncertain input, `k` is the draw's
    conditioned kernel times `T`, rescaled to the reported variance as
    `_pathwise_simulate` does. Seeds stay keyed by node name, so batch
    invariance holds and the draws at `Z` can be cached. `Spherical` and
    `Cubic` bases are refused, having no features, and so are tapers whose
    realizations leave a gap (`MultiStructureGP`, `AdditiveGP`, `TaperedGP`).
  - **The weights enter through the moments of `q(w)`**, as every
    expected-kernel node's input does, so the bound is not comparable with
    a `Multiply` network's. The report's tapered models reached much
    higher objectives without better maps: choose on held-out data.
- **Stages**, on a branch `tapered-gp` in its own worktree. Stages 1-4 do not
  wait for the research project; stage 5 reads its derivation of the
  uncertain-input moments:
  1. `_pair_covariance` on the nodes above, with the Expert blend: against
     draws of the posterior, PSD, the diagonal its own. (The reported variance
     on the diagonal, as first written here, is indefinite: smallest eigenvalue
     -0.11 to -0.047. `C(x, x)` sits RMS 0.07-0.08 from the reported variance,
     which moves `T` by RMS 0.03; the effect on the fold is not measured.)
  2. `TaperedGP` at a certain input and one Expert, refusing the rest with
     messages in `_check_expected_kernel`. The report's Propositions 7 and 8
     as unit tests (`verification/prop7_svc_taper.py`,
     `prop8_gaussian_construction.py`): the covariance against draws of the
     generative model, positive definiteness, and `kappa -> infinity` giving
     `MultiStructureGP`. The catalogue's declaration comes here, since the
     completeness test demands it: `parents` slots, the `len` size rule,
     `gaussian` true, `trainable` (`weights`, `ranges_{i}`, `offset`), a
     stand-in in `test_catalogue.py`, the refusals in `_conditioned_builds`.
  3. The full draw at a certain input: the ensemble's two moments against
     the node's.
  4. **Gate, the fold at a certain input**, six seeds, basin hits counted:
     the hierarchical indicator, waste a `BasicGP`, ore this node, against the
     stationary model and `GPWalk` rerun on today's code. With 5 holes it must
     beat stationary on IoU, Brier and log score on at least 5 of 6 seeds,
     which also tests the departures from the report (no rotations,
     compositional shares, range priors, geoML's variational family). 3 holes
     is recorded without blocking, and the docstring says it. **Failing stops
     the item, which moves to Settled with its numbers.**
     **Measured 2026-10-31** (`docs/benchmarks/fold_regression/fold_tapered.py`
     on `tapered-gp`): IoU, Brier, log score with 5 holes, stationary 0.790,
     0.0458, 0.172. At x/2, the node directly on the coordinates **fails**, 0 of
     6 (0.782, 0.0473, 0.174). With a `Linear(4, unit_norm=False)` as its parent,
     then read as the design's replacement for rotations, it **passes at the
     minimum**, 5 of 6 on all three together (0.864, 0.0316, 0.167); seed 3
     stands at a bound 20 nats lower at 500 steps with IoU 0.759. The log score's
     margin is small on two of the five (0.1709 and 0.1633 against 0.1715), and
     `GPWalk` stays far ahead on it (0.101). A unit-norm `Linear` fails (0.767, 1
     of 6); why was not looked into. With 3 holes the free-weight form loses on
     all six (0.681, 0.0796, 0.448, 2.2 bodies, against 0.701, 0.0676, 0.306, 1),
     at a higher bound. The `GPWalk` rows are the research project's at
     `fcd2a02`, this branch's base, not rerun here. The node matches the research
     prototype to the third decimal of the bound over 100 steps.
     **Why the margin is thin, off the gate's maps**: the free-weight form's
     good seeds halve the area wrong at p = 0.5 (3.2% of the section against
     6.0%) but not the area confidently wrong, error above 0.9 (2.1% against
     1.8%), with 2.9% of the section undecided against 16.7%. With 3 holes the
     area wrong is the same (9.0% against 8.8%) and the area confidently wrong
     nearly double (5.8% against 3.2%): the node cuts the vein at the gap between
     holes and is sure of the cut.
     **The research project's follow-up, 2026-10-31**
     (`experiments/tg_fold.py`, its `report/geoml_note.pdf` and `main.pdf`), on
     its prototype, 6 seeds unless stated, the same section, 500 steps. IoU,
     Brier, log score, seeds beating stationary:
     - **Long scale, no `Linear`** (the coordinates at x/20 instead of x/2):
       0.884, 0.027, 0.148, 6 of 6, one body, 0.26 s/step. So the node on the
       coordinates can beat stationary, and the sentence saying otherwise in its
       docstring on `tapered-gp` (written from the x/2 run alone) is false and
       has to go, with "needs a parent with directions of its own". All four
       bases of the `Linear` form share one azimuth (145-149 degrees), main ranges
       97-146 m by 23-26 m on a 16 by 20 m section, and the taper makes the
       structure: ore-waste correlation at 4-8 m falls from 0.83 to -0.04 with
       it. The x/20 run changes the start and the ceiling of the ranges together,
       so it does not say which matters, nor whether that is why the unit-norm
       `Linear`, which cannot shrink the coordinates, fails.
     - **`Linear(4)` with the offset fixed at 1**: 0.889, 0.025, 0.124, 6 of 6,
       seed 3's poor start gone, 0.23 s/step. Trained, the main basis's offset
       falls to about 0.09. `parameters["offset"].fix()` does this on the node
       with no new code. Offset fixed at 10 or 100, or started at 10: about 0.709,
       3 bodies. A small fixed offset costs too (0.01: 0.815, and 0.566 with 3
       holes). Per-column tapers also clear seed 3 (0.890, 0.025, 0.143, 6 of 6) at
       twice the time. Fixed offset with long scale was not tried together.
     - **With the taper off** (`Linear(4)`, offset fixed at 1e4; 3 seeds): 0.697,
       0.073, 0.298, 3 bodies.
     - **Steps**: at 2000, seed 1, stationary reaches 0.820, log 0.141 (3 holes
       0.730, 0.290); `Linear(4)` 0.890, 0.117 (0.703, 0.472); offset fixed at 1
       0.888, 0.099 (0.706, 0.452). Seed 3 recovers to 0.838 with one body, so its
       poor start is partly slow convergence. One seed each, and `GPWalk` was not
       rerun at 2000. The gate does not state a step count; it should.
     - **3 holes**: no setting tried keeps stationary's scores along with the
       5-hole gain. Best with the gain: per-column 0.699, log 0.449; offset fixed
       at 1 0.685, 0.424. `GPWalk` alone wins here (0.733, 0.054, 0.239).
     - **The bound does not pick the form**, at 5 holes: mean bound -66 for long
       scale (log 0.148), -80 for the fixed offset (0.124), -81 for the `Linear`
       with a trained offset (0.167), -93 for `GPWalk` (0.101), -104 for
       stationary (0.172).
     - **Time**, one job at a time, 12 threads, tracing excluded, 5 holes, in
       s/step: stationary 0.089, `Linear(4)` 0.239, long scale 0.263, fixed offset
       0.230, per-column 0.514, `GPWalk` 1.721.
     **Still to do at this stage**: run long scale, the fixed offset, and both
     together on the node itself, since only the two forms above were compared
     with it. These forms were chosen among about thirty on this one section and
     these seeds, and a seed only changes the start, so rerunning the same gate
     cannot confirm them: state the step count and add a hole layout that no
     form was chosen on. Then correct the docstring on `tapered-gp`, including
     its "settle at a poorer fit". Whether long scale can become a default is
     open: x/20 is a choice for this section's size.
  5. Uncertain inputs: `_conditioned_sums` and the pathwise draw generalized
     to several structures with the taper factor; the moments and the
     ensemble's variance against Monte Carlo, the ensemble within 10% before
     the rescale is relied on. **Gate, the fold** with GP, `Linear`, this
     node, against stage 4 and `GPWalk`, with time per step and the prediction
     cost. Failing keeps the node at certain inputs, refusing the rest.
     **Measured 2026-10-31, on the research prototype, not built here, and the
     experiment was cancelled partway** (9 by 11 inducing grid, 4 of the planned
     six seeds; stationary 0.760, the design 0.853): GP, node 0.747, 1 of 4; GP,
     `Linear(4)`, node 0.620, 0 of 4, 3.75 bodies. Offsets stay at 2.5-31, and in
     the two runs diagnosed the correlation is unchanged by the taper to 0.01:
     the taper is off and the hidden GP makes the domain. A stationary reader on
     the same layers also scores only 0.70 and 0.63, so the deep network may be
     unconverged at 500 steps; this says the gate would fail, not that the node
     cannot work. Cost about `2J`: 5.2 and 11.5 s/step against 0.70 and 1.43 with
     a stationary reader. **The condition "within 10% before the rescale" cannot
     pass**: the unscaled ratio has median 1.65 (1.20-2.20), 1.77 at width 4, none
     within 10%, because the moments are of the mixture at a fixed `K`, not of
     the network; compare the ensemble with the network, or drop the condition.
     The moment formula itself is right (variance to 1e-15), the clamp harmless
     (3.5% of nodes, mean excess 0.0026). A wider `Linear` also meets the
     independent nodes of the item "The conditioned quadrature takes a wide
     `Linear`'s columns as independent". Both the research note and the reading
     here recommend holding this stage until a use appears that a stationary reader or
     `GPWalk` cannot serve; that reverses the decision that uncertain inputs must
     be supported, so **it is the user's**, not settled.
  6. **Gate, Jura** at J = 3, 5 and 7, independent outputs, geoML's warping,
     against plain independent GPs by paired bootstrap on the validation set:
     CRPS and NLPD no worse.
     **Measured 2026-10-31 on the research prototype, six seeds, 500 steps: it
     would fail on CRPS as written.** CRPS/sd against 0.462: J = 3 +0.0019
     [+0.0006, +0.0033], J = 5 +0.0021 [-0.0000, +0.0042], J = 7 +0.0031 [+0.0004,
     +0.0057]; NLPD indistinguishable; 0.28-0.45 s/step against 0.09. At most
     0.7%, and the taper is not the cause: with it off the scores agree to 0.0002.
     So the node brings nothing on Jura. **Open for the user**: whether "no worse"
     allows a margin. Recommended: no, a margin chosen after the result proving
     little; record it as no gain on Jura and describe the node as a soft
     domain, for data that constrain the domain.
  7. Experts: dense, then the slots path, after checking how `train_by_expert`
     treats a taper on a second root. GeoScape trains by Expert wherever a Run
     has Experts, so this is what lets it offer the node.
  8. The skill, the changelog, a section in the manual if a gate earned one.

**What the gridded-GP research gives geoML as it stands** (assessed
2026-10-14 from the research project `OneDrive\Claude\Research\Kronecker
GP\`, report `report/main.pdf`, which compared against `master` at 0.9.0).
The project built a node whose inducing values sit on a regular grid --
Kronecker prior written as a positive mixture of Gaussians, cubic
interpolation to the data, one Gaussian site per datum, CG and Lanczos in
place of dense algebra -- and measured it against an exact GP and SGPR,
not against geoML's experts. What carries over:
- **S — Correct the kernel-mixture error figures** (**done 2026-10-16**,
  0.9.3: recomputed over 2.6 million distances, 5.32e-4, 1.03e-5 and
  9.15e-6, the comment, the design record and `CLAUDE.md` corrected and
  the 0.9.0 figures noted in the 0.9.3 changelog). The comment above
  `_KERNEL_MIXTURES` (`latent/network.py`), `docs/expected-kernel.md` and
  the 0.9.0 changelog give the largest error over [0, 6] ranges as 5.1e-4,
  9.8e-6 and 9.1e-6 (exponential, Matern32, Matern52); those are maxima
  over `kernel_mixtures.py`'s own distances. Over 2.4 million distances the
  supremum is 5.32e-4, 1.03e-5 and 9.15e-6 (the project's
  `verification/matern_mixture.py`, which also confirms the tables are the
  ones in the code, to the bit). The bound is per entry: as a matrix the
  mixture differs from the closed form by about a nugget in 2-D and one to
  two orders more in 3-D (its E1), which matters only if the mixtures are
  ever used where the inputs are certain; today they serve the expected
  kernel at uncertain inputs alone.
- **The realizations' missing residual has a second consumer.** A
  `BasicGP` draw is `mu + cov_cross·chol_r·normals` and carries only the
  variance the inducing points explain -- known, and the subject of the
  "Pathwise conditioning" research. Anything that reads realizations
  through a GP node (the item below, a gridded child) inherits the
  under-dispersion, so that fix comes before either.
- **Not a replacement for the experts.** The gridded node is the exact GP
  of its own interpolated model (posterior to 7e-12 standard deviations)
  and on Walker Lake matches the exact GP (rmse 5.137 against 5.136),
  1.5-6% ahead of SGPR at equal time -- but it has **no time advantage over
  an exact dense GP below about 1.4e4 data in 2-D and none was reached in
  3-D** (1.9-28x slower at 1e4), at N = 17 600 the exact GP was 9x faster,
  and its advantage is memory alone (from about 3e4 data on this machine).
  A 1% covariance error needs 15 grid nodes per range for the Matern 3/2,
  2-6x more for predictions within 1% of the exact GP's, and the
  exponential kernel cannot be gridded at any practical spacing -- in 3-D
  that is the cost. It stays a candidate for one case: a dense 2-D (or
  thin) field held in memory, the "gridded tree for geophysics" above.
- **Consistent with decisions already taken, nothing to change:** a
  coregionalized model did not help held-out Jura (equal on one triple,
  worse on the other, with 132-154 nats more training evidence; the
  isotopic case, where cokriging is known to gain little), as
  `LinearCombination(per_component=True)` measured neutral; site updates
  (CVI) work there because a single grid has no expert blend, the reason
  the natural-gradient step for the experts was rejected (below); and a
  hidden node trained well only from an informative start (its E11), as
  chapter 17's deep model needed reseeding -- an observation, not a
  measurement on geoML.

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

**Want (1) measured 2026-10-16** (prototype and every run in
`docs/benchmarks/doubly_stochastic/`, `run_all.sh` remaking them, results
and figures in `docs/benchmarks/figures/doubly_stochastic/`; the folded
section of `docs/benchmarks/expected_kernel.py`, gate 2's two
deep networks, Gaussian likelihood, 2000 Adam iterations, three seeds,
scored a new hole as there). The leaf reads S draws of its parent instead
of the expected kernel, each point from the parent's marginal, the last
layer's expectation kept closed. **The leaf's inducing inputs are
conditioned on each draw** through the chain's covariance (`Z | h`: mean
moved by `c/v (h - mu)`, variance cut by `c²/v`), so the model is the one
the joint rule trains: averaged over 2e4 draws the conditioned kernel is
the joint expected kernel to Monte Carlo error (0.004, against 0.42 from
the uncoupled one), expert by expert too. Leaving them independent of the
draw -- Salimbeni & Deisenroth as written, whose inducing inputs are free
-- scored about a nat a hole worse on both networks.

- **Why the analytic path did not match the draws: the second moment, left
  out on purpose.** `_takes_second_moment` completes the moments only where
  the inducing points are certain, and a GP reading a GP has uncertain ones
  (0.9.0's second moment in `BasicGP`: deep networks "stay out"). Trained on the moment bound, the moment path reports 10-14% of the
  model's own predictive variance on the GP on a GP and 27-32% beside the
  coordinates; nearly all the rest is the spread of the leaf's mean over
  the hidden layer. A Gaussian with the exact two moments scores within 0.5
  a hole of the full mixture on all 30 trained models: the variance is
  everything, the shape nothing. With a Gaussian likelihood and an affine
  warping the expected log-likelihood depends on the two moments alone, so
  **the analytic bound with the full second moment is the doubly
  stochastic bound at infinite S**.
- **Training on the corrected bound** (new-hole log score, mean of three
  seeds; the flat GP -12.2):

  | Training | GP on a GP | Beside the coordinates | ms a step |
  |---|---|---|---|
  | moment bound (today) | -9.97 | -10.27 | 7.4 / 8.2 |
  | 64 draws a step | -8.72 | -3.73 | 33 / 41 |
  | second moment, 5x5 Hermite grid | -8.87 | -8.29 | 21 / 24 |
  | 200 steps of 64 draws, then the grid | -8.81 | **-3.71** | 33-41, then 21-24 |
  | one draw a step, 10 000 steps | -10.53 | -8.98 | 7.1 / 7.7 |

  Beside the coordinates the 64-draw models recover the fold (field CRPS
  against the truth 0.16, today's 0.37); the GP on a GP recovers it under
  no training. The moment bound had converged: 10 000 steps move no seed
  by more than 0.2.
- **Rechecked 2026-10-30 on 0.9.12: the 64-draw row beside the coordinates
  does not reproduce.** The same seeds (2026, 11, 12) score -8.60, -2.17,
  -8.32 against the table's -3.91, -2.20, -5.07: mean -6.36, not -3.73. The
  package on those seeds scores -1.90, -8.51, -1.97, where 0.9.4's step 3
  recorded -2.04 for seed 11. So this network's scores are bimodal, about
  -2 or about -8, and a seed can change side between versions; what moves
  it is not established. On six seeds the package reaches the good side on
  three (mean -5.22), 64 draws on one of five (-7.32). A mean over three
  seeds here mostly counts those hits, and "the grid starts in a worse
  basin" below describes the 5x5 grid of 2026-10-16, not today's rule.
- **The grid is correct but starts in a worse basin.** A Gauss-Hermite
  grid over the hidden outputs, the inducing inputs conditioned at each
  node, matches 4000 draws to a standard error at 15 nodes a dimension, the
  5x5 grid within 0.5-0.7 of a data term of -13. From scratch it trains to
  -8.3 beside the coordinates (15x15 the same, so not its bias); continued
  on 64 draws it stays there; the 64-draw model continued on the grid
  improves to -1.5, and the grid rates it higher (bound -84 against its own
  -109). So 200 sampled steps, then the grid, reproduce the 64-draw runs
  seed by seed at 0.6 of the time; a one-draw warm-up does not (-9.2,
  -10.7). Why light noise picks the basin and none or heavy noise do not
  is open -- though the noise must be fresh: the same draws kept for the
  whole run land where the grid did (-9.99 at 16 draws, -7.85 at 64; fresh
  training draws are the default since 0.9.3).
- **One draw a step** costs a moment step and is unbiased, but at a
  constant learning rate its bound stops rising early (table). A decaying
  rate is untried. One hidden function shared by every point was measured
  at 16 and 64 draws (2026-10-30, "A GP node reading whole draws of its
  parent" below); at one draw it is untried too.
- **Reading a moment-trained model by draws makes it worse** (settled
  below), so the prediction-only route planned above is dropped.
- **Realizations**: today's leaf realizations cover the true field's 90%
  band at 32-45% of the grid whatever the training; drawn through the
  hidden realization with the variance the inducing points leave added as
  independent noise, 69-87% at 16 draws a step.
- **Cost** (the GP on a GP at 2 400 and 9 600 points, 100 and 300 inducing
  points): each draw a step adds 18-27% of a moment step's time and holds
  7-11 float64 `[n, m, d]` tensors for the gradient (35 MB at 2 400 x 100,
  337 MB at 9 600 x 300) -- 16 draws are 4-5x the time and 1.4-3.4x the
  memory, and 64 at 9 600 x 300 ran out of 15 GB. Prediction alike: a
  256-draw mixture 1 s against the moments' 0.05 s at 2 400 points. A grid
  of Q nodes costs Q draws, and Q is k to the power of the hidden width, so
  past two or three hidden outputs a Sobol set (the 64 nodes `_input_nodes`
  takes) is the rule. Draws chunked under the gradient would hold memory at
  one chunk; not built.
- **Experts**: the coupling carries -- each expert draws its own parent
  expert's chain and conditions its own inducing inputs, blended draw by
  draw by the package's weights; every expert's gate within 1.2e-3. The
  gain did not: on 60 inducing points beside the coordinates, one expert
  scores -6.62 on the moment bound (-2.3 to -9.2) and -5.34 on 64 draws
  (-4.7 to -6.1); three experts of 24-28, -4.23 (-4.1 to -4.3) and -4.95,
  the warm-up then the grid -5.08. With three experts the moment path
  already reports 83% of the exact variance (hidden sd 0.51 at the new
  holes), with one of the same 60 points 13% (0.71). The dense path reads
  every expert at every point: 64 draws 77 ms a step against 15. Slots
  untried.
- **Next: the plan agreed 2026-10-18, being executed.** The goal: a GP
  node reading another node's output -- a deep layer, a `GPWalk` -- reports
  the full variance, and **its realizations carry that same variance**:
  the per-point variance over the ensemble equals `latent_variance` to
  Monte Carlo error, with the spatial structure of draws of the deep
  model. Want (2) is untouched.
  1. **The second moment at uncertain inducing inputs.**
     `_takes_second_moment` stops requiring certain inducing points: the
     moments take a quadrature over each point's input `h ~ N(mean, var)`
     per expert chain, the inducing inputs conditioned on each node `h_q`
     (`Z | h_q`: mean `u + (c/v)(h_q - mean)`, variance `u_var - c²/v`),
     the cross-covariance `k(h_q, Z | h_q)` averaged over `Z`'s remaining
     spread, and the mixture's moments (mean of `m_q`, mean of the node
     variances plus the variance of `m_q`). A 5x5 Gauss-Hermite grid up to
     two hidden dimensions, the 64 symmetric Sobol nodes of `_input_nodes`
     from three. The nodes run in a `while_loop` under a gradient of their
     own that accumulates, as `_mixture_sum` does, so one node's arrays are
     alive at a time -- naively 64x the cross-kernel, ~1 GB a reader on the
     GeoScape walk network. Gates: (1a) mean and variance against 4000
     draws of the deep model on the folded section's two networks, one
     and three experts, Gaussian and Matern kernels, to Monte Carlo error;
     (1b) no chain covariance gives today's second moment exactly and no
     input variance the plain path; (1c) peak memory and epoch time on the
     GeoScape network against 3.7 GB and 153 s.
  2. **Realizations with the same variance.** At an uncertain input,
     realization `s` reads the parent's realization `s` (paired with the
     parent's own, as the walk rides the field's), plus the variance the
     parent's realizations leave out as independent noise per point; the
     inducing inputs conditioned on it; `k(h_s, Z | h_s)(alpha + R eta_s)
     + bias`, plus the reader's own unexplained variance as independent
     noise per point -- **at uncertain inputs only** (the user's choice;
     whether every GP node should add it is decided at step 4, once its
     size on shallow models is measured). The location-uncertainty case
     is folded in: a GP on a `GaussianInput` draws the same way and the
     separate `_input_jitter` goes (the user's choice). Gates: (2a) at
     2000 realizations the per-point variance against `latent_variance`
     within chi-square bounds, the mean against the prediction within its
     standard error, on the folded networks, a two-layer Jura or Walker
     network, chapter 16's walk and the GeoScape network; (2b) the
     realizations' variogram against draws of the deep model on the same
     pairs; (2c) the folded section's true-field 90% band covered by more
     than today's 32-45% (the prototype's 69-87%).
  3. **Training.** Arms: today's moment bound; the second moment from
     scratch; a sampled warm-up then the second moment; one draw a step
     with a decaying rate -- three seeds, on the folded section, a
     two-layer Walker network, chapter 16 and the GeoScape network; held-
     out log score, CRPS, coverage and `spread_check`, time and memory.
     Fresh training draws (0.9.3's default) bear on the warm-up.
  4. **Defaults and saves.** An option on for new models and off for a
     save made before (a class default, as `propagation`); every deep
     model's realizations change -- changelog, chapters 16 and 17 rerun,
     the skill, one full run at the end.
- **Steps 1-2 built 2026-10-19, branch `deep-second-moment`** (worktree
  geoML-second; gates in `docs/benchmarks/deep_second_moment/`, figures in
  `docs/benchmarks/figures/deep_second_moment/`). Step 1 as planned, the 64
  nodes chunked four at a time; the conditional variance clamped node by
  node, because conditioned on a node the inducing points sit closer than
  the expected kernel among them allows and `k (K + D)^-1 kᵀ` passed one
  at half the draws. Step 2 departed from the plan in three places, each
  on a measurement:
  - **Every GP node carries its residual, by random features** (the user's
    1(b)), not independent noise: Wilson et al.'s `f(x) - k K^-1 f(Z)`,
    whose correlation is the kernel's. Independent noise was the
    0.9.3-shaped nugget; the nugget in the realizations turned out to be
    mostly the model's own short-scale roughness, and more inducing points
    made it worse (lag-1 semivariance over the sill 0.23 -> 0.66 -> 0.93
    without noise), so correlated noise was not pursued. The prediction
    integrates the residual analytically from the realizations without it
    (`smooth_draws`, the user's 2(b)); the stored realizations keep it.
  - **Training on samples draws each location's marginal**
    (`marginal_draws`, the user's choice): the full variance, where the
    realizations used to carry only what the inducing points explain. The
    `GaussianMixture` regimes gate looked lost under it (seed 1234: RMSE
    0.218 against the GP's 0.192); over seven seeds the true bound finds
    the switch five times (RMSE 0.126-0.145), the explained-only bound
    none -- the fit is bimodal by its start, and 1234 is a bad one.
  - **The experts.** Each expert alone held its variance (1.00-1.03) and
    its mean; the blend did not -- a certain-input GP's realizations held
    0.51 of its variance where two experts carried weight, because the
    experts' normals, shared by index, met at unrelated inducing points
    and were independent in effect. A reader above it then read a parent
    too narrow and its mean moved (half of a GP-on-a-GP's points outside
    their interval; the user's diagnosis: experts too small to reach the
    tails before the blend, not the blending of means). Fixed in two
    steps: the blend rescaled to `sum w v` (`166ef0e`), then **every expert
    drawing from one prior draw** (the user's three steps: each expert's
    conditioned draw from its own `S`, blended, Matheron's residual added).
    The coupling `u - mu = M f(Z)`, `M K Mᵀ = S`, measured on the hidden
    layer at 1-3 shared points an expert, before any rescale (blend over
    reported where two experts carry weight; step ratio overlap/elsewhere):
    independent 0.47 / 0.98; `chol(S) L^-1` 0.84 / 1.20; `S^½ K^-½` 0.87 /
    1.23; optimal transport 0.86 / 1.17; Matheron's own `D (K+D)^-1 f(Z) -
    K (K+D)^-1 eps`, eps keyed by location, 0.70 / 1.16 (its pseudo-noise
    dominates where the data are dense, and is independent but at shared
    points); sequential conditioning on an earlier expert's values 0.72,
    with 74% of the means pulled off. **Chosen: `chol(S) L^-1`**, which is
    `k R G^-1 Phi(Z) w` in the realization and needs no inverse of K --
    the symmetric and transport maps need `K^-½`, which a `SelectInput`
    network's K (conditioned at 1e18) would not survive; the exact rescale
    closes what the coupling leaves. `G` is the Cholesky of the features'
    Gram matrix at the inducing points, so each expert's draw is exactly
    standard and its posterior exact. Experts drawn alone with one seed
    correlate at 0.72 where they meet.
  - **Gates on the folded section** (2000 realizations; ratio of their
    variance to the reported one, median): one expert, GP on a GP 1.00,
    beside the coordinates 1.00, the walk 0.99 (0.9.3: 0.59, 0.35, 0.45);
    three experts 0.93, 0.99, 0.99, means within their intervals (2-8%
    outside). The 0.93 left is the experts' mean functions disagreeing
    (their spread 28% of the variance on the GP on a GP), which a weighted
    blend of means loses by convexity and no coupling recovers. A shallow
    GP's realizations against its exact posterior covariance: variance
    1.00, correlation error 0.023 median, 0.07 at the 95th percentile (the
    Monte Carlo floor at 2000 realizations 0.022) -- the draw and the
    residual share `w`, so the features' error adds a small
    cross-covariance; 0.9.3 had 0.07 and 0.15 there. **The GeoScape
    exchange network** (32 experts, three walk readers on Matern 3/2 and
    exponential kernels, 600 conditioning points): every reader's
    realizations at 0.994-1.006 of the reported variance (before: 0.82-0.91),
    4-6.5% of the points outside the chi-square band, 5-7.5% of the means
    outside theirs; an epoch 435 s (441 before), drawing a reader's
    realizations 233 s at a GPU peak of 9.9-11 GB.
  - **The features' random numbers are drawn on Philox by name**:
    `auto_select` picks another generator under XLA, and a compiled
    prediction drew another basis (realizations apart by their own size);
    plain and compiled now agree to 1.4e-7 of the field's scale.
  - **Training** (step 3, three seeds each, `1b84ed5`). The folded
    section, new-hole log score: the second moment from scratch -2.04,
    -1.94, -2.42 beside the coordinates (the moment bound -9.6 to -11.2),
    -10.2, -7.5, -8.4 on the GP on a GP (-12.4 to -13.9). **Walker Lake**,
    the same two networks on the 470 samples (120 inducing points, one
    expert, chapter 15's warping, 2000 steps), scored against the
    exhaustive field at every sixth row and column: the second moment
    beats the moment bound on every seed and network -- RMSE 174-188 ->
    160-161, CRPS of measurement samples 92-103 -> 85, their 90% coverage
    0.96-0.99 -> 0.89-0.90, log score -6.30/-6.52 -> -6.20 -- and both
    networks converge to one model, where the moment bound leaves the GP
    on a GP far behind; its maps are scattered high-grade blobs, a
    different set in every realization, the second moment's the field's
    belt, arc and hole. 13-17x the training time (500-700 s against 40).
    Against an exhaustive truth the realizations of the *ground* cover only
    ~35%: the model puts Walker's fine texture in the noise, so the fair
    check is samples of a measurement (`predict_measurements`). **Chapter
    16** (250 steps): the rock's log score 0.3224 -> 0.3204 and Brier
    0.0971 -> 0.0966 on every seed, the metals unchanged (RMSE/sd 0.912,
    CRPS/sd 0.512), 3.7x the time (1342 s against 360); the maps alike,
    the rock's realizations equally fragmented under both bounds -- five
    latent draws of full variance on a rough Matern 3/2, a property of the
    realizations rather than of the training. **The sampled warm-up adds
    nothing** on any case (Walker within 0.003 of a log score). Reading for
    step 4: the second moment pays where a hidden layer carries the model
    and buys little where one reader sits on a walk beside a stationary
    tree -- an argument for a default that depends on the network, or for
    a cheaper quadrature, before a blanket default. The GeoScape network
    untried; step 4 not started.
  - **The quadrature made cheaper** (2026-10-20, `d6998c6`, `d6e6f19`).
    Profiled on Walker's GP on a GP: 88% of a training step is the
    conditioned quadrature's loop, 72% of that its backward pass -- the 64
    nodes four at a time, recomputed under the gradient. **Chunks sized by
    a memory budget**, `GPOptions(quadrature_memory=2e9)`: every node at
    once, differentiated plainly, where the estimated arrays fit it, else
    the largest chunk that does and never fewer than four; the realizations
    the same way. A fixed number rather than the memory free, so a model
    chunks and rounds alike in every process. A step 136 -> 60 ms on the
    GPU and 234 -> 147 on the CPU, the bound the same to 1e-9; the exchange
    network unchanged (447 s an epoch against 435, a reader's realizations
    229-234 s, 0.99-1.01 of the variance, GPU peak 11.5-12.8 GB against
    9.9-11.1). What a per-call estimate cannot see, measured on that
    network, each run out of the 24 GB device: rows hinted at the whole
    data set (a chunk of one, 510 s an epoch) -- every call now states the
    rows it carries; 32 experts' quadratures run together, each inside the
    budget -- they share it now; the kernel components' loop unrolled, its
    eight Gaussians run together -- reverted; a 6e9 budget giving 32 nodes
    beside two sibling readers -- 2e9 and the floor of four; a reader's
    realizations at 16 bytes an element -- 48, times its experts.
    `[..., None]` on the large arrays became `expand_dims`, whose gradient
    is a reshape and not a full-size `StridedSliceGrad` a chunk.
    **Gauss-Hermite up to two dimensions** (`gate_nodes.py`, against 16000
    coupled draws on the folded section's GP on a GP, median error of the
    variance): Sobol 64 2.3%, 256 ~1.0%, 1024 0.65-0.77%; Hermite 5x5 1.0%,
    7x7 matching Sobol 1024. A step at 5x5 1.5x faster than 64 Sobol nodes
    on the GPU, 2.3x on the CPU. **7x7 is the default** (the user's choice,
    2026-10-20): more accurate than the 64 Sobol points and cheaper; Sobol
    stays beyond two dimensions. **XLA on the chunk's body** 1.1x on the GPU
    at a third more memory, 1.3x on the CPU: left an experiment, off.
  - **Step 4 built 2026-10-20, released in 0.9.4.** `GPOptions(full_variance=
    True)` for a new model, the class default off for a save; off is 0.9.3
    to the bit on four models (shallow over two experts, a GP on a GP over
    two, a walk reader, a `GaussianMixture`), predictions, realizations,
    measurement samples and training log, rebuilt and reopened
    (`gate_off.py`) -- which took putting back 0.9.3's expert draw: every
    expert under the node's name and the old summation order, the new blend
    having moved a mixture's log by 6e-12. **Memory, found by chapter 17**:
    its walk reader's refinement held 37 GB per pass against 0.9.3's 10.5;
    profiled one process a setting (a second prediction in one process
    reuses the allocator's pages and reads zero), the conditioned
    quadrature costs no memory beyond 0.9.3's 7.9 GB at 20000 rows and the
    pathwise realizations 16 GB -- one realization 3-4 GB there, the
    four-a-chunk floor overruling the 2 GB budget. Each model now sizes its
    own prediction batch (`_prediction_rows`, 3375 rows on chapter 17):
    9.6 GB and no slower. **A latent bug the smaller batches exposed**: the
    four block reductions divided rows by blocks as a float and cast, and in
    a trace with the rows unknown the division became a multiplication by
    the reciprocal -- 1496 / 187 truncated to 7, chapter 17's refinement
    failing on a short last batch; integer division now. **Chapters**: 16's
    metals' goodness rose (Ni 0.81 -> 0.86, Zn 0.86 -> 0.90); 17 ran 89
    min against 48, peak 14.8 GB. With `bias=False` the stationary vein of
    17 spread fully uncertain sheets over unsampled ground (the field
    returns to zero, both log-odds on the contour level), and chapters 6, 16
    and 17 now give the indicator `bias=True` -- the class prior belongs to
    the likelihood, not to a field's mean.
  - **The resolution gap, found by chapter 13** (2026-10-21,
    `walker_gap.py`, `walker_fan.py`). Under the full variance Walker's
    shallow model drew a variogram fan above the data: about 10% with the
    spherical, 20-70% with the exponential, and the exponential's intervals
    over their promise (`nominal(0.9)` 0.847, goodness 0.95). Training and
    the moments are identical on and off; what differs is the realizations,
    which with it on draw the gap `1 - k K^-1 kᵀ` (random-feature kernels
    only; the spherical and cubic have no random features and carry the
    explained part, independent experts rescaled). At the 470 samples the
    gap was 0.67 of the reported variance on the 10-unit lattice (0.59 for
    the spherical), and 0.60 on a 5-unit lattice -- training shortened the
    range from 122 to 77 -- so not a shortage of points as usually counted:
    a sample informs the field only through the inducing values and never
    reduces the gap at its own location. Adding the samples to the lattice
    before cutting the experts: gap 0.08 at the samples, bound -3199 against
    -3292, fan on the data, `nominal(0.9)` 0.925 (exponential) and 0.909
    with RMSE 207 (spherical). The user's reading: rough kernels need
    infinite resolution, the ideal is an inducing point at every test
    location -- a lattice as fine as the data -- and where that is out of
    reach the gap is foreseeable. Released so: the manual's VGP models take
    the exponential (honesty: its realizations carry everything), chapter 13
    names the gap and the cure, spherical and cubic GP nodes warn.
- **Open: the gap where resolution is out of reach** (S-M). A block model
  of millions of cells cannot hold an inducing point per cell, so its
  realizations carry the gap of a rough kernel between lattice nodes. To
  measure: how large it is on Tom East and Assen at their usual lattices,
  and whether a likelihood-side correction (the data's own conditional
  given the inducing values) is worth its cost. Also open: random features
  for the spherical and cubic (their spectral densities are known in
  closed form, though not as mixtures of Gaussians), which would let them
  carry the gap too.

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
  rather than noisy (fixed by fresh training draws, the default since
  0.9.3).

Open: the real-deposit gate against hard domains (likely Jura) before the
node is public; Monte Carlo paths for the categorical likelihoods, so a rock-type
leaf that is not Gaussian can train on its realizations; and the gate read
from a rock-type leaf, where the indicator rule (the largest latent) and
the mixture's softmax read one latent through two different links.

**L — `LithologicalControl`: the logged rock type as an observation of the
population** (requested 2026-10-31; proposed, nothing measured yet). Reading
a rock type and a `LikelihoodMixture`'s shares off one latent node ties the
*field*, and the effect is weak. Given the share columns `g`, the rock `c` and
the population are independent:
`log p(c | g) + log Σ_k π_k(g) exp(t_k)`, with `t_k` population k's expected
log-density of the metals. The rock reaches a row's population only by moving
`g`, and three things blunt that. A row's log-odds of population is
`Δlog π + Δt`, and `Δt` likely dominates at an assayed row: in-sample agreement
with `Code_Simple` above is 0.94 in every arm, including the one that never
sees the rock. The mixture's own `amplitude` and `bias` let the shares' contact
sit away from the rock's. And one field has several read-outs: a separate
probit per column in training, `probability`, the largest column in a
realization, a scaled softmax in the shares. With the rock on every logged
interval, the rock's own call has recall 0.219 on Tom East and the shares
0.555, off the same columns.

This class ties the *label*: one likelihood over both variables, with a trained
`P(c | k)`, populations by rock categories, and per row

`log Σ_k π_k(g) · P(c | k) · exp(t_k)`

`exp(t_k)` dropping out where a row has no assay and `P(c | k)` where it has
no log. The rock then sits in the same sum as the metals, so it decides
ambiguous rows' population directly, and through them each population's noise
and warping. The rock's probability is `Σ_k E[π_k] P(c | k)` and its
realizations come from the `population` store, so there is one contact.
`amplitude` and `bias` stay and can no longer separate the two. An identity
`P(c | k)` would be hard domains, forcing the rows where the log and the assay
disagree (6% on Tom East) into the wrong population; trained, it absorbs them
and reports how often they disagree. The number of populations need not match
the number of categories.

What it asks for:
- **One likelihood reading two variables.** `_data_log_lik` gives each
  likelihood its own variable and adds the results, and a sum inside a log does
  not split that way. How a likelihood is named against two variables, and how
  `predict` writes both, is the first design question; `update`,
  `cross_validate`, `predict_measurements` and the catalogue all follow from it.
- **Contact rows.** `is_boundary` needs a term; `log π_a + log π_b`, which
  peaks at equal shares, is the natural candidate.
- **The start.** `P(c | k)` starts from the clustering's overlap with the log,
  or the population labels can switch.

**Check before building**, on Tom East's tied arm: count the rows whose
`responsibilities` change population with and without the rock. If almost none
do, the assay decides the populations, and the class would change mainly the
rock-only rows and the rock's contact, not the grades.

**Gates**, against the tied and the metals-alone arms above, out of fold by
hole: CRPS per metal; balanced accuracy of the rock on every logged interval;
and the rock's probabilities against their own frequency, since the indicator
they replace overstates its confidence (0.10.0: stated 0.821, right 0.680 on
chapter 16). The rock type is barely predictable between holes on Tom East, so
a large CRPS gain is not to be expected. The rock's scores in the tables above
were taken before 0.10.0, when `predict` read a density; on chapters 6 and 16
the fix moved under 1% of the grid's labels, and its effect on Tom East has not
been measured.

**S — Say when a heavy-tailed noise under a log-like warping has no
finite mean** (requested 2026-10-15, after a GeoScape mixture that would
not show its two populations; **the warning built 2026-10-16, 0.9.3**:
`_ContinuousLikelihood.initialize` calls `_warn_log_tail`, each family
declaring `_tail_rates`, the reach read off the data by least squares of
the logarithm on the warped values -- on Jura's vector chain a variance
pole at 1.28 against a start of 1; the safe start below stays open, its
measurement needing the Macpass record). `EpsilonInsensitive` trains its epsilon to
zero and is a Laplace; pushed back through a log-like link (`Log`,
`CenteredLogRatio`, `BoxCox` near zero) its exponential has a finite mean
only where the noise rate exceeds the spread it is scaled by, the pole the
Jura metals sat at (`2·sigma_log/c = 1` for the second moment, half that
for the first). Measured on the Macpass assays (record 6ba2eb70, a
`LikelihoodMixture` of two `EpsilonInsensitive` populations over
`CenteredLogRatio(4) → PCA(4, 3)`, 120 epochs of `train_by_expert`): the
waste population's clr spread is 1.86 on its first axis and the rate
starts at 1, so its integrated ground value read 1.67 % Zn against a
centre of 0.04 %, came down only as the rate climbed (2.6 by epoch 120),
and the ground simulations reached the data's low-Pb mode at 0.011 against
the data's 0.100. The same network with Gaussian populations: 0.040, the
measurement draws at exactly the data's 0.100, the waste population's Zn
0.20 % against 0.33 % -- at a bound 345 nats lower, as on Jura. Two
remedies, either or both:
- **A warning** at construction (or at `initialize`, where the spread is
  known) when a Laplace-type or other exponential-tailed noise sits under
  a log-like warping, naming the pole and the Gaussian alternative --
  cheap, and it reaches GeoScape through the log.
- **A start on the safe side**: `initialize` setting the noise rate from
  the spread the warping leaves (above `sigma` for a finite mean, `2·sigma`
  for a finite variance), so training never passes through the explosive
  region. Changes the starting point of every such model, so measure it on
  Jura and the Macpass record before making it the default.
Initialization of the mixture itself was ruled out on the way: its
clustering put population 0 at waste (0.04 % Zn) and population 1 at ore
(8.8 % Zn), and they stayed apart throughout.

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

**M — A sampling-based conditioned quadrature for training** (filed
2026-10-30, from the GP-NN link research project, whose stage A is the
measurement below). The conditioned quadrature's fixed nodes, 7, 49 or 64
for one, two or more hidden outputs, replaced in training by S fresh draws
per location and step: the same integrand (the inducing inputs conditioned
on each draw, the kernel averaged over what is left of their spread), the
chain kept, the model unchanged, the Gaussian expectation closed per draw.
So not the doubly stochastic route of "Propagate individual realizations",
whose fixed-input form is a different model and scored worse; this is that
item's 64-draw arm, taken at small S, as a cheaper quadrature (its reading
for step 4 asked for one). Its 16 draws were "4-5x the time" against the
moment bound, which has no second moment to pay for; the comparison here is
against the second moment.

- **Accuracy at fixed parameters** (four GP-on-a-GP models trained on the
  joint rule, Gaussian kernels, one expert: a synthetic composition at 1500
  points with 225 inducing points, Walker at 470 with 225, one and two
  hidden outputs; against 4096 draws per location, relative error of one
  gradient of the data term):

  | Case | 7 or 49 nodes | 4 draws | 16 draws |
  |---|---|---|---|
  | synthetic, one output | 0.28 | 0.09 | 0.05 |
  | synthetic, two outputs | 0.05 | 0.07 | 0.04 |
  | Walker, one output | 0.06 | 0.38 | 0.17 |
  | Walker, two outputs | 0.01 | 0.29 | 0.15 |

  The nodes' error is a bias, kept step after step; the draws' is noise,
  with no bias detected over 24 settings (z-rms 0.84-1.50). The bias is
  all in the variance, since the mean is closed: the worst test point's 90%
  interval covers 35% on the synthetic model at seven nodes (mean over the
  points 0.92), and the data term sits 62 nats (24 standard errors) from
  the draws' on the synthetic model at 49. Those Walker models fit poorly
  (the noise took 0.9 of the variance), so the child is nearly linear
  there, which is the quadrature's easy case. This does not overturn
  `gate_nodes.py`'s 7x7, which is a median over the folded section.
- **Cost** (one model's value and gradient, CPU). Run eagerly, 3.5-6.3x
  less at 16 draws and 6.0-8.4x at 4. Eager calls flatter the draws;
  traced, on the Walker model of `phase3_walker.py` (120 k-means points, two
  outputs), 88 ms a step against 20 at 4 draws, on 200 steps.
- **Training, stopped at the user's request with 48 runs** (read 2026-11-01
  under "The quadrature's rule should follow the input" below; stage B of
  that project, started
  2026-10-30, Adam from one start, three seeds, one held-out scorer for
  every arm): 1, 4, 16 draws (and 64 on the folded section) against the
  joint rule, on the folded section, Walker as in `phase3_walker.py`, the
  synthetic models, and at 10000 points with two and three hidden outputs.
  The adoption rule set before the runs: a draw count reaching the rule's
  held-out log score within one seed's standard error, at 3x less time to
  target or peak memory. Basins may well decide it rather than cost: fresh
  noise is what found the better basin with fresh training draws
  (0.9.3, `docs/benchmarks/fresh_draws.py`).
- **The change, if it passes** (S-M): a value of the quadrature's rule
  beside `"hermite"` (`_Setting.rule`, `_input_rule`), drawing in
  `_conditioned_sums` under `_training_seed`, so that chunks and replays
  stay as they are, for training only: prediction keeps the fixed nodes and
  stays deterministic. Draws per location where the nodes are per call, so
  `kernel_at`'s offset takes a location axis; the memory budget's
  `_NODE_BYTES` then counts draws as nodes. Not chunked in the prototype;
  under the gradient it must be. A class default off for a save, as
  `propagation` did. Its gates: the integrand at the rule's own nodes
  equals the moments (measured: 1e-15 of the variance, and the data term
  against `_data_log_lik` to 1e-9); a Matérn kernel's eight Gaussians and
  experts, neither tried yet.

**M — A GP node reading whole draws of its parent, with its own `K` at
each** (measured 2026-10-30, open; the prototype is
`C:\Users\Public\cr\paths\paths.py`, outside the repository, with its
results beside it). The question: the expected kernel and the moments
computed through a node's realizations instead of in closed form. The
folded section's two deep networks, a Gaussian likelihood, warping fixed,
2000 Adam steps, one expert, CPU. Each step draws S functions of the hidden
node, one per draw, at the data and at the inducing points together:

```
u_s    = alpha + R eta_s
Z_s    = K_h u_s + b
h_s(x) = k_x u_s + b + sqrt(1 - k_x K_h^-1 k_x') eps_s
```

The residual is independent of the inducing values, so a row's joint with
`Z_s` is exact with no random features; rows are independent given `u_s`,
which a factorized likelihood does not read. The leaf then takes the plain
kernel, `k_s = k(h_s(x), Z_s)`, `K_s = k(Z_s, Z_s)`, `alpha_s = L_s^-T w`,
`v_s = 1 - k_s (K_s + D)^-1 k_s'`, with `w` and `D` shared by the draws and
the leaf's KL averaged over them. The inducing inputs being the hidden
node's inducing values, this is the deep GP's own bound, and `v_s` needs no
clamp. Want (1) above draws each point from its marginal and conditions the
inducing inputs on it, with one `K`, and its bound in expectation is the
second moment's; this is a different model. One `K` read through whole
draws is settled below (rejected).

New-hole log score, a mean over the seeds shown (the flat GP -12.2; the
package read by its moments, the draws by their mixture at S = 256):

| Network | Package | `K_s`, 16 draws | `K_s`, 64 draws |
|---|---|---|---|
| GP on a GP, Gaussian leaf | -8.71 (3) | -13.03 (3) | -12.40 (3) |
| GP on a GP, Matérn 3/2 leaf | -7.86 (3) | | -12.05 (3) |
| Beside the coordinates, Gaussian leaf | -5.22 (6) | -7.28 (3) | -7.46 (3) |
| Beside the coordinates, Matérn 3/2 leaf | -5.43 (6) | -8.28 (3) | -5.79 (6) |

- **Beside the coordinates at 64 draws it ties, on the Matérn leaf.** Per
  seed -4.06, -5.51, -8.07, -6.07, -4.27, -4.59 against -5.02, -5.89,
  -5.35, -5.09, -5.02, -8.38; held-out CRPS 0.267 against 0.262, CRPS
  against the true field 0.185 against 0.173, its 90% coverage 0.91 against
  0.94, the draws' figures from per-location marginals, since they define
  no field. At three seeds it looked better; three more made it a tie. The
  Gaussian leaf's row compares six seeds with three on a bimodal network
  (see the recheck under "Propagate individual realizations"), so it is no
  verdict either way.
- **On a GP on a GP it stays near the flat GP**, on both leaves and at 64
  draws. The cause is not established. Sharing `w` across draws ties the
  inducing means together (`L_s w`) and is the first thing to test.
- **Untying the means helps by about a nat and does not close the gap**
  (2026-11-02; `C:\Users\Public\qm\neg\site`, the prototype with one change).
  Pseudo-data `y` and `D` shared by the draws, `q(u | Z_s) = N(K_s B_s y, K_s -
  K_s B_s K_s)`, the form of Ober et al. 2021: a leaf's values stay at `y` while
  their locations move, and it is a true bound. Gaussian leaf, 64 draws, seeds
  2026, 1, 2, 11, 12, 13, against the package on the same seeds:

  | Network | Package | `K_s` | Pseudo-data |
  |---|---|---|---|
  | GP on a GP | -9.02 | -12.32 | -11.23, worse on 6 of 6 |
  | Beside the coordinates | -5.22, 3 of 6 in the -2 basin | -7.66, none | -6.65, one (-3.3), worse on 4 of 6 |

  Held-out CRPS beside the coordinates 0.299 against the package's 0.269, CRPS
  against the true field 0.212 against 0.186. The fit trades the hidden layer
  for noise (0.215 against the clamp's 0.153, 0.400 against 0.279 on the GP on a
  GP), as the research report's narrow-hidden-layer argument expects. The
  per-draw collapse (its H) was not run: it needs the closed `E[k_i k_j | Z]`
  and a per-draw solve at prediction, and it differs only by a freer leaf. A
  Matérn leaf was not run on this form either; there `K_s` already ties.
- **Cost** (the Matérn leaf beside the coordinates, the median single step,
  CPU, four threads; a `train_full` called in blocks retraces inside the
  timed window, so time single steps):

  | Rows x inducing points | Package | `K_s`, 16 | `K_s`, 64 |
  |---|---|---|---|
  | 2 400 x 100 | 2.14 s, 4.3 GB | 0.29 s, 1.5 GB | 1.09 s, 2.8 GB |
  | 4 800 x 200 | 18.9 s, 4.5 GB | 1.09 s, 2.9 GB | 3.8 s, 8.0 GB |

  So 2-5x faster at 64 draws, and more memory at the larger size: the
  package is chunked to its budget and the prototype is not. Want (1)'s
  conditioned draws, also unchunked, cost 1.00 s, 4.1 GB and 3.7 s, 12.3 GB
  at 16.
- **Noise.** The bound's standard deviation at fixed parameters, beside the
  coordinates, 8.7, 3.3, 1.6 at 16, 64, 256 draws (at the package-trained
  parameters), and on the GP on a GP 35, 9.5, 8.4. A per-location variance
  from one `K`'s draws errs by 20-25%, 10-13%, 5-6% at the same counts,
  against 0.7-1% for the 7x7 grid, so reported variances would need many
  draws at prediction, or the quadrature.
- **What it would remove, if it won everywhere**: the named expected-kernel
  helpers (about 900 of `latent/network.py`'s 7120 lines), most of the 18
  `propagate` methods' 477 lines, the refusals of `Spherical`, `Cubic` and
  `Cosine` on an uncertain input, and "Covariance between a node's outputs".
  Stored realizations would still need random features for their spatial
  structure.
- **Next, in order; each stops if it fails.** (1) A reader of a `GPWalk`,
  the case GeoScape's network and chapter 17 use and where the quadrature
  is 83% of a step: the folded section, one expert, six seeds against the package,
  counting basin hits; it goes on if it ties and is at least 2x faster. (2)
  Experts. (3) A non-Gaussian likelihood: with a Gaussian one the data term
  reads two moments alone, so nothing above has tested dropping moment
  matching itself. Separately, why the GP on a GP fails. If (1) fails, this
  moves to Settled with these numbers.

**M — The quadrature's rule should follow the input** (measured 2026-11-01;
nothing in the package changed; harness and tables in
`C:\Users\Public\qm`, outside the repository). The models fall into two
regimes, and one rule serves neither well. Against a Hermite grid of 31 points
a dimension (13 in three dimensions; chapter 17's against 9, the finer one out
of memory), at the rows an expert carries:

| Model | nodes | input sd / range | nodes with `k S kᵀ` > 1 | clamp's addition / variance | spread's share | variance error, median, p95 | gradient error |
|---|---|---|---|---|---|---|---|
| GeoScape walk network | 64 | 0.02 | 0 | 0 | at most 0.007 | under 1e-5 | 0.6-1.5e-4 |
| Chapter 16 | 49 | 0.05-0.07 | 3-7% | 0 | at most 0.02 | 1e-5, 3e-5 to 2e-4 | 4e-5 to 7e-4 |
| Chapter 17 | 64 | 0.07 | 37% | 0.13 | 0.45 | 0.023, 0.31 | 0.021 |
| Folded section, GP on a GP, 2000 steps | 49 | 0.13-0.17 | 86% | 1.2 | 0.75 | 0.020, 0.090 | 0.051 |
| Folded section, walk, 2000 steps | 49 | 0.01-0.08 | 70% | 0.46 | 0.39 | 0.033, 0.79 | 0.042 |
| Walker, GP on a GP, 2000 steps | 49 | 0.33-0.43 | 90% | 19 | 1.00 | 0.036, 0.26 | 0.092 |

`gate_nodes.py` reported a median of 0.7% for the 7x7 grid; the medians here
are 2-4% on the deep networks, and the tails are what that gate did not show.
The walk's p95 is 0.10 over the larger half of the variances, so those errors
sit at rows of small variance.

- **Where the input is nearly certain the nodes are wasted.** On the GeoScape
  walk network (`A_model`, 32 experts, two epochs of `train_by_expert` on the
  GPU, the same seed, `_input_rule` replaced for three dimensions):

  | Rule | seconds an epoch | GPU peak | bound after two epochs | parameters, against 64 nodes |
  |---|---|---|---|---|
  | 64 Sobol nodes | 404 | 8.3 GB | -16919.363 | |
  | Stroud's degree 5, 14 nodes | 195 | 7.2 GB | -16919.369 | 2e-5 |
  | The axes, degree 3, 6 nodes | 151 | 6.0 GB | -16919.337 | 4e-5 |

  The bound rose 619 over those epochs. At fixed parameters the 14 nodes are
  more accurate than the 64 (gradient 1e-5 against 1e-4). This is not the
  rejected middle ground: the chain and the conditioning stay. Chapter 16 holds
  at 9 nodes too (p95 1e-3, gradient 1e-3). Not tried: held-out scores, longer
  training, prediction.
- **The rule cannot be chosen from the input's spread alone.** Chapter 17 and
  chapter 16 have about the same sd / range, and 14 nodes do not hold on 17
  (p95 0.55 against the 64 nodes' 0.31). What separates them is the clamp. The
  question left is the switch: an option a model sets, or a choice made in the
  graph from a cheap pilot.
- **The corners of the 7x7 grid carry nothing.** Dropping the nodes that hold
  the last 1e-3 of the weight leaves 31 of 49. Variance p95 0.088 against 0.090,
  0.79 against 0.79, 0.26 against 0.26 on the three deep cases, 5e-4 against
  2e-4 on chapter 16; Walker's median alone moves, 0.036 to 0.057. Time 0.6-0.9
  of the full grid, about 0.63 at the larger sizes. Dropping to 21 nodes starts
  to cost on Walker.
  Numbers move, so it is not bit-compatible.
- **Exact arithmetic that is cheaper** (isolated calls, the same sums in one
  chunk; a whole step not measured). On a tensor grid each Gaussian component
  of the kernel is a product over the coordinates, so a coordinate needs its
  own 7 nodes rather than the grid's 49 and a certain one none: 5e-16 from
  the package's kernel, 0.69-1.0 of the time on the CPU, 0.75-0.91 on the GPU;
  beside the coordinates, also 49 nodes against 64, 0.39-0.51, though no gain
  at the folded section's 192 rows on the GPU. And `k S kᵀ` with a gradient by hand, the
  forward product reused: 0.64-0.93 on either device, 3e-15 from
  autodifferentiation. With the corners dropped, the three together are 0.39-0.44
  on the GPU and 0.46-0.61 on the CPU with a Gaussian kernel. On chapter 16's
  CPU, eight components under XLA, they are slower than the package, 1.2,
  which is unexplained and comes first. The prototype's components are unrolled,
  so its memory is not yet what the package's would be.
- **Where the input is uncertain the bias is real and the clamp is why.**
  Conditioned on a node the kernel `k` is larger than the expected kernel in
  `(K + D)^-1` allows, so `k S kᵀ` passes one. After 2000 steps on the deep
  networks it does at 70-90% of the nodes, where the variance is then zero and
  the gradient through it is zero, and what the clamp adds is up to 19 times the
  variance (raw, the conditional variance is far below zero). The integrand has
  a kink, which is why a rule's polynomial degree buys little there and even
  the 15-point grid is 32% off at the walk's p95. This is the model rather than
  the arithmetic, and it is the larger question: chapter 17 is in this regime.
  Whether a consistent `K` per node can be afforded, or a different limit than
  a clamp is right, is open.
- **Draws per location suit this regime and not the other.** A gradient by 16
  draws is 0.13-0.36 off on chapter 17 and the folded section against the fixed
  nodes' 0.02-0.05, but its error averages and theirs does not; on the GeoScape
  network it is 0.18-0.31 off against the nodes' 1e-4. Time 0.23-0.62 of the
  package. The stopped stage B runs, 2000 steps, new-location log score, mean of
  two seeds (one on `d1_dp2`), the 49 nodes against 4 and 16 draws, seconds a
  step in brackets: `d1_dp2` -0.268 (0.72), -0.296 (0.09), -0.290 (0.27);
  folded -0.720 (0.010), -0.798 (0.004), -0.644 (0.008); Walker -6.203 (0.081),
  -6.210 (0.020), -6.211 (0.041). Against 7 nodes the draws save little: `d1_dp1`
  -0.075 (0.080), -0.139 (0.057), -0.040 (0.172). Too few seeds to decide. Plain
  draws, unchunked; randomized QMC draws (Buchholz et al., 2018) are untried.
- **Where a step goes** (one call, one chunk): the backward pass is 60-80%.
  Eight-component kernels spend their time in the kernel (GeoScape 0.6 of the
  CPU's call, `k S kᵀ` 0.11-0.17), a Gaussian kernel at large `m` in `k S kᵀ`
  (chapter 17 about half). Forcing the floor of four nodes a chunk costs 3x on
  the GPU in an isolated GeoScape call, at half the peak.
- **In order**: (1) the rule as an option, off by default, gated on the GeoScape
  network to convergence and held out; (2) the corners dropped; (3) the kernel
  by coordinates and `k S kᵀ` by hand, after the chapter 16 CPU regression is
  understood; (4) the clamp, as a modelling question of its own; (5) stage B
  finished, if (4) leaves it worth doing.

---

**M — Experts with fixed weights** (requested 2026-11-03; **built
2026-11-06** for 0.10.6 as `GPOptions(expert_weights="coverage",
expert_temperature=, expert_blend=)`, the record and the measurements in
`docs/expert-weights.md`: better on Tom East and the folded vein's walk,
4-7% worse rmse on Walker and Jura, a GeoScape walk network's prediction
5x faster at half the memory; fixed weights are the default for a new
model wherever the network takes them, by the user's choice. Left
open: a reference length tied to the kernel's range rather than to the
inducing spacing, which Walker's long range suggests. Found on the way: on a GPU, Tom East's 20 m grid predicts in 9.0 s
under the variance rule with XLA and in 1.3 s with XLA off -- a second
call, so not a first compilation -- against 0.93 s under fixed weights
with XLA; not looked into.) What was asked: today
an expert's weight at a location is read off its own prediction there:
`_GPNode.get_expert_weights` gives each expert its explained variance over
the variance it leaves, normalized over the experts (`_blend`; under the
second moment `1 - explained`). The weights therefore move with training
and with the input's uncertainty, and every expert is predicted at every
location to compute them. Fixed weights would be a function of the
location alone, set when the experts are built -- a partition of unity over
the experts' regions -- so that a location reads only the experts whose
weight is not zero there. To settle before building: what fixes them (the
inducing points' clusters, a taper over the overlap `experts()` and
`grid_experts` already make), whether they ride on the root as the expert
sets do, how they meet slots and prediction by expert, what a save records,
and the gate against today's weights (held-out scores, seams in a map, time
and memory on a many-expert model).

**M — What a walk still costs** (measured 2026-11-05, after 0.10.3 cut the
folded section's training step from 555 to 349 ms on the GPU and from 808 to
609 on the CPU; `docs/benchmarks/walk_cost.py`, `walk_ops.py`). Three parts
remain, largest first:
- *The reader's pathwise draws* -- **done in 0.10.4** (profiled 2026-11-06,
  `docs/benchmarks/realization_cost.py`): the triangle solved once per call
  and the two products joined, the features' products in single precision
  under `quadrature_precision="float32"` (the GPU's default), the chunks
  compiled with XLA on a GPU outside a tape; the folded section's grid
  prediction 12.3 -> 5.9 s on the GPU, 22.6 -> 21.7 s on the CPU in float64
  and 17.8 in float32. What is left of a GPU prediction is the moments,
  4.6 s at one realization. Contracting the feature weights first would
  still halve the products, the norms that rescale each draw reading the
  features whole; not tried.
- *The moments, compiled with XLA on a GPU* -- **built and dropped
  2026-11-06**. Profiled on the folded section after 0.10.4 (15 batches of
  2158 rows): the walk's ten steps 70% of a batch, about 18 ms each, the
  reader's expert moments 27%; XLA on each step alone bought 14% of the
  walk, and the prediction's whole graph is refused by XLA on a GPU
  ("Invalid TensorList shape" where the realizations come in several
  chunks). Compiling `GPWalk._joint_walk` and a GP node's
  `_expert_moments` over an uncertain input whole -- a fresh
  `tf.function(jit_compile=True)` per call inside the outer trace, so it
  captures no other graph's tensors -- took the folded section's grid from
  5.9 to 4.5 s (4.6 to 3.1 at one realization), the realizations moving by
  1.6e-7, the GPU's peak 0.42 GB against 0.63. On the 2026-10-22 GeoScape
  walk network (many experts, three readers) the first `predict` of a
  process took **897 s against 655** plain, the second 109 against 206,
  and the process was killed at a 28 GB cap where the plain path peaks at
  15.9 GB: every expert and every outer trace is an XLA compilation of its
  own, kept alive. GeoScape predicts once per process, so the first call
  is the one that counts. Worth retrying only with one compilation per
  node and shape kept across calls, which needs the moments' inputs
  (the node's refreshed state included) passed as arguments rather than
  captured. Patch kept outside the repository.
- *The refresh grows as the cube of the inducing points*: they walk
  themselves every step, m walkers against m points, and at 775 a step is
  2.0 s against 0.35 at 357. Experts divide it.
- *XLA for the expected kernel during training on the CPU*: 9.2 ms to 6.7
  with its gradient, isolated, at the folded section's shapes; never tried
  in a training step, where XLA as a whole was slower and went NaN before,
  and on the GPU it bought nothing in the step.

**S — A reader's whitened prior is not standard** (found 2026-11-06,
profiling the realizations). `BasicGP._whitened_prior` promises that `G^-1
Phi(Z) w` is "exactly standard whatever the features", `G` the Cholesky of
the features' Gram `Phi(Z) Phi(Z)^T + 1e-9 I`. At the folded section
reader's 357 walked inducing points the Gram of its 1025 Matern32 features
has 55 eigenvalues under the 1e-9 jitter (the exact kernel's smallest is
1.2e-3, no two points closer than 0.15 ranges), so the whitened rows'
variances fall to 0.33 and 212 of 357 rows are off by more than 1e-3; the
draw's comment in `_pathwise_simulate` that the rows are orthonormal says
the same thing. The rescale to the moments' variance restores every
location's marginal, so what is wrong is the realizations' covariance
between locations, by an amount not measured. Measure first: the
realizations' empirical covariance between locations against the one the
posterior implies, at the folded section's grid; then, if it matters, draw
the null
space of the Gram with normals of its own or take more features where the
Gram is short of rank.

## 2. Fitting and initialization

The theme is measured rather than assumed: **training does not leave the
basin it starts in**, so where a start can be computed rather than guessed,
it should be.

**M — Inducing points by cover tree** (requested 2026-10-10; **built
2026-10-15, 0.9.3**: `inducing.from_cover_tree(data, separation,
transform=)`, the samples farthest first, divided into experts by
`experts`; read against the paper since: the guarantees of its Algorithm
12 (eq. 20, separation and spatial resolution both `separation`), but in
O(N·M) where its breadth-first tree is O(N log N), and without its two
optional steps -- the local averaging that places a node between samples
when that keeps it a radius from the others, so fewer nodes are needed,
and the Voronoi repartition that balances the tree. Either is the next
step if the measurement below asks for it; the clustered-data
approximation of its §3.3 is Gaussian-likelihood only. What remains is
that measurement). Terenin,
Burt, Artemev, Flaxman, van der Wilk, Rasmussen & Ge (2024), *Numerically
stable sparse Gaussian processes via minimum separation using cover
trees*, JMLR 25(26), arXiv 2210.07893 (in the Paper Library; read from the
abstract so far): conditions on the inducing points -- a minimum
separation -- under which the sparse computations are numerically stable,
an automated way to place points satisfying them in low dimension through
a modified cover tree, and an alternative sparse approximation for a
Gaussian likelihood trading a little accuracy for more stability. Where it
meets geoML: the inducing helpers place points by k-means, lattice or hull
(`from_kmeans`/`from_grid`/`from_hull`) and nothing guarantees a
separation beyond `combine(tolerance=)`'s greedy merge; stability rests on
the 1e-6 jitter and the whitened root of 0.6.9, which was needed because a
fault displacement put restored inducing points 0.03 apart and the
Cholesky went NaN in graph mode. The candidate is an
`inducing.from_cover_tree(data, separation)` beside the others, the
separation in the input's transformed units, and a measurement of what the
guarantee costs or buys against `from_kmeans` at equal counts (held-out
scores, the conditioning of `K`, training stability) on Walker, Jura and a
3-D case. Read the paper in full first; the alternative approximation is
Gaussian-likelihood only, which most of geoML's models are not.

**M — Gradient-free training experiments.** Closed against three conditions
in 2026-09-03 (see below) and kept here only for the part that survived: the
memory saving is real and available today, since taking the ranges off the
tape cuts peak GPU memory 3–4× and the step 1.5×.

**L — A conjugate trainer for shallow single-expert networks** (requested
2026-10-26). One expert, and the variational parameters leave training:
what made the natural-gradient step fail for the experts (Settled by
measurement, 2026-10-09) was the blend, which is not conjugate, and a
single expert has none. The user's proposal: site updates, conjugate
gradients and gradient-free training together; the parts, as assessed:
- **q eliminated.** Under a Gaussian likelihood (its warping fixed for the
  solve) the optimal q(u) is closed form -- Titsias's collapsed bound --
  and nothing is trained but the hyperparameters. Under any other
  likelihood with a per-datum expected log-likelihood (the indicators,
  Laplace, Huber, the mixtures), CVI site updates, conjugate-shaped once
  there is no blend: tens of iterations at steps near 1 where the
  likelihood is log-concave, damped where it is not (`LikelihoodMixture`).
  Their per-datum derivatives in the marginal mean and variance come from
  autodiff of `log_lik`, with no tape through a Cholesky.
- **Operation nodes stay conjugate.** `Add`, `Linear` and
  `LinearCombination` over GP nodes keep the model jointly Gaussian, so q
  is joint over every node's inducing values -- a block system, still
  closed -- which covers LMC-style models.
- **Out of reach.** Deep networks: a GP reading another's uncertain output
  (`GPWalk`, concatenation) has no conjugate update for the hidden layer,
  the expected kernel coupling it nonlinearly; `Multiply` and
  `GaussianMixture` leaves are not Gaussian. They keep the current
  machinery, so this is a second trainer, not a replacement.
- **Size.** Dense Cholesky takes one expert to about 1-2e4 inducing
  points: GeoScape's 6605 points in seconds, 2e4 in tens of seconds and
  3.2 GB; the data side, `sum K_uf D^-1 K_fu`, accumulates over row
  batches, so N is not the limit. Conjugate gradients only help past that,
  and only matrix-free (the kernel recomputed per product, memory O(m)),
  since an unstructured K has no fast product the way a grid has; its
  conditioning (up to 1e9 recorded) calls for a preconditioner (pivoted
  Cholesky), and the bound's log-determinant then comes from stochastic
  Lanczos, a noisy objective that also weakens the reproducibility
  contract. Cholesky first.
- **The hyperparameters.** Once q is eliminated the bound is a
  deterministic, smooth function of them, whose gradient through a
  Cholesky is cheap and holds no variational parameters on the tape:
  L-BFGS first, as SGPR does. Bayesian optimization earns its place where
  gradients are gone (with CG) or the objective is multimodal, and scales
  poorly past about 20 dimensions -- a shallow Jura LMC already has 30+
  (ranges per node and dimension, warping links, noise, `Linear`
  weights). It will not find better optima: the bound was measured
  unimodal along the range on Jura and flat on Walker (Settled by
  measurement, 2026-09-03), so its value is speed and robustness. The MAP
  priors join either objective as they are.
- **What it ends, if it holds:** the experts' slots, weight sweeps and
  late rise, and the optimizer's decaying clock (the next item), for every
  model it covers. It agrees with the measurement that fewer, larger
  experts scored better than many small ones at a fixed total of inducing
  points.
- **Steps and gates**:
  1. A prototype `train_conjugate` for shallow single-expert networks:
     the collapsed q under Gaussian likelihoods, CVI otherwise, L-BFGS on
     the hyperparameters; BO as a second arm on the same objective.
  2. Against `train_full` and `train_by_expert` on Walker, Jura, Tom East
     and GeoScape's shallow Macpass network (the 2026-10-25 Run's, one
     expert of its 6605 points): wall time to equal held-out scores
     (rmse, CRPS, goodness), peak memory, the spread over seeds. It goes
     on only if it wins time or memory at equal scores.
  3. Only if the single expert holds at Macpass's size: matrix-free CG for
     larger m, with BO there, since CG takes the gradients away.
  4. A deep node met by the trainer refuses with a message, or falls back
     to the gradient path for the non-conjugate part -- decided at step 1.
  5. The catalogue (`fit.train` lists the trainers GeoScape offers),
     persistence (nothing new in a save if q is recomputed from the data
     and the hyperparameters on load -- decide), progress (an evaluation
     or an iteration as the unit) and the reproducibility page.

**M — The optimizer's defaults, measured** (requested 2026-10-15). Every
trainer builds the same optimizer -- Adam with AMSGrad at 1e-2, decaying
0.999 a step (`ExponentialDecay(1e-2, 1, 0.999)`, written out in each
model's constructor, `set_learning_rate` and the by-expert state in
`models.py`) -- and none of those numbers was ever chosen by measurement.
The decay is a clock: at 0.999 a step the rate is at 37% after 1 000 steps
and 5% after 3 000, so how far a model can travel depends on how many steps
its trainer takes, which `train_by_expert` already ran into (item 3 under
the batched experts, the experts' clocks against the shared parameters').
The experiments: the starting rate and the decay (including none, and a
decay per epoch rather than per step), AMSGrad on and off, and other
optimizers -- Nadam first, then plain Adam, AdamW and whatever else is
cheap to try -- each on `train_full`, `train_svi` and `train_by_expert`,
on Walker, Jura and a 3-D case at several seeds. Score on held-out data
(rmse, CRPS, goodness) and on the bound reached for a given wall time,
with `training_tolerance` stopping both arms, since a faster optimizer
that stops earlier is the point. A new default changes how every re-run
script trains, so it goes in the changelog, and the old settings should
stay reachable: today `set_learning_rate` takes a rate only, so the
outcome may also be an option for the decay and the optimizer.

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

**M — Latent variables as a model's input** (split 2026-09-28 from
batched prediction from a latent node, `predict_node` since 0.8.5). A `LatentVariable` may enter a new model as a `GaussianInput`
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

**S — A refined surface in the manual again** (filed 2026-10-29). Chapter
17 moved to a 2D section of a folded vein, so the manual no longer shows
the 3D surface workflow end to end: `refine` on a `BlockSet3D`, the
contour of an indicator at zero with `get_contour`, and a prediction onto
the mesh, coloured by the uncertainty. Chapter 12 teaches `refine` and
`get_contour` apart; a small synthetic body in 3D there would carry the
rest. The 3D vein drillholes that chapter 17 used took 57 minutes a run
at 0.9.12, so the example should be one a release can afford.

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
- `BoxCox`'s default shift from the data (`shift=None`, built 0.9.3 as an
  opt-in): the default stays `1e-6` today only because a save replays the
  arguments it was given, so an old `BoxCox(1)` reloaded under a new
  default would register the extra `shift` parameter and refuse its own
  slots. Here the default becomes `None` and the float path's slot layout
  stops mattering; the manual's chains drop their hand-set `shift=1.0`.
  Measured already (Walker, 2026-10-16): the found shift trains as the
  hand-set one does, and on a column with no zeros it is the old `1e-6`.
- **Deprecated in 0.9.0** (ignored under `"joint"`, removed here):
  `GPWalk`'s `precision` parameter and the variance shrinking it drives,
  and the walk's displacement term, `1/2 sum (moved - start)² / spread²`
  over the walked inducing points -- replaced under `"joint"` by an
  isotropic GP reading the walk, documented (measured 2026-10-09: -6.1 a
  new hole on the folded section against -10.7).
Measure nothing new for it: it removes code whose replacement the 0.9.0
gates passed.

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
is met** (the last, item 47, in 0.9.13; the changelog has each one); what
is left below is the paperwork outside the code.

What GeoScape builds on, where a change must be flagged to it rather than
made quietly: dotted class paths importable forever, already policy;
constructor arguments as the whole persisted story, since the editor
exposes nothing else; the Fourier-feature classes kept internal; **batch
invariance**, a location's realizations independent of the call that
computed them given the same fit, seed and count
(`test_predicting_only_the_new_blocks_gives_the_same_answer`), which a
per-location residual draw would break; and node names seeding the draws,
GeoScape never renaming a node once created.

Outside the code, from the same list: a contributor licence agreement
before the first outside contribution is merged; and title in writing, with
no institutional claim, before any geoML code ships inside something
RockAnalytica distributes -- the desktop helper (M7), a self-hosted tier,
or a `geoscape-cli` that imports geoML. Version 1 and the cloud need
neither, Scripts being text and portal services not distribution.

---

## 7. Geometallurgy

Variables that describe how the ground will behave in the plant rather
than what it contains: support, non-additivity and the rules that carry
a sample's value to a block's.

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

**L–XL — Custom aggregation rules for variables that do not add up**
(requested 2026-10-14). Every block value today is the volume-weighted
mean of its sub-blocks (`_aggregate`), which is right for an additive
variable -- a grade, a density -- and wrong for the geometallurgical ones:
a hardness or work index, a recovery, a throughput, a comminution
parameter, a ratio of two grades. A block's work index is not the mean of
its parts' (energies add along the circuit, not the indices), a recovery is
weighted by the metal it recovers, a ratio is the ratio of the sums. So a
variable needs its own rule for going from points to a block: a declared
aggregation (a weighted mean with a named weight -- the metal content, the
tonnage --, a harmonic or power mean, a ratio of two additive variables, a
user function applied realization by realization as `derive` does), read
by `_aggregate`'s callers, `as_blocks3d`, `aggregate(data)` and the
grade-tonnage curves, which all assume the mean today. Three problems, in
order of difficulty:
- **Prediction at the data's own support.** Interpolating a non-additive
  variable from samples of one support to points of the same support
  should be fine as it is -- the model never sums anything -- but that
  must be verified, not assumed: a synthetic field built from additive
  parts, the non-additive variable computed from them at the samples, and
  the model's prediction of it against the truth at the same support,
  beside predicting the parts and computing the variable from their
  realizations (`derive`). If the second wins, the honest route is to model
  the additive parts and derive.
- **Blocks under a rule.** A block's value is the rule applied to its
  sub-blocks realization by realization (never to the mean: the rule is
  nonlinear), then summarized; the `dispersion`, the quantiles and the
  cut-off shares follow from those realizations rather than from the mean's
  formulas. Needs the rule to reach the sub-block fan-out, and the weights
  of a weighted rule to be predicted alongside.
- **Samples of different supports combined under the rule** -- the
  hardest. A composite, a bulk sample and a plant reconciliation measure
  the same variable over different volumes, and for a non-additive variable
  a larger support is not the mean of the smaller ones, so the likelihood
  has to aggregate the latent field to each sample's support under the rule
  before comparing -- the change of support of the item above, made
  nonlinear. It shares that item's prerequisite (the support visible to the
  model) and should be designed with it.

---

## 8. Optimizations

**M — Faster mesh operations** (requested 2026-11-11). Speed up the mesh
work: contouring, the Manifold booleans, `simplify`, `split` and the mesh
sets. Profile first on the Assen and Tom v6 models, then work on whatever
takes the most time.

---

## Settled by measurement

These were tried. The numbers are why they are, or are not, in the package.

**The split criteria that lost to coherence** (2026-10-29, item M on
refinement). Scored on four known fields (the pod, the ore body, the rock
plane, a rough random field; 8 and 20 realizations; coarse blocks and
blocks half their size) against Run 12's far blocks, by the benefit a cut
buys against the field (`docs/benchmarks/split_coherence.py`):
- *The contrast* -- the largest difference across a block's sub-blocks in
  the probability of being above the cut-off, at least a half and above
  an exact binomial noise floor. Built first. The floor, taken at its
  worst (sub-blocks independent), was 0.70 at 20 realizations and left the
  pod's shell uncut (99 blocks against 533, 14 750 m3 on the wrong side
  of the field against 12 750); at a fixed half it still lost 2% there and
  passed none of a gradual rock contact, whose probabilities move at most
  0.26 across a block though its realizations agree (coherence 0.24 to
  0.68). How *far* the field moves across a block is not whether the
  model places it.
- *The F test of the sub-blocks alone* -- with 200 realizations a tiny
  real gradient in unsampled ground is significant, and it marked 1% of
  Run 12's far blocks; an effect size is needed, which is the coherence.
- *The within-block share of the variance* (dispersion over the total) --
  100% of the far blocks pass any threshold the worthy ones pass.
- *The same measures on the indicator of the cut-off* -- lose benefit at
  thresholds the values keep it at.

**A plain cross-covariance at certain inputs -- not worth it** (2026-10-29,
item L's 6). The first layer's `covariance_matrix` builds the
uncertain-input normalization (`total_var`, the determinants, a `norm` of
exactly one) at inputs whose variance is zero. A training step that reads
`_plain_covariance` there instead, against the step as it is, three
repeats each (`docs/benchmarks/compute_review.py`, the `step` and `plain`
parts): Walker (one node, 200 iterations) +1.4% on the CPU and +0.9% on
the GPU, slower; chapter 15's experts (761 batches) -1.3% on the CPU and
-5.1% on the GPU. A twentieth of an expert model's step on the GPU and
nothing elsewhere does not pay for a second path through every GP node's
cross-covariance.

**The conditioned quadrature's two loops merged into one -- not worth it**
(2026-10-28, item L's 3). `_mixture_moments` runs `sums` and `root_sums`
as two `_chunked_sums` loops in a prediction, both computing the component
kernel and `ksk`. Answering the root loop with zeros, which bounds what a
merge could save, moved a prediction batch 0.8% on chapter 17's walked
model on the CPU, 0.7% on the GPU, and -1.1% on the GeoScape walk network
on the CPU (`docs/benchmarks/compute_review.py`, the `roots` part): the
duplicated kernel is not where a batch's time goes.

**Four cheaper forms of the conditioned quadrature -- rejected** (2026-11-01;
`C:\Users\Public\qm`, under "The quadrature's rule should follow the input").
- **`k S kᵀ` by `matmul` in place of `einsum`**: 1.0-1.4 of the time on the CPU
  and 1.0-1.15 on the GPU; the `einsum` is not the cost.
- **One Gram matrix per row, then one trace per output** (`n q m²` once, not per
  output): it cannot clamp node by node, which is up to 19 times the variance,
  and it is no faster at one output (Walker 2.1x slower on the CPU).
- **A rule turned at random each step** (Genz and Monahan's stochastic
  spherical rules, the radii left fixed): the mean of 16 turns is as far from
  the reference as one turn (gradient 0.131 against 0.133 on the folded section,
  0.492 against 0.493 on Walker). The error is in the radii, and turning does
  not move it.
- **Cubatures of fewer nodes for the same degree** (Möller's 12 at degree 7,
  rings of 7, 16 and 21): no better than the Hermite grid of the same size on
  the deep cases, whose integrand has a kink. Rings of 36 at degree 11 did beat
  the 49 on the folded walk (p95 0.35 against 0.79) and are the one worth a
  second look.
- Also mixed, and not adopted alone: a Hermite grid over only the uncertain
  coordinates beside certain ones. 0.75 of the time; better than 64 Sobol
  nodes on the folded section (median 4e-4 against 0.009), worse on Walker
  (gradient 0.087 against 0.018).

**Repairing the conditional variance's clamp -- rejected** (2026-11-02;
`C:\Users\Public\qm\neg`, theory in the research project's
`Spatial cross-validation/report/kq_closure.tex`). `1 - k_q (K + D)^-1 k_qᵀ` at
an input node is a simple-kriging variance on weights solved with the wrong
covariance, the expected `K` where the node's own is `K_q`, and it goes negative:
at a fit on the folded section the unclamped mean is -0.039 on the GP on a GP,
and 76% of rows are negative across their nodes. The clamp is what keeps the
objective a number, and an active clamp shows the family it would bound is not
a distribution there, so the objective is not a bound. Every repair removes the
negatives and predicts no better. New-hole log score on the folded section,
Gaussian leaf, 2000 steps, seeds 2026, 1, 2, 11, 12, 13:

| Arm | GP on a GP | Beside the coordinates | Walk |
|---|---|---|---|
| the clamp (the package) | **-9.02** | **-5.22**, 3 of 6 in the -2 basin | -9.18 |
| the kriging variance of the same weights (K1) | -10.08 | -8.05, none | -9.09 |
| K1, one CG step on the weights, mean and variance | -10.81 | -9.34, none | -9.54 |
| K1, three CG steps | -10.17 | -8.23, none | not run |
| `K_q` whitened per node, shared `S_w` | -10.76 | -8.76, two seeds, none | not run |
| first moment only, no spread | -10.75 | -9.37, none | -10.25 |

K1 and the steps lose to the clamp on 5 of 6 seeds on the GP on a GP and 4 or 5
of 6 beside the coordinates; on the walk K1 ties (3 of 6) and one step is
slightly behind. The steps cap the largest node variance (2.2 to 0.99 at the
first fit), so the tail nodes are not why K1 loses. `K_q` per node cost 160x the
clamp a step. With the hyperparameters trained, K1 narrows the hidden layer
(sd/range 0.204 to 0.065) and raises the noise (0.279 to 0.312, and 0.153 to
0.237 beside the coordinates), the opposite of the report's toy, whose
hyperparameters were fixed. Why the clamp's objective finds the better basin is
open. The kernel was Gaussian throughout; only three Matérn 3/2 seeds ran, for
the clamp. `network.py`'s `_mixture_moments` and `models.py` should stop calling
this quantity a bound; not done yet.

**Whitening the second moment into one-dimensional quadratures -- rejected**
(2026-11-02; `C:\Users\Public\qm\neg\q1.py`). With independent input
coordinates and a Gaussian kernel component, `E[k_i k_j]` factors over the
coordinates into closed one-dimensional integrals: it matches 101 points per
coordinate to 1e-15, and 7 points per coordinate equal the 7x7 grid to 8e-16.
It computes the expectation, so it cannot clamp node by node, and the raw
conditional variance is negative for whole rows (shares 0.76 on the folded
GP on a GP, 0.99 on Walker, 0.58 on Matern 5/2, 0.50 on chapter 17). It is also
slower at these sizes: 0.061 s against 0.025, 0.351 against 0.101, 2.166 against
0.070, 4.03 against 0.397. The spread's quadrature error stays (3e-3 to 9.5e-3).

**The second moment over certain inducing points (the "middle ground") --
rejected** (2026-10-24). On a GeoScape network of three GP nodes reading a
`GPWalk` (4429 rows, 32 experts, a likelihood mixture) the profile put 83%
of a training step's op time in the conditioned quadrature
(`_conditioned_sums`): the inducing inputs conditioned on each quadrature
node of the input, `[n, nodes, m, d]` arrays a reader. The middle ground
kept the joint first moment and took the second moment, and the pathwise
realizations, with the inducing points at their means -- the certain-points
route, a plain kernel at the realizations (a switch, since removed). It is
fast and looks harmless in-sample:
- a training step 6.09 s against 2.60 (mean 7.72 against 2.98), the bound
  over the same eleven batches within 0.01-0.18%, the middle ground's
  always higher;
- at fixed parameters, predictions at 500 data locations moved a median
  0.0003 sd, 0.015-0.022 sd at the 99th percentile, 0.4-0.9 sd at worst on
  a few grades; probabilities 0.9 points at most.
Scored on the folded section's 200 new holes (`expected_kernel.py`, seeds
2026, 1 and 2, 2000 iterations, the walk network and the GP on a GP) it is
worse on five cases of six, a log score of -3.4 a hole on average, and its
intervals too narrow:

| seed | network | score a hole | calibration (1 right) | realizations/variance | bound |
|---|---|---|---|---|---|
| 2026 | walk | -9.16 / -10.17 | 1.43 / 1.50 | 1.00 / 1.00 | -102.0 / -119.1 |
| 1 | walk | -11.74 / -11.25 | 2.23 / 2.12 | 0.99 / 1.00 | -68.2 / -73.4 |
| 2 | walk | -2.46 / -8.47 | 0.69 / 1.79 | 1.00 / 1.40 | -63.2 / -82.5 |
| 2026 | GP on a GP | -7.03 / -12.56 | 0.76 / 1.96 | 0.98 / 384 | -150.5 / -138.3 |
| 1 | GP on a GP | -11.99 / -13.14 | 1.62 / 2.26 | 0.97 / 72 | -168.0 / -121.4 |
| 2 | GP on a GP | -9.55 / -15.88 | 1.35 / 3.24 | 1.01 / 18 | -143.7 / -63.5 |

(conditioned / middle ground). On the GP on a GP the bound rises by up to
80 while the held-out score falls: the inducing points' spread dropped from
the second moment makes the model tighter than its data allow, and the
bound rewards it -- a check on the training data alone would have taken
it. Its realizations lose the moments (18-384 times the variance), the
refresh still sizing `alpha` and the solves for uncertain inducing inputs;
consistent, the refresh would treat them as certain too, which is the
marginal rule's position the joint rule was built to leave. The step's cost
is the conditioned quadrature's work, and it is worth what it costs.

**Single precision past the quadrature -- rejected** (2026-10-24, planned
and run the same day). After the conditioned quadrature's arrays went to
float32 (0.9.8), two more places looked large enough to follow, under the
same option:
- **The realizations' feature products** (`_pathwise_simulate`'s `draw`:
  `k · root_prior`, the residual, the closing sum over the ~1025
  features). In float32 the residual `phi_h - projection`, a difference of
  two terms of the features' own size, moved a realization by 5e-5-9e-5 of
  the spread between realizations, against a gate of 1e-5 (the quadrature
  alone: 3.7e-7); kept in float64, the rest passed at 1.4e-6 -- and bought
  nothing. A prediction of the GeoScape walk network at its 4429
  locations, 10 realizations, on the RTX 4060 Ti: 1238 s in float64, 972 s
  with the quadrature in float32, 975 s with the features too.
- **`ksk = k (K + D)^-1 kᵀ`**, the `O(n · nodes · m² · size)` einsum the
  conditional variance `1 - ksk` comes from, behind a per-expert guard
  (`eps · ‖(K + D)^-1‖₂ · |k|²` under 1e-4 of each row's conditional
  variance). Measured at eight draws of the input on 400 of each model's
  own locations, TF32 off:

  | model | m | `‖(K + D)^-1‖₂` | guard clears | worst change where cond > 1e-3 |
  |---|---|---|---|---|
  | GeoScape walk network, 128 expert calls | 185-227 | 0.47-2.5 | 100% | 8e-6 |
  | folded section, GP on a GP | 30 | 6.3 | 21% | 6.7e-4 |
  | folded section, walk | 30 | 14 | 6% | 1.4e-3 |
  | chapter 17's walked model | 617 | 61 | 0.7% | 1.8e-2 |

  The gate was 1e-4: wherever data sit near a reader's inducing points the
  inverse grows and float32 loses the conditional variance, and that is the
  case a model is built for.

**Two traps on the way, both measured**: on the GPU
TensorFlow runs float32 matmuls in TF32 by default -- `k · root_prior` 2.9e-4
from float64 with it, 7.5e-7 without; the switch is read when a kernel runs,
so turning it off for the length of a call works on traced functions. On the
CPU oneDNN's batched float32 matmul is wrong past a contraction of about
20 000 (tensorflow/tensorflow#129498). The shipped float32 quadrature is
elementwise with reductions and meets neither; anything that adds a float32
matmul must handle both.

**A confident variational start (`delta` 0.01) as the default -- rejected**
(2026-10-21; the folded-ore ablation's finding carried to the manual). On
the folded section under the joint rule a small starting `delta` selects
the fold (one body, limbs 1.00) where the default start settles on five
blobs at a better bound, so every chapter of the manual was run with every
variational node's `delta` starting at 0.01 (`BasicGP`, `MultiStructureGP`,
`GradientConstrainedInput`, and `cross_validate`'s fresh state with them;
worktree `geoML-confident`). All green, and worse where it was scored:
Walker's out-of-fold rmse 220.8 -> 267.1 and CRPS 125.9 -> 152.1 (both
+21%; the folds started confident too, so this mixes the training start
with the short fold refit -- but the trained model's own maps changed,
smoother and striped, the median's top 1200 -> 950); Jura's held-out
metals 1-6% worse in rmse on five of seven, CRPS a draw, rock kappa mixed
(Argovian 0.79 -> 0.73, Sequanian 0.62 -> 0.69). Goodness rose everywhere
(Jura 0.61-0.75 -> 0.68-0.82, Walker's conformal cut 0.847 -> 0.903): the
confident start buys calibration with wider intervals, not better
predictions. Chapter 17 is the decisive one: the deep model's walk moved
points 0.09 ranges at most against 0.38, and its contact came back as the
stationary model's tubes along the holes (113 755 m2 against the default's
one sheet, 193 518 m2), its first bound near -35 000 against -9 000. The
fold the small start finds on three holes is a basin of that section, not
a better start in general; `delta` keeps starting at 1.

**Reading a moment-trained deep network by draws -- rejected**
(2026-10-16; the first step "Propagate individual realizations" planned).
The model trained on the moment bound, read through draws of its hidden
layer instead of its moments: -12.62 a new hole against -8.99 on the folded
section's GP on a GP, -11.46 against -11.12 beside the coordinates. The
moment bound reports 10-32% of such a model's own variance, so training
leaves the hidden layer uncertain at no cost and the exact read charges for
it. A deep model is read the way it was trained.

**One `K` read through whole draws of the parent -- rejected** (2026-10-30;
prototype and setting under "A GP node reading whole draws of its parent").
The literal form of computing the expected kernel through realizations:
`K` the mean of `k(Z_s, Z_s)`, or the package's own, with `f_s = k_s alpha`
and `v_s = 1 - k_s (K + D)^-1 k_s'` at each joint draw. At a model the
package trained, against 16384 draws, the mean agrees to 0.006 and 0.003
sd, but the package's variance is 0.057x (GP on a GP) and 0.446x (beside the
coordinates) of the draws', nearly all of them the spread of `f_s`. With
`C = Cov[k(h, Z) | h]` and the clamp ignored, the draws exceed the package
by `E_h[alpha' C alpha - tr((K + D)^-1 C)]`: `alpha` has large cancelling
entries that cancel only for the averaged kernel, and the package integrates
`Z`'s remaining spread inside the kernel before squaring. Sampling that
spread away needs per-location `Z | h`, the same `[n, S, m, d]` arrays the
quadrature holds. Trained on it the hidden layer is priced out: new-hole log
scores of -11.13 to -11.64 on the GP on a GP (the package -8.71, the flat GP
-12.2) and -10.28 to -10.76 beside the coordinates (the package -6.14 on the
same three seeds), whether `K` was the package's, the mean over the same 16
or 64 draws, or the mean over 256 draws of its own. `[[1, k_s], [k_s', K]]`
need not be positive semi-definite either, so `v_s` needs its clamp and no
joint model was shown to stand behind the bound.

**Natural-gradient steps for the experts -- rejected** (2026-10-09; Level 2
of `report/design_dual_parameter_geoml.md` in the research project
`OneDrive\Claude\Research\Batched experts\`, record in its
`report/results_2026-10-09_natural_gradient.md`, prototype in
`experiments/natural_gradient/`). The idea: replace `train_by_expert`'s
per-expert AMSGrad on `alpha_white` and `delta` with a natural-gradient
step in the site natural parameters of the family `BasicGP` already uses
(`S = (K^-1 + diag r)^-1`, `r = 1/delta`), taken on the stored
coordinates, warmed up and then held as Salimbeni et al. (2018) do, in the
hope that a step that never decays would end the late rise of item 1
above. On the whole data the direction is right (the bound's slope along
it 55 237 against 55 239 predicted) and gains 15 times the plain
gradient's best step (90 nats against 6 after 10 epochs). But the bound
takes steps up to rho = 0.03-0.1 only, whether one expert moves or all
sixteen, and loses 850-25 000 nats at rho = 1, the conjugate case's
one-step optimum: the blend is not conjugate. Batched, a step divides the
batch's gradient by the expert's KL share (about 1/20 at 4 visits), so the
stable rho is the whole-data limit times that share: 0.001 trained, 0.002
and above diverged. At that step an expert moves about 2% of the way to
what its batch implies and falls behind. J = 16 synthetic field, seed 1,
60 epochs, rmse / CRPS: at 4 visits AMSGrad 0.088 / 0.128 against 0.134 /
0.143; at 16 visits 0.096 / 0.130 against 0.132 / 0.142 (rho = 0.0005);
19% more time per epoch. The late rise stayed: with no clock to decay,
rmse rose from epoch 40 to 60 by 0.010 and 0.020 (rho = 0.0005, 0.001)
against AMSGrad's 0.008, so the experts' decaying rates are not the whole
of item 1 (one seed). Adam normalizes each coordinate's step, which keeps
a batch's extrapolation to the whole epoch from compounding; a scale per
inducing point rather than per expert would need per-datum sites, the
note's Level 3, whose full-covariance family is the `FullGP` measured
below as no gain held out. The note's Level 1 (storing a site mean in
place of `alpha_white`) only changes the coordinates Adam sees, where
whitening is the better-conditioned choice; it goes with Level 2. Stepped
in `r` itself, `r` went negative from rho = 0.1; in `log r` it stays
positive.

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
