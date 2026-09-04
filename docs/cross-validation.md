# Cross-validation: the folds, the driver, and the calibration

Status: **executed 2026-08-13** (planned 2026-08-12). Five pieces, built in
order: the scoring metrics, the fold builder, the refit driver, the variogram
figure, and the conformal calibration. This file records the design decisions
and the measurements that settled them.

## The problem

Random k-fold cross-validation assumes exchangeable errors. Under spatial
autocorrelation a held-out point next to a training point is answered by
proximity rather than by the model, so random CV flatters every score — worst
with clustered sampling, which is what drillhole data is. The literature's
first correction (spatial blocks, buffers — Roberts et al. 2017) overshoots:
folds pushed as far away as possible test an extrapolation nobody asked for
(Wadoux et al. 2021). The synthesis the field converged on is
**prediction-task matching**: the right folds are the ones whose
held-out-to-training nearest-neighbour distances are distributed like the
distances from the actual prediction target to the sample (NNDM — Milà et
al. 2022; kNNDM — Linnenbrink et al. 2024, GMD 17:5897). geoML has an unfair
advantage here: ecology guesses its prediction domain, while `spatial_k_fold`
is handed the actual grid or block model.

## The fold builder: `PointData.spatial_k_fold(test_data, k, groups, seed)`

The earlier prototype optimized a continuous per-sample fold membership plus
a sample weighting until the distance distributions matched. It overfitted —
the distributions matched perfectly while the folds were spatially wrong —
because ~n·k continuous degrees of freedom can tailor the match without
tailoring the geometry. The rewrite makes the folds **discrete**:

1. **Atoms that never split** — a metadata column (`groups="hole"`: a drill
   hole stands or falls together) or, without one, k-means into
   `min(n, 25·k)` small spatial groups.
2. **The target statistic is the nearest-neighbour distance** (changed from
   the prototype's full-pairwise ECDF): the ECDF of NN distances from
   `test_data`'s locations to the sample is what governs prediction
   difficulty; the full-pairwise ECDF mostly reflects large-scale geometry.
3. **Agglomeration under W**: cut a Ward dendrogram of the atom centroids at
   every count from `k` up to one cluster per atom, deal each cut's clusters
   to the emptiest fold largest-first, and keep the cut whose pooled
   held-out-to-training NN distances are closest (Wasserstein) to the
   target's.

The result lands in a `"fold"` metadata column; the two distance samples come
back for a diagnostic figure. Everything is deterministic given `seed` (the
k-means pre-clustering is the only draw). Coordinates only — the common case;
a feature-space variant would be a different distance matrix into the same
builder.

## The driver: `models.cross_validate(model, folds, refit, method)`

The VGP has no closed-form LOO (Rasmussen & Williams §5.4.2 is for the
closed-form GP), and retraining from scratch per fold is what the driver
exists to avoid. The translation of kriging's fixed-variogram CV:

- the trained model is saved once (`persistence.save_model`);
- each fold gets a copy rebuilt around the reduced data
  (`load_model(path, data=data[fold != i])`);
- its **variational state** — `alpha_white_*`, `delta_*`, `bias_*` on every
  latent node, which is where the data lives in a trained VGP — is
  **re-initialized**, so the fold model is structurally ignorant of the
  held-out rows; everything else (kernels, warpings, likelihood noise) is
  frozen;
- a short refit of that state alone (the fast, well-behaved part of the
  optimization), then the held-out rows are predicted into one shared
  container via `predict(..., where=held)` — folds partition the data, so
  the loop ends with every location predicted by the one model that never
  saw it.

Scores are of **measurements** (`predict_measurements` — the container's own
simulations are of the ground, which no assay observes): per fold and pooled,
`rmse`/`mae`/`bias` of the predictive mean, `crps`, and Deutsch's `goodness`.
The conceded leakage, stated in the docstring: hyperparameters and warping
saw all the data — the same concession kriging makes when it keeps the
variogram. `refit="all"` trades the other way (warm-start everything, freeze
nothing new). Leave-one-hole-out is `folds` pointed at a hole-id column.

**The refit can be minibatched.** `method="svi"` sends each fold to
`train_svi` instead of `train_full`, in batches of
`options.training_batch_size` — which the fold copy already carries, the
options riding in the saved model, so nothing new is passed in. It matters
because freezing parameters does not make an iteration cheaper: a full-batch
refit costs the whole reduced data set per step whatever is on the tape, so
a model too large to train full-batch is too large to cross-validate that
way. The passes are counted by a **separate** argument, `epochs`, because an
epoch is one visit to the data in batches and therefore many gradient steps:
a number that suits `iterations` is wrong for `epochs`, and one argument
serving both would silently be read the wrong way. Two things follow the
choice. `options.training_tolerance`, if set, judges once an epoch on the
mean bound over its batches rather than once an iteration, so a short refit
may never give it enough to fire and the cap does the stopping. And
`train_svi` does not index the variables' `training_input` per batch (a
commented-out attempt sits beside it), which costs nothing while the only
payload is `RockTypeVariable`'s `is_boundary` — accepted by both categorical
likelihoods and read by neither — and would need fixing first the day a
censoring mask or a per-datum support rides that channel.

**The scoring is what costs memory, not the refit.** Each fold is scored
through `predict_measurements`, which returns its whole answer as one array
per variable — `n_sim * n_nodes * 8` bytes a row for each column, 5 KB a row
at the defaults, and **twice that at the peak**: a 92 MB answer was measured
to cost 182 MB resident, the concatenate holding the parts and the assembled
whole at once. Nothing checked that until 2026-09-04, when a
notebook running a cross-validation had its kernel killed by the Linux OOM
killer: 63 GB resident against WSL's 62 GB, host RAM, the GPU never
involved. That call now refuses past `models.MEASUREMENT_LIMIT` (2 GB)
before doing any work.

**So the driver stopped materializing them.** Every statistic taken of these
samples reduces the sample axis one row at a time — `coverage` builds a
central interval per location and counts the fraction inside, CRPS is per
row, so are the point errors and the PIT — so nothing needs two rows'
samples at once. `VGPNetwork.measurement_batches` yields `(rows, samples)`
per batch and the fold loop folds each batch into **sufficient statistics**:
counts and sums, never a mean of means, since batches differ in size and a
short one must not weigh like a long one. Those are the same statistics the
pooled row already used, so one accumulator now serves both, a fold read in
batches and a pooling read fold by fold being the same arithmetic. Measured
on 18 800 rows whose samples come to 92 MB: **+75 MB peak materialized
against +0 MB streamed**, the same rmse to six decimals, slightly faster.
The two tests that make it safe are the equivalence ones — the batches
concatenate to exactly what the whole call returns, and the scores agree to
1e-12 whether a fold arrives in one piece or in eighty.

One trap that test found: two consecutive `cross_validate` runs are not
comparable at all unless the seed is reset between them, because
`_fresh_variational_state` draws `alpha_white_` from the package RNG, so the
second run's fold models start somewhere else. Nothing to do with batching —
but it is what a naive equivalence test measures first.

### E1 — the measurement that settled the refit question

Walker Lake `V` (470 points), single-GP model (10×10 inducing grid,
isotropic 50, Gaussian kernel), base model trained 500 iterations, 5 spatial
folds (`spatial_k_fold` against the Walker grid, W = 5.8), RTX 4060 Ti.
Pooled out-of-fold scores; `scratch-400` rebuilds and trains a fresh model
per fold (400 iterations, hypers and warping fitted on fold data alone — the
honest gold standard). Time is the whole 5-fold loop.

| scheme       | rmse   | mae    | bias  | crps   | goodness | time    |
|--------------|--------|--------|-------|--------|----------|---------|
| in-sample    | 212.59 | 168.45 | 11.06 | 121.22 | 0.935    | —       |
| scratch-400  | 244.92 | 198.93 |  1.28 | 139.52 | 0.984    | 162.9 s |
| fresh-50     | 257.59 | 208.10 | 16.15 | 147.25 | 0.969    |  35.9 s |
| fresh-200    | 238.16 | 191.00 | 10.79 | 135.64 | 0.970    |  70.4 s |
| warm-50      | 231.98 | 186.54 |  5.41 | 131.79 | 0.979    |  35.4 s |
| warm-200     | 230.83 | 185.71 |  2.29 | 131.40 | 0.982    |  78.2 s |
| all-200      | 224.92 | 180.79 |  3.32 | 128.05 | 0.966    | 105.6 s |

Three findings:

1. **Every CV scheme sits well above in-sample** — the leakage the whole
   exercise exists to expose is exposed.
2. **Fresh-variational at 200 iterations matches the scratch gold within a
   few percent** (rmse 238 vs 245, crps 136 vs 140) at 2.3× less cost.
   50 iterations is undertrained (worse everywhere); the default budget of
   200 was right on this case. Iterations too few read as *pessimism*, so
   the failure mode is the safe direction.
3. **Warm-starting the variational state scores better than the honest gold
   itself** (rmse 231 vs 245; `refit="all"` better still at 225). A scheme
   cannot beat the honest reference by honest means: the ~3–8% edge *is*
   the residual memory of the held-out rows surviving 200 iterations of
   supposed forgetting. This is the measured answer to "how many iterations
   to forget": more than anyone would pay, which is why the default
   re-initializes instead — fresh init is structurally ignorant, and the
   question dissolves.

## The variogram: `prepare.variogram`, drawn by both backends

The experimental semivariogram of the measurements, with one thin curve per
realization computed **on the same pairs** — the spatial-continuity check the
marginal figures (`accuracy`, `spread_check`) cannot make. A kernel too
smooth sags below the data's curve at short lags; a nugget fitted into the
range lifts it there. Omnidirectional or along a named direction (the
anisotropy ellipsoid's principal axes are the ones worth asking);
`residuals=True` gives the variogram of `measured − predicted` (fan dropped),
honest on the OOF container. The pair budget is met by deterministic
striding; realizations are read one column at a time, never the store whole.
`metrics.variogram_score` (Scheuerer & Hamill 2015, p = 0.5) is the numeric
side: the proper score over pairs that catches an ensemble with the right
histograms and the wrong spatial structure. Given `coordinates` it
declusters its pairs the same way the figure does, but it cannot correct the
noise footing — `|difference| ** p` is not a second moment, so there is no
constant to add and the alternative is a draw, which would put a seed inside
a metric. It is therefore an estimate that never reaches zero, to be read as
a ranking between models on the same data; the figure is what to reach for
when the size of the disagreement matters.

## The calibration: `models.conformalize` / `ConformalCalibration`

Split conformal on the out-of-fold PITs `cross_validate` leaves behind as
metadata (`pit_<variable>[_<component>]` — where each assay fell inside its
own out-of-fold predictive distribution, mid-rank on ties). The score is
`|u − ½|`; `nominal(q)` is twice its `⌈(n+1)q⌉`-th smallest value — the level
to cut a central interval at so it covers a share `q` of fresh measurements,
with the standard finite-sample guarantee. The honest limits, stated where
they belong:

- **Exchangeability** fails point-by-point on spatial data; folds that mimic
  deployment (`spatial_k_fold`) are what the guarantee is worth.
- The intervals are of **measurements** — the ground is never observed, so
  there is nothing to calibrate a ground interval against.
- The repair is **bounded by the ensemble's own range**: `nominal(q) == 1.0`
  means the model was too sure for its samples to say how much wider the
  interval must be (measured: a ×0.5 overconfidence with 400 samples
  saturates at the sample range and stalls at ~0.86 coverage; ×0.7 with
  1000 samples repairs to 0.90 exactly). Raise `n_sim`/`n_nodes`, or fix
  the model rather than the interval.

## What was deliberately not built

- **PSIS-style importance-weighted LOO**: degrades exactly when a whole hole
  (a correlated group) is deleted, and pointwise LOO is the leaky scheme
  being avoided.
- **Per-sample fold weights** (`sample_weight`): the overfitting channel the
  discrete rewrite exists to close.
- **Categorical rows in the score table**: a categorical likelihood has no
  measurement distribution; subset the OOF container by fold and use the
  variable's own `compute_metrics`.
- **True-scratch `refit` mode inside the driver**: `load_model` replays
  constructors and then restores trained parameters, so honest scratch
  needs user-built models (as E1 did by hand); the driver documents the two
  tiers it owns.

## Tests

`geoml/test/test_cross_validation.py` (18): the fold builder (beats random on
W, atoms never split, deterministic, refusals), the driver (every row
out-of-fold, table pooled, held-out worse than in-sample, original model
untouched, fresh state ignorant with everything else frozen), and the
calibration (planted overconfidence and hedge repaired to nominal coverage
on data the calibration never saw, calibrated forecasts left alone,
monotone and saturating). Variograms in `test_plots.py` /
`test_interactive.py` (worked example, iid flat vs smooth rising,
directional pair filter, fan = per-realization curves, residual mode,
column-at-a-time watchdog); scores in `test_metrics.py` (CRPS against the
Gaussian closed form and propriety; variogram score against a worked
example and broken-structure detection).
