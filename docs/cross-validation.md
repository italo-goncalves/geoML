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

## The driver: `models.cross_validate(model, folds, refit, iterations)`

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
histograms and the wrong spatial structure.

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
