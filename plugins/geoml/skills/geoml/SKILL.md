---
name: geoml
description: >
  Working knowledge of the geoML Python package (github.com/italo-goncalves/geoML):
  variational Gaussian processes for spatial data, implicit geological modelling,
  block models, drillhole data, compositional and categorical variables, warping and
  likelihoods, inducing points and local experts, cross-validation and calibration.
  Use this skill for any task involving geoML -- writing modelling code or notebooks,
  reading its output, choosing a kernel/warping/likelihood, or navigating the
  package. It says what to build and why; the installed package's catalogue says
  exactly what each class accepts.
---

# geoML

How to use the geoML package: the object model, the workflow, the
recommendations and the geostatistical intuitions its API takes for granted.
It says *what to build and why*. It does not list arguments: the installed
package answers that exactly, through its catalogue (§0). Where anything here
disagrees with the code, trust the code.

**Writing a notebook or example code? Read `references/notebook-style.md`
first**, for the import aliases, the workflow arc, the plotting conventions
and the tree paths. **Worked examples are the manual's chapters**, in
`references/manual/`, each run against the package at every release (§4).

---

## 0. Looking things up: the catalogue

Every class and function a model or a script may use is described by the
installed geoML itself: arguments, defaults, bounds, how a node's size
follows from its arguments, which parents it takes, and how stable it is.
Ask it rather than guessing, and before writing any constructor call:

```bash
python -m geoml.catalogue --show BoxCox          # one entry, as text
python -m geoml.catalogue --show BasicGP --json  # the same, as JSON
python -m geoml.catalogue --list                 # the categories
python -m geoml.catalogue --list warping         # one category's entries
```

A short name that means two things (`Gaussian` is a kernel and a likelihood)
lists the dotted candidates instead; pass the dotted one. The answer is for
the version installed where the command runs, which is the version the code
will run against.

**Stability.** Every entry is one of three:

- **public**: use freely.
- **experimental**: works and is tested, but its interface or its
  recommendation may still change. Say so when you use one.
- **internal**: kept so that old saved models load. Never build a new model
  with one; `--list` leaves them out.

Where geoML is not installed, the same descriptions are on the reference
pages of the documentation site, https://italo-goncalves.github.io/geoML/reference/.

---

## 1. The package

**Install:**
```bash
pip install git+https://github.com/italo-goncalves/geoML
```
**Backend:** TensorFlow 2.x + TensorFlow-Probability, GPU-accelerated. All
computation is in `float64`: geostatistical matrices are ill-conditioned and
`float32` breaks the Cholesky factorizations. Realizations are *stored* as
`float32` (half the disk, every read widened back to `float64`);
`geoml.set_realization_dtype("float64")` keeps them wide.

**License:** GPL-3 (dual-licensed; see README). **Version:** 0.9.0.

**Layout:** five subpackages (`data`, `latent`, `math`, `stats`, `viz`) plus
the older `plots`, around modules left flat on purpose: `models`,
`likelihood`, `kernels`, `transform`, `warping`, `parameter`, `datasets`,
`metrics`, `persistence`, `storage`. A saved model records the dotted path
of every class in it and imports it by that path when it loads, which is
why those modules never move.

### 1.1 The object model

- **Containers hold data and predictions.** `PointData`, the grids
  (`Grid1D`/`Grid2D`/`Grid3D`), the variable-size block model `BlockSet3D`,
  and triangulated meshes (`Surface3D`, `Solid3D`). A container holds
  *variables* (continuous, vector, compositional, categorical), and each
  variable holds *columns*: measurements, a prediction, variances,
  quantiles, realizations.
- **Everything in a container is reached by tree path.**
  `container.values("V/prediction")` is a numpy array,
  `container.get("V/quantiles/0.5")` the column itself (for `as_image()` or
  `get_contour()`), `container.tree()` the picture of what is there.
  `_metadata/...` holds per-location facts the model never reads: hole ids,
  depths, folds, filters. Paths are in `references/notebook-style.md` §0.
- **A model is a latent network plus one likelihood per variable.**
  `models.VGPNetwork` is the core. The network is built from `latent` nodes:
  an input node carrying the inducing points, GP nodes, and operations that
  combine them. Its **leaves** are where the likelihoods attach: pass one
  leaf per likelihood, or name them together with
  `variables={"Rock": likelihood, ...}`. A likelihood carries a **warping**,
  the monotone map from the data to the Gaussian scale the model works on.
- **Predictions are written into the target container, not returned.**
  `model.predict(grid)` fills the grid's columns. `grid.unpredicted()` names
  what no prediction reached yet, so a cancelled or partial prediction is
  finished with `model.predict(grid, where=grid.unpredicted())`.
- **Long calls report and can be cancelled.** Inside
  `with geoml.progress(callback):` training, prediction, refinement,
  cross-validation and mesh sets report each unit they finish; the callback
  raising cancels the call.
- **Everything trainable is a `Parametric`** holding `RealParameter`s,
  constrained through their transforms and trained by Adam. Bounds a
  constructor clips to are in the catalogue.

### 1.2 Worked example (Jura, categorical)

```python
import geoml
import geoml.latent as gl, geoml.transform as tr
import geoml.kernels as kr, geoml.likelihood as lk

geoml.set_seed(1234)                      # BEFORE constructing anything
jura_train, _ = geoml.datasets.jura()
labels = list(jura_train.get("Landuse").labels)

network_input = gl.BasicInput(
    inducing_points=jura_train, transform=tr.Isotropic(0.5))
network_output = gl.BasicGP(
    parent=network_input, size=len(labels), kernel=kr.Matern32(),
    fix_range=True)

model = geoml.models.VGPNetwork(
    data=jura_train, variables="Landuse", latent_network=network_output,
    likelihoods=lk.CategoricalGaussianIndicator(n_components=len(labels)),
    options=geoml.models.GPOptions(verbose=False))
model.train_full(5)

grid = geoml.data.Grid2D(start=[0, 0], end=[6, 6], n=[21, 21])
model.predict(grid)                       # writes into the container
image = grid.get("Landuse/predicted").as_image()
```

`model.to_dot()` renders the network as a diagram. The full modelling arcs
are the case-study chapters: Walker Lake (15), Jura with a non-stationary
multivariate network (16), and a folded quartz vein in 3D (17).

### 1.3 Where things live

| Module | For |
|---|---|
| `data` | the containers and variables, meshes, block models, drillholes (`DrillholeData`, converted to points and never fed to a model), Zarr storage, and `data.inducing` for building inducing points |
| `latent` | the nodes a network is built from |
| `likelihood`, `warping` | the observation model and the map to the Gaussian scale |
| `kernels`, `transform` | covariance functions, and the transforms of the input they read (anisotropy, projections, faults) |
| `models` | `VGPNetwork`, `GPOptions`, and the workflows as free functions: `refine`, `cross_validate`, `search_throw` |
| `metrics`, `plots` | scores, and figures in two backends (`Explorer` for print, `Interactive` for plotly) |
| `datasets` | the bundled data (§3) |

`python -m geoml.catalogue --list CATEGORY` gives the classes of each.

### 1.4 Recommendations

What was measured to work, as of this version. Arguments for each class are
in the catalogue.

- **Warpings.** A single positive grade: `BoxCox` then `ZScore`; where the
  data hold zeros, give `BoxCox` a `shift` of the order of the smallest
  positive value, not the default. A variable centred on zero:
  `YeoJohnson` then `ZScore`. A vector of grades: `BoxCox`, `RobustPCA`,
  `ZScore`, `SinhArcsinh`, `ZScore`, chained with `ChainedWarping`. These
  parametric links replaced `Spline`, which stays for saved models (with
  `backbone="rq"` if you use it anyway). One marginal transform is where
  the benefit stops: stacking rotation-and-spline pairs made held-out
  scores worse.
- **Noise law.** For a skewed grade under a log-like link, prefer a
  Gaussian likelihood: a heavy-tailed law (`Laplace`, `EpsilonInsensitive`)
  pushed back through the link can have an unbounded variance in data units
  and over-wide intervals, though its bound may look better. For data with
  gross errors, `likelihood.Mixture` (experimental) names the bad readings,
  with its warping led by `ZScore(robust=True)`.
- **Several populations.** Where one variable is drawn from populations that
  differ in level or skew, ore and waste say, use
  `likelihood.LikelihoodMixture` over continuous likelihoods, each with its
  own warping. With `shares="latent"` the shares change from place to place:
  read them from a GP node of their own, apart from the populations', and
  where a domain was logged let a `CategoricalGaussianIndicator` with
  `bias=True`, trained on every logged interval rather than the assayed ones
  alone, read the same node. On one deposit it beat a single likelihood on
  every metal out of fold, but its intervals came out narrower than the
  held-out data supported; check them.
- **Inducing points.** The data's own locations plus a regular backbone,
  divided into overlapping experts: `data.inducing.experts` over
  `data.inducing.combine`. Where the survey does not fill its box -- a fan
  of drillholes -- take the backbone from `data.inducing.from_hull`, which
  keeps the lattice inside the data's hull and a margin around it. Experts
  come out compact and share points with every neighbour once each borrows
  at least as many points as it has neighbours: keep experts large enough
  for their overlap (an overlap of 0.1 on 25-point experts cannot reach six
  neighbours). How many points a model can absorb depends on the whole
  configuration; measure it on held-out data rather than carrying a number
  across problems.
- **Structure.** One variable influencing another: `Linear` off the first's
  field, added to the second's GP through `LinearCombination`, with
  `unit_norm=False` so the model can decline the influence. A field that is
  not stationary: `GPWalk` moves the coordinates and a stationary kernel
  reads the moved space. Build the GP reading the walk with
  `isotropic=True`: a range per direction there is a second description
  of the anisotropy, which belongs in the input's transform. Independent parts of a model can sit on
  independent trees, one leaf each. Chapter 16 builds the first two, one
  leaf per variable.
- **Depth.** Since 0.9.0 a GP node reading another node's uncertain output
  averages its kernel over it, the covariance between locations included
  (the expected kernel, `GPOptions(propagation="joint")`, the default); a
  model saved before keeps the old rule, which made deep networks
  overconfident (`propagation="marginal"`). Build a deep network with the
  coordinates concatenated beside the inner GP (`Concatenate(root,
  inner)`). A GP on an uncertain input takes the Gaussian, exponential,
  Matérn or rational quadratic kernel; spherical and cubic are refused
  there, and the error message says so. On a `GaussianInput` a `BasicGP`
  takes the mixture's moments (the variance of the mean over the input
  included) and the likelihood integrates what its realizations leave
  out; `UncertainInputGP` is deprecated. A location error is still better
  left untold: a noise term absorbs it. That variance costs time and
  memory in proportion to the inducing points; experts do not shorten it,
  `train_by_expert` holds a fraction of it in memory.
- **Training.** `GPOptions(training_tolerance=0.01)` stops once the bound
  has settled; the last few percent of the bound buys sharpness held-out
  data does not support.
- **Many experts.** When memory grows with their number, train and predict
  an expert at a time: `model.train_by_expert`, `model.predict_by_expert`,
  and `refine(..., by_expert=True)` for a block model. It matched or beat
  `train_svi` on CRPS and on a deposit's rock types in a third to a half of
  the device memory, a little behind on rmse on a smooth synthetic field.
  Fewer, larger experts scored better than many small ones at a fixed
  total of inducing points. A deep GP fed by another needs the coordinates
  concatenated into its input to stay local; one fed by a GP alone spreads
  every expert over the field. Five nodes are refused; the message names
  them.
- **Validation.** `PointData.spatial_k_fold` writes folds that resemble the
  real prediction task, `models.cross_validate` scores the model out of
  fold, and `models.conformalize` recalibrates the intervals. In-sample scores
  flatter; only out-of-fold ones speak for the ground between samples.
- **Block models.** A `BlockSet3D` refined by `models.refine`: predict
  coarse, split only the blocks the prediction's surface runs through (a
  block the model is merely unsure about is left whole), predict what the
  split made. `MeshSet` contours a column for the prediction and every
  realization at once, with volumes measured as it goes.

### 1.5 Theory → code

The papers and the code name the same things differently.

| Paper concept | In the code |
|---|---|
| Inducing points $\mathbf{T}$, $\mathbf{u}$ | the container given to `BasicInput(inducing_points=…)`, read by the GP nodes above it |
| DGP uncertainty propagation | each node hands on a mean, a variance and its covariance with the inducing points; there is no separate class |
| Expected kernel | inside the GP nodes, under `GPOptions(propagation="joint")` |
| Paciorek kernel | the propagation before 0.9.0, kept for older saves as `propagation="marginal"` |
| SDE node | `latent.GPWalk` |
| Local experts | overlapping inducing sets from `data.inducing.experts` |
| CLR transform | `warping.CenteredLogRatio` |
| PCA after CLR | `warping.PCA` / `warping.RobustPCA` |
| ε-insensitive likelihood | `likelihood.EpsilonInsensitive` |
| Boundary / contact likelihood | `likelihood.CategoricalGaussianIndicator` and its hierarchical twin |
| Structural / dip-strike field | `latent.GradientConstrainedInput` with directional data |
| Warped GP | a warping chain on the likelihood (§1.4) |
| Multivariate weight matrix $\mathbf{M}$ | `latent.LinearCombination` |

### 1.6 Gotchas

- **Reproducibility:** call `geoml.set_seed(seed)` *before constructing
  anything*. It is the only knob: parameter initialization and a model's
  simulation stream both draw from it, and a saved model keeps its seed.
  `cross_validate` draws each fold's fresh variational state from the same
  generator, so two runs in a row differ; set the seed right before each
  run that must repeat. Under `GPOptions(jit_predict=True)` XLA draws other
  normals from the same seed: the latent moments agree, the realizations do
  not.
- **`values()` for what you compute with, `get()` for what you draw.**
  Never `values()` a bare `simulations` path: it reads every realization
  into memory at once, fatal on a block model. Read one realization
  (`"V/simulations/7"`) or reduce in row bands.
- **Open a store you only read with `mode="r"`.** A container's `open`
  defaults to `"r+"`, and a prediction into it writes into the store.
  Read-only, a write is refused and the store stays as it was; `to_zarr`
  never writes over the store a container reads from.
- **The stored realizations are of the ground; a measurement scatters
  around it.** Comparing a prediction with an assay needs
  `model.predict_measurements`, not the stored realizations.
- **A node inside the tree is predicted with `model.predict_node`**, into
  a `LatentVariable` on the latent scale: what a `GPWalk` did to the
  coordinates, what a trend adds. Points only, never a block model; its
  realizations match the leaf's through operation nodes, not across a GP.
- **Do not build models in a loop and expect the memory back.** TensorFlow
  keeps a trained model's graph machinery after the model is gone.
  `cross_validate` swaps data into one model for that reason; do the same.
- **Scripts use bare aliases** (`import geoml.latent as gl`); package source
  uses underscore-prefixed ones. Never mix the two.

---

## 2. Common intuitions

- **Kriging = GP posterior mean**, exactly. The VGP generalizes it to
  non-Gaussian likelihoods.
- **Inducing points are pseudo-data summarizing the real data.** More is a
  better approximation and slower training.
- **The ELBO's KL term is automatic regularization**: it keeps the model
  from fitting noise.
- **The ε-insensitive likelihood is the GP analogue of SVM regression.**
- **A warping maps a Gaussian latent field to a non-Gaussian marginal.** The
  GP still models a Gaussian field.
- **Non-stationarity in geology often comes from geometry** (folds,
  faults). Moving the input space makes a stationary kernel non-stationary
  in the original space.
- **In implicit modelling the sign of the potential field matters, not its
  magnitude.** A contact constrains the field to cross a threshold.
- **Compositional data live on a simplex.** A log-ratio transform maps them
  to real space; back-transforming must restore the closure.
- **A held-out score measured at sampled locations cannot speak for the
  ground between them.** A regular backbone of inducing points is there for
  the map, and a validation set from the same campaign cannot notice it.

---

## 3. Datasets

`geoml.datasets` bundles `walker()`, `jura()`, `ararangua()`, `andrade()`,
`arctic_lake()`, `example_fold()` and `sunspot_number()` (which downloads
from sidc.be). `macpass(path)` reads a drillhole database the user
downloads themselves. The research line also used data that are not
bundled: the Passo Feio dip/strike data, a quartz-vein drillhole set, the
Zhang gold data and the Thalanga VHMS assays.

---

## 4. The manual

`references/manual/` holds the manual's chapters, copied at each release
from the repository, where every code block in them is run. Read the one a
task touches before writing code for it; the case studies are the complete
arcs.

| Chapter | Covers |
|---|---|
| `01-why-another-geostatistics.md` | what the approach offers over kriging |
| `02-the-gp-is-kriging.md` | the GP posterior as kriging |
| `03-inducing-points-and-the-elbo.md` | inducing points, experts, the bound |
| `04-warpings-likelihoods-and-the-gaussian.md` | warpings and likelihoods |
| `05-latent-networks.md` | building networks, deep models |
| `06-categories-and-boundaries.md` | categorical variables and contacts |
| `07-simulation.md` | realizations and derived variables |
| `08-change-of-support.md` | blocks and support |
| `09-from-database-to-data.md` | drillhole databases to point data |
| `10-containers-and-addressing.md` | containers and tree paths |
| `11-building-and-training.md` | constructing and training a model |
| `12-prediction-blocks-and-surfaces.md` | predicting into grids, blocks and meshes |
| `13-validation.md` | folds, cross-validation, calibration |
| `14-reporting.md` | grade-tonnage, swaths, figures |
| `15-case-study-walker-lake.md` | a positive grade in 2D |
| `16-case-study-jura.md` | a non-stationary multivariate model with rock types |
| `17-case-study-quartz-vein.md` | implicit modelling of a folded vein in 3D |

Figure links point at the documentation site. Chapter 17 reads two CSV files
from `docs/manual/data/` in the repository.
