---
name: geo-ml
description: >
  Working knowledge of the geoML Python package (github.com/italo-goncalves/geoML):
  variational Gaussian processes for spatial data, implicit geological modelling,
  block models, drillhole data, compositional and categorical variables, warping and
  likelihoods, inducing points and local experts, cross-validation and calibration.
  Use this skill for any task involving geoML -- writing modelling code or notebooks,
  reading its output, choosing a kernel/warping/likelihood, or navigating the module
  layout. Also covers the geostatistical intuitions the API assumes.
---

# geoML

Working knowledge of the geoML package: the object model, the module layout,
the workflow, and the geostatistical intuitions its API takes for granted.
Written for the version named below -- check `geoml.__version__` if something
here disagrees with the code, and trust the code.

**Writing a notebook or example code? Read `references/notebook-style.md`
first.** It carries the import aliases, the workflow arc, the plotting
conventions and the verified tree-path table for reaching variables and
attributes.

---

## 1. The package

**Install:**
```bash
pip install git+https://github.com/italo-goncalves/geoML
```
**Backend:** TensorFlow 2.x + TensorFlow-Probability, GPU-accelerated. All
computation is in `float64` — geostatistical matrices are ill-conditioned and
`float32` breaks the Cholesky factorizations.

**License:** GPL-3 (dual-licensed; see README). **Version:** 0.6.6.

**Layout:** since 0.6.0, `geoml/` is five subpackages — `data/`, `latent/`,
`math/`, `stats/`, `viz/` (plus the older `plots/`) — around a set of modules
deliberately left flat: `models`, `likelihood`, `kernels`, `transform`,
`warping`, `parameter`, `datasets`, `metrics`, `persistence`, `storage`. No
`src/` layout. The first six are pinned flat on purpose: **saved models replay
dotted import paths**, so `persistence._resolve` reaches for
`geoml.likelihood.Gaussian` by name when a model is loaded. A class recorded
in a save must keep its old path importable *forever*, not just politely —
which is why one-line shims still sit at every pre-0.6.0 path
(`geoml/tftools.py`, `geoml/drillhole.py`, `geoml/random.py`, …) and the
subpackage facades re-export everything the old flat modules held.

### 1.1 The object model

- **Everything trainable subclasses `parameter.Parametric`**, which holds a
  `parameters` dict and a flat `_all_parameters` list. Composite objects call
  `self._register(child)` to bubble a child's parameters up, and
  `self._add_parameter(name, RealParameter(...))` for their own. Training reads
  `get_unfixed_variables()` and optimizes with Adam. All training is
  gradient-based.
- **`RealParameter` wraps a `tf.Variable`** with a transform plus min/max
  bounds, kept in range by `refresh()`. Constraints are encoded *in the
  transform*, not by projection: `PositiveParameter` (log),
  `CompositionalParameter` (softmax/logit), `CircularParameter`,
  `OrthonormalMatrix`.
- **`VGPNetwork` is the heart of the package.** You compose a network from
  `latent` nodes, attach one `likelihood` per variable, and optionally a
  `warping`. Prefer it over the legacy closed-form `GP` for all new work.

### 1.2 Worked example (Jura, categorical)

Taken from `geoml/test/test_real_cases.py`, which runs green:

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
    options=geoml.models.GPOptions())
model.train_full(5)

grid = geoml.data.Grid2D(start=[0, 0], end=[6, 6], n=[21, 21])
model.predict(grid)                       # writes into the container
grid.get("Landuse/predicted").as_image()
```

Predictions are written *into the target container*, not returned. Reach them
by tree path — `get(path)` for an attribute's own methods, `values(path)` for
a numpy array; see `references/notebook-style.md` §0. `model.to_dot()` renders
the network as a diagram.

### 1.3 Worked example (Jura, non-stationary and multivariate)

The pattern to reach for when one variable should influence another and the
field is not stationary. From `references/08_Jura_non_stationary.ipynb`,
which is verified end to end.

```python
import geoml
import geoml.latent as gl, geoml.transform as tr
import geoml.kernels as kr, geoml.likelihood as lk, geoml.warping as wp

geoml.set_seed(1234)
jura, jura_validation = geoml.datasets.jura()
elements = list(jura.get("Elements").labels)
rocks = list(jura.get("Rock").labels)

# the data's own locations plus a regular backbone, split into experts
inducing = geoml.data.inducing.experts(
    geoml.data.inducing.combine(
        jura, geoml.data.Grid2D(start=[0, 0], n=[21, 21], end=[6, 6])),
    4)
net_input = gl.BasicInput(inducing_points=inducing,
                          transform=tr.Isotropic(0.05))

# an SDE moves the coordinates: a stationary kernel in the moved space is
# non-stationary in the real one
field = gl.BasicGP(net_input, size=2, kernel=kr.Gaussian())
coords = gl.GPWalk(field, n_steps=5)
cat = gl.BasicGP(coords, size=len(rocks), kernel=kr.Matern32())

# how the geology reaches the grades: a trainable map off the categorical
# fields, *added* to the numerical ones
trend = gl.Linear(cat, size=len(elements), unit_norm=False)
num = gl.BasicGP(net_input, size=len(elements), kernel=kr.Spherical())
net_out = gl.Concatenate(cat, gl.LinearCombination(trend, num))

likelihoods = [
    lk.CategoricalGaussianIndicator(len(rocks)),
    lk.Laplace(warping=wp.ChainedWarping(
        wp.Log(len(elements)),                       # non-negativity
        wp.RobustPCA(len(elements), len(elements)),  # decorrelate
        wp.Spline(len(elements), knots_per_arm=5),   # asymmetry
        wp.ZScore(len(elements)))),
]

model = geoml.models.VGPNetwork(
    data=jura, variables=["Rock", "Elements"], likelihoods=likelihoods,
    latent_network=net_out,
    options=geoml.models.GPOptions(prediction_batch_size=1000, jitter=1e-6))
model.train_full(250)
```

Four things in that network are worth carrying to other problems:

- **`Linear` → `LinearCombination` is how one variable influences another**,
  and the influence is *additive*. The alternative — making the categorical
  node a **parent** of the numerical one, `BasicGP(Concatenate(root, cat))` —
  also works and is a genuine deep GP, but it forces the grades to read the
  rock fields through the uncertainty propagation, which is a much stronger
  commitment than adding a trend.
- **`unit_norm=False` on the `Linear` is deliberate**: it lets the mixing
  weights shrink towards zero, so the model can conclude that the geology
  says nothing about a particular element. With the unit-norm constraint it
  cannot decline the influence.
- **`GPWalk` is what makes it non-stationary.** An inner GP of size 2 moves
  the coordinates over a few steps, and the categorical fields are modelled
  in the moved space. The numerical fields read the *unmoved* input here —
  the two need not share a geometry.
- **The input transform starts small** (`Isotropic(0.05)`) because the walked
  coordinates do the long-range work. Do not copy a range across from a
  stationary model.

Measured on Jura's 100 held-out sites: the metals come back at 0.91 times
their own standard deviation and rock balanced accuracy at 0.72, against
0.95 and 0.67 for a flat stationary model on the same data. The two differ
in more than the network — likelihood, warping chain and inducing set all
change — so read that as the recipe being better rather than as an isolated
effect of any one piece. 250 iterations is enough; 500 measured identically.

### 1.4 Module map

| Module | Role |
|---|---|
| `parameter.py` | `Parametric` + the `RealParameter` family. The foundation. |
| `models.py` | `GP` (legacy, closed-form), **`VGPNetwork`**, `StructuralField`, `GPEnsemble`, `ProjectedVGP`, `Normalizer`, `GPOptions`. Plus two **free functions** rather than methods, each being a workflow rather than something a model *is*: `refine(model, blocks)` drives the block-model passes — predict coarse, ask which blocks the model cannot decide, cut only those, predict only what the cut made, stopping at the lattice's `max_levels`; and `cross_validate(model, folds="fold")` is the VGP translation of kriging's fixed-variogram CV — per fold it rebuilds around the reduced data, **re-initializes the variational state** while freezing the rest (a fresh init is structurally ignorant of the held-out rows; warm-starting measured 3–8% better than the honest scratch gold, which is memory rather than skill), refits briefly, and predicts only the held-out rows into one shared out-of-fold container. Its PITs land as `pit_*` metadata for `conformalize`. |
| `latent/` | `network.py` holds the nodes and the facade re-exports them, so `geoml.latent.BasicGP` is what saves replay: `BasicInput`, `BasicGP`, `AdditiveGP`, `Add`, `Multiply`, `Linear`, `LinearCombination`, `ProductOfExperts`, `Exponentiation`, `Bias`, `Scale`, `RadialTrend`, `GPWalk`, `MultiStructureGP`, `Stack`, `Concatenate`, `SelectInput`, `GradientConstrainedInput`. `fourier.py` (the old `projnet.py`) is dark on purpose — unadvertised, untested, reachable only as `geoml.latent.fourier`. |
| `likelihood.py` | `Gaussian`, `Laplace`, `Gamma`, `StudentT`, `EpsilonInsensitive`, `Huber`, `Mixture` (0.6.3 — the noise as several scales of one family, for contaminated data), their `Multivariate*` twins, `Bernoulli`, `BernoulliMaximumMargin`, `CategoricalGaussianIndicator`, `HierarchicalGaussianIndicator`, `OrderedGaussianIndicator`, `GradientIndicator`. Since 0.6.3 there is **no scalar/multivariate split** — one `_ContinuousLikelihood` serves any size, the twins being thin subclasses that only size the default warping; what differs is decided by `warping.elementwise`. |
| `kernels.py` | `Gaussian`, `Spherical`, `Exponential`, `Cubic`, `Constant`, `Cosine`, `Matern32`, `Matern52`, `RationalQuadratic`, plus `Covariance` composition (`Sum`, `Product`, `Scale`, `Linear`). |
| `transform.py` | Input transforms: `Isotropic`, `Anisotropy2D/3D` (+`Math`/`Dynamic`), `AnisotropyARD`, `ProjectionTo1D`, `Periodic`, `RandomProjections`, `ChainedTransform`, `BellFault2D`. |
| `warping.py` | Output warpings: `Spline`, `ZScore`, `Softplus`, `Log`, `Sigmoid`, `PCA`, `RobustPCA`, **`CenteredLogRatio`**, `ScaledSimplex`, `Rotation`, `ContinuousNormalizingFlow`, `ChainedWarping`. |
| `data/` | The containers, split since 0.6.0 but re-exported by `data/__init__.py`, so `geoml.data.X` always resolves: `base.py` (errors, `BoundingBox`, `VariablePath`, the tree traversal), `variables.py`, `containers.py` (`PointData`, `DirectionalData`, `Section3D`), `grids.py` (`Grid1D/2D/3D`), `meshes.py` (`Mesh3D` primitive with `Surface3D`/`Solid3D` siblings and `DTM3D`; DXF I/O; booleans), `blocks.py` (`BlockSet3D`, the variable-size block model), `io.py` (Zarr), `drillhole.py`, `inducing.py`. `assign_from_surface`/`assign_from_solid` tag locations above/below a sheet or inside a body. Largest area (~4500 lines in blocks alone). |
| `data/drillhole.py` | `DrillholeData` + `IntervalTable`: minimum-curvature desurveying, compositing, conversion to `PointData`. Never fed to a model — only converted. |
| `data/inducing.py` | Builds the inducing points: `from_kmeans`, `from_grid`, `combine` for one set; `grid_experts`, `experts` for one per expert. Divides inducing points, not data. |
| `datasets.py` | `walker()`, `jura()`, `ararangua()`, `andrade()`, `arctic_lake()`, `sunspot_number()`, `example_fold()`, plus `macpass(path)` for a database the user downloads. |
| `storage.py` | `ArrayStore` — every array a container holds, NumPy in RAM or chunked Zarr on disk by size. `row_quantiles`/`row_cdf`/`row_bands` reduce without materializing. |
| `persistence.py` | Whole-model save/load by replaying recorded constructor calls. Containers use `to_zarr`/`open`. |
| `stats/random.py` | The package RNG behind `geoml.set_seed` (shim at `geoml/random.py`). |
| `stats/probability.py` | Custom TFP distributions (`EpsilonInsensitive`, `Huber`, spline/empirical). |
| `math/` | `geometry.py` (rotations/angles, the sub-block lattice arithmetic, triangulated-mesh maths — arrays in, arrays out), `tf.py` (the four helpers in everyday use), `linalg.py` (seventeen solvers parked for future work, imported by nothing today), `interpolate.py` (TF interpolators). |
| `metrics.py` | `rmse`/`mae`/`bias`, sample-based `crps`, `coverage`/`goodness`, `variogram_score`, `interval_score`. |
| `plots/` | EDA and model figures in two backends: `Explorer` (matplotlib) and `Interactive` (plotly), plus a linked `Dashboard`. `prepare.py` holds the arithmetic and imports no plotting library. |
| `viz/` | `graphviz.py` (DOT diagrams), `plotly.py`, `pyvista.py` (visualization export). |

### 1.5 Theory → code

The papers and the code frequently use **different names for the same thing**.
This table is the bridge:

| Paper concept (§) | In the code |
|---|---|
| Inducing points / $\mathbf{T}$, $\mathbf{u}$ | `BasicInput(inducing_points=…)` feeding a `BasicGP` |
| DGP uncertainty propagation | the `x_var` argument threaded through every node's `predict`/`propagate`/`interpolate` in `latent.py` — each node takes *and returns* a mean and a variance |
| **Paciorek kernel** | **no class of that name exists.** The non-stationary maths is embedded in the node propagation above; do not send anyone looking for `kernels.Paciorek` |
| SDE node | `latent.GPWalk` |
| Local experts | `models.GPEnsemble` |
| CLR transform | `warping.CenteredLogRatio` |
| PCA after CLR | `warping.PCA` / `RobustPCA` |
| $\epsilon$-insensitive likelihood | `likelihood.EpsilonInsensitive` (+ `Multivariate…`) |
| Boundary / contact likelihood | `likelihood.CategoricalGaussianIndicator` and the `…GaussianIndicator` family |
| Structural / dip-strike field | `models.StructuralField`, `likelihood.GradientIndicator`, `latent.GradientConstrainedInput` |
| Warped GP | `warping.Spline` + `Softplus`, attached to the model |
| Multivariate weight matrix $\mathbf{M}$ | `latent.LinearCombination` |

### 1.6 Gotchas

- **Reproducibility:** call `geoml.set_seed(seed)` *before constructing the
  objects*. Parameter initialization draws from the package RNG in
  `geoml/stats/random.py` — **not** from the global NumPy/TF RNG. Seeding
  after construction does nothing. Since 0.6.2 this is the *only* knob:
  a model's options draw their own `seed` (training's Monte Carlo, the
  simulation stream) from the same generator at construction, so
  `GPOptions(seed=…)` is a `TypeError` now and a saved model keeps the number
  it drew.
- **Imports inside the package** use underscore aliases (`import numpy as _np`,
  `import geoml.parameter as _gpr`). Match this in module code; test scripts use
  a bare `import geoml`.
- Every source file carries the GPL-3 header. **`setup.py` is gone** since
  0.6.0 — packaging is `pyproject.toml` (PEP 621) and subpackages are
  discovered automatically, so a new module needs nothing registered. The
  public surface is still curated by hand: `geoml/__init__.py`'s `__all__`
  names the user-facing modules, and internals (`parameter`, `persistence`,
  `storage`) plus the deprecation shims stay importable but unadvertised.
- **Tests:** `geoml/test/` is a real pytest suite (1006 tests, 33 files,
  ~7 min). Run **only the files covering what changed** — the full run is for
  a release or the last piece of a change that reaches every container. The
  bulk of the time is `test_real_cases.py` and `test_vgp.py`, which train
  actual models. geoML is usually not pip-installed, so run from the repo
  root. CI runs the structural tier (~600 tests, no training) on every push
  and the full suite on release tags.
- Rendering with pyvista can fail on an old conda env under WSL for reasons that
  look like X11 but are not; see `docs/wsl-pyvista-rendering.md` in the repo.

Many worked examples are in the `references` folder next to this file.

---

## 2. Common intuitions and "folk knowledge"

These are the intuitions that are rarely stated explicitly but underlie the work:

- **Kriging = GP posterior mean** (exactly). The VGP generalizes this to non-Gaussian likelihoods.
- **Inducing points ≈ pseudo-data that summarize the real data.** More inducing points = better approximation, slower training. Place them where data is most informative.
- **The ELBO's KL term is automatic regularization.** It prevents the model from fitting noise.
- **ε-insensitive likelihood is the GP analog of SVM regression.** It is more robust than the Gaussian likelihood in the presence of outliers, without requiring data transformation.
- **Warping = applying a monotonic function to the latent field** to match a non-Gaussian marginal. The GP still models a Gaussian field; the warping maps it to the observed space.
- **Non-stationarity in geology often comes from geometric complexity** (folds, faults). The DGP/SDE node "warps" the input space so that a stationary kernel in the warped space is non-stationary in the original space.
- **The Paciorek kernel is the unique kernel that allows analytical uncertainty propagation** through a GP layer. This is the key mathematical innovation of the DGP paper.
- **In implicit modeling, what matters is the sign of the potential field**, not its magnitude. A contact point constrains the field to cross a specific threshold.
- **Compositional data (Cu-Pb-Zn) lives on a simplex.** The CLR transform maps it to $\mathbb{R}^D$ where standard GP assumptions apply. Back-transforming requires care to enforce the closure constraint.
- **The number of inducing points $U$ is the main hyperparameter.** For smooth functions, $U \approx 100$–$500$ suffices. For rough/high-frequency patterns, $U \to N$ is needed.
- **How many inducing points a model can absorb is a property of the whole configuration, not a number to carry between problems.** Measured on Jura (259 samples, 100 held out): a stationary model with a Gaussian likelihood degraded from 0.96 to 1.13 times the data's own standard deviation when its inducing set doubled from 81 to 169, the extra capacity going into interpolating its own samples. The non-stationary network of §1.3, with a Laplace likelihood, is *flat* from 259 to 1220 — five times the points, five and a half times the training, no measurable change. Do not port a count across a change of network or likelihood; measure it on held-out data for the configuration you intend to ship.
- **A held-out score measured at sampled locations cannot speak for the ground between them.** A regular backbone of inducing points is there for the *map*, and a validation set drawn from the same campaign is structurally unable to notice it. A number that does not move is not always a number that has looked.
- **Variogram modeling = GP training with a Gaussian likelihood.** Maximum likelihood training is more principled but requires the full $N \times N$ matrix unless sparse approximations are used.

---

## 3. Benchmarks and datasets

Standard datasets used in this line of work:
- **Walker Lake**: classic 2D synthetic dataset; used in 2018 paper
- **Jura**: 259 heavy metal measurements; multivariate; used in 2021 paper
- **Passo Feio Metamorphic Complex**: dip/strike data; used in 2020 paper
- **Quartz vein dataset**: 53 drillholes; thin folded vein; implicit modeling; used in 2022 and DGP papers
- **Zhang 2021 gold dataset**: 1260 points, Au + sedimentological covariates; used in DGP paper
- **Thalanga VHMS deposit**: 18,603 drillhole assays; Cu-Pb-Zn; used in scalable VGP paper
- **Sunspot number 2.0**: annual averages 1700–2018; time-series warped GP

**Which of these ship with the package.** `geoml.datasets` bundles `walker()`,
`jura()`, `sunspot_number()` (downloads from sidc.be, so it needs a connection),
`ararangua()`, `andrade()`, `arctic_lake()` and `example_fold()`. The Passo
Feio, quartz vein, Zhang gold and Thalanga datasets are **not** bundled — they
live only in the papers. `macpass(path)` reads a Fireweed Metals database the
user downloads themselves (559 drillholes, Ag-Pb-Zn, published under terms the
user accepts on download); it is the one loader with no test coverage.

Performance metrics used: **RMSE**, **MAE**, **bias**, visual variogram reproduction, CDF reproduction.

---
