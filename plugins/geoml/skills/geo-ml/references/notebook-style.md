# geoML notebook style

How Ítalo's geoML notebooks are written — imports, cell rhythm, the shape of a
modelling workflow, plotting habits. Distilled from the seven current tutorial
notebooks in `references/` (`01_…` – `07_…`, refreshed 2026-08-13). Follow this
when writing or extending a geoML notebook.

The notebooks in `_superseded_2026-04-26/` and the unnumbered ones (`08 …`,
`09 …`, `Inducing points demo`, `VGP_training`, `Showcase/`) are older. They
still show real cases, but where they disagree with this file, this file wins.

---

## 0. The one rule that overrides the notebooks

**Access variables and attributes by tree path.** The notebooks were written
across the transition and most still use the old attribute chains; `05` and
parts of `07` use paths. Always write the path form.

```python
# yes
v = walker.values("V/measurements")                    # ndarray
img = grid.get("V/quantiles/0.5").as_image()           # 2-D array for imshow
sim = grid.get("Elements/Cd/simulations/3").as_image()

# no — what the older notebooks do
v = walker.variables["V"].measurements.values.to_numpy()
img = grid.variables["V"].quantiles[0.5].as_image()
sim = grid.variables["Elements"].components["Cd"].simulation(3).as_image()
```

Three entry points, and the difference between them matters:

| call | returns |
|---|---|
| `container.values(path)` | a **numpy array**, decoded if the column is coded |
| `container.get(path)` | the `_Attribute` (has `.as_image()`, `.labels`, `.to_numpy()`) |
| `container.get(path).values` | an **`ArrayStore`**, *not* an ndarray — rarely what you want |
| `container.select(pattern)` | every match, e.g. `select("**/prediction")` |
| `container.tree()` | the printable picture of what the container holds |

`get(path, key)` is shorthand for a dict family: `get("V/quantiles", 0.5)` is
`get("V/quantiles/0.5")`.

**Prefer `values(path)`.** It is the default for anything you compute with —
it materializes to numpy and decodes a coded column back to its labels, so
you never have to know whether the column holds `int8` codes or floats:

```python
jura.values("Landuse/measurements_a")        # ['Meadow' 'Pasture' ...]
jura.get("Landuse/measurements_a").values    # ArrayStore holding [1 2 ...]
```

Reach for `get(path)` when you want the attribute's own methods rather than
its numbers — `as_image()` for an `imshow`, `get_contour()` for a surface,
`.labels` for a category list. Roughly: **`values()` for what you compute
with, `get()` for what you draw.**

Never `values()` a bare `simulations` path. It resolves to the 2-D store
rather than an attribute, so the call materializes every realization at once —
fatal on a block model:

```python
grid.values("V/simulations")     # the whole ensemble into RAM
grid.values("V/simulations/7")   # one realization, which is what you want
```

### Verified path table

Every row below was executed against geoML 0.6.3.

**Continuous variable** (`ContinuousVariable`, and each component of a vector one)

```
V/measurements      V/prediction        V/latent_mean      V/latent_variance
V/dispersion        V/noise_variance    V/quantiles/0.5    V/simulations/0
V/proportions/<cutoff>   V/divided/<cutoff>   V/responsibilities/<k>
```

**Vector / compositional variable** — components are child nodes

```
Elements                    -> the VectorVariable; .labels lists the components
Elements/uncertainty        Elements/responsibilities/<k>
Elements/Cd                 -> a ContinuousVariable, then any row above
Elements/Cd/prediction      Elements/Cd/quantiles/0.975
```

**Categorical variable** — categories are child nodes

```
Landuse                     -> the CategoricalVariable; .labels lists categories
Landuse/predicted           Landuse/entropy        Landuse/uncertainty
Landuse/measurements_a      Landuse/measurements_b Landuse/boundary
Landuse/Forest              -> a _Category
Landuse/Forest/probability             Landuse/Forest/indicator
Landuse/Forest/indicator_mean          Landuse/Forest/indicator_variance
Landuse/Forest/indicator_predicted
```

**Metadata** (per-location facts the models never see)

```
_metadata/HOLEID    _metadata/LENGTH    _metadata/fold
```

Quantiles exist only **after** a prediction that drew simulations:

```python
model.predict(grid, n_sim=100)
grid.get("V").reset_quantiles([0.025, 0.5, 0.975])   # then V/quantiles/0.5
```

---

## 1. Preamble

Cell 1 installs, always with `%%capture`, always from the branch being taught:

```python
%%capture
!pip install cmcrameri  # scientific color maps
!pip install git+https://github.com/italo-goncalves/geoML.git@claude
```

Cell 2 imports, in a fixed shape — third-party, blank line, `import geoml`,
blank line, the submodule aliases:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm

import geoml

import geoml.kernels as kr
import geoml.transform as tr
import geoml.likelihood as lk
import geoml.latent as gl
import geoml.warping as wp
```

The aliases are fixed and worth memorizing: **`kr` `tr` `lk` `gl` `wp`**. Import
only the ones the notebook uses. Note these are the *notebook* aliases — package
source code uses underscore-prefixed ones (`import geoml.kernels as _kr`)
instead; never mix the two conventions.

Constants go next, upper case:

```python
CMAP = cm.batlow
MAX_V = 1700.0
ELEMENTS = jura_train.get("Elements").labels
```

A Colab-targeted notebook says so early: *"Do not forget to enable the GPU in
Colab."*

---

## 2. Voice and rhythm

- A markdown cell before nearly every code cell — one to three sentences saying
  what the next cell does and *why*, not what the code says.
- First person plural: "In this notebook we'll see…", "Let us extract a
  subset…", "First we generate the dataset."
- Markdown *after* results too, reading them out: "It can be seen that the
  predictions are non-negative", "a stationary model struggles to adapt to the
  small details in the data."
- Title is `# Topic` with a one-paragraph statement of the exercise.
- Terms are introduced in *italics* on first use — the *kernel*, the *transform*,
  *inducing points*, the *warping function*.
- A `## References` section at the end when the method has a paper, author–date
  with DOI links.
- Sometimes an exercise closes the notebook: "Try making a model for the `Rock`
  variable."

---

## 3. The workflow arc

The VGP notebooks number their steps, and the numbering is the canonical order:

```
### Step #0: exploratory data analysis
### Step #1: inducing points
### Step #2: building a latent variable network
### Step #3: likelihoods and normalization
### Step #4: training
### Step #5: diagnostics
```

**Step 0 — EDA.** `Explorer` figures, closed with a semicolon to swallow the
return value:

```python
exp_data = geoml.plots.Explorer(jura_train, continuous='Elements')
exp_data.pairs(log=True, principal_components=3, upper='density');
exp_data.pca(0.95, log=True);
```

**Step 1 — inducing points.** Data plus a coarse grid, divided into experts:

```python
inducing_points = geoml.data.inducing.experts(
    geoml.data.inducing.combine(
        jura_train,
        geoml.data.Grid2D(start=[0, 0], n=[21, 21], end=[6, 6])),
    4)
```

**Step 2 — the latent network**, built input → GP → output:

```python
net_input = gl.BasicInput(inducing_points, tr.Isotropic(1))
net_output = gl.BasicGP(net_input, size=7, kernel=kr.Spherical())
```

For a categorical variable the output is linear, sized to the categories:

```python
net_out = gl.Linear(gp, size=2)   # to match the 2 categories
```

Deep models insert the walk between the field and the model:

```python
field = gl.BasicGP(deep_input, size=3)      # a vector field in 3D
new_coords = gl.GPWalk(field, n_steps=5)    # moves the points
deep_gp = gl.BasicGP(new_coords, size=1)    # modelled on moved coordinates
```

**Step 3 — warping and likelihood.** The chain reads top to bottom with a
comment naming each link's job; the size is positional and repeated:

```python
warp = wp.ChainedWarping(
    wp.Log(7),                      # non-negativity
    wp.ZScore(7),                   # centering / scaling
    wp.RobustPCA(7, 7),             # decorrelation
    wp.Spline(7, knots_per_arm=5),  # asymmetry
    wp.ZScore(7),
)
likelihood = lk.EpsilonInsensitive(warp)
```

**Step 4 — training.** Keyword arguments, always in this order:

```python
model = geoml.models.VGPNetwork(
    data=jura_train,
    variables='Elements',
    likelihoods=likelihood,
    latent_network=net_output,
    options=geoml.models.GPOptions(prediction_batch_size=2000))
model.set_learning_rate(5e-2)
model.train_full(100)
```

`train_full` for full-batch, `train_svi` for minibatches. Deep models reset the
learning rate between rounds, with the reason in a comment:

```python
# Deep models may have more local optima,
# so the optimizer is reset after some epochs
deep_model.set_learning_rate(5e-2)
deep_model.train_svi(20)
deep_model.set_learning_rate(1e-2)
deep_model.train_svi(60)
```

**Step 5 — diagnostics.** Predict back onto the training data, then the five
figures, each introduced by a markdown cell saying what a good one looks like:

```python
model.predict(jura_train, n_sim=100)
exp_data = geoml.plots.Explorer(jura_train, continuous='Elements', model=model)
exp_data.training_curve();     # convergence
exp_data.accuracy();           # 1:1 curve if the intervals are right
exp_data.spread_check();       # is local uncertainty proportional to variability
exp_data.transformed_pairs(upper='density');   # normalized and decorrelated?
exp_data.simulation_pairs();   # was the data distribution captured?
```

**Then prediction on a grid**, and only then plotting:

```python
grid = geoml.data.Grid2D(start=[0, 0], n=[251, 251], end=[6, 6])
model.predict(grid, n_sim=100, include_noise=True)
grid.get("Elements").reset_quantiles([0.025, 0.5, 0.975])
```

The plain `GP` model is the same arc without steps 1–3: build a `Covariance`,
construct, `train(max_iter)`, `predict(grid)`.

---

## 4. Idioms

**Print the object.** A bare name in its own cell shows the parameter table —
after construction and again after training:

```python
points          # the container
model_1d        # the trained model, with its parameters
print(dh_point.tree())
```

**Leave the alternatives commented out.** This is deliberate and frequent — the
notebook shows the options it isn't taking:

```python
# Isotropic
transf = tr.Isotropic(100)

# Anisotropic
# transf = tr.Anisotropy2D(maxrange=100, minrange_fct=0.5, azimuth=135)

# Automatic anisotropic
# transf = tr.Anisotropy2DDynamic(n_directions=9)
```

**Comment the surprising argument, not the obvious one.** `# higher = more
flexibility`, `# to match the 2 categories`, `# The Gaussian covariance requires
the addition of a small value to the main diagonal.`

**Synthetic data gets `np.random.seed(0)`** and is built to expose a specific
behaviour — a gap in the middle to show uncertainty growing, added noise "for
the sake of realism".

---

## 5. Plotting

Matplotlib for 2D and diagnostics, plotly for 3D, pyvista to *produce* meshes
rather than to display them.

- `fig, ax = plt.subplots(n, m, figsize=[9, 9])`, then `fig.show()` at the end.
- `ax.set_aspect("equal")` on every map.
- `imshow` of a grid always carries `origin="lower"` and
  `extent=(x0, x1, y0, y1)`.
- Shared `vmin`/`vmax` across panels that are meant to be compared; a single
  `plt.colorbar(im, ax=ax, shrink=0.9)` for the group.
- Colour maps come from `cmcrameri`, chosen by role: `cm.batlow` / `cm.davos`
  for grades, `cm.roma` for categories and diverging potentials, `cm.nuuk` for
  uncertainty.
- Training curve: `plt.plot(model.training_log)`, xlabel `"Iteration"`, ylabel
  `"Log-likelihood"` for `GP` and `"ELBO"` for `VGPNetwork`.
- `as_image(sigma=…)` smooths quantiles for display — quantiles are estimated
  from a finite set of simulations, so the tails are noisy.
- 3D: `go.Figure()` with `go.Scatter3d` for the data and `go.Mesh3d` for the
  surface, `scene=dict(aspectmode='data')`, explicit `width`/`height`. The mesh
  comes from geoML and is unpacked for plotly:

```python
surf = grid.get("SIMPLE LITO/Vein/indicator_predicted").get_contour(0.001)
model.predict(surf)                  # transfer properties onto the surface
pv_surf = surf.as_pyvista()
vertices = pv_surf.points
faces = pv_surf.faces.reshape(-1, 4)[:, 1:]   # [3, p1, p2, p3] per triangle
```

Seaborn appears only for categorical scatter over a data frame
(`data.as_data_frame()`), with an explicit `hue_order` and a palette built from
a cmcrameri map via `rgb2hex`.

---

## 6. Datasets and cases

`geoml.datasets` loaders return train/validation pairs or tuples:

```python
walker, walker_ex = geoml.datasets.walker()        # 470 samples + full grid
jura_train, jura_validation = geoml.datasets.jura()
points, tangents, _ = geoml.datasets.example_fold()
```

The recurring cases and what each teaches: **Walker Lake** — warping an
asymmetric non-negative variable; **Jura** — multivariate correlation, and its
`Landuse`/`Rock` columns for categoricals; **example_fold** — gradients and the
potential-field method; **the quartz vein drillholes** — 3D implicit modelling,
stationary against deep, compared by confusion matrix.
