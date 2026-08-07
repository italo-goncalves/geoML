# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---
# Communication between LLM and user within code

## Locked-code convention (MUST follow)

Some code is user-maintained and must never be modified by Claude.

- Block marker — never edit anything between these lines (not even formatting
  or imports). Reading, calling, and wrapping the code is fine:

  ```python
  # === LOCKED: <owner/reason> ===
  ...
  # === END LOCKED ===
  ```

- File marker — the whole file is user-maintained (placed at the top, after
  the module docstring):

  ```python
  # LOCKED FILE — user-maintained. Do not edit; propose changes instead.
  ```

Rules:
1. Before editing ANY file, check it for `LOCKED` markers.
2. If a locked region has a bug, or its interface doesn't fit, STOP and propose
   the change to the user (diff or description) — never apply it directly.
3. Renaming, moving, or deleting a locked file also requires asking first.
4. Unmarked code is normal — edit as usual.

## Notes-to-Claude convention

Comments starting with `# CLAUDE:` are notes addressed to Claude:

```python
# CLAUDE: this expects NHWC float32 in [0,1]; write the adapter accordingly.
# CLAUDE: is dropout=0.3 too high here? suggest, don't change.
```

Rules:
1. When the user says "check my notes" (or similar), grep the repo for
   `# CLAUDE:` and handle every hit.
2. Act on notes found while reading a file, even if not explicitly told.
3. After resolving a note, delete the comment (quote it in the reply so the
   resolution is traceable). Notes ending with `(keep)` stay in place.
4. A note inside a LOCKED region may be read and acted on, but the region
   itself still must not be edited — including removing the note; ask instead.


## Communicating changes

1. After completing a coding task, give a small summary about the files and lines changed, as well as the functions, methods created, etc.
2. Do not commit the changes immediately; allow the user the opportunity to open the IDE and see the changes.
3. When you do commit, include the user's own uncommitted changes as well —
   never stage only your own files. Describe them in the commit message as you
   would your own, reading the diff to see what they do.

---

# Project: geoML

Machine learning models for spatial/geoscientific data, centered on **variational Gaussian processes**. TensorFlow / TensorFlow-Probability backend, all computation in `float64`. GPL-3 (dual-licensed; see README). Single flat package `geoml/` (no `src/` layout).

## Module map

| Module | Role |
|---|---|
| `parameter.py` | `Parametric` base class + `RealParameter` family. Foundation everything inherits from. |
| `data.py` | Data containers (`PointData`, `Grid1D/2D/3D`, `DirectionalData`, `Section3D`, `Surface3D`), variable types (`ContinuousVariable`, `CategoricalVariable`, `CompositionalVariable`, `RockTypeVariable`, …), and point-wise metadata (`add_metadata`/`get_metadata` — per-location information the models never see). Triangulated meshes are `Mesh3D` (the primitive, measuring `area`/`closed`/`consistent` at construction) with `Surface3D` (never closes) and `Solid3D` (always closes, consistently wound, has `volume`) as siblings, plus `DTM3D(Surface3D)` for a terrain (never folds over); `mesh3d(...)` picks by geometry (never `DTM3D` — that is a promise to make, not a fact to detect), `heal()` repairs, `split()` separates pieces, `Solid3D` does `union`/`intersection`/`difference`, and cutting across kinds keeps the caller's kind (sheet ∩ body → sheet; body ∩ sheet → the volume below it, via extruding the sheet). Bad combinations raise `NotClosedError`/`InconsistentMeshError`/`NotSingleValuedError`/`MeshTypeError`, all `ValueError` subclasses. `get_contour(close="above"|"below")` pads the cube so a contour reaching the grid edge closes into a body. They read and write DXF (`from_dxf`/`export_dxf`, geometry only), and every container can be told where its locations sit relative to one (`assign_from_surface` above/below a sheet, `assign_from_solid` inside a body — both write a metadata column; blocks can also report the fraction cut). Largest module (~4300 lines). |
| `drillhole.py` | `DrillholeData` (collar + optional survey + named `IntervalTable`s) and `IntervalTable`. Minimum-curvature desurveying, per-column roles, validation, compositing, subsetting/category grouping, and conversion to `PointData`. Never fed to a model — only converted. The constructor **renames as it ingests**: whatever the source columns were called, `dh.collar` comes back with `X`/`Y`/`Z`/`LENGTH`/`DIP`/`AZIMUTH` indexed by `HOLEID`, and interval tables use `HOLEID`/`FROM`/`TO` (the constants at the top of the module) — reach for the original names and you get a `KeyError`. Interval coordinates are never stored; `coordinates_at()` produces them on demand, so compositing stays exact. |
| `datasets.py` | Loaders for bundled `sample_data/` (`walker()`, `jura()`, `ararangua()`, …), plus `macpass()`, which reads a database the user downloads themselves. |
| `models.py` | User-facing models: `GP` (legacy, closed-form), `VGPNetwork` (the core variational model), `StructuralField`, `GPEnsemble`, `ProjectedVGP`, `Normalizer`. |
| `latent.py` | Composable latent-variable network nodes (`BasicGP`, `Add`, `Multiply`, `LinearCombination`, `ProductOfExperts`, …) — the building blocks passed to `VGPNetwork`. |
| `inducing.py` | Builds the inducing points `BasicInput` is given: `from_kmeans`/`from_grid`/`combine` for one set, `grid_experts`/`experts` for a list of them, one per expert. Both overlap neighbouring experts, so a prediction shows no seam where one gives way to the next. `grid_experts` extends regular blocks by one step, keeping every expert the same size with the Moore neighbourhood (8 in 2D, 26 in 3D) known in advance; `experts` is the unordered version — it clusters the points it is handed, then has each cluster borrow a further `overlap` of its own count from the nearest points (Mahalanobis) belonging to others. Counting the overlap in points, not distance, is deliberate: a radius keeps the experts wildly uneven, since a crowded cluster swallows far more than an isolated one. It divides inducing points, not data: `experts(from_kmeans(data, 1500), 12)`. |
| `likelihood.py` | Likelihoods (`Gaussian`, `MultivariateGaussian`, `Bernoulli`, `CategoricalGaussianIndicator`, `GradientIndicator`, …). |
| `kernels.py` | Kernels + `Covariance` objects. |
| `transform.py` | Spatial transforms fed to kernels (anisotropy ellipsoids, projections, chaining). |
| `warping.py` | Output warpings (`ZScore`, `Spline`, `Softplus`, `PCA`, `ContinuousNormalizingFlow`, …). |
| `projnet.py` | Projected / variational Fourier-feature latent variables. |
| `interpolation.py` | Cubic-convolution and cubic-spline interpolators (TF). |
| `probability.py` | Custom TFP distributions (`EpsilonInsensitive`, `Huber`, spline/empirical). |
| `tftools.py` | TF numerical helpers (CG solver, Lanczos, pairwise dist, …). |
| `storage.py` | `ArrayStore`: one array backed by NumPy in RAM or chunked Zarr on disk, chosen by size. Everything a container holds (coordinates, variable attributes, simulations, metadata) goes through it. |
| `persistence.py` | Whole-model save/load, by replaying the constructor calls recorded on every `Parametric`. Container persistence is `to_zarr`/`open` in `data.py`. |
| `random.py` | The package RNG. `geoml.set_seed(seed)` before construction is what makes parameter initialization reproducible. |
| `geometry.py`, `metrics.py` | Rotation/angle helpers, plus the triangulated-surface maths behind `Surface3D` and the assignments (`weld`, `fan_triangulation`, `vertex_normals`, `area`, `components`, `single_valued`, `open_edges`/`reversed_edges`/`signed_volume` — closed? consistently wound? facing out? — and `sheet_interpolator`/`sheet_elevation`, `inside_solid`) — arrays in, arrays out, no containers; scoring metrics. |
| `plotly.py`, `pyvista.py` | Visualization export. `DrillholeData.as_pyvista(table)` gives one line cell per interval, carrying `HOLEID` and every value column as `cell_data`; `.tube(radius=…)` fattens it for display, but keep the radius under the interval length or each interval renders as a disc rather than a rod. |
| `plots/` | EDA and model figures, in two backends. `plots/base.py`'s `Selection` holds the container + a continuous/categorical choice (+ an optional model) and the colours; `Explorer` (matplotlib, to print) and `Interactive` (plotly, to look at) subclass it and draw the same set under the same names — `histogram`/`pairs`/`pca`/`scene`, plus `training_curve`/`transformed_pairs`/`simulation_pairs`/`prediction_scatter`/`accuracy`/`grade_tonnage`. `plots/dashboard.py`'s `Dashboard` puts plotly figures on one page with their selections linked (matched on the row index every location-drawn trace carries in `customdata`), writes self-contained HTML and renders in a notebook via an iframe. `plots/prepare.py` holds the arithmetic both backends read and imports no plotting library; `plots/style.py` is the palette, the `rc_context` and the plotly `TEMPLATE`. Only sub-package in the codebase — add it to `packages=` in `setup.py` when adding modules. |
| `graphviz.py` | `to_dot(model_or_network)` — the model as a DOT diagram (`model.to_dot()` / `network.to_dot()`). Writes text and imports nothing; Graphviz is only needed to render. |

## Core architecture

- **Everything trainable subclasses `parameter.Parametric`.** It holds a `parameters` dict and a flat `_all_parameters` list. Composite objects call `self._register(child)` to bubble a child's parameters up, and `self._add_parameter(name, RealParameter(...))` for their own. Training reads `get_unfixed_variables()` and optimizes with Adam — all training is gradient-based.
- **`RealParameter` wraps a `tf.Variable`** with a transform + min/max bounds, kept in bounds via `refresh()`. Subclasses encode constraints via the transform: `PositiveParameter` (log), `CompositionalParameter` (softmax/logit), `CircularParameter`, `OrthonormalMatrix`, etc. Persistence is `save_state`/`load_state` (pickle of parameter values).
- **`VGPNetwork` is the heart of the package:** `VGPNetwork(data, variables, likelihoods, latent_network, ...)`. You compose a network from `latent` nodes, attach a `likelihood` per variable, and optionally `warping`. Prefer it over the legacy `GP` for new work.
- **`predict_raw` is a dispatcher, `_predict_raw` is the body** (`VGPNetwork`, inherited by `ProjectedVGP`). `jit_compile` is settled when a `tf.function` is built, so honouring `GPOptions(jit_predict=...)` means holding one traced function per setting in `self._compiled` rather than a flag on a single one; off uses `jit_compile=None`, which is exactly the old behaviour. Keeping the public name on the dispatcher is what lets `predict` still call `self.predict_raw` (a test replaces it with a spy). The other models' `predict_raw` (`GP`, `StructuralField`) are unrelated methods and still plainly decorated. XLA is worth 3–5×, is prediction-only (on training it was slower *and* went NaN), and is off by default because it raises rather than falling back on anything it cannot compile.

## Conventions

- **Imports:** sibling modules are imported with underscore aliases — `import geoml.parameter as _gpr`, `import geoml.tftools as _tftools`. Third-party too: `import numpy as _np`, `import tensorflow as _tf`, `import tensorflow_probability as _tfp`. Match this in any module code. (Test scripts use a bare `import geoml`.)
- Every source file carries the GPL-3 header block — keep it on new files.
- Public API is re-exported via `geoml/__init__.py`'s `__all__`; add new modules there.

## Build / test / gotchas

- No `pyproject.toml`, `requirements.txt`, or CI. Install is `pip install -e .` from `setup.py`; deps: numpy, scipy, pandas, scikit-learn, scikit-image, tensorflow, tensorflow-probability, zarr, xarray, dask, pyvista, matplotlib, plotly, ezdxf.
- **Rendering (pyvista/VTK) fails on an old conda env under WSL**, and the error blames X11 rather than the real cause: VTK falls through GLX, EGL and OSMesa, then segfaults. Mesa's llvmpipe needs a newer `libstdc++` than a 2022-vintage `libstdcxx-ng` provides, and the env's copy shadows the system one. `docs/wsl-pyvista-rendering.md` has the diagnosis and the fix. This bites the visualization work only — nothing else in the package touches OpenGL, so the test suite passes regardless.
- **Tests:** `geoml/test/` is a runnable pytest suite — 685 tests in 26 files, ~4 min, on synthetic data or the bundled `sample_data/`, except `test_sunspot_deep_model`, which downloads from sidc.be and so fails offline (`datasets.macpass()` is the one loader left uncovered). By area: `test_drillhole.py` (90 tests) covers desurveying, interval tables, compositing, renaming and conversion; `test_storage.py`, `test_containers.py`, `test_coordinates_variance.py` , `test_metadata.py` and `test_coded_attributes.py` the `ArrayStore` backend, the batched-write contract every container owes the models, the point-wise metadata columns and the coded (text-as-integer) attributes; `test_persistence.py`, `test_variable_persistence.py` and `test_model_persistence.py` the Zarr and saved-model round-trips; `test_repr.py`, `test_latent_names.py`, `test_graphviz.py`, `test_seed.py`, `test_pyvista_export.py`, `test_simulation_attribute.py`, `test_surface_io.py`, `test_mesh3d.py`, `test_mesh_operations.py`, `test_assignment.py` and `test_block_prediction.py` the string forms, the latent-node names, the DOT diagrams (one test renders and skips when `dot` is absent), `set_seed`, the export selectors, the DXF round trip (written as a `MESH`, read back from `MESH`/`POLYFACE`/`3DFACE`), the mesh hierarchy (invariants, healing, splitting and the booleans — including the disjoint and nested cases VTK returns nothing for), closing a contour and cutting across mesh kinds, the surface assignments (one test pins the grid's per-column shortcut against the general per-cell path) and the block fan-out; `test_plots.py` the EDA and model figures, mostly through `plots.prepare` since a picture is hard to assert on, `test_interactive.py` the plotly twins and the dashboard — a plotly figure is a document rather than a picture, so trace counts, subplot axes and the `customdata` row index the linked selection rests on can all be read back — and `test_metrics.py` the accuracy-plot numbers; `test_experts.py` the inducing point helpers and everything that only happens with more than one expert (the quadratic propagation loop, the traced refresh, a multi-expert save/load); `test_vgp.py` and `test_real_cases.py` build, train and predict end to end, the latter mirroring the real modelling cases (Jura, Arctic lake, sunspots, Walker Lake) at a few iterations; `test_vgp.py` also covers `options.jit_predict` — that XLA gives the same answer on both network roots, that the flag can be flipped on a live model, and that options are per-model rather than shared. geoML isn't pip-installed here, so run from the repo root (so `import geoml` picks up the working tree) with an env that has TF/TFP. No pytest config or CI yet. **Run the files that cover what you changed, not the whole suite** — it takes ~3 min, most of it in `test_real_cases.py` and `test_vgp.py`, which train real models; keep the full run for a release or a change that reaches every container. Note: a model is only reproducible if `geoml.set_seed()` is called *before* the objects are constructed — parameter init draws from the package RNG in `geoml/random.py`, not from `options.seed` (which seeds training) and not from the global NumPy/TF RNG.
- **Version:** `__init__.py`, `setup.py` and `docs/source/conf.py` all say `0.5.5`, the last released version; the changelog's top section is the work in progress, and the version is bumped at release. They drift apart easily — `conf.py` sat at `0.3.5` through several releases before anyone noticed — so bump all three together. A release is a `Version X.Y.Z` commit on `claude`, a `--no-ff` merge into `master` titled `Merge branch 'claude' for version X.Y.Z`, and an annotated `vX.Y.Z` tag.
