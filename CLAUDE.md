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
| `data.py` | Data containers (`PointData`, `Grid1D/2D/3D`, `BlockSet3D`, `DirectionalData`, `Section3D`, `Surface3D`), variable types (`ContinuousVariable`, `CategoricalVariable`, `CompositionalVariable`, `RockTypeVariable`, …), and point-wise metadata (`add_metadata`/`get_metadata` — per-location information the models never see). Triangulated meshes are `Mesh3D` (the primitive, measuring `area`/`closed`/`consistent` at construction) with `Surface3D` (never closes) and `Solid3D` (always closes, consistently wound, has `volume`) as siblings, plus `DTM3D(Surface3D)` for a terrain (never folds over); `mesh3d(...)` picks by geometry (never `DTM3D` — that is a promise to make, not a fact to detect), `heal()` repairs, `split()` separates pieces, `Solid3D` does `union`/`intersection`/`difference`, and cutting across kinds keeps the caller's kind (sheet ∩ body → sheet; body ∩ sheet → the volume below it, via extruding the sheet). Bad combinations raise `NotClosedError`/`InconsistentMeshError`/`NotSingleValuedError`/`MeshTypeError`, all `ValueError` subclasses. `get_contour(close="above"|"below")` pads the cube so a contour reaching the grid edge closes into a body. They read and write DXF (`from_dxf`/`export_dxf`, geometry only), and every container can be told where its locations sit relative to one (`assign_from_surface` above/below a sheet, `assign_from_solid` inside a body — both write a metadata column; blocks can also report the fraction cut). `BlockSet3D(PointData)` is the variable-size block model: every block's origin and size are whole numbers of a base cell (`step / discretization ** max_levels`), so it is built full and `split` keeps it tiling exactly (`is_full()` counts base cells — volume alone cannot tell a gap from a cancelling overlap). **`discretization` is also the refinement ratio** — a block splits into its own sub-blocks, one child each, per axis and not necessarily by two, so `[2, 2, 1]` refines in plan and leaves the bench height alone, and sub-block `j` and child `j` are the same corner (which is what lets a coarse prediction speak about the blocks a split would make). Being constant across levels is why nothing downstream changed: every block fans out into the same number of rows, so `_aggregate` still reshapes and batches need not be level-sorted. Per-block geometry is `block_size`/`block_volume`, deliberately not `step_size` (which means one size for the whole object). `split` keeps what the unsplit blocks hold (same block, same support) and leaves the children missing; `unpredicted()` names them and `predict(..., where=...)` fills only those, which is bit-identical to predicting the lot because a location's value does not depend on its batch. Metadata is inherited by children — it describes the ground — while predictions are not. **It cannot be subsetted**: `__getitem__`/`subset_region` raise, because `PointData`'s would return a plain `PointData` with no origin, level or size — a thing that looks like a block model and cannot report a tonnage. Exclude ground with `refine(..., where=mask)` — those blocks are never predicted at any pass *and* never cut, the mask being carried across each split — and hand it off through `as_data_frame` (a size per row), `as_pyvista` (hexahedra) or `to_zarr`. `predict(..., where=)` on a container that lacks the variable creates it and leaves the unnamed rows at NaN (what `unpredicted` reads); it never reallocates an existing one, which would wipe what the untouched locations hold. It exports as welded hexahedra (`as_pyvista`) and contours through VTK rather than marching cubes (`get_contour`, which also accepts the name of a *component* — only those hold a grade), values being averaged from blocks onto corners first. A hexahedron is contoured from its own eight corners, so a coarse block cannot see the corners its finer neighbours put in the middle of the shared face and the two sides tear the surface apart along it; `_cut_to_contour` therefore cuts the blocks a surface runs through down to the finest size **in the mesh handed to VTK, never in the model**, one ring of neighbours included. Nothing is predicted: a child takes its parent's value plus what the corners say about the shape across it (`_trilinear_weights`), and because those weights are symmetric the correction cancels over the children, so a block's estimate stays the mean of the children replacing it — copying the parent instead costs 0.8% of a surface's area, interpolating without the anchor drifts 1.5% in mass. Widening the *refinement* fixes the tears equally well and costs 5x the blocks, which is why it is not what happens. `supersample=1` (the default) cuts that mesh a further level past the model's finest block: what VTK reads between corners is trilinear, creased at every face, and averaging onto corners again at each level composes into a rounder reconstruction — so the surface both turns less sharply and sits *closer* to the true level set, matching a model 3.4x the size for no prediction. Smoothing the triangles was measured and rejected (Taubin: a sixth less faceting, 50% further from the truth) — do not add it. `needs_splitting`/`block_shares` ask each variable for its own decisions via `_Variable.split_shares` (union over variables, `split_on` to narrow) — never reach in for `divided` from outside, it is a column on a category and a dict on a grade. `unbalanced(gap=1)` is the *second* criterion: a block whose sub-blocks agree is never marked by `needs_splitting`, yet if a neighbour was cut twice over the field turns sharply nearby and the block's eight corners are too crude a guess across its span — levelling those jumps measured 3x closer to the true surface for a third more blocks, where a whole level of extra refinement bought nothing for 2.6x as many (deeper refinement widens the jumps as fast as it narrows the blocks). It searches **from the fine side** — each fine block steps one cell past its faces and names the coarse block covering that point — because asking the coarse block what lies beyond its face is not exact (a cell of its own size holds blocks that do not touch it) and painting the base lattice is the memory the model exists to avoid. `crossed_by(mesh)` is the *third*: the blocks whose sub-blocks fall on both sides of a surface or closed body, so a topography or a vein wall forces a cut before anything is predicted (it needs no model — loop it against `split`). `BlockSet3D` also carries the `assign_from_surface`/`assign_from_solid` overrides with `fraction=`, which it lacked (it is **not** decorated `@_blockdata` — it extends `PointData`, so it fell back to the centre-only version); the shared geometry is `_sub_block_shares`/`_blocks_from_surface`/`_blocks_from_solid` at module level, called by both `Blocks3D` and `BlockSet3D`. `models.refine` drives the passes with all three criteria and stops itself: the lattice's `max_levels` bounds them; `where=` takes a mask or the name of a boolean metadata column (a stored filter), and those blocks are neither predicted nor cut. `group(mask)` is `split`'s inverse — every child of a parent or none, a partial family being refused rather than averaged — and it carries what the *ungrouped* blocks hold while leaving the parents missing, since coarsening is a change of support (metadata crosses from the first child; it describes the ground). `index_data` answers with **which block**, not a cell index per axis as a grid does, and `aggregate_numeric`/`_categorical`/`_binary` are built on it. Design and measurements in `docs/variable-block-models.md`. Largest module (~4500 lines). |
| `drillhole.py` | `DrillholeData` (collar + optional survey + named `IntervalTable`s) and `IntervalTable`. Minimum-curvature desurveying, per-column roles, validation, compositing, subsetting/category grouping, and conversion to `PointData`. Never fed to a model — only converted. The constructor **renames as it ingests**: whatever the source columns were called, `dh.collar` comes back with `X`/`Y`/`Z`/`LENGTH`/`DIP`/`AZIMUTH` indexed by `HOLEID`, and interval tables use `HOLEID`/`FROM`/`TO` (the constants at the top of the module) — reach for the original names and you get a `KeyError`. Interval coordinates are never stored; `coordinates_at()` produces them on demand, so compositing stays exact. |
| `datasets.py` | Loaders for bundled `sample_data/` (`walker()`, `jura()`, `ararangua()`, …), plus `macpass()`, which reads a database the user downloads themselves. |
| `models.py` | User-facing models: `GP` (legacy, closed-form), `VGPNetwork` (the core variational model), `StructuralField`, `GPEnsemble`, `ProjectedVGP`, `Normalizer`. Plus `refine(model, blocks)`, a free function rather than a method because it is a workflow rather than something a model *is*: predict coarse, ask `needs_splitting`, cut, predict only what the cut made. |
| `latent.py` | Composable latent-variable network nodes (`BasicGP`, `Add`, `Multiply`, `LinearCombination`, `ProductOfExperts`, …) — the building blocks passed to `VGPNetwork`. |
| `inducing.py` | Builds the inducing points `BasicInput` is given: `from_kmeans`/`from_grid`/`combine` for one set, `grid_experts`/`experts` for a list of them, one per expert. Both overlap neighbouring experts, so a prediction shows no seam where one gives way to the next. `grid_experts` extends regular blocks by one step, keeping every expert the same size with the Moore neighbourhood (8 in 2D, 26 in 3D) known in advance; `experts` is the unordered version — it clusters the points it is handed, then has each cluster borrow a further `overlap` of its own count from the nearest points (Mahalanobis) belonging to others. Counting the overlap in points, not distance, is deliberate: a radius keeps the experts wildly uneven, since a crowded cluster swallows far more than an isolated one. It divides inducing points, not data: `experts(from_kmeans(data, 1500), 12)`. |
| `likelihood.py` | Likelihoods (`Gaussian`, `MultivariateGaussian`, `Bernoulli`, `CategoricalGaussianIndicator`, `GradientIndicator`, …). Holds no reference to a data container — pure tensors. Three reductions over a block's sub-blocks: `_aggregate` (mean — 25 call sites, so leave its signature alone), `_dispersion` (variance) and the pair `_proportions`/`_divided`. All read *after* the warping is undone and *before* the noise is added, so a block value and everything said about its interior are in the variable's own units and free of what refining cannot resolve. `_proportions` is the share of a block below a cut-off (the recoverable share); `_divided` is how often the cut-off passes through it, judging each realization separately — **only the second licenses a split**, since realizations either side of a cut-off are the model not knowing, which cutting cannot mend. A category's cut-off is zero on `ind_skew` (its log-odds against its best rival), so a contact is that zero level set and one reduction serves both kinds. `_CategoricalLikelihood._resolve` computes entropy/indicators from the sub-blocks and aggregates afterwards. |
| `kernels.py` | Kernels + `Covariance` objects. |
| `transform.py` | Spatial transforms fed to kernels (anisotropy ellipsoids, projections, chaining). |
| `warping.py` | Output warpings (`ZScore`, `Spline`, `Softplus`, `PCA`, `ContinuousNormalizingFlow`, …). |
| `projnet.py` | Projected / variational Fourier-feature latent variables. |
| `interpolation.py` | Cubic-convolution and cubic-spline interpolators (TF). |
| `probability.py` | Custom TFP distributions (`EpsilonInsensitive`, `Huber`, spline/empirical). |
| `tftools.py` | TF numerical helpers (CG solver, Lanczos, pairwise dist, …). |
| `storage.py` | `ArrayStore`: one array backed by NumPy in RAM or chunked Zarr on disk, chosen by size. Everything a container holds (coordinates, variable attributes, simulations, metadata) goes through it. Chunking splits the leading (location) axis only, so a chunk holds whole rows — every realization of a band of blocks. That is what the row-wise reductions are built on: `row_quantiles`/`row_cdf` return lazy dask arrays, and `row_bands()` hands back the slices to read a store in for anything reducing over locations by hand. `np.asarray(store)` materializes the lot and on a block model's simulations (hundreds of GB) will kill the session — reach for a band instead. |
| `persistence.py` | Whole-model save/load, by replaying the constructor calls recorded on every `Parametric`. Container persistence is `to_zarr`/`open` in `data.py`. |
| `random.py` | The package RNG. `geoml.set_seed(seed)` before construction is what makes parameter initialization reproducible. |
| `geometry.py`, `metrics.py` | Rotation/angle helpers, plus the triangulated-surface maths behind `Surface3D` and the assignments (`weld`, `fan_triangulation`, `vertex_normals`, `area`, `components`, `single_valued`, `open_edges`/`reversed_edges`/`signed_volume` — closed? consistently wound? facing out? — and `sheet_interpolator`/`sheet_elevation`, `inside_solid`) — arrays in, arrays out, no containers; scoring metrics. |
| `plotly.py`, `pyvista.py` | Visualization export. `DrillholeData.as_pyvista(table)` gives one line cell per interval, carrying `HOLEID` and every value column as `cell_data`; `.tube(radius=…)` fattens it for display, but keep the radius under the interval length or each interval renders as a disc rather than a rod. |
| `plots/` | EDA and model figures, in two backends. `plots/base.py`'s `Selection` holds the container + a continuous/categorical choice (+ an optional model) and the colours; `Explorer` (matplotlib, to print) and `Interactive` (plotly, to look at) subclass it and draw the same set under the same names — `histogram`/`pairs`/`pca`/`scene`, plus `training_curve`/`transformed_pairs`/`simulation_pairs`/`prediction_scatter`/`accuracy`/`grade_tonnage`. `plots/dashboard.py`'s `Dashboard` puts plotly figures on one page with their selections linked (matched on the row index every location-drawn trace carries in `customdata`), writes self-contained HTML and renders in a notebook via an iframe. `plots/prepare.py` holds the arithmetic both backends read and imports no plotting library; `plots/style.py` is the palette, the `rc_context` and the plotly `TEMPLATE`. Anything in `prepare.py` that touches simulations reads them a band at a time (`realization_store` rather than `realizations`, `ArrayStore.row_bands()`) — a block model carries more of them than memory holds. `grade_tonnage` places each block at the highest cut-off it clears and cumulates from the top down, so it costs one pass over the grade rather than one per cut-off; the volume goes into each block's weight (via `block_volume`, a number or a column) rather than multiplying the finished curve, which is what lets blocks differ in size. Only sub-package in the codebase — add it to `packages=` in `setup.py` when adding modules. |
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
- **A variable that holds components does not copy what belongs to them.** `VectorVariable`, `CompositionalVariable` and the `RockTypeVariable` family build their components inside `__init__`, so anything a *component* owns — its `cutoffs`, and whatever is added next — is left behind by every method that rebuilds a variable: `from_variable`, `carry_to`, `copy_to`, the persistence round trip. Each has to carry it across by hand, and each has been fixed once for forgetting to. The failure is quiet: the rebuilt variable is structurally complete and merely does nothing, so **add a test with a vector variable, not only a scalar and a categorical one** — categorical components are `_Category`, which is a different class with different attributes, and testing it proves nothing about the graded case.
- **Do not reach into a variable for an attribute whose type depends on the class.** `divided` is one column on `_Category` and a dict keyed by cut-off on `ContinuousVariable`; `proportion`/`proportions` likewise. Ask the variable — `split_shares()` is the pattern — rather than `getattr`-ing and hoping.

## Build / test / gotchas

- No `pyproject.toml`, `requirements.txt`, or CI. Install is `pip install -e .` from `setup.py`; deps: numpy, scipy, pandas, scikit-learn, scikit-image, tensorflow, tensorflow-probability, zarr, xarray, dask, pyvista, matplotlib, plotly, ezdxf.
- **Rendering (pyvista/VTK) fails on an old conda env under WSL**, and the error blames X11 rather than the real cause: VTK falls through GLX, EGL and OSMesa, then segfaults. Mesa's llvmpipe needs a newer `libstdc++` than a 2022-vintage `libstdcxx-ng` provides, and the env's copy shadows the system one. `docs/wsl-pyvista-rendering.md` has the diagnosis and the fix. This bites the visualization work only — nothing else in the package touches OpenGL, so the test suite passes regardless.
- **Tests:** `geoml/test/` is a runnable pytest suite — 763 tests in 28 files, ~5.5 min, on synthetic data or the bundled `sample_data/`, except `test_sunspot_deep_model`, which downloads from sidc.be and so fails offline (`datasets.macpass()` is the one loader left uncovered). By area: `test_drillhole.py` (90 tests) covers desurveying, interval tables, compositing, renaming and conversion; `test_storage.py`, `test_containers.py`, `test_coordinates_variance.py` , `test_metadata.py` and `test_coded_attributes.py` the `ArrayStore` backend, the batched-write contract every container owes the models, the point-wise metadata columns and the coded (text-as-integer) attributes; `test_persistence.py`, `test_variable_persistence.py` and `test_model_persistence.py` the Zarr and saved-model round-trips; `test_repr.py`, `test_latent_names.py`, `test_graphviz.py`, `test_seed.py`, `test_pyvista_export.py`, `test_simulation_attribute.py`, `test_surface_io.py`, `test_mesh3d.py`, `test_mesh_operations.py`, `test_assignment.py`, `test_block_prediction.py` and `test_blockset.py` the string forms, the latent-node names, the DOT diagrams (one test renders and skips when `dot` is absent), `set_seed`, the export selectors, the DXF round trip (written as a `MESH`, read back from `MESH`/`POLYFACE`/`3DFACE`), the mesh hierarchy (invariants, healing, splitting and the booleans — including the disjoint and nested cases VTK returns nothing for), closing a contour and cutting across mesh kinds, the surface assignments (one test pins the grid's per-column shortcut against the general per-cell path), the `BlockSet3D` lattice (tiling preserved by `split`, mixed-level batches still rectangular, an unrefined set matching the equivalent `Blocks3D` bit for bit) and the block fan-out — including the within-block `dispersion` column, checked against the variance of the sub-blocks predicted as bare points, which only lines up with the noise off (the block path shares one noise pattern across blocks, a point path draws one per row); `test_plots.py` the EDA and model figures, mostly through `plots.prepare` since a picture is hard to assert on — including that the simulations are read in bands and never held whole, which a fixture small enough to sit in RAM as one piece would not catch, so those tests re-chunk it onto disk first, `test_interactive.py` the plotly twins and the dashboard — a plotly figure is a document rather than a picture, so trace counts, subplot axes and the `customdata` row index the linked selection rests on can all be read back — and `test_metrics.py` the accuracy-plot numbers; `test_experts.py` the inducing point helpers and everything that only happens with more than one expert (the quadratic propagation loop, the traced refresh, a multi-expert save/load); `test_vgp.py` and `test_real_cases.py` build, train and predict end to end, the latter mirroring the real modelling cases (Jura, Arctic lake, sunspots, Walker Lake) at a few iterations; `test_vgp.py` also covers `options.jit_predict` — that XLA gives the same answer on both network roots, that the flag can be flipped on a live model, and that options are per-model rather than shared. geoML isn't pip-installed here, so run from the repo root (so `import geoml` picks up the working tree) with an env that has TF/TFP. No pytest config or CI yet. **Run the files that cover what you changed, not the whole suite** — it takes ~5 min, most of it in `test_real_cases.py` and `test_vgp.py`, which train real models; keep the full run for a release or a change that reaches every container. Note: a model is only reproducible if `geoml.set_seed()` is called *before* the objects are constructed — parameter init draws from the package RNG in `geoml/random.py`, not from `options.seed` (which seeds training) and not from the global NumPy/TF RNG.
- **Version:** `__init__.py`, `setup.py` and `docs/source/conf.py` all say `0.5.6`, the last released version; the changelog's top section is the work in progress, and the version is bumped at release. They drift apart easily — `conf.py` sat at `0.3.5` through several releases before anyone noticed — so bump all three together. A release is a `Version X.Y.Z` commit on `claude`, a `--no-ff` merge into `master` titled `Merge branch 'claude' for version X.Y.Z`, and an annotated `vX.Y.Z` tag.
