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
| `data.py` | Data containers (`PointData`, `Grid1D/2D/3D`, `DirectionalData`, `Section3D`, `Surface3D`), variable types (`ContinuousVariable`, `CategoricalVariable`, `CompositionalVariable`, `RockTypeVariable`, …), and point-wise metadata (`add_metadata`/`get_metadata` — per-location information the models never see). Largest module (~3800 lines). |
| `drillhole.py` | `DrillholeData` (collar + optional survey + named `IntervalTable`s) and `IntervalTable`. Minimum-curvature desurveying, per-column roles, validation, compositing, subsetting/category grouping, and conversion to `PointData`. Never fed to a model — only converted. |
| `datasets.py` | Loaders for bundled `sample_data/` (`walker()`, `jura()`, `ararangua()`, …), plus `macpass()`, which reads a database the user downloads themselves. |
| `models.py` | User-facing models: `GP` (legacy, closed-form), `VGPNetwork` (the core variational model), `StructuralField`, `GPEnsemble`, `ProjectedVGP`, `Normalizer`. |
| `latent.py` | Composable latent-variable network nodes (`BasicGP`, `Add`, `Multiply`, `LinearCombination`, `ProductOfExperts`, …) — the building blocks passed to `VGPNetwork`. |
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
| `geometry.py`, `metrics.py` | Rotation/angle helpers; scoring metrics. |
| `plotly.py`, `pyvista.py` | Visualization export. |

## Core architecture

- **Everything trainable subclasses `parameter.Parametric`.** It holds a `parameters` dict and a flat `_all_parameters` list. Composite objects call `self._register(child)` to bubble a child's parameters up, and `self._add_parameter(name, RealParameter(...))` for their own. Training reads `get_unfixed_variables()` and optimizes with Adam — all training is gradient-based.
- **`RealParameter` wraps a `tf.Variable`** with a transform + min/max bounds, kept in bounds via `refresh()`. Subclasses encode constraints via the transform: `PositiveParameter` (log), `CompositionalParameter` (softmax/logit), `CircularParameter`, `OrthonormalMatrix`, etc. Persistence is `save_state`/`load_state` (pickle of parameter values).
- **`VGPNetwork` is the heart of the package:** `VGPNetwork(data, variables, likelihoods, latent_network, ...)`. You compose a network from `latent` nodes, attach a `likelihood` per variable, and optionally `warping`. Prefer it over the legacy `GP` for new work.

## Conventions

- **Imports:** sibling modules are imported with underscore aliases — `import geoml.parameter as _gpr`, `import geoml.tftools as _tftools`. Third-party too: `import numpy as _np`, `import tensorflow as _tf`, `import tensorflow_probability as _tfp`. Match this in any module code. (Test scripts use a bare `import geoml`.)
- Every source file carries the GPL-3 header block — keep it on new files.
- Public API is re-exported via `geoml/__init__.py`'s `__all__`; add new modules there.

## Build / test / gotchas

- No `pyproject.toml`, `requirements.txt`, or CI. Install is `pip install -e .` from `setup.py`; deps: numpy, scipy, pandas, scikit-learn, scikit-image, tensorflow, tensorflow-probability, pyvista, plotly.
- **Tests:** `geoml/test/` is a runnable pytest suite — 342 tests in 15 files, ~13 min, on synthetic data or the bundled `sample_data/`, except `test_sunspot_deep_model`, which downloads from sidc.be and so fails offline (`datasets.macpass()` is the one loader left uncovered). By area: `test_drillhole.py` (90 tests) covers desurveying, interval tables, compositing, renaming and conversion; `test_storage.py`, `test_containers.py`, `test_coordinates_variance.py` and `test_metadata.py` the `ArrayStore` backend, the batched-write contract every container owes the models, and the point-wise metadata columns; `test_persistence.py`, `test_variable_persistence.py` and `test_model_persistence.py` the Zarr and saved-model round-trips; `test_repr.py`, `test_seed.py`, `test_pyvista_export.py`, `test_simulation_attribute.py` and `test_block_prediction.py` the string forms, `set_seed`, the export selectors and the block fan-out; `test_vgp.py` and `test_real_cases.py` build, train and predict end to end, the latter mirroring the real modelling cases (Jura, Arctic lake, sunspots, Walker Lake) at a few iterations. geoML isn't pip-installed here, so run from the repo root (so `import geoml` picks up the working tree) with an env that has TF/TFP. No pytest config or CI yet. **Run the files that cover what you changed, not the whole suite** — it takes ~13 min, most of it in `test_real_cases.py` and `test_vgp.py`, which train real models; keep the full run for a release or a change that reaches every container. Note: a model is only reproducible if `geoml.set_seed()` is called *before* the objects are constructed — parameter init draws from the package RNG in `geoml/random.py`, not from `options.seed` (which seeds training) and not from the global NumPy/TF RNG.
- **Version:** `__init__.py` and `setup.py` both say `0.5.2`, the last released version; the changelog's top section (`0.5.3`) is the work in progress and the version is bumped at release. They drifted apart once — bump both together.
