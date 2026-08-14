## version 0.6.4
* **The repository is now a Claude Code plugin marketplace**, shipping one
plugin (`geoml`) whose single skill carries the package's own working
knowledge. Installed with `/plugin marketplace add italo-goncalves/geoML`
then `/plugin install geoml@geoml`; `README.md` says so.
  - The point is that an assistant writing geoML code should not have to
  infer the API from its names. The skill carries the object model, the
  module layout (which moved wholesale in 0.6.0), the workflow arc, and
  `references/notebook-style.md` -- the import aliases, the plotting
  conventions and a **tree-path table checked by execution**, every row run
  against 0.6.3 rather than written from memory. The seven tutorial
  notebooks ride along with their outputs stripped: 15 MB of `.ipynb` holds
  33 KB of code, and unstripped a single one fills a context window with
  base64.
  - It costs ~194 tokens of always-on description and is only read when it
  fires, which is the whole reason this is a skill rather than more
  `CLAUDE.md`.
  - `plugins/` sits outside the distribution: `[tool.setuptools.packages.find]`
  includes `geoml*` only, so nothing here reaches the wheel.

## version 0.6.3
* **Type annotations, and the documentation site rebuilt around them.** One
workstream, because one pass over a signature does both. The reason for
annotating at all is editor completion: hints are what an IDE surfaces
without being asked, which docstrings are not.
  - The scope is the **user-facing surface**, and the tensor internals are
  left bare deliberately: `tf.Tensor` in and `tf.Tensor` out says nothing
  about the rank, dtype or axis order that actually goes wrong in there, and
  tensorflow-probability ships no stubs, so a checker could not verify that
  half anyway. `typings/tensorflow/__init__.pyi` says so out loud -- it
  declares TensorFlow untyped, which took the first `pyright` run on
  `models.py` from 127 diagnostics (mostly `tf.matmul` overloads) to 24 that
  were all about geoML's own code.
  - `geoml/_types.py` holds the aliases the API's several-shapes arguments
  needed: `Where` (a mask, indices, or the name of a metadata column),
  `Cutoffs`, `Bins`, `PathLike`, `FloatArray`, `Labels`.
  - **The declarations paid better than the signatures.** Declaring
  `_SpatialData._n_data: int` and the `ContinuousVariable` columns closed
  nineteen of the remaining diagnostics at once, since every `np.zeros(
  container.n_data)` downstream had been reading a `None`. `models.py` is
  now clean, and three things the checker found were real: `save()` returns
  the path it wrote rather than `None`, `VGPNetwork` wants a `PointData`
  rather than any container (it reads `coordinates` and
  `get_batched_variance`), and the `variables`/`likelihoods` arguments now
  normalize on `isinstance(str)` / `isinstance(_Likelihood)`, so a tuple or
  any other sequence of names is accepted rather than silently wrapped.
  - **Docstrings are technical now**, scikit-learn style: summary,
  description, parameters, returns, raises, see-also, a brief notes,
  references, examples. The design notes and measured comparisons that had
  accumulated in them -- addressed to a maintainer, and read by users in an
  IDE tooltip -- moved to code comments, the changelog and the design
  records. `models.py` is done; the rest follow module by module.
  - **Sixteen modules are annotated and type-checked**: `models`,
  `metrics`, `storage`, `likelihood`, `kernels`, `transform`,
  `plots/prepare`, `plots/explorer`, `plots/interactive`, `data/inducing`,
  `data/containers`, `data/variables`, `data/blocks`, `data/drillhole`.
  `data/grids` and `warping` are annotated but not yet checked -- the
  first still has numpy-stub and scalar-arithmetic diagnostics, the second
  has its base declared and its families to go.
  - **The checker found thirteen defects, and every one was a promise the
  code did not keep** -- not a type slip, and not one a test would have
  caught, because in each case no test called that path with that input:
  `save()` returns the path it wrote rather than None; `ArrayStore.copy()`
  returns an array, not a store; `metrics.interval_score` and
  `bias_variance_decomposition` never converted the array-likes they
  document, and three metrics returned NumPy scalars where they promised
  floats; `VGPNetwork` wants a `PointData` rather than any container;
  `BlockSet3D.unpredicted` read a prediction column off variables that may
  have none; `IntervalTable.rename` keyed its roles by a value that need
  not be a string; `add_intervals` handed a set to a method taking a
  sequence, so dropped holes reordered the rest; `_Warping.forward` and
  `backward` returned None instead of refusing, so an unimplemented warping
  failed three frames away; `OrderedGaussianIndicator` takes a *count* of
  levels; `Anisotropy3DMath` declared integer angles that
  `Anisotropy3DDynamic` computes as floats; and `Grid3D.make_interpolator`
  calls a function that does not exist (left as found, filed with the RBF
  item -- wiring it to `CubicConv3DSeparable` or deleting it is a decision).
  - Two design changes came out of it. `prepare.continuous_parts` names the
  kind of variable a figure needs, in place of four copies of a `getattr`
  idiom; and `_Variable._sim_store()` is the one place that asks for a
  simulations store, so a variable that was never allocated says so instead
  of raising from `NoneType` inside a batch loop. The grid constructors keep
  their `step` argument and their `step_vector` apart, which is what makes
  the argument's type stateable at all.
  - **The documentation site is rebuilt.** The five-file `sphinx-quickstart`
  skeleton and the 94 tracked build artifacts (a stale HTML build of the
  pre-0.6.0 layout, with a vendored theme and its fonts) are gone. Sphinx +
  MyST + autodoc: a Markdown front page with a worked Walker example, an API
  reference that grows as modules are annotated, and the design records
  included from `docs/*.md` rather than copied. A new `docs` workflow builds
  it with warnings as errors on every push, runs `pyright` beside it, and
  publishes to GitHub Pages on a release tag; the README carries the badge
  and the worked example.

* **The mixture likelihood, settled: one family, several scales, and the
mixture taken over the row.** Four changes, all of them decisions the
first implementation left open or got half right:
  - **It was fitting one model and diagnosing another.** `log_lik` took
  the mixture per *cell* -- a component chosen independently for each
  column, sharing one weight -- while `responsibilities` took it per
  *row*. The two coincide on one column, which is everything that had
  been measured; on a two-column toy they differ by 1.14 nats, one
  calling a row bad and the other calling one of its columns bad. The row
  is right for a vector or compositional variable, whose columns are one
  observation in sample space: a measurement wrong in one component is a
  wrong measurement. So the densities are multiplied across the columns
  before the components are weighted, and because that no longer
  factorizes, a mixture on a vector variable takes its latent expectation
  over the joint posterior samples rather than each column's own
  quadrature (`_column_quadrature`; a scalar mixture keeps Gauss-Hermite,
  where the two readings are the same thing and the quadrature is exact).
  - **One family, built by the mixture.**
  `Mixture(warping, n_components=2, family="gaussian", separation=3.0)`
  replaces the list of hand-built component likelihoods -- a **breaking
  change**, taken deliberately. Mixing families was never sound: a
  Student's t is itself a scale mixture of Gaussians, so a
  Gaussian-plus-Student mixture is an unidentifiable reparameterization
  of a scale mixture, and shape is the warping's job in this package
  anyway. Every contaminated-noise GP in the literature (Kuss 2006,
  Stegle et al. 2008, and the 2024 sparse variational version) is
  same-family scale inflation. With the family known, the sharp edge that
  needed a docstring warning -- symmetric components never pull apart --
  is closed by construction: each family declares which parameters carry
  its width and with which exponent (`_WIDTH_PARAMETERS`: a Gaussian's
  `noise` is a variance, so it moves with the square; an
  epsilon-insensitive `c_rate` is a rate, so it moves with the inverse;
  a Student's `df` is shape and does not move), and the components are
  spread by `separation` at construction, bounds moving with them where a
  ceiling chosen for one noise would clamp the wide one back. The
  contamination test that used to hand-set `noise=9.0` now passes on the
  default construction alone. `Gamma` is refused as a family, with the
  reason: its spread is tied to its mean.
  - **Contamination is declared, not assumed.** The default is now
  `contamination=[False] * n_components` -- the mixture is a noise model
  first, and nothing is silently excluded from the ground. Marking a
  component says its readings replace a measurement rather than report
  one, and only then does the ground part company with the measurement.
  - **`noise_variance` is of a measurement again**, as its own docstring
  always claimed. It was the inlier-only spread, so `spread_check` scored
  a model as over-confident for correctly calling a row contaminated. The
  nodes now carry two weightings through one fold -- the genuine
  components renormalized for the value, the mixture's own for the spread
  -- and the second moment is taken **about the reported value**, which
  is what a residual is measured from (`E[(y - prediction)^2]`, exactly
  the extra `(mean_full - mean_genuine)^2` the contamination shifts by).
  With nothing declared as contamination the two weightings are equal and
  this is the ordinary noise integral. Cross-validation was never
  affected: it scores through `predict_measurements`, which always kept
  the full mixture.
* **Responsibilities are filed on the variable**, as a dict family keyed
by component (`assay/responsibilities/0`, `.../1`) -- the same shape as
quantiles and cut-off shares, so the data frame, the pyvista export, the
Zarr round trip, subsetting and `select("**/responsibilities/*")` all
have it off one declaration. `VGPNetwork.responsibilities(newdata)`
computes the latent moments itself (no prior `predict`, no stale-column
trap), skips every variable whose likelihood is not a mixture, leaves a
row without a measurement missing, and takes `store=False` to look
without writing. Read it out-of-fold: at a training location the model
interpolates its own measurement, so a row's own responsibility
understates it -- the honest version is the container `cross_validate`
returns.

## version 0.6.2
* **Cross-validation, whole: folds that mimic the task, a driver that
never retrains from scratch, and intervals held to their word.** Four
pieces, one pipeline (design record and measurements in
`docs/cross-validation.md`):
  - `spatial_k_fold(test_data, k, groups=...)` works now, and differently
  from the prototype it replaces: discrete groups that are never split (a
  drill hole stands or falls together; k-means atoms otherwise),
  agglomerated by cutting a Ward dendrogram of group centroids at every
  count and keeping the cut whose held-out-to-training nearest-neighbour
  distances best match -- Wasserstein -- the distances from the *actual
  prediction target* to the data (Linnenbrink et al., 2024). The
  soft-membership optimizer matched the distributions perfectly while the
  folds were spatially wrong; ~n·k continuous degrees of freedom was the
  overfit, and discreteness is the fix. Writes a `"fold"` metadata column
  (`name=` renames it, so two labellings can sit side by side).
  - `models.cross_validate` is the VGP translation of kriging's
  fixed-variogram cross-validation: the trained model saved once, each
  fold a copy rebuilt around the reduced data, its variational state --
  where the data lives -- re-initialized so it is *structurally* ignorant
  of the held-out rows, everything else frozen, a short refit, and the
  held-out rows predicted into one shared container that ends the loop
  fully out-of-fold. Scored on measurements, per fold and pooled (rmse,
  mae, bias, crps, goodness). Measured on Walker, 5 spatial folds:
  fresh-variational at 200 iterations matches a 400-iteration scratch
  retrain within a few percent (rmse 238 vs 245) at 2.3x less cost --
  while *warm-starting* scores better than the honest gold itself (231),
  which is the residual memory of the held-out data surviving 200
  iterations of supposed forgetting, and is why fresh initialization is
  the default rather than a knob.
  - `models.conformalize` on top: split conformal on the out-of-fold PITs
  the driver leaves behind, a monotone map from the coverage an interval
  should have to the level it must be cut at, with the finite-sample
  guarantee -- and its limits said out loud: folds that mimic deployment
  are what the exchangeability is worth, the intervals are of
  measurements, and no repair reaches past the ensemble's own range.
  - The variogram joins the figures, in both backends: the data's curve
  with one thin curve per realization on the same pairs -- the
  spatial-continuity check the marginal figures cannot make -- with
  directions, `residuals=True` for what the model missed, deterministic
  pair budgeting, and realizations read one column at a time.
  `metrics.variogram_score` (Scheuerer & Hamill) is its number, and
  `rmse`/`mae`/`bias`/`crps` join the metrics for the fold reports.
  `ContinuousVariable.compute_metrics` now reports the probabilistic
  scores beside the point errors it always had: bias, CRPS, goodness and
  the variogram score, all off the stored simulations.
* **The back-transform takes all realizations at once.** Prediction with
`include_noise=True` at high `n_sim` had become the slow step, and the
arithmetic was innocent: the warping's backward ran through a `map_fn`,
one realization at a time -- a sequential loop whose per-step launch
overhead was measured at 96% of a noise-free 100-simulation prediction
-- and the noise integration paid that loop again for every quadrature
node, 8 elementwise and 64 for a mixing warping: 6400 sequential little
ops at `n_sim=100`. The realization axis is now folded into the row
axis, so the backward runs once over `n x n_sim` rows -- exact, a
warping acting on each row alone, and the peak tensor unchanged.
Measured on the Walker 78k grid: `n_sim=100` with noise **14.1 s to
0.40 s**, without 6.6 to 0.36; `n_sim=20` with noise 6.7 to 0.26.
Integrating the noise now costs a tenth on top of a prediction instead
of multiplying it, and the node-by-node fold keeps its memory promise
untouched. One regression caught on the real Macpass model and fixed the
next day: the reshape assumed the backward keeps its width, which a chain
holding a PCA does not (3 latent columns back to 4 composition parts) --
the width now comes from the warping's own declaration, and a test pins
the width-changing case the suite's square fixtures had missed.
* **Experts may keep their own counsel.**
`GPOptions(expert_propagation="independent")` lets each expert of a deep
network predict its own inducing set alone, where the default
(`"consensus"`, the historical behavior) predicts every set from every
expert and combines by precision weighting -- the network's one O(K^2)
step. Measured on a deep Walker model and a 3000-point synthetic:
training 1.6x faster at 5 experts growing to 6.3x at 40 (the consensus
cost quadruples per doubling of K, the independent cost doubles), and
prediction up to 8x (54 s down to 6.8 s on a 6.4k grid at K=40).
Duplicated inducing points in overlapping sets do genuinely diverge --
several latent standard deviations, where the consensus made them
identical by construction -- and the data-side weighting in
`interpolate`, which is untouched, arbitrates: quality lands within a
few percent of consensus on Walker (four seeds) and ahead of it on the
synthetic at 20 experts, where the consensus coupling of every expert's
parameters appears to slow optimization. One option in `GPOptions`
governs every `BasicGP` in the network at once, read at trace time
through the same context-flag pattern as `qmc_simulations`; the traced
refresh and the prediction dispatcher are keyed by it, so flipping the
option on a live model takes effect at the next call and flipping it
back reproduces the original numbers. Single-layer networks are
untouched -- below a terminal node the propagation never runs.
* **`RobustPCA` initializes quietly.** scikit-learn's FastMCD chatters on
its concentration steps ("Determinant has increased" on perfectly normal
iterations) and flags a not-full-rank estimate on the way out; neither is
actionable during an initialization, so exactly those two warnings are
filtered around the fit alone, and anything else sklearn says still comes
through.
* **A variable can be derived from others, realization by realization.**
`container.derive(names, function, arguments)` builds `DerivedVariable`s --
the middle ground between metadata (a constant the models never see) and a
modelled variable: full simulations and everything built on them, with all
of the uncertainty inherited. The function is applied once per realization
-- which is what keeps a nonlinear one honest, `f(E[grades])` not being
`E[f(grades)]` -- and the prediction column is the mean of the derived
realizations. One derived variable per output name; metadata paths
(`"_metadata/density"`) come in as per-location constants; a variable
without simulations is refused, everything being realization-wise; and a
function that accepts a `simulation=` keyword receives the realization's
index, which is how a per-realization price scenario knows which draw it
is in. The walk is banded, so a block model's stores are never held whole
(pinned by the same tripwire as everywhere else). Being a
`ContinuousVariable` underneath, cut-offs, quantiles, contours,
grade-tonnage curves, subsetting and the Zarr round trip all come free --
the recipe itself is not persisted (a function does not survive a store
honestly): a reloaded derived variable is data, and rerunning the script
that derived it is the refresh. Models refuse it by name at every door
(`training_input`, `get_measurements`, `update`).
* **One realization at a time, guaranteed.** A block model's simulations
are hundreds of gigabytes, and the workflows coming (a random NSR per
realization, and anything else that walks the realizations sequentially)
need one of them without paying for all of them. `variable.simulation(i)`
already read a single column out of the store; it is now *pinned* to stay
that way -- the same tripwire that guards subsetting fails any read of a
whole `(n_data, n_sim)` store -- and its docstring carries the cost
model: only one column is ever in memory, but the store is chunked by
location, so a column read visits every chunk and walking all
realizations costs one pass per realization; anything that decomposes
over locations should read row bands instead. `compute_metrics`, the
last consumer that materialized the store (to keep only its measured
rows), now indexes those rows out of the chunks the way subsetting has
since 0.5.9, and the `get_simulations` methods say in-source that they
are the materializers, for data that fits.
* **One seed to rule them all.** `geoml.set_seed` was already what made
the initial parameters reproducible; now it is the only knob. A model's
options draw their `seed` -- the number training's Monte Carlo and the
simulation stream read through stateless sampling -- from the package
generator when they are built, so the same call that fixes the starting
parameters fixes the training trajectory and every simulation, and there
is no second seed to forget. `GPOptions(seed=...)` is gone (a
`TypeError` now -- set the attribute directly in the unlikely case a
training seed must be forced), while `options.seed` itself remains: a
saved model keeps the number it drew, since persistence restores the
options `vars` wholesale, so old saves keep their stored seed and any
saved model replays its simulations exactly on reload. A model built
without `set_seed` now varies its training draws from run to run -- as
its initialization always did, so nothing that was reproducible before
has stopped being reproducible. Every
dependency carries a lower bound, and the package declares
`requires-python >= 3.10`. Four floors are API facts: zarr 3
(`zarr.create_array` is the v3 surface the stores are built on), pyvista
0.47 (`select_interior_points`, added there as the replacement for the
filter whose rename once bit), `tensorflow-probability[tf]` 0.24 (the
extra that brings tf-keras), and vtk 9.1 -- declared as a dependency for
the first time, because `data/meshes.py` imports vtk directly rather
than through pyvista. The remaining floors are era markers, versions
comfortably older than anything verified, so a truly ancient environment
gets a legible refusal from pip instead of a strange crash later.
* **Float64 earned its keep.** The mixed-precision idea — kernel
arithmetic in float32, Choleskys kept in float64 — was measured on the
Walker VGP and rejected, with the trail in `docs/mixed-precision.md`.
The gradients do flow through the casts (median deviation 2.6%, loss
identical to 8 digits at the same parameters), but the scheme needs
local coordinates as a matter of arithmetic (covariance errors of 1e-2
at UTM scale against 6e-7 in a local frame), needs the jitter raised to
1e-5 to survive training (1e-9 dies at once, 1e-6 halfway) -- and that
jitter moves the model sixteen times further than float32 itself does.
For all that it buys 1.1-1.2x end to end and nothing on top of XLA,
which already removed the memory traffic float32 would have halved. The
one real speed lever for prediction remains `jit_predict`: exact, 3-5x,
already shipped.
* **The tests now run themselves.** A GitHub Actions workflow
(`.github/workflows/tests.yml`) runs the structural test files -- the 600
tests that train no real model, two and a half minutes locally -- on every
push, and the full suite on release tags and on demand from the Actions
tab, minus the one test that downloads from sidc.be (a release gate should
not depend on a third-party server being up). Every run starts from a
machine with nothing on it, so `pip install -e .` is re-proved from
`pyproject.toml` alone each time, and a pyvista or TensorFlow release that
breaks something turns a commit red before any user hits it. A new test
file runs on every push until it earns its way onto the heavy list. Its
very first runs earned their keep: recent tensorflow-probability does
not install Keras support by itself, so a fresh `pip install geoml`
could not import the package, and pandas treats the Excel engine behind
`datasets.ararangua()` as optional -- the dependencies are now
`tensorflow-probability[tf]` and `openpyxl`, both of which every
environment built by hand already had by accident.

## version 0.6.1
* **Everything below the topography, in one call.** Three pieces, built
for the grade-shells-under-the-DTM workflow. A sheet can now be cut by a
terrain: it has no inside, but a single-valued sheet has an *underneath*,
so `surface.intersection(topo)` keeps the piece lying under it (and
`difference` the piece above), through the same extruded ground a body is
cut with -- a folded sheet still refuses, `"below"` not being one region.
`BlockSet3D.get_contour` gained `close=` -- `"above"`/`"below"`/`True`,
the grid's own argument -- closing a shell that runs out of the model
against its box: a shell of ghost cells mirrors the boundary blocks, each
its partner's exact size so the cap cannot tear at the box face, and the
boundary cells the cap covers join the fine-cutting like any other piece
of the surface (the first attempt left the cap tearing between
mismatched ghost sizes exactly as the method's own docstring predicts for
hanging nodes; cutting the cap's cells is the same cure the interior
always had). And `topo.clip_meshes([...])` is the batch form: one ground
body extruded once serves every cut, bodies coming back closed bodies,
sheets coming back sheets, in input order.
* **The console belongs to the user again.** A boolean the exact engine is
*allowed* to fail -- falling back is the design -- no longer prints VTK's
wall of red on the way: the error catcher keeps the messages off the
Python log (`send_to_logging=False`) and the VTK logger is held off
stderr for the attempt, restored after. What remains is the one honest
`UserWarning` naming the implicit grid's step. And importing geoml sets
`TF_CPP_MIN_LOG_LEVEL=1` (via `setdefault`, so a user's own choice wins
and `0` brings everything back), which silences TensorFlow's C++ INFO
wall -- oneDNN notices, XLA compilation, ptxas register spills -- from
about twenty-five lines at import and first prediction down to two: the
absl preamble, and the one line saying which GPU came up with how much
memory, which is the line worth keeping.
* **A mesh can be simplified to an error budget, and smoothed with its
eyes open.** `Mesh3D.simplify(max_error)` decimates to the fewest
triangles a *geometric* budget allows -- how far, in the mesh's own units,
the simplified surface may sit from the original -- so the same call means
the same thing on a coarse shell and a fine one, which a fraction of
triangles would not. The budget is enforced by measurement: vtkDecimatePro
carries the only error-bounded decimator and its own metric was measured
running 7x loose at tight budgets, so the simplified faces are probed
against the original and the internal bound tightened until the promise
holds. Measured on a contoured shell (unit steps): budget 0.02 gives 1.2x
fewer triangles at true deviation 0.017, budget 0.5 gives 9x at 0.306,
budget 2 gives 13x at 0.404 -- closed at every level, the caller's kind
kept, the kind's own invariants refusing a reduction that breaks its
promise. `get_contour(..., simplify=budget)` applies it on the way out, on
grids and block sets both. `Mesh3D.smooth(iterations, pass_band)` is
Taubin's non-shrinking filter (0.04% volume drift measured), priced
honestly in its docstring: on a block-model contour it was measured to
take a sixth of the faceting and move the surface 50% further from the
true level set, which is why no contour smooths itself and `supersample`
remains the accuracy-preserving route. `simplify` scales: a mesh past
100k triangles takes a fast quadric pre-pass first -- verified against
the original like every other step and given half the budget, so the
error contract stays global -- and the error-bounded decimator then works
a fraction of the triangles; the distance function behind the
measurements is built once per call rather than per loop pass. Measured
on an 835k-triangle shell: 17 s to 7 s, with the pre-pass leaving
better-conditioned input and the output smaller at the same budget (4.8k
against 7.8k triangles). Decimation can also leave a few triangles wound
against their neighbours -- reported from a real model as an
`InconsistentMeshError` at the rebuild -- and winding is bookkeeping, not
shape: on that error alone the triangles are made to agree and face
outward, as `heal()` would, and the build is retried once. A mesh failing
for any other reason (a solid opened, a terrain folded) still refuses,
since closing a hole or unfolding a sheet would be inventing geometry.
* **The booleans gained a second engine, and the failing cases now come
back right.** Found while measuring: VTK's exact filter produces
structurally wrong *output* on contour-derived meshes -- a contoured shell
against a box came back with 21 open and 305 reversed edges, and the
damage is not a repairable seam: not T-junctions (measured directly -- no
vertex rests on any broken edge), not coincidence (an offset cut fails the
same), not slivers (simplified inputs fail the same); the filter drops or
fabricates entire patches, beyond `heal` at any hole size and welding at
any tolerance. So wherever the exact engine fails -- errors alongside
output, a product failing the `Solid3D` invariants, or an empty answer
when the vertex-sides test says the surfaces genuinely cross --
`_implicit_combine` answers instead: each body's signed distance sampled
on a ~2M-cell grid over the region the answer can occupy, `max`/`min`/
`max(a,-b)` for intersection/union/difference, and the zero surface
contoured back. There is no seam geometry to walk, which is what makes it
robust; the price is honest (exact to the grid's step) and said out loud
in a `UserWarning` naming the step. The previously failing shell-vs-box
cut now returns a closed body within 0.11% of a Monte-Carlo reference.
The fields are evaluated in a *band*: a distance field is 1-Lipschitz, so
a coarse pre-pass (one sample per 4x4x4 fine cells) proves which fine
points can hold no zero crossing and settles their signs without a query
-- exact values survive everywhere they can matter, measured at perfect
sign agreement and zero in-band error. Each body's band is further masked
by the other's coarse field, so a small shell against a whole topography
resolves the topography around the shell alone. Measured on a 21k-triangle
shell cut by a 29k-triangle terrain: the distance sweeps went from 68 s to
under 4 s, the whole intersection from 72 s to 8 s, bit-identical output.
A simplified-proxy variant (evaluating against decimated inputs) measured
2.5x more on top and was rejected: it trades the answer's exactness at
grid resolution for speed the banding already found.
`get_contour`'s own output was never the defect -- everything it returns
passes the closed/consistent invariants at construction.
* **`BlockSet3D.get_contour` takes the tree path.** The pair of string
arguments (`variable`, `attribute="prediction"`) collapsed into the one
address the container already understands: `get_contour("metals/zn", 1.5)`
contours the component's prediction, `get_contour("g/average_sim", 1.5)`
names another column outright, and a bare `get_contour("zn", 1.5)` stays as
sugar over the same tree — resolved by a `select("**/zn")` search, and
refused with the candidates listed when the name sits in more than one
place. The `attribute=` keyword is gone (breaking; the path carries it).
Underneath, the pyvista label is looked up in the `geoml_paths` table the
export itself writes rather than reconstructed from name and role — the
last constructed-label reader in the package, which is what used to break
whenever a label's spelling changed. Errors ride the tree too: a wrong
path reports from as deep as it reached, and a composition still says
"made of components; name one of ...".
* **The mesh booleans survive mine-grid coordinates, and a failed boolean is
refused rather than guessed at.** Reported on a real case: `intersection` on
a solid with a DTM logged VTK errors ("No cell with correct orientation
found") and handed the solid back unchanged. Two defects compounded. VTK's
intersection filter works to absolute tolerances, so at UTM-scale
coordinates (~1e6) it fails on geometry that cuts cleanly at the origin —
the booleans and the sheet clip now run in a local frame (both meshes
translated by their shared box corner, rounded so the shift costs nothing,
and translated back). And a failed boolean answers with an empty mesh,
exactly what two non-crossing bodies answer — with VTK logging errors in
the legitimate cases too, so the errors cannot arbitrate. The vertices can:
a surface crossing another has vertices on both sides of it, so the
containment fallback now tests every vertex (it read one), returns the
nested/disjoint answers as before when each body lies wholly on one side,
and hands mixed sides — the case that used to come back as a wrong body
with no remark — to the implicit engine above. Regression tests cut
the same irregular terrain at the origin and at (500 000, 7 000 000):
volumes agree to VTK's own cut-line noise (~2e-6), where the old code came
back empty or unclosed.

## version 0.6.0
* **The package has a shape.** What was twenty-two flat modules is now five
subpackages and the flat survivors. `geoml/data/` holds the 7,200-line
monolith split into seven modules, each answering one question — `base` (the
container tree: errors, paths, the traversal, the attribute), `variables`,
`containers`, `grids`, `meshes`, `blocks`, `io` (the Zarr writers) — plus
`drillhole` and `inducing`, which moved in beside them because both are
data. `geoml/latent/` holds the modelling paradigms side by side:
`network.py` is latent.py as it was, `fourier.py` is projnet.py — dark on
purpose: unadvertised, untested, imported only by name. `geoml/math/` holds
`geometry` (now also owning the sub-block lattice arithmetic and
`bounding_box`, drained from the monolith under the arrays-in/arrays-out
rule), `interpolate`, and the tftools split — `tf.py` the four helpers in
everyday use, `linalg.py` the seventeen solvers parked for future work.
`geoml/stats/` holds `probability` and `random`; `geoml/viz/` holds
`plotly`, `pyvista` and `graphviz`, ending three modules shadowing the
packages they build for. Where a mesh method names a container class or a
container names the Zarr writers, the import happens late and says why.
* **Nothing a user or a saved model holds breaks.** Every old dotted path
resolves: the subpackage facades re-export what the flat modules held
(`geoml.data.PointData`, `geoml.latent.BasicGP` — the paths saved models
replay), and a one-line shim sits at each old flat path (`geoml.tftools`,
`geoml.drillhole`, ...) for one release. Saves written from here on record
the new module paths, so older geoML versions will not open them — the
reverse direction is guaranteed, that one is not.
* **The public surface is curated.** `geoml.__all__` names the thirteen
modules a user reaches for; the shims and the internals (`parameter`,
`persistence`, `storage`) stay importable but unadvertised. The everyday
names sit at the root: `geoml.PointData`, `geoml.Grid1D/2D/3D`,
`geoml.BlockSet3D`, `geoml.DrillholeData`, `geoml.VGPNetwork`, `set_seed`.
Kernels, likelihoods and warpings stay module-qualified — `Gaussian` alone
names three different things. `models.__all__` gains `refine` and
`Normalizer`, which were always public in practice; `ProjectedVGP` stays
out deliberately, keeping `fourier`'s company until it earns tests.
* **Packaging is declared once.** `pyproject.toml` (PEP 621) replaces
`setup.py`: same metadata, same dependencies, but subpackages are
discovered instead of hand-listed — the list that had to be edited on every
addition and would have been wrong five times over in this release alone.
* Removed on the way: `data.py`'s stale `__all__`, which named 15 of its
~40 public classes (no `VectorVariable`, no `BlockSet3D`, no
`GaussianData`, none of the errors). It filtered nothing while the module
was reached by attribute and broke the facade the day a star import became
load-bearing; with every import underscore-aliased, bare star semantics
exposes exactly the public surface, with nothing left to drift.

## version 0.5.9
* **A piece of data inside a container has an address.** A container holds
variables, a variable holds components or attributes, and an attribute holds
one array per location; that tree can now be named the way a file system
names a file. `container.get("assay/Zn/noise_variance")` is the column,
`get("assay/Zn")` the component, `get("")` the container itself;
`values(path)` is the array, decoded; `get("assay/Zn/quantiles/1.5")` reaches
into a dict family, with `get(path, 1.5)` as sugar; `_metadata/HOLEID` is a
metadata column, under a reserved root so a fold over the modelled columns
can leave it alone; `assay/Zn/simulations/7` is one realization, read by
indexing the store rather than materializing it. The scheme is not new — the
Zarr persistence has composed exactly these strings since it was written;
this promotes them out of the store and into the API. Design and prior art in
`docs/variable-paths.md`
* `select(pattern)` asks the tree for a set at once: `**/prediction` is every
prediction anywhere, `assay/*` one segment down, `assay/**` a subtree. `*`
does not cross a `/`; `**` spans zero or more segments; `filled=True` keeps
only columns holding something — the one thing no pattern can express. A bare
`**` deliberately does not unroll the realization axis, or a default export
of a simulated model would emit a hundred arrays per variable
* `container.tree()` prints the whole thing — every variable, component and
column, its dtype, and `empty` where nothing was ever written, which is the
question that used to take a session of reading a ParaView array list.
`status=False` skips the filled check on disk-backed models
* **One enumeration behind every export, and one spelling.** The four ways of
naming the same quantity — `assay/Zn/noise_variance` in Zarr,
`assay - Zn - noise_variance` in one pyvista path, `assay - Zn - noise
variance` in another, `assay_Zn_noise_variance` in a frame — collapse to
`render(path, style)` over one walk. Thirty-five per-class export bodies
(fourteen `as_data_frame`, twenty-one `fill_pyvista_*`) become one fold each,
which is where seven of the eight bugs in the components family lived: a
hand-written list cannot miss half of itself once it is not written by hand.
`proportions` and `divided` reach the exports for the first time, on every
path; the measured column is `au_measurements` now, not bare `au`; a column
nothing wrote is left out instead of exported empty; a collision made by
flattening (`noise/variance` beside `noise_variance`) is resolved
deterministically and warned about
* `as_data_frame(include="**/prediction")` and
`as_pyvista(include="assay/**")` choose what leaves the container with one
pattern instead of a flag per family; `simulations=` keeps its selector
meaning and composes with it. `columns="multi"` returns a `MultiIndex` level
per path segment for whoever is staying in pandas — off by default, because
written to CSV it makes several header rows that other software reads as data
* Every pyvista export now carries the metadata columns (only the block set
did before — an air code or a fold is as useful draped over a grid) and
`field_data["geoml_paths"]`, a JSON table from each array's label back to the
path that produced it: the flat spellings are not invertible, so the file
carries the mapping instead of anyone guessing
* **Subsetting no longer reads every realization of everything.** Three
nested `deepcopy`s each read whole stores — the container copied all its
variables before subsetting them, a variable its `(n_data, n_sim)`
simulations, and an attribute's back-reference dragged in every other
variable of the container. On a block model whose simulations live on disk
precisely because they do not fit in memory, any one of the three was the end
of the session. A test now watches a subset go through with a store that
raises on any whole read
* **A rebuilt variable keeps what its nodes know.** Cut-offs declared on a
composition's components were dropped by `from_variable`, so a block model
predicted from one could not compute its shares and `refine` had nothing to
split on — the third time cut-offs specifically were the casualty. Node facts
are declared (`_NODE_ATTRS`) and carried wholesale by every rebuild, so the
next fact added is carried the day it is declared
* **A category keeps the same share dicts a grade does.** `proportions` and
`divided`, keyed by the one cut-off a category has: zero on `ind_skew`, whose
zero level set the contact is. The last storage asymmetry between the two
kinds of decision, and with it `split_shares` collapses to one body and one
naming hook — a grade renders `@ 1.5` because someone declared that number, a
category stays bare because its zero is an artefact of the log-odds.
`block_shares()` labels are pinned unchanged by a test
* Subsetting a `BinaryVariable` never cut its `probability`: the old override
wrote the subset into a dead `average` attribute nothing ever read. Fixed by
the fold rather than by hand
* **The Zarr store format moves to 2** — the dict families and node facts
persist off the declarations, under the same keys `get` takes
(`v/quantiles/0.5`, not `quantile_0.5`) — and `open` now checks the format,
refusing a mismatched store with what was written and what it reads rather
than half-loading it. No shim: agreed before the work started, everything in
use being refreshed after it
* **`plots.prepare` takes paths, and the dotted form is gone.**
`"Elements/uncertainty"` where `"Elements.uncertainty"` used to be — removed
rather than aliased, refused with the replacement spelled out, since a `.`
inside a label would make a wrong guess look like a working one. The
duplicate component search in `prepare` (a near-copy of `data.py`'s, and the
original symptom of the whole problem) is deleted; the one resolver lives on
`_SpatialData`, every container inherits it, and the plots see components of
every variable kind the way `get_contour` always did
* `set_coordinates` — six hand-written lists, four missing columns — is one
walk; the subset, the carry, the frame, the fills, the persistence and the
repr all run on the same traversal, so a column or a family added to the
declarations reaches all of them the day it is added
* **Every grid and block class builds itself around data:
`from_data(data, step, margin=0.1, decimals=0)`.** `data` is any spatial
object — drillholes included, whose desurveyed cloud stands in for the
coordinates they do not have — `margin` a fraction of its extent (one number,
per side, per axis, or both), and `decimals` how many decimals the box corner
is floored to: round numbers read better on a section, flooring rather than
rounding means the margin is never eaten, and the count of steps grows to
keep the far side covered. On the rotated classes `decimals` also rounds the
fitted azimuth, dip and rake (degrees) — a grid at 47.3182 degrees is
nobody's intention — and the rounding happens *before* the box is measured,
so the data stays covered by the rounded frame
* **`RotatedBlockSet3D`: the variable-size block model, rotated.** The
lattice is `BlockSet3D`'s untouched — splitting, grouping, refinement and the
integer arithmetic all happen in the unrotated frame, which is what keeps
them exact — and the rotation is applied where coordinates leave (centres,
the sub-block fan-out a prediction reads, the exported hexahedra) and removed
where they come in (`index_data`). Every mesh test and assignment reads
sub-block positions through `get_batched_coordinates`, so geometry against
surfaces and solids works in world coordinates with nothing overridden;
`split` keeps the rotation, the Zarr round trip keeps the angles, and a model
predicts onto it unchanged
* **One `aggregate(data, variables=None, metadata=True)` replaces the nine
per-class `aggregate_numeric`/`_categorical`/`_binary`.** Each variable says
what it is and the operation follows: continuous values (and each part of a
vector) average; categories keep the label most often measured in the cell,
both sides of a contact voting; a composition is averaged and *closed again*,
a cell missing any part coming back missing whole; numeric metadata averages
and coded metadata keeps the dominant label. **What is truly ambiguous is
empty**: two labels tied for a cell name no winner, where the old aggregators
picked whichever sorted last and reported an answer that was not there. Built
on `index_data`, so a grid, a rotated grid and a block model of several sizes
all aggregate through the same body — the grids used to re-derive their cell
ids in near-duplicate pandas per dimension
* Fixed, pre-existing: `RotatedGrid3D.index_data` applied the forward
rotation where the inverse was needed, so almost nothing landed in the cell
that held it — probed: the odd node of the grid's own coordinates found its
own cell, by luck. Nothing ever noticed because everything downstream of it
raised `NotImplementedError` until the aggregates were unified
* **`container.drop(names)` removes variables.** Whole variables only: a
composition without one part is a different composition, so a component's
name is refused with its owner named — `'a' is a component of 'assay'` —
rather than half a variable being left behind
* **Interval tables can be renamed and dropped.**
`holes.rename_table("assay", "chemistry")` keeps the table's position (the
order is what `as_point_data` merges by default) and refuses a name already
taken; `holes.drop_table(name)` removes one and re-derives the bounding box
* **`recovery` is a column role of its own** in drillhole tables, between
`density` and `flag`: numeric, the share of each interval actually
recovered. It composites as a length-weighted mean — a fraction of a length
is exactly what length-weighting averages — and is never applied as a weight
to the grades, since how much a poorly recovered assay should count is a
modelling decision, not a compositing one. `as_point_data` carries it as
*metadata*, beside `HOLEID` and `LENGTH`: it describes the sample rather
than the ground, so the models never see it. The `flag` docstring loses
"recovery" from its examples, that having been the stopgap
* **`Mixture`: a likelihood whose noise comes from one of several
mechanisms.** Component likelihoods of any kind share the latent location
with a trainable simplex of proportions; the classic pair is a narrow
component for the natural short-range variability and a wide one for
contaminated measurements, which one nugget cannot tell apart — a handful
of bad assays inflates it and blurs every prediction — and which a
heavy-tailed likelihood can absorb but never *name*. The mixture names them
(`responsibilities`, AUC 0.94-0.97 on planted contamination), counts them
(the weight recovered 0.03-0.08 against a true 0.05), and recovers
clean-data accuracy from contaminated data where a Gaussian loses 30-140%
and a Student-t half of that. Every component after the first is taken as
contamination unless said otherwise, and what that means sits entirely at
prediction: training and `measurement_samples` keep the full mixture — the
data must be explained as it is, and a fresh assay can be a bad one — while
`integrated_backward` averages the ground over the genuine components
alone, a contaminated reading replacing the measurement and saying nothing
about the ground it displaced. On a nonlinear warping that exclusion is
worth +6 to +17% of bias; zeroing the contamination's variance instead of
dropping it was measured within 0.15 pp and rejected as the muddier
statement. The noise integral runs each component on its own quadrature
nodes (the new `_noise_values` hook — exact, where a joint quantile would
need root-finding), and the mixture quantile that `measurement_samples`
does need is sixty bisections of the closed-form CDF, once per trace
* **The scalar/multivariate likelihood split is gone.** One
`_ContinuousLikelihood` serves any number of components — `size` is the
warping's — and everywhere the two classes genuinely differed is decided by
`warping.elementwise`, the same flag that already chose the noise nodes:
the training expectation is Gauss-Hermite quadrature when it holds and
Monte Carlo over the latent samples when the warping mixes, and a row
missing a component is dropped whole only in that same case. The
`Multivariate*` twins remain as thin subclasses that only size the default
warping, keeping their historical parameter names and initial values so a
saved model still loads. A vector variable with an elementwise warping now
trains through quadrature rather than sampling — a better integral at the
same cost. `use_monte_carlo` is removed everywhere, having selected nothing
a user should choose; a save that explicitly recorded it no longer replays
* **`ZScore(size, robust=True)` initializes from a winsorized copy of the
data** (clipped at its [1%, 99%] quantiles, the clipped points still
counting from the fence), so a gross outlier cannot set the scale
everything else is normalized by — one was measured compressing the genuine
values into a sliver of a trainable Spline's working window before training
began, and no amount of refinement recovers from that start. Built for the
warping under a `Mixture`, whose docstring prescribes the pairing: the
pathological contamination draw went from 4.3x the clean-data error to 1.3x
with it, the healthy draws and clean data unchanged to three decimals.
Median/IQR initialization was measured and rejected — the same rescue, but
10-16% worse on clean skewed data, legitimate skew being exactly what it
misreads. `Mixture`'s warping is a required argument: the components' own
warpings are inert (frozen at registration), so the mixture's size can only
honestly come from its own
* **`GPOptions(qmc_simulations=True)` draws the posterior simulations from a
seeded-scramble Sobol sequence** — each simulation one point of a
`size x inducing`-dimensional sequence pushed through the normal quantile, so
the ensemble covers the posterior evenly rather than by chance. Measured on
the Walker Lake model at 16-256 simulations: the ensemble mean lands 7-37x
closer to the exact posterior mean the same call reports (converging at
~N^-1.1 against Monte Carlo's N^-0.5), proportions below a cut-off and the
outer quantiles about a quarter closer — the accuracy of half again to twice
the simulations — while the ensemble's own spread and the correlation between
nearby locations gain nothing, and the wall time is identical. Off by
default. The rule is settled by the seed at trace time, so it stays
deterministic, batch-invariant and XLA-compatible; the traced-function cache
now keys on (jit, qmc), one graph per combination, and `predict_measurements`
follows the same option. Monte Carlo remains exactly what it was, down to the
drawn numbers
* **The likelihood noise is integrated out of a prediction rather than drawn.**
What a prediction reports is `E[g(z + eps)]` — the value the ground would show
once the measurement error and the variability below the model's resolution
were averaged over — instead of `g(z + eps)` for some `eps` pulled out of a
random number generator. There is no point in mapping noise, and a block of
any size integrates it away in any case; that is the conceptual difference
from a conventional geostatistical simulation, which conflates signal and
noise. It matters more than it sounds: on the Macpass assays, not integrating
costs 2.8% of a standard deviation on silver and 17.3% on a composition, and
the error is a bias, always in the same direction, because a back-transform
that bends is convex where grades are skewed
* **The accuracy plot asks the model what a measurement would read.** Its
intervals used to come from the stored simulations, which since this release
hold the ground with the noise integrated out — so scoring them against assays
compared an interval for one quantity against observations of another, and
read as wild over-confidence (on the Macpass composition, a spread 4.7 times
too narrow: G fell to 0.22 where the ensemble's own histogram was matching the
data well). `model.predict_measurements(newdata)` returns the predictive
distribution of a *sample* instead — `n_sim * n_nodes` equally likely values
per location, the same computation `predict` makes but stopped one step
earlier, before the nodes are averaged. Equal-share nodes rather than
Gauss-Hermite: a rule built to make an integral exact carries a few far-flung
points at weights of 1e-4, which is excellent for a mean and useless as a
picture of a distribution. Nothing is stored, `newdata` is untouched, and
`variable.simulations` keeps its one meaning. Measured on a synthetic
reproduction of the Macpass case, this puts G at 0.96-0.99 where the stored
simulations gave 0.14-0.24, and where widening them by `noise_variance` — the
cheap alternative, tried and rejected — reached only 0.78-0.87 while
over-covering in the middle, a symmetric interval being the wrong shape for a
skewed conditional. `Explorer.accuracy` and `Interactive.accuracy` now need
the model the selection was built with, and build the array once per
selection. A likelihood carrying no warping is passed over rather than asked:
a categorical one keeps its noise in the probabilities and has no value for a
sample to scatter around, which `_Likelihood.warped` now says out loud instead
of leaving to whoever reaches for `warping` first
* **`spread_check`**, a new figure in both backends: what the model claims a
value's spread is, against what it turned out to be, along the predicted
value. A residual holds two things at once — how wrong the model was about the
ground, and how far the assay fell from the ground — so it can only be read
against the two together, which is why the figure carries three things: the
noise as a band, the whole claim as a line, and the observed root mean square
residual as points. On the line is calibrated, below it hedging, above it
over-confident. The level axis is what says *which* term is at fault: a
warping bends, so the noise grows with the value while the model's own
uncertainty does not, and a shortfall widening with the grade is the noise
where a flat one is the posterior. Points sitting inside the noise band alone
are the plainest case — the fitted noise over-explains the errors by itself,
which is what an under-trained model looks like. Unlike `accuracy` it needs no
model: everything it reads is on the container. `bins` takes a count or the
positions, as a histogram does; a count gives **equal-count** bins, a
predicted grade being skewed enough that equal width would leave the top bins
holding a sample each. The per-bin values are drawn as steps rather than as a
line through the centres — one number per bin is not a curve — and the points
span their bin, so a wide bar reads as a thin stretch of data
* Subsetting a composition carries every column. `_Component.__getitem__`
listed the ones to cut by hand and stopped at `prediction`, so a held-out set
came back with `dispersion` and `noise_variance` still at the *original*
length — wrong only later, in whatever read one against the new length. Both
it and `ContinuousVariable.__getitem__` now walk `_ZARR_ATTRS`, the list the
Zarr round trip and `carry_to` already use, so a column added in one place is
subset without being named a fourth time
* A composition's parts now carry a `dispersion`, which they never had:
`CompositionalVariable.update` overrides `VectorVariable`'s and its parts are
`_Component`, which overrides `ContinuousVariable`'s and reads different keys,
so the column added when block discretization went in reached every kind of
variable except this one. A compositional block model reported no within-block
spread on any part, silently
* `grade_tonnage(log_mass=True)` puts the tonnage on a logarithmic scale, in
both backends. Most of a deposit clears the low cut-offs, so on a linear axis
the high ones are a flat line along the bottom and the spread between the
realizations there — which is where the decision usually is — cannot be seen
at all. Only the tonnage: the grade axis spans one order of magnitude at most
and a log scale there would say nothing
* A continuous variable gained a third variance, `noise_variance`, and with it
a third question. `latent_variance` is how sure the model is of the value,
`dispersion` is how much the ground varies inside a block, and this is how far
a fresh *measurement* of the value would fall from it — the likelihood noise
carried into the variable's own units. It is the second moment of the same
quadrature that produces the value, so it costs nothing beyond an accumulator,
and it is what has to be added back to compare a prediction with an assay,
since the prediction itself reports the ground. On a block it averages over
the sub-blocks, being what a sample taken inside the block would read. Missing
rather than zero where the prediction was made with `include_noise=False`:
nobody claimed a measurement there is exact
* The consequences run through the whole prediction path. There is no longer a
noisy field and a clean one to keep apart, so a block's cut-off shares, its
dispersion and its simulations are read from one set of numbers and the
back-transform runs **once instead of twice**. Nothing is drawn, so the noise
follows no seed and cannot depend on how a prediction was batched. And the
noise no longer contributes randomness that refining could not resolve, which
`refine`'s criteria used to have to look past
* `include_noise` is now a boolean and defaults to `True`. The `'delta'`
method is gone: measured against a 40 001-node reference on a Walker-style
chain it left an error of 7.0% of a standard deviation — *exactly* what not
correcting at all costs, the third-order terms being the same size as the
second. `white_noise` and `_add_noise` are gone with it
* How the integral is done is decided by the warping, and the two cases differ
only in which array of nodes is used — nothing in `likelihood.py` asks which
one it got. A warping that works on each component alone gets **eight
Gauss-Hermite nodes**, the same node applied to every component, so the cost
does not grow with their number; it is exact to five figures. One that mixes
them has to be integrated over all of them at once and gets **64 scrambled
Sobol points**, reaching 0.2-0.6% where plain Monte Carlo of the same size
gives 4-9%. The scramble is seeded, so the rule is fixed rather than random.
`_Warping.elementwise` is what picks, `False` for `PCA`/`RobustPCA`,
`CenteredLogRatio`, `Rotation`, `ScaledSimplex` and
`ContinuousNormalizingFlow`, and a chain is elementwise only if every link is
— one mixing link and everything after it sees the mixture, which is worth
3-4% of a standard deviation on the Macpass chains. A test checks every
warping's declaration against a numerical Jacobian
* Non-Gaussian noise needs nothing special. The nodes live in the unit cube
and reach the noise through the likelihood's own quantile function, which is
how it was ever drawn, so `Laplace`, `Huber` and `EpsilonInsensitive` go
through the same code
* `warping.Center` could not be constructed at all: `_np.ones[size]` where
`_np.ones(size)` was meant
* Points made from drillholes remember where they came from. Every conversion
— `as_point_data`, `get_contacts` and `as_classification_input` — carries the
hole as the metadata column `HOLEID` and the sample's own length as `LENGTH`.
Metadata rather than variables, so the models never see them while subsetting,
`as_data_frame` and `to_zarr` all keep them: `HOLEID` is what a
leave-one-hole-out split reads, and `LENGTH` is the support each value stands
for, which any weighting by it needs. A contact has no length, so
`get_contacts` records none; in `as_classification_input` the contacts come
through at zero, the zero support being the whole reason they are added
* **A category is scored against the best of the others in two row maxima.**
`_CategoricalLikelihood.entropy_and_indicators` built `ind_skew` by masking
one category out at a time — a scatter over the whole `(n, n_cat)` indicator
matrix per category, driven by a `tf.map_fn`, so the work grew with the
number of categories and the loop went through a `while_loop` on the way. It
is the same quantity read differently: the rival of whoever wins is the
runner-up and the rival of everybody else is the winner, so the row maximum
and the row maximum with the winners dropped answer for every category at
once. Where two categories share the maximum they are each other's rival, so
both still come out at zero and the contact stays the zero level set — which
is why the count of winners is taken rather than assumed to be one. Bit for
bit the same numbers (the old sentinel was -999 and an indicator is at worst
`log(1e-6)`, so the masking never bound), and on the GPU 4.8x faster at two
categories, 11.9x at twelve — 0.5-0.8 ms a batch of 20 000 rows against
2.6-9.7, and flat in the category count where it used to grow with it. This
matters at every prediction on a discretized block model, where the call
happens before aggregation and so sees `prod(discretization)` times as many
rows. A top-2 pass was the obvious replacement and is *not* what went in:
`tf.math.top_k` costs a flat ~19 ms a call on the GPU whatever its shape,
dtype or `k`, so it would have been two to seven times slower than the
scatters it replaced while looking like an optimization on paper
* The two nested Python loops that built `log_prob_final` in
`CategoricalGaussianIndicator.predict` and `HierarchicalGaussianIndicator.
predict` — "the probability of being class i and not being the others",
assembled a column at a time — are one broadcast: the row of negative
log-probabilities summed, with each category's own swapped for its positive
one. **This is a graph-size change, not a speed one**, and it is worth saying
which: prediction runs 1.0-1.3x faster, because the time goes into the
`log_prob`/`log_survival_function` pair rather than the summation, but the
graph drops from 2427 operations to 102 at 24 categories (687 at 12), which
is 12x less tracing and, with `jit_predict`, a 4.0 s XLA compilation instead
of 1.6. The reformulation sums the whole row and takes one term back out
where the loop never added it, so the numbers are no longer bit-identical:
checked against the pre-change file across every output of `predict` for both
classes, the worst disagreement is 4e-15 relative, four ulps, and it stays
there when the variance is small enough to push the log-probabilities to 1e7
* **A composition's parts export everything the other continuous variables
do.** `as_data_frame` gave `assay_a_prediction` and stopped there, leaving out
`dispersion` and `noise_variance`, so a block model's composition reported
neither — the columns were written, filled and persisted, and only the export
could not see them. `_Component.as_data_frame` was a copy of
`ContinuousVariable`'s minus the latent columns, so every column added
upstream had to be added there too and the omission was silent. It delegates
now, with `latent=False`, which is what the copy was for; what the parent
writes is guarded on the attribute being there and holding something, so a
component's absent latent space and a column no prediction ever filled are
both simply skipped. This is the fifth bug of that family — `update`,
`__getitem__` and two rounds of `carry_to` were the others — and the second
fixed by deleting a duplicate rather than adding to it
* **An exported component carries the name of the variable it belongs to.**
`assay - a - prediction` in pyvista, where it used to be `a - prediction`: a
bare `a` says nothing about which assay it is, and two variables may each hold
a component so named, in which case one quietly overwrote the other.
Categories already did this (`rock - granite - probability`) — the continuous
side accepted a `prefix` argument in all three `fill_pyvista_*` and dropped it
on the floor. `BlockSet3D.get_contour` reads those labels back, so it now
takes the owner's name from `_variable_or_component`, which returns it
alongside the component

## version 0.5.7
* `BlockSet3D.index_data` locates sample data in a model of several block
sizes, and `aggregate_numeric`/`aggregate_categorical`/`aggregate_binary`
follow from it — the last of the block model's methods that a grid had and it
did not. It answers with **which block**, not with a cell index per axis:
blocks of several sizes have no per-axis index to give. The lattice makes it
cheap, a location's base cell being arithmetic and the block covering it the
one whose origin is that cell's ancestor at its own level, so one
`searchsorted` per level settles every location
* `BlockSet3D.group(mask)` is the inverse of `split`: whole families of
children back into the parent they came from. The mask must name every child
of a parent or none — a partial family would average over children that are
not there, the mass-conservation error the lattice exists to prevent, so it is
refused rather than approximated. A block that was not grouped keeps what it
holds, as in `split` and for the same reason; the parents come back missing,
because coarsening is a change of support and a parent's spread, within-block
dispersion and categories are not its children's. Metadata crosses from the
first child, describing the ground rather than the model
* `refine(..., where="column")` takes the name of a boolean metadata column,
which is what a stored filter is here — `assign_from_solid` writing one being
the usual way to come by it
* A mesh can force the blocks it runs through to split.
`BlockSet3D.crossed_by(mesh)` marks the blocks whose sub-blocks fall on both
sides of a surface or a closed body — the question `needs_splitting` asks of a
cut-off, asked of geometry instead. A block a topography or a vein wall passes
through holds two answers whatever is predicted into it, and geometry says so
before any prediction does; one entirely above or entirely below is left alone
however close it lies, which is what keeps it from refining a whole domain.
Hand the mask to `split`, or give `models.refine(..., meshes=[...])` the
meshes and it unions this with the other two criteria. It needs no model at
all, so a block model can be refined at a topography before anything has been
predicted into it — the loop is three lines and the docstring has it
* `BlockSet3D` gained the `assign_from_surface`/`assign_from_solid` overrides
`Blocks3D` already had, so `fraction=` measures the share of a block below a
sheet or inside a body over its own sub-blocks. Before this it fell back to
`PointData`'s, which knows only the block centre, so partial blocks were not
available on a variable-size model at all. The shared parts of both now live
in `_sub_block_shares`/`_blocks_from_surface`/`_blocks_from_solid` at module
level, which `Blocks3D` and `BlockSet3D` both call rather than keeping two
copies of the geometry
* `models.refine(..., where=mask)` names the ground worth modelling. The blocks
left out are never predicted at any pass and never cut either — they hold
nothing to decide and draw no surface, so cutting them would only make more of
nothing. This is what a block model has instead of being subsettable: air above
a topography or ground beyond a lease boundary costs nothing, rather than
costing a prediction that is then filtered out of the reports. The mask is
given once, against the blocks as they stand, and carried across each split.
`predict(..., where=)` was relaxed to make the first pass possible: naming only
some locations of a container that does not carry the variable yet used to
raise, and now creates the variable and leaves the rest at `nan` — which is
what `unpredicted` reads and what the reporting layer already skips. An
existing variable is still never reallocated under `where`, since that would
wipe what the untouched locations hold
* `BlockSet3D` now refuses to be subsetted — `blocks[mask]` and
`subset_region` raise `TypeError` instead of quietly handing back a plain
`PointData`. That object had no origin, no level and no block size, so the
size columns vanished from `as_data_frame`, `as_pyvista` drew points rather
than blocks and `grade_tonnage` refused it, all while looking like a block
model. The class is structurally complete by design: every block tiles its box,
which is what keeps a coarsening group from averaging over children that are
no longer there. Ground is excluded by value instead, and the message names
the ways — a metadata column for the exclusion, `predict(..., where=...)` to
visit part of a model without making a smaller one, and
`as_data_frame`/`as_pyvista`/`to_zarr` for a handoff, each of which already
carries the per-block size
* Fixed: `BlockSet3D.get_contour` tore the surface open wherever it crossed a
change of block size — on a real refined model, 1455 holes and 13% of the area
missing. A hexahedron is contoured from its own eight corners, so a coarse
block cannot see the corners its finer neighbours place in the middle of the
face they share; the two sides draw different curves there. The blocks a
surface runs through are now cut to the finest size the lattice allows before
contouring, in the mesh handed to VTK and not in the model, which puts the
interfaces out of the surface's way. Nothing is predicted: a child takes its
parent's value plus what the block's corners say about the shape running
across it, and since the trilinear weights are symmetric about the centre that
correction cancels over the children, so a block's estimate stays exactly the
mean of the children standing in for it. The surface comes back the one a
model carried at the finest size throughout would have drawn — 0.01% in area
against 13% before — for 13% more cells in the mesh and none in the model.
Widening the refinement instead, which `docs/variable-block-models.md` §5.1
used to recommend, is equally correct and costs five times the blocks to
predict; §5.1.1 has the comparison
* `BlockSet3D.get_contour(..., supersample=1)` cuts that mesh a further level
past the model's finest block, which costs no prediction and makes the surface
both rounder and closer to the level set it stands for. What VTK reads between
block corners is trilinear — continuous, but creased at every face it crosses,
and on a field with structure at a few blocks the creases are what looks
blocky. Averaging onto corners again at each level composes into a rounder
reconstruction, so the extra levels converge on the field rather than on the
trilinear caricature of it: on a rough test field one level took the mean
angle between neighbouring triangles from 9.1° to 5.7° and halved the distance
from the true surface, matching a model of 3.4 times as many blocks. Smoothing
the triangles instead was measured and rejected — Taubin bought a sixth of the
faceting for 50% more distance from the true surface, which is a rounder
picture at the price of a less true one. `supersample=0` leaves the mesh at the
model's own resolution
* `models.refine` also cuts a block whose neighbour ended up more than one
level finer than itself, and `BlockSet3D.unbalanced(gap=1)` is what finds
them. Such a block is undecided by its own reckoning — its sub-blocks agree,
so `needs_splitting` rightly leaves it alone — but a neighbour cut twice over
is evidence that the field turns sharply nearby, and that evidence sits
outside the block where nothing was looking. It shows up in what gets drawn: a
contour reads a block through its eight corners, so a coarse block beside much
finer ones lays a crude straight guess right where the surface runs. Levelling
those jumps measured **three times closer to the true surface for a third more
blocks**, where refining a whole level deeper without it bought almost nothing
for 2.6 times as many — deeper refinement widens the jumps as fast as it
narrows the blocks. The search runs from the fine side, each fine block
stepping one cell past its faces to name the coarse block covering that point,
which is exact where asking the coarse block is not and needs no array the
shape of the base lattice
* Fixed: `BlockSet3D.get_contour` could not name a component of a vector or
compositional variable, only the variable holding it — but only the components
carry a grade, so there was nothing to contour under the name it accepted
* Fixed: refining a model whose variable has components — a vector or
compositional one — raised `AttributeError: 'collections.OrderedDict' object
has no attribute '_has_content'`. `divided` was two different things depending
on the class, a single column on a category and a dict keyed by cut-off on a
grade, and the criterion reached in for it from outside and hoped. Each
variable now reports its own decisions through `split_shares`, so nothing has
to guess. Cut-offs are declared per component (two grades are judged against
two different numbers) and travel to the model as a matrix with a row each,
short rows padded with infinity so the spare columns come back empty
* Fixed: `VectorVariable.from_variable` built its components afresh and left
their cut-offs behind, so a model predicted into a new container had none and
quietly refined nothing
* `docs/variable-block-models.md` §11 works the whole thing through on the
Macpass drillholes, from declaring a cut-off to the grade-tonnage curve and
the grade shell, with the output of an actual run against each snippet
* `geoml.models.refine(model, blocks)` predicts on a coarse block
model, works out which blocks cannot decide, cuts those, and predicts on what
the cutting made — until there is nothing left to cut. It needs no telling how
many passes that is: each one takes the blocks it splits a level finer and the
criterion never marks a block already at the lattice's `max_levels`, so it
runs out on its own within that many passes. How fine the model may go was
settled when the block set was made, which is the one place it belongs. Only
the new blocks are visited at each pass, a block that was not split being the
same block on the same support. On a compact ore body in a barren domain it reached 4309 blocks where
the equivalent uniform model needed 131 072, the coarse ones sitting where the
grade is plainly below the cut-off and the fine ones spanning it
* What decides a split is one question asked of every variable: *what does a
value have to cross for the decision to change?* A grade crosses the cut-offs
someone declared — `variable.set_cutoffs([...])`, carried to whatever is
predicted from that data, and `None` for a variable that takes no part, which
is the answer for a composition's rest component. A category crosses zero:
`ind_skew` is its log-odds against its best rival, so the contact between two
categories is exactly its zero level set. One reduction serves both, and the
categorical case needs no special handling at all
* Two numbers come back per cut-off, and they answer different questions.
`proportions` is how much of the block sits below it — the recoverable share,
and per category on a categorical variable the partial-block domaining number,
worth having whether or not anything is ever refined. `divided` is how often
the cut-off passes *through* the block. Only the second licenses a split, and
the difference is the whole of what refining can fix: realizations either side
of a cut-off are the model not knowing, which no amount of cutting will settle
— that wants another drillhole — while sub-blocks either side *within one
realization* are two answers in one block, which cutting separates
* The splitting criterion, and `dispersion` with it, are read from the
**noise-free** field. Noise is the part of a block's spread that cutting
cannot resolve, so a block straddling a cut-off only on account of it would be
cut for nothing, and a dispersion carrying it would report a variability that
is not the ground's. The predictions themselves still carry noise as
`include_noise` asks
* A categorical likelihood now works out its entropy, uncertainty and
indicators from the sub-blocks and averages afterwards, where it used to
average the probabilities first and work from those. Averaged first, a block
half granite and half schist looks like a place where the model cannot decide
rather than one where it decides differently in different corners. Two of the
three call sites were also passing block-averaged probabilities alongside
sub-block variances, which this makes moot
* Grade-tonnage counts every block at its own size. It used to take one volume
from `step_size` and multiply the finished curve by it, which a model of one
block size allows and `BlockSet3D` does not; the volume now goes into each
block's weight as the bands are read, so a model half refined reports exactly
the tonnage the coarse one did. `prepare.block_volume` is what asks a
container how big its blocks are — a number where they are all alike, a column
where they are not
* `BlockSet3D.get_contour` draws an isosurface through blocks of more than one
size. `marching_cubes` needs a rectangular array and there is none to give it,
so the blocks go to VTK as hexahedra, which contours an unstructured grid
directly; on a model of one block size it lands within a few percent of what
marching cubes gives on the equivalent grid, and refining away from the
surface leaves it exactly where it was. Values live on the blocks and an
isosurface needs them on the corners, so they are averaged there first — the
blocks meeting at a corner are what decide where the surface passes. A mesh
refined right where the surface runs may still come back open, a coarse face
and the fine faces against it interpolating along their own edges and not
agreeing, so leave a level or two of margin; a cracked answer arrives as a
`Surface3D` rather than a `Solid3D`, never as a body quietly enclosing the
wrong volume, and `heal()` is there for it
* `BlockSet3D.as_pyvista` writes one hexahedron per block, welded so that
blocks sharing a face share its corners — implicit `ImageData` can only state
one spacing, and the point of this container is that there is more than one.
Cell values come from `_Variable.fill_pyvista_cells`, one implementation
serving every variable type rather than the per-type methods a structured
export needs
* `BlockSet3D` is a block model whose blocks need not all be the same size —
fine where the ground is worth resolving, coarse everywhere else. On a real
deposit that is most of the volume: refining to 5 m only around the drillholes
takes a 29-million-cell model under 700 000, and its simulations from 22 GiB
to half of one. Every block's position and size are whole numbers of a **base
cell**, `step / discretization ** max_levels`, and working in those integers
rather than in
metres is what makes the arithmetic exact — blocks meet with no tolerance, a
block is a whole number of its own children, and regrouping conserves mass to
the last digit. It is built full, as a uniform grid, and `split` keeps it that
way, so the blocks always tile their box: `is_full()` says so by counting base
cells, since volume alone cannot tell a gap from an overlap that cancels it.
`discretization` does two jobs, and they are the same job: it is how finely a
block is sampled in order to average it, and it is how that block splits —
each sub-block becomes a child. So the refinement ratio is the
discretization, per axis and not necessarily two, which means an axis can be
left alone (`[2, 2, 1]` refines in plan and keeps the bench height) and
sub-block `j` and child `j` are the same corner of the parent, so what a
coarse prediction already says about its sub-blocks describes the blocks a
split would make. Being the same at every level is the reason nothing
downstream had to change: a block of any size fans out into the same number of
rows, so `_aggregate` still reshapes, batches need not be sorted by level, and
the model never learns that levels exist. Splitting keeps what the blocks that were *not*
split already hold: such a block is the same block on the same support, so its
value is still the right answer for it and arriving at it again would be work
for nothing. Only the children start missing — a parent's value is never
handed down, which would manufacture children agreeing exactly, the one thing
refining is meant to find out rather than assume. `unpredicted()` names them
and `predict(..., where=...)` visits only those, which gives bit-identical
numbers to predicting the whole refined set because a location's simulated
value does not depend on what else is in its batch. Metadata *is* inherited by
the children, unlike a prediction: it describes the ground, and a child sits
on its parent's ground. Per-block sizes are `block_size` and `block_volume`, not
`step_size`: that name means one size for the whole object, and anything
reading it would take the product of the array for a volume. Saves and reopens
through `to_zarr`. The plan it comes from, and the measurements behind it, are
in `docs/variable-block-models.md`
* A predicted `ContinuousVariable` now carries a `dispersion` column: how much
the locations inside each block differ among themselves. It is the variance
over a block's sub-blocks, averaged over the realizations, and it answers a
different question from `latent_variance` — that one says how sure the model
is *of* a block, this one says how much the block varies *within itself*. A
well-drilled block can still be heterogeneous, and it is the second number,
not the first, that says whether cutting the block finer would tell anyone
anything. Nothing in the package reported change of support before this.
It is taken from the sub-blocks before they are averaged away and *after* the
warping has been undone, so it is a spread of grades rather than of the latent
field — the two are not the same thing under a warping and only the first is
reportable. Where a container does not discretize the column stays missing
rather than zero: a location has no interior, and a block the model treats as
its own centre has one it knows nothing about, whereas zero would read as
"uniform inside", which is a claim nobody made. It follows the variable
through subsetting, `as_data_frame`, the pyvista exports and `to_zarr`, and a
vector variable's components each carry their own. See
`docs/variable-block-models.md`, which is what this is the first step of

## version 0.5.6
* `geoml.inducing` builds the inducing points a network is given, which until
now had to be assembled by hand. `from_kmeans` puts them where the data is,
`from_grid` on a regular lattice, and `combine` merges sets — the usual
mixture of a regular backbone and the data's own locations is
`combine(from_grid(data, 50), from_kmeans(data, 200))`. The two `*_experts`
functions build a list, one set per expert, which is what `BasicInput` wants
when a model is divided into local experts. Both make neighbouring experts
overlap, which is what stops a prediction showing a seam where one gives way
to the next. `grid_experts` cuts space into regular blocks, each taking its own
block plus one node of margin all round: every expert holds the same number of
points and its neighbours are known in advance rather than measured — the
Moore neighbourhood, 8 in the plane and 26 in space. `experts` is the unordered
counterpart, for points that follow the data rather than a lattice, as a
drillhole survey does since it does not fill its bounding box. It clusters the
points into groups of about the same size, and each group then borrows a
further `overlap` of its own count from around it — the nearest points, in
Mahalanobis distance, among those belonging to other groups. A borrowed point
keeps its own group as well, so the experts come to share the points between
them. Counting the overlap in points rather than in distance is what keeps the
experts the same size: growing each group's radius instead lets one in a
crowded part of the survey swallow far more than one out on its own. The usual
call divides a set chosen beforehand, `experts(from_kmeans(data, 1500), 12)`.
A cluster of samples taken along one drillhole is nearly a line, whose
covariance is singular and whose Mahalanobis distance across the line would be
unbounded, so the eigenvalues are floored before the distances are measured
* A GP node no longer propagates its inducing points when nothing is built on
top of it. Every expert's set is predicted from every other, so this is the
one step in the network that is quadratic in the number of experts, and on the
last node of a network it was pure waste: nothing ever read the result. On a
single-layer model with 32 experts that alone took a prediction from 1.9 s to
0.18 s. Deeper networks keep paying it where it is genuinely needed
* The refresh that `predict` runs once per call is now traced rather than run
in eager Python. It is arithmetic over parameters that do not move during a
prediction, but run eagerly it pays Python overhead for each covariance block,
which with several experts was most of the call. The trace is kept on the
network so predicting again does not rebuild it, and it reads the parameters
live, so a model trained further still predicts from its new state
* A training step — the ELBO, its gradient and the update — is now traced as
one graph instead of being driven from eager Python. Run eagerly, Adam issues
its update variable by variable at about 1.5 ms each whatever the variable's
size, which is invisible on a small model and dominant on a large one: a GP
node holds three parameters per expert, so 40 experts spent more time in the
optimizer than in the model, 278 ms a step against 98 ms traced. The step is
built once per call to `train_full`/`train_svi` and reused. The cost is a
larger graph to build the first time, which a short run does not repay — the
mixed Jura case, five iterations, went from 30.3 s to 38.5 s — while a real
run recovers it within a few hundred iterations
* `train_svi` passes its directional data correctly. It was building each of
the four directional tensors with a trailing comma, so a one-element tuple
reached the model in place of the tensor
* `GPOptions(jit_predict=True)` compiles the prediction graph with XLA, which
is worth 3 to 5 times on a grid of any size: on a 200 000-node grid a
prediction went from 2.46 s to 0.53 s on the processor, and from 1.05 s to
0.36 s on a GPU; a block model, 16 000 blocks discretized 3x3x3, from 5.27 s to
1.06 s and from 2.26 s to 0.73 s. The gain is not in the arithmetic but in the
memory it saves: the covariance between a batch of locations and the inducing
points builds a difference tensor of one entry per pair per dimension, several
times over, and every one of those is a full pass through memory to compute
something that is only read once. XLA fuses that chain into a single loop and
the intermediates are never written down at all — which is also why the gain is
smaller on a GPU, whose memory is fast enough that the unfused version hurt
less. Two things to know. XLA compiles for the exact shape it is given, so a
grid that `prediction_batch_size` does not divide pays that compilation twice,
about half a second; and it refuses to run anything it cannot compile rather
than falling back quietly, which is why this is an option and not simply how
prediction works. It is off by default. Prediction only, and named so: the same
treatment applied to training was both slower and unstable
* A model built without an `options` argument now gets its own `GPOptions`.
The default was written as `options=GPOptions()`, which Python evaluates once,
when the module is imported, so every model that did not bring its own options
shared a single object and setting an option on one model set it on all of them
* A grade-tonnage curve no longer needs the simulations in memory. It opened
by asking for all of them as one array, which on a real block model — 280 GB
of them on disk — killed the session before any arithmetic happened. They are
now read a band of blocks at a time and reduced as they arrive, so what memory
holds is one band and the answer, which is a few numbers per cut-off per
realization however many blocks there were. The chunking never had to change:
the simulations are already split along the block axis only, so a chunk is a
band of blocks holding every realization of each — exactly what a reduction
over blocks wants. A simulated density is streamed alongside the grade rather
than materialized, since it is as large as the grade itself, and a density
given as a number is now kept as one instead of being spread over an array of
one value per block
* The same curve is also about five times faster. Each cut-off used to take
its own pass over every block, so thirty cut-offs meant thirty passes; now
each block is placed once at the highest cut-off it clears, and summing those
from the top down at the end turns them into the curve. Giving `cutoffs` as
values rather than as a count saves a further pass, the one that would
otherwise go looking for their range. On two million blocks by forty
realizations: 11.2 s and 2.7 GB of working memory, against 2.4 s and none
* `simulation_sample`, which thins the simulations down to what a scatter
matrix can draw, now strides through them while reading rather than
materializing a whole component and throwing most of it away. Which values it
takes follows from the stride alone, so two components chunked differently are
still thinned to the same locations and the pairs stay pairs
* `ArrayStore.row_bands()` is the slices to read a store in, each holding
whole chunks. This is what the above is written against, and what anything
reducing over locations should use
* Fixed: a variable with no simulations was meant to fall back on its
prediction as the only realization there is, but never could. Asked for
simulations it does not have, it hands back `asarray(None)` — a dimensionless
nan, which reads as a value right up to the point where it is asked for its
length, and the fallback was only reached when something raised
* Networks with more than one expert are now tested — building, training,
predicting in one batch and in many, propagating through a second layer, and
surviving a save and load

## version 0.5.5
* `get_contour(value, close="above")` closes a surface where it runs out of
the grid, so what comes back is a body with a volume rather than a sheet with
a hole in its side. It works by padding the cube with one cell of "well
outside" before contouring, which gives the surface somewhere to close inside
the padded volume — no capping geometry, and nothing to go wrong at a corner.
`"above"` keeps the region where the values exceed the contour, a grade shell;
`"below"` the region under it
* `DTM3D` is a terrain: a sheet that promises never to fold back over itself,
so that it stands at one height over each (x, y). That promise is what lets a
body be divided into what lies under it and what lies over it, and it is
checked where the object is made, by asking whether every triangle projects
onto the ground the same way round. Not what `mesh3d` returns — an ordinary
sheet stays a `Surface3D` until a terrain is asked for, this being a promise
to make rather than a fact to detect
* Meshes of different kinds can now be cut against each other, and the answer
keeps the caller's kind. `surface.intersection(solid)` is the part of the
sheet inside the body and `surface.difference(solid)` the part outside, both
sheets, cut where they cross the body rather than along their own triangles.
`solid.intersection(sheet)` is the part of the body below the sheet and
`solid.difference(sheet)` the part above, both bodies — worked by extruding
the sheet downwards into the ground beneath itself and letting the ordinary
body-to-body operations do the rest
* The edge cases raise rather than answer quietly, through four new errors,
all of them `ValueError`: `NotClosedError`, `InconsistentMeshError`,
`NotSingleValuedError` (a sheet that folds has no single "below" to keep) and
`MeshTypeError` (a sheet has no volume to add to a body; a sheet that does not
reach across a body would cut it at its own edge, and says so with both
extents in the message)
* Triangulated meshes are now three classes rather than one. `Mesh3D` is the
primitive — vertices, triangles, normals — and it measures itself as it is
built: its `area`, and whether it is `closed` and `consistent`, meaning its
triangles agree which way is out. `Surface3D` and `Solid3D` are siblings on
top of it, a sheet and a body, each checking its promise where the object is
made. A `Solid3D` therefore always has a `volume` worth reading, and
`assign_from_solid` can trust what it is handed instead of measuring it again
* `mesh3d(points, triangles, normals)` returns whichever of the three the
geometry calls for, and is what `get_contour` and `Mesh3D.from_dxf` now answer
with — so a contour that closes inside the grid comes back as a body, with its
volume, while one the grid cuts off comes back as a sheet
* A body wound inwards is turned round as it is built rather than refused:
nothing about it is ambiguous, only reversed. A closed mesh whose triangles
disagree with each other is a plain `Mesh3D`, being neither sheet nor body
* `Mesh3D.heal()` returns a repaired copy — vertices welded, holes up to a
size covered over, triangles made to agree and turned to face outward. That
last step is not a nicety: filling a hole leaves the new triangles wound
however they came, which would leave the mesh closed and still untestable.
What comes back is whichever class the repair earned, and an empty `Mesh3D` if
nothing survived
* `Mesh3D.split()` returns the connected pieces, each as an object of its own.
A boolean readily answers with a body in several pieces — an ore shell cut in
two — which is one legitimate mesh until you want them apart
* `Solid3D` does `union`, `intersection` and `difference`. VTK is asked first,
but it answers with nothing at all whenever the two surfaces have no face
crossing another — whether they stand apart or one contains the other, and
with no error either way — so an empty answer is not believed but worked out
from which body contains which. A difference where the other body lies wholly
inside is a body with a cavity, written as both surfaces with the inner one
turned inwards, so the volumes subtract and a location in the hollow tests as
outside
* A `Surface3D` can now be read from and written to a DXF file —
`Surface3D.from_dxf(filename)` and `surface.export_dxf(filename)`. Until now a
surface could only be *made*, by contouring a grid; a triangulation somebody
else built had no way in
* The export writes a single `MESH` entity, which holds the vertex list and the
triangles that index into it, so a surface comes back exactly as it went out.
The `POLYFACE` mesh that DXF triangulations are more often written as would
have been the obvious choice, but its vertex indices are 16-bit in most
readers — a ceiling near 32,000 vertices, which a contour off a real grid goes
through without trying
* The import is more forgiving than the export, foreign files not being ours to
choose: `MESH`, `POLYFACE` and loose `3DFACE` entities are all read. The last
two spell out the coordinates of every corner, so a vertex shared by six
triangles arrives six times; those are welded back into shared vertices, and a
face with more than three corners is split into a fan of triangles. Every mesh
in the file is read, so a file holding several bodies comes back as one surface
in several disconnected pieces
* Normals are computed on the way in — the area-weighted mean of the triangles
meeting at each vertex — since a DXF file carries none and `Surface3D` requires
them
* Only the geometry travels. A DXF file has nowhere to put the variables and
metadata a surface carries: `to_zarr` keeps a container whole, and
`as_pyvista` carries the values onto a mesh object
* `ezdxf` is now a dependency
* Surfaces can be assigned to a container's locations, which is what a
surface is usually read in for. `assign_from_surface(surface, name)` records
which side of a sheet each location falls on — above or below a topography, a
seam roof, a weathering front — by comparing it with the sheet's elevation
directly over it, interpolated across the triangle its (x, y) lands in.
`assign_from_solid(solid, name)` asks instead whether a location falls inside
a closed body, an ore envelope or a stope. Both are on every container
* The answer is a metadata column, which is what a domain code should be:
point-wise, carried through `as_data_frame()` and `to_zarr()`, and never seen
by the models. Locations the sheet does not reach are left empty rather than
guessed at
* A grid reads the sheet once for each column of cells rather than once for
each cell — the same (x, y) repeats at every level, and a sheet depends on
nothing else. A `RotatedGrid3D` is not on an axis-aligned lattice and takes
the general path
* A block model can measure the blocks a surface cuts through. The flag
follows the block centre, as a whole-block code does everywhere else, but
naming a `fraction` column as well records the share of each block below the
sheet, or inside the body, over the sub-blocks `discretization` already
defines — what a tonnage near surface needs, where counting a half-buried
block whole is the error
* The mesh arithmetic behind both — splitting faces into triangles, normals
from the triangles meeting at a vertex, counting the edges that belong to one
triangle, interpolating a sheet, testing a body — is in `geoml.geometry`,
taking arrays and returning arrays. `data.py` keeps only what touches a
container, which is the two lines that turn a `Surface3D` into either
* A sheet need not cover everything it is assigned to, and what happens where
it does not is the caller's to choose. Its flag is empty there in any case —
neither above nor below, since the surface says nothing — and the `uncovered`
argument decides the rest: `numpy.nan` by default, so an uncovered block
cannot pass for an empty one in the `fraction` column, `0.0` to count it as
nothing, or `"raise"` to refuse a surface that leaves any location out, which
is the right answer when it was meant to cover the whole model. The footprint
is the triangulation's own, not its convex hull, so a survey boundary, an
L-shaped sheet or a hole around a lake are all respected
* A surface is refused by the assignment it cannot answer: a closed body has
no single elevation above a location, and a sheet has no inside. The two are
told apart by counting the edges that belong to only one triangle — after
welding the vertices, since a mesh can be closed in space while indexing every
triangle's corners separately, as `pyvista.Cylinder` does. A cylinder with one
end open is a sheet by this reckoning, which is the right answer
* A body is also refused when its triangles disagree about which way is out,
because the inside/outside test reads a disagreement as a hole rather than as
an error — quietly, and only where the offending faces are. One wound inwards
throughout is not ambiguous, only reversed, and is turned round rather than
refused; the sign of `geometry.signed_volume` is what tells the two apart, and
its size is the volume the body encloses
* Fixed: `export_contour` raised `TypeError` on every call. It still unpacked
the four arrays `get_contour` once returned, but `get_contour` builds a
`Surface3D` now and hands back that; nothing tested either one, so nothing
noticed

## version 0.5.4
* Printing a variable now says what is on it. `repr` is unchanged — it names
which variable this is — but `str` adds the parts it is made of and the columns
that can be read off it, so they need not be looked up in the source: a
`VectorVariable` lists its components, a categorical one its categories, and
every variable its columns, its quantiles and how many simulations it holds.
Only the names are listed, not whether anything has been written into each
column: knowing that means reading the column, and a column on a large model
lives on disk, which is too much work for a print
* A new `geoml.plots` sub-package, for looking at a data set before modelling
it. `Explorer` holds a container and a choice of variables — one continuous or
vector, one categorical — and answers with figures: `histogram()`, `pairs()`,
`pca(explained=0.9)` and `scene()`. The categorical variable splits and colours
every one of them, which is the question worth asking of most geoscientific
data: does this population behave as one, or as several
* `pca()` draws the components carrying a share of the variance you name, with
the loadings as arrows over the scores. A `CompositionalVariable` is opened up
with the centred log-ratio first — proportions carry a constant sum, so their
covariance is singular and a PCA of them spends a component describing the
closure rather than the data
* `scene()` draws the data where it is, in 1D, 2D or 3D, coloured by a variable
or by one component of a vector variable (`scene(color="Zn")`). For more than a
look at a 3D data set, `as_pyvista()` remains the better road
* The arithmetic behind the figures is in `geoml.plots.prepare`, which draws
nothing and imports no plotting library, so the numbers can be had on their own
and a second way of showing them will read the same functions
* Given a model, `Explorer` reports on it too. `training_curve()` draws the
ELBO with a running mean over it — every value is an estimate, and under
`train_svi` one per *batch*, so the curve is noisy whether or not training has
settled. `transformed_pairs()` passes the measurements through the model's own
warping, already fitted, and asks the two questions a fitted model rests on:
whether the warping made each variable Gaussian, against a normal of the same
mean and spread down the diagonal, and whether what is left is independent,
with the correlation named on every panel. `prediction_scatter()` puts
predicted against measured — one variable with its two distributions along the
sides, where a bias shows that the scatter alone hides, or a panel per
component. `accuracy()` asks whether the simulated spread is the spread the
errors actually have
* `pairs(principal_components=2)` draws the components over the data in the
data's own axes — the reverse of `pca()`, which puts the data on the
components' axes. Each is a line through the mean, one standard deviation of
that component either way, so its length is in the measurements' own units:
long means that direction carries much of the spread, and flat along an axis
means that measurement moves on its own. A line is drawn rather than an arrow
because the sign of a component means nothing, and it is named only where it is
long enough to hang a name on. A `CompositionalVariable` refuses: its
components belong to the log-ratios and are not directions in the proportions
being drawn — `pca()` is the figure for those
* `pairs(upper=...)` fills the upper triangle, which was empty: `"hist2d"` or
`"density"` for where the data sits with the categories pooled, or
`"correlation"` for the coefficient alone, sized by its strength. The lower
half answers which category a point belongs to; the upper answers where the
data as a whole is, which a colour-split scatter hides behind whichever
category was drawn last
* Every figure made of points — `pairs()`, `pca()`, `transformed_pairs()` and
`prediction_scatter()` — takes `kind="hist2d"` for a block model, where the
points are past counting and a scatter is a solid mass whatever its
transparency, `alpha=` for the middling case where transparency is still
enough, and `log_counts=True` to colour the cells by the logarithm of the
count, which is worth having whenever a few cells hold most of the data. Empty
cells are left unpainted rather than drawn as the bottom of the colour scale,
which would fill the panel with a background that looks like data. Counts make
one surface, so a categorical split is pooled rather than drawn five times over
* `transformed_pairs()` numbers its axes `Variable 1`, `Variable 2`, … A
warping may rotate the data or bend it, so a column is generally a mixture of
what was measured rather than any one of it, and naming it after an element
would be wrong in a way nobody would catch
* `pca()` names the variance share to a decimal place — `PC1 (72.2%)` rather
than `PC1 (72%)`
* The figures that compare against observations say so when there is nothing to
compare against. A grid predicted onto carries values everywhere and
measurements nowhere, and `accuracy()`, `histogram()`, `pairs()`, `pca()` and
`transformed_pairs()` now all name that rather than failing somewhere inside
NumPy
* `grade_tonnage()` draws how much material clears each cut-off and how good
it is, on two scales. Without a density it is in volume — or in *area*, for a
two-dimensional grid, which the axis says rather than calling it a volume.
A density may be a number, a metadata column, or a `ContinuousVariable`, and
in that last case its simulations are matched with the grade's **one to one**:
pairing them any other way invents a correlation nobody modelled. Realizations
are carried through separately and drawn as a family with the median over them,
since the curve of the mean model is not the mean of the curves — a cut-off is
a threshold, and averaging either side of it gives different answers
* `grade_tonnage(max_uncertainty=...)` leaves out the blocks the model doubts
most, reading what `Explorer(..., uncertainty=...)` points at. Which column
that is depends on what was modelled — `latent_variance` on a continuous
variable, `uncertainty` on a vector or categorical one, or a column written
alongside — so it is named rather than guessed at. A bare name is looked for on
the variable being drawn and then on the variable *containing* it, which is the
ordinary case rather than an exotic one: grading one component of a vector
variable, the uncertainty that was written belongs to the parent. Failing that,
the metadata. `"Variable.column"` names exactly where to read it, and an array
of one value per location is taken as it comes, which is the way out of every
case a name does not reach. A column that was never filled does not count as
found, so an empty `latent_variance` on a component does not hide the parent's. A block the model cannot
speak for is not tonnage, and counting it flatters the answer at exactly the
cut-offs with least data behind them; the title says how many blocks were kept
* `simulation_pairs()` sets the measurements against the simulations: the
data fills the lower triangle and the simulations the upper, in the same form
and between the same limits, so a simulation that reproduces the data has an
upper half mirroring the lower one. The diagonal carries the measured
histogram with the simulated density drawn over it. Cells rather than points
by default — simulations arrive in location-by-realization blocks running to
millions of values, and a scatter of those is a filled rectangle whatever its
transparency. The two halves are binned according to how many values each
holds, since a few hundred measurements against a hundred thousand simulated
values would otherwise leave the sparse half as scattered single counts
* The simulations are thinned by striding through the block rather than read
whole, one component at a time, so the high-water mark is one component's
simulations and not the whole variable's. The **same stride serves every
component**: thinning them separately would pair the value simulated at one
location with the value simulated at another, and the joint shape — the thing
the figure is for — would be an artefact of the thinning
* `metrics.coverage` and `metrics.goodness` are new, and are plain numbers:
the share of true values falling inside each simulated interval, and Deutsch's
statistic summing that up. Intervals that are too wide count at half the weight
of intervals that are too narrow — claiming a precision the model does not have
is the mistake someone acts on
* The colours are the Explorer's: `palette=` takes a list, or a dict naming
the categories — `{"granite": "#d99"}`, which is what a mapping convention
wants, since a category keeps its colour wherever it lands in the order — and
`cmap=` takes any colormap for the continuous values. The default is `cividis`,
which reads the same to colour-vision-deficient eyes and keeps both its ends
visible against a white background
* Figures are drawn under `geoml.plots.style`, applied through an
`rc_context` to the figure being drawn: importing geoML does not change how
anyone else's plots look. `geoml.plots.PALETTE` is one list, so a rock type
keeps its colour from a histogram to a map; `geoml.plots.use()` takes the
settings globally if you want them
* **matplotlib and plotly are now declared dependencies.** plotly was already
used by `geoml.plotly` and had never been declared. `setup.py` also lists
`geoml.plots`, without which the sub-package would be left out of an install
* `geoml.plots.Interactive` is every one of those figures again, in plotly.
Same class, same method names, same arguments — `histogram`, `pairs`, `pca`,
`scene`, `training_curve`, `transformed_pairs`, `simulation_pairs`,
`prediction_scatter`, `accuracy`, `grade_tonnage` — differing only in taking a
size in pixels rather than in inches. `Explorer` is the one to save and print;
`Interactive` is the one to look at, and what it adds is what a printed figure
cannot do: zooming a panel of a scatter matrix takes its whole row and column
with it, so a cluster can be followed across every pair at once; clicking a
category in the legend takes it out of every panel; and a 3D scene can be
turned around
* Both classes now share `plots.base.Selection`, which holds the container,
the choice of variables and the colours, and settles what a figure is *about*
before anything is drawn. The numbers still come from `plots.prepare` — read
by both, so the two backends cannot drift into showing different things
* `geoml.plots.Dashboard` puts several figures on one page and **links their
selections**: drag a box or a lasso over any panel and the same locations light
up in every other one, which is the question no single figure answers — those
samples out on the tail of the histogram, where are they on the map? Double-
click to clear. Every trace `Interactive` draws from locations carries, in
`customdata`, the row of the container each point came from, and selections are
matched on those rows rather than on position, so a sample may be a bar in one
panel and a dot in another and still be the same sample
* What carries no rows is left alone, which is the honest answer rather than a
guess: a counted cell holds a number of points and not one of them, a simulated
value is a realization and not a place, and a 3D scene has neither a box to drag
over it nor per-point selection to show a brush made elsewhere
* `scene(clip=[0, 0.99])`, on both `Explorer` and `Interactive`, ends the
colour scale at a pair of quantiles rather than at the extremes. One value far
from the rest otherwise takes the whole scale with it and leaves every other
point in a single shade, which is the usual fate of an assay: a Cd with one
value twenty-five times the next draws as a uniformly dark map, and the same
map clipped at the 99th percentile shows the structure that was there all
along. Nothing is dropped and nothing is altered — the points past the end take
the end colour, and a hover still reports what was measured. `[0.01, 0.99]`
takes both tails; a menu clips each of its choices by its own quantiles
* A categorical variable with no category measured anywhere now says so.
Every series is drawn per category, so none of them meant no traces at all --
a blank figure and no complaint. The commonest way to arrive there is passing
the values to `add_categorical_variable` where the *categories* belong, and
the message says as much
* The dashboard's linked selection got out of its own way. Plotly redraws
every trace of a figure whatever changed, at about a millisecond each, so a
selection used to freeze the page for the sum over every panel at once. Now a
panel drawn from no locations — a training curve, a grade-tonnage — is skipped
rather than redrawn to no effect; a panel already clear is not cleared again;
the panels are done lightest first, so the map answers at once and the scatter
matrix follows; and the browser is handed back between them, so each appears
as it is finished instead of all of them after a frozen half-second. A
selection arriving mid-walk replaces the pending one rather than being dropped.
What is left is the cost of the heaviest single panel, which is plotly's to
pay: a seven-by-seven matrix split five ways is 161 traces and about 185ms of
redraw. Fewer categories, fewer components, or `kind="hist2d"` are the levers
on that
* `Interactive.scene(color=[...])` puts a menu on the figure instead of
drawing one variable, one entry per choice — and `color=["Elements"]` names
every component of a vector variable, which is the answer to the error that
says a vector is not one colour scale. The ground is drawn once and the values
are swapped over it rather than a trace being carried per choice: lighter, a
block model's coordinates being the bulk of it, and truer to what is being
asked, since the cloud stays where it is while the variable over it changes.
Only the locations measured in all of them are drawn, for that same reason. In
1D, where the value is the vertical axis and carries no colour, the menu moves
the points rather than repainting them
* 3D scenes are linked the other way about. There is no brushing them, but
there is turning them, and a page of several answers a question one of them
cannot: the same body of ground under two variables, seen from the same angle.
Turn one and the rest follow. The exchange ends by each panel ignoring a view
it already holds — plotly answers a camera move with a camera event of its own,
which arrives too late for a flag to catch, so the panels would otherwise hand
the view back and forth for ever
* The rows of a dashboard need not be equal. Nest an entry in `figures` and it
becomes a row of its own, however many panels are in it — a map across the full
width, a histogram and a training curve side by side beneath it, the scatter
matrix under those. Nest nothing and the figures are dealt out `columns` to a
row as before. A `(caption, figure)` pair is still a captioned figure and not a
row of two: both arrive as a tuple, and what tells them apart is that a caption
is a name and a figure is not
* `board.write_html("jura.html")` writes a page that carries plotly with it —
some four megabytes — so it opens anywhere, with no network and no geoML.
`plotlyjs="cdn"` fetches the library instead, for a file small enough to keep
in a repository. `eda.dashboard()` is the short way, naming the figures to
draw; `Dashboard([...])` takes figures instead of names, drawn with whatever
arguments you like and captioned where the caption is a better title
* A dashboard renders in a notebook, inside an iframe. A notebook is not
obliged to run a script it is handed and JupyterLab does not, which is how a
bare `<script>` works in one notebook and silently does nothing in the next; an
iframe is a document in its own right and runs its own, everywhere. Panels keep
the proportions they were drawn at rather than being stretched to their column
— a scatter matrix poured into half a page turns every panel into a slot
* `plots.style` grew `TEMPLATE`, the same look for plotly, as a plain dict:
reading the package's colours costs no import of plotly, the same way
`geoml.plotly` builds a figure without importing it
* `plots.prepare` gained the arithmetic both backends now share —
`counts_2d`, `density_grid`, `density_curve`, `normal_curve` and `cells` — and
`prediction_values` also returns which rows it drew, without which a
predicted-against-measured panel could not say who its outliers are

* A model can be drawn: `model.to_dot()` writes the whole thing as a Graphviz
diagram — the coordinates that go in, the latent network, the warpings on the
way out and the variables that come out — and `network.to_dot()` draws a latent
network on its own. Boxes are coloured by the part they play, a `Stack` or
`Concatenate` is the circle the hand-drawn diagrams use, and every arrow is
labelled with the number of variables travelling along it. The warpings are
drawn the way the model *generates* a value rather than the way it reads one,
so all the arrows run together. Render with `dot -Tpng network.dot -o
network.png`
* The new `geoml.graphviz` module imports nothing, the way `geoml.plotly`
builds a figure without importing plotly: Graphviz is needed to look at a
diagram, never to write one. Its `PALETTE` is a plain dict, so the colours are
yours to change
* A diagram says one thing the printed tree cannot: a node feeding several
branches is drawn once, with an arrow to each. `str` has to repeat it, which
reads as though there were two
* Latent nodes have names. Every node in `latent.py` takes an optional `name`,
and one built without it is numbered against the nodes it is connected to —
`BasicGP_1`, `BasicGP_2` — so the branches of a network can be told apart in
`str(network)` without naming anything. The numbering looks along the network
in both directions, not only upwards, because a new node's siblings are not
among its ancestors; two subnetworks built separately and joined only later are
the one case it cannot see through, and `get_node` says so if it is ever asked
for a name they share
* `network.get_node(name)` returns a node by name, among the top node and
everything feeding it. Names given by the user are recorded as constructor
arguments, so they are saved with the model; automatic ones are regenerated by
the replay, in the same order, and come back identical
* The node incompatibility errors name the nodes involved. A size mismatch
used to read `All parents must have the same size. Found [1, 2]`, and now reads
`Add_1: all parents must have the same size. Found BasicGP_1 (size 1),
BasicGP_2 (size 2)`; a GP built on a parent that cannot propagate inducing
points names both

## version 0.5.3
* Point-wise information can be attached to any container with
`obj.add_metadata(name, values)` and read back with `obj.get_metadata(name)` —
an air/solid code, a cross-validation fold, a sample weight: things known per
location that the models never see. It follows the object through subsetting,
`as_data_frame()` and now `to_zarr()`/`open()`, which used to drop it silently
* **Breaking:** `obj.metadata` is a dict of columns, not a `DataFrame`. A
column is the same `_Attribute` a variable is built from, so it spills to Zarr
when large and carries the same helpers — `grid.metadata["air"].as_cube()` and
`fill_pyvista_blocks()` work on it. Text is kept as integer codes plus labels:
as an object column a rock code on a two-million-block model costs 105 MB
against 1.9 MB as codes, and object arrays can never spill to disk. Code that
read `obj.metadata["fold"]` expecting a column of values wants
`obj.get_metadata("fold")`
* `_Attribute` is a module-level class instead of one nested inside
`_Variable`, and it is where the text-as-codes handling lives
(`_Attribute.encoded`). Two fixes came with it: an attribute built from values
now chooses its backend by size like an empty one does, instead of being pinned
in RAM whatever its length; and reopening one from disk no longer allocates a
throwaway array first
* **Breaking:** the categorical variables hold codes too.
`RockTypeVariable.predicted` / `measurements_a` / `measurements_b`,
`BinaryVariable.measurements` / `predicted` and their subclasses were arrays of
Python strings — a predicted rock type on a 27-million-block model is about
1.4 GB that cannot be spilled to disk, against 27 MB as `int8`. Read them with
`attribute.to_numpy()`, which gives the labels back; `attribute.values` is now
the codes. Anything that writes a prediction writes the position of the winning
label. -1 means "not measured here" and reads back as the empty string, as it
did before. A measurement column carries **its own** categories, not the
variable's, so an `AnomalyVariable` measured as "hit"/"none" still reports
"none" even though the variable calls that class `_dummy`. Containers saved by
an earlier version are re-encoded when they are opened
* Slicing a categorical variable now drops the categories that are no longer
present, all the way through: the components go with them, and the prediction
is re-coded to the shorter list, `update` writing the winning label's position.
Before, only `labels` was reset, leaving a variable that claimed one category
and carried three. An `OrderedRockType`'s implicit values are recomputed rather
than carried over — they are positions in the label sequence, so dropping a
category shifts every one after it, and a slice would otherwise be trained
against a scale that no longer existed. A slice with nothing measured in it
keeps the categories it was declared with
* Categories that are not text are left as they are. Deriving them stringified
them, so a variable measured as `1`/`2` compared its predictions against `'1'`
and `'2'` and every metric came out at chance. They are stringified only on the
way to disk, as the variable's own labels always have been — which also fixes a
crash, older than this release, that made a variable with integer categories
impossible to save at all
* Interval columns can be renamed with `holes.rename(table, {"Pb_pct_ICP":
"Pb"})`, or `table.rename(...)` on the table itself. The roles travel with the
columns: renaming the data frame directly left them behind, keyed by the old
name, so the column quietly stopped being a grade and printing or compositing
the table raised a `KeyError`. Renaming the frame before it is added still
works as it always did — this is for a database that was built for you, by
`datasets.macpass()` or by compositing, where the frames were never in reach.
Column names become variable names in `as_point_data()`, so it is also the last
chance to tidy them before they reach a model

## version 0.5.2
* Simulated predictions are reproducible. The likelihood noise was drawn from
TensorFlow's global generator with a hard-coded seed, so `options.seed` never
reached it and two predictions of the same model gave different simulations.
The draws are now stateless and seeded from `options.seed`, in a stream of
their own so they do not follow the latent draws
* The noise on discretized blocks is drawn per sub-block position: the `k`
sub-blocks of a block get independent values, and every block gets the same `k`
values. Averaging them is what integrates the noise over the block, and the
shared pattern means the noise shifts the map rather than roughening it, which
is what makes simulations of a block model look like the field and not like
static. Since the draw is indexed by sub-block rather than by row, simulations
no longer depend on how the prediction was batched
* `prediction_batch_size` now counts the rows the model actually evaluates. A
block with a [5, 5, 5] discretization fans out into 125 of them, so the default
20000 handed the model 2.5 million rows at once and had to be shrunk by hand to
avoid running out of memory; the setting now means the same thing for every
container
* The block fan-out is built by broadcasting instead of a Python loop over
blocks — 8x faster on a 40000-block model — and the noise is broadcast over the
blocks instead of being tiled into a second copy of the largest tensor in the
prediction
* The two `white_noise` implementations, which disagreed on what "coherent"
meant, are now one on the base likelihood

* **Breaking:** `as_pyvista()` and the `fill_pyvista_*` methods no longer export
simulations by default. Each one is a full-length array in the exported object,
so a block model with twenty of them carried twenty copies of itself into
memory whether or not they were going to be looked at. Pass `simulations=True`
for all of them, an `int` for the first n, or a sequence of indices —
`grid.as_pyvista(simulations=[0, 5])`
* The simulations that are exported are now read in a single pass. The store is
chunked along rows only, so a chunk holds every simulation of a row band and
reading one column at a time decompressed the whole array once per simulation.
On a 4 million point grid with 20 simulations the read went from 2.9 s to 1.5 s,
and the default export — which no longer touches the simulations at all — from
2.9 s to 0.02 s
* `as_data_frame()` read its simulations the same way, one column at a time,
and now takes the same single pass. Its `simulations` argument also accepts an
`int` or a sequence of indices, not only a flag, so a data frame can carry a
couple of realizations instead of all or nothing. Its defaults are unchanged
* The check that skips an empty attribute is vectorized. It used the built-in
`all()` over a NumPy array, walking it one element at a time in Python, for
every attribute of every export

## version 0.5.1
* A model can now be reproduced from a single call. Parameters are initialized
at random when an object is built, which happens before any model exists to
carry `options.seed`, so the draws now come from one generator held in the new
`random` module: call `geoml.set_seed(1234)` before building and the same
objects start from the same values, in this run or the next. Left alone, the
generator is seeded by the operating system, as before. Scripts that pinned a
build with `np.random.seed()` must call `geoml.set_seed()` instead —
initialization no longer reads NumPy's or TensorFlow's global generators
* Objects now print themselves. `repr` identifies one on a single line, as the
call that would build it — `Covariance(Gaussian(), Isotropic(range=111.0))` —
reading the arguments each object already records for persistence, and showing
the *current* value of a parameter rather than the one it was built with, since
training moves it. Objects holding many or large parameters, such as a GP node,
fall back to naming their configuration. Data containers, variables and models
keep summarizing themselves as they did. This also fixes `repr` raising
`NotImplementedError` for every kernel and every likelihood, which took
debuggers and failing assertions down with it
* `str` lays out an object's state: its parameter values, and everything
registered inside it, indented. A latent network therefore reads as its
composition, from the output node down through its parents to the input, with
each node's size. Large parameter arrays are summarized by shape and range
instead of being printed in full. The four near-identical `pretty_print`
implementations in `kernels` and `transform` are now one, inherited from
`Parametric`

* geoML no longer disturbs the generators the caller is using: building a
`RandomProjections` transform used to reset NumPy's global seed outright, and
training did the same before shuffling its batches. Both now draw from a
generator of their own, and remain reproducible from the seed they are given

## version 0.5.0
* `DrillholeData` was rewritten from scratch, in a new `drillhole` module.
It is now built from a collar and an optional survey table, and interval files
(assay, lithology, and any others) are added to it separately, each as an
`IntervalTable`. Hole paths are desurveyed by minimum curvature, so deviated
holes are positioned properly; holes with no survey are taken as straight,
along the collar attitude. Databases that record a downward hole with a
negative dip are supported through `dip_positive_down=False`
* Interval tables declare the role of each column — grade, categorical,
density, flag or ignore — which decides how it is composited. Roles can be
redeclared afterwards
* Interval data is checked on entry: overlapping intervals, non-positive
lengths, missing depths and holes with no collar are reported as errors, and
gaps and intervals running past the end of the hole as warnings, with the
policy chosen by `on_error`
* New compositing methods, all of which return a new object with every table on
one shared support: `composite()` (fixed length, honouring the boundaries of a
named categorical domain), `composite_fixed()` and `composite_to()` (onto the
intervals of another table). Numeric columns are averaged weighted by length,
or by length times density where a density column is declared, skipping
missing values; categories are decided by the greatest length in the composite
* `as_point_data()` converts the interval data to points, merging the tables
row for row. Compositional and vector variables are assembled here, with the
processing raw assays need, in order: each column is divided by its unit
("%", "ppm", "g/t", …, or a number), which has to be declared and is what
lets parts measured in different units be added together; a sample missing any
one part is marked missing entirely, since the parts only carry information
relative to each other; non-positive parts are replaced by half the smallest
positive value of their own column, the usual substitution for values below
detection, which a log-ratio transform cannot otherwise take; and a rest part
is added, holding whatever is left of the whole, so a barren sample is nearly
all rest and a rich one much less so. Where the parts leave no room, the rest
is held at the smallest part found and the sample is scaled down to fit, with
a warning — which usually means a unit was declared wrongly
* The database can be cut down before it is used: `subset_region()` keeps the
holes whose collar falls within a bounding box, `subset_holes()` those named
explicitly, and `filter_intervals()` the intervals holding a given value — or,
with `whole_holes`, every hole in which that value was logged
* Logging codes can be lumped into modelling domains with
`group_categories()`, which takes a mapping from each new category to the codes
it takes in, and reports the codes left out. Since there is no GUI to click
them together, `category_legend()` lists the distinct codes with the number of
intervals and the metreage each accounts for, sorted by length and with a
`group` column ready to be edited: write it to a spreadsheet, fill the column
in, and read it straight back into `group_categories()`. Intervals carrying no
value get a row of their own, so the legend accounts for the whole table
* `fill_unlogged()` gives a category to the ground a log does not account for,
which is what a database holds when only the intervals of interest were
logged. Intervals that exist but carry no value take the new label, and the
depths no interval covers become intervals of their own, reaching from the
collar to the bottom of the hole — so a hole nobody logged comes out labelled
end to end, and the log covers every hole continuously afterwards
* `as_classification_input()` still produces points of zero effective support,
including the contacts between rock types carrying both neighbouring classes
* New `datasets.macpass()` loader for the Macmillan Pass drillhole database,
published by Fireweed Metals Corp. The data is not distributed with geoML; see
the docstring for where to get it and the terms that apply
* Memory-efficient data storage: variable arrays are now backed by NumPy in RAM
or chunked Zarr on disk, chosen automatically by size, so large projects no
longer need to hold every array in memory
* Simulations are stored as a single `(n_data, n_sim)` array (a dedicated
dimension) rather than a list of separate arrays
* New `variable.simulation(i)` and `variable.n_sim`: a single simulation can
again be handled as an attribute, with the usual `as_image()`, `as_cube()`,
`smooth()` and contouring helpers
* Grid and block coordinates are now generated lazily: a regular grid no longer
holds its full `(n_data, n_dim)` coordinate array in memory, but produces the
requested rows on demand, so very large grids can be built and predicted into
without the coordinates dominating RAM
* Point coordinates and `GaussianData`'s input variance are stored the same way
as variable arrays (NumPy in RAM or Zarr on disk, by size), so a large point
cloud no longer has to stay in memory
* Fixed a bug where the input variance was rebuilt for the whole object on every
prediction batch, making prediction quadratic in the number of points; it is now
generated for the requested batch only
* Fixed `GaussianData`: predicting into one no longer fails, subsetting keeps the
object's class and variance instead of degrading to `PointData`, and the variance
is included in `as_data_frame()`
* Containers can be persisted to and reloaded from a single Zarr store with
`to_zarr()` / `open()`, covering all point-based containers and variable types
* Trained models can be saved and loaded whole with `model.save()` and
`Model.open()`, in a new `persistence` module: structure, trained parameters
and training data are all kept, so a reloaded model predicts exactly as it did
and can be trained further. It remembers its variables and their types, and so
creates them on any new data object it predicts on. `Model.open(path, data=...)`
rebuilds the same model around a different data set
* Quantiles and probabilities are computed lazily in a single pass over the
simulations, without materializing them
* `reset_probabilities()` is now the inverse of `reset_quantiles()`: it takes
cutoff values and returns cumulative probabilities in (0, 1)
* Fixed prediction of categorical variables, which failed with a `TypeError`
after the prediction posterior started being cached between batches
* Fixed variables being degraded to a more basic type when built from an
existing one, either by predicting on a new data object or through
`from_data_frame()`: a categorical variable became a rock type variable, an
ordered rock type became an unordered one, and an anomaly became a plain binary
variable. The same applied to data objects, where `Blocks2D.from_bounding_box()`
and its relatives returned a plain grid
* `CategoricalVariable` can now be built from a data frame with a single
measurements column, instead of the two contact columns a rock type needs
* Fixed the `n_sim` argument being ignored when predicting with the standard
`GP` model, which failed for any value other than 50
* Fixed construction of `OrderedRockType` and rock-type/binary variables from
string measurements
* Fixed a bug where a parameter shared between components (e.g. a transform
reused by several kernels) was counted more than once by `get_parameter_values`
and `save_state`
* New dependencies: `zarr`, `xarray`, `dask`

## version 0.4.1
* Support for predictions over blocks with discretization
* Delta method for change of support

## version 0.4.0
* Implemented multivariate data and likelihoods
* Warpings are now multivariate
* New warpings: `PCA`, `RobustPCA`, `Rotation`, and `ContinuousNormalizingFlow`
* Streamlined code to deal with compositional data
* Streamlined quantiles and percentiles computation, now directly from simulations
* Added option to smooth data when converting it to data to image or cube format
* Added convenience methods to compute model metrics


## version 0.3.5
* Support for rotated 3D grids
* Simulations now have the option to include the likelihood variance
* Improvements and bug fixes
* Product of Experts is now the default for GP latent variables

## version 0.3.4
* Improved pyvista integration
* Multiscale modeling did not work
* Compatibility with TensorFlow v2.11

## version 0.3.3
* Improvements and optimizations
* Heteroscedastic likelihoods
* A likelihood for conformable layers
* A likelihood for limited time-aware implicit modeling
* Ability to use a random field to move data points
* Ellipsoidal trends
* Limited support for faults
* (Experimental) multiscale modeling (refinement with increasingly denser inducing point sets)

## version 0.3.2
* Tensorized latent variable removed for now (back to the old chalkboard).
* Full deep GP implementation.
* Added Jura dataset.
* Added likelihoods for compositional variables.
* `kernels` module now has a specific object for covariance functions.
* Implemented the ability to convert data objects to `pyvista`.
* Implemented `Surface3D` data object.

## version 0.3.1
* New autoregressive latent variable.
* New tensorized latent variable.
* New Product of Experts latent variable.
* Spline warping is now centered at zero.
* Ensembles now handled in a specific model.

## version 0.3.0
* Introduced variational models.
* All training is now gradient-based.
* Changed the data manipulation system, defining variable types (continuous,
categorical, etc.) and Attributes (mean, variance, entropy, etc.) that can be
accessed and plotted directly.
* Plotly interface is now in a specific module.
* New `Section3D` data object.

## version 0.2.0
* Code updated to TensorFlow 2.0.
* Cubic convolution implemented in TensorFlow.
* Simplified kernel API.
* New auxiliary functions in `tftools` module.
* Cubic splines implemented in TensorFlow.
* Added Cerro do Andrade dataset.
* Notebooks moved to Google Colaboratory.
* Training is now based on Particle Swarm Optimization.
* Included a `GPOptions` class to control verbosity, batch size and
other model options.
* The covariance matrix for directional data is now computed with a 
finite difference method.
* Drillholes can now be segmented in roughly fixed intervals.

## version 0.1.3
* Introduced parallel sparse cubic convolution interpolation.
* Introduced the sparse GP model.
* Introduced the SPICE model for scalability.
* New auxiliary functions in `tftools` module, such as 
conjugate gradient solver and Lanczos decomposition.
* Limited support for modeling non-spatial data.

## version 0.1.2
* Introduced the `jitter` parameter in the models, for improved 
numerical stability.
* Corrected a bug where setting the value of a parameter beyond its limits
would not extend the limits.
* Implemented the `prod_n()` and `safe_logdet()` functions in the 
`tftools` module.
* Implemented the `aggregate_categorical()` method for point data methods.
* Optimized the batch prediction process.
* Implemented saving and loading for model parameters.
* Implemented the neural network transform.
* Now multiple transforms can be chained and/or shared between multiple
kernels.
* The classification algorithms now output an uncertainty metric, weighting
the entropy and predictive variance.

## version 0.1.1
* Added support for products of kernels.
* Fixed bug in `geoml.tftools.prod_n()` function.
* `geoml.kernels.Gaussian.covariance_matrix(x, y)` now
adds some jitter to the covariance matrix when x == y, to
avoid Cholesky decomposition problems.
* `geoml.warping.Softplus` now corrects non-positive values
in its input.
* Corrected a bug where the wrong nugget would be added
while making a prediction.
* Added sunspot number data and example notebook.
* Implemented cubic convolution fully in TensorFlow.
* Set the default `n_knots=5` in `geoml.warping.CubicSpline.__init__()`.