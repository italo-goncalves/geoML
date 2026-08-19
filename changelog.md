## version 0.6.8
* **One declustering column, and everything downstream shares it.**
`container.decluster(on=, cell=)` computes cell-declustering weights and
keeps them as the metadata column `"declustering"` — a first-class
per-location column with subsetting, Zarr and export for free, which is
what metadata columns are. The census that forced it: the two existing
consumers each swept their own cell size on their own values, so the
same container could get different weights in different figures; with
the initializers joining them, the inconsistency would have multiplied.
  - **One call turns declustering on everywhere.** The `variogram`
  figure reads the stored column when present (its internal
  `variogram_score` then rides the very same weights);
  `metrics.variogram_score` — arrays in, arrays out, no container —
  gains an explicit `weights=` argument instead. Precedence everywhere:
  explicit argument > stored column > computed on the fly. `on=` names
  the variable driving the automatic cell sweep, defaulting to the only
  continuous variable and refusing to guess among several.
  - **The warping initializers start declustered.** The model fetches
  the stored column at construction and threads it down
  `likelihood.initialize` → `warping.initialize(x, weights=None)`:
  `ZScore` takes weighted moments (and weighted winsor fences when
  robust), `Spline` places its knots on the weighted empirical CDF —
  with the tail floor counted in **Kish's effective sample size** rather
  than in rows, since a region carried by many light rows is still the
  sample it weighs, and equal weights reproduce `count/(n+1)` exactly.
  Warpings with nothing to weight accept and ignore; `None` everywhere
  is the start it always was, bit for bit. The declustered start on a
  duplicated-region sample lands where the plain start on the clean
  sample does, which is the test.
  - A subset carries its parent's weights by value — the `"fold"`
  column's convention — so they are a snapshot, recomputed after
  subsetting when it matters. Said in the docstring rather than solved
  by magic.
  - **`Rotation`'s ICA is seeded from the package RNG now.** Found by a
  test threshold that moved when the batch around it changed: unseeded,
  FastICA draws from numpy's *global* state, which made the rotation the
  one data-dependent start `geoml.set_seed` did not reproduce — a gap in
  the 0.6.2 one-knob unification, closed. The affected test seeds itself
  and means one thing now.
  - Also here, the type check over the touched files: two diagnostics
  that had shipped silently with the reliability commit (bare `labels`
  access on the `_Variable` base) and the `weights` argument rebound in
  `variogram_score` against the house rule — repaired, the list clean.
  - The variogram-based kernel initialization that was planned alongside
  this is **dropped, on the user's call**: too much work for machinery
  this library exists to replace. The experimental variogram stays what
  the manual sells it as — a check, never a fitting surface.
* **A reliability diagram, on both plotting backends.** `reliability` on
`Explorer` and `Interactive`, the counting in `prepare.reliability` — the
categorical half of what `accuracy` asks of a continuous variable, and
the first of the categorical follow-ups the confusion matrix opened.
  - One curve per category: the locations binned by the probability the
  model assigned to it (**equal-count** bins — for a rare category the
  claims pile up near zero, and equal width would leave most bins
  holding nothing; pass edges for equal width), each bin's mean claim
  against the share of its locations actually measured as that
  category. The legend carries each curve's expected calibration error,
  the count-weighted mean distance from the diagonal. A constant claim
  comes back as one point rather than an error, so one degenerate
  category cannot blank the figure.
  - The same locations count as in `confusion_matrix` — the shared truth
  side now lives in `_measured_categories` — and **the predicted label
  is what says a location was predicted at all**: a category's
  `probability` initializes to zero, not to absence, so an unpredicted
  container would otherwise read as a model claiming zero everywhere.
  Caught by the test that asks an unpredicted container to say so.
  - The plotly twin keeps the bin counts in `text`, deliberately not
  `customdata`: that slot is the dashboard's contract for container
  rows, and a bin is not a location.
  - Same honesty caveat as the whole model-checking family: only honest
  on data the model has not seen, the out-of-fold container being the
  honest input. Five tests on the counting, one per backend on the
  figures.
* **geoh5 interchange, phase one: surfaces and points.** The first cut of
the geoh5 plan (the octree and block-model halves are the later phases):
`Surface3D`/`Solid3D`/`Mesh3D` and `PointData` now read and write Mira
Geoscience's workspace format, which is what **Geoscience ANALYST — a
free viewer — opens**, so `to_geoh5` turns that viewer into a 3D screen
for geoML's meshes and point predictions.
  - The dependency is an optional extra (`pip install geoml[geoh5]`,
  `geoh5py >= 0.13` — floored at the verified version, the `value_map`
  representation having moved between geoh5py releases), imported lazily
  inside `data/geoh5.py` and never before a file is asked for. CI
  installs the extra so the tests run rather than skip; without it they
  skip whole, and the package works untouched.
  - A workspace accumulates: writing into an existing file adds the
  object beside what is there, `name=` says which object to read when a
  file holds several, and `geoml.data.geoh5.contents` lists what there
  is to name. Reading a path that does not exist raises instead of
  quietly creating an empty workspace, which is what `geoh5py` itself
  would do with a typo.
  - Surfaces travel as geometry and classify on the way back through
  `mesh3d`, exactly the `from_dxf` convention — including the
  degenerate-face drop, a foreign file not being ours to trust. Points
  carry their columns off the same `_export_leaves` enumeration every
  export drives, named in the `pretty` style, with coded attributes as
  geoh5's own *referenced* data (codes shifted by one: geoh5 reserves 0
  for "Unknown" where geoML's missing is -1) and the name-to-path table
  in the object's metadata under `geoml_paths`. Imported columns become
  measured variables — float as continuous, referenced as categorical,
  "Unknown" as not measured; what the file has no room to say (which
  column was a prediction, whose tree it belonged to) is not guessed at.
  - One geoh5py fact recorded where it bit: a `Surface` *is a* `Points`
  by inheritance, so the object lookup matches on exact type — reading
  "the points" out of a workspace must not hand back a surface.
  - Nine tests, round-tripping both mesh kinds, the accumulating
  workspace, the name errors, missing values in both column kinds, and
  the paths table; `pytest.importorskip` guards the lot.
* **geoh5 interchange, phase two: block models.** `BlockSet3D.to_geoh5`
and `from_geoh5`, against geoh5's Octree object. The subclass question
the phase opened with — a `BlockSet3D` subclass in 1:1 correspondence
with geoh5 — dissolved under measurement: **the correspondence already
exists at `discretization=(2, 2, 2)`, which is the constructor's
default**, because a block's origin and size in base cells are an octree
cell's `I J K NCells` as they stand. The two facts that could have
justified the class both fell to geoh5py's own behaviour, probed before
any code was written: axis counts must be powers of two, but **partial
coverage is legal**, so export pads the counts and writes no padding
cells — nothing contaminates a round trip, no phantom blocks, and the
box comes back exact; and cell order is preserved end to end, so values
ride with no reordering. A model built any other way is refused with the
message naming the fix, never resampled.
  - **Rotation maps exactly, both directions**: geoh5 carries one
  counter-clockwise rotation about the vertical axis about its origin
  (verified numerically against geoh5py's own centroids), which is the
  negated mining-convention azimuth once the row-vector application is
  accounted for. A `RotatedBlockSet3D` with dip or rake is refused —
  there is nowhere for them to go — and a rotated octree comes back a
  `RotatedBlockSet3D` whose world centres match to the last figure. The
  pivot difference (geoML turns about its first block's centre, geoh5
  about its corner) cancels by exporting the *world* corner as the
  origin.
  - **`max_levels` rides in the object metadata**: it is refinement
  capacity, which the cells alone cannot say once every coarse block has
  been split; a foreign file without it gets the largest cell as the
  root.
  - **A foreign file is validated, then completed.** Cells must be
  aligned power-of-two cubes free of overlaps — `dyadic_overlaps` in
  `math/geometry.py`, one ancestor lookup per cell, since aligned dyadic
  cubes are disjoint or nested and never partial — and whatever they do
  not cover inside their own box is filled with unvalued blocks by
  `dyadic_complement` (top-down, largest aligned cells first), a
  metadata column named `imported` marking the difference: the
  always-full design, not bent for the vendor octrees that carry only
  the domain of interest. Both helpers are the lattice-validation core
  the CSV importer item was scoped to need, built once here. An octree
  written with its vertical axis running downward — origin at the top,
  negative w, common in the wild — is normalized on the way in; a
  flipped *horizontal* axis under a rotation would compose a reflection
  no rotated block model can hold, and is refused.
  - Nine more tests: refined and rotated round trips whole (geometry,
  levels, values, box), the two export refusals, the partial fill, the
  upside-down file, the overlap and reflection refusals.
  - Also here, found by running the type check over the touched files:
  two diagnostics latent in `blocks.py` since the 0.6.7 contour rewrite
  (an unannotated `_contour_column` contract and a `cap` narrowing the
  checker could not see) — repaired, the file list is clean again.
* **geoh5 interchange: the `Workspace` object, and replace-on-rewrite.**
From the first real ANALYST session: the files loaded — which also
confirms the rotation sense and the partial-coverage padding against the
actual viewer — but one file per object opens as one *project* per
object, and a model's pieces belong together. So
`geoml.data.geoh5.Workspace(path)` opens (or creates) one file and every
`to_geoh5`/`from_geoh5`/`contents` now takes it in place of a path — a
context manager, the file opened once for a whole model's exports
instead of per call, readable back through the same open handle.
  - The workspace's name, as ANALYST shows it, **is the file's name**:
  geoh5py accepts a display name at creation and does not persist it —
  measured, it reads back as "GEOSCIENCE" — so no name argument is
  offered that would silently vanish on reload. Per-object names were
  already there (`name=` on every `to_geoh5`).
  - **Rewriting a name replaces the object** (`replace=True`, the new
  default): re-running an export script is what the workflow *is*, and
  without replacement every run doubled the project and made the name
  unreadable back. `replace=False` keeps both, said out loud in the
  ambiguity error. One geoh5py fact carried the design: entity removal
  is deferred to `close`, so the live listings keep a ghost of anything
  replaced — the wrapper remembers the removed uids and filters them,
  which is what keeps a read-after-replace honest inside one session.
  - Four more tests: a whole model through one open workspace and read
  back without closing, replacement across sessions with the new values
  winning, `replace=False` keeping both, and the in-session ghost.
* **geoh5 interchange, the remaining pieces: regular block models,
drillholes, and the workspace that reads like a dict.** The last three
items of the geoh5 plan, plus the import ergonomics the user asked for.
  - **`Workspace` is a mapping now.** Printing one lists what it holds —
  name against geoh5 kind, drillhole groups included with their hole
  counts — and `project["ore body"]` returns the geoML container the
  object's kind calls for: a Surface classifies through `mesh3d`, Points
  become `PointData`, an Octree a `BlockSet3D`, a BlockModel a
  `Blocks3D`, a drillhole — or a whole drillhole group named as one — a
  `DrillholeData`. Converted on demand, one name at a time, deliberately
  not an eager dict: a workspace can hold more than fits in memory.
  `in`, `len` and iteration over names come with it.
  - **`Blocks3D.to_geoh5`/`from_geoh5`, against geoh5's BlockModel** —
  the regular-grid sibling of the Octree. Cell data is stored u-fastest
  where geoML's grids run x slowest, so every column transposes on the
  way through, the ordering pinned against geoh5py's own centroids; a
  rotated BlockModel comes back a `RotatedBlockSet3D` with
  `max_levels=0` (there is no rotated regular-grid class, and a uniform
  rotated model is exactly that — everything works, nothing refines); a
  true tartan grid is refused with the uneven axis named, geoML having
  no container for uneven blocks. The first-edge offset travels
  separately from the origin, because it is local and must turn with
  the rotation.
  - **`DrillholeData.from_geoh5`.** Every hole a collar row, surveys
  across with **geoh5's dip convention converted** (there a vertical
  downward hole reads -90, so the database is built with
  `dip_positive_down=False`), and each named property group holding
  FROM/TO becomes one interval table of that name gathered across the
  holes — referenced columns take the categorical role, depth-associated
  readings (no interval to cover) stay out, and `name` narrows to one
  group or one hole. What comes back composites, validates and converts
  like any other drillhole database.
  - A reference page for the module joins the docs, and the `data` page
  stops advertising `RotatedBlocks3D` — a class that does not exist,
  listed since the reference was written.
* **geoh5 folders.** Every `to_geoh5` takes `folder="Surfaces/Ore"` — a
path whose segments are geoh5 *container groups*, the folders ANALYST
shows in its project tree, each created when it does not exist and
reused when it does, so repeated exports share one tree rather than
planting a second `Surfaces`. What follows from a tree follows
everywhere: `contents()`, the workspace repr and iteration come
folder-qualified (`Surfaces/Ore/hematite`; a root object keeps its bare
name); `project[...]` and every `from_geoh5` accept the qualified name,
with the bare one staying valid while it is unique and the ambiguity
error answering with the qualified alternatives; and `replace` is
**scoped to the folder**, so "hematite" under Ore and "hematite" under
Waste can each be rewritten alone. Foreign files with folders were
already readable — geoh5py lists objects flat — so this changes what
geoML writes and how names render, not what it can open. Three tests:
the tree (nesting, reuse, qualified listing, the ambiguity answer),
the scoped replace, and the qualified name reaching the classmethods.
## version 0.6.7
* **`BlockSet3D.get_contour(close=...)` returns a body again.** Reported
against a real model: `close="above"` and `close="below"` both promise a
closed shell, and a `Mesh3D` was coming back instead. Two independent
defects, both in the closing, and the second is the one the report named.
  - **The ghost shell had no diagonals.** `_ghost_shell` mirrored each
  boundary block across the faces it touched and skipped the edge and
  corner directions, on the reasoning that nothing crosses a box edge from
  inside. Nothing does — but the closing cap itself runs *along* the
  boundary, and where two face ghosts meet without the diagonal between
  them the cap reaches the shell's own edge and stops. That is a slit, and
  it is the common case rather than a corner one: `close="below"` keeps the
  region under the value, which on most models touches every face, so its
  cap ran the length of all twelve box edges and tore along each. Measured
  on a ball model, `close="below"` returned an open `Surface3D` with 178 to
  630 boundary edges in every geometry that reached the boundary — the
  open edges sitting at the ghost centres, one cell outside the box, which
  is what pointed at the missing diagonals. All 26 directions are mirrored
  now, and the same sweep returns `Solid3D` throughout.
  - **VTK's contour emits faces that bound nothing.** Where the level set
  passes exactly through a cell corner it pinches to a point, and the
  marching-cubes case covering that writes out its zero-area triangles
  anyway. They carry no geometry, but each makes an edge appear twice the
  same way round, so `reversed_edges` counts them and `mesh3d` reads the
  surface as inconsistently wound. One geometry produced 28 slivers, 4
  twinned faces and 22 non-manifold edges: closed, consistent everywhere it
  had area, and classed `Mesh3D`. `math.geometry.drop_degenerate_faces`
  welds and drops both kinds, and `get_contour` runs it before asking what
  class the geometry calls for. How many copies of a twin go is decided by its
  edges (see the sweep below), which is what keeps a doubled wall from
  being read as a flap.
  - Neither fix moves a surface that was already closing: the volumes on
  every previously-working case are unchanged to five figures.
  - Six tests in `test_blockset.py` (both sides at three supersampling
  levels, the three geometries that reach a face, an edge and a corner, and
  the pinch that produced the `Mesh3D`) and one in `test_mesh3d.py` for the
  dropping itself, including what it deliberately costs.
  - **Not fixed, and worth knowing:** the cap does not sit on the box face.
  It lands where the boundary block's value and the fill put it — the fill
  being one past the data's range — so a closed body can extend up to half
  a block beyond the model and report a volume larger than the model's own
  box: measured +2.8% on a `Grid3D` and up to +6% on blocks. This is not
  new and not specific to blocks; the grid path pads its cube with the same
  constant. Landing the cap on the face would mean giving each ghost the
  reflection of its partner's value about the contour level rather than one
  shared constant.
* **`simplify` on a closed body raised instead of simplifying.** The same
family as the contour fix above, one function further on, and reported from
the same workflow: `get_contour(..., simplify=0.5)` came back
`InconsistentMeshError: this mesh closes, but its triangles disagree about
which way is out`.
  - Decimation collapses edges, and collapsing one through a thin feature
  folds it into a **zero-thickness flap** — the same triangle twice,
  bounding nothing. Every edge of it then runs twice the same way round, so
  the constructor reads the body as inconsistently wound and refuses.
  - `_rebuilt_as` already anticipated inconsistent winding and reoriented
  the triangles as `heal()` would. **That cannot work here, and the failure
  is diagnostic**: a flap is non-manifold, VTK's orientation pass cannot
  walk across it, and it returns having spread the disagreement rather than
  settled it — measured 2 reversed edges in and 3 out on a decimated shell,
  and 16 in and 24 out in the report. So the flaps are dropped first, by the
  same `drop_degenerate_faces`, and only genuine winding is reoriented.
  Second repair now runs on the cleaned arrays rather than the mesh as it
  arrived, so the orientation pass walks a manifold surface.
  - Measured over 80 combinations — four rough fields, both sides, two
  supersampling levels, five error budgets — two raised before and none
  does now, with volumes stable across budgets to under a percent.
  - `test_mesh_operations.py` plants a flap on a box and rebuilds through
  `_rebuilt_as` directly, which is deterministic where reproducing a
  decimator artifact is not.
* **`Solid3D`'s booleans no longer call VTK, because VTK was crashing the
session.** Reported as a hard crash from `intersection` — not an exception,
a segfault: `vtkIntersectionPolyDataFilter` dies on contour-derived bodies,
which are the shapes this package exists to produce.
  - **What the design assumed, and why it could not hold.** `_combine` ran
  the exact filter first and fell back to `_implicit_combine` on an error,
  an empty answer or invalid output. A SIGSEGV is none of those: the
  process is gone, `VtkErrorCatcher` never gets a turn, and a notebook
  loses its state. So the meshes most in need of the fallback were exactly
  the ones that could not reach it.
  - **It is not about size, quality or coordinates.** Measured: a box
  against a box (12 triangles) and two spheres (1680 each) go through
  exactly; a block-model shell of **836** triangles segfaults, one of 3128
  hangs past 300 s and then segfaults. The same shells cleaned, stripped of
  slivers (they had none — smallest triangle area 1.5e-3) and decimated to
  1460 triangles crash identically, at the origin and at 1e6. No property
  measured here separates the survivors from the casualties, so no screen
  would be safe.
  - **The exact engine is therefore gone**, on the user's call, rather than
  guarded behind a subprocess. Two bodies now go straight to `_resolved`:
  the vertices say whether the surfaces cross, the disjoint and nested
  cases are answered exactly and cheaply as before, and anything that
  crosses goes to `_implicit_combine`. The cost is exactness on the cases
  VTK could do — a box against a box now answers to the grid step like
  everything else, and the warning says so. `_without_crossing` is
  `_resolved` renamed, since it is now the whole of the decision rather
  than the branch VTK left over.
  - **The fallback needed the same repair as the contour**: it ends in a
  marching-cubes contour of its distance field, and a sphere differenced
  from a sphere was coming back a `Mesh3D` with a NaN volume for want of
  dropping the faces that bound nothing. `_Attribute.get_contour` now
  drops them too, so both contour producers are clean and the implicit
  engine answers with a body.
  - `test_mesh_operations.py` combines two block-model shells three ways.
  A regression there does not fail the test — it takes the runner down,
  which is the loudest signal available and the reason the engine was
  dropped rather than guarded.
* **`heal()` did not heal the thing everything sends it.** Every mesh error
message in the module ends "heal() puts this right", and for the commonest
cause it did not: a zero-thickness flap or a zero-area sliver makes an edge
run twice the same way round, which reads as a winding failure and is the
one such failure reorienting cannot mend — the surface is non-manifold
there, so VTK has nowhere to walk. Traced step by step on a box carrying one
flap, `clean`, `compute_normals(consistent_normals=True,
auto_orient_normals=True)` and `triangulate` left all three of its reversed
edges exactly as they were, and the mesh came back a `Mesh3D` still, with
pyvista warning that its own output was invalid. `heal` now drops the faces
that bound nothing before any of that, and the same box comes back a
`Solid3D` of the right volume. An open tube still closes at `hole_size`, as
before.
  - The refusal in `_closed_body` was recommending the repair that does not
  work — "rebuild from as_pyvista().compute_normals(...)" — and now points
  at `heal()` like every other message.
* **A sweep of the mesh code: four repairs.**
  - **One welding where there were two.** Every mesh asks `open_edges` and
  `reversed_edges` of the same triangulation as it is built, and each
  welded for itself — a rounding and a `unique` over the vertices, which is
  the expensive half. `geometry.edge_defects` answers both from one
  welding: measured on 27 656 triangles, 15 ms became 11 ms and building
  the mesh 18 ms became 13 ms. It compounds, `simplify` building one mesh
  and a boolean three.
  - **Triangles that index nothing are refused where they are given.** The
  constructor checked that its arrays had three columns and nothing else,
  so an out-of-range index surfaced as `IndexError: index 99 is out of
  bounds` from inside `area` — a confusing place to meet it. It now says
  which range the triangles reach and how many points there are to reach.
  - **How many copies of a twinned face to drop is worked out on the
  surface, not per face — it took two field reports from the same model to
  land here, and the trail is worth keeping.** Dropping every copy is
  right for a zero-thickness flap and turned a doubled *wall* into a hole
  (report one: `NotClosedError` out of `simplify`). Scoring each twin
  group by its own edges — drop all copies unless some edge would be left
  on exactly one face — repaired the wall and still tore a doubled
  *patch*, because the groups couple: the middle face of a collapsed
  membrane sees every neighbour as a twin, so its own edges all look
  twice-supported and it drops while the rim stays (report two, same
  error, 32 open edges). The rule that survives predicts nothing:
  duplicated faces start out dropped, and copies are put back one at a
  time wherever the surface as it stands has an edge on exactly one face —
  the rim of a doubled patch comes back on the first pass, the middle on
  the next, once the rim's return leaves its edges half-supported. A
  detached flap stays dropped (its edges land on zero faces, which is no
  boundary), a doubled wall keeps one copy, a clean mesh is untouched.
  - **`simplify` no longer raises at all.** Decimation can break a body
  however clean its input, and `simplify` holds a remedy no other rebuild
  has: cutting less, whose error only shrinks, down to not cutting at
  all — the original is within any budget trivially. So where the rebuilt
  kind is refused it retries at a quarter of the internal bound, three
  times, and as the last resort returns the mesh as it came with a warning
  naming the triangle count. Measured over 64 rough-shell
  contour-then-simplify pipelines: zero raises, four fallbacks — the four
  that used to come out of a notebook as `NotClosedError`.
  - **The guard reaches the last two mesh producers**: `Surface3D._clipped`,
  where cutting along a surface leaves slivers wherever the cut passes near
  a vertex, and `from_dxf`, a foreign file being nobody's promise. Doing
  that made `drop_degenerate_faces` **order-preserving** — it welds only to
  judge coincidence now, and reports on the caller's own vertices — because
  welding sorts, and `from_dxf` promises a file's vertices back in the
  file's own order, which its round-trip test rightly pins.
  - **The closing cap lands on the box face.** Each ghost now holds the
  reflection of the block it mirrors about the contour level, `2*value - v`,
  which puts the crossing halfway between the two centres — on the shared
  face, which is the boundary. One shared constant far past the data's range
  put it hard against the block's own centre instead, half a block short, so
  a grade shell was **cut back** wherever it met the model's edge and its
  complement let out by the same amount. Measured against Monte Carlo
  volumes, a ball meeting the box at a face, an edge, a corner and half
  outside it: `close="above"` read −6.3%, −12.9%, −20.0% and −13.9%, and now
  reads within 1%; `close="below"` read +2.5% to +5.1% and now reads within
  0.2%. Worst of the eight, 20.0% to 0.99%. Blocks on the far side of the
  level set keep their value and make no cap, so every ghost still lands
  outside the kept region — which is what stops the surface reaching the
  shell's own outer face and leaving the body open.
  - **A correction, and the reason it is worth recording.** This same change
  was tried earlier in the day, measured, and reported as a regression that
  doubled the error. That measurement was against a box I had got wrong:
  `BlockSet3D(start, ...)` takes the first block's *centre*, so a model built
  from `[0,0,0]` with a step of 10 spans [−5, 155] and not [0, 160]. Every
  "exact" volume in that table was the wrong reference, and it reversed the
  sign of the finding — the cap was reported as bulging *out* by 10–31% when
  it was in fact falling *short*. The lesson is cheap to state and was not
  cheap to learn: **take the box from the object** (`bounding_box`), never
  from the arguments it was built with, and prefer Monte Carlo over a closed
  form whose derivation encodes the same assumption twice.
* **The mesh pipeline profiled and sped where the profile said, not where
it looked slow.** At 33k blocks and 56k-triangle shells: the boolean was
8.6 s with 92% of it inside `vtkImplicitPolyDataDistance.FunctionValue`,
`get_contour` 0.7 s, `simplify` 0.17 s — so only the boolean was worth
real work, and it is the operation the exact engine's removal made
universal.
  - **The distance queries run on a pool of forked workers.**
  `_signed_distance` splits any query past 20k points over up to 16
  processes, one locator each. Processes and not threads, and it is not a
  style choice: **VTK holds the GIL through `FunctionValue`**, measured at
  1.2x on 8 threads where forked processes reach 7.9x on the same
  1.2M-point benchmark, bit-identical. End to end the crossing boolean
  went **8.6 s to 2.9 s**; the remaining time is locator builds, pool
  spin-up and the contouring, not queries. Verified stable with CUDA live
  in the parent — the workers touch VTK and numpy alone — and Python's
  fork-in-threads `DeprecationWarning` is suppressed at that one call
  site with the reasoning in a comment, since the alternatives lose the
  race: threads sit behind the GIL, and spawned workers would import the
  package, TensorFlow and all, per pool. Platforms without `fork`
  (Windows outside WSL) take the serial path unchanged.
  - **The crossing test probes before it commits.** `_resolved` asked
  every vertex of each body which side of the other it falls on — 0.79 s
  at 55k vertices a side — when crossing, the common case, is usually
  provable from a couple of thousand. A probe of ~2k vertices a side runs
  first; the full queries are paid only when the probes come back
  one-sided, which is the only time their answer is needed in full.
  - **`get_contour` reads its column from the container**, by the path
  `_contour_column` had already resolved, instead of building the whole
  `as_pyvista` export — every corner welded, every variable's every
  attribute filled — just to read one column back out of it. On the
  one-variable profiling model that was worth little (0.05 s); the cost
  it removes scales with columns times blocks, which is exactly what a
  real categorical model carries. The `geoml_paths` label lookup went
  with it, there being no label to look up any more.
* **The implicit engine validated on real ore-body meshes, and hardened
where they broke it.** The user supplied the practical case: two closed
bodies from the Assen Fe project — an uncertainty envelope of 650k
triangles and a hematite envelope of 2.5M, several lobes each, flat
clipping walls on exact box planes, at true mine-grid coordinates. The
intersection came back an **open `Surface3D`**. Three findings, each
measured:
  - **The distance locator lies, and no oracle tells the truth.**
  `vtkImplicitPolyDataDistance` signs by the closest feature's
  pseudonormal, and on the hematite mesh it called a whole cone of points
  forty metres *outside* the body inside — -43 where +40 was true — which
  carried the boolean's zero surface out through the grid margin, where
  the contour is sliced open. Dropping the mesh's 1228 zero-area faces
  changed nothing. Replacing the sign with `vtkSelectEnclosedPoints`' ray
  casting was built, measured and **reverted**: the same 58 probes stayed
  wrong and whole columns of good ones flipped — a ray through a
  degenerate patch corrupts the parity of every point in its shadow — and
  the outputs quintupled and opened everywhere. A pseudonormal error is
  local; a parity error is a column.
  - **So closure is made unconditional instead of the field truthful.**
  The answer cannot exist outside the region box (the boxes' overlap for
  an intersection, their union for a union, the first body's for a
  difference), so the box's own signed distance — analytic, no locator —
  caps the combined field 1.5 cells out. Whatever the locators report
  inside, the zero surface cannot cross the cap. 1.5 cells rather than
  0.5 because a body's own clipping wall sits *on* the region boundary,
  and two zero surfaces closer than a cell pinch marching cubes, which
  has one crossing per edge to give.
  - **The residue is pinholes, and the engine heals them at its own
  resolution.** After the cap, the difference still returned 6 open edges
  of 350k triangles — sub-cell pinches where walls meet the cap region.
  Where the contour of the combined field is not a body, the engine now
  tries `heal(hole_size=2*step)` before answering; a pinhole is honestly
  repairable at the resolution the answer is exact to, which is the step.
  - **End state, all measured on the real pair**: intersection, difference
  and union all `Solid3D`, ~14-16 s each, and the algebra closes —
  intersection + difference matches the first body's volume to **0.05%**,
  and inclusion-exclusion for the union to 0.06%. The meshes are too large
  to bundle, so the regression test injects the locator's lie directly
  (`test_a_lying_locator_cannot_carry_the_surface_out_of_the_region`) and
  requires the boolean to close anyway.
* **`BlockSet3D.get_contour` paints and runs flying edges instead of
welding hexahedra.** The user pointed at Micromine contouring a mixed-size
block model fast and asked what the trick was; profiled at 912k blocks,
ours had no trick to miss — 9 of 11.9 s went to building the unstructured
mesh the contour ran on (welding eight corners per cell through `unique`,
then VTK's single-threaded unstructured contour at 3.8 s) rather than to
the sizes themselves.
  - The repair uses what `_cut_to_contour` already guarantees: near the
  surface every cell is at the one finest size, so the same field can be
  **painted onto slabs of a regular grid** — array writes, no welding —
  and contoured by `vtkFlyingEdges3D`, which is threaded and measured at
  0.85 s where the unstructured contour took 3.8. Slabs are capped at 8M
  cells with one ghost cell layer either side, so the corner averaging at
  a slab face sees the same neighbours the full grid would and the pieces
  weld back seamlessly; the closing ghosts paint like any cell, with the
  hollows beyond the smaller ones filled far-outside so the cap stays
  shut. A rotated set contours on its lattice and turns the finished
  vertices, exactly as its `_hex_mesh` did.
  - **What Micromine's trick was not copied**: resampling everything to a
  common coarse size is where its rounded corners and inflated limits
  come from. The painting is at the contour lattice's own resolution —
  the field is represented differently, not changed — and the volumes
  agree with the welded path to seven significant figures on every case
  measured.
  - Two numpy repairs rode along, found by the same profile: the size
  classes were looked up by `unique` over rows (void-record sort, ~1.5 s)
  and are now one integer key; and `_at_corners` summed through
  `np.add.at`, which is several times slower than the `bincount` that
  replaced it.
  - Measured at 912k blocks (base lattice 192x192x64, supersample 1):
  `close="above"` 11.9 to 8.5 s, `close="below"` 21.8 to 12.5 s, open
  contour 9.5 to 6.6 s, supersample 0 2.9 to 1.6 s. What remains is
  `_cut_to_contour`'s corner matching and the finished mesh's own
  classification welds, both numpy sorts — the next lever if contouring
  registers again.
  - **The real model then failed the painted route, and the repair is a
  router, not another patch.** On the 908k-block Assen model, every rock
  type's indicator contoured at 1e-4 came back a closed-but-inconsistent
  `Mesh3D` — 46 same-way edges of 1.2M triangles — where the welded-hex
  route had been clean on the same calls. The cause is semantic, not a
  bug in either: an indicator field crosses every cell within a whisker
  of a corner, and at fine/coarse interfaces the painted grid's corner
  values are volume-weighted where the welded mesh's are cell-equal,
  which is enough to pinch a surface that hugs the lattice. Two source
  repairs were built, measured against the real model, and reverted:
  snapping vertices to the lattice planes left 44 of the 46 bad edges
  standing, and dropping collinear zero-area faces tore 46 902 edges open
  — collinear faces are structural on a lattice-hugging surface in their
  tens of thousands. What stands instead: fields that pinch **announce
  themselves** (the share of near-level cells within a thousandth of the
  span from the level measured 0.034-0.134 on the six real indicator
  fields against 0.000-0.003 on grade fields), so past 0.01 the welded
  route is taken directly; and whichever route runs, a `Mesh3D`
  classification sends the painted result back through the welded one,
  so the routing is a cost decision and never a correctness one.
  - **End state on the real model**: all six rock types close as
  `Solid3D` where the session began with `NotClosedError`s, and the
  Hematite contour's volume matches the surface saved from it days
  earlier to five figures. The painted route serves grade-like fields at
  its measured 1.4-1.8x; indicator fields pay the welded price they
  always did, once, instead of twice.
* **`supersample` defaults to 0, and the disjoint boolean conclusion got
cheap.** Both from running the whole workflow on the real Assen model, on
the user's call.
  - The supersample's measured benefit — rounder and *closer*, worth a
  model 3.4x the size — was established on smooth fields, and the fields a
  domains workflow contours most are near-binary indicators, which are
  corner-locked either way: at 908k blocks, one level cost 3.5-5.8x the
  time and moved the volumes 0.03-0.17%. The six-rock contour loop went
  673 s to 147 s at the new default; pass `supersample=1` for a smooth
  grade at an interior cut-off, which is what the docstring now says. One
  test pins `supersample=1` explicitly, its 3% tolerance having been
  measured there.
  - Rock shells are mutually exclusive, so their pairwise intersections
  are the boolean the workflow actually runs — and a disjoint answer never
  goes to the implicit grid, it is concluded exactly, which used to mean
  querying every vertex of both meshes. Now: boxes apart conclude with no
  query at all; overlapping-box disjoint pairs pay one full scan of the
  smaller vertex set, since two crossing surfaces put vertices of *each*
  on both sides of the other and one side's scan therefore decides for
  both; and the second scan is owed only where nesting is still possible,
  which the bounding boxes say for free — plus always after an
  all-inside verdict, nesting being exactly where a crossing could hide
  from one scan. All 15 Assen rock-pair intersections: 95 s, 14 through
  the implicit grid (adjacent domains overlap by thin films of 400-8900
  m³ — real, not noise), one exactly disjoint.
* **The two contour routes are one, and the router above is gone.** The
router was an admission — the painted route disagreed with the welded
mesh on fields that hug the lattice — and closing the disagreement at
its sources turned out to dissolve it. Two sources, found in sequence on
the same real model.
  - **The painted grid now carries the welded mesh's own point field.**
  Painting cell values and letting `cell_data_to_point_data` average
  them puts volume-weighted values at every fine/coarse interface corner
  where the welded mesh's are cell-equal. So the cells are not painted
  at all any more: `_painted_contour` builds the corner table itself —
  one integer key per block corner, one vote per block whatever its size
  — and fills the interior of any block larger than one cell with the
  trilinear reading of its own eight corner means, which is exactly what
  VTK interpolates across a hexahedron, and trilinear survives
  subdivision, so the fine lattice reproduces the coarse cell's surface
  rather than resampling it. A block holding no value paints its points
  absent rather than background — unpredicted ground is not outside the
  model. This alone took the real model's worst field (BIF, 46 same-way
  edges) to a clean `Solid3D` and cut the failures from six fields in
  six to two in seven; painting points needs no slab ghost layers and
  no `cell_data_to_point_data` pass, so it is also mildly faster.
  - **The last two pinches were float32, not geometry.** Full-precision
  inspection of a surviving pinch showed every output coordinate
  quantized: VTK's image contours write single-precision points with no
  say in the matter (flying edges has no output-precision setting at
  all), and at mine-grid coordinates that is a northing of 2.8e6
  resolved in steps of **0.25 m** and any crossing within ~1e-4 of a
  lattice plane snapped onto it — both surviving pinches sat at a bench
  boundary the surface grazed, welded shut by the snap. The welded mesh
  never had the problem only because an unstructured contour inherits
  its input's float64. The image now lives in the lattice's own frame —
  origin zero, spacing one, where float32 resolves ~1e-5 of a cell —
  and the world enters after the contour, in float64. The grid path was
  checked for the same defect and is already built this way (skimage in
  the local frame, origin added afterwards).
  - **End state on the real model**: all seven fields — six rock
  indicators and the uncertainty — close as `Solid3D` straight through
  the painted route, the welded fallback firing zero times; it stays as
  the classification net it always was, a closed-but-inconsistent
  `Mesh3D` never being what a level set means. The six-rock loop runs
  52 s where the router had it at 147 (indicators paid the welded price)
  and the release began at 673. Volumes unchanged to five figures.
* **`simplify` measures on a pool of forked workers.** Profiled on the
real uncertainty shell (522k triangles), half of a `simplify(0.5)` was
the deviation measurements — 4.9 of 10.0 s in `FunctionValue`, which
holds the GIL, so threads were never an option (the boolean engine
measured that lesson already: 8 threads 1.2x, 16 forked processes 7.9x).
`_DistanceQueries` is the keep-the-pool sibling of `_signed_distance`:
the fork and the per-worker locator builds are paid once per `simplify`
call and every measurement rides them — three to six measurements, all
probing the same original. The call went 10.0 to 6.2 s, bit-identical
output; what remains is `vtkDecimatePro` itself, which is sequential by
nature. A mesh too small to repay the spin-up keeps the serial path on
one kept locator, and the boolean's own `_signed_distance` is untouched.
* **A confusion matrix, on both plotting backends.** `confusion_matrix`
on `Explorer` and `Interactive`, with the counting in
`prepare.confusion_matrix` as everything drawn twice is.
  - Rows are what was measured, columns what the model called there, in
  the variable's own label order — with any measured category the
  variable does not model appended after, so a stranger in the data gets
  a row whose diagonal can never light rather than being dropped. The
  shading is each cell's share of its measured row (categories are as
  unbalanced as rock types usually are, and raw-count colour lights the
  dominant row and hides what happens to a rare one); the counts are
  written in the cells, and the title carries the overall agreement.
  - Three kinds of location do not count, each for its own reason: a
  contact carries two measurements and neither alone is a truth to be
  right about (the `boundary` flag, the same rule `compute_metrics`
  applies); an unpredicted location has no claim to score; an unmeasured
  one nothing to score it against. All in `prepare`, so the two backends
  cannot disagree about who counted.
  - The docstrings say what every model-checking figure here says: only
  honest on data the model has not seen — at a training location the
  prediction interpolates its own measurement, and the diagonal
  congratulates the model on remembering it. The out-of-fold container
  `cross_validate` fills is the honest input.
  - `Selection` grew `_require_categorical`, the missing sibling of
  `_require_continuous`. Seven tests on the counting and one per
  backend on the figures.
## version 0.6.6
* **The kernel derivatives are exact.** `covariance_matrix_d1` and
`covariance_matrix_d2` — the point-to-direction and direction-to-direction
covariances that directional data are built on — were central differences
over a hard-coded step of 1e-3. They are now differentiated, and the way
they are differentiated is the interesting part.
  - **Differentiating the covariance matrix does not work, and cannot.** A
  covariance goes through a Euclidean distance, and `|h|` is not twice
  differentiable at `h = 0` — which is exactly the diagonal every d2 call
  has, since `self_covariance_matrix_d2` asks for a point against itself.
  Forward-mode autodiff over the matrix returns NaN there. So the kernel is
  differentiated instead: `kernelize` is a one-dimensional elementwise
  function, two nested `GradientTape`s give `f'(r)` and `f''(r)` exactly,
  and the spatial part is carried by the chain rule
  `dK/dy.u = -phi A`, `d2K/dx dy = -(psi A B + phi C)`, with
  `phi = f'(r)/r` and `psi = (f''(r) - phi)/r**2`. Both have removable
  singularities at the origin, `phi(0) = f''(0)`, and A and B vanish there —
  so the coincident case is *computed*, not stepped around.
  - Directions are pushed through the transform's Jacobian by a
  `ForwardAccumulator`, which is what keeps an anisotropy, a projection or
  a periodic transform correct without any of them declaring a derivative.
  - **Measured**, against the closed forms written out by hand: relative
  error 5e-16 where the central differences gave 2.3e-7 (d2) and 5e-10
  (d1), and 5e-16 on the derivative variance where they gave 2.5e-5 for
  Cubic and 6.7e-5 for Matern32. The error the differences carried grows
  with the square of the range — the answer shrinks as `1/range**2` while
  the rounding amplified by dividing out `1e-3` twice does not — reaching
  1.3e-4 at range 5000. **At which point the derivative block stops being
  positive definite and the Cholesky in `GradientConstrainedInput.refresh`,
  which adds no jitter at all, returns NaN.** It now factorizes.
  - Also faster, since it is one distance matrix rather than four: 4.1 ms
  to 1.9 ms for a 600x600 d2. On the fold notebook the model is unchanged
  (ELBO -125.591 either way) — the old error was far below what that
  well-conditioned case could feel, which is the honest summary of when
  this matters and when it does not.
  - **`Linear`'s directional covariances were wrong** and are now right.
  The old code measured its step about a shifted origin, which cancels only
  when the covariance depends on differences alone; `Linear` does not, so
  `dK/dy.u` came back as `(x - min(y)).u` instead of `x.u`. It now takes
  the base class's generic forward-mode route, which for a covariance with
  no distance in it is exact.
  - `point_variance_d2` loses its `step` argument, having no step any more.
  - Kernels with no derivative process are unchanged in status and changed
  in number: `Exponential` and `Spherical` have a kink at the origin, so
  the variance of their derivative is infinite. The differences reported a
  finite number manufactured by the step size; the chain rule reports one
  implied by the kernel's own `epsilon`. Both are arbitrary. Using them
  with directional data is a modelling error either way.
  - **`Product`'s directional covariances were wrong** and now follow the
  product rule. `_NodeCovariance` applies its operation to the components'
  derivatives, which is the derivative of the operation only where that
  operation is *linear* — true of `Sum`, and not of a product, where it
  returned the product of the derivatives. Measured on a case with an exact
  answer (a product of Gaussian kernels is a Gaussian kernel, with
  `1/c**2 = 1/a**2 + 1/b**2`): **98% wrong**, now exact to 9e-16. The rule
  needs the derivative in the *first* argument, which the interface does
  not offer and does not need to — a covariance is symmetric, so
  `dK/dx = transpose(covariance_matrix_d1(y, x, dir_x))`, the same identity
  that already assembles the mixed blocks in
  `full_directional_covariance_d1` and `GradientConstrainedInput.refresh`.
  `self_covariance_matrix_d2` is overridden alongside, for the same reason.
* **`Spline.backward` is a solve, not a second guess.** It used to
interpolate the knots the other way round, which is not an inverse: the
inverse of a cubic is not a cubic, so the two curves met at the knots and
parted between them. `forward(backward(y))` came back **1.3e-1** from `y`,
a tenth of a standard deviation, meaning the model inverted a different map
from the one it fitted. `MonotonicCubicSpline.invert` now solves the
forward polynomial by Newton, and the round trip is **1e-11 or better in
both directions**.
  - The interval is located once, in `y` — a monotone map puts the answer
  in the interval the query occupies among the y-knots, so the bracket
  cannot move and no step searches again. The Hermite form is converted to
  monomial coefficients once, so each iteration is a Horner pass rather
  than four basis polynomials; that is what keeps twelve steps cheaper than
  three were before it. `backward` costs 2.6x a forward pass.
  - The last correction is taken from a detached iterate, so the derivative
  is the implicit `1 / f'(t)` exactly rather than the derivative of an
  unrolled loop. Checked against a finite difference.
  - **The accuracy floor is the transform's, not the solver's**, and that
  is worth knowing when reading the tolerance: a normal-score fit leaves
  half its intervals nearly flat — 40% have a span below 1e-4 even at the
  default five knots per arm — and where the forward map compresses by `s`,
  a residual at machine precision returns magnified by `1/s`. The measured
  5e-11 floor is exactly `2.2e-16 / 1e-5`. It is information `forward`
  discarded, and no inverse recovers it.
  - Twelve iterations is the default because that is where the error stops
  falling, measured over six samples at knot counts from 11 to 161: 1e-3 at
  three steps, 1e-7 at eight, 5e-11 at twelve, unchanged at thirty-two. A
  single sample suggested three was enough, which is why the default is set
  from a sweep and not from one run.
* **The interpolators' epsilons no longer perturb the intervals that were
fine.** `w + 1e-6` in the derivative rule and `h + 1e-6` in the evaluation
guarded against a zero-width interval by shifting *every* interval; at a
knot spacing of 0.125 that is a relative error of 8e-6, which put a floor
under the spline's own self-consistency and was enough to stall the Newton
inversion above. `_safe_width` replaces only the zeros, so a legitimate
width is divided by itself. Interpolating a straight line now returns the
line to 1e-12 at every spacing tested. The base class used 1e-12 for the
same guard and the monotone subclass 1e-6, a millionfold apart with no
reason recorded; both are exact now.
* **`Spline` initializes on the data, and the projection-pursuit question
it was blocking is answered.** The warping's knots have fixed inputs on a
regular grid and trainable outputs; they used to start on a uniform
partition, which reads out as exactly that grid — the identity. So a
`Spline` began every fit with nothing to say, and a chain of
`Rotation → Spline` pairs rotated repeatedly with nothing gaussianizing
between the rotations. Measured, that initialization moved *backwards*: the
projection-pursuit index on the worst direction went 47 to 111, because
stacked rotations without a marginal transform only remix heavy tails.
  - `Spline.initialize` now fits the knots to the marginal normal-score
  transform — the empirical CDF at each knot through the normal quantile,
  which is the geostatistician's anamorphosis and PPMT's gaussianization
  step. Closed form, monotone by construction, and a *start* rather than a
  fit, as `Rotation`'s ICA already was.
  - Two compositional parameters carry the map, so it is pinned at
  `(-5, -5)`, `(0, 0)` and `(5, 5)`; each arm is rescaled to reach the
  anchors, which keeps the transform's shape and not its scale. That is the
  one approximation: on the Jura elements the index lands at 0.19 against
  0.15 for the exact transform, where the raw data reads 4.65 and Gaussian
  data of the same size reads 0.04.
  - **Because `ChainedWarping.initialize` already threads the data link by
  link, this makes a chain initialize as a projection-pursuit sweep** with
  no new machinery: each rotation runs its ICA on data the previous spline
  has gaussianized. Four sweeps reach the Gaussian null at Jura's sample
  size, not the fifteen PPMT asks for.
  - **And the measurement says to stop at one.** Trained on Jura and scored
  held out, one initialized marginal transform gives the best rmse (0.927
  against 0.933 as it was, 0.938 with no transform) and the best CRPS; a
  single rotation sweep is worse on both; two and four sweeps blow up the
  back-transform, 56 and 189 non-finite predictions of 700, since a chain
  of splines is steep in the tails and `Log.backward` exponentiates what
  comes out. The ELBO improves monotonically with sweeps the whole time,
  which is worth knowing on its own. **Those two blow-up numbers were
  mostly this initializer's own defect** — see the entry below: repaired,
  one and two sweeps are clean on the same draws and four leaves 0.2%, so
  the recommendation stands on the held-out scores rather than on
  arithmetic falling over.
  - Recorded in passing, pre-existing and unrelated to the initializer:
  `Spline.backward` swaps the two knot sets and interpolates again, which
  is not the analytic inverse of a cubic. The round trip is exact only
  where the map is straight — 1e-6 for the identity, 1e-1 at five knots per
  arm, 1e-2 at forty — and an initialized spline is curved from the first
  iteration rather than after training, so it meets that sooner.
  - `geoml/test/test_spline_initialization.py`: the anchors, monotonicity
  at the knots and densely between them, the constant-column fallback, the
  round trip against the identity that isolates whose approximation it is,
  and that a chain leaves its second rotation something to find.
* **The initializer left the map uninvertible where the data runs out, and
now does not.** Reported against a Jura prediction, as
`invalid value encountered in subtract` from `np.quantile` — which raises
only for `inf - inf`, so the simulations held infinities.
  - The knots span [-5, 5] and data does not fill that. Every knot past the
  sample's range therefore reads the same empirical quantile, asks to sit
  on top of its neighbour, and was held apart only by an absolute floor of
  1e-6 on each share. On Jura that put the outer three knots of each arm
  within 1e-5 of the anchor: a forward map essentially flat out there, and
  **a flat forward map is a vertical inverse**. A latent draw a little past
  the knots came back at 1e5, `Log.backward` exponentiated it to infinity,
  and the quantiles taken over the simulations returned NaN. At one
  standard deviation, 1943 of 140 000 back-transformed draws were
  non-finite.
  - The repair is one rule, and which segment it applies to is the whole
  point: **only the outermost segment of each arm gets a real floor**
  (`_OUTERMOST_SHARE`, 0.2 of a uniform share). `backward` locates its
  bracket first and clips every iterate into it, so a flat segment
  *between* knots costs accuracy across one interval and cannot leave it;
  past the last knot the spline extrapolates along the end segment with
  nothing to clip against. That gives an exact bound, independent of the
  knot count: the slope there is at least 0.2, so nothing comes back
  further than five units per unit past the knots. Measured at `y = 8`:
  20.0, against 1.7e5 before.
  - **Two alternatives were measured and rejected.** Continuing the
  transform linearly past the data — the statistically obvious fix — costs
  an order of magnitude of gaussianization, because the arms are rescaled
  to the anchors and the continuation eats the range the data needed
  (lognormal departure 1.6 to 20.5). Flooring *every* segment relatively
  costs a quarter of it (2.0). Flooring only the outermost costs nothing
  measurable: the Jura projection index reads 0.0426 against 0.0428 for the
  broken version, where the raw data reads 0.111 and a Gaussian sample
  0.008.
  - Training does not re-create the flatness: after 150 iterations on Jura
  the outermost gaps sit at 0.28 to 1.62, and the prediction that reported
  this is clean.
  - Four tests in `test_spline_initialization.py`: the outermost share at
  four knot counts, the extrapolation against its own bound, a mass point
  keeping the partition positive, and a `Log`-bottomed chain staying finite
  — which is the reported failure in miniature.
* **A capacity warning the manual made, withdrawn on measurement.**
Chapter 3 taught that a stationary Jura model went from 0.96 to 1.13 times
the data's standard deviation when its inducing set doubled, and explained
it by the kernel range being an unpriced point estimate that lets the field
roughen. Re-measured across 81, 169, 324 and 625 inducing points on three
seeds: **rmse/sd is flat at 0.93** and the number does not reproduce.
  - What does move is the width of the intervals, and reading *that* is
  where the trap was. `compute_metrics` scores the container's stored
  simulations — the **ground**, with the likelihood's noise integrated out.
  Against assays that leaves the noise out of every interval, so a nominal
  90% band reads 0.59 falling to 0.49; through `predict_measurements` the
  same models read **0.936, 0.926, 0.911, 0.894**. Well calibrated
  throughout, drifting gently. The artifact grows with capacity because a
  model with more capacity calls less of the variance noise, so the piece
  being omitted is larger. `cross_validate` and the `accuracy` figure were
  always right — both ask for measurements — and
  `ContinuousVariable.compute_metrics` now says so in a `Notes` section.
  - The range explanation is backwards. The fitted range does fall (3.9 to
  2.7 km across the sweep), but freezing it at the value the smallest model
  chose makes calibration *worse* at every count. It is the model
  compensating, not failing.
  - Recorded in passing, because it bears on when a MAP prior is worth
  reaching for: `range_prior=2.0` moves the fitted range by 0.003 and
  changes no score to four decimals. A log-prior does not scale with the
  data, so against ~1800 likelihood terms two scalars decide nothing.
  - Chapter 3's section is rewritten around the measured table and chapter
  16's cross-reference now matches it.
* **Two directional entry points that could not run at all.** Neither has
test coverage, which is how both rotted unnoticed; both were found while
checking the derivative work end to end.
  - `GP` with `directional_data` died in training: the branch that bypasses
  the warping set `log_derivative` to a Python float, which is float32
  against the float64 it is added to.
  - `StructuralField.predict` died writing its answer, calling `update` with
  `mean`/`variance` where a variable is filled with `average_sim` (the
  prediction) and `mean`/`variance` (the latent pair). This model has no
  likelihood and no warping to separate them, so the prediction is the mean
  itself.
  - Both now train and predict. On the fold dataset the structural field's
  potential is flat along the tangents that constrain it — slope 4e-4 rms
  against 0.76 across them — and `predict_raw_directions` returns 1e-10
  along them.
  - `geoml/test/test_kernel_derivatives.py` is new: 29 tests over the
  closed forms, the transform's Jacobian, coincident points at mine-grid
  coordinates, the positive-definiteness case above, the product rule
  against both an exact and a numerical reference, tracing with a changing
  batch shape, one direction broadcast over every point, and that the
  training gradient still reaches the kernel and transform parameters
  through a derivative of a derivative.

## version 0.6.5
* **The documentation site covers the whole public surface, and the type
check covers every module behind it.** Both had been growing module by
module; this closes them together.
  - **`data/grids.py` is annotated and checked**, the last module the typing
  workstream had left. Thirty-three diagnostics, almost all one idiom: the
  grid constructors take `step` and `margin` as either a number or one value
  per axis and rebind the parameter, so a checker follows the float branch
  down. Each converts into a differently named local now, which is the rule
  this project has now learned three times. `Grid1D` is narrowed to the
  scalar `step`/`end` its `float(step)` always required.
  - **The checked list is every user-facing module** — twenty-seven files,
  the thirteen in `geoml.__all__` and everything under them, all clean. It
  had held twelve. The gap mattered: `plots/prepare`, `explorer` and
  `interactive` were annotated back in 0.6.4 but never added to the list, so
  their annotations were read by editors and verified by nobody, and they
  had drifted to twelve diagnostics. A module annotated but unchecked is the
  worst of both worlds, and `pyproject.toml` now says so where the list is.
  - Real things the checker found while being pointed at more code:
  `math/geometry` called `np.atan2`, which only exists in NumPy 2 (now
  `arctan2`, spelled the same in every version); `viz/plotly` used
  `collections.abc` while importing only `collections`, which works until
  nothing else has imported the submodule first; `prepare.color_choices` and
  `prepare.moving_average` both declared they return an array where they
  return a tuple. `Grid3D.make_interpolator` still calls a function that
  does not exist — left as found, since wiring it to `CubicConv3DSeparable`
  or deleting it is a decision, and now flagged in place rather than only in
  a changelog entry nobody rereads.
  - **Eight new reference pages**: `latent`, `kernels`, `transform`,
  `warping`, `plots`, `viz`, `math` and `stats`, plus grids, blocks, meshes
  and variables joining the data page, which had documented only the point
  containers. The two design records written since the site was built join
  `internals`.
  - **The manual is part of the site.** A `builder-inited` hook copies
  `docs/manual/` into the source tree at build time, which is what keeps its
  relative figure links working; the copy is gitignored, so the chapters
  still live in one place and are still run from there.
  - Getting `-W` back to green took two decisions worth recording. The
  duplicate-object warning is filtered in `conf.py`, because classes here
  share one function object on purpose (`_blockdata` hands the block fan-out
  to the grid classes) and autodoc identifies a member by the qualified name
  of whoever defined it; and `stats.md` excludes the distribution methods
  TFP copies into every subclass, whose docstrings are TFP's. Nothing else
  is suppressed: a broken reference or a malformed docstring still fails the
  build, and five docstrings were fixed to keep it that way — an indented
  code example is a definition list to RST unless it says
  `.. code-block:: python`.
  - **`test_manual.py`** runs every chapter through `run_blocks.py`. It is a
  release gate only: a full pass trains real models and takes about twenty
  minutes, so it sits on the structural job's ignore list with the other
  heavy files.

* **The point-estimated parameters can carry priors now, and the important
ones do by default.** Only the variational state was ever priced by a KL;
the ranges, mixing weights and friends were optimized against the ELBO with
box bounds and nothing else. They stay point estimates — no posterior is
integrated. A parameter that declares a `prior` adds its log-density to the
training objective next to the KL, making the whole a bound on
`log p(y, theta)`: MAP, one differentiable term, and the gradient does the
rest. A model with no priors trains on bit-for-bit the objective it always
did, and a saved model reloaded and *refit* gains the defaults, since
persistence replays constructors.
  - The defaults are canonical because the network works in whitened space,
  and each accepts a number for the experienced user or `None` to switch
  off: `BasicGP(range_prior=2.0)` is a Gamma whose **mode** sits at 1 — MAP
  pulls toward the mode, so "mean 1" would actually drag ranges toward 0.5
  — falling to minus infinity as a range collapses and linearly as it
  grows; `Linear(weight_prior=1.0)` is a standard Gaussian on the free
  weights (the parameter whose count scales with the network), whose hard
  [-1, 1] walls step back to a ±10 safety net;
  `MultiStructureGP(weight_concentration="staircase")` is a Dirichlet whose
  peak puts shares in proportion to each structure's starting range, and
  its range priors peak at each structure's own staircase position rather
  than at a common 1, which would fight the ordering.
  - **Measured before shipping, and one idea died.** On Walker Lake against
  the exhaustive truth (the one dataset where overfitting is measurable
  rather than inferred), the `MultiStructureGP` priors improved every
  truth-facing number on every seed — rmse against the field 200.1 to
  199.1, variogram score 42.38 to 42.11, held-out goodness 0.642 to 0.652 —
  while the staircase *initialization* worsened all of them (206.2, 43.61):
  training never leaves that basin, where a uniform start finds the right
  weights with or without the prior. So the prior ships and the staircase
  start is recorded as rejected. On the chapter-16 Jura network the priors
  are neutral-to-slightly-positive: CRPS under the Linear prior beats the
  unpriored model pairwise on every seed, by under a percent, and nothing
  is ever worse.
  - `LinearCombination(per_component=True)` is new alongside this: one set
  of mixing weights per output component instead of one for the node, so
  one element can lean on a trend another ignores, with a symmetric
  Dirichlet holding each component's shares near equal until its data
  argues. Off by default — measured neutral on Jura, whose seven elements
  evidently agree about how much trend they want.
  - Design record with the census, the MAP argument and both tables:
  `docs/parameter-priors.md`. The census is worth one line here: on the
  deep Jura network the unpriced parameters number 45 against 21 628
  priced ones, so the density ceiling measured earlier in this release was
  never about them.

* **Training can stop when the bound settles**, through the new
`GPOptions(training_tolerance=...)`. The bound is smoothed with an
exponential moving average, and training ends once its gain over the last
twenty iterations falls below that fraction of everything gained since the
call began. `max_iter` becomes a cap rather than a count. **Off by default**,
so nothing trains differently unless asked.
  - Normalizing by the progress made so far is the load-bearing part. An
  ELBO's magnitude means nothing on its own — it grows with the number of
  data points and with whatever normalization the likelihood carries — so a
  fraction of the *value* would not transfer between models. A fraction of
  the gain does, and it self-protects early on, when little has been gained
  and the threshold is correspondingly small.
  - **The comparison is over a window, not between consecutive iterations.**
  For a bound approaching its limit with time constant `tau`, a
  per-iteration test fires once `exp(-t/tau) < tolerance * tau`, so a slowly
  converging fit stops almost at once and how early depends on a `tau`
  nobody knows in advance. Measured on three recorded curves at tolerance
  0.01: a window of 1 fires after 26 to 36 iterations having kept 81-84% of
  the total gain, a window of 20 keeps 96.2-97.2%, and a window of 40 keeps
  98.1-98.6%. Twenty is the internal window, and it is not a knob: one
  number behaved the same way on every case tried, including SVI, where the
  criterion reads one value an epoch (the mean over its batches) rather than
  one per minibatch.
  - **Where it fires**, at 0.01: Walker Lake stops at iteration 118 of 600,
  the Jura seven-element model at 117 of 600, and the non-stationary Jura
  network of the skill's worked example at 130 of 400 — which is close to
  the 250 that network was given by hand. Under SVI, epoch 92 of 300.
  - **What that costs, which is the only gate that matters.** Held out, over
  three seeds each: Walker's rmse is 1.5-2.0% *worse* and its CRPS 6% better;
  Jura's rmse is 1% better and its CRPS 6.5% better. Deep Jura, one seed:
  rmse 1.4% better, CRPS 2.0% better, in 46% of the time.
  - **Calibration is the surprise.** Deutsch's goodness went from 0.49-0.51
  to 0.73-0.75 on Walker and from 0.54-0.58 to 0.77-0.82 on Jura, every seed,
  and from 0.45 to 0.57 under SVI. The last few percent of the bound buys
  sharper posteriors that held-out data does not support, which is early
  stopping doing what early stopping has always done. Worth knowing before
  reading a training curve as something to maximize.
  - Two behaviours worth stating because they are deliberate. A run whose
  bound goes to NaN **never satisfies the test**, so it goes to its cap and
  leaves the NaNs in the log where they can be seen, rather than stopping and
  reporting success. And a call that begins already converged has nothing to
  take a fraction of, so it also runs to its cap: the rule can end a phase of
  training, never skip one. That is what keeps the phased pattern
  (`train`, `set_learning_rate`, `train`) working, since each call is judged
  from its own starting point.
  - `cross_validate` needs no argument for this: each fold is rebuilt from
  the saved model, options included, so a tolerance set on the model sizes
  every fold's refit instead of the hard-coded 200.

* **The `geo-ml` skill gained the non-stationary multivariate recipe**, from
a notebook verified end to end: `08_Jura_non_stationary.ipynb` joins the
references (stripped of outputs, 5.6 MB to 20 kB) and a new worked example
sits beside the categorical one. The pattern it records is the one that was
missing — how a categorical variable *influences* a numerical one, and how
to make a model non-stationary without leaving the API.
  - `Linear(cat, size=n, unit_norm=False)` → `LinearCombination(trend, num)`
  adds a trend read off the rock fields to the grades. `unit_norm=False` is
  the load-bearing argument: it lets the mixing weights shrink to zero, so
  the model can conclude the geology says nothing about an element. The
  alternative — making the categorical node a *parent* of the numerical one
  — is a genuine deep GP and a much stronger commitment.
  - `BasicGP(size=2)` → `GPWalk` → `BasicGP` puts the categorical fields in
  a space the model bends, and the numerical fields read the *unmoved*
  input: the two need not share a geometry. The input transform starts small
  (`Isotropic(0.05)`) because the walked coordinates do the long-range work.
  - Measured on Jura's 100 held-out sites: metals at 0.90 times their own
  standard deviation and rock balanced accuracy 0.72, against 0.95 and 0.67
  for the stationary model the manual's case study used before. 250
  iterations is enough; 500 measures identically.
  - **A finding that corrects an earlier one.** 0.6.5's variogram work
  established that this dataset's stationary model degrades sharply with
  extra inducing points (0.96 → 1.13 from 81 to 169). The non-stationary
  network is *flat* from 259 to 1220 — five times the points, five and a
  half times the training, no measurable change. The ceiling belongs to a
  configuration, not to the method, and both the skill and the manual now
  say so rather than quoting a number.
* **TensorFlow's retracing notice is quiet for this package's own graphs**,
and the retracing it was reporting is largely gone. Two changes, and the
first is the one that matters.
  - `predict_raw`'s traced function is built with `reduce_retracing=True`.
  The batch shape is the one input that varies without meaning anything: a
  grid divides into equal batches and a short last one, and every container
  of a different size started the count again. Measured over eight
  predictions of differing size, **eight traces and 7.6 s become two and
  2.2 s**, bit-identical, and XLA agrees with the eager path either way
  (checked at `jit_predict` both ways, gap 1e-14). What still retraces —
  `n_sim`, `include_noise` — is baked into the graph and has to.
  - `geoml.math.tf.silence_retracing_notices()` drops the notice for
  `_predict_raw` and `refresh_cached`'s inner function, and is installed
  when `geoml` is imported. Call it with `False` to hear them again, which
  is worth doing if a prediction seems to be compiling rather than
  computing. The filter matches those two names only, so anything
  TensorFlow says about a user's own `tf.function` still comes through.
  - Why suppress at all, when retracing notices are usually worth reading:
  the message interpolates the repr of the function it names, and for a
  bound method that is the **entire model** — every node, every parameter —
  so one notice can run to hundreds of lines and bury whatever it lands in.
  `refresh_cached`'s function meanwhile takes no arguments and is traced
  once per network; a session holding many models (a cross-validation, the
  manual) trips TensorFlow's counter without anything being retraced twice.
  - Worth knowing, and not a bug: with `jit_predict` on and off, the
  *deterministic* columns agree to 1e-14 (`latent_mean`, `latent_variance`,
  `noise_variance`) while `prediction` does not, because it is the mean of
  the simulations and the two backends draw different random numbers. The
  gap falls as `1/sqrt(n_sim)` — 79, 19, 5.5 at n_sim 4, 40, 400 — which is
  Monte Carlo noise, and is why `test_jit_prediction_matches_the_plain_one`
  compares the latent mean.
* **The 0.6.0 aliases now say they are going away.** The ten shims left at
the pre-0.6.0 flat paths (`geoml.tftools`, `geoml.drillhole`, `geoml.random`,
…) promised "one release" in a docstring and said nothing at runtime, so
0.6.1 through 0.6.4 all passed without anyone being told. Each now raises a
`DeprecationWarning` naming where the module went, ahead of removal in 0.7.0.
  - **The notice is lazy, and that is the whole design.** `geoml/__init__.py`
  imported all ten eagerly so that `geoml.geometry` kept working as an
  attribute; warning from there would have fired ten notices at every
  `import geoml`, for the majority who never touch an old path — which is how
  a warning teaches people to ignore warnings. A module-level `__getattr__`
  (PEP 562) keeps the attribute working and moves the notice to the moment
  someone reaches for it. Verified in a subprocess: a plain `import geoml`
  emits nothing and loads no shim.
  - It is also re-issued from `__getattr__` rather than passed through from
  the shim body, because a `DeprecationWarning` attributed to library code is
  one Python's default filter *hides* — attributed to the caller, it shows.
  One notice, pointing at the line that asked.
  - **`kernels.py` was itself importing through the `tftools` shim** (three
  helpers, two of which had landed on the `math.linalg` side of the split),
  so the notice would have fired for every user on its own. Repointed at
  `geoml.math.tf` and `geoml.math.linalg` directly, and the test suite —
  which reached for `geoml.geometry`, `geoml.inducing`, `geoml.drillhole`
  and `geoml.graphviz` in nine files — now uses the current paths too. The
  last of those was found by running the suite and reading its warnings
  rather than by grep: making the shims speak is what made it audible.
  - `test_deprecated_paths.py` pins the contract: every alias resolves,
  warns exactly once, names its destination and 0.7.0, and is attributed to
  the caller; a plain import stays silent; an unknown attribute still raises.
* **`warping.Rotation` documented, and its FastICA warning silenced.** The
class had no docstring; it now says what it is — an orthogonal, volume-
preserving rotation whose matrix is trainable and **initialized by
independent component analysis**, putting the axes along the directions of
maximum non-Gaussianity. That is what makes `Rotation -> Spline` the usual
pairing: the rotation finds where the marginals depart most from a Gaussian,
and the spline is applied where that departure lives.
  - The `ConvergenceWarning` sklearn raised when the ICA fit hit its
  iteration limit (visible on the Jura case in the suite) is suppressed at
  that one call, with the reason recorded: ICA is asked for a *starting
  point*, not a converged answer, since the rotation is a trainable
  parameter that training moves from wherever the fit stopped. Raising
  `max_iter` instead would slow every initialization to chase a number that
  is overwritten anyway.
  - `Rotation` and `ScaledSimplex` join `warping.__all__`, which had left
  them out — reachable as module attributes but missed by a star import.
  Every warping class in the module is now named there.
* **`warping.Log` was returning the Jacobian instead of its logarithm**, and
`ChainedWarping` was seeding its accumulator with the column count. Both are
one-line fixes to the same quantity: the second value `forward` returns is
the *log* of the Jacobian determinant, which is the only form that composes
by addition along a chain.
  - `Log.forward` computed `sum(1 / (x + shift))` where the truth is
  `-sum(log(x + shift))`, while `Softplus` and `ZScore` next to it both
  returned logs. Found while writing the manual's Jura chapter, which leads
  with `Log -> RobustPCA -> Spline`; verified against a finite-difference
  Jacobian, where `Log(2)` on `[[1, 2], [3, 4]]` reported `[1.5, 0.583]`
  against a true `[-0.693, -2.485]`.
  - **What it cost in practice was small, and worth stating precisely.**
  `Log` carries no trainable parameter, so as the *first* link of a chain
  its term is a constant that no gradient sees, and every model in the
  package and the manual used it that way. Put anything trainable ahead of
  it and the model optimizes the wrong objective. The chain's accumulator
  was likewise a constant offset of `size_in`. So no fitted model changes;
  what changes is the ELBO's reported value, and printed training curves for
  chained warpings shift down by the chain's width.
  - `test_noise_integration.py` now checks nine warpings' reported
  log-determinant against a numerical Jacobian, and pins the chain's
  addition rule and the zero for a chain of identities. Both were confirmed
  to fail on the old code before the fix landed.
  - The four that were left alone in the first pass — `PCA`, `RobustPCA`,
  `CenteredLogRatio` and `ScaledSimplex`, each reporting zero where the
  truth was something else — were finished a few days later. See the entry
  below.
* **The last four zero log-determinants say what they are worth.** Each was
reporting zero for a map that changes volume, wrong in the ELBO's value and
invisible in its gradient, so no fitted model moves and no prediction
changes; what changes is any comparison of ELBOs across warpings.
  - `PCA` and `RobustPCA` rotate and then divide by the square roots of the
  eigenvalues, so the term is `-0.5 * sum(log(eigvals))`. Measured against a
  finite-difference Jacobian on three variables: formula `-1.764258740`,
  numerical `-1.764258740`. With **fewer components than variables** there
  is no square Jacobian and no determinant to report at all, since the map
  projects rather than transforms; that case keeps its zero and now says so
  in a comment and in a test of its own.
  - `ScaledSimplex` divides each part by a constant of its own, a diagonal
  Jacobian, so the term is `-sum(log(scale))`: `3.298589068` against a
  numerical `3.298589068`.
  - `CenteredLogRatio` was the interesting one. Between arrays of `n_dim`
  columns its Jacobian is **singular**, not merely constant — the
  transformation ignores a rescaling of the whole row, so the determinant is
  exactly zero and there is nothing to report. Read as the bijection it
  really is, from the simplex onto the hyperplane where the components sum
  to zero (both one dimension smaller), the volume factor is
  `-log(n_dim) - sum(log(x))`, which unlike the others **varies with the
  row**, growing without bound as a part approaches zero.
  - **No balance matrix had to be chosen**, which was the objection that
  parked this. Any two orthonormal bases of that hyperplane differ by a
  rotation of determinant one, so the volume factor is the same in all of
  them — measured with four (the SVD basis, the classical Helmert balance
  matrix, and two random rotations) at `n_dim` 3 and 5, agreeing to ten
  decimals with each other and with the closed form, which mentions no basis
  at all. A test pins that invariance, since it is the reason the one-line
  implementation is legitimate.
  - The redundant latent direction that comes with it is the familiar
  softmax one — `backward(z + c)` is `backward(z)` — so it is
  unidentifiable and harmless rather than a defect to design around. The
  consequence for the objective is stated in the class docstring: it is a
  density on the hyperplane, comparable between compositional models and
  only loosely against a warping that is one to one.
  - Three of the four join the parametrized log-determinant test, which now
  covers twelve warpings; the log-ratio has two tests of its own for the two
  claims a numerical determinant cannot make.
* **The variogram fan was a nugget below the data, and looked like a
verdict.** `plots.prepare.variogram` drew the measurements' experimental
semivariogram against the stored realizations, which are of the *ground*
with the likelihood noise integrated out. The two are not the same
quantity, so the fan sat low at every lag by the fitted nugget and every
model read as over-smooth. On Walker Lake the shortest lag showed 8 555
against the data's 55 013.
  - The fan is now raised onto the measurements' footing before it is
  drawn. An error of its own at each location adds `(var_i + var_j) / 2` to
  every pair's contribution, so the correction is that quantity averaged
  over the pairs in each bin, read from the `noise_variance` column. Same
  example after: 54 344 against 55 013, with the fan running about 15%
  under the data through the mid-range — which is the *real* finding, a
  range fitted slightly long, previously buried under the offset.
  - **It is analytic, not simulated.** Only the noise's variance enters a
  semivariogram, never its shape, so nothing is drawn: no RNG, no seed
  inside a figure, and exact in expectation rather than approximated. A
  Monte Carlo check over 200 draws of independent per-location noise agrees
  with the closed form, and is in the suite.
  - `predict_measurements` is **not** the right source here, though it is
  the one `accuracy` uses. `likelihood.measurement_samples` adds a
  per-node *scalar* to the whole field at once, so each of its columns is a
  spatially coherent field plus a constant — exactly the right marginal per
  location, and a constant cancels out of `(y_i - y_j)`. Measured: it moves
  the fan by 2%, not by the nugget. The two figures need different things
  from the same noise, and only one of them needs it to be independent
  between locations.
  - Panels gained a `noise` key holding the per-bin correction, or None
  where the container carries no `noise_variance` (predicted with
  `include_noise=False`), in which case the fan is drawn uncorrected rather
  than refused. `residuals=True` is unaffected: it has no fan.
* **The variogram's pairs are declustered now, by default**, and this was
the larger of the two errors in that figure. Samples follow the ore, so an
experimental variogram on raw pairs describes the sampling as much as the
field. `geoml.math.geometry.declustering_weights` is the new primitive: lay
a lattice, split one vote among the samples sharing a cell, average over
shifted lattice origins so that where the lattice starts does not decide
the answer.
  - The cell size is chosen the way `declus` has always chosen it (Deutsch
  & Journel), by sweeping and keeping the departure of the declustered mean
  from the naive one — taken in absolute value, so clustering in high and
  in low values are both handled without being told which happened. Both
  ends of the sweep return the naive mean, so the extremum is interior.
  - **Checked against a known truth rather than argued.** Walker Lake ships
  its exhaustive field, so the honest variogram is knowable: its 470 samples
  have a mean of 435 against the field's 278, and their raw variogram runs
  1.4× the true one at the sill and 2.3× at the shortest lag. Declustering
  (chosen cell 22.3 units) brings the mean to 291 and the average departure
  across lags from 56% to 12%, with the sill from 89 700 to 64 400 against a
  truth of 61 800. Two tests pin exactly this, and they are the reason to
  keep the exhaustive grid in the bundled data.
  - The weights apply to the fan and the sill as well as the data curve, or
  the sides would again be estimating different things. `decluster=False`
  restores the raw pairs and a number fixes the cell; panels carry the cell
  used. The shortest lag is the one bin declustering cannot mend, since the
  closest pairs exist mostly *inside* the clusters, and the manual now says
  so where it reads the figure.
  - **`metrics.variogram_score` declusters too**, given the `coordinates`
  its pairs come from: each pair is weighted by `w_i * w_j`, and
  `compute_metrics` passes them, so the number reported beside the CRPS is
  the declustered one. Its other bias stays, deliberately. The truth carries
  the measurement noise and the realizations do not, and unlike a
  semivariogram there is no constant to add to `|difference| ** p`, so
  putting the two on one footing would mean drawing noise into the
  realizations — a seed inside a metric, and a number that moves between
  calls. The docstring now says plainly that this is an estimate which never
  reaches zero, to be read as a ranking between models on the same data,
  and points at the figure for when the size of the gap is the question.
  - **The figure carries the score**, in each panel's title as `VS = ...`,
  in both backends. It is computed on the locations the figure kept and
  with the cell the figure chose, so the number and the curves are the same
  comparison; the realizations it needs are the columns the fan already
  read, kept side by side rather than re-read, which is bounded by the pair
  budget however large the container is. Panels without a fan (`residuals`,
  nothing simulated) carry `None` and title themselves as before. The score
  is over every pair of the kept locations rather than the binned ones, so
  it does not move with `max_lag` or `direction` — it judges the ensemble,
  and the curves are what the direction changes.
  - This started from the observation that the fan's remaining shortfall
  looked like preferential sampling rather than a bad model, and it was.
  With both corrections in place the fan tracks the data across the whole
  range on Walker Lake, where before the first it sat at a sixth of the
  data's curve at short lags. The figure's verdict on that model reversed
  entirely, which is the argument for being scrupulous about what goes on
  each axis of a diagnostic: an unfair comparison convicts an innocent
  model and looks like evidence while doing it.

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
