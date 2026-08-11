# Naming, reaching and exporting a variable's parts — analysis and plan

**Status: executed, 0.5.9** — thirteen commits, one per step of §7, each
tested against the files it reached and the full suite at the end. Three
deviations from the plan as written, all recorded here because the reasoning
changed, not just the order:

- **Step 10 folded into step 6's persistence commit, with no shim.** The plan
  paired the Zarr key alignment with the `_Category` format bump and required
  a read shim plus a fixture store. Before the work started the user ruled
  that old stores need not reopen ("everything will be refreshed"), which
  made the shim dead weight; instead `open` now *checks*
  `_GEOML_ZARR_FORMAT` — it never did — and refuses a mismatched store
  outright rather than half-loading it.
- **The `split_shares` fold ran after step 7, not inside step 6.** Folding it
  first would have meant one implementation handling both the scalar and the
  dict shape of a category's `divided`, deleted a commit later when the dict
  promotion landed. Order within the plan was by blast radius; these two
  steps interact, and doing 7 first made 6's last piece three lines.
- **Two repairs surfaced by the folds were taken as part of them.**
  `set_coordinates` was six hand-written lists with four missing columns and
  became a walk (step 4); subsetting a `BinaryVariable` had never cut its
  `probability` — the override wrote the subset into a dead `average`
  attribute — and the generic `_subset_into` fixed it by construction
  (step 6).

The section below is the analysis as it stood before the work, on `master`
at 0.5.8, kept because the bug table and the prior art are why the design is
what it is.

A container holds variables, a variable holds other variables or attributes,
and an attribute holds one array per location. That tree is deliberate — it is
what stopped the code being written once per variable type — but it has never
been given a way to *say* where something is. The result is four different
spellings of the same quantity, two near-duplicate resolvers, and a family of
bugs that has now been fixed seven times without the cause being touched.

The proposal is a path: `points.get("assay/Zn/noise_variance")`. The argument
for it is not novelty. It is that **the Zarr store already writes exactly that
string**, so the scheme is not being invented here, only promoted out of the
persistence layer into the API and the exports.

---

## 1. The tree that exists today

| Node | Child nodes | Its own leaves | Node-level facts |
|---|---|---|---|
| `_SpatialData` | `variables`, `metadata` | `coordinates` | `coordinate_labels`, `n_data` |
| `ContinuousVariable` | — | `measurements`, `latent_mean`, `latent_variance`, `prediction`, `dispersion`, `noise_variance`, `simulations`, and the dicts `quantiles`, `probabilities`, `proportions`, `divided` | `cutoffs` |
| `VectorVariable` | components (`ContinuousVariable`) | `uncertainty` | `labels` |
| `CompositionalVariable` | components (`_Component`) | `uncertainty` | `labels` |
| `_Component` | — | as `ContinuousVariable`, minus the latent pair (`None` by design) | `cutoffs` |
| `RockTypeVariable` | components (`_Category`) | `predicted`, `entropy`, `uncertainty`, `measurements_a`, `measurements_b`, `boundary` | `labels` |
| `_Category` | — | `probability`, `indicator`, `indicator_mean`, `indicator_variance`, `indicator_predicted`, `proportion`, `divided`, `simulations` | — |
| `OrderedRockType` | components | the above plus `implicit_values` | `labels` |
| `BinaryVariable` | — | `indicator`, `measurements`, `weights`, `predicted`, `probability`, `entropy`, `uncertainty`, `latent_mean`, `latent_variance` | `labels` |

Two asymmetries in that table are the source of most of the trouble, and both
are defensible in isolation:

- **`divided` is a dict on a grade and a single column on a category.** A grade
  is judged against several cut-offs, a category against one contact. Nothing
  outside can therefore `getattr` it; `_Variable.split_shares` (`data.py:804`)
  exists to ask instead, and is the only place in the codebase that has this
  right.
- **A component is a variable, but not the same kind of variable as its
  parent.** `_Component` and `_Category` are different classes with different
  leaves, and `VectorVariable`'s components are plain `ContinuousVariable`s —
  three cases where the code says "component".

### 1.1 The four spellings

The same number, `noise_variance` on component `Zn` of variable `assay`:

| Where | Spelling | Built by |
|---|---|---|
| Zarr | `assay/Zn/noise_variance` | `_save_attr`, `data.py:994` |
| `as_pyvista`, cells | `assay - Zn - noise_variance` | `fill_pyvista_cells`, `data.py:826` |
| `as_pyvista`, cube/points/blocks | `assay - Zn - noise variance` | the per-type fills, hand-written |
| `as_data_frame` | `assay_Zn_noise_variance` | `VectorVariable.as_data_frame` prefixing |
| `plots.prepare` | `assay.noise_variance` — two levels only | `uncertainty_values` |

Even Zarr is not internally consistent: components become path segments, but a
parameterized leaf is flattened into the name (`quantile_1.5`,
`proportion_1.5`) rather than nested.

### 1.2 The decision already taken

`_zarr_save` (`data.py:1022`) composes keys by recursion, passing
`prefix + "/" + component_name` down and appending the role at the leaf, with
metadata under a reserved `_metadata/` root. A saved container is already a
POSIX-style tree:

```
assay/Zn/prediction        assay/Zn/simulations     assay/uncertainty
assay/Zn/noise_variance    assay/Zn/quantile_1.5    _metadata/HOLEID
```

So the question is not whether geoML should have a path scheme. It has one.
The question is why it stops at the store.

---

## 2. What the absence costs

Each of the following was one hand-written list of columns falling out of date
with another. None was a hard bug to fix; the point is that they are one bug.

| # | What was dropped | Where | Status |
|---|---|---|---|
| 1 | component cut-offs | `carry_to` | fixed 0.5.7 |
| 2 | component cut-offs | `VectorVariable.from_variable` | fixed 0.5.7 |
| 3 | `divided` reached by `getattr` | `block_shares` | fixed 0.5.7 (`split_shares`) |
| 4 | `dispersion`, `noise_variance` | `_Component.update` | fixed 0.5.8 |
| 5 | every column after `prediction` | `_Component.__getitem__` | fixed 0.5.8 |
| 6 | `dispersion`, `noise_variance` | `_Component.as_data_frame` | fixed 0.5.8 |
| 7 | component cut-offs | `CompositionalVariable.from_variable` | **open** |

Number 7 is worth stating on its own, because it is not only an export
nuisance: a cut-off declared on a composition never reaches a block model, so
`proportions`/`divided` are never computed for it and **a block model cannot be
refined on an assay grade at all**.

Two more of the same shape, both open:

- `proportions` and `divided` reach **no** pyvista export. The block-set path
  walks `_ZARR_ATTRS` and never looks at the dicts; the grid path names its
  columns by hand and stops before them; `_Category`'s copy stops at
  `probability`. Net: a block set exports the categorical shares and no
  quantiles, a grid exports quantiles and no shares.
- `data.py` carries **fourteen** `as_data_frame` definitions and **twenty-five**
  `fill_pyvista_*` definitions. Every column of every variable type is written
  out by hand between two and five times.

The failure mode is always the same and it is always silent: the column exists,
is allocated, is filled, is persisted — and one of the five lists does not
mention it, so it is simply absent from wherever that list is read.

---

## 3. Prior art

Six systems solve a recognisably identical problem. What each one settles:

**h5py and Zarr groups.** POSIX paths in one namespace, `visit()`/`visititems()`
to traverse, `.tree()` to display, and — the part that matters most here — the
split between *datasets* and *`.attrs`*. Arrays hang in the path; scalars that
describe a node sit beside it. That is the answer to `cutoffs` and `labels`:
they are node attributes, not children, and the reason they keep being dropped
is that nothing enumerates them the way `_ZARR_ATTRS` at least enumerates the
arrays.

**xarray `DataTree`** (in xarray core since late 2024, already a dependency). A
tree of Datasets addressed by absolute or relative path, `.subtree` iteration,
and coordinates *inherited* downward. The inheritance maps directly: geoML's
`coordinates` belong to the container and every leaf beneath has the same
length, which is an invariant currently enforced by each attribute holding a
back-reference.

**PyTorch `nn.Module`** — the closest structural analogue, and the one to steal
from. A tree of objects that must flatten for persistence and re-inflate:
`named_parameters()` yields `("encoder.layer.0.weight", tensor)`,
`get_submodule("encoder.layer.0")` resolves a path, `state_dict()` is a flat
path→array mapping, and `__repr__` prints the tree. **One traversal derives
lookup, persistence and printing.** That is the whole of §2 in a sentence.

**pandas MultiIndex.** The flat/nested tension. A DataFrame can keep the
structure as tuple columns; a CSV cannot.

**JSON Pointer (RFC 6901).** Escaping: `~1` for a literal `/`. Relevant because
labels are user data — a rock type called `Qtz/Fsp` breaks a naive split.

**CF/netCDF conventions.** A flat variable name plus attributes carrying the
meaning (`standard_name`, `cell_methods`). The lesson for exports: use the
target's attribute mechanism where there is one. VTK arrays have none, so the
name has to carry everything.

**ArviZ `InferenceData`.** Groups by *kind* — `posterior`, `observed_data`,
`log_likelihood` — rather than by variable. This is the one genuine
alternative to what follows, and §5 rejects it.

---

## 4. The proposal

### 4.1 Grammar

`/`, absolute from the container, with the roots the Zarr layout already uses:
a bare first segment is a variable, `_metadata/…` is a metadata column,
`coordinates` is the coordinates.

A small `VariablePath` value object, in the shape of `PurePosixPath` — `/`
composition, `.parts`, `.parent`, `.name` — with plain strings accepted
everywhere a path is taken. A label containing `/` is refused at construction
rather than escaped: the escape is cheap to write and expensive to read, and
`Qtz/Fsp` is a naming mistake worth catching early. (If a real database forces
one, JSON Pointer's `~1` is the convention to adopt.)

**The container is a node.** `get("")` returns it, `walk()` starts there, and
a variable walked on its own uses the same code with its own name as the root.
That is what lets one traversal serve `container.to_zarr` and
`variable._zarr_save` instead of the two that exist now.

### 4.2 `get()`, not `__getitem__`

`PointData.__getitem__` already takes a boolean mask and returns a subset
container. `points["assay/Zn"]` would be a trap for both readers and callers.
The path API is `get`, `select`, `walk`, `leaves` — all new names, none
overloaded.

### 4.3 Parameterized leaves

Three forms are available for a quantile:

```python
points.get("assay/Zn/quantiles", 1.5)     # (a) two arguments
points.get("assay/Zn/quantiles/1.5")      # (b) one path
points.get("assay/Zn/quantiles[1.5]")     # (c) bracketed
```

**(b) is canonical, (a) is kept as sugar.** The reason is that one path naming
exactly one array is what lets globbing, export rendering and Zarr keys line
up; a two-argument form cannot appear inside a glob pattern or a column name,
which means every one of those places would need a special case for exactly the
four dict-valued families. Float segments resolve by `float(segment)`, so
`1.50` and `1.5` land on the same key, and formatting on the way out uses
Python's shortest round-trip repr, which is what `str(p)` already gives the
Zarr key.

The Zarr layout can keep `quantile_1.5` for now. The reconstruction metadata
records each key explicitly, so the on-disk spelling is an implementation
detail rather than a contract — aligning it is a format change, and §7 now has
one to ride along with.

### 4.3.1 A single realization

`assay/Zn/simulations/7` addresses one realization by the same rule, and it
**must not be implemented as `np.asarray(store)[:, 7]`** — that is the line
that killed a session on a 280 GB model. `store[:, 7]` on a Zarr backend
returns `(n,)` floats, so the peak memory is one column whatever `n_sim` is,
and the result is wrapped as an `_Attribute` so `.smooth()`, `.as_cube()` and
`.get_contour()` work on it like any other leaf. `variable.simulation(i)`
already does exactly this and stays as the shorthand.

What the path cannot do is make it *cheap*. Chunking splits the location axis
only, so every chunk holds all `n_sim` columns and reading one decompresses
the lot: measured on `(2M, 40)`, one realization is 0.104 s against 0.008 s for
one row band. Chunking per simulation would make it 0.013 s and cost **5.8x to
64x on every prediction write**, which is why the layout is what it is and is
not up for revision. So the rule the docs already give stands, and the path
does not soften it: **when the reduction is over locations, ask for a row band,
not a realization.** `simulations/7` is for looking at one map, not for
building a statistic.

### 4.4 Arrays and node facts are different things

Following h5py: a node has **leaves** (one array per location) and **attrs**
(scalars describing the node). `cutoffs`, `labels`, `length` and `name` are
attrs. Declaring them per class, the way `_ZARR_ATTRS` declares the arrays, is
what makes `carry_to`, `from_variable` and the persistence round trip copy them
wholesale — and is the fix for bugs 1, 2 and 7 as a class rather than one at a
time.

### 4.5 What `get` returns

The node — an `_Attribute`, a variable, or the container — as h5py returns a
Dataset rather than its contents. That keeps `.smooth()`, `.as_cube()`,
`.get_contour()`, `.draw_*()` reachable from a path. A `values(path)` shortcut
returns the NumPy array for the common case.

### 4.6 One traversal

`walk()` yields `(path, node)`; `leaves()` yields `(path, attribute)`. Defined
once per node kind — which is the only per-class code the design needs — every
list that is written by hand today becomes a fold over it:

| Written by hand today | Derived from `leaves()` |
|---|---|
| `_ZARR_ATTRS` + `_zarr_save` + `_zarr_load` | write each leaf at its path, each node's attrs beside it |
| fourteen `as_data_frame` | one column per leaf, named `render(path, "flat")` |
| twenty-five `fill_pyvista_*` | one array per leaf, named `render(path, "pretty")` |
| `carry_to` / `_carry_into` / `__getitem__` / `copy_to` | subset each leaf, copy each node's attrs |
| `split_shares` | the leaves whose role is `divided` |
| `get_contour(name)` | round-trips by construction: it renders the path it was given |

Every bug in §2 is one of those rows being out of date.

### 4.7 Rendering, and what cannot be inverted

One function, three styles:

| Style | Example | For |
|---|---|---|
| `path` | `assay/Zn/noise_variance` | Zarr, `get`, anything internal |
| `flat` | `assay_Zn_noise_variance` | DataFrame columns, CSV, mining software |
| `pretty` | `assay - Zn - noise variance` | pyvista/ParaView array names |

`flat` is **not invertible**: a variable called `noise` with a component
`variance` renders identically to a leaf called `noise_variance`. So nothing
may parse a rendered name back into a path. Anything that needs the path must
be *given* the path — which is what `BlockSet3D.get_contour` now does, after
it broke the day the pyvista labels changed. Where the target format has an
attribute mechanism, the mapping can travel with the file: pyvista's
`field_data` can carry a JSON path↔name table, which makes a round trip
possible for VTK without making it possible in general.

### 4.7.1 Collisions belong to the flattening, not to the tree

A path cannot collide: `/` is refused inside a name (§4.1), so two distinct
paths are two distinct strings. The collision is created by the join, and only
`flat` creates it readily — `_` appears in nearly every role name, so
`noise/variance` and `noise_variance` land together. `pretty` can collide too,
but only if a label contains the literal `" - "`.

So deduplication is a property of *rendering a set of paths into a flat
namespace*, not of any single path, and it lives in one place:

```python
render_all(paths, style="flat")   ->  {path: name}, all names distinct
```

The rule, which has to be deterministic or an export changes shape between
runs: render every path; group the ones that agree; **sort each colliding group
by its path** and leave the first name as it is, suffixing the rest `_2`,
`_3`, …; warn once per export, naming every path in the group and the name each
received. Sorting by path rather than by traversal order is what makes it
stable — a container walked in a different order renders identically.

Two consequences worth stating rather than discovering:

- **Only the colliding group is affected.** Every other column keeps its
  natural name, which is why suffixing the whole group was rejected: it
  penalizes the innocent column to spare the pathological one.
- **Adding a variable can rename a column inside a colliding group**, since
  which path sorts first depends on what else is present. That is unavoidable
  when the namespace is genuinely ambiguous, and it is exactly why the warning
  exists rather than a silent fix.

The `path` style never needs this, which is another reason it is the one thing
stored and passed around internally.

### 4.8 Selection

`select(pattern)` returns `{path: node}` in traversal order, matching
segment-wise against the full path:

| Pattern | Matches |
|---|---|
| `assay/Zn/prediction` | exactly that |
| `assay/*/prediction` | one segment — every component of `assay`, not deeper |
| `assay/**` | `assay` and everything beneath it |
| `**/prediction` | every prediction anywhere in the tree |
| `**/quantiles/*` | every quantile of every grade |

`*` does not cross `/`; `**` matches zero or more segments, so
`assay/**/prediction` finds both `assay/prediction` and `assay/Zn/prediction`.
Matching is case-sensitive, because labels are.

**Patterns only, with one exception.** A `role=` keyword was considered and
rejected: `role="prediction"` is exactly `**/prediction`, and a second way to
say the same thing is the disease this document treats. The exception is
`filled=`, which no pattern can express and which every export needs — the
all-NaN guard applied by hand in a dozen places today. So the whole surface is
`select(pattern="**", filled=None)`.

**`**` does not expand a realization axis.** The leaf is
`assay/Zn/simulations`, the `(n, n_sim)` array; `assay/Zn/simulations/7` is an
explicit sub-address (§4.3.1) reachable by `get` or by an explicit
`**/simulations/*`, never produced by a bare `**`. Without that rule a default
export of a 100-realization model would emit 100 arrays per variable, and the
existing `simulations=` selector — which says *how many* to expand — would have
nothing to attach to. With it, `include=` and `simulations=` compose cleanly:
the pattern chooses which leaves, the selector chooses how far the realization
axis is unrolled.

### 4.9 The tree, printed

```
BlockSet3D — 8243 blocks, levels 1-3
├── assay                CompositionalVariable  labels=[Ag, Pb, Zn, rest]
│   ├── uncertainty      ● float64
│   ├── Ag               _Component             cutoffs=[1.5]
│   │   ├── prediction        ● float64
│   │   ├── dispersion        ● float64
│   │   ├── noise_variance    ○ empty
│   │   ├── quantiles/        1.5
│   │   └── simulations       ● (8243, 100)
│   └── Pb               _Component
│       └── …
├── Rock                 CategoricalVariable    labels=[Ore, Waste]
│   ├── predicted        ● <U8
│   ├── boundary         ● bool
│   └── Ore              _Category
│       ├── probability  ● float64
│       └── divided      ● float64
└── _metadata
    └── HOLEID           ● <U12
```

`●` filled, `○` allocated and empty. The empty `noise_variance` above is the
real case that took a session to diagnose from a ParaView array list; it is one
glance here.

---

## 5. Decisions taken

**Variable-first, not role-first.** ArviZ's arrangement — `prediction/assay/Zn`
— makes "every prediction" a subtree, which is genuinely attractive for
exports. It is rejected because a variable would stop being one object: every
operation that geoML actually performs on this tree (`carry_to`, subsetting,
`copy_to`, `update`, persistence, refinement) is per-variable, and a role-first
layout scatters each one across as many subtrees as there are roles. The
bulk-by-role case is served by `select("**/prediction")` instead, which costs
one pattern rather than a reorganisation.

**MultiIndex columns are optional, and off by default.** `as_data_frame` keeps
returning flat, path-rendered column names. A CSV written from a MultiIndex
frame carries several header rows, which every other piece of software then
reads as data — the export exists to leave geoML, so the default has to be the
form that survives leaving. `as_data_frame(columns="multi")` returns
`("assay", "Zn", "prediction")` tuples for the reader who is staying in pandas
and wants `df.xs("prediction", level=-1, axis=1)`.

**A realization is addressable by path.** `assay/Zn/simulations/7`, backed by a
strided read rather than a materialization (§4.3.1). The tree stays uniform —
every path names one array — and the cost of asking is documented rather than
hidden.

**The container is a node.** One traversal from the root, and a variable can be
walked standalone with the same code.

**`_Category.proportion`/`divided` become dicts**, keyed by `0.0`, and the
plural names `proportions`/`divided` match a grade's. The zero is not
arbitrary: a category's cut-off *is* zero on `ind_skew`, its log-odds against
its best rival, which is why one reduction in `likelihood.py` already serves
both kinds. After this a block's shares are the same structure whatever holds
them, `_ZARR_HAS_QUANTILES` (the flag meaning "this node has dict families")
becomes true for `_Category`, and the dict machinery is written once.

Two things this does **not** change, and both matter:

- `split_shares` stays. It is no longer papering over a type difference, but
  the *label* still differs — a grade renders `au @ 1.5` because someone
  declared that number, while a category renders the bare `rock granite`
  because the zero is an artefact of the log-odds and would read as noise in
  `block_shares()`. Uniform storage, deliberate rendering.
- `probability` stays a plain leaf. It answers a different question from
  `proportions` — how sure the model is that the whole block is granite, not
  how much of the block granite holds — and nothing about it is keyed by a
  cut-off.

It is a store-format change, so it carries a `_GEOML_ZARR_FORMAT` bump and a
read shim for the scalar form, which is why §7 pairs it with the Zarr key
alignment that was otherwise going to wait indefinitely.

**A flat namespace deduplicates itself, loudly.** `render_all` resolves
collisions deterministically and warns (§4.7.1). Not an error: the collision is
legal today, and raising would break a working script on the day it upgrades,
over a name nobody has complained about.

**`select` takes patterns and one flag.** `select(pattern="**",
filled=None)` — no `role=` keyword, because it would duplicate
`**/prediction`, and `**` deliberately does not unroll realizations (§4.8).

**`plots.prepare`'s dotted `"Variable.column"` is removed, not aliased.** The
user's call, and the right one: an alias is a second grammar to maintain for
the lifetime of the package, and this one is two levels deep in a tree that is
four. It is removed with a message that names the replacement —

> `"Elements.uncertainty" is no longer accepted; use the path "Elements/uncertainty"`

— rather than silently reinterpreted, since `.` inside a label would otherwise
make a wrong guess look like a working one. The second resolver
(`prepare._variable_or_component`, a near-copy of `data.py`'s) goes with it,
which was the original symptom that started this document.

---

## 6. What breaks

**One thing, deliberately.** `plots.prepare`'s dotted `"Variable.column"` is
removed (§5), which changes a documented argument and is the reason step 9 gets
a release of its own. Everything else is additive:
`points.variables["assay"].components["Zn"].noise_variance` keeps working —
paths are a second way in, not a replacement, and the traversal they rest on
generalizes `_ZARR_ATTRS` rather than replacing it.

Three things change visibly without breaking a call, and each is a fix:

- export column names become consistent, so a script reading
  `au - noise variance` from a grid or `assay - Zn - noise_variance` from a
  block set will need the one spelling instead;
- `as_data_frame` gains the columns the hand-written lists were missing;
- a category's shares are reached as `proportions[0.0]` rather than
  `proportion`, and reach an export for the first time.

The `flat` collision (§4.7.1) is not a new risk — it exists today, silently,
and the difference after this work is that the export says so.

---

## 7. Order of work

Nine steps, each independently useful and independently testable. The existing
suite is the check at every stage, and the rule throughout is that **no step
is allowed to change a number** — only where a number is written and what it
is called. The two that do change stored bytes (6 and 9) are deliberately
adjacent and share one format bump.

| # | Step | Touches | Why here |
|---|---|---|---|
| 1 | `walk`, `leaves`, `attrs` per node kind; `VariablePath`; `get`, `values`; container as node | new code only | nothing can be folded before there is something to fold onto |
| 2 | `select` (patterns + `filled=`) and `render`/`render_all` with the dedup warning | new code only | the two things every later step consumes; testable against a hand-built tree before anything depends on them |
| 3 | the tree `__repr__` | new code only | makes every later step reviewable by eye, and by itself answers "why is this column empty" |
| 4 | `simulations/i` as a strided read; the same fix to `_Variable.__getitem__`, which materializes today | `data.py`, `storage.py` | one line of `np.asarray(store)[item]` is both the path feature and a known memory bug |
| 5 | `cutoffs` declared an attr; attrs copied wholesale in `from_variable`/`carry_to` | `data.py` | closes open bug 7 — a composition can be refined on grade again |
| 6 | fold the five families onto `leaves()`, one commit each: `as_data_frame` → pyvista fills → `__getitem__`/`carry_to` → `split_shares` → persistence | `data.py` | ordered by blast radius, persistence last because it is the one with files on disk behind it |
| 7 | `_Category.proportion`/`divided` → dicts keyed `0.0` | `data.py`, `likelihood.py`, `models.refine` | nearly free once step 6 has generic dict handling; needs the format bump |
| 8 | include patterns in the exports replacing the per-family booleans; the `field_data` path table | `data.py`, `pyvista.py` | the shares finally reach pyvista, and `simulations=` stops being a special case |
| 9 | `plots.prepare` onto paths; **remove** the dotted form; delete the second resolver | `plots/prepare.py` | the duplicate `_variable_or_component` was the original symptom |
| 10 | Zarr keys aligned with paths (`quantiles/1.5`, not `quantile_1.5`) | `data.py` | same format bump as 7, so old stores get one read shim rather than two |

**Steps 1–5 stand alone.** They give the tree a vocabulary, a query, a picture,
remove a materialization, and close the open bug — worth doing even if nothing
after them ever happens. Steps 6–10 are where the duplication actually dies,
and each is a strict deletion: fourteen `as_data_frame` bodies become one,
twenty-five `fill_pyvista_*` become three renderers over one traversal.

**Step 9 is the only one that breaks a caller.** Everything else is additive or
internal; removing the dotted form changes a documented argument. It should
therefore land in its own release, be named in the changelog under its own
heading rather than inside a list, and carry the error message from §5 so the
fix is readable without opening the docs.

### 7.1 What steps 7 and 10 owe the reader

Both change what is on disk, so together they carry:

- one `_GEOML_ZARR_FORMAT` increment;
- a read shim that accepts a scalar `proportion`/`divided` and the
  `quantile_1.5` key spelling, so every store written up to 0.5.8 still opens.
  There is precedent — the text-attribute change in 0.5.3 re-encodes old
  stores on open, and this is the same shape of compatibility;
- a test that opens a store written before the change. That means keeping a
  small fixture store in the repository rather than generating one, since a
  generated one is written by the new code and proves nothing.

### 7.2 What step 7 will break in the tests

Named now so it is not discovered as a surprise: `test_blockset.py` reads
`component.proportion.values.to_numpy()` directly in
`test_a_category_splits_a_block_where_its_boundary_runs`, and asserts
`sorted(blocks.block_shares()) == ["rock granite", "rock schist"]`. The first
becomes `component.proportions[0.0]`; the second must **not** change, which is
the rendering decision in §5 and is worth an assertion of its own so a later
tidy-up does not "make it consistent".

---

## 8. Nothing is outstanding

Every question this document opened has been answered, and the answers are
in §5. Three things are still left to the implementation rather than settled
here, because settling them on paper would be guessing:

- **How noisy the dedup warning is on a real model.** It should fire on
  approximately nothing. If a Macpass export raises it, the naming rule is
  wrong rather than the model, and §4.7.1 is what to revisit.
- **Whether the folded `as_data_frame` is fast enough.** It walks a tree per
  call where it read a fixed list before. Expected to be lost in the noise of
  the array reads it wraps — but that is a claim, and step 6 is where to
  measure it rather than assume it.
- **Whether the pre-change fixture store (§7.1) belongs in the repository.** A
  binary in a source tree is a cost; generating it from a pinned older geoML
  is a bigger one. Decide when step 7 is written.
