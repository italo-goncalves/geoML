# Mesh sets

*Built 2026-09-11 (0.6.10). `geoml/data/meshsets.py`; tests in
`geoml/test/test_meshsets.py`; the measurements in
`docs/benchmarks/mesh_sets.py`.*

A block model contoured at one cut-off is a mesh; a model contoured at
every cut-off a variable declares, for the prediction and for each of its
realizations, is a set of them, and most of the questions worth asking are
about the set: the volume between two shells, whether the shells nest, how
far one realization's shell sits from another's, how much the prediction's
volume misreports what the realizations say. `MeshSet` is that set.

## The shape

`MeshSet(blocks, "Comp/Fe")` contours the column at every cut-off the
variable declares -- `cutoffs=` otherwise, and refused with neither -- and
holds the bodies as a read-only mapping: `shells[0.5]` is the body where the
prediction clears 0.5. A categorical variable gives one body per category,
keyed by name: `shells["BIF"]`. Every mesh is a closed `Solid3D`: a set of
sheets has no volumes, so `close=False` is refused.

**Realizations come first in the access, and each is a set of its own**:
`shells.simulations[4]` is realization 4 at every cut-off, a `MeshSet`
with the same keys, table, check and exports, and `shells.simulations[4][0.5]`
one of its meshes. A realization's shells are level sets of one field, so
they belong together the way the prediction's do. Two spellings were
considered and not taken: `shells[0.5, 4]` puts two axes in one key, which
`keys()` could list only one of; `shells[0.5][4]` makes `shells[0.5]` a
container of meshes, so the common case would need `.prediction`. The
realizations live in a Zarr store laid out as `prediction/0.5` and
`simulations/4/0.5`, every mesh a container `Solid3D.open` reads on its own,
and are read when asked for; the ensemble is a lazy sequence that slices.

## Limits

`limits={name: mesh}` cuts every mesh to a sheet's underneath -- a
topography -- or a body's inside; `exclude={name: mesh}` takes them away. A
sheet becomes the ground beneath it once for the whole set, as
`clip_meshes` does, so every cut is a body boolean. They are named so the
table can say what each took (`removed: topography`), in the order they
cut. All the booleans of a set are worked out by Manifold in **one frame**,
the model's rounded corner, so a body converted once -- a limit, the
prediction's shell a realization is compared against -- serves every
operation after it, and a cut that takes nothing hands the contour back
untouched rather than re-triangulated.

A cut that takes **everything** is said out loud. A limit keeps its inside,
so one around the wrong ground empties every shell, and the set used to be
built, stored and reopened with nothing in it and no word of why: on the
Tom v6 model, `get_contour('Rock/uncertainty', 1e-4)` gave a body around
the few pockets where the rock was that certain, 0.02% of the model, and as
a limit it took all three metals' shells -- the Zn shell at 5% held 10.2
million m3 before it and nothing after. A shell that had ground and has none
left after the limits now raises a warning naming the cut-offs and the share
each limit or exclusion took, from `_build` before any realization is
contoured and from `limit`/`exclude` on a set already made. A shell empty
before any cut, the field never reaching its level, is not reported.

## Realizations

Every realization is contoured at creation (`simulations=True`, or an int
for the first n, a list of numbers, or False). The work goes to forked
workers, which inherit the model, the realizations of the group being
contoured, the limits and the prediction's shells from the parent rather
than receive them -- `_BUILD`, set before the pool forks -- and hand back
the arrays of each mesh with what was measured on it. The simulations are
chunked by rows, so one realization costs a pass over every chunk, and a
group of realizations is read in one pass, as many as fit a gigabyte.

Each shell is measured as it is made -- its volume before and after the
limits, how many pieces it is in and the largest one's share, and the
volume it gains and loses against the prediction's shell -- so
`volume_dispersion()` and the figure behind it never load a mesh. One
realization's failure is recorded (`failures`, a warning at the end) rather
than allowed to throw away a build that can run for an hour.

**How many workers.** As many as the CPUs allow, eight at most, and no
more than memory holds (2026-09-11). A worker costs what one contour of
the model does, and that grows with the model: on the Tom v6 model, 6.8
million blocks, one contour took 4 to 8 GB, most of it `_cut_to_contour`'s
corner tables over the cut mesh, and the eight workers the default then
started took the 62 GB of the WSL running them down with them -- the
global OOM killer takes WSL's own processes, a hang that outlives the job.
So the set measures the parent's high-water growth over the prediction's
contours, which a worker's contour costs about as much as (5.2 GB against
4.75 GB a worker, measured), reads the memory there is, and takes as many
workers as fit 80% of it at 1.25 times that cost, with a warning when that
is fewer than the CPUs. A worker killed anyway is a `RuntimeError`: the
pool is a `ProcessPoolExecutor`, where `multiprocessing.Pool` replaced a
dead worker and waited for its task forever.

## Categories

A category's body is where it holds the ground. For the prediction that is
its `indicator_predicted` above zero -- the log-odds against its best rival,
whose zero set the contact is. For a realization, which carries each
category's latent draw, the container does not record which likelihood
drew them, so the set asks: `rule="largest"` takes the category with the
largest draw, which is `CategoricalGaussianIndicator`'s rule -- in the limit
of no variance, its probability is non-zero only where one category's draw
is positive and every other's negative, and the largest draw extends that
to where the rule says nothing -- and `rule="priority"` lets a later
category override an earlier one wherever its draw is positive,
`HierarchicalGaussianIndicator`'s rule, under which ground no category
claims is a gap. The Assen rocks were trained with the first. An ordered
rock type is read off one implicit field and is refused.

## What a set checks

In principle the shells nest and the categories do not overlap. `check()`
measures, exactly, the volume of each shell outside the one around it, and
for categories every pair's overlap and the gap left in the model's box;
`repair()` -- or `repair=True` at construction, for every realization too
-- cuts each shell to the one outside it, or each category away from the
ones before it in a priority order, and says what that took. A gap is not
filled: that would be inventing ground.

**None of the gap is the model's own edges any more** (2026-09-14). A body
closed against the box used to have its caps painted onto the faces, and
rounded the edges where two faces meet: the painted value at an edge
corner averaged one cell inside with three reflected outside, so it fell
below the level and the surface cut the corner by about half a boundary
block. Three slabs filling an 80 m box of 10 m blocks left 5% of it that
way. The ghosts past the box now carry copies of the blocks they mirror,
the surface closes a cell beyond them, and Manifold cuts the body at the
box, so the three slabs tile it exactly.

## A contour that will not close

`BlockSet3D.get_contour(close=...)` can meet its own cap edge-on. On the
Assen block model, FeO_total at 0.7 kept a layer one cell thick against
the top of the lattice, and the painted surface came back closed but with
one edge shared by four triangles; the welded-mesh fallback it is routed to
then came back *open*, at 44 edges, in small loops on the planes where the
closing ghosts change size. No winding repair settles an edge shared by
four triangles. Contoured 1e-9 lower the same shell closes, so a set
retries a contour that will not close a hair either side of its level --
from 1e-9 of the field's span up to 1e-4, lower first -- and records the
move as `nudge` in the mesh's provenance. One realization's shell (19, at
0.8) stayed open at 534 edges through every move up to 1e-5 and closed at
1e-4 lower, a few millimetres of shell. The cap itself was the cure, and
it came on 2026-09-14 (below).

Since 2026-09-11 a set does not route such a surface through the welded
mesh at all (`_contour_values(..., fallback=False)`); it heals it and moves
the level. The fallback closed none of the shells it was tried on, and on
the Tom v6 model, 6.8 million blocks, it was the costliest thing a contour
did: an Ag realization at 10 ppm fell back at every level tried, 14
million welded cells at 17 GB and two minutes an attempt, twice what a
worker is sized for, and came back open each time. Without it an attempt
costs the painted contour, 12 GB and 95 s there. That realization still
closes at no level up to 1e-5 either side -- the cap again -- so it will be
recorded as a failure, after 13 attempts and about twenty minutes.

Since 2026-09-13 the edge is split before the level is moved
(`math.geometry.split_touching_edges`). Where two pieces touch along it,
the four triangles alternate in the direction they walk it; each is paired
with its neighbour across a wedge of inside, and the vertices get a copy for
every piece meeting there, so the layer and the cap touch rather than share.
Counted on Assen, every realization contoured at its own level only
(`docs/benchmarks/contour_stages.py`): 16 shells would have been retried and
8 still are. Those are the cap folding flat onto itself -- two of the four
triangles lying in the box face on the same side of the edge, walking it
both ways -- which is no touch and is left alone; the cure is the cap's,
drawn as it is through lattice points valued exactly at the level. Each of
the eight the split closed sits between the bodies a ten-thousandth of the
span either side, and matches the retried body to four parts in a million
where the retry moved a billionth or a hundred-millionth; where it had to
move a ten-thousandth, as for realization 19 at 0.8 and Limestone's 3, it
returned that neighbour's body, 0.07-0.08% larger
(`docs/benchmarks/contour_split_neighbours.py`). The split is sometimes the
better answer as well as the cheaper one: on a small model
where the retry moved the level a billionth of the span, it returned a body
1.1% smaller than the bodies a millionth either side, and the split body
sits between them.

Since 2026-09-14 the cap is not drawn at all: the ghosts carry copies of
the blocks they mirror, the surface closes a cell past them, and Manifold
cuts the body at the box. Nothing lies on a box face at the level any
more, so nothing can fold flat there. The same census: none of the 100
FeO_total shells or the 150 rock bodies needs a retry, and the 25
FeO_total realizations took 1198 s in one process instead of 1395. The
nudges stay, for whatever else will not close.

## A band that touches itself

Two shells closed against the same face of the model have caps on the same
plane, so the band between them touches itself along the box -- every band
of the Assen FeO_total set did. Manifold's answer is a valid manifold, the
touch held as vertices at one position, and geoML's welding merged them
into edges four triangles share: the band came back a `Mesh3D` with no
volume. The fix went into the booleans themselves (`_from_manifold`), since
any difference meeting itself along an edge had the same trouble: each copy
of such a vertex moves a hundred-thousandth of a unit into its own side.

## Reports and new sets

`table(density=, grade=)` gives each cut-off's volume, the band from it to
the next, what each limit took, the pieces, the crossing, and the volume of
the blocks on the kept side of the cut-off for comparison with the raw
contour; with a density or a grade, each band's tonnage, mean grade and
metal, and the P10 to P90 of the metal with every realization's grade
filling the prediction's bands. A block the surface passes through counts
the share of its sub-blocks inside -- and only those blocks are asked,
since a block farther from the surface than half its own diagonal lies
wholly on one side (the shortcut is pinned exact against every sub-block).
`volume_dispersion()`, `connectivity()`, `spacing()`, `compare(other)` and
`section(axis, value)` answer the rest, and `probability(blocks, path,
cutoff, levels=)` builds the set of bodies where a cut-off is cleared with
given probability, from the realizations a band of rows at a time.

`clip`, `exclude`, `simplify` (nested again afterwards), `drop_pieces` and
`repair` return new sets of the prediction's meshes; the realizations are
not carried, since cutting them means building the set again with the
limit. `assign(container, name, fraction=)` writes the band or body each
location falls in as a coded metadata column, `crossed_by(blocks)` the
blocks any mesh passes through. `to_zarr`/`open`, `to_geoh5(folder=)` (one
Surface per mesh, the realizations asked for in `simulations/<n>`),
`export_dxf` (a layer per mesh), `as_pyvista` and `plot` take it out.
Figures: `volume_dispersion`, `connectivity` and `section`, in both
backends.

Every mesh a contour makes now says what it was made from -- `provenance`,
the column, the level, the side it closes on and its budgets, and for a
set's meshes the limits and the realization -- carried by `to_zarr` and
into geoh5 metadata. The Assen `Uncertainty.zarr` that prompted it is a
`Solid3D` with empty metadata.

## Measured on Assen

`docs/benchmarks/mesh_sets.py`, 2026-09-11: the Assen block model, 908 237
blocks and 25 realizations per variable, eight workers on 32 CPUs. The
results are in `docs/benchmarks/figures/mesh_sets.txt`.

| | FeO_total at 0.6, 0.7, 0.8, 0.85 | Simple Rock, six bodies |
|---|---|---|
| The prediction's meshes | 128 s | in the next row |
| Every realization besides, eight workers | 800 s | 1024 s |
| One realization, in one process | 59 s | |
| Peak memory, the set / its largest worker | 5.5 / 6.8 GB | 5.5 / 9.1 GB |
| Meshes retried off their level | 7 of 104 | 8 of 156 |
| Meshes that would not close even so | none | 1 of 156 |

**The gate: the shells nest.** The prediction's four Fe shells cross each
other by 1.8e-15 m³ as contoured, and no realization's set by more than
7e-15 m³; simplified to 1 m and nested again, the nesting took nothing
back. `repair` stays off by default: on this model it has nothing to do.
Two of the four shells were returned whole by `simplify`, the weakness the
roadmap already recorded at 2 m, then at 1 m -- a quadric pre-pass that
came back open, every cut starting from it, dropped since 2026-09-14.

**The volumes the prediction misreports.** FeO_total's median is 0.73:

| Cut-off | Prediction | Realizations, P10 / P50 / P90 | Rank | Gained | Lost |
|---|---|---|---|---|---|
| 0.60 | 9.72 Mm³ | 8.33 / 9.33 / 9.97 | 0.76 | 0.08 | 0.12 |
| 0.70 | 6.43 Mm³ | 6.17 / 6.86 / 7.61 | 0.24 | 0.19 | 0.12 |
| 0.80 | 3.68 Mm³ | 3.51 / 4.03 / 4.85 | 0.24 | 0.32 | 0.20 |
| 0.85 | 1.42 Mm³ | 1.65 / 2.18 / 3.00 | 0.08 | 0.87 | 0.26 |

Below the median the prediction's shell is larger than three realizations
in four; above it, smaller -- at 0.85 smaller than 23 of the 25, and 35%
under the median. The realizations also put the ground elsewhere: at 0.85
a realization's shell holds, outside the prediction's, 87% of the
prediction's volume. And they fragment: the prediction's shell at 0.85 is
five pieces, 63% of it in the largest; a realization's, ten.

**The rocks.** The prediction's bodies overlap by 0 to 217 m³ a pair,
about 1 100 m³ in all -- 0.008% of the model -- and leave 12 900 m³, 0.1%,
uncovered, most of it the box's rounded edges (0.053% since the box cuts
the bodies). Their volumes against the
realizations':

| Rock | Prediction | Realizations, P10 / P50 / P90 | Rank |
|---|---|---|---|
| BIF | 0.82 Mm³ | 0.97 / 1.27 / 1.60 | 0.00 |
| Calcitic Hematite | 3.59 Mm³ | 2.76 / 3.32 / 3.61 | 0.88 |
| Diabase | 2.01 Mm³ | 1.60 / 2.20 / 2.49 | 0.40 |
| Hematite | 2.46 Mm³ | 2.27 / 2.68 / 3.25 | 0.24 |
| Limestone | 2.87 Mm³ | 1.67 / 2.28 / 2.74 | 0.92 |
| Shale | 1.42 Mm³ | 0.94 / 1.50 / 2.02 | 0.44 |

Every realization holds more BIF than the prediction, by a third at the
median. The prediction's body is where BIF is the most probable rock, and
a rock that is often likely and seldom the most likely loses ground there
that its realizations keep. **A realization's bodies did not tile the
model:** three measured overlapped by 141 000 to 162 000 m³, about 1.1% of
the model, and left 0.75 to 0.84% uncovered. Each category was contoured on
its own field -- its draw against the best of the others, taken block by
block -- and where the draws are rough from block to block, the corner
values two neighbours' fields paint disagree about where their contact
runs. The prediction's fields are smooth, and agree to 0.008%.

Since 2026-09-14 a realization's categories come from one cut and one
paint of all its draws (`BlockSet3D._contour_fields`): the draws are
averaged onto the corners first and each category's field read off those
means, so along a contact between two categories the one's field is the
other's negated, and their surfaces pass through the same points. Measured
on realizations 0, 12 and 24 (`docs/benchmarks/category_partition.py`):

| | Overlap, of the model | Gap | A realization |
|---|---|---|---|
| Each category on its own | 1.07-1.21% | 0.73-0.77% | 72 s |
| All at once | 0.0000% | 0.089-0.095% | 54 s |
| The prediction's bodies | 0.008% | 0.053% | |

The gap left is where three categories meet inside one cell of the
painted lattice: each field's marching-cubes piece cuts off the corners it
wins and none claims the middle. Taking a winner's margin against its
neighbours' winners rather than its runner-up was tried on a plain
lattice and closed a tenth of it (15.8% to 14.0% on rough draws), so the
cause is the cell and not the edge crossings; a multi-material contour,
each interface drawn once, would close the rest (roadmap). Six fields at
once peak at 2.06 GB where one peaks at 0.92, so the pool is sized to the
first realization, made in the parent and kept.

**Workers scaled 2.2 times on eight**: 25 realizations in 672 s past the
prediction, 27 s each, against 59 s in one process. Measured on
2026-09-14 (`docs/benchmarks/mesh_set_workers.py`, eight realizations of
FeO_total, after the leaner contours): 45.8, 30.4, 18.0 and 12.8 s a
realization on 1, 2, 4 and 8 workers, 3.6 times at eight. The parent's
reads and writes are a few seconds of it, and VTK runs one thread here. A
forked worker runs Manifold on one thread, the parent having started its
thread pool, so a task is one core's work, 93 s against 45 s on two cores
in one process; and more workers do not help, 12, 24 of them giving 10.8
and 11.2 s a realization with every stage slower in step, the 16-core,
two-channel machine out of memory bandwidth and then cores. OpenBLAS keeps
a spinning thread per CPU in every process, 80 s of a task's 173 s of
CPU; the workers now hold it to one.
