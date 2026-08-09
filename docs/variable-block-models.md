# Variable block sizes in geoML — analysis, measurements, plan

Nothing in `geoml/` has been changed. Everything below is measured on the
current `master` (0.5.6), with benchmark scripts kept in the session scratchpad
(`octree_study`, `subset_study`, `contour_study`, `crack_study`,
`support_study`, `lattice_study`, `support_theory`, `disc222`), run in the WSL
`geoml` conda env.

The starting question was memory: on a real block model the interesting ground
is a small fraction of the volume, so most cells pay full price to say nothing.
The saving is large — 44x to 434x fewer cells, measured on the Macpass
drillholes — and the prediction path needs almost nothing doing to it. Two
things turned out to matter more than the memory:

- refining by proximity to data biases a grade-tonnage curve by **-40% at high
  cut-offs** (§6). The criterion has to be about heterogeneity, not about
  distance from data and not about posterior variance;
- constraining every block to an integer lattice (§4) makes conversion between
  supports exact and, more usefully, makes that bias **auditable per cut-off**
  before anyone signs a report.

---

## 1. What a block model is in the code today

`Blocks3D(Grid3D)` plus the `_blockdata` decorator (`data.py:5003`), which is
how block behaviour is added without multiple inheritance:

| Piece | Where | What it does |
|---|---|---|
| `discretization` | `_blockdata.new_init` | `[3, 3, 3]` — the sub-block lattice inside every block |
| `sub_grid` | same | `(k, n_dim)` offsets from a block centre, `k = prod(discretization)`, **shared by every block** |
| `rows_per_location` | `data.py:5058` | `k`. `predict` divides the batch size by it |

A block model's coordinates are never materialized: `_GriddedData` is its own
lazy coordinate provider (`data.py:3221`), regenerating rows from the per-axis
node vectors on demand, and `grid.coordinates is grid`. **So the cells
themselves cost nothing today.** What costs is the variables — `n_data x
(n_attributes + n_sim)` — which is what the 280 GB behind the 0.5.6 streaming
work was.

An irregular mesh must store explicit geometry, so it starts from behind. §3
shows by how little.

---

## 2. The prediction path barely cares

`VGPNetwork.predict` (`models.py:975`) asks a container for exactly six things:

```
n_dim   n_data   rows_per_location
get_batched_coordinates(batch) -> (coords, splits)
get_batched_variance(batch)
variables[v].update(batch, **output)
```

There is **no regularity assumption anywhere in it.** The GP sees coordinates
and nothing else. Confirmed by counting container references per module:

| Module | References to prediction-target containers |
|---|---|
| `kernels.py`, `transform.py`, `parameter.py`, `tftools.py`, `storage.py`, `persistence.py` | none |
| `likelihood.py` | **none** — pure tensors |
| `latent.py` | only `ip.coordinates`, on *inducing* points |
| `inducing.py` | takes training data, by design |
| `models.py` | `predict` only |

**The variational machinery does not know what a block is**, and none of it
needs to change.

### 2.1 Constant discretization keeps it that way

`_aggregate` (`likelihood.py:129`) and `_add_noise` (`likelihood.py:205`)
reshape rather than split:

```python
agg = _tf.reshape(x, _tf.concat([[n_splits, n], _tf.shape(x)[1:]], axis=0))
```

so `n = rows / n_splits` must be a whole number — every block in a batch must
fan out to the same number of sub-blocks.

**If `discretization` is held constant across levels, that condition is
satisfied for free.** `rows_per_location` is then the same at every level, so
`n_splits = len(index)` for any batch, mixed levels or not. Nothing about
batching has to change, `_batch_rows` is untouched, and `_aggregate` and
`_add_noise` keep working unmodified — which matters, because `_aggregate` has
**25 call sites** across six likelihood classes and changing its signature
would reach all of them.

The whole prediction-side change is then one factor in
`get_batched_coordinates`, scaling the unit sub-grid by each block's own size:

```python
coords = centers[:, None, :] + unit_sub_grid[None, :, :] * size[:, None, :]
```

Same single temporary, same block-major layout that `_aggregate` averages back.

`white_noise` also needs no change: it draws `k` rows indexed by sub-block
position and shares them across blocks, and `k` is now constant everywhere.

---

## 3. What would actually be saved

Measured on the real Macpass drillholes (`octree_study`): 559 holes composited
to 5 m, clustered at 900 m single linkage, largest deposit taken — 5667 samples
over a 2752 x 1260 x 1056 m domain. Octree refined to 5 m wherever data lies
within `reach` cell widths of a cell centre, walked level by level from 80 m
down.

Uniform block models over the same domain:

| block size | cells |
|---|---|
| 80 m | 7 840 |
| 40 m | 59 616 |
| 20 m | 460 782 |
| 10 m | 3 686 256 |
| **5 m** | **29 436 624** |

Octree refined to 5 m near the holes:

| refinement margin | cells | vs uniform 5 m | 100 sims |
|---|---|---|---|
| 3.0 cell widths | 661 722 | **44x fewer** | 0.5 GiB |
| 2.0 cell widths | 361 018 | 82x fewer | 0.3 GiB |
| 1.0 cell width | 112 014 | 263x fewer | 0.1 GiB |
| 0.75 cell widths | 67 844 | 434x fewer | 0.1 GiB |

Uniform 5 m at 100 simulations is 22 GiB; the most generous octree is 0.5 GiB.
Explicit geometry costs 13 bytes a block (§4.4), or 8.6 MB at 660k cells.
Against 800 bytes a block of simulations at `n_sim = 100`, it does not
register.

*(This table refines by proximity to data, which §6.1 shows is the wrong
criterion. It is kept because it bounds the geometry: the achievable saving is
of this order whatever drives the refinement.)*

---

## 4. The integer-lattice constraint

Every block's origin and size an integer multiple of a base cell. Alignment
becomes integer arithmetic, with no floating-point tolerance anywhere.

### 4.1 "Multiple of the smallest" is not enough

Two blocks of size 2 and 3 tile a 5-cell box exactly, both being multiples of
the base, and neither uniform target is reachable (`lattice_study`):

```
-> uniform 2: 1 block(s) are not a whole number of 2-cells
-> uniform 3: 1 block(s) are not a whole number of 3-cells
```

Only the base itself is reachable — the case that is memory-infeasible. The
working invariant is stronger:

> **every size present divides every larger size present** — a divisibility
> chain, not merely a common divisor.

Powers of two is the canonical instance, but it need not be cubic or binary:
`10x10x5 -> 5x5x2.5 -> 2.5x2.5x1.25` is a valid chain and matches how mining
packages sub-block. A second condition falls out of the same test: the
**bounding box** must also be a whole number of the coarsest level, or the
coarse levels do not fit.

This is why `BlockSet3D.__init__` should take `discretization` and
`max_levels` — the latter is what fixes the base cell
(`step / ratio ** max_levels`) and so makes the lattice well defined.

### 4.2 Conversion is exact and mass-conserving

An octree over 64³ base cells, sizes 1–16, brought to every uniform support —
splitting the blocks above the target and grouping those below it, since
neither direction alone suffices:

| target | blocks | full? | volume-weighted mean |
|---|---|---|---|
| source (mixed) | 26 552 | yes | −0.015204 |
| 1-cell | 262 144 | yes | −0.015204 |
| 2-cell | 32 768 | yes | −0.015204 |
| 8-cell | 512 | yes | −0.015204 |
| 32-cell | 8 | yes | −0.015204 |

Identical to every digit: **metal content is conserved under regrouping**,
which is the property a resource model lives or dies by. Grouping directly and
grouping via the base agree to `0.00e+00`.

Grouping simulations must average **matching realization indices** — column j
of a parent is the mean of column j of its children. Any other pairing invents
a correlation the model never produced. This is the same rule `block_density`
already follows in `plots/prepare.py`.

### 4.3 Splitting is replication, and should be named as such

Verified: splitting the largest block into 4096 children gives spread
`0.00e+00`. Arithmetically right and statistically loaded — it manufactures a
fine-support model with zero within-block variance, which is the §6 bias in
another costume. Grouping is honest; splitting is not a prediction and must
not be presented as one. If a fine-support answer is wanted, re-predict.

### 4.4 Checking and storing

Exact tiling cannot be checked by volume alone — a gap and an overlap of equal
size cancel — so base-cell coverage is counted:

| base lattice | blocks | check time |
|---|---|---|
| 64³ (262k) | 25 040 | 0.05 s |
| 128³ (2.1M) | 196 568 | 0.36 s |
| 256³ (16.8M) | 1 741 048 | 3.05 s |

Macpass scale (29.4M base cells) is about 5 s and 29 MB as `uint8`, 3.7 MB as
a bitset. But split and group both preserve tiling by construction, so for a
model built from a full grid **fullness is an invariant**: check on import,
maintain by construction.

Storage is `origin` (3 x int32) plus `level` (uint8) = 13 bytes a block; size
is derived from the level.

---

## 5. Contouring: solved, but not for free

`skimage.measure.marching_cubes` needs a rectangular array, so `as_cube` and
`get_contour` cannot work on an irregular mesh. VTK contours unstructured
grids directly, and pyvista is already a dependency.

`contour_study`, same analytic field, 61 nodes per axis:

| representation | cells | triangles | area | contour time |
|---|---|---|---|---|
| skimage `marching_cubes` | 226 981 | 11 096 | 4.1065 | 0.005 s |
| VTK, explicit hexahedra | 216 000 | 11 096 | 4.1065 | 0.005 s |
| VTK, two-level octree | 48 980 | 11 096 | 4.1065 | 0.003 s |
| VTK, Delaunay tetrahedra | 266 573 | 12 144 | 4.0868 | 0.006 s |

Identical triangles and identical area from the octree, on 4.4x fewer cells,
slightly faster.

### 5.1 That was the easy case

The octree above refined 4 cells either side of the surface, so the isosurface
lay wholly inside the fine region and never met a coarse/fine face. Tightening
the band until the surface crosses level transitions (`crack_study`):

| refine band | cells | triangles | area | open edges | watertight |
|---|---|---|---|---|---|
| 4 cells | 48 980 | 11 096 | 4.1065 | 0 | yes |
| 2 cells | 38 032 | 11 072 | 4.1063 | 648 | **no** |
| 1 cell | 32 656 | 9 592 | 4.0914 | 3592 | **no** |
| 0.5 cells | 29 632 | 6 608 | 4.0914 | 3344 | **no** |
| sign-change only | 36 772 | 11 096 | 4.1065 | 0 | yes |
| full fine grid | 216 000 | 11 096 | 4.1065 | 0 | yes |

The classic hanging-node crack: a coarse face meets four fine faces, each side
interpolates along its own edges, and the two disagree. The last row came out
watertight, but that is a property of a field smooth at this resolution — a
sign change on a face interior implying one at a corner — and should not be
relied on.

*A correction worth recording.* The first run of `crack_study` placed child
cells at `±step` instead of `±step/2`, so they did not tile the parent at all.
It reported open edges for every band, which read as a far worse crack problem
than there is. Those gaps were geometric, not contouring artefacts. The table
above is the corrected run.

### 5.1.1 Why the margin was the wrong fix, and what replaced it

The mitigation this section first recommended — refine with 2–4 cells of
margin — works, and is exact. It is also the wrong trade. `refine` cuts
precisely the blocks that straddle, which is zero margin, so the workflow the
feature exists for produced exactly the failure documented above: a real run
lost **13% of the surface area** to tears, 1455 of them.

Widening the refinement closes them, at a price the model pays for a display
problem. Measured on one analytic field, against the same surface drawn on a
uniform model of the finest size:

| | blocks predicted | mesh cells | tears | area error |
|---|---|---|---|---|
| no margin | 1 674 | 1 674 | 1018 | −12.8% |
| margin in the model | 8 107 | 8 107 | 0 | 0.00% |
| **cut the mesh, not the model** | **1 674** | 7 015 | **0** | **−0.01%** |

So the cut happens in `get_contour`, on the mesh handed to VTK, and the model
is left alone. Blocks the surface runs through go to the finest size the
lattice allows, one ring of neighbours included so the surface has room to
shift as its values are read at the finer size, and the interfaces end up
away from it. Nothing is predicted.

A child cannot simply copy its parent, which costs 0.8% in area — it takes the
parent's value **plus what the block's corners say about the shape running
across it**, read at the sub-block centres with the trilinear weights. Because
those weights are symmetric about the centre, the correction averages to zero
over the children, so a block's estimate stays exactly the mean of the children
standing in for it and no metal is invented. Interpolating without that anchor
is worse than copying: it inherits the corner averaging's smoothing and drifts
1.5% in mass.

Tetrahedralising into a conforming mesh — the textbook fix — was tried and
abandoned: a face collects midpoints from edge-adjacent blocks as well as
face-adjacent ones, so no simple triangulation rule makes both sides agree,
and 2:1 balancing does not rescue it. VTK's `vtkHyperTreeGridContour` is the
off-the-shelf version but is limited to a branch factor of 2 or 3, which would
rule out a discretization like `(4, 4, 2)`.

### 5.1.2 Faceting, and why the cure is not smoothing

Closed, the surface still looked blocky. Two candidates: the cut itself, or
the resolution. On a sphere the refined surface came back indistinguishable
from a uniform model at the finest size — 0.091 m against 0.089 m from the
true sphere, mean angle between neighbouring triangles 1.68° against 1.62° —
so the cut is not what does it.

What does is the reconstruction. VTK reads a **trilinear** field between block
corners: continuous, but with a crease at every face it crosses. On a field
with structure at a few blocks those creases are the faceting. On a
deliberately rough test field, against the true level set:

| | angle between triangles | off true surface | blocks predicted |
|---|---|---|---|
| as it stands (5 m) | 9.07° | 0.51 m | 19 853 |
| + Taubin smoothing, 40 iterations | 7.63° | **0.78 m** | 19 853 |
| **+ one level of supersampling** | **5.73°** | **0.22 m** | **19 853** |
| + supersample and a smoothing pass | 4.88° | 0.36 m | 19 853 |
| 2.5 m blocks, actually predicted | 5.29° | 0.19 m | 67 705 |

**Smoothing the triangles is the wrong answer.** Taubin buys a sixth of the
faceting and pays 50% more distance from the surface it is meant to be — it
makes the picture rounder by making it less true, which on an ore body is a
bad trade. It is not offered.

**Supersampling the display mesh is the right one, and it is nearly free.**
Cutting one level past the model's finest block, which costs no prediction at
all, removes more faceting than smoothing does *and* halves the distance from
the true surface. It lands on the quality of a model with 3.4 times as many
blocks in it. The reason both improve together is that averaging onto corners
again at each level composes the box filters into a rounder reconstruction, so
the extra levels are not empty interpolation — they converge on the underlying
field rather than on the trilinear caricature of it.

`get_contour(..., supersample=1)` is the default. A second smoothing pass over
the corner field is what the fourth row measures: less faceting again, but
paid for in accuracy, so it is not done either. Past one level the returns
flatten while the mesh keeps multiplying, which is why the default is one and
not more. On the end-to-end `refine` case the whole thing costs a mesh of
96 055 cells against a model of 24 711 and 0.4 s.

### 5.1.3 The jump the block itself cannot see

Supersampling left it blocky still, and this is why. A block whose own
sub-blocks agree is never marked by `needs_splitting`, and rightly so —
cutting it would not change the answer it gives. But the field can still turn
sharply inside it, and **nothing in the block says so**. That a *neighbour*
was cut twice while this block was not cut at all is the evidence, and it
lives outside the block where nothing was looking.

It costs accuracy, not just looks: a contour reads a block through its eight
corners, so a coarse block beside much finer ones lays a crude straight guess
across a long span, right where the surface runs. On the rough field, 5.0% of
blocks had a neighbour more than one level finer:

| | blocks | angle between triangles | off true surface |
|---|---|---|---|
| straddle criterion only | 11 117 | 7.24° | 0.779 m |
| **plus levelling the jumps** | **14 841** | **5.89°** | **0.266 m** |
| a whole level deeper, jumps kept | 28 393 | 5.64° | 0.758 m |
| a level deeper, jumps levelled | 53 166 | 3.68° | 0.081 m |

Levelling the jumps is **three times closer to the true surface for a third
more blocks**. Refining a whole level deeper without it buys almost nothing
for 2.6 times as many — deeper refinement widens the jumps as fast as it
narrows the blocks, which is why resolution alone does not fix this and why
the third row is the trap to avoid. `models.refine` now cuts both.

`BlockSet3D.unbalanced(gap=1)` finds them **from the fine side**: each fine
block steps one cell past each of its faces and names the coarse block
covering that point. Asking the coarse block instead is not exact — a cell of
its own size beyond its face holds blocks that do not touch it — and painting
the base lattice to be sure is the memory the model exists to avoid. Levels
were already stored per block, so the criterion needed no new bookkeeping,
only the lookup. It agrees exactly with a brute-force raster across
anisotropic and non-power-of-two discretizations, which is a test.

### 5.2 An existing safety net

`mesh3d()` picks its class from measured geometry, and `Solid3D` checks
closure at construction. A cracked contour therefore returns a `Surface3D` or
a plain `Mesh3D`, never a silently wrong `Solid3D`, and `heal()` exists for it.
The failure mode is detected, not hidden.

---

## 6. Change of support

### 6.1 The trap

Krige's relation: the variance of a block average falls as the block grows, so
a coarse block clears a high cut-off *less often* than the fine blocks filling
the same volume — not because the model is wrong, but because a coarse block
genuinely is a coarser-support estimate.

`support_study`: a lognormal field on 240³ fine cells correlated over ~6
cells; coarse blocks are exact averages of 8³ fine cells. Variance ratio
coarse/fine **0.189**. Half the domain fine and half coarse, against all-fine
over the same volume and the same field:

| cut-off | mixed tonnage | all-fine tonnage | error |
|---|---|---|---|
| 0.80 | 7 029 108 | 5 780 243 | **+21.6%** |
| 1.00 | 5 071 188 | 4 488 508 | +13.0% |
| 1.20 | 3 589 257 | 3 532 673 | +1.6% |
| 1.50 | 2 183 526 | 2 523 729 | −13.5% |
| 2.00 | 1 057 831 | 1 523 636 | **−30.6%** |
| 2.50 | 586 645 | 971 971 | **−39.6%** |

A reserve-reporting error, not a rounding error.

### 6.2 Two criteria that look right and are not

**"Refine near the data"** is the premise this document started from, and it is
backwards. Far from data means *uncertain*, which is the opposite of safe to
coarsen. Refining by proximity to data coarsens exactly the high-grade,
poorly-drilled ground where the support bias then removes 30–40% of the
tonnage.

**"Refine where posterior variance is high"** is also wrong, for a different
reason. **Refinement reduces support error, not estimation error.** Splitting a
block far from data yields eight equally uncertain blocks; the ignorance is
finer-grained, not smaller. High posterior variance is a reason to drill, not
to split.

What licenses splitting is *within-block heterogeneity relative to a decision*
— whether the children would fall on different sides of a cut-off. That is a
dispersion question, and it is a different quantity from the posterior
variance the model already reports.

### 6.3 Krige's relation is the justification, not the algorithm

Theoretically, the within-block dispersion is
`D2(v|V) = Cbar(v,v) - Cbar(V,V)`, a function of the kernel and two geometries
only, and it checks out numerically (`support_theory`) against the measured
spread of eight children's averages, to 6e-3 or better.

**It is of limited practical use, and the code should not be built on it.**
The model is Gaussian in *warped* space; under a `Spline` or `Softplus` the
block average of the grade is not the back-transform of the block average of
the latent field. An average of covariance functions describes a field nobody
reports on.

What it does supply is the licence for the whole approach: a coarse block's
value is not a blurry approximation of the fine answer, it is **the correct
answer at coarse support**. Blocks left unrefined keep a valid estimate.

### 6.4 The operational route is already in the code

`_ContinuousLikelihood.predict` (`likelihood.py:261`) back-transforms
*before* it aggregates:

```python
sims = self._add_noise(sims, seed=seed, n_splits=n_splits)   # 265
sims = _tf.map_fn(lambda x: self.warping.backward(x), sims)  # 267  <- data space
...
sims = _aggregate(sims, n_splits=n_splits)                   # 276  <- then averaged
```

Sub-blocks are converted to data space individually and then averaged, so the
block value is a mean of grades and not a warped mean. **Non-linear change of
support is already handled correctly.**

Which means a second reduction over the same axis at line 276 yields the
within-block dispersion **in data space, after warping and after noise** —
the quantity that matters, with no kernel integration anywhere. (`mu` and
`var` at 277–278 stay in latent space, as their names say.)

### 6.5 What the lattice buys: an audit

Because every block is a known number of base cells, the tonnage resting on
coarse support is computable, and more sharply the tonnage on coarse blocks
*near a cut-off* — a coarse block far above or below one being insensitive to
its support:

| cut-off | tonnage on coarse blocks | coarse **and** within 0.5 of the cut-off |
|---|---|---|
| 0.5 | 77.8% | **68.5%** |
| 1.0 | 29.7% | 29.7% |
| 1.5 | 0.0% | **0.0%** |
| 2.0 | 0.0% | **0.0%** |

At cut-offs 1.5 and 2.0 the report is provably support-clean; at 0.5 it is
provably not. §6.1 says "this may be biased and we cannot tell". The lattice
turns that into a per-cut-off guarantee, computable before the report is
signed. This is the strongest argument for the constraint — stronger than
convertibility.

---

## 7. The two-pass workflow

Predict coarse, decide what needs subdividing, predict only that.

### 7.1 Only one of the two regimes saves anything

**Regime (a), fixed sub-block count.** Every block gets the same
`discretization` regardless of size. Sub-block *spacing* then differs by level,
so a parent's quadrature points are not the union of its children's.

**Regime (b), fixed sub-block spacing.** A 2x block carries 8x the sub-blocks
and children's points partition the parent's exactly. Pathwise consistent.

Cost in GP rows evaluated, `p` the refined fraction (`support_theory`):

| refined fraction | coarse + refine | direct fine | saving |
|---|---|---|---|
| 2% | 1.16 | 8.00 | **6.90x** |
| 5% | 1.40 | 8.00 | 5.71x |
| 10% | 1.80 | 8.00 | **4.44x** |
| 25% | 3.00 | 8.00 | 2.67x |
| 100% | 9.00 | 8.00 | 0.89x |
| *regime (b), any p* | 8.80 | 8.00 | **0.91x** |

Regime (b) is theoretically clean and **saves nothing**: the coarse pass alone
costs the whole fine model, and then refinement is paid on top. Regime (a) is
therefore the design, and exact nesting is given up deliberately.

### 7.2 What regime (a) costs, and why 2x2x2 is still the default

Quadrature error in the block variance `Cbar(V,V)`, Gaussian kernel
(`disc222`):

| L / range | exact | 2x2x2 | 3x3x3 | 4x4x4 | 5x5x5 |
|---|---|---|---|---|---|
| 0.10 | 0.98528 | +0.4% | +0.2% | +0.1% | +0.0% |
| 0.25 | 0.91301 | +2.2% | +0.9% | +0.5% | +0.3% |
| 0.50 | 0.71029 | +7.7% | +3.1% | +1.6% | +0.9% |
| 1.00 | 0.33319 | +19.7% | +7.6% | +3.9% | +2.2% |
| 2.00 | 0.07935 | **+82.2%** | +16.0% | +7.2% | +4.0% |

2x2x2 is the **least** accurate of the four, not the most. Two things make it
the right default anyway.

**The error is self-correcting.** It always *overstates* block variance — a
`k^3` quadrature samples the interior and misses the corners where covariance
is lowest — and most severely at large L/range. An overstated variance
triggers a split, and the children sit at half the L/range where the estimate
is far better. The error is largest exactly where it causes the refinement
that removes it.

**And it wins on cost while over-splitting.** At `k(1 + 8p)` rows: 2x2x2
splitting 15% costs `8 x 2.2 = 17.6`; 3x3x3 splitting 10% costs
`27 x 1.8 = 48.6`. 2x2x2 is 2.8x cheaper even assuming it splits half again as
many blocks.

The same conservatism appears twice more. With fixed `k`, block-averaged noise
variance is `sigma0^2 / k` whatever the block's size, though a larger block
should average out more — again an overstatement, again on coarse blocks. And
with binary refinement the 8 sub-blocks sit at the children's *centres*, so
the criterion reads point values, which vary more than block averages. All
three errors push toward splitting too much rather than too little.

### 7.3 The splitting criterion

Every likelihood answers one question: *what does a value have to cross for
the decision to change?* A grade crosses the cut-offs someone declared; an
indicator crosses zero, `ind_skew` being a category's log-odds against its
best rival, so that the contact between two categories is its zero level set.
One reduction then serves both, and the categorical case needs no special
handling — it simply has no realization axis.

- **Continuous with cut-offs**: `cutoffs` declared on the variable and carried
  to whatever is predicted from it. `cutoffs = None` is the opt-out — a
  composition's rest component takes no part in any decision.
- **Categorical**: the crossing is always zero, so nothing is declared;
  opting out belongs to `split_on`.
- **Geometric**: already built. `assign_from_surface(..., fraction=...)` gives
  the share of a block's sub-blocks below a sheet, and a fraction strictly
  between 0 and 1 means the surface cuts the block.
- **Cap**: `max_levels`.

A union over the participating variables, conservative on purpose: one
variable demanding a split is enough.

#### The criterion this section first specified was wrong

It said: split when the proportion of the block above the cut-off, over
sub-blocks *and* realizations, lies strictly inside `(eps, 1 - eps)`. Built
that way it marked **239 of 256** blocks, then 1766, then 13 460 — refining
almost everything, which is the opposite of the point.

The reason is the distinction §6.2 of this document already draws and the
formula walks straight past. Averaging over sub-blocks and realizations
together mixes two different things back into one number:

- realizations either side of a cut-off are **the model not knowing**, and no
  amount of cutting will settle it — the answer to that is another drillhole;
- sub-blocks either side **within one realization** are two answers inside one
  block, and cutting is exactly what separates them.

Only the second licenses a split. So each realization is judged on its own —
*is this block divided, in this realization?* — and only then averaged over
realizations. That is `likelihood._divided`, against
`likelihood._proportions` which keeps the first reading.

Both are stored, because they answer different questions. `proportions[c]` is
the recoverable share of the block, and per category for a categorical
variable it is the partial-block domaining number — worth having whether or
not anything is ever refined. `divided[c]` is the criterion.

On a compact ore body in a barren domain, cut-off 1.0, three passes from a
256-block start: **4309 blocks against 131 072** for the equivalent uniform
model, 30x fewer, with the level-0 blocks at grade -0.17 to 0.66 and the
finest spanning the cut-off.

**The criterion reads the noise-free field.** Noise is the part of a block's
spread that cutting cannot resolve, so a block straddling a cut-off only on
account of it would be cut for nothing; the same argument makes `dispersion`
noise-free, that being a statement about the ground rather than about the
measurement. The predictions themselves still carry noise as `include_noise`
asks. Verified: proportions, `divided` and dispersion all agree to `0.0e+00`
between a noisy and a noise-free prediction, while the simulations differ.

**The test is direction-agnostic.** A cut-off sometimes keeps what is below it
— a contaminant limit — but a block is divided by the cut-off either way.
Direction matters for reporting, never for the decision, so the proportion is
stored in one convention (at-or-below, matching `row_cdf` and
`reset_probabilities`) and the reporting layer flips it.

Resolution at 2x2x2: a block carries `8 x n_sim` samples of its own interior —
160 at `n_sim = 20`, giving a share to sd 0.038, and never falsely saturating
at 0 or 1 in 20 000 trials. `tolerance` defaults to 0.05, so one realization
in twenty finding a block divided does not carry it.

### 7.4 The saving erodes with variable count

Break-even is `p = 87.5%`. If V variables each independently mark a fraction
q, the union is `1 - (1-q)^V`:

| variables | q = 10% | saving | q = 15% | saving |
|---|---|---|---|---|
| 1 | 10% | 4.44x | 15% | 3.64x |
| 3 | 27% | 2.52x | 39% | 1.94x |
| 8 | 57% | 1.44x | 73% | 1.17x |
| 12 | 72% | 1.20x | 86% | **1.02x** |

The bound assumes independence and is pessimistic — the elements of a real
polymetallic deposit are strongly correlated and the union is much closer to q
— but the direction is a genuine constraint. The mitigation is to let the user
name which variables drive splitting, rather than defaulting to all of them.

An economic criterion such as NSR would collapse the union to one variable,
but it depends on prices and recoveries, which are not modelling inputs.
**Out of scope here**: `split_on` names variables, and any economic
combination belongs downstream of the model.

### 7.5 Where the numbers live

Two additions to `ContinuousVariable`, both fed by `update`:

- **`dispersion`** — one column beside `latent_variance`, the within-block
  variance averaged over realizations. Not `(n_data, n_sim)`: per-realization
  dispersion would double the storage this exercise exists to reduce.
- **`proportions`** — an `OrderedDict` keyed by cut-off, the same shape as the
  existing `probabilities` but at **sub-block** support rather than block
  support. This is the recoverable share within a block, and it is also
  exactly the splitting criterion.

`cutoffs` becomes a declared property of the variable, set on the raw data and
carried to the blocks by `copy_to`. `cutoffs = None` means the variable
neither reports proportions nor participates in splitting — which is the
opt-out for a composition's rest component. **One declaration serves both.**

Both go in `_ZARR_ATTRS`; the reader uses `.get`, so old stores still open.
`VectorVariable` needs nothing special — its components are
`ContinuousVariable`s and each carries its own.

---

## 8. What breaks

Each run against a `Blocks3D(n=[20,20,10], discretization=[3,3,3])` and a
subset of it (`subset_study`).

| Capability | Why it breaks | Difficulty |
|---|---|---|
| `as_cube` / `as_image` | `reshape(values, grid_size)` | **fundamental** — no rectangular array exists |
| `get_contour` | needs `as_cube` | **solved** — VTK contour on hexahedra (§5) |
| `smooth` | needs `as_cube` | needs a graph/neighbour smoother; new code |
| `Blocks3D.as_pyvista` | `pv.ImageData`, implicit geometry | `UnstructuredGrid`; ~40 MB per 660k cells |
| `index_data`, `aggregate_*` | O(1) cell arithmetic off a uniform step | tree descent or KD-tree, O(log n) |
| `grade_tonnage` | scalar `volume = prod(step_size)` | per-cell volume; small |
| `Grid3D.assign_from_surface` fast path | column periodicity of `_generate` | already falls back to the general per-cell path |
| `to_zarr` / `open` | stores `start`/`n`/`step` | store the lattice |
| `interpolation.py` | `grid_size` | out of scope — not a block-model path |

Batching is **not** on this list: §2.1 removes it.

---

## 9. Other cell formats

### 9.1 Tetrahedra

Contour cleanly (§5), and their structural advantage is real: no hanging
nodes, so no cracks. Against that:

- **Higher cell count** for equal resolution — 266 573 tets from 40 000 points,
  about 6.7 tets per point.
- **Build cost** — 2.23 s for 40 000 points by Delaunay, and the result holds
  slivers of near-zero volume, unpleasant as tonnage weights.
- **Wrong shape for the answer.** A reserve is reported on a selective mining
  unit, which is a *block* because that is the shape of mining selectivity. A
  tetrahedron corresponds to nothing that can be excavated.

Worth having as a geometry format eventually; not as the support a block model
reports on.

### 9.2 Corner-point and curved cells

**Corner-point geometry is not a memory optimization.** It stores eight
corners a cell, more than a regular grid. What it buys is geometric fidelity —
cells following stratigraphy and fault offsets.

It is **logically structured** (i, j, k) while geometrically deformed, so
`as_cube` keeps working on the logical index; but a contour extracted in
logical space must be pushed through each cell's trilinear map to reach
physical space, and that is a different surface from contouring in physical
space. Contour in physical space, via VTK.

Two consequences: block averaging needs the reference sub-grid pushed through
each cell's trilinear map rather than added as a shared offset (the same
per-cell change §2.1 already makes); and faults appear as pillar offsets, so
logical neighbours need not be geometric neighbours. That is the topology the
fault-transform item identified as its hard part, and it belongs with that
work rather than this.

Non-planar faces make a hexahedron ambiguous; VTK decomposes them into tets
internally, so "the" contour depends on that decomposition.

---

## 10. Always full, with filters

The model is **structurally complete at all times**. Blocks are never removed;
regions are excluded by a filter.

This is what makes grouping unconditionally safe. A partial group would average
over missing children and silently mis-weight — precisely the mass-conservation
error the lattice exists to prevent. Filtering is a *value* property, structure
stays complete, so every coarsening group is always populated.

Filters are boolean metadata columns — existing machinery (`add_metadata`), no
new state class.

Two consequences worth having:

**Filters make prediction cheaper, not just reports cleaner.** A filtered block
need never be predicted. Air above topography costs nothing, and it composes
with the two-pass: filter, predict coarse, refine. A filtered block's value
stays NaN, and since `grade_tonnage` already skips non-finite values, most of
the reporting layer needs no filter awareness at all.

**Done**, as `where=` rather than as a stored filter: `refine(model, blocks,
where=mask)` names the ground worth modelling, and the blocks left out are
never predicted at any pass *and* never cut — they hold nothing to decide and
draw no surface, so cutting them would only make more of nothing. The mask is
given once, against the blocks as they stand, and carried across each split,
because `split` keeps the unsplit blocks first and then each parent's children
in sub-block order. `predict(..., where=)` had to be relaxed for the first
pass: naming only some locations of a container that does not carry the
variable yet used to be refused, and now creates the variable and leaves the
rest at NaN, which is what `unpredicted` reads. No `set_filter` and no stored
state — the argument is enough, and a mask that ought to persist is an
ordinary metadata column the caller passes in.

**Fullness is cheap to keep** for the same reason it is worth keeping: the
excluded regions are the uninteresting ones, which coarsen to almost nothing.

### 10.1 Do not overload `subset_region`

`_GriddedData.subset_region` (`data.py:3288`) already returns a `PointData`,
carrying variables, simulations and metadata, and dropping `step_size`,
`discretization`, `grid_size`, `grid` and `sub_grid`:

| method | `Blocks3D` | subset |
|---|---|---|
| `as_cube`, `get_contour`, `smooth` | ok | `ValueError` / `NotGriddedDataError` |
| `grade_tonnage` | ok | `TypeError: needs a gridded container` |
| `aggregate_numeric`, `index_data` | ok | `AttributeError` |
| `as_pyvista`, `as_data_frame`, `to_zarr` | ok | ok |
| `assign_from_surface` | ok | ok |

Having `subset_region` mutate a filter on `BlockSet3D` would break a contract
the rest of the package relies on, and returning a view over shared mutable
`ArrayStore`s is the footgun `__deepcopy__` already avoids by materializing.
Keep the two capabilities under distinct names:

- `subset_region(...)` — still exports a `PointData`, one way, for handoff
- `set_filter(...)` / a `where=` argument — the in-place workflow

### 10.2 What a `PointData` export should carry

**Mostly done, and not the way this section proposed.** Nothing had to be
handed to an export, because the block model answers for itself:
`BlockSet3D.as_data_frame` appends a `_X`/`_Y`/`_Z` column of per-row block
size, `as_pyvista` writes explicit hexahedra (volume exact against
`block_volume.sum()`), and `grade_tonnage` reads `prepare.block_volume`, which
asks for `block_volume` and falls back to `step_size` — so a container with a
size per block is served without knowing it is one. All three goals below are
met.

The **subset** used to fail quietly: `blocks[mask]` returned a plain
`PointData` with no `_origin`, no `_level` and no size, so `_X`/`_Y`/`_Z`
vanished from the data frame, `as_pyvista` gave points rather than blocks and
`grade_tonnage` raised — on an object that looked usable. It now **raises**.
`__getitem__` and `subset_region` are overridden to refuse, which is §10's
conclusion carried into the code: a block model is structurally complete, and
ground is excluded by value rather than by removing blocks. The message names
the three ways out — a metadata column for the exclusion, `predict(...,
where=...)` to visit part of a model without making a smaller one, and
`as_data_frame`/`as_pyvista`/`to_zarr` for a handoff.

That is the whole of the "should it carry the size or refuse" question below:
refusing, because carrying it would make a half-capable object that still
could not group, and grouping is the point of the lattice.

Original notes below.

Block size as metadata is under-powered: `grade_tonnage` reads
`container.step_size`. Give the export **per-row `step_size` plus
`discretization`** and it recovers `grade_tonnage` (which needs only a
per-cell volume, and the 0.5.6 streaming loop already reads mass a band at a
time), `rows_per_location`, and `as_pyvista` as explicit hexahedra.

Not recoverable, and not worth faking: `as_cube`, `as_image`, `smooth` and
`aggregate_*` all need a rectangular lattice.

`Section3D` (`data.py:4178`) is the precedent — a `PointData` that keeps
`grid_shape` and rebuilds its faces for export.

---

## 11. Using it

Worked on the Macpass drillholes, which are not distributed with geoML —
`datasets.macpass` takes the directory you downloaded them into. Every number
below is from running it; the whole thing takes about a minute and a half on
a GPU.

### 11.1 The data, and one deposit

The field is 27 x 18 km in three widely separated groups, so a single block
model over all of it would be mostly air.

```python
import numpy as np
from scipy.cluster import hierarchy
import geoml

geoml.set_seed(1234)
holes = geoml.datasets.macpass("path/to/Macpass")
samples = holes.composite(5.0).as_point_data()

xy = np.asarray(samples.coordinates)[:, :2]
group = hierarchy.fcluster(hierarchy.linkage(xy, method="single"),
                           900.0, criterion="distance")
biggest = np.argmax(np.bincount(group))
lo, hi = xy[group == biggest].min(axis=0), xy[group == biggest].max(axis=0)
z = np.asarray(samples.coordinates)[:, 2]
deposit = samples.subset_region([lo[0] - 100, lo[1] - 100, z.min() - 10],
                                [hi[0] + 100, hi[1] + 100, z.max() + 10])
```

    deposit: 5667 samples over [2352.  860.  656.] m

### 11.2 Declaring what the decision is

A cut-off is a property of the variable, not of the blocks. Declared on the
data, it travels to whatever is predicted from it, so nothing downstream has
to be told a second time.

```python
deposit.variables["Zn_pct"].set_cutoffs([4.0])
```

    Zn 0.000 .. 45.7 %, 656 of 5667 samples over 4%

`set_cutoffs(None)` is the opt-out: that variable takes no part in any
decision and reports no shares. It is the right answer for a composition's
rest component, and for anything carried along that nobody mines on.

A **categorical** variable declares nothing. Its cut-off is zero on
`ind_skew`, a category's log-odds against its best rival, so the contact
between two categories is that quantity's zero level set and the criterion is
the same one a grade gets. Excluding a category from the decision is
`split_on`'s job, not the variable's.

### 11.3 A model, and a coarse block model to start from

Nothing here is new — the model is built and trained as always.

```python
warping = geoml.warping.ChainedWarping(
    geoml.warping.Softplus(1), geoml.warping.ZScore(1),
    geoml.warping.Spline(1, 8), geoml.warping.ZScore(1))
inducing = geoml.inducing.from_kmeans(deposit, 400, seed=0)
root = geoml.latent.BasicInput(
    [inducing], transform=geoml.transform.Anisotropy3D(
        maxrange=120.0, midrange_fct=0.6, minrange_fct=0.25, azimuth=45.0))
gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())

model = geoml.models.VGPNetwork(
    deposit, "Zn_pct", geoml.likelihood.Gaussian(warping=warping), gp,
    options=geoml.models.GPOptions(verbose=False, seed=1234,
                                   training_samples=10))
model.train_full(max_iter=120)
```

The block model starts **coarse**, and `max_levels` fixes how fine it may
ever go. `discretization` is both the sub-block lattice and the ratio a block
splits by, so `(2, 2, 2)` with `max_levels=3` means 40 m blocks that can
reach 5 m.

```python
corner = np.asarray(deposit.coordinates).min(axis=0)
span = np.ptp(np.asarray(deposit.coordinates), axis=0)
step = np.array([40.0, 40.0, 40.0])

blocks = geoml.data.BlockSet3D(
    start=corner, n=np.ceil(span / step).astype(int) + 1, step=step,
    discretization=(2, 2, 2), max_levels=3)
```

    coarse model: 24840 blocks of [40.0, 40.0, 40.0] m, base cell [5.0, 5.0, 5.0] m
    a uniform model at the finest level would be 12718080 blocks

### 11.4 The multi-pass prediction

```python
refined = geoml.models.refine(model, blocks, n_sim=20, verbose=True)
```

    pass 1: cut 5485 block(s), 63235 now
    pass 2: cut 18106 block(s), 189977 now
    pass 3: cut 43802 block(s), 496591 now
    refined in 74 s

    496591 blocks, 25.6x fewer than uniform, full=True
       level 0:  19355 blocks of  40.0 m
       level 1:  25774 blocks of  20.0 m
       level 2: 101046 blocks of  10.0 m
       level 3: 350416 blocks of   5.0 m

Half a million blocks against the 12.7 million a uniform 5 m model would need,
and the 5 m ones are where the 4% surface runs.

Three passes, and nobody asked for three. `refine` runs until nothing needs
cutting, and needs no count: each pass takes the blocks it splits one level
finer, and the criterion never marks a block already at `max_levels`, so
within that many passes every block is either settled or as fine as the
lattice allows. How fine that is was decided when the block set was made,
which is the one place it belongs.

`refine` takes `split_on` to name which variables get a say, `tolerance` for
how often a block must be found divided, and `include_noise`, which is passed
to the predictions. The criterion never reads the noise (§7.3).

### 11.5 Doing it by hand

`refine` is a short loop over three calls, and there is no reason not to write
it out — to stop part way and look at what a pass cost, or where it went, or
to decide on something `needs_splitting` does not offer:

```python
model.predict(blocks, n_sim=20)

for _ in range(3):
    mask = blocks.needs_splitting(split_on="Zn_pct", tolerance=0.05)
    if not mask.any():
        break
    blocks = blocks.split(mask)
    model.predict(blocks, n_sim=20, where=blocks.unpredicted())
```

`split` keeps what the blocks that were not split already hold, `unpredicted()`
names the ones it created, and `where=` visits only those. The result is
bit-identical to predicting the whole refined model.

What the mask is made from is readable on its own:

```python
blocks.block_shares()            # {'Zn_pct @ 4': array([...])}
variable = blocks.variables["Zn_pct"]
variable.proportions[4.0]        # how much of each block is under 4% Zn
variable.divided[4.0]            # how often 4% passes through the block
variable.dispersion              # how much the block varies inside itself
```

`proportions` is the recoverable share and is worth having whether or not
anything is refined; on a categorical variable it is per category, and it is
the partial-block domaining number:

```python
for label, part in blocks.variables["rock"].components.items():
    part.proportion   # the share of each block this rock type holds
    part.divided      # whether the block straddles its boundary
```

### 11.6 Reporting on the result

Everything downstream counts each block at its own size.

```python
import geoml.plots.prepare as prep

curves = prep.grade_tonnage(refined, "Zn_pct", density=2.96,
                            cutoffs=[1.0, 2.0, 4.0, 8.0])
```

    >=  1.0 %Zn :    1.491e+09 t, at 3.71 %
    >=  2.0 %Zn :    8.830e+08 t, at 5.72 %
    >=  4.0 %Zn :    4.566e+08 t, at 8.42 %
    >=  8.0 %Zn :    1.815e+08 t, at 12.55 %

A grade shell comes from VTK rather than marching cubes, there being no
rectangular array to give it:

```python
shell = refined.get_contour("Zn_pct", 4.0)
```

    4% shell: Surface3D, 57301 triangles, closed=False

Note the type. It came back a `Surface3D` and not a `Solid3D`, which is the
honest answer: the shell is cut by the edge of the modelled box, so it is not
a body and cannot say what volume it encloses. That is the only cause left —
tearing where the mesh changes level is handled inside `get_contour` now
(§5.1.1), and this run predates that. `heal()` is there when the geometry is
nearly right.

### 11.7 Saving it

```python
refined.to_zarr("macpass_zn.zarr")
back = geoml.data.BlockSet3D.open("macpass_zn.zarr")
```

The lattice is stored as integers, so what comes back tiles its box exactly
as what went in did, and can be refined further.

---

## 12. Plan

1. **Within-block dispersion from the coarse pass.** A second reduction beside
   `_aggregate` (`likelihood.py:276`), a `dispersion` column and a
   `proportions` dict on `ContinuousVariable`, and `cutoffs` declared on the
   variable. A few lines in one function, no signature change to `_aggregate`
   and so none of its 25 call sites touched. Independently useful: nothing in
   the package currently reports change of support.
2. **`BlockSet3D`** — integer lattice, `discretization` and `max_levels`
   required, initialized as a full grid in lieu of `Blocks3D`. Split and group
   preserving fullness and conserving mass. Filters as metadata.
3. **Per-cell volume in `grade_tonnage`**, and the VTK contour path.
4. **The refinement criterion and the two-pass driver** — `models.refine`,
   `BlockSet3D.needs_splitting`, `split_on` naming the variables, the union
   rule. **Done**, with the correction in §7.3.
5. **Point location** for `aggregate_*` and `index_data`.
6. **Crack-free contouring across levels**, so a refined mesh does not tear
   where a contour runs through it. **Done**, but not as §5.1 proposed: the
   mesh is cut, not the model. See §5.1.1.
7. **`group`**, the inverse of `split`, with the per-group completeness check
   of §10 — what makes conversion between supports two-directional.

Deliberately excluded: level-aware batching (§2.1 makes it unnecessary),
economic criteria such as NSR (§7.4 — not modelling inputs), tetrahedra as a
block support (§9.1), and corner-point geometry (§9.2).

Noted for later rather than done: `_CategoricalLikelihood.
entropy_and_indicators` builds `ind_skew` with one full-size scatter per
category, which a single top-2 pass would replace. It matters more since
§7.3 moved that call before aggregation, where it sees
`prod(discretization)` times as many rows. And the noise integration behind
§7.2 is still crude — with a fixed sub-block count a coarse block averages
out no more noise than a fine one, which it should.
