# Implicit surfaces and faults as transforms

A design record for `geoml.math.rbf`, `geoml.math.geometry.point_normals`
and the fault transforms in `geoml.transform` (0.6.9). Its first purpose
is attribution: structural modelling is a mature field, and almost
everything here is somebody else's method written into this package's
idiom. The second is the measurements that decided the design.

## What is standard, and whose

| Piece | Source |
|---|---|
| Implicit surface as the zero level set of an RBF field fitted to points on it | Carr et al. (2001), the graphics reconstruction; Cowan et al. (2003) for the mining practice |
| Normals as gradient constraints of the field (the Hermite form), the cubic basis `r³` | Macêdo, Gois and Velho (2011); Hillier et al. (2014) for geological surfaces |
| Normals as pairs of displaced points with values `±d` (the off-surface form), thin-plate and linear bases | Carr et al. (2001) |
| The same system read as cokriging with gradient data, i.e. the potential-field method | Lajaunie, Courrioux and Manuel (1997); Calcagno et al. (2008) |
| Polynomial augmentation and conditional positive definiteness | Duchon (1977); Wendland (2005) |
| Greedy centre selection to a tolerance | Carr et al. (2001), section 4.2 |
| Normals from local PCA, sign propagated along a spanning tree | Hoppe et al. (1992) |
| A fault as a drift or feature that separates its two sides in the interpolant | Calcagno et al. (2008); de la Varga et al. (2019), GemPy |
| A fault as a displacement field with an envelope that dies out past the fault's extent, slip along the fault | Laurent et al. (2013); Georgsen et al. (2012) |
| Several faults restored youngest first, older surfaces refitted in the restored frame, abutting relations | Grose et al. (2021), LoopStructural; the series of Calcagno et al. (2008) |

What may be original is small and said so in the docstrings: the rule
that orients derived normals toward the surface's estimated concavity
(the user's, 2026-09-02), and the repulsion coordinate `amp · f(s/range)
· taper` as a kernel-space form of the fault drift, which `BellFault2D`
already did for a straight segment.

## Decisions, with the numbers behind them

**Parameter-free.** The user's condition: the surface is an interpolant,
nothing about it trains. The polyharmonic bases have no range. The
thin-plate spline and the linear basis, the classic parameter-free
choices, are not twice differentiable at the origin, so the gradient
block of the Hermite system is singular on its diagonal with them; `r³`
is the next member of the family and admits the constraints, hence the
default, with a linear drift for uniqueness. The thin-plate and linear
bases remain for the off-surface form, at the price of one implicit
scale, the offset.

**A fitted frame of unit extent.** A cubic basis at metre scale puts
entries of a million beside entries of one: the condition number of a
50 m sphere's system measured `1.1e14`. The solve was still exact to
`1e-12`, but the scaling costs nothing and is done: the fit lives in a
shifted, unit-extent frame; values keep the caller's units and gradients
are scaled back on the way out. A gradient is a covector, so a given
normal goes through a transform's Jacobian as `J⁻ᵀ n` and is *not*
renormalized; renormalizing it scaled the recovered gradient by the
transform (measured: a tenth, under an isotropic range of ten).

**Nothing per point.** Gradient constraints may sit on a subset of the
points, with NaN rows for the rest, or at their own locations, as the
potential-field method's orientation data usually do. One is enough to
remove the trivial zero field. A plane is reproduced exactly from zero
values and five normals; a sphere from four hundred zero values and one
normal comes back negative inside and positive outside.

**Derived normals.** Local PCA fixes each normal up to sign. The sign is
propagated along a minimum spanning tree of the neighbour graph from one
root per connected piece; the root's own sign comes from where its
neighbours' centroid falls relative to the tangent plane, which is the
concave side of a curved surface. On a flat patch that offset is noise,
so the function warns that the sign is consistent but arbitrary and
accepts a reference vector instead. A single line of points in three
dimensions is refused, one line fixing no normal.

**Greedy selection.** With `max_error` the fit starts from a tenth of
the points, adds the worst-fitting until every point is within the
budget, and reports how many it kept, as in Carr et al. A sphere of 1500
points fits to 0.05 with a fraction of them. The residual criterion is
on values only; gradients ride with their points.

**Faults as transforms.** Two variants on one base: the repulsion
feature, `BellFault2D` for any surface and dimension, which needs no
topology and simply concatenates, and the displacement, which restores
the hanging wall and lets one kernel read across the fault. Both take the
observations as arrays and refit the surface at construction, so a saved
model replays exactly; a fitted object would not persist.

**Slip in the fault's frame.** Training the displacement on a synthetic
layer offset by 20 m with a free slip vector let it drift to a component
of 23 m along the fault's normal. That component only pulls the walls
apart and is not identified by the data, and it is not what a fault
does. The slip is therefore said in the fault's own frame, the fault
frame of Laurent et al.: a `throw` along the local up-dip direction
(`n × (ẑ × n)`, whose vertical component is `|ẑ × n| ≥ 0`) and, in three
dimensions, a `strike_slip` along the local strike (`ẑ × n`), both unit
and tangent; in two dimensions the fault has one tangent, the normal
turned a quarter turn. A horizontal fault has no strike and takes the
east axis in its plane.

**Where the frame is read.** At the point's foot on the surface, not at
the point. The first version read it at the point, and on a sinuous
fault the restored coordinates piled up where the field's level sets
converged away from the surface and tore where they diverged, which the
user spotted in the figure: two points on one normal line were sent
different ways. Read at the foot, found by two Newton steps of the
field, everything on a normal line slides together along the surface,
the way material does, and the deformation that remains is the
fault-bend kind, which a restoration along a curved fault cannot avoid.
**What remains is the geometry, measured.** The normal lines of any
curve cross beyond a bend's centre of curvature, so a hanging wall
sliding along the surface must fold wherever the throw exceeds the
radius of curvature, whatever frame is used. On the first sinuous fault
(amplitude 1.5, radius of curvature 1.7 against a throw of 2.5) 1.9% of
the hanging wall folds — the restoration's Jacobian determinant runs
from −0.74 to 2.55 — and two points on one true normal line get feet
0.045 apart, the field's projection not being the Euclidean foot far
from so tight a bend; that is the pile-up and the wedge the user saw.
At amplitude 0.5 (radius 5.1) nothing folds, the determinant stays in
0.80–1.27 and the two moves agree within 0.6%. The demonstration now
uses amplitude 0.8 (radius 3.2), bends wider than the throw, as real
faults have; `test_faults.py` pins the fold-free case. A rigid translation would
remove it and would no longer close a curved surface; the full answer,
curvilinear fault-frame coordinates with the slip moving the along-fault
coordinate (Georgsen et al. 2012; LoopStructural), needs a
parameterization of the surface and is not built.

**The throw search.** With a continuous variable the throw trains from
zero at the phased learning rate. With a categorical likelihood, or with
two faults, it was measured not to move off zero in a thousand
iterations: the bound is flat in the throw away from the right value.
`models.search_throw` does what gradient descent cannot — for each
candidate throw the model is restored to its starting state through
`update_parameters`, the throw set, a fresh optimizer run for a short
burst, and the bound at its end recorded; the best candidate is set on
the restored model and training proceeds. On the 20 m layer it picks 20
against 0, 10 and 30 in four bursts of forty iterations.

**The gate for the displacement variant** (the user's condition: build
it if the tests support it). Synthetic layer `v = z − 20·H(x − 50)`,
noise 0.5, 400 training samples, 216 inducing points, held-out RMSE:

| schedule | plain `Isotropic(30)` | with `FaultDisplacement` | slip found |
|---|---|---|---|
| 300 iterations, default rate | 5.55 | 4.60 | (−0.7, −1.7, 2.7) |
| 1500 iterations, default rate | – | 1.55 | (−7.8, 1.9, 10.6) |
| rate 0.1, 500 iterations | 2.37 | 0.60 | (0.0, 4.5, 20.4), normal component pinned |

The default schedule under-trains everything, the true-slip run
included (3.3 after 300 iterations). At the rate the phased pattern uses
the model reaches the noise floor and reads the slip. The docstring says
so.

**Restoring observations.** Samples take a smooth step of width `width`
about the surface, which is what lets the slip train through it. Another
fault's observations take a hard step: inside the band the smooth step
had half-restored them, and the refitted surface bent through the
half-restored points.

**Abutting.** A fault declared to stop against an older one neither
displaces that fault's observations nor acts beyond its surface: its
displacement is multiplied by a step on the declared side of the older
field *as observed*. That is an approximation at the junction, where the
older field is read in a frame only partly restored; it is exact for
planes and is the documented semantics.

## Synthetic demonstrations (2026-09-02)

Three two-dimensional cases, run from the user's fault notebook as the
starting point; the script is `docs/benchmarks/fault_cases.py`.

**A. The notebook's contact.** A horizontal contact offset by 3 across
an inclined fault, modelled as rock types from 36 labelled points and 11
contact points, with the notebook's `Linear` head between the
concatenated transform and the GP, which learns anisotropy in the
expanded space. The base model bends the contact smoothly across the
fault; `BellFault2D` breaks it cleanly; `ImplicitFault` fitted to 25
points on the fault line breaks it just as cleanly **once its step is a
hard sign** — the first version used `tanh(s / range)` with the range
initialized at a fifth of the extent, which is a smooth ramp over the
whole box and bent the contact like no fault at all: the kernel already
sees a smooth ramp in the coordinates, the feature has to jump. The
displacement did not move its throw off zero with the categorical
likelihood in a thousand iterations. `search_throw` over seventeen
candidates in ±8 chose 6 — the bound after a forty-iteration burst on
categorical data rises from the true 3.2 to about 6 and only then falls,
the labels alone leaving the contact anywhere between the grid rows —
and training from there brought it to 3.92 against the exact 3.23 along
the fault, with the contact restored to one straight line.

**B. A grade offset along a curved fault.** A vertical gradient offset
by a throw of 2.5 along a sinuous fault (amplitude 0.8, radius of
curvature 3.2 at the bends) fitted from 41 trace points with normals
derived toward the concavity, 180 noisy samples. RMSE against the
truth: plain isotropic 0.43, repulsion 0.17, displacement 0.07 with the
throw chosen as 3 by the search among nine candidates and trained to
2.61 against 2.50; the restored grid slides along the surface without
folding. On its own, from zero, the throw also trains to 2.53
in two hundred iterations at the phased rate; in the script's run order,
from the initialization that draws, it stayed at zero, the kernel range
having shrunk to explain the jump first, which is why the search is
used here too. The restored coordinates show the grade continuous again
across the fault.

**C. Two faults, the younger cutting the older.** F1 (older, throw
2.15 along it, 2 of vertical offset) crossed by F2 (younger, throw
−1.62, 1.5 down), so F1 is observed in two pieces; a vertical gradient
as the layering, 220 noisy samples. From zero both throws stayed near
zero in a thousand iterations. Searched in turn — the younger over nine
candidates with the older at zero, then the older with the younger as
chosen — and trained jointly: RMSE against the truth plain 0.49, two
repulsion coordinates without topology 0.23, `FaultNetwork` 0.09, the
throws read as −1.65 against −1.62 for F2 and 2.32 against 2.15 for F1,
and F1's two observed pieces put back on one line by the trained F2. A
first run of this case, with faults that did not cross inside the box,
returned a NaN prediction map; that was the second defect below.

**D. Two curvilinear faults ending on a third.** An older inclined,
gently curved fault F0 (throw 1.5 along it, hanging wall above) with two
younger steep curvilinear faults F1 and F2 (throws 1.0 and −0.8) that
exist only above F0 and end on it; a vertical gradient as the layering,
260 noisy samples. Four models. Plain isotropic: 0.29. Three
`ImplicitFault` coordinates behind a `Linear(size=5)` head, no topology:
0.19 — every contact breaks cleanly, but F1's and F2's coordinates
continue past F0 until their tapers fade over the reach, since nothing
tells them where a fault ends. `FaultNetwork` with F1 and F2 declared to
stop on F0's upper side, throws searched oldest first then trained
jointly: 0.07, throws 1.64, 1.01 and −1.12 — F2 over by 0.3, its trace
being the shortest and its samples the fewest. The control, the same
network with the abutting left undeclared, so that F1 and F2 cut through
F0 and displace its observations: 0.23, worse than the repulsion, with
F0's throw lost entirely (0.02) because the younger faults' displacement
below F0 contradicts it. The topology is not decoration: declared, it is
what lets the older throw be found at all. `ImplicitFaultBlocks`, the
same three repulsions with F1 and F2 declared to stop on F0: 0.19, the
same as untrimmed. The partition is now right by construction — the
summed coordinates show F1 and F2 identically zero below F0 where the
untrimmed ones continued over their reach — but the repulsion's error on
this case is not the leak, which was a thin band; it is that a
repulsion only decorrelates across a fault and restores nothing, so each
block is fitted from its own samples alone, and that is the gap to the
network's 0.07. The trimming earns its place where a terminating fault's
phantom continuation would cut through data, which a reach of one unit
here did not.

**Topology in the repulsion, by composition of implicit functions**
(the user's suggestion, 2026-09-02). A fault F1 that ends on F0 is its
zero set trimmed to one side of F0's field, `{s1 = 0} ∩ {σ0 s0 ≥ 0}`,
constructive solid geometry on implicit functions. A repulsion only
reads the sign of its field, and the sign of the trimmed surface is
`sign(s1) · H(σ0 s0)`, a product of steps: ±1 on the declared side of
F0, identically 0 beyond it. The distance to the trimmed surface is, to
the usual approximation, `sqrt(s1² + max(0, −σ0 s0)²)`, exact for
orthogonal planes, which is what the decay mode uses so that the feature
fades past F0 as it fades away from F1. A chain of terminations is the
product of its steps; a fault's own tip line would be the same
construction with an extent field, LoopStructural's third fault-frame
field, in place of `s0`. `ImplicitFaultBlocks(faults, abutting)` is
that, the repulsion's twin of `FaultNetwork`; nothing is restored, so
every field is read as observed and no age order is needed. The one
artefact is at the junction: crossing F0 next to F1 costs
`sqrt((2a0)² + a1²)` in the expanded space where crossing F0 elsewhere
costs `2a0`, because F1's coordinate drops from `±a1` to 0 there; no
per-fault encoding avoids it, and `a1` trains.

**Improving the displacement (2026-09-02, the user's request, four
points).** (1) *Fault-parallel flow*: the move is no longer a straight
step along the frame at the foot but an integration along the level set
of the field through the point, `flow_steps` midpoint steps with the
frame read where the point is at each, the standard kinematics of
restoration (Egan et al. 1997); on a plane it is the straight step to
1e-9, on a curved fault the moved points stay on their level set to 2%
of the throw where the straight step drifts three times as far. Two
points on one normal line now follow parallel curves of different
length, so their chords differ by a few percent by design. (2) *A throw
profile*: `profile="bell"` scales the throw by the taper's bell over the
foot's coordinates in the fault's mean frame, the along-fault axes from
the mean normal of the observations, with trainable extents starting a
fifth past the observations — largest at the centre, zero at the tip
lines, the displacement profile of Walsh and Watterson (1987) in the
shape LoopStructural uses. (3) *Finding the throw*: `search_throw`
refines coarse then fine (`refine=`); the step's width is a parameter
that `set_width` anneals between phases, wide where the bound is smooth
in the throw and sharp at the end; and `throw_from_markers` reads the
throw off one horizon seen on both walls by Gauss-Newton on the
footwall markers' implicit surface, which is how a geologist measures
it and needs no bound. (4) *Drag*: `drag=True` trains the width within
bounds as the width of a drag zone.

**E. Case D's geometry with the improved displacement.** The truth now
gives F1 and F2 bell-shaped throws, peaks 1.4 and −1.2, and F0 its
constant 1.5, all moved by fault-parallel flow; 260 noisy samples of a
vertical gradient. Plain isotropic 0.22, `ImplicitFaultBlocks` 0.19, the
network as it was — straight step, one throw per fault, fixed width,
coarse search — 0.19 with throws 1.18, 0.63 and −0.62, a constant throw
being the wrong shape for a profile and settling on its average. The
improved network — flow, bell profiles on F1 and F2, trainable drag
widths, F0's throw started from the marker horizon (1.70 read against
1.50), coarse-then-fine searches for F1 and F2 at a step width of 1,
then three training phases with the width annealed 1 → 0.4 → 0.15 —
0.05, throws 1.57, 1.39 and −1.64: F0 and F1 within a tenth, F2's peak
over by 0.44 on the shortest trace with the fewest samples, its learned
profile the true shape but deeper. The drag widths trained to 0.08,
0.05 and 0.33. The price is time: 1019 s against 297 s, the flow's ten
field evaluations per call against three and the two refinement passes.

**What the demonstrations settle.** The repulsion is `BellFault2D` for
any surface and must jump. The displacement restores exactly when its
throw is right; the throw trains from zero on continuous data when it
moves before the kernel range shrinks to explain the jump, and not
otherwise — with a categorical likelihood, with two faults, or from an
unlucky initialization — which is what the search on the bound is for,
and training refines what it chooses. The network's refit-in-the-graph
of the older surface works end to end.

**Two defects found on the way, and fixed.** The notebook's faulted
model puts a `Linear` head between the concatenated transform and the
GP. That configuration gave a NaN bound at the first iteration with any
transform: `Linear.refresh` built its inducing variance from the
parent's inducing *points*, so coordinates entered the kernel as
variances. One word, and a test in `test_node_protocol.py`. The second
was the NaN prediction map that appeared in two fits out of three of
case B and once in case C: a displacement moves the inducing lattice
too, the restored lattice had points 0.03 apart, and `BasicGP.refresh`
formed the simulations' square root as the Cholesky of `K⁻¹ − (K+D)⁻¹`,
`K⁻¹` at 1e9 against `(K+D)⁻¹` at 1e3 — a cancellation that came back
NaN in graph mode and finite eagerly, so the mean survived and every
simulation, and the prediction built on them, did not. The root is now
the whitened `L⁻ᵀ chol(W)`, `W = (I + LᵀD⁻¹L)⁻¹`, the same matrix with
nothing cancelled.

## Not built, on purpose

- No model overlay in the fault transforms beyond what a transform is.
- No compactly supported basis: it kills extrapolation across data gaps,
  which faults have; the parked conjugate gradient in `math/linalg.py`
  stays parked until a measurement asks for it.
- No manual chapter yet: no bundled dataset carries a fault.
- The locally-varying-anisotropy item can read orientation off
  `HermiteRBF.gradient`; nothing was done for it here.

## Deleted

The commented-out `_RadialBasisFunction` family in `kernels.py`
(polyharmonic forms normalized to a `max_distance`, from a source the
user could no longer name, which never gave good results), and the
`make_interpolator` methods of `Grid3D` and `RotatedGrid3D`, which called
a function that no longer existed and that nothing called.

## References

- Calcagno, P., Chilès, J. P., Courrioux, G. and Guillen, A. (2008).
  Geological modelling from field data and geological knowledge, Part I.
  *Physics of the Earth and Planetary Interiors* 171, 147–157.
- Carr, J. C., Beatson, R. K., Cherrie, J. B., Mitchell, T. J., Fright,
  W. R., McCallum, B. C. and Evans, T. R. (2001). Reconstruction and
  representation of 3D objects with radial basis functions. *SIGGRAPH
  2001*, 67–76.
- Cowan, E. J., Beatson, R. K., Ross, H. J., Fright, W. R., McLennan,
  T. J., Evans, T. R., Carr, J. C., Lane, R. G., Bright, D. V., Gillman,
  A. J., Oshust, P. A. and Titley, M. (2003). Practical implicit
  geological modelling. *Fifth International Mining Geology Conference*,
  89–99.
- de la Varga, M., Schaaf, A. and Wellmann, F. (2019). GemPy 1.0:
  open-source stochastic geological modeling and inversion. *Geoscientific
  Model Development* 12, 1–32.
- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in
  Sobolev spaces. In *Constructive Theory of Functions of Several
  Variables*, Springer, 85–100.
- Georgsen, F., Røe, P., Syversveen, A. R. and Lia, O. (2012). Fault
  displacement modelling using 3D vector fields. *Computational
  Geosciences* 16, 247–259.
- Grose, L., Ailleres, L., Laurent, G. and Jessell, M. (2021).
  LoopStructural 1.0: time-aware geological modelling. *Geoscientific
  Model Development* 14, 3915–3937.
- Hillier, M. J., Schetselaar, E. M., de Kemp, E. A. and Perron, G.
  (2014). Three-dimensional modelling of geological surfaces using
  generalized interpolation with radial basis functions. *Mathematical
  Geosciences* 46, 931–953.
- Hoppe, H., DeRose, T., Duchamp, T., McDonald, J. and Stuetzle, W.
  (1992). Surface reconstruction from unorganized points. *SIGGRAPH
  1992*, 71–78.
- Lajaunie, C., Courrioux, G. and Manuel, L. (1997). Foliation fields and
  3D cartography in geology: principles of a method based on potential
  interpolation. *Mathematical Geology* 29, 571–584.
- Laurent, G., Caumon, G., Bouziat, A. and Jessell, M. (2013). A
  parametric method to model 3D displacements around faults with
  volumetric vector fields. *Tectonophysics* 590, 83–93.
- Macêdo, I., Gois, J. P. and Velho, L. (2011). Hermite radial basis
  functions implicits. *Computer Graphics Forum* 30(1), 27–42.
- Wendland, H. (2005). *Scattered Data Approximation*. Cambridge
  University Press.
