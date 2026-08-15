# 5. Latent networks

Everything so far was one GP behind one variable. The package's actual
model is a **network of latent nodes**, a directed graph the data flows
through, and the reason is not architectural ambition. The network is
where geological hypotheses go. "The grade rides on a regional trend plus
a local structure", "these seven elements share three underlying
processes", "this orebody is folded, so distances should be measured in an
unfolded space": each of these is a network shape, trained end to end by
the same ELBO.

## 5.1 The nodes

**Sources.** `BasicInput` turns coordinates into the network's first
latent variables and holds the inducing points. It is *deterministic by
design*, because coordinates are facts, and its `transform` is the door
where anisotropy, projections or a fault step in. That is chapter 2's
ellipsoid, made composable.

**Workers.** `BasicGP(parent, size=k)` is the GP node of chapter 3. The
`size` argument gives it $k$ latent columns that share one kernel and one
set of inducing points, which is cheap: one covariance structure serves
all of them.

It is worth being precise about what those $k$ columns are, because the
shorthand "one node, several variables" invites the wrong reading. **The
columns are independent fields.** They share a kernel, so they have the
same range, the same shape at the origin and the same anisotropy, but
knowing one of them tells you nothing about the others. Correlation is not
something a `BasicGP` produces, and it has to be introduced deliberately,
in one of two places:

- **In the network**, by passing the node's output through a `Linear`
  node, whose trainable matrix mixes the independent columns into
  correlated outputs. This is the package's reading of the linear model of
  coregionalization, with the loadings learned rather than declared.
- **On the data side**, through a mixing warping. `PCA`, `RobustPCA` and
  `Rotation` decorrelate the measured variables before they reach the
  field, so independent latent columns come back as correlated variables
  after the back-transform. Chapter 16 takes this route.

**Combiners**, each a hypothesis:

- `Add` is superposition. A long-range node plus a short-range node is the
  trainable version of a nested variogram, with the split between
  structures estimated rather than declared.
- `Multiply` is interaction, one field modulating another's amplitude.
- `Linear` maps its parent's columns onto however many outputs are wanted,
  through a trainable matrix. Close to the root it rotates coordinates, in
  the middle it is a bottleneck, and at the end it is where correlation
  between outputs comes from.
- `LinearCombination` mixes several parents of the same size with positive
  weights that sum to one, keeping the output's variance under control.
- `Concatenate` stacks parents side by side into a joint input for a
  deeper node.
- `Bias` is a trainable constant, for when the mean is not the data's.

**Depth.** A `BasicGP` whose parent is another `BasicGP` receives a
*distribution* rather than a point. The parent's uncertainty rides along,
which is what makes the stack a deep GP instead of two models glued
together. The enabling mathematics is the Paciorek non-stationary kernel,
the one covariance that absorbs Gaussian input uncertainty analytically,
derived for this setting in the 2025 deep-GP paper. The practical reading
is simpler: **the inner layers warp space**. A stationary kernel in the
warped space is non-stationary in the real one, so folded veins and curved
orebodies stop being kernel problems and become network problems, and the
network is trainable.

Depth comes with one habit that is close to mandatory. **Concatenate the
inner node with the original coordinates before feeding the next layer.**
An inner GP is free to map two distant regions onto the same place, and
if the outer layer sees only the inner node's output it has no way to tell
them apart, so the space collapses and points that are far away become
artificially correlated. Keeping the coordinates in the joint input costs
a couple of columns and removes the failure mode.

## 5.2 Choosing a shape

Start with `BasicInput → BasicGP` and earn every addition:

- variance left at short range that the kernel cannot take: `Add` a
  short-range node;
- several variables that should share structure: one `BasicGP(size=k)`
  followed by a `Linear`, or a decorrelating warping on the data side;
- mapped geometry the ellipsoid cannot express: depth, an inner GP or an
  SDE node warping space, which is the subject of the 2025 paper;
- more data than one inducing set carries: experts (chapter 3).

The network mirrors belief, and it also *documents* it. `to_dot()` renders
the graph, and the model's `repr` prints every node with its parameters,
worth reading before training the way one reads a variogram model before
kriging with it.

## 5.3 A deep model, drawn

Two layers on Walker Lake: a two-column inner GP warping space, and a
one-column outer GP reading the warped coordinates alongside the real
ones. The inducing points are chapter 3's grid of experts, and the warping
is the positive chain of chapter 4 with a `Scale` in front, so the
predictions stay in assay units and cannot come back negative.

```python
import geoml
import numpy as np

geoml.set_seed(1234)
walker, walker_grid = geoml.datasets.walker()

experts = geoml.data.inducing.grid_experts(walker_grid, 10.0, block=8)

root = geoml.latent.BasicInput(
    experts,
    transform=geoml.transform.Isotropic(50))

# the inner layer moves the ground: two smooth fields that the outer layer
# will read as if they were coordinates
inner = geoml.latent.BasicGP(
    root,
    size=2,
    kernel=geoml.kernels.Gaussian())

# the real coordinates travel alongside the warped ones, so that two
# distant places cannot end up at the same address
deep_input = geoml.latent.Concatenate(root, inner)

outer = geoml.latent.BasicGP(
    deep_input,
    size=1,
    kernel=geoml.kernels.Spherical())

warping = geoml.warping.ChainedWarping(
    geoml.warping.Scale(1),
    geoml.warping.Softplus(1),
    geoml.warping.ZScore(1),
    geoml.warping.Spline(1, knots_per_arm=4))

model = geoml.models.VGPNetwork(
    walker, "V",
    geoml.likelihood.Gaussian(warping),
    outer,
    options=geoml.models.GPOptions(verbose=False))

model.train_full(max_iter=100)

print([node.name for node in [outer] + outer.get_unique_parents()])
print(model.to_dot().splitlines()[0])
```

The node list is the graph the ELBO trains, and the DOT text renders with
any Graphviz viewer. Prediction confirms what the warping promised:

```python
subset = geoml.data.PointData.from_array(
    np.asarray(walker_grid.coordinates)[::20])
model.predict(subset, n_sim=20)

print("lowest prediction:",
      round(float(np.min(subset.values("V/prediction"))), 1))
```

The value is positive, and it is positive by construction rather than by
luck. Chapter 2's model could return a negative grade wherever it was
unsure. This one cannot, at any location and in any realization, because
the softplus in the chain has no negative branch.

Depth costs iterations. A two-layer model settles more slowly than a flat
one, and it repays the wait only where the geometry is genuinely curved,
which Walker Lake's is only mildly. The folded vein of chapter 17 is the
honest showcase.

> **In the code.** Every node class lives in `geoml.latent`, and
> `viz/graphviz.py` renders any model or network through `to_dot()`. The
> expert propagation of chapter 3 keys off this same graph, so what
> `to_dot` draws is exactly what runs.

## Further reading

Damianou & Lawrence (2013) for deep GPs; Paciorek & Schervish (2003) for
the non-stationary covariance that carries the input uncertainty; the 2025
paper for the analytical propagation and the SDE node; the 2026
scalable-VGP paper for a deposit-scale network combining most of this
chapter.

## References

Damianou, A., & Lawrence, N. D. (2013). Deep Gaussian processes.
*Proceedings of the 16th International Conference on Artificial
Intelligence and Statistics (AISTATS)*, 207–215.

Gonçalves, Í. G. *et al.* (2025). Uncertainty propagation in deep Gaussian
process networks. *Mathematical Geosciences*.
<https://doi.org/10.1007/s11004-025-10187-4>

Gonçalves, Í. G. *et al.* (2026). Scalable variational Gaussian process
framework for implicit geological modelling and compositional grade
interpolation. *Artificial Intelligence in Geosciences*.
<https://doi.org/10.1016/j.aiig.2026.100218>

Paciorek, C. J., & Schervish, M. J. (2003). Nonstationary covariance
functions for Gaussian process regression. *Advances in Neural Information
Processing Systems 16*.
