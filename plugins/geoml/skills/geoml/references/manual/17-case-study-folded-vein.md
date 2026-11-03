# 17. Case study: a folded vein in section

The last case study is the one the first two could not be: a body whose
shape changes direction, where one anisotropy ellipsoid cannot follow the
ground and the deep network of chapter 5 has something to do. A vein is
folded through a vertical section, five drillholes cut it, and the
question is the usual one: where is the vein between the holes, and how
sure are we?

The section is synthetic, and that is the point of it. The truth is
known everywhere, so every comparison in this chapter is made against the
ground rather than against the samples the models were trained on, and a
section is small enough to train both models in minutes and to draw what
the deep model does to the coordinates. Both models are built and both
maps are drawn, so the cost and the gain are on the same page.

## 17.1 The section and the holes

The truth is a raster of the vein on a 0.1 m grid, 16 m across and 20 m
deep, stored beside the chapter. Five holes are collared at the surface,
four of them inclined, and each is sampled every 0.2 m: a sample is ore or
waste by the cell it falls in. Every hole cuts the vein at least twice.

```python
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import geoml

os.makedirs("figures", exist_ok=True)
geoml.set_seed(1234)

# 201 rows from -20 m to 0, 161 columns from -8 m to 8
truth = np.load("data/vein-truth.npz")["ore"]
EXTENT = [-8, 8, -20, 0]


def ore_at(x, y):
    """Whether the truth holds ore at each point, by its nearest cell."""
    row = np.clip(np.round((y + 20) / 0.1).astype(int), 0, 200)
    column = np.clip(np.round((x + 8) / 0.1).astype(int), 0, 160)
    return truth[row, column]


# where each hole is collared, and its angle from vertical in degrees
HOLES = [(-4.5, 20), (-1.5, -20), (1.0, 0), (3.0, -40), (7.5, -20)]

coordinates, hole_of = [], []
for number, (collar, angle) in enumerate(HOLES):
    depth = np.arange(0, 40, 0.2)
    x = collar + depth * np.sin(np.radians(angle))
    y = -depth * np.cos(np.radians(angle))
    inside = (x >= -8) & (x <= 8) & (y >= -20)
    coordinates.append(np.stack([x[inside], y[inside]], axis=1))
    hole_of.append(np.full(np.sum(inside), number))
coordinates = np.concatenate(coordinates)
hole_of = np.concatenate(hole_of)
ore = ore_at(coordinates[:, 0], coordinates[:, 1])

samples = geoml.data.PointData.from_array(coordinates)
samples.add_categorical_variable("Rock", ["waste", "ore"],
                                 np.where(ore, "ore", "waste"))
print(samples.n_data, "samples,", int(np.sum(ore)), "of them ore")
```

```python
def draw_holes(axes, points=coordinates):
    """Each hole as a line, its ore samples in red."""
    for number in range(len(HOLES)):
        hole = points[hole_of == number]
        axes.plot(hole[:, 0], hole[:, 1], color="k", lw=0.8)
    axes.scatter(points[ore, 0], points[ore, 1], s=5, color="#d62728",
                 zorder=3)


figure, axes = plt.subplots(figsize=(4.6, 5.6))
axes.imshow(truth, origin="lower", extent=EXTENT, cmap="Greys",
            vmin=0, vmax=2.5)
draw_holes(axes)
axes.set_aspect("equal")
axes.set_xlabel("x (m)")
axes.set_ylabel("depth (m)")
axes.set_title("the vein (grey) and the holes")

figure.savefig("figures/17-section.png", dpi=150, bbox_inches="tight")
```

![The folded vein and the five holes that cut it](https://italo-goncalves.github.io/geoML/_images/17-section.png)

The vein comes up from the left, turns back on itself in a tight hinge
near the middle hole, and leaves to the right with a second closure below.
The limbs dip in different directions on either side of the hinge, so
there is no single direction along which the vein is continuous: that is
the shape a stationary model cannot follow.

## 17.2 Two models that differ in one node

Both models are the two-scale indicator model of chapter 6. One input
reads the coordinates through two transforms, one starting at a 20 m
range and one at 2 m, and hands each GP its own pair of columns: a
long-range GP carries the broad split between ore and waste, and a
short-range GP carries the vein itself. The hierarchical indicator turns the
two into the probability of ore. Every inducing point sits on a 1 m grid
over the section. The transforms' ranges train; the GPs' own ranges are
held where a transform already sets the scale (`fix_range=True`).

The stationary model's short GP reads the coordinates directly, with a
range per axis, so the best it can do is one ellipse for the whole
section. The deep model puts a `GPWalk` under it (chapter 5): a GP with
two outputs is a vector field over the section, the walk moves every
point along it in ten small steps, and the short GP reads the moved
coordinates. That GP is isotropic, because the walk already says how the
space is stretched; a range per axis above a walk would be a second way
of saying the same thing. Both short GPs take the Gaussian kernel. The
deep one reads the walk, so it averages its kernel over where each point
may have been carried, which is one closed-form term for the Gaussian
kernel and eight for a Matérn one; the stationary one takes it too, so
that the two models differ in the walk alone.

```python
inducing = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])


def vein_model(deep):
    """The two-scale vein model, with a walk under the short GP if deep."""
    root = geoml.latent.BasicInput(
        inducing,
        geoml.transform.Concatenate(geoml.transform.Isotropic(20),
                                    geoml.transform.Isotropic(2)),
        center=True)
    long_gp = geoml.latent.BasicGP(
        geoml.latent.SelectInput(root, [0, 1]), size=1, fix_range=True)
    short_input = geoml.latent.SelectInput(root, [2, 3])
    walk = None
    if deep:
        field = geoml.latent.BasicGP(short_input, size=2, fix_range=True)
        walk = geoml.latent.GPWalk(field)
        short_gp = geoml.latent.BasicGP(walk, kernel=geoml.kernels.Gaussian(),
                                        isotropic=True)
    else:
        short_gp = geoml.latent.BasicGP(short_input,
                                        kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        samples, "Rock",
        geoml.likelihood.HierarchicalGaussianIndicator(2),
        geoml.latent.Concatenate(long_gp, short_gp),
        options=geoml.models.GPOptions(jitter=1e-6, verbose=False))
    return model, walk
```

Both train for 500 full steps from the same seed, with the optimizer each
model builds for itself.

```python
geoml.set_seed(1234)
flat_model, _ = vein_model(deep=False)
flat_model.train_full(max_iter=500)

geoml.set_seed(1234)
deep_model, walk = vein_model(deep=True)
deep_model.train_full(max_iter=500)

figure, axes = plt.subplots(figsize=(7, 4.2))
axes.plot(flat_model.training_log, label="stationary")
axes.plot(deep_model.training_log, label="deep (GPWalk)")
axes.set_xlabel("iteration")
axes.set_ylabel("ELBO")
axes.legend()
print("bound: stationary %.1f, deep %.1f"
      % (flat_model.training_log[-1], deep_model.training_log[-1]))

figure.savefig("figures/17-elbo.png", dpi=150, bbox_inches="tight")
```

![The two models' bounds during training](https://italo-goncalves.github.io/geoML/_images/17-elbo.png)

The stationary model climbs faster, having only its variational state
and its ranges to fit; the deep model also has a vector field to learn.
It overtakes near iteration 250 and ends higher, at -90.1 against -94.9.
A bound is the training data's verdict, though, and the next section asks
the ground.

## 17.3 The maps, against the truth

Each model predicts the section on the truth's own grid, so every cell of
the raster has a probability of ore to be scored against. Four scores, and
a count:

- **accuracy**, the share of cells called right at a probability of 0.5;
- **IoU**, the intersection of the predicted and the true vein over their
  union, which ignores the easy waste far from the holes;
- the **Brier** and **log** scores of the probability (chapter 13), which
  reward a model for being sure where it is right and unsure where it is
  not, and are lower when better;
- the **bodies**, the separate pieces of the predicted vein. The truth
  is one.

```python
def scores(grid):
    """The predicted vein against the truth, cell by cell."""
    p = np.asarray(grid.values("Rock/ore/probability"),
                   dtype=float).reshape(201, 161)
    called = p > 0.5
    pieces, count = ndimage.label(called, structure=np.ones((3, 3)))
    sizes = ndimage.sum(called, pieces, range(1, count + 1))
    right = np.clip(np.where(truth, p, 1 - p), 1e-6, 1)
    return dict(accuracy=np.mean(called == truth),
                iou=np.sum(called & truth) / np.sum(called | truth),
                brier=np.mean((p - truth) ** 2),
                log_score=np.mean(-np.log(right)),
                bodies=int(np.sum(sizes >= 20)))


maps = {}
for name, model in [("stationary", flat_model), ("deep", deep_model)]:
    grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
    model.predict(grid)
    maps[name] = grid
    print("%-10s accuracy %.3f  IoU %.3f  Brier %.3f  log %.3f  "
          "bodies %d" % (name, *scores(grid).values()))
```

```python
figure, axes = plt.subplots(2, 2, figsize=(9, 9.6), sharex=True,
                            sharey=True)

for column, name in enumerate(["stationary", "deep"]):
    grid = maps[name]
    for row, (path, colours, label) in enumerate([
            ("Rock/ore/probability", "RdBu_r", "P(ore)"),
            ("Rock/uncertainty", "cividis", "uncertainty")]):
        ax = axes[row, column]
        values = np.asarray(grid.values(path), dtype=float)
        image = ax.imshow(values.reshape(201, 161), origin="lower",
                          extent=EXTENT, cmap=colours, vmin=0, vmax=1)
        ax.contour(truth.astype(float), levels=[0.5], extent=EXTENT,
                   colors="0.4", linewidths=0.8, linestyles="dashed")
        draw_holes(ax)
        ax.set_aspect("equal")
        ax.set_title("%s: %s" % (name, label))
        figure.colorbar(image, ax=ax, shrink=0.8)

figure.savefig("figures/17-maps.png", dpi=150, bbox_inches="tight")
```

![The two models' maps, the true vein dashed](https://italo-goncalves.github.io/geoML/_images/17-maps.png)

The deep model is better on every score: accuracy 0.942 against 0.933,
an IoU of 0.801 against 0.769, a Brier score of 0.047 against 0.069 and a
log score of 0.182 against 0.253, and it draws the vein as one body where
the stationary model draws two. The maps show where the difference lies.
The stationary model reads the vein through one ellipse, which suits the
limbs that dip one way and not the others: it breaks the lower limb short
of its tip and leaves the tip, under the hinge, as a lobe of its own. The
deep model carries the lower limb to its tip as one band.

The uncertainty tells the same story from the other side. The stationary
model's pale patches cover the ground between the holes on the lower
left, where it cannot tell which way the vein runs; the deep model is
unsure along the contacts and past the last hole on the right, and almost
nowhere else. The truth is what makes this fair. With real holes, chapter
13's cross-validation is the nearest thing to it.

## 17.4 Where the walk moved the section

The deep model's answer rests on a node nobody measured: the walk.
`predict_node` stops the model's prediction at any node of the tree and
writes what that node says into a container, as `predict` writes what the
likelihood says. Asked at the walk's start (`walk.walker`, the coordinates
the walk reads) and at the walk itself, it gives every point of the
section before and after the move, both in the units the short GP reads:
metres divided by the range its transform trained to.

```python
def walked(model, walk, target):
    """Each point of `target` where the walk starts it and where it ends."""
    ends = []
    for node in (walk.walker, walk):
        moved = model.predict_node(node, target, n_sim=1, name=node.name)
        ends.append(moved.get_predictions())
    return ends


section = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[81, 101])
start, end = walked(deep_model, walk, section)
sample_start, sample_end = walked(deep_model, walk, samples)
shift = np.linalg.norm(end - start, axis=1)
amp = float(np.asarray(walk.parameters["amp"].get_value()))
print("reach %.2f; the walk moves a point by %.2f at most, %.2f on average"
      % (walk.step * walk.n_steps * amp, shift.max(), shift.mean()))
```

```python
plan = np.asarray(section.coordinates)
inside = ore_at(plan[:, 0], plan[:, 1])


def draw_walked(axes, end):
    """The section's metre grid and the truth where the walk puts them."""
    lattice = end.reshape(101, 81, 2)
    for j in range(0, 81, 5):
        axes.plot(lattice[:, j, 0], lattice[:, j, 1], color="0.75", lw=0.5)
    for i in range(0, 101, 5):
        axes.plot(lattice[i, :, 0], lattice[i, :, 1], color="0.75", lw=0.5)
    axes.scatter(end[~inside, 0], end[~inside, 1], s=1.5, color="#9ecae1",
                 lw=0)
    axes.scatter(end[inside, 0], end[inside, 1], s=1.5, color="#d62728",
                 lw=0)
    axes.set_aspect("equal")


figure, axes = plt.subplots(1, 3, figsize=(15, 5.8))

image = axes[0].imshow(shift.reshape(101, 81), origin="lower",
                       extent=EXTENT, cmap="viridis")
every = (slice(None, None, 5), slice(None, None, 5))
arrows = (end - start).reshape(101, 81, 2)
axes[0].quiver(plan[:, 0].reshape(101, 81)[every],
               plan[:, 1].reshape(101, 81)[every],
               arrows[..., 0][every], arrows[..., 1][every], color="w",
               angles="xy", scale_units="xy", scale=0.25)
axes[0].contour(truth.astype(float), levels=[0.5], extent=EXTENT,
                colors="r", linewidths=0.8)
axes[0].set_aspect("equal")
axes[0].set_title("the displacement (arrows four times longer)")
figure.colorbar(image, ax=axes[0], shrink=0.8, label="transformed units")

draw_walked(axes[1], end)
axes[1].set_title("the section in the walked coordinates")

draw_walked(axes[2], end)
draw_holes(axes[2], sample_end)
axes[2].set_title("the holes in the walked coordinates")

figure.savefig("figures/17-walk.png", dpi=150, bbox_inches="tight")
```

![What the walk does to the section](https://italo-goncalves.github.io/geoML/_images/17-walk.png)

The walk is short: its reach trained to 0.14. On a section 3.1
transformed units wide it moves a point by 0.23 of a unit at most and
0.07 on average, and in the walked coordinates (the middle panel) the
vein keeps its shape: the walk does not unfold the fold into a flat
layer. What it does is change the spacing locally. The largest moves are
in the hinge, pulling the limbs together across the waste between them,
so the stationary GP above sees those intercepts closer than they are on
the section; the holes (right panel) bend into gentle S-shapes. The deep
model's gain is a range that changes from place to place, not a
restoration of the fold. The scores could only say that the deep model is
better; `predict_node` is what shows why.

## 17.5 How far the walk may reach

The walk's **reach**, `step * n_steps * amp`, is how far it can carry a
point: the field's strength, in transformed units. Above, the amplitude
trained and the bound chose a reach of 0.14. `GPWalk(reach=)` holds it
instead, and a long reach does not unfold the vein either. A free field
is allowed to converge: held at a full unit, the walk collapses whole
stretches of the section into a few sinks, drags the holes in with them
and stretches the ground between over the rest. Ore intercepts from
different limbs land side by side in those coordinates, and the GP above
joins them across the waste. On this section that model scored an
accuracy of 0.911 and an IoU of 0.723, broke the vein into two bodies,
and ended 85 nats below the trained walk's bound.

There the bound and the truth agree, but they need not. On three holes
instead of five, the benchmark behind this chapter found the best bound
on a walk that barely moved anything, and a better map at a lower bound.
`reach` is therefore a modelling choice, to be made on held-out data
(chapter 13). Unfolding the layer outright would take a field that cannot
converge, such as a divergence-free one built from a stream function,
and a reader longer along the layering than across it.

> **In the code.** `geoml.latent.GPWalk` is the walk, with `reach=` to
> hold its strength, and `geoml.latent.SelectInput` hands each GP its own
> columns of one input. `VGPNetwork.predict_node` writes any node's
> prediction as a `geoml.data.LatentVariable`: one part per output, with
> the node's `latent_mean`, `latent_variance` and realizations on the
> latent scale, read like any other variable. A node is predicted at
> points only; a block's value is an average a likelihood takes, and a
> node has none. Chapter 9 turns real drillhole logs into data like
> `samples`, and chapter 12 contours a refined block model into a surface.

## Further reading

The potential-field formulation is Lajaunie et al. (1997), geoML's
machine-learning reading of it is the 2017 paper, and the categorical
treatment used here is the 2023 one. The deep model is the subject of the
2025 paper, where a folded vein drilled in three dimensions is examined
at length.

## References

Gonçalves, Í. G. *et al.* (2017). A machine learning approach to the
potential-field method for implicit modeling of geological structures.
*Computers & Geosciences*.

Gonçalves, Í. G. *et al.* (2023). Variational Gaussian processes for
implicit geological modeling. *Computers & Geosciences*.
<https://linkinghub.elsevier.com/retrieve/pii/S0098300423000274>

Gonçalves, Í. G. *et al.* (2025). Uncertainty propagation in deep Gaussian
process networks. *Mathematical Geosciences*.
<https://doi.org/10.1007/s11004-025-10187-4>

Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and 3D
cartography in geology: principles of a method based on potential
interpolation. *Mathematical Geology*, 29(4), 571–584.
