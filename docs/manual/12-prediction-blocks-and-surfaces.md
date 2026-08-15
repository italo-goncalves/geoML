# 12. Prediction, blocks and surfaces

A model's output is not a map, it is a container full of columns. This
chapter is the road from those columns to the objects a mining workflow
actually exchanges: a refined block model, a grade shell or domain
boundary as a triangulated surface, volumes and tonnages cut against
geometry, and files other software opens.

The running example rebuilds chapter 8's ore pod, refined.

```python
import os
import tempfile

import geoml
import numpy as np

geoml.set_seed(1234)
rng = np.random.default_rng(1234)

xyz = rng.uniform(0, 160, size=[500, 3])
radius = np.linalg.norm(xyz - np.array([80.0, 80.0, 80.0]), axis=1)

point = geoml.data.PointData.from_array(xyz)
point.add_continuous_variable(
    "au", 4.0 * np.exp(-(radius / 28.0) ** 2) + 0.02)
point.get("au").set_cutoffs([1.0])

inducing = geoml.data.inducing.from_kmeans(point, 150, seed=0)

root = geoml.latent.BasicInput(
    inducing,
    transform=geoml.transform.Isotropic(22.0))

gp = geoml.latent.BasicGP(
    root,
    size=1,
    kernel=geoml.kernels.Gaussian())

warping = geoml.warping.ChainedWarping(
    geoml.warping.Softplus(1),
    geoml.warping.ZScore(1))

model = geoml.models.VGPNetwork(
    point, "au",
    geoml.likelihood.Gaussian(warping),
    gp,
    options=geoml.models.GPOptions(verbose=False))

model.train_full(max_iter=150)

blocks = geoml.data.BlockSet3D(
    start=[10, 10, 5],
    n=[8, 8, 16],
    step=[20.0, 20.0, 10.0],
    discretization=(2, 2, 2),
    max_levels=2)

blocks = geoml.models.refine(model, blocks, n_sim=20)
print(blocks.n_data, "blocks after refinement")
```

## 12.1 From a field to a surface

Any contoured value is a surface: a grade shell at a cut-off, a domain
boundary at a categorical's log-odds zero (chapter 6), a topography at an
elevation. On a block model the extraction is careful in a way worth
knowing about. A coarse block beside finer neighbours would tear the
surface along their shared face, so the mesh handed to the contouring is
locally cut down to the finest size, in the export only and never in the
model. The result is smoother *and* closer to the true level set than a
uniformly fine model several times the size.

```python
shell = blocks.get_contour("au", 1.0)

print(type(shell).__name__,
      "closed:", shell.closed,
      "volume:", round(float(shell.volume), 0))
```

What comes back is a real mesh object with invariants rather than a soup
of triangles. `Surface3D` never closes, `Solid3D` always closes and knows
its `volume`, and a mesh that satisfies neither is refused rather than
passed along quietly. Solids support `union`, `intersection` and
`difference`, with a robust fallback where exact geometry fails, which
warns and states its resolution when it is used. `simplify(max_error)`
decimates against a geometric budget it actually verifies, and `from_dxf`
and `export_dxf` exchange geometry with the rest of the mining world.

## 12.2 Geometry cutting data

The opposite direction, geometry deciding about locations, is one call on
any container: `assign_from_surface` for above or below a sheet, and
`assign_from_solid` for inside or outside a body, each writing a metadata
column. Blocks add `fraction=`, the share of each block inside, measured
on the sub-blocks the model already discretizes into.

```python
blocks.assign_from_solid(shell, "shell", fraction="share")

share = blocks.get_metadata("share")
inside = float((blocks.block_volume * share).sum())

print("volume inside the shell:", round(inside, 0),
      "of", round(float(shell.volume), 0))
```

The two volumes agree to a couple of percent, which is the resolution of
the test rather than a disagreement about the shape: the block-side number
asks eight sub-blocks per block whether they are inside, while the mesh
knows its own volume exactly. Refine further and the two converge. The
second number is the one to quote when the geometry is the reference, and
the point of computing both is that they *check each other*. The same
calls put a topography or a lease boundary into the workflow: assign, then
hand the mask to `refine(..., where=...)` so that ground nobody asked
about is never predicted at all.

## 12.3 Leaving the package

Three doors, by destination:

```python
frame = blocks.as_data_frame()
print(list(frame.columns)[:6])

mesh = blocks.as_pyvista()
print(mesh.n_cells, "hexahedra for a 3D viewer")

out = os.path.join(tempfile.mkdtemp(), "shell.dxf")
shell.export_dxf(out)
print("dxf written:", os.path.getsize(out) > 0)
```

`as_data_frame` gives one row per block with a size per row, which reaches
spreadsheets and general software. `as_pyvista` reaches 3D viewers and
VTK-based pipelines, with every column travelling as cell data and its
path carried alongside. DXF reaches CAD and the mine-planning packages,
geometry only. And `to_zarr` (chapter 10) remains the lossless hand-off
between geoML scripts: the others are exports, this one is the container
itself.

> **In the code.** `BlockSet3D.get_contour` and the mesh hierarchy in
> `data/meshes.py`, with `topo.clip_meshes` for batch topography cuts;
> assignments in `data/containers.py` and `data/blocks.py`; the pure mesh
> mathematics in `math/geometry.py`. Design record:
> `docs/variable-block-models.md`.

## Further reading

Chapter 8 for what the blocks contain, chapter 6 for boundaries as level
sets, and chapter 17 for topography, domains and grade shells on a real
deposit at once.
