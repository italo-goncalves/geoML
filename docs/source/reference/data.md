# geoml.data

The containers, the variables they hold, and the geometry they are cut
against. Everything is addressed by **tree path** —
`container.values("assay/Zn/prediction")` — rather than by attribute chain.

## Point containers

```{eval-rst}
.. automodule:: geoml.data.containers
   :members: PointData, GaussianData, DirectionalData, Section3D
   :show-inheritance:
```

## Grids

```{eval-rst}
.. automodule:: geoml.data.grids
   :members: Grid1D, Grid2D, Grid3D, GridND, RotatedGrid3D
   :show-inheritance:
```

## Blocks

`Blocks3D` is the regular block model and `RotatedBlocks3D` the same one
turned; `BlockSet3D` is the variable-size one, where every block's origin
and size are whole numbers of a base cell so that splitting keeps it tiling
exactly. `BlockSet3D.as_blocks3d` hands a refined model back as a regular
one at its coarsest level, a gathered block averaged from its parts by
volume. Design record: {doc}`../internals/variable-block-models`.

```{eval-rst}
.. automodule:: geoml.data.blocks
   :members: Blocks3D, RotatedBlocks3D, BlockSet3D, RotatedBlockSet3D
   :show-inheritance:
```

## Meshes

```{eval-rst}
.. automodule:: geoml.data.meshes
   :members: Mesh3D, Surface3D, Solid3D, DTM3D, mesh3d,
             NotClosedError, InconsistentMeshError, NotSingleValuedError,
             MeshTypeError
   :show-inheritance:
```

## Mesh sets

Every contour of one column at once: a block model contoured at each of a
variable's cut-offs, or once per category, for the prediction and for each
realization, as one read-only mapping -- `shells[0.5]`,
`shells["BIF"]`, `shells.simulations[4][0.5]`. The set cuts its meshes to
limits, checks that shells nest and categories do not overlap, and
measures volumes, bands, tonnage and how the realizations' volumes spread.
Design record: {doc}`../internals/mesh-sets`.

```{eval-rst}
.. automodule:: geoml.data.meshsets
   :members: MeshSet
   :show-inheritance:
```

## Variables

What a container holds at each location: measurements, and everything a
model writes back.

A variable may declare what it is measured in — `unit="g/t"`, `unit="%"`,
or a number to divide by. On a variable a model reads directly the unit is
a label: it names an axis and rides an export, and no number changes. On a
part of a `CompositionalVariable` it is also the divisor, because parts in
different units cannot be added up and a composition is defined by its sum;
there the parts are stored, predicted and simulated in the units they were
assayed in, becoming fractions of the whole only where the model reads and
writes them. `UNITS` is the table of names the package can divide by.

```{eval-rst}
.. automodule:: geoml.data.variables
   :members: ContinuousVariable, VectorVariable, CompositionalVariable,
             CategoricalVariable, RockTypeVariable, BinaryVariable,
             DerivedVariable, UNITS
   :show-inheritance:
```

## Paths

How a variable's columns are named and addressed. Design record:
{doc}`../internals/variable-paths`.

```{eval-rst}
.. automodule:: geoml.data.base
   :members: VariablePath, BoundingBox, render
```
