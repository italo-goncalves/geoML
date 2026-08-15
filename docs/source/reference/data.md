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

`Blocks3D` is the regular block model; `BlockSet3D` is the variable-size
one, where every block's origin and size are whole numbers of a base cell
so that splitting keeps it tiling exactly. Design record:
{doc}`../internals/variable-block-models`.

```{eval-rst}
.. automodule:: geoml.data.blocks
   :members: Blocks3D, BlockSet3D, RotatedBlocks3D, RotatedBlockSet3D
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

## Variables

What a container holds at each location: measurements, and everything a
model writes back.

```{eval-rst}
.. automodule:: geoml.data.variables
   :members: ContinuousVariable, VectorVariable, CompositionalVariable,
             CategoricalVariable, RockTypeVariable, BinaryVariable,
             DerivedVariable
   :show-inheritance:
```

## Paths

How a variable's columns are named and addressed. Design record:
{doc}`../internals/variable-paths`.

```{eval-rst}
.. automodule:: geoml.data.base
   :members: VariablePath, BoundingBox, render
```
