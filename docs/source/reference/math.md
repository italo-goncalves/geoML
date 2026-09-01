# geoml.math

Arrays in, arrays out: the geometry the containers and meshes are built on,
and the TensorFlow helpers the models use. Nothing here holds a container.

## Geometry

Rotations and angles, the triangulated-surface predicates behind
{class}`~geoml.data.Surface3D` and the assignments, the sub-block lattice
arithmetic, and cell declustering.

```{eval-rst}
.. automodule:: geoml.math.geometry
   :members: rotation_matrix, rotation_matrix_from_points,
             angles_from_rotation_matrix, azimuth_from_xy, dip_from_vec,
             vector_product, bounding_box, declustering_weights,
             weld, fan_triangulation, vertex_normals, area, components,
             single_valued, open_edges, reversed_edges, signed_volume,
             sheet_interpolator, sheet_elevation, inside_solid,
             sub_block_index, unit_sub_grid, trilinear_weights
```

## TensorFlow helpers

```{eval-rst}
.. automodule:: geoml.math.tf
   :members: pairwise_dist, ensure_rank_2, batched_dataset, training_step,
             silence_retracing_notices
```

## Interpolation

```{eval-rst}
.. automodule:: geoml.math.interpolate
   :members: CubicConv1D, CubicConv2DSeparable, CubicConv3DSeparable,
             CubicConvND, CubicSpline, MonotonicCubicSpline,
             MonotonicRationalQuadraticSpline
   :show-inheritance:
```
