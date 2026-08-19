# geoml.data.geoh5

Interchange with Mira Geoscience's geoh5 format — the workspace files
Geoscience ANALYST, a free viewer, opens as projects. The everyday surface
is on the containers themselves (`to_geoh5`/`from_geoh5` on the mesh
classes, `PointData`, `Blocks3D` and `BlockSet3D`, and
`DrillholeData.from_geoh5` for reading a drillhole database); this module
holds the workspace handle those methods share, and the listing helper.

The dependency is an optional extra: `pip install geoml[geoh5]`.

```{eval-rst}
.. automodule:: geoml.data.geoh5
   :members: Workspace, contents
```
