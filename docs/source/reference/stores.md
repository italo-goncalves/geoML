# Store formats

What geoML writes to disk, for programs that read it without geoML: a
container (`to_zarr`, read back by `open`) and a mesh set
(`MeshSet.to_zarr`, read back by `MeshSet.open`). Both are Zarr version 3
groups written by zarr-python 3 with its default codecs, `bytes` then
`zstd` today. Read each array's dtype, chunk shape and codecs from its own
metadata rather than assuming them.

## Attributes and versions

geoML writes its root attributes under keys starting with `geoml`: `geoml`
on a container, `geoml_meshset` on a mesh set and `geoml_model` on a saved
model. Every other root attribute belongs to whoever wrote it. Rewriting a
store keeps those and replaces the arrays and groups.

Each store records a format number. It changes when a reader of the old
layout would misread the new one: a group moved or renamed, a field removed,
or a field given another meaning. An added field keeps the number, so a
reader ignores the fields it does not know. JSON has no NaN; a missing
number is written as `null`.

## A mesh

Every mesh is a container of its own, which `Solid3D.open` reads alone.

| Array | Content |
|---|---|
| `_coordinates` | `float64`, `(n, 3)`: the vertices |
| `_triangles` | integer, `(m, 3)`: each triangle's vertex indices |
| `_normals` | `float64`, `(n, 3)`: one normal per vertex |

A `Solid3D`'s triangles face outwards: counter-clockwise seen from outside.

The root attribute `geoml` holds `geoml_format` (2), `container`, and
`metadata` and `variables`, both empty for a mesh. `container["class"]` is
`Solid3D`, `Surface3D`, `DTM3D` or `Mesh3D`. `container["provenance"]`, when
present, records how the mesh was made. In a mesh set it holds:

- `source`, the contoured column's path;
- `value`, the level contoured, and `key`, the set's key;
- `close`, `supersample` and `simplify`, as below;
- `limits` and `exclude`, the cuts applied;
- `nudge`, how far off its level the contour was retried, zero when it
  closed at the level itself;
- `realization`, null for the prediction's body.

## A mesh set

Format 1: `geoml_meshset["format"]`.

```
<store>/
    prediction/<group>/          the prediction's body for each key
    simulations/<n>/<group>/     realization n's body for each key
    limits/<name>/               each limit, as given
    excluded/<name>/             each exclusion, as given
```

Every leaf group is a mesh. `<group>` is a key's group name, listed in
`groups` in the order of `keys`. `<n>` is a realization number from
`numbers`, written in decimal. A realization missing from `numbers` was not
stored; a body that could not be made is listed in `failures`.

The root attribute `geoml_meshset` holds:

| Field | Meaning |
|---|---|
| `format` | The layout's number, 1 |
| `kind` | `"cutoff"` for grade shells, `"category"` for one body per category |
| `path` | The contoured column's path in its container |
| `name`, `unit` | The variable's name, and its unit or null |
| `keys` | The cut-offs, as numbers, or the category names, in order |
| `groups` | Each key's group name, in the order of `keys` |
| `levels` | The level each key was contoured at |
| `close` | The side of its level a body keeps: `"above"` or `"below"` |
| `supersample`, `simplify` | The contour's extra refinement levels, and its error budget or null |
| `rule` | How a categorical realization picks its winner; null for grades |
| `limits`, `excluded` | The names of the limits and exclusions |
| `provenance` | The set's own provenance |
| `realization` | Null for a whole set; a number for one realization written alone |
| `numbers` | The realizations stored |
| `summary` | The prediction's bodies, one value per key: `volume`, `raw` (the volume before any cut), `pieces`, `largest` (the largest piece's share of the volume) and `triangles`; plus `taken`, the volume each cut removed, one list per cut |
| `measures` | The same for every realization: one table per name among `volume`, `raw`, `pieces`, `largest`, `triangles`, `gained`, `lost` and `nudge`, a row per entry of `numbers` and a column per key; `gained` and `lost` are the volume gained and lost against the prediction's body |
| `taken` | The volume each cut removed from each realization's bodies, indexed `[cut][realization][key]`, the cuts being the limits and then the exclusions |
| `nudge` | Per key, how far off its level the prediction's body was retried |
| `failures` | `{"realization", "key", "error"}` for each body that could not be made |
| `repairs` | `{"keys", "values"}`, the volume `repair` removed per key, or null |
| `corners`, `shift` | The model's box and the frame the cuts were computed in; a reader needs neither |

Stores written before geoML 0.6.13 lack `groups`. Their group names spell
each cut-off as Python's `str(float(key))` does, as in `30.0` or `1e-05`,
and each category by its name, unless the name reads as a number.

## A container

Format 2: `geoml["geoml_format"]`. The root attribute `geoml` holds
`geoml_format`, `container`, `metadata` and `variables`. `container` names
the class and what rebuilds its geometry: a point cloud's coordinates are
the array `_coordinates`, and a grid's are generated from its `start`, `n`
and `step`. Each metadata column and each variable's column gives the
array holding it under `key`, such as `_metadata/HOLEID` or
`zn/prediction`, and a column of text is stored as integer codes into its
`labels`. Follow the keys rather than building the paths. Stores at format
1 are not read.
