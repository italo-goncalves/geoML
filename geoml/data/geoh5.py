# geoML - machine learning models for geospatial data
# Copyright (C) 2021  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Interchange with Mira Geoscience's geoh5 format.

A ``.geoh5`` file is a *workspace* holding any number of named objects, and
it is what Geoscience ANALYST — a free viewer — opens as **one** project,
so writing one turns that viewer into a 3D screen for geoML's surfaces,
point predictions and block models. The user-facing surface is on the
containers — ``to_geoh5``/``from_geoh5`` on the mesh classes, `PointData`
and `BlockSet3D` — each taking a path, or an open :class:`Workspace` from
here when several exports belong together; :func:`contents` lists what a
workspace holds. Interchange, not persistence: ``to_zarr`` is what keeps a
geoML container whole, and what comes back from a geoh5 file is data as
the file spells it, not the tree that produced it.

The dependency is optional: ``pip install geoml[geoh5]`` brings ``geoh5py``,
and nothing in this module is imported until one of these functions runs.
"""
import json as _json
import os as _os
from collections.abc import Sequence

import numpy as _np
import pandas as _pd

import geoml._types as _types
from geoml.data.base import METADATA_ROOT, PATH_SEP, render


def _geoh5py():
    """The optional dependency, or a message saying how to get it."""
    try:
        import geoh5py
        import geoh5py.data
        import geoh5py.groups
        import geoh5py.objects
        import geoh5py.workspace
    except ImportError as error:
        raise ImportError(
            "reading and writing geoh5 files needs the geoh5py package; "
            "install it with `pip install geoml[geoh5]`") from error
    return geoh5py


class Workspace:
    """
    An open geoh5 workspace, holding several exports together.

    A ``.geoh5`` file is what Geoscience ANALYST opens as **one**
    project, so a model's pieces — surfaces, samples, block models —
    belong in one workspace rather than a file each. Passing the same
    path to every ``to_geoh5`` already lands them together, at the price
    of opening and closing the file per call; this object opens it once
    and every ``to_geoh5`` and ``from_geoh5`` given it writes and reads
    through the open handle:

    .. code-block:: python

        with geoml.data.geoh5.Workspace("assen.geoh5") as project:
            topo.to_geoh5(project, name="topography")
            ore.to_geoh5(project, name="ore body")
            blocks.to_geoh5(project, name="block model")

    The file is created when it does not exist and appended to when it
    does. The workspace's own name, as ANALYST shows it, **is the file's
    name**: geoh5py accepts a display name at creation but does not
    persist it — measured, it reads back as "GEOSCIENCE" — so none is
    offered here; name the file.

    Parameters
    ----------
    filename : path
        The ``.geoh5`` file to open or create.
    """

    def __init__(self, filename: _types.PathLike):
        geoh5 = _geoh5py()
        self.filename = str(filename)
        if _os.path.exists(self.filename):
            self._handle = geoh5.workspace.Workspace(self.filename)
        else:
            self._handle = geoh5.workspace.Workspace.create(self.filename)
        # entity removal is deferred to `close`, so anything replaced in
        # this session still shows in the live listings; remembering the
        # ghosts is what keeps reads honest meanwhile
        self._removed = set()

    @classmethod
    def _existing(cls, filename):
        """For readers: a typo in a path must read as a typo, where
        geoh5py's own `Workspace(path)` would quietly create an empty
        file."""
        if not _os.path.exists(str(filename)):
            raise FileNotFoundError("no geoh5 file at %s" % filename)
        return cls(filename)

    def close(self) -> None:
        """Writes everything out and releases the file."""
        self._handle.close()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def _live(self):
        """The objects, without the ghosts of anything replaced."""
        return [obj for obj in self._handle.objects
                if obj.uid not in self._removed]

    def _discard(self, name, kind, parent=None):
        """What `replace=True` means: same name, same exact kind, same
        folder, gone — scoped to the folder so "hematite" under Ore and
        "hematite" under Waste can each be rewritten alone."""
        target = parent.uid if parent is not None else self._handle.root.uid
        for obj in self._live():
            if type(obj) is kind and str(obj.name) == str(name) \
                    and getattr(obj.parent, "uid", None) == target:
                self._handle.remove_entity(obj)
                self._removed.add(obj.uid)

    def contents(self) -> "dict[str, str]":
        """As `geoml.data.geoh5.contents`, on the open workspace.

        Names come folder-qualified — `Surfaces/Ore/hematite` — an
        object at the root by its bare name.
        """
        return {_qualified(obj): type(obj).__name__
                for obj in self._live()}

    def _drillhole_groups(self):
        geoh5 = _geoh5py()
        return [group for group in self._handle.groups
                if isinstance(group, geoh5.groups.DrillholeGroup)]

    def __repr__(self):
        lines = ["Workspace(%r) holding:" % self.filename]
        for name, kind in sorted(self.contents().items()):
            lines.append("    %s: %s" % (name, kind))
        for group in self._drillhole_groups():
            lines.append("    %s: %s (%d holes)"
                         % (str(group.name), type(group).__name__,
                            len(group.children)))
        if len(lines) == 1:
            lines[0] = "Workspace(%r), empty" % self.filename
        return "\n".join(lines)

    def __len__(self):
        return len(self._live())

    def __iter__(self):
        return iter(sorted(self.contents()))

    def __contains__(self, name):
        listing = self.contents()
        return str(name) in listing \
            or any(key.split("/")[-1] == str(name) for key in listing) \
            or any(str(group.name) == str(name)
                   for group in self._drillhole_groups())

    def __getitem__(self, name: str):
        """The named object, as the geoML container its kind calls for.

        A Surface classifies through `mesh3d`, Points become `PointData`,
        an Octree a `BlockSet3D`, a BlockModel a `Blocks3D` (rotated ones
        a `RotatedBlockSet3D`), and a drillhole — or a whole drillhole
        group, named as one — a `DrillholeData`. Converted on demand,
        one name at a time: a workspace can hold more than fits in
        memory at once, which is why this is not an eager dict.
        """
        # late, through the facade: the containers this dispatches to
        # live above this module
        import geoml.data as _gdata
        geoh5 = _geoh5py()
        matches = [obj for obj in self._live()
                   if str(obj.name) == str(name)
                   or _qualified(obj) == str(name)]
        if len(matches) == 0:
            if any(str(group.name) == str(name)
                   for group in self._drillhole_groups()):
                return _gdata.DrillholeData.from_geoh5(self, name)
            raise KeyError(
                "nothing named %r in %s; it holds %s"
                % (str(name), self.filename,
                   ", ".join(sorted(self.contents())) or "nothing"))
        if len(matches) > 1:
            raise ValueError(
                "%d objects answer to %r in %s: say which — %s"
                % (len(matches), str(name), self.filename,
                   ", ".join(sorted(_qualified(obj)
                                    for obj in matches))))
        kind = type(matches[0])
        if kind is geoh5.objects.Surface:
            return _gdata.Mesh3D.from_geoh5(self, name)
        if kind is geoh5.objects.Points:
            return _gdata.PointData.from_geoh5(self, name)
        if kind is geoh5.objects.Octree:
            return _gdata.BlockSet3D.from_geoh5(self, name)
        if kind is geoh5.objects.BlockModel:
            return _gdata.Blocks3D.from_geoh5(self, name)
        if isinstance(matches[0], geoh5.objects.Drillhole):
            return _gdata.DrillholeData.from_geoh5(self, name)
        raise TypeError(
            "no geoML container reads a geoh5 %s" % kind.__name__)


def _qualified(obj):
    """The object's name with its folders in front — `Surfaces/Ore/body`
    — walked up the parent chain to the root."""
    parts = [str(obj.name)]
    node = getattr(obj, "parent", None)
    while node is not None and type(node).__name__ != "RootGroup":
        parts.append(str(node.name))
        node = getattr(node, "parent", None)
    return "/".join(reversed(parts))


def _folder(wrapper, path):
    """The container group a folder path names, created where it does not
    exist yet and reused where it does — geoh5's groups are what ANALYST
    shows as folders in its project tree. `None` stays `None`: the root.
    """
    if path is None:
        return None
    geoh5 = _geoh5py()
    current = wrapper._handle.root
    for segment in str(path).split("/"):
        segment = segment.strip()
        if not segment:
            continue
        found = [group for group in wrapper._handle.groups
                 if str(group.name) == segment
                 and getattr(group.parent, "uid", None) == current.uid]
        if found:
            current = found[0]
        else:
            current = geoh5.groups.ContainerGroup.create(
                wrapper._handle, name=segment, parent=current)
    return None if current is wrapper._handle.root else current


def _borrowed(target, reading):
    """The `Workspace` behind a call, and whether the call owns it.

    A caller's own `Workspace` stays open — that is what it is for; a
    path is opened here and closed by the caller's `finally`.
    """
    if isinstance(target, Workspace):
        return target, False
    if reading:
        return Workspace._existing(target), True
    return Workspace(target), True


def contents(workspace: "_types.PathLike | Workspace") -> "dict[str, str]":
    """
    What a geoh5 workspace holds: object names against their kinds.

    Parameters
    ----------
    workspace
        Path of the workspace to look into, or an open `Workspace`.

    Returns
    -------
    listing : dict
        One entry per object, name to geoh5 type name (`"Surface"`,
        `"Points"`, `"Octree"`, ...).
    """
    wrapper, owned = _borrowed(workspace, reading=True)
    try:
        return wrapper.contents()
    finally:
        if owned:
            wrapper.close()


def _named(wrapper, name, kind):
    """The one object of `kind` the caller means, or a listing of what is
    there to mean.

    Matched on the exact type: in geoh5py a `Surface` *is a* `Points`
    (it has vertices), and reading "the points" out of a workspace must
    not hand back a surface.
    """
    found = [obj for obj in wrapper._live() if type(obj) is kind]
    if name is not None:
        found = [obj for obj in found if str(obj.name) == str(name)
                 or _qualified(obj) == str(name)]
    what = kind.__name__
    if len(found) == 0:
        listing = ", ".join(
            "%r (%s)" % (_qualified(obj), type(obj).__name__)
            for obj in wrapper._live()) or "nothing"
        raise ValueError(
            "no %s%s in %s; the workspace holds %s"
            % (what, "" if name is None else " named %r" % str(name),
               wrapper.filename, listing))
    if len(found) > 1:
        raise ValueError(
            "%d %s objects%s in %s: pass `name=` to say which — %s"
            % (len(found), what,
               "" if name is None else " named %r" % str(name),
               wrapper.filename,
               ", ".join(repr(_qualified(obj)) for obj in found)))
    return found[0]


# ------------------------------------------------------------------ #
# surfaces
# ------------------------------------------------------------------ #
def write_surface(mesh, workspace, name, replace=True, folder=None):
    """`Surface3D.to_geoh5` and its siblings land here."""
    geoh5 = _geoh5py()
    wrapper, owned = _borrowed(workspace, reading=False)
    try:
        parent = _folder(wrapper, folder)
        if replace:
            wrapper._discard(name, geoh5.objects.Surface, parent)
        geoh5.objects.Surface.create(
            wrapper._handle,
            vertices=_np.asarray(mesh.coordinates, dtype=float),
            cells=_np.asarray(mesh.triangles),
            name=str(name),
            **({"parent": parent} if parent is not None else {}))
    finally:
        if owned:
            wrapper.close()


def read_surface(workspace, name=None):
    """`Mesh3D.from_geoh5` lands here: the geometry, classified after."""
    geoh5 = _geoh5py()
    wrapper, owned = _borrowed(workspace, reading=True)
    try:
        surface = _named(wrapper, name, geoh5.objects.Surface)
        points = _np.asarray(surface.vertices, dtype=float)
        triangles = _np.asarray(surface.cells)
    finally:
        if owned:
            wrapper.close()
    return points, triangles


# ------------------------------------------------------------------ #
# points
# ------------------------------------------------------------------ #
def _as_geoh5_data(attribute):
    """One geoML column as the dict `add_data` takes.

    A coded attribute becomes *referenced* data — geoh5's own
    text-as-integer — with every code moved up by one, since geoh5
    reserves 0 for "Unknown" where geoML's missing code is -1.
    """
    if attribute.labels is not None:
        codes = _np.asarray(attribute.values).astype(_np.int64)
        return {"values": (codes + 1).astype(_np.int32),
                "value_map": {i + 1: str(label)
                              for i, label in enumerate(attribute.labels)},
                "type": "referenced"}
    values = _np.asarray(attribute.values)
    if values.dtype == bool:
        values = values.astype(float)
    return {"values": _np.asarray(values, dtype=float)}


def _payload(container, include, simulations, association=None):
    """Every exportable column as the dict `add_data` takes, plus the
    name-to-path table.

    The same enumeration every export drives off (`_export_leaves`), named
    in the `pretty` style as the pyvista export names them; none of the
    rendered names parses back, which is what the table is for.
    """
    table, payload = {}, {}
    for variable in container.variables.values():
        for path, attribute in variable._export_leaves(include, simulations):
            label = render(path, "pretty")
            payload[label] = _as_geoh5_data(attribute)
            table[label] = str(path)
    for column_name, column in container.metadata.items():
        payload[column_name] = _as_geoh5_data(column)
        table[column_name] = METADATA_ROOT + PATH_SEP + column_name
    if association is not None:
        for entry in payload.values():
            entry["association"] = association
    return payload, table


def write_points(container, workspace, name, include: str = "**",
                 simulations: "bool | int | Sequence[int]" = False,
                 replace: bool = True, folder: "str | None" = None):
    """`PointData.to_geoh5` lands here."""
    geoh5 = _geoh5py()
    wrapper, owned = _borrowed(workspace, reading=False)
    try:
        parent = _folder(wrapper, folder)
        if replace:
            wrapper._discard(name, geoh5.objects.Points, parent)
        points = geoh5.objects.Points.create(
            wrapper._handle,
            vertices=_np.asarray(container.coordinates, dtype=float),
            name=str(name),
            **({"parent": parent} if parent is not None else {}))
        payload, table = _payload(container, include, simulations)
        if payload:
            points.add_data(payload)
        if table:
            points.metadata = {"geoml_paths": _json.dumps(table)}
    finally:
        if owned:
            wrapper.close()


# ------------------------------------------------------------------ #
# block models
# ------------------------------------------------------------------ #
_OCTREE_CELLS = [("I", "<i4"), ("J", "<i4"), ("K", "<i4"),
                 ("NCells", "<i4")]


def _next_power(count):
    """The smallest power of two holding `count` cells: geoh5py refuses
    any other axis count, and cells past the model's own box are simply
    not written — partial coverage is legal, so the padding costs
    nothing."""
    return 1 << max(0, int(_np.ceil(_np.log2(max(1, int(count))))))


def write_blocks(blocks, workspace, name, include: str = "**",
                 simulations: "bool | int | Sequence[int]" = False,
                 replace: bool = True, folder: "str | None" = None):
    """`BlockSet3D.to_geoh5` lands here.

    The lattice maps one to one: a block's origin and size in base cells
    are an octree cell's `I J K NCells` as they stand, in the model's own
    row order, so the cell data rides with no reordering. The counts are
    padded to geoh5's required power of two with nothing written in the
    padding, and the rotation — geoh5 carries exactly one, counter-
    clockwise about the vertical axis about the origin — is the negated
    azimuth, geoML's rotation being the mining-convention clockwise one.
    `max_levels` rides in the object's metadata: it is refinement
    *capacity*, which the cells alone cannot say once every coarse block
    has been split.
    """
    geoh5 = _geoh5py()
    if list(blocks.discretization) != [2, 2, 2]:
        raise ValueError(
            "a geoh5 octree subdivides strictly 2x2x2 per level, and this "
            "model's discretization is %s; build the set with "
            "discretization=(2, 2, 2) — the default — to make it "
            "interchangeable" % (tuple(blocks.discretization),))
    azimuth = float(getattr(blocks, "azimuth", 0.0))
    if float(getattr(blocks, "dip", 0.0)) != 0.0 \
            or float(getattr(blocks, "rake", 0.0)) != 0.0:
        raise ValueError(
            "a geoh5 octree carries one rotation, about the vertical axis; "
            "this model dips or rakes (azimuth=%g, dip=%g, rake=%g)"
            % (azimuth, float(blocks.dip), float(blocks.rake)))

    origin = _np.asarray(blocks._origin, dtype=_np.int64)
    size = _np.asarray(blocks._size, dtype=_np.int64)
    cells = _np.zeros(len(origin), dtype=_OCTREE_CELLS)
    cells["I"], cells["J"], cells["K"] = (origin[:, 0], origin[:, 1],
                                          origin[:, 2])
    cells["NCells"] = size[:, 0]

    corner = _np.asarray(blocks._to_world(
        _np.asarray(blocks.box_corner, dtype=float)[None, :]))[0]
    step = _np.asarray(blocks.base_step, dtype=float)

    wrapper, owned = _borrowed(workspace, reading=False)
    try:
        parent = _folder(wrapper, folder)
        if replace:
            wrapper._discard(name, geoh5.objects.Octree, parent)
        tree = geoh5.objects.Octree.create(
            wrapper._handle,
            **({"parent": parent} if parent is not None else {}),
            origin=corner,
            u_count=_next_power(blocks.lattice_shape[0]),
            v_count=_next_power(blocks.lattice_shape[1]),
            w_count=_next_power(blocks.lattice_shape[2]),
            u_cell_size=float(step[0]), v_cell_size=float(step[1]),
            w_cell_size=float(step[2]), rotation=-azimuth,
            octree_cells=cells, name=str(name))
        payload, table = _payload(container=blocks, include=include,
                                  simulations=simulations,
                                  association="CELL")
        if payload:
            tree.add_data(payload)
        tree.metadata = {
            "geoml_paths": _json.dumps(table),
            "geoml_lattice": _json.dumps(
                {"max_levels": int(blocks.max_levels)})}
    finally:
        if owned:
            wrapper.close()


def read_blocks(workspace, name=None):
    """`BlockSet3D.from_geoh5` lands here: the octree as plain arrays.

    Negative cell sizes — a workspace with its origin at the top and the
    w axis running down is common — are normalized here: the axis is
    flipped to a positive step, the origin moved to the true low corner,
    and the cell indices re-counted from it. A flipped horizontal axis
    under a rotation would compose a reflection into the rotation, which
    no rotated block model can hold, and is refused.
    """
    geoh5 = _geoh5py()
    wrapper, owned = _borrowed(workspace, reading=True)
    try:
        tree = _named(wrapper, name, geoh5.objects.Octree)
        counts = _np.array([int(tree.u_count), int(tree.v_count),
                            int(tree.w_count)], dtype=_np.int64)
        step = _np.array([float(tree.u_cell_size), float(tree.v_cell_size),
                          float(tree.w_cell_size)], dtype=float)
        corner = _np.asarray(tree.origin.tolist(), dtype=float)
        rotation = float(tree.rotation or 0.0)
        raw = _np.asarray(tree.octree_cells.tolist(), dtype=_np.int64)
        origin, size = raw[:, :3].copy(), raw[:, 3].copy()

        for axis in range(3):
            if step[axis] >= 0:
                continue
            if rotation != 0.0 and axis < 2:
                raise ValueError(
                    "%s runs its %s axis backwards under a rotation, which "
                    "composes a reflection no rotated block model can "
                    "hold" % (wrapper.filename, "uv"[axis]))
            # a plain world shift either way: horizontal flips only reach
            # here unrotated, and a rotation about z never touches z
            offset = _np.zeros(3)
            offset[axis] = counts[axis] * step[axis]
            corner = corner + offset
            step[axis] = -step[axis]
            origin[:, axis] = counts[axis] - origin[:, axis] - size

        floats, coded = [], []
        for data in tree.children:
            values = getattr(data, "values", None)
            if values is None or len(_np.shape(values)) != 1 \
                    or len(values) != len(origin):
                continue
            if isinstance(data, geoh5.data.ReferencedData):
                mapping = _decoded_map(data)
                labels = [mapping[key] for key in sorted(mapping)]
                decoded = _np.array(
                    [mapping.get(int(code), "") for code in values],
                    dtype=object)
                coded.append((str(data.name), labels, decoded))
            elif isinstance(data, geoh5.data.FloatData):
                floats.append((str(data.name),
                               _np.asarray(values, dtype=float)))

        lattice = {}
        metadata = tree.metadata or {}
        if "geoml_lattice" in metadata:
            lattice = _json.loads(metadata["geoml_lattice"])
    finally:
        if owned:
            wrapper.close()
    return {"origin": origin, "size": size, "counts": counts,
            "step": step, "corner": corner, "rotation": rotation,
            "floats": floats, "coded": coded, "lattice": lattice,
            "filename": wrapper.filename}


def _to_geoh5_order(values, shape):
    """geoML's grid order (x slowest) to geoh5's BlockModel order (u
    fastest) — the same transposition the pyvista export makes."""
    return _np.asarray(values).reshape(tuple(shape)).transpose(2, 1, 0) \
        .ravel()


def _from_geoh5_order(values, shape):
    """The inverse of `_to_geoh5_order`."""
    return _np.asarray(values).reshape(
        (shape[2], shape[1], shape[0])).transpose(2, 1, 0).ravel()


def write_grid_blocks(blocks, workspace, name, include: str = "**",
                      simulations: "bool | int | Sequence[int]" = False,
                      replace: bool = True, folder: "str | None" = None):
    """`Blocks3D.to_geoh5` lands here: a uniform model as a BlockModel.

    A geoh5 BlockModel is a tensor grid — per-axis edge positions from
    its origin — which a uniform model fills with equal steps. Cell data
    is stored u-fastest where geoML's grids run x slowest, so every
    column is transposed on the way through, values and order agreeing
    with what geoh5py's own centroids say.
    """
    geoh5 = _geoh5py()
    shape = _np.asarray(blocks.grid_size, dtype=_np.int64)
    step = _np.asarray(blocks.step_size, dtype=float)
    corner = _np.asarray(blocks.coordinates, dtype=float).min(axis=0) \
        - step / 2.0

    wrapper, owned = _borrowed(workspace, reading=False)
    try:
        parent = _folder(wrapper, folder)
        if replace:
            wrapper._discard(name, geoh5.objects.BlockModel, parent)
        model = geoh5.objects.BlockModel.create(
            wrapper._handle,
            **({"parent": parent} if parent is not None else {}),
            origin=corner,
            u_cell_delimiters=_np.arange(shape[0] + 1) * step[0],
            v_cell_delimiters=_np.arange(shape[1] + 1) * step[1],
            z_cell_delimiters=_np.arange(shape[2] + 1) * step[2],
            rotation=0.0, name=str(name))
        payload, table = _payload(blocks, include, simulations,
                                  association="CELL")
        for entry in payload.values():
            entry["values"] = _to_geoh5_order(entry["values"], shape)
        if payload:
            model.add_data(payload)
        if table:
            model.metadata = {"geoml_paths": _json.dumps(table)}
    finally:
        if owned:
            wrapper.close()


def read_grid_blocks(workspace, name=None):
    """`Blocks3D.from_geoh5` lands here: a BlockModel as plain arrays.

    Only a uniform spacing has a geoML container — a true tartan grid is
    refused with its uneven axis named — and the cell data comes back in
    geoML's own order.
    """
    geoh5 = _geoh5py()
    wrapper, owned = _borrowed(workspace, reading=True)
    try:
        model = _named(wrapper, name, geoh5.objects.BlockModel)
        corner = _np.asarray(model.origin.tolist(), dtype=float)
        rotation = float(model.rotation or 0.0)
        step, shape, offset = [], [], []
        for axis, edges in enumerate((model.u_cell_delimiters,
                                      model.v_cell_delimiters,
                                      model.z_cell_delimiters)):
            widths = _np.diff(_np.asarray(edges, dtype=float))
            if len(widths) == 0 or _np.any(widths <= 0):
                raise ValueError(
                    "%s's %s axis is empty or runs backwards"
                    % (wrapper.filename, "uvz"[axis]))
            if not _np.allclose(widths, widths[0]):
                raise ValueError(
                    "%s is a tartan block model — its %s axis spacing "
                    "runs from %g to %g — and geoML has no container "
                    "for uneven blocks"
                    % (wrapper.filename, "uvz"[axis], widths.min(),
                       widths.max()))
            step.append(float(widths[0]))
            shape.append(len(widths))
            # the first edge is a *local* offset from the origin: it must
            # turn with the rotation, so it travels separately
            offset.append(float(_np.asarray(edges, dtype=float)[0]))
        step, shape = _np.asarray(step), _np.asarray(shape, dtype=_np.int64)
        offset = _np.asarray(offset)

        floats, coded = [], []
        n_cells = int(shape.prod())
        for data in model.children:
            values = getattr(data, "values", None)
            if values is None or len(_np.shape(values)) != 1 \
                    or len(values) != n_cells:
                continue
            if isinstance(data, geoh5.data.ReferencedData):
                mapping = _decoded_map(data)
                labels = [mapping[key] for key in sorted(mapping)]
                decoded = _np.array(
                    [mapping.get(int(code), "") for code in values],
                    dtype=object)
                coded.append((str(data.name), labels,
                              _from_geoh5_order(decoded, shape)))
            elif isinstance(data, geoh5.data.FloatData):
                floats.append((str(data.name), _from_geoh5_order(
                    _np.asarray(values, dtype=float), shape)))
    finally:
        if owned:
            wrapper.close()
    return {"corner": corner, "step": step, "shape": shape,
            "offset": offset, "rotation": rotation, "floats": floats,
            "coded": coded, "filename": wrapper.filename}


def read_drillholes(workspace, name=None):
    """`DrillholeData.from_geoh5` lands here: collars, surveys and the
    interval tables, as plain frames.

    geoh5 stores one object per hole, its interval data as FROM/TO
    columns inside named *property groups*; the group names become the
    table names, gathered across every hole. Depth-associated data — a
    reading at a point down the hole rather than over an interval — has
    no place in an interval table and is left out. `name` narrows to one
    drillhole group, or to one hole.
    """
    geoh5 = _geoh5py()
    wrapper, owned = _borrowed(workspace, reading=True)
    try:
        holes = [obj for obj in wrapper._live()
                 if isinstance(obj, geoh5.objects.Drillhole)]
        if name is not None:
            named = [hole for hole in holes
                     if str(hole.name) == str(name)
                     or str(getattr(hole.parent, "name", "")) == str(name)]
            if len(named) == 0:
                raise ValueError(
                    "no drillhole or drillhole group named %r in %s; the "
                    "workspace holds %s"
                    % (str(name), wrapper.filename,
                       ", ".join(sorted(str(hole.name)
                                        for hole in holes)) or "no holes"))
            holes = named
        if len(holes) == 0:
            raise ValueError(
                "no drillholes in %s; the workspace holds %s"
                % (wrapper.filename,
                   ", ".join(sorted(wrapper.contents())) or "nothing"))

        collar_rows, survey_rows, tables = [], [], {}
        for hole in holes:
            hole_id = str(hole.name)
            x, y, z = _np.asarray(hole.collar.tolist(), dtype=float)
            collar_rows.append((hole_id, x, y, z))
            surveys = _np.asarray(hole.surveys.tolist(), dtype=float)
            for depth, azimuth, dip in surveys.reshape(-1, 3):
                survey_rows.append((hole_id, depth, azimuth, dip))

            by_uid = {child.uid: child for child in hole.children}
            for group in (hole.property_groups or []):
                columns = {}
                bounds = {}
                for uid in group.properties:
                    child = by_uid.get(uid)
                    if child is None:
                        continue
                    label = str(child.name)
                    if label in ("FROM", "TO"):
                        bounds[label] = _np.asarray(child.values,
                                                    dtype=float)
                    elif isinstance(child, geoh5.data.ReferencedData):
                        mapping = _decoded_map(child)
                        columns[label] = _np.array(
                            [mapping.get(int(code), "")
                             for code in child.values], dtype=object)
                    elif hasattr(child, "values") \
                            and child.values is not None:
                        columns[label] = _np.asarray(child.values,
                                                     dtype=float)
                if "FROM" not in bounds or "TO" not in bounds \
                        or not columns:
                    continue
                frame = _pd.DataFrame({"HOLEID": hole_id,
                                       "FROM": bounds["FROM"],
                                       "TO": bounds["TO"], **columns})
                tables.setdefault(str(group.name), []).append(frame)
    finally:
        if owned:
            wrapper.close()

    collar = _pd.DataFrame(collar_rows, columns=["HOLEID", "X", "Y", "Z"])
    survey = _pd.DataFrame(survey_rows,
                           columns=["HOLEID", "DEPTH", "AZIMUTH", "DIP"])
    gathered = {label: _pd.concat(frames, ignore_index=True)
                for label, frames in tables.items()}
    return collar, survey, gathered


def _decoded_map(data):
    """A referenced column's `{code: label}`, whatever geoh5py wrapped it
    in — the representation has moved between releases — with geoh5's own
    "Unknown" zero left out, since it is that format's missing code."""
    value_map = getattr(data.value_map, "map", data.value_map)
    decoded = {}
    for key, label in _np.asarray(value_map).tolist():
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        if int(key) != 0:
            decoded[int(key)] = str(label)
    return decoded


def read_points(workspace, name=None):
    """`PointData.from_geoh5` lands here.

    Returns the vertices and the vertex-associated columns: float data as
    `(name, values)` and referenced data as `(name, labels, decoded)`,
    for the classmethod to turn into continuous and categorical
    variables.
    """
    geoh5 = _geoh5py()
    wrapper, owned = _borrowed(workspace, reading=True)
    try:
        points = _named(wrapper, name, geoh5.objects.Points)
        vertices = _np.asarray(points.vertices, dtype=float)
        floats, coded = [], []
        for data in points.children:
            values = getattr(data, "values", None)
            if values is None or len(_np.shape(values)) != 1 \
                    or len(values) != len(vertices):
                continue
            if isinstance(data, geoh5.data.ReferencedData):
                mapping = _decoded_map(data)
                labels = [mapping[key] for key in sorted(mapping)]
                decoded = _np.array(
                    [mapping.get(int(code), "") for code in values],
                    dtype=object)
                coded.append((str(data.name), labels, decoded))
            elif isinstance(data, geoh5.data.FloatData):
                floats.append((str(data.name),
                               _np.asarray(values, dtype=float)))
    finally:
        if owned:
            wrapper.close()
    return vertices, floats, coded
