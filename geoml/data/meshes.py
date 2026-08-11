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
Triangulated meshes: `Mesh3D` the primitive, `Surface3D` and `Solid3D` as
siblings, `DTM3D` the terrain, `mesh3d` picking by geometry, the booleans,
the DXF round trip, and the adapters every container's assignments read
(`_sheet_interpolator`, `_closed_body`, `_side_codes`). The arithmetic
itself is `geoml.math.geometry`; what lives here is what touches a
container or holds an error message.
"""
import numpy as _np
import pandas as _pd
import pyvista as _pv
import ezdxf as _ezdxf
from ezdxf.render import MeshVertexMerger as _MeshVertexMerger

import geoml.math.geometry as _gmt
from geoml.math.geometry import bounding_box

from geoml.data.base import *
from geoml.data.containers import *
from geoml.data.containers import _PointBased

def _sheet_interpolator(surface):
    """Prepares a surface to be asked its elevation, refusing a closed body.

    A closed body stands at two heights over most of its footprint, so it has
    no elevation in the sense meant here -- and `matplotlib` takes such a
    triangulation without complaint, answering with whichever of the two
    sheets it happens to find, which is why this is checked before it is
    handed over.
    """
    if surface.closed:
        raise ValueError(
            "this surface is closed -- every edge belongs to two triangles -- "
            "so there is no single elevation above a location, and 'above' "
            "and 'below' do not describe it; use assign_from_solid for a body")

    return _gmt.sheet_interpolator(
        _np.asarray(surface.coordinates, dtype=float), surface.triangles)


def _uncovered_rule(uncovered):
    """Splits the `uncovered` argument into refuse-or-not and a fill value."""
    if isinstance(uncovered, str):
        if uncovered != "raise":
            raise ValueError(
                "uncovered takes 'raise', or the value to record where the "
                "sheet does not reach; got %r" % uncovered)
        return True, _np.nan
    return False, float(uncovered)


def _check_covered(elevation, uncovered):
    """Refuses a sheet that leaves locations out, when asked to.

    Returns the value to record for those locations, which is of no use where
    there is nothing numeric to record it in -- only a block model's fraction
    column has room for it, the flag being empty either way.
    """
    refuse, fill = _uncovered_rule(uncovered)
    if refuse:
        missing = int(_np.sum(_np.isnan(elevation)))
        if missing > 0:
            raise ValueError(
                "the surface does not reach %d of the %d locations; pass "
                "uncovered=0.0, or numpy.nan, to record those as unknown "
                "rather than refuse them" % (missing, elevation.shape[0]))
    return fill


def _side_codes(height, elevation):
    """0 above the sheet, 1 below it, -1 where the sheet does not reach.

    The comparison is false wherever the elevation is NaN, so those start out
    as "above" and are corrected; -1 is what the metadata layer reads as
    missing, and decodes to the empty string.
    """
    codes = _np.where(height < elevation, 1, 0).astype(_np.int8)
    codes[_np.isnan(elevation)] = -1
    return codes


def _closed_body(solid):
    """A body as a pyvista mesh, refusing one that is not watertight.

    Checked here rather than left to `select_interior_points`, which would
    repeat the check for every chunk of a block model's sub-blocks.
    """
    if not solid.closed:
        open_edges = _gmt.open_edges(solid.coordinates, solid.triangles)
        raise ValueError(
            "this surface is not closed -- %d of its edges belong to a single "
            "triangle, so it is a sheet, or a body with a face missing -- and "
            "it has no inside to test for; use assign_from_surface for a "
            "sheet" % open_edges)

    # A closed body still says nothing about which side is out unless its
    # triangles agree, and the test reads a disagreement as a hole rather
    # than as an error, so it is caught here instead.
    if not solid.consistent:
        reversed_edges = _gmt.reversed_edges(solid.coordinates,
                                             solid.triangles)
        raise ValueError(
            "this surface is closed, but its triangles disagree about which "
            "way is out -- %d of its edges are walked the same way round by "
            "both triangles sharing them -- so part of it would be read as a "
            "hole; reverse the offending triangles, or rebuild the surface "
            "from as_pyvista().compute_normals(consistent_normals=True, "
            "auto_orient_normals=True)" % reversed_edges)

    # A body can be closed and consistent and still be wound inwards, which
    # the test answers the exact complement of. Nothing is ambiguous about
    # one of those -- only its idea of "out" is reversed -- so it is turned
    # round rather than refused. The variables are left behind with it: the
    # test wants the shape and nothing else.
    triangles = _np.asarray(solid.triangles)
    if solid._signed_volume < 0:
        triangles = triangles[:, ::-1]

    faces = _np.concatenate(
        [_np.full([triangles.shape[0], 1], 3, int), triangles], axis=1)
    return _pv.PolyData(_np.asarray(solid.coordinates, dtype=float),
                        faces.ravel())


class Mesh3D(_PointBased):
    """
    A triangulated surface: vertices, the triangles indexing them, normals.

    The primitive `Surface3D` and `Solid3D` are built on, and the only one of
    the three that promises nothing about its shape — which is what a mesh
    must be allowed to be while it is still being repaired. What it does do
    is measure itself as it is built, so that everything downstream can ask
    rather than work it out again: `area`, and whether it is `closed` and
    `consistent`. Those cost a few milliseconds on a mesh of tens of
    thousands of triangles.

    `mesh3d(points, triangles, normals)` builds whichever of the three the
    geometry calls for, and is what the readers use.

    Attributes
    ----------
    area : float
        The surface area, whether or not the mesh closes.
    closed : bool
        Whether every edge is shared by two triangles, so that the mesh
        bounds a volume. Vacuously true of an empty mesh.
    consistent : bool
        Whether the triangles agree about which way is out. A closed mesh
        that is not consistent bounds nothing that can be tested.
    """

    def __init__(self, points, triangles, normals):
        super().__init__()

        if points.shape[1] != 3:
            raise ValueError("points must be an array with 3 columns")
        if triangles.shape[1] != 3:
            raise ValueError("triangles must be an array with 3 columns")
        if normals.shape[1] != 3:
            raise ValueError("normals must be an array with 3 columns")

        self.coordinates = points
        self.triangles = triangles
        self.normals = normals

        self._n_dim = 3
        self._n_data = self.coordinates.shape[0]
        if self._n_data > 0:
            self._bounding_box = BoundingBox.from_array(self.coordinates)
        else:
            self._bounding_box = BoundingBox.from_array(
                _np.zeros([2, self.n_dim]))

        self.area = _gmt.area(points, triangles)
        self.closed = _gmt.open_edges(points, triangles) == 0
        self.consistent = _gmt.reversed_edges(points, triangles) == 0
        self._signed_volume = _gmt.signed_volume(points, triangles)

    def _polydata(self):
        """The bare geometry as a pyvista mesh, carrying no variables."""
        triangles = _np.asarray(self.triangles)
        faces = _np.concatenate(
            [_np.full([triangles.shape[0], 1], 3, int), triangles], axis=1)
        return _pv.PolyData(_np.asarray(self.coordinates, dtype=float),
                            faces.ravel())

    def split(self):
        """
        The mesh's connected pieces, each as an object of its own.

        A boolean operation readily answers with a body in several pieces —
        an ore shell cut in two by a fault — and each piece is a body in its
        own right, while together they are still one legitimate mesh. This is
        how to take them apart; each piece comes back as whichever class its
        own geometry calls for.

        Returns
        -------
        pieces : list
            One mesh per connected piece, longest-standing order. A mesh
            already in one piece returns `[self]`.
        """
        count, labels = _gmt.components(self.coordinates, self.triangles)
        if count <= 1:
            return [self]

        points = _np.asarray(self.coordinates, dtype=float)
        triangles = _np.asarray(self.triangles)
        normals = _np.asarray(self.normals)

        pieces = []
        for piece in range(count):
            keep = labels == piece
            if not keep.any():
                continue
            used = _np.unique(triangles[keep])
            index = _np.zeros(points.shape[0], dtype=int)
            index[used] = _np.arange(used.size)
            pieces.append(mesh3d(points[used], index[triangles[keep]],
                                 normals[used]))
        return pieces

    def heal(self, hole_size=None):
        """
        A repaired copy of this mesh.

        Three things are put right, in the order that works: coincident
        vertices are welded, so that seams stop reading as boundaries; holes
        smaller than `hole_size` are covered over; and the triangles are made
        to agree about which way is out, then turned to face outward. That
        last step is not optional — filling a hole leaves the new triangles
        wound however they came, which would leave the mesh closed and still
        untestable.

        What comes back is whichever class the repaired geometry calls for,
        which may be the same one, and may be an empty `Mesh3D` if nothing
        survived. Healing is not guaranteed: a mesh with a hole larger than
        `hole_size`, or one self-intersecting, can come back no better.

        Parameters
        ----------
        hole_size : float, optional
            The largest hole to cover, in the mesh's own units. None to weld
            and reorient only, leaving every boundary where it is.

        Returns
        -------
        mesh : Mesh3D, Surface3D or Solid3D
        """
        mesh = self._polydata().clean()
        if hole_size is not None:
            mesh = mesh.fill_holes(float(hole_size))
        mesh = mesh.compute_normals(consistent_normals=True,
                                    auto_orient_normals=True).triangulate()

        if mesh.n_points == 0 or mesh.n_cells == 0:
            return Mesh3D(_np.zeros([0, 3]), _np.zeros([0, 3], dtype=int),
                          _np.zeros([0, 3]))

        points = _np.asarray(mesh.points, dtype=float)
        triangles = mesh.faces.reshape(-1, 4)[:, 1:]
        return mesh3d(points, triangles,
                      _gmt.vertex_normals(points, triangles))

    @classmethod
    def from_dxf(cls, filename):
        """
        Reads a triangulated surface from a DXF file.

        Three ways of writing a triangulation are understood. The `MESH`
        entity that `export_dxf` writes already holds a vertex list and the
        faces that index into it, and is taken as it stands. `POLYFACE`
        meshes and loose `3DFACE` entities instead repeat the coordinates of
        every corner they share, and are welded back into shared vertices,
        matched to six decimal places. Faces with more than three corners are
        split into a fan of triangles.

        Every mesh in the file is read and the results are concatenated, so a
        file holding several bodies comes back as one surface in several
        disconnected pieces. Each `MESH` entity keeps its own vertices, while
        the welded entities share one vertex list, so pieces that meet there
        are joined. Entities nested inside blocks are not searched.

        Only the geometry is read: see `export_dxf` on what a DXF file has no
        room for.

        Parameters
        ----------
        filename : str
            Path of the file to read.

        Returns
        -------
        mesh : Surface3D, Solid3D or Mesh3D
            Whichever the geometry read calls for, with normals computed from
            the triangles (see `geometry.vertex_normals`), since a DXF file
            carries none.
        """
        model = _ezdxf.readfile(filename).modelspace()

        blocks = [(_np.asarray(mesh.vertices, dtype=float),
                   [list(face) for face in mesh.faces])
                  for mesh in model.query("MESH")]

        # A 3DFACE carries no vertex list at all -- each one spells out the
        # coordinates of its corners, so a vertex shared by six triangles
        # arrives six times -- and a POLYFACE is read one face at a time.
        # The merger is what turns those back into shared vertices.
        merger = _MeshVertexMerger()
        for polyline in model.query("POLYLINE"):
            if not polyline.is_poly_face_mesh:
                continue
            body = _MeshVertexMerger.from_polyface(polyline)
            vertices = _np.asarray(body.vertices, dtype=float)
            for face in body.faces:
                merger.add_face(vertices[list(face)].tolist())
        for face in model.query("3DFACE"):
            merger.add_face(face.wcs_vertices())
        if len(merger.faces) > 0:
            blocks.append((_np.asarray(merger.vertices, dtype=float),
                           [list(face) for face in merger.faces]))

        if len(blocks) == 0:
            raise ValueError(
                "no triangulated surface found in %s: the file holds none of "
                "the MESH, POLYFACE or 3DFACE entities a surface is written "
                "as" % filename)

        points, triangles, start = [], [], 0
        for block_points, block_faces in blocks:
            points.append(block_points)
            triangles.append(_gmt.fan_triangulation(block_faces) + start)
            start += block_points.shape[0]
        points = _np.concatenate(points, axis=0)
        triangles = _np.concatenate(triangles, axis=0)

        return mesh3d(points, triangles,
                      _gmt.vertex_normals(points, triangles))

    def export_dxf(self, filename, offset=None):
        """
        Writes this surface to a DXF file, as a single MESH entity.

        A `MESH` holds the vertex list and the triangles that index into it,
        so the surface comes back from `from_dxf` exactly as it went out --
        nothing is welded and there is no ceiling on the number of vertices,
        unlike the `POLYFACE` mesh DXF is more often written as.

        Only the geometry travels. A DXF file has nowhere to put the
        variables and metadata a surface carries: `to_zarr` keeps a container
        whole, and `as_pyvista` carries the values onto a mesh object.

        Parameters
        ----------
        filename : str
            Path of the file to write.
        offset : array-like
            Added to the coordinates on the way out, as in
            `export_micromine`, for writing into a local grid. It is not
            recorded in the file, so reading it back gives the shifted
            coordinates.
        """
        points = _np.asarray(self.coordinates, dtype=float)
        if offset is not None:
            points = points + _np.asarray(offset, dtype=float).reshape([1, 3])

        document = _ezdxf.new()
        mesh = document.modelspace().add_mesh()
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = points.tolist()
            mesh_data.faces = _np.asarray(self.triangles, dtype=int).tolist()
        document.saveas(filename)

    def export_micromine(self, points_filename="points",
                         triangles_filename="triangles",
                         offset=[0, 0, 0], **kwargs):
        points_df = [
            _pd.DataFrame({"id": _np.arange(self.n_data)}),
            _pd.DataFrame(self.coordinates, columns=["EAST", "NORTH", "RL"])
        ]
        for variable in self.variables.values():
            points_df.append(variable.as_data_frame(**kwargs))
        points_df = _pd.concat(points_df, axis=1)
        points_df["EAST"] += offset[0]
        points_df["NORTH"] += offset[1]
        points_df["RL"] += offset[2]
        points_df.to_csv(points_filename + ".csv", index=False)

        triangles_df = _pd.DataFrame(
            self.triangles, columns=["PointId1", "PointId2", "PointId3"])
        triangles_df.to_csv(triangles_filename + ".csv", index=False)

    def as_pyvista(self, simulations=False, include="**"):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        faces = _np.concatenate(
            [_np.full([self.triangles.shape[0], 1], 3, int), self.triangles],
            axis=1
        )

        pv_surf = _pv.PolyData(self.coordinates, faces.ravel())

        return self._finish_pyvista(pv_surf, "points", simulations, include)


class Surface3D(Mesh3D):
    """
    A mesh that does not close: a sheet, with an edge to it.

    A topography, a seam roof, a weathering front, a fault plane — anything
    that has two sides rather than an inside. The promise is checked where it
    is made, so `assign_from_surface` need only be given one of these.
    """

    def __init__(self, points, triangles, normals):
        super().__init__(points, triangles, normals)

        # an empty mesh keeps every promise, having nothing to break them
        # with, and is what an operation that comes to nothing returns
        if self.n_data > 0 and self.closed:
            raise MeshTypeError(
                "this mesh closes -- every edge belongs to two triangles -- "
                "so it bounds a volume rather than being a sheet, and has no "
                "single elevation above a location; build a Solid3D, or "
                "mesh3d(...) for whichever the geometry calls for")

    def intersection(self, solid):
        """
        The part of this sheet lying inside a body.

        The sheet is cut where it crosses the body's surface, so what comes
        back follows the body's shape rather than the triangles' — the piece
        of a fault plane inside an ore envelope, say. A sheet lying wholly
        outside comes back empty.

        Parameters
        ----------
        solid : Solid3D
            The body to cut against.

        Returns
        -------
        surface : Surface3D
        """
        return self._clipped(solid, inside=True)

    def difference(self, solid):
        """
        The part of this sheet lying outside a body.

        The complement of `intersection`: together the two hold the whole
        sheet. A sheet lying wholly inside comes back empty.

        Parameters
        ----------
        solid : Solid3D
            The body to cut away.

        Returns
        -------
        surface : Surface3D
        """
        return self._clipped(solid, inside=False)

    def _clipped(self, solid, inside):
        """The sheet cut by a body, keeping one side of it."""
        if not isinstance(solid, Solid3D):
            raise MeshTypeError(
                "a sheet can only be cut by a Solid3D, a body being what has "
                "an inside to cut against; got %s" % type(solid).__name__)

        if self.n_data == 0 or solid.n_data == 0:
            return self if inside is False else _empty_surface()

        clipped = self._polydata().clip_surface(solid._polydata(),
                                                invert=inside)
        if clipped.n_points == 0 or clipped.n_cells == 0:
            return _empty_surface()

        clipped = clipped.triangulate()
        points = _np.asarray(clipped.points, dtype=float)
        triangles = clipped.faces.reshape(-1, 4)[:, 1:]
        return Surface3D(points, triangles,
                         _gmt.vertex_normals(points, triangles))


class Solid3D(Mesh3D):
    """
    A mesh that closes: a body, with an inside.

    An ore envelope, a stope, a dyke, a contoured shell. Both promises are
    checked where they are made — the mesh must close, and its triangles must
    agree which way is out — so `assign_from_solid` need only be given one of
    these, and `volume` always means something.

    A body wound inwards is turned round on the way in rather than refused:
    nothing about it is ambiguous, only reversed. The triangles are what get
    reversed; the normals are left as they were given.

    Attributes
    ----------
    volume : float
        The volume enclosed, always positive. Zero for an empty body, which
        is what an intersection of two bodies that do not meet comes to.
    """

    def __init__(self, points, triangles, normals):
        super().__init__(points, triangles, normals)

        if self.n_data > 0 and not self.closed:
            raise NotClosedError(
                "this mesh does not close -- %d of its edges belong to a "
                "single triangle -- so it has no inside; build a Surface3D "
                "for a sheet, or heal() it if a body is what was meant"
                % _gmt.open_edges(points, triangles))
        if not self.consistent:
            raise InconsistentMeshError(
                "this mesh closes, but its triangles disagree about which "
                "way is out -- %d of its edges are walked the same way round "
                "by both triangles sharing them -- so what it bounds is not "
                "defined; heal() puts this right"
                % _gmt.reversed_edges(points, triangles))

        if self._signed_volume < 0:
            self.triangles = _np.ascontiguousarray(
                _np.asarray(self.triangles)[:, ::-1])
            self._signed_volume = -self._signed_volume
        self.volume = self._signed_volume

    def union(self, other):
        """
        A body covering everything either of these two covers.

        Two bodies that do not meet make a union in two pieces, which is one
        legitimate body; `split()` takes it apart.
        """
        return self._combine(other, "union")

    def intersection(self, other):
        """
        A body covering what both of these two cover, empty where they do not
        meet at all.
        """
        return self._combine(other, "intersection")

    def difference(self, other):
        """
        A body covering what this one covers and the other does not.

        Where `other` lies wholly inside this one the answer is this body
        with a cavity in it, which is written as both surfaces, the inner one
        turned inwards — so `volume` comes to the difference of the two, and
        a location in the cavity tests as outside.
        """
        return self._combine(other, "difference")

    def _combine(self, other, operation):
        """Works the boolean out, VTK being unable to when they do not cross.

        VTK answers with nothing at all whenever the two surfaces have no
        face crossing another -- whether they stand apart or one contains the
        other, and with no error either way -- so an empty answer is not
        taken at face value but worked out from which body contains which.
        """
        if isinstance(other, Surface3D):
            return self._cut_by_sheet(other, operation)
        if not isinstance(other, Solid3D):
            raise MeshTypeError(
                "a body can only be combined with another Solid3D, or cut by "
                "a Surface3D; got %s" % type(other).__name__)

        if self.n_data > 0 and other.n_data > 0:
            combined = getattr(self._polydata(),
                               "boolean_" + operation)(other._polydata())
            if combined.n_points > 0:
                points = _np.asarray(combined.points, dtype=float)
                triangles = combined.triangulate().faces.reshape(-1, 4)[:, 1:]
                return Solid3D(points, triangles,
                               _gmt.vertex_normals(points, triangles))

        return self._without_crossing(other, operation)

    def _without_crossing(self, other, operation):
        """The answer where neither surface cuts the other."""
        here, there = _empty_solid(), _empty_solid()
        if self.n_data > 0:
            here = self
        if other.n_data > 0:
            there = other

        mine_inside = theirs_inside = False
        if here.n_data > 0 and there.n_data > 0:
            mine_inside = bool(_gmt.inside_solid(
                there._polydata(), _np.asarray(here.coordinates)[:1])[0])
            theirs_inside = bool(_gmt.inside_solid(
                here._polydata(), _np.asarray(there.coordinates)[:1])[0])

        if operation == "union":
            if mine_inside:
                return there
            if theirs_inside or there.n_data == 0:
                return here
            if here.n_data == 0:
                return there
            return _joined([here, there])

        if operation == "intersection":
            if mine_inside:
                return here
            if theirs_inside:
                return there
            return _empty_solid()

        if mine_inside:
            return _empty_solid()
        if theirs_inside:
            # a body with a cavity: the inner surface turned inwards, so the
            # volumes subtract and a location in the hollow reads as outside
            return _joined([here, there], reverse=[False, True])
        return here


    def _cut_by_sheet(self, sheet, operation):
        """This body divided by a sheet, keeping what lies under or over it.

        A sheet has no volume of its own, so there is nothing to add to or
        subtract from directly. What it does have, being single valued, is an
        underneath: extruded downwards past everything here, it becomes the
        ground beneath itself, and the ordinary body-to-body operations do
        the rest. `intersection` therefore keeps what lies below the sheet
        and `difference` what lies above it.
        """
        if operation == "union":
            raise MeshTypeError(
                "a sheet encloses no volume, so there is nothing in it to "
                "add to a body; use intersection to keep what lies below it, "
                "or difference to keep what lies above")

        if sheet.n_data == 0:
            return _empty_solid() if operation == "intersection" else self
        if self.n_data == 0:
            return _empty_solid()

        if not _gmt.single_valued(sheet.coordinates, sheet.triangles):
            raise NotSingleValuedError(
                "this sheet folds over, so 'below' and 'above' it are not "
                "one region each and a body cannot be divided by it; a DTM3D "
                "is the kind of surface this works with")

        here = self.bounding_box
        there = sheet.bounding_box
        low, high = _np.ravel(here.min), _np.ravel(here.max)
        sheet_low, sheet_high = _np.ravel(there.min), _np.ravel(there.max)
        if _np.any(sheet_low[:2] > low[:2]) \
                or _np.any(sheet_high[:2] < high[:2]):
            raise MeshTypeError(
                "the sheet does not reach across the whole body -- it spans "
                "x %g to %g and y %g to %g, against the body's x %g to %g "
                "and y %g to %g -- so it would cut at its own edge and leave "
                "a face that means nothing; extend it, or trim the body first"
                % (sheet_low[0], sheet_high[0], sheet_low[1], sheet_high[1],
                   low[0], high[0], low[1], high[1]))

        return self._combine(_ground_below(sheet, here), operation)


def _ground_below(sheet, box):
    """The body under a sheet, reaching below everything in `box`.

    The extrusion arrives with its walls and its floor wound against its lid,
    so it is reoriented before it can be a body at all.
    """
    top = float(_np.ravel(sheet.bounding_box.max)[2])
    floor = float(min(_np.ravel(box.min)[2],
                      _np.ravel(sheet.bounding_box.min)[2]))
    drop = (top - floor) + max(abs(top - floor), 1.0)

    body = sheet._polydata().extrude((0, 0, -drop), capping=True)
    body = body.clean().compute_normals(
        consistent_normals=True, auto_orient_normals=True).triangulate()

    points = _np.asarray(body.points, dtype=float)
    triangles = body.faces.reshape(-1, 4)[:, 1:]
    return Solid3D(points, triangles, _gmt.vertex_normals(points, triangles))


def _empty_solid():
    """A body enclosing nothing, which is what an empty answer looks like."""
    return Solid3D(_np.zeros([0, 3]), _np.zeros([0, 3], dtype=int),
                   _np.zeros([0, 3]))


def _empty_surface():
    """A sheet covering nothing, for an operation that clips everything away."""
    return Surface3D(_np.zeros([0, 3]), _np.zeros([0, 3], dtype=int),
                     _np.zeros([0, 3]))


def _joined(meshes, reverse=None):
    """One mesh holding several, each keeping its own vertices."""
    if reverse is None:
        reverse = [False] * len(meshes)

    points, triangles, normals, start = [], [], [], 0
    for mesh, turn in zip(meshes, reverse):
        block = _np.asarray(mesh.triangles) + start
        points.append(_np.asarray(mesh.coordinates, dtype=float))
        triangles.append(block[:, ::-1] if turn else block)
        normals.append(_np.asarray(mesh.normals))
        start += mesh.n_data

    return mesh3d(_np.concatenate(points, axis=0),
                  _np.ascontiguousarray(_np.concatenate(triangles, axis=0)),
                  _np.concatenate(normals, axis=0))


class DTM3D(Surface3D):
    """
    A terrain: a sheet standing at one height over each (x, y).

    A digital terrain model, and the shape most of the surfaces in a project
    have — a topography, a seam roof, a weathering front. The promise is that
    it never folds back over itself, checked where the object is made, which
    is what lets a body be divided into what lies under it and what lies over
    it, and what makes "the elevation here" a question with one answer.

    Not what `mesh3d` returns: an ordinary sheet is a `Surface3D` unless a
    terrain is asked for, this being a promise to make rather than a fact to
    detect. Triangles standing exactly vertical are allowed, a cliff being
    single valued everywhere but along the line of its face.
    """

    def __init__(self, points, triangles, normals):
        super().__init__(points, triangles, normals)

        if self.n_data > 0 and not _gmt.single_valued(points, triangles):
            raise NotSingleValuedError(
                "this sheet folds over: some of its triangles face the "
                "ground and some face away from it, so it stands at more "
                "than one height over some of its footprint and is not a "
                "terrain. A Surface3D holds it without that promise")


def mesh3d(points, triangles, normals):
    """
    A mesh of whichever class its geometry calls for.

    A `Solid3D` where the triangles close and agree which way is out, a
    `Surface3D` where they do not close, and a plain `Mesh3D` where they
    close but disagree — the one case that is neither a sheet nor a body, and
    what `Mesh3D.heal` exists for.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    normals : array
        An (n, 3) array of vertex normals.

    Returns
    -------
    mesh : Surface3D, Solid3D or Mesh3D
    """
    if _gmt.open_edges(points, triangles) > 0:
        return Surface3D(points, triangles, normals)
    if _gmt.reversed_edges(points, triangles) > 0:
        return Mesh3D(points, triangles, normals)
    return Solid3D(points, triangles, normals)


def _below_sheet(surface):
    """The test a sheet poses of a location: is it under the surface?"""
    interpolator = _sheet_interpolator(surface)

    def below(coordinates):
        return coordinates[:, 2] < _gmt.sheet_elevation(interpolator,
                                                        coordinates)
    return below


def _within_body(solid):
    """The test a closed body poses of a location: is it inside?"""
    mesh = _closed_body(solid)
    return lambda coordinates: _gmt.inside_solid(mesh, coordinates)


def _mesh_test(mesh):
    """Which side of `mesh` a location falls on, whichever kind it is."""
    if isinstance(mesh, Solid3D):
        return _within_body(mesh)
    if isinstance(mesh, Surface3D):
        return _below_sheet(mesh)
    raise MeshTypeError(
        "a %s says nothing about which side a block is on: a sheet has an "
        "above and a below, a body an inside and an outside, and a mesh that "
        "is neither has no sides to speak of. `heal()` is what turns one into "
        "a body" % type(mesh).__name__)


