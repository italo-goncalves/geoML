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
import multiprocessing as _mp
import os as _os
import warnings as _warnings

import numpy as _np
import pandas as _pd
import pyvista as _pv
import vtk as _vtk
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
            "hole; heal() puts this right" % reversed_edges)

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
        if triangles.size > 0:
            reach = (int(_np.min(triangles)), int(_np.max(triangles)))
            if reach[0] < 0 or reach[1] >= points.shape[0]:
                # said here rather than left to whichever measure indexes
                # first, which reports an IndexError from inside `area`
                raise ValueError(
                    "the triangles index vertices from %d to %d, and there "
                    "are %d points to index"
                    % (reach[0], reach[1], points.shape[0]))

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
        # both counts from one welding: it is the expensive half, and every
        # mesh is built asking both questions of the same triangulation
        open_count, reversed_count = _gmt.edge_defects(points, triangles)
        self.closed = open_count == 0
        self.consistent = reversed_count == 0
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

        Four things are put right, in the order that works: coincident
        vertices are welded, so that seams stop reading as boundaries; the
        faces that bound nothing are dropped; holes smaller than `hole_size`
        are covered over; and the triangles are made to agree about which
        way is out, then turned to face outward. That last step is not
        optional — filling a hole leaves the new triangles wound however
        they came, which would leave the mesh closed and still untestable.

        **The dropping has to come first**, and this method did not do it
        until 0.6.7. A zero-thickness flap or a zero-area sliver makes an
        edge run twice the same way round, so it reads as a winding failure
        — and it is the one winding failure reorienting cannot mend, the
        surface being non-manifold there for VTK to walk. Measured on a box
        carrying one flap: `clean`, `compute_normals(consistent_normals=
        True, auto_orient_normals=True)` and `triangulate` left all three of
        its reversed edges exactly as they were, and every error message in
        this module sends the user here to have them fixed.

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
        points, triangles = _gmt.drop_degenerate_faces(
            _np.asarray(self.coordinates, dtype=float),
            _np.asarray(self.triangles))
        faces = _np.concatenate(
            [_np.full([triangles.shape[0], 1], 3, int), triangles], axis=1)
        mesh = _pv.PolyData(points, faces.ravel()).clean()
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

    def simplify(self, max_error):
        """
        The same shape on as few triangles as the error budget allows.

        Built for what `get_contour` returns: a contoured surface carries a
        triangle for every block corner it crosses, most of them slivers
        saying nothing the budget would miss. The argument is geometric --
        how far, in the mesh's own units, the simplified surface may sit
        from the original -- so the same call means the same thing on a
        coarse shell and a fine one, which a fraction of triangles does not.

        The caller's kind is kept: a body stays a body, a terrain a
        terrain. If a cut breaks the kind's own promise -- a solid opened,
        a terrain folded over -- the mesh is cut more gently until the
        promise holds, since a gentler cut only sits closer to the
        original; if no cut survives, the mesh comes back as it came, with
        a warning saying so. Simplification never trades the shape away.

        Parameters
        ----------
        max_error : float
            The largest distance the simplified surface may sit from the
            original, in the mesh's own units. Enforced by measurement: the
            simplified faces are probed against the original surface, and
            the decimation tightened until the promise holds.

        Returns
        -------
        mesh : the same class as this one.
        """
        max_error = float(max_error)
        if max_error <= 0:
            raise ValueError(
                "max_error is a distance in the mesh's own units and must "
                "be positive; got %g" % max_error)
        if self.n_data == 0:
            return self

        box = self.bounding_box
        diagonal = float(_np.linalg.norm(
            _np.ravel(box.max) - _np.ravel(box.min)))
        original = self._polydata()

        # one distance oracle for every measurement in the loop, its
        # workers and locators built once -- the measurements are half of
        # a real simplify (4.9 of 10.0 s on a 522k-triangle shell), and
        # VTK holds the GIL through them, so the pool is what threads can
        # not do (see `_signed_distance`); a mesh too small to repay the
        # spin-up measures serially on the one kept locator
        with _DistanceQueries(
                original,
                parallel=4 * original.n_cells >= _PARALLEL_QUERIES
        ) as measure:

            def deviation_of(mesh):
                # probed on the simplified faces -- centroids and edge
                # midpoints; the surviving vertices lie on the original by
                # construction and would measure nothing
                pts = _np.asarray(mesh.points, dtype=float)
                tri = mesh.faces.reshape(-1, 4)[:, 1:]
                probes = _np.concatenate([
                    pts[tri].mean(axis=1),
                    (pts[tri[:, 0]] + pts[tri[:, 1]]) / 2,
                    (pts[tri[:, 1]] + pts[tri[:, 2]]) / 2,
                    (pts[tri[:, 2]] + pts[tri[:, 0]]) / 2])
                return float(_np.abs(measure.query(probes)).max())

            # A large mesh takes a fast quadric pre-pass first, so the
            # error-bounded decimator works a fraction of the triangles: on
            # an 835k-triangle shell this is most of a 4x speedup. The
            # pre-pass is verified against the original like everything
            # else, and given half the budget; where it overspends, the
            # mesh is taken as it came.
            working = original
            n_triangles = original.n_cells
            if n_triangles > 100_000:
                rough = original.decimate(1.0 - 50_000.0 / n_triangles,
                                          volume_preservation=True)
                rough = rough.clean().triangulate()
                if deviation_of(rough) <= 0.5 * max_error:
                    working = rough

            def cut_at(bound):
                # vtkDecimatePro is the one decimator that takes an error
                # bound, as a fraction of the bounding-box diagonal;
                # preserving topology is what keeps a closed body closed,
                # and the error accumulates against its input rather than
                # being re-granted per collapse
                decimate = _vtk.vtkDecimatePro()
                decimate.SetInputData(working)
                decimate.SetTargetReduction(1.0)
                decimate.SetMaximumError(bound / diagonal)
                decimate.AccumulateErrorOn()
                decimate.PreserveTopologyOn()
                decimate.BoundaryVertexDeletionOff()
                decimate.Update()
                return _pv.wrap(decimate.GetOutput()).clean().triangulate()

            # the decimator's own error metric runs loose at tight budgets
            # (measured 7x over at 0.02 of a unit step on a contoured
            # shell), so the true deviation -- always against the original,
            # whatever the pre-pass did -- is measured and the internal
            # bound tightened until the promise holds
            bound = max_error
            for _ in range(4):
                mesh = cut_at(bound)
                deviation = deviation_of(mesh)
                if deviation <= max_error:
                    break
                bound *= 0.5 * max_error / deviation

            # Decimation can also break the kind's own promise -- collapse
            # a thin feature into a membrane the rebuild cannot always
            # repair -- and that is no reason to raise out of a workflow,
            # because unlike the other rebuilds this one holds a remedy:
            # cutting less. A gentler cut avoids the collapse, its error
            # only shrinks, and the original mesh is within any budget at
            # all -- so the budget is spent more timidly until the kind
            # survives, and not at all as the last resort, said out loud
            # rather than silently.
            for _ in range(3):
                try:
                    return _rebuilt_as(type(self), mesh)
                except (NotClosedError, InconsistentMeshError,
                        NotSingleValuedError):
                    bound *= 0.25
                    mesh = cut_at(bound)
            try:
                return _rebuilt_as(type(self), mesh)
            except (NotClosedError, InconsistentMeshError,
                    NotSingleValuedError):
                _warnings.warn(
                    "decimation broke this mesh's own shape however gently "
                    "it was applied, so the mesh is returned as it came, "
                    "with all its %d triangles" % len(self.triangles))
                return self

    def smooth(self, iterations=20, pass_band=0.1):
        """
        A smoothed copy, by Taubin's non-shrinking filter.

        Cosmetic, and priced honestly: applied to a block-model contour this
        was measured to take away a sixth of the faceting while moving the
        surface 50% further from the true level set -- the creases go, and
        accuracy goes with them, which is why no contour smooths itself. For
        a surface that is both rounder and *closer* to the truth, contour
        with `supersample` instead; smooth when the look of the mesh is what
        matters.

        The caller's kind is kept, as in `simplify`.

        Parameters
        ----------
        iterations : int
            Passes of the filter; more is smoother.
        pass_band : float
            The filter's pass band, in (0, 2): lower smooths more.

        Returns
        -------
        mesh : the same class as this one.
        """
        if self.n_data == 0:
            return self
        mesh = self._polydata().smooth_taubin(
            n_iter=int(iterations), pass_band=float(pass_band)).triangulate()
        return _rebuilt_as(type(self), mesh)

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

        # a foreign file is not ours to trust: a face repeated between two
        # entities, or one whose corners collapse, would be read as a
        # winding failure rather than as the nothing it bounds
        points, triangles = _gmt.drop_degenerate_faces(points, triangles)
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

    def intersection(self, other):
        """
        The part of this sheet lying inside a body, or under a terrain.

        Against a body, the sheet is cut where it crosses the body's
        surface, so what comes back follows the body's shape rather than
        the triangles' — the piece of a fault plane inside an ore envelope,
        say. Against a single-valued sheet — a topography — the cut is
        against the ground below it, keeping what lies under. A sheet lying
        wholly outside comes back empty.

        Parameters
        ----------
        other : Solid3D or Surface3D
            The body to cut against, or the terrain whose underneath to
            keep.

        Returns
        -------
        surface : Surface3D
        """
        return self._clipped(other, inside=True)

    def difference(self, other):
        """
        The part of this sheet lying outside a body, or over a terrain.

        The complement of `intersection`: together the two hold the whole
        sheet. A sheet lying wholly inside comes back empty.

        Parameters
        ----------
        other : Solid3D or Surface3D
            The body to cut away, or the terrain whose underneath to cut
            away.

        Returns
        -------
        surface : Surface3D
        """
        return self._clipped(other, inside=False)

    def clip_meshes(self, meshes):
        """
        Everything below this sheet, each mesh cut to its own kind.

        The batch form of cutting against a terrain: one ground body is
        extruded under the sheet and serves every cut, where cutting one by
        one would rebuild it per mesh. A body comes back a closed body (the
        boolean engines see to it), a sheet comes back a sheet — a shell
        that runs out of its model is open, and stays open here; contour it
        with `close=` first if a body is what is wanted.

        Parameters
        ----------
        meshes : sequence of Mesh3D
            Bodies and sheets to cut below this one.

        Returns
        -------
        list
            One cut mesh per input, in the same order.
        """
        meshes = list(meshes)
        if len(meshes) == 0:
            return []
        for mesh in meshes:
            if not isinstance(mesh, (Solid3D, Surface3D)):
                raise MeshTypeError(
                    "clip_meshes cuts bodies and sheets; got %s"
                    % type(mesh).__name__)
            if mesh.n_data > 0:
                _reaches_across(self, mesh.bounding_box)

        full = [mesh for mesh in meshes if mesh.n_data > 0]
        if len(full) == 0:
            return meshes
        low = _np.min([_np.ravel(m.bounding_box.min) for m in full], axis=0)
        high = _np.max([_np.ravel(m.bounding_box.max) for m in full], axis=0)
        ground = _ground_under(
            self, BoundingBox.from_array(_np.stack([low, high])))

        return [mesh._combine(ground, "intersection")
                if isinstance(mesh, Solid3D)
                else mesh._clipped(ground, inside=True)
                for mesh in meshes]

    def _clipped(self, other, inside):
        """The sheet cut by a body, keeping one side of it."""
        if isinstance(other, Solid3D):
            solid = other
        elif isinstance(other, Surface3D):
            # a sheet has no inside, but a single-valued one has an
            # underneath: the ground below it is the body to cut against,
            # exactly as when it divides a Solid3D
            if other.n_data == 0:
                return self if inside is False else _empty_surface()
            solid = _ground_under(other, self.bounding_box)
        else:
            raise MeshTypeError(
                "a sheet can only be cut by a Solid3D, a body being what "
                "has an inside to cut against, or by a single-valued sheet, "
                "whose underneath is one; got %s" % type(other).__name__)

        if self.n_data == 0 or solid.n_data == 0:
            return self if inside is False else _empty_surface()

        # the same local frame as the booleans, for the same numerics
        shift = _local_frame(self, solid)
        clipped = self._polydata().translate(-shift).clip_surface(
            solid._polydata().translate(-shift), invert=inside)
        if clipped.n_points == 0 or clipped.n_cells == 0:
            return _empty_surface()

        clipped = clipped.triangulate().translate(shift)
        points = _np.asarray(clipped.points, dtype=float)
        triangles = clipped.faces.reshape(-1, 4)[:, 1:]
        # cutting along a surface leaves slivers where the cut passes close
        # to a vertex, and they would be read as defects rather than as the
        # nothing they cover
        points, triangles = _gmt.drop_degenerate_faces(points, triangles)
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
        """Works the boolean out, on the geometry rather than on a filter.

        A sheet is handled by extruding it into the ground beneath itself;
        two bodies go to `_resolved`, which decides whether they cross and
        answers accordingly.
        """
        if isinstance(other, Surface3D):
            return self._cut_by_sheet(other, operation)
        if not isinstance(other, Solid3D):
            raise MeshTypeError(
                "a body can only be combined with another Solid3D, or cut by "
                "a Surface3D; got %s" % type(other).__name__)

        return self._resolved(other, operation)

    def _resolved(self, other, operation):
        """Which case this is, and the answer for it.

        Two bodies either cross or they do not, and the vertices say which:
        a surface crossing another has vertices on both sides of it. Where
        they do not cross -- apart, or one inside the other -- the answer is
        one of the two bodies, or both, or nothing, and is exact. Where they
        do, `_implicit_combine` works it out on a grid.

        **VTK's exact boolean is not consulted at all**, and that is a
        deliberate retreat from what 0.6.1 shipped. The filter does answer
        exactly on simple analytic meshes, but on the contour-derived bodies
        this package exists to produce it **segfaults**, taking the session
        with it -- measured on shells of 836 to 3128 triangles, at the
        origin and at mine-grid coordinates, and on the same shells cleaned,
        de-slivered and decimated. A crash cannot be caught, so the
        error-and-empty-output fallback the old code wrapped it in never
        got a turn; the meshes most likely to need the fallback were the
        ones that never reached it. What is given up is exactness on the
        cases VTK could do (a box against a box), which now answer to the
        implicit engine's grid step like everything else, and say so.
        """
        here, there = _empty_solid(), _empty_solid()
        if self.n_data > 0:
            here = self
        if other.n_data > 0:
            there = other

        mine_inside = theirs_inside = False
        if (here.n_data > 0 and there.n_data > 0
                and here.bounding_box.overlaps_with(there.bounding_box)):
            # boxes apart need none of this: nothing crosses, nothing is
            # inside anything, and a domains workflow is full of exactly
            # such pairs -- mutually exclusive rock shells
            shift = _local_frame(here, there)
            here_poly = here._polydata().translate(-shift)
            there_poly = there._polydata().translate(-shift)
            here_points = _np.asarray(here.coordinates) - shift
            there_points = _np.asarray(there.coordinates) - shift

            # Crossing is the common case and rarely needs every vertex to
            # prove: a couple of thousand from either body usually land on
            # both sides already. The full queries below are only paid when
            # the probes come back one-sided, which is also the only time
            # their answer is needed in full -- "all inside" is a claim
            # about every vertex. Measured on 55k-vertex shells: 0.79 s of
            # crossing test becomes ~0.05 s wherever the bodies do cross.
            for probe_points, probe_body in ((here_points, there_poly),
                                             (there_points, here_poly)):
                sample = probe_points[
                    ::max(1, len(probe_points) // 2048)]
                seen = _gmt.inside_solid(probe_body, sample)
                if seen.any() and not seen.all():
                    return _implicit_combine(here, there, operation)

            # One full scan usually settles it: two closed surfaces that
            # cross put vertices of *each* on both sides of the other, so
            # the smaller vertex set decides crossing for both, and
            # disjoint bodies with overlapping boxes -- interleaved rock
            # lobes -- pay one scan instead of two. The second scan is
            # owed in two cases only: the second body could still lie
            # wholly inside the first (its bounding box says so for free),
            # or the first came back wholly inside -- nesting is exactly
            # where a vertex the probes and the first scan never see could
            # hide a crossing, and it is rare enough to afford the check
            # the old code always paid.
            flipped = len(there_points) < len(here_points)
            if flipped:
                first = _gmt.inside_solid(here_poly, there_points)
            else:
                first = _gmt.inside_solid(there_poly, here_points)
            if first.any() and not first.all():
                return _implicit_combine(here, there, operation)
            first_inside = bool(first.all())

            second_inside = False
            inner, outer = ((here, there) if flipped else (there, here))
            enclosable = bool(
                _np.all(_np.ravel(inner.bounding_box.min)
                        >= _np.ravel(outer.bounding_box.min))
                and _np.all(_np.ravel(inner.bounding_box.max)
                            <= _np.ravel(outer.bounding_box.max)))
            if first_inside or enclosable:
                if flipped:
                    second = _gmt.inside_solid(there_poly, here_points)
                else:
                    second = _gmt.inside_solid(here_poly, there_points)
                if second.any() and not second.all():
                    return _implicit_combine(here, there, operation)
                second_inside = bool(second.all())

            if flipped:
                mine_inside, theirs_inside = second_inside, first_inside
            else:
                mine_inside, theirs_inside = first_inside, second_inside

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

        return self._combine(
            _ground_under(sheet, self.bounding_box), operation)


def _rebuilt_as(cls, mesh):
    """A pyvista mesh back as `cls`, past the two artifacts decimation leaves.

    The first is faces that bound nothing. Collapsing an edge through a thin
    feature folds it into a zero-thickness flap -- the same triangle twice,
    or one with no area -- and those are dropped before anything else, since
    they carry no shape and every one of them makes an edge run twice the
    same way round. **They are why the winding repair below is not enough on
    its own**: a flap is non-manifold, VTK's orientation pass cannot walk
    across it, and it comes back having moved the disagreement rather than
    settled it (measured on a decimated shell: 2 reversed edges in, 3 out).

    The second is genuine: a few triangles wound against their neighbours,
    which the constructor rightly refuses and which is repairable without
    touching the geometry -- winding is bookkeeping, not shape -- so the
    triangles are made to agree and face outward, as `heal()` would, and the
    build is tried once more. A mesh that fails for any other reason (a
    solid opened, a terrain folded) fails the second time too, and that
    error stands: closing a hole or unfolding a sheet would be inventing
    geometry.
    """
    points, triangles = _gmt.drop_degenerate_faces(
        _np.asarray(mesh.points, dtype=float),
        mesh.faces.reshape(-1, 4)[:, 1:])
    try:
        return cls(points, triangles,
                   _gmt.vertex_normals(points, triangles))
    except InconsistentMeshError:
        # on the cleaned arrays rather than on `mesh`, so the orientation
        # pass walks a manifold surface -- across a flap it cannot, and
        # returns having spread the disagreement instead of settling it
        faces = _np.concatenate(
            [_np.full([triangles.shape[0], 1], 3, int), triangles], axis=1)
        repaired = _pv.PolyData(points, faces.ravel()).compute_normals(
            consistent_normals=True, auto_orient_normals=True).triangulate()
        points = _np.asarray(repaired.points, dtype=float)
        triangles = repaired.faces.reshape(-1, 4)[:, 1:]
        return cls(points, triangles,
                   _gmt.vertex_normals(points, triangles))


# the cell budget of the implicit fallback: the step is chosen so the grid
# stays near this many cells whatever the bodies span
_IMPLICIT_CELLS = 2_000_000

# how many fine steps a coarse cell spans in the Lipschitz pre-pass
_IMPLICIT_COARSE = 4


# One distance locator per worker process, built by the pool initializer and
# read by every chunk that worker answers. Module-level because a pool can
# only call what it can import.
_QUERY_STATE = {}

# Below this many points the pool costs more than it saves: forking and
# feeding ~16 workers is ~0.2 s, and the serial rate is ~25 us a point -- at 20k points the pool already returns twice as fast as the serial call, which is what pools the boolean's coarse passes (~31k points on a 2M-cell grid) and not the small probes.
_PARALLEL_QUERIES = 20_000


def _distance_worker(points, faces):
    measure = _vtk.vtkImplicitPolyDataDistance()
    measure.SetInput(_pv.PolyData(points, faces))
    _QUERY_STATE["measure"] = measure


def _distance_chunk(chunk):
    cloud = _pv.PolyData(_np.ascontiguousarray(chunk))
    out = _vtk.vtkDoubleArray()
    _QUERY_STATE["measure"].FunctionValue(cloud.GetPoints().GetData(), out)
    return _np.asarray(_pv.convert_array(out), dtype=float)


class _DistanceQueries:
    """Signed distances to one surface, over several queries.

    The keep-the-pool sibling of `_signed_distance`, which spins its
    workers up per call: `simplify` measures its deviations three to six
    times against the same original, so the fork and the per-worker
    locator builds are paid once here and every measurement after the
    first rides them. Probing 260k points against a 522k-triangle shell
    measured 1.6 s a call serial; the pool answers the lot of a
    `simplify` in about that. Serial wherever a pool cannot or should not
    come up -- `parallel=False`, no `fork`, one CPU, or the fork failing
    -- with the one locator likewise kept across queries.
    """

    def __init__(self, polydata, parallel=True):
        self._polydata = polydata
        self._pool = None
        self._measure = None
        if not (parallel and "fork" in _mp.get_all_start_methods()):
            return
        self._workers = max(1, min(16, _os.cpu_count() or 1))
        if self._workers < 2:
            return
        try:
            with _warnings.catch_warnings():
                # the same suppression as `_signed_distance`, for the
                # same measured reason: the workers touch VTK and numpy
                # alone, never TF, never the GPU
                _warnings.filterwarnings(
                    "ignore", message=".*fork\\(\\)",
                    category=DeprecationWarning)
                self._pool = _mp.get_context("fork").Pool(
                    self._workers, initializer=_distance_worker,
                    initargs=(_np.asarray(polydata.points, dtype=float),
                              _np.asarray(polydata.faces)))
        except OSError:
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None
        return False

    def query(self, points):
        points = _np.ascontiguousarray(points, dtype=float)
        if self._pool is not None:
            return _np.concatenate(self._pool.map(
                _distance_chunk,
                _np.array_split(points, 2 * self._workers)))
        if self._measure is None:
            self._measure = _vtk.vtkImplicitPolyDataDistance()
            self._measure.SetInput(self._polydata)
        cloud = _pv.PolyData(points)
        out = _vtk.vtkDoubleArray()
        self._measure.FunctionValue(cloud.GetPoints().GetData(), out)
        return _np.asarray(_pv.convert_array(out), dtype=float)


def _signed_distance(body, points):
    """Each point's signed distance to the body, negative inside.

    **The sign is the pseudonormal's, imperfect and measured to be the
    best available.** On a real 2.5M-triangle ore envelope carrying
    degenerate faces, `vtkImplicitPolyDataDistance` signed a small cone of
    points forty metres outside the body as -43 — and replacing its sign
    with `vtkSelectEnclosedPoints`' ray casting was built, measured and
    reverted, because a ray through a degenerate patch corrupts the
    crossing parity of **every point along its shadow**: the same 58
    probes stayed wrong and whole columns of good ones flipped, speckling
    the fields and quintupling the output. A pseudonormal error is local
    to the defect; a parity error is a column. What actually contains the
    damage is the region clamp in `_implicit_combine`, which no sign
    error can cross.

    A large query is spread over forked worker processes, each holding its
    own locator; the answers are the same filter either way, and were
    measured bit-identical. **Processes, not threads, and it is not a
    style choice**: VTK holds the GIL through `FunctionValue`, so eight
    threads measured 1.2x where sixteen forked processes measured 7.9x.
    Everything else takes the serial path: small queries, where the pool
    costs more than it saves, platforms without `fork` (Windows outside
    WSL — a spawned worker would re-import the package, TensorFlow and
    all), and any pool that fails to come up.
    """
    points = _np.ascontiguousarray(points)
    if (len(points) >= _PARALLEL_QUERIES
            and "fork" in _mp.get_all_start_methods()):
        workers = max(1, min(16, _os.cpu_count() or 1))
        if workers > 1:
            polydata = body._polydata()
            try:
                with _warnings.catch_warnings():
                    # Python 3.12 warns that forking a multi-threaded
                    # process can deadlock, and TensorFlow's thread pools
                    # are always up by the time a mesh is cut. The workers
                    # here touch VTK and numpy alone -- never the GPU,
                    # never TF -- and the combination was measured stable
                    # and bit-identical with CUDA live in the parent.
                    # Suppressed rather than designed around because every
                    # alternative loses: threads sit behind VTK's GIL hold,
                    # and spawn or forkserver workers would import the
                    # package, TensorFlow and all, per pool.
                    _warnings.filterwarnings(
                        "ignore", message=".*fork\\(\\)",
                        category=DeprecationWarning)
                    with _mp.get_context("fork").Pool(
                            workers, initializer=_distance_worker,
                            initargs=(
                                _np.asarray(polydata.points, dtype=float),
                                _np.asarray(polydata.faces))) as pool:
                        return _np.concatenate(pool.map(
                            _distance_chunk,
                            _np.array_split(points, 2 * workers)))
            except OSError:
                # no processes to be had; the serial path always is
                pass
    cloud = _pv.PolyData(points)
    return _np.asarray(cloud.compute_implicit_distance(
        body._polydata())["implicit_distance"], dtype=float)


def _banded_distance(body, points, shape, low, step, relevant):
    """The signed field, exact wherever `relevant` allows a zero crossing.

    A distance field is 1-Lipschitz, so a coarse sample bounds every fine
    value near it: where the nearest coarse value stands further from zero
    than the anchor distance plus a cell's diagonal, the fine sign is
    settled without a query. The exact queries collapse to a band around
    the body's surface -- measured at 29% of the lattice on a contoured
    shell, for 12x the speed at zero error, since inside the band the
    values are the same queries they always were. `relevant` narrows the
    band further to where the *combined* field can cross at all, which is
    what spares a vast terrain's field being resolved far from a small
    body.
    """
    k = _IMPLICIT_COARSE
    axes = [low[i] + k * step * _np.arange(shape[i] // k + 2)
            for i in range(3)]
    gz, gy, gx = _np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    coarse = _signed_distance(body, _np.column_stack(
        [gx.ravel(), gy.ravel(), gz.ravel()]))

    index = [_np.clip(_np.round((points[:, i] - low[i]) / (k * step))
                      .astype(int), 0, len(axes[i]) - 1) for i in range(3)]
    field = coarse.reshape(len(axes[2]), len(axes[1]), len(axes[0]))[
        index[2], index[1], index[0]]

    reach = _implicit_reach(step)
    band = (_np.abs(field) <= reach) & relevant
    if band.any():
        field = field.copy()
        field[band] = _signed_distance(body, points[band])
    return field


def _implicit_reach(step):
    """How far from zero an anchored coarse value may sit and still leave a
    fine crossing possible: the anchor offset plus two cell diagonals."""
    return (_IMPLICIT_COARSE / 2.0 + 2.0) * _np.sqrt(3.0) * step


def _implicit_combine(here, there, operation):
    """The boolean as signed fields on a grid, contoured back to a body.

    The engine of last resort, for the meshes VTK's exact filter fails on
    (measured on contour-derived shells: whole patches dropped or
    fabricated, unrepairable after the fact). Each body becomes its signed
    distance sampled on a grid over the region the answer can occupy;
    `max` of the fields is the intersection, `min` the union, `max(a, -b)`
    the difference; the zero surface of the combined field is the answer.
    There is no seam geometry to walk, which is what makes it robust, and
    the price is honest: the surface is exact to the grid's step rather
    than to the inputs' triangles.

    The fields are evaluated in a band (see `_banded_distance`), and each
    body's band is masked by the other's coarse field, so that neither is
    resolved where the other has already decided the outcome -- a small
    shell against a whole topography queries the topography around the
    shell alone.
    """
    # imported late: the grids subclass the containers meshes sit beside
    from geoml.data import Grid3D

    a_low, a_high = (_np.ravel(here.bounding_box.min),
                     _np.ravel(here.bounding_box.max))
    b_low, b_high = (_np.ravel(there.bounding_box.min),
                     _np.ravel(there.bounding_box.max))
    if operation == "intersection":
        low, high = _np.maximum(a_low, b_low), _np.minimum(a_high, b_high)
        if _np.any(high <= low):
            return _empty_solid()
    elif operation == "union":
        low, high = _np.minimum(a_low, b_low), _np.maximum(a_high, b_high)
    else:
        low, high = a_low, a_high

    span = high - low
    step = float((span.prod() / _IMPLICIT_CELLS) ** (1.0 / 3.0))
    _warnings.warn(
        "the %s of these meshes crosses, so it is worked out on an implicit "
        "grid and is exact to its step of %.3g" % (operation, step))

    # the region the answer can occupy, kept before the margin is added:
    # the combined field is clamped against this box below
    region_low, region_high = low.copy(), high.copy()

    # two cells of margin, so the zero surface closes inside the grid
    low = low - 2 * step
    n = _np.ceil((span + 4 * step) / step).astype(int) + 1
    grid = Grid3D(start=low, n=n, step=[step, step, step])
    points = _np.asarray(grid.coordinates, dtype=float)

    # each body is only exact where the other leaves the outcome open; the
    # first pass has no other field to ask, so it answers everywhere
    reach = _implicit_reach(step)
    everywhere = _np.ones(len(points), dtype=bool)
    field_a = _banded_distance(here, points, n, low, step, everywhere)
    if operation == "intersection":
        field_b = _banded_distance(there, points, n, low, step,
                                   field_a <= reach)
        field = _np.maximum(field_a, field_b)
    elif operation == "union":
        field_b = _banded_distance(there, points, n, low, step,
                                   field_a >= -reach)
        field = _np.minimum(field_a, field_b)
    else:
        field_b = _banded_distance(there, points, n, low, step,
                                   field_a <= reach)
        field = _np.maximum(field_a, -field_b)

    # The answer cannot exist outside the region box, so the box's own
    # signed distance caps the field. This is what seals the surface at
    # the region boundary *whatever the locators do inside*: their sign is
    # a pseudonormal's and was measured wrong forty metres outside a real
    # ore envelope, which carried the zero surface out through the grid
    # and handed back an open sheet for the intersection of two closed
    # bodies. The cap stands 1.5 cells out -- far enough that a body's own
    # wall on the region boundary and the cap fall in different cells
    # (half a step apart they pinch, marching cubes having one crossing
    # per edge to give), near enough that the two-cell grid margin still
    # holds a node beyond the cap for its surface to close through.
    gap = _np.maximum((region_low - 1.5 * step) - points,
                      points - (region_high + 1.5 * step))
    clamp = (_np.linalg.norm(_np.maximum(gap, 0.0), axis=1)
             + _np.minimum(gap.max(axis=1), 0.0))
    field = _np.maximum(field, clamp)

    grid.add_continuous_variable("distance", field)
    if _np.all(field > 0):
        return _empty_solid()
    # the variable was just added as a continuous one, which is what carries
    # the `measurements` this contours
    distance = grid.variables["distance"]
    assert isinstance(distance, ContinuousVariable)
    body = distance.measurements.get_contour(0.0)
    if not isinstance(body, Solid3D):
        # Marching cubes pinches wherever two zero surfaces run closer
        # than a cell, and real bodies invite that: their flat walls sit
        # on the region boundary the cap runs along. The residue is
        # pinholes -- measured 6 open edges of 350k triangles on a real
        # pair -- and a pinhole is honestly repairable at the engine's own
        # resolution, which is what the step is.
        healed = body.heal(hole_size=2 * step)
        if isinstance(healed, Solid3D):
            return healed
    return body


def _local_frame(mine, theirs):
    """The corner both meshes are translated by before VTK sees them.

    Rounded so the shift itself costs no precision on the way back.
    """
    corner = _np.minimum(_np.ravel(mine.bounding_box.min),
                         _np.ravel(theirs.bounding_box.min))
    return _np.round(corner)


def _reaches_across(sheet, box):
    """Refuses a sheet whose footprint does not span `box`.

    A sheet cutting a region it does not cover would cut at its own edge
    and leave a face that means nothing.
    """
    low, high = _np.ravel(box.min), _np.ravel(box.max)
    there = sheet.bounding_box
    sheet_low, sheet_high = _np.ravel(there.min), _np.ravel(there.max)
    if _np.any(sheet_low[:2] > low[:2]) \
            or _np.any(sheet_high[:2] < high[:2]):
        raise MeshTypeError(
            "the sheet does not reach across the whole mesh -- it spans "
            "x %g to %g and y %g to %g, against the mesh's x %g to %g "
            "and y %g to %g -- so it would cut at its own edge and leave "
            "a face that means nothing; extend it, or trim the mesh first"
            % (sheet_low[0], sheet_high[0], sheet_low[1], sheet_high[1],
               low[0], high[0], low[1], high[1]))


def _ground_under(sheet, box):
    """The extruded ground below a sheet, after the checks every cut
    against a sheet makes: it must not fold over -- 'below' has to be one
    region -- and it must reach across everything it is to divide."""
    if not _gmt.single_valued(sheet.coordinates, sheet.triangles):
        raise NotSingleValuedError(
            "this sheet folds over, so 'below' and 'above' it are not "
            "one region each and nothing can be divided by it; a DTM3D "
            "is the kind of surface this works with")
    _reaches_across(sheet, box)
    return _ground_below(sheet, box)


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


