# geoML - machine learning models for geospatial data
# Copyright (C) 2025  Ítalo Gomes Gonçalves
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


import itertools as _iter

import numpy as _np
import pyvista as _pv
from sklearn.decomposition import PCA as _PCA
import scipy.spatial as _spatial
import warnings as _warnings
from scipy.sparse import coo_matrix as _coo_matrix
from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.sparse.csgraph import minimum_spanning_tree as _minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order as _breadth_first_order

# geometry, not drawing: `tri` locates a point in a triangulation and
# interpolates over it, which is what asking a sheet its elevation amounts to
from matplotlib import tri as _mtri


def rotation_matrix(azimuth=0.0, dip=0.0, rake=0.0):
    # conversion to radians
    azimuth = azimuth * (_np.pi / 180)
    dip = dip * (_np.pi / 180)
    rake = rake * (_np.pi / 180)

    # conversion to mathematical coordinates
    dip = - dip

    # rotation matrix
    # x and y axes are switched
    # rotation over z is with sign reversed
    rx = _np.stack([_np.cos(rake), 0, _np.sin(rake),
                    0, 1, 0,
                    -_np.sin(rake), 0, _np.cos(rake)], -1)
    rx = _np.reshape(rx, [3, 3])
    ry = _np.stack([1, 0, 0,
                    0, _np.cos(dip), -_np.sin(dip),
                    0, _np.sin(dip), _np.cos(dip)], -1)
    ry = _np.reshape(ry, [3, 3])
    rz = _np.stack([_np.cos(azimuth), _np.sin(azimuth), 0,
                    -_np.sin(azimuth), _np.cos(azimuth), 0,
                    0, 0, 1], -1)
    rz = _np.reshape(rz, [3, 3])

    rot = _np.matmul(_np.matmul(rz, ry), rx)
    return rot.T


def rotation_matrix_from_points(points):
    pca = _PCA()
    pca.fit(points)
    rotmat = pca.components_[[1, 0, 2]]
    normal = vector_product(rotmat[0], rotmat[1])
    prod = _np.sum(normal * rotmat[2])
    if prod < 0:
        rotmat[1] *= -1
    # elif rotmat[0, 0] > 0:
    #     rotmat[0] *= -1
    #     rotmat[1] *= -1
    return rotmat


def azimuth_from_xy(x, y):
    ang = _np.degrees(_np.arctan2(y, x))
    ang = 90 - ang
    if ang < 0:
        ang += 360
    return ang


def dip_from_vec(vec):
    if vec.shape != (3,):
        raise ValueError('Vector must be 3D')
    x, y, z = vec
    proj = _np.sqrt(x**2 + y**2)
    dip = - _np.degrees(_np.arctan2(z, proj))
    return dip


def angles_from_rotation_matrix(rotmat):
    rotmat = rotmat.T

    if rotmat.shape == (2, 2):
        return azimuth_from_xy(rotmat[0, 1], rotmat[1, 1])
    elif rotmat.shape != (3, 3):
        raise ValueError('Rotation matrix must be 2D or 3D')

    az = azimuth_from_xy(rotmat[0, 1], rotmat[1, 1])
    dip = dip_from_vec(rotmat[:, 1])

    rotmat_2 = rotation_matrix(az, dip, 0)
    rotmat_3 = _np.matmul(rotmat_2, rotmat).T
    rake = - dip_from_vec(rotmat_3[:, 0])

    if dip < 0:
        dip = - dip
        rake = - rake
        if az < 180:
            az += 180
        else:
            az -= 180

    return az, dip, rake


def vector_product(vec1, vec2):
    vec1 = _np.asarray(vec1)
    vec2 = _np.asarray(vec2)

    normalvec = vec1[[1, 2, 0]] * vec2[[2, 0, 1]] \
                - vec1[[2, 0, 1]] * vec2[[1, 2, 0]]
    return normalvec


# -- triangulated surfaces ------------------------------------------------- #
# Arrays of points and triangles, not containers: `data.Surface3D` is what
# holds them together, and what calls these.

def fan_triangulation(faces):
    """
    Splits faces, given as vertex indices, into triangles.

    A DXF `3DFACE` has four corners and a `MESH` face may have more, neither
    of which a `Surface3D` has room for. A fan from the first corner is the
    standard split, and is exact for the convex faces a triangulated surface
    is made of.

    Parameters
    ----------
    faces : sequence
        One sequence of vertex indices per face, of any length.

    Returns
    -------
    triangles : array
        An (n, 3) array of vertex indices.
    """
    if all(len(face) == 3 for face in faces):
        return _np.asarray(faces, dtype=int).reshape([-1, 3])

    triangles = []
    for face in faces:
        for corner in range(1, len(face) - 1):
            triangles.append((face[0], face[corner], face[corner + 1]))
    return _np.asarray(triangles, dtype=int).reshape([-1, 3])


def vertex_normals(points, triangles):
    """
    The unit normal at each vertex, from the triangles meeting there.

    A DXF file carries no normals and `Surface3D` keeps one per vertex, as
    `marching_cubes` hands them over. The cross product of a triangle's edges
    has twice the triangle's area for its length, so summing the face normals
    before normalizing weights each one by its area — which keeps a large face
    from being outvoted by the slivers around it.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.

    Returns
    -------
    normals : array
        An (n, 3) array of unit vectors, one per vertex.
    """
    corners = points[triangles]
    face_normals = _np.cross(corners[:, 1] - corners[:, 0],
                             corners[:, 2] - corners[:, 0])

    normals = _np.zeros_like(points)
    index = triangles.ravel()
    for axis in range(3):
        normals[:, axis] = _np.bincount(
            index, weights=_np.repeat(face_normals[:, axis], 3),
            minlength=points.shape[0])

    length = _np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / _np.where(length > 0, length, 1)


def weld(points, triangles, precision=6):
    """
    Merges vertices sitting at the same place, remapping the triangles.

    Whether a surface is closed is a question about its edges, and an edge is
    only shared if the triangles meeting along it say so with the same two
    indices. Plenty of meshes are closed in space while indexing every
    triangle's corners separately — `pyvista.Cylinder` is one — and welding is
    what lets the seams be seen for what they are.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    precision : int
        Decimal places the coordinates are matched to.

    Returns
    -------
    points : array
        The distinct vertices.
    triangles : array
        The triangles, indexing into them.
    """
    keys = _np.round(_np.asarray(points, dtype=float), precision)
    unique, index = _np.unique(keys, axis=0, return_inverse=True)
    return unique, index.ravel()[_np.asarray(triangles)]


def drop_degenerate_faces(points, triangles, precision=6):
    """
    Removes the faces of a triangulation that bound nothing.

    Two kinds go: a triangle whose corners are not three distinct places,
    which has no area, and a face carrying a twin, which encloses no volume
    with it. A contour of an unstructured grid emits both wherever the
    surface passes exactly through a cell corner — the level set pinches to
    a point there, and the marching cubes case that covers it writes the
    slivers out anyway.

    They matter because they are read as a winding failure. Both make an
    edge appear twice the same way round, so `reversed_edges` counts them
    and `mesh3d` returns a plain `Mesh3D` for a surface that is closed and
    consistent everywhere it has area.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    precision : int
        Decimal places the coordinates are welded to, as in `weld`.

    Returns
    -------
    points : array
        The vertices, exactly as they came.
    triangles : array
        The faces that survived, in the order they came, still indexing the
        original vertices.

    Notes
    -----
    Coincidence is judged on welded indices and reported on the caller's,
    so a vertex list is never reordered or renumbered. That matters: `weld`
    sorts, and `from_dxf` promises a round trip that returns a file's own
    vertices in its own order. Vertices left unused by a dropped face stay
    where they are, costing a row and confusing nothing — every measure
    that cares welds for itself.

    **How many copies of a twinned face to drop is worked out on the
    surface, not predicted.** Dropping every copy is right for a
    zero-thickness flap, whose edges are either its own alone or already
    carried by the surface it lies against; it is wrong for a wall that
    happens to be recorded twice, where it would leave each edge bounding
    one face — an open edge where there is no hole. And the two cannot be
    told apart one group at a time, because the groups couple: in a doubled
    *patch* — a membrane several faces wide, which is what decimation makes
    of a collapsed thin feature — the middle face sees every neighbour as a
    twin, and any rule that scores its edges in isolation drops it while
    its neighbours stay, tearing a hole down the middle of the patch.

    So duplicated faces start out dropped, and copies are put back one at a
    time wherever the surface as it stands is left with an edge on exactly
    one face — a boundary the mesh did not have. Each pass resurrects at
    least one group or stops, so the loop is bounded by the number of
    groups, and each resurrection can heal the edges of the next: the rim
    of a doubled patch comes back first, and the middle on the pass after,
    once the rim's return has left its edges half-supported.
    """
    triangles = _np.asarray(triangles)
    _, welded = weld(points, triangles, precision)

    distinct = ((welded[:, 0] != welded[:, 1])
                & (welded[:, 1] != welded[:, 2])
                & (welded[:, 0] != welded[:, 2]))
    # collinear faces -- three distinct corners on one line -- are NOT
    # dropped, and the restraint is measured: on a real 1.2M-triangle
    # indicator contour they are structural in their tens of thousands,
    # and dropping them tore 46 902 edges open while fixing nothing
    rows = _np.flatnonzero(distinct)
    good = welded[distinct]
    if len(good) == 0:
        return points, triangles[rows]

    _, first, inverse, counts = _np.unique(
        _np.sort(good, axis=1), axis=0, return_index=True,
        return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    span = int(good.max()) + 1
    edges = _np.sort(_np.stack(
        [good[:, [0, 1]], good[:, [1, 2]], good[:, [2, 0]]]), axis=-1)
    keys = edges[..., 0].astype(_np.int64) * span + edges[..., 1]

    keep = counts[inverse] == 1
    waiting = _np.flatnonzero(counts > 1)
    for _ in range(len(waiting) + 1):
        seen, times = _np.unique(keys[:, keep].ravel(), return_counts=True)
        open_keys = seen[times == 1]
        if len(open_keys) == 0 or len(waiting) == 0:
            break
        touches = _np.isin(keys[:, first[waiting]], open_keys).any(axis=0)
        if not touches.any():
            break
        keep[first[waiting[touches]]] = True
        waiting = waiting[~touches]

    return points, triangles[rows[keep]]


def _counts_of(n_points, triangles, directed):
    """How often each edge of a *welded* triangulation appears."""
    edges = _np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                             triangles[:, [2, 0]]], axis=0)
    if not directed:
        edges = _np.sort(edges, axis=1)
    key = edges[:, 0].astype(_np.int64) * n_points + edges[:, 1]
    _, counts = _np.unique(key, return_counts=True)
    return counts


def _edge_counts(points, triangles, precision, directed):
    """How often each edge appears, undirected (shared) or directed (wound)."""
    points, triangles = weld(points, triangles, precision)
    return _counts_of(points.shape[0], triangles, directed)


def edge_defects(points, triangles, precision=6):
    """
    Both edge counts a mesh is judged on, from one welding.

    `open_edges` and `reversed_edges` ask two questions of the same welded
    triangulation, and every mesh is built asking both. Welding is the
    expensive half — a rounding and a `unique` over the vertices — so doing
    it once for the pair is worth the one extra function: measured on 27 656
    triangles, the two separately cost 9 ms and 7 ms of which 4 ms was each
    one's welding, and together they cost 12 ms.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    precision : int
        Decimal places the coordinates are welded to.

    Returns
    -------
    open_count : int
        Edges belonging to a single triangle: none on a closed mesh.
    reversed_count : int
        Edges walked the same way round by both triangles sharing them:
        none where the winding is consistent.

    See Also
    --------
    open_edges, reversed_edges : the same numbers, one question at a time.
    """
    points, triangles = weld(points, triangles, precision)
    n_points = points.shape[0]
    return (int(_np.sum(_counts_of(n_points, triangles, False) == 1)),
            int(_np.sum(_counts_of(n_points, triangles, True) > 1)))


def open_edges(points, triangles, precision=6):
    """
    How many of a surface's edges belong to a single triangle.

    None on a closed body, where every edge is shared by two faces; at least
    the outline on a sheet. It is what tells the two apart, and so which
    questions a surface can answer — a body has no elevation above a location,
    and a sheet has no inside. Vertices are welded first, so a mesh that is
    closed in space counts as closed however its corners are indexed.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    precision : int
        Decimal places the coordinates are welded to.

    Returns
    -------
    count : int
        The number of edges belonging to one triangle only.
    """
    return int(_np.sum(_edge_counts(points, triangles, precision, False) == 1))


def reversed_edges(points, triangles, precision=6):
    """
    How many edges the triangles sharing them walk the same way round.

    None where the winding is consistent: two triangles meeting along an edge
    traverse it in opposite directions, which is what makes "outward" mean one
    thing over a whole closed surface. Any at all and some triangle faces the
    wrong way, which an inside/outside test reads as a hole in the body —
    quietly, and only in the region the offending faces bound.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    precision : int
        Decimal places the coordinates are welded to.

    Returns
    -------
    count : int
        The number of edges traversed more than once in the same direction.
    """
    return int(_np.sum(_edge_counts(points, triangles, precision, True) > 1))


def area(points, triangles):
    """
    The surface area of a triangulation.

    Meaningful whether or not the surface closes, unlike its volume.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.

    Returns
    -------
    area : float
    """
    corners = _np.asarray(points, dtype=float)[_np.asarray(triangles)]
    crossed = _np.cross(corners[:, 1] - corners[:, 0],
                        corners[:, 2] - corners[:, 0])
    return float(_np.sum(_np.linalg.norm(crossed, axis=1)) / 2)


def components(points, triangles, precision=6):
    """
    Labels the triangles by the connected piece of surface they belong to.

    A boolean operation readily answers with a surface in several pieces — an
    ore body cut in two, a shell around a cavity — and each piece is a body in
    its own right. Vertices are welded first, since pieces that touch only
    through unwelded corners are one piece in space.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    precision : int
        Decimal places the coordinates are welded to.

    Returns
    -------
    count : int
        How many pieces there are.
    labels : array
        One piece number per triangle.
    """
    welded_points, welded = weld(points, triangles, precision)
    if welded.shape[0] == 0:
        return 0, _np.zeros([0], dtype=int)

    rows = _np.concatenate([welded[:, 0], welded[:, 1], welded[:, 2]])
    cols = _np.concatenate([welded[:, 1], welded[:, 2], welded[:, 0]])
    graph = _coo_matrix((_np.ones(rows.size), (rows, cols)),
                        shape=(welded_points.shape[0],) * 2)
    count, vertex_label = _connected_components(graph, directed=False)

    # a triangle belongs to the piece its corners do -- all three of them,
    # an edge between them being what put them in the same piece
    return int(count), vertex_label[welded[:, 0]]


def single_valued(points, triangles, tolerance=1e-9):
    """
    Whether a surface stands at one height over each (x, y).

    True where every triangle projects onto the ground the same way round. A
    fold or an overhang turns some of them over, and a closed body turns its
    whole underside over, so both are caught. Triangles standing vertically
    project to nothing at all and are allowed: a cliff is single valued
    everywhere except along the line of its face.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    tolerance : float
        Projected areas this much smaller than the largest count as nothing.

    Returns
    -------
    single_valued : bool
    """
    corners = _np.asarray(points, dtype=float)[_np.asarray(triangles)]
    edge_a = corners[:, 1, :2] - corners[:, 0, :2]
    edge_b = corners[:, 2, :2] - corners[:, 0, :2]
    twice_area = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    if twice_area.size == 0:
        return True

    flat = tolerance * _np.max(_np.abs(twice_area))
    return not (_np.any(twice_area > flat) and _np.any(twice_area < -flat))


def signed_volume(points, triangles):
    """
    The volume a closed surface encloses, negative if it is wound inwards.

    Each triangle forms a tetrahedron with the origin, whose signed volume is
    a sixth of the determinant of its corners; over a closed surface those
    add up to what it encloses, wherever the origin happens to be. The sign
    is the useful part: it says which way the triangles face taken together,
    which is what an inside/outside test must know and cannot learn from any
    one of them.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices, of a closed surface.

    Returns
    -------
    volume : float
        Positive where the triangles face outwards, negative where they face
        in. Meaningless for a surface that is not closed.
    """
    corners = _np.asarray(points, dtype=float)[_np.asarray(triangles)]
    return float(_np.sum(_np.einsum(
        "ij,ij->i", corners[:, 0],
        _np.cross(corners[:, 1], corners[:, 2]))) / 6.0)


def sheet_interpolator(points, triangles):
    """
    Prepares a sheet to be asked its elevation.

    The sheet must be single valued — checking that is the caller's business,
    and `open_edges` is what tells a sheet from a body. `matplotlib` takes a
    folded triangulation without complaint, answering with whichever of its
    sheets it happens to find.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.

    Returns
    -------
    interpolator : matplotlib.tri.LinearTriInterpolator
        To be handed to `sheet_elevation`.
    """
    mesh = _mtri.Triangulation(points[:, 0], points[:, 1],
                               _np.asarray(triangles))
    return _mtri.LinearTriInterpolator(mesh, points[:, 2])


def sheet_elevation(interpolator, coordinates):
    """
    The sheet's height over each location, NaN past its edge.

    Parameters
    ----------
    interpolator : matplotlib.tri.LinearTriInterpolator
        From `sheet_interpolator`.
    coordinates : array
        An (n, 2) or (n, 3) array; only the first two columns are read.

    Returns
    -------
    elevation : array
        One height per location, NaN where the sheet does not reach.
    """
    elevation = interpolator(coordinates[:, 0], coordinates[:, 1])
    return _np.ma.filled(elevation.astype(float), _np.nan)


def inside_solid(mesh, coordinates):
    """
    Whether each location falls within a closed body, asking VTK.

    Parameters
    ----------
    mesh : pyvista.PolyData
        The body, which must be watertight — checking that is the caller's
        business, and `open_edges` is what tells it.
    coordinates : array
        An (n, 3) array of locations.

    Returns
    -------
    inside : array
        One boolean per location.
    """
    cloud = _pv.PolyData(_np.ascontiguousarray(coordinates))
    selected = cloud.select_interior_points(mesh, check_surface=False)
    return _np.asarray(selected["selected_points"]).astype(bool)


def _circumradius(vertices):
    """Circumradius of each simplex, `[n_simplices]`, from its vertices
    `[n_simplices, n_dim + 1, n_dim]`. Solves for the circumcentre: the point
    equidistant from every vertex. A flat simplex has no circumsphere and
    reads as infinite, which is what excludes it from any hull."""
    first = vertices[:, :1, :]
    rest = vertices[:, 1:, :]
    system = 2.0 * (rest - first)                              # [n, d, d]
    target = (rest ** 2).sum(axis=2) - (first ** 2).sum(axis=2)
    determinant = _np.linalg.det(system)
    scale = _np.abs(system).max(axis=(1, 2)) + 1e-300
    flat = _np.abs(determinant) < 1e-12 * scale ** system.shape[1]
    safe = _np.where(flat[:, None, None], _np.eye(system.shape[1])[None],
                     system)
    centre = _np.linalg.solve(safe, target[:, :, None])[:, :, 0]
    radius = _np.sqrt(((centre - first[:, 0, :]) ** 2).sum(axis=1))
    return _np.where(flat, _np.inf, radius)


class ConcaveHull:
    """
    The alpha shape of a point set: a concave hull at a chosen length.

    The Delaunay triangulation's simplices are kept where their circumradius
    is below `length`, so the hull follows the data at that scale -- it
    fills the interior between drill fences closer than `length` apart and
    leaves out a notch wider than that, where a convex hull would bridge
    the notch and a ball around each sample would leave the interior out.
    Built by `concave_hull`.

    Attributes
    ----------
    points : array
        The `(n_points, n_dim)` input.
    length : float
        The circumradius the simplices were kept under.
    kept : array
        One boolean per Delaunay simplex.
    simplices : array
        The kept simplices, as vertex indices into `points`.
    boundary : array
        The facets of kept simplices that face an unkept one or nothing --
        `(n_facets, n_dim)` vertex indices: segments in 2D, triangles in 3D.
    """

    def __init__(self, points, length):
        from scipy.spatial import Delaunay
        points = _np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be an (n_points, n_dim) array")
        self.points = points
        self.length = float(length)
        self._delaunay = Delaunay(points)
        radius = _circumradius(points[self._delaunay.simplices])
        self.kept = radius < self.length
        self.simplices = self._delaunay.simplices[self.kept]

        # a facet is on the boundary when the simplex across it is unkept
        # or absent; `neighbors[i, j]` is the simplex across the facet
        # opposite vertex `j`
        neighbors = self._delaunay.neighbors
        across = _np.where(neighbors >= 0, self.kept[neighbors], False)
        facets = []
        n_vertices = self._delaunay.simplices.shape[1]
        for i in _np.flatnonzero(self.kept):
            for j in range(n_vertices):
                if not across[i, j]:
                    facets.append(_np.delete(self._delaunay.simplices[i], j))
        self.boundary = _np.asarray(facets, dtype=int).reshape(
            -1, n_vertices - 1)

    def contains(self, coordinates):
        """
        Whether each location lies inside a kept simplex.

        Parameters
        ----------
        coordinates
            `(n, n_dim)` locations.

        Returns
        -------
        inside : array
            `(n,)` booleans.
        """
        coordinates = _np.asarray(coordinates, dtype=float)
        index = self._delaunay.find_simplex(coordinates)
        return _np.where(index >= 0, self.kept[_np.maximum(index, 0)], False)


def concave_hull(points, length):
    """
    The concave hull of a point set at a length scale -- an alpha shape.

    Parameters
    ----------
    points
        `(n_points, n_dim)` sample locations, 2D or 3D.
    length
        The largest circumradius a Delaunay simplex may have and still be
        part of the hull: a length in the coordinates' units, so a hull
        "at 100 m" spans gaps in the data narrower than that and stops at
        wider ones.

    Returns
    -------
    ConcaveHull
        With `contains(coordinates)` for the inside test and `boundary` for
        the facets.

    Raises
    ------
    ValueError
        If the points do not span their space (fewer than `n_dim + 1` of
        them, or all on a line or plane), which leaves nothing to
        triangulate.
    """
    from scipy.spatial import QhullError
    try:
        return ConcaveHull(points, length)
    except QhullError as error:
        raise ValueError(
            "a concave hull needs points that span their space -- at least "
            "n_dim + 1 of them, not all on one line or plane; Qhull said: %s"
            % str(error).splitlines()[0]) from None


def bounding_box(points):
    """
    Computes a point set's bounding box and its diagonal.

    Parameters
    ----------
    points : array
        A set of coordinates.

    Returns
    -------
    bbox : array-like
        Array with the box's minimum and maximum values in each direction.
    d : float
        The box's diagonal length.
    """
    if len(points.shape) < 2:
        points = _np.expand_dims(points, axis=0)
    bbox = _np.array([[_np.min(points[:, i]) for i in range(points.shape[1])],
                      [_np.max(points[:, i]) for i in range(points.shape[1])]])
    d = _np.sqrt(sum([
        _np.diff(bbox[:, i]) ** 2 for i in range(bbox.shape[1])]))
    d = _np.squeeze(d)
    return bbox, d


def _cell_weights(coordinates, cell, origins):
    """Weights for one cell size, averaged over shifted cell origins.

    Where the lattice happens to start is arbitrary, and on a small sample it
    moves the answer, so the weights are averaged over `origins` evenly
    shifted lattices rather than computed on one.
    """
    total = _np.zeros(coordinates.shape[0])
    for k in range(origins):
        shift = cell * k / origins
        key = _np.floor((coordinates + shift) / cell).astype(_np.int64)
        _, inverse, counts = _np.unique(key, axis=0, return_inverse=True,
                                        return_counts=True)
        total = total + 1.0 / counts[inverse]
    return total / origins


def declustering_weights(coordinates, values=None, cell=None, origins=4,
                         n_sizes=24):
    """
    Cell-declustering weights, one per location.

    Samples are rarely laid down evenly. Drilling follows the ore, so the
    interesting ground is crowded and the rest is sparse, and every statistic
    that treats the samples as equal votes then describes the *sampling*
    rather than the field. Cell declustering is the classical repair: lay a
    lattice over the data, and split one vote among the samples sharing a
    cell, so a crowded cell speaks once rather than twenty times.

    Parameters
    ----------
    coordinates
        `(n_data, n_dim)` sample locations.
    values
        Sample values, needed only to choose `cell` automatically.
    cell
        Cell side. Chosen automatically when absent, which needs `values`.
    origins
        How many shifted lattices to average the weights over. Where the
        lattice starts is arbitrary and on a small sample it moves the
        answer.
    n_sizes
        Cell sides tried when choosing one.

    Returns
    -------
    weights : array
        `(n_data,)`, summing to `n_data`, so that a set of evenly spread
        samples comes back at one apiece.
    cell : float
        The size used, whether given or chosen.

    Notes
    -----
    The automatic choice follows the usual practice (Deutsch & Journel's
    `declus`): sweep the cell size and keep the one whose declustered mean
    departs furthest from the naive one. Both extremes of the sweep return
    the naive mean -- a cell below the sample spacing gives every point its
    own vote, and one larger than the domain puts them all in one cell -- so
    the departure has an interior maximum, and taking it by absolute value
    handles clustering in high and in low values alike without being told
    which happened.

    References
    ----------
    Deutsch, C. V., & Journel, A. G. (1998). *GSLIB: Geostatistical Software
    Library and User's Guide* (2nd ed.). Oxford University Press.
    """
    coordinates = _np.asarray(coordinates, dtype=float)
    n_data = coordinates.shape[0]
    if n_data < 2:
        return _np.ones(n_data), float(cell or 1.0)

    if cell is None:
        if values is None:
            raise ValueError(
                "choosing a cell size needs the values; pass `values`, or "
                "give `cell` directly")
        values = _np.asarray(values, dtype=float).ravel()
        usable = _np.isfinite(values)
        if usable.sum() < 2:
            return _np.ones(n_data), 0.0

        span = coordinates.max(axis=0) - coordinates.min(axis=0)
        largest = float(_np.max(span))
        if largest <= 0:
            return _np.ones(n_data), 0.0

        # from about the sample spacing to about the domain: below the first
        # every point is alone in its cell, above the second they all share
        # one, and the weights are flat at both ends
        spacing = largest / max(n_data ** (1.0 / coordinates.shape[1]), 1.0)
        sizes = _np.geomspace(max(spacing, largest * 1e-3), largest, n_sizes)

        naive = float(values[usable].mean())
        best, cell = -1.0, float(sizes[0])
        for size in sizes:
            weights = _cell_weights(coordinates, size, origins)
            share = weights[usable]
            mean = float((share * values[usable]).sum() / share.sum())
            departure = abs(mean - naive)
            if departure > best:
                best, cell = departure, float(size)

    weights = _cell_weights(coordinates, cell, origins)
    return weights * (n_data / weights.sum()), float(cell)


def sub_block_index(discretization):
    """Which sub-block sits where, as integer counts along each axis.

    Axis 0 varies fastest, the order `_blockdata` has always used and the one
    the likelihood's noise is indexed by. Both the sub-block offsets and, in a
    `BlockSet3D`, the children of a split are built from this, so sub-block
    `j` of a block and child `j` of that same block are the same corner of it.
    """
    return _np.array(
        list(_iter.product(*[_np.arange(d) for d in discretization[::-1]])),
        dtype=_np.int64)[:, ::-1]


def unit_sub_grid(discretization):
    """Sub-block offsets from a block's centre, as fractions of its size.

    The same layout `_blockdata` builds, but divided through by the block so
    that one array serves every size. Scaling it per block is the whole of
    what a variable-size block model has to do differently when it fans out.
    """
    counts = _np.array(discretization)[None, :]
    return (sub_block_index(discretization) - (counts - 1) / 2) / counts


# a hexahedron's eight corners, in the order VTK reads them
HEX_CORNERS = _np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                        dtype=_np.int64)


def trilinear_weights(discretization):
    """What each of a block's eight corners is worth at each sub-block centre.

    A corner carries what the blocks meeting there say, so reading the corners
    at the sub-blocks is how a child learns the shape running across its
    parent. The layout is symmetric about the centre, so the weights average
    to an eighth apiece and a correction built from them cancels over the
    children -- which is what keeps a block's own estimate the mean of the
    children standing in for it.
    """
    t = unit_sub_grid(discretization) + 0.5
    return _np.prod(_np.where(HEX_CORNERS[None, :, :] == 1,
                              t[:, None, :], 1.0 - t[:, None, :]), axis=2)


def dyadic_overlaps(origin, size):
    """The cells of a dyadic tiling that sit inside a larger one.

    Cells aligned to their own power-of-two size — origin a multiple of it,
    per axis — are either disjoint or nested, never partially overlapping,
    so overlap detection collapses to an ancestor lookup: a cell overlaps
    iff the cube one of the present sizes above it is itself present. Built
    for reading foreign octrees, whose files are not ours to trust.

    Parameters
    ----------
    origin : array
        `(n, 3)` integer cell origins, each a multiple of its cell's size.
    size : array
        `(n,)` integer cell sizes, powers of two.

    Returns
    -------
    offending : array
        Indices of the cells contained in some larger present cell; empty
        when the tiling is sound.
    """
    origin = _np.asarray(origin, dtype=_np.int64)
    size = _np.asarray(size, dtype=_np.int64)
    present = {(int(i), int(j), int(k), int(s))
               for (i, j, k), s in zip(origin, size)}
    offending = []
    for row, ((i, j, k), s) in enumerate(zip(origin, size)):
        for bigger in _np.unique(size[size > s]):
            bigger = int(bigger)
            parent = (int(i) // bigger * bigger, int(j) // bigger * bigger,
                      int(k) // bigger * bigger, bigger)
            if parent in present:
                offending.append(row)
                break
    return _np.asarray(offending, dtype=_np.int64)


def dyadic_complement(origin, size, shape, coarsest):
    """The gaps of a dyadic tiling, as the largest aligned cells that fit.

    What makes a partial octree full again: a foreign file usually carries
    only the cells inside a domain of interest, and the always-full design
    wants the rest present and marked rather than absent. Walks the box top
    down — a candidate cube that is a present cell, or sits inside one, is
    covered; one holding a present descendant splits into its eight
    children; one holding nothing is a gap, emitted at that size.

    Parameters
    ----------
    origin, size : arrays
        The present cells, as in `dyadic_overlaps` — already checked not to
        overlap, since a nested pair here would be read as coverage.
    shape : array
        `(3,)` the box to fill, in cells; each axis a multiple of
        `coarsest`.
    coarsest : int
        The largest cell size to emit, a power of two; the recursion
        starts on the grid of cubes this size.

    Returns
    -------
    gap_origin, gap_size : arrays
        The cells that complete the tiling; empty when it already is one.
    """
    origin = _np.asarray(origin, dtype=_np.int64)
    size = _np.asarray(size, dtype=_np.int64)
    shape = _np.asarray(shape, dtype=_np.int64)
    coarsest = int(coarsest)

    present = {(int(i), int(j), int(k), int(s))
               for (i, j, k), s in zip(origin, size)}
    # every dyadic cube standing above a present cell, so "holds something
    # finer" is one lookup
    above = set()
    for (i, j, k), s in zip(origin, size):
        bigger = int(s) * 2
        while bigger <= coarsest:
            above.add((int(i) // bigger * bigger,
                       int(j) // bigger * bigger,
                       int(k) // bigger * bigger, bigger))
            bigger *= 2

    gap_origin, gap_size = [], []
    stack = [(int(i), int(j), int(k), coarsest)
             for i in range(0, int(shape[0]), coarsest)
             for j in range(0, int(shape[1]), coarsest)
             for k in range(0, int(shape[2]), coarsest)]
    while stack:
        cube = stack.pop()
        if cube in present:
            continue
        i, j, k, s = cube
        if cube in above:
            half = s // 2
            stack.extend(
                (i + di * half, j + dj * half, k + dk * half, half)
                for di in (0, 1) for dj in (0, 1) for dk in (0, 1))
            continue
        gap_origin.append((i, j, k))
        gap_size.append(s)
    if not gap_origin:
        return (_np.zeros((0, 3), dtype=_np.int64),
                _np.zeros((0,), dtype=_np.int64))
    return (_np.asarray(gap_origin, dtype=_np.int64),
            _np.asarray(gap_size, dtype=_np.int64))


def grow(corners, marked, rings):
    """Add `rings` of neighbouring blocks, through the corners blocks share.

    Sparse on purpose: dilating a mask over the base lattice would cost a cell
    for every one the model exists to avoid carrying.
    """
    for _ in range(rings):
        touched = _np.zeros(int(corners.max()) + 1, dtype=bool)
        touched[corners[marked].ravel()] = True
        marked = touched[corners].any(axis=1)
    return marked


def point_normals(points, k=12, orient="concave"):
    """
    Unit normals of a scattered point set lying on a surface.

    Each normal is the direction of least spread among a point's nearest
    neighbours (Hoppe et al. 1992), which fixes it up to sign. The sign is
    made consistent over the whole set by propagating it along a minimum
    spanning tree of the neighbour graph, from one root per connected
    piece, so that the field can serve as a gradient constraint.

    Parameters
    ----------
    points
        `(n, 2)` or `(n, 3)`.
    k
        Neighbours per point.
    orient
        `"concave"` (default): the root's normal points toward the side the
        surface curves to, read from where its neighbours' centroid falls
        relative to the tangent plane; the root is the most curved point of
        its piece. A surface that reads as flat has no such side, which is
        warned about, and the sign is then whatever the root drew. A vector
        instead orients every normal to have a positive component along it.

    Returns
    -------
    normals : array
        `(n, d)` unit normals.

    Raises
    ------
    ValueError
        In three dimensions, when the points read as a single line, which
        cannot fix a normal: give the normals, or points off the line.

    References
    ----------
    Hoppe, H., DeRose, T., Duchamp, T., McDonald, J. and Stuetzle, W.
    (1992). Surface reconstruction from unorganized points. SIGGRAPH 1992,
    71-78.
    """
    points = _np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must be (n, 2) or (n, 3)")
    n, d = points.shape
    if n < d + 1:
        raise ValueError("at least %d points are needed" % (d + 1))
    k = int(min(k, n - 1))
    tree = _spatial.cKDTree(points)
    distance, index = tree.query(points, k=k + 1)
    neighbours = points[index]                       # [n, k + 1, d], self first
    centroid = neighbours.mean(axis=1)
    centred = neighbours - centroid[:, None, :]
    covariance = _np.einsum("nki,nkj->nij", centred, centred) / (k + 1)
    eigenvalues, eigenvectors = _np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].copy()
    if d == 3:
        # a line spreads along one direction only: the two smallest
        # eigenvalues vanish together and no normal is defined
        flat = eigenvalues[:, 1] <= 1e-6 * eigenvalues[:, 2] + 1e-300
        if flat.mean() > 0.5:
            raise ValueError(
                "the points read as a single line, which cannot fix a "
                "normal in three dimensions; give the normals, or points "
                "off the line")

    if not isinstance(orient, str):
        vector = _np.asarray(orient, dtype=float).ravel()
        if len(vector) != d:
            raise ValueError("orient must be a %d-vector" % d)
        sign = _np.sign(normals @ vector)
        sign[sign == 0] = 1.0
        return normals * sign[:, None]
    if orient != "concave":
        raise ValueError("orient must be 'concave' or a vector")

    # how far, and to which side, the neighbourhood bends off the tangent
    # plane, relative to its own spacing: the concavity signal
    offset = _np.einsum("ni,ni->n", centroid - points, normals)
    spacing = _np.median(distance[:, 1:], axis=1)
    confidence = _np.abs(offset) / _np.where(spacing > 0, spacing, 1.0)

    # the neighbour graph, weighted by how much adjacent normals disagree
    rows = _np.repeat(_np.arange(n), k)
    cols = index[:, 1:].ravel()
    agreement = _np.abs(_np.einsum("ni,ni->n", normals[rows], normals[cols]))
    weight = 1.0 - agreement + 1e-9
    graph = _coo_matrix((weight, (rows, cols)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    forest = _minimum_spanning_tree(graph)
    forest = forest.maximum(forest.T)
    n_pieces, piece = _connected_components(graph, directed=False)

    weakest = _np.inf
    for p in range(n_pieces):
        members = _np.where(piece == p)[0]
        root = members[_np.argmax(confidence[members])]
        weakest = min(weakest, confidence[root])
        if offset[root] < 0:
            normals[root] = -normals[root]
        order, predecessor = _breadth_first_order(
            forest, root, directed=False, return_predecessors=True)
        for node in order[1:]:
            parent = predecessor[node]
            if normals[node] @ normals[parent] < 0:
                normals[node] = -normals[node]
    if weakest < 0.02:
        _warnings.warn(
            "the surface reads as flat: it has no concave side, so the "
            "normals' sign is consistent but arbitrary; pass a vector as "
            "`orient` to fix it", UserWarning)
    return normals
