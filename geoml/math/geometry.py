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
from scipy.sparse import coo_matrix as _coo_matrix
from scipy.sparse.csgraph import connected_components as _connected_components

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
    ang = _np.degrees(_np.atan2(y, x))
    ang = 90 - ang
    if ang < 0:
        ang += 360
    return ang


def dip_from_vec(vec):
    if vec.shape != (3,):
        raise ValueError('Vector must be 3D')
    x, y, z = vec
    proj = _np.sqrt(x**2 + y**2)
    dip = - _np.degrees(_np.atan2(z, proj))
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


def _edge_counts(points, triangles, precision, directed):
    """How often each edge appears, undirected (shared) or directed (wound)."""
    points, triangles = weld(points, triangles, precision)
    edges = _np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                             triangles[:, [2, 0]]], axis=0)
    if not directed:
        edges = _np.sort(edges, axis=1)
    key = edges[:, 0].astype(_np.int64) * points.shape[0] + edges[:, 1]
    _, counts = _np.unique(key, return_counts=True)
    return counts


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
