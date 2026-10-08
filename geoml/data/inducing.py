# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
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
Building the inducing points a latent network is given.

`latent.BasicInput` takes either one `PointData` of inducing points or a list
of them, one per expert, and until now both had to be assembled by hand. The
functions here produce them:

    from_kmeans(data, n)          one set, at the k-means centroids of the data
    from_grid(data, step)         one set, on a regular lattice
    from_hull(data, step, d)      one set, on a lattice kept where the data reach
    combine(a, b, ...)            one set out of several, duplicates dropped
    grid_experts(data, step)      a list of sets, laid out as overlapping blocks
    experts(points, n_experts)    a list of sets, from overlapping clusters

The two `*_experts` functions divide a set of inducing points among experts so
that neighbouring ones overlap, which is what keeps a prediction from showing a
seam where the experts meet. They differ only in how the division is made.
`grid_experts` cuts space into regular blocks and extends each by one step, so
every expert is the same size and its neighbours are known in advance -- the
Moore neighbourhood, 8 in the plane and 26 in space. `experts` is the unordered
counterpart: it clusters whatever points it is given into compact clusters of
about the same size and lends each the points nearest its own, which suits a
survey that does not fill its bounding box, such as drillholes or a
shoreline.

`experts` takes inducing points rather than data, so the usual way to build an
irregular network is to choose the points first and then divide them:

    sets = experts(from_kmeans(data, 1500), 12)
"""

__all__ = ["from_kmeans", "from_grid", "from_hull", "combine", "grid_experts",
           "experts"]

from typing import cast as _cast

import numpy as _np
import scipy.optimize as _optimize
import scipy.sparse as _sparse
import scipy.spatial as _spatial
from sklearn.cluster import KMeans as _KMeans

import geoml._types as _types
import geoml.data.containers as _data


def _coordinates(source):
    """The coordinates of a container, or an array taken as they are."""
    if hasattr(source, "coordinates"):
        return _np.asarray(source.coordinates, dtype=float)
    return _np.array(source, ndmin=2, dtype=float)


def _as_points(coordinates, labels=None):
    return _data.PointData.from_array(
        _np.ascontiguousarray(coordinates, dtype=float), labels)


def from_kmeans(data: "_data._SpatialData | _types.ArrayLike", n: int,
                seed: int | None = None) -> "_data.PointData":
    """
    Inducing points at the k-means centroids of the data.

    The centroids follow the data's density: many inducing points where the
    samples are crowded, few where they are sparse, which is where a sparse
    GP needs them. Deterministic for a given `seed`.

    Parameters
    ----------
    data
        A spatial container, or an `(n_data, n_dim)` array of coordinates.
    n
        Number of inducing points. Must not exceed the number of data
        points.
    seed
        Passed to `sklearn.cluster.KMeans` for a reproducible result. This
        is separate from :func:`geoml.set_seed`, which governs the
        model's parameter initialization.

    Returns
    -------
    geoml.data.PointData
        `n` points, in no particular order.
    """
    coordinates = _coordinates(data)
    n = int(n)
    if not 1 <= n <= coordinates.shape[0]:
        raise ValueError(
            "n must be between 1 and the number of data points (%d), got %d"
            % (coordinates.shape[0], n))

    centers = _KMeans(n, n_init=10, random_state=seed).fit(
        coordinates).cluster_centers_
    return _as_points(centers, getattr(data, "coordinate_labels", None))


def _lattice_axes(coordinates, step, nodes_per_axis=None):
    """
    Node positions per axis for a regular lattice covering the data.

    Centred on the data, so the padding needed to reach a whole number of
    steps is split between the two ends instead of piling up at one.
    """
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    middle = 0.5 * (low + high)

    axes = []
    for d in range(coordinates.shape[1]):
        if nodes_per_axis is None:
            count = int(_np.ceil((high[d] - low[d]) / step[d])) + 1
        else:
            count = int(nodes_per_axis[d])
        count = max(count, 2)
        start = middle[d] - 0.5 * (count - 1) * step[d]
        axes.append(start + _np.arange(count) * step[d])
    return axes


def _lattice(axes):
    """The Cartesian product of the axes, first axis varying slowest."""
    mesh = _np.meshgrid(*axes, indexing="ij")
    return _np.stack([m.ravel() for m in mesh], axis=1)


def _step_vector(step, n_dim):
    step = _np.array(step, ndmin=1, dtype=float)
    if step.size == 1:
        step = _np.repeat(step, n_dim)
    if step.size != n_dim:
        raise ValueError(
            "step must be a scalar or have one entry per dimension (%d), "
            "got %d" % (n_dim, step.size))
    if _np.any(step <= 0):
        raise ValueError("step must be positive")
    return step


def from_grid(data: "_data._SpatialData | _types.ArrayLike",
              step: "float | _types.ArrayLike") -> "_data.PointData":
    """
    Inducing points on a regular lattice covering the data.

    Evenly spread whatever the data's density, so the model has something
    to say away from the samples too. Often combined with `from_kmeans`,
    through `combine`.

    Parameters
    ----------
    data
        A spatial container, or an `(n_data, n_dim)` array of coordinates.
    step
        Spacing between neighbouring inducing points, one value per
        dimension or a single value for all of them.

    Returns
    -------
    geoml.data.PointData
        The lattice nodes, the first axis varying slowest.
    """
    coordinates = _coordinates(data)
    step = _step_vector(step, coordinates.shape[1])
    nodes = _lattice(_lattice_axes(coordinates, step))
    return _as_points(nodes, getattr(data, "coordinate_labels", None))


def _inside_hull(coordinates, nodes):
    """Which `nodes` lie inside the convex hull of `coordinates`. Data that
    span fewer dimensions than they have -- a section in space, a single
    line -- enclose no volume, and nothing is inside."""
    if coordinates.shape[1] == 1:
        return (nodes[:, 0] >= coordinates[:, 0].min())             & (nodes[:, 0] <= coordinates[:, 0].max())
    try:
        triangulation = _spatial.Delaunay(coordinates)
    except _spatial.QhullError:
        return _np.zeros(len(nodes), dtype=bool)
    return triangulation.find_simplex(nodes) >= 0


def from_hull(data: "_data._SpatialData | _types.ArrayLike",
              step: "float | _types.ArrayLike",
              distance: float) -> "_data.PointData":
    """
    Inducing points on a regular lattice, kept where the data reach.

    The lattice of `from_grid`, extended by `distance` beyond the data's box,
    loses the nodes the data say nothing about: every node inside the
    data's convex hull stays, and a node outside it stays only within
    `distance` of a sample. A survey that does not fill its box -- a fan of
    drillholes, a shoreline -- keeps an even backbone where it is, and a
    margin of `distance` around it, without the nodes in the empty corners.

    Parameters
    ----------
    data
        A spatial container, or an `(n_data, n_dim)` array of coordinates.
    step
        Spacing between neighbouring nodes, one value per dimension or a
        single value for all of them.
    distance
        How far outside the convex hull a node may lie from the nearest
        sample and stay. Zero keeps the hull alone. Data that enclose no
        volume -- every sample on one plane in space, say -- have nothing
        inside, and the distance alone decides.

    Returns
    -------
    geoml.data.PointData
        The nodes kept, the first axis varying slowest.

    See Also
    --------
    from_grid : the whole lattice over the data's box.
    """
    coordinates = _coordinates(data)
    step = _step_vector(step, coordinates.shape[1])
    if distance < 0:
        raise ValueError("distance must not be negative, got %r" % distance)
    margin = _np.vstack([coordinates.min(axis=0) - distance,
                         coordinates.max(axis=0) + distance])
    nodes = _lattice(_lattice_axes(margin, step))
    keep = _inside_hull(coordinates, nodes)
    outside = _np.flatnonzero(~keep)
    if outside.size:
        gap = _spatial.cKDTree(coordinates).query(nodes[outside])[0]
        keep[outside[gap <= distance]] = True
    return _as_points(nodes[keep], getattr(data, "coordinate_labels", None))

def combine(*sources: "_data._SpatialData | _types.ArrayLike",
            tolerance: float = 0.0) -> "_data.PointData":
    """
    One inducing point set out of several, dropping duplicates.

    Useful for the usual mixture of a regular backbone and the data's own
    locations, `combine(from_grid(data, 50), from_kmeans(data, 200))`.

    Parameters
    ----------
    sources
        Spatial containers or coordinate arrays, all of the same
        dimension.
    tolerance
        Points closer than this to one already kept are dropped. The
        default of zero removes only exact repeats.

    Returns
    -------
    geoml.data.PointData
    """
    if len(sources) == 0:
        raise ValueError("combine needs at least one set of points")

    arrays = [_coordinates(s) for s in sources]
    dims = {a.shape[1] for a in arrays}
    if len(dims) != 1:
        raise ValueError(
            "all sources must have the same dimension, found %s"
            % ", ".join(str(d) for d in sorted(dims)))

    merged = _np.concatenate(arrays, axis=0)
    labels = getattr(sources[0], "coordinate_labels", None)
    if tolerance <= 0:
        _, keep = _np.unique(merged, axis=0, return_index=True)
        return _as_points(merged[_np.sort(keep)], labels)

    # Greedy in input order: a point is kept unless one kept before it lies
    # closer than the tolerance. Snapping to a grid of the tolerance, which
    # this replaced, kept two close points either side of a cell boundary
    # and merged two up to tolerance * sqrt(d) apart inside one cell.
    tree = _spatial.cKDTree(merged)
    dropped = _np.zeros(len(merged), dtype=bool)
    for i, near in enumerate(tree.query_ball_point(merged, tolerance)):
        if dropped[i]:
            continue
        near = _np.asarray(near, dtype=int)
        near = near[near > i]
        close = _np.linalg.norm(merged[near] - merged[i], axis=1) < tolerance
        dropped[near[close]] = True
    return _as_points(merged[~dropped], labels)


def grid_experts(data: "_data._SpatialData | _types.ArrayLike",
                 step: "float | _types.ArrayLike",
                 block: int = 4) -> "list[_data.PointData]":
    """
    Experts laid out as overlapping blocks of a regular lattice.

    The space is cut into blocks of `block` inducing points per side, and each
    expert takes its own block plus one node of margin all around, so
    neighbouring experts overlap by one step. Two things follow from that
    layout, and both matter to the model:

    - every expert holds exactly ``(block + 2) ** n_dim`` inducing points, so
      the per-expert state is rectangular;
    - an expert's neighbours are known from the block indices rather than
      measured -- the Moore neighbourhood, 8 in the plane and 26 in space.

    Parameters
    ----------
    data
        A spatial container, or an `(n_data, n_dim)` array of coordinates.
    step
        Spacing between neighbouring inducing points.
    block
        Inducing points per block side, before the margin is added.

    Returns
    -------
    list of geoml.data.PointData
        One set per expert, ordered with the first axis varying slowest.
    """
    coordinates = _coordinates(data)
    n_dim = coordinates.shape[1]
    step = _step_vector(step, n_dim)
    block = int(block)
    if block < 1:
        raise ValueError("block must be at least 1, got %d" % block)

    extent = coordinates.max(axis=0) - coordinates.min(axis=0)
    n_blocks = _np.maximum(
        1, _np.ceil(extent / (step * block)).astype(int))

    # one node of margin at each end, so the outermost blocks have the same
    # surroundings as the inner ones and every expert comes out the same size
    axes = _lattice_axes(coordinates, step, n_blocks * block + 2)
    labels = getattr(data, "coordinate_labels", None)

    sets = []
    for corner in _np.ndindex(*n_blocks):
        block_axes = [axes[d][c * block:c * block + block + 2]
                      for d, c in enumerate(corner)]
        sets.append(_as_points(_lattice(block_axes), labels))
    return sets


def _bounded_assignment(cost, low, high, candidates=None):
    """
    Each point's cluster, minimizing the summed `cost` with every cluster
    holding between `low` and `high` points, or None where no assignment
    fits.

    A transport problem: every point sends one unit, every cluster takes
    between its bounds. Its constraint matrix is totally unimodular, so the
    simplex returns a vertex whose entries are whole -- one cluster per
    point -- and the largest entry of each row names it. With `candidates`,
    a point may go only to that many of its nearest clusters: the far ones
    never take it at the optimum, and leaving them out divides the problem
    by the number of clusters over the candidates.
    """
    n_points, n_clusters = cost.shape
    if candidates is None or candidates >= n_clusters:
        options = _np.broadcast_to(_np.arange(n_clusters),
                                   (n_points, n_clusters))
    else:
        options = _np.argpartition(cost, candidates - 1, axis=1)[
            :, :candidates]
    width = options.shape[1]
    columns = _np.arange(n_points * width)
    ones = _np.ones(n_points * width)
    each = _sparse.csr_matrix(
        (ones, (_np.repeat(_np.arange(n_points), width), columns)),
        shape=(n_points, n_points * width))
    taken = _sparse.csr_matrix(
        (ones, (options.ravel(), columns)),
        shape=(n_clusters, n_points * width))
    result = _optimize.linprog(
        _np.take_along_axis(cost, options, axis=1).ravel(),
        A_ub=_sparse.vstack([taken, -taken]),
        b_ub=_np.concatenate([_np.full(n_clusters, high),
                              _np.full(n_clusters, -low)]),
        A_eq=each, b_eq=_np.ones(n_points), bounds=(0, 1),
        method="highs-ds")
    if result.status != 0:
        return None
    chosen = _np.argmax(result.x.reshape(n_points, width), axis=1)
    return options[_np.arange(n_points), chosen]


def _balanced_labels(coordinates, n_clusters, balance, seed, rounds=30):
    """
    Compact clusters whose sizes stay within `balance` of the mean.

    k-means with its assignment step bounded (Bradley, Bennett & Demiriz
    2000): every round assigns the points to the centres at the least
    summed squared distance that keeps each cluster between
    `(1 - balance)` and `(1 + balance)` times `n / n_clusters` points, then
    moves each centre to its points' mean, until nothing moves. Solved
    whole, the assignment lets clusters trade points; placing them one at a
    time, as an earlier version did, filled the nearby clusters first and
    sent what came last to whichever cluster still had room, however far.
    """
    n_points = coordinates.shape[0]
    mean = n_points / n_clusters
    low = max(1, int(_np.floor(mean * (1 - balance))))
    high = int(_np.ceil(mean * (1 + balance)))
    centres = _KMeans(n_clusters, n_init=10, random_state=seed).fit(
        coordinates).cluster_centers_
    labels = None
    for _ in range(rounds):
        cost = ((coordinates[:, None, :] - centres[None]) ** 2).sum(-1)
        assigned = _bounded_assignment(cost, low, high, candidates=8)
        if assigned is None:
            # with every cluster a candidate there is always a solution, the
            # bounds holding n between low * k and high * k
            assigned = _cast(_np.ndarray,
                             _bounded_assignment(cost, low, high))
        if labels is not None and _np.array_equal(assigned, labels):
            break
        labels = assigned
        centres = _np.stack([coordinates[labels == j].mean(axis=0)
                             for j in range(n_clusters)])
    return labels


def experts(points: "_data._SpatialData | _types.ArrayLike",
            n_experts: int, overlap: float = 0.1,
            seed: int | None = None,
            balance: float = 0.1) -> "list[_data.PointData]":
    """
    Experts from overlapping clusters of an unstructured point set.

    The unordered counterpart to `grid_experts`, for inducing points that
    follow the data rather than a lattice. The points are split into compact
    clusters of about the same size -- k-means whose assignment keeps every
    cluster within `balance` of the mean size, solved for all the points at
    once, so that clusters trade points rather than fill up -- and each
    cluster then borrows up to `overlap` of its own count, rounded up, from
    its neighbours evenly: one point from each neighbour a round, that
    neighbour's nearest to any of its own members, so a cluster with many
    neighbours spreads its overlap over all of them. A borrowed point keeps
    its own cluster too, so neighbouring experts come to share the points
    between them — which is what stops a prediction showing a seam where one
    expert gives way to the next, and is the irregular equivalent of the one
    step of margin `grid_experts` adds to each block. An expert reaches every
    neighbour once its overlap is at least its number of neighbours, which
    asks for experts that are not too small.

    Counting the overlap in points rather than in distance is what keeps the
    experts the same size. Growing each cluster by a radius instead lets a
    cluster in a crowded part of the survey swallow far more than one out on
    its own, and the experts come out wildly uneven.

    Since this divides inducing points rather than data, the usual call is
    ``experts(from_kmeans(data, 1500), 12)``.

    Parameters
    ----------
    points
        The inducing points to divide: a spatial container, or an
        `(n_points, n_dim)` array.
    n_experts
        Number of experts. Must not exceed the number of points.
    overlap
        How many points each expert borrows from its neighbours, as a fraction
        of its own count, rounded up, so an expert ends up with about
        `1 + overlap` times the points its cluster holds. Zero leaves the
        experts a strict partition, sharing nothing.
    seed
        Passed to `sklearn.cluster.KMeans` for a reproducible result.
    balance
        How far a cluster's size may stray from `n_points / n_experts`, as a
        fraction of it: the room the clusters have to trade points for
        compactness. Zero makes them all the same size, within one point.

    Returns
    -------
    list of geoml.data.PointData
        One set per expert: its own cluster, plus what it borrowed.
    """
    coordinates = _coordinates(points)
    n_points = coordinates.shape[0]
    n_experts = int(n_experts)
    if not 1 <= n_experts <= n_points:
        raise ValueError(
            "n_experts must be between 1 and the number of points (%d), "
            "got %d" % (n_points, n_experts))
    if overlap < 0:
        raise ValueError("overlap must not be negative, got %r" % overlap)
    if not 0 <= balance < 1:
        raise ValueError("balance must be in [0, 1), got %r" % balance)

    labels = _balanced_labels(coordinates, n_experts, balance, seed)
    coordinate_labels = getattr(points, "coordinate_labels", None)
    neighbours = _neighbours(coordinates, labels, n_experts)

    sets = []
    for j in range(n_experts):
        core = labels == j
        keep = core.copy()
        budget = int(_np.ceil(overlap * core.sum()))
        if budget > 0 and neighbours[j]:
            keep[_borrowed(coordinates, labels, j, neighbours[j], budget)]                 = True
        sets.append(_as_points(coordinates[keep], coordinate_labels))
    return sets


def _neighbours(coordinates, labels, n_experts):
    """Each cluster's neighbours: the clusters some member of it has as
    its nearest point outside it, or that have it so. Read off the points
    themselves, it needs no distance to tune and adapts to how far apart
    the points of a survey are."""
    faced = [set() for _ in range(n_experts)]
    for j in range(n_experts):
        core = labels == j
        if core.all():
            continue
        outside = _np.flatnonzero(~core)
        nearest = _spatial.cKDTree(coordinates[outside]).query(
            coordinates[core])[1]
        for other in _np.unique(labels[outside[nearest]]):
            faced[j].add(int(other))
            faced[int(other)].add(j)
    return [sorted(f) for f in faced]


def _borrowed(coordinates, labels, j, neighbours, budget):
    """The points cluster `j` borrows: from each neighbour in turn, nearest
    first, the neighbour's point nearest any of `j`'s members, one each a
    round, until `budget` points or the neighbours run out. Spread over the
    neighbours, the overlap reaches every side of the cluster -- taken by
    nearness alone it came from the one or two nearest, and most touching
    experts shared nothing."""
    tree = _spatial.cKDTree(coordinates[labels == j])
    queues = []
    for other in neighbours:
        members = _np.flatnonzero(labels == other)
        distance = tree.query(coordinates[members])[0]
        order = _np.argsort(distance)
        queues.append((distance[order[0]], members[order]))
    queues.sort(key=lambda queue: queue[0])
    taken = []
    depth = 0
    while len(taken) < budget:
        before = len(taken)
        for _, members in queues:
            if depth < members.size and len(taken) < budget:
                taken.append(members[depth])
        if len(taken) == before:
            break
        depth += 1
    return _np.asarray(taken, dtype=int)
