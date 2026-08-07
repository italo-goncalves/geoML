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
    combine(a, b, ...)            one set out of several, duplicates dropped
    grid_experts(data, step)      a list of sets, laid out as overlapping blocks
    experts(points, n_experts)    a list of sets, from overlapping clusters

The two `*_experts` functions divide a set of inducing points among experts so
that neighbouring ones overlap, which is what keeps a prediction from showing a
seam where the experts meet. They differ only in how the division is made.
`grid_experts` cuts space into regular blocks and extends each by one step, so
every expert is the same size and its neighbours are known in advance -- the
Moore neighbourhood, 8 in the plane and 26 in space. `experts` is the unordered
counterpart: it clusters whatever points it is given and grows each cluster by
a fraction of its own Mahalanobis radius, which suits a survey that does not
fill its bounding box, such as drillholes or a shoreline.

`experts` takes inducing points rather than data, so the usual way to build an
irregular network is to choose the points first and then divide them:

    sets = experts(from_kmeans(data, 1500), 12)
"""

__all__ = ["from_kmeans", "from_grid", "combine", "grid_experts", "experts"]

import numpy as _np
from sklearn.cluster import KMeans as _KMeans

import geoml.data as _data


def _coordinates(source):
    """The coordinates of a container, or an array taken as they are."""
    if hasattr(source, "coordinates"):
        return _np.asarray(source.coordinates, dtype=float)
    return _np.array(source, ndmin=2, dtype=float)


def _as_points(coordinates, labels=None):
    return _data.PointData.from_array(
        _np.ascontiguousarray(coordinates, dtype=float), labels)


def from_kmeans(data, n, seed=None):
    """
    Inducing points at the k-means centroids of the data.

    Parameters
    ----------
    data
        A spatial container, or an (n_data, n_dim) array of coordinates.
    n : int
        Number of inducing points. Must not exceed the number of data points.
    seed : int
        Passed to `sklearn.cluster.KMeans` for a reproducible result. Note that
        this is separate from `geoml.set_seed`, which governs the model's
        parameter initialization.

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


def from_grid(data, step):
    """
    Inducing points on a regular lattice covering the data.

    Parameters
    ----------
    data
        A spatial container, or an (n_data, n_dim) array of coordinates.
    step : float or array
        Spacing between neighbouring inducing points, one value per dimension
        or a single value for all of them.

    Returns
    -------
    geoml.data.PointData
        The lattice nodes, the first axis varying slowest.
    """
    coordinates = _coordinates(data)
    step = _step_vector(step, coordinates.shape[1])
    nodes = _lattice(_lattice_axes(coordinates, step))
    return _as_points(nodes, getattr(data, "coordinate_labels", None))


def combine(*sources, tolerance=0.0):
    """
    One inducing point set out of several, dropping duplicates.

    Useful for the usual mixture of a regular backbone and the data's own
    locations, `combine(from_grid(data, 50), from_kmeans(data, 200))`.

    Parameters
    ----------
    sources
        Spatial containers or coordinate arrays, all of the same dimension.
    tolerance : float
        Points closer than this to one already kept are dropped. The default
        of zero removes only exact repeats.

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
    if tolerance > 0:
        # snapping to a grid of the tolerance turns "near duplicates" into
        # exact ones, which `unique` can then remove in one pass
        keys = _np.round(merged / tolerance)
    else:
        keys = merged
    _, keep = _np.unique(keys, axis=0, return_index=True)
    labels = getattr(sources[0], "coordinate_labels", None)
    return _as_points(merged[_np.sort(keep)], labels)


def grid_experts(data, step, block=4):
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
        A spatial container, or an (n_data, n_dim) array of coordinates.
    step : float or array
        Spacing between neighbouring inducing points.
    block : int
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


def _balanced_labels(coordinates, n_clusters, seed):
    """
    Cluster assignment with every cluster capped at the same size.

    Plain k-means leaves clusters of wildly different sizes, and an expert
    cannot draw more inducing points than its cluster holds. Points are
    offered their preferred cluster in order of how much the choice costs
    them -- the ones with the least to gain elsewhere go first -- and a
    cluster stops accepting once it is full.
    """
    n_data = coordinates.shape[0]
    centers = _KMeans(n_clusters, n_init=10, random_state=seed).fit(
        coordinates).cluster_centers_

    distance = ((coordinates[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    capacity = int(_np.ceil(n_data / n_clusters))

    order = _np.argsort(distance.min(axis=1) - distance.max(axis=1))
    preference = _np.argsort(distance, axis=1)
    labels = _np.full(n_data, -1, dtype=int)
    counts = _np.zeros(n_clusters, dtype=int)

    for i in order:
        for j in preference[i]:
            if counts[j] < capacity:
                labels[i] = j
                counts[j] += 1
                break
    return labels


def _cluster_covariance(members, n_dim, floor=1e-3):
    """
    A cluster's covariance, kept invertible.

    A cluster of samples taken along a single drillhole is very nearly a line,
    which leaves the covariance singular and the Mahalanobis distance across
    that line unbounded — the expert would reach over the whole survey. Raising
    the smallest eigenvalues to a fraction of the largest bounds the ellipsoid
    in every direction while leaving the shape alone where it is well
    determined.
    """
    if members.shape[0] > n_dim:
        covariance = _np.atleast_2d(_np.cov(members, rowvar=False))
    else:
        covariance = _np.eye(n_dim) * max(float(members.var()), 1.0)

    values, vectors = _np.linalg.eigh(covariance)
    if values.max() <= 0:  # every member in the same place
        return _np.eye(n_dim)
    values = _np.maximum(values, floor * values.max())
    return (vectors * values) @ vectors.T


def _mahalanobis(points, center, covariance):
    chol = _np.linalg.cholesky(covariance)
    solved = _np.linalg.solve(chol, (points - center).T)
    return _np.sqrt((solved ** 2).sum(axis=0))


def experts(points, n_experts, overlap=0.1, seed=None):
    """
    Experts from overlapping clusters of an unstructured point set.

    The unordered counterpart to `grid_experts`, for inducing points that
    follow the data rather than a lattice. The points are split into clusters
    of about the same size, and each cluster then borrows a further `overlap`
    of its own count from its surroundings: the points nearest its centre, in
    Mahalanobis distance, among those belonging to other clusters. A borrowed
    point keeps its own cluster too, so neighbouring experts come to share the
    points between them — which is what stops a prediction showing a seam where
    one expert gives way to the next, and is the irregular equivalent of the
    one step of margin `grid_experts` adds to each block.

    Counting the overlap in points rather than in distance is what keeps the
    experts the same size. Growing each cluster's Mahalanobis radius instead
    lets a cluster in a crowded part of the survey swallow far more than one
    out on its own, and the experts come out wildly uneven.

    Since this divides inducing points rather than data, the usual call is
    ``experts(from_kmeans(data, 1500), 12)``.

    Parameters
    ----------
    points
        The inducing points to divide: a spatial container, or an
        (n_points, n_dim) array.
    n_experts : int
        Number of experts. Must not exceed the number of points.
    overlap : float
        How many points each expert borrows from its neighbours, as a fraction
        of its own count, so an expert ends up with about `1 + overlap` times
        the points its cluster holds. Zero leaves the experts a strict
        partition, sharing nothing.
    seed : int
        Passed to `sklearn.cluster.KMeans` for a reproducible result.

    Returns
    -------
    list of geoml.data.PointData
        One set per expert: its own cluster, plus what it borrowed.
    """
    coordinates = _coordinates(points)
    n_points, n_dim = coordinates.shape
    n_experts = int(n_experts)
    if not 1 <= n_experts <= n_points:
        raise ValueError(
            "n_experts must be between 1 and the number of points (%d), "
            "got %d" % (n_points, n_experts))
    if overlap < 0:
        raise ValueError("overlap must not be negative, got %r" % overlap)

    labels = _balanced_labels(coordinates, n_experts, seed)
    coordinate_labels = getattr(points, "coordinate_labels", None)

    sets = []
    for j in range(n_experts):
        core = labels == j
        keep = core.copy()

        members = coordinates[core]
        outside = _np.flatnonzero(~core)
        borrowed = min(int(round(overlap * core.sum())), outside.size)

        if borrowed > 0:
            covariance = _cluster_covariance(members, n_dim)
            distance = _mahalanobis(coordinates, members.mean(axis=0),
                                    covariance)[outside]
            if borrowed < outside.size:
                outside = outside[_np.argpartition(distance, borrowed - 1)
                                  [:borrowed]]
            keep[outside] = True

        sets.append(_as_points(coordinates[keep], coordinate_labels))
    return sets
