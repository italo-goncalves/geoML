"""Where the data can speak: the concave hull, and the reach it defines.

A swath plot compares a data mean with a model mean slab by slab, and the
model side is only fair over ground the data informs. The hull here is an
alpha shape at a length: it fills the interior between drill fences closer
than that length -- where a ball around each sample would leave a gap --
and leaves out a notch wider than it, where a convex hull would bridge
across. The two cases below are the definition.
"""
import numpy as np
import pytest

import geoml.math.geometry as geometry


def _c_shape(spacing=1.0):
    """Points on a thick C: an annulus with its right quarter cut out."""
    rng = np.random.default_rng(0)
    angle = rng.uniform(0.25 * np.pi, 1.75 * np.pi, 600)
    radius = rng.uniform(6.0, 10.0, 600)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def test_the_hull_leaves_a_notch_out_and_the_convex_hull_would_not():
    points = _c_shape()
    hull = geometry.concave_hull(points, length=2.0)

    # the C's opening -- inside the convex hull (the lips of the C reach
    # x = 10 cos 45 = 7.07, so a point at x = 6 on the axis is behind the
    # chord between them), outside the concave one
    notch = np.array([[6.0, 0.0], [5.0, 0.5]])
    assert not hull.contains(notch).any()
    # the C's own arm is inside
    assert hull.contains(np.array([[-8.0, 0.0], [0.0, 8.0]])).all()
    # and the convex hull is the counter-example
    from scipy.spatial import Delaunay
    assert (Delaunay(points).find_simplex(notch) >= 0).all()


def test_the_hull_fills_between_fences_closer_than_its_length():
    """Two drill fences ten units apart. A hull at fifteen bridges them --
    the ground between the fences is inside -- and a hull at five does
    not, which is the length being a statement about the data."""
    along = np.arange(0.0, 20.0, 1.0)
    fences = np.concatenate([
        np.stack([along, np.zeros_like(along)], axis=1),
        np.stack([along, np.full_like(along, 10.0)], axis=1)])
    fences = fences + np.random.default_rng(1).normal(scale=0.05,
                                                     size=fences.shape)
    between = np.array([[10.0, 5.0], [4.0, 4.0]])

    assert geometry.concave_hull(fences, length=15.0).contains(between).all()
    assert not geometry.concave_hull(fences, length=5.0).contains(between).any()


def test_a_three_dimensional_hull_holds_its_interior():
    rng = np.random.default_rng(2)
    cloud = rng.uniform(0.0, 10.0, size=[400, 3])
    hull = geometry.concave_hull(cloud, length=4.0)

    assert hull.contains(np.array([[5.0, 5.0, 5.0]]))[0]
    assert not hull.contains(np.array([[30.0, 30.0, 30.0]]))[0]
    # boundary facets are triangles whose vertices are points of the cloud
    assert hull.boundary.shape[1] == 3
    assert hull.boundary.min() >= 0 and hull.boundary.max() < len(cloud)


def test_points_that_do_not_span_the_space_are_refused():
    line = np.stack([np.arange(5.0), np.zeros(5)], axis=1)
    with pytest.raises(ValueError, match="span"):
        geometry.concave_hull(line, length=2.0)


# --------------------------------------------------------------------------- #
# the reach, as a container column
# --------------------------------------------------------------------------- #
def _fence_data():
    import pandas as pd
    import geoml
    along = np.arange(0.0, 20.0, 1.0)
    fences = np.concatenate([
        np.stack([along, np.zeros_like(along)], axis=1),
        np.stack([along, np.full_like(along, 10.0)], axis=1)])
    fences = fences + np.random.default_rng(3).normal(scale=0.05,
                                                     size=fences.shape)
    return geoml.data.PointData(pd.DataFrame(fences, columns=["X", "Y"]),
                                ["X", "Y"])


def _row(container, point):
    """The row of the location nearest a point."""
    coordinates = np.asarray(container.coordinates, dtype=float)
    return int(np.argmin(np.linalg.norm(coordinates - np.asarray(point),
                                        axis=1)))


def test_the_reach_is_a_boolean_column_by_distance_or_hull():
    import geoml
    data = _fence_data()
    grid = geoml.data.Grid2D(start=[-10, -10], end=[30, 20], n=[41, 31])
    between = _row(grid, [10.0, 5.0])
    far = _row(grid, [28.0, 18.0])
    on_fence = _row(grid, [10.0, 0.0])

    reach = grid.assign_from_data(data, distance=1.5)
    assert reach.dtype == bool
    assert np.array_equal(grid.get_metadata("near_data"), reach)
    assert reach[on_fence] and not reach[between] and not reach[far]

    hull = grid.assign_from_data(data, hull=15.0, name="hull")
    assert hull[between] and hull[on_fence] and not hull[far]

    with pytest.raises(ValueError, match="distance"):
        grid.assign_from_data(data)


def test_the_transform_carries_the_anisotropy():
    """Under `Isotropic(r)` the distance is read in units of `r`; under an
    ellipsoid the reach stretches along its long axis."""
    import geoml
    data = _fence_data()
    grid = geoml.data.Grid2D(start=[-10, -10], end=[30, 20], n=[41, 31])
    between = _row(grid, [10.0, 5.0])

    plain = grid.assign_from_data(data, distance=5.0)
    scaled = grid.assign_from_data(
        data, distance=1.0, transform=geoml.transform.Isotropic(5.0),
        name="scaled")
    assert np.array_equal(plain, scaled)

    # a range of 20 along y (azimuth 0 is north) and 2 along x: the reach
    # crosses the gap between the fences and rises well above the top
    # one, but barely extends past a fence's end
    stretched = grid.assign_from_data(
        data, distance=1.0, name="stretched",
        transform=geoml.transform.Anisotropy2D(maxrange=20.0,
                                               minrange_fct=0.1,
                                               azimuth=0.0))
    above = _row(grid, [10.0, 17.0])
    past_the_end = _row(grid, [23.0, 0.0])  # four past the last sample
    assert stretched[between] and stretched[above]
    assert not stretched[past_the_end]
    assert plain[past_the_end] and not plain[above]


def test_a_block_model_keeps_the_reach_through_a_split():
    import pandas as pd
    import geoml
    rng = np.random.default_rng(4)
    cloud = rng.uniform(20.0, 60.0, size=[80, 3])
    data = geoml.data.PointData(pd.DataFrame(cloud, columns=["X", "Y", "Z"]),
                                ["X", "Y", "Z"])
    blocks = geoml.data.BlockSet3D(start=[0, 0, 0], n=[8, 8, 8],
                                   step=[10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2), max_levels=2)
    reach = blocks.assign_from_data(data, hull=15.0, distance=5.0)
    assert 0 < reach.sum() < blocks.n_data

    finer = blocks.split(reach)
    inherited = finer.get_metadata("near_data")
    assert inherited.dtype == bool
    # every child of a split block was in reach, and the untouched blocks
    # keep what they held
    assert inherited.sum() == reach.sum() * 8 + 0

