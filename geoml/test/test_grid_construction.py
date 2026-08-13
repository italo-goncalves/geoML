"""Building grids and block models around data, and aggregating onto them.

`from_data` covers another object's bounding box with a margin, flooring the
box corner to a chosen number of decimals -- round numbers read better on a
section -- and growing the count so the rounding never eats the margin. The
rotated classes fit their angles to the data and round those too.

`aggregate` is one method instead of one per kind: each variable says what it
is and the operation follows -- continuous values average, categories keep
the dominant label, what is truly ambiguous comes back empty.
"""
import numpy as np
import pandas as pd
import pytest

import geoml


def _points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([2.3, -7.9, 11.1], [48.7, 33.3, 29.9], size=(n, 3))
    point = geoml.data.PointData.from_array(xyz, ["E", "N", "RL"])
    return point, xyz


# --------------------------------------------------------------------------- #
# from_data
# --------------------------------------------------------------------------- #
def test_the_grid_covers_the_data_with_its_margin():
    point, xyz = _points()
    grid = geoml.data.Grid3D.from_data(point, step=5.0, margin=0.1)

    low, high = xyz.min(axis=0), xyz.max(axis=0)
    extent = high - low
    assert np.all(grid.bounding_box.min[0] <= low - 0.1 * extent)
    assert np.all(grid.bounding_box.max[0] >= high + 0.1 * extent)
    assert grid.coordinate_labels == ["E", "N", "RL"]


def test_the_box_corner_is_floored_to_the_decimals():
    point, _ = _points()
    grid = geoml.data.Grid3D.from_data(point, step=5.0, margin=0.0)
    assert np.allclose(grid.origin, np.round(grid.origin, 0))

    fine = geoml.data.Grid3D.from_data(point, step=5.0, margin=0.0,
                                       decimals=1)
    assert np.allclose(fine.origin, np.round(fine.origin, 1))
    # flooring, not rounding: neither start moves up into the data, and the
    # coarser flooring reaches at most as high as the finer one
    _, xyz = _points()
    assert np.all(grid.origin <= xyz.min(axis=0))
    assert np.all(grid.origin <= fine.origin + 1e-12)


def test_each_dimension_builds_its_own_kind():
    point, xyz = _points()
    flat = geoml.data.PointData.from_array(xyz[:, :2], ["E", "N"])
    line = geoml.data.PointData.from_array(xyz[:, :1], ["E"])

    assert geoml.data.Grid1D.from_data(line, step=2.0).n_dim == 1
    assert geoml.data.Grid2D.from_data(flat, step=2.0).n_dim == 2
    assert geoml.data.GridND.from_data(flat, step=2.0).n_dim == 2

    with pytest.raises(geoml.data.DimensionMismatchError):
        geoml.data.Grid2D.from_data(point, step=2.0)


def test_blocks_inherit_it_and_the_block_set_counts_cells():
    point, xyz = _points()
    blocks = geoml.data.Blocks3D.from_data(point, step=[10.0, 10.0, 5.0])
    assert isinstance(blocks, geoml.data.Blocks3D)

    bs = geoml.data.BlockSet3D.from_data(
        point, step=[10.0, 10.0, 5.0], margin=0.1,
        discretization=(2, 2, 1), max_levels=1)
    assert bs.is_full()
    assert bs.discretization == [2, 2, 1]
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    extent = high - low
    assert np.all(bs.bounding_box.min[0] <= low - 0.1 * extent)
    assert np.all(bs.bounding_box.max[0] >= high + 0.1 * extent)
    # the corner, not the first centre, is the round number
    assert np.allclose(bs.box_corner, np.round(bs.box_corner, 0))


def test_drillholes_serve_as_data():
    collar = pd.DataFrame({
        "HoleID": ["H1", "H2"], "X": [0.0, 30.0], "Y": [0.0, 10.0],
        "Z": [100.0, 105.0], "Length": [50.0, 60.0],
        "Dip": [90.0, 90.0], "Azimuth": [0.0, 0.0]})
    holes = geoml.drillhole.DrillholeData(
        collar, hole="HoleID", x="X", y="Y", z="Z", length="Length",
        dip="Dip", azimuth="Azimuth")

    grid = geoml.data.Grid3D.from_data(holes, step=10.0, margin=0.0)
    assert np.all(grid.bounding_box.min[0] <= [0.0, 0.0, 45.0])
    assert np.all(grid.bounding_box.max[0] >= [30.0, 10.0, 105.0])

    rotated = geoml.data.RotatedGrid3D.from_data(holes, step=10.0)
    assert isinstance(rotated, geoml.data.RotatedGrid3D)


def test_the_fitted_angles_come_out_rounded():
    rng = np.random.default_rng(1)
    plane = rng.uniform(0, 100, size=(300, 3)) * [1.0, 0.3, 0.05]
    mat = geoml.geometry.rotation_matrix(37.3182, 12.6541, 0.0)
    tilted = geoml.data.PointData.from_array(plane @ mat)

    grid = geoml.data.RotatedGrid3D.from_data(tilted, step=5.0, decimals=0)
    assert grid.azimuth == np.round(grid.azimuth, 0)
    assert grid.dip == np.round(grid.dip, 0)
    assert grid.rake == np.round(grid.rake, 0)
    assert np.allclose(grid.origin, np.round(grid.origin, 0))

    finer = geoml.data.RotatedGrid3D.from_data(tilted, step=5.0, decimals=1)
    assert finer.azimuth == np.round(finer.azimuth, 1)

    # rounded or not, the data stays covered
    coords = np.asarray(tilted.coordinates)
    assert np.all(grid.bounding_box.min[0] <= coords.min(axis=0) + 1e-9)
    assert np.all(grid.bounding_box.max[0] >= coords.max(axis=0) - 1e-9)


# --------------------------------------------------------------------------- #
# the rotated block set
# --------------------------------------------------------------------------- #
def _rotated_blocks(**kwargs):
    kwargs.setdefault("azimuth", 30.0)
    kwargs.setdefault("dip", 10.0)
    return geoml.data.RotatedBlockSet3D(
        [10.0, 20.0, 30.0], [4, 3, 2], [10.0, 10.0, 5.0],
        discretization=(2, 2, 2), max_levels=2, **kwargs)


def test_unrotated_it_is_the_block_set_it_extends():
    plain = geoml.data.BlockSet3D([10.0, 20.0, 30.0], [4, 3, 2],
                                  [10.0, 10.0, 5.0],
                                  discretization=(2, 2, 2), max_levels=2)
    zero = _rotated_blocks(azimuth=0.0, dip=0.0)
    assert np.allclose(np.asarray(zero.coordinates),
                       np.asarray(plain.coordinates))
    coords, _ = zero.get_batched_coordinates()
    plain_coords, _ = plain.get_batched_coordinates()
    assert np.allclose(coords, plain_coords)


def test_every_block_holds_its_own_centre():
    blocks = _rotated_blocks().split([0, 5, 7])
    centres = geoml.data.PointData.from_array(
        np.asarray(blocks.coordinates))
    assert np.array_equal(blocks.index_data(centres),
                          np.arange(blocks.n_data))


def test_the_fan_out_turns_with_the_blocks():
    blocks = _rotated_blocks()
    coords, splits = blocks.get_batched_coordinates()
    per_block = coords.reshape(blocks.n_data, -1, 3)

    # the sub-blocks average back to their block's centre
    assert np.allclose(per_block.mean(axis=1),
                       np.asarray(blocks.coordinates))
    # and the spread is rigid: rotation changes no distances
    plain = geoml.data.BlockSet3D([10.0, 20.0, 30.0], [4, 3, 2],
                                  [10.0, 10.0, 5.0],
                                  discretization=(2, 2, 2), max_levels=2)
    flat, _ = plain.get_batched_coordinates()
    flat = flat.reshape(plain.n_data, -1, 3)
    d_rot = np.linalg.norm(per_block[:, :, None] - per_block[:, None], axis=3)
    d_flat = np.linalg.norm(flat[:, :, None] - flat[:, None], axis=3)
    assert np.allclose(d_rot, d_flat)


def test_a_split_keeps_the_rotation():
    blocks = _rotated_blocks()
    cut = blocks.split([0, 1])

    assert isinstance(cut, geoml.data.RotatedBlockSet3D)
    assert cut.azimuth == blocks.azimuth
    centres = geoml.data.PointData.from_array(np.asarray(cut.coordinates))
    assert np.array_equal(cut.index_data(centres), np.arange(cut.n_data))


def test_the_exported_hexahedra_keep_their_volume():
    blocks = _rotated_blocks().split([0])
    mesh = blocks.as_pyvista()

    sizes = mesh.compute_cell_sizes(volume=True)
    assert np.allclose(np.sort(sizes.cell_data["Volume"]),
                       np.sort(blocks.block_volume))
    # cell centres land on the (rotated) block centres
    assert np.allclose(
        np.sort(mesh.cell_centers().points, axis=0),
        np.sort(np.asarray(blocks.coordinates), axis=0), atol=1e-8)


def test_it_survives_a_zarr_round_trip(tmp_path):
    blocks = _rotated_blocks().split([0, 3])
    blocks.add_continuous_variable("g", np.arange(float(blocks.n_data)))

    path = str(tmp_path / "rot.zarr")
    blocks.to_zarr(path)
    back = geoml.data.BlockSet3D.open(path)

    assert isinstance(back, geoml.data.RotatedBlockSet3D)
    assert back.azimuth == blocks.azimuth
    assert np.allclose(np.asarray(back.coordinates),
                       np.asarray(blocks.coordinates))
    assert np.allclose(back.values("g/measurements"),
                       np.arange(float(blocks.n_data)))


def test_from_data_fits_and_covers():
    rng = np.random.default_rng(2)
    cloud = rng.uniform(0, 60, size=(300, 3)) * [1.0, 0.4, 0.1]
    tilted = geoml.data.PointData.from_array(
        cloud @ geoml.geometry.rotation_matrix(25.0, 5.0, 0.0))

    blocks = geoml.data.RotatedBlockSet3D.from_data(
        tilted, step=[10.0, 10.0, 2.0], margin=0.05,
        discretization=(2, 2, 2), max_levels=1)

    assert blocks.is_full()
    assert blocks.azimuth == np.round(blocks.azimuth, 0)
    coords = np.asarray(tilted.coordinates)
    assert np.all(blocks.bounding_box.min[0] <= coords.min(axis=0) + 1e-9)
    assert np.all(blocks.bounding_box.max[0] >= coords.max(axis=0) - 1e-9)
    # and the fit is real: every point lands in some block
    assert np.all(blocks.index_data(tilted) >= 0)


# --------------------------------------------------------------------------- #
# one aggregate
# --------------------------------------------------------------------------- #
def _measured_points():
    """Two cells' worth of data with every variable kind on it."""
    xyz = np.array([[1.0, 1.0, 1.0], [1.2, 0.8, 1.1], [1.1, 1.2, 0.9],
                    [11.0, 1.0, 1.0], [11.2, 0.8, 1.1]])
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable("g", np.array([1.0, 2.0, 3.0, 10.0, 20.0]))
    point.add_categorical_variable(
        "rock", measurements=np.array(
            ["ore", "ore", "waste", "ore", "waste"]))
    point.add_vector_variable("vec", ["p", "q"],
                              np.arange(10.0).reshape(5, 2))
    point.add_compositional_variable(
        "assay", ["a", "b"],
        np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.5, 0.5],
                  [0.7, 0.3]]))
    point.add_metadata("weight", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    point.add_metadata("source", np.array(["lab", "lab", "field",
                                           "lab", "field"]))
    return point


def _two_cells():
    return geoml.data.Grid3D(start=[1.0, 1.0, 1.0], n=[2, 1, 1],
                             step=[10.0, 10.0, 10.0])


def test_each_kind_aggregates_its_own_way():
    grid = _two_cells().aggregate(_measured_points())

    assert np.allclose(grid.values("g/measurements"), [2.0, 15.0])
    # dominant in the first cell; a 1-1 tie in the second is no answer
    assert list(grid.values("rock/measurements_a")) == ["ore", ""]
    assert np.allclose(grid.values("vec/p/measurements"), [2.0, 7.0])
    # a composition is averaged and closed again
    parts = np.stack([grid.values("assay/a/measurements"),
                      grid.values("assay/b/measurements")], axis=1)
    assert np.allclose(parts.sum(axis=1), 1.0)
    assert np.allclose(parts[0], [0.4, 0.6])
    # metadata follows the same rule: numeric averages, coded keeps the
    # dominant label -- and a tie is empty there too
    assert np.allclose(grid.get_metadata("weight"), [2.0, 4.5])
    assert list(grid.get_metadata("source")) == ["lab", ""]


def test_a_cell_nothing_fell_in_is_empty():
    grid = geoml.data.Grid3D(start=[1.0, 1.0, 1.0], n=[3, 1, 1],
                             step=[10.0, 10.0, 10.0])
    grid.aggregate(_measured_points(), variables=["g", "rock"])

    assert np.isnan(grid.values("g/measurements")[2])
    assert grid.values("rock/measurements_a")[2] == ""


def test_the_variables_can_be_named_and_a_wrong_name_says_so():
    grid = _two_cells()
    grid.aggregate(_measured_points(), variables="g", metadata=False)

    assert list(grid.variables) == ["g"]
    assert len(grid.metadata) == 0
    with pytest.raises(ValueError, match="no variable named 'nope'"):
        grid.aggregate(_measured_points(), variables="nope")


def test_a_rotated_grid_aggregates_through_its_rotation():
    """`index_data` used to apply the forward map instead of the inverse;
    nothing noticed while everything downstream of it raised."""
    grid = geoml.data.RotatedGrid3D(
        start=[10.0, 20.0, 30.0], n=[4, 3, 2], step=[10.0, 10.0, 5.0],
        azimuth=30.0, dip=10.0)
    centres = np.asarray(grid.coordinates)
    point = geoml.data.PointData.from_array(centres)
    point.add_continuous_variable("g", np.arange(float(grid.n_data)))

    grid.aggregate(point)
    assert np.allclose(grid.values("g/measurements"),
                       np.arange(float(grid.n_data)))


def test_blocks_of_several_sizes_aggregate_the_same_way():
    blocks = geoml.data.BlockSet3D([5.0, 5.0, 5.0], [2, 2, 2],
                                   [10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2),
                                   max_levels=1).split([0])
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0.0, 20.0, size=(300, 3))
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable("g", xyz[:, 0])

    blocks.aggregate(point)
    held = blocks.index_data(point)
    got = blocks.values("g/measurements")
    for b in np.unique(held):
        assert np.isclose(got[b], xyz[held == b, 0].mean())


def test_a_model_predicts_onto_the_rotated_blocks():
    """The whole prediction path -- fan-out, batching, aggregation back to
    the block -- runs through `get_batched_coordinates`, which is the one
    thing the rotated class changes about it."""
    import tensorflow as tf
    geoml.set_seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    rng = np.random.default_rng(42)
    xyz = rng.uniform(0.0, 60.0, size=(40, 3))
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable("v", np.sin(xyz[:, 0] / 15.0))

    inducing = geoml.data.PointData.from_array(
        rng.uniform(0.0, 60.0, size=(6, 3)))
    network = geoml.latent.BasicGP(
        geoml.latent.BasicInput(inducing,
                                transform=geoml.transform.Isotropic(30.0)),
        size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), network,
        options=geoml.models.GPOptions(verbose=False, training_samples=4))
    model.train_full(max_iter=2)

    blocks = geoml.data.RotatedBlockSet3D.from_data(
        point, step=[20.0, 20.0, 20.0], margin=0.0,
        discretization=(2, 2, 2), max_levels=1)
    model.predict(blocks, n_sim=4)

    prediction = blocks.values("v/prediction")
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(blocks.values("v/dispersion")))
