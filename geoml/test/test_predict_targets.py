"""What `predict` writes into, and what it refuses.

A prediction into part of a container writes into the realizations the rest
of it keeps, so the number of them must agree -- `n_sim=None` takes the
stored one, and a different number is refused before any batch runs. The
part can be named by a stored filter, as `refine` names it. And anything of
the model's dimension is a target: a triangulated surface predicts at its
vertices.
"""
import numpy as np
import pytest

import geoml


def _walker_model():
    geoml.set_seed(1234)
    point, _ = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[5, 5], step=[55, 62])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "V", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False,
                                       prediction_batch_size=50))
    model.train_full(max_iter=3)
    return model


@pytest.fixture(scope="module")
def trained():
    return _walker_model()


def _grid():
    return geoml.data.Grid2D(start=[10, 10], n=[10, 10], step=[20, 20])


# --------------------------------------------------------------------------- #
# the number of realizations
# --------------------------------------------------------------------------- #
def test_a_fresh_target_gets_twenty_realizations(trained):
    grid = _grid()
    trained.predict(grid)
    assert grid.variables["V"].n_sim == 20


def test_none_takes_the_number_the_target_holds(trained):
    grid = _grid()
    trained.predict(grid, n_sim=3)
    half = np.arange(grid.n_data) < 50
    grid.variables["V"].simulations[~half, :] = np.nan
    trained.predict(grid, where=~half)
    assert grid.variables["V"].n_sim == 3
    assert not np.isnan(np.asarray(grid.variables["V"].simulations)).any()


@pytest.mark.parametrize("n_sim", [1, 5])
def test_a_different_number_under_where_is_refused(trained, n_sim):
    """With one realization the old code copied it into all three stored
    columns without a word; with five it failed part way through."""
    grid = _grid()
    trained.predict(grid, n_sim=3)
    before = np.asarray(grid.variables["V"].simulations).copy()
    with pytest.raises(ValueError, match="3 realization"):
        trained.predict(grid, n_sim=n_sim, where=np.arange(10))
    assert np.array_equal(before, np.asarray(grid.variables["V"].simulations))


def test_predicting_everything_again_may_change_the_number(trained):
    grid = _grid()
    trained.predict(grid, n_sim=3)
    trained.predict(grid, n_sim=5)
    assert grid.variables["V"].n_sim == 5


# --------------------------------------------------------------------------- #
# a stored filter
# --------------------------------------------------------------------------- #
def test_where_takes_a_metadata_column_by_name(trained):
    named, masked = _grid(), _grid()
    inside = np.asarray(named.coordinates[:, 0] < 100)
    named.add_metadata("inside", inside)
    trained.predict(named, n_sim=3, where="inside")
    trained.predict(masked, n_sim=3, where=inside)

    assert np.array_equal(named.unpredicted(), ~inside)
    for path in ("V/prediction", "V/latent_variance"):
        assert np.array_equal(np.asarray(named.get(path).values),
                              np.asarray(masked.get(path).values),
                              equal_nan=True)


# --------------------------------------------------------------------------- #
# a mesh as a target
# --------------------------------------------------------------------------- #
def test_a_surface_is_predicted_at_its_vertices_and_saved(tmp_path):
    geoml.set_seed(7)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 80.0, (60, 3))
    point = geoml.data.PointData.from_array(coords)
    point.add_continuous_variable("V", coords[:, 2] / 40.0 - 1.0)
    root = geoml.latent.BasicInput(
        geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3], step=[40, 40, 40]),
        transform=geoml.transform.Isotropic(40))
    model = geoml.models.VGPNetwork(
        point, "V", geoml.likelihood.Gaussian(),
        geoml.latent.BasicGP(root, size=1),
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=3)

    points = np.array([[10.0, 10, 40], [70, 10, 40], [70, 70, 40],
                       [10, 70, 40]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    sheet = geoml.data.Surface3D(
        points, triangles,
        geoml.math.geometry.vertex_normals(points, triangles))
    model.predict(sheet, n_sim=3)

    at_points = geoml.data.PointData.from_array(points)
    model.predict(at_points, n_sim=3)
    assert np.allclose(sheet.get("V/prediction").values,
                       at_points.get("V/prediction").values)

    path = sheet.to_zarr(str(tmp_path / "sheet.zarr"))
    reopened = geoml.data.Surface3D.open(path)
    assert np.array_equal(np.asarray(reopened.get("V/prediction").values),
                          np.asarray(sheet.get("V/prediction").values))
    assert reopened.variables["V"].n_sim == 3


# --------------------------------------------------------------------------- #
# what a prediction into measured data leaves for the figures
# --------------------------------------------------------------------------- #
def test_a_prediction_into_data_records_where_each_measurement_fell(trained):
    data = trained.data
    trained.predict(data, n_sim=5)
    pit = data.get_metadata("pit_V")
    measured = np.isfinite(np.asarray(data.get("V/measurements").values))
    assert np.all(np.isfinite(pit[measured]))
    assert np.all((pit[measured] >= 0.0) & (pit[measured] <= 1.0))
    # in-sample, the assays sit near the middle of their distributions
    assert 0.3 < np.mean(pit[measured]) < 0.7


def test_the_warped_measurements_are_the_model_s_view_of_the_data(trained):
    data = trained.data
    trained.predict(data, n_sim=3)
    expected, measured, _ = geoml.plots.prepare.warped_values(trained, "V")
    stored = data.get_metadata("warped_V_0")
    np.testing.assert_allclose(stored[measured], expected[:, 0])
    assert np.all(np.isnan(stored[~measured]))


def test_a_grid_gets_no_measurement_columns(trained):
    grid = _grid()
    trained.predict(grid, n_sim=3)
    assert not [k for k in grid.metadata
                if k.startswith(("pit_", "warped_"))]


def test_a_partial_prediction_fills_the_columns_in_parts(trained):
    """Resumed, the PITs are those of one call over everything: a
    location's sample does not depend on which others share its batch."""
    point, _ = geoml.datasets.walker()
    whole = point[np.arange(200)]
    trained.predict(whole, n_sim=4)
    parts = point[np.arange(200)]
    first = np.arange(200) < 90
    trained.predict(parts, n_sim=4, where=first)
    assert np.all(np.isnan(parts.get_metadata("pit_V")[~first]))
    trained.predict(parts, where=~first)
    np.testing.assert_array_equal(parts.get_metadata("pit_V"),
                                  whole.get_metadata("pit_V"))


# --------------------------------------------------------------------------- #
# emptying what was predicted at some locations
# --------------------------------------------------------------------------- #
def _everything(container):
    """Every column the container exports, and every realization store."""
    held = {"frame": container.as_data_frame()}

    def walk(node, path):
        for name, store in node.realization_stores():
            held["%s/%s" % (path, name)] = np.asarray(store).copy()
        for label, child in node.child_nodes().items():
            walk(child, "%s/%s" % (path, label))

    for name, variable in container.variables.items():
        walk(variable, name)
    return held


def _same(a, b, rows=slice(None)):
    assert set(a) == set(b)
    for key in a:
        if key == "frame":
            import pandas as pd
            pd.testing.assert_frame_equal(a[key].iloc[rows], b[key].iloc[rows])
        else:
            assert np.array_equal(a[key][rows], b[key][rows], equal_nan=True), key


def test_emptied_locations_are_unpredicted_and_fill_again_to_the_bit(trained):
    """GeoScape's item 48, and its gate: what `clear_predictions` empties
    is what `unpredicted` names, the other locations keep every bit, and
    predicting the emptied ones gives back what was there."""
    grid = _grid()
    trained.predict(grid, n_sim=4)
    whole = _everything(grid)
    gone = np.zeros(grid.n_data, dtype=bool)
    gone[20:55] = True

    assert grid.clear_predictions(where=gone) is grid
    assert np.array_equal(grid.unpredicted(), gone)
    _same(_everything(grid), whole, rows=np.flatnonzero(~gone))
    emptied = _everything(grid)
    assert np.all(np.isnan(emptied["V/simulations"][gone]))
    assert np.all(np.isnan(np.asarray(
        grid.values("V/noise_variance"), dtype=float)[gone]))

    trained.predict(grid, where=grid.unpredicted())
    _same(_everything(grid), whole)


def test_emptying_takes_indices_a_stored_filter_and_names(trained):
    grid = _grid()
    trained.predict(grid, n_sim=3)
    grid.add_metadata("west", np.asarray(grid.coordinates[:, 0] < 100))
    grid.clear_predictions("V", where="west")
    assert np.array_equal(grid.unpredicted(),
                          np.asarray(grid.coordinates[:, 0] < 100))
    grid.clear_predictions(["V"], where=[0, 99])
    assert grid.unpredicted()[[0, 99]].all()
    with pytest.raises(ValueError, match="no variable named 'W'"):
        grid.clear_predictions("W")
    grid.clear_predictions()
    assert grid.unpredicted().all()


def test_emptying_keeps_the_measurements_and_the_metadata(trained):
    point, _ = geoml.datasets.walker()
    data = point[np.arange(120)]
    measured = np.asarray(data.values("V/measurements"), dtype=float).copy()
    trained.predict(data, n_sim=3)
    pit = data.get_metadata("pit_V").copy()
    data.clear_predictions()

    assert data.unpredicted().all()
    assert np.array_equal(
        np.asarray(data.values("V/measurements"), dtype=float), measured,
        equal_nan=True)
    assert np.array_equal(data.get_metadata("pit_V"), pit, equal_nan=True)


def test_every_kind_of_variable_is_emptied_and_filled_again(tmp_path):
    """A category's parts, a vector variable's components, their coded and
    keyed columns and every realization store, on a container reopened from
    its store."""
    geoml.set_seed(1234)
    train, held = geoml.datasets.jura()
    n_rock = len(train.get("Rock").labels)
    n_el = len(train.get("Elements").labels)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(train, 40, seed=0),
        transform=geoml.transform.Isotropic(1.0))
    model = geoml.models.VGPNetwork(
        train, {"Rock": geoml.likelihood.CategoricalGaussianIndicator(n_rock),
                "Elements": geoml.likelihood.MultivariateGaussian(n_el)},
        latent_network=[geoml.latent.BasicGP(root, size=n_rock),
                        geoml.latent.BasicGP(root, size=n_el)],
        options=geoml.models.GPOptions(verbose=False, training_samples=8))
    model.train_full(max_iter=3)
    model.predict(held, n_sim=3)
    held.to_zarr(str(tmp_path / "held.zarr"))
    stored = type(held).open(str(tmp_path / "held.zarr"))
    whole = _everything(stored)
    gone = np.arange(stored.n_data) % 3 == 0

    stored.clear_predictions(where=gone)
    for name in ("Rock", "Elements"):
        assert np.array_equal(stored.unpredicted(name), gone)
    _same(_everything(stored), whole, rows=np.flatnonzero(~gone))
    assert set(np.asarray(stored.get("Rock/predicted").to_numpy())[gone]) \
        == {""}
    assert np.array_equal(
        np.asarray(stored.values("Rock/measurements_a")),
        np.asarray(held.values("Rock/measurements_a")))

    model.predict(stored, where=stored.unpredicted())
    _same(_everything(stored), whole)


def test_a_block_model_is_emptied_where_a_stored_filter_says(trained):
    blocks = geoml.data.Blocks2D(
        start=[10, 10], n=[8, 8], step=[20, 20], discretization=[2, 2])
    trained.predict(blocks, n_sim=3)
    whole = _everything(blocks)
    blocks.add_metadata("north", np.asarray(blocks.coordinates[:, 1] > 90))
    blocks.clear_predictions(where="north")
    assert np.array_equal(blocks.unpredicted(),
                          blocks.get_metadata("north").astype(bool))
    trained.predict(blocks, where="north")
    _same(_everything(blocks), whole)
