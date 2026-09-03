"""The swath plot's arithmetic: declustered data against a model in reach.

A swath compares the data's mean with the model's slab by slab along one
axis, which is only fair once the data are declustered and the model is
read only where the data can speak. The tests plant a known trend and
check that each correction does exactly what it claims and nothing else.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.plots.prepare as prep


def _points(coordinates, name, values):
    frame = pd.DataFrame(coordinates, columns=["X", "Y"][:coordinates.shape[1]])
    data = geoml.data.PointData(frame, list(frame.columns))
    data.add_continuous_variable(name, values)
    return data


def _trend_case(seed=0, n=400):
    """Samples scattered over [0, 100]^2 with value = x, and a grid over the
    same square predicting exactly that."""
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 100.0, size=[n, 2])
    data = _points(coordinates, "v", coordinates[:, 0])

    grid = geoml.data.Grid2D(start=[0, 0], end=[100, 100], n=[51, 51])
    grid.add_continuous_variable("v", np.full(grid.n_data, np.nan))
    grid.variables["v"].prediction.values[:] = \
        np.asarray(grid.coordinates)[:, 0]
    return data, grid


def test_a_linear_trend_is_recovered_on_both_sides():
    data, grid = _trend_case()
    (panel,) = prep.swath(data, grid, "v", axis="X", bins=10)

    assert panel["axis"] == "X"
    assert len(panel["lo"]) == 10
    # the grid is discrete -- nodes at 0, 2, ..., 8 average to 4, not 5 --
    # so the model side is checked against exactly that, the data side
    # against the centres it scatters around
    x = np.asarray(grid.coordinates)[:, 0]
    slab = np.minimum(np.searchsorted(panel["hi"], x, side="right"), 9)
    expected = np.bincount(slab, weights=x) / np.bincount(slab)
    assert np.allclose(panel["model_mean"], expected, atol=1e-9)
    assert np.allclose(panel["data_mean"], panel["centre"], atol=2.0)
    assert panel["data_count"].sum() == data.n_data
    assert panel["model_count"].sum() == grid.n_data
    assert panel["band_lo"] is None and panel["band_hi"] is None


def test_declustering_moves_the_data_mean_where_the_cluster_is():
    """A dense patch of high values inside one slab drags the raw mean up;
    the stored declustering column pulls it back toward the trend."""
    rng = np.random.default_rng(1)
    spread = rng.uniform(0.0, 100.0, size=[300, 2])
    patch = np.array([45.0, 50.0]) + rng.normal(scale=1.0, size=[300, 2])
    coordinates = np.concatenate([spread, patch])
    values = np.concatenate([spread[:, 0], np.full(300, 95.0)])
    data = _points(coordinates, "v", values)
    _, grid = _trend_case()

    raw = prep.swath(data, grid, "v", axis="X", bins=10,
                     weights=np.ones(data.n_data))[0]
    data.decluster(on="v")
    declustered = prep.swath(data, grid, "v", axis="X", bins=10)[0]

    slab = 4  # [40, 50), where the patch sits
    assert raw["data_mean"][slab] > 80.0
    assert declustered["data_mean"][slab] < raw["data_mean"][slab]
    # elsewhere the two agree, the patch being the only clustering
    others = np.arange(10) != slab
    assert np.allclose(raw["data_mean"][others],
                       declustered["data_mean"][others], atol=3.0)


def test_the_reach_removes_ground_the_data_never_saw():
    """Samples cover x in [0, 50] and the grid runs to 100, predicting
    nonsense beyond the data. Without `where` the nonsense makes slabs;
    with the reach column it is not there."""
    rng = np.random.default_rng(2)
    coordinates = rng.uniform([0.0, 0.0], [50.0, 100.0], size=[300, 2])
    data = _points(coordinates, "v", coordinates[:, 0])
    _, grid = _trend_case()
    # the nonsense starts past the reach, not at the data's edge: a reach of
    # three from samples up to x = 50 legitimately holds nodes at 52
    x = np.asarray(grid.coordinates)[:, 0]
    grid.variables["v"].prediction.values[:] = np.where(x <= 56, x, 999.0)

    unfiltered = prep.swath(data, grid, "v", axis="X", bins=10)[0]
    assert unfiltered["hi"][-1] == pytest.approx(100.0)
    assert unfiltered["model_mean"][-1] == pytest.approx(999.0)

    grid.assign_from_data(data, distance=3.0)
    filtered = prep.swath(data, grid, "v", axis="X", bins=10,
                          where="near_data")[0]
    assert filtered["hi"][-1] <= 53.0
    assert np.all(filtered["model_mean"] < 60.0)
    assert np.allclose(filtered["model_mean"], filtered["centre"], atol=1.5)


def test_blocks_count_at_their_own_volume():
    """One slab holding one coarse block at 10 and its neighbour's eight
    children at 0: the mean is the volume-weighted one, not the
    count-weighted one."""
    blocks = geoml.data.BlockSet3D(start=[0, 0, 0], n=[2, 1, 1],
                                   step=[10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2), max_levels=1)
    blocks = blocks.split(np.array([False, True]))
    assert blocks.n_data == 9
    blocks.add_continuous_variable("v", np.full(blocks.n_data, np.nan))
    volume = np.asarray(blocks.block_volume, dtype=float)
    values = np.where(volume == volume.max(), 10.0, 0.0)
    blocks.variables["v"].prediction.values[:] = values

    rng = np.random.default_rng(3)
    coordinates = rng.uniform(0.0, 20.0, size=[40, 3])
    frame = pd.DataFrame(coordinates, columns=["X", "Y", "Z"])
    data = geoml.data.PointData(frame, ["X", "Y", "Z"])
    data.add_continuous_variable("v", np.full(40, 5.0))

    (panel,) = prep.swath(data, blocks, "v", axis="Y", bins=1)
    expected = (values * volume).sum() / volume.sum()
    assert panel["model_mean"][0] == pytest.approx(expected)
    assert panel["model_mean"][0] == pytest.approx(5.0)  # equal volumes
    assert not np.isclose(panel["model_mean"][0], values.mean())


def test_the_band_brackets_the_mean_and_is_read_in_bands(tmp_path,
                                                          monkeypatch):
    data, grid = _trend_case()
    rng = np.random.default_rng(4)
    x = np.asarray(grid.coordinates)[:, 0]
    sims = x[:, None] + rng.normal(scale=5.0, size=[grid.n_data, 30])
    var = grid.variables["v"]
    var.prediction.values[:] = sims.mean(axis=1)
    var.simulations = geoml.storage.ArrayStore.allocate(
        sims.shape, backend="zarr", store=str(tmp_path / "sims"),
        chunks=(500, sims.shape[1]))
    var.simulations[:] = sims

    # reading the simulations whole is the one thing a block model cannot
    # afford; the store's own class refuses it, the prediction's stays
    class Refusing(geoml.storage.ArrayStore):
        def __array__(self, dtype=None, copy=None):
            raise AssertionError("the simulations were read whole")
    var.simulations.__class__ = Refusing

    (panel,) = prep.swath(data, grid, "v", axis="X", bins=8)
    assert panel["band_lo"].shape == (8,)
    assert np.all(panel["band_lo"] <= panel["model_mean"] + 1e-9)
    assert np.all(panel["band_hi"] >= panel["model_mean"] - 1e-9)
    assert np.all(panel["band_hi"] - panel["band_lo"] > 0)


def test_categorical_shares_sum_to_one_on_both_sides():
    rng = np.random.default_rng(5)
    coordinates = rng.uniform(0.0, 100.0, size=[300, 2])
    rock = np.where(coordinates[:, 0] < 50, "granite", "basalt")
    frame = pd.DataFrame(coordinates, columns=["X", "Y"])
    data = geoml.data.PointData(frame, ["X", "Y"])
    data.add_categorical_variable("rock", ["granite", "basalt"], rock)

    grid = geoml.data.Grid2D(start=[0, 0], end=[100, 100], n=[41, 41])
    grid.add_categorical_variable("rock", ["granite", "basalt"],
                                  np.full(grid.n_data, "", dtype=object))
    x = np.asarray(grid.coordinates)[:, 0]
    granite = np.clip(1.0 - x / 100.0, 0.0, 1.0)
    var = grid.variables["rock"]
    var.components["granite"].probability.values[:] = granite
    var.components["basalt"].probability.values[:] = 1.0 - granite

    result = prep.categorical_swath(data, grid, "rock", axis="X", bins=5)
    assert result["labels"] == ["granite", "basalt"]
    assert np.allclose(result["data_share"].sum(axis=1), 1.0)
    assert np.allclose(result["model_share"].sum(axis=1), 1.0)
    # granite fades with x on both sides
    assert np.all(np.diff(result["model_share"][:, 0]) < 0)
    assert result["data_share"][0, 0] == pytest.approx(1.0)
    assert result["data_share"][-1, 0] == pytest.approx(0.0)
    assert not result["declustered"]


def test_both_backends_draw_the_continuous_swath():
    import matplotlib
    matplotlib.use("Agg")
    data, grid = _trend_case()
    explorer = geoml.plots.Explorer(data, continuous="v")
    figure = explorer.swath(grid, axis="X", bins=6)
    # means above, samples below, and the twin axis for the model's cells
    assert len(figure.axes) == 3
    matplotlib.pyplot.close(figure)

    interactive = geoml.plots.Interactive(data, continuous="v")
    figure = interactive.swath(grid, axis="X", bins=6)
    names = [trace.name for trace in figure.data]
    assert "model mean" in names and "data mean" in names
    # no simulations: no band -- model line, data line, data markers,
    # the samples bar and the cells line
    assert len(figure.data) == 5
    assert all(getattr(trace, "customdata", None) is None
               or trace.mode == "markers" for trace in figure.data)


def test_both_backends_draw_the_categorical_swath():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(6)
    coordinates = rng.uniform(0.0, 100.0, size=[200, 2])
    rock = np.where(coordinates[:, 0] < 50, "granite", "basalt")
    frame = pd.DataFrame(coordinates, columns=["X", "Y"])
    data = geoml.data.PointData(frame, ["X", "Y"])
    data.add_categorical_variable("rock", ["granite", "basalt"], rock)
    grid = geoml.data.Grid2D(start=[0, 0], end=[100, 100], n=[21, 21])
    grid.add_categorical_variable("rock", ["granite", "basalt"],
                                  np.full(grid.n_data, "", dtype=object))
    x = np.asarray(grid.coordinates)[:, 0]
    var = grid.variables["rock"]
    var.components["granite"].probability.values[:] = 1.0 - x / 100.0
    var.components["basalt"].probability.values[:] = x / 100.0

    figure = geoml.plots.Explorer(data, categorical="rock").swath(
        grid, axis="X", bins=4)
    assert len(figure.axes) == 2
    matplotlib.pyplot.close(figure)

    figure = geoml.plots.Interactive(data, categorical="rock").swath(
        grid, axis="X", bins=4)
    # two categories, data and model each, plus the samples bar
    assert len(figure.data) == 2 * 2 + 1
    assert figure.layout.barmode == "stack"


def test_axes_are_named_or_indexed_and_checked():
    data, grid = _trend_case()
    by_index = prep.swath(data, grid, "v", axis=1, bins=4)[0]
    by_label = prep.swath(data, grid, "v", axis="Y", bins=4)[0]
    assert by_index["axis"] == "Y"
    assert np.allclose(by_index["model_mean"], by_label["model_mean"],
                       equal_nan=True)
    with pytest.raises(ValueError, match="coordinate"):
        prep.swath(data, grid, "v", axis="Z")
    with pytest.raises(ValueError, match="axis"):
        prep.swath(data, grid, "v", axis=2)
