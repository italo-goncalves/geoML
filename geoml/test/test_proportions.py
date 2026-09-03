"""The proportions figure's arithmetic: the data's declustered category
shares against the model's expected shares over the ground in reach.

The categorical swath with no slabs, so the same corrections are checked
the same way: a planted split recovered on both sides, declustering
touching only the data's side, blocks counting at their own volume and
`where` cutting the model's.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.plots.prepare as prep

LABELS = ["granite", "basalt"]


def _rocks(coordinates, split=50.0):
    """Samples that are granite left of `split` along x, basalt right."""
    columns = ["X", "Y", "Z"][:coordinates.shape[1]]
    data = geoml.data.PointData(pd.DataFrame(coordinates, columns=columns),
                                columns)
    data.add_categorical_variable(
        "rock", LABELS,
        np.where(coordinates[:, 0] < split, "granite", "basalt"))
    return data


def _fading_grid(n=41):
    """Granite's probability fades from one to zero along x."""
    grid = geoml.data.Grid2D(start=[0, 0], end=[100, 100], n=[n, n])
    grid.add_categorical_variable("rock", LABELS,
                                  np.full(grid.n_data, "", dtype=object))
    x = np.asarray(grid.coordinates)[:, 0]
    var = grid.variables["rock"]
    var.components["granite"].probability.values[:] = 1.0 - x / 100.0
    var.components["basalt"].probability.values[:] = x / 100.0
    return grid


def test_a_planted_split_is_recovered_on_both_sides():
    rng = np.random.default_rng(5)
    coordinates = rng.uniform(0.0, 100.0, size=[300, 2])
    data = _rocks(coordinates)

    result = prep.proportions(data, _fading_grid(), "rock")
    assert result["labels"] == LABELS
    assert result["data_share"][0] == pytest.approx(
        np.mean(coordinates[:, 0] < 50))
    # the grid is symmetric about x = 50: granite's expected share is a half
    assert result["model_share"][0] == pytest.approx(0.5)
    assert result["data_share"].sum() == pytest.approx(1.0)
    assert result["model_share"].sum() == pytest.approx(1.0)
    assert result["data_count"] == 300
    assert result["model_count"] == 41 * 41
    assert not result["declustered"]


def test_declustering_weights_move_only_the_data_side():
    """A dense patch of granite samples inflates the raw share; weights
    that silence the patch bring it back to the scattered samples' own."""
    rng = np.random.default_rng(1)
    spread = rng.uniform(0.0, 100.0, size=[300, 2])
    patch = np.array([45.0, 50.0]) + rng.normal(scale=1.0, size=[300, 2])
    data = _rocks(np.concatenate([spread, patch]))
    grid = _fading_grid()

    raw = prep.proportions(data, grid, "rock")
    weights = np.concatenate([np.ones(300), np.full(300, 1e-3)])
    declustered = prep.proportions(data, grid, "rock", weights=weights)
    assert raw["data_share"][0] > 0.7
    assert declustered["data_share"][0] == pytest.approx(
        np.mean(spread[:, 0] < 50), abs=0.01)
    assert declustered["declustered"] and not raw["declustered"]
    assert np.allclose(raw["model_share"], declustered["model_share"])


def test_blocks_count_at_their_own_volume_and_where_cuts_the_model():
    """One coarse block of granite against its neighbour's eight children
    of basalt: equal volumes, so a half -- not one block in nine."""
    blocks = geoml.data.BlockSet3D(start=[0, 0, 0], n=[2, 1, 1],
                                   step=[10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2), max_levels=1)
    blocks = blocks.split(np.array([False, True]))
    blocks.add_categorical_variable("rock", LABELS,
                                    np.full(blocks.n_data, "", dtype=object))
    volume = np.asarray(blocks.block_volume, dtype=float)
    coarse = volume == volume.max()
    var = blocks.variables["rock"]
    var.components["granite"].probability.values[:] = coarse.astype(float)
    var.components["basalt"].probability.values[:] = 1.0 - coarse
    rng = np.random.default_rng(3)
    data = _rocks(rng.uniform(0.0, 20.0, size=[40, 3]), split=10.0)

    result = prep.proportions(data, blocks, "rock")
    assert result["model_share"][0] == pytest.approx(0.5)
    assert not np.isclose(result["model_share"][0], coarse.mean())

    children = prep.proportions(data, blocks, "rock", where=~coarse)
    assert children["model_share"][0] == pytest.approx(0.0)
    assert children["model_count"] == 8
    with pytest.raises(ValueError, match="where"):
        prep.proportions(data, blocks, "rock",
                         where=np.zeros(blocks.n_data, dtype=bool))


def test_a_category_the_model_lacks_is_refused():
    rng = np.random.default_rng(2)
    data = _rocks(rng.uniform(0.0, 100.0, size=[100, 2]))
    grid = geoml.data.Grid2D(start=[0, 0], end=[100, 100], n=[11, 11])
    grid.add_categorical_variable("rock", ["granite", "gneiss"],
                                  np.full(grid.n_data, "", dtype=object))
    grid.variables["rock"].components["granite"].probability.values[:] = 1.0
    with pytest.raises(ValueError, match="no category"):
        prep.proportions(data, grid, "rock")


def test_both_backends_draw_one_bar_pair_per_category():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(6)
    data = _rocks(rng.uniform(0.0, 100.0, size=[200, 2]))
    grid = _fading_grid(21)

    figure = geoml.plots.Explorer(data, categorical="rock").proportions(grid)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].containers) == 2 * len(LABELS)
    matplotlib.pyplot.close(figure)

    figure = geoml.plots.Interactive(data, categorical="rock").proportions(
        grid)
    assert [trace.name for trace in figure.data] == ["data", "model"]
    assert figure.layout.barmode == "group"
    assert all(getattr(trace, "customdata", None) is None
               for trace in figure.data)
