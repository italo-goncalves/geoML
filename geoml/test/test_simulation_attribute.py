"""Access to a single simulation as an ``_Attribute``.

Simulations used to be a list of ``_Attribute`` objects, so each one could be
reshaped, smoothed and contoured on its own. They now live in a single
``(n_data, n_sim)`` store; ``variable.simulation(i)`` restores that behavior by
wrapping one column.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.storage as storage

from test_containers import _synth_point_model


def _grid2d_with_simulations(n_sim=4, seed=0):
    grid = geoml.data.Grid2D(start=[0, 0], n=[6, 5], step=[1, 1])
    grid.add_continuous_variable("v")
    var = grid.variables["v"]

    rng = np.random.default_rng(seed)
    sims = rng.normal(size=(grid.n_data, n_sim))
    var.allocate_simulations(n_sim)
    var.simulations[:, :] = sims
    return grid, var, sims


def test_simulation_wraps_the_matching_column():
    grid, var, sims = _grid2d_with_simulations()

    assert var.n_sim == 4
    for i in range(4):
        attr = var.simulation(i)
        assert isinstance(attr, geoml.data._Variable._Attribute)
        assert attr.coordinates is grid
        assert np.allclose(np.asarray(attr.values), sims[:, i])


def test_simulation_supports_the_attribute_helpers():
    grid, var, sims = _grid2d_with_simulations()

    image = var.simulation(2).as_image()
    assert image.shape == tuple(grid.grid_size)[::-1]
    assert np.allclose(image, np.reshape(sims[:, 2], grid.grid_size,
                                         order="F").transpose())

    # smoothing changes the values, and the unsmoothed one is still available
    smoothed = var.simulation(2)
    smoothed.smooth(sigma=1.0)
    assert not np.allclose(np.asarray(smoothed.values), sims[:, 2])
    assert np.allclose(np.asarray(var.simulation(2).values), sims[:, 2])


def test_simulation_is_a_copy_that_can_be_written_back():
    grid, var, sims = _grid2d_with_simulations()

    attr = var.simulation(1)
    attr.smooth(sigma=1.0)
    assert np.allclose(np.asarray(var.simulations[:, 1]), sims[:, 1])

    var.simulations[:, 1] = attr.values
    assert np.allclose(np.asarray(var.simulations[:, 1]),
                       np.asarray(attr.values))
    # the other simulations are untouched
    assert np.allclose(np.asarray(var.simulations[:, 0]), sims[:, 0])


def test_simulation_without_simulations():
    grid = geoml.data.Grid2D(start=[0, 0], n=[4, 4], step=[1, 1])
    grid.add_continuous_variable("v")
    var = grid.variables["v"]

    assert var.n_sim == 0
    with pytest.raises(geoml.data.NoDataError):
        var.simulation(0)


def test_simulation_on_a_zarr_backed_store(monkeypatch):
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    grid, var, sims = _grid2d_with_simulations()

    assert var.simulations.backend == "zarr"
    assert np.allclose(np.asarray(var.simulation(3).values), sims[:, 3])
    assert var.simulation(3).as_image().shape == tuple(grid.grid_size)[::-1]


def test_simulation_on_other_variable_types():
    """Binary variables and the components of a rock type keep their own store."""
    grid = geoml.data.Grid2D(start=[0, 0], n=[4, 4], step=[1, 1])
    grid.add_binary_variable("b", labels=("no", "yes"))
    grid.add_rock_type_variable("rt", labels=("a", "b", "c"))

    rng = np.random.default_rng(1)
    binary = grid.variables["b"]
    binary.allocate_simulations(3)
    values = rng.normal(size=(grid.n_data, 3))
    binary.simulations[:, :] = values

    assert binary.n_sim == 3
    assert np.allclose(np.asarray(binary.simulation(2).values), values[:, 2])

    rock_type = grid.variables["rt"]
    rock_type.allocate_simulations(3)
    component = rock_type.components["a"]
    assert component.n_sim == 3
    assert component.simulation(0).values.shape == (grid.n_data,)


def test_simulation_after_a_real_prediction():
    model = _synth_point_model(3)
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[4, 5, 6], step=[2, 2, 2])
    model.predict(grid, n_sim=3)

    var = grid.variables["v"]
    assert var.n_sim == 3

    cube = var.simulation(1).as_cube()
    assert cube.shape == tuple(grid.grid_size)
    assert np.all(np.isfinite(cube))
    assert np.allclose(cube.ravel(order="F"),
                       np.asarray(var.simulations[:, 1]))
