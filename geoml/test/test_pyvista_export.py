"""Exporting variables to pyvista.

Simulations are the expensive part: each one is a full-length array in the
exported object, and the store is chunked along rows only, so reading them one
column at a time decompresses the whole array once per simulation. They are now
left out unless asked for, and the ones asked for are read in a single pass.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.storage as storage

N_SIM = 8


def _grid(n_sim=N_SIM, on_disk=False, threshold=None):
    """A small grid carrying a continuous variable with simulations."""
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[4, 5, 3], step=[1, 1, 1])
    grid.add_continuous_variable("v", np.full(grid.n_data, np.nan))
    variable = grid.variables["v"]

    if on_disk:
        threshold(storage, "DEFAULT_THRESHOLD", 0)
    variable.allocate_simulations(n_sim)

    rng = np.random.default_rng(0)
    values = rng.normal(size=(grid.n_data, n_sim))
    variable.simulations[:, :] = values
    variable.prediction.values[:] = values[:, 0]
    return grid, values


def _simulation_arrays(exported):
    return sorted(name for name in exported.point_data.keys()
                  if "simulation" in name)


def test_simulations_stay_out_unless_asked_for():
    grid, _ = _grid()
    assert _simulation_arrays(grid.as_pyvista()) == []
    assert "v - prediction" in grid.as_pyvista().point_data


def test_every_simulation_comes_when_asked_for_all():
    grid, _ = _grid()
    names = _simulation_arrays(grid.as_pyvista(simulations=True))
    assert len(names) == N_SIM


@pytest.mark.parametrize("selection,expected", [
    (False, []),
    (None, []),
    (3, [0, 1, 2]),
    ([0, 5], [0, 5]),
    ((2,), [2]),
    (True, list(range(N_SIM))),
])
def test_the_selection_says_which_ones(selection, expected):
    grid, _ = _grid()
    exported = grid.as_pyvista(simulations=selection)
    assert _simulation_arrays(exported) == \
        sorted("v - simulation %d" % i for i in expected)


def test_asking_for_more_than_there_are_takes_what_exists():
    grid, _ = _grid()
    names = _simulation_arrays(grid.as_pyvista(simulations=N_SIM + 5))
    assert len(names) == N_SIM


def test_an_index_that_does_not_exist_is_an_error():
    grid, _ = _grid()
    with pytest.raises(IndexError, match="not among the %d" % N_SIM):
        grid.as_pyvista(simulations=[0, N_SIM])


def test_the_exported_values_are_the_stored_ones():
    grid, values = _grid()
    exported = grid.as_pyvista(simulations=[1, 4])
    for i in (1, 4):
        got = np.asarray(exported.point_data["v - simulation %d" % i])
        # a cube is written in pyvista's own axis order
        assert np.allclose(np.sort(got), np.sort(values[:, i]))


def test_the_store_is_read_once_however_many_are_asked_for(monkeypatch):
    """The point of the change: one pass over the chunks, not one per column."""
    grid, _ = _grid()
    original = storage.ArrayStore.as_dask
    reads = []

    def counted(self):
        if len(self.shape) == 2:          # the simulations store
            reads.append(self.shape)
        return original(self)

    monkeypatch.setattr(storage.ArrayStore, "as_dask", counted)

    grid.as_pyvista(simulations=True)
    assert len(reads) == 1

    reads.clear()
    grid.as_pyvista(simulations=[0, 3, 6])
    assert len(reads) == 1

    reads.clear()
    grid.as_pyvista()
    assert len(reads) == 0                # nothing read when none are wanted


def test_it_works_the_same_on_the_zarr_backend(monkeypatch):
    grid, values = _grid(on_disk=True, threshold=monkeypatch.setattr)
    assert grid.variables["v"].simulations.backend == "zarr"

    exported = grid.as_pyvista(simulations=[2])
    got = np.asarray(exported.point_data["v - simulation 2"])
    assert np.allclose(np.sort(got), np.sort(values[:, 2]))


def test_points_and_blocks_take_the_same_argument():
    coords = np.stack([np.arange(12.0)] * 3, axis=1)
    point = geoml.data.PointData(
        pd.DataFrame(coords, columns=["X", "Y", "Z"]), ["X", "Y", "Z"])
    point.add_continuous_variable("v", np.arange(12.0))
    point.variables["v"].allocate_simulations(4)
    point.variables["v"].simulations[:, :] = np.zeros((12, 4))

    assert _simulation_arrays(point.as_pyvista()) == []
    assert len(_simulation_arrays(point.as_pyvista(simulations=True))) == 4

    blocks = geoml.data.Blocks3D(start=[0, 0, 0], n=[3, 3, 2], step=[1, 1, 1])
    blocks.add_continuous_variable("v", np.zeros(blocks.n_data))
    blocks.variables["v"].allocate_simulations(3)
    blocks.variables["v"].simulations[:, :] = np.zeros((blocks.n_data, 3))

    exported = blocks.as_pyvista(simulations=2)
    written = [name for name in exported.cell_data.keys() if "simulation" in name]
    assert len(written) == 2


def test_a_categorical_variable_passes_the_argument_to_its_categories():
    coords = np.stack([np.arange(12.0)] * 3, axis=1)
    point = geoml.data.PointData(
        pd.DataFrame(coords, columns=["X", "Y", "Z"]), ["X", "Y", "Z"])
    point.add_categorical_variable(
        "rock", np.array(["ore", "waste"] * 6))
    variable = point.variables["rock"]
    categories = list(variable.components.keys())
    for label in categories:
        variable.components[label].allocate_simulations(3)
        variable.components[label].simulations[:, :] = np.zeros((12, 3))

    assert _simulation_arrays(point.as_pyvista()) == []
    assert _simulation_arrays(point.as_pyvista(simulations=1)) == \
        sorted("rock - %s - simulation 0" % label for label in categories)


def test_the_data_frame_reads_the_store_once_too(monkeypatch):
    """`as_data_frame` had the same column-by-column loop."""
    grid, values = _grid()
    original = storage.ArrayStore.as_dask
    reads = []

    def counted(self):
        if len(self.shape) == 2:
            reads.append(self.shape)
        return original(self)

    monkeypatch.setattr(storage.ArrayStore, "as_dask", counted)

    df = grid.variables["v"].as_data_frame(simulations=True)
    assert len(reads) == 1
    assert np.allclose(df["v_sim_3"].to_numpy(), values[:, 3])


def test_the_data_frame_takes_a_selection_as_well():
    """The flag was a bool; the same values now mean the same as elsewhere."""
    grid, values = _grid()
    variable = grid.variables["v"]

    columns = [c for c in variable.as_data_frame(simulations=False) if "_sim_" in c]
    assert columns == []

    columns = [c for c in variable.as_data_frame(simulations=True) if "_sim_" in c]
    assert len(columns) == N_SIM

    df = variable.as_data_frame(simulations=[2, 6])
    assert [c for c in df if "_sim_" in c] == ["v_sim_2", "v_sim_6"]
    assert np.allclose(df["v_sim_6"].to_numpy(), values[:, 6])


def test_an_empty_attribute_is_still_left_out():
    """The all-NaN guard is now vectorized; it must decide the same way."""
    grid, _ = _grid()
    exported = grid.as_pyvista()
    assert "v" not in exported.point_data          # measurements are all NaN

    grid.variables["v"].measurements.values[0] = 1.0
    assert "v" in grid.as_pyvista().point_data
