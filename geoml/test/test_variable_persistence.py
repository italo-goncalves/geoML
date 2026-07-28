"""Persistence + sim-collapse round-trips for the non-continuous variable types.

Each variable is built on a small PointData, its attributes are populated
directly (as ``update`` would), then the container is written with ``to_zarr``
and reloaded. This exercises the collapsed (n_data, n_sim) simulation store, the
coded (text) and bool attribute encodings, and component recursion, without
needing a trained categorical/binary model.
"""
import numpy as np
import pandas as pd
import pytest

import geoml


N = 6


def _point():
    df = pd.DataFrame({"c0": np.arange(N, dtype=float),
                       "c1": np.arange(N, dtype=float) * 2})
    return geoml.data.PointData(df, ["c0", "c1"])


def _codes(labels, values):
    """What a coded attribute stores for these labels — what `update` writes."""
    return np.array([list(labels).index(value) for value in values])


def test_vector_variable_roundtrip(tmp_path):
    pt = _point()
    pt.add_vector_variable("vec", labels=["a", "b"],
                           measurements=np.random.random((N, 2)))
    var = pt.variables["vec"]
    var.allocate_simulations(4)
    var.uncertainty.values[:] = np.random.random(N)
    for lb in var.labels:
        comp = var.components[lb]
        comp.prediction.values[:] = np.random.random(N)
        comp.simulations[:, :] = np.random.random((N, 4))

    unc = np.asarray(var.uncertainty.values).copy()
    pred_a = np.asarray(var.components["a"].prediction.values).copy()
    sims_a = np.asarray(var.components["a"].simulations).copy()

    pt.to_zarr(str(tmp_path / "vec.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "vec.zarr"))
    rv = r.variables["vec"]

    assert list(rv.labels) == ["a", "b"]
    assert np.allclose(np.asarray(rv.uncertainty.values), unc)
    assert np.allclose(np.asarray(rv.components["a"].prediction.values), pred_a)
    assert np.allclose(np.asarray(rv.components["a"].simulations), sims_a)


def test_compositional_variable_roundtrip(tmp_path):
    pt = _point()
    comp_meas = np.random.random((N, 3))
    comp_meas /= comp_meas.sum(axis=1, keepdims=True)
    pt.add_compositional_variable("comp", labels=["x", "y", "z"],
                                  measurements=comp_meas)
    var = pt.variables["comp"]
    var.allocate_simulations(3)
    for lb in var.labels:
        var.components[lb].prediction.values[:] = np.random.random(N)
        var.components[lb].simulations[:, :] = np.random.random((N, 3))
    pred_x = np.asarray(var.components["x"].prediction.values).copy()

    pt.to_zarr(str(tmp_path / "comp.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "comp.zarr"))
    rv = r.variables["comp"]

    assert list(rv.labels) == ["x", "y", "z"]
    assert np.allclose(np.asarray(rv.components["x"].prediction.values), pred_x)


def test_categorical_variable_roundtrip(tmp_path):
    pt = _point()
    meas = np.array(
        ["granite", "basalt", "granite", "basalt", "granite", "basalt"])
    pt.add_categorical_variable("rock", labels=["basalt", "granite"],
                                measurements=meas)
    rock = pt.variables["rock"]
    rock.allocate_simulations(3)
    rock.predicted.values[:] = _codes(rock.labels, meas)     # coded labels
    rock.entropy.values[:] = np.random.random(N)
    for lb in rock.labels:
        c = rock.components[lb]
        c.indicator_predicted.values[:] = np.random.random(N)
        c.simulations[:, :] = np.random.random((N, 3))

    predicted = rock.predicted.to_numpy().copy()
    meas_a = rock.measurements_a.to_numpy().copy()
    boundary = np.asarray(rock.boundary.values).copy()
    entropy = np.asarray(rock.entropy.values).copy()
    ind = np.asarray(rock.components["granite"].indicator_predicted.values).copy()
    sims = np.asarray(rock.components["granite"].simulations).copy()

    pt.to_zarr(str(tmp_path / "rock.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "rock.zarr"))
    rr = r.variables["rock"]

    assert list(rr.labels) == ["basalt", "granite"]
    # stored as codes, read back as the labels they stand for
    assert np.asarray(rr.predicted.values).dtype == np.int8
    assert rr.predicted.labels == ["basalt", "granite"]
    assert np.array_equal(rr.predicted.to_numpy(), predicted)
    assert np.array_equal(rr.measurements_a.to_numpy(), meas_a)
    assert np.asarray(rr.boundary.values).dtype == bool          # bool kept
    assert np.array_equal(np.asarray(rr.boundary.values), boundary)
    assert np.allclose(np.asarray(rr.entropy.values), entropy)
    assert np.allclose(
        np.asarray(rr.components["granite"].indicator_predicted.values), ind)
    assert np.allclose(np.asarray(rr.components["granite"].simulations), sims)


def test_binary_variable_roundtrip(tmp_path):
    pt = _point()
    meas = np.array(["ore", "waste", "ore", "waste", "ore", "waste"])
    pt.add_binary_variable("ore", labels=["waste", "ore"], measurements=meas)
    var = pt.variables["ore"]
    var.allocate_simulations(4)
    var.predicted.values[:] = _codes(var.labels, meas)
    var.latent_mean.values[:] = np.random.random(N)
    var.probability.values[:] = np.random.random(N)
    var.simulations[:, :] = np.random.random((N, 4))

    predicted = var.predicted.to_numpy().copy()
    measurements = var.measurements.to_numpy().copy()
    weights = np.asarray(var.weights.values).copy()
    mean = np.asarray(var.latent_mean.values).copy()
    sims = np.asarray(var.simulations).copy()

    pt.to_zarr(str(tmp_path / "ore.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "ore.zarr"))
    rv = r.variables["ore"]

    assert list(rv.labels) == ["waste", "ore"]
    assert np.array_equal(rv.predicted.to_numpy(), predicted)
    assert np.array_equal(rv.measurements.to_numpy(), measurements)
    assert np.allclose(np.asarray(rv.weights.values), weights)
    assert np.allclose(np.asarray(rv.latent_mean.values), mean)
    assert np.allclose(np.asarray(rv.simulations), sims)


def test_anomaly_variable_roundtrip(tmp_path):
    pt = _point()
    meas = np.array(["hit", "none", "hit", "none", "hit", "none"])
    pt.add_anomaly_variable("anom", label="hit", measurements=meas)
    var = pt.variables["anom"]
    var.allocate_simulations(2)
    # this variable only ever predicts its own two labels, whatever the
    # measurements say the other class was called
    var.predicted.values[:] = [0, 1] * (N // 2)
    var.simulations[:, :] = np.random.random((N, 2))
    predicted = var.predicted.to_numpy().copy()
    measurements = var.measurements.to_numpy().copy()
    sims = np.asarray(var.simulations).copy()

    pt.to_zarr(str(tmp_path / "anom.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "anom.zarr"))
    rv = r.variables["anom"]

    assert isinstance(rv, geoml.data.AnomalyVariable)
    assert rv.labels[0] == "hit"
    assert np.array_equal(rv.predicted.to_numpy(), predicted)
    # "none" is not one of the variable's labels, and is kept all the same
    assert np.array_equal(measurements, meas)
    assert np.array_equal(rv.measurements.to_numpy(), meas)
    assert np.allclose(np.asarray(rv.simulations), sims)


def test_rock_type_from_python_lists():
    """Regression: list measurements used to collapse the element-wise
    comparisons to scalars (boundary length 1 -> crash; silently wrong
    indicators)."""
    pt = _point()
    meas_a = ["granite", "basalt", "granite", "basalt", "granite", "basalt"]
    meas_b = ["granite", "granite", "granite", "basalt", "basalt", "basalt"]
    pt.add_rock_type_variable("rock", measurements_a=meas_a,
                              measurements_b=meas_b)
    var = pt.variables["rock"]

    boundary = np.asarray(var.boundary.values)
    assert boundary.shape == (N,)
    assert np.array_equal(
        boundary, np.array(meas_a) != np.array(meas_b))

    granite = np.asarray(var.components["granite"].indicator.values)
    expected = 0.5 * ((np.array(meas_a) == "granite") * 1.0
                      + (np.array(meas_b) == "granite") * 1.0)
    assert np.allclose(granite, expected)


def test_binary_from_python_lists():
    """Regression: list measurements used to leave the indicator all-NaN
    (scalar boolean index made the assignments no-ops)."""
    pt = _point()
    meas = ["ore", "waste", "ore", "waste", "ore", "waste"]
    pt.add_binary_variable("ore", labels=["ore", "waste"], measurements=meas)
    var = pt.variables["ore"]

    indicator = np.asarray(var.indicator.values)
    assert np.array_equal(indicator, np.array([1, 0, 1, 0, 1, 0], dtype=float))
    assert not np.any(np.isnan(np.asarray(var.weights.values)))


def test_ordered_rock_type_construction_from_strings():
    """Regression: ``-0.5 * np.ones_like(measurements_a)`` crashed on string
    measurements, making the class unconstructable in real use."""
    pt = _point()
    meas_a = ["low", "low", "mid", "mid", "high", "high"]
    meas_b = ["low", "mid", "mid", "high", "high", "low"]
    pt.add_rock_type_variable("seq", labels=["low", "mid", "high"],
                              measurements_a=meas_a, measurements_b=meas_b,
                              ordered=True)
    var = pt.variables["seq"]

    implicit = np.asarray(var.implicit_values.values)
    # interiors at i -/+ 0.5, adjacent contacts at i; non-adjacent pairs keep
    # the -0.5 default.
    assert np.allclose(implicit, [-0.5, 0.0, 0.5, 1.0, 1.5, -0.5])

    values, has_value = var.get_measurements()
    assert values.shape == (N, 1)
    assert np.all(has_value == 1.0)


def test_ordered_rock_type_roundtrip(tmp_path):
    pt = _point()
    meas_a = ["low", "low", "mid", "mid", "high", "high"]
    meas_b = ["low", "mid", "mid", "high", "high", "low"]
    pt.add_rock_type_variable("seq", labels=["low", "mid", "high"],
                              measurements_a=meas_a, measurements_b=meas_b,
                              ordered=True)
    var = pt.variables["seq"]
    var.allocate_simulations(2)
    var.predicted.values[:] = _codes(var.labels, meas_a)
    var.entropy.values[:] = np.random.random(N)
    for lb in var.labels:
        var.components[lb].simulations[:, :] = np.random.random((N, 2))

    implicit = np.asarray(var.implicit_values.values).copy()
    predicted = var.predicted.to_numpy().copy()
    boundary = np.asarray(var.boundary.values).copy()
    sims_mid = np.asarray(var.components["mid"].simulations).copy()

    pt.to_zarr(str(tmp_path / "seq.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "seq.zarr"))
    rv = r.variables["seq"]

    assert isinstance(rv, geoml.data.OrderedRockType)
    assert list(rv.labels) == ["low", "mid", "high"]
    assert np.allclose(np.asarray(rv.implicit_values.values), implicit)
    assert np.array_equal(rv.predicted.to_numpy(), predicted)
    assert np.array_equal(np.asarray(rv.boundary.values), boundary)
    assert np.allclose(np.asarray(rv.components["mid"].simulations), sims_mid)
