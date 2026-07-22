"""Persistence + sim-collapse round-trips for the non-continuous variable types.

Each variable is built on a small PointData, its attributes are populated
directly (as ``update`` would), then the container is written with ``to_zarr``
and reloaded. This exercises the collapsed (n_data, n_sim) simulation store, the
string (object) and bool attribute encodings, and component recursion, without
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
    rock.predicted.values[:] = np.array(meas, dtype=object)   # string labels
    rock.entropy.values[:] = np.random.random(N)
    for lb in rock.labels:
        c = rock.components[lb]
        c.indicator_predicted.values[:] = np.random.random(N)
        c.simulations[:, :] = np.random.random((N, 3))

    predicted = np.asarray(rock.predicted.values).copy()
    meas_a = np.asarray(rock.measurements_a.values).copy()
    boundary = np.asarray(rock.boundary.values).copy()
    entropy = np.asarray(rock.entropy.values).copy()
    ind = np.asarray(rock.components["granite"].indicator_predicted.values).copy()
    sims = np.asarray(rock.components["granite"].simulations).copy()

    pt.to_zarr(str(tmp_path / "rock.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "rock.zarr"))
    rr = r.variables["rock"]

    assert list(rr.labels) == ["basalt", "granite"]
    assert np.asarray(rr.predicted.values).dtype == object      # strings kept
    assert np.array_equal(np.asarray(rr.predicted.values), predicted)
    assert np.array_equal(np.asarray(rr.measurements_a.values), meas_a)
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
    var.predicted.values[:] = np.array(meas, dtype=object)
    var.latent_mean.values[:] = np.random.random(N)
    var.probability.values[:] = np.random.random(N)
    var.simulations[:, :] = np.random.random((N, 4))

    predicted = np.asarray(var.predicted.values).copy()
    measurements = np.asarray(var.measurements.values).copy()
    weights = np.asarray(var.weights.values).copy()
    mean = np.asarray(var.latent_mean.values).copy()
    sims = np.asarray(var.simulations).copy()

    pt.to_zarr(str(tmp_path / "ore.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "ore.zarr"))
    rv = r.variables["ore"]

    assert list(rv.labels) == ["waste", "ore"]
    assert np.array_equal(np.asarray(rv.predicted.values), predicted)
    assert np.array_equal(np.asarray(rv.measurements.values), measurements)
    assert np.allclose(np.asarray(rv.weights.values), weights)
    assert np.allclose(np.asarray(rv.latent_mean.values), mean)
    assert np.allclose(np.asarray(rv.simulations), sims)


def test_anomaly_variable_roundtrip(tmp_path):
    pt = _point()
    meas = np.array(["hit", "none", "hit", "none", "hit", "none"])
    pt.add_anomaly_variable("anom", label="hit", measurements=meas)
    var = pt.variables["anom"]
    var.allocate_simulations(2)
    var.predicted.values[:] = np.array(meas, dtype=object)
    var.simulations[:, :] = np.random.random((N, 2))
    predicted = np.asarray(var.predicted.values).copy()
    sims = np.asarray(var.simulations).copy()

    pt.to_zarr(str(tmp_path / "anom.zarr"))
    r = geoml.data.PointData.open(str(tmp_path / "anom.zarr"))
    rv = r.variables["anom"]

    assert isinstance(rv, geoml.data.AnomalyVariable)
    assert rv.labels[0] == "hit"
    assert np.array_equal(np.asarray(rv.predicted.values), predicted)
    assert np.allclose(np.asarray(rv.simulations), sims)


# Note: OrderedRockType is intentionally not supported by to_zarr/open (its
# constructor needs measurements to rebuild the implicit values). It also can't
# be constructed from string measurements in the current code base
# (`np.ones_like` on a string array), so it isn't exercised here. The
# unsupported-type guard is covered in test_persistence.py.
