"""The derived variable: computed from others, realization by realization.

``container.derive(names, function, arguments)`` walks the parents'
simulations in bands, applies the function once per realization, and stores
the results as ``DerivedVariable``s -- full variables (quantiles, cut-offs,
paths, exports) whose uncertainty is entirely inherited. These tests pin the
arithmetic, the realization-wise contract (unsimulated variables refused,
metadata as per-location constants, the optional ``simulation=`` index), the
banded walk, the model-facing refusals, and the Zarr round trip -- values and
ancestry survive; the function stays in the script that ran ``derive``.
"""
import numpy as np
import pytest

import geoml
import geoml.storage as storage


def _point(n=40, n_sim=6, seed=0):
    rng = np.random.default_rng(seed)
    point = geoml.data.PointData.from_array(rng.uniform(0, 10, (n, 2)))
    for name, scale in (("zn", 3.0), ("pb", 1.0)):
        point.add_continuous_variable(name)
        var = point.variables[name]
        var.allocate_simulations(n_sim)
        var.simulations[:, :] = rng.lognormal(size=(n, n_sim)) * scale
        var.prediction.values[:] = np.asarray(var.simulations).mean(axis=1)
    point.add_metadata("density", rng.uniform(2.5, 3.0, n))
    return point


def test_the_function_is_applied_per_realization():
    point = _point()
    nsr = point.derive("nsr", lambda zn, pb: 40.0 * zn + 15.0 * pb,
                       ["zn", "pb"])

    zn = np.asarray(point.variables["zn"].simulations)
    pb = np.asarray(point.variables["pb"].simulations)
    expected = 40.0 * zn + 15.0 * pb

    assert isinstance(nsr, geoml.data.DerivedVariable)
    assert point.variables["nsr"] is nsr
    assert nsr.parents == ["zn", "pb"]
    assert np.allclose(np.asarray(nsr.simulations), expected)
    # the prediction is the mean of the derived realizations, not the
    # function of the parents' predictions -- the nonlinear case is the
    # reason the class exists
    assert np.allclose(nsr.prediction.values.to_numpy(),
                       expected.mean(axis=1))


def test_one_variable_per_output_name():
    point = _point()
    revenue, contained = point.derive(
        ["revenue", "contained"],
        lambda zn, rho: (40.0 * zn * rho, zn * rho),
        ["zn", "_metadata/density"])

    zn = np.asarray(point.variables["zn"].simulations)
    rho = point.get_metadata("density")[:, None]
    assert isinstance(revenue, geoml.data.DerivedVariable)
    assert isinstance(contained, geoml.data.DerivedVariable)
    assert np.allclose(np.asarray(revenue.simulations), 40.0 * zn * rho)
    assert np.allclose(np.asarray(contained.simulations), zn * rho)


def test_metadata_comes_in_as_a_constant():
    point = _point()
    contained = point.derive("contained", lambda zn, rho: zn * rho,
                             ["zn", "_metadata/density"])
    zn = np.asarray(point.variables["zn"].simulations)
    rho = point.get_metadata("density")
    assert np.allclose(np.asarray(contained.simulations), zn * rho[:, None])


def test_an_unsimulated_variable_is_refused():
    point = _point()
    point.add_continuous_variable("rho")
    with pytest.raises(ValueError, match="no simulations"):
        point.derive("nsr", lambda zn, rho: zn * rho, ["zn", "rho"])


def test_realization_counts_must_agree():
    point = _point()
    point.add_continuous_variable("extra")
    var = point.variables["extra"]
    var.allocate_simulations(3)
    var.simulations[:, :] = 1.0
    with pytest.raises(ValueError, match="in step"):
        point.derive("nsr", lambda a, b: a + b, ["zn", "extra"])


def test_metadata_alone_has_nothing_to_walk():
    point = _point()
    with pytest.raises(ValueError, match="at least one simulated"):
        point.derive("x", lambda rho: rho, ["_metadata/density"])


def test_the_function_may_ask_which_realization():
    """A quantity drawn per realization -- a price scenario, say -- needs to
    know which realization it is in; a keyword named `simulation` receives
    the index, and a function without it never sees one."""
    point = _point()
    prices = np.linspace(30.0, 50.0, point.variables["zn"].n_sim)
    nsr = point.derive("nsr",
                       lambda zn, simulation: zn * prices[simulation],
                       ["zn"])
    zn = np.asarray(point.variables["zn"].simulations)
    assert np.allclose(np.asarray(nsr.simulations), zn * prices[None, :])


def test_wrong_output_count_is_refused():
    point = _point()
    with pytest.raises(ValueError, match="output"):
        point.derive(["a", "b"], lambda zn: zn, ["zn"])


def test_component_paths_resolve():
    point = _point()
    rng = np.random.default_rng(7)
    point.add_vector_variable("metals", labels=["cu", "au"])
    for label in ("cu", "au"):
        component = point.variables["metals"].components[label]
        component.allocate_simulations(6)
        component.simulations[:, :] = rng.lognormal(size=(40, 6))

    value = point.derive("value", lambda cu, au: cu + 60.0 * au,
                         ["metals/cu", "metals/au"])
    cu = np.asarray(point.variables["metals"].components["cu"].simulations)
    au = np.asarray(point.variables["metals"].components["au"].simulations)
    assert np.allclose(np.asarray(value.simulations), cu + 60.0 * au)


def test_derive_walks_in_bands(monkeypatch):
    """On a disk-backed model the parents are read and the results written a
    band at a time -- no store is ever held whole."""
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    monkeypatch.setattr(storage, "_TARGET_CHUNK_BYTES", 256)  # several bands
    point = _point()
    assert point.variables["zn"].simulations.backend == "zarr"
    assert len(point.variables["zn"].simulations.row_bands()) > 1

    def explode(self, *args, **kwargs):
        if len(self.shape) == 2:
            raise AssertionError("a whole simulations store was read")
        return np.asarray(self._array)

    monkeypatch.setattr(storage.ArrayStore, "__array__", explode)
    nsr = point.derive("nsr", lambda zn, pb: zn + pb, ["zn", "pb"])

    zn = np.asarray(point.variables["zn"].simulations._array)
    pb = np.asarray(point.variables["pb"].simulations._array)
    assert np.allclose(np.asarray(nsr.simulations._array), zn + pb)
    assert np.allclose(np.asarray(nsr.prediction.values._array),
                       (zn + pb).mean(axis=1))


def test_a_derived_variable_is_not_for_models():
    point = _point()
    nsr = point.derive("nsr", lambda zn: 2.0 * zn, ["zn"])
    with pytest.raises(TypeError, match="cannot train"):
        nsr.training_input()
    with pytest.raises(TypeError, match="no measurements"):
        nsr.get_measurements()
    with pytest.raises(TypeError, match="predict into"):
        nsr.update([0])


def test_everything_a_continuous_variable_has_comes_free():
    point = _point()
    nsr = point.derive("nsr", lambda zn, pb: 40.0 * zn + 15.0 * pb,
                       ["zn", "pb"])
    nsr.set_cutoffs([50.0])
    nsr.reset_quantiles([0.1, 0.9])

    assert point.get("nsr") is nsr
    assert point.get("nsr/quantiles/0.1") is nsr.quantiles[0.1]
    assert "nsr" in point.tree(status=False)
    frame = point.as_data_frame()
    assert any(name.startswith("nsr") for name in frame.columns)


def test_subsetting_keeps_the_derivation():
    point = _point()
    nsr = point.derive("nsr", lambda zn: 2.0 * zn, ["zn"])
    kept = point[np.arange(0, 40, 2)]
    sub = kept.variables["nsr"]

    assert type(sub) is geoml.data.DerivedVariable
    assert sub.parents == ["zn"]
    assert np.allclose(np.asarray(sub.simulations),
                       np.asarray(nsr.simulations)[::2])


def test_deriving_again_replaces_the_variable():
    point = _point()
    point.derive("nsr", lambda zn: zn, ["zn"])
    second = point.derive("nsr", lambda zn: 2.0 * zn, ["zn"])
    assert point.variables["nsr"] is second
    assert np.allclose(np.asarray(second.simulations),
                       2.0 * np.asarray(point.variables["zn"].simulations))


def test_the_round_trip_keeps_the_values_and_the_ancestry(tmp_path):
    point = _point()
    nsr = point.derive("nsr", lambda zn, pb: 40.0 * zn + 15.0 * pb,
                       ["zn", "pb"])
    nsr.set_cutoffs([60.0])
    sims = np.asarray(nsr.simulations).copy()

    path = str(tmp_path / "derived.zarr")
    point.to_zarr(path)
    reloaded = geoml.data.PointData.open(path).variables["nsr"]

    assert type(reloaded) is geoml.data.DerivedVariable
    assert reloaded.parents == ["zn", "pb"]
    assert reloaded.cutoffs == [60.0]
    assert np.allclose(np.asarray(reloaded.simulations), sims)
    assert np.allclose(np.asarray(reloaded.prediction.values),
                       np.asarray(nsr.prediction.values))
