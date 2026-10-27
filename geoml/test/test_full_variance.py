# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""`GPOptions(full_variance=)`: on for a new model, off for an older save.

On, a deep model's realizations carry the variance its moments report; off
they carry what 0.9.3's did, the explained part, and every expert draws
under the node's own name. Bit-for-bit agreement with 0.9.3 itself needs
the 0.9.3 tree beside this one and is checked by
`docs/benchmarks/deep_second_moment/gate_off.py`.
"""
import warnings

import numpy as np
import pytest

import geoml
import geoml.latent.network as _net


def _deep_model(full_variance=True):
    geoml.set_seed(7)
    x = np.random.default_rng(3).uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 20) + 0.3 * np.cos(x[:, 1] / 15))
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 24, seed=0),
        geoml.transform.Isotropic(40))
    leaf = geoml.latent.BasicGP(geoml.latent.BasicGP(root, size=2), size=1)
    return geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.Identity(1)),
        leaf,
        options=geoml.models.GPOptions(
            verbose=False, propagation="joint",
            expert_propagation="independent", full_variance=full_variance))


def _variance_ratio(model, n_sim=1000):
    """The realizations' variance over the reported latent variance, the
    median over 30 locations."""
    x = np.random.default_rng(5).uniform(0, 100, [30, 2])
    target = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(target, n_sim=n_sim)
    sims = np.asarray(target.variables["v"].get_simulations(), dtype=float)
    reported = np.asarray(target.values("v/latent_variance"), dtype=float)
    return float(np.median(np.var(sims, axis=1) / reported.ravel()))


def test_a_new_model_takes_it_and_an_older_save_does_not(tmp_path):
    model = _deep_model()
    assert model.options.full_variance and model._full_variance()
    # a save made before 0.9.4 records no such option
    del model.options.__dict__["full_variance"]
    model.save(tmp_path / "old")
    reopened = geoml.persistence.load_model(tmp_path / "old")
    assert not reopened._full_variance()
    assert reopened._propagation_key() != _deep_model()._propagation_key()


def test_off_every_expert_draws_under_the_node_name():
    with _net.propagation_rule("independent", joint=True, full=False):
        assert _net._expert_key("gp", 2) == "gp"
    with _net.propagation_rule("independent", joint=True, full=True):
        assert _net._expert_key("gp", 2) == "gp/expert2"
        assert _net._expert_key("gp", 0) == "gp"


def test_the_realizations_carry_the_reported_variance_only_when_on():
    on, off = _deep_model(True), _deep_model(False)
    for model in (on, off):
        model.train_full(30)
    ratio_on, ratio_off = _variance_ratio(on), _variance_ratio(off)
    print("variance ratio: on %.3f, off %.3f" % (ratio_on, ratio_off))
    assert abs(ratio_on - 1.0) < 0.1
    assert ratio_off < 0.8


def test_a_deep_model_sizes_its_prediction_batches_to_the_budget():
    model = _deep_model(True)
    assert _deep_model(False)._prediction_rows() == 20000
    model.options.quadrature_memory = 1e7
    # four realizations of the reader's pathwise draw: 24 inducing points
    # in two dimensions, 1024 features and the constant, 48 bytes each
    rows = model._prediction_rows()
    assert rows == int(1e7 // (4 * 48 * (24 * 2 + 1025)))
    x = np.random.default_rng(5).uniform(0, 100, [120, 2])
    small = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(small, n_sim=4)
    model.options.quadrature_memory = 1e12
    whole = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(whole, n_sim=4)
    for path in ("v/prediction", "v/latent_variance"):
        np.testing.assert_allclose(small.values(path), whole.values(path),
                                   rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        small.variables["v"].get_simulations(),
        whole.variables["v"].get_simulations(), rtol=1e-8, atol=1e-10)


def _shallow_model(kernel, full_variance=True):
    x = np.random.default_rng(3).uniform(0, 100, [40, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", np.sin(x[:, 0] / 20))
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 12, seed=0),
        geoml.transform.Isotropic(40))
    gp = geoml.latent.BasicGP(root, size=1,
                              kernel=getattr(geoml.kernels, kernel)())
    return geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False,
                                       full_variance=full_variance))


@pytest.mark.parametrize("kernel", ["Spherical", "Cubic"])
def test_a_kernel_without_random_features_is_warned_of(kernel):
    with pytest.warns(UserWarning, match="only the variance its inducing"):
        _shallow_model(kernel)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _shallow_model(kernel, full_variance=False)
        _shallow_model("Exponential")


def test_the_quadrature_compiles_with_xla_on_the_cpu_only(monkeypatch):
    # an XLA-compiled chunk crashed the CUDA driver on the GPU: a GPU in
    # sight turns the compilation off, and the sums stay the same
    import tensorflow as tf
    compiled = []
    function = tf.function

    def spy(fn=None, **kwargs):
        compiled.append(kwargs.get("jit_compile"))
        return function(fn, **kwargs)

    def chunk(c, x):
        return (tf.reduce_sum(x * tf.cast(c + 1, tf.float64)),)

    x = tf.constant(np.arange(6.0))
    sums = {}
    for gpu in (False, True):
        monkeypatch.setattr(_net, "_gpu_visible", lambda gpu=gpu: gpu)
        monkeypatch.setattr(tf, "function", spy)
        compiled.clear()
        sums[gpu] = float(_net._chunked_sums(chunk, [x], 3)[0])
        monkeypatch.setattr(tf, "function", function)
        assert _net._quadrature_xla() is (not gpu)
        assert compiled == ([] if gpu else [True])
    assert sums[False] == sums[True] == 15.0 * 6


def test_the_default_budget_comes_from_the_device_s_memory(monkeypatch):
    # a new model's budget: a fifth of the device's memory, read as a
    # power of two, shared among the GP nodes taking the quadrature, never
    # more than the fixed 2e9; a number in the options wins, and an older
    # save, whose options lack the attribute, keeps 2e9
    assert geoml.models.GPOptions().quadrature_memory is None
    device = _net._device_bytes()
    assert device is None or float(np.log2(device)).is_integer()

    model = _deep_model(True)
    model.options.quadrature_memory = None
    assert sum(1 for n in model._nodes() if hasattr(n, "_takes_quadrature")
               and n._takes_quadrature()) == 1
    for size, budget in ((2.0 ** 34, 2e9), (2.0 ** 33, 0.2 * 2.0 ** 33),
                         (None, 2e9)):
        monkeypatch.setattr(_net, "_device_bytes", lambda size=size: size)
        model.__dict__.pop("_automatic_quadrature_memory", None)
        assert model._quadrature_memory() == budget
    monkeypatch.setattr(_net, "_device_bytes", lambda: 2.0 ** 34)
    assert _net._automatic_budget(3) == 0.2 * 2.0 ** 34 / 3
    model.options.quadrature_memory = 5e8
    assert model._quadrature_memory() == 5e8
    del model.options.quadrature_memory
    assert model._quadrature_memory() == 2e9


def _batches_seen(model, monkeypatch, n=100):
    """The locations of every batch a prediction of `n` points makes, and the
    prediction."""
    x = np.random.default_rng(5).uniform(0, 100, [n, 2])
    target = geoml.data.PointData.from_array(x, ["X", "Y"])
    sizes = []
    batched = target.get_batched_coordinates

    def recorded(rows):
        sizes.append(len(rows))
        return batched(rows)

    monkeypatch.setattr(target, "get_batched_coordinates", recorded)
    model.predict(target, n_sim=4)
    return sizes, target


def test_a_prediction_on_a_gpu_measures_its_batches(monkeypatch):
    # GeoScape's item 45: the estimate counts what the GP nodes hold and
    # nothing else, so on a GPU the first batches take a quarter of it, the
    # second and third are measured, and the rest take the rows that fit 0.7
    # of the device's limit -- never more than the estimate where a node
    # chunks by the rows, as here; the answer is the same as with batches of
    # any other size
    import tensorflow as tf
    model = _deep_model(True)
    model.train_full(5)
    model.options.prediction_batch_size = 40
    plain, reference = _batches_seen(model, monkeypatch)
    assert plain == [40, 40, 20]

    # a row measured at 1 MB, and room for 25 of them
    monkeypatch.setattr(_net, "_device_limit", lambda: 25e6 / 0.7)
    monkeypatch.setattr(tf.config.experimental, "reset_memory_stats",
                        lambda device: None)
    monkeypatch.setattr(tf.config.experimental, "get_memory_info",
                        lambda device: {"current": 0, "peak": 1e7})
    sizes, measured = _batches_seen(model, monkeypatch)
    # the third measured too, at the size the second gave: 25 rows at 1 MB
    # held, a row then 0.4 MB and room for 62, the estimate's 40
    assert sizes == [10, 10, 25, 40, 15]
    for path in ("v/prediction", "v/latent_variance"):
        np.testing.assert_allclose(measured.values(path),
                                   reference.values(path),
                                   rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(measured.variables["v"].get_simulations(),
                               reference.variables["v"].get_simulations(),
                               rtol=1e-12, atol=1e-14)

    # a measurement that leaves more room keeps the estimate
    monkeypatch.setattr(_net, "_device_limit", lambda: 1e12)
    sizes, _ = _batches_seen(model, monkeypatch)
    assert sizes == [10, 10, 40, 40]

    # a call the estimate takes whole is measured all the same -- each group
    # of a prediction by expert is one, and on a GeoScape Run every group
    # ran out of memory as a single batch -- and one a probe batch holds is
    # left whole
    monkeypatch.setattr(_net, "_device_limit", lambda: 25e6 / 0.7)
    model.options.prediction_batch_size = 120
    sizes, measured = _batches_seen(model, monkeypatch)
    assert sizes == [30, 30, 40]
    np.testing.assert_allclose(measured.values("v/prediction"),
                               reference.values("v/prediction"),
                               rtol=1e-12, atol=1e-14)
    model.options.prediction_batch_size = 400
    sizes, _ = _batches_seen(model, monkeypatch)
    assert sizes == [100]


def test_a_measured_batch_grows_past_the_estimate_where_nothing_chunks(
        monkeypatch):
    """A measured batch may take more rows than the estimate, up to the
    options' batch size, unless a node sizes a quadrature chunk by the rows a
    call states -- the GP on a GP here, not a GP on the input. The estimate
    cannot see what XLA saves."""
    import tensorflow as tf
    # an estimate of 20 rows, a row measured at 0.5 MB and room for 50
    monkeypatch.setattr(_net, "_device_limit", lambda: 25e6 / 0.7)
    monkeypatch.setattr(tf.config.experimental, "reset_memory_stats",
                        lambda device: None)
    monkeypatch.setattr(tf.config.experimental, "get_memory_info",
                        lambda device: {"current": 0, "peak": 2.5e6})
    for model, expected in ((_input_gp_model(), [5, 5, 50, 40]),
                            (_deep_model(True), [5, 5, 20, 20, 20, 20, 10])):
        model.train_full(5)
        model.options.prediction_batch_size = 1000
        monkeypatch.setattr(model, "_prediction_rows",
                            lambda one_expert=False: 20)
        sizes, _ = _batches_seen(model, monkeypatch)
        assert sizes == expected


def _input_gp_model():
    """A GP on the input, which takes no quadrature."""
    geoml.set_seed(7)
    x = np.random.default_rng(3).uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 20) + 0.3 * np.cos(x[:, 1] / 15))
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 24, seed=0),
        geoml.transform.Isotropic(40))
    return geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.Identity(1)),
        geoml.latent.BasicGP(root, size=1),
        options=geoml.models.GPOptions(verbose=False))


def test_a_batch_out_of_memory_raises_with_its_size(monkeypatch):
    import tensorflow as tf
    model = _deep_model(True)

    def exhausted(*args, **kwargs):
        raise tf.errors.ResourceExhaustedError(None, None, "OOM")

    monkeypatch.setattr(model, "predict_raw", exhausted)
    x = np.random.default_rng(5).uniform(0, 100, [30, 2])
    with pytest.raises(MemoryError, match="30 locations"):
        model.predict(geoml.data.PointData.from_array(x, ["X", "Y"]),
                      n_sim=4)
