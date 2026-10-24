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

"""`GPOptions(quadrature_precision=)`: the conditioned quadrature's large
arrays in single precision, everything else in double.

At fixed parameters the leaf's moments, their gradient and the
realizations move by single precision's rounding and no more, at the
origin and at mine-grid coordinates; the rational quadratic, whose
components are a grid about its trained scale, stays in double.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.latent.network as _net


def _model(network, kernel, shift=0.0):
    geoml.set_seed(7)
    x = np.random.default_rng(3).uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x + shift, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 20) + 0.3 * np.cos(x[:, 1] / 15))
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 16, seed=0),
        geoml.transform.Isotropic(40), center=True)
    hidden = geoml.latent.BasicGP(root, size=2)
    below = geoml.latent.GPWalk(hidden, n_steps=3) if network == "walk" \
        else hidden
    leaf = geoml.latent.BasicGP(below, size=1,
                                kernel=getattr(geoml.kernels, kernel)(),
                                isotropic=network == "walk")
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), leaf,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(10)
    return model, data


def _leaf(model, data, precision):
    """The leaf's mean and variance at the data, and the gradient of their
    total over every trained parameter."""
    model.options.quadrature_precision = precision
    x = tf.constant(np.asarray(data.coordinates), tf.float64)
    variables = model.get_unfixed_variables()
    with tf.GradientTape() as tape:
        with model._propagation():
            model._refresh(model.options.jitter)
            mu, var, _, _ = model.leaves[0].predict(x, n_sim=1)
        total = tf.reduce_sum(mu) + tf.reduce_sum(var)
    grads = tape.gradient(total, variables)
    return (np.asarray(mu).ravel(), np.asarray(var).ravel(),
            np.concatenate([np.ravel(g) for g in grads if g is not None]))


def test_a_new_model_takes_single_on_a_gpu_and_an_older_save_double(
        tmp_path, monkeypatch):
    assert geoml.models.GPOptions().quadrature_precision is None
    with pytest.raises(ValueError, match="quadrature_precision"):
        geoml.models.GPOptions(quadrature_precision="float16")
    model, _ = _model("dgp", "Gaussian")
    for gpu, precision in ((False, "float64"), (True, "float32")):
        monkeypatch.setattr(_net, "_gpu_visible", lambda gpu=gpu: gpu)
        assert model._quadrature_precision() == precision
    monkeypatch.setattr(_net, "_gpu_visible", lambda: True)
    model.options.quadrature_precision = "float64"
    assert model._quadrature_precision() == "float64"
    key = model._propagation_key()
    model.options.quadrature_precision = "float32"
    assert model._propagation_key() != key
    # a save made before 0.9.8 records no such option
    del model.options.__dict__["quadrature_precision"]
    assert model._quadrature_precision() == "float64"
    assert model._propagation_key() == key
    model.save(tmp_path / "old")
    reopened = geoml.persistence.load_model(tmp_path / "old")
    assert reopened._quadrature_precision() == "float64"


@pytest.mark.parametrize("shift", [0.0, 1e6])
@pytest.mark.parametrize("kernel", ["Gaussian", "Exponential"])
@pytest.mark.parametrize("network", ["walk", "dgp"])
def test_single_precision_moves_the_moments_by_its_rounding(network, kernel,
                                                            shift):
    model, data = _model(network, kernel, shift)
    m64, v64, g64 = _leaf(model, data, "float64")
    m32, v32, g32 = _leaf(model, data, "float32")
    assert np.all(np.isfinite(v32)) and np.all(np.isfinite(g32))
    np.testing.assert_allclose(m32, m64, rtol=0, atol=1e-6)
    np.testing.assert_allclose(v32, v64, rtol=1e-5)
    assert np.abs(g32 - g64).max() < 1e-4 * np.abs(g64).max()


def test_single_precision_moves_the_realizations_by_its_rounding():
    model, _ = _model("walk", "Exponential")
    x = np.random.default_rng(5).uniform(0, 100, [30, 2])
    sims = {}
    for precision in ("float64", "float32"):
        model.options.quadrature_precision = precision
        target = geoml.data.PointData.from_array(x, ["X", "Y"])
        model.predict(target, n_sim=20)
        sims[precision] = np.asarray(
            target.variables["v"].get_simulations(), float)
    spread = sims["float64"].std()
    assert np.abs(sims["float32"] - sims["float64"]).max() < 1e-4 * spread


def test_the_rational_quadratic_stays_in_double():
    model, data = _model("dgp", "RationalQuadratic")
    _, v64, g64 = _leaf(model, data, "float64")
    _, v32, g32 = _leaf(model, data, "float32")
    np.testing.assert_array_equal(v32, v64)
    np.testing.assert_array_equal(g32, g64)
    with _net.quadrature_rule(precision="float32"):
        assert _net._node_bytes(geoml.kernels.Gaussian()) == \
            _net._NODE_BYTES / 2
        assert _net._node_bytes(geoml.kernels.RationalQuadratic()) == \
            _net._NODE_BYTES
    assert _net._node_bytes(geoml.kernels.Gaussian()) == _net._NODE_BYTES
