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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Linear tails on the log-like links and the spline, and the warning.

A warping with `linear_tails` must be the same map wherever the data it was
initialized on lie -- so training is unchanged -- and beyond them a straight
continuation along its tangent: value and derivative continuous, inverted
exactly, finite gradients however far a draw lands. The warning before
training names a chain whose way back carries a latent draw far past the
data, and keeps quiet once the tails are linear or the model was only
opened.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import geoml
import geoml.likelihood as lk
import geoml.warping as wp


def _skewed(n=300, size=2, seed=0, zeros=False):
    rng = np.random.default_rng(seed)
    x = np.exp(rng.normal(0.0, 1.0, (n, size)))
    if zeros:
        x[:10, 0] = 0.0
    return x


def _spline_data(n=300, size=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, size)) * 1.5 + rng.gamma(2.0, 1.0, (n, size))


def _built(kind, linear):
    if kind == "log":
        return wp.Log(2, linear_tails=linear), _skewed()
    if kind == "boxcox":
        return wp.BoxCox(2, linear_tails=linear), _skewed()
    if kind == "boxcox-zeros":
        return wp.BoxCox(2, shift=None, linear_tails=linear), \
            _skewed(zeros=True)
    backbone = "rq" if kind == "spline-rq" else "cubic"
    return wp.Spline(2, backbone=backbone, linear_tails=linear), \
        _spline_data()


KINDS = ["log", "boxcox", "boxcox-zeros", "spline", "spline-rq"]


def _initialized(kind, linear):
    geoml.set_seed(3)
    warping, x = _built(kind, linear)
    warping.initialize(x)
    if kind.startswith("boxcox"):
        # an exponent at the logarithm, the case that overshoots
        warping.parameters["exponent"].set_value(np.array([0.0, 0.3]))
    return warping, x


@pytest.mark.parametrize("kind", KINDS)
def test_the_data_see_the_same_map(kind):
    plain, x = _initialized(kind, False)
    linear, _ = _initialized(kind, True)
    a, da = plain.forward(tf.constant(x))
    b, db = linear.forward(tf.constant(x))
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(da), np.asarray(db))


def _beyond(kind, x):
    """Points past the data on each side the tails straighten."""
    hi = x.max(axis=0)
    out = [hi + 0.1, hi * 3 + 5, hi * 1e3 + 100]
    if kind.startswith("spline"):
        lo = x.min(axis=0)
        out += [lo - 0.1, lo * 3 - 5, lo * 1e3 - 100]
    return np.array(out)


@pytest.mark.parametrize("kind", KINDS)
def test_beyond_the_data_the_map_is_its_tangent(kind):
    warping, x = _initialized(kind, True)
    lo, hi = x.min(axis=0), x.max(axis=0)
    h = 1e-9

    def slope(v):
        # elementwise, so the gradient of the sum is each column's slope
        v = tf.constant(v)
        with tf.GradientTape() as tape:
            tape.watch(v)
            out = tf.reduce_sum(warping.forward(v)[0])
        return np.asarray(tape.gradient(out, v))

    edges = [(hi, 1.0)] + ([(lo, -1.0)] if kind.startswith("spline") else [])
    for edge, side in edges:
        edge = edge[None, :]
        f = lambda v: np.asarray(warping.forward(tf.constant(v))[0])
        value = f(edge)
        inner, outer = slope(edge - side * h), slope(edge + side * h)
        # the derivative is continuous across the data's limit
        np.testing.assert_allclose(inner, outer, rtol=1e-6)
        # and beyond it the map is straight
        far = f(edge + side * 50.0)
        np.testing.assert_allclose(far, value + side * 50.0 * outer,
                                   rtol=1e-6)
        # the log-determinant is the tangent's
        log_det = np.asarray(
            warping.forward(tf.constant(edge + side * 50.0))[1])
        np.testing.assert_allclose(log_det, np.log(outer).sum(), rtol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_the_way_back_inverts_the_tails(kind):
    warping, x = _initialized(kind, True)
    v = _beyond(kind, x)
    warped = warping.forward(tf.constant(v))[0]
    back = np.asarray(warping.backward(warped))
    if kind == "log":
        # `Log.backward` is `exp` and leaves the shift in, as it always has
        back = back - warping.shift
    np.testing.assert_allclose(back, v, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kind", ["log", "boxcox", "spline"])
def test_a_draw_far_out_comes_back_linearly(kind):
    plain, x = _initialized(kind, False)
    linear, _ = _initialized(kind, True)
    z = np.asarray(linear.forward(tf.constant(x.max(axis=0)[None, :]))[0])
    far = z + np.array([[10.0], [20.0], [40.0]])
    a = np.asarray(linear.backward(tf.constant(far)))
    steps = np.diff(a, axis=0)
    # equal latent steps, equal steps in the data
    np.testing.assert_allclose(steps[1], 2 * steps[0], rtol=1e-8)
    if kind != "spline":
        # where the plain map is a power or an exponential, far lower
        b = np.asarray(plain.backward(tf.constant(far)))
        assert np.all(a[-1] < b[-1])


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_stay_finite_far_beyond(kind):
    warping, x = _initialized(kind, True)
    z = np.asarray(warping.forward(tf.constant(x))[0])
    far = tf.constant(np.concatenate([z, z.max(axis=0)[None] + 1e4,
                                      z.min(axis=0)[None] - 1e4]))
    variables = [p.variable for p in warping._all_parameters]
    with tf.GradientTape() as tape:
        out = tf.reduce_sum(warping.backward(far)) \
            + tf.reduce_sum(warping.forward(tf.constant(
                np.concatenate([x, x.max(axis=0)[None] * 1e4])))[0])
    grads = tape.gradient(out, variables)
    assert np.isfinite(np.asarray(out))
    for g in grads:
        if g is not None:
            assert np.all(np.isfinite(np.asarray(g)))


def test_without_the_option_nothing_is_kept():
    warping, _ = _initialized("boxcox", False)
    assert warping.upper is None
    spline, _ = _initialized("spline", False)
    assert spline.lower is None and spline.upper is None


# --------------------------------------------------------------------------- #
# through a model
# --------------------------------------------------------------------------- #
COLS = ["X", "Y"]


def _model(chain, seed=5, n=120):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    point.add_continuous_variable(
        "v", np.exp(2.5 * np.sin(coords[:, 0] / 20.0)
                    + 1.0 * rng.standard_normal(n)))
    geoml.set_seed(seed)
    inducing = geoml.data.PointData(
        pd.DataFrame(rng.uniform(0, 100, (8, 2)), columns=COLS), COLS)
    gp = geoml.latent.BasicGP(geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(40)), size=1)
    return geoml.models.VGPNetwork(
        point, "v", lk.Gaussian(chain), gp,
        options=geoml.models.GPOptions(verbose=False))


def _caught(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [str(w.message) for w in caught
            if "carries a latent value" in str(w.message)]


def _log_chain(linear):
    return wp.ChainedWarping(wp.Log(1, linear_tails=linear), wp.ZScore(1))


def test_a_log_link_is_warned_of_before_training():
    model = _model(_log_chain(False))
    found = _caught(lambda: model.train_full(2))
    assert len(found) == 1 and "Variable 'v'" in found[0]
    assert "linear_tails=True" in found[0]
    # once per model
    assert not _caught(lambda: model.train_full(2))


def test_linear_tails_silence_the_warning():
    model = _model(_log_chain(True))
    assert not _caught(lambda: model.train_full(2))


def test_a_tame_chain_is_not_warned_of():
    # a square root: the way back is a square, which grows with the draw
    # as the data's own tail does
    link = wp.BoxCox(1, exponent=0.5)
    link.parameters["exponent"].fix()
    model = _model(wp.ChainedWarping(link, wp.ZScore(1)))
    assert not _caught(lambda: model.train_full(2))


def test_the_limits_survive_a_save_and_opening_warns_of_nothing(tmp_path):
    chain = wp.ChainedWarping(wp.Log(1, linear_tails=True), wp.ZScore(1),
                              wp.Spline(1, linear_tails=True))
    model = _model(chain)
    model.train_full(3)
    path = str(tmp_path / "model.zarr")
    model.save(path)
    log, _, spline = model.likelihoods[0].warping.warpings
    far = tf.constant([[-30.0], [30.0]], tf.float64)
    before = np.asarray(model.likelihoods[0].warping.backward(far))

    found = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = geoml.models.VGPNetwork.open(path)
    found = [w for w in caught if "carries a latent value" in str(w.message)]
    assert not found
    log2, _, spline2 = loaded.likelihoods[0].warping.warpings
    assert np.array_equal(np.asarray(log.upper), np.asarray(log2.upper))
    assert np.array_equal(np.asarray(spline.lower), np.asarray(spline2.lower))
    assert np.array_equal(np.asarray(spline.upper), np.asarray(spline2.upper))
    after = np.asarray(loaded.likelihoods[0].warping.backward(far))
    np.testing.assert_allclose(after, before, rtol=1e-12)


def test_an_opened_log_model_is_not_warned_until_it_trains(tmp_path):
    model = _model(_log_chain(False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train_full(2)
    path = str(tmp_path / "model.zarr")
    model.save(path)
    found = _caught(lambda: geoml.models.VGPNetwork.open(path))
    assert not found
