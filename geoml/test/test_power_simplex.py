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
"""The simplex warpings: `ScaledSimplex` and the family it heads,
`PowerSimplex`.

`PowerSimplex` maps each share by Box-Cox, ``((x/m)**a - 1)/a``, and back
by a weighted alpha-entmax, ``m [1 + a (w - tau)]_+ ** (1/a)``, the
threshold making the parts sum to one; `ScaledSimplex` is its member at
``a = 1``. Pinned here: the round trip, zeros included where they are
allowed; the way back as the maximizer it is said to be, against a general
optimizer; the limits (the softmax at 0, the chi-square projection at 1);
any number of zeros in a row; the ignored shift; the Jacobian's closed
form and the derivatives reaching the power; the refusals; a save; and a
model with them. The log-determinant against a numerical Jacobian lives in
`test_noise_integration.py`.
"""
import numpy as np
import pytest
import tensorflow as tf
from scipy.optimize import minimize

import geoml
import geoml.persistence as persistence
import geoml.warping as wp

D = 4
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.9, 1.0]


def _parts(n=300, seed=0, zeros=False):
    x = np.random.default_rng(seed).dirichlet(np.ones(D) * 2.0, size=n)
    if zeros:
        x[: n // 4, 1] = 0.0
        x[n // 8: n // 4, 3] = 0.0
        x = x / x.sum(axis=1, keepdims=True)
    return x


def _back(warping, latent):
    return np.asarray(warping.backward(tf.constant(latent)))


def _power(alpha, data=None):
    warping = wp.PowerSimplex(D, alpha=alpha)
    warping.initialize(_parts() if data is None else data)
    return warping


def _divergence(x, scale, alpha):
    """Half the Cressie-Read divergence of `x` from `scale`."""
    if alpha == 0.0:
        return np.sum(x * np.log(np.maximum(x, 1e-300) / scale))
    r = x / scale
    return np.sum(scale * (r ** (1 + alpha) - (1 + alpha) * r + alpha)
                  / (alpha * (1 + alpha)))


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_way_back_returns_the_composition(alpha):
    data = _parts()
    warping = _power(alpha, data)
    latent = np.asarray(warping.forward(tf.constant(data))[0])
    np.testing.assert_allclose(_back(warping, latent), data, atol=1e-14)


@pytest.mark.parametrize("warping", [
    lambda: wp.ScaledSimplex(D), lambda: wp.PowerSimplex(D, alpha=1.0)],
    ids=["scaled", "power-1"])
def test_zeros_come_back_exactly_at_alpha_one(warping):
    data = _parts(zeros=True)
    warping = warping()
    latent = warping.initialize(data)
    back = _back(warping, np.asarray(latent))
    np.testing.assert_allclose(back, data, atol=1e-15)
    assert np.all(back[data == 0.0] == 0.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_way_back_is_the_maximizer_on_the_simplex(alpha):
    """``argmax_x w'x - Omega(x)`` over the simplex, with `Omega` half the
    Cressie-Read divergence from the scale: a weighted alpha-entmax."""
    warping = _power(alpha)
    scale = np.asarray(warping.scale)[0]
    latent = np.random.default_rng(1).normal(size=[15, D]) * 1.5
    back = _back(warping, latent)
    for row, found in zip(latent, back):
        best = minimize(
            lambda x: _divergence(x, scale, alpha) - row @ x,
            np.full(D, 1.0 / D), method="SLSQP", bounds=[(0.0, 1.0)] * D,
            constraints={"type": "eq", "fun": lambda x: x.sum() - 1.0},
            options={"ftol": 1e-15, "maxiter": 1000})
        np.testing.assert_allclose(found, best.x, atol=1e-6)


def test_the_limits_are_the_softmax_and_the_chi_square_projection():
    data = _parts()
    latent = np.random.default_rng(2).normal(size=[200, D]) * 2.0
    soft = _power(0.0, data)
    scale = np.asarray(soft.scale)
    expected = np.exp(np.log(scale) + latent)
    np.testing.assert_allclose(
        _back(soft, latent), expected / expected.sum(axis=1, keepdims=True),
        atol=1e-15)

    # the chi-square projection of the point of unit sum the latent stands
    # for, which is `ScaledSimplex`'s way back
    linear = _power(1.0, data)
    scaled = wp.ScaledSimplex(D)
    scaled.initialize(data)
    np.testing.assert_array_equal(_back(linear, latent),
                                  _back(scaled, latent + 1.0))
    for row, found in zip(latent[:10] + 1.0, _back(scaled, latent[:10] + 1.0)):
        r = scale[0] * row + (1.0 - scale[0] @ row) * scale[0]
        best = minimize(
            lambda x: np.sum((x - r) ** 2 / scale[0]), np.full(D, 1.0 / D),
            method="SLSQP", bounds=[(0.0, 1.0)] * D,
            constraints={"type": "eq", "fun": lambda x: x.sum() - 1.0},
            options={"ftol": 1e-15, "maxiter": 1000})
        np.testing.assert_allclose(found, best.x, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_a_row_can_come_back_with_any_number_of_zeros(alpha):
    """The shrink `ScaledSimplex` used until 0.10.13 gave one zero at most;
    a threshold gives every face of the simplex some of the latent space."""
    warping = _power(alpha)
    latent = 1.0 + 1.5 * np.random.default_rng(3).normal(size=[20000, D])
    back = _back(warping, latent)
    counts = np.bincount(np.sum(back == 0.0, axis=1), minlength=D)
    assert np.all(counts > 0)
    assert np.all(back >= 0.0)
    np.testing.assert_allclose(back.sum(axis=1), 1.0, atol=1e-14)


@pytest.mark.parametrize("alpha", [1e-6, 0.02] + ALPHAS)
def test_the_parts_sum_to_one_and_ignore_a_shift(alpha):
    warping = _power(alpha)
    latent = np.random.default_rng(4).normal(size=[2000, D]) * 3.0
    back = _back(warping, latent)
    np.testing.assert_allclose(back.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(_back(warping, latent + 2.5), back,
                               atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 1.0])
def test_the_jacobian_is_a_diagonal_less_a_rank_one_term(alpha):
    """``diag(g) - g g' / sum(g)``, ``g = x**(1 - a) * m**a`` -- the
    gradient the implicit step gives the threshold, against differences."""
    warping = _power(alpha)
    scale = np.asarray(warping.scale)[0]
    latent = tf.constant(np.random.default_rng(5).normal(size=[6, D]) * 1.2)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent)
        back = warping.backward(latent)
        columns = [back[:, k] for k in range(D)]
    jacobian = np.stack([np.asarray(tape.gradient(c, latent))
                         for c in columns], axis=1)
    x = np.asarray(back)
    for i in range(x.shape[0]):
        g = np.where(x[i] > 0, x[i] ** (1 - alpha) * scale ** alpha, 0.0)
        np.testing.assert_allclose(
            jacobian[i], np.diag(g) - np.outer(g, g) / g.sum(), atol=1e-12)
        step = np.zeros([1, D])
        for j in range(D):
            step[:] = 0.0
            step[0, j] = 1e-6
            numeric = (_back(warping, np.asarray(latent)[[i]] + step)
                       - _back(warping, np.asarray(latent)[[i]] - step))
            np.testing.assert_allclose(jacobian[i][:, j], numeric[0] / 2e-6,
                                       atol=1e-7)


def test_the_derivative_reaches_the_power_both_ways():
    data = _parts()
    warping = wp.PowerSimplex(D, alpha=0.6, fixed=False)
    warping.initialize(data)
    latent = tf.constant(np.random.default_rng(6).normal(size=[20, D]))
    weights = np.arange(D, dtype=float)
    variable = warping.parameters["alpha"].variable
    with tf.GradientTape(persistent=True) as tape:
        back = tf.reduce_sum(warping.backward(latent) * weights)
        w, log_det = warping.forward(tf.constant(data))
        forth = tf.reduce_sum(w) + tf.reduce_sum(log_det)

    def at(alpha):
        other = wp.PowerSimplex(D, alpha=alpha)
        other.scale = warping.scale
        b = float(tf.reduce_sum(other.backward(latent) * weights))
        w, log_det = other.forward(tf.constant(data))
        return b, float(tf.reduce_sum(w) + tf.reduce_sum(log_det))

    up, down = at(0.6 + 1e-6), at(0.6 - 1e-6)
    # the parameter is held in its own transform: read the slope through it
    with tf.GradientTape() as inner:
        value = warping.parameters["alpha"].get_value()
    per_unit = float(inner.gradient(value, variable))
    assert float(tape.gradient(back, variable)) / per_unit == \
        pytest.approx((up[0] - down[0]) / 2e-6, rel=1e-6)
    assert float(tape.gradient(forth, variable)) / per_unit == \
        pytest.approx((up[1] - down[1]) / 2e-6, rel=1e-6)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arguments", [
    {"alpha": 0.5}, {"alpha": 1.0, "fixed": False}])
def test_zeros_are_refused_where_the_power_can_be_below_one(arguments):
    warping = wp.PowerSimplex(D, **arguments)
    with pytest.raises(wp.DomainError) as raised:
        warping.initialize(_parts(zeros=True))
    assert raised.value.column == 1
    assert raised.value.count == 75
    assert "fixed alpha of 1" in str(raised.value)


def test_negative_parts_are_refused():
    data = _parts()
    data[3, 2] = -0.01
    with pytest.raises(wp.DomainError):
        wp.PowerSimplex(D).initialize(data)


def test_the_scale_and_the_power_survive_a_save(tmp_path):
    rng = np.random.default_rng(7)
    coords = rng.uniform(0.0, 100.0, (60, 2))
    point = geoml.data.PointData.from_array(coords, ["x", "y"])
    point.add_vector_variable("c", ["a", "b", "c", "d"], _parts(60, seed=8))
    inducing = geoml.data.PointData.from_array(
        rng.uniform(0.0, 100.0, (12, 2)), ["x", "y"])
    geoml.set_seed(1234)
    link = wp.PowerSimplex(D, alpha=0.5, fixed=False)
    gp = geoml.latent.BasicGP(geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(40.0)), size=D)
    model = geoml.models.VGPNetwork(
        point, "c", geoml.likelihood.Gaussian(
            wp.ChainedWarping(link, wp.ZScore(D))), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=5)
    assert np.isfinite(model.training_log).all()
    assert float(link.parameters["alpha"].get_value()) != 0.5

    query = geoml.data.PointData.from_array(
        rng.uniform(0.0, 100.0, (15, 2)), ["x", "y"])
    model.predict(query, n_sim=4)
    before = np.asarray(query.values("c/a/prediction"), dtype=float).copy()
    assert np.isfinite(before).all()
    sims = np.stack([np.asarray(query.values(f"c/{k}/simulations"))
                     for k in "abcd"], axis=2)
    np.testing.assert_allclose(sims.sum(axis=2), 1.0, atol=1e-12)

    persistence.save_model(model, tmp_path / "model")
    restored = persistence.load_model(tmp_path / "model")
    again = restored.likelihoods[0].warping.warpings[0]
    np.testing.assert_array_equal(np.asarray(again.scale),
                                  np.asarray(link.scale))
    restored.predict(query, n_sim=4)
    np.testing.assert_allclose(
        np.asarray(query.values("c/a/prediction")), before, rtol=1e-10)


def _simulated(alpha, n=500, seed=0):
    """Compositions drawn from the family: a smooth field per part, noise,
    and the way back at `alpha`; rows with a zero left out, which the
    usual likelihood cannot take below one."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, (n, 2))
    d2 = ((coords[:, None] - coords[None]) ** 2).sum(-1)
    field = np.linalg.cholesky(np.exp(-d2 / (2 * 25.0 ** 2))
                               + 1e-8 * np.eye(n)) @ rng.normal(size=(n, D))
    truth = wp.PowerSimplex(D, alpha=alpha)
    truth.scale = tf.constant([[0.4, 0.3, 0.2, 0.1]], tf.float64)
    x = _back(truth, field + 0.3 * rng.normal(size=(n, D)))
    keep = np.all(x > 0.0, axis=1)
    return coords[keep], x[keep]


def _trained_alpha(alpha):
    coords, x = _simulated(alpha)
    geoml.set_seed(1)
    point = geoml.data.PointData.from_array(coords, ["x", "y"])
    point.add_vector_variable("c", ["a", "b", "c", "d"], x)
    link = wp.PowerSimplex(D, alpha=1.0, fixed=False)
    gp = geoml.latent.BasicGP(geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(point, 60, seed=0),
        transform=geoml.transform.Isotropic(30.0)), size=D)
    model = geoml.models.VGPNetwork(
        point, "c", geoml.likelihood.Gaussian(
            wp.ChainedWarping(link, wp.ZScore(D))), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=600)
    return float(link.parameters["alpha"].get_value())


def test_a_trained_power_moves_the_right_way_but_not_all_the_way():
    """Why the power is fixed by default. Under the usual likelihood it
    leaves 1 toward the truth and keeps the order, but settles toward the
    middle: 0.44, 0.58 and 0.89 for true powers of 0.25, 0.5 and 1 after
    3000 iterations (0.40 and 0.84 after the 600 here, for 0.25 and 1)."""
    low, high = _trained_alpha(0.25), _trained_alpha(1.0)
    assert low < 0.5 < high < 0.95
