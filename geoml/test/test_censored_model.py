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
"""`likelihood.CensoredSimplex` in a model.

The row forms themselves are `test_censored_simplex.py`'s. Pinned here:
the warping it takes and refuses; a start with zeros, which `PowerSimplex`
refuses under `Gaussian`; training on a `MassFractionVariable` with zeros
and rows missing parts, with only the rows missing too many left out;
predictions closed and holding exact zeros; a likelihood that does not
depend on how the rows are batched; and a save.
"""
import warnings

import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.likelihood as lk
import geoml.persistence as persistence
import geoml.warping as wp

D = 4


def _fractions(n, seed):
    """Compositions drawn from the family at a power of one half, with
    zeros where the latent row leaves the simplex."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, (n, 2))
    d2 = ((coords[:, None] - coords[None]) ** 2).sum(-1)
    chol = np.linalg.cholesky(np.exp(-d2 / (2 * 25.0 ** 2))
                              + 1e-6 * np.eye(n))
    latent = chol @ rng.standard_normal((n, D)) \
        + 0.6 * rng.standard_normal((n, D))
    warping = wp.PowerSimplex(D, alpha=0.5)
    warping.scale = tf.constant([[0.4, 0.3, 0.2, 0.1]], tf.float64)
    x = warping.backward(tf.constant(latent, tf.float64)).numpy()
    x[x < 1e-12] = 0.0
    return coords, x


def _model(x, coords, likelihood, variable="mass_fraction"):
    point = geoml.data.PointData.from_array(coords, ["x", "y"])
    if variable == "mass_fraction":
        point.add_mass_fraction_variable("m", list("abcd"), x)
    else:
        point.add_compositional_variable("m", list("abcd"), x)
    inducing = geoml.data.PointData.from_array(
        np.random.default_rng(1).uniform(0.0, 100.0, (12, 2)), ["x", "y"])
    geoml.set_seed(1234)
    gp = geoml.latent.BasicGP(geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(30.0)), size=D)
    return point, geoml.models.VGPNetwork(
        point, "m", likelihood, gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=8))


def _censored(alpha=0.5, **kwargs):
    return lk.CensoredSimplex(wp.ChainedWarping(
        wp.PowerSimplex(D, alpha=alpha), wp.ZScore(D)), **kwargs)


# --------------------------------------------------------------------------- #
# the warping
# --------------------------------------------------------------------------- #
def test_the_warping_starts_with_a_simplex_warping():
    lk.CensoredSimplex(wp.PowerSimplex(D))
    lk.CensoredSimplex(wp.ScaledSimplex(D))
    lk.CensoredSimplex(wp.ChainedWarping(
        wp.PowerSimplex(D), wp.SinhArcsinh(D), wp.ZScore(D)))
    with pytest.raises(TypeError, match="start with PowerSimplex"):
        lk.CensoredSimplex(wp.ZScore(D))


@pytest.mark.parametrize("link", [lambda: wp.BoxCox(D), lambda: wp.Log(D),
                                  lambda: wp.Softplus(D),
                                  lambda: wp.RobustPCA(D)])
def test_a_link_off_the_whole_line_or_mixing_is_refused(link):
    made = link()
    with pytest.raises(TypeError, match=type(made).__name__):
        lk.CensoredSimplex(wp.ChainedWarping(wp.PowerSimplex(D), made))


def test_a_power_of_zero_takes_no_zero():
    coords, x = _fractions(60, 0)
    assert (x == 0).any()
    with pytest.raises(wp.DomainError, match="power of zero"):
        _model(x, coords, _censored(alpha=0.0))


# --------------------------------------------------------------------------- #
# training and prediction
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained():
    coords, x = _fractions(80, 2)
    x[:6, 1] = np.nan
    x[6:9, 2:] = np.nan
    x[9, :3] = np.nan                     # three missing, beyond the limit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        point, model = _model(x, coords, _censored(max_hidden=2))
    model.train_full(max_iter=30)
    return point, model, x, [str(w.message) for w in caught]


def test_only_the_rows_missing_too_many_parts_are_left_out(trained):
    _, model, x, messages = trained
    assert any("1 row(s) cannot be scored" in m and "more than 2" in m
               for m in messages)
    assert (model.has_value[9] == 0).all()
    np.testing.assert_array_equal(model.has_value[0], [1, 0, 1, 1])
    np.testing.assert_array_equal(model.has_value[7], [1, 1, 0, 0])


def test_it_trains_and_the_bound_rises(trained):
    _, model, _, _ = trained
    log = np.asarray(model.training_log)
    assert np.isfinite(log).all()
    assert log[-1] > log[0]


def test_predictions_are_closed_and_hold_exact_zeros(trained):
    _, model, _, _ = trained
    query = geoml.data.PointData.from_array(
        np.random.default_rng(3).uniform(0.0, 100.0, (40, 2)), ["x", "y"])
    model.predict(query, n_sim=16)
    sims = np.stack([np.asarray(query.values(f"m/{k}/simulations"))
                     for k in "abcd"], axis=2)
    np.testing.assert_allclose(sims.sum(axis=2), 1.0, atol=1e-10)
    assert (sims >= 0.0).all()
    assert type(query.variables["m"]) is geoml.data.MassFractionVariable
    # the zeros are the noise's: a measurement of the ground leaves the
    # simplex where the ground itself seldom does
    samples = np.asarray(model.predict_measurements(query, n_sim=8)["m"])
    assert (samples == 0.0).any()


def test_the_likelihood_does_not_depend_on_the_batch(trained):
    _, model, x, _ = trained
    lik = model.likelihoods[0]
    y, has_value = model.y, model.has_value
    samples = tf.constant(np.random.default_rng(4).normal(
        0.0, 1.0, (len(y), D, 3)), tf.float64)
    whole = lik.log_lik(None, None, tf.constant(y), tf.constant(has_value),
                        samples=samples)
    halves = sum(lik.log_lik(None, None, tf.constant(y[i]),
                             tf.constant(has_value[i]),
                             samples=tf.gather(samples, i))
                 for i in (np.arange(0, 37), np.arange(37, len(y))))
    assert float(whole) == pytest.approx(float(halves), rel=1e-12)


def test_a_composition_trains_too():
    coords, x = _fractions(50, 5)
    x = np.where(x <= 0, 1e-3, x)
    x = x / x.sum(axis=1, keepdims=True)
    _, model = _model(x, coords, _censored(), variable="compositional")
    model.train_full(max_iter=5)
    assert np.isfinite(model.training_log).all()


def test_a_save_keeps_the_likelihood(trained, tmp_path):
    _, model, _, _ = trained
    query = geoml.data.PointData.from_array(
        np.random.default_rng(6).uniform(0.0, 100.0, (10, 2)), ["x", "y"])
    model.predict(query, n_sim=4)
    before = np.asarray(query.values("m/a/prediction"), dtype=float).copy()
    persistence.save_model(model, tmp_path / "model")
    restored = persistence.load_model(tmp_path / "model")
    assert type(restored.likelihoods[0]) is lk.CensoredSimplex
    assert restored.likelihoods[0].max_hidden == 2
    restored.predict(query, n_sim=4)
    np.testing.assert_allclose(np.asarray(query.values("m/a/prediction")),
                               before, rtol=1e-10)


@pytest.mark.parametrize("method", ["train_full", "train_svi",
                                    "train_by_expert"])
def test_a_step_out_of_memory_names_what_to_make_smaller(method,
                                                         monkeypatch):
    coords, x = _fractions(40, 3)
    _, model = _model(x, coords, _censored(max_hidden=3))

    def exhausted(*args, **kwargs):
        raise tf.errors.ResourceExhaustedError(None, None, "OOM")

    monkeypatch.setattr(model.likelihoods[0], "log_lik", exhausted)
    with pytest.raises(MemoryError) as raised:
        getattr(model, method)(1)
    message = str(raised.value)
    assert "max_hidden (now 3" in message and "on the CPU" in message
    assert {"train_full": "train_svi or train_by_expert",
            "train_svi": "training_batch_size",
            "train_by_expert": "visits (now 1)"}[method] in message
    assert isinstance(raised.value.__cause__,
                      tf.errors.ResourceExhaustedError)
