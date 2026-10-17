"""An exponential or heavier noise tail under a log-like link warns at
`initialize`, naming the rate below which the value with the noise
integrated out has no finite mean or variance; anything else stays quiet.

Under `Log -> ZScore` one warped unit is the standard deviation of the
logarithm, so the warning's figures can be checked by hand.
"""
import re
import warnings

import numpy as np
import pytest

import geoml.likelihood as lk
import geoml.warping as wp


def _lognormal(n=500, k=1, sigma=1.5, seed=0):
    return np.exp(np.random.default_rng(seed).normal(0.0, sigma, (n, k)))


def _caught(likelihood, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        likelihood.initialize(y)
    return [str(w.message) for w in caught
            if "log-like" in str(w.message)]


def test_the_pole_is_the_spread_of_the_logarithm():
    y = _lognormal()
    messages = _caught(lk.EpsilonInsensitive(
        wp.ChainedWarping(wp.Log(1), wp.ZScore(1))), y)
    assert len(messages) == 1
    numbers = [float(v) for v in re.findall(r"[-+]?\d*\.\d+(?:e-?\d+)?",
                                            messages[0])]
    sd = np.std(np.log(y + 1e-6))
    # the reach, the mean's pole, the variance's pole
    np.testing.assert_allclose(numbers[:3], [sd, sd, 2 * sd], rtol=1e-3)


@pytest.mark.parametrize("likelihood", [
    lambda w: lk.Laplace(w),
    lambda w: lk.Huber(w),
    lambda w: lk.MultivariateLaplace(1, w),
    lambda w: lk.Mixture(w, family="laplace"),
])
def test_every_exponential_tail_warns(likelihood):
    assert len(_caught(likelihood(
        wp.ChainedWarping(wp.BoxCox(1), wp.ZScore(1))), _lognormal())) == 1


def test_a_student_tail_has_no_mean_at_any_scale():
    messages = _caught(lk.StudentT(
        wp.ChainedWarping(wp.Log(1), wp.ZScore(1))), _lognormal())
    assert len(messages) == 1 and "any scale" in messages[0]


def test_a_composition_through_a_projection_warns():
    y = _lognormal(k=4)
    y = y / y.sum(axis=1, keepdims=True)
    assert len(_caught(lk.MultivariateEpsilonInsensitive(
        3, wp.ChainedWarping(wp.CenteredLogRatio(4), wp.PCA(4, 3))), y)) == 1


@pytest.mark.parametrize("likelihood,warping", [
    (lk.Gaussian, wp.ChainedWarping(wp.Log(1), wp.ZScore(1))),
    (lk.EpsilonInsensitive, wp.ZScore(1)),
    (lk.Laplace, wp.ChainedWarping(wp.YeoJohnson(1), wp.ZScore(1))),
])
def test_a_gaussian_noise_or_a_link_that_is_not_logarithmic_is_quiet(
        likelihood, warping):
    assert _caught(likelihood(warping), _lognormal()) == []
