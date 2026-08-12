"""Scoring metrics.

`coverage` and `goodness` are the numbers behind an accuracy plot, and they are
here rather than in the plotting code so that they can be had without drawing
anything -- and tested against distributions whose answer is known in advance.
"""
import numpy as np

import geoml.metrics as metrics


def test_coverage_finds_the_share_that_is_actually_inside():
    """Truth and simulations drawn from the same distribution: an interval
    holding `p` of the simulations should hold `p` of the truth."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=4000)
    simulations = rng.normal(size=(4000, 200))

    nominal, observed = metrics.coverage(y_true, simulations)
    assert np.allclose(observed, nominal, atol=0.03)


def test_coverage_catches_intervals_that_are_too_narrow():
    """The failure that matters: a model sure of itself and wrong."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=2000)
    simulations = rng.normal(size=(2000, 200)) * 0.2

    _, observed = metrics.coverage(y_true, simulations)
    assert np.all(observed < np.linspace(0.05, 0.95, 19))


def test_coverage_takes_the_probabilities_it_is_given():
    rng = np.random.default_rng(2)
    nominal, observed = metrics.coverage(
        rng.normal(size=200), rng.normal(size=(200, 50)),
        probabilities=[0.5, 0.9])

    assert list(nominal) == [0.5, 0.9]
    assert len(observed) == 2


def test_goodness_is_one_when_the_line_is_met():
    nominal = np.linspace(0.05, 0.95, 19)
    assert metrics.goodness(nominal, nominal) == 1.0


def test_goodness_counts_optimism_twice_as_heavily():
    """Claiming a precision the model does not have is the worse mistake: it
    is the one someone acts on."""
    nominal = np.linspace(0.05, 0.95, 19)

    hedging = metrics.goodness(nominal, np.clip(nominal + 0.1, 0, 1))
    optimistic = metrics.goodness(nominal, np.clip(nominal - 0.1, 0, 1))

    assert hedging > optimistic
    assert np.isclose(1 - optimistic, 2 * (1 - hedging), rtol=0.05)


def test_the_point_errors_on_a_worked_example():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([1.0, 1.0, 0.0])

    assert np.isclose(metrics.rmse(y_true, y_pred), np.sqrt(5 / 3))
    assert np.isclose(metrics.mae(y_true, y_pred), 1.0)
    assert np.isclose(metrics.bias(y_true, y_pred), -1 / 3)


def test_crps_matches_the_gaussian_closed_form():
    """`sigma * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / sqrt(pi))` for a
    Gaussian forecast -- the sample estimator should land on it."""
    from scipy import stats

    rng = np.random.default_rng(3)
    z = np.array([-2.0, -0.5, 0.0, 1.0, 2.5])
    samples = rng.normal(size=[z.size, 40000])

    exact = z * (2 * stats.norm.cdf(z) - 1) \
        + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    for i, (zi, ei) in enumerate(zip(z, exact)):
        assert np.isclose(metrics.crps(np.array([zi]), samples[i:i + 1]), ei,
                          atol=0.01)


def test_crps_reduces_to_the_absolute_error_for_one_sample():
    y_true = np.array([0.0, 1.0, -2.0])
    y_pred = np.array([[1.5], [1.0], [0.5]])
    assert np.isclose(metrics.crps(y_true, y_pred),
                      metrics.mae(y_true, y_pred.ravel()))


def test_crps_is_proper():
    """The truth's own distribution beats both the hedged and the
    overconfident forecast of it -- the property that makes the score worth
    reporting."""
    rng = np.random.default_rng(4)
    y_true = rng.normal(size=3000)

    honest = rng.normal(size=(3000, 400))
    hedged = rng.normal(size=(3000, 400)) * 3.0
    overconfident = rng.normal(size=(3000, 400)) * 0.2

    score = metrics.crps(y_true, honest)
    assert score < metrics.crps(y_true, hedged)
    assert score < metrics.crps(y_true, overconfident)


def test_variogram_score_on_a_worked_example():
    """Three locations, one realization, every pair by hand at p=1."""
    y_true = np.array([0.0, 1.0, 3.0])
    y_pred = np.array([[0.0], [2.0], [3.0]])

    truth = np.array([1.0, 3.0, 2.0])
    ensemble = np.array([2.0, 3.0, 1.0])
    expected = np.mean((truth - ensemble) ** 2)
    assert np.isclose(metrics.variogram_score(y_true, y_pred, p=1.0),
                      expected)


def test_variogram_score_punishes_broken_spatial_structure():
    """Same values at every location, dependence destroyed: `crps` cannot
    tell the two ensembles apart, and this score is the number that can."""
    rng = np.random.default_rng(6)
    n, m = 60, 200
    honest = np.cumsum(rng.normal(size=(n, m)), axis=0)
    y_true = np.cumsum(rng.normal(size=n))

    shuffled = honest.copy()
    for i in range(n):
        shuffled[i] = shuffled[i, rng.permutation(m)]

    assert metrics.variogram_score(y_true, honest) < \
        metrics.variogram_score(y_true, shuffled)
