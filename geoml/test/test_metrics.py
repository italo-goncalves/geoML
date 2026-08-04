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
