"""What `CategoricalGaussianIndicator` reports as a category's probability.

A category holds a location where its own latent value is positive and every
other category's is negative. Both are probabilities of a sign under the
latent marginals, so a density has no place in the formula: until 0.9.15 the
negative side was one, which left a `log(sd)` in the winner's score.
"""
import numpy as np
import pytest
import tensorflow as tf
from scipy.special import log_ndtr, softmax

import geoml.likelihood as lk


def _predicted(mu, sd, **kwargs):
    mu = tf.constant(mu, tf.float64)
    var = tf.constant(np.asarray(sd) ** 2, tf.float64)
    sims = tf.zeros(mu.shape + [5], tf.float64)
    return lk.CategoricalGaussianIndicator(
        mu.shape[1], **kwargs).predict(mu, var, sims, var)


def test_the_probability_is_of_a_sign_in_every_column():
    """Each category's positive column and the others' negative ones, against
    the same expression written with SciPy's normal CDF."""
    mu = np.array([[1.0, -0.5, 0.2], [-1.0, -2.0, -0.1], [0.3, 0.4, -3.0]])
    sd = np.array([[0.5, 1.0, 0.8], [0.3, 0.9, 1.0], [1.0, 0.2, 0.6]])
    tol = 1e-3

    negative = log_ndtr((-tol - mu) / sd)
    positive = log_ndtr((mu - tol) / sd)
    reference = softmax(negative.sum(axis=1, keepdims=True) - negative
                        + positive, axis=1)

    np.testing.assert_allclose(_predicted(mu, sd)["probability"], reference)


def test_the_more_uncertain_category_gets_no_credit_for_it():
    """Category 0 stands further above zero in its own units than category 1,
    and category 1 has twice its standard deviation. The density's `log(sd)`
    handed the location to category 1."""
    out = _predicted([[0.05, -0.1]], [[0.5, 1.0]])
    assert np.argmax(out["probability"], axis=1)[0] == 0
    assert float(out["indicators"][0, 0]) > 0.0


@pytest.mark.parametrize("sd", [0.2, 1.0])
def test_a_clear_call_reads_as_one(sd):
    """Both columns several standard deviations the right way: the density's
    term cancelled the evidence of the column saying no."""
    out = _predicted([[3 * sd, -3 * sd]], [[sd, sd]])
    assert float(out["probability"][0, 0]) > 0.9999
