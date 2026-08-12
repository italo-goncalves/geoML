# geoML - machine learning models for geospatial data
# Copyright (C) 2025  Ítalo Gomes Gonçalves
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


import numpy as _np


def rmse(y_true, y_pred):
    """
    Root mean squared error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    error : float
        The error, in the variable's units.
    """
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float).ravel()
    return float(_np.sqrt(_np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    """
    Mean absolute error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    error : float
        The error, in the variable's units.
    """
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float).ravel()
    return float(_np.mean(_np.abs(y_pred - y_true)))


def bias(y_true, y_pred):
    """
    Mean error, prediction minus truth. Positive means overestimation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    error : float
        The signed error, in the variable's units.
    """
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float).ravel()
    return float(_np.mean(y_pred - y_true))


def crps(y_true, y_pred):
    """
    Continuous ranked probability score, from samples. Lower is better.

    The proper score for a probabilistic prediction against a measured value:
    it rewards putting probability near the truth and nothing else, so neither
    hedging with wide intervals nor feigning precision can improve it. With a
    single sample per location it reduces to the absolute error, which is the
    scale to read it on. Estimated by the energy form
    ``E|X - y| - E|X - X'| / 2``, the pairwise term taken from the sorted
    samples in one pass.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples, n_predictions)
        Candidate predictions (e.g. measurement samples).

    Returns
    -------
    score : float
        The average score, in the variable's units.
    """
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float)

    m = y_pred.shape[1]
    ordered = _np.sort(y_pred, axis=1)
    position = 2 * _np.arange(1, m + 1) - m - 1
    spread = (ordered * position).sum(axis=1) / m ** 2

    error = _np.mean(_np.abs(y_pred - y_true[:, None]), axis=1)
    return float(_np.mean(error - spread))


def interval_score(y_true, y_pred, alpha=0.05):
    """
    Interval score, based on confidence intervals estimated from `y_pred`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples, n_predictions)
        Candidate predictions.
    alpha : float
        Confidence interval.

    Returns
    -------
    isc : float
        Interval score.
    """
    lower = _np.quantile(y_pred, alpha / 2, axis=1)
    upper = _np.quantile(y_pred, 1 - alpha / 2, axis=1)

    isc = _np.mean(
        upper - lower
        + 2 / alpha * _np.maximum(lower - y_true, 0.0)
        + 2 / alpha * _np.maximum(y_true - upper, 0.0)
    )
    return isc


def bias_variance_decomposition(y_true, y_pred):
    """
    Compute bias and variance from predictions and true values.

    Assumes multiple predictions for each true value (e.g., from bootstrapping or ensemble).

    Parameters
    ----------
    y_true: array-like, shape (n_samples,)
        True values.
    y_pred: array-like, shape (n_samples, n_predictions)
        Candidate predictions.

    Returns
    -------
    bias: float
        Mean squared bias.
    var: float
        Mean variance.
    """
    # Mean prediction across models
    y_pred_mean = _np.mean(y_pred, axis=1)

    # Bias^2: difference between average prediction and true value
    bias_squared = _np.mean((y_pred_mean - y_true) ** 2)

    # Variance: variability of predictions across models
    variance = _np.mean(_np.var(y_pred, axis=1))

    return bias_squared, variance


def coverage(y_true, y_pred, probabilities=None):
    """
    How often the truth falls inside an interval of a given probability.

    The numbers behind an accuracy plot. At every location the central interval
    holding a share `p` of the simulated values is built, and the fraction of
    true values inside it is counted. A model that knows what it does not know
    puts those fractions on the 1:1 line: below it the intervals are narrower
    than the errors they have to cover, above it the model is hedging.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples, n_predictions)
        Simulated values at the same locations.
    probabilities : array-like
        The nominal probabilities to check. Defaults to 0.05 to 0.95.

    Returns
    -------
    probabilities : array of shape (n_probabilities,)
        The nominal probabilities, as given.
    observed : array of shape (n_probabilities,)
        The share of true values actually inside each interval.
    """
    if probabilities is None:
        probabilities = _np.linspace(0.05, 0.95, 19)
    probabilities = _np.asarray(probabilities, dtype=float)

    y_true = _np.asarray(y_true).ravel()
    y_pred = _np.asarray(y_pred)

    observed = _np.zeros_like(probabilities)
    for i, p in enumerate(probabilities):
        lower = _np.quantile(y_pred, (1 - p) / 2, axis=1)
        upper = _np.quantile(y_pred, (1 + p) / 2, axis=1)
        observed[i] = _np.mean((y_true >= lower) & (y_true <= upper))

    return probabilities, observed


def goodness(probabilities, observed):
    """
    How close an accuracy plot sits to the 1:1 line. One is perfect.

    Deutsch's statistic. Intervals that are too wide are counted at half the
    weight of intervals that are too narrow: claiming a precision the model
    does not have is the worse mistake, since it is the one that leads someone
    to act on the number.

    Parameters
    ----------
    probabilities : array-like
        Nominal probabilities, as returned by `coverage`.
    observed : array-like
        Observed shares, as returned by `coverage`.

    Returns
    -------
    g : float
        One when the two agree everywhere, less as they part.
    """
    probabilities = _np.asarray(probabilities, dtype=float)
    observed = _np.asarray(observed, dtype=float)

    weight = _np.where(observed >= probabilities, 1.0, 2.0)
    departure = weight * _np.abs(observed - probabilities)

    # the trapezoid rule, written out rather than taken from NumPy, whose name
    # for it changed in 2.0
    width = _np.diff(probabilities)
    area = _np.sum(width * 0.5 * (departure[:-1] + departure[1:]))
    span = probabilities[-1] - probabilities[0]

    return 1.0 - area / span if span > 0 else 1.0


def aitchison_distance(comp_true, comp_pred):
    clr_true = _np.log(comp_true)
    clr_true = clr_true - _np.mean(clr_true, axis=1, keepdims=True)
    clr_pred = _np.log(comp_pred)
    clr_pred = clr_pred - _np.mean(clr_pred, axis=1, keepdims=True)

    dist = _np.sqrt(_np.sum((clr_true - clr_pred)**2, axis=1))
    return _np.mean(dist)
