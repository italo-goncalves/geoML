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


def aitchison_distance(comp_true, comp_pred):
    clr_true = _np.log(comp_true)
    clr_true = clr_true - _np.mean(clr_true, axis=1, keepdims=True)
    clr_pred = _np.log(comp_pred)
    clr_pred = clr_pred - _np.mean(clr_pred, axis=1, keepdims=True)

    dist = _np.sqrt(_np.sum((clr_true - clr_pred)**2, axis=1))
    return _np.mean(dist)
