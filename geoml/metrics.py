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

import geoml._types as _types
import geoml.math.geometry as _geom


def rmse(y_true: _types.ArrayLike, y_pred: _types.ArrayLike) -> float:
    """
    Root mean squared error.

    Parameters
    ----------
    y_true
        True values, one per location.
    y_pred
        Predicted values, one per location.

    Returns
    -------
    error : float
        The error, in the variable's units.
    """
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float).ravel()
    return float(_np.sqrt(_np.mean((y_pred - y_true) ** 2)))


def mae(y_true: _types.ArrayLike, y_pred: _types.ArrayLike) -> float:
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


def bias(y_true: _types.ArrayLike, y_pred: _types.ArrayLike) -> float:
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


def crps(y_true: _types.ArrayLike, y_pred: _types.ArrayLike) -> float:
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
    y_true
        True values, one per location.
    y_pred
        Candidate predictions per location, of shape
        `(n_data, n_predictions)` -- measurement samples, normally.

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


def variogram_score(y_true: _types.ArrayLike, y_pred: _types.ArrayLike,
                    p: float = 0.5, max_pairs: int = 50000,
                    coordinates: "_types.ArrayLike | None" = None,
                    decluster: "bool | float" = True) -> float:
    """
    How well the ensemble reproduces the differences between locations.

    Scheuerer and Hamill's (2015) proper score over pairs: for every pair of
    locations, the truth's absolute difference to the power `p` against the
    ensemble's mean one, squared and averaged. `crps` judges each location's
    marginal and cannot see dependence; this is the score that punishes an
    ensemble whose realizations have the right histograms and the wrong
    spatial structure. Past the budget the pairs are strided down
    deterministically.

    Given `coordinates`, each pair is weighted by `w_i * w_j` from
    :func:`geoml.math.geometry.declustering_weights`, so that a crowded
    patch of drilling counts once rather than once per hole. Without them
    the pairs are raw and the score describes the sampling as much as the
    field.

    This is an **estimate, and a biased one**. The truth carries the
    likelihood noise while the realizations are of the ground with that
    noise integrated out, so a measured difference is systematically the
    wider of the two and the score never reaches zero however good the model
    is. The variogram figure corrects the same bias by raising its fan,
    which works there because a semivariogram is a second moment and
    independent noise adds a known variance to it. `|difference| ** p` is
    not a second moment and has no such constant, so putting the two sides
    on one footing here would mean drawing noise into the realizations,
    which is a seed inside a metric and a number that changes between calls.
    The bias is instead left in place: it is common to any two ensembles on
    the same locations, so read the score as a **comparison between models
    on the same data** rather than as an absolute quantity.
    :func:`geoml.plots.prepare.variogram` is the honest picture of the same
    question, and the one to reach for when the size of the disagreement
    matters rather than its ordering.

    Parameters
    ----------
    y_true
        True values, one per location.
    y_pred
        Realizations at the same locations, of shape
        `(n_data, n_realizations)` -- simulations, not measurement
        samples: dependence between locations is the thing under test.
    p
        The power. 0.5 is the authors' recommendation.
    max_pairs
        The pair budget.
    coordinates
        `(n_data, n_dim)` sample locations, needed to decluster.
    decluster
        Weight pairs by cell-declustering weights. `True` chooses the cell
        size, a number fixes it, `False` leaves the pairs raw. Ignored when
        there are no `coordinates` to lay a lattice over.

    Returns
    -------
    score : float
        Lower is better.

    References
    ----------
    Scheuerer, M., & Hamill, T. M. (2015). Variogram-based proper scoring
    rules for probabilistic forecasts of multivariate quantities. *Monthly
    Weather Review*, 143(4), 1321-1334.
    """
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float)

    weights = None
    if coordinates is not None and decluster is not False:
        weights = _geom.declustering_weights(
            _np.asarray(coordinates, dtype=float), y_true,
            cell=None if decluster is True else float(decluster))[0]

    i_idx, j_idx = _np.triu_indices(y_true.size, k=1)
    if i_idx.size > max_pairs:
        stride = int(_np.ceil(i_idx.size / max_pairs))
        i_idx, j_idx = i_idx[::stride], j_idx[::stride]

    truth = _np.abs(y_true[i_idx] - y_true[j_idx]) ** p
    ensemble = _np.mean(
        _np.abs(y_pred[i_idx, :] - y_pred[j_idx, :]) ** p, axis=1)
    squared = (truth - ensemble) ** 2

    if weights is None:
        return float(_np.mean(squared))
    share = weights[i_idx] * weights[j_idx]
    return float((share * squared).sum() / share.sum())


def interval_score(y_true: _types.ArrayLike, y_pred: _types.ArrayLike,
                   alpha: float = 0.05) -> float:
    """
    Interval score, based on confidence intervals estimated from `y_pred`.

    Parameters
    ----------
    y_true
        True values, one per location.
    y_pred
        Candidate predictions per location, of shape
        `(n_data, n_predictions)`.
    alpha
        One minus the interval's nominal coverage.

    Returns
    -------
    isc : float
        Interval score.
    """
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float)

    lower = _np.quantile(y_pred, alpha / 2, axis=1)
    upper = _np.quantile(y_pred, 1 - alpha / 2, axis=1)

    isc = _np.mean(
        upper - lower
        + 2 / alpha * _np.maximum(lower - y_true, 0.0)
        + 2 / alpha * _np.maximum(y_true - upper, 0.0)
    )
    return float(isc)


def bias_variance_decomposition(y_true: _types.ArrayLike,
                                y_pred: _types.ArrayLike
                                ) -> tuple[float, float]:
    """
    Compute bias and variance from predictions and true values.

    Assumes multiple predictions for each true value (e.g., from bootstrapping or ensemble).

    Parameters
    ----------
    y_true
        True values, one per location.
    y_pred
        Candidate predictions per location, of shape
        `(n_data, n_predictions)`.

    Returns
    -------
    bias: float
        Mean squared bias.
    var: float
        Mean variance.
    """
    # taken as arrays first, as every other function here does: the docstring
    # accepts anything array-like, and a list would not survive the arithmetic
    y_true = _np.asarray(y_true, dtype=float).ravel()
    y_pred = _np.asarray(y_pred, dtype=float)

    # Mean prediction across models
    y_pred_mean = _np.mean(y_pred, axis=1)

    # Bias^2: difference between average prediction and true value
    bias_squared = _np.mean((y_pred_mean - y_true) ** 2)

    # Variance: variability of predictions across models
    variance = _np.mean(_np.var(y_pred, axis=1))

    return float(bias_squared), float(variance)


def coverage(y_true: _types.ArrayLike, y_pred: _types.ArrayLike,
             probabilities: _types.ArrayLike | None = None
             ) -> tuple[_types.FloatArray, _types.FloatArray]:
    """
    How often the truth falls inside an interval of a given probability.

    The numbers behind an accuracy plot. At every location the central interval
    holding a share `p` of the simulated values is built, and the fraction of
    true values inside it is counted. A model that knows what it does not know
    puts those fractions on the 1:1 line: below it the intervals are narrower
    than the errors they have to cover, above it the model is hedging.

    Parameters
    ----------
    y_true
        True values, one per location.
    y_pred
        Simulated values at the same locations, of shape
        `(n_data, n_realizations)`.
    probabilities
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


def goodness(probabilities: _types.ArrayLike,
             observed: _types.ArrayLike) -> float:
    """
    How close an accuracy plot sits to the 1:1 line. One is perfect.

    Deutsch's statistic. Intervals that are too wide are counted at half the
    weight of intervals that are too narrow: claiming a precision the model
    does not have is the worse mistake, since it is the one that leads someone
    to act on the number.

    Parameters
    ----------
    probabilities
        Nominal probabilities, as :func:`coverage` returns them.
    observed
        Observed shares, as :func:`coverage` returns them.

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


def aitchison_distance(comp_true: _types.ArrayLike,
                       comp_pred: _types.ArrayLike) -> float:
    """
    Mean distance between two compositions, in the simplex's own geometry.

    The Euclidean distance between the centred log-ratios, which is what
    "close" means for parts of a whole: it ignores the closure and answers in
    ratios rather than in percentage points.

    Parameters
    ----------
    comp_true
        True compositions, of shape `(n_data, n_parts)`, strictly positive.
    comp_pred
        Predicted compositions, of the same shape.

    Returns
    -------
    float
        The distance, averaged over the locations.
    """
    clr_true = _np.log(comp_true)
    clr_true = clr_true - _np.mean(clr_true, axis=1, keepdims=True)
    clr_pred = _np.log(comp_pred)
    clr_pred = clr_pred - _np.mean(clr_pred, axis=1, keepdims=True)

    dist = _np.sqrt(_np.sum((clr_true - clr_pred)**2, axis=1))
    return float(_np.mean(dist))
