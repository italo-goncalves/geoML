# geoML - machine learning models for geospatial data
# Copyright (C) 2019  Ítalo Gomes Gonçalves
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

__all__ = ["Identity",
           "Spline",
           "ZScore",
           "Softplus",
           "Log",
           "ChainedWarping",
           "Scale",
           "Sigmoid",
           "Center",
           "ContinuousNormalizingFlow",
           "TensorProductFlow",
           "CenteredLogRatio",
           "PCA",
           "RobustPCA",
           "Rotation",
           "ScaledSimplex"
           ]

import geoml.math.interpolate as _gint
import geoml.parameter as _gpr
import geoml.math.tf as _tftools
import geoml.data as _data
import geoml.stats.random as _rnd

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp
import warnings as _warnings

from scipy.special import ndtri as _ndtri
from sklearn.covariance import MinCovDet as _MCD
from sklearn.decomposition import FastICA as _ICA
from sklearn.exceptions import ConvergenceWarning as _ConvergenceWarning


# The share of a uniform partition that `Spline.initialize` keeps for the
# outermost segment of each arm. That segment's slope is the one `backward`
# extrapolates along, and an inverse crosses `1 / slope` per unit: at 0.2 the
# slope is at least 0.2 whatever the knot count, so nothing beyond the knots
# comes back further than five units per unit out. The absolute 1e-6 floor
# this replaced left that slope at 1e-5, and the Jura chain answered a latent
# draw three standard deviations out with 1e5 -- which the `Log` at the
# bottom of the chain turned into an infinity, and the quantiles over the
# simulations into NaN.
_OUTERMOST_SHARE = 0.2

# Every share must also be strictly positive: the partitions are
# compositional, so a zero is minus infinity in logit coordinates and takes
# the whole column with it.
_SMALLEST_SHARE = 1e-6


def _arm_shares(widths, outer):
    """
    One arm's compositional shares, from the widths its knots ask for.

    Parameters
    ----------
    widths
        Gaps between consecutive warped knot positions on one arm.
    outer
        Index of the segment touching the arm's outer anchor: `0` for the
        left arm, `-1` for the right.

    Returns
    -------
    The widths as a composition summing to one, the outermost holding at
    least `_OUTERMOST_SHARE` of a uniform share.

    Notes
    -----
    Only the outermost segment is held away from zero by a real margin, and
    the asymmetry is the point. `backward` locates its bracket first and
    clips each iterate into it, so a flat segment *between* knots costs
    accuracy across one interval and cannot leave it. Past the last knot the
    spline extrapolates along that segment's slope with nothing to clip
    against, which is where a share near zero becomes an answer near
    infinity.

    A run of flat segments at the end of an arm is the normal case rather
    than a pathology: the knots span [-5, 5] and data rarely fills it, so
    every knot beyond the sample's range reads the same empirical quantile.
    """
    widths = _np.maximum(_np.asarray(widths, dtype=float), _SMALLEST_SHARE)
    shares = widths / widths.sum()
    floor = _OUTERMOST_SHARE / len(shares)
    if shares[outer] < floor:
        # the others keep their proportions to each other, rescaled into
        # whatever the outermost segment leaves them
        shares = shares * (1.0 - floor) / (shares.sum() - shares[outer])
        shares[outer] = floor
    return shares / shares.sum()


class _Warping(_gpr.Parametric):
    """
    Base warping class.
    """

    # Whether this warping lets one component of its input reach another
    # component of its output. It decides how the likelihood noise is
    # integrated out: a warping that works on each component alone needs one
    # integration rule per component, and one that mixes them needs a rule
    # over all of them at once, which costs an order of magnitude more nodes.
    # Keep it truthful -- `test_warping_integration` checks every warping's
    # declaration against a numerical Jacobian.
    _mixes = False

    # How wide this warping is on each side. Every subclass sets both in its
    # own constructor, from the size it is built for; they start at zero
    # rather than None because a width is a count.
    _size_in: int
    _size_out: int

    def __init__(self, **kwargs):
        super().__init__()
        self._size_in = 0
        self._size_out = 0

    @property
    def size_in(self) -> int:
        """How many components this warping takes, in the data's units."""
        return self._size_in

    @property
    def size_out(self) -> int:
        """How many it produces, on the latent scale."""
        return self._size_out

    @property
    def elementwise(self):
        """Whether this warping acts on each component on its own.

        With a single component there is nothing to mix, so a rotation is a
        sign and a one-component PCA is a scale.
        """
        return not self._mixes or self.size_out == 1


    def refresh(self):
        """
        Recomputes whatever internal state `forward` and `backward` read.

        A no-op for the closed-form warpings. The likelihood calls it once
        at each of its entry points, so a warping with real state (the
        flow) pays for it per call rather than per invocation --
        `integrated_backward` alone runs the backward once per noise node.
        Deliberately a per-call recomputation and never a stored-Variable
        cache, which would cut the training gradients to the state's
        parameters.
        """
        pass

    def forward(self, x):
        """
        Passes values through the class's warping function.

        Parameters
        ----------
        x : array-like
            Vector with values to warp.

        Returns
        -------
        x : array-like
            Vector with warped values.
        log_det : array-like
            Log-derivative of warping function.
        """
        raise NotImplementedError

    def backward(self, x):
        """
        Transforms values back to the original units.

        Parameters
        ----------
        x : array-like
            Vector with values to warp back to the original units.

        Returns
        -------
        x : array-like
            Vector with warped back values.
        """
        raise NotImplementedError

    def initialize(self, x, weights=None):
        """
        Uses the provided values to initialize the object's parameters.

        A data-dependent *start*, not a fit — and `weights` is how a
        declustered start arrives: one weight per row, as
        `container.decluster()` stores them, so a crowded patch of
        drilling does not set the target distribution. A warping with
        nothing to weight accepts and ignores them; `None` reproduces
        the unweighted start exactly.

        Parameters
        ----------
        x : array-like
            Vector with values to warp.
        weights : array-like, optional
            One declustering weight per row of `x`.

        Returns
        -------
        x : array-like
            Vector with warped values.
        """
        x, _ = self.forward(x)
        return x


class Identity(_Warping):
    """Identity warping."""
    def __init__(self, size):
        super().__init__()
        self._size_in = size
        self._size_out = size

    def forward(self, x):
        return x, _tf.reduce_sum(_tf.zeros_like(x), axis=1)
    
    def backward(self, x):
        return x


class Spline(_Warping):
    """
    Uses a monotonic spline to convert from original to warped space and
    back.

    The spline is assumed to work with normalized (z-score) values. It is
    centered at the origin and the arms span up to +/- 5 units. Its main
    use is to transform an asymmetric distribution to one closer to a Gaussian.

    Attributes
    ----------
    n_knots : int
        Total number of knots.
    """
    def __init__(self, size, knots_per_arm=5, backbone="cubic"):
        """
        Initializer for Spline.

        Parameters
        ----------
        knots_per_arm : int
            The number of knots used to build each side (positive and negative)
            of the spline.
        backbone : str
            The interpolant between the knots: `"cubic"` (the monotonic
            cubic, the default) or `"rq"` (the monotonic rational
            quadratic, whose inverse is a closed form rather than an
            iteration). Same knots and slopes either way.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size
        self.n_knots = knots_per_arm * 2 + 1

        comp = _np.ones(knots_per_arm) / knots_per_arm
        for i in range(size):
            self._add_parameter(f"warped_partition_left_{i}",
                                _gpr.CompositionalParameter(comp))
            self._add_parameter(f"warped_partition_right_{i}",
                                _gpr.CompositionalParameter(comp))
        if backbone == "cubic":
            self.spline = _gint.MonotonicCubicSpline()
        elif backbone == "rq":
            self.spline = _gint.MonotonicRationalQuadraticSpline()
        else:
            raise ValueError(
                "backbone must be 'cubic' or 'rq', got %r" % (backbone,))
        self.backbone = backbone
        x_original = _tf.constant(
            _np.linspace(-5, 5, knots_per_arm * 2 + 1)[:, None],
            _tf.float64
        )
        self.x_original = _tf.tile(x_original, [1, self.size_in])

    def _get_warped_coordinates(self, dim):
        warped_left = _tf.cumsum(
            self.parameters[f"warped_partition_left_{dim}"].get_value())
        warped_right = _tf.cumsum(
            self.parameters[f"warped_partition_right_{dim}"].get_value()) + 1.0
        warped_coordinates = _tf.concat(
            [_tf.constant([0.0], _tf.float64), warped_left, warped_right],
            axis=0) / 2
        warped_coordinates = 10 * warped_coordinates - 5
        return warped_coordinates
    
    def forward(self, x):
        warped_coordinates = _tf.stack(
            [self._get_warped_coordinates(i) for i in range(self.size_in)],
            axis=1
        )
        x_warp = self.spline.interpolate(self.x_original, warped_coordinates, x)
        xd = self.spline.interpolate_d1(self.x_original, warped_coordinates, x)
        log_det = _tf.reduce_sum(_tf.math.log(xd), axis=1)

        return x_warp, log_det
    
    def backward(self, x):
        warped_coordinates = _tf.stack(
            [self._get_warped_coordinates(i) for i in range(self.size_in)],
            axis=1
        )
        # solved rather than approximated: interpolating the knots the
        # other way round meets `forward` at the knots and parts between
        # them, which used to leave `forward(backward(y))` a tenth of a
        # standard deviation from `y`
        x_back = self.spline.invert(self.x_original, warped_coordinates, x)
        return x_back

    def initialize(self, x, weights=None):
        """
        Places the knots on the data's own normal-score transform.

        The spline's input knots are fixed on a regular grid; what moves is
        where each one lands. Setting them to the marginal Gaussian
        anamorphosis -- the empirical CDF at each knot, read through the
        normal quantile -- starts the warping at the transform a
        geostatistician would apply by hand, and training refines it from
        there.

        Parameters
        ----------
        x
            Values reaching this link of the chain, one column per
            dimension. Expected to be roughly standardized, as the class
            docstring says: the knots span [-5, 5].
        weights
            One declustering weight per row: the empirical CDF the knots
            are placed on is then the weighted one, so the target
            distribution is the field's rather than the sampling's.

        Returns
        -------
        The warped values, so that a chain's next link initializes on them.

        Notes
        -----
        Two compositional parameters carry each arm, so the map is monotone
        by construction and pinned at `(-5, -5)`, `(0, 0)` and `(5, 5)` --
        a compositional vector sums to one, and those three points are what
        the sum buys. The fit therefore keeps the *shape* of the normal
        score transform on each arm while rescaling it to reach the
        anchors, which is the one approximation involved.

        The knots span [-5, 5] and data rarely fills that, so the outermost
        knots of each arm read the same empirical quantile and ask to sit on
        top of one another. `_arm_shares` keeps the last segment of each arm
        at `_OUTERMOST_SHARE` of a uniform share instead, since that segment
        is the slope `backward` extrapolates along and a flat one inverts
        into an answer that leaves the scale entirely.

        Without this the knots keep their uniform partition, which reads
        out as exactly the input grid -- the identity -- so a chain of
        `Rotation` and `Spline` pairs would rotate repeatedly with nothing
        gaussianizing in between, and every rotation after the first sees
        data the previous one already made as independent as it knows how.

        An initialized spline is curved from the first iteration rather
        than after training, which used to matter: `backward` interpolated
        the swapped knots, an approximation good only where the map is
        straight, and the round trip was off by a tenth of a standard
        deviation. It solves the forward polynomial now
        (`MonotonicCubicSpline.invert`), so what is left is set by the
        transform's own conditioning rather than by the inverse.

        See Also
        --------
        Rotation : the usual partner before this one, likewise initialized
            from the data and likewise only as a starting point.
        """
        values = _np.asarray(x, dtype=float)
        if weights is not None:
            weights = _np.asarray(weights, dtype=float).ravel()
        knots = self.x_original.numpy()[:, 0]
        per_arm = (len(knots) - 1) // 2
        for dim in range(values.shape[1]):
            order = _np.argsort(values[:, dim])
            column = values[order, dim]
            if not _np.ptp(column) > 0:
                # nothing to transform, and no CDF worth inverting: leave
                # this column on the uniform partition, i.e. the identity
                continue
            found = _np.searchsorted(column, knots, side="right")
            if weights is None:
                floor = 1.0 / (len(column) + 1.0)
                share = found * floor
            else:
                # the weighted empirical CDF, with the tail floor counted
                # in Kish's effective sample size rather than in rows: a
                # region carried by many light rows is still the sample
                # it weighs, and equal weights reproduce count/(n+1)
                # exactly
                share_w = weights[order]
                cumulative = _np.concatenate([[0.0], _np.cumsum(share_w)])
                effective = cumulative[-1] ** 2 / (share_w ** 2).sum()
                floor = 1.0 / (effective + 1.0)
                share = cumulative[found] / cumulative[-1] \
                    * (effective * floor)
            share = _np.clip(share, floor, 1.0 - floor)
            target = _np.clip(_ndtri(share), -5.0, 5.0)

            # invert `_get_warped_coordinates`, then normalize each arm --
            # which is what enforces the anchors.
            scaled = (target + 5.0) / 5.0
            self.parameters[f"warped_partition_left_{dim}"].set_value(
                _arm_shares(_np.diff(scaled[:per_arm + 1]), outer=0))
            self.parameters[f"warped_partition_right_{dim}"].set_value(
                _arm_shares(_np.diff(scaled[per_arm:]), outer=-1))

        return self.forward(_tf.constant(values, _tf.float64))[0]


def _weighted_quantiles(x, q, weights):
    """Per-column weighted quantiles, by the midpoint rule.

    Equal weights land within a sample spacing of `np.quantile`, which is
    all a winsor fence needs; written out because the runtime floor's
    numpy has no weighted quantile of its own.
    """
    x = _np.asarray(x, dtype=float)
    weights = _np.asarray(weights, dtype=float).ravel()
    q = _np.asarray(q, dtype=float)
    out = _np.empty((len(q), x.shape[1]))
    for dim in range(x.shape[1]):
        order = _np.argsort(x[:, dim])
        column = x[order, dim]
        share = weights[order]
        cdf = (_np.cumsum(share) - 0.5 * share) / share.sum()
        out[:, dim] = _np.interp(q, cdf, column)
    return out


class ZScore(_Warping):
    """
    A Warping that simply normalizes the values to z-scores.
    """

    # The quantile band a robust initialization trusts; what lies outside is
    # clipped to the fence before the moments are taken.
    _ROBUST_QUANTILES = (0.01, 0.99)

    def __init__(self, size, mean=None, std=None, robust=False):
        """
        Initializer for ZScore.

        Parameters
        ----------
        mean : double
            The desired mean of the data.
        std : double
            The desired standard deviation of the data.
        robust : bool
            Whether to initialize from a winsorized copy of the data, so a
            handful of gross outliers cannot set the scale everything else
            is normalized by -- one was measured squashing the genuine
            values into a sliver of a trainable warp's working window. The
            clipped points still count, at the fence, so clean data fits
            the same moments. Meant for the warping under a `Mixture`
            likelihood, whose contamination component expects such values.

        The mean and standard deviation can be computed from the data
        (if omitted) or specified.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size
        self.robust = bool(robust)

        self._add_parameter(
            "mean",
            _gpr.RealParameter(
                _np.zeros([size]),
                _np.zeros([size]) - 1e9,
                _np.zeros([size]) + 1e9
            )
        )
        if mean is not None:
            self.parameters["mean"].set_value(mean)
            # self.parameters["mean"].set_limits(mean - 2*_np.abs(mean),
            #                                    mean + 2*_np.abs(mean))

        self._add_parameter(
            "std",
            _gpr.PositiveParameter(
                _np.ones([size]),
                _np.ones([size]) * 1e-9,
                _np.ones([size]) * 1e9
            )
        )
        if std is not None:
            self.parameters["std"].set_value(std)
            self.parameters["std"].set_limits(std / 100, std * 10)
        
    def forward(self, x):
        mean = self.parameters["mean"].get_value()[None, :]
        std = self.parameters["std"].get_value()[None, :]
        x = (x - mean) / std
        log_det = _tf.zeros_like(x) - _tf.math.log(std)
        return x, _tf.reduce_sum(log_det, axis=1)
    
    def backward(self, x):
        mean = self.parameters["mean"].get_value()[None, :]
        std = self.parameters["std"].get_value()[None, :]
        # x = _tftools.ensure_rank_2(x)
        return x * std + mean

    def _trusted(self, x, weights=None):
        """The data the initialization believes: winsorized when robust."""
        if not self.robust:
            return x
        if weights is None:
            lo, hi = _np.quantile(x, self._ROBUST_QUANTILES, axis=0)
        else:
            lo, hi = _weighted_quantiles(x, self._ROBUST_QUANTILES, weights)
        return _np.clip(x, lo, hi)

    def initialize(self, x, weights=None):
        fit = self._trusted(x, weights)
        if weights is None:
            mean = _np.mean(fit, axis=0)
            std = _np.std(fit, axis=0)
        else:
            share = _np.asarray(weights, dtype=float).ravel()[:, None]
            fit = _np.asarray(fit, dtype=float)
            mean = (share * fit).sum(axis=0) / share.sum()
            std = _np.sqrt(
                (share * (fit - mean) ** 2).sum(axis=0) / share.sum())

        if not self.parameters["mean"].fixed:
            self.parameters["mean"].set_value(mean)
            self.parameters["mean"].set_limits(mean - 3 * std, mean + 3 * std)

        if not self.parameters["std"].fixed:
            self.parameters["std"].set_value(std)
            self.parameters["std"].set_limits(std / 100, std * 10)
        return super().initialize(x, weights)


class Center(ZScore):
    """
    A Warping that simply centers the data.
    """

    def __init__(self, size, mean=None):
        """
        Initializer for Center.

        The mean can be computed from the data (if omitted) or specified.

        Parameters
        ----------
        mean : double
            The desired mean of the data.
        """
        super().__init__(size, mean, std=_np.ones(size))
        self.parameters['std'].fix()

    def initialize(self, x, weights=None):
        mean = _np.mean(x, axis=0)
        if not self.parameters["mean"].fixed:
            self.parameters["mean"].set_value(mean)
        return super().initialize(x, weights)


class Softplus(_Warping):
    """
    Transforms the data using the inverse of the softplus function. 
    All the data must be positive.
    """
    def __init__(self, size, shift=1e-6):
        """
        Initializer for Softplus.

        Parameters
        ----------
        shift : float
            A positive value to add to the data. Use it if you have zeros.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size

        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    # computation only for x < 50.0 to avoid overflow
    def forward(self, x):
        # x = _tftools.ensure_rank_2(x)
        x_s = x + self.shift
        x_warp = _tf.where(_tf.greater(x_s, 50.0),
                           x_s,
                           _tf.math.log(_tf.math.expm1(x_s))
                           )
        # x_warp = _tf.where(_tf.math.is_nan(x), x, x_warp)

        log_det = _tf.where(_tf.greater(x_s, 50.0),
                            _tf.ones_like(x_s),
                            # 1 / (- _tf.math.expm1(-x_warp))
                            _tf.math.exp(x_s) / _tf.math.expm1(x_s)
                            )
        log_det = _tf.reduce_sum(_tf.math.log(log_det), axis=1)
        return x_warp, log_det
    
    def backward(self, x):
        # x = _tftools.ensure_rank_2(x)
        x_back = _tf.where(_tf.greater(x, 50.0),
                           x,
                           _tf.math.log1p(_tf.math.exp(x)))
        return x_back


class Log(_Warping):
    """
    Log-scale warping.

    Forward function: log
    Backward function: exp
    """
    def __init__(self, size, shift=1e-6):
        """
        Initializer for Log.

        Parameters
        ----------
        shift : float
            A positive value to add to the data. Use it if you have zeros.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size

        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    def forward(self, x):
        x_warp = _tf.math.log(x + self.shift)
        # the derivative is 1 / (x + shift); what goes out is its log, as
        # everywhere else, so that a chain can add its links together
        log_det = - _tf.reduce_sum(_tf.math.log(x + self.shift), axis=1)
        return x_warp, log_det

    def backward(self, x):
        return _tf.math.exp(x)


class Scale(ZScore):
    """Linear scaling, assuming a mean of zero."""
    def __init__(self, size, scale=1):
        super().__init__(
            size,
            mean=_np.full([size], -1e-6),
            std=_np.full([size], scale)
        )
        self.parameters["mean"].fix()

    def initialize(self, x, weights=None):
        sc = _np.max(x, axis=0) - _np.min(x, axis=0) + 1e-6
        if not self.parameters["std"].fixed:
            self.parameters["std"].set_value(sc)
        x, _ = self.forward(x)
        return x

    def backward(self, x):
        std = self.parameters["std"].get_value()[None, :]
        return x * std


def _profile_start(x, weights, candidates, transform, log_jacobian):
    """The candidate that makes a transformed column most Gaussian.

    Box and Cox's criterion: the Gaussian log-likelihood of the transformed
    values with their mean and variance profiled out, plus the log-Jacobian
    of the transform -- on declustering weights where given. Every
    parametric link below takes its data-dependent start from it; training
    refines the value from there.
    """
    x = _np.asarray(x, dtype=float).ravel()
    w = _np.ones_like(x) if weights is None \
        else _np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()
    best, best_value = candidates[0], -_np.inf
    for candidate in candidates:
        with _np.errstate(all="ignore"):
            z = transform(x, candidate)
            jac = log_jacobian(x, candidate)
        if not (_np.all(_np.isfinite(z)) and _np.all(_np.isfinite(jac))):
            continue
        mean = (w * z).sum()
        var = (w * (z - mean) ** 2).sum()
        if not var > 0:
            continue
        value = -0.5 * _np.log(var) + (w * jac).sum()
        if value > best_value:
            best, best_value = candidate, value
    return best


def _power_np(u, lam):
    return u if abs(lam) < 1e-8 else _np.expm1(lam * u) / lam


def _log_cosh(a):
    """`log(cosh(a))` without overflow: `|a| - log 2 + log1p(exp(-2|a|))`."""
    a = _tf.abs(a)
    return a - _np.log(2.0) + _tf.math.log1p(_tf.exp(-2.0 * a))


class BoxCox(_Warping):
    """
    Box-Cox power transform with a trainable exponent.

    Maps ``x`` to ``((x + shift)**λ - 1) / λ``, the logarithm at ``λ = 0``,
    with one exponent per component. The exponent starts at the value that
    makes the column most Gaussian -- the profile likelihood of Box and Cox
    -- and trains with the model. For positive data it is the power link
    between the logarithm and the identity, and its inverse is a root
    rather than an exponential, which is what keeps a latent draw in the
    tail from blowing up the way `Log.backward` does.

    Parameters
    ----------
    size
        Number of components.
    shift
        A positive value added to the data before the power, for zeros.
    exponent
        The starting exponent, one value or one per component;
        `initialize` replaces it.

    Notes
    -----
    The exponent is kept in ``[0, 2]``. With ``λ > 0`` the transformed
    values are bounded below by ``-1/λ``; a latent draw past that bound has
    no pre-image and `backward` returns the floor, a value of zero, which
    is finite and harmless. A negative exponent bounds them *above*, and
    the inverse then blows up toward that bound -- measured on Jura at 826
    times the data's maximum on tail draws, with the mean prediction
    destroyed in one arm -- so negative exponents are not offered.

    References
    ----------
    Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations.
    *Journal of the Royal Statistical Society: Series B*, 26(2), 211-252.
    """
    _EXPONENT_LIMITS = (0.0, 2.0)
    _GRID = _np.linspace(0.0, 2.0, 81)

    def __init__(self, size, shift=1e-6, exponent=1.0):
        super().__init__()
        self._size_in = size
        self._size_out = size
        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = float(shift)
        low, high = self._EXPONENT_LIMITS
        self._add_parameter("exponent", _gpr.RealParameter(
            _np.broadcast_to(_np.asarray(exponent, dtype=float), [size]).copy(),
            _np.full([size], low), _np.full([size], high)))

    @staticmethod
    def _power(u, lam):
        """`(exp(λ u) - 1) / λ`, and `u` at `λ = 0`; every branch finite,
        so the `where` differentiates cleanly."""
        small = _tf.abs(lam) < 1e-8
        safe = _tf.where(small, _tf.ones_like(lam), lam)
        return _tf.where(small, u, _tf.math.expm1(safe * u) / safe)

    @staticmethod
    def _root(y, lam):
        """The inverse of `_power`: `log(1 + λ y) / λ`, and `y` at `λ = 0`,
        the argument floored where a draw has no pre-image."""
        small = _tf.abs(lam) < 1e-8
        safe = _tf.where(small, _tf.ones_like(lam), lam)
        inner = _tf.maximum(1.0 + safe * y, 1e-12)
        return _tf.where(small, y, _tf.math.log(inner) / safe)

    def forward(self, x):
        lam = self.parameters["exponent"].get_value()[None, :]
        u = _tf.math.log(x + self.shift)
        log_det = _tf.reduce_sum((lam - 1.0) * u, axis=1)
        return self._power(u, lam), log_det

    def backward(self, x):
        lam = self.parameters["exponent"].get_value()[None, :]
        return _tf.exp(self._root(x, lam)) - self.shift

    def initialize(self, x, weights=None):
        x = _np.asarray(x, dtype=float)
        shift = self.shift
        starts = [
            _profile_start(
                x[:, i], weights, self._GRID,
                lambda v, c: _power_np(_np.log(v + shift), c),
                lambda v, c: (c - 1.0) * _np.log(v + shift))
            for i in range(self.size_in)]
        if not self.parameters["exponent"].fixed:
            self.parameters["exponent"].set_value(_np.array(starts))
        return super().initialize(x, weights)


class YeoJohnson(_Warping):
    """
    Yeo-Johnson power transform with a trainable exponent.

    Box-Cox extended to the whole real line: the power ``λ`` acts on
    ``1 + x`` for ``x >= 0`` and the power ``2 - λ`` on ``1 - x`` for
    ``x < 0``, so the map is smooth through zero, the identity at
    ``λ = 1``, and needs no shift. One exponent per component, started at
    the most Gaussian value by the profile likelihood and trained with the
    model. The link for a centred or residual column, where `BoxCox`
    cannot go.

    Parameters
    ----------
    size
        Number of components.
    exponent
        The starting exponent, one value or one per component;
        `initialize` replaces it.

    Notes
    -----
    The exponent is kept in ``[0, 2]``, where both powers are non-negative
    and the inverse is defined for every real value. Outside it one side
    of the map is bounded and its inverse blows up toward the bound, the
    hazard `BoxCox` documents.

    References
    ----------
    Yeo, I.-K., & Johnson, R. A. (2000). A new family of power
    transformations to improve normality or symmetry. *Biometrika*, 87(4),
    954-959.
    """
    _EXPONENT_LIMITS = (0.0, 2.0)
    _GRID = _np.linspace(0.0, 2.0, 81)

    def __init__(self, size, exponent=1.0):
        super().__init__()
        self._size_in = size
        self._size_out = size
        low, high = self._EXPONENT_LIMITS
        self._add_parameter("exponent", _gpr.RealParameter(
            _np.broadcast_to(_np.asarray(exponent, dtype=float), [size]).copy(),
            _np.full([size], low), _np.full([size], high)))

    def forward(self, x):
        lam = self.parameters["exponent"].get_value()[None, :]
        # each branch sees only its own side, so the unselected one stays
        # finite and the `where` differentiates cleanly
        up = _tf.math.log1p(_tf.maximum(x, 0.0))
        down = _tf.math.log1p(_tf.maximum(-x, 0.0))
        positive = x >= 0.0
        warped = _tf.where(positive, BoxCox._power(up, lam),
                           -BoxCox._power(down, 2.0 - lam))
        log_det = _tf.reduce_sum(
            _tf.where(positive, (lam - 1.0) * up, (1.0 - lam) * down), axis=1)
        return warped, log_det

    def backward(self, x):
        lam = self.parameters["exponent"].get_value()[None, :]
        up = _tf.maximum(x, 0.0)
        down = _tf.maximum(-x, 0.0)
        return _tf.where(x >= 0.0,
                         _tf.math.expm1(BoxCox._root(up, lam)),
                         -_tf.math.expm1(BoxCox._root(down, 2.0 - lam)))

    @staticmethod
    def _forward_np(v, c):
        return _np.where(v >= 0, _power_np(_np.log1p(_np.maximum(v, 0)), c),
                         -_power_np(_np.log1p(_np.maximum(-v, 0)), 2.0 - c))

    @staticmethod
    def _log_jacobian_np(v, c):
        return _np.where(v >= 0, (c - 1.0) * _np.log1p(_np.maximum(v, 0)),
                         (1.0 - c) * _np.log1p(_np.maximum(-v, 0)))

    def initialize(self, x, weights=None):
        x = _np.asarray(x, dtype=float)
        starts = [
            _profile_start(x[:, i], weights, self._GRID,
                           self._forward_np, self._log_jacobian_np)
            for i in range(self.size_in)]
        if not self.parameters["exponent"].fixed:
            self.parameters["exponent"].set_value(_np.array(starts))
        return super().initialize(x, weights)


class Arcsinh(_Warping):
    """
    Inverse hyperbolic sine warping, with a trainable scale.

    Maps ``x`` to ``asinh(x / scale)``: the identity for values small
    against the scale, the logarithm of ``2 |x| / scale`` far from it, on
    both sides of zero. Tolerates zeros and negatives with no shift, which
    is what recommends it over `Log` for a variable that is skewed but not
    strictly positive. The scale starts at the value that makes the column
    most Gaussian and trains with the model.

    Parameters
    ----------
    size
        Number of components.
    scale
        The starting scale, one value or one per component; `initialize`
        replaces it.

    References
    ----------
    Johnson, N. L. (1949). Systems of frequency curves generated by methods
    of translation. *Biometrika*, 36(1-2), 149-176.

    Burbidge, J. B., Magee, L., & Robb, A. L. (1988). Alternative
    transformations to handle extreme values of the dependent variable.
    *Journal of the American Statistical Association*, 83(401), 123-127.
    """
    def __init__(self, size, scale=1.0):
        super().__init__()
        self._size_in = size
        self._size_out = size
        self._add_parameter("scale", _gpr.PositiveParameter(
            _np.broadcast_to(_np.asarray(scale, dtype=float), [size]).copy(),
            _np.full([size], 1e-9), _np.full([size], 1e9)))

    def forward(self, x):
        scale = self.parameters["scale"].get_value()[None, :]
        log_det = -0.5 * _tf.reduce_sum(_tf.math.log(x ** 2 + scale ** 2), axis=1)
        return _tf.asinh(x / scale), log_det

    def backward(self, x):
        scale = self.parameters["scale"].get_value()[None, :]
        return scale * _tf.sinh(x)

    def initialize(self, x, weights=None):
        x = _np.asarray(x, dtype=float)
        starts = []
        for i in range(self.size_in):
            spread = float(_np.std(x[:, i])) or 1.0
            starts.append(_profile_start(
                x[:, i], weights, spread * _np.logspace(-2.0, 2.0, 81),
                lambda v, c: _np.arcsinh(v / c),
                lambda v, c: -0.5 * _np.log(v ** 2 + c ** 2)))
        if not self.parameters["scale"].fixed:
            self.parameters["scale"].set_value(_np.array(starts))
        return super().initialize(x, weights)


class SinhArcsinh(_Warping):
    """
    Sinh-arcsinh warping, with trainable skewness and tail weight.

    Maps ``x`` to ``sinh(δ asinh(x) - ε)``: ``ε`` skews the distribution,
    ``δ`` sets its tail weight (below one heavier than the Gaussian, above
    one lighter), and ``ε = 0, δ = 1`` is the identity. The inverse is
    closed form, ``sinh((asinh(z) + ε) / δ)``. Meant for a column already
    centred and scaled, as by `ZScore`: the curvature of the map sits at
    unit scale. The two parameters start at the values that make the
    column most Gaussian and train with the model. Between them they cover
    what `Arcsinh` and Johnson's S_U system do.

    Parameters
    ----------
    size
        Number of components.
    skewness
        The starting ``ε``, one value or one per component; `initialize`
        replaces it.
    tailweight
        The starting ``δ``, likewise.

    References
    ----------
    Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions.
    *Biometrika*, 96(4), 761-780.
    """
    _SKEW_GRID = _np.linspace(-2.0, 2.0, 21)
    _TAIL_GRID = _np.logspace(_np.log10(0.25), _np.log10(4.0), 21)

    def __init__(self, size, skewness=0.0, tailweight=1.0):
        super().__init__()
        self._size_in = size
        self._size_out = size
        self._add_parameter("skewness", _gpr.RealParameter(
            _np.broadcast_to(_np.asarray(skewness, dtype=float), [size]).copy(),
            _np.full([size], -3.0), _np.full([size], 3.0)))
        self._add_parameter("tailweight", _gpr.PositiveParameter(
            _np.broadcast_to(_np.asarray(tailweight, dtype=float), [size]).copy(),
            _np.full([size], 0.1), _np.full([size], 10.0)))

    def forward(self, x):
        skew = self.parameters["skewness"].get_value()[None, :]
        tail = self.parameters["tailweight"].get_value()[None, :]
        inner = tail * _tf.asinh(x) - skew
        log_det = _tf.reduce_sum(
            _tf.math.log(tail) + _log_cosh(inner)
            - 0.5 * _tf.math.log1p(x ** 2), axis=1)
        return _tf.sinh(inner), log_det

    def backward(self, x):
        skew = self.parameters["skewness"].get_value()[None, :]
        tail = self.parameters["tailweight"].get_value()[None, :]
        return _tf.sinh((_tf.asinh(x) + skew) / tail)

    @staticmethod
    def _forward_np(v, c):
        skew, tail = c
        return _np.sinh(tail * _np.arcsinh(v) - skew)

    @staticmethod
    def _log_jacobian_np(v, c):
        skew, tail = c
        inner = _np.abs(tail * _np.arcsinh(v) - skew)
        return (_np.log(tail) + inner - _np.log(2.0) + _np.log1p(_np.exp(-2.0 * inner))
                - 0.5 * _np.log1p(v ** 2))

    def initialize(self, x, weights=None):
        x = _np.asarray(x, dtype=float)
        candidates = [(e, d) for e in self._SKEW_GRID for d in self._TAIL_GRID]
        starts = [
            _profile_start(x[:, i], weights, candidates,
                           self._forward_np, self._log_jacobian_np)
            for i in range(self.size_in)]
        if not self.parameters["skewness"].fixed:
            self.parameters["skewness"].set_value(
                _np.array([s for s, _ in starts]))
        if not self.parameters["tailweight"].fixed:
            self.parameters["tailweight"].set_value(
                _np.array([t for _, t in starts]))
        return super().initialize(x, weights)


class ChainedWarping(_Warping):
    """
    Chains multiple Warping objects.
    """
    def __init__(self, *warpings):
        """

        Parameters
        ----------
        warpings : list
            List with Warping objects to apply in sequence.
        """
        super().__init__()
        self.warpings = list(warpings)
        for wp in warpings:
            self._register(wp)

        for i in range(len(self.warpings) - 1):
            size_out = self.warpings[i].size_out
            size_in = self.warpings[i + 1].size_in
            if size_out != size_in:
                raise ValueError(
                    f'Chained warping dimension mismatch at position {i}: {size_out} != {size_in}'
                )
        self._size_in = self.warpings[0].size_in
        self._size_out = self.warpings[-1].size_out

    @property
    def elementwise(self):
        """A chain is elementwise only if every link is.

        One mixing link is enough to spread a component over the others, and
        everything applied after it sees the mixture.
        """
        return all(wp.elementwise for wp in self.warpings)

    def refresh(self):
        for wp in self.warpings:
            wp.refresh()

    def forward(self, x):
        # log-determinants add along a chain, so the accumulator starts at
        # zero; `ones_like` seeded it with the column count until 0.6.5,
        # which offset every chained warping's value by its own width
        d = _tf.reduce_sum(_tf.zeros_like(x, dtype=_tf.float64), axis=1)
        for wp in self.warpings:
            x, log_d = wp.forward(x)
            d = d + log_d
        return x, d

    def backward(self, x):
        warping_rev = self.warpings.copy()
        warping_rev.reverse()
        for wp in warping_rev:
            x = wp.backward(x)
        return x

    def initialize(self, x, weights=None):
        for wp in self.warpings:
            x = wp.initialize(x, weights)
        return x


class Sigmoid(_Warping):
    """
    Sigmoid warping, for values constrained to the ]0, 1[ interval.

    Forward function: inverse sigmoid
    Backward function: sigmoid
    """

    def __init__(self, size, shift=1e-6):
        """
        Initializer for Sigmoid.

        Parameters
        ----------
        shift : float
            A positive value to ensure the data is constrained to the ]0, 1[ interval.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size

        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    def forward(self, x):
        x = x * (1 - 2 * self.shift) + self.shift
        x_warp = - _tf.math.log(1 / x - 1)
        log_det = _tf.reduce_sum(- _tf.math.log(x - x**2), axis=1)
        return x_warp, log_det

    def backward(self, x):
        return 1 / (1 + _tf.math.exp(-x))


class _ContinuousFlow(_Warping):
    """What the continuous flows share: time knots, the solver, both
    integration directions and the sweep contract.

    A subclass owns one representation of the velocity field -- its
    parameters, `refresh` (which settles the field into `_state`, a dict
    of tensors), `_field` and `_field_and_divergence`. The settled state
    rides the solver's `constants` rather than the closure: the adjoint
    computes gradients for the state being integrated, for Variables read
    inside the field and for `constants`, but a tensor merely captured is
    cut -- which silently detached the field's weights from training when
    this was first written the obvious way.

    `forward` and `backward` deliberately do not refresh themselves: the
    likelihood calls `refresh()` once per entry point, where the old
    per-invocation refresh was most of the flow's measured cost.
    """
    _mixes = True

    def __init__(self, size, n_steps=10, rtol=1e-6):
        super().__init__()
        self._size_in = size
        self._size_out = size
        self.n_knots = n_steps
        self.rtol = float(rtol)
        self._solver = _tfp.math.ode.DormandPrince(
            rtol=self.rtol, atol=self.rtol * 1e-2)
        self.knots = _tf.constant(
            _np.linspace(0.0, 1.0, n_steps), _tf.float64)
        self._state = None

        self._add_parameter(
            'rng_time', _gpr.PositiveParameter(1.0, 0.1, 10.0)
        )

    def _time_cross(self, t):
        """`k_time(t, knots)`, `[n_knots]` for a scalar `t`."""
        rng = self.parameters['rng_time'].get_value()
        dif = (t - self.knots) / rng
        return _tf.exp(-3.0 * dif ** 2)

    def _time_cholesky(self):
        knot_col = self.knots[:, None]
        rng = self.parameters['rng_time'].get_value()
        dif = _tftools.pairwise_dist(knot_col, knot_col) / rng
        cov = _tf.exp(-3.0 * dif ** 2)
        return _tf.linalg.cholesky(
            cov + _tf.eye(self.n_knots, dtype=_tf.float64) * 1e-6)

    def refresh(self):
        raise NotImplementedError

    def _field(self, x, t, **state):
        raise NotImplementedError

    def _field_and_divergence(self, x, t, **state):
        raise NotImplementedError

    def _settled(self):
        if self._state is None:
            raise RuntimeError(
                "the flow's field is not settled: call refresh() first "
                "(the likelihood does this once per entry point)")
        return self._state

    def forward(self, x):
        state = self._settled()
        x = _tf.convert_to_tensor(x, _tf.float64)
        augmented = _tf.concat(
            [x, _tf.zeros([_tf.shape(x)[0], 1], _tf.float64)], axis=1)

        def ode_fn(t, y, **state):
            field, divergence = self._field_and_divergence(
                y[:, :self.size_in], t, **state)
            return _tf.concat([field, divergence[:, None]], axis=1)

        out = self._solver.solve(
            ode_fn, 0.0, augmented, solution_times=[1.0],
            constants=state).states[0]
        return out[:, :self.size_in], out[:, self.size_in]

    def backward(self, x):
        state = self._settled()
        x = _tf.convert_to_tensor(x, _tf.float64)

        # the same settled field, time reversed: the round trip misses by
        # the solver tolerance, not by a scheme mismatch
        def ode_fn(t, y, **state):
            return -self._field(y, 1.0 - t, **state)

        return self._solver.solve(
            ode_fn, 0.0, x, solution_times=[1.0],
            constants=state).states[0]

    def flow_history(self, x):
        """The trajectory at the knot times, for inspection."""
        self.refresh()
        x = _tf.convert_to_tensor(x, _tf.float64)

        def ode_fn(t, y, **state):
            return self._field(y, t, **state)

        times = _np.linspace(0.0, 1.0, self.n_knots + 1)[1:]
        states = self._solver.solve(
            ode_fn, 0.0, x, solution_times=times,
            constants=self._settled()).states
        return [_np.asarray(x)] + [_np.asarray(s) for s in states]

    def initialize(self, x, weights=None):
        """Settles the field and passes the data through.

        Nothing here is data-dependent -- the field's geometry is fixed by
        construction -- so unlike every other data-dependent start this
        one only exists to hand the next link what the chain threads.
        """
        self.refresh()
        x, _ = self.forward(x)
        return x


class ContinuousNormalizingFlow(_ContinuousFlow):
    """A continuous-time flow on a fixed, separable space-time field.

    The velocity field is a deterministic Gaussian-kernel interpolant over
    anchors that never move: Sobol points covering the `[-5, 5]` box (the
    chain leads with a `ZScore`, so the flow sees whitened units) crossed
    with regular time knots on `[0, 1]`, the covariance factorizing as
    `K_space (x) K_time`. `refresh` settles the field with one Cholesky per
    factor; `forward` then integrates the augmented state `(x, log_det)`
    with an adaptive Dormand-Prince solver, and `backward` integrates the
    *same* field with time reversed, so the round trip is exact to the
    solver tolerance rather than to a scheme mismatch. The divergence
    driving the log-determinant is analytic -- the derivative of the
    spatial kernel factor -- and the map decays to the identity away from
    the box, so nothing steep ever reaches the tails.
    """

    # Built by `refresh` rather than by the constructor, which starts them
    # at None: the declarations say what they become, so the methods that
    # read them are not read as indexing None.
    alpha: "_tf.Tensor | None"
    chol_space: "_tf.Tensor | None"
    chol_time: "_tf.Tensor | None"

    def __init__(self, size, inducing_points=20, n_steps=10, step=0.01,
                 rtol=1e-6):
        """
        Initializer for ContinuousNormalizingFlow.

        Parameters
        ----------
        size
            Number of components. The flow exists to mix them; a single
            component is `Spline`'s job.
        inducing_points
            Number of spatial anchors -- Sobol points covering the
            `[-5, 5]` box, fixed by construction.
        n_steps
            Number of time knots of the field over `[0, 1]`.
        step
            Accepted so saves from versions before 0.6.9 replay, and
            ignored: the integration interval is fixed at `[0, 1]` and the
            amplitude parameter carries the scale.
        rtol
            Relative tolerance of the solver, shared by both directions
            and by the log-determinant.
        """
        super().__init__(size, n_steps=n_steps, rtol=rtol)
        self.n_ip = inducing_points

        # fixed by construction: deterministic Sobol coverage of the box,
        # so there is nothing data-dependent to persist and nothing for
        # `initialize` to place. The generator is local on purpose -- the
        # anchors are geometry, not a draw the package seed should move
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            unit = _rnd.sobol_engine(
                size, _np.random.default_rng(0)).random(inducing_points)
        self.anchors = _tf.constant((unit - 0.5) * 10.0, _tf.float64)

        self.alpha = None
        self.chol_space = None
        self.chol_time = None

        self._add_parameter(
            'alpha_white',
            _gpr.RealParameter(
                _rnd.rng().normal(
                    scale=1e-2, size=[inducing_points, size, n_steps]),
                _np.full([inducing_points, size, n_steps], -10),
                _np.full([inducing_points, size, n_steps], 10)
            )
        )
        self._add_parameter(
            'amp', _gpr.PositiveParameter(1, 0.01, 100)
        )
        self._add_parameter(
            'rng_space', _gpr.PositiveParameter(3.0, 0.5, 20.0)
        )

    def _space_cross(self, x):
        """`k_space(x, anchors)`, `[n, m]`."""
        rng = self.parameters['rng_space'].get_value()
        dist = _tftools.pairwise_dist(x, self.anchors) / rng
        return _tf.exp(-3.0 * dist ** 2)

    def refresh(self):
        amp = self.parameters['amp'].get_value()
        alpha_white = self.parameters['alpha_white'].get_value()

        cov_space = self._space_cross(self.anchors)
        chol_space = _tf.linalg.cholesky(
            cov_space + _tf.eye(self.n_ip, dtype=_tf.float64) * 1e-6)
        chol_time = self._time_cholesky()

        # whitened weights to interpolation coefficients, one triangular
        # solve per Kronecker factor: the coefficients are
        # `L_space^{-T} W L_time^{-1}`, so the field's values at the
        # anchor-knot lattice come out as `L_space W L_time^T` -- a unit-
        # variance draw's scale, whatever the ranges hold
        white = _tf.transpose(alpha_white, [1, 0, 2])  # [size, m, K]
        half = _tf.linalg.triangular_solve(
            chol_space[None], white, lower=True, adjoint=True)
        coef = _tf.transpose(_tf.linalg.triangular_solve(
            chol_time[None], _tf.transpose(half, [0, 2, 1]),
            lower=True, adjoint=True), [0, 2, 1])

        self.alpha = _tf.transpose(coef, [1, 0, 2]) * amp  # [m, size, K]
        self.chol_space = chol_space
        self.chol_time = chol_time
        self._state = {'alpha': self.alpha}

    def _coefficients(self, t, alpha):
        """The spatial coefficients at time `t`, `[m, size]`."""
        return _tf.einsum('msk,k->ms', alpha, self._time_cross(t))

    def _field(self, x, t, alpha):
        return _tf.matmul(self._space_cross(x), self._coefficients(t, alpha))

    def _field_and_divergence(self, x, t, alpha):
        rng = self.parameters['rng_space'].get_value()
        coef = self._coefficients(t, alpha)
        ks = self._space_cross(x)
        field = _tf.matmul(ks, coef)

        # d k(x, x_i) / d x_s is analytic, so the divergence needs no
        # estimator: differentiate the spatial factor and contract
        dif = x[:, None, :] - self.anchors[None, :, :]
        grad = -6.0 * dif / rng ** 2 * ks[:, :, None]
        divergence = _tf.einsum('nms,ms->n', grad, coef)
        return field, divergence


class TensorProductFlow(_ContinuousFlow):
    """A continuous-time flow on a gridded field with CP-structured weights.

    The field lives on the full `(size + 1)`-dimensional lattice -- a
    regular grid on each `[-5, 5]` axis crossed with the time knots -- and
    escapes the curse of dimensionality by never forming the lattice: the
    weight tensor is a rank-`R` canonical polyadic sum of per-axis
    vectors, `sum_r lambda_r a_r^(1) (x) ... (x) a_r^(size) (x) a_r^(t)
    (x) a_r^(c)` with a component axis, and because the kernel is
    separable the field factorizes through it into one-dimensional kernel
    sums: `O(R (size * grid + knots))` per evaluation, with the grid
    tensor's `grid^size * knots` entries never materialized. Whitening
    survives the structure too -- a Kronecker-factored operator acts
    axis-wise on CP factors -- so the coefficients cost one small
    triangular solve per axis. The divergence is exact, the derivative
    landing on one axis' factor at a time.

    A zero factor would gate its whole rank's gradients, so every factor
    initializes at unit scale and the per-rank amplitude `weights` alone
    holds the field near the identity at the start.
    """

    def __init__(self, size, grid=9, rank=5, n_steps=10, rtol=1e-6):
        """
        Initializer for TensorProductFlow.

        Parameters
        ----------
        size
            Number of components. The flow exists to mix them.
        grid
            Nodes per axis of the regular `[-5, 5]` grid.
        rank
            Number of canonical-polyadic terms -- the capacity knob.
        n_steps
            Number of time knots of the field over `[0, 1]`.
        rtol
            Relative tolerance of the solver, shared by both directions
            and by the log-determinant.
        """
        super().__init__(size, n_steps=n_steps, rtol=rtol)
        self.grid = grid
        self.rank = rank
        self.nodes = _tf.constant(_np.linspace(-5.0, 5.0, grid), _tf.float64)

        rng = _rnd.rng()
        self._add_parameter(
            'factors_space',
            _gpr.RealParameter(
                rng.normal(size=[size, grid, rank]),
                _np.full([size, grid, rank], -10.0),
                _np.full([size, grid, rank], 10.0)))
        self._add_parameter(
            'factors_time',
            _gpr.RealParameter(
                rng.normal(size=[n_steps, rank]),
                _np.full([n_steps, rank], -10.0),
                _np.full([n_steps, rank], 10.0)))
        self._add_parameter(
            'factors_component',
            _gpr.RealParameter(
                rng.normal(size=[size, rank]),
                _np.full([size, rank], -10.0),
                _np.full([size, rank], 10.0)))
        self._add_parameter(
            'weights',
            _gpr.RealParameter(
                rng.normal(scale=1e-2, size=[rank]),
                _np.full([rank], -10.0), _np.full([rank], 10.0)))
        self._add_parameter(
            'rng_space',
            _gpr.PositiveParameter(
                _np.full([size], 3.0), _np.full([size], 0.5),
                _np.full([size], 20.0)))

    def _axis_cross(self, x):
        """The one-dimensional kernel of every coordinate against the grid,
        `[n, size, grid]`."""
        rng = self.parameters['rng_space'].get_value()
        dif = (x[:, :, None] - self.nodes[None, None, :]) / rng[None, :, None]
        return _tf.exp(-3.0 * dif ** 2)

    def refresh(self):
        rng = self.parameters['rng_space'].get_value()
        dif = (self.nodes[:, None] - self.nodes[None, :])[None, :, :] \
            / rng[:, None, None]
        cov = _tf.exp(-3.0 * dif ** 2) \
            + _tf.eye(self.grid, dtype=_tf.float64)[None] * 1e-6
        chol_space = _tf.linalg.cholesky(cov)  # [size, grid, grid]
        chol_time = self._time_cholesky()

        # whitening acts axis by axis on the factors and keeps the rank
        c_space = _tf.linalg.triangular_solve(
            chol_space, self.parameters['factors_space'].get_value(),
            lower=True, adjoint=True)  # [size, grid, rank]
        c_time = _tf.linalg.triangular_solve(
            chol_time, self.parameters['factors_time'].get_value(),
            lower=True, adjoint=True)  # [knots, rank]
        components = self.parameters['factors_component'].get_value() \
            * self.parameters['weights'].get_value()[None, :]  # [size, rank]

        self._state = {'c_space': c_space, 'c_time': c_time,
                       'components': components}

    def _pieces(self, x, t, c_space, c_time):
        axis = self._axis_cross(x)
        along = _tf.einsum('ndm,dmr->ndr', axis, c_space)  # [n, size, rank]
        in_time = _tf.einsum('k,kr->r', self._time_cross(t), c_time)
        return axis, along, in_time

    def _field(self, x, t, c_space, c_time, components):
        _, along, in_time = self._pieces(x, t, c_space, c_time)
        product = _tf.reduce_prod(along, axis=1)  # [n, rank]
        return _tf.einsum('nr,r,sr->ns', product, in_time, components)

    def _field_and_divergence(self, x, t, c_space, c_time, components):
        rng = self.parameters['rng_space'].get_value()
        axis, along, in_time = self._pieces(x, t, c_space, c_time)
        product = _tf.reduce_prod(along, axis=1)
        field = _tf.einsum('nr,r,sr->ns', product, in_time, components)

        # d f_s / d x_s differentiates the s-th axis' factor and leaves the
        # others' product alone: exact, one axis at a time
        dif = (x[:, :, None] - self.nodes[None, None, :]) \
            / rng[None, :, None] ** 2
        d_along = _tf.einsum('ndm,dmr->ndr', -6.0 * dif * axis, c_space)
        divergence = _tf.zeros([_tf.shape(x)[0]], _tf.float64)
        for s in range(self.size_in):
            others = _tf.reduce_prod(
                _tf.concat([along[:, :s], along[:, s + 1:]], axis=1), axis=1)
            divergence = divergence + _tf.einsum(
                'nr,nr,r,r->n', d_along[:, s], others, in_time,
                components[s])
        return field, divergence


class PCA(_Warping):
    _mixes = True

    def __init__(self, n_dim, n_components=None):
        super().__init__()
        if n_components is None:
            n_components = n_dim
        self._size_in = n_dim
        self._size_out = n_components

        self.mean = None
        self.eigvals = None
        self.eigvecs = None

    def forward(self, x):
        x = x - self.mean
        x = _tf.matmul(x, self.eigvecs)
        x = x / _tf.sqrt(self.eigvals)

        # The rotation preserves volume and the scaling does not, so the map
        # multiplies it by prod(eigvals) ** -0.5 -- a constant, settled when
        # the warping was initialized. With fewer components than variables
        # there is no square Jacobian and so no determinant to report: the
        # map projects, and what the likelihood sees is that projection.
        log_det = _tf.reduce_sum(_tf.zeros_like(x), axis=1)
        if self.size_out == self.size_in:
            log_det = log_det \
                - 0.5 * _tf.reduce_sum(_tf.math.log(self.eigvals))
        return x, log_det

    def backward(self, x):
        x = x * _tf.sqrt(self.eigvals)
        x = _tf.matmul(x, self.eigvecs, False, True)
        x = x + self.mean
        return x

    def initialize(self, x, weights=None):
        self.mean = _tf.constant(_np.mean(x, axis=0, keepdims=True), _tf.float64)
        x_center = x - self.mean
        cov = _np.matmul(_np.transpose(x_center), x_center) / x_center.shape[0]
        vals, vecs = _np.linalg.eigh(cov)
        self.eigvals = _tf.constant(vals[::-1][None, :self.size_out], _tf.float64)
        self.eigvecs = _tf.constant(vecs[:, ::-1][:, :self.size_out], _tf.float64)

        x, _ = self.forward(x)
        return x


class RobustPCA(PCA):
    def __init__(self, n_dim, n_components=None, support_fraction=0.75):
        super().__init__(n_dim, n_components)
        self.support_fraction = support_fraction

    def initialize(self, x, weights=None):
        with _warnings.catch_warnings():
            # FastMCD chatters while it searches: "Determinant has increased"
            # on its concentration steps, and a not-full-rank notice on the
            # final estimate. Neither is actionable here -- the fit is an
            # initialization -- so only these two stay off the console;
            # anything else sklearn says still comes through.
            _warnings.filterwarnings(
                "ignore", message="Determinant has increased")
            _warnings.filterwarnings(
                "ignore",
                message="The covariance matrix associated to your dataset "
                        "is not full rank")
            mcd = _MCD(support_fraction=self.support_fraction).fit(x)
        self.mean = _tf.constant(mcd.location_[None, :], _tf.float64)
        cov = mcd.covariance_
        vals, vecs = _np.linalg.eigh(cov)
        self.eigvals = _tf.constant(vals[::-1][None, :self.size_out], _tf.float64)
        self.eigvecs = _tf.constant(vecs[:, ::-1][:, :self.size_out], _tf.float64)

        x, _ = self.forward(x)
        return x


class CenteredLogRatio(_Warping):
    """
    The centered log-ratio transformation of compositional data.

    Takes the logarithm of each part and subtracts the row's mean log, which
    frees the composition from the constraint that its parts sum to one. The
    inverse is the softmax.

    Notes
    -----
    The transformation maps the simplex onto the hyperplane where the
    components sum to zero, and both are one dimension smaller than the
    number of parts. The log-determinant reported is the volume factor of
    that map, so the objective is a density on the hyperplane: values are
    comparable between compositional models and only loosely against a
    warping that transforms a variable one to one.
    """
    _mixes = True

    def __init__(self, n_dim):
        super().__init__()
        self._size_in = n_dim
        self._size_out = n_dim

    def forward(self, x):
        # trick to convert dtype
        x = x + _tf.constant(0.0, _tf.float64)

        x_log = _tf.math.log(x)
        x_log = x_log - _tf.reduce_mean(x_log, axis=1, keepdims=True)

        # The transformation ignores a rescaling of the whole row, so between
        # arrays of n_dim columns its Jacobian is singular and has no
        # determinant at all. Read as the bijection it is -- from the simplex
        # to the hyperplane where the components sum to zero, both of them
        # one dimension smaller -- the volume factor is this. Unlike the
        # other constants here it varies with the row, growing without bound
        # as a part approaches zero, where the transformation stretches. It
        # is the same whichever orthonormal basis of that hyperplane it is
        # measured in, which is why no balance matrix has to be chosen (nor
        # defended) to state it.
        log_det = - _np.log(self.size_in) \
            - _tf.reduce_sum(_tf.math.log(x), axis=1)
        return x_log, log_det

    def backward(self, x):
        return _tf.nn.softmax(x, axis=1)


class Rotation(Identity):
    """
    An orthogonal rotation of the variables.

    Multiplies the data by a square orthonormal matrix, which is a trainable
    parameter. The transformation is volume preserving, so the log-determinant
    it contributes is zero and the rotation neither stretches nor compresses
    the density.

    The matrix is **initialized by independent component analysis**
    (`sklearn.decomposition.FastICA`), which positions the axes along the
    directions of maximum non-Gaussianity in the data. This makes it the
    natural partner of a per-component transformation placed after it: the
    rotation finds the directions along which the marginals depart most from
    a Gaussian, and the following warping is then applied where that departure
    lives. `Rotation` followed by `Spline` is the usual pairing.

    Parameters
    ----------
    n_dim : int
        Number of variables, and the size of the rotation matrix.
    fixed : bool
        Whether to keep the matrix at its initial value instead of training
        it. The ICA initialization is used either way.

    Notes
    -----
    This warping mixes its inputs, so `elementwise` is False for any chain
    containing it, and the likelihood integrates its noise over Sobol points
    rather than per-column Gauss-Hermite nodes.

    The ICA fit is a starting point rather than a result: training moves the
    matrix from wherever ICA stopped, so a fit that reaches the iteration
    limit is not an error and its convergence warning is suppressed.

    References
    ----------
    Hyvärinen, A., & Oja, E. (2000). Independent component analysis:
    algorithms and applications. Neural Networks, 13(4-5), 411-430.

    See Also
    --------
    PCA, RobustPCA : decorrelating transformations, by variance rather than
        by non-Gaussianity.
    Spline : the per-component warping usually placed after this one.
    """
    _mixes = True

    def __init__(self, n_dim, fixed=False):
        super().__init__(n_dim)

        self._add_parameter(
            'rotation',
            _gpr.OrthonormalMatrix(n_dim, n_dim)
        )
        self.parameters['rotation'].set_value(_np.eye(n_dim))
        if fixed:
            self.parameters['rotation'].fix()

    def forward(self, x):
        rot = self.parameters['rotation'].get_value()
        x = _tf.matmul(x, rot)
        log_det = _tf.reduce_sum(_tf.zeros_like(x), axis=1)
        return x, log_det

    def backward(self, x):
        rot = self.parameters['rotation'].get_value()
        x = _tf.matmul(x, rot, False, True)
        return x

    def initialize(self, x, weights=None):
        # ICA is asked for a starting point, not a converged answer: the
        # rotation is trainable, and training moves it from wherever the
        # fit stopped. Hitting the iteration limit therefore costs nothing
        # worth telling the user about, and the ConvergenceWarning it
        # raises is noise in a fit that is working as intended -- it fired
        # on the Jura case in the test suite, where the model is fine.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", _ConvergenceWarning)
            # seeded from the package RNG: unseeded, FastICA draws from
            # numpy's global state, which made this the one data-dependent
            # start `geoml.set_seed` did not reproduce -- and left a test
            # threshold riding on whatever ran before it
            ica = _ICA(whiten=False,
                       random_state=int(_rnd.rng().integers(2 ** 31))
                       ).fit(x)
        self.parameters['rotation'].set_value(ica.components_)
        rot = self.parameters['rotation'].get_value()
        x = _tf.matmul(x, rot)
        return x


class ScaledSimplex(Identity):
    """
    Compositional data transformation without log-ratios.

    Accepts zeros.
    """
    _mixes = True

    def __init__(self, size):
        super().__init__(size)
        self.scale = None

    def forward(self, x):
        # each part divided by a constant of its own, so the Jacobian is
        # diagonal and the log-determinant is minus the log of the scales
        log_det = _tf.reduce_sum(_tf.zeros_like(x), axis=1) \
            - _tf.reduce_sum(_tf.math.log(self.scale))
        return x / self.scale, log_det

    def backward(self, x):
        x = x * self.scale
        total = _tf.reduce_sum(x, axis=1, keepdims=True)
        x = x + (1.0 - total) * self.scale

        denom = _tf.where(_tf.less(x, 0.0), self.scale - x, 1.0)
        shift = _tf.where(_tf.less(x, 0.0), -x / denom, 0.0)

        max_s = _tf.reduce_max(shift, axis=1, keepdims=True)
        x = (1.0 - max_s) * x + max_s * self.scale
        x = _tf.maximum(x, 0.0)
        return x

    def initialize(self, x, weights=None):
        missing = _np.isnan(x)
        complete = ~_np.any(missing, axis=1, keepdims=True)
        x_new = _np.where(missing, 0, x) * complete
        scale = _np.sum(x_new, axis=0, keepdims=True) / _np.sum(complete)
        self.scale = _tf.constant(scale, _tf.float64)
        return x_new / self.scale
