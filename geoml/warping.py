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
import warnings as _warnings

from scipy.special import ndtri as _ndtri
from sklearn.covariance import MinCovDet as _MCD
from sklearn.cluster import KMeans as _KMeans
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
    def __init__(self, size, knots_per_arm=5):
        """
        Initializer for Spline.

        Parameters
        ----------
        knots_per_arm : int
            The number of knots used to build each side (positive and negative)
            of the spline.
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
        self.spline = _gint.MonotonicCubicSpline()
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


class ContinuousNormalizingFlow(_Warping):
    _mixes = True

    # Built by `refresh` rather than by the constructor, which starts them at
    # None: the declarations say what they become, so the methods that index
    # them are not read as indexing None.
    base_ip: "_tf.Tensor | None"
    inducing_points: "_tf.Tensor"
    alpha: "_tf.Tensor"
    chol_space: "_tf.Tensor | None"
    chol_time: "_tf.Tensor | None"

    def __init__(self, size, inducing_points=20, n_steps=10, step=0.01):
        super().__init__()
        self._size_in = size
        self._size_out = size
        self.n_ip = inducing_points
        self.base_ip = None
        # self.ip_weight = None
        self.inducing_points = None
        self.n_steps = n_steps
        self.step = step

        self.alpha = None
        self.chol_space = None
        self.chol_time = None
        self.time = _tf.constant(_np.arange(self.n_steps)[:, None], _tf.float64)
        # self.mean = None
        # self.std = None

        self._add_parameter(
            'alpha_white',
            _gpr.RealParameter(
                _rnd.rng().normal(scale=1e-3, size=[inducing_points, size, n_steps]),
                _np.full([inducing_points, size, n_steps], -10),
                _np.full([inducing_points, size, n_steps], 10)
            )
        )
        self._add_parameter(
            'amp', _gpr.PositiveParameter(1, 0.01, 100, fixed=False)
        )
        self._add_parameter(
            'rng_space', _gpr.PositiveParameter(
                _np.ones([n_steps]),  _np.ones([n_steps]) * 0.1, _np.ones([n_steps]) * 10
            )
        )

    def covariance_matrix_space(self, x_1, x_2, t):
        rng_space = self.parameters['rng_space'].get_value()[t]
        dist_space = _tftools.pairwise_dist(x_1, x_2) / rng_space
        cov_space = _tf.exp(- 3 * dist_space ** 2)
        return cov_space

    def covariance_matrix_space_d1(self, x_1, x_2, t):
        rng_space = self.parameters['rng_space'].get_value()[t]
        dif = - (x_1[:, None, :] - x_2[None, :, :]) / rng_space  # [data, data, size]
        cov_space = self.covariance_matrix_space(x_1, x_2, t)  # [data, data]
        cov_d1 = 6 * cov_space[:, :, None] * dif
        return cov_d1

    def refresh(self):
        alpha_white = self.parameters['alpha_white'].get_value()
        amp = self.parameters['amp'].get_value()
        inducing_points = _tf.constant(self.base_ip, _tf.float64)
        all_ip = []
        fields = []
        all_cov_inv = []

        for i in range(self.n_steps):
            cov_space = self.covariance_matrix_space(inducing_points, inducing_points, i)
            cov_space = cov_space + _tf.eye(self.n_ip, dtype=_tf.float64) * 1e-6
            chol_space = _tf.linalg.cholesky(cov_space)
            cov_space_inv = _tf.linalg.cholesky_solve(
                chol_space,
                _tf.eye(self.n_ip, dtype=_tf.float64)
            )
            field = _tf.matmul(chol_space, alpha_white[:, :, i]) * amp

            # Midpoint
            x_mid = inducing_points + self.step / 2 * field
            cov_space = self.covariance_matrix_space(x_mid, x_mid, i)
            cov_space = cov_space + _tf.eye(self.n_ip, dtype=_tf.float64) * 1e-6
            chol_space = _tf.linalg.cholesky(cov_space)
            field_mid = _tf.matmul(chol_space, alpha_white[:, :, i]) * amp

            all_ip.append(inducing_points)
            fields.append(field)
            all_cov_inv.append(cov_space_inv)

            inducing_points = inducing_points + self.step * field_mid
        self.inducing_points = _tf.stack(all_ip, axis=-1)
        fields = _tf.stack(fields, axis=-1)
        cov_space_inv = _tf.stack(all_cov_inv, axis=0)

        alpha = _tf.einsum('top,pst->ost', cov_space_inv, fields) / amp**2
        self.alpha = alpha

        # last_ip = self.inducing_points[:, :, -1]
        # self.mean = _tf.reduce_sum(self.ip_weight[:, None] * last_ip, axis=0, keepdims=True)
        # self.std = _tf.sqrt(_tf.reduce_sum(self.ip_weight[:, None] * (last_ip - self.mean)**2,
        #                                    axis=0, keepdims=True))

    def get_field(self, x, t):
        amp = self.parameters['amp'].get_value()
        cov_space = self.covariance_matrix_space(x, self.inducing_points[:, :, t], t)
        field = _tf.einsum('op,ps->os', cov_space, self.alpha[:, :, t])
        return field * amp**2

    def get_gradient(self, x, t):
        amp = self.parameters['amp'].get_value()
        cov_space = self.covariance_matrix_space_d1(x, self.inducing_points[:, :, t], t)
        grad = _tf.einsum('ops,ps->os', cov_space, self.alpha[:, :, t])
        return grad * amp**2

    # def forward(self, x):
    #     self.refresh()
    #     for i in range(self.n_steps):
    #         # Midpoint
    #         field = self.get_field(x, i)
    #         x_mid = x + self.step / 2 * field
    #         field_mid = self.get_field(x_mid, i)
    #         x = x + self.step * field_mid
    #     # x = (x - self.mean) / self.std
    #     return x

    def backward(self, x):
        self.refresh()
        # x = x * self.std + self.mean
        for i in range(self.n_steps):
            j = self.n_steps - 1 - i
            # Midpoint
            field = self.get_field(x, j)
            x_mid = x - self.step / 2 * field
            field_mid = self.get_field(x_mid, j)
            x = x - self.step * field_mid
        return x

    def forward(self, x):
        self.refresh()
        grads = []
        norm = []
        for i in range(self.n_steps):
            # Midpoint
            field = self.get_field(x, i)
            x_mid = x + self.step / 2 * field
            field_mid = self.get_field(x_mid, i)
            grad = self.get_gradient(x_mid, i)
            grads.append(_tf.reduce_sum(grad, axis=1, keepdims=True))
            norm.append(_tf.reduce_sum(grad**2, axis=1, keepdims=True))
            x = x + self.step * field_mid
        total_grad = _tf.add_n(grads) * self.step
        total_norm = _tf.add_n(norm) * self.step

        log_det = _tf.reduce_sum(total_grad - total_norm * 0.5, axis=1)
        return x, log_det

    def flow_history(self, x):
        self.refresh()
        history = [x]
        for i in range(self.n_steps):
            field = self.get_field(x, i)
            x_mid = x + self.step / 2 * field
            field_mid = self.get_field(x_mid, i)
            x = x + self.step * field_mid
            history.append(x.numpy())
        return history

    def initialize(self, x, weights=None):
        cluster = _KMeans(self.n_ip).fit(x)

        cl_mean = _np.mean(cluster.cluster_centers_, axis=0, keepdims=True)

        self.base_ip = (cluster.cluster_centers_ - cl_mean) * 1.1 + cl_mean
        # self.ip_weight = _np.array([_np.sum(cluster.labels_ == i)
        #                             for i in range(self.n_ip)])
        # self.ip_weight = self.ip_weight / _np.sum(self.ip_weight)

        # x_min = _np.min(x, axis=0, keepdims=True)
        # x_max = _np.max(x, axis=0, keepdims=True)
        # self.base_ip = _np.random.uniform(x_min, x_max, size=[self.n_ip, self.size_out])

        x, _ = self.forward(x)
        return x


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
