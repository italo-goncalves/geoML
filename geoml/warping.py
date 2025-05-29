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
           "ChainedWarping"]

import geoml.interpolation as _gint
import geoml.parameter as _gpr
import geoml.tftools as _tftools

import numpy as _np
import tensorflow as _tf


class _Warping(_gpr.Parametric):
    """
    Base warping class.
    """

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        return s
        
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
        """
        pass
    
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
        pass
    
    def derivative(self, x):
        """
        Computes the derivative of the warping function.

        Parameters
        ----------
        x : array-like
            Vector with values to warp.

        Returns
        -------
        x : array-like
            Vector with the derivatives.
        """
        pass

    def initialize(self, x):
        """
        Uses the provided values to initialize the object's parameters.

        Parameters
        ----------
        x : array-like
            Vector with values to warp.

        Returns
        -------
        x : array-like
            Vector with warped values.
        """
        return self.forward(x)


class Identity(_Warping):
    """Identity warping."""
    def forward(self, x):
        return x
    
    def backward(self, x):
        return x
    
    def derivative(self, x):
        x_warp = _tf.ones_like(x)
        x_warp = _tf.where(_tf.math.is_nan(x), x, x_warp)
        return x_warp


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
    def __init__(self, knots_per_arm=5):
        """
        Initializer for Spline.

        Parameters
        ----------
        knots_per_arm : int
            The number of knots used to build each side (positive and negative)
            of the spline.
        """
        super().__init__()
        self.n_knots = knots_per_arm * 2 + 1
        comp = _np.ones(knots_per_arm) / knots_per_arm
        self._add_parameter("warped_partition_left",
                            _gpr.CompositionalParameter(comp))
        self._add_parameter("warped_partition_right",
                            _gpr.CompositionalParameter(comp))
        self.spline = _gint.MonotonicCubicSpline()
        self.x_original = _np.linspace(-5, 5, knots_per_arm * 2 + 1)

    def _get_warped_coordinates(self):
        warped_left = _tf.cumsum(
            self.parameters["warped_partition_left"].get_value())
        warped_right = _tf.cumsum(
            self.parameters["warped_partition_right"].get_value()) + 1.0
        warped_coordinates = _tf.concat(
            [_tf.constant([0.0], _tf.float64), warped_left, warped_right],
            axis=0) / 2
        warped_coordinates = 10 * warped_coordinates - 5
        return warped_coordinates
    
    def forward(self, x):
        warped_coordinates = self._get_warped_coordinates()

        x = _tftools.ensure_rank_2(x)
        x_warp = self.spline.interpolate(
            self.x_original, warped_coordinates, x)
        return x_warp
    
    def backward(self, x):
        warped_coordinates = self._get_warped_coordinates()

        x = _tftools.ensure_rank_2(x)
        x_back = self.spline.interpolate(
            warped_coordinates, self.x_original, x)
        return x_back
    
    def derivative(self, x):
        warped_coordinates = self._get_warped_coordinates()

        x = _tftools.ensure_rank_2(x)
        x_warp = self.spline.interpolate_d1(
            self.x_original, warped_coordinates, x)
        return x_warp


class ZScore(_Warping):
    """
    A Warping that simply normalizes the values to z-scores.
    """
    def __init__(self, mean=None, std=None):
        """
        Initializer for ZScore.

        Parameters
        ----------
        mean : double
            The desired mean of the data.
        std : double
            The desired standard deviation of the data.

        The mean and standard deviation can be computed from the data
        (if omitted) or specified.
        """
        super().__init__()
        self._add_parameter("mean", _gpr.RealParameter(0, -1e9, 1e9))
        if mean is not None:
            self.parameters["mean"].set_value(mean)
            # self.parameters["mean"].set_limits(mean - 2*_np.abs(mean),
            #                                    mean + 2*_np.abs(mean))

        self._add_parameter("std", _gpr.PositiveParameter(1, 1e-9, 1e9))
        if std is not None:
            self.parameters["std"].set_value(std)
            self.parameters["std"].set_limits(std / 100, std * 10)
        
    def forward(self, x):
        mean = self.parameters["mean"].get_value()
        std = self.parameters["std"].get_value()
        x = _tftools.ensure_rank_2(x)
        return (x - mean) / std
    
    def backward(self, x):
        mean = self.parameters["mean"].get_value()
        std = self.parameters["std"].get_value()
        x = _tftools.ensure_rank_2(x)
        return x * std + mean
    
    def derivative(self, x):
        std = self.parameters["std"].get_value()
        x = _tftools.ensure_rank_2(x)
        return _tf.ones_like(x) / std

    def initialize(self, x):
        mean = _np.mean(x)
        std = _np.std(x)
        self.parameters["mean"].set_value(mean)
        self.parameters["std"].set_value(std)
        self.parameters["mean"].set_limits(mean - 3*std, mean + 3*std)
        self.parameters["std"].set_limits(std / 100, std * 10)
        return super().initialize(x)


class Center(_Warping):
    """
    A Warping that simply centers the data.
    """

    def __init__(self, mean=None):
        """
        Initializer for Center.

        The mean can be computed from the data (if omitted) or specified.

        Parameters
        ----------
        mean : double
            The desired mean of the data.
        """
        super().__init__()
        self._add_parameter("mean", _gpr.RealParameter(0, -1e9, 1e9))
        if mean is not None:
            self.parameters["mean"].set_value(mean)

    def forward(self, x):
        mean = self.parameters["mean"].get_value()
        x = _tftools.ensure_rank_2(x)
        return x - mean

    def backward(self, x):
        mean = self.parameters["mean"].get_value()
        x = _tftools.ensure_rank_2(x)
        return x + mean

    def derivative(self, x):
        x = _tftools.ensure_rank_2(x)
        return _tf.ones_like(x)

    def initialize(self, x):
        mean = _np.mean(x)
        self.parameters["mean"].set_value(mean)
        self.parameters["mean"].set_limits(mean - 3, mean + 3)
        return super().initialize(x)


class Softplus(_Warping):
    """
    Transforms the data using the inverse of the softplus function. 
    All the data must be positive.
    """
    def __init__(self, shift=1e-6):
        """
        Initializer for Softplus.

        Parameters
        ----------
        shift : float
            A positive value to add to the data. Use it if you have zeros.
        """
        super().__init__()
        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    # computation only for x < 50.0 to avoid overflow
    def forward(self, x):
        x = _tftools.ensure_rank_2(x)
        x_warp = x + self.shift
        x_warp = _tf.where(_tf.greater(x_warp, 50.0),
                           x_warp,
                           _tf.math.log(_tf.math.expm1(x_warp)))
        x_warp = _tf.where(_tf.math.is_nan(x), x, x_warp)
        return x_warp
    
    def backward(self, x):
        x = _tftools.ensure_rank_2(x)
        x_back = _tf.where(_tf.greater(x, 50.0),
                           x,
                           _tf.math.log1p(_tf.math.exp(x)))
        return x_back
    
    def derivative(self, x):
        x = _tftools.ensure_rank_2(x)
        x_warp = x + self.shift
        x_warp = _tf.where(_tf.greater(x_warp, 50.0),
                           _tf.ones_like(x_warp),
                           1/(- _tf.math.expm1(-x_warp)))
        return x_warp


class Log(_Warping):
    """
    Log-scale warping.

    Forward function: log
    Backward function: exp
    """
    def __init__(self, shift=1e-6):
        """
        Initializer for Log.

        Parameters
        ----------
        shift : float
            A positive value to add to the data. Use it if you have zeros.
        """
        super().__init__()
        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    def forward(self, x):
        return _tf.math.log(x + self.shift)

    def backward(self, x):
        return _tf.math.exp(x)

    def derivative(self, x):
        return 1 / (x + self.shift)


class Scale(ZScore):
    """Linear scaling, assuming a mean of zero."""
    def __init__(self, scale):
        super().__init__(mean=-1e-3, std=scale)
        self.parameters["mean"].fix()

    def initialize(self, x):
        return self.forward(x)


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

    def __repr__(self):
        s = "".join([repr(wp) for wp in self.warpings])
        return s

    def forward(self, x):
        x = _tftools.ensure_rank_2(x)
        for wp in self.warpings:
            x = wp.forward(x)
        return x

    def backward(self, x):
        x = _tftools.ensure_rank_2(x)
        warping_rev = self.warpings.copy()
        warping_rev.reverse()
        for wp in warping_rev:
            x = wp.backward(x)
        return x

    def derivative(self, x):
        x = _tftools.ensure_rank_2(x)
        d = _tf.ones_like(x)
        for wp in self.warpings:
            d = d * wp.derivative(x)
            x = wp.forward(x)
        return d

    def initialize(self, x):
        x = _tftools.ensure_rank_2(x)
        for wp in self.warpings:
            x = wp.initialize(x)
        return x


class Sigmoid(_Warping):
    """
    Sigmoid warping, for values constrained to the ]0, 1[ interval.

    Forward function: inverse sigmoid
    Backward function: sigmoid
    """

    def __init__(self, shift=1e-6):
        """
        Initializer for Sigmoid.

        Parameters
        ----------
        shift : float
            A positive value to ensure the data is constrained to the ]0, 1[ interval.
        """
        super().__init__()
        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    def forward(self, x):
        x = x * (1 - 2 * self.shift) + self.shift
        return - _tf.math.log(1 / x - 1)

    def backward(self, x):
        return 1 / (1 + _tf.math.exp(-x))

    def derivative(self, x):
        x = x * (1 - 2 * self.shift) + self.shift
        return 1 / (x - x**2)
