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
# MERCHANTABILITY or FITNESS FOR matrix PARTICULAR PURPOSE.  See the
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


class _Warping(object):
    """
    Base warping class.

    Attributes
    ----------
    parameters : dict
        matrix dictionary with RealParameter objects.
    """
    def __init__(self):
        self.parameters = {}
        self._all_parameters = [pr for pr in self.parameters.values()]

    @property
    def all_parameters(self):
        return self._all_parameters

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        return s
        
    def forward(self, x):
        pass
    
    def backward(self, x):
        pass
    
    def derivative(self, x):
        pass

    def get_parameter_values(self, complete=False):
        value = []
        shape = []
        position = []
        min_val = []
        max_val = []

        for index, parameter in enumerate(self._all_parameters):
            if (not parameter.fixed) | complete:
                value.append(_tf.reshape(parameter.value_transformed, [-1]).
                                 numpy())
                shape.append(_tf.shape(parameter.value_transformed).numpy())
                position.append(index)
                min_val.append(_tf.reshape(parameter.min_transformed, [-1]).
                               numpy())
                max_val.append(_tf.reshape(parameter.max_transformed, [-1]).
                               numpy())

        if len(value) > 0:
            min_val = _np.concatenate(min_val, axis=0)
            max_val = _np.concatenate(max_val, axis=0)
            value = _np.concatenate(value, axis=0)
        else:
            min_val = _np.array(min_val)
            max_val = _np.array(max_val)
            value = _np.array(value)

        return value, shape, position, min_val, max_val

    def update_parameters(self, value, shape, position):
        sizes = _np.array([int(_np.prod(sh)) for sh in shape])
        value = _np.split(value, _np.cumsum(sizes))[:-1]
        value = [_np.squeeze(val) if len(sh) == 0 else val
                 for val, sh in zip(value, shape)]

        for val, sh, pos in zip(value, shape, position):
            self._all_parameters[pos].set_value(
                _np.reshape(val, sh) if len(sh) > 0 else val,
                transformed=True
            )


class Identity(_Warping):
    """Identity warping"""
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
        self.parameters = {
            "warped_partition_left": _gpr.CompositionalParameter(comp),
            "warped_partition_right": _gpr.CompositionalParameter(comp)
        }
        self._all_parameters = [pr for pr in self.parameters.values()]
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
        self.parameters["mean"] = _gpr.RealParameter(0, -1e9, 1e9)
        if mean is not None:
            self.parameters["mean"].set_value(mean)
            self.parameters["mean"].set_limits(mean - 2*_np.abs(mean),
                                               mean + 2*_np.abs(mean))
        self.parameters["std"] = _gpr.PositiveParameter(1, 1e-9, 1e9)
        if std is not None:
            self.parameters["std"].set_value(std)
            self.parameters["std"].set_limits(std / 10, std * 10)
        self._all_parameters.append(self.parameters["mean"])
        self._all_parameters.append(self.parameters["std"])
        
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
    def __init__(self, scale):
        super().__init__(mean=-1e-3, std=scale)
        self.parameters["mean"].fix()


class ChainedWarping(_Warping):
    def __init__(self, *warpings):
        super().__init__()
        count = -1
        for wp in warpings:
            count += 1
            names = list(wp.parameters.keys())
            names = [s + "_" + str(count) for s in names]
            self.parameters.update(zip(names, wp.parameters.values()))
        self.warpings = list(warpings)
        self._all_parameters = [wp.all_parameters for wp in warpings]
        self._all_parameters = [item for sublist in self._all_parameters
                                for item in sublist]

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
