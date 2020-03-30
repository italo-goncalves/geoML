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

import numpy as _np
import tensorflow as _tf


class _Warping(object):
    """
    Base warping class.

    Attributes
    ----------
    parameters : dict
        matrix dictionary with Parameter objects.
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
    
    def refresh(self, y):
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
        return _tf.ones_like(x)


class Spline(_Warping):
    """
    Uses a monotonic spline to convert from original to warped space and
    back.

    Attributes
    ----------
    n_knots : int
        The number of knots to build the spline.
    base_spline :
        matrix spline object to convert from the original to warped space.
    reverse_spline :
        matrix spline object to convert from the warped to original space.
    """
    def __init__(self, n_knots=5):
        """
        Initializer for Spline.

        Parameters
        ----------
        n_knots : int
            The number of knots to build the spline
        """
        super().__init__()
        self.n_knots = n_knots
        self.parameters = {
            "warped_partition": _gpr.CompositionalParameter(
                _np.ones([n_knots + 1]) / (n_knots + 1))}
        self._all_parameters = [pr for pr in self.parameters.values()]

        dummy_vals = _tf.range(0, n_knots, dtype=_tf.float64) / n_knots
        self.base_spline = _gint.MonotonicCubicSpline(dummy_vals, dummy_vals)
        self.reverse_spline = _gint.MonotonicCubicSpline(dummy_vals, dummy_vals)
        
    def refresh(self, y):
        """
        Rebuilds the spline using the given data's range.

        The range of the warped coordinates is fixed within the [-5, 5]
        interval.
        """
        y_seq = _tf.linspace(_tf.reduce_min(y), _tf.reduce_max(y), self.n_knots)
        warped_coordinates = _tf.cumsum(
            self.parameters["warped_partition"].value)[:self.n_knots]
        warped_coordinates = 10*warped_coordinates - 5

        self.base_spline.refresh(y_seq, warped_coordinates)

        # modeling in the opposite direction with more points
        # to preserve the base curve's shape
        y_dif = _tf.reduce_max(y) - _tf.reduce_min(y)
        y_seq_expanded = _tf.linspace(_tf.reduce_min(y) - 0.1*y_dif,
                                      _tf.reduce_max(y) + 0.1*y_dif,
                                      self.n_knots*10)
        warped_coordinates_expanded = self.base_spline.interpolate(
            y_seq_expanded)
        self.reverse_spline.refresh(
            warped_coordinates_expanded, y_seq_expanded)
    
    def forward(self, x):
        return self.base_spline.interpolate(x)
    
    def backward(self, x):
        return self.reverse_spline.interpolate(x)
    
    def derivative(self, x):
        return self.base_spline.interpolate_d1(x)


class ZScore(_Warping):
    """
    matrix Warping that simply normalizes the values to z-scores.
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
        self.def_mean = mean
        self.def_std = std
        self.y_mean = None
        self.y_std = None
    
    def refresh(self, y):
        if self.def_mean is None:
            self.y_mean = _tf.reduce_mean(y)
        else:
            self.y_mean = _tf.constant(self.def_mean, _tf.float64)
        if self.def_std is None:
            self.y_std = _tf.sqrt(_tf.reduce_mean((y - self.y_mean)**2))
        else:
            self.y_std = _tf.constant(self.def_std, _tf.float64)
        
    def forward(self, x):
        return (x - self.y_mean) / self.y_std
    
    def backward(self, x):
        return x * self.y_std + self.y_mean
    
    def derivative(self, x):
        return _tf.ones_like(x) / self.y_std


class Softplus(_Warping):
    """
    Transforms the data using the inverse of the softplus function. 
    All the data must be positive. Negative values will be replaced with
    half of the smallest positive value.
    """
    # computation only for x < 50.0 to avoid overflow
    def forward(self, x):
        x_positive = _tf.gather(x, _tf.squeeze(_tf.where(_tf.greater(x, 0.0))))
        x_warp = _tf.maximum(x, 0.5 * _tf.reduce_min(x_positive))

        x_warp = _tf.where(_tf.greater(x_warp, 50.0),
                           x_warp,
                           _tf.math.log(_tf.math.expm1(x_warp)))
        return x_warp
    
    def backward(self, x):
        x_warp = _tf.where(_tf.greater(x, 50.0),
                           x,
                           _tf.math.log1p(_tf.math.exp(x)))
        return x_warp
    
    def derivative(self, x):
        x_positive = _tf.gather(x, _tf.squeeze(_tf.where(_tf.greater(x, 0.0))))
        x_warp = _tf.maximum(x, 0.5 * _tf.reduce_min(x_positive))

        x_warp = _tf.where(_tf.greater(x_warp, 50.0),
                           _tf.ones_like(x_warp),
                           1/(- _tf.math.expm1(-x_warp)))
        return x_warp


class Log(_Warping):
    """
    Log-scale warping.
    """
    def __init__(self, shift=None):
        """
        Initializer for Log.

        Parameters
        ----------
        shift : float
            matrix positive value to add to the data. Use it if you have zeros.
            The default is half of the smallest positive value in data.
        """
        super().__init__()
        self.shift = shift
        if shift is not None:
            self.shift = _tf.constant(shift, _tf.float64)

    def forward(self, x):
        return _tf.math.log(x + self.shift)

    def backward(self, x):
        return _tf.math.exp(x)

    def derivative(self, x):
        return 1 / (x + self.shift)

    def refresh(self, y):
        if self.shift is None:
            y_positive = _tf.gather(y,
                                    _tf.squeeze(_tf.where(_tf.greater(y, 0.0))))
            self.shift = 0.5 * _tf.reduce_min(y_positive)


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
        for wp in self.warpings:
            x = wp.forward(x)
        return x

    def backward(self, x):
        warping_rev = self.warpings.copy()
        warping_rev.reverse()
        for wp in warping_rev:
            x = wp.backward(x)
        return x

    def refresh(self, y):
        for wp in self.warpings:
            wp.refresh(y)
            y = wp.forward(y)

    def derivative(self, x):
        d = _tf.ones_like(x)
        for wp in self.warpings:
            d = d * wp.derivative(x)
            x = wp.forward(x)
        return d
