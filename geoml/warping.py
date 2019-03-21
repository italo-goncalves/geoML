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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["Identity", "Spline", "ZScore", "Scaling", "Softplus"]

import numpy as _np
import geoml.interpolation as _gint
import geoml.parameter as _gpr


class _Warping(object):
    """
    Base warping class.

    Attributes
    ----------
    params : dict
        A dictionary with Parameter objects.
    """
    def __init__(self):
        self.params = {}
        
    def forward(self, x):
        pass
    
    def backward(self, x):
        pass
    
    def refresh(self, y):
        pass
    
    def derivative(self, x):
        pass


class Identity(_Warping):
    """Identity warping"""
    def forward(self, x):
        return x
    
    def backward(self, x):
        return x
    
    def derivative(self, x):
        return _np.repeat(1, len(x))


class Spline(_Warping):
    """
    Uses a monotonic spline to convert from original to warped space and
    back.

    Attributes
    ----------
    n_knots : int
        The number of knots to build the spline.
    base_spline :
        A spline object to convert from the original to warped space.
    reverse_spline :
        A spline object to convert from the warped to original space.
    """
    def __init__(self, n_knots):
        """
        Initializer for Spline.

        Parameters
        ----------
        n_knots : int
            The number of knots to build the spline
        """
        self.n_knots = n_knots
        default_seq = _np.linspace(-3, 3, n_knots)
        default_seq = _np.concatenate([[default_seq[0]], _np.diff(default_seq)])
        self.params = {
            "warp": _gpr.Parameter(
                default_seq,
                min_val=_np.concatenate([[-5], _np.repeat(0.1, n_knots - 1)]),
                max_val=_np.concatenate([[0], _np.repeat(1, n_knots - 1)])),
            "original": _gpr.Parameter(default_seq,
                                       min_val=default_seq,
                                       max_val=default_seq,
                                       fixed=True)}
        self.base_spline = None
        self.reverse_spline = None
        
    def refresh(self, y):
        """Rebuilds the spline using the given data's range."""
        y_seq = _np.linspace(_np.min(y), _np.max(y), self.n_knots)
        self.params["original"].set_value(y_seq)
        warp = self.params["warp"].value.cumsum()
        orig = y_seq
        self.base_spline = _gint.MonotonicSpline(orig, warp)
        # modeling in the opposite direction with more points
        # to preserve the base curve's shape
        orig_swap = _np.linspace(
                orig[0] - 0.1*(orig[-1] - orig[0]),
                orig[-1] + 0.1*(orig[-1] - orig[0]),
                10 * orig.size)
        warp_swap = self.base_spline(orig_swap)
        self.reverse_spline = _gint.MonotonicSpline(warp_swap, orig_swap)
#        self.reverse_spline = gint.MonotonicSpline(warp, orig)
    
    def forward(self, x):
        return self.base_spline(x)
    
    def backward(self, x):
        return self.reverse_spline(x)
    
    def derivative(self, x):
        return self.base_spline(x, n_deriv=1)
    
#    def set_limits(self, data):
#        s = self.params["warp"].value.size
#        self.params["warp"].set_limits(
#                min_val = np.concatenate([[-5], np.repeat(0.1, s - 1)]),
#                max_val = np.concatenate([[0], np.repeat(1, s - 1)]))


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

        The mean and standard deviation can be computed from the data (if omitted) or specified.
        """
        super().__init__()
        self.def_mean = mean
        self.def_std = std
        self.y_mean = None
        self.y_std = None
    
    def refresh(self, y):
        if self.def_mean is None:
            self.y_mean = y.mean()
        else:
            self.y_mean = self.def_mean
        if self.def_std is None:
            self.y_std = y.std()
        else:
            self.y_std = self.def_std
        
    def forward(self, x):
        return (x - self.y_mean) / self.y_std
    
    def backward(self, x):
        return x * self.y_std + self.y_mean
    
    def derivative(self, x):
        return _np.repeat(1 / self.y_std, len(x))


class Softplus(_Warping):
    """
    Transforms the data using the inverse of the softplus function. 
    All the data must be positive. Negative values will be replaced with
    half of the smallest positive value.
    """
    def forward(self, x):
        x = _np.maximum(x, 0.5 * _np.min(x[x > 0]))
        x_warp = x
        # computation only for x < 50.0 to avoid overflow
        x_warp[x_warp < 50] = _np.log(_np.expm1(x_warp[x_warp < 50]))
        return x_warp
    
    def backward(self, x):
        x_warp = x
        # computation only for x < 50.0 to avoid overflow
        x_warp[x_warp < 50] = _np.log1p(_np.exp(x_warp[x_warp < 50]))
        return x_warp
    
    def derivative(self, x):
        x = _np.maximum(x, 0.5 * _np.min(x[x > 0]))
        x_warp = _np.ones_like(x)
        # computation only for x < 50.0 to avoid overflow
        x_warp[x < 50] = 1/(-_np.expm1(-x[x < 50]))
        return x_warp


class Scaling(_Warping):
    """
    A Warping that normalizes the data to lie within the [0,1] interval.
    The positive option replaces the <=0 values with half of the smallest
    positive value.
    """
    def __init__(self, positive=False):
        """
        Initializer for ZScore.

        Parameters
        ----------
        positive : bool
            Whether to enforce positivity in the data. The <=0 values will be replaced with half of the smallest
            positive value.
        """
        super().__init__()
        self.positive = positive
        self.y_max = None
        self.y_min = None
    
    def refresh(self, y):
        self.y_max = y.max()
        self.y_min = y.min()
        
    def forward(self, x):
        if self.positive:
            x = _np.maximum(x, 0.5 * _np.min(x[x > 0]))
        return (x - self.y_min) / (self.y_max - self.y_min)
    
    def backward(self, x):
        return x * (self.y_max - self.y_min) + self.y_min
    
    def derivative(self, x):
        return _np.repeat(1 / (self.y_max - self.y_min), len(x))
