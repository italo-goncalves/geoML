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

__all__ = ['cubic_conv_1d', 'MonotonicSpline']

import numpy as _np
import tensorflow as _tf
import scipy.interpolate as _interp


def cubic_conv_1d(x, xnew, output="numpy"):
    """
    Generates a sparse matrix for interpolating from a regular grid to
    a new set of positions in one dimension.
    
    Parameters
    ----------
    x : array
        Array of regularly spaced values.
    xnew : array
        Positions to receive the interpolated values. Its limits must be
        confined within the limits of x.
    output : str
        The output format. Either 'numpy' or 'tensorflow'.
    
    Returns
    -------
    w : a sparse matrix
        A standard numpy matrix or a SparseTensor, if output == 'tensorflow'.
    """
    if _np.std(_np.diff(x)) > 1e-9:
        raise Exception("array x not equally spaced")
    if (_np.min(xnew) < _np.min(x)) | (_np.max(xnew) > _np.max(x)):
        raise Exception("xnew out of bounds")
        
    # interval and data position
    h = _np.diff(x)[0]
    x_len = x.size
    s = (xnew - _np.min(x)) / h
    base_pos = _np.linspace(-1, x_len, x_len + 2)
    
    # weight matrix
    if output == "numpy":
        # vectorized version
        w = _np.abs(_np.resize(s, [x_len + 2, _np.size(s)]).transpose()
                    - _np.resize(base_pos, [_np.size(s), x_len + 2]))
        pos = w <= 1
        w[pos] = 1.5 * _np.power(w[pos], 3) - 2.5 * _np.power(w[pos], 2) + 1
        pos = (w > 1) & (w <= 2)
        w[pos] = - 0.5 * _np.power(w[pos], 3) + 2.5 * _np.power(w[pos], 2) - 4 * w[pos] + 2
        w[w > 2] = 0
        # borders
        w[:, 1] = w[:, 1] + 3*w[:, 0]
        w[:, 2] = w[:, 2] - 3*w[:, 0]
        w[:, 3] = w[:, 3] + 1*w[:, 0]
        w[:, x_len] = w[:, x_len] + 3 * w[:, x_len + 1]
        w[:, x_len - 1] = w[:, x_len - 1] - 3 * w[:, x_len + 1]
        w[:, x_len - 2] = w[:, x_len - 2] + 1 * w[:, x_len + 1]
        w = w[:, range(1, x_len + 1)]
    elif output == "tensorflow":
        ix = _np.empty(0, dtype=_np.int64)
        iy = _np.empty(0, dtype=_np.int64)
        values = _np.empty(0)
        # looping through s to save memory
        for i in range(_np.size(s)):
            si = _np.abs(s[i] - base_pos)
            pos = si <= 1
            si[pos] = 1.5 * _np.power(si[pos], 3) - 2.5 * _np.power(si[pos], 2) + 1
            pos = (si > 1) & (si <= 2)
            si[pos] = - 0.5 * _np.power(si[pos], 3) + 2.5 * _np.power(si[pos], 2) - 4 * si[pos] + 2
            si[si > 2] = 0
            # borders
            si[1] = si[1] + 3*si[0]
            si[2] = si[2] - 3*si[0]
            si[3] = si[3] + 1*si[0]
            si[x_len] = si[x_len] + 3 * si[x_len + 1]
            si[x_len - 1] = si[x_len - 1] - 3 * si[x_len + 1]
            si[x_len - 2] = si[x_len - 2] + 1 * si[x_len + 1]
            si = si[range(1, x_len + 1)]
            # indices
            pos = si.nonzero()[0]
            iy = _np.append(iy, pos)
            ix = _np.append(ix, _np.repeat(i, _np.size(pos)))
            values = _np.append(values, si[pos])
        # tensorflow object
        indices = _np.append(_np.matrix(ix), _np.matrix(iy), 0).transpose()
        with _tf.name_scope("cubic_conv_1D"):
            w = _tf.SparseTensor(indices=indices, values=values,
                                 dense_shape=[_np.size(s), x_len])
            w = _tf.sparse.reorder(w)
    else:
        raise Exception("invalid output type")
        
    return w


class Spline(object):
    """
    Abstract spline class
    
    Objects can be called like a function to interpolate at new positions.
    """
    
    # basic interpolation methods
    @staticmethod
    def _phi(t, n_deriv=0):
        if n_deriv == 0:
            return 3*t**2 - 2*t**3
        elif n_deriv == 1:
            return 6*t - 6*t**2
        else:
            raise ValueError("n_deriv must be 0 or 1")

    @staticmethod
    def _psi(t, n_deriv=0):
        if n_deriv == 0:
            return t**3 - t**2
        elif n_deriv == 1:
            return 3*t**2 - 2*t
        else:
            raise ValueError("n_deriv must be 0 or 1")

    @staticmethod
    def _h1(x, x1, x2, n_deriv=0):
        h = x2 - x1
        if n_deriv == 0:
            return Spline._phi((x2 - x)/h)
        elif n_deriv == 1:
            return Spline._phi((x2 - x)/h, 1)*(-1/h)
        else:
            raise ValueError("n_deriv must be 0 or 1")

    @staticmethod
    def _h2(x, x1, x2, n_deriv=0):
        h = x2 - x1
        if n_deriv == 0:
            return Spline._phi((x - x1)/h)
        elif n_deriv == 1:
            return Spline._phi((x - x1)/h, 1)/h
        else:
            raise ValueError("n_deriv must be 0 or 1")
        
    @staticmethod
    def _h3(x, x1, x2, n_deriv=0):
        h = x2 - x1
        if n_deriv == 0:
            return - h * Spline._psi((x2 - x)/h)
        elif n_deriv == 1:
            return Spline._psi((x2 - x)/h, 1)
        else:
            raise ValueError("n_deriv must be 0 or 1")
            
    @staticmethod
    def _h4(x, x1, x2, n_deriv=0):
        h = x2 - x1
        if n_deriv == 0:
            return + h * Spline._psi((x - x1)/h)
        elif n_deriv == 1:
            return Spline._psi((x - x1)/h, 1)
        else:
            raise ValueError("n_deriv must be 0 or 1")

    @staticmethod
    def _poly(x, x1, x2, f1, f2, d1, d2, n_deriv=0):
        return f1*Spline._h1(x, x1, x2, n_deriv) \
               + f2*Spline._h2(x, x1, x2, n_deriv) \
               + d1*Spline._h3(x, x1, x2, n_deriv) \
               + d2*Spline._h4(x, x1, x2, n_deriv)


class MonotonicSpline(Spline):
    """
    Implementation of the monotonic spline algorithm by Fritsch and Carlson (1980).

    Attributes
    ----------
    x, y : array
        The knots that define the spline.
    d : array
        The derivative of y at the points x, found internally to preserve monotonicity.
    region : int
        May affect the spline's shape. Refer to the original paper for details.
    """
    
    def __find_d(self, region=1):
        """
        Starting from a standard cubic spline, adjusts the derivatives
        at the data points to ensure monotonicity. 
        
        The region argument takes values from 1 to 4, each one increasing 
        the constraints in the possible values for the derivatives. Details
        can be found in the original paper, but the default will be
        sufficient for most applications.
        """
        x = self.x
        y = self.y
        if not (region in [1, 2, 3, 4]):
            raise Exception("region must be either 1, 2, 3, or 4")
        if not ((_np.diff(x) >= 0).all()):
            raise Exception("x is not monotonic")
        if not ((_np.diff(y) >= 0).all() | (_np.diff(y) <= 0).all()):
            raise Exception("y is not monotonic")
        
        delta = _np.diff(y) / _np.diff(x)
        sp = _interp.CubicSpline(x, y)
        d = sp(x, 1)
        d[d < 0] = 0
        if d[0] == 0:
            d[0] = delta[0]
        if d[-1] == 0:
            d[-1] = delta[-1]
        d1 = d[range(0, d.size - 1)]
        d2 = d[range(1, d.size)]
        alpha = d1 / (delta + 1e-6)
        beta = d2 / (delta + 1e-6)
        
        for i in range(d1.size):
            # region 1
            if (alpha[i] > beta[i]) & (alpha[i] > 3):
                k = alpha[i] / (beta[i] + 1e-6)
                alpha[i] = 3
                beta[i] = 3 / k
            elif beta[i] > 3:
                k = beta[i] / (alpha[i] + 1e-6)
                beta[i] = 3
                alpha[i] = 3 / k
            # region 2
            if region >= 2:
                r = _np.sqrt(alpha[i] ** 2 + beta[i] ** 2) + 1e-6
                if r > 3:
                    alpha[i] = alpha[i] * 3 / r
                    beta[i] = beta[i] * 3 / r
            # region 3
            if region >= 3:
                s = alpha[i] + beta[i]
                if s > 3:
                    alpha[i], beta[i] = 3*alpha[i]/s, 3*beta[i]/s
            # region 4
            if region == 4:
                if alpha[i] > beta[i]:
                    if alpha[i] > (3-beta[i])/2:
                        k = alpha[i] / (beta[i] + 1e-6)
                        beta[i] = 3 / (2*k + 1)
                        alpha[i] = (3-beta[i])/2
                else:
                    if beta[i] > (3-alpha[i])/2:
                        k = beta[i] / (alpha[i] + 1e-6)
                        alpha[i] = 3 / (2*k + 1)
                        beta[i] = (3-alpha[i])/2        
            # updating derivatives
            d1[i] = alpha[i] * delta[i]
            d2[i] = beta[i] * delta[i]
            if i < alpha.size - 1:
                d1[i+1] = d2[i]
                alpha[i+1] = d1[i+1] / (delta[i+1] + 1e-6)
        
        self.d = _np.append(d1, d2[-1])
        self._alpha = alpha
        self._beta = beta
        
    def __call__(self, x2, n_deriv=0):
        """
        Interpolation at new positions. Extrapolation is done linearly.
        """
        yout = _np.repeat(0.0, x2.size)
        x = self.x
        y = self.y
        d = self.d
        
        # values outside limits
        if n_deriv == 0:
            yout[x2 < x[0]] = y[0] + d[0] * (x2[x2 < x[0]] - x[0])
            yout[x2 >= x[-1]] = y[-1] + d[-1] * (x2[x2 >= x[-1]] - x[-1])
        elif n_deriv == 1:
            yout[x2 < x[0]] = d[0]
            yout[x2 >= x[-1]] = d[-1]
        else:
            raise ValueError("n_deriv must be 0 or 1")
            
        # values inside limits
        for i in range(d.size - 1):
            pos = ((x2 >= x[i]) & (x2 < x[i + 1]))
            yout[pos] = Spline._poly(x2[pos],
                                     x[i], x[i+1],
                                     y[i], y[i+1],
                                     d[i], d[i+1],
                                     n_deriv)
                      
        return yout
    
    def __init__(self, x, y, region=1):
        self.x = x
        self.y = y
        self.region = region
        self.__find_d(region)
