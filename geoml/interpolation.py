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

__all__ = ['cubic_conv_1d',
           'cubic_conv_1d_sparse',
           'cubic_conv_2d_sparse',
           'cubic_conv_3d_sparse',
           'cubic_conv_1d_parallel',
           'cubic_conv_2d_parallel',
           'cubic_conv_3d_parallel',
           'MonotonicSpline']

import numpy as _np
# import tensorflow as _tf
import scipy.interpolate as _interp
import scipy.sparse as _sp
import pathos.multiprocessing as _mult


def _batch_id(n_data, batch_size):
    n_batches = int(_np.ceil(n_data / batch_size))
    batch_id = [_np.arange(i * batch_size,
                           _np.minimum((i + 1) * batch_size, n_data))
                for i in range(n_batches)]
    return batch_id


def cubic_conv_1d(x, xnew):
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
    
    Returns
    -------
    w : array
        The weights matrix.
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
    w = _np.abs(_np.resize(s, [x_len + 2, _np.size(s)]).transpose()
                - _np.resize(base_pos, [_np.size(s), x_len + 2]))
    pos = w <= 1
    w[pos] = 1.5 * _np.power(w[pos], 3) - 2.5 * _np.power(w[pos], 2) + 1
    pos = (w > 1) & (w <= 2)
    w[pos] = - 0.5 * _np.power(w[pos], 3) \
             + 2.5 * _np.power(w[pos], 2) \
             - 4 * w[pos] + 2
    w[w > 2] = 0
    # borders
    w[:, 1] = w[:, 1] + 3*w[:, 0]
    w[:, 2] = w[:, 2] - 3*w[:, 0]
    w[:, 3] = w[:, 3] + 1*w[:, 0]
    w[:, x_len] = w[:, x_len] + 3 * w[:, x_len + 1]
    w[:, x_len - 1] = w[:, x_len - 1] - 3 * w[:, x_len + 1]
    w[:, x_len - 2] = w[:, x_len - 2] + 1 * w[:, x_len + 1]
    w = w[:, range(1, x_len + 1)]

    return w


def cubic_conv_1d_sparse(x, xnew):
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

    Returns
    -------
    w : csc_matrix
        The weights matrix.
    """
    if _np.std(_np.diff(x)) > 1e-9:
        raise Exception("array x not equally spaced")
    if (_np.min(xnew) < _np.min(x)) | (_np.max(xnew) > _np.max(x)):
        raise Exception("xnew out of bounds")

    # interval and data position
    h = x[1] - x[0]
    x_len = x.shape[0]
    x_new_len = xnew.shape[0]
    dense_shape = [x_new_len, x_len]

    s = (xnew - _np.min(x)) / h
    base_cols = _np.array([-1, 0, 1, 2])

    rows = []
    cols = []
    vals = []
    for i in range(x_new_len):
        # position
        si = s[i] + base_cols
        cols_i = _np.floor(si).tolist()
        si = _np.abs(s[i] - _np.floor(si))

        # weights
        wi = [- 0.5 * si[0]**3 + 2.5 * si[0]**2 - 4.0 * si[0] + 2.0,
              1.5 * si[1]**3 - 2.5 * si[1]**2 + 1.0,
              1.5 * si[2]**3 - 2.5 * si[2]**2 + 1.0,
              - 0.5 * si[3]**3 + 2.5 * si[3]**2 - 4.0 * si[3] + 2.0]

        # borders
        if cols_i[0] == -1:
            wi = [wi[1] + 3.0 * wi[0],
                  wi[2] - 3.0 * wi[0],
                  wi[3] + 1.0 * wi[0]]
            cols_i = cols_i[1:4]
        elif cols_i[-1] == x_len:
            wi = [wi[0] + 1.0 * wi[3],
                  wi[1] - 3.0 * wi[3],
                  wi[2] + 3.0 * wi[3]]
            cols_i = cols_i[0:3]
        elif cols_i[-1] == x_len + 1:
            wi = [1]
            cols_i = [x_len - 1]

        # output
        rows.extend([i] * len(cols_i))
        cols.extend(cols_i)
        vals.extend(wi)

    w_new = _sp.coo_matrix((vals, (rows, cols)), shape=dense_shape)
    return w_new


def cubic_conv_2d_sparse(grid_x, grid_y, coords):
    int_x = cubic_conv_1d_sparse(grid_x, coords[:, 0])
    int_y = cubic_conv_1d_sparse(grid_y, coords[:, 1])

    n_data = coords.shape[0]
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]

    rows = []
    cols = []
    vals = []
    for i in range(n_data):
        cols_x = int_x.col[int_x.row == i]
        data_x = int_x.data[int_x.row == i]

        cols_y = int_y.col[int_y.row == i]
        data_y = int_y.data[int_y.row == i]

        data_xy = _np.kron(data_y, data_x)
        cols_xy = _np.concatenate([nx * j + cols_x for j in cols_y], axis=0)

        rows.extend([i] * len(cols_x) * len(cols_y))
        cols.extend(cols_xy.tolist())
        vals.extend(data_xy.tolist())

    w_new = _sp.coo_matrix((vals, (rows, cols)), shape=[n_data, nx * ny])
    return w_new


def cubic_conv_3d_sparse(grid_x, grid_y, grid_z, coords):
    int_x = cubic_conv_1d_sparse(grid_x, coords[:, 0])
    int_y = cubic_conv_1d_sparse(grid_y, coords[:, 1])
    int_z = cubic_conv_1d_sparse(grid_z, coords[:, 2])

    n_data = coords.shape[0]
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]
    nz = grid_z.shape[0]

    rows = []
    cols = []
    vals = []
    for i in range(n_data):
        cols_x = int_x.col[int_x.row == i]
        data_x = int_x.data[int_x.row == i]

        cols_y = int_y.col[int_y.row == i]
        data_y = int_y.data[int_y.row == i]

        data_xy = _np.kron(data_y, data_x)
        cols_xy = _np.concatenate([nx * j + cols_x for j in cols_y], axis=0)

        cols_z = int_z.col[int_z.row == i]
        data_z = int_z.data[int_z.row == i]

        data_xyz = _np.kron(data_z, data_xy)
        cols_xyz = _np.concatenate([nx * ny * j + cols_xy for j in cols_z],
                                   axis=0)

        rows.extend([i] * len(cols_x) * len(cols_y) * len(cols_z))
        cols.extend(cols_xyz.tolist())
        vals.extend(data_xyz.tolist())

    w_new = _sp.coo_matrix((vals, (rows, cols)), shape=[n_data, nx * ny * nz])
    return w_new


def cubic_conv_1d_parallel(x, xnew, batch_size=1000):
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
    batch_size : int
        Number of points per batch.

    Returns
    -------
    w : csc_matrix
        The weights matrix.
    """
    n_data = len(xnew)
    batch_id = _batch_id(n_data, batch_size)

    def loop_fn(bid):
        return cubic_conv_1d_sparse(x, xnew[bid])

    pool = _mult.ProcessingPool()
    out = pool.map(loop_fn, batch_id)

    w_new = _sp.vstack(out)
    return w_new


def cubic_conv_2d_parallel(grid_x, grid_y, coords, batch_size=1000):
    n_data = coords.shape[0]
    batch_id = _batch_id(n_data, batch_size)

    def loop_fn(bid):
        return cubic_conv_2d_sparse(grid_x, grid_y, coords[bid])

    pool = _mult.ProcessingPool()
    out = pool.map(loop_fn, batch_id)

    w_new = _sp.vstack(out)
    return w_new


def cubic_conv_3d_parallel(grid_x, grid_y, grid_z, coords, batch_size=1000):
    n_data = coords.shape[0]
    batch_id = _batch_id(n_data, batch_size)

    def loop_fn(bid):
        return cubic_conv_3d_sparse(grid_x, grid_y, grid_z, coords[bid])

    pool = _mult.ProcessingPool()
    out = pool.map(loop_fn, batch_id)

    w_new = _sp.vstack(out)
    return w_new


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
                                     x[i], x[i + 1],
                                     y[i], y[i + 1],
                                     d[i], d[i + 1],
                                     n_deriv)

        return yout


class MonotonicSpline(Spline):
    """
    Implementation of the monotonic spline algorithm by Fritsch and Carlson
    (1980).

    Attributes
    ----------
    x, y : array
        The knots that define the spline.
    d : array
        The derivative of y at the points x, found internally to preserve
        monotonicity.
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
    
    def __init__(self, x, y, region=1):
        self.x = x
        self.y = y
        self.region = region
        self.__find_d(region)
