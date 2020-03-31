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

__all__ = ['cubic_conv_1d',
           'cubic_conv_2d',
           'cubic_conv_3d',
           'CubicSpline',
           'MonotonicCubicSpline']


import geoml.tftools as _tftools

import tensorflow as _tf


def cubic_conv_1d(x, xnew):
    """
    Generates a sparse matrix for interpolating from a regular grid to
    a new set of positions in one dimension.

    Parameters
    ----------
    x : Tensor
        Array of regularly spaced values.
    xnew : Tensor
        Positions to receive the interpolated values. Its limits must be
        confined within the limits of x.

    Returns
    -------
    w : SparseTensor
        The weights matrix.
    """
    # TODO: throw exceptions in TensorFlow
    # if _tf.math.reduce_std(x[1:]-x[:-1]) > 1e-9:
    #     raise Exception("array x not equally spaced")
    # if (_tf.reduce_min(xnew) < _tf.reduce_min(x)) \
    #         | (_tf.reduce_max(xnew) > _tf.reduce_max(x)):
    #     raise Exception("xnew out of bounds")

    # interval and data position
    h = x[1] - x[0]
    x_len = _tf.shape(x, out_type=_tf.int64)[0]
    x_new_len = _tf.shape(xnew, out_type=_tf.int64)[0]
    dense_shape = [x_new_len, x_len]

    s = _tf.expand_dims((xnew - _tf.reduce_min(x)) / h, axis=1)
    base_cols = _tf.constant([[-1, 0, 1, 2]], dtype=_tf.float64)

    # distance and positions
    s_relative = s + base_cols
    cols = _tf.math.floor(s_relative)
    s_relative = _tf.math.abs(s - cols)
    cols = _tf.cast(cols, _tf.int64)

    # weights
    s_0 = s_relative[:, 0]
    s_1 = s_relative[:, 1]
    s_2 = s_relative[:, 2]
    s_3 = s_relative[:, 3]
    w = _tf.stack(
        [- 0.5*s_0**3 + 2.5*s_0**2 - 4.0*s_0 + 2.0,
         1.5*s_1**3 - 2.5*s_1**2 + 1.0,
         1.5*s_2**3 - 2.5*s_2**2 + 1.0,
         - 0.5*s_3**3 + 2.5*s_3**2 - 4.0*s_3 + 2.0],
        axis=1
    )

    # borders
    left_border = _tf.stack(
        [w[:, 1] + 3.0*w[:, 0],
         w[:, 2] - 3.0*w[:, 0],
         w[:, 3] + 1.0*w[:, 0],
         0*w[:, 0]],
        axis=1
    )
    right_border = _tf.stack(
        [0 * w[:, 0],
         w[:, 0] + 1.0*w[:, 3],
         w[:, 1] - 3.0*w[:, 3],
         w[:, 2] + 3.0*w[:, 3]],
        axis=1
    )
    last_point = _tf.concat([_tf.zeros([x_new_len, 3], _tf.float64),
                             _tf.ones([x_new_len, 1], _tf.float64)], axis=1)

    w_final = _tf.where(
        _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 0], -1), 1), [1, 4]),
        left_border, w
    )
    w_final = _tf.where(
        _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len), 1), [1, 4]),
        right_border, w_final
    )
    w_final = _tf.where(
        _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len + 1), 1), [1, 4]),
        last_point, w_final
    )

    cols_final = _tf.where(
        _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 0], -1), 1), [1, 4]),
        cols + 1, cols
    )
    cols_final = _tf.where(
        _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len), 1), [1, 4]),
        cols - 1, cols_final
    )
    cols_final = _tf.where(
        _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len + 1), 1), [1, 4]),
        cols - 2, cols_final
    )

    # sparse indices
    rows = _tf.tile(_tf.expand_dims(_tf.range(x_new_len, dtype=_tf.int64), 1),
                    [1, 4])
    rows = _tf.reshape(rows, [-1])
    # cols = _tf.maximum(cols, 0)
    # cols = _tf.minimum(cols, x_len - 1)
    cols = _tf.reshape(cols_final, [-1])
    vals = _tf.reshape(w_final, [-1])

    keep = _tf.squeeze(_tf.where(_tf.not_equal(vals, 0.0)))
    rows = _tf.gather(rows, keep)
    cols = _tf.gather(cols, keep)
    vals = _tf.gather(vals, keep)

    # output
    indices = _tf.stack([rows, cols], axis=1)
    w_new = _tf.sparse.SparseTensor(indices, vals, dense_shape)
    return w_new


def cubic_conv_2d(grid_x, grid_y, coords):
    int_x = cubic_conv_1d(grid_x, coords[:, 0])
    int_y = cubic_conv_1d(grid_y, coords[:, 1])

    w_new = _tftools.sparse_kronecker_by_rows(int_x, int_y)
    return w_new


def cubic_conv_3d(grid_x, grid_y, grid_z, coords):
    int_x = cubic_conv_1d(grid_x, coords[:, 0])
    int_y = cubic_conv_1d(grid_y, coords[:, 1])
    int_z = cubic_conv_1d(grid_z, coords[:, 2])

    w_xy = _tftools.sparse_kronecker_by_rows(int_x, int_y)
    w_new = _tftools.sparse_kronecker_by_rows(w_xy, int_z)
    return w_new


class CubicSpline(object):
    """
    Cubic splines.
    """

    # Hermite spline functions
    @staticmethod
    def _phi(t):
        return 3 * t ** 2 - 2 * t ** 3

    @staticmethod
    def _phi_d1(t):
        return 6 * t - 6 * t ** 2

    @staticmethod
    def _psi(t):
        return t ** 3 - t ** 2

    @staticmethod
    def _psi_d1(t):
        return 3 * t ** 2 - 2 * t

    @staticmethod
    def _h1(x, x1, x2):
        h = x2 - x1
        return CubicSpline._phi((x2 - x) / h)

    @staticmethod
    def _h1_d1(x, x1, x2):
        h = x2 - x1
        return CubicSpline._phi_d1((x2 - x) / h) * (-1 / h)

    @staticmethod
    def _h2(x, x1, x2):
        h = x2 - x1
        return CubicSpline._phi((x - x1) / h)

    @staticmethod
    def _h2_d1(x, x1, x2):
        h = x2 - x1
        return CubicSpline._phi_d1((x - x1) / h) / h

    @staticmethod
    def _h3(x, x1, x2):
        h = x2 - x1
        return - h * CubicSpline._psi((x2 - x) / h)

    @staticmethod
    def _h3_d1(x, x1, x2):
        h = x2 - x1
        return CubicSpline._psi_d1((x2 - x) / h)

    @staticmethod
    def _h4(x, x1, x2):
        h = x2 - x1
        return h * CubicSpline._psi((x - x1) / h)

    @staticmethod
    def _h4_d1(x, x1, x2):
        h = x2 - x1
        return CubicSpline._psi_d1((x - x1) / h)

    @staticmethod
    def _poly(x, x1, x2, f1, f2, d1, d2):
        return f1 * CubicSpline._h1(x, x1, x2) \
               + f2 * CubicSpline._h2(x, x1, x2) \
               + d1 * CubicSpline._h3(x, x1, x2) \
               + d2 * CubicSpline._h4(x, x1, x2)

    @staticmethod
    def _poly_d1(x, x1, x2, f1, f2, d1, d2):
        return f1 * CubicSpline._h1_d1(x, x1, x2) \
               + f2 * CubicSpline._h2_d1(x, x1, x2) \
               + d1 * CubicSpline._h3_d1(x, x1, x2) \
               + d2 * CubicSpline._h4_d1(x, x1, x2)

    def __init__(self, x, y):
        self.x = _tf.Variable(x, dtype=_tf.float64, validate_shape=False)
        self.y = _tf.Variable(y, dtype=_tf.float64, validate_shape=False)
        self.d = _tf.Variable(_tf.ones_like(self.x), validate_shape=False)
        self.refresh(x, y)

    def refresh(self, x, y):
        self.x.assign(x)
        self.y.assign(y)

        w = x[1:] - x[:-1]
        s = (y[1:] - y[:-1]) / w
        d = _tf.concat([
            _tf.expand_dims(s[0], axis=0),
            (s[:-1] * w[1:] + s[1:] * w[:-1]) / (w[1:] + w[:-1]),
            _tf.expand_dims(s[-1], axis=0),
        ], axis=0)
        self.d.assign_add(d)

    def interpolate(self, xnew):
        with _tf.name_scope("cubic_interpolation"):
            x = self.x
            y = self.y
            d = self.d

            n_knots = _tf.shape(x)[0]

            x = _tf.expand_dims(x, 1)
            xnew = _tf.expand_dims(xnew, 0)
            le = _tf.less_equal(x, xnew)
            pos = _tf.reduce_sum(_tf.cast(le, _tf.int32), axis=0)
            x = _tf.squeeze(x)
            xnew = _tf.squeeze(xnew)

            ynew = _tf.where(
                _tf.equal(pos, 0),
                y[0] + d[0] * (xnew - x[0]),
                _tf.where(
                    _tf.equal(pos, n_knots),
                    y[-1] + d[-1] * (xnew - x[-1]),
                    CubicSpline._poly(
                        xnew,
                        _tf.gather(x, _tf.maximum(pos-1, 0)),
                        _tf.gather(x, _tf.minimum(pos, n_knots-1)),
                        _tf.gather(y, _tf.maximum(pos-1, 0)),
                        _tf.gather(y, _tf.minimum(pos, n_knots-1)),
                        _tf.gather(d, _tf.maximum(pos-1, 0)),
                        _tf.gather(d, _tf.minimum(pos, n_knots-1)))))
        return ynew

    def interpolate_d1(self, xnew):
        with _tf.name_scope("cubic_interpolation_d1"):
            x = self.x
            y = self.y
            d = self.d

            n_knots = _tf.shape(x)[0]

            x = _tf.expand_dims(x, 1)
            xnew = _tf.expand_dims(xnew, 0)
            le = _tf.less_equal(x, xnew)
            pos = _tf.reduce_sum(_tf.cast(le, _tf.int32), axis=0)
            x = _tf.squeeze(x)
            xnew = _tf.squeeze(xnew)

            ynew = _tf.where(
                _tf.equal(pos, 0),
                d[0],
                _tf.where(
                    _tf.equal(pos, n_knots),
                    d[-1],
                    CubicSpline._poly_d1(
                        xnew,
                        _tf.gather(x, _tf.maximum(pos-1, 0)),
                        _tf.gather(x, _tf.minimum(pos, n_knots-1)),
                        _tf.gather(y, _tf.maximum(pos-1, 0)),
                        _tf.gather(y, _tf.minimum(pos, n_knots-1)),
                        _tf.gather(d, _tf.maximum(pos-1, 0)),
                        _tf.gather(d, _tf.minimum(pos, n_knots-1)))))
        return ynew


class MonotonicCubicSpline(CubicSpline):
    """
    Implementation of the monotonic spline algorithm by Steffen (1990).

    Attributes
    ----------
    x, y : Tensor
        The knots that define the spline.
    d : Tensor
        The derivative of y at the points x, adjusted to preserve
        monotonicity.
    """
    def refresh(self, x, y):
        self.x = x
        self.y = y

        w = x[1:] - x[:-1]
        s = (y[1:] - y[:-1]) / w

        p = (s[:-1] * w[1:] + s[1:] * w[:-1]) / (w[1:] + w[:-1])
        s_max = 2 * _tf.reduce_min(_tf.stack([s[:-1], s[1:]], axis=0), axis=0)

        d = _tf.where(_tf.greater(p, s_max), s_max, p)

        self.d = _tf.concat([
            _tf.expand_dims(s[0], axis=0),
            d,
            _tf.expand_dims(s[-1], axis=0),
        ], axis=0)
