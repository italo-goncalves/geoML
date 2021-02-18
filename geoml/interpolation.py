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


class InterpolationMatrix:
    def __init__(self, weights, index, full_size):
        self.weights = weights
        self.index = index
        self.full_size = full_size

    @_tf.function
    def matmul(self, x):
        with _tf.name_scope("interpolation_matmul"):
            n = _tf.shape(self.weights)[0]
            m = _tf.shape(self.weights)[1]
            p = _tf.shape(x)[1]

            idx = _tf.reshape(self.index, [-1])
            x_g = _tf.gather(x, idx)  # [m * n, p]
            x_g = _tf.reshape(_tf.transpose(x_g), [p, n, m])

            interp = _tf.einsum("nm,pnm->np", self.weights, x_g)
            return interp

    def outer_product(self, other):
        with _tf.name_scope("interpolation_outer_product"):
            new_size = self.full_size * other.full_size
            size_self = _tf.cast(self.full_size, _tf.int64)
            compact_size = _tf.shape(self.weights)[1] \
                           * _tf.shape(other.weights)[1]

            w_outer = _tf.einsum("ix,iy->iyx", self.weights, other.weights)
            w_outer = _tf.reshape(w_outer, [-1, compact_size])

            idx_outer = size_self * other.index[:, :, None] \
                        + self.index[:, None, :]
            idx_outer = _tf.reshape(idx_outer, [-1, compact_size])

            return InterpolationMatrix(w_outer, idx_outer, new_size)


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
    w : InterpolationMatrix
        The interpolator object.
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

    return InterpolationMatrix(w_final, cols_final, x_len)


def cubic_conv_2d(data, grid_x, grid_y):
    mat_x = cubic_conv_1d(grid_x, data[:, 0])
    mat_y = cubic_conv_1d(grid_y, data[:, 1])

    return mat_x.outer_product(mat_y)


def cubic_conv_3d(data, grid_x, grid_y, grid_z):
    mat_x = cubic_conv_1d(grid_x, data[:, 0])
    mat_y = cubic_conv_1d(grid_y, data[:, 1])
    mat_z = cubic_conv_1d(grid_z, data[:, 2])

    mat_yz = mat_y.outer_product(mat_z)

    return mat_x.outer_product(mat_yz)


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

    def _get_derivative(self, x, y):
        w = x[1:, :] - x[:-1, :]
        s = (y[1:, :] - y[:-1, :]) / (w + 1e-12)
        d = _tf.concat([
            _tf.expand_dims(s[0, :], axis=0),
            (s[:-1, :] * w[1:, :] + s[1:, :] * w[:-1, :])
            / (w[1:, :] + w[:-1, :] + 1e-12),
            _tf.expand_dims(s[-1, :], axis=0),
        ], axis=0)
        return d

    def interpolate(self, x, y, xnew):
        with _tf.name_scope("cubic_interpolation"):
            # x, y, xnew = [knots, batch]
            x = _tftools.ensure_rank_2(x)
            y = _tftools.ensure_rank_2(y)
            xnew = _tftools.ensure_rank_2(xnew)

            d = self._get_derivative(x, y)

            n_knots = _tf.shape(x)[0]

            x_ex = x[:, None, :]
            y_ex = y[:, None, :]
            d_ex = d[:, None, :]
            xnew_ex = xnew[None, :, :]

            idx = _tf.range(n_knots - 1)

            all_interp = _tf.concat([
                y[0] + d[0] * (xnew_ex - x[0]),
                CubicSpline._poly(
                    xnew_ex,
                    _tf.gather(x_ex, idx),
                    _tf.gather(x_ex, idx + 1),
                    _tf.gather(y_ex, idx),
                    _tf.gather(y_ex, idx + 1),
                    _tf.gather(d_ex, idx),
                    _tf.gather(d_ex, idx + 1)),
                y[-1] + d[-1] * (xnew_ex - x[-1])
            ], axis=0)

            le = _tf.less_equal(x_ex, xnew_ex)
            pos = _tf.reduce_sum(_tf.cast(le, _tf.int32), axis=0)

            idx_1 = _tf.range(_tf.shape(xnew)[0])
            idx_2 = _tf.range(_tf.shape(xnew)[1])
            idx_1, idx_2 = _tf.meshgrid(idx_1, idx_2)
            idx_1 = _tf.reshape(idx_1, [-1, 1])
            idx_2 = _tf.reshape(idx_2, [-1, 1])
            pos = _tf.reshape(_tf.transpose(pos), [-1, 1])
            pos = _tf.concat([pos, idx_1, idx_2], axis=1)
            ynew = _tf.gather_nd(all_interp, pos)
            ynew = _tf.reshape(ynew, _tf.shape(xnew)[::-1])
            ynew = _tf.transpose(ynew)

        return ynew

    def interpolate_d1(self, x, y, xnew):
        with _tf.name_scope("cubic_interpolation_d1"):
            # x, y, xnew = [knots, batch]
            x = _tftools.ensure_rank_2(x)
            y = _tftools.ensure_rank_2(y)
            xnew = _tftools.ensure_rank_2(xnew)

            d = self._get_derivative(x, y)

            n_knots = _tf.shape(x)[0]

            x_ex = x[:, None, :]
            y_ex = y[:, None, :]
            d_ex = d[:, None, :]
            xnew_ex = xnew[None, :, :]

            idx = _tf.range(n_knots - 1)

            all_interp = _tf.concat([
                d[0] * _tf.ones_like(xnew_ex),
                CubicSpline._poly_d1(
                    xnew_ex,
                    _tf.gather(x_ex, idx),
                    _tf.gather(x_ex, idx + 1),
                    _tf.gather(y_ex, idx),
                    _tf.gather(y_ex, idx + 1),
                    _tf.gather(d_ex, idx),
                    _tf.gather(d_ex, idx + 1)),
                d[-1] * _tf.ones_like(xnew_ex)
            ], axis=0)

            le = _tf.less_equal(x_ex, xnew_ex)
            pos = _tf.reduce_sum(_tf.cast(le, _tf.int32), axis=0)

            idx_1 = _tf.range(_tf.shape(xnew)[0])
            idx_2 = _tf.range(_tf.shape(xnew)[1])
            idx_1, idx_2 = _tf.meshgrid(idx_1, idx_2)
            idx_1 = _tf.reshape(idx_1, [-1, 1])
            idx_2 = _tf.reshape(idx_2, [-1, 1])
            pos = _tf.reshape(_tf.transpose(pos), [-1, 1])
            pos = _tf.concat([pos, idx_1, idx_2], axis=1)
            ynew = _tf.gather_nd(all_interp, pos)
            ynew = _tf.reshape(ynew, _tf.shape(xnew)[::-1])
            ynew = _tf.transpose(ynew)

        return ynew


class MonotonicCubicSpline(CubicSpline):
    """
    Implementation of the monotonic spline algorithm by Steffen (1990).
    """

    def _get_derivative(self, x, y):
        w = x[1:, :] - x[:-1, :]
        s = (y[1:, :] - y[:-1, :]) / (w + 1e-12)

        p = (s[:-1, :] * w[1:, :] + s[1:, :] * w[:-1, :]) \
            / (w[1:, :] + w[:-1, :] + 1e-12)
        s_max = 2 * _tf.reduce_min(_tf.stack([s[:-1, :], s[1:, :]], axis=0),
                                   axis=0)

        d = _tf.where(_tf.greater(p, s_max), s_max, p)

        d = _tf.concat([_tf.expand_dims(s[0, :], axis=0),
                        d,
                        _tf.expand_dims(s[-1, :], axis=0)], axis=0)
        return d
