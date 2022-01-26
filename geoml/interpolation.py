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

__all__ = ['CubicSpline',
           'MonotonicCubicSpline',
           'CubicConv1D',
           'CubicConv2DSeparable',
           'CubicConv3DSeparable',
           'CubicConv2DFull']

import geoml.tftools as _tftools

import tensorflow as _tf
import numpy as _np


class _Interpolator:
    def __init__(self, grid):
        self.grid = grid.grid
        self. _n_dim = grid.n_dim

    @property
    def n_dim(self):
        return self._n_dim

    def make_interpolation_matrix(self, coordinates, derivative=-1):
        raise NotImplementedError

    def interpolate(self, values, coordinates):
        interp_mat = self.make_interpolation_matrix(coordinates)
        return interp_mat.matmul(values)

    def interpolate_gradient(self, values, coordinates, directions):
        interp_mats = [self.make_interpolation_matrix(coordinates, d)
                       for d in range(self.n_dim)]
        interpolated = _tf.stack([mat.matmul(values) for mat in interp_mats],
                                 axis=2)
        return _tf.reduce_sum(interpolated * directions[:, None, :], axis=2)

    @staticmethod
    def cubic_conv_1d(x, xnew, derivative=-1):
        with _tf.name_scope("cubic_conv_1d"):
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

            if derivative == -1:
                w = _tf.stack(
                    [- 0.5 * s_0 ** 3 + 2.5 * s_0 ** 2 - 4.0 * s_0 + 2.0,
                     1.5 * s_1 ** 3 - 2.5 * s_1 ** 2 + 1.0,
                     1.5 * s_2 ** 3 - 2.5 * s_2 ** 2 + 1.0,
                     - 0.5 * s_3 ** 3 + 2.5 * s_3 ** 2 - 4.0 * s_3 + 2.0],
                    axis=1
                )
            elif derivative == 0:
                w = _tf.stack(
                    [- 1.5 * s_0 ** 2 + 5 * s_0 - 4.0,
                     4.5 * s_1 ** 2 - 5 * s_1,
                     - (4.5 * s_2 ** 2 - 5 * s_2),
                     - (- 1.5 * s_3 ** 2 + 5 * s_3 - 4.0)],
                    axis=1
                )
            else:
                raise ValueError("invalid direction for derivative")

            # borders
            left_border = _tf.stack(
                [w[:, 1] + 3.0 * w[:, 0],
                 w[:, 2] - 3.0 * w[:, 0],
                 w[:, 3] + 1.0 * w[:, 0],
                 0 * w[:, 0]],
                axis=1
            )
            right_border = _tf.stack(
                [0 * w[:, 0],
                 w[:, 0] + 1.0 * w[:, 3],
                 w[:, 1] - 3.0 * w[:, 3],
                 w[:, 2] + 3.0 * w[:, 3]],
                axis=1
            )
            if derivative == -1:
                last_point = _tf.concat([_tf.zeros([x_new_len, 3], _tf.float64),
                                         _tf.ones([x_new_len, 1], _tf.float64)],
                                        axis=1)
            else:
                # last_point = _tf.zeros([x_new_len, 4], _tf.float64)
                last_point = right_border

            w_final = _tf.where(
                _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 0], -1), 1), [1, 4]),
                left_border, w
            )
            w_final = _tf.where(
                _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len), 1),
                         [1, 4]),
                right_border, w_final
            )
            w_final = _tf.where(
                _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len + 1), 1),
                         [1, 4]),
                last_point, w_final
            )

            cols_final = _tf.where(
                _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 0], -1), 1), [1, 4]),
                cols + 1, cols
            )
            cols_final = _tf.where(
                _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len), 1),
                         [1, 4]),
                cols - 1, cols_final
            )
            cols_final = _tf.where(
                _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len + 1), 1),
                         [1, 4]),
                cols - 2, cols_final
            )

            return w_final, cols_final

    class _InterpolationMatrix:
        def __init__(self, weights, index, full_size):
            self.weights = weights
            self.index = index
            self.full_size = full_size

        def matmul(self, x, power=1.0):
            with _tf.name_scope("interpolation_matmul"):
                n = _tf.shape(self.weights)[0]
                m = _tf.shape(self.weights)[1]
                p = _tf.shape(x)[1]

                idx = _tf.reshape(self.index, [-1])
                x_g = _tf.gather(x, idx)  # [m * n, p]
                x_g = _tf.reshape(_tf.transpose(x_g), [p, n, m])

                interp = _tf.einsum("nm,pnm->np", self.weights ** power, x_g)
                return interp

        def outer_product(self, other):
            with _tf.name_scope("interpolation_outer_product"):
                new_size = self.full_size * other.full_size
                size_self = _tf.cast(self.full_size, _tf.int64)
                compact_size = _tf.shape(self.weights)[1] \
                               * _tf.shape(other.weights)[1]

                w_outer = _tf.einsum("ix,iy->ixy", self.weights, other.weights)
                w_outer = _tf.reshape(w_outer, [-1, compact_size])

                idx_outer = size_self * other.index[:, None, :] \
                            + self.index[:, :, None]
                idx_outer = _tf.reshape(idx_outer, [-1, compact_size])

                return self.__class__(w_outer, idx_outer, new_size)


class CubicConv1D(_Interpolator):
    def __init__(self, grid):
        super().__init__(grid)

        if grid.n_dim != 1:
            raise ValueError("grid must have be 1-dimensional")

    def make_interpolation_matrix(self, coordinates, derivative=-1):
        x = self.grid[0]
        xnew = coordinates[:, 0]

        w_final, cols_final = self.cubic_conv_1d(x, xnew, derivative)

        return self._InterpolationMatrix(w_final, cols_final, len(x))


class CubicConv2DSeparable(_Interpolator):
    def __init__(self, grid):
        super().__init__(grid)

        if grid.n_dim != 2:
            raise ValueError("grid must be 2-dimensional")

    def make_interpolation_matrix(self, coordinates, derivative=-1):
        """
        Generates a sparse matrix for interpolating from a regular grid to
        a new set of positions in one dimension.

        Parameters
        ----------
        coordinates : array-like
            Coordinates to interpolate on.
        derivative : int
            Direction to derivate on (-1 for no derivative).

        Returns
        -------
        interp : InterpolationMatrix
            The interpolator object.
        """
        x, y = self.grid
        xnew = coordinates[:, 0]
        ynew = coordinates[:, 1]

        if derivative == -1:
            w_x, cols_x = self.cubic_conv_1d(x, xnew)
            interp_x = self._InterpolationMatrix(w_x, cols_x, len(x))
            w_y, cols_y = self.cubic_conv_1d(y, ynew)
            interp_y = self._InterpolationMatrix(w_y, cols_y, len(y))
        elif derivative == 0:
            w_x, cols_x = self.cubic_conv_1d(x, xnew, derivative=0)
            interp_x = self._InterpolationMatrix(w_x, cols_x, len(x))
            w_y, cols_y = self.cubic_conv_1d(y, ynew)
            interp_y = self._InterpolationMatrix(w_y, cols_y, len(y))
        elif derivative == 1:
            w_x, cols_x = self.cubic_conv_1d(x, xnew)
            interp_x = self._InterpolationMatrix(w_x, cols_x, len(x))
            w_y, cols_y = self.cubic_conv_1d(y, ynew, derivative=0)
            interp_y = self._InterpolationMatrix(w_y, cols_y, len(y))
        else:
            raise ValueError("invalid direction for derivative")

        return interp_x.outer_product(interp_y)


class CubicConv3DSeparable(_Interpolator):
    def __init__(self, grid):
        super().__init__(grid)

        if grid.n_dim != 3:
            raise ValueError("grid must be 2-dimensional")

    def make_interpolation_matrix(self, coordinates, derivative=-1):
        x, y, z = self.grid
        xnew = coordinates[:, 0]
        ynew = coordinates[:, 1]
        znew = coordinates[:, 2]

        if derivative == -1:
            w_x, cols_x = self.cubic_conv_1d(x, xnew)
            interp_x = self._InterpolationMatrix(w_x, cols_x, len(x))
            w_y, cols_y = self.cubic_conv_1d(y, ynew)
            interp_y = self._InterpolationMatrix(w_y, cols_y, len(y))
            w_z, cols_z = self.cubic_conv_1d(z, znew)
            interp_z = self._InterpolationMatrix(w_z, cols_z, len(z))
        elif derivative == 0:
            w_x, cols_x = self.cubic_conv_1d(x, xnew, derivative=0)
            interp_x = self._InterpolationMatrix(w_x, cols_x, len(x))
            w_y, cols_y = self.cubic_conv_1d(y, ynew)
            interp_y = self._InterpolationMatrix(w_y, cols_y, len(y))
            w_z, cols_z = self.cubic_conv_1d(z, znew)
            interp_z = self._InterpolationMatrix(w_z, cols_z, len(z))
        elif derivative == 1:
            w_x, cols_x = self.cubic_conv_1d(x, xnew)
            interp_x = self._InterpolationMatrix(w_x, cols_x, len(x))
            w_y, cols_y = self.cubic_conv_1d(y, ynew, derivative=0)
            interp_y = self._InterpolationMatrix(w_y, cols_y, len(y))
            w_z, cols_z = self.cubic_conv_1d(z, znew)
            interp_z = self._InterpolationMatrix(w_z, cols_z, len(z))
        elif derivative == 2:
            w_x, cols_x = self.cubic_conv_1d(x, xnew)
            interp_x = self._InterpolationMatrix(w_x, cols_x, len(x))
            w_y, cols_y = self.cubic_conv_1d(y, ynew)
            interp_y = self._InterpolationMatrix(w_y, cols_y, len(y))
            w_z, cols_z = self.cubic_conv_1d(z, znew, derivative=0)
            interp_z = self._InterpolationMatrix(w_z, cols_z, len(z))
        else:
            raise ValueError("invalid direction for derivative")

        return interp_x.outer_product(interp_y.outer_product(interp_z))


class CubicConv2DFull(_Interpolator):
    def __init__(self, grid):
        super().__init__(grid)

        if grid.n_dim != 2:
            raise ValueError("grid must be 2-dimensional")

    def make_interpolation_matrix(self, coordinates, derivative=-1):
        xg, yg = self.grid
        xnew = coordinates[:, 0]
        ynew = coordinates[:, 1]

        # interval and data position
        h_x = xg[1] - xg[0]
        x_len = _tf.shape(xg, out_type=_tf.int64)[0]
        x_new_len = _tf.shape(xnew, out_type=_tf.int64)[0]

        h_y = yg[1] - yg[0]
        y_len = _tf.shape(yg, out_type=_tf.int64)[0]

        s_x = _tf.expand_dims((xnew - _tf.reduce_min(xg)) / h_x, axis=1)
        s_y = _tf.expand_dims((ynew - _tf.reduce_min(yg)) / h_y, axis=1)
        base_pos = _tf.constant([[-1, 0, 1, 2]], dtype=_tf.float64)

        # distance and positions
        s_relative_x = s_x + base_pos
        cols_x = _tf.math.floor(s_relative_x)
        s_relative_x = _tf.math.abs(s_x - cols_x)
        cols_x = _tf.cast(cols_x, _tf.int64)

        s_relative_y = s_y + base_pos
        cols_y = _tf.math.floor(s_relative_y)
        s_relative_y = _tf.math.abs(s_y - cols_y)
        cols_y = _tf.cast(cols_y, _tf.int64)

        # weights
        def f_11(x, y):
            return 25/4 * x**3 * y**3 \
                            - 35/4 * (x**3 * y**2 + x**2 * y**3) \
                            + 5/2 * (x**3 + y**3) + 49/4 * x**2 * y**2 \
                            - 7/2 * (x**2 + y**2) + 1

        def f_12(x, y):
            return (y - 2)**2 / 4 * (5 * x**3 * y - 5 * x**3
                                               - 7 * x**2 * y + 7 * x**2
                                               + 2 * y - 2)

        def f_21(x, y):
            return (x - 2)**2 / 4 * (5 * y**3 * x - 5 * y**3
                                               - 7 * y**2 * x + 7 * y**2
                                               + 2 * x - 2)

        def f_22(x, y):
            return (x - 2)**2 * (y - 2)**2 / 4 * (x*y - x - y + 1)

        funs = _np.array([[f_11, f_12], [f_21, f_22]])
        funs = _np.concatenate([_np.flip(funs, axis=0), funs], axis=0)
        funs = _np.concatenate([_np.flip(funs, axis=1), funs], axis=1)

        def f_11_dx(x, y):
            return x / 4 * (75 * x * y**3 - 105 * x * y**2
                                        + 30 * x - 70 * y**3
                                        + 98 * y**2 - 28)

        def f_12_dx(x, y):
            return x * (y - 2) ** 2 / 4 * (15 * x * y - 15 * x
                                                       - 14 * y + 14)

        def f_21_dx(x, y):
            return (x - 2)**2 / 2 * (5/2 * y**3 - 7/2 * y**2 + 1) \
                               + (x - 2) * (5/2 * x * y**3 - 7/2 * x * y**2 + x
                                            - 5/2 * y**3 + 7/2 * y**2 - 1)

        def f_22_dx(x, y):
            return (x - 2) * (y - 2) ** 2 / 4 * (
                2 * x * y - 2 * x - 2 * y + (x - 2) * (y - 1) + 2)

        funs_dx = _np.array([[f_11_dx, f_12_dx], [f_21_dx, f_22_dx]])
        funs_dx = _np.concatenate([_np.flip(funs_dx, axis=0), funs_dx], axis=0)
        funs_dx = _np.concatenate([_np.flip(funs_dx, axis=1), funs_dx], axis=1)

        def f_11_dy(x, y):
            return y / 4 * (75 * x**3 * y - 70 * x**3
                                        - 105 * x**2 * y + 98 * x**2
                                        + 30 * y - 28)

        def f_12_dy(x, y):
            return (y - 2) / 4 * (10 * x**3 * y - 10 * x**3
                                              - 14 * x**2 * y + 14 * x**2
                                              + 4 * y + (y - 2)
                                              * (5 * x**3 - 7 * x**2 + 2) - 4)

        def f_21_dy(x, y):
            return y * (x - 2)**2 / 4 * (15 * x * y - 14 * x - 15 * y + 14)

        def f_22_dy(x, y):
            return (x - 2)**2 * (y - 2) / 4 * (
                    2 * x * y - 2 * x - 2 * y + (x - 1) * (y - 2) + 2)

        funs_dy = _np.array([[f_11_dy, f_12_dy], [f_21_dy, f_22_dy]])
        funs_dy = _np.concatenate([_np.flip(funs_dy, axis=0), funs_dy], axis=0)
        funs_dy = _np.concatenate([_np.flip(funs_dy, axis=1), funs_dy], axis=1)

        # weights.shape = [n_data, 4, 4]
        if derivative == -1:
            weights = _tf.stack(
                [_tf.stack(
                    [funs[i, j](s_relative_x[:, i], s_relative_y[:, j])
                     for i in range(4)], axis=-1)
                    for j in range(4)], axis=-1
            )
        elif derivative == 0:
            weights = _tf.stack(
                [_tf.stack(
                    [funs_dx[i, j](s_relative_x[:, i], s_relative_y[:, j])
                     for i in range(4)], axis=-1)
                    for j in range(4)], axis=-1
            )
        elif derivative == 1:
            weights = _tf.stack(
                [_tf.stack(
                    [funs_dy[i, j](s_relative_x[:, i], s_relative_y[:, j])
                     for i in range(4)], axis=-1)
                    for j in range(4)], axis=-1
            )
        else:
            raise ValueError("invalid direction for derivative")

        # borders
        def left_border(w):
            return _tf.stack(
                [w[:, 1, :] - 5.0 * w[:, 0, :],
                 w[:, 2, :] + 5.0 * w[:, 0, :],
                 w[:, 3, :] + 1.0 * w[:, 0, :],
                 0 * w[:, 0, :]],
                axis=1
            )

        def right_border(w):
            return _tf.stack(
                [0 * w[:, 0, :],
                 w[:, 0, :] + 1.0 * w[:, 3, :],
                 w[:, 1, :] + 5.0 * w[:, 3, :],
                 w[:, 2, :] - 5.0 * w[:, 3, :]],
                axis=1
            )

        def bottom_border(w):
            return _tf.stack(
                [w[:, :, 1] - 5.0 * w[:, :, 0],
                 w[:, :, 2] + 5.0 * w[:, :, 0],
                 w[:, :, 3] + 1.0 * w[:, :, 0],
                 0 * w[:, :, 0]],
                axis=2
            )

        def top_border(w):
            return _tf.stack(
                [0 * w[:, :, 0],
                 w[:, :, 0] + 1.0 * w[:, :, 3],
                 w[:, :, 1] + 5.0 * w[:, :, 3],
                 w[:, :, 2] - 5.0 * w[:, :, 3]],
                axis=2
            )

        bool_switch = {
            "left_border": _tf.logical_and(
                _tf.equal(cols_x[:, 0], -1),
                _tf.logical_and(
                    _tf.greater(cols_y[:, 0], -1),
                    _tf.less_equal(cols_y[:, 3], y_len)
                )
            )[:, None, None],
            "right_border": _tf.logical_and(
                _tf.greater_equal(cols_x[:, 3], x_len),
                _tf.logical_and(
                    _tf.greater(cols_y[:, 0], -1),
                    _tf.less_equal(cols_y[:, 3], y_len)
                )
            )[:, None, None],
            "top_border": _tf.logical_and(
                _tf.logical_and(
                    _tf.less_equal(cols_x[:, 3], x_len),
                    _tf.greater(cols_x[:, 0], -1)
                ),
                _tf.greater_equal(cols_y[:, 3], y_len)
            )[:, None, None],
            "bottom_border": _tf.logical_and(
                _tf.logical_and(
                    _tf.less_equal(cols_x[:, 3], x_len),
                    _tf.greater(cols_x[:, 0], -1)
                ),
                _tf.equal(cols_y[:, 0], -1)
            )[:, None, None],
            "bottom_left_corner": _tf.logical_and(
                _tf.equal(cols_x[:, 0], -1),
                _tf.equal(cols_y[:, 0], -1)
            )[:, None, None],
            "bottom_right_corner": _tf.logical_and(
                _tf.greater_equal(cols_x[:, 3], x_len),
                _tf.equal(cols_y[:, 0], -1)
            )[:, None, None],
            "top_left_corner": _tf.logical_and(
                _tf.equal(cols_x[:, 0], -1),
                _tf.greater_equal(cols_y[:, 3], y_len)
            )[:, None, None],
            "top_right_corner": _tf.logical_and(
                _tf.greater_equal(cols_x[:, 3], x_len),
                _tf.greater_equal(cols_y[:, 3], y_len)
            )[:, None, None],
        }

        w_final = weights
        w_final = _tf.where(
            _tf.tile(bool_switch["left_border"], [1, 4, 4]),
            left_border(weights), w_final
        )
        w_final = _tf.where(
            _tf.tile(bool_switch["right_border"], [1, 4, 4]),
            right_border(weights), w_final
        )
        w_final = _tf.where(
            _tf.tile(bool_switch["top_border"], [1, 4, 4]),
            top_border(weights), w_final
        )
        w_final = _tf.where(
            _tf.tile(bool_switch["bottom_border"], [1, 4, 4]),
            bottom_border(weights), w_final
        )
        w_final = _tf.where(
            _tf.tile(bool_switch["bottom_left_corner"], [1, 4, 4]),
            left_border(bottom_border(weights)), w_final
        )
        w_final = _tf.where(
            _tf.tile(bool_switch["bottom_right_corner"], [1, 4, 4]),
            right_border(bottom_border(weights)), w_final
        )
        w_final = _tf.where(
            _tf.tile(bool_switch["top_left_corner"], [1, 4, 4]),
            left_border(top_border(weights)), w_final
        )
        w_final = _tf.where(
            _tf.tile(bool_switch["top_right_corner"], [1, 4, 4]),
            right_border(top_border(weights)), w_final
        )
        # w_final = _tf.transpose(w_final, [0, 2, 1])
        w_final = _tf.reshape(w_final, [-1, 16])

        # positions
        cols_final_x = cols_x
        cols_final_x = _tf.where(
            _tf.tile(_tf.equal(cols_x[:, 0], -1)[:, None], [1, 4]),
            cols_x + 1, cols_final_x
        )
        cols_final_x = _tf.where(
            _tf.tile(_tf.equal(cols_x[:, 3], x_len)[:, None], [1, 4]),
            cols_x - 1, cols_final_x
        )
        cols_final_x = _tf.where(
            _tf.tile(_tf.equal(cols_x[:, 3], x_len + 1)[:, None], [1, 4]),
            cols_x - 2, cols_final_x
        )

        cols_final_y = cols_y
        cols_final_y = _tf.where(
            _tf.tile(_tf.equal(cols_y[:, 0], -1)[:, None], [1, 4]),
            cols_y + 1, cols_final_y
        )
        cols_final_y = _tf.where(
            _tf.tile(_tf.equal(cols_y[:, 3], y_len)[:, None], [1, 4]),
            cols_y - 1, cols_final_y
        )
        cols_final_y = _tf.where(
            _tf.tile(_tf.equal(cols_y[:, 3], y_len + 1)[:, None], [1, 4]),
            cols_y - 2, cols_final_y
        )

        cols_final = x_len * cols_final_y[:, None, :] + cols_final_x[:, :, None]
        cols_final = _tf.reshape(cols_final, [-1, 16])

        return self._InterpolationMatrix(w_final, cols_final, x_len * y_len)


# class InterpolationMatrix:
#     def __init__(self, weights, index, full_size):
#         self.weights = weights
#         self.index = index
#         self.full_size = full_size
#
#     @_tf.function
#     def matmul(self, x, power=1.0):
#         with _tf.name_scope("interpolation_matmul"):
#             n = _tf.shape(self.weights)[0]
#             m = _tf.shape(self.weights)[1]
#             p = _tf.shape(x)[1]
#
#             idx = _tf.reshape(self.index, [-1])
#             x_g = _tf.gather(x, idx)  # [m * n, p]
#             x_g = _tf.reshape(_tf.transpose(x_g), [p, n, m])
#
#             interp = _tf.einsum("nm,pnm->np", self.weights**power, x_g)
#             return interp
#
#     def outer_product(self, other):
#         with _tf.name_scope("interpolation_outer_product"):
#             new_size = self.full_size * other.full_size
#             size_self = _tf.cast(self.full_size, _tf.int64)
#             compact_size = _tf.shape(self.weights)[1] \
#                            * _tf.shape(other.weights)[1]
#
#             # w_outer = _tf.einsum("ix,iy->iyx", self.weights, other.weights)
#             w_outer = _tf.einsum("ix,iy->ixy", self.weights, other.weights)
#             w_outer = _tf.reshape(w_outer, [-1, compact_size])
#
#             # idx_outer = size_self * other.index[:, :, None] \
#             #             + self.index[:, None, :]
#             idx_outer = size_self * other.index[:, None, :] \
#                         + self.index[:, :, None]
#             idx_outer = _tf.reshape(idx_outer, [-1, compact_size])
#
#             return InterpolationMatrix(w_outer, idx_outer, new_size)
#
#     # @_tf.function
#     # def matmul_diag(self, diag):
#     #     with _tf.name_scope("interpolation_matmul_diag"):
#     #         n = _tf.shape(self.weights)[0]
#     #         m = _tf.shape(self.weights)[1]
#     #         p = _tf.shape(diag)[1]
#     #
#     #         idx = _tf.reshape(self.index, [-1])
#     #         x_g = _tf.gather(diag, idx)  # [m * n, p]
#     #         x_g = _tf.reshape(_tf.transpose(x_g), [p, n, m])
#     #
#     #         prod = self.weights[None, :, :]**2 * x_g
#     #         interp = _tf.transpose(_tf.reduce_sum(prod, axis=-1))
#     #         return interp
#
#
# def cubic_conv_1d(x, xnew, a=-0.5):
#     """
#     Generates a sparse matrix for interpolating from a regular grid to
#     a new set of positions in one dimension.
#
#     Parameters
#     ----------
#     x : Tensor
#         Array of regularly spaced values.
#     xnew : Tensor
#         Positions to receive the interpolated values. Its limits must be
#         confined within the limits of x.
#     a : double
#         Derivative of weight function at one unit from the origin.
#
#     Returns
#     -------
#     w : InterpolationMatrix
#         The interpolator object.
#     """
#     # TODO: throw exceptions in TensorFlow
#     # if _tf.math.reduce_std(x[1:]-x[:-1]) > 1e-9:
#     #     raise Exception("array x not equally spaced")
#     # if (_tf.reduce_min(xnew) < _tf.reduce_min(x)) \
#     #         | (_tf.reduce_max(xnew) > _tf.reduce_max(x)):
#     #     raise Exception("xnew out of bounds")
#
#     # interval and data position
#     h = x[1] - x[0]
#     x_len = _tf.shape(x, out_type=_tf.int64)[0]
#     x_new_len = _tf.shape(xnew, out_type=_tf.int64)[0]
#
#     s = _tf.expand_dims((xnew - _tf.reduce_min(x)) / h, axis=1)
#     base_cols = _tf.constant([[-1, 0, 1, 2]], dtype=_tf.float64)
#
#     # distance and positions
#     s_relative = s + base_cols
#     cols = _tf.math.floor(s_relative)
#     s_relative = _tf.math.abs(s - cols)
#     cols = _tf.cast(cols, _tf.int64)
#
#     # weights
#     s_0 = s_relative[:, 0]
#     s_1 = s_relative[:, 1]
#     s_2 = s_relative[:, 2]
#     s_3 = s_relative[:, 3]
#     # w = _tf.stack(
#     #     [- 0.5*s_0**3 + 2.5*s_0**2 - 4.0*s_0 + 2.0,
#     #      1.5*s_1**3 - 2.5*s_1**2 + 1.0,
#     #      1.5*s_2**3 - 2.5*s_2**2 + 1.0,
#     #      - 0.5*s_3**3 + 2.5*s_3**2 - 4.0*s_3 + 2.0],
#     #     axis=1
#     # )
#     w = _tf.stack(
#         [a * s_0 ** 3 - 5 * a * s_0 ** 2 + 8 * a * s_0 - 4 * a,
#          (a + 2) * s_1 ** 3 - (a + 3) * s_1 ** 2 + 1.0,
#          (a + 2) * s_2 ** 3 - (a + 3) * s_2 ** 2 + 1.0,
#          a * s_3 ** 3 - 5 * a * s_3 ** 2 + 8 * a * s_3 - 4 * a],
#         axis=1
#     )
#
#     # borders
#     left_border = _tf.stack(
#         [w[:, 1] + 3.0*w[:, 0],
#          w[:, 2] - 3.0*w[:, 0],
#          w[:, 3] + 1.0*w[:, 0],
#          0*w[:, 0]],
#         axis=1
#     )
#     right_border = _tf.stack(
#         [0 * w[:, 0],
#          w[:, 0] + 1.0*w[:, 3],
#          w[:, 1] - 3.0*w[:, 3],
#          w[:, 2] + 3.0*w[:, 3]],
#         axis=1
#     )
#     last_point = _tf.concat([_tf.zeros([x_new_len, 3], _tf.float64),
#                              _tf.ones([x_new_len, 1], _tf.float64)], axis=1)
#
#     w_final = _tf.where(
#         _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 0], -1), 1), [1, 4]),
#         left_border, w
#     )
#     w_final = _tf.where(
#         _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len), 1), [1, 4]),
#         right_border, w_final
#     )
#     w_final = _tf.where(
#         _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len + 1), 1), [1, 4]),
#         last_point, w_final
#     )
#
#     cols_final = _tf.where(
#         _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 0], -1), 1), [1, 4]),
#         cols + 1, cols
#     )
#     cols_final = _tf.where(
#         _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len), 1), [1, 4]),
#         cols - 1, cols_final
#     )
#     cols_final = _tf.where(
#         _tf.tile(_tf.expand_dims(_tf.equal(cols[:, 3], x_len + 1), 1), [1, 4]),
#         cols - 2, cols_final
#     )
#
#     return InterpolationMatrix(w_final, cols_final, x_len)
#
#
# def cubic_conv_2d(data, grid_x, grid_y, a=-0.5):
#     mat_x = cubic_conv_1d(grid_x, data[:, 0], a=a)
#     mat_y = cubic_conv_1d(grid_y, data[:, 1], a=a)
#
#     return mat_x.outer_product(mat_y)
#
#
# def cubic_conv_3d(data, grid_x, grid_y, grid_z, a=-0.5):
#     mat_x = cubic_conv_1d(grid_x, data[:, 0], a=a)
#     mat_y = cubic_conv_1d(grid_y, data[:, 1], a=a)
#     mat_z = cubic_conv_1d(grid_z, data[:, 2], a=a)
#
#     mat_yz = mat_y.outer_product(mat_z)
#
#     return mat_x.outer_product(mat_yz)


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
        with _tf.name_scope("cubic_interpolation_find_derivative"):
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
        with _tf.name_scope("monotonic_interpolation_find_derivative"):
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
