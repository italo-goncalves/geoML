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

__all__ = ["Gaussian",
           "Spherical",
           "Exponential",
           "Cubic",
           "Constant",
           "Linear",
           "Cosine",
           "Sum",
           "Product",
           "Matern32",
           "Matern52",
           "Scale"]

import geoml.parameter as _gpr
import geoml.transform as _gt

from geoml.tftools import pairwise_dist as _pairwise_dist
from geoml.tftools import pairwise_dist_l1 as _pairwise_dist_l1
from geoml.tftools import prod_n as _prod_n

import tensorflow as _tf
import numpy as _np


class _Kernel(_gpr.Parametric):
    def __init__(self):
        super().__init__()
        self._has_compact_support = False

    @property
    def has_compact_support(self):
        return self._has_compact_support

    def kernelize(self, x):
        raise NotImplemented

    def implicit_matmul(self, coordinates):
        """
        Implicit matrix-vector multiplication.

        Returns a function that multiplies the kernel's covariance matrix
        (defined at the given coordinates) with a vector efficiently.
        """
        raise NotImplemented


class Gaussian(_Kernel):
    """Gaussian kernel"""
    def kernelize(self, x):
        return _tf.exp(-3 * x**2)


class Spherical(_Kernel):
    """Spherical kernel"""
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self._has_compact_support = True
        self.epsilon = epsilon  # required to be able to compute gradients

    def kernelize(self, x):
        d = _tf.sqrt(x ** 2 + self.epsilon)
        k = 1 - 1.5 * d + 0.5 * _tf.pow(d, 3)
        k = _tf.where(_tf.less(d, 1.0), k, _tf.zeros_like(k))
        return k


class Exponential(_Kernel):
    """Exponential kernel"""
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon  # required to be able to compute gradients

    def kernelize(self, x):
        d = _tf.sqrt(x ** 2 + self.epsilon)
        k = _tf.exp(-3 * d)
        return k


class Cubic(_Kernel):
    """Cubic kernel"""
    def __init__(self):
        super().__init__()
        self._has_compact_support = True

    def kernelize(self, x):
        k = 1 - 7 * _tf.pow(x, 2) + 35 / 4 * _tf.pow(x, 3) \
            - 7 / 2 * _tf.pow(x, 5) + 3 / 4 * _tf.pow(x, 7)
        k = _tf.where(_tf.less(x, 1.0), k, _tf.zeros_like(k))
        return k


class Constant(_Kernel):
    """Constant kernel"""
    def kernelize(self, x):
        return _tf.ones_like(x)

    def implicit_matmul(self, coordinates):
        def matmul_fn(vector):
            result = _tf.ones_like(vector) * _tf.reduce_sum(vector)
            return result

        return matmul_fn


class Cosine(_Kernel):
    """Cosine kernel"""
    def kernelize(self, x):
        return _tf.cos(2.0 * _np.pi * x)


class Matern32(_Kernel):
    """Once differentiable Matérn kernel."""
    def kernelize(self, x):
        return (1 + 5*x)*_tf.math.exp(-5*x)


class Matern52(_Kernel):
    """Twice differentiable Matérn kernel."""
    def kernelize(self, x):
        return (1 + 6*x + 12*x**2)*_tf.math.exp(-6*x)


class RationalQuadratic(_Kernel):
    """Rational quadratic (a.k.a. Cauchy) kernel."""
    def __init__(self, scale=1):
        super().__init__()
        self._add_parameter("scale", _gpr.PositiveParameter(scale, 1e-3, 100))

    def kernelize(self, x):
        alpha = self.parameters["scale"].get_value()
        cov = (1 + 3 * x ** 2 / alpha) ** (-alpha)
        return cov


# class _RadialBasisFunction(_Kernel):
#     def __init__(self, max_distance, epsilon=1e-12):
#         super().__init__()
#         self.max_distance = max_distance
#         self.epsilon = epsilon  # required to be able to compute gradients
#
#
# class RBF3D(_RadialBasisFunction):
#     """RBF 3D"""
#     def kernelize(self, x):
#         d = _tf.sqrt(x ** 2 + self.epsilon)
#         k = 2 * d**3 + 3 * d**2 * self.max_distance + self.max_distance**3
#         k = k / self.max_distance**3
#         return k
#
#
# class RBF2D(_RadialBasisFunction):
#     """Thin plate spline RBF (2D)"""
#     def kernelize(self, x):
#         d = _tf.sqrt(x ** 2 + self.epsilon)
#         k = 2 * _tf.math.log(d) * d ** 2 \
#             - (1 + 2*_np.log(self.max_distance)) * d**2 \
#             + self.max_distance**2
#         k = k / self.max_distance ** 2
#         return k
#
#
# class RBF1D(_RadialBasisFunction):
#     """Cubic RBF (1D)"""
#     def kernelize(self, x):
#         d = _tf.sqrt(x ** 2 + self.epsilon)
#         k = 2*d**3 - 3 * d**2 * self.max_distance + self.max_distance**3
#         k = k / self.max_distance ** 3
#         return k


class _AbstractCovariance(_gpr.Parametric):
    """Abstract covariance function class"""

    def __init__(self):
        super().__init__()
        self._has_compact_support = False

    @property
    def has_compact_support(self):
        return self._has_compact_support

    def covariance_matrix(self, x, y):
        """Computes point-point covariance matrix between x and y tensors."""
        raise NotImplementedError()

    def covariance_matrix_d1(self, x, y, dir_y):
        """
        Computes point-direction covariance matrix between x and y tensors.
        """
        # if step is None:
        step = 1e-3

        min_coords = _tf.reduce_min(y, axis=0, keepdims=True)
        x = x - min_coords
        y = y - min_coords

        k1 = self.covariance_matrix(x, y + 0.5*step*dir_y)
        k2 = self.covariance_matrix(x, y - 0.5*step*dir_y)
        return (k1 - k2)/step

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        """
        Computes direction-direction covariance matrix between x and y tensors.
        """
        # if step is None:
        step = 1e-3

        min_coords = _tf.reduce_min(y, axis=0, keepdims=True)
        x = x - min_coords
        y = y - min_coords

        k1 = self.covariance_matrix_d1(x + 0.5*step*dir_x, y, dir_y)
        k2 = self.covariance_matrix_d1(x - 0.5*step*dir_x, y, dir_y)
        return (k1 - k2) / step

    def point_variance(self, x):
        """
        Computes the data points' self variance (covariance between the point
        and itself).
        """
        raise NotImplementedError()

    def self_covariance_matrix(self, x):
        return self.covariance_matrix(x, x)

    def self_covariance_matrix_d2(self, x, dir_x):
        return self.covariance_matrix_d2(x, x, dir_x, dir_x)

    def point_variance_d2(self, x, dir_x, step=None):
        if step is None:
            step = 1e-3

        min_coords = _tf.reduce_min(x, axis=0, keepdims=True)
        x = x - min_coords

        def loop_fn(elems):
            y = _tf.expand_dims(elems[0], 0)
            dir_y = _tf.expand_dims(elems[1], 0)
            k = self.covariance_matrix(y, y + dir_y*step)
            return _tf.squeeze(k)

        cov_2 = _tf.map_fn(loop_fn, [x, dir_x], dtype=_tf.float64,
                           parallel_iterations=1000)
        cov_0 = self.point_variance(x)

        return 2 * (cov_0 - cov_2) / step**2

    def set_limits(self, data):
        pass

    def implicit_matmul(self, coordinates):
        """
        Implicit matrix-vector multiplication.

        Returns a function that multiplies the kernel's covariance matrix
        (defined at the given coordinates) with a vector efficiently.
        """
        pass

    def sparse_covariance_matrix(self, x, y):
        """
        Sparse covariance matrix.

        The kernel must have compact support in order to build a sparse matrix.
        The shorter the kernel range relative to the spatial region, the higher
        the matrix's sparsity.
        """
        if not self.has_compact_support:
            raise AssertionError("kernel must have compact support")

        with _tf.name_scope("sparse_covariance_matrix"):
            cov_mat = self.covariance_matrix(x, y)
            cov_mat = _tf.sparse.from_dense(cov_mat)
        return cov_mat

    def self_full_directional_covariance(self, x):
        ndim = _tf.shape(x)[1]
        n_data = _tf.shape(x)[0]
        eye = _tf.eye(ndim, dtype=_tf.float64)
        directions = _tf.tile(eye, [1, n_data])
        directions = _tf.reshape(directions, [n_data * ndim, ndim])
        x_tile = _tf.tile(x, [ndim, 1])

        cov_d0 = self.self_covariance_matrix(x)
        cov_d1 = self.covariance_matrix_d1(x, x_tile, directions)
        cov_d2 = self.self_covariance_matrix_d2(x_tile, directions)

        full_cov = _tf.concat(
            [_tf.concat([cov_d0, cov_d1], axis=1),
             _tf.concat([_tf.transpose(cov_d1), cov_d2], axis=1)],
            axis=0)
        return full_cov

    def full_directional_covariance(self, x, base):
        ndim = _tf.shape(x)[1]
        n_base = _tf.shape(base)[0]
        eye = _tf.eye(ndim, dtype=_tf.float64)
        directions = _tf.tile(eye, [1, n_base])
        directions = _tf.reshape(directions, [n_base * ndim, ndim])

        cov_d0 = self.covariance_matrix(x, base)

        base = _tf.tile(base, [ndim, 1])
        cov_d1 = self.covariance_matrix_d1(x, base, directions)

        full_cov = _tf.concat([cov_d0, cov_d1], axis=1)
        return full_cov

    def full_directional_covariance_d1(self, x, directions, base):
        ndim = _tf.shape(x)[1]
        n_base = _tf.shape(base)[0]
        eye = _tf.eye(ndim, dtype=_tf.float64)
        base_directions = _tf.tile(eye, [1, n_base])
        base_directions = _tf.reshape(base_directions, [n_base * ndim, ndim])

        cov_d1 = _tf.transpose(self.covariance_matrix_d1(base, x, directions))

        base = _tf.tile(base, [ndim, 1])
        cov_d2 = self.covariance_matrix_d2(x, base, directions, base_directions)

        full_cov = _tf.concat([cov_d1, cov_d2], axis=1)
        return full_cov


class Covariance(_AbstractCovariance):
    """Covariance function."""

    def __init__(self, kernel, transform=_gt.Identity()):
        """
        Initializer for Covariance.

        Parameters
        ----------
        kernel
            A kernel object.
        transform
            An object from the `transform` module.
        """
        super().__init__()
        self.kernel = self._register(kernel)
        self.transform = self._register(transform)
        self._has_compact_support = self.kernel.has_compact_support

    def covariance_matrix(self, x, y):
        with _tf.name_scope(self.__class__.__name__ + "_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            d = _pairwise_dist(x, y)
            k = self.kernel.kernelize(d)
        return k

    def point_variance(self, x):
        with _tf.name_scope("Kernel_point_var"):
            v = _tf.ones([_tf.shape(x)[0]], dtype=_tf.float64)
        return v

    def pretty_print(self, depth=0):
        s = ""
        s += "  " * depth + self.__class__.__name__
        if self.has_compact_support:
            s += " (compact)"
        s += "\n"
        depth += 1
        for name, parameter in self.parameters.items():
            s += "  " * depth + name + ": " \
                 + str(parameter.get_value().numpy())
            if parameter.fixed:
                s += " (fixed)"
            s += "\n"
        s += self.transform.pretty_print(depth)
        return s

    def set_limits(self, data):
        self.transform.set_limits(data)

    def implicit_matmul(self, coordinates):
        cov_mat = self.self_covariance_matrix(coordinates)

        def matmul_fn(vector):
            return _tf.matmul(cov_mat, vector)

        return matmul_fn

    def feature_matrix(self, x):
        raise NotImplementedError()


class _NodeCovariance(_AbstractCovariance):
    """A covariance operation on another covariance"""

    def __init__(self, *args):
        super().__init__()
        self.components = args
        for arg in args:
            self._register(arg)

    def _operation(self, arg_list):
        raise NotImplementedError

    def covariance_matrix(self, x, y):
        k = self._operation(
            [kernel.covariance_matrix(x, y)
             for kernel in self.components]
        )
        return k

    def covariance_matrix_d1(self, x, y, dir_y):
        k = self._operation(
            [kernel.covariance_matrix_d1(x, y, dir_y)
             for kernel in self.components]
        )
        return k

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        k = self._operation(
            [kernel.covariance_matrix_d2(x, y, dir_x, dir_y)
             for kernel in self.components]
        )
        return k

    def point_variance(self, x):
        v = self._operation(
            [kernel.point_variance(x) for kernel in self.components]
        )
        return v

    def self_covariance_matrix(self, x):
        k = self._operation(
            [kernel.self_covariance_matrix(x)
             for kernel in self.components]
        )
        return k

    def self_covariance_matrix_d2(self, x, dir_x):
        k = self._operation(
            [kernel.self_covariance_matrix_d2(x, dir_x)
             for kernel in self.components]
        )
        return k

    def pretty_print(self, depth=0):
        s = ""
        s += "  " * depth + self.__class__.__name__
        if self.has_compact_support:
            s += " (compact)"
        s += "\n"
        for name, parameter in self.parameters.items():
            s += "  " * depth + name + ": " \
                 + str(parameter.get_value().numpy())
            if parameter.fixed:
                s += " (fixed)"
            s += "\n"
        for kernel in self.components:
            s += kernel.pretty_print(depth + 1)
        return s

    def set_limits(self, data):
        for comp in self.components:
            comp.set_limits(data)

    def implicit_matmul(self, coordinates,
                        points_to_honor=None):
        funs = [comp.implicit_matmul(coordinates, points_to_honor)
                for comp in self.components]

        def matmul_fn(vector):
            results = [fun(vector) for fun in funs]
            return self._operation(results)

        return matmul_fn


class _WrapperCovariance(_AbstractCovariance):
    def __init__(self, base_covariance):
        super().__init__()
        self.base_covariance = self._register(base_covariance)
        self._has_compact_support = self.base_covariance.has_compact_support

    def pretty_print(self, depth=0):
        s = ""
        s += "  " * depth + self.__class__.__name__
        if self.has_compact_support:
            s += " (compact)"
        s += "\n"
        depth += 1
        for name, parameter in self.parameters.items():
            s += "  " * depth + name + ": " \
                 + str(parameter.get_value().numpy())
            if parameter.fixed:
                s += " (fixed)"
            s += "\n"
        s += self.base_covariance.pretty_print(depth)
        return s


class Linear(_AbstractCovariance):
    """Linear covariance"""
    def __init__(self, transform=_gt.Identity()):
        super().__init__()
        self.transform = self._register(transform)

    def covariance_matrix(self, x, y):
        with _tf.name_scope("Linear_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            k = _tf.matmul(x, y, False, True)
        return k

    def point_variance(self, x):
        with _tf.name_scope("Linear_point_var"):
            x = self.transform.__call__(x)
            v = _tf.reduce_sum(_tf.pow(x, 2), 1)
        return v

    def implicit_matmul(self, coordinates):
        def matmul_fn(vector):
            result = _tf.matmul(coordinates, vector, True, False)
            result = _tf.matmul(coordinates, result)
            return result

        return matmul_fn

    def feature_matrix(self, x):
        return self.transform(x)


class Sum(_NodeCovariance):
    """Kernel sum"""
    def __init__(self, *args):
        """
        Kernel sum.

        Parameters
        ----------
        args
            Kernels to compute the sum.
        """
        n_comp = len(args)
        v = _gpr.CompositionalParameter(_tf.ones([n_comp], _tf.float64)/n_comp)

        super().__init__(*args)
        self.parameters = {"variance": v}
        self._all_parameters.append(v)
        self._has_compact_support = all([kernel.has_compact_support
                                         for kernel in args])

    def _operation(self, arg_list):
        k = _tf.zeros_like(arg_list[0])
        for i, comp in enumerate(arg_list):
            k = k + self.parameters["variance"].get_value()[i] * comp
        return k


class Product(_NodeCovariance):
    """Kernel product"""
    def __init__(self, *args):
        """
        Kernel product.

        Parameters
        ----------
        args
            Kernels to compute the product.
        """
        super().__init__(*args)
        self._has_compact_support = any([kernel.has_compact_support
                                         for kernel in args])

    def _operation(self, arg_list):
        return _prod_n(arg_list)


class Scale(_WrapperCovariance):
    """
    Kernel scaling.

    Add a parameter allowing for non-unit variance.
    """
    def __init__(self, base_covariance):
        super().__init__(base_covariance)
        self._add_parameter("amplitude", _gpr.PositiveParameter(1.0, 1e-4, 1e4))

    def covariance_matrix(self, x, y):
        return self.parameters["amplitude"].get_value() \
               * self.base_covariance.covariance_matrix(x, y)

    def point_variance(self, x):
        return self.parameters["amplitude"].get_value() \
               * self.base_covariance.point_variance(x)
