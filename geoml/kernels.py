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

__all__ = ["Gaussian",
           "Spherical",
           "Exponential",
           "Cubic",
           "Constant",
           "Linear",
           "Cosine",
           "Nugget",
           "Sum",
           "Product"]

import geoml.parameter as _gpr
import geoml.transform as _gt

from geoml.tftools import pairwise_dist as _pairwise_dist
from geoml.tftools import prod_n as _prod_n

import tensorflow as _tf
import numpy as _np


class _Kernel(object):
    """Abstract kernel class"""

    def __init__(self):
        self._all_parameters = []
        self.parameters = {}
        self._has_compact_support = False

    @property
    def all_parameters(self):
        return self._all_parameters

    @property
    def has_compact_support(self):
        return self._has_compact_support

    def covariance_matrix(self, x, y):
        """Computes point-point covariance matrix between x and y tensors."""
        pass

    def covariance_matrix_d1(self, x, y, dir_y):
        """
        Computes direction-point covariance matrix between x and y tensors.
        """
        step = 1e-3
        k1 = self.covariance_matrix(x, y + 0.5*step*dir_y)
        k2 = self.covariance_matrix(x, y - 0.5*step*dir_y)
        return (k1 - k2)/step

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        """
        Computes direction-direction covariance matrix between x and y tensors.
        """
        step = 1e-3
        k1 = self.covariance_matrix_d1(x + 0.5*step*dir_x, y, dir_y)
        k2 = self.covariance_matrix_d1(x - 0.5*step*dir_x, y, dir_y)
        return (k1 - k2) / step

    def point_variance(self, x):
        """
        Computes the data points' self variance (covariance between the point
        and itself).
        """
        pass

    def self_covariance_matrix(self, x, points_to_honor=None):
        return self.covariance_matrix(x, x)

    def self_covariance_matrix_d2(self, x, dir_x):
        return self.covariance_matrix_d2(x, x, dir_x, dir_x)

    def pretty_print(self, depth=0):
        pass

    def __repr__(self):
        return self.pretty_print()

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

        min_val = _np.concatenate(min_val, axis=0)
        max_val = _np.concatenate(max_val, axis=0)
        value = _np.concatenate(value, axis=0)

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

    def set_limits(self, data):
        pass

    def implicit_matmul(self, coordinates, points_to_honor=None):
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


class _LeafKernel(_Kernel):
    """Kernel that acts on actual coordinates"""

    def __init__(self, transform=_gt.Identity()):
        super().__init__()
        self.transform = transform
        self._all_parameters = [pr for pr in self.parameters.values()] \
                             + [pr for pr in self.transform.parameters.values()]

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
                 + str(parameter.value.value().numpy())
            if parameter.fixed:
                s += " (fixed)"
            s += "\n"
        s += self.transform.pretty_print(depth)
        return s

    def set_limits(self, data):
        self.transform.set_limits(data)

    def implicit_matmul(self, coordinates, points_to_honor=None):
        cov_mat = self.self_covariance_matrix(coordinates, points_to_honor)

        def matmul_fn(vector):
            return _tf.matmul(cov_mat, vector)

        return matmul_fn


class _NodeKernel(_Kernel):
    """A kernel operation on other kernel"""

    def __init__(self, *args):
        super().__init__()

        self.components = args
        self._all_parameters = [kernel.all_parameters for kernel in args]
        self._all_parameters = [item for sublist in self._all_parameters
                                for item in sublist]
        self._all_parameters += [pr for pr in self.parameters.values()]
        self.nugget_position = [i for i, comp in enumerate(self.components)
                                if isinstance(comp, Nugget)]

    def _operation(self, arg_list):
        raise NotImplementedError

    def covariance_matrix(self, x, y):
        k = self._operation(
            [kernel.covariance_matrix(x, y) for kernel in self.components]
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

    def self_covariance_matrix(self, x, points_to_honor=None):
        k = self._operation(
            [kernel.self_covariance_matrix(x, points_to_honor)
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
                 + str(parameter.value.value().numpy())
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

    def nugget_matmul(self, coordinates,
                      points_to_honor=None):
        funs = [self.components[i].implicit_matmul(
            coordinates, points_to_honor)
            for i in self.nugget_position]

        def matmul_fn(vector):
            results = [fun(vector) for fun in funs]
            return self._operation(results)

        return matmul_fn

    def nugget_variance(self, x):
        v = self._operation(
            [kernel.point_variance(x) for kernel in self.components
             if isinstance(kernel, Nugget)]
        )
        return v


class Gaussian(_LeafKernel):
    """Gaussian kernel"""

    def covariance_matrix(self, x, y):
        with _tf.name_scope("Gaussian_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            d2 = _tf.pow(_pairwise_dist(x, y), 2)
            k = _tf.exp(-3 * d2)
        return k


class Spherical(_LeafKernel):
    """Spherical kernel"""
    def __init__(self, transform=_gt.Identity()):
        super().__init__(transform)
        self._has_compact_support = True

    def covariance_matrix(self, x, y):
        with _tf.name_scope("Spherical_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            d = _pairwise_dist(x, y)
            k = 1 - 1.5 * d + 0.5 * _tf.pow(d, 3)
            k = _tf.where(_tf.less(d, 1.0), k, _tf.zeros_like(k))
        return k


class Exponential(_LeafKernel):
    """Exponential kernel"""

    def covariance_matrix(self, x, y):
        with _tf.name_scope("Exponential_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            d = _pairwise_dist(x, y)
            k = _tf.exp(-3 * d)
        return k


class Cubic(_LeafKernel):
    """Cubic kernel"""
    def __init__(self, transform=_gt.Identity()):
        super().__init__(transform)
        self._has_compact_support = True

    def covariance_matrix(self, x, y):
        with _tf.name_scope("Cubic_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            d = _pairwise_dist(x, y)
            k = 1 - 7 * _tf.pow(d, 2) + 35 / 4 * _tf.pow(d, 3) \
                - 7 / 2 * _tf.pow(d, 5) + 3 / 4 * _tf.pow(d, 7)
            k = _tf.where(_tf.less(d, 1.0), k, _tf.zeros_like(k))
        return k


class Constant(_LeafKernel):
    """Constant kernel"""

    def covariance_matrix(self, x, y):
        with _tf.name_scope("Constant_cov"):
            k = _tf.ones([_tf.shape(x)[0], _tf.shape(y)[0]], dtype=_tf.float64)
        return k

    def implicit_matmul(self, coordinates, points_to_honor=None):
        def matmul_fn(vector):
            result = _tf.ones_like(vector) * _tf.reduce_sum(vector)
            return result

        return matmul_fn


class Linear(_LeafKernel):
    """Linear kernel"""

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

    def implicit_matmul(self, coordinates, points_to_honor=None):
        def matmul_fn(vector):
            result = _tf.matmul(coordinates, vector, True, False)
            result = _tf.matmul(coordinates, result)
            return result

        return matmul_fn


class Cosine(_LeafKernel):
    """Cosine kernel"""

    def covariance_matrix(self, x, y):
        with _tf.name_scope("Cosine_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            d = _pairwise_dist(x, y)
            k = _tf.cos(2.0 * _np.pi * d)
        return k

    def covariance_matrix_d1(self, x, y, dir_y):
        with _tf.name_scope("Cosine_point_dir_cov"):
            x = self.transform.__call__(x)
            y = self.transform.__call__(y)
            dir_y = self.transform.__call__(dir_y)
            sx = _tf.shape(x, name="shape_x")
            x_prod = _tf.matmul(x, dir_y, False, True)
            y_prod = _tf.reduce_sum(y * dir_y, axis=1, keepdims=True)
            y_prod = _tf.transpose(y_prod)
            y_prod = _tf.tile(y_prod, [sx[0], 1])
            dif = x_prod - y_prod
            d = _pairwise_dist(x, y)
            k = - _tf.sin(2.0 * _np.pi * d)
            k = k * dif * 2.0 * _np.pi / (d + 1e-9)
        return k

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        raise NotImplementedError()


class Nugget(_LeafKernel):
    """Nugget effect"""
    def __init__(self):
        super().__init__()
        self._has_compact_support = True

    def covariance_matrix(self, x, y):
        sx = _tf.shape(x)[0]
        sy = _tf.shape(y)[0]
        return _tf.zeros([sx, sy], _tf.float64)

    def covariance_matrix_d1(self, x, y, dir_y):
        return self.covariance_matrix(x, y)

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        return self.covariance_matrix(x, y)

    def self_covariance_matrix(self, x, points_to_honor=None):
        if points_to_honor is None:
            return _tf.eye(x.shape[0], dtype=_tf.float64)
        else:
            nugget_vector = _tf.where(
                points_to_honor,
                _tf.zeros(_tf.shape(points_to_honor), dtype=_tf.float64),
                _tf.ones(_tf.shape(points_to_honor), dtype=_tf.float64))
            return _tf.linalg.diag(nugget_vector)

    def implicit_matmul(self, coordinates, points_to_honor=None):
        if points_to_honor is None:
            return lambda vector: vector
        else:
            nugget_vector = _tf.where(
                points_to_honor,
                _tf.zeros([coordinates.shape[0]], dtype=_tf.float64),
                _tf.ones([coordinates.shape[0]], dtype=_tf.float64))
            return lambda vector: vector * _tf.expand_dims(nugget_vector, 1)


class Sum(_NodeKernel):
    """Kernel sum"""
    def __init__(self, *args):
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
            k = k + self.parameters["variance"].value[i] * comp
        return k

    def nugget_matmul(self, coordinates,
                      points_to_honor=None):
        funs = [self.components[i].implicit_matmul(
            coordinates, points_to_honor)
            for i in self.nugget_position]

        def matmul_fn(vector):
            results = [fun(vector)*self.parameters["variance"].value[i]
                       for fun, i in zip(funs, self.nugget_position)]
            return _tf.add_n(results)

        return matmul_fn

    def nugget_variance(self, x):
        nugget_comp = [self.components[i] for i in self.nugget_position]

        v = [self.parameters["variance"].value[i] * comp.point_variance(x)
             for comp, i in zip(nugget_comp, self.nugget_position)]
        return _tf.add_n(v)


class Product(_NodeKernel):
    """Kernel product"""
    def __init__(self, *args):
        super().__init__(*args)
        self._has_compact_support = any([kernel.has_compact_support
                                         for kernel in args])

    def _operation(self, arg_list):
        return _prod_n(arg_list)


class Matern32(_LeafKernel):
    def covariance_matrix(self, x, y):
        x = self.transform(x)
        y = self.transform(y)
        d = _pairwise_dist(x, y)
        cov = (1 + 5*d)*_tf.math.exp(-5*d)
        return cov


class Matern52(_LeafKernel):
    def covariance_matrix(self, x, y):
        x = self.transform(x)
        y = self.transform(y)
        d = _pairwise_dist(x, y)
        cov = (1 + 6*d + 12*d**2)*_tf.math.exp(-6*d)
        return cov
