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

from geoml.math.tf import pairwise_dist as _pairwise_dist
# the other two landed on the linalg side of the 0.6.0 tftools split, which
# is why this used to reach for them through the `geoml.tftools` shim
from geoml.math.linalg import pairwise_dist_l1 as _pairwise_dist_l1
from geoml.math.linalg import prod_n as _prod_n

import tensorflow as _tf
import numpy as _np


# How much of `na - 2 a.b + nb` is rounding, relative to the magnitudes that
# go into it. A squared distance below this is indistinguishable from zero
# and is read as zero, so coincident points reach the exact `r = 0` branch of
# the chain rule instead of a distance at rounding level -- which for a
# kernel with an odd term in r (Cubic, both Materns) would carry straight
# into f'(r)/r. Generous by an order of magnitude: it only ever zeroes
# separations below ~1e-8 of the point cloud's own extent.
_DISTANCE_ROUNDING = 8 * float(_np.finfo(_np.float64).eps)


def _kernel_derivatives(kernel, dist):
    """
    First and second derivatives of a kernel with respect to distance.

    `kernelize` is elementwise, so its Jacobian is diagonal and one reverse
    pass per order returns the derivative of every entry at once.

    Parameters
    ----------
    kernel
        A kernel object.
    dist
        Distances at which to differentiate.

    Returns
    -------
    first, second
        Tensors shaped like `dist`.
    """
    with _tf.GradientTape() as outer:
        outer.watch(dist)
        with _tf.GradientTape() as inner:
            inner.watch(dist)
            k = kernel.kernelize(dist)
        first = inner.gradient(k, dist)
        if first is None:                       # a kernel with no distance in it
            first = _tf.zeros_like(dist)
    second = outer.gradient(first, dist)
    if second is None:
        second = _tf.zeros_like(dist)
    return first, second


def _transform_jvp(transform, x, direction):
    """
    A transform applied to `x`, and a direction pushed through its Jacobian.

    Parameters
    ----------
    transform
        An object from the `transform` module.
    x
        Coordinates.
    direction
        One direction per row of `x`, in the untransformed space.

    Returns
    -------
    transformed, pushed
        The transformed coordinates and the directions carried with them.
    """
    # callers reach here with plain arrays as often as with tensors, and an
    # accumulator can only watch a tensor. It also wants one tangent per
    # row: a single direction meant for every point (`StructuralField`
    # predicts along one mean vector) has to be spelled out first, where
    # the arithmetic it replaces would have broadcast it.
    x = _tf.convert_to_tensor(x, _tf.float64)
    direction = _tf.broadcast_to(
        _tf.convert_to_tensor(direction, _tf.float64), _tf.shape(x))
    with _tf.autodiff.ForwardAccumulator(x, direction) as acc:
        transformed = transform.__call__(x)
    pushed = acc.jvp(transformed)
    if pushed is None:                          # a transform ignoring its input
        pushed = _tf.zeros_like(transformed)
    return transformed, pushed


class _Kernel(_gpr.Parametric):
    def __init__(self):
        super().__init__()
        self._has_compact_support = False

    @property
    def has_compact_support(self):
        return self._has_compact_support

    def _summary_line(self):
        s = self.__class__.__name__
        if self.has_compact_support:
            s += " (compact)"
        return s

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
    def __init__(self, epsilon: float = 1e-12):
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
    def __init__(self, epsilon: float = 1e-12):
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
    def __init__(self, scale: float = 1):
        super().__init__()
        self._add_parameter("scale", _gpr.PositiveParameter(scale, 1e-3, 100))

    def kernelize(self, x):
        alpha = self.parameters["scale"].get_value()
        cov = (1 + 3 * x ** 2 / alpha) ** (-alpha)
        return cov


class _AbstractCovariance(_gpr.Parametric):
    """Abstract covariance function class"""

    def __init__(self):
        super().__init__()
        self._has_compact_support = False

    @property
    def has_compact_support(self):
        return self._has_compact_support

    def _summary_line(self):
        s = self.__class__.__name__
        if self.has_compact_support:
            s += " (compact)"
        return s

    def covariance_matrix(self, x, y):
        """Computes point-point covariance matrix between x and y tensors."""
        raise NotImplementedError()

    def covariance_matrix_d1(self, x, y, dir_y):
        """
        Computes point-direction covariance matrix between x and y tensors.
        """
        # Forward mode fits the shape of the answer: entry (i, j) involves
        # only row i of x and row j of y, so a single pass along `dir_y`
        # fills the whole matrix, where reverse mode would contract it.
        y = _tf.convert_to_tensor(y, _tf.float64)
        dir_y = _tf.broadcast_to(
            _tf.convert_to_tensor(dir_y, _tf.float64), _tf.shape(y))
        with _tf.autodiff.ForwardAccumulator(y, dir_y) as acc:
            k = self.covariance_matrix(x, y)
        return acc.jvp(k)

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        """
        Computes direction-direction covariance matrix between x and y tensors.
        """
        # `self_covariance_matrix_d2` passes one tensor twice, and each
        # accumulator has to watch a tensor of its own; the copy is made
        # before the outer context opens, or it inherits the outer tangent
        # and the whole thing differentiates K(x + t.v, x + t.v) instead.
        x = _tf.convert_to_tensor(x, _tf.float64)
        y = _tf.identity(_tf.convert_to_tensor(y, _tf.float64))
        dir_x = _tf.broadcast_to(
            _tf.convert_to_tensor(dir_x, _tf.float64), _tf.shape(x))
        dir_y = _tf.broadcast_to(
            _tf.convert_to_tensor(dir_y, _tf.float64), _tf.shape(y))
        with _tf.autodiff.ForwardAccumulator(x, dir_x) as acc_x:
            with _tf.autodiff.ForwardAccumulator(y, dir_y) as acc_y:
                k = self.covariance_matrix(x, y)
            k_d1 = acc_y.jvp(k)
        return acc_x.jvp(k_d1)

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

    def point_variance_d2(self, x, dir_x):
        return _tf.linalg.diag_part(
            self.covariance_matrix_d2(x, x, dir_x, dir_x))

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

    def __init__(self, kernel: "_Kernel",
                 transform: "_gt._Transform" = _gt.Identity()):
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

    # The base class differentiates the covariance matrix itself, which
    # cannot work here: this covariance goes through a Euclidean distance,
    # and |h| is not twice differentiable at h = 0 -- exactly the diagonal
    # every d2 call has. Differentiating the *kernel* instead moves the
    # autodiff onto a one-dimensional elementwise function, where it is
    # exact, and leaves the spatial part to the chain rule below, which is
    # regular at the origin.

    def _radial_derivatives(self, p, q):
        """
        The radial factors of the directional covariances.

        With `K = f(r)` and `r` the distance between transformed
        coordinates, every directional derivative is built from
        `phi = f'(r)/r` and `psi = (f''(r) - phi)/r ** 2`. Both have
        removable singularities at the origin, where `phi` tends to `f''(0)`
        and `psi` is multiplied by a factor that vanishes.

        Parameters
        ----------
        p, q
            Transformed coordinates.

        Returns
        -------
        phi, psi
            Matrices with one entry per pair.
        """
        # Distances are measured about a common origin: only differences
        # reach the kernel, and subtracting it keeps `na - 2 p.q + nb` away
        # from the cancellation that mine-grid coordinates would cause.
        origin = _tf.stop_gradient(_tf.reduce_min(q, axis=0, keepdims=True))
        p_local = p - origin
        q_local = q - origin
        na = _tf.reduce_sum(p_local ** 2, axis=1)[:, None]
        nb = _tf.reduce_sum(q_local ** 2, axis=1)[None, :]
        sq_dist = na - 2 * _tf.matmul(p_local, q_local, False, True) + nb
        sq_dist = _tf.where(sq_dist > _DISTANCE_ROUNDING * (na + nb),
                            sq_dist, _tf.zeros_like(sq_dist))
        dist = _tf.sqrt(sq_dist)

        first, second = _kernel_derivatives(self.kernel, dist)
        apart = dist > 0.0
        safe = _tf.where(apart, dist, _tf.ones_like(dist))
        phi = _tf.where(apart, first / safe, second)
        psi = _tf.where(apart, (second - phi) / safe ** 2,
                        _tf.zeros_like(dist))
        return phi, psi

    def covariance_matrix_d1(self, x, y, dir_y):
        with _tf.name_scope(self.__class__.__name__ + "_cov_d1"):
            p = self.transform.__call__(x)
            q, q_dir = _transform_jvp(self.transform, y, dir_y)
            phi, _ = self._radial_derivatives(p, q)
            # a[i, j] = (p_i - q_j) . q_dir_j, without ever forming h
            a = (_tf.matmul(p, q_dir, False, True)
                 - _tf.reduce_sum(q * q_dir, axis=1)[None, :])
            k = -phi * a
        return k

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        with _tf.name_scope(self.__class__.__name__ + "_cov_d2"):
            p, p_dir = _transform_jvp(self.transform, x, dir_x)
            q, q_dir = _transform_jvp(self.transform, y, dir_y)
            phi, psi = self._radial_derivatives(p, q)
            a = (_tf.matmul(p, q_dir, False, True)
                 - _tf.reduce_sum(q * q_dir, axis=1)[None, :])
            b = (_tf.reduce_sum(p * p_dir, axis=1)[:, None]
                 - _tf.matmul(p_dir, q, False, True))
            c = _tf.matmul(p_dir, q_dir, False, True)
            k = -(psi * a * b + phi * c)
        return k

    def point_variance_d2(self, x, dir_x):
        with _tf.name_scope("Kernel_point_var_d2"):
            # a and b vanish where a point meets itself, leaving -f''(0) c
            _, p_dir = _transform_jvp(self.transform, x, dir_x)
            _, second = _kernel_derivatives(
                self.kernel, _tf.zeros([_tf.shape(x)[0]], _tf.float64))
            v = -second * _tf.reduce_sum(p_dir ** 2, axis=1)
        return v

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

    # Applying the operation to the components' derivatives is the
    # derivative of the operation only where that operation is linear --
    # true of `Sum`, and the reason `Product` overrides all three.
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

    # A product is not linear in its components, so the delegation above
    # would return the product of the derivatives where the product rule
    # is wanted. The rule needs the derivative in the *first* argument,
    # which the interface does not offer -- and does not need to: a
    # covariance is symmetric, `K(x, y) = K(y, x)'`, so the derivative in x
    # is the derivative in y with the arguments swapped and the result
    # transposed. The same identity assembles the mixed blocks in
    # `full_directional_covariance_d1` and in `GradientConstrainedInput`.
    @staticmethod
    def _all_but(matrices, *skip):
        """The product of every component's matrix except the ones named."""
        kept = [mat for i, mat in enumerate(matrices) if i not in skip]
        if len(kept) == 0:
            return _tf.ones_like(matrices[0])
        return _prod_n(kept)

    def covariance_matrix_d1(self, x, y, dir_y):
        values = [kernel.covariance_matrix(x, y)
                  for kernel in self.components]
        d_y = [kernel.covariance_matrix_d1(x, y, dir_y)
               for kernel in self.components]
        return _tf.add_n([d * self._all_but(values, i)
                          for i, d in enumerate(d_y)])

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        values = [kernel.covariance_matrix(x, y)
                  for kernel in self.components]
        d_y = [kernel.covariance_matrix_d1(x, y, dir_y)
               for kernel in self.components]
        d_x = [_tf.transpose(kernel.covariance_matrix_d1(y, x, dir_x))
               for kernel in self.components]
        d_xy = [kernel.covariance_matrix_d2(x, y, dir_x, dir_y)
                for kernel in self.components]

        terms = [d * self._all_but(values, i) for i, d in enumerate(d_xy)]
        terms += [d_y[i] * d_x[j] * self._all_but(values, i, j)
                  for i in range(len(values))
                  for j in range(len(values)) if j != i]
        return _tf.add_n(terms)

    def self_covariance_matrix_d2(self, x, dir_x):
        return self.covariance_matrix_d2(x, x, dir_x, dir_x)


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

    # Scaling passes through a derivative, so these delegate rather than
    # inherit: the base class would differentiate `covariance_matrix` above,
    # and with a distance-based covariance underneath that is the one thing
    # it cannot do.
    def covariance_matrix_d1(self, x, y, dir_y):
        return self.parameters["amplitude"].get_value() \
               * self.base_covariance.covariance_matrix_d1(x, y, dir_y)

    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        return self.parameters["amplitude"].get_value() \
               * self.base_covariance.covariance_matrix_d2(x, y, dir_x, dir_y)

    def point_variance_d2(self, x, dir_x):
        return self.parameters["amplitude"].get_value() \
               * self.base_covariance.point_variance_d2(x, dir_x)
