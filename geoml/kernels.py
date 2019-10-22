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

__all__ = ["GaussianKernel", "SphericalKernel", "ExponentialKernel",
           "CubicKernel", "ConstantKernel", "LinearKernel", "Nugget"]

import tensorflow as _tf
import numpy as _np
import pickle as _pickle
import geoml.parameter as _gpr
import geoml.transform as _gt

from geoml.tftools import pairwise_dist as _pairwise_dist
from geoml.tftools import prod_n as _prod_n
from geoml.tftools import extract_features as _extract_features


class _Kernel(object):
    """Abstract kernel class"""
    def __init__(self, transform=_gt.Identity()):
        self.transform = transform
        
    def covariance_matrix(self, x, y=None):
        """Computes point-point covariance matrix between x and y tensors."""
        pass
    
    def covariance_matrix_d1(self, x, y, dir_y):
        """
        Computes direction-point covariance matrix between x and y tensors.
        """
        raise Exception("Point-direction covariance not available for this " 
                        + "kernel")
    
    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        """
        Computes direction-direction covariance matrix between x and y tensors.
        """
        raise Exception("Direction-direction covariance not available for "  
                        + "this kernel")
    
    def point_variance(self, x):
        """
        Computes the data points' self variance (covariance between the point
        and itself).
        """
        with _tf.name_scope("Kernel_point_var"):
            v = _tf.ones([_tf.shape(x)[0]], dtype=_tf.float64)
        return v

    def feature_matrix(self, x, min_var=0.999):
        """
        Factorizes the covariance matrix in a compressed form.

        Parameters
        ----------
        x : Tensor
            The coordinates.
        min_var : double
            The minimum amount of variance to retain.

        Returns
        -------
        features : Tensor
            A feature matrix F, such that F^T*F approximates the original
            covariance matrix.
        """
        with _tf.name_scope("Feature_matrix"):
            k = self.covariance_matrix(x)
            features = _extract_features(k, min_var)
        return features


class GaussianKernel(_Kernel):
    """Gaussian kernel"""
    def covariance_matrix(self, x, y=None):
        with _tf.name_scope("Gaussian_cov"):
            if y is None:
                x = self.transform.backward(x)
                d2 = _tf.pow(_pairwise_dist(x, x), 2)
                k = _tf.exp(-3 * d2)
                # adding jitter to avoid Cholesky decomposition problems
                sx = _tf.shape(x)
                k = k + _tf.diag(_tf.ones(sx[0], dtype=_tf.float64) * 1e-9)
            else:
                x = self.transform.backward(x)
                y = self.transform.backward(y)
                d2 = _tf.pow(_pairwise_dist(x, y), 2)
                k = _tf.exp(-3 * d2)
        return k
    
    def covariance_matrix_d1(self, x, y, dir_y):
        with _tf.name_scope("Gaussian_point_dir_cov"):
            k_base = self.covariance_matrix(x, y)
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            dir_y = self.transform.backward(dir_y)
            sx = _tf.shape(x, name="shape_x")
            x_prod = _tf.matmul(x, dir_y, False, True)
            y_prod = _tf.reduce_sum(y * dir_y, axis=1, keepdims=True)
            y_prod = _tf.transpose(y_prod)
            y_prod = _tf.tile(y_prod, [sx[0], 1])
            k = _tf.multiply(6.0 * k_base, x_prod - y_prod)
        return k
    
    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        with _tf.name_scope("Gaussian_dir_dir_cov"):
            k_base = self.covariance_matrix(x, y)
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            dir_x = self.transform.backward(dir_x)
            dir_y = self.transform.backward(dir_y)
            sx = _tf.shape(x, name="shape_x")
            sy = _tf.shape(y, name="shape_y")
            dx_x = _tf.reduce_sum(dir_x * x, axis=1, keepdims=True)
            dx_x = _tf.tile(dx_x, [1, sy[0]])
            dy_y = _tf.reduce_sum(dir_y * y, axis=1, keepdims=True)
            dy_y = _tf.tile(_tf.transpose(dy_y), [sx[0], 1])
            dx_y = _tf.matmul(dir_x, y, False, True)
            dy_x = _tf.matmul(x, dir_y, False, True)
            prod = (dx_x - dx_y)*(dy_x - dy_y)
            k = 6.0 * k_base * (_tf.matmul(dir_x, dir_y, False, True)
                                - 6.0 * prod)
        return k


class SphericalKernel(_Kernel):
    """Spherical kernel"""
    def covariance_matrix(self, x, y=None):
        with _tf.name_scope("Spherical_cov"):
            if y is None:
                y = x
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            d = _pairwise_dist(x, y)
            k = 1 - 1.5 * d + 0.5 * _tf.pow(d, 3)
            k = _tf.where(_tf.less(d, 1.0), k, _tf.zeros_like(k))
        return k


class ExponentialKernel(_Kernel):
    """Exponential kernel"""
    def covariance_matrix(self, x, y=None):
        with _tf.name_scope("Exponential_cov"):
            if y is None:
                y = x
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            d = _pairwise_dist(x, y)
            k = _tf.exp(-3 * d)
        return k


class CubicKernel(_Kernel):
    """Cubic kernel"""
    def covariance_matrix(self, x, y=None):
        with _tf.name_scope("Cubic_cov"):
            if y is None:
                x = self.transform.backward(x)
                d = _pairwise_dist(x, x)
                k = 1 - 7 * _tf.pow(d, 2) + 35 / 4 * _tf.pow(d, 3) \
                    - 7 / 2 * _tf.pow(d, 5) + 3 / 4 * _tf.pow(d, 7)
                k = _tf.where(_tf.less(d, 1.0), k, _tf.zeros_like(k))

                # adding jitter to avoid Cholesky decomposition problems
                sx = _tf.shape(x)
                k = k + _tf.diag(_tf.ones(sx[0], dtype=_tf.float64) * 1e-9)
            else:
                x = self.transform.backward(x)
                y = self.transform.backward(y)
                d = _pairwise_dist(x, y)
                k = 1 - 7 * _tf.pow(d, 2) + 35 / 4 * _tf.pow(d, 3) \
                    - 7 / 2 * _tf.pow(d, 5) + 3 / 4 * _tf.pow(d, 7)
                k = _tf.where(_tf.less(d, 1.0), k, _tf.zeros_like(k))
        return k
    
    def covariance_matrix_d1(self, x, y, dir_y):
        with _tf.name_scope("Cubic_point_dir_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            dir_y = self.transform.backward(dir_y)
            sx = _tf.shape(x, name="shape_x")
            x_prod = _tf.matmul(x, dir_y, False, True)
            y_prod = _tf.reduce_sum(y * dir_y, axis=1, keepdims=True)
            y_prod = _tf.transpose(y_prod)
            y_prod = _tf.tile(y_prod, [sx[0], 1])
            dif = x_prod-y_prod
            d = _pairwise_dist(x, y)
            k = 14 \
                - 105 / 4 * d \
                + 35 / 2 * _tf.pow(d, 3) \
                - 21 / 4 * _tf.pow(d, 5)
            k = _tf.where(_tf.less(d, 1.0), k, _tf.zeros_like(k))
            k = k*dif
        return k
    
    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        with _tf.name_scope("Cubic_dir_dir_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            dir_x = self.transform.backward(dir_x)
            dir_y = self.transform.backward(dir_y)
            sx = _tf.shape(x, name="shape_x")
            sy = _tf.shape(y, name="shape_y")
            dx_x = _tf.reduce_sum(dir_x * x, axis=1, keepdims=True)
            dx_x = _tf.tile(dx_x, [1, sy[0]])
            dy_y = _tf.reduce_sum(dir_y * y, axis=1, keepdims=True)
            dy_y = _tf.tile(_tf.transpose(dy_y), [sx[0], 1])
            dx_y = _tf.matmul(dir_x, y, False, True)
            dy_x = _tf.matmul(x, dir_y, False, True)
            prod = (dx_x - dx_y)*(dy_x - dy_y)
            dir_prod = _tf.matmul(dir_x, dir_y, False, True)
            d = _pairwise_dist(x, y)
            k1 = 14 - 105 / 4 * d + 35 / 2 * _tf.pow(d, 3) \
                 - 21 / 4 * _tf.pow(d, 5)
            k2 = -105 / 4 / (d+1e-9) + 105 / 2 * d - 105 / 4 * _tf.pow(d, 3)
            # k1 = _tf.Variable(k1, dtype=_tf.float64, validate_shape=False)
            # k2 = _tf.Variable(k2, dtype=_tf.float64, validate_shape=False)
            # indices = _tf.where(_tf.greater(d, 1))
            # updates = _tf.zeros(_tf.shape(indices)[0], dtype=k1.dtype)
            # k1 = _tf.scatter_nd_update(k1, indices, updates)
            # k2 = _tf.scatter_nd_update(k2, indices, updates)
            k1 = _tf.where(_tf.less(d, 1.0), k1, _tf.zeros_like(k1))
            k2 = _tf.where(_tf.less(d, 1.0), k2, _tf.zeros_like(k2))
            k = k1*dir_prod + k2*prod
        return k


class ConstantKernel(_Kernel):
    """Constant kernel"""
    def covariance_matrix(self, x, y=None):
        with _tf.name_scope("Constant_cov"):
            if y is None:
                y = x
            k = _tf.ones([_tf.shape(x)[0], _tf.shape(y)[0]],
                         dtype=_tf.float64)
        return k


class LinearKernel(_Kernel):
    """Linear kernel"""       
    def covariance_matrix(self, x, y=None):
        with _tf.name_scope("Linear_cov"):
            if y is None:
                y = x
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            k = _tf.matmul(x, y, False, True)
        return k
    
    def point_variance(self, x):
        with _tf.name_scope("Linear_point_var"):
            x = self.transform.backward(x)
            v = _tf.reduce_sum(_tf.pow(x, 2), 1)
        return v

    def feature_matrix(self, x, min_var=0.999):
        with _tf.name_scope("Feature_matrix"):
            x = self.transform.backward(x)
        return x


class CosineKernel(_Kernel):
    """Cosine kernel"""
    def covariance_matrix(self, x, y=None):
        with _tf.name_scope("Cosine_cov"):
            if y is None:
                y = x
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            d = _pairwise_dist(x, y)
            k = _tf.cos(2.0 * _np.pi * d)
        return k

    def covariance_matrix_d1(self, x, y, dir_y):
        with _tf.name_scope("Cosine_point_dir_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            dir_y = self.transform.backward(dir_y)
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


class Nugget(object):
    """Nugget effect"""
    
    @staticmethod
    def nugget_matrix(x, interpolate=None):
        """
        Computes a matrix with the nugget variance of each data point.

        Parameters
        ----------
        x :
            A 2-dimensional tensor.
        interpolate :
            A boolean vector indicating which data points must be interpolated.
            The variance at these points is set to 1e-9.

        Returns
        -------
        A 2-dimensional tensor.
        """
        with _tf.name_scope("Nugget"):
            ones = _tf.ones(_tf.shape(x)[0], dtype=_tf.float64)
            if interpolate is not None:
                ones = _tf.where(interpolate,
                                 _tf.ones_like(ones) * 1e-9,
                                 _tf.ones_like(ones))
            mat = _tf.linalg.tensor_diag(ones)
        return mat


class _CovarianceModel(object):
    """
    Abstract class for kernel managers.
    """
    def __repr__(self):
        return self.__str__()


class CovarianceModelRegression(_CovarianceModel):
    """
    Gathers all the specified kernels, manages the model parameters and builds
    the covariance matrices of the appropriate size.
    
    Attributes
    ----------
    kernels :  list
        A list of lists containing the desired kernels.
    warping : list
        A list containing the Warpings for the dependent variable.
    nugget : Nugget
        A nugget object.
    variance :
        A parameter object with the relative variance of each object.
    jitter : double
        A small number to increase numerical stability during training.

    In this covariance model, it is assumed that all variances add to 1.
    The warping objects are used to scale and transform the data accordingly.
    """
    def __init__(self, kernels, warping, jitter=1e-9):
        """
        Initializer for CovarianceModelRegression object.

        Parameters
        ----------
        kernels :  list
            A list of lists containing the desired kernels.
        warping : list
            A list containing the Warpings for the dependent variable.
        jitter : double
            A small number to increase numerical stability during training.

        The kernels must be contained in a nested list with two levels.
        When building the covariance matrices, the kernels in the inner
        level are multiplied, and the results are added.
        """
        # kernels parsing
        kernels_internal = []
        for k in kernels:
            if isinstance(k, _Kernel):
                kernels_internal.append([k])
            else:
                is_kernel = [isinstance(kk, _Kernel) for kk in k]
                if not all(is_kernel):
                    raise Exception("kernels must be a nested list of "
                                    "Kernel objects")
                kernels_internal.append(k)
        self.kernels = kernels_internal

        # warping
        self.warping = warping

        # nugget and variance
        self.nugget = Nugget()
        # initializing with 10% nugget
        v = _np.concatenate([_np.repeat(0.9 / len(self.kernels),
                                        len(self.kernels)),
                             _np.array([0.1])])
        self.variance = _gpr.CompositionalParameter(v)
        self.jitter = jitter
        
    def __str__(self):
        s = "A " + self.__class__.__name__ + " object\n"

        # variance
        s += "\nVariance is"
        if self.variance.fixed:
            s += " fixed\n"
        else:
            s += " free\n"
        
        # nugget
        s += "\n" + self.nugget.__class__.__name__ + ": " \
             + str(self.variance.value[-1]) + "\n"
        
        # kernels
        for i in range(len(self.kernels)):
            # outer index
            s += "\nPosition " + str(i) + ": " \
                    "variance = " + str(self.variance.value[i]) + "\n"

            # inner index
            for j in range(len(self.kernels[i])):
                s += "\n\tPosition [" + str(i) + "][" + str(j) + "]: " \
                     + self.kernels[i][j].__class__.__name__ + " - " \
                     + self.kernels[i][j].transform.__class__.__name__ + "\n"

                # parameters
                params = self.kernels[i][j].transform.params
                if len(params) > 0:
                    par_names = [x for x in params.keys()]
                    for name in par_names:
                        par = params[name]
                        s += "\t\t" + name + ": " + str(par.value)
                        if par.fixed:
                            s += " (fixed)\n"
                        else:
                            s += " (free)\n"
            
        # warping
        s += "\nWarping:\n"
        for i in range(len(self.warping)):
            s += "\nPosition " + str(i) + ": " \
                 + self.warping[i].__class__.__name__ + "\n"
            
            # parameters
            params = self.warping[i].params
            if len(params) > 0:
                par_names = [x for x in params.keys()]
                for name in par_names:
                    par = params[name]
                    s += "\t" + str(name) + ": "
                    s += _np.array_str(par.value,
                                       max_line_width=200,
                                       precision=4)
                    if par.fixed:
                        s += " (fixed)\n"
                    else:
                        s += " (free)\n"
        return s
        
    def fix_kernel_parameter(self, position, parameter):
        """
        Fixes a parameter's current value, so that the genetic algorithm will
        not change it.

        Parameters
        ----------
        position : list or tuple of ints
            Positions of outer and inner levels in the list of kernel objects.
        parameter : str
            The parameter's name.

        The positions and names can be seen by printing the model object.
        """
        i, j = position
        if parameter not in self.kernels[i][j].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[i][j].transform.params[parameter].fix()
        
    def unfix_kernel_parameter(self, position, parameter):
        """
        Allows the genetic algorithm to change a parameter's value within
        its set limits.

        Parameters
        ----------
        position : list or tuple of ints
            Positions of outer and inner levels in the list of kernel objects.
        parameter : str
            The parameter's name.

        The positions and names can be seen by printing the model object.
        """
        i, j = position
        if parameter not in self.kernels[i][j].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[i][j].transform.params[parameter].unfix()
        
    def set_kernel_parameter(self, position, parameter, value, transf=False):
        """
        Allows the setting a parameter's value manually.

        Parameters
        ----------
        position : list or tuple of ints
            Positions of outer and inner levels in the list of kernel objects.
        parameter : str
            The parameter's name.
        value :
            The parameter's value.
        transf : bool
            Whether to change the parameter's original or transformed (log,
            softplus, etc.) value. The user should not change this.

        The positions and names can be seen by printing the model object.
        """
        i, j = position
        if parameter not in self.kernels[i][j].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[i][j].transform.params[parameter].set_value(
            value, transf)
        
    def set_kernel_parameter_limits(self, position, parameter,
                                    min_val, max_val):
        """
        Allows setting a parameter's limits manually.

        Parameters
        ----------
        position : list or tuple of ints
            Positions of outer and inner levels in the list of kernel objects.
        parameter : str
            The parameter's name.
        min_val, max_val :
            The parameter's limits.

        The positions and names can be seen by printing the model object.
        """
        i, j = position
        if parameter not in self.kernels[i][j].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[i][j].transform.params[parameter].set_limits(
            min_val, max_val)
        
    def fix_warping_parameter(self, position, parameter):
        """
        Fixes a parameter's current value, so that the genetic algorithm will
        not change it.

        Parameters
        ----------
        position : int
            Position in the list of warping objects.
        parameter : str
            The parameter's name.

        The positions and names can be seen by printing the model object.
        """
        if parameter not in self.warping[position].params.keys():
            raise Exception("warping does not have parameter " + parameter)
        self.warping[position].params[parameter].fix()
        
    def unfix_warping_parameter(self, position, parameter):
        """
        Allows the genetic algorithm to change a parameter's value within its
        set limits.

        Parameters
        ----------
        position : int
            Position in the list of warping objects.
        parameter : str
            The parameter's name.

        The positions and names can be seen by printing the model object.
        """
        if parameter not in self.warping[position].params.keys():
            raise Exception("warping does not have parameter " + parameter)
        self.warping[position].params[parameter].unfix()
        
    def set_warping_parameter(self, position, parameter, value, transf=False):
        """
        Allows the setting a parameter's value manually.

        Parameters
        ----------
        position : int
            Position in the list of warping objects.
        parameter : str
            The parameter's name.
        value :
            The parameter's value.
        transf : bool
            Whether to change the parameter's original or transformed (log,
            softplus, etc.) value. The user should not change this.

        The positions and names can be seen by printing the model object.
        """
        if parameter not in self.warping[position].params.keys():
            raise Exception("warping does not have parameter " + parameter)
        self.warping[position].params[parameter].set_value(value, transf)
        
    def set_warping_parameter_limits(self, position, parameter,
                                     min_val, max_val):
        """
        Allows setting a parameter's limits manually.

        Parameters
        ----------
        position : int
            Position in the list of warping objects.
        parameter : str
            The parameter's name.
        min_val, max_val :
            The parameter's limits.

        The positions and names can be seen by printing the model object.
        """
        if parameter not in self.warping[position].params.keys():
            raise Exception("warping does not have parameter " + parameter)
        self.warping[position].params[parameter].set_limits(min_val, max_val)
        
    def auto_set_kernel_parameter_limits(self, data):
        """
        Sets reasonable limits for the parameters based on the data's
        bounding box and such.
        """
        for k in self.kernels:
            for kk in k:
                kk.transform.set_limits(data)
        
    def init_tf_placeholder(self):
        """To be called within the context of a TensorFlow graph."""
        self.variance.init_tf_placeholder()
        for k in self.kernels:
            for kk in k:
                params = kk.transform.params
                if len(params) > 0:
                    par_names = [x for x in params.keys()]
                    for name in par_names:
                        params[name].init_tf_placeholder()
                    
    def feed_dict(self):
        """Gathers all parameters to feed a TensorFlow graph."""
        feed = {}
        feed.update(self.variance.tf_feed_entry)
        
        for k in self.kernels:
            for kk in k:
                params = kk.transform.params
                if len(params) > 0:
                    par_names = [x for x in params.keys()]
                    for name in par_names:
                        feed.update(params[name].tf_feed_entry)

        return feed
        
    def covariance_matrix(self, x, y):
        """Adds all point-point covariance matrices."""
        with _tf.name_scope("Cov_model_mat"):
            v = self.variance.tf_val
            cov_mat_outer = []
            # start from inner index - kernel products
            for k in self.kernels:
                cov_mat_inner = [kk.covariance_matrix(x, y) for kk in k]
                if len(cov_mat_inner) > 1:
                    cov_mat_inner = [_prod_n(cov_mat_inner)]
                cov_mat_outer.extend(cov_mat_inner)
            # weighting and adding up
            for i in range(len(cov_mat_outer)):
                cov_mat_outer[i] = cov_mat_outer[i] * v[i]
            cov_mat = _tf.add_n(cov_mat_outer)
        return cov_mat
    
    def covariance_matrix_d1(self, x, y, dir_y):
        """Adds all point-direction covariance matrices."""
        with _tf.name_scope("Cov_model_d1_mat"):
            v = self.variance.tf_val
            cov_mat_outer = []
            # start from inner index - kernel products
            for k in self.kernels:
                cov_mat_inner = [kk.covariance_matrix_d1(x, y, dir_y)
                                 for kk in k]
                if len(cov_mat_inner) > 1:
                    cov_mat_inner = [_prod_n(cov_mat_inner)]
                cov_mat_outer.extend(cov_mat_inner)
            # weighting and adding up
            for i in range(len(cov_mat_outer)):
                cov_mat_outer[i] = cov_mat_outer[i] * v[i]
            cov_mat = _tf.add_n(cov_mat_outer)
        return cov_mat
    
    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        """Adds all direction-direction covariance matrices."""
        with _tf.name_scope("Cov_model_d2_mat"):
            v = self.variance.tf_val
            cov_mat_outer = []
            # start from inner index - kernel products
            for k in self.kernels:
                cov_mat_inner = [kk.covariance_matrix_d2(x, y, dir_x, dir_y)
                                 for kk in k]
                if len(cov_mat_inner) > 1:
                    cov_mat_inner = [_prod_n(cov_mat_inner)]
                cov_mat_outer.extend(cov_mat_inner)
            # weighting and adding up
            for i in range(len(cov_mat_outer)):
                cov_mat_outer[i] = cov_mat_outer[i] * v[i]
            cov_mat = _tf.add_n(cov_mat_outer)
        return cov_mat
    
    def point_variance(self, x):
        """Adds all point variances."""
        with _tf.name_scope("Point_var_all"):
            v = self.variance.tf_val
            cov_mat_outer = []
            # start from inner index - kernel products
            for k in self.kernels:
                cov_mat_inner = [kk.point_variance(x) for kk in k]
                if len(cov_mat_inner) > 1:
                    cov_mat_inner = [_prod_n(cov_mat_inner)]
                cov_mat_outer.extend(cov_mat_inner)
            # weighting and adding up
            for i in range(len(cov_mat_outer)):
                cov_mat_outer[i] = cov_mat_outer[i] * v[i]
            cov_mat = _tf.add_n(cov_mat_outer)
        return cov_mat
    
    def warp_backward(self, y):
        """
        Applies its warping to the original variable, returning the transformed
        variable.
        """
        warping_rev = self.warping.copy()
        warping_rev.reverse()
        for W in warping_rev:
            y = W.backward(y)
        return y
    
    def warp_forward(self, y):
        """
        Applies its warping to the transformed variable, returning the original
        variable.
        """
        for W in self.warping:
            y = W.forward(y)
        return y
    
    def warp_derivative(self, y):
        """Calculates the derivative of the forward warping."""
        d = _np.ones(len(y))
        for W in self.warping:
            d *= W.derivative(y)
            y = W.forward(y)
        return d
    
    def warp_refresh(self, y):
        """Applies the given y to each warping."""
        for W in self.warping:
            W.refresh(y)
            y = W.forward(y)
            
    def params_dict(self, complete=False):
        """
        Writes all the model parameters in a dictionary, for easy reading later.

        The complete option is used to save the parameters to disk.
        """
        pd = {"param_type": [],
              "param_pos_outer": [],
              "param_pos_inner": [],
              "param_id": [],
              "param_val": [],
              "param_min": [],
              "param_max": [],
              "param_name": []}
        count = -1
        
        # variance and nugget
        if (not self.variance.fixed) | complete:
            count += 1
            val_length = len(self.variance.value_transf)
            pd["param_type"].extend(_np.repeat("variance", val_length).tolist())
            pd["param_pos_outer"].extend(_np.repeat(0, val_length).tolist())
            pd["param_pos_inner"].extend(_np.repeat(0, val_length).tolist())
            pd["param_id"].extend(_np.repeat(count, val_length).tolist())
            pd["param_val"].extend(self.variance.value_transf)
            pd["param_min"].extend(self.variance.min_transf)
            pd["param_max"].extend(self.variance.max_transf)
            pd["param_name"].extend(_np.repeat("variance", val_length).tolist())
        
        # kernels
        # the same transform can appear at multiple kernels, so we keep
        # track of their hashes in order not to duplicate their parameters
        hashes = []
        for i in range(len(self.kernels)):
            for ii in range(len(self.kernels[i])):
                params = self.kernels[i][ii].transform.params
                if len(params) > 0:
                    par_names = [x for x in params.keys()]
                    for j in range(len(params)):
                        if hash(params[par_names[j]]) not in hashes:
                            hashes.append(hash(params[par_names[j]]))
                            if (not params[par_names[j]].fixed) | complete:
                                count += 1
                                val_length = len(params[par_names[j]]
                                                 .value_transf)
                                pd["param_type"].extend(_np.repeat(
                                    "kernel_param", val_length).tolist())
                                pd["param_pos_outer"].extend(_np.repeat(
                                    i, val_length).tolist())
                                pd["param_pos_inner"].extend(_np.repeat(
                                    ii, val_length).tolist())
                                pd["param_id"].extend(_np.repeat(
                                    count, val_length).tolist())
                                pd["param_val"].extend(
                                    params[par_names[j]].value_transf)
                                pd["param_min"].extend(
                                    params[par_names[j]].min_transf)
                                pd["param_max"].extend(
                                    params[par_names[j]].max_transf)
                                pd["param_name"].extend(_np.repeat(
                                    par_names[j], val_length).tolist())
        
        # warping
        for i in range(len(self.warping)):
            params = self.warping[i].params
            if len(params) > 0:
                par_names = [x for x in params.keys()]
                for j in range(len(params)):
                    if (not params[par_names[j]].fixed) | complete:
                        count += 1
                        val_length = len(params[par_names[j]].value_transf)
                        pd["param_type"].extend(_np.repeat(
                            "warping_param", val_length).tolist())
                        pd["param_pos_outer"].extend(_np.repeat(
                            i, val_length).tolist())
                        pd["param_pos_inner"].extend(_np.repeat(
                            i, val_length).tolist())  # not used
                        pd["param_id"].extend(_np.repeat(
                            count, val_length).tolist())
                        pd["param_val"].extend(
                            params[par_names[j]].value_transf)
                        pd["param_min"].extend(params[par_names[j]].min_transf)
                        pd["param_max"].extend(params[par_names[j]].max_transf)
                        pd["param_name"].extend(_np.repeat(
                            par_names[j], val_length).tolist())

        return pd
    
    def update_params(self, params_dict):
        """
        Updates the model parameters, which will be fed to a TensorFlow graph
        later.
        """
        n_updates = _np.max(params_dict["param_id"])
        for i in range(n_updates + 1):
            pos = _np.where(_np.array(params_dict["param_id"]) == i)[0]
            pos2 = slice(pos[0], pos[-1] + 1)
            if params_dict["param_type"][pos[0]] == "variance":
                v = _np.array(params_dict["param_val"][pos2])
                self.variance.set_value(v, transf=True)
            elif params_dict["param_type"][pos[0]] == "kernel_param":
                kernel_pos = [params_dict["param_pos_outer"][pos[0]],
                              params_dict["param_pos_inner"][pos[0]]]
                self.set_kernel_parameter(kernel_pos,
                                          params_dict["param_name"][pos[0]],
                                          params_dict["param_val"][pos2], 
                                          transf=True)
            elif params_dict["param_type"][pos[0]] == "warping_param":
                warp_pos = params_dict["param_pos_outer"][pos[0]]
                self.set_warping_parameter(warp_pos,
                                           params_dict["param_name"][pos[0]],
                                           params_dict["param_val"][pos2], 
                                           transf=True)
            else:
                raise ValueError("invalid param_id")

    def save_state(self, file):
        """
        Saves a model's current parameters to disk.

        Parameters
        ----------
        file : str
            The file name.
        """
        d = self.params_dict(complete=True)
        with open(file, 'wb') as f:
            _pickle.dump(d, f)

    def load_state(self, file):
        """
        Restores a model's parameters from disk.

        Parameters
        ----------
        file : str
            The file name.

        The object that calls this method must have the same structure
        (kernels, warping, etc.) as the one that created the saved file,
        otherwise the system is likely to throw an error.
        """
        with open(file, 'rb') as f:
            d = _pickle.load(f)
        self.update_params(d)


class CovarianceModelSparse(CovarianceModelRegression):
    def __init__(self, kernels, warping, pseudo_inputs, jitter=1e-9):
        super().__init__(kernels, warping, jitter)

        coords = pseudo_inputs.coords.flatten(order="C")
        min_val = pseudo_inputs.coords.min(axis=0)
        min_val = _np.repeat(min_val, pseudo_inputs.coords.shape[0])
        max_val = pseudo_inputs.coords.max(axis=0)
        max_val = _np.repeat(max_val, pseudo_inputs.coords.shape[0])
        self.ps_coords = _gpr.Parameter(coords, min_val, max_val, fixed=True)

    def fix_pseudo_inputs(self):
        self.ps_coords.fix()

    def unfix_pseudo_inputs(self):
        self.ps_coords.unfix()

    def init_tf_placeholder(self):
        super().init_tf_placeholder()
        self.ps_coords.init_tf_placeholder()

    def feed_dict(self):
        feed = super().feed_dict()
        feed.update(self.ps_coords.tf_feed_entry)
        return feed

    def params_dict(self, complete=False):
        pd = super().params_dict()
        count = max(pd["param_id"])

        if (not self.ps_coords.fixed) | complete:
            count += 1
            val_length = len(self.ps_coords.value_transf)
            pd["param_type"].extend(_np.repeat("pseudo_inputs", val_length)
                                    .tolist())
            pd["param_pos_outer"].extend(_np.repeat(0, val_length).tolist())
            pd["param_pos_inner"].extend(_np.repeat(0, val_length).tolist())
            pd["param_id"].extend(_np.repeat(count, val_length).tolist())
            pd["param_val"].extend(self.ps_coords.value_transf)
            pd["param_min"].extend(self.ps_coords.min_transf)
            pd["param_max"].extend(self.ps_coords.max_transf)
            pd["param_name"].extend(_np.repeat("pseudo_inputs", val_length)
                                    .tolist())
        return pd

    def update_params(self, params_dict):
        """
        Updates the model parameters, which will be fed to a TensorFlow graph
        later.
        """
        n_updates = _np.max(params_dict["param_id"])
        for i in range(n_updates + 1):
            pos = _np.where(_np.array(params_dict["param_id"]) == i)[0]
            pos2 = slice(pos[0], pos[-1] + 1)
            if params_dict["param_type"][pos[0]] == "variance":
                v = _np.array(params_dict["param_val"][pos2])
                self.variance.set_value(v, transf=True)
            elif params_dict["param_type"][pos[0]] == "kernel_param":
                kernel_pos = [params_dict["param_pos_outer"][pos[0]],
                              params_dict["param_pos_inner"][pos[0]]]
                self.set_kernel_parameter(kernel_pos,
                                          params_dict["param_name"][pos[0]],
                                          params_dict["param_val"][pos2],
                                          transf=True)
            elif params_dict["param_type"][pos[0]] == "warping_param":
                warp_pos = params_dict["param_pos_outer"][pos[0]]
                self.set_warping_parameter(warp_pos,
                                           params_dict["param_name"][pos[0]],
                                           params_dict["param_val"][pos2],
                                           transf=True)
            elif params_dict["param_type"][pos[0]] == "pseudo_inputs":
                ps = _np.array(params_dict["param_val"][pos2])
                self.ps_coords.set_value(ps, transf=True)
            else:
                raise ValueError("invalid param_id")

