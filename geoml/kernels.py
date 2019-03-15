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
import geoml.parameter as _gpr
import geoml.transform as _gt

from geoml.tftools import pairwise_dist as _pairwise_dist


class _Kernel(object):
    """Abstract kernel class"""
    def __init__(self, transform=_gt.Identity()):
        self.transform = transform
        
    def covariance_matrix(self, x, y):
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


class GaussianKernel(_Kernel):
    """Gaussian kernel"""
    def covariance_matrix(self, x, y):
        with _tf.name_scope("Gaussian_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            d2 = _tf.pow(_pairwise_dist(x, y), 2)
            k = _tf.Variable(_tf.exp(-3 * d2), dtype=_tf.float64,
                             validate_shape=False)
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
    def covariance_matrix(self, x, y):
        with _tf.name_scope("Spherical_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            d = _pairwise_dist(x, y)
            k = 1 - 1.5 * d + 0.5 * _tf.pow(d, 3)
            k = _tf.Variable(k, dtype=_tf.float64, validate_shape=False)
            indices = _tf.where(_tf.greater(d, 1))
            updates = _tf.zeros(_tf.shape(indices)[0], dtype=k.dtype)
            k = _tf.scatter_nd_update(k, indices, updates)
        return k


class ExponentialKernel(_Kernel):
    """Exponential kernel"""
    def covariance_matrix(self, x, y):
        with _tf.name_scope("Exponential_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            d = _pairwise_dist(x, y)
            k = _tf.Variable(_tf.exp(-3 * d), dtype=_tf.float64,
                             validate_shape=False)
        return k


class CubicKernel(_Kernel):
    """Cubic kernel"""
    def covariance_matrix(self, x, y):
        with _tf.name_scope("Cubic_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            d = _pairwise_dist(x, y)
            k = 1 - 7 * _tf.pow(d, 2) + 35 / 4 * _tf.pow(d, 3) \
                - 7 / 2 * _tf.pow(d, 5) + 3 / 4 * _tf.pow(d, 7)
            k = _tf.Variable(k, dtype=_tf.float64, validate_shape=False)
            indices = _tf.where(_tf.greater(d, 1))
            updates = _tf.zeros(_tf.shape(indices)[0], dtype=k.dtype)
            k = _tf.scatter_nd_update(k, indices, updates)
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
            k = _tf.Variable(k, dtype=_tf.float64, validate_shape=False)
            indices = _tf.where(_tf.greater(d, 1))
            updates = _tf.zeros(_tf.shape(indices)[0], dtype=k.dtype)
            k = _tf.scatter_nd_update(k, indices, updates)
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
            k1 = _tf.Variable(k1, dtype=_tf.float64, validate_shape=False)
            k2 = _tf.Variable(k2, dtype=_tf.float64, validate_shape=False)
            indices = _tf.where(_tf.greater(d, 1))
            updates = _tf.zeros(_tf.shape(indices)[0], dtype=k1.dtype)
            k1 = _tf.scatter_nd_update(k1, indices, updates)
            k2 = _tf.scatter_nd_update(k2, indices, updates)
            k = k1*dir_prod + k2*prod
        return k


class ConstantKernel(_Kernel):
    """Constant kernel"""
    def covariance_matrix(self, x, y):
        with _tf.name_scope("Constant_cov"):
            k = _tf.ones([_tf.shape(x)[0], _tf.shape(y)[0]], dtype=_tf.float64)
        return k


class LinearKernel(_Kernel):
    """Linear kernel"""       
    def covariance_matrix(self, x, y):
        with _tf.name_scope("Linear_cov"):
            x = self.transform.backward(x)
            y = self.transform.backward(y)
            k = _tf.matmul(x, y, False, True)
        return k
    
    def point_variance(self, x):
        with _tf.name_scope("Linear_point_var"):
            x = self.transform.backward(x)
            v = _tf.reduce_sum(_tf.pow(x, 2), 1)
        return v


class CosineKernel(_Kernel):
    """Cosine kernel"""
    def covariance_matrix(self, x, y):
        with _tf.name_scope("Cosine_cov"):
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
            mat = _tf.diag(ones)
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
        A list containing the desired kernels.
    warping : list
        A list containing the Warpings for the dependent variable.
    nugget : Nugget
        A nugget object.
    variance :
        A parameter object with the relative variance of each object.

    In this covariance model, it is assumed that all variances add to 1.
    The warping objects are used to scale and transform the data accordingly.
    """
    def __init__(self, kernels, warping):
        """
        Initializer for CovarianceModelRegression object.

        Parameters
        ----------
        kernels :  list
            A list containing the desired kernels.
        warping : list
            A list containing the Warpings for the dependent variable.
        """
        is_kernel = [isinstance(k, _Kernel) for k in kernels]
        if not all(is_kernel):
            raise Exception("kernels must be a list of Kernel objects")
        self.kernels = kernels  
        self.warping = warping
        self.nugget = Nugget()
        # initializing with 10% nugget
        v = _np.concatenate([_np.repeat(0.9 / len(kernels), len(kernels)),
                             _np.array([0.1])])
        self.variance = _gpr.CompositionalParameter(v)
        
    def __str__(self):
        s = "A " + self.__class__.__name__ + " object\n\n"
        s += "Variance is"
        if self.variance.fixed:
            s += " fixed\n"
        else:
            s += " free\n"
        
        # nugget
        s += "\n" + self.nugget.__class__.__name__ + ": " \
             + str(self.variance.value[-1]) + "\n"
        
        # kernels
        for i in range(len(self.kernels)):
            s += "\nPosition " + str(i) + ": " \
                 + self.kernels[i].__class__.__name__ + "\n"
            
            # variance
            s += "\tVariance: " + str(self.variance.value[i]) + "\n"
            
            # parameters
            params = self.kernels[i].transform.params
            if len(params) > 0:
                s += "\tParameters:\n"
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
                s += "\tParameters:\n"
                par_names = [x for x in params.keys()]
                for name in par_names:
                    par = params[name]
                    s += "\t\t" + str(name) + ": " 
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
        position : int
            Position in the list of warping objects.
        parameter : str
            The parameter's name.

        The positions and names can be seen by printing the model object.
        """
        if parameter not in self.kernels[position].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[position].transform.params[parameter].fix()
        
    def unfix_kernel_parameter(self, position, parameter):
        """
        Allows the genetic algorithm to change a parameter's value within
        its set limits.

        Parameters
        ----------
        position : int
            Position in the list of warping objects.
        parameter : str
            The parameter's name.

        The positions and names can be seen by printing the model object.
        """
        if parameter not in self.kernels[position].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[position].transform.params[parameter].unfix()
        
    def set_kernel_parameter(self, position, parameter, value, transf=False):
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
        if parameter not in self.kernels[position].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[position].transform.params[parameter].set_value(
            value, transf)
        
    def set_kernel_parameter_limits(self, position, parameter,
                                    min_val, max_val):
        """
        Allows setting a parameter's limits manually.

        Parameters
        ----------
        position : int
            Position in the list of kernel objects.
        parameter : str
            The parameter's name.
        min_val, max_val :
            The parameter's limits.

        The positions and names can be seen by printing the model object.
        """
        if parameter not in self.kernels[position].transform.params.keys():
            raise Exception("kernel does not have parameter " + parameter)
        self.kernels[position].transform.params[parameter].set_limits(
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
            k.transform.set_limits(data)
        
    def init_tf_placeholder(self):
        """To be called within the context of a TensorFlow graph."""
        self.variance.init_tf_placeholder()
        for k in self.kernels:
            params = k.transform.params
            if len(params) > 0:
                par_names = [x for x in params.keys()]
                for name in par_names:
                    params[name].init_tf_placeholder()
                    
    def feed_dict(self):
        """Gathers all parameters to feed a TensorFlow graph."""
        feed = {}
        feed.update(self.variance.tf_feed_entry)
        
        for k in self.kernels:
            params = k.transform.params
            if len(params) > 0:
                par_names = [x for x in params.keys()]
                for name in par_names:
                    feed.update(params[name].tf_feed_entry)

        return feed
        
    def covariance_matrix(self, x, y):
        """Adds all point-point covariance matrices."""
        with _tf.name_scope("Cov_model_mat"):
            v = self.variance.tf_val
            cov_mat = [k.covariance_matrix(x, y) for k in self.kernels]
            for i in range(len(cov_mat)):
                cov_mat[i] = cov_mat[i] * v[i]
            cov_mat = _tf.add_n(cov_mat)
        return cov_mat
    
    def covariance_matrix_d1(self, x, y, dir_y):
        """Adds all point-direction covariance matrices."""
        with _tf.name_scope("Cov_model_d1_mat"):
            v = self.variance.tf_val
            cov_mat = [k.covariance_matrix_d1(x, y, dir_y)
                       for k in self.kernels]
            for i in range(len(cov_mat)):
                cov_mat[i] = cov_mat[i] * v[i]
            cov_mat = _tf.add_n(cov_mat)
        return cov_mat
    
    def covariance_matrix_d2(self, x, y, dir_x, dir_y):
        """Adds all direction-direction covariance matrices."""
        with _tf.name_scope("Cov_model_d2_mat"):
            v = self.variance.tf_val
            cov_mat = [k.covariance_matrix_d2(x, y, dir_x, dir_y)
                       for k in self.kernels]
            for i in range(len(cov_mat)):
                cov_mat[i] = cov_mat[i] * v[i]
            cov_mat = _tf.add_n(cov_mat)
        return cov_mat
    
    def point_variance(self, x):
        """Adds all point variances."""
        with _tf.name_scope("Point_var_all"):
            v = self.variance.tf_val
            cov_mat = [k.point_variance(x) for k in self.kernels]
            for i in range(len(cov_mat)):
                cov_mat[i] = cov_mat[i] * v[i]
            cov_mat = _tf.add_n(cov_mat)
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
            
    def params_dict(self):
        """
        Writes all the model parameters in a dictionary, for easy reading later.
        """
        pd = {"param_type": [],
              "param_pos": [],
              "param_id": [],
              "param_val": [],
              "param_min": [],
              "param_max": [],
              "param_name": []}
        count = -1
        
        # nugget
        if not self.variance.fixed:
            count += 1
            val_length = len(self.variance.value_transf)
            pd["param_type"].extend(_np.repeat("variance", val_length).tolist())
            pd["param_pos"].extend(_np.repeat(0, val_length).tolist())
            pd["param_id"].extend(_np.repeat(count, val_length).tolist())
            pd["param_val"].extend(self.variance.value_transf)
            pd["param_min"].extend(self.variance.min_transf)
            pd["param_max"].extend(self.variance.max_transf)
            pd["param_name"].extend(_np.repeat("variance", val_length).tolist())
        
        # kernels
        for i in range(len(self.kernels)):
            params = self.kernels[i].transform.params
            if len(params) > 0:
                par_names = [x for x in params.keys()]
                for j in range(len(params)):
                    if not params[par_names[j]].fixed:
                        count += 1
                        val_length = len(params[par_names[j]].value)
                        pd["param_type"].extend(_np.repeat(
                            "kernel_param", val_length).tolist())
                        pd["param_pos"].extend(_np.repeat(
                            i, val_length).tolist())
                        pd["param_id"].extend(_np.repeat(
                            count, val_length).tolist())
                        pd["param_val"].extend(
                            params[par_names[j]].value_transf)
                        pd["param_min"].extend(params[par_names[j]].min_transf)
                        pd["param_max"].extend(params[par_names[j]].max_transf)
                        pd["param_name"].extend([par_names[j]])
        
        # warping
        for i in range(len(self.warping)):
            params = self.warping[i].params
            if len(params) > 0:
                par_names = [x for x in params.keys()]
                for j in range(len(params)):
                    if not params[par_names[j]].fixed:
                        count += 1
                        val_length = len(params[par_names[j]].value)
                        pd["param_type"].extend(_np.repeat(
                            "warping_param", val_length).tolist())
                        pd["param_pos"].extend(_np.repeat(
                            i, val_length).tolist())
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
                self.set_kernel_parameter(params_dict["param_pos"][pos[0]],
                                          params_dict["param_name"][pos[0]],
                                          params_dict["param_val"][pos2], 
                                          transf=True)
            elif params_dict["param_type"][pos[0]] == "warping_param":
                self.set_warping_parameter(params_dict["param_pos"][pos[0]],
                                           params_dict["param_name"][pos[0]],
                                           params_dict["param_val"][pos2], 
                                           transf=True)
            else:
                raise ValueError("invalid param_id")
