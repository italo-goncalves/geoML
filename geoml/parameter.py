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

__all__ = ["RealParameter",
           "PositiveParameter",
           "CompositionalParameter",
           "CircularParameter"]

import tensorflow as _tf
import numpy as _np

# import geoml.tftools as _tftools


class RealParameter(object):
    """
    Trainable model parameter. Can be a vector, matrix, or scalar.

    The `fixed` property applies to the array as a whole.
    """
    def __init__(self, value, min_val, max_val, fixed=False,
                 name="Parameter"):
        self.name = name
        self.fixed = fixed

        value = _np.array(value)
        min_val = _np.array(min_val)
        max_val = _np.array(max_val)

        if not max_val.shape == value.shape:
            raise ValueError(
                "Shape of max_val do not match shape of value: expected %s "
                "and found %s" % (str(value.shape),
                                  str(max_val.shape)))

        if not min_val.shape == value.shape:
            raise ValueError(
                "Shape of min_val do not match shape of value: expected %s "
                "and found %s" % (str(value.shape),
                                  str(min_val.shape)))

        self.shape = value.shape

        self.variable = _tf.Variable(self._transform(value),
                                     dtype=_tf.float64, name=name)

        self.max_transformed = _tf.Variable(
            self._transform(max_val), dtype=_tf.float64)
        self.min_transformed = _tf.Variable(
            self._transform(min_val), dtype=_tf.float64)

        self.refresh()

    def _transform(self, x):
        return x

    def _back_transform(self, x):
        return x

    def fix(self):
        self.fixed = True

    def unfix(self):
        self.fixed = False

    def set_limits(self, min_val=None, max_val=None):
        if min_val is not None:
            self.min_transformed.assign(self._transform(min_val))

        if max_val is not None:
            self.max_transformed.assign(self._transform(max_val))
        self.refresh()

    def set_value(self, value, transformed=False):
        if transformed:
            self.variable.assign(value)
        else:
            self.variable.assign(self._transform(value))
        self.refresh()

    def get_value(self):
        return self._back_transform(self.variable)

    def refresh(self):
        value = _tf.maximum(self.min_transformed,
                            _tf.minimum(self.max_transformed, self.variable))
        self.variable.assign(value)

    def randomize(self):
        val = (self.variable - self.min_transformed) \
              / (self.max_transformed - self.min_transformed)
        val = val + _np.random.uniform(size=self.shape, low=-0.05, high=0.05)
        val = _tf.maximum(0, _tf.minimum(1, val))
        val = val * (self.max_transformed - self.min_transformed) \
              + self.min_transformed
        self.variable.assign(val)


class PositiveParameter(RealParameter):
    """Parameter in log scale"""

    def _transform(self, x):
        return _tf.math.log(_tf.cast(x, _tf.float64))

    def _back_transform(self, x):
        return _tf.math.exp(x)


class CompositionalParameter(RealParameter):
    """
    A vector parameter in logit coordinates
    """
    def __init__(self, value, fixed=False, name="Parameter"):
        super().__init__(value, value, value, fixed, name=name)
        self.min_transformed.assign(-10 * _tf.ones_like(self.variable))
        self.max_transformed.assign(10 * _tf.ones_like(self.variable))
        self.variable.assign(self._transform(value))

    def _transform(self, x):
        x_tr = _tf.math.log(_tf.cast(x, _tf.float64))
        return x_tr - _tf.reduce_mean(x_tr)

    def _back_transform(self, x):
        return _tf.nn.softmax(x)


class CircularParameter(RealParameter):
    def refresh(self):
        amp = self.max_transformed - self.min_transformed
        n_laps = _tf.floor((self.variable - self.min_transformed) / amp)
        value = self.variable - n_laps * amp
        self.variable.assign(value)


class UnitColumnNormParameter(RealParameter):
    def __init__(self, value, min_val, max_val, fixed=False, name="Parameter"):
        value = _np.array(value)
        if len(value.shape) != 2:
            raise ValueError("value must be rank 2")
        super().__init__(value, min_val, max_val, fixed, name)

    def refresh(self):
        value = self.get_value()
        normalized = value / (_tf.math.reduce_euclidean_norm(
            value, axis=0, keepdims=True) + 1e-6)
        self.variable.assign(normalized)


class UnitColumnSumParameter(RealParameter):
    def __init__(self, value, fixed=False, name="Parameter"):
        value = _np.array(value)
        if len(value.shape) != 2:
            raise ValueError("value must be rank 2")
        super().__init__(value, value, value, fixed, name)
        self.min_transformed.assign(-100 * _tf.ones_like(self.variable))
        self.max_transformed.assign(100 * _tf.ones_like(self.variable))
        self.variable.assign(self._transform(value))

    def _transform(self, x):
        x_tr = _tf.math.log(_tf.cast(x, _tf.float64))
        return x_tr - _tf.reduce_mean(x_tr, axis=0, keepdims=True)

    def _back_transform(self, x):
        return _tf.nn.softmax(x, axis=0)


class NaturalParameter(RealParameter):
    def __init__(self, dim, n_latent):
        mat = _np.tile(_np.eye(dim)[None, :, :], [n_latent, 1, 1])
        vec = _np.zeros([n_latent, dim])
        start = _np.concatenate(
            [_np.reshape(mat, [-1]), _np.reshape(vec, [-1])],
            axis=0)

        self.dim = dim
        self.n_latent = n_latent
        self.theta = _tf.Variable(_tf.constant(-0.5 * start, _tf.float64))
        super().__init__(start, start - 20, start + 20)

    def get_value(self):
        # self.variable = [eta_2, eta_1]
        n_cov = self.n_latent * self.dim**2
        eta_2 = _tf.reshape(self.variable[:n_cov],
                            [self.n_latent, self.dim, self.dim])
        eta_1 = _tf.reshape(self.variable[n_cov:],
                            [self.n_latent, self.dim, 1])
        mat_s = eta_2 - _tf.matmul(eta_1, eta_1, False, True)
        return eta_1, mat_s

    def refresh(self):
        n_cov = self.n_latent * self.dim ** 2
        theta_2 = _tf.reshape(self.theta[:n_cov],
                              [self.n_latent, self.dim, self.dim])
        theta_1 = _tf.reshape(self.theta[n_cov:],
                              [self.n_latent, self.dim, 1])

        eye = _tf.eye(self.dim, dtype=_tf.float64, batch_shape=[self.n_latent])
        theta_2 = 0.5 * (theta_2 + _tf.transpose(theta_2, [0, 2, 1]))
        theta_2_chol = _tf.linalg.cholesky(-2 * theta_2 + eye * 0.01)

        # eta_1 = -0.5 * _tf.linalg.solve(theta_2, theta_1)
        # eta_2 = -0.5 * _tf.linalg.solve(
        #     theta_2, # + eye * 1e-6,
        #     _tf.matmul(theta_1, eta_1, False, True) + eye)
        eta_1 = _tf.linalg.cholesky_solve(theta_2_chol, theta_1)
        eta_2 = _tf.linalg.cholesky_solve(
            theta_2_chol,
            _tf.matmul(theta_1, eta_1, False, True) + eye)
        eta = _tf.concat(
            [_tf.reshape(eta_2, [-1]), _tf.reshape(eta_1, [-1])],
            axis=0
        )
        self.variable.assign(eta)
