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
           "CompositionalParameter"]

import tensorflow as _tf


class Parameter(object):
    """
    Trainable model parameter. Can be a vector, matrix, or scalar.

    The `fixed` property applies to the array as a whole.
    """
    def __init__(self, value, min_val, max_val, fixed=False):
        self.value = _tf.Variable(value, dtype=_tf.float64)
        self.fixed = fixed

        self.max = _tf.constant(max_val, dtype=_tf.float64)
        self.min = _tf.constant(min_val, dtype=_tf.float64)

        if not self.max.shape == self.value.shape:
            raise ValueError(
                "Shape of max_val do not match shape of value: expected %s "
                "and found %s" % (str(self.value.shape), str(self.max.shape)))

        if not self.min.shape == self.value.shape:
            raise ValueError(
                "Shape of min_val do not match shape of value: expected %s "
                "and found %s" % (str(self.value.shape), str(self.min.shape)))

        self.value_transformed = _tf.Variable(value, dtype=_tf.float64)
        self.max_transformed = _tf.constant(max_val, dtype=_tf.float64)
        self.min_transformed = _tf.constant(min_val, dtype=_tf.float64)
        self.refresh()

    def fix(self):
        self.fixed = True

    def unfix(self):
        self.fixed = False

    def set_limits(self, min_val=None, max_val=None):
        if min_val is not None:
            min_val = _tf.constant(min_val, dtype=_tf.float64)
            if not min_val.shape == self.value.shape:
                raise ValueError(
                    "Shape of min_val do not match shape of value: expected %s "
                    "and found %s" % (
                        str(self.value.shape), str(min_val.shape)))

            self.min = min_val
            self.min_transformed = _tf.constant(min_val, dtype=_tf.float64)

        if max_val is not None:
            max_val = _tf.constant(max_val, dtype=_tf.float64)
            if not max_val.shape == self.value.shape:
                raise ValueError(
                    "Shape of max_val do not match shape of value: expected %s "
                    "and found %s" % (
                        str(self.value.shape), str(max_val.shape)))

            self.max = max_val
            self.max_transformed = _tf.constant(max_val, dtype=_tf.float64)

        self.refresh()

    def set_value(self, value, transformed=False):
        self.value.assign(value)
        self.refresh()

    def refresh(self):
        pass


class RealParameter(Parameter):
    """
    Trainable model parameter. Can be a vector, matrix, or scalar.
    
    The `fixed` property applies to the array as a whole.
    """
    def refresh(self):
        self.value.assign(_tf.minimum(self.value, self.max))
        self.value.assign(_tf.maximum(self.value, self.min))

        self.value_transformed.assign(self.value.value())
        self.value_transformed.assign(
            _tf.minimum(self.value_transformed, self.max_transformed))
        self.value_transformed.assign(
            _tf.maximum(self.value_transformed, self.min_transformed))


class PositiveParameter(Parameter):
    """Parameter in log scale"""

    def set_value(self, value, transformed=False):
        if transformed:
            self.value_transformed.assign(value)
            self.value.assign(_tf.math.exp(value))
        else:
            self.value.assign(value)
        self.refresh()

    def refresh(self):
        self.value.assign(_tf.minimum(self.value, self.max))
        self.value.assign(_tf.maximum(self.value, self.min))

        self.value_transformed.assign(_tf.math.log(self.value))
        self.max_transformed = _tf.math.log(self.max)
        self.min_transformed = _tf.math.log(self.min)


class CompositionalParameter(Parameter):
    """
    matrix vector parameter in logit coordinates
    """
    def __init__(self, value, fixed=False):
        value = _tf.constant(value, dtype=_tf.float64)
        min_val = _tf.zeros_like(value)
        max_val = _tf.ones_like(value)
        super().__init__(value, min_val, max_val, fixed)
        self.min_transformed = _tf.ones_like(value) * -10
        self.max_transformed = _tf.ones_like(value) * 10
        
    def set_limits(self, min_val=None, max_val=None):
        pass
    
    def set_value(self, value, transformed=False):
        if transformed:
            # self.value_transformed.assign(value)

            value = value - _tf.reduce_mean(value)  # to avoid overflow
            value = _tf.math.exp(value)
            value = value / _tf.reduce_sum(value)
            self.value.assign(value)
        else:
            self.value.assign(value)
        self.refresh()

    def refresh(self):
        self.value.assign(self.value / _tf.reduce_sum(self.value))

        logit = _tf.math.log(self.value)
        self.value_transformed.assign(logit - _tf.reduce_mean(logit))
