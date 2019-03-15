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

import numpy as _np
import tensorflow as _tf


class Parameter(object):
    """
    Trainable model parameter. Can be a vector or scalar.
    
    The fixed property applies to the array as a whole.
    """
    def __init__(self, value, min_val, max_val, fixed=False):
        self.value = _np.array(value, ndmin=1)
        self.fixed = fixed
        self.max = _np.array(max_val, ndmin=1)
        self.min = _np.array(min_val, ndmin=1)
        self.value_transf = _np.array(value, ndmin=1)
        self.max_transf = _np.array(max_val, ndmin=1)
        self.min_transf = _np.array(min_val, ndmin=1)
        self.tf_val = None
        self.tf_feed_entry = None
        self.refresh()
        
    def fix(self):
        self.fixed = True
    
    def unfix(self):
        self.fixed = False
        
    def set_limits(self, min_val=None, max_val=None):
        if min_val is not None:
            self.min = _np.array(min_val, ndmin=1)
            self.min_transf = _np.array(min_val, ndmin=1)
        if max_val is not None:
            self.max = _np.array(max_val, ndmin=1)
            self.max_transf = _np.array(max_val, ndmin=1)
        self.refresh()
            
    def set_value(self, value, transf=False):
        self.value = _np.array(value, ndmin=1)
        self.value_transf = _np.array(value, ndmin=1)
        self.refresh()
        
    def refresh(self):
        if (self.value > self.max).any():
            self.value = self.max
        if (self.value_transf > self.max_transf).any(): 
            self.value_transf = self.max_transf
        if (self.value < self.min).any():
            self.value = self.min
        if (self.value_transf < self.min_transf).any(): 
            self.value_transf = self.min_transf
        if self.tf_val is not None:
            self.tf_feed_entry = {self.tf_val: self.value}
            
    def init_tf_placeholder(self):
        self.tf_val = _tf.placeholder(_tf.float64, shape=self.value.shape)
        self.tf_feed_entry = {self.tf_val: self.value}


class PositiveParameter(Parameter):
    """Parameter in log scale"""
    def __init__(self, value, min_val, max_val, fixed=False):
        super().__init__(value, min_val, max_val, fixed)
        self.value_transf = _np.array(_np.log(value), ndmin=1)
        self.max_transf = _np.array(_np.log(max_val), ndmin=1)
        self.min_transf = _np.array(_np.log(min_val), ndmin=1)
        self.refresh()
        
    def set_limits(self, min_val=None, max_val=None):
        if min_val is not None:
            self.min = _np.array(min_val, ndmin=1)
            self.min_transf = _np.array(_np.log(min_val), ndmin=1)
        if max_val is not None:
            self.max = _np.array(max_val, ndmin=1)
            self.max_transf = _np.array(_np.log(max_val), ndmin=1)
        self.refresh()
            
    def set_value(self, value, transf=False):
        if transf:
            self.value_transf = _np.array(value, ndmin=1)
            self.value = _np.array(_np.exp(value), ndmin=1)
        else:
            self.value = _np.array(value, ndmin=1)
            self.value_transf = _np.array(_np.log(value), ndmin=1)
        self.refresh()


class CompositionalParameter(Parameter):
    """
    A vector parameter in clr coordinates
    """
    def __init__(self, value, fixed=False):
        s = value.size
        self.value = value
        self.fixed = fixed
        self.max = _np.repeat(1, s)
        self.min = _np.repeat(0, s)
        self.value_transf = _np.log(value) - _np.log(value).mean()
        self.value_transf = self.value_transf[0:(s-1)]
        self.max_transf = _np.repeat(5, s - 1)
        self.min_transf = _np.repeat(-5, s - 1)
        self.tf_val = None
        self.tf_feed_entry = None
        
    def set_limits(self, min_val=None, max_val=None):
        pass
    
    def set_value(self, value, transf=False):
        if transf:
            v = _np.concatenate([value, [-value.sum()]])
            self.value = _np.exp(v) / sum(_np.exp(v))
            self.value_transf = value
        else:
            s = value.size
            self.value = value
            self.value_transf = _np.log(value) - _np.log(value).mean()
            self.value_transf = self.value_transf[0:(s-1)]
        self.refresh()
