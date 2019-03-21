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

__all__ = ["Identity", "Isotropic", "Anisotropy2D", "Anisotropy3D",
           "Projection2DTo1D"]

import numpy as _np
import tensorflow as _tf
import geoml.parameter as _gpr


class _Transform(object):
    """An abstract class for variable transformations"""
    def __init__(self):
        self.params = {}
        
    def refresh(self):
        pass
    
    def set_limits(self, data):
        pass
    
    def forward(self, x):
        pass
    
    def backward(self, x):
        pass


class Identity(_Transform):
    """The identity transformation"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        with _tf.name_scope("Identity_forward"):
            return _tf.constant(x, dtype=_tf.float64)
    
    def backward(self, x):
        with _tf.name_scope("Identity_backward"):
            return _tf.constant(x, dtype=_tf.float64)


class Isotropic(_Transform):
    """Isotropic range"""
    def __init__(self, r):
        """
        Initializer for Isotropic.

        Parameters
        ----------
        r : double
            The range. Must be positive.
        """
        super().__init__()
        self.params = {"range": _gpr.PositiveParameter(r, 0.1, 10000)}
        
    def forward(self, x):
        with _tf.name_scope("Isotropic_forward"):
            r = self.params["range"].tf_val
            return x * r
    
    def backward(self, x):
        with _tf.name_scope("Isotropic_backward"):
            r = self.params["range"].tf_val
            return x / r
    
    def set_limits(self, data):
        self.params["range"].set_limits(
                min_val=data.diagonal / 100, max_val=data.diagonal * 10)


class Anisotropy2D(_Transform):
    """Anisotropy in two dimensions"""
    def __init__(self, maxrange, minrange_fct=1, azimuth=0):
        """
        Builds an anisotropy matrix, to be multiplied by a coordinate matrix
        from the right.
        
        Parameters
        ----------
        azimuth : double
            Defined clockwise from north, and is aligned with maxrange.
        maxrange : double
            The maximum range. Must be positive.
        minrange_fct : double
            A multiple of maxrange, contained in the [0,1) interval.
        """
        super().__init__()
        self.params = {"maxrange": _gpr.PositiveParameter(maxrange, 0.1, 10000),
                       "minrange_fct": _gpr.Parameter(minrange_fct, 0.05, 1),
                       "azimuth": _gpr.Parameter(azimuth, 0, 180)}
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy2D_refresh"):
            azimuth = self.params["azimuth"].tf_val
            maxrange = self.params["maxrange"].tf_val
            minrange = _tf.multiply(
                self.params["minrange_fct"].tf_val, maxrange)
            # conversion to radians
            azimuth = azimuth * (_np.pi / 180)
            # conversion to mathematical coordinates
            azimuth = _np.pi / 2 - azimuth
            # rotation matrix
            rot = _tf.concat([_tf.cos(azimuth), -_tf.sin(azimuth),
                              _tf.sin(azimuth), _tf.cos(azimuth)], -1)
            rot = _tf.reshape(rot, [2, 2])
            # scaling matrix
            sc = _tf.linalg.diag(_tf.concat([maxrange, minrange], -1))
            # anisotropy matrix
            self._anis = _tf.transpose(_tf.matmul(rot, sc))
            self._anis_inv = _tf.linalg.inv(self._anis)
    
    def forward(self, x):
        with _tf.name_scope("Anisotropy2D_forward"):
            self.refresh()
            return _tf.matmul(x, self._anis)
    
    def backward(self, x):
        with _tf.name_scope("Anisotropy2D_backward"):
            self.refresh()
            return _tf.matmul(x, self._anis_inv)
    
    def set_limits(self, data):
        self.params["maxrange"].set_limits(
                min_val=data.diagonal / 100, max_val=data.diagonal * 10)


class Anisotropy3D(_Transform):
    """Anisotropy in two dimensions"""
    def __init__(self, maxrange, midrange_fct=1, minrange_fct=1,
                 azimuth=0, dip=0, rake=0):
        """
        Builds an anisotropy matrix, to be multiplied by a coordinate matrix
        from the right.
        
        Parameters
        ----------
        maxrange : double
            The maximum range. Must be positive.
        midrange_fct : double
            A multiple of maxrange, contained in the [0,1) interval.
        minrange_fct : double
            A multiple of midrange, contained in the [0,1) interval.
        azimuth : double
            Defined clockwise from north, and is aligned with maxrange.
        dip : double
            Dip angle, from 0 to 90 degrees.
        rake : double
            Rake angle, from -90 to 90 degrees.
        """
        super().__init__()
        self.params = {"maxrange": _gpr.PositiveParameter(maxrange, 0.1, 10000),
                       "midrange_fct": _gpr.Parameter(midrange_fct, 0.05, 1),
                       "minrange_fct": _gpr.Parameter(minrange_fct, 0.05, 1),
                       "azimuth": _gpr.Parameter(azimuth, 0, 180),
                       "dip": _gpr.Parameter(dip, 0, 90),
                       "rake": _gpr.Parameter(rake, -90, 90)}
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy3D_refresh"):
            azimuth = self.params["azimuth"].tf_val
            dip = self.params["dip"].tf_val
            rake = self.params["rake"].tf_val
            maxrange = self.params["maxrange"].tf_val
            midrange = _tf.multiply(
                self.params["midrange_fct"].tf_val, maxrange)
            minrange = _tf.multiply(
                self.params["minrange_fct"].tf_val, midrange)
            # conversion to radians
            azimuth = azimuth * (_np.pi / 180)
            dip = dip * (_np.pi / 180)
            rake = rake * (_np.pi / 180)
            # conversion to mathematical coordinates
            dip = - dip
            rng = _tf.linalg.diag(_tf.concat(
                [midrange, maxrange, minrange], -1))
            # rotation matrix
            rx = _tf.concat([_tf.cos(rake), [0], _tf.sin(rake),
                             [0], [1], [0],
                             -_tf.sin(rake), [0], _tf.cos(rake)], -1)
            rx = _tf.reshape(rx, [3, 3])
            ry = _tf.concat([[1], [0], [0],
                             [0], _tf.cos(dip), -_tf.sin(dip),
                             [0], _tf.sin(dip), _tf.cos(dip)], -1)
            ry = _tf.reshape(ry, [3, 3])
            rz = _tf.concat([_tf.cos(azimuth), _tf.sin(azimuth), [0],
                             -_tf.sin(azimuth), _tf.cos(azimuth), [0],
                             [0], [0], [1]], -1)
            rz = _tf.reshape(rz, [3, 3])
            # anisotropy matrix
            anis = _tf.matmul(_tf.matmul(_tf.matmul(rz, ry), rx), rng)
            self._anis = _tf.transpose(anis)
            self._anis_inv = _tf.linalg.inv(self._anis)
    
    def forward(self, x):
        with _tf.name_scope("Anisotropy3D_forward"):
            self.refresh()
            return _tf.matmul(x, self._anis)
    
    def backward(self, x):
        with _tf.name_scope("Anisotropy3D_backward"):
            self.refresh()
            return _tf.matmul(x, self._anis_inv)
    
    def set_limits(self, data):
        self.params["maxrange"].set_limits(
                min_val=data.diagonal / 100, max_val=data.diagonal * 10)


class Projection2DTo1D(_Transform):
    """
    Projection of 2D data to a line.
    """
    def __init__(self, azimuth, rng):
        """
        Initializer for Projection2DTo1D.

        Parameters
        ----------
        azimuth : double
            Azimuth of the line to project on, between 0 and 180 degrees.
        rng : double
            Range to scale the data after projection. Must be positive.
        """
        super().__init__()
        self.params = {"azimuth": _gpr.Parameter(azimuth, 0, 180, fixed=True),
                       "range": _gpr.PositiveParameter(rng, 0.1, 10000)}
        self.vector = None

    def refresh(self):
        with _tf.name_scope("Projection2DTo1D_refresh"):
            az = self.params["azimuth"].tf_val
            az = az * (_np.pi / 180)
            self.vector = _tf.concat([_tf.sin(az), _tf.cos(az)], axis=0)
            self.vector = _tf.reshape(self.vector, [2, 1])

    def backward(self, x):
        with _tf.name_scope("Projection2DTo1D_backward"):
            self.refresh()
            r = self.params["range"].tf_val
            x = _tf.matmul(x, self.vector)
            x = x / r
        return x

    def forward(self, x):
        with _tf.name_scope("Projection2DTo1D_forward"):
            self.refresh()
            r = self.params["range"].tf_val
            x = x * r
            az = self.params["azimuth"].tf_val
            az = az * (_np.pi / 180)
            coords = _tf.concat([x * _tf.sin(az), x * _tf.cos(az)], axis=1)
        return coords

    def set_limits(self, data):
        self.params["range"].set_limits(
                min_val=data.diagonal / 100, max_val=data.diagonal * 10)
