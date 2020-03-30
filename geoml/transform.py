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

__all__ = ["Identity",
           "Isotropic",
           "Anisotropy2D",
           "Anisotropy3D",
           "ProjectionTo1D",
           "AnisotropyARD",
           "ChainedTransform",
           "SelectVariables"]

import geoml.parameter as _gpr

import numpy as _np
import tensorflow as _tf


class _Transform(object):
    """An abstract class for variable transformations"""
    def __init__(self):
        self.parameters = {}
        self._all_parameters = [pr for pr in self.parameters.values()]

    @property
    def all_parameters(self):
        return self._all_parameters
        
    def refresh(self):
        pass
    
    def set_limits(self, data):
        pass
    
    def __call__(self, x):
        pass

    def pretty_print(self, depth=0):
        s = "  " * depth + self.__class__.__name__ + "\n"
        depth += 1
        for name, parameter in self.parameters.items():
            s += "  " * depth + name + ": " + str(parameter.value.numpy())
            if parameter.fixed:
                s += " (fixed)"
            s += "\n"
        return s

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


class Identity(_Transform):
    """The identity transformation"""
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        with _tf.name_scope("Identity_transform"):
            return x


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
        self.parameters = {"range": _gpr.PositiveParameter(r, 0.1, 10000)}
        self._all_parameters = [pr for pr in self.parameters.values()]
    
    def __call__(self, x):
        with _tf.name_scope("Isotropic_transform"):
            r = self.parameters["range"].value
            return x / r
    
    def set_limits(self, data):
        self.parameters["range"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)


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
            matrix multiple of maxrange, contained in the [0,1) interval.
        """
        super().__init__()
        self.parameters = {
            "maxrange": _gpr.PositiveParameter(maxrange, 0.1, 10000),
            "minrange_fct": _gpr.RealParameter(minrange_fct, 0.05, 1),
            "azimuth": _gpr.RealParameter(azimuth, 0, 180)}
        self._all_parameters = [pr for pr in self.parameters.values()]
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy2D_refresh"):
            azimuth = self.parameters["azimuth"].value
            maxrange = self.parameters["maxrange"].value
            minrange = self.parameters["minrange_fct"].value * maxrange

            # conversion to radians
            azimuth = azimuth * (_np.pi / 180)

            # conversion to mathematical coordinates
            azimuth = _np.pi / 2 - azimuth

            # rotation matrix
            rot = _tf.stack([_tf.cos(azimuth), -_tf.sin(azimuth),
                             _tf.sin(azimuth), _tf.cos(azimuth)], axis=0)
            rot = _tf.reshape(rot, [2, 2])

            # scaling matrix
            sc = _tf.linalg.diag(_tf.stack([maxrange, minrange], axis=0))

            # anisotropy matrix
            self._anis = _tf.transpose(_tf.matmul(rot, sc))
            self._anis_inv = _tf.linalg.inv(self._anis)
    
    def __call__(self, x):
        with _tf.name_scope("Anisotropy2D_transform"):
            self.refresh()
            return _tf.matmul(x, self._anis_inv)
    
    def set_limits(self, data):
        self.parameters["maxrange"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)


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
            matrix multiple of maxrange, contained in the [0,1) interval.
        minrange_fct : double
            matrix multiple of midrange, contained in the [0,1) interval.
        azimuth : double
            Defined clockwise from north, and is aligned with maxrange.
        dip : double
            Dip angle, from 0 to 90 degrees.
        rake : double
            Rake angle, from -90 to 90 degrees.
        """
        super().__init__()
        self.parameters = {
            "maxrange": _gpr.PositiveParameter(maxrange, 0.1, 10000),
            "midrange_fct": _gpr.RealParameter(midrange_fct, 0.05, 1),
            "minrange_fct": _gpr.RealParameter(minrange_fct, 0.05, 1),
            "azimuth": _gpr.RealParameter(azimuth, 0, 180),
            "dip": _gpr.RealParameter(dip, 0, 90),
            "rake": _gpr.RealParameter(rake, -90, 90)}
        self._all_parameters = [pr for pr in self.parameters.values()]
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy3D_refresh"):
            azimuth = self.parameters["azimuth"].value
            dip = self.parameters["dip"].value
            rake = self.parameters["rake"].value
            maxrange = self.parameters["maxrange"].value
            midrange = _tf.multiply(
                self.parameters["midrange_fct"].value, maxrange)
            minrange = _tf.multiply(
                self.parameters["minrange_fct"].value, midrange)

            # conversion to radians
            azimuth = azimuth * (_np.pi / 180)
            dip = dip * (_np.pi / 180)
            rake = rake * (_np.pi / 180)

            # conversion to mathematical coordinates
            dip = - dip
            azimuth = _np.pi / 2 - azimuth
            rng = _tf.linalg.diag(_tf.stack(
                [midrange, maxrange, minrange], -1))

            # rotation matrix
            rx = _tf.stack([_tf.cos(rake), 0, _tf.sin(rake),
                             0, 1, 0,
                             -_tf.sin(rake), 0, _tf.cos(rake)], -1)
            rx = _tf.reshape(rx, [3, 3])
            ry = _tf.stack([1, 0, 0,
                             0, _tf.cos(dip), -_tf.sin(dip),
                             0, _tf.sin(dip), _tf.cos(dip)], -1)
            ry = _tf.reshape(ry, [3, 3])
            rz = _tf.stack([_tf.cos(azimuth), _tf.sin(azimuth), 0,
                             -_tf.sin(azimuth), _tf.cos(azimuth), 0,
                             0, 0, 1], -1)
            rz = _tf.reshape(rz, [3, 3])

            # anisotropy matrix
            anis = _tf.matmul(_tf.matmul(_tf.matmul(rz, ry), rx), rng)
            self._anis = _tf.transpose(anis)
            self._anis_inv = _tf.linalg.inv(self._anis)
    
    def __call__(self, x):
        with _tf.name_scope("Anisotropy3D_transform"):
            self.refresh()
            return _tf.matmul(x, self._anis_inv)
    
    def set_limits(self, data):
        self.parameters["maxrange"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)


class ProjectionTo1D(_Transform):
    """
    Projection of high-dimensional data to a line.
    """
    def __init__(self, n_dim):
        """
        Initializer for ProjectionTo1D.

        Parameters
        ----------
        n_dim : int
            The number of dimensions. May be greater than 3 when used in
            conjunction with a space-expanding transform.
        """
        super().__init__()
        self.parameters = {"directions": _gpr.PositiveParameter(
            _np.ones(n_dim), _np.ones(n_dim) * 0.001, _np.ones(n_dim))}
        self._all_parameters = [pr for pr in self.parameters.values()]

    def __call__(self, x):
        with _tf.name_scope("ProjectionTo1D_transform"):
            vector = _tf.expand_dims(self.parameters["directions"].value,
                                     axis=1)
            vector = vector / _tf.math.reduce_euclidean_norm(vector)
            x = _tf.matmul(x, vector)
        return x


class AnisotropyARD(_Transform):
    """Automatic Relevance Detection"""

    def __init__(self, n_dim):
        """
        Initializer for Isotropic.

        Parameters
        ----------
        n_dim : int
            The number of dimensions. May be greater than 3 when used in
            conjunction with a space-expanding transform.
        """
        super().__init__()
        self.parameters = {"ranges": _gpr.PositiveParameter(
            _np.ones(n_dim), _np.ones(n_dim)*0.001, _np.ones(n_dim)*10000)}
        self._all_parameters = [pr for pr in self.parameters.values()]

    def __call__(self, x):
        with _tf.name_scope("ARD_transform"):
            ranges = _tf.expand_dims(self.parameters["ranges"].value, axis=0)
            x_tr = x / ranges
        return x_tr


class ChainedTransform(_Transform):
    def __init__(self, *transforms):
        super().__init__()
        count = -1
        for tr in transforms:
            count += 1
            names = list(tr.parameters.keys())
            names = [s + "_" + str(count) for s in names]
            self.parameters.update(zip(names, tr.parameters.values()))
        self.transforms = transforms
        self._all_parameters = [tr.all_parameters for tr in transforms]
        self._all_parameters = [item for sublist in self._all_parameters
                                for item in sublist]

    def __call__(self, x):
        for tr in self.transforms:
            x = tr.__call__(x)
        return x

    def set_limits(self, data):
        self.transforms[0].set_limits(data)


class SelectVariables(_Transform):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __call__(self, x):
        x = _tf.gather(x, self.index, axis=1)
        r = _tf.rank(x)
        x = _tf.cond(_tf.equal(r, 1),
                     lambda: _tf.expand_dims(x, 1),
                     lambda: x)
        return x
