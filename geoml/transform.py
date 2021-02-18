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

# __all__ = ["Identity",
#            "Isotropic",
#            "Anisotropy2D",
#            "Anisotropy3D",
#            "ProjectionTo1D",
#            "AnisotropyARD",
#            "ChainedTransform",
#            "SelectVariables"]

import geoml.parameter as _gpr
# import geoml.interpolation as _gint
# import geoml.tftools as _tftools

import numpy as _np
import tensorflow as _tf


class _Transform(object):
    """An abstract class for variable transformations"""
    def __init__(self):
        if not hasattr(self, "parameters"):
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
            s += "  " * depth + name + ": " \
                 + str(parameter.get_value().numpy())
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
                value.append(_tf.reshape(parameter.variable, [-1]).
                                 numpy())
                shape.append(_tf.shape(parameter.variable).numpy())
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
    def __init__(self, r=1.0):
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
            r = self.parameters["range"].get_value()
            return x / r
    
    def set_limits(self, data):
        self.parameters["range"].set_limits(
            min_val=data.diagonal / 1000,
            max_val=data.diagonal * 2)


class Anisotropy2D(_Transform):
    """Anisotropy in two dimensions"""
    def __init__(self, maxrange=1.0, minrange_fct=1, azimuth=0):
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
            "azimuth": _gpr.CircularParameter(azimuth, 0, 180)}
        self._all_parameters = [pr for pr in self.parameters.values()]
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy2D_refresh"):
            azimuth = self.parameters["azimuth"].get_value()
            maxrange = self.parameters["maxrange"].get_value()
            minrange = self.parameters["minrange_fct"].get_value() * maxrange

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


class Anisotropy2DMath(_Transform):
    """Anisotropy in two dimensions"""

    def __init__(self, range_x=1.0, range_y=1.0, theta=0):
        """
        Anisotropy matrix in mathematical parametrization.

        Parameters
        ----------
        range_x, range_y : double
            The ellipsoid semi-length in each direction. Must be positive.
        theta : double
            The rotation angle in degrees.
        """
        super().__init__()
        self.parameters = {
            "range_x": _gpr.PositiveParameter(range_x, 0.1, 10000),
            "range_y": _gpr.PositiveParameter(range_y, 0.1, 10000),
            "theta": _gpr.CircularParameter(theta, 0, 360)}
        self._all_parameters = [pr for pr in self.parameters.values()]
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy2DMath_refresh"):
            range_x = self.parameters["range_x"].get_value()
            range_y = self.parameters["range_y"].get_value()
            theta = self.parameters["theta"].get_value()

            # conversion to radians
            theta = theta * (_np.pi / 180)

            # rotation matrix
            rz = _tf.stack([_tf.cos(theta), - _tf.sin(theta),
                            _tf.sin(theta), _tf.cos(theta)], -1)
            rz = _tf.reshape(rz, [2, 2])

            rng = _tf.linalg.diag(_tf.stack(
                [range_x, range_y], -1))

            # anisotropy matrix
            anis = _tf.matmul(rz, rng)
            self._anis = _tf.transpose(anis)
            self._anis_inv = _tf.linalg.inv(self._anis)

    def __call__(self, x):
        with _tf.name_scope("Anisotropy2D_transform"):
            self.refresh()
            return _tf.matmul(x, self._anis_inv)

    def set_limits(self, data):
        self.parameters["range_x"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)
        self.parameters["range_y"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)


class Anisotropy3D(_Transform):
    """Anisotropy in two dimensions"""
    def __init__(self, maxrange=1.0, midrange_fct=1, minrange_fct=1,
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
        self.parameters = {
            "maxrange": _gpr.PositiveParameter(maxrange, 0.1, 10000),
            "midrange_fct": _gpr.RealParameter(midrange_fct, 0.05, 1),
            "minrange_fct": _gpr.RealParameter(minrange_fct, 0.01, 1),
            "azimuth": _gpr.CircularParameter(azimuth, 0, 360),
            "dip": _gpr.RealParameter(dip, 0, 90),
            "rake": _gpr.RealParameter(rake, -90, 90)}
        self._all_parameters = [pr for pr in self.parameters.values()]
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy3D_refresh"):
            azimuth = self.parameters["azimuth"].get_value()
            dip = self.parameters["dip"].get_value()
            rake = self.parameters["rake"].get_value()
            maxrange = self.parameters["maxrange"].get_value()
            midrange = _tf.multiply(
                self.parameters["midrange_fct"].get_value(), maxrange)
            minrange = _tf.multiply(
                self.parameters["minrange_fct"].get_value(), midrange)

            # conversion to radians
            azimuth = azimuth * (_np.pi / 180)
            dip = dip * (_np.pi / 180)
            rake = rake * (_np.pi / 180)

            # conversion to mathematical coordinates
            dip = - dip
            # azimuth = _np.pi / 2 - azimuth
            rng = _tf.linalg.diag(_tf.stack(
                [midrange, maxrange, minrange], -1))

            # rotation matrix
            # x and y axes are switched
            # rotation over z is with sign reversed
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


class Anisotropy3DMath(_Transform):
    """Anisotropy in two dimensions"""

    def __init__(self, range_x=1.0, range_y=1.0, range_z=1.0,
                 theta_x=0, theta_y=0, theta_z=0):
        """
        Anisotropy matrix in mathematical parametrization.

        Parameters
        ----------
        range_x, range_y, range_z : double
            The ellipsoid semi-length in each direction. Must be positive.
        theta_x, theta_y, theta_z : double
            The rotation angles in degrees.
        """
        super().__init__()
        self.parameters = {
            "range_x": _gpr.PositiveParameter(range_x, 0.1, 10000),
            "range_y": _gpr.PositiveParameter(range_y, 0.1, 10000),
            "range_z": _gpr.PositiveParameter(range_z, 0.1, 10000),
            "theta_x": _gpr.CircularParameter(theta_x, 0, 360),
            "theta_y": _gpr.CircularParameter(theta_y, 0, 360),
            "theta_z": _gpr.CircularParameter(theta_z, 0, 360)}
        self._all_parameters = [pr for pr in self.parameters.values()]
        self._anis = None
        self._anis_inv = None

    def refresh(self):
        with _tf.name_scope("Anisotropy3DMath_refresh"):
            range_x = self.parameters["range_x"].get_value()
            range_y = self.parameters["range_y"].get_value()
            range_z = self.parameters["range_z"].get_value()
            theta_x = self.parameters["theta_x"].get_value()
            theta_y = self.parameters["theta_y"].get_value()
            theta_z = self.parameters["theta_z"].get_value()

            # conversion to radians
            theta_x = theta_x * (_np.pi / 180)
            theta_y = theta_y * (_np.pi / 180)
            theta_z = theta_z * (_np.pi / 180)

            # rotation matrix
            rx = _tf.stack([1, 0, 0,
                            0, _tf.cos(theta_x), - _tf.sin(theta_x),
                            0, _tf.sin(theta_x), _tf.cos(theta_x)], -1)
            rx = _tf.reshape(rx, [3, 3])
            ry = _tf.stack([_tf.cos(theta_y), 0, _tf.sin(theta_y),
                            0, 1, 0,
                            - _tf.sin(theta_y), 0, _tf.cos(theta_y)], -1)
            ry = _tf.reshape(ry, [3, 3])
            rz = _tf.stack([_tf.cos(theta_z), - _tf.sin(theta_z), 0,
                            _tf.sin(theta_z), _tf.cos(theta_z), 0,
                            0, 0, 1], -1)
            rz = _tf.reshape(rz, [3, 3])

            rng = _tf.linalg.diag(_tf.stack(
                [range_x, range_y, range_z], -1))

            # anisotropy matrix
            anis = _tf.matmul(_tf.matmul(_tf.matmul(rz, ry), rx), rng)
            self._anis = _tf.transpose(anis)
            self._anis_inv = _tf.linalg.inv(self._anis)

    def __call__(self, x):
        with _tf.name_scope("Anisotropy3D_transform"):
            self.refresh()
            return _tf.matmul(x, self._anis_inv)

    def set_limits(self, data):
        self.parameters["range_x"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)
        self.parameters["range_y"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)
        self.parameters["range_z"].set_limits(
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
            vector = _tf.expand_dims(self.parameters["directions"].get_value(),
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
            The number of dimensions.
        """
        super().__init__()
        self.parameters = {"ranges": _gpr.PositiveParameter(
            _np.ones(n_dim), _np.ones(n_dim)*0.001, _np.ones(n_dim)*1000)}
        self._all_parameters = [pr for pr in self.parameters.values()]

    def __call__(self, x):
        with _tf.name_scope("ARD_transform"):
            ranges = _tf.expand_dims(
                self.parameters["ranges"].get_value(), axis=0)
            x_tr = x / ranges
        return x_tr

    def set_limits(self, data):
        dif = data.bounding_box[1, :] - data.bounding_box[0, :]
        self.parameters["ranges"].set_limits(
            min_val=dif / 100,
            max_val=dif * 2)
        self.parameters["ranges"].set_value(dif/2)


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


class NormalizeWithBoundingBox(_Transform):
    def __init__(self, box):
        super().__init__()
        self.box = _tf.constant(box, _tf.float64)

    def __call__(self, x):
        with _tf.name_scope("NormalizeWithBoundingBox_transform"):
            coords_min = _tf.expand_dims(self.box[0, :], axis=0)
            coords_dif = _tf.expand_dims(self.box[1, :] - self.box[0, :],
                                         axis=0)
            return 2*(x - coords_min)/coords_dif - 1


class Periodic(_Transform):
    def __call__(self, x):
        with _tf.name_scope("Periodic_transform"):
            features = _tf.concat([_tf.sin(2.0 * _np.pi * x),
                                   _tf.cos(2.0 * _np.pi * x)], axis=1)
            return features


class Concatenate(ChainedTransform):
    def __call__(self, x):
        transformed = [tr.__call__(x) for tr in self.transforms]
        return _tf.concat(transformed, axis=1)

    def set_limits(self, data):
        for tr in self.transforms:
            tr.set_limits(data)


class Sine(_Transform):
    def __call__(self, x):
        with _tf.name_scope("Sine_transform"):
            return _tf.sin(2*_np.pi * x)


class Linear(_Transform):
    def __init__(self, dim_in, dim_out, bias=True):
        w = _tf.random.normal([dim_in, dim_out], dtype=_tf.float64)
        b = _tf.zeros([1, dim_out], dtype=_tf.float64)
        self.parameters = {
            "weights": _gpr.RealParameter(
                w, -1000 * _tf.ones_like(w), 1000 * _tf.ones_like(w)),
        }
        if bias:
            self.parameters["bias"] = _gpr.RealParameter(
                b, b - 1000, b + 1000)
        self.dim_in = dim_in
        self.dim_out = dim_out
        super().__init__()

    def __call__(self, x):
        with _tf.name_scope("Linear_transform"):
            w = self.parameters["weights"].get_value()
            b = self.parameters["bias"].get_value() \
                if "bias" in self.parameters.keys() else 0.0
            return _tf.matmul(x, w) + b


class Normalize(_Transform):
    def __call__(self, x):
        return x / (_tf.math.reduce_euclidean_norm(
            x, axis=1, keepdims=True) + 1e-6)


class Swish(Linear):
    def __call__(self, x):
        with _tf.name_scope("Swish_transform"):
            x = super().__call__(x)
            return x * _tf.nn.sigmoid(x)


class ReLU(Linear):
    def __call__(self, x):
        with _tf.name_scope("Swish_transform"):
            x = super().__call__(x)
            return _tf.nn.relu(x)


class Tanh(Linear):
    def __call__(self, x):
        with _tf.name_scope("Tanh_transform"):
            x = super().__call__(x)
            return _tf.math.tanh(x)
