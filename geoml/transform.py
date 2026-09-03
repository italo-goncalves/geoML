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

# __all__ = ["Identity",
#            "Isotropic",
#            "Anisotropy2D",
#            "Anisotropy3D",
#            "ProjectionTo1D",
#            "AnisotropyARD",
#            "ChainedTransform",
#            "SelectVariables"]

import geoml.math.rbf as _rbf
import geoml.parameter as _gpr
import geoml.stats.random as _rnd
# import geoml.interpolation as _gint
# import geoml.tftools as _tftools

import numpy as _np
import scipy.spatial as _spatial
import warnings as _warnings
import tensorflow as _tf


class _Transform(_gpr.Parametric):
    """An abstract class for variable transformations"""
    # affine -- the Jacobian is the same at every point -- which is what
    # lets `latent.GaussianInput` carry an input variance through in one
    # matrix product; the nonlinear ones (`Periodic`, the faults) keep the
    # default and are differentiated at every point. Declared per class and
    # checked against a numerical Jacobian by `test_gaussian_input.py`.
    _linear = False

    @property
    def linear(self):
        """Whether the transform is affine, its Jacobian constant."""
        return self._linear

        
    def refresh(self):
        pass
    
    def set_limits(self, data):
        pass
    
    def __call__(self, x):
        pass

class Identity(_Transform):
    """The identity transformation"""
    
    _linear = True

    def __call__(self, x):
        with _tf.name_scope("Identity_transform"):
            return x


class Isotropic(_Transform):
    """Isotropic range"""
    _linear = True

    def __init__(self, r: float = 1.0):
        """
        Initializer for Isotropic.

        Parameters
        ----------
        r : double
            The range. Must be positive.
        """
        super().__init__()
        self._add_parameter("range", _gpr.PositiveParameter(r, 0.1, 10000))
    
    def __call__(self, x):
        with _tf.name_scope("Isotropic_transform"):
            r = self.parameters["range"].get_value()
            return x / r
    
    def set_limits(self, data):
        base_range = data.diagonal / (data.n_data ** (1/data.n_dim))

        # self.parameters["range"].set_limits(
        #     min_val=data.diagonal / 1000,
        #     max_val=data.diagonal / 2)

        self.parameters["range"].set_limits(
            min_val=base_range * 2,
            max_val=data.diagonal * 2)


class _Ellipsoidal(_Transform):
    _linear = True

    def __init__(self):
        super().__init__()
        self._anis = None
        self._anis_inv = None

    @property
    def anis(self):
        return self._anis

    def __call__(self, x):
        with _tf.name_scope("Ellipsoidal_transform"):
            self.refresh()
            return _tf.matmul(x, self._anis_inv)


class Anisotropy2D(_Ellipsoidal):
    """Anisotropy in two dimensions"""
    def __init__(self, maxrange: float = 1.0, minrange_fct: float = 1,
                 azimuth: float = 0):
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
        self._add_parameter("maxrange",
                            _gpr.PositiveParameter(maxrange, 0.1, 10000))
        self._add_parameter("minrange_fct",
                            _gpr.RealParameter(minrange_fct, 0.05, 1))
        self._add_parameter("azimuth",
                            _gpr.CircularParameter(azimuth, 0, 180))

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
    
    def set_limits(self, data):
        self.parameters["maxrange"].set_limits(
            min_val=data.diagonal / 100,
            max_val=data.diagonal * 2)


class Anisotropy2DMath(_Ellipsoidal):
    """Anisotropy in two dimensions"""

    def __init__(self, range_x: float = 1.0, range_y: float = 1.0,
                 theta: float = 0):
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
        self._add_parameter("range_x",
                            _gpr.PositiveParameter(range_x, 0.1, 10000))
        self._add_parameter("range_y",
                            _gpr.PositiveParameter(range_y, 0.1, 10000))
        self._add_parameter("theta",
                            _gpr.CircularParameter(theta, 0, 360))

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

    def set_limits(self, data):
        base_range = data.diagonal / (data.n_data ** (1 / data.n_dim))

        self.parameters["range_x"].set_limits(
            min_val=base_range * 2,
            max_val=data.diagonal * 2)
        self.parameters["range_y"].set_limits(
            min_val=base_range * 2,
            max_val=data.diagonal * 2)


class Anisotropy2DDynamic(_Ellipsoidal):
    def __init__(self, n_directions: int = 9):
        super().__init__()
        self._base_transforms = []
        for i in range(n_directions):
            tr = Anisotropy2DMath(theta=i * 90 / n_directions)
            tr.parameters["theta"].fix()
            self._base_transforms.append(self._register(tr))

        rnd = _rnd.rng().normal(size=(n_directions, 1))
        rnd = rnd / _np.sqrt(_np.sum(rnd ** 2, axis=0, keepdims=True))
        self._add_parameter(
            "weights",
            _gpr.UnitColumnNormParameter(
                rnd, - _np.ones_like(rnd), _np.ones_like(rnd)
            )
        )

    def refresh(self):
        with _tf.name_scope("Anisotropy2DDynamic_refresh"):
            for tr in self._base_transforms:
                tr.refresh()

            # anisotropy matrix
            w = self.parameters["weights"].get_value()[:, :, None]
            anis = _tf.stack([tr.anis for tr in self._base_transforms],
                             axis=0)

            self._anis = _tf.reduce_sum(anis * w, axis=0)
            self._anis_inv = _tf.linalg.inv(self._anis)

    def set_limits(self, data):
        for tr in self._base_transforms:
            tr.set_limits(data)
            tr.parameters["range_x"].set_value(data.diagonal / 10)
            tr.parameters["range_y"].set_value(data.diagonal / 10)


class Anisotropy3D(_Ellipsoidal):
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
        self._add_parameter("maxrange",
                            _gpr.PositiveParameter(maxrange, 0.1, 10000))
        self._add_parameter("midrange_fct",
                            _gpr.RealParameter(midrange_fct, 0.05, 1))
        self._add_parameter("minrange_fct",
                            _gpr.RealParameter(minrange_fct, 0.01, 1))
        self._add_parameter("azimuth",
                            _gpr.CircularParameter(azimuth, 0, 360))
        self._add_parameter("dip",
                            _gpr.RealParameter(dip, 0, 90))
        self._add_parameter("rake",
                            _gpr.RealParameter(rake, -90, 90))

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
            self._anis_inv = _tf.linalg.inv(self._anis) # + _tf.eye(3, dtype=_tf.float64) * 1e-3)
    
    def set_limits(self, data):
        self.parameters["maxrange"].set_limits(
            min_val=data.diagonal / 1000,
            max_val=data.diagonal * 2)


class Anisotropy3DMath(_Ellipsoidal):
    """Anisotropy in two dimensions"""

    def __init__(self, range_x: float = 1.0, range_y: float = 1.0,
                 range_z: float = 1.0, theta_x: float = 0,
                 theta_y: float = 0, theta_z: float = 0):
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
        self._add_parameter("range_x",
                            _gpr.PositiveParameter(range_x, 0.1, 10000))
        self._add_parameter("range_y",
                            _gpr.PositiveParameter(range_y, 0.1, 10000))
        self._add_parameter("range_z",
                            _gpr.PositiveParameter(range_z, 0.1, 10000))
        self._add_parameter("theta_x",
                            _gpr.CircularParameter(theta_x, 0, 360))
        self._add_parameter("theta_y",
                            _gpr.CircularParameter(theta_y, 0, 360))
        self._add_parameter("theta_z",
                            _gpr.CircularParameter(theta_z, 0, 360))

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


class Anisotropy3DDynamic(_Ellipsoidal):
    def __init__(self, n_directions_per_axis=3):
        super().__init__()
        self._base_transforms = []
        for i in range(n_directions_per_axis):
            for j in range(n_directions_per_axis):
                for k in range(n_directions_per_axis):
                    tr = Anisotropy3DMath(
                        theta_x=i * 90 / n_directions_per_axis,
                        theta_y=j * 90 / n_directions_per_axis,
                        theta_z=k * 90 / n_directions_per_axis
                    )
                    tr.parameters["theta_x"].fix()
                    tr.parameters["theta_y"].fix()
                    tr.parameters["theta_z"].fix()
                    self._base_transforms.append(self._register(tr))

        rnd = _rnd.rng().normal(size=(n_directions_per_axis**3, 1))
        rnd = rnd / _np.sqrt(_np.sum(rnd ** 2, axis=0, keepdims=True))
        self._add_parameter(
            "weights",
            _gpr.UnitColumnNormParameter(
                rnd, - _np.ones_like(rnd), _np.ones_like(rnd)
            )
        )

    def refresh(self):
        with _tf.name_scope("Anisotropy3DDynamic_refresh"):
            for tr in self._base_transforms:
                tr.refresh()

            # anisotropy matrix
            w = self.parameters["weights"].get_value()[:, :, None]
            anis = _tf.stack([tr.anis for tr in self._base_transforms],
                             axis=0)

            self._anis = _tf.reduce_sum(anis * w, axis=0)
            self._anis_inv = _tf.linalg.inv(
                self._anis + _tf.eye(3, dtype=_tf.float64) * 1e-9
            )

    def set_limits(self, data):
        for tr in self._base_transforms:
            tr.set_limits(data)
            tr.parameters["range_x"].set_value(data.diagonal / 10)
            tr.parameters["range_y"].set_value(data.diagonal / 10)
            tr.parameters["range_z"].set_value(data.diagonal / 10)


class ProjectionTo1D(_Transform):
    """
    Projection of high-dimensional data to a line.
    """
    _linear = True

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
        self._add_parameter("directions", _gpr.PositiveParameter(
            _np.ones(n_dim), _np.ones(n_dim) * 0.001, _np.ones(n_dim)))

    def __call__(self, x):
        with _tf.name_scope("ProjectionTo1D_transform"):
            vector = _tf.expand_dims(self.parameters["directions"].get_value(),
                                     axis=1)
            vector = vector / _tf.math.reduce_euclidean_norm(vector)
            x = _tf.matmul(x, vector)
        return x


class AnisotropyARD(_Transform):
    """Automatic Relevance Detection"""

    _linear = True

    def __init__(self, n_dim):
        """
        Initializer for Isotropic.

        Parameters
        ----------
        n_dim : int
            The number of dimensions.
        """
        super().__init__()
        self._add_parameter("ranges", _gpr.PositiveParameter(
            _np.ones(n_dim), _np.ones(n_dim)*0.001, _np.ones(n_dim)*1000))

    def __call__(self, x):
        with _tf.name_scope("ARD_transform"):
            ranges = _tf.expand_dims(
                self.parameters["ranges"].get_value(), axis=0)
            x_tr = x / ranges
        return x_tr

    def set_limits(self, data):
        box = data.bounding_box.as_array()
        dif = box[1, :] - box[0, :]
        self.parameters["ranges"].set_limits(
            min_val=dif / 100,
            max_val=dif * 2)
        self.parameters["ranges"].set_value(dif/10)


class ChainedTransform(_Transform):
    """
    Chained transform.

    This object allows multiple transforms to be called sequentially. Useful for complex transformations.
    """
    def __init__(self, *transforms):
        """
        Chained transform.

        Parameters
        ----------
        transforms
            Transformes to chain.
        """
        super().__init__()
        self.transforms = transforms
        for tr in transforms:
            self._register(tr)

    @property
    def linear(self):
        return all(tr.linear for tr in self.transforms)

    def __call__(self, x):
        for tr in self.transforms:
            x = tr.__call__(x)
        return x

    def set_limits(self, data):
        self.transforms[0].set_limits(data)


class SelectVariables(_Transform):
    """
    Variable selection.

    Returns the specified columns of the input, discarding the others.
    """
    _linear = True

    def __init__(self, index):
        """
        Variable selection.

        Parameters
        ----------
        index : list
            Indices of variables to select.
        """
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
    """
    Normalization with a bounding box.

    Uses a `BoundingBox` object as guide to normalize the data. All columns will be contained in the [-3, 3] interval.

    """
    _linear = True

    def __init__(self, box):
        """
        Normalization with a bounding box.

        Parameters
        ----------
        box
            The bounding box to use as reference for normalization.
        """
        super().__init__()
        self.box = _tf.constant(box.as_array(), _tf.float64)

    def __call__(self, x):
        with _tf.name_scope("NormalizeWithBoundingBox_transform"):
            coords_min = _tf.expand_dims(self.box[0, :], axis=0)
            coords_dif = _tf.expand_dims(self.box[1, :] - self.box[0, :],
                                         axis=0)
            return 6*(x - coords_min)/coords_dif - 3


class Periodic(_Transform):
    """
    Periodic transform.

    Returns sines and cosines doubling the number of columns in the data. The periods can be specified by
    chaining an `Anisotropy*` transform with this one.
    """
    def __call__(self, x):
        with _tf.name_scope("Periodic_transform"):
            features = _tf.concat([_tf.sin(2.0 * _np.pi * x),
                                   _tf.cos(2.0 * _np.pi * x)], axis=1)
            return features


class Concatenate(ChainedTransform):
    """
    Concatenation of transforms.

    Consolidates a list of inputs into a single one.
    """
    def __call__(self, x):
        transformed = [tr.__call__(x) for tr in self.transforms]
        return _tf.concat(transformed, axis=1)

    def set_limits(self, data):
        for tr in self.transforms:
            tr.set_limits(data)


class RandomProjections(_Transform):
    _linear = True

    def __init__(self, n_dim, n_directions, seed=1234):
        super().__init__()
        self.n_directions = n_directions
        self.n_dim = n_dim

        if n_dim == 1:
            raise ValueError("Invalid n_dim: must be 2 or greater")

        if n_dim == 2:
            angles = _np.linspace(0, _np.pi, n_directions + 1)[:-1]
            projections = _np.stack([_np.cos(angles), _np.sin(angles)], axis=1)
            self.projections = _tf.constant(projections.T, _tf.float64)
        else:
            # a generator of its own: this transform is reproducible from its
            # own seed, and seeding the global one would reach every draw made
            # afterwards
            projections = _np.random.default_rng(seed).normal(
                size=[n_dim, n_directions])
            norm = _np.sqrt(_np.sum(projections**2, axis=0, keepdims=True))
            self.projections = _tf.constant(projections / norm, _tf.float64)

    def __call__(self, x):
        return _tf.matmul(x, self.projections)


class BellFault2D(_Transform):
    """Fault simulation."""
    def __init__(self, start, end):
        """
        Fault simulation.

        Creates a discontinuity in space, returning an additional coordinate that can be used to artificially
        repel points on opposite sides of a line.

        Parameters
        ----------
        start : array-like
            Starting point of fault.
        end : array-like
            Endpoint of fault.
        """
        super().__init__()
        self.start = _np.array(start)
        self.end = _np.array(end)
        self.midpoint = 0.5 * (self.start + self.end)

        r = _np.sqrt(_np.sum((start - end)**2))
        self._add_parameter("range", _gpr.PositiveParameter(r/5, r/10, r*10))
        self._add_parameter("amp", _gpr.PositiveParameter(r/5, r/20, r))

    @staticmethod
    def kernelize(x):
        x = _tf.minimum(x, 1.0)
        amp = 1 - 7*x**2 + 35/4*x**3 - 7/2*x**5 + 3/4*x**7
        return amp

    def __call__(self, x):
        with _tf.name_scope("BellFault2D_transform"):
            dif = self.end - self.midpoint
            length = _np.sqrt(_np.sum(dif**2))

            x_along = _tf.reduce_sum(
                (x - self.midpoint[None, :]) * dif[None, :],
                axis=1, keepdims=True) \
                      * dif[None, :] / length ** 2
            x_across = (x - self.midpoint[None, :]) - x_along

            proj_along = _tf.math.reduce_euclidean_norm(x_along, axis=1) / length
            proj_across = _tf.math.reduce_euclidean_norm(x_across, axis=1) / length

            vector_prod = x_across[:, 0] * dif[1] - x_across[:, 1] * dif[0]
            rng = self.parameters["range"].get_value()
            sign = _tf.sign(vector_prod) * _tf.math.exp(- 3 * proj_across / rng)

            amp = self.parameters["amp"].get_value()
            out = self.kernelize(proj_along) * amp * sign
            return out[:, None]


def _bell(x):
    """One at zero, zero from one on, twice differentiable: the taper
    `BellFault2D` has always used."""
    x = _tf.minimum(x, 1.0)
    return 1 - 7 * x ** 2 + 35 / 4 * x ** 3 - 7 / 2 * x ** 5 + 3 / 4 * x ** 7


class _ImplicitSurface(_Transform):
    """
    A surface fitted from its observations, with the taper that confines
    whatever is built on it to the surface's extent.

    The observations are recorded, not the fitted object, so a saved model
    refits the same surface on reload. The field is
    `geoml.math.rbf.HermiteRBF`; the taper projects a point onto the
    surface with one Newton step, measures the projection's distance to
    the nearest observation past twice the observations' own spacing, and
    lets the feature fade to nothing over `reach` beyond the footprint --
    a fault dies out past where it was seen, as the displacement envelopes
    of Laurent et al. (2013) and Georgsen et al. (2012) have it.
    """
    def __init__(self, points, normals=None, basis="cubic", reach=None,
                 k=12):
        super().__init__()
        self.points = _np.asarray(points, dtype=float)
        if normals is None or normals is False:
            self.normals = normals
        elif isinstance(normals, (tuple, list)) and len(normals) == 2 \
                and _np.ndim(normals[0]) == 2:
            self.normals = (_np.asarray(normals[0], dtype=float),
                            _np.asarray(normals[1], dtype=float))
        else:
            self.normals = _np.asarray(normals, dtype=float)
        self.surface = _rbf.HermiteRBF(self.points, normals=self.normals,
                                       basis=basis, k=k)
        low, high = self.points.min(axis=0), self.points.max(axis=0)
        self._extent = float(_np.linalg.norm(high - low))
        spacing, _ = _spatial.cKDTree(self.points).query(self.points, k=2)
        self._spacing = float(_np.median(spacing[:, 1]))
        self.reach = self._extent / 4 if reach is None else float(reach)
        self._centres = _tf.constant(self.points)

    @staticmethod
    def _foot(x, value, gradient):
        """Each point's foot on the surface, by one Newton step of the
        field: where along the fault the point sits."""
        norm2 = _tf.reduce_sum(gradient ** 2, axis=1, keepdims=True)
        return x - _tf.math.divide_no_nan(value[:, None] * gradient, norm2)

    def _taper(self, x, value, gradient, centres):
        return self._taper_at(self._foot(x, value, gradient), centres)

    def _taper_at(self, projected, centres):
        delta = projected[:, None, :] - centres[None]
        # floored: at zero distance the square root's infinite derivative
        # met the zero gradient of `maximum` below and made NaN
        nearest = _tf.sqrt(_tf.maximum(
            _tf.reduce_min(_tf.reduce_sum(delta ** 2, axis=2), axis=1),
            (1e-3 * self._spacing) ** 2))
        beyond = _tf.maximum(nearest - 2.0 * self._spacing, 0.0)
        return _bell(beyond / self.reach)


class ImplicitFault(_ImplicitSurface):
    """
    A fault fitted from its observations, as one extra coordinate that
    repels points across it.

    `BellFault2D` for any surface and any dimension: the field is fitted to
    the fault's points and normals, and the coordinate returned is
    `amp * sign(s) * g(|s|) * taper`, opposite in sign on the two sides,
    so a kernel reading it beside the spatial coordinates sees points
    across the fault as far apart. `mode="step"` is the fault-block
    indicator, `±amp` on the two sides; `mode="decay"` fades with distance
    from the fault over `range` as `BellFault2D` does. The jump is the
    point: a smooth ramp, however narrow, is what the kernel already sees
    in the coordinates. It is the kernel-space form of the fault
    drift of the potential-field method (Calcagno et al. 2008; de la Varga
    et al. 2019), not a new idea; it needs no fault topology, several of
    them simply concatenate. `amp` and `range` train.

    Parameters
    ----------
    points
        The fault observations, `(n, d)`.
    normals
        The gradient constraints, as `HermiteRBF` takes them: aligned with
        the points with NaN rows where there is none, a `(locations,
        vectors)` pair, or None to derive one per point toward the
        surface's concavity (`geoml.math.geometry.point_normals`). One is
        enough.
    basis
        As in `HermiteRBF`.
    reach
        How far past the observations the feature persists; a quarter of
        their extent by default.
    mode
        `"step"` or `"decay"`.
    k
        Neighbours used to derive the normals.

    Notes
    -----
    Use it in a `Concatenate` beside the spatial transform, with the input
    node's `center=False` (the default), since the surface is fitted in
    the coordinates the observations came in.

    References
    ----------
    Calcagno, P., Chilès, J. P., Courrioux, G. and Guillen, A. (2008).
    Geological modelling from field data and geological knowledge, Part I.
    Physics of the Earth and Planetary Interiors 171, 147-157.

    de la Varga, M., Schaaf, A. and Wellmann, F. (2019). GemPy 1.0:
    open-source stochastic geological modeling and inversion. Geoscientific
    Model Development 12, 1-32.
    """
    def __init__(self, points, normals=None, basis="cubic", reach=None,
                 mode="step", k=12):
        super().__init__(points, normals, basis, reach, k)
        if mode not in ("step", "decay"):
            raise ValueError("mode must be 'step' or 'decay'")
        self.mode = mode
        extent = self._extent
        self._add_parameter(
            "range", _gpr.PositiveParameter(extent / 5, extent / 50,
                                            extent * 10))
        self._add_parameter(
            "amp", _gpr.PositiveParameter(extent / 5, extent / 20, extent))

    def _coordinate(self, value, taper, bounds=()):
        """The extra coordinate from the field's value, the taper and the
        fields of the faults this one stops on, as `(value, side)` pairs.

        A fault that ends on another is its zero set trimmed to one side of
        the other's field, `{s = 0} ∩ {side * s_j >= 0}`: in step mode the
        trimmed sign is `sign(s) * H(side * s_j)`, a product of steps; in
        decay mode the distance to the trimmed surface,
        `sqrt(s^2 + sum max(0, -side * s_j)^2)`, does the trimming, the
        feature fading past the bounding fault as it fades away from this
        one.
        """
        rng = self.parameters["range"].get_value()
        amp = self.parameters["amp"].get_value()
        # the feature must jump at the fault: a smooth ramp, however
        # narrow, is what the kernel already sees in the coordinates
        # (measured -- a tanh at a fifth of the extent bent the contact
        # like no fault at all, where the jump breaks it cleanly)
        if self.mode == "step":
            across = _tf.sign(value)
            for other, side in bounds:
                across = across * _tf.cast(side * other > 0, _tf.float64)
        else:
            beyond = sum((_tf.maximum(0.0, -side * other) ** 2
                          for other, side in bounds),
                         _tf.zeros_like(value))
            distance = _tf.sqrt(value ** 2 + beyond)
            across = _tf.sign(value) * _tf.exp(-3.0 * distance / rng)
        return (amp * across * taper)[:, None]

    def __call__(self, x):
        with _tf.name_scope("ImplicitFault_transform"):
            value, gradient = self.surface.evaluate(x)
            taper = self._taper(x, value, gradient, self._centres)
            return self._coordinate(value, taper)


class FaultDisplacement(_ImplicitSurface):
    """
    A fault fitted from its observations, as the displacement that
    restores its hanging wall.

    Returns `x - H(s) * taper * slip`: the side the normals point to, where
    the field is positive, is moved back by the slip, over the fault's
    extent, so a sequence displaced by the fault becomes continuous again
    in the transformed coordinates and one kernel can read across it. `H`
    is a smooth step of width `width` about the surface. The slip is said
    in the fault's own frame (Laurent et al. 2013): a `throw` along the
    up-dip direction and, in three dimensions, a `strike_slip` along the
    strike, both read at the point's foot on the surface, so everything
    on one normal line slides together along the fault and the slip stays
    tangent, since a component along the normal is not what a fault does
    and is not identified by the data either. In two dimensions the fault
    has one tangent, the normal turned a quarter turn, and `throw` is
    along it. On a curved fault the restoration is not rigid -- a hanging
    wall sliding on a curved surface bends -- and it is exact on a plane.
    Both train; their sign says which way the hanging wall goes, so the
    normals' orientation only fixes the convention. A translation, as the
    vector fields of Laurent et al. (2013) and Georgsen et al. (2012)
    reduce to over one envelope; the restoration ordering of several
    faults is `FaultNetwork`'s.

    The move itself is fault-parallel flow: each point follows the level
    set of the field through it, by `flow_steps` midpoint steps, so
    material slides along a curved fault rather than stepping straight
    off it; `flow_steps=0` is the straight step along the frame at the
    foot. With `profile="bell"` the throw varies along the fault as
    displacement profiles do, largest at the centre and zero at the tip
    lines, over trainable extents along the fault's mean axes.

    Training note: the throw moves slowly at the default learning rate;
    the phased pattern, `set_learning_rate(0.1)` for a few hundred
    iterations, recovered a 20 m throw to within a metre on a synthetic
    layer with a continuous variable, where the default schedule left it
    at a tenth. With a categorical likelihood, or with several faults, it
    did not move off zero at all: `throw_from_markers` reads it off one
    horizon seen on both walls, `models.search_throw` chooses it among
    candidates on the bound, and `set_width` anneals the step from wide,
    where the bound is smooth in the throw, to sharp.

    Parameters
    ----------
    points, normals, basis, reach, k
        As in `ImplicitFault`.
    throw
        The initial throw along the up-dip direction; zero by default.
    strike_slip
        The initial slip along the strike, three dimensions only; zero
        by default.
    width
        The step's half-width about the surface; a hundredth of the
        observations' extent by default.
    drag
        Whether the width trains, as the width of a drag zone.
    profile
        None for one throw over the fault, `"bell"` for a profile.
    flow_steps
        Midpoint steps of the fault-parallel flow; zero for the straight
        step.

    References
    ----------
    Laurent, G., Caumon, G., Bouziat, A. and Jessell, M. (2013). A
    parametric method to model 3D displacements around faults with
    volumetric vector fields. Tectonophysics 590, 83-93.

    Georgsen, F., Røe, P., Syversveen, A. R. and Lia, O. (2012). Fault
    displacement modelling using 3D vector fields. Computational
    Geosciences 16, 247-259.
    """
    def __init__(self, points, normals=None, throw=0.0, strike_slip=0.0,
                 basis="cubic", reach=None, width=None, drag=False,
                 profile=None, flow_steps=4, k=12):
        super().__init__(points, normals, basis, reach, k)
        extent = self._extent
        d = self.points.shape[1]
        self._add_parameter(
            "throw", _gpr.RealParameter(float(throw), -extent, extent))
        if d == 3:
            self._add_parameter(
                "strike_slip",
                _gpr.RealParameter(float(strike_slip), -extent, extent))
        elif strike_slip:
            raise ValueError("a strike slip needs three dimensions")
        # the step's half-width: fixed by default, trainable as the drag
        # zone's width when asked, and settable either way for annealing
        width = extent / 100 if width is None else float(width)
        self._add_parameter(
            "width", _gpr.PositiveParameter(width, extent / 1e4, extent / 2,
                                            fixed=not drag))
        self.drag = bool(drag)
        self.flow_steps = int(flow_steps)
        if profile not in (None, "bell"):
            raise ValueError("profile must be None or 'bell'")
        self.profile = profile
        centre, axes, half = self._mean_frame()
        self._origin = _tf.constant(centre)
        self._axes = _tf.constant(axes)
        if profile is not None:
            # the profile's extents along the fault's mean axes, trainable,
            # starting a fifth past the observations
            names = ("extent",) if d == 2 else ("extent_strike", "extent_dip")
            for name, h in zip(names, half):
                self._add_parameter(
                    name, _gpr.PositiveParameter(1.2 * h, 0.5 * h, 4.0 * h))
            self._extent_names = names

    def _mean_frame(self):
        """The observations' centre, the fault's mean along-fault axes (the
        tangent in 2-D; strike and dip in 3-D) and the observations'
        half-extents along them."""
        centre = self.points.mean(axis=0)
        gradient = _np.asarray(self.surface.gradient(self.points))
        normal = gradient.mean(axis=0)
        normal = normal / max(_np.linalg.norm(normal), 1e-12)
        if len(normal) == 2:
            axes = _np.array([[-normal[1], normal[0]]])
        else:
            strike = _np.cross([0.0, 0.0, 1.0], normal)
            if _np.linalg.norm(strike) < 1e-6:
                strike = _np.cross([1.0, 0.0, 0.0], normal)
            strike = strike / _np.linalg.norm(strike)
            axes = _np.stack([strike, _np.cross(normal, strike)])
        along = (self.points - centre) @ axes.T
        half = _np.maximum(_np.abs(along).max(axis=0), 1e-6 * self._extent)
        return centre, axes, half

    @property
    def width(self):
        """The step's half-width about the surface."""
        return float(self.parameters["width"].get_value())

    def set_width(self, width):
        """
        Sets the step's half-width, for annealing it between training
        phases: wide, the bound is smooth in the throw and gradient
        descent finds it from afar; narrow, the fault is sharp.
        """
        self.parameters["width"].set_value(float(width))

    def _profile(self, foot):
        """The throw's share at each foot: one everywhere without a
        profile, a bell over the fault's extents with one -- largest at
        the centre, zero at the tip lines, as displacement profiles are."""
        if self.profile is None:
            return _tf.ones([_tf.shape(foot)[0]], _tf.float64)
        along = _tf.linalg.matmul(foot - self._origin[None, :],
                                  self._axes, transpose_b=True)
        extents = _tf.stack([self.parameters[name].get_value()
                             for name in self._extent_names])
        radius = _tf.sqrt(_tf.reduce_sum((along / extents[None, :]) ** 2,
                                         axis=1))
        return _bell(radius)

    def _frame(self, gradient):
        """The local slip directions from the field's gradient: in two
        dimensions the tangent (the unit normal turned a quarter turn), in
        three the up-dip direction and the strike, both unit and tangent."""
        unit = gradient / _tf.maximum(
            _tf.norm(gradient, axis=1, keepdims=True), 1e-12)
        if int(unit.shape[1]) == 2:
            return _tf.stack([-unit[:, 1], unit[:, 0]], axis=1), None
        n = _tf.shape(unit)[0]
        up = _tf.tile(_tf.constant([[0.0, 0.0, 1.0]], _tf.float64), [n, 1])
        east = _tf.tile(_tf.constant([[1.0, 0.0, 0.0]], _tf.float64), [n, 1])
        strike = _tf.linalg.cross(up, unit)
        length = _tf.norm(strike, axis=1, keepdims=True)
        # a horizontal fault has no strike: take the east axis in its plane
        other = _tf.linalg.cross(east, unit)
        other = other / _tf.maximum(_tf.norm(other, axis=1, keepdims=True),
                                    1e-12)
        strike = _tf.where(length > 1e-6, strike / _tf.maximum(length, 1e-12),
                           other)
        dip = _tf.linalg.cross(unit, strike)   # up-dip: its z is |up x n| >= 0
        return dip, strike

    def _slip(self, gradient, share):
        """The slip vector at points whose field gradient is `gradient`,
        scaled by `share` (the step, the taper, the profile): tangent to
        the level set through each point by construction."""
        dip, strike = self._frame(gradient)
        slip = self.parameters["throw"].get_value() * dip
        if strike is not None:
            slip = slip + self.parameters["strike_slip"].get_value() * strike
        return share[:, None] * slip

    def _restore(self, x, value, gradient, centres, evaluate, confine=None,
                 hard=False):
        # samples take the smooth step, so the throw trains through it;
        # another fault's observations take a hard one, or the ones inside
        # the band would be half-restored and bend the refitted surface
        width = self.parameters["width"].get_value()
        step = _tf.cast(value > 0, _tf.float64) if hard \
            else _tf.sigmoid(value / width)
        # the foot on the surface, by two Newton steps: where along the
        # fault the point sits, for the taper and the throw's profile
        foot = self._foot(x, value, gradient)
        at_foot, along = evaluate(foot)
        foot = self._foot(foot, at_foot, along)
        share = step * self._taper_at(foot, centres) * self._profile(foot)
        if confine is not None:
            share = share * confine
        if self.flow_steps == 0:
            # a straight step along the frame at the foot: everything on
            # one normal line slides together (a frame read at the point
            # itself sent neighbours different ways, measured), and the
            # slip is tangent by construction -- a free vector was measured
            # to drift into the component along the normal, which grades
            # cannot identify
            _, along = evaluate(foot)
            return x - self._slip(along, share)
        # fault-parallel flow: the point follows the level set of the field
        # through it, the frame read where it is at each step, by the
        # midpoint rule -- material slides along a curved fault rather than
        # stepping straight off it, and the hanging wall keeps its shape
        # within the radius of curvature
        position = x
        for _ in range(self.flow_steps):
            _, here = evaluate(position)
            midpoint = position - 0.5 * self._slip(here, share) \
                / self.flow_steps
            _, there = evaluate(midpoint)
            position = position - self._slip(there, share) / self.flow_steps
        return position

    def __call__(self, x):
        with _tf.name_scope("FaultDisplacement_transform"):
            value, gradient = self.surface.evaluate(x)
            return self._restore(x, value, gradient, self._centres,
                                 self.surface.evaluate)

    def throw_from_markers(self, hanging, footwall, iterations=3):
        """
        Sets the throw from one horizon seen on both walls.

        What a geologist measures: the same marker on the hanging wall
        and on the footwall, offset by the fault. The footwall markers
        are fitted as an implicit surface; the throw is the one whose
        restoration brings the hanging-wall markers onto that surface, by
        a few Gauss-Newton steps on the surface's field. Needs no bound,
        so it serves as the start `search_throw` and training refine, and
        it reads the profile's peak when there is one.

        Parameters
        ----------
        hanging
            Marker points on the side the restoration moves, `(m, d)`.
        footwall
            Marker points on the other side, `(n, d)`.
        iterations
            Gauss-Newton steps.

        Returns
        -------
        float
            The throw set.
        """
        with _warnings.catch_warnings():
            # a flat horizon has no concave side, which is fine here
            _warnings.simplefilter("ignore")
            reference = _rbf.HermiteRBF(_np.asarray(footwall, dtype=float))
        hanging = _tf.constant(_np.asarray(hanging, dtype=float))
        parameter = self.parameters["throw"]
        throw = float(parameter.get_value())
        h = 1e-3 * self._extent

        def residual(t):
            parameter.set_value(t)
            return _np.asarray(reference(self(hanging)))

        for _ in range(iterations):
            r0 = residual(throw)
            slope = (residual(throw + h) - r0) / h
            denominator = float(_np.sum(slope ** 2))
            if denominator == 0.0:
                break
            throw = throw - float(_np.sum(r0 * slope)) / denominator
        parameter.set_value(throw)
        return float(parameter.get_value())


class FaultNetwork(_Transform):
    """
    Several faults restored in age order, youngest first.

    The youngest fault is undone with its surface as observed. Each older
    fault's observations are then moved by the restorations of the younger
    faults that cut it, so its pieces become one surface again, and its
    field is refitted on those restored positions before its own slip is
    undone -- inside the graph, since the slips train. This is the
    ordering of LoopStructural (Grose et al. 2021) and of the series of
    the potential-field method (Calcagno et al. 2008). A fault that stops
    against an older one, declared in `abutting`, neither displaces that
    fault's observations nor acts beyond its surface: its displacement is
    confined to the declared side of the older fault as observed.

    Parameters
    ----------
    faults
        `FaultDisplacement` objects, youngest first.
    abutting
        Triples `(younger, older, side)` of fault positions and `+1` or
        `-1`: the younger fault stops against the older one and acts only
        on the older field's positive or negative side.

    References
    ----------
    Grose, L., Ailleres, L., Laurent, G. and Jessell, M. (2021).
    LoopStructural 1.0: time-aware geological modelling. Geoscientific
    Model Development 14, 3915-3937.

    Calcagno, P., Chilès, J. P., Courrioux, G. and Guillen, A. (2008).
    Geological modelling from field data and geological knowledge, Part I.
    Physics of the Earth and Planetary Interiors 171, 147-157.
    """
    def __init__(self, faults, abutting=()):
        super().__init__()
        self.faults = tuple(faults)
        if not all(isinstance(f, FaultDisplacement) for f in self.faults):
            raise ValueError("faults must be FaultDisplacement objects")
        for fault in self.faults:
            self._register(fault)
        self.abutting = tuple((int(a), int(b), int(c)) for a, b, c in abutting)
        for younger, older, side in self.abutting:
            if not 0 <= younger < older < len(self.faults):
                raise ValueError("abutting names a younger fault (lower "
                                 "position) stopping against an older one")
            if side not in (-1, 1):
                raise ValueError("the side is +1 or -1")

    def _cuts(self, younger, older):
        return not any(a == younger and b == older
                       for a, b, _ in self.abutting)

    def _confinement(self, k, x):
        """The observed sides of the older faults `k` stops against."""
        confine = None
        for younger, older, side in self.abutting:
            if younger != k:
                continue
            fault = self.faults[older]
            value = fault.surface(x)
            gate = _tf.sigmoid(side * value
                               / fault.parameters["width"].get_value())
            confine = gate if confine is None else confine * gate
        return confine

    def _restored(self, k, points):
        """`points` moved by the faults younger than `k` that cut it."""
        points = _tf.constant(points)
        for j in range(k):
            if self._cuts(j, k):
                points = self._undo(j, points, hard=True)
        return points

    def _restored_points(self, k):
        """Fault `k`'s observations, moved by the younger faults that cut
        it, as a tensor."""
        return self._restored(k, self.faults[k].points)

    def _undo(self, k, x, hard=False):
        fault = self.faults[k]
        centres = self._restored_points(k)
        surface = fault.surface
        constraints = surface.working_gradients
        if constraints is None:
            gradient_centres, gradients = None, None
        else:
            # the constraints' locations move with the observations; the
            # vectors do not, a restoration being a translation
            gradient_centres = surface.to_working(
                self._restored(k, surface.gradient_points))
            gradients = _tf.constant(constraints[1])
        alpha, beta, drift = _rbf.solve_hermite(
            surface.to_working(centres),
            _tf.zeros([int(centres.shape[0])], _tf.float64),
            gradient_centres, gradients, surface.basis)
        working_centres = surface.to_working(centres)

        def evaluate(points):
            value, grad = _rbf.field(surface.to_working(points),
                                     working_centres, alpha,
                                     gradient_centres, beta, drift,
                                     surface.basis)
            return value, grad / surface.scale

        value, grad = evaluate(x)
        return fault._restore(x, value, grad, centres, evaluate,
                              self._confinement(k, x), hard=hard)

    def __call__(self, x):
        with _tf.name_scope("FaultNetwork_transform"):
            for k in range(len(self.faults)):
                x = self._undo(k, x)
            return x


class ImplicitFaultBlocks(_Transform):
    """
    Several repulsions with their terminations: the fault-block partition
    as one extra coordinate per fault.

    The repulsion's twin of `FaultNetwork`, for `ImplicitFault` objects. A
    fault that ends on another is its zero set trimmed to one side of the
    other's field, `{s = 0} ∩ {side * s_j >= 0}`, and its coordinate is
    the trimmed sign, `amp * sign(s) * H(side * s_j) * taper`, a product
    of steps -- identically zero beyond the bounding fault instead of
    fading over a reach, so no coordinate jumps where no fault exists and
    two points in different fault blocks differ in at least one
    coordinate. In decay mode the distance to the trimmed surface,
    `sqrt(s^2 + sum max(0, -side * s_j)^2)`, does the trimming smoothly.
    A chain of terminations composes as the product of its steps. Nothing
    is restored, so every field is read as observed, and the faults need
    no age order. The one artefact is at a junction: crossing the bounding
    fault next to the one that ends on it costs the ending fault's
    amplitude on top of the bounding fault's, since that coordinate drops
    to zero there.

    Parameters
    ----------
    faults
        `ImplicitFault` objects, in any order.
    abutting
        Triples `(stopping, bounding, side)` of fault positions and `+1`
        or `-1`: the first fault exists only on that side of the second's
        field.
    """
    def __init__(self, faults, abutting=()):
        super().__init__()
        self.faults = tuple(faults)
        if not all(isinstance(f, ImplicitFault) for f in self.faults):
            raise ValueError("faults must be ImplicitFault objects")
        for fault in self.faults:
            self._register(fault)
        self.abutting = tuple((int(a), int(b), int(c)) for a, b, c in abutting)
        n = len(self.faults)
        for stopping, bounding, side in self.abutting:
            if not (0 <= stopping < n and 0 <= bounding < n) \
                    or stopping == bounding:
                raise ValueError("abutting names two different faults by "
                                 "position")
            if side not in (-1, 1):
                raise ValueError("the side is +1 or -1")

    def __call__(self, x):
        with _tf.name_scope("ImplicitFaultBlocks_transform"):
            fields = [fault.surface.evaluate(x) for fault in self.faults]
            columns = []
            for k, fault in enumerate(self.faults):
                value, gradient = fields[k]
                taper = fault._taper(x, value, gradient, fault._centres)
                bounds = [(fields[j][0], side)
                          for a, j, side in self.abutting if a == k]
                columns.append(fault._coordinate(value, taper, bounds))
            return _tf.concat(columns, axis=1)
