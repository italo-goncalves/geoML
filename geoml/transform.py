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
           "ProjectionTo1D"]

import numpy as _np
import tensorflow as _tf
import geoml.parameter as _gpr
import geoml.tftools as _tftools

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
            return x
    
    def backward(self, x):
        with _tf.name_scope("Identity_backward"):
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
        self.params = {"ranges": _gpr.PositiveParameter(
            _np.ones(n_dim), _np.ones(n_dim) * 0.001, _np.ones(n_dim) * 10000)}

    def backward(self, x):
        with _tf.name_scope("ProjectionTo1D_backward"):
            vector = _tf.expand_dims(1 / self.params["ranges"].tf_val, axis=1)
            x = _tf.matmul(x, vector)
        return x

    def forward(self, x):
        with _tf.name_scope("ProjectionTo1D_forward"):
            vector = self.params["ranges"].tf_val
            x = x * vector
        return x

    # def set_limits(self, data):
    #     self.params["range"].set_limits(
    #             min_val=data.diagonal / 100, max_val=data.diagonal * 10)


class NeuralNetwork(_Transform):
    """A neural network transform."""

    def __init__(self, n_dim, layers=[10]):
        """
        Initializer for NeuralNetwork.

        Parameters
        ----------
        n_dim : int
            Dimensionality of the input.
        layers : List[int]
            The hidden layers in the network. A list of ints whose size
            determines the number of layers and each value corresponds to
            the layer size.
        """
        super().__init__()
        self.n_dim = n_dim
        self.layers = layers
        self.center = _np.zeros([n_dim])
        self.scale = 1.0

        n_weights = 0
        n_bias = 0
        tmp = [n_dim] + layers
        for i in range(len(layers)):
            n_weights += tmp[i] * tmp[i + 1]
            n_bias += tmp[i + 1]
        self.params = {
            "weights": _gpr.Parameter(
                _np.random.normal(size=n_weights, scale=0.1),
                _np.ones(n_weights) * (-10),
                _np.ones(n_weights) * 10),
            "bias": _gpr.Parameter(
                _np.zeros(n_bias),
                _np.ones(n_bias) * (-10),
                _np.ones(n_bias) * 10)
        }
        self.bias = None
        self.weights = None

    def refresh(self):
        with _tf.name_scope("NeuralNetwork_refresh"):
            # weights
            weights_array = self.params["weights"].tf_val
            self.weights = []
            pos_1 = 0
            pos_2 = self.n_dim * self.layers[0]
            self.weights.append(_tf.reshape(
                weights_array[pos_1:pos_2],
                shape=[self.n_dim, self.layers[0]]))
            if len(self.layers) > 1:
                for i in range(1, len(self.layers)):
                    pos_1 = pos_2
                    pos_2 += self.layers[i] * self.layers[i - 1]
                    self.weights.append(_tf.reshape(
                        weights_array[pos_1:pos_2],
                        shape=[self.layers[i - 1], self.layers[i]]))

            # bias
            bias_array = self.params["bias"].tf_val
            self.bias = []
            pos_1 = 0
            pos_2 = self.layers[0]
            self.bias.append(bias_array[pos_1:pos_2])
            if len(self.layers) > 1:
                for i in range(1, len(self.layers)):
                    pos_1 = pos_2
                    pos_2 += self.layers[i]
                    self.bias.append(bias_array[pos_1:pos_2])

    def forward(self, x):
        raise Exception("forward transform not available for this class")

    def backward(self, x):
        with _tf.name_scope("NeuralNetwork_backward"):
            self.refresh()
            # scaling
            with _tf.name_scope("scaling"):
                center = _tf.expand_dims(_tf.constant(self.center, _tf.float64),
                                         axis=0)
                scale = _tf.constant(self.scale, _tf.float64)
                s = _tf.shape(x)
                center = _tf.tile(center, [s[0], 1])
                x_tr = (x - center) / scale
            # ReLU layers
            for i in range(len(self.weights)):
                with _tf.name_scope("dense_layer_" + str(i)):
                    x_tr = _tf.nn.xw_plus_b(x_tr, self.weights[i], self.bias[i])
                    x_tr = x_tr * _tf.nn.sigmoid(x_tr)  # swish activation
            return x_tr

    def set_limits(self, data):
        self.center = (data.bounding_box[1, :] + data.bounding_box[0, :]) / 2.0
        self.scale = data.diagonal


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
        self.params = {"ranges": _gpr.PositiveParameter(
            _np.ones(n_dim), _np.ones(n_dim)*0.001, _np.ones(n_dim)*10000)}

    def forward(self, x):
        with _tf.name_scope("Isotropic_forward"):
            ranges = self.params["ranges"].tf_val
            x_tr = _tf.matmul(x, _tf.diag(ranges))
        return x_tr

    def backward(self, x):
        with _tf.name_scope("Isotropic_backward"):
            ranges = self.params["ranges"].tf_val
            x_tr = _tf.matmul(x, _tf.diag(1 / ranges))
        return x_tr


class ChainedTransform(_Transform):
    def __init__(self, *transforms):
        super().__init__()
        count = -1
        for tr in transforms:
            count += 1
            names = list(tr.params.keys())
            names = [s + "_" + str(count) for s in names]
            self.params.update(zip(names, tr.params.values()))
        self.transforms = transforms

    def backward(self, x):
        for tr in self.transforms:
            x = tr.backward(x)
        return x

    def set_limits(self, data):
        for tr in self.transforms:
            tr.set_limits(data)
