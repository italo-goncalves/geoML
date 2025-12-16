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

__all__ = ["Identity",
           "Spline",
           "ZScore",
           "Softplus",
           "Log",
           "ChainedWarping",
           "Scale",
           "Sigmoid",
           "Center",
           "ContinuousNormalizingFlow",
           "CenteredLogRatio",
           "PCA",
           "RobustPCA"
           ]

import geoml.interpolation as _gint
import geoml.parameter as _gpr
import geoml.tftools as _tftools
import geoml.data as _data

import numpy as _np
import tensorflow as _tf

from sklearn.covariance import MinCovDet as _MCD
from sklearn.cluster import KMeans as _KMeans
from sklearn.decomposition import FastICA as _ICA


class _Warping(_gpr.Parametric):
    """
    Base warping class.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self._size_in = None
        self._size_out = None

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        return s

    @property
    def size_in(self):
        return self._size_in

    @property
    def size_out(self):
        return self._size_out
        
    def forward(self, x):
        """
        Passes values through the class's warping function.

        Parameters
        ----------
        x : array-like
            Vector with values to warp.

        Returns
        -------
        x : array-like
            Vector with warped values.
        log_det : array-like
            Log-derivative of warping function.
        """
        pass
    
    def backward(self, x):
        """
        Transforms values back to the original units.

        Parameters
        ----------
        x : array-like
            Vector with values to warp back to the original units.

        Returns
        -------
        x : array-like
            Vector with warped back values.
        """
        pass
    
    def initialize(self, x):
        """
        Uses the provided values to initialize the object's parameters.

        Parameters
        ----------
        x : array-like
            Vector with values to warp.

        Returns
        -------
        x : array-like
            Vector with warped values.
        """
        x, _ = self.forward(x)
        return x


class Identity(_Warping):
    """Identity warping."""
    def __init__(self, size):
        super().__init__()
        self._size_in = size
        self._size_out = size

    def forward(self, x):
        return x, _tf.reduce_sum(_tf.zeros_like(x), axis=1)
    
    def backward(self, x):
        return x


class Spline(_Warping):
    """
    Uses a monotonic spline to convert from original to warped space and
    back.

    The spline is assumed to work with normalized (z-score) values. It is
    centered at the origin and the arms span up to +/- 5 units. Its main
    use is to transform an asymmetric distribution to one closer to a Gaussian.

    Attributes
    ----------
    n_knots : int
        Total number of knots.
    """
    def __init__(self, size, knots_per_arm=5):
        """
        Initializer for Spline.

        Parameters
        ----------
        knots_per_arm : int
            The number of knots used to build each side (positive and negative)
            of the spline.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size
        self.n_knots = knots_per_arm * 2 + 1

        comp = _np.ones(knots_per_arm) / knots_per_arm
        for i in range(size):
            self._add_parameter(f"warped_partition_left_{i}",
                                _gpr.CompositionalParameter(comp))
            self._add_parameter(f"warped_partition_right_{i}",
                                _gpr.CompositionalParameter(comp))
        self.spline = _gint.MonotonicCubicSpline()
        x_original = _tf.constant(
            _np.linspace(-5, 5, knots_per_arm * 2 + 1)[:, None],
            _tf.float64
        )
        self.x_original = _tf.tile(x_original, [1, self.size_in])

    def _get_warped_coordinates(self, dim):
        warped_left = _tf.cumsum(
            self.parameters[f"warped_partition_left_{dim}"].get_value())
        warped_right = _tf.cumsum(
            self.parameters[f"warped_partition_right_{dim}"].get_value()) + 1.0
        warped_coordinates = _tf.concat(
            [_tf.constant([0.0], _tf.float64), warped_left, warped_right],
            axis=0) / 2
        warped_coordinates = 10 * warped_coordinates - 5
        return warped_coordinates
    
    def forward(self, x):
        warped_coordinates = _tf.stack(
            [self._get_warped_coordinates(i) for i in range(self.size_in)],
            axis=1
        )
        x_warp = self.spline.interpolate(self.x_original, warped_coordinates, x)
        xd = self.spline.interpolate_d1(self.x_original, warped_coordinates, x)
        log_det = _tf.reduce_sum(_tf.math.log(xd), axis=1)

        return x_warp, log_det
    
    def backward(self, x):
        warped_coordinates = _tf.stack(
            [self._get_warped_coordinates(i) for i in range(self.size_in)],
            axis=1
        )
        x_back = self.spline.interpolate(warped_coordinates, self.x_original, x)
        return x_back


class ZScore(_Warping):
    """
    A Warping that simply normalizes the values to z-scores.
    """
    def __init__(self, size, mean=None, std=None):
        """
        Initializer for ZScore.

        Parameters
        ----------
        mean : double
            The desired mean of the data.
        std : double
            The desired standard deviation of the data.

        The mean and standard deviation can be computed from the data
        (if omitted) or specified.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size

        self._add_parameter(
            "mean",
            _gpr.RealParameter(
                _np.zeros([size]),
                _np.zeros([size]) - 1e9,
                _np.zeros([size]) + 1e9
            )
        )
        if mean is not None:
            self.parameters["mean"].set_value(mean)
            # self.parameters["mean"].set_limits(mean - 2*_np.abs(mean),
            #                                    mean + 2*_np.abs(mean))

        self._add_parameter(
            "std",
            _gpr.PositiveParameter(
                _np.ones([size]),
                _np.ones([size]) * 1e-9,
                _np.ones([size]) * 1e9
            )
        )
        if std is not None:
            self.parameters["std"].set_value(std)
            self.parameters["std"].set_limits(std / 100, std * 10)
        
    def forward(self, x):
        mean = self.parameters["mean"].get_value()[None, :]
        std = self.parameters["std"].get_value()[None, :]
        x = (x - mean) / std
        log_det = _tf.zeros_like(x) - _tf.math.log(std)
        return x, _tf.reduce_sum(log_det, axis=1)
    
    def backward(self, x):
        mean = self.parameters["mean"].get_value()[None, :]
        std = self.parameters["std"].get_value()[None, :]
        # x = _tftools.ensure_rank_2(x)
        return x * std + mean

    def initialize(self, x):
        mean = _np.mean(x, axis=0)
        std = _np.std(x, axis=0)
        self.parameters["mean"].set_value(mean)
        self.parameters["std"].set_value(std)
        self.parameters["mean"].set_limits(mean - 3*std, mean + 3*std)
        self.parameters["std"].set_limits(std / 100, std * 10)
        return super().initialize(x)


class Center(ZScore):
    """
    A Warping that simply centers the data.
    """

    def __init__(self, size, mean=None):
        """
        Initializer for Center.

        The mean can be computed from the data (if omitted) or specified.

        Parameters
        ----------
        mean : double
            The desired mean of the data.
        """
        super().__init__(size, mean, std=_np.ones[size])
        self.parameters['std'].fix()

    def initialize(self, x):
        mean = _np.mean(x, axis=0)
        self.parameters["mean"].set_value(mean)
        return super().initialize(x)


class Softplus(_Warping):
    """
    Transforms the data using the inverse of the softplus function. 
    All the data must be positive.
    """
    def __init__(self, size, shift=1e-6):
        """
        Initializer for Softplus.

        Parameters
        ----------
        shift : float
            A positive value to add to the data. Use it if you have zeros.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size

        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    # computation only for x < 50.0 to avoid overflow
    def forward(self, x):
        # x = _tftools.ensure_rank_2(x)
        x_s = x + self.shift
        x_warp = _tf.where(_tf.greater(x_s, 50.0),
                           x_s,
                           _tf.math.log(_tf.math.expm1(x_s))
                           )
        # x_warp = _tf.where(_tf.math.is_nan(x), x, x_warp)

        log_det = _tf.where(_tf.greater(x_s, 50.0),
                            _tf.ones_like(x_s),
                            # 1 / (- _tf.math.expm1(-x_warp))
                            _tf.math.exp(x_s) / _tf.math.expm1(x_s)
                            )
        log_det = _tf.reduce_sum(_tf.math.log(log_det), axis=1)
        return x_warp, log_det
    
    def backward(self, x):
        # x = _tftools.ensure_rank_2(x)
        x_back = _tf.where(_tf.greater(x, 50.0),
                           x,
                           _tf.math.log1p(_tf.math.exp(x)))
        return x_back


class Log(_Warping):
    """
    Log-scale warping.

    Forward function: log
    Backward function: exp
    """
    def __init__(self, size, shift=1e-6):
        """
        Initializer for Log.

        Parameters
        ----------
        shift : float
            A positive value to add to the data. Use it if you have zeros.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size

        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    def forward(self, x):
        x_warp = _tf.math.log(x + self.shift)
        log_det = _tf.reduce_sum(1 / (x + self.shift), axis=1)
        return x_warp, log_det

    def backward(self, x):
        return _tf.math.exp(x)


class Scale(ZScore):
    """Linear scaling, assuming a mean of zero."""
    def __init__(self, size, scale=1):
        super().__init__(
            size,
            mean=_np.full([size], -1e-6),
            std=_np.full([size], scale)
        )
        self.parameters["mean"].fix()

    def initialize(self, x):
        sc = _np.max(x, axis=0) - _np.min(x, axis=0) + 1e-6
        self.parameters["std"].set_value(sc)
        x, _ = self.forward(x)
        return x

    def backward(self, x):
        std = self.parameters["std"].get_value()[None, :]
        return x * std


class ChainedWarping(_Warping):
    """
    Chains multiple Warping objects.
    """
    def __init__(self, *warpings):
        """

        Parameters
        ----------
        warpings : list
            List with Warping objects to apply in sequence.
        """
        super().__init__()
        self.warpings = list(warpings)
        for wp in warpings:
            self._register(wp)

        for i in range(len(self.warpings) - 1):
            size_out = self.warpings[i].size_out
            size_in = self.warpings[i + 1].size_in
            if size_out != size_in:
                raise ValueError(
                    f'Chained warping dimension mismatch at position {i}: {size_out} != {size_in}'
                )
        self._size_in = self.warpings[0].size_in
        self._size_out = self.warpings[-1].size_out

    def __repr__(self):
        s = "".join([repr(wp) for wp in self.warpings])
        return s

    def forward(self, x):
        d = _tf.reduce_sum(_tf.ones_like(x, dtype=_tf.float64), axis=1)
        for wp in self.warpings:
            x, log_d = wp.forward(x)
            d = d + log_d
        return x, d

    def backward(self, x):
        warping_rev = self.warpings.copy()
        warping_rev.reverse()
        for wp in warping_rev:
            x = wp.backward(x)
        return x

    def initialize(self, x):
        for wp in self.warpings:
            x = wp.initialize(x)
        return x


class Sigmoid(_Warping):
    """
    Sigmoid warping, for values constrained to the ]0, 1[ interval.

    Forward function: inverse sigmoid
    Backward function: sigmoid
    """

    def __init__(self, size, shift=1e-6):
        """
        Initializer for Sigmoid.

        Parameters
        ----------
        shift : float
            A positive value to ensure the data is constrained to the ]0, 1[ interval.
        """
        super().__init__()
        self._size_in = size
        self._size_out = size

        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = shift

    def forward(self, x):
        x = x * (1 - 2 * self.shift) + self.shift
        x_warp = - _tf.math.log(1 / x - 1)
        log_det = _tf.reduce_sum(- _tf.math.log(x - x**2), axis=1)
        return x_warp, log_det

    def backward(self, x):
        return 1 / (1 + _tf.math.exp(-x))


class ContinuousNormalizingFlow(_Warping):
    def __init__(self, size, inducing_points=20, n_steps=10, step=0.01):
        super().__init__()
        self._size_in = size
        self._size_out = size
        self.n_ip = inducing_points
        self.base_ip = None
        # self.ip_weight = None
        self.inducing_points = None
        self.n_steps = n_steps
        self.step = step

        self.alpha = None
        self.chol_space = None
        self.chol_time = None
        self.time = _tf.constant(_np.arange(self.n_steps)[:, None], _tf.float64)
        # self.mean = None
        # self.std = None

        self._add_parameter(
            'alpha_white',
            _gpr.RealParameter(
                _np.random.normal(scale=1e-3, size=[inducing_points, size, n_steps]),
                _np.full([inducing_points, size, n_steps], -10),
                _np.full([inducing_points, size, n_steps], 10)
            )
        )
        self._add_parameter(
            'amp', _gpr.PositiveParameter(1, 0.01, 100, fixed=False)
        )
        self._add_parameter(
            'rng_space', _gpr.PositiveParameter(
                _np.ones([n_steps]),  _np.ones([n_steps]) * 0.1, _np.ones([n_steps]) * 10
            )
        )

    def covariance_matrix_space(self, x_1, x_2, t):
        rng_space = self.parameters['rng_space'].get_value()[t]
        dist_space = _tftools.pairwise_dist(x_1, x_2) / rng_space
        cov_space = _tf.exp(- 3 * dist_space ** 2)
        return cov_space

    def covariance_matrix_space_d1(self, x_1, x_2, t):
        rng_space = self.parameters['rng_space'].get_value()[t]
        dif = - (x_1[:, None, :] - x_2[None, :, :]) / rng_space  # [data, data, size]
        cov_space = self.covariance_matrix_space(x_1, x_2, t)  # [data, data]
        cov_d1 = 6 * cov_space[:, :, None] * dif
        return cov_d1

    def refresh(self):
        alpha_white = self.parameters['alpha_white'].get_value()
        amp = self.parameters['amp'].get_value()
        inducing_points = _tf.constant(self.base_ip, _tf.float64)
        all_ip = []
        fields = []
        all_cov_inv = []

        for i in range(self.n_steps):
            cov_space = self.covariance_matrix_space(inducing_points, inducing_points, i)
            cov_space = cov_space + _tf.eye(self.n_ip, dtype=_tf.float64) * 1e-6
            chol_space = _tf.linalg.cholesky(cov_space)
            cov_space_inv = _tf.linalg.cholesky_solve(
                chol_space,
                _tf.eye(self.n_ip, dtype=_tf.float64)
            )
            field = _tf.matmul(chol_space, alpha_white[:, :, i]) * amp

            # Midpoint
            x_mid = inducing_points + self.step / 2 * field
            cov_space = self.covariance_matrix_space(x_mid, x_mid, i)
            cov_space = cov_space + _tf.eye(self.n_ip, dtype=_tf.float64) * 1e-6
            chol_space = _tf.linalg.cholesky(cov_space)
            field_mid = _tf.matmul(chol_space, alpha_white[:, :, i]) * amp

            all_ip.append(inducing_points)
            fields.append(field)
            all_cov_inv.append(cov_space_inv)

            inducing_points = inducing_points + self.step * field_mid
        self.inducing_points = _tf.stack(all_ip, axis=-1)
        fields = _tf.stack(fields, axis=-1)
        cov_space_inv = _tf.stack(all_cov_inv, axis=0)

        alpha = _tf.einsum('top,pst->ost', cov_space_inv, fields) / amp**2
        self.alpha = alpha

        # last_ip = self.inducing_points[:, :, -1]
        # self.mean = _tf.reduce_sum(self.ip_weight[:, None] * last_ip, axis=0, keepdims=True)
        # self.std = _tf.sqrt(_tf.reduce_sum(self.ip_weight[:, None] * (last_ip - self.mean)**2,
        #                                    axis=0, keepdims=True))

    def get_field(self, x, t):
        amp = self.parameters['amp'].get_value()
        cov_space = self.covariance_matrix_space(x, self.inducing_points[:, :, t], t)
        field = _tf.einsum('op,ps->os', cov_space, self.alpha[:, :, t])
        return field * amp**2

    def get_gradient(self, x, t):
        amp = self.parameters['amp'].get_value()
        cov_space = self.covariance_matrix_space_d1(x, self.inducing_points[:, :, t], t)
        grad = _tf.einsum('ops,ps->os', cov_space, self.alpha[:, :, t])
        return grad * amp**2

    # def forward(self, x):
    #     self.refresh()
    #     for i in range(self.n_steps):
    #         # Midpoint
    #         field = self.get_field(x, i)
    #         x_mid = x + self.step / 2 * field
    #         field_mid = self.get_field(x_mid, i)
    #         x = x + self.step * field_mid
    #     # x = (x - self.mean) / self.std
    #     return x

    def backward(self, x):
        self.refresh()
        # x = x * self.std + self.mean
        for i in range(self.n_steps):
            j = self.n_steps - 1 - i
            # Midpoint
            field = self.get_field(x, j)
            x_mid = x - self.step / 2 * field
            field_mid = self.get_field(x_mid, j)
            x = x - self.step * field_mid
        return x

    def forward(self, x):
        self.refresh()
        grads = []
        norm = []
        for i in range(self.n_steps):
            # Midpoint
            field = self.get_field(x, i)
            x_mid = x + self.step / 2 * field
            field_mid = self.get_field(x_mid, i)
            grad = self.get_gradient(x_mid, i)
            grads.append(_tf.reduce_sum(grad, axis=1, keepdims=True))
            norm.append(_tf.reduce_sum(grad**2, axis=1, keepdims=True))
            x = x + self.step * field_mid
        total_grad = _tf.add_n(grads) * self.step
        total_norm = _tf.add_n(norm) * self.step

        log_det = _tf.reduce_sum(total_grad - total_norm * 0.5, axis=1)
        return x, log_det

    def flow_history(self, x):
        self.refresh()
        history = [x]
        for i in range(self.n_steps):
            field = self.get_field(x, i)
            x_mid = x + self.step / 2 * field
            field_mid = self.get_field(x_mid, i)
            x = x + self.step * field_mid
            history.append(x.numpy())
        return history

    def initialize(self, x):
        cluster = _KMeans(self.n_ip).fit(x)

        cl_mean = _np.mean(cluster.cluster_centers_, axis=0, keepdims=True)

        self.base_ip = (cluster.cluster_centers_ - cl_mean) * 1.1 + cl_mean
        # self.ip_weight = _np.array([_np.sum(cluster.labels_ == i)
        #                             for i in range(self.n_ip)])
        # self.ip_weight = self.ip_weight / _np.sum(self.ip_weight)

        # x_min = _np.min(x, axis=0, keepdims=True)
        # x_max = _np.max(x, axis=0, keepdims=True)
        # self.base_ip = _np.random.uniform(x_min, x_max, size=[self.n_ip, self.size_out])

        x, _ = self.forward(x)
        return x


class PCA(_Warping):
    def __init__(self, n_dim, n_components=None):
        super().__init__()
        if n_components is None:
            n_components = n_dim
        self._size_in = n_dim
        self._size_out = n_components

        self.mean = None
        self.eigvals = None
        self.eigvecs = None

    def forward(self, x):
        x = x - self.mean
        x = _tf.matmul(x, self.eigvecs)
        x = x / _tf.sqrt(self.eigvals)
        return x, _tf.reduce_sum(_tf.zeros_like(x), axis=1)

    def backward(self, x):
        x = x * _tf.sqrt(self.eigvals)
        x = _tf.matmul(x, self.eigvecs, False, True)
        x = x + self.mean
        return x

    def initialize(self, x):
        self.mean = _tf.constant(_np.mean(x, axis=0, keepdims=True), _tf.float64)
        x_center = x - self.mean
        cov = _np.matmul(_np.transpose(x_center), x_center) / x_center.shape[0]
        vals, vecs = _np.linalg.eigh(cov)
        self.eigvals = _tf.constant(vals[::-1][None, :self.size_out], _tf.float64)
        self.eigvecs = _tf.constant(vecs[:, ::-1][:, :self.size_out], _tf.float64)

        x, _ = self.forward(x)
        return x


class RobustPCA(PCA):
    def __init__(self, n_dim, n_components=None, support_fraction=0.75):
        super().__init__(n_dim, n_components)
        self.support_fraction = support_fraction

    def initialize(self, x):
        mcd = _MCD(support_fraction=self.support_fraction).fit(x)
        self.mean = _tf.constant(mcd.location_[None, :], _tf.float64)
        cov = mcd.covariance_
        vals, vecs = _np.linalg.eigh(cov)
        self.eigvals = _tf.constant(vals[::-1][None, :self.size_out], _tf.float64)
        self.eigvecs = _tf.constant(vecs[:, ::-1][:, :self.size_out], _tf.float64)

        x, _ = self.forward(x)
        return x


class CenteredLogRatio(_Warping):
    def __init__(self, n_dim):
        super().__init__()
        self._size_in = n_dim
        self._size_out = n_dim

    def forward(self, x):
        # trick to convert dtype
        x = x + _tf.constant(0.0, _tf.float64)

        x_log = _tf.math.log(x)
        x_log = x_log - _tf.reduce_mean(x_log, axis=1, keepdims=True)
        log_det = _tf.reduce_sum(1 / x, axis=1) * 0.0
        return x_log, log_det

    def backward(self, x):
        return _tf.nn.softmax(x, axis=1)


class Rotation(Identity):
    def __init__(self, n_dim, fixed=False):
        super().__init__(n_dim)

        self._add_parameter(
            'rotation',
            _gpr.OrthonormalMatrix(n_dim, n_dim)
        )
        self.parameters['rotation'].set_value(_np.eye(n_dim))
        if fixed:
            self.parameters['rotation'].fix()

    def forward(self, x):
        rot = self.parameters['rotation'].get_value()
        x = _tf.matmul(x, rot)
        log_det = _tf.reduce_sum(_tf.zeros_like(x), axis=1)
        return x, log_det

    def backward(self, x):
        rot = self.parameters['rotation'].get_value()
        x = _tf.matmul(x, rot, False, True)
        return x

    def initialize(self, x):
        ica = _ICA(whiten=False).fit(x)
        self.parameters['rotation'].set_value(ica.components_)
        rot = self.parameters['rotation'].get_value()
        x = _tf.matmul(x, rot)
        return x
