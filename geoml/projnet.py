# geoML - machine learning models for geospatial data
# Copyright (C) 2024  Ítalo Gomes Gonçalves
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

import geoml.parameter as _gpr
import geoml.tftools as _tftools
import geoml.kernels as _kr
import geoml.transform as _tr
# import geoml.interpolation as _gint
import geoml.data as _data

import numpy as _np
import tensorflow as _tf


def uniform_directions(n, dim, n_iter=100):
    if dim < 2:
        raise ValueError("Invalid dim: must be 2 or greater.")
    if dim == 2:
        angles = _np.linspace(0, _np.pi, n + 1)[:-1]
        vectors = _np.stack([_np.cos(angles), _np.sin(angles)], axis=1)
        return vectors

    # vectors = _np.random.normal(size=[n, dim])
    # vectors /= _np.sqrt(_np.sum(vectors**2, axis=1, keepdims=True))
    #
    # for i in range(n_iter):
    #     dif = vectors[:, None, :] - vectors[None, :, :]
    #     dist_sq = _np.sum(dif**2, axis=2, keepdims=True) + 0.01
    #
    #     dif /= _np.sqrt(dist_sq)
    #     force = dim * _np.sum(dif / dist_sq, axis=1)
    #
    #     vectors += force
    #     vectors /= _np.sqrt(_np.sum(vectors**2, axis=1, keepdims=True))
    #
    # return vectors

    mask = _np.ones([n, n])
    mask = mask - _np.eye(n)
    mask = _tf.constant(mask, _tf.float64)

    vectors = _np.random.normal(size=[n, dim])
    vectors = _tf.Variable(vectors, dtype=_tf.float64)
    vectors.assign(vectors / _tf.math.reduce_euclidean_norm(vectors, axis=1, keepdims=True))

    def loss():
        dist = _tftools.pairwise_dist(vectors, vectors) + 1e-6
        return _tf.reduce_sum(mask / dist ** 2)

    opt = _tf.keras.optimizers.Adam(1e-2)

    for _ in range(n_iter):
        with _tf.GradientTape() as gt:
            l_i = loss()

        grads = gt.gradient(l_i, vectors)

        opt.apply_gradients(zip([grads], [vectors]))
        vectors.assign(vectors / _tf.math.reduce_euclidean_norm(vectors, axis=1, keepdims=True))

    return vectors.numpy()


class _ProjectedLatentVariable(_gpr.Parametric):
    def __init__(self):
        super().__init__()
        self._size = 0
        self.children = []
        self.root = None
        self.min = None
        self.max = None

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        return s

    @property
    def size(self):
        return self._size

    def set_parameter_limits(self, data):
        pass

    def refresh(self, jitter=1e-9):
        pass

    def get_unique_parents(self):
        raise NotImplementedError

    def predict(self, x, n_sim=1, seed=(0, 0)):
        raise NotImplementedError

    def predict_directions(self, x, dir_x, step=1e-3):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError


class _RootProjectedVariable(_ProjectedLatentVariable):
    def __init__(self):
        super().__init__()
        self.root = self

    def get_unique_parents(self):
        return []


class _FunctionalProjectedVariable(_ProjectedLatentVariable):
    def __init__(self, parent):
        super().__init__()
        self.parent = self._register(parent)
        parent.children.append(self)
        self.root = parent.root

    def get_unique_parents(self):
        return [self.parent] + self.parent.get_unique_parents()

    def set_parameter_limits(self, data):
        self.parent.set_parameter_limits(data)

    def refresh(self, jitter=1e-9):
        self.parent.refresh(jitter)


class _Operation(_ProjectedLatentVariable):
    def __init__(self, *latent_variables):
        super().__init__()
        self.parents = list(latent_variables)
        self.root = latent_variables[0].root
        for lat in latent_variables:
            self._register(lat)
            lat.children.append(self)

    def get_unique_parents(self):
        all_parents = self.parents.copy()
        for p in self.parents:
            all_parents.extend(p.get_unique_parents())
        return list(set(all_parents))

    def set_parameter_limits(self, data):
        for p in self.parents:
            p.set_parameter_limits(data)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class ProjectedInput(_RootProjectedVariable):
    def __init__(self, bounding_box, inducing_points=100):
                 # transform=_tr.Identity(), fix_transform=False):
        super().__init__()
        self._size = bounding_box.n_dim
        self.min = _tf.constant(_np.zeros([self.size]) - 5, _tf.float64)
        self.max = _tf.constant(_np.zeros([self.size]) + 5, _tf.float64)

        self.transform = self._register(_tr.NormalizeWithBoundingBox(bounding_box))
        # if fix_transform:
        #     for p in self.transform.all_parameters:
        #         p.fix()

        self.n_ip = inducing_points

    def refresh(self, jitter=1e-9):
        self.transform.refresh()

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def set_parameter_limits(self, data):
        self.transform.set_limits(data)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        x_tr = self.transform(x)
        return _tf.tile(x_tr[:, :, None], [1, 1, n_sim])


class Concatenate(_Operation):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        self._size = sum([p.size for p in self.parents])
        self.root = latent_variables[0].root
        self.min = _tf.concat([lat.min for lat in self.parents], axis=0)
        self.max = _tf.concat([lat.max for lat in self.parents], axis=0)

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        pred = [lat.predict(x, n_sim, seed) for lat in self.parents]
        return _tf.concat(pred, axis=1)


class BasicProjectedGP(_FunctionalProjectedVariable):
    def __init__(self, parent, size=1, kernel=_kr.Gaussian(),
                 fix_range=False, #isotropic=False,
                 n_directions=30):
        super().__init__(parent)

        self._size = size
        self.kernel = self._register(kernel)
        # self.isotropic = isotropic

        self.cov = None
        self.cov_inv = None
        self.cov_chol = None
        self.cov_smooth = None
        self.cov_smooth_chol = None
        self.cov_smooth_inv = None
        self.chol_r = None
        self.alpha = None

        self.angular_cov = None
        self.angular_chol = None
        self.angular_inv = None

        self.prior_cov = None
        self.prior_cov_inv = None
        self.prior_cov_chol = None

        self.projections = uniform_directions(n_directions, parent.size)
        self.n_directions = n_directions
        self.dot = _np.matmul(self.projections, self.projections.T)
        self.dot = _np.minimum(self.dot, 1.0)
        self.dot = _np.maximum(self.dot, -1.0)

        self.fix_range = fix_range
        self._set_parameters()

        self.min = _tf.constant(_np.zeros([self.size]) - 5, _tf.float64)
        self.max = _tf.constant(_np.zeros([self.size]) + 5, _tf.float64)
        self.inducing_points = None

    def _set_parameters(self):
        n_ip = self.root.n_ip
        self._add_parameter(
            "alpha_white",
            _gpr.RealParameter(
                _np.random.normal(
                    scale=1e-3,
                    size=[self.n_directions, self.size, n_ip, 1]
                ),
                _np.zeros([self.n_directions, self.size, n_ip, 1]) - 10,
                _np.zeros([self.n_directions, self.size, n_ip, 1]) + 10
            ))
        self._add_parameter(
            "delta",
            _gpr.PositiveParameter(
                _np.ones([self.n_directions, self.size, n_ip]),
                _np.ones([self.n_directions, self.size, n_ip]) * 1e-6,
                _np.ones([self.n_directions, self.size, n_ip]) * 1e2
            ))
            # _gpr.PositiveParameter(
            #     _np.ones([1, self.size, n_ip]),
            #     _np.ones([1, self.size, n_ip]) * 1e-6,
            #     _np.ones([1, self.size, n_ip]) * 1e2
            # ))
            # _gpr.PositiveParameter(
            #     _np.ones([1, self.size, 1]),
            #     _np.ones([1, self.size, 1]) * 1e-6,
            #     _np.ones([1, self.size, 1]) * 1e2
            # ))

        # if self.isotropic:
        #     self._add_parameter(
        #         "ranges",
        #         _gpr.PositiveParameter(
        #             _np.ones([1, 1, 1, 1]),
        #             _np.ones([1, 1, 1, 1]) * 1e-6,
        #             _np.ones([1, 1, 1, 1]) * 10,
        #             fixed=self.fix_range
        #         )
        #     )
        # else:
        #     self._add_parameter(
        #         "ranges",
        #         _gpr.PositiveParameter(
        #             _np.ones([1, 1, self.n_directions, 1]),
        #             _np.ones([1, 1, self.n_directions, 1]) * 1e-6,
        #             _np.ones([1, 1, self.n_directions, 1]) * 10,
        #             fixed=self.fix_range
        #         )
        #     )
        self._add_parameter(
            "range",
            _gpr.PositiveParameter(
                _np.ones([1, 1]),
                _np.ones([1, 1]) * 0.01,
                _np.ones([1, 1]) * 10,
                fixed=self.fix_range
            )
        )

        self._add_parameter(
            "angular_range",
            _gpr.PositiveParameter(
                _np.ones([1, 1]) * 0.1,
                _np.ones([1, 1]) * 0.01,
                _np.ones([1, 1]) * 10,
                fixed=self.fix_range
            )
        )

    def cross_covariance_matrix(self, x):
        with _tf.name_scope("projected_covariance_matrix"):
            # ranges = self.parameters["ranges"].get_value()

            # # # [n_data, n_data, n_dim, n_sims]
            # dif = x[:, None, :, :] - y[None, :, :, :]
            # dist = _tf.math.abs(dif) / ranges
            # cov = self.kernel.kernelize(dist)
            # cov = _tf.reduce_mean(cov, axis=3)
            # #
            # # # [n_dim, n_data, n_data]
            # cov = _tf.transpose(cov, [2, 0, 1])

            # [n_data, n_data, n_dim, n_sims, n_sims]
            # dif = x[:, None, :, :, None] - y[None, :, :, None, :]
            # dist = _tf.math.abs(dif) / ranges
            # cov = self.kernel.kernelize(dist)
            # cov = _tf.reduce_mean(cov, axis=3)
            # cov = _tf.reduce_mean(cov, axis=3)

            # [n_dim, n_data, n_data]
            # cov = _tf.transpose(cov, [2, 0, 1])

            range = self.parameters["range"].get_value()

            x = _tf.einsum('abc,db->adc', x, self.projections)  # [n_data, n_proj, n_sim]

            # # [n_data, n_ip, n_dim, n_sims]
            dif = x[:, None, :, 0, None] - self.inducing_points[None, :, None, None]
            # dif = x[:, None, :, :] - self.inducing_points[None, :, None, None]
            dist = _tf.math.abs(dif) / range
            cov = self.kernel.kernelize(dist)
            cov = _tf.reduce_mean(cov, axis=3)

            # # [n_dim, n_data, n_ip]
            cov = _tf.transpose(cov, [2, 0, 1])

            return cov

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("basic_refresh"):
            self.parent.refresh(jitter)

            # prior
            # ip = _tf.linspace(
            #     _tf.reduce_sum(self.parent.min[None, :] * self.projections, axis=1),
            #     _tf.reduce_sum(self.parent.max[None, :] * self.projections, axis=1),
            #     self.root.n_ip
            # )[:, :, None]
            ip_min = _tf.reduce_min(self.parent.min[None, :] * self.projections)
            ip_max = _tf.reduce_max(self.parent.max[None, :] * self.projections)
            ip = _tf.linspace(ip_min, ip_max, self.root.n_ip)
            self.inducing_points = ip

            range = self.parameters["range"].get_value()
            dist = _tf.math.abs(ip[:, None] - ip[None, :]) / range
            eye = _tf.eye(self.root.n_ip, dtype=_tf.float64)
            cov = self.kernel.kernelize(dist) + eye * jitter  # [n_ip, n_ip]

            # eye = _tf.eye(self.root.n_ip, dtype=_tf.float64)
            # eye = _tf.tile(eye[None, :, :], [self.n_directions, 1, 1])

            # cov = self.covariance_matrix(ip, ip) + eye * jitter
            chol = _tf.linalg.cholesky(cov)
            cov_inv = _tf.linalg.cholesky_solve(chol, eye)

            self.cov = cov
            self.cov_chol = chol
            self.cov_inv = cov_inv

            # posterior cov
            eye = _tf.tile(eye[None, None, :, :], [self.n_directions, self.size, 1, 1])
            delta = self.parameters["delta"].get_value()  # [n_proj, size, n_ip]
            # delta = _tf.tile(delta, [self.n_directions, 1, self.root.n_ip])
            delta_diag = _tf.linalg.diag(delta)
            self.cov_smooth = self.cov[None, None, :, :] + delta_diag
            self.cov_smooth_chol = _tf.linalg.cholesky(
                self.cov_smooth + eye * jitter)
            self.cov_smooth_inv = _tf.linalg.cholesky_solve(
                self.cov_smooth_chol, eye)
            self.chol_r = _tf.linalg.cholesky(
                self.cov_inv[None, None, :, :] - self.cov_smooth_inv + eye * jitter)  # [n_proj, size, n_ip, n_ip]

            # angular_cov
            ang_rng = self.parameters['angular_range'].get_value()
            ang_eye = _tf.eye(self.n_directions, dtype=_tf.float64)
            self.angular_cov = _tf.math.exp(- (1 - self.dot) / ang_rng)  # [n_proj, n_proj]
            self.angular_chol = _tf.linalg.cholesky(self.angular_cov + ang_eye * jitter)
            self.angular_inv = _tf.linalg.cholesky_solve(self.angular_chol, ang_eye)

            # posterior mean
            alpha_white = self.parameters["alpha_white"].get_value()  # [n_proj, size, n_ip, 1]
            pred_inputs = _tf.einsum("mp,dsmi->dspi", self.cov_chol, alpha_white)
            self.alpha = _tf.einsum("mp,dsmi->dspi", self.cov_inv, pred_inputs)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("basic_prediction"):
            x = self.parent.predict(x, n_sim, seed)

            # x = _tf.einsum('abc,db->adc', x, self.projections)

            # [n_dim, n_data, n_ip]
            cov_cross = self.cross_covariance_matrix(x)

            mu = _tf.einsum("dnm,dsmi->dnsi", cov_cross, self.alpha)  # [n_dim, n_data, size, 1]

            # explained_var = _tf.reduce_sum(
            #     _tf.einsum("dnm,dsmp->dnsp", cov_cross, self.cov_smooth_inv) * cov_cross[:, :, None, :],
            #     axis=-1
            # )  # [n_dim, n_data, size]
            # var = _tf.maximum(1.0 - explained_var, 0.0)
            # std = _tf.sqrt(var)
            #
            # avg_var = _tf.reduce_mean(var, axis=0)
            # avg_std = _tf.sqrt(avg_var)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_directions, self.size, self.root.n_ip, n_sim],
                seed=seed, dtype=_tf.float64
            )
            sims = _tf.einsum("dnm,dsmp,dspi->dnsi", cov_cross, self.chol_r, rnd) + mu
            # sims = _tf.einsum("dnm,dmp,dspi->dnsi", cov_cross, _tf.matmul(self.cov_inv, self.cov_chol), rnd) + mu

            # sims = _tf.einsum("dnm,dsmp,dspi->dnsi", cov_cross, self.chol_r, rnd)  # [n_dim, n_data, size, n_sim]
            # norm_sims = sims / std[:, :, :, None]
            # avg_sims = _tf.reduce_sum(norm_sims, axis=0) / _np.sqrt(self.n_directions) * avg_std[:, :, None]
            #
            # avg_mu = _tf.reduce_sum(mu, axis=0) / _np.sqrt(self.n_directions)
            #
            # return avg_sims + avg_mu

            return _tf.reduce_sum(sims, axis=0) / _np.sqrt(self.n_directions)

    def kl_divergence(self):
        with _tf.name_scope("basic_KL_divergence"):
            delta = self.parameters["delta"].get_value()
            # delta = _tf.tile(delta, [self.n_directions, 1, self.root.n_ip])
            alpha_white = self.parameters["alpha_white"].get_value()

            tr = _tf.reduce_sum(self.cov_smooth_inv * self.cov[None, None, :, :])
            fit = _tf.reduce_sum(alpha_white**2)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol)))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))

            kl = 0.5 * (- tr + fit + det_1 - det_2)

            return kl


class SelectInput(_FunctionalProjectedVariable):
    def __init__(self, parent, columns):
        super().__init__(parent)
        self.columns = _tf.constant(columns)
        self._size = len(columns)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        x = self.parent.predict(x, n_sim, seed)
        x = _tf.gather(x, self.columns, axis=1)
        return x


class Linear(_FunctionalProjectedVariable):
    def __init__(self, parent, size=1, unit_norm=True):
        super().__init__(parent)
        self._size = size

        if unit_norm:
            rnd = _np.random.normal(size=(parent.size, self.size))
            rnd = rnd / _np.sqrt(_np.sum(rnd ** 2, axis=0, keepdims=True))
            self._add_parameter(
                "weights",
                _gpr.UnitColumnNormParameter(
                    rnd, - _np.ones_like(rnd), _np.ones_like(rnd)
                )
            )
        else:
            rnd = _np.random.normal(size=(parent.size, self.size), scale=1e-4)
            self._add_parameter(
                "weights",
                _gpr.RealParameter(
                    _np.zeros([parent.size, self.size]) + rnd + 1/parent.size,
                    _np.zeros([parent.size, self.size]) - 1,
                    _np.zeros([parent.size, self.size]) + 1
                )
            )

        # binary classification
        if (parent.size == 1) & (self.size == 2):
            self.parameters["weights"].set_value([[1, -1]])
            self.parameters["weights"].fix()

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        weights = self.parameters["weights"].get_value()

        x = self.parent.predict(x, n_sim, seed)
        pred = _tf.einsum("axb,xy->ayb", x, weights)

        return pred


class Exponentiation(_FunctionalProjectedVariable):
    def __init__(self, parent):
        super().__init__(parent)
        # self._add_parameter("amp_mean", _gpr.RealParameter(0, -5, 5))
        # self._add_parameter(
        #     "amp_scale", _gpr.PositiveParameter(0.25, 0.01, 10))
        self._size = parent.size

        self.min = _tf.math.exp(self.parent.min)
        self.max = _tf.math.exp(self.parent.max)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("exponentiation_prediction"):
            # amp_mean = self.parameters["amp_mean"].get_value()
            # amp_scale = self.parameters["amp_scale"].get_value()

            x = self.parent.predict(x, n_sim, seed)

            return _tf.exp(x)


class Multiply(_Operation):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("all parents must have the same size")

        self._size = sizes[0]

        self.min = _tf.reduce_prod(_tf.stack([p.min for p in self.parents], axis=0), axis=0)
        self.max = _tf.reduce_prod(_tf.stack([p.max for p in self.parents], axis=0), axis=0)

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        all_pred = []

        for i, v in enumerate(self.parents):
            x_i = v.predict(x, n_sim, [seed[0] + i, seed[1]])
            all_pred.append(x_i)

        all_pred = _tf.stack(all_pred, axis=0)

        return _tf.reduce_prod(all_pred, axis=0)


class Add(_Operation):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("all parents must have the same size")

        self._size = sizes[0]

        self.min = _tf.reduce_sum(_tf.stack([p.min for p in self.parents], axis=0), axis=0)
        self.max = _tf.reduce_sum(_tf.stack([p.max for p in self.parents], axis=0), axis=0)

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)

    def predict(self, x, n_sim=1, seed=(0, 0)):
        all_pred = []

        for i, v in enumerate(self.parents):
            x_i = v.predict(x, n_sim, [seed[0] + i, seed[1]])
            all_pred.append(x_i)

        all_pred = _tf.stack(all_pred, axis=0)

        return _tf.reduce_sum(all_pred, axis=0)


class LinearCombination(_Operation):
    """
    Linear combination.

    This node combines the inputs linearly with positive weights.
    """
    def __init__(self, *latent_variables, unit_variance=True):
        """
        Initializer for LinearCombination.

        Parameters
        ----------
        latent_variables
            Nodes to combine. They must all have the same number of variables.
        unit_variance : bool
            If `True`, constrains the weights to unit sum to control the variance of the output.
        """
        super().__init__(*latent_variables)
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise SizeIncompatibilityError(
                f"All parents must have the same size. Found {sizes}."
            )

        self._size = sizes[0]

        if unit_variance:
            self._add_parameter(
                "weights",
                _gpr.CompositionalParameter(
                    _np.ones(len(latent_variables)) / len(latent_variables))
            )
        else:
            self._add_parameter(
                "weights",
                _gpr.PositiveParameter(
                    _np.ones(len(latent_variables)) / len(latent_variables),
                    _np.ones(len(latent_variables)) * 0.01,
                    _np.ones(len(latent_variables)) * 100
                )
            )

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        weights = self.parameters["weights"].get_value()

        all_pred = []

        for i, v in enumerate(self.parents):
            x_i = v.predict(x, n_sim, [seed[0] + i, seed[1]])
            all_pred.append(x_i)

        all_pred = _tf.stack(all_pred, axis=-1)

        all_pred = _tf.reduce_sum(all_pred * weights, axis=-1)

        return all_pred

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class _VariationalFourierFeatures(_FunctionalProjectedVariable):
    def __init__(self, parent, size=1, fix_range=False, isotropic=False, n_directions=30):
        super().__init__(parent)
        self._size = size
        self.isotropic = isotropic
        self.n_columns = None

        self.diag = None
        self.mat_b = None
        self.diag_posterior = None
        self.chol = None
        self.diag_b = None
        self.diag_b_post = None
        self.chol_r = None

        self.fix_range = fix_range
        self.projections = uniform_directions(n_directions, parent.size)
        self.n_directions = n_directions
        self._set_parameters()

        self.min = _tf.constant(_np.zeros([self.size]) - 5, _tf.float64)
        self.max = _tf.constant(_np.zeros([self.size]) + 5, _tf.float64)

        self.harm_cos = _tf.range(self.root.n_ip + 1, dtype=_tf.float64)
        self.harm_sin = _tf.range(self.root.n_ip, dtype=_tf.float64) + 1

        pmax = _tf.reduce_sum(self.parent.max[None, :] * self.projections, axis=1)
        pmin = _tf.reduce_sum(self.parent.min[None, :] * self.projections, axis=1)
        pmax, pmin = _tf.maximum(pmax, pmin), _tf.minimum(pmax, pmin)
        dif = pmax - pmin
        self.lower = pmin - dif / 2
        self.upper = pmax + dif / 2

        # [n_ip, n_dim]
        self.freq_cos = 2 * _np.pi * self.harm_cos[:, None] / (self.upper[None, :] - self.lower[None, :])
        self.freq_sin = 2 * _np.pi * self.harm_sin[:, None] / (self.upper[None, :] - self.lower[None, :])

    def _set_parameters(self):
        n_ip = 2 * self.root.n_ip + 1
        self._add_parameter(
            "mean",
            _gpr.RealParameter(
                _np.random.normal(
                    scale=1e-3,
                    size=[self.n_directions, self.size, n_ip, 1]
                ),
                _np.zeros([self.n_directions, self.size, n_ip, 1]) - 100,
                _np.zeros([self.n_directions, self.size, n_ip, 1]) + 100
            ))
        self._add_parameter(
            "delta",
            _gpr.PositiveParameter(
                _np.ones([self.n_directions, self.size, n_ip]) * 1e3,
                _np.ones([self.n_directions, self.size, n_ip]) * 1e-6,
                _np.ones([self.n_directions, self.size, n_ip]) * 1e12
            ))

        if self.isotropic:
            self._add_parameter(
                "ranges",
                _gpr.PositiveParameter(
                    _np.ones([1, 1]),
                    _np.ones([1, 1]) * 1e-6,
                    _np.ones([1, 1]) * 10,
                    fixed=self.fix_range
                )
            )
        else:
            self._add_parameter(
                "ranges",
                _gpr.PositiveParameter(
                    _np.ones([1, self.n_directions]),
                    _np.ones([1, self.n_directions]) * 1e-6,
                    _np.ones([1, self.n_directions]) * 10,
                    fixed=self.fix_range
                )
            )

    def fourier_features(self, x):
        features = _tf.concat([
            _tf.cos(self.freq_cos[None, :, :, None] * (x[:, None, :, :] - self.lower[None, None, :, None])),
            _tf.sin(self.freq_sin[None, :, :, None] * (x[:, None, :, :] - self.lower[None, None, :, None]))
        ], axis=1)  # [n_data, n_ip, n_dim, n_sims]
        features = _tf.reduce_mean(features, axis=3)
        return _tf.transpose(features, [2, 0, 1])

    def spectrum(self, frequencies):
        raise NotImplementedError

    def refresh(self, jitter=1e-9):
        raise NotImplementedError

    def predict(self, x, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("VFF_prediction"):
            x = self.parent.predict(x, n_sim, seed)

            x = _tf.einsum('abc,db->adc', x, self.projections)

            # [n_dim, n_data, n_ip]
            cov_cross = self.fourier_features(x)

            mean = self.parameters['mean'].get_value()
            prod = _tf.matmul(self.diag_b, self.chol)
            m_1 = _tf.matmul(prod, mean, True)
            m_1 = _tf.matmul(prod, m_1)
            m_2 = mean / self.diag[:, :, :, None]
            mu = _tf.einsum("dnm,dsmi->dnsi", cov_cross, m_2 - m_1)

            # mean = self.parameters['mean'].get_value()
            # k_12 = _tf.concat([
            #     _tf.linalg.diag(_tf.sqrt(self.diag)),
            #     _tf.tile(self.mat_b, [self.parent.size, 1, 1, 1])
            # ], axis=3)
            # prod = _tf.matmul(k_12, mean)
            # m_1 = _tf.matmul(self.diag_b, self.chol)
            # m_1 = _tf.matmul(m_1, prod, True)
            # m_1 = _tf.matmul(m_1, m_1)
            # m_2 = prod / self.diag[:, :, :, None]
            # mu = _tf.einsum("dnm,dsmi->dnsi", cov_cross, m_2 - m_1)

            n_ip = 2 * self.root.n_ip + 1
            rnd_1 = _tf.random.stateless_normal(
                shape=[self.n_directions, self.size, n_ip, n_sim],
                seed=[seed[0], seed[1] + 1], dtype=_tf.float64
            )
            rnd_2 = _tf.random.stateless_normal(
                shape=[self.n_directions, self.size, self.n_columns, n_sim],
                seed=[seed[0], seed[1] + 2], dtype=_tf.float64
            )
            rnd_3 = _tf.random.stateless_normal(
                shape=[self.n_directions, self.size, self.n_columns, n_sim],
                seed=[seed[0], seed[1] + 3], dtype=_tf.float64
            )

            sims_1 = _tf.sqrt(1/self.diag - 1/self.diag_posterior)[:, :, :, None] * rnd_1
            sims_2 = - _tf.einsum("dsmc,dsci->dsmi", _tf.matmul(self.diag_b, self.chol), rnd_2)
            sims_3 = _tf.einsum("dsmc,dsci->dsmi", _tf.matmul(self.diag_b_post, self.chol), rnd_3)
            sims = _tf.einsum("dnm,dsmi->dnsi", cov_cross, sims_1 + sims_2 + sims_3) + mu

            # sims = _tf.einsum("dnm,dsmi->dnsi", cov_cross, _tf.matmul(self.chol_r, rnd_1)) + mu

            return _tf.reduce_sum(sims, axis=0) / _np.sqrt(self.n_directions)

    def kl_divergence(self):
        mean = self.parameters['mean'].get_value()
        prod_1 = _tf.matmul(self.diag_b, self.chol)
        prod_2 = _tf.matmul(prod_1, mean, True)
        prod_3 = mean**2 / self.diag[:, :, :, None]
        fit = _tf.reduce_sum(prod_3) - _tf.reduce_sum(prod_2**2)
        # fit = _tf.reduce_sum(mean**2)

        det_1 = - _tf.reduce_sum(_tf.math.log(self.parameters['delta'].get_value()))
        det_2 = _tf.reduce_sum(_tf.math.log(self.diag_posterior))

        eye = _tf.eye(self.n_columns, dtype=_tf.float64, batch_shape=[self.n_directions, self.size])
        prod_4 = _tf.matmul(self.mat_b, self.diag_b_post, True)
        chol = _tf.linalg.cholesky(eye + prod_4)
        det_3 = 2 * _tf.reduce_sum(_tf.math.log(_tf.linalg.diag_part(chol)))

        tr_1 = - _tf.reduce_sum(self.diag / self.diag_posterior)
        tr_2 = - _tf.reduce_sum(_tf.linalg.diag_part(prod_4))

        prod_5 = _tf.sqrt(self.diag[:, :, :, None]) * _tf.matmul(self.diag_b_post, self.chol)
        tr_3 = _tf.reduce_sum(prod_5**2)

        prod_6 = _tf.matmul(prod_4, self.chol)
        tr_4 = _tf.reduce_sum(prod_6**2)

        kl = 0.5 * (fit + det_1 + det_2 + det_3 + tr_1 + tr_2 + tr_3 + tr_4)
        return kl


class VFFMatern0(_VariationalFourierFeatures):
    def __init__(self, parent, size=1, fix_range=False, isotropic=False, n_directions=30):
        super().__init__(parent, size, fix_range, isotropic, n_directions)
        self.n_columns = 1

    def spectrum(self, frequencies):
        lbd = 1 / self.parameters['ranges'].get_value()
        return 2 * lbd / (lbd**2 + frequencies**2)

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("VFF_refresh"):
            self.parent.refresh(jitter)

            spectrum_cos = 0.5 * (self.upper[None, :] - self.lower[None, :]) / self.spectrum(self.freq_cos)
            w = _np.ones([self.root.n_ip + 1, 1])
            w[0] *= 2
            spectrum_cos = spectrum_cos * w
            spectrum_sin = 0.5 * (self.upper[None, :] - self.lower[None, :]) / self.spectrum(self.freq_sin)

            # [n_dim, size, n_ip]
            self.diag = _tf.transpose(_tf.concat([spectrum_cos, spectrum_sin], axis=0))[:, None, :]
            self.diag_posterior = self.diag + self.parameters['delta'].get_value()

            beta = _tf.constant(_np.concatenate([
                _np.ones([self.root.n_ip + 1]), _np.zeros([self.root.n_ip])
            ]), _tf.float64)
            self.mat_b = beta[None, None, :, None]  # [n_dim, size, n_ip, n_col]
            self.diag_b = self.mat_b / self.diag[:, :, :, None]
            self.diag_b_post = self.mat_b / self.diag_posterior[:, :, :, None]

            eye = _tf.eye(self.n_columns, dtype=_tf.float64, batch_shape=[self.n_directions, self.size])
            mat = eye + _tf.matmul(self.mat_b, self.mat_b, True)
            self.chol = _tf.linalg.cholesky(_tf.linalg.solve(mat, eye))  # [n_dim, size, n_col, n_col]

            # prod_1 = _tf.matmul(self.diag_b, self.chol)
            # prod_2 = _tf.matmul(self.diag_b_post, self.chol)
            # mat_r = _tf.linalg.diag(1 / self.diag) \
            #     - _tf.matmul(prod_1, prod_1, False, True) \
            #     - _tf.linalg.diag(1 / self.diag_posterior) \
            #     + _tf.matmul(prod_2, prod_2, False, True)
            # self.chol_r = _tf.linalg.cholesky(mat_r)


class VFFMatern1(_VariationalFourierFeatures):
    def __init__(self, parent, size=1, fix_range=False, isotropic=False, n_directions=30):
        super().__init__(parent, size, fix_range, isotropic, n_directions)
        self.n_columns = 2

    def spectrum(self, frequencies):
        lbd = _np.sqrt(3) / self.parameters['ranges'].get_value()
        return 4 * lbd**3 / (lbd**2 + frequencies**2)**2

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("VFF_refresh"):
            self.parent.refresh(jitter)

            spectrum_cos = 0.5 * (self.upper[None, :] - self.lower[None, :]) / self.spectrum(self.freq_cos)
            w = _np.ones([self.root.n_ip + 1, 1])
            w[0] *= 2
            spectrum_cos = spectrum_cos * w
            spectrum_sin = 0.5 * (self.upper[None, :] - self.lower[None, :]) / self.spectrum(self.freq_sin)

            # [n_dim, size, n_ip]
            self.diag = _tf.transpose(_tf.concat([spectrum_cos, spectrum_sin], axis=0))[:, None, :]
            self.diag_posterior = self.diag + self.parameters['delta'].get_value()

            beta_1 = _tf.constant(_np.concatenate([
                _np.ones([self.root.n_ip + 1]), _np.zeros([self.root.n_ip])
            ]), _tf.float64)
            beta_1 = _tf.tile(beta_1[None, None, :, None], [self.n_directions, self.size, 1, 1])

            lbd = _tf.transpose(_np.sqrt(3) / self.parameters['ranges'].get_value())
            beta_2 = _tf.concat([
                _tf.zeros_like(beta_1)[:, :, :(self.root.n_ip + 1), :],
                _tf.tile(_tf.transpose(self.freq_sin)[:, None, :, None] / lbd[:, :, None, None], [1, self.size, 1, 1])
            ], axis=2)

            self.mat_b = _tf.concat([beta_1, beta_2], axis=3)  # [n_dim, size, n_ip, n_col]
            self.diag_b = self.mat_b / self.diag[:, :, :, None]
            self.diag_b_post = self.mat_b / self.diag_posterior[:, :, :, None]

            eye = _tf.eye(self.n_columns, dtype=_tf.float64, batch_shape=[self.n_directions, self.size])
            mat = eye + _tf.matmul(self.mat_b, self.mat_b, True)
            self.chol = _tf.linalg.cholesky(_tf.linalg.solve(mat, eye))  # [n_dim, size, n_col, n_col]


class VFFMatern2(_VariationalFourierFeatures):
    def __init__(self, parent, size=1, fix_range=False, isotropic=False, n_directions=30):
        super().__init__(parent, size, fix_range, isotropic, n_directions)
        self.n_columns = 3

    def spectrum(self, frequencies):
        lbd = _np.sqrt(5) / self.parameters['ranges'].get_value()
        return 16/3 * lbd**5 / (lbd**2 + frequencies**2)**3

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("VFF_refresh"):
            self.parent.refresh(jitter)

            spectrum_cos = 0.5 * (self.upper[None, :] - self.lower[None, :]) / self.spectrum(self.freq_cos)
            w = _np.ones([self.root.n_ip + 1, 1])
            w[0] *= 2
            spectrum_cos = spectrum_cos * w
            spectrum_sin = 0.5 * (self.upper[None, :] - self.lower[None, :]) / self.spectrum(self.freq_sin)

            # [n_dim, size, n_ip]
            self.diag = _tf.transpose(_tf.concat([spectrum_cos, spectrum_sin], axis=0))[:, None, :]
            self.diag_posterior = self.diag + self.parameters['delta'].get_value()

            beta_1 = _tf.constant(_np.concatenate([
                _np.ones([self.root.n_ip + 1]), _np.zeros([self.root.n_ip])
            ]), _tf.float64)
            beta_1 = _tf.tile(beta_1[None, None, :, None], [self.n_directions, self.size, 1, 1])

            lbd = _tf.transpose(_np.sqrt(5) / self.parameters['ranges'].get_value())
            beta_2 = _tf.concat([
                3 * _tf.tile(_tf.transpose(self.freq_cos)[:, None, :, None] / lbd[:, :, None, None],
                             [1, self.size, 1, 1])**2 - 1,
                _tf.zeros_like(beta_1)[:, :, :self.root.n_ip, :]
            ], axis=2)

            beta_3 = _np.sqrt(3) * _tf.concat([
                _tf.zeros_like(beta_1)[:, :, :(self.root.n_ip + 1), :],
                _tf.tile(_tf.transpose(self.freq_sin)[:, None, :, None] / lbd[:, :, None, None], [1, self.size, 1, 1])
            ], axis=2)

            self.mat_b = _tf.concat([beta_1, beta_2, beta_3], axis=3)  # [n_dim, size, n_ip, n_col]
            self.diag_b = self.mat_b / self.diag[:, :, :, None]
            self.diag_b_post = self.mat_b / self.diag_posterior[:, :, :, None]

            eye = _tf.eye(self.n_columns, dtype=_tf.float64, batch_shape=[self.n_directions, self.size])
            mat = eye + _tf.matmul(self.mat_b, self.mat_b, True)
            self.chol = _tf.linalg.cholesky(_tf.linalg.solve(mat, eye))  # [n_dim, size, n_col, n_col]