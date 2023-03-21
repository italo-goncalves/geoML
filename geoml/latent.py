# geoML - machine learning models for geospatial data
# Copyright (C) 2021  Ítalo Gomes Gonçalves
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
import geoml.interpolation as _gint
import geoml.data as _data

import numpy as _np
import tensorflow as _tf


class _LatentVariable(_gpr.Parametric):
    def __init__(self):
        super().__init__()
        self._size = 0
        self.children = []
        self.root = None
        self.inducing_points = None
        self.inducing_points_variance = None
        # self._is_deterministic = []

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        return s

    @property
    def size(self):
        return self._size

    # @property
    # def is_deterministic(self):
    #     return self._is_deterministic

    def set_parameter_limits(self, data):
        pass

    def refresh(self, jitter=1e-9):
        pass

    def get_unique_parents(self):
        raise NotImplementedError

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        raise NotImplementedError

    def predict_directions(self, x, dir_x, step=1e-3):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError

    def propagate(self, x, x_var=None):
        raise NotImplementedError

    @staticmethod
    def add_offset(x):
        ones = _tf.ones([_tf.shape(x)[0], 1], _tf.float64)
        return _tf.concat([ones, x], axis=1)

    @staticmethod
    def add_offset_grad(x):
        zeros = _tf.zeros([_tf.shape(x)[0], 1], _tf.float64)
        return _tf.concat([zeros, x], axis=1)


class _RootLatentVariable(_LatentVariable):
    def __init__(self):
        super().__init__()
        self.root = self

    def get_unique_parents(self):
        return []

    def get_root_inducing_points(self):
        return NotImplementedError


class _FunctionalLatentVariable(_LatentVariable):
    def __init__(self, parent):
        super().__init__()
        self.parent = self._register(parent)
        parent.children.append(self)
        self.root = parent.root

    def get_unique_parents(self):
        return [self.parent] + self.parent.get_unique_parents()

    def propagate(self, x, x_var=None):
        mu, var = self.predict(x, x_var, n_sim=0)
        mu = _tf.transpose(mu[:, :, 0])
        var = _tf.transpose(var)
        return mu, var

    def set_parameter_limits(self, data):
        self.parent.set_parameter_limits(data)


class _Operation(_LatentVariable):
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


class BasicInput(_RootLatentVariable):
    def __init__(self, inducing_points, transform=_tr.Identity(),
                 fix_inducing_points=True, fix_transform=False,
                 center=False):
        super().__init__()
        test_point = _np.ones([1, inducing_points.n_dim])
        test_point = transform(test_point)
        self._size = test_point.shape[1]
        self.bounding_box = inducing_points.bounding_box

        self.transform = self._register(transform)
        if fix_transform:
            for p in self.transform.all_parameters:
                p.fix()
        self.transform.set_limits(inducing_points)

        self.n_ip = inducing_points.coordinates.shape[0]
        self._add_parameter(
            "inducing_points",
            _gpr.RealParameter(
                inducing_points.coordinates,
                _np.tile(self.bounding_box.min, [self.n_ip, 1]),
                _np.tile(self.bounding_box.max, [self.n_ip, 1]),
                fixed=fix_inducing_points
            ))

        self.inducing_points_variance = _tf.zeros(
            [self.n_ip, self.size], _tf.float64)

        self.center = _np.zeros_like(transform(self.bounding_box.max))
        if center:
            self.center = 0.5 * (transform(self.bounding_box.min)
                                 + transform(self.bounding_box.max))

    def get_root_inducing_points(self):
        ip = self.parameters["inducing_points"].get_value()
        return ip, _tf.zeros_like(ip)

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("basic_input_refresh"):
            self.transform.refresh()
            self.inducing_points = self.transform(
                self.parameters["inducing_points"].get_value()) - self.center

    def propagate(self, x, x_var=None):
        x_tr = self.transform(x) - self.center
        return x_tr, _tf.zeros_like(x_tr)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def set_parameter_limits(self, data):
        self.transform.set_limits(data)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        x_tr = _tf.transpose(self.transform(x) - self.center)
        var = _tf.zeros_like(x_tr)
        if n_sim > 0:
            sims = _tf.tile(x_tr[:, :, None], [1, 1, n_sim])
            return x_tr[:, :, None], var, sims, \
                   _tf.zeros_like(var), _tf.zeros_like(var)
        else:
            return x_tr[:, :, None], var

    def predict_directions(self, x, dir_x, step=1e-3):
        x_plus = self.transform(x + dir_x*step/2) - self.center
        x_minus = self.transform(x - dir_x * step / 2) - self.center

        mu = _tf.transpose((x_plus - x_minus) / step)
        return mu[:, :, None], _tf.zeros_like(mu), _tf.zeros_like(mu)


class Concatenate(_Operation):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        self._size = sum([p.size for p in self.parents])
        self.root = latent_variables[0].root

    def propagate(self, x, x_var=None):
        means, variances = [], []
        for lat in self.parents:
            m, v = lat.propagate(x, x_var)
            means.append(m)
            variances.append(v)

        mean = _tf.concat(means, axis=1)
        var = _tf.concat(variances, axis=1)
        return mean, var

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)
        if all([lat.inducing_points is not None for lat in self.parents]):
            self.inducing_points = _tf.concat(
                [lat.inducing_points for lat in self.parents],
                axis=1)
            self.inducing_points_variance = _tf.concat(
                [lat.inducing_points_variance for lat in self.parents],
                axis=1)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        if n_sim > 0:
            means, variances, sims, exp_vars, influences = [], [], [], [], []
            for lat in self.parents:
                m, v, s, ev, inf = lat.predict(x, x_var, n_sim, seed)
                means.append(m)
                variances.append(v)
                sims.append(s)
                exp_vars.append(ev)
                influences.append(inf)

            mean = _tf.concat(means, axis=0)
            var = _tf.concat(variances, axis=0)
            sims = _tf.concat(sims, axis=0)
            exp_var = _tf.concat(exp_vars, axis=0)
            influence = _tf.concat(influences, axis=0)

            return mean, var, sims, exp_var, influence
        else:
            means, variances = [], []
            for lat in self.parents:
                m, v = lat.predict(x, x_var, n_sim=0)
                means.append(m)
                variances.append(v)

            mean = _tf.concat(means, axis=0)
            var = _tf.concat(variances, axis=0)
            return mean, var


class BasicGP(_FunctionalLatentVariable):
    def __init__(self, parent, size=1, kernel=_kr.Gaussian(),
                 fix_range=False):
        super().__init__(parent)
        self._size = size
        self.kernel = self._register(kernel)

        self.cov = None
        self.cov_inv = None
        self.cov_chol = None
        self.cov_smooth = None
        self.cov_smooth_chol = None
        self.cov_smooth_inv = None
        self.chol_r = None
        self.alpha = None

        self.prior_cov = None
        self.prior_cov_inv = None
        self.prior_cov_chol = None

        n_ip = self.root.n_ip
        self._add_parameter(
            "alpha_white",
            _gpr.RealParameter(
                _np.random.normal(
                    scale=1e-3,
                    size=[self.size, n_ip, 1]
                ),
                # _np.ones([self.size, n_ip, 1])*0.1,
                _np.zeros([self.size, n_ip, 1]) - 10,
                _np.zeros([self.size, n_ip, 1]) + 10
            ))
        self._add_parameter(
            "delta",
            _gpr.PositiveParameter(
                _np.ones([self.size, n_ip]),
                _np.ones([self.size, n_ip]) * 1e-6,
                _np.ones([self.size, n_ip]) * 1e2
            ))

        self._add_parameter(
            "ranges",
            _gpr.PositiveParameter(
                _np.ones([1, 1, self.parent.size]),
                _np.ones([1, 1, self.parent.size]) * 1e-6,
                _np.ones([1, 1, self.parent.size]) * 10,
                fixed=fix_range
            )
        )

    def covariance_matrix(self, x, y, var_x=None, var_y=None):
        with _tf.name_scope("basic_covariance_matrix"):
            ranges = self.parameters["ranges"].get_value()
            if var_x is None:
                var_x = _tf.zeros_like(x)
            if var_y is None:
                var_y = _tf.zeros_like(y)
            var_x = var_x[:, None, :]
            var_y = var_y[None, :, :]

            # [n_data, n_data, n_dim]
            dif = x[:, None, :] - y[None, :, :]

            total_var = ranges**2 + (var_x + var_y) / 2
            dist = _tf.sqrt(_tf.reduce_sum(dif ** 2 / total_var, axis=-1))
            cov = self.kernel.kernelize(dist)

            # normalization
            det_x = _tf.reduce_prod(var_x + ranges**2, axis=-1) ** (1 / 4)
            det_y = _tf.reduce_prod(var_y + ranges**2, axis=-1) ** (1 / 4)
            det_2 = _tf.sqrt(_tf.reduce_prod(total_var, axis=-1))

            norm = det_x * det_y / det_2
            # norm = _tf.reduce_prod(ranges) / det_2

            # output
            cov = cov * norm
            return cov

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("basic_refresh"):
            self.parent.refresh(jitter)

            # prior
            ip = self.parent.inducing_points
            ip_var = self.parent.inducing_points_variance

            eye = _tf.eye(self.root.n_ip, dtype=_tf.float64)

            cov = self.covariance_matrix(ip, ip, ip_var, ip_var) + eye * jitter
            chol = _tf.linalg.cholesky(cov)
            cov_inv = _tf.linalg.cholesky_solve(chol, eye)

            self.cov = cov  # _tf.tile(cov[None, :, :], [self.size, 1, 1])
            self.cov_chol = chol  # = _tf.tile(chol[None, :, :], [self.size, 1, 1])
            self.cov_inv = cov_inv  #_tf.tile(cov_inv[None, :, :], [self.size, 1, 1])

            # posterior
            eye = _tf.tile(eye[None, :, :], [self.size, 1, 1])
            delta = self.parameters["delta"].get_value()
            delta_diag = _tf.linalg.diag(delta)
            self.cov_smooth = self.cov[None, :, :] + delta_diag
            self.cov_smooth_chol = _tf.linalg.cholesky(
                self.cov_smooth + eye * jitter)
            self.cov_smooth_inv = _tf.linalg.cholesky_solve(
                self.cov_smooth_chol, eye)
            self.chol_r = _tf.linalg.cholesky(
                self.cov_inv[None, :, :] - self.cov_smooth_inv + eye * jitter)

            # inducing points
            alpha_white = self.parameters["alpha_white"].get_value()
            # pred_inputs = _tf.matmul(self.cov_chol, alpha_white)
            pred_inputs = _tf.einsum("ab,sbc->sac", self.cov_chol, alpha_white)
            self.inducing_points = _tf.transpose(pred_inputs[:, :, 0])
            # self.alpha = _tf.matmul(self.cov_inv, pred_inputs)
            self.alpha = _tf.einsum("ab,sbc->sac", self.cov_inv, pred_inputs)
            # pred_var = 1.0 - _tf.reduce_sum(
            #         _tf.matmul(self.cov, self.cov_smooth_inv) * self.cov,
            #         axis=2, keepdims=False
            #     )
            pred_var = 1.0 - _tf.reduce_sum(
                _tf.einsum("ab,sbc->sac", self.cov, self.cov_smooth_inv)
                * self.cov[None, :, :],
                axis=2, keepdims=False
            )
            self.inducing_points_variance = _tf.transpose(pred_var)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("basic_prediction"):
            x, x_var = self.parent.propagate(x, x_var)

            cov_cross = self.covariance_matrix(
                x, self.parent.inducing_points,
                x_var, self.parent.inducing_points_variance)

            mu = _tf.einsum("ab,sbc->sac", cov_cross, self.alpha)

            explained_var = _tf.reduce_sum(
                _tf.einsum("ab,sbc->sac", cov_cross, self.cov_smooth_inv)
                * cov_cross[None, :, :],
                axis=2, keepdims=False)
            var = _tf.maximum(1.0 - explained_var, 0.0)

            influence = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_inv) * cov_cross,
                axis=1, keepdims=False)
            influence = _tf.tile(influence[None, :], [self.size, 1])

            if n_sim > 0:
                rnd = _tf.random.stateless_normal(
                    shape=[self.size, self.root.n_ip, n_sim],
                    seed=seed, dtype=_tf.float64
                )
                sims = _tf.einsum(
                    "ab,sbc->sac",
                    cov_cross,
                    _tf.matmul(self.chol_r, rnd)) + mu

                return mu, var, sims, explained_var, influence

            else:
                return mu, var

    def kl_divergence(self):
        with _tf.name_scope("basic_KL_divergence"):
            delta = self.parameters["delta"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()

            tr = _tf.reduce_sum(self.cov_smooth_inv * self.cov)
            fit = _tf.reduce_sum(alpha_white**2)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol)))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))
            kl = 0.5 * (- tr + fit + det_1 - det_2)

            return kl

    def covariance_matrix_d1(self, y, dir_y, step=1e-3):
        with _tf.name_scope("basic_covariance_matrix_d1"):
            x_pr = self.parent.inducing_points
            x_var = self.parent.inducing_points_variance
            y_pr_plus, y_var_plus = self.parent.propagate(
                y + 0.5 * step * dir_y)
            y_pr_minus, y_var_minus = self.parent.propagate(
                y - 0.5 * step * dir_y)

            cov_1 = self.covariance_matrix(x_pr, y_pr_plus, x_var, y_var_plus)
            cov_2 = self.covariance_matrix(x_pr, y_pr_minus, x_var,
                                           y_var_minus)

            return (cov_1 - cov_2) / step

    def point_variance_d2(self, x, dir_x, step=1e-3):
        with _tf.name_scope("basic_point_variance_d2"):
            mu_1, var_1 = self.parent.propagate(x + 0.5 * dir_x * step)
            mu_2, var_2 = self.parent.propagate(x - 0.5 * dir_x * step)

            ranges = self.parameters["ranges"].get_value()[0, :, :]
            var_1 = var_1 + ranges ** 2
            var_2 = var_2 + ranges ** 2

            dif = mu_1 - mu_2
            avg_var = 0.5 * (var_1 + var_2)
            dist_sq = _tf.reduce_sum(dif ** 2 / avg_var, axis=1, keepdims=True)

            cov_step = self.kernel.kernelize(_tf.sqrt(dist_sq))

            det_avg = _tf.reduce_prod(avg_var, axis=1, keepdims=True) ** (1/2)
            det_1 = _tf.reduce_prod(var_1, axis=1, keepdims=True) ** (1/4)
            det_2 = _tf.reduce_prod(var_2, axis=1, keepdims=True) ** (1/4)

            # norm = _tf.reduce_prod(ranges) / det_avg
            norm = det_1 * det_2 / det_avg
            cov_step = cov_step * norm

            point_var = 2 * (1.0 - cov_step) / step ** 2
            point_var = _tf.tile(point_var, [1, self.size])
            point_var = _tf.transpose(point_var)

            return point_var

    def predict_directions(self, x, dir_x, step=1e-3):
        with _tf.name_scope("basic_prediction_directions"):

            cov_cross = self.covariance_matrix_d1(x, dir_x, step)
            cov_cross = _tf.transpose(cov_cross)

            mu = _tf.einsum("ab,sbc->sac", cov_cross, self.alpha)

            explained_var = _tf.reduce_sum(
                _tf.einsum("ab,sbc->sac", cov_cross, self.cov_smooth_inv)
                * cov_cross[None, :, :],
                axis=2, keepdims=False)

            point_var = self.point_variance_d2(x, dir_x, step)
            var = _tf.maximum(point_var - explained_var, 0.0)

            return mu, var, explained_var

    def covariance_matrix_gradient(self, x, y, var_x=None, var_y=None,
                                   step=1e-3):
        with _tf.name_scope("basic_covariance_matrix_gradient"):
            s = self.parent.size

            base_dir = _tf.unstack(_tf.eye(s, dtype=_tf.float64), axis=0)

            cov_mat = []
            for i in range(s):
                cov_1 = self.covariance_matrix(
                    x + step * base_dir[i][None, :], y, var_x, var_y)
                cov_2 = self.covariance_matrix(
                    x - step * base_dir[i][None, :], y, var_x, var_y)
                cov_mat.append((cov_1 - cov_2) / step)

            return _tf.stack(cov_mat, axis=0)

    def gradient_variance(self, x, dir_x, var_x=None, step=1e-3):
        with _tf.name_scope("basic_gradient_variance"):
            if var_x is None:
                var_x = _tf.zeros_like(x)

            mu_1 = x + dir_x * step
            mu_2 = x - dir_x * step

            ranges = self.parameters["ranges"].get_value()[0, :, :]
            var_x = var_x + ranges ** 2

            dif = mu_1 - mu_2
            dist_sq = _tf.reduce_sum(dif ** 2 / var_x, axis=1, keepdims=True)

            cov_step = self.kernel.kernelize(_tf.sqrt(dist_sq))

            # det_avg = _tf.reduce_prod(var_x, axis=1, keepdims=True) ** (1/2)
            # det_1 = _tf.reduce_prod(var_x, axis=1, keepdims=True) ** (1/4)
            # det_2 = _tf.reduce_prod(var_x, axis=1, keepdims=True) ** (1/4)
            #
            # # norm = _tf.reduce_prod(ranges) / det_avg
            # norm = det_1 * det_2 / det_avg
            # cov_step = cov_step * norm

            point_var = 2 * (1.0 - cov_step) / step ** 2
            point_var = _tf.tile(point_var, [1, self.size])
            point_var = _tf.transpose(point_var)

            return point_var

    class GPGradient(_FunctionalLatentVariable):
        def __init__(self, parent):
            super().__init__(parent)
            self._size = self.parent.size * self.root.size

        def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
            dir_x = _tf.split(_tf.eye(self.root.size, dtype=_tf.float64),
                              self.root.size)
            grads, grads_var = [], []

            for dir_i in dir_x:
                g, gv = self.parent.predict_directions(x, dir_i)
                grads.append(g)
                grads_var.append(gv)

            grads = _tf.concat(grads, axis=0)
            grads_var = _tf.concat(grads_var, axis=0)
            return grads, grads_var

        def refresh(self, jitter=1e-9):
            self.parent.refresh(jitter)
            ip, ip_var, _ = self.parent.predict_directions(
                self.root.inducing_points,
                self.root.inducing_points_variance
            )
            self.inducing_points = _tf.transpose(ip[:, :, 0])
            self.inducing_points_variance = _tf.transpose(ip_var)

        def kl_divergence(self):
            return _tf.constant(0.0, _tf.float64)

    def gradient(self):
        return self.GPGradient(self)


class AdditiveGP(BasicGP):
    def covariance_matrix(self, x, y, var_x=None, var_y=None):
        with _tf.name_scope("basic_covariance_matrix"):
            ranges = self.parameters["ranges"].get_value()
            if var_x is None:
                var_x = _tf.zeros_like(x)
            if var_y is None:
                var_y = _tf.zeros_like(y)
            var_x = var_x[:, None, :]
            var_y = var_y[None, :, :]

            # [n_data, n_data, n_dim]
            dif = x[:, None, :] - y[None, :, :]

            total_var = ranges**2 + (var_x + var_y) / 2
            dist = dif / _tf.sqrt(total_var)
            cov = self.kernel.kernelize(dist)

            # normalization
            det_x = (var_x + ranges**2) ** (1 / 4)
            det_y = (var_y + ranges**2) ** (1 / 4)
            det_2 = _tf.sqrt(total_var)

            norm = det_x * det_y / det_2

            # output
            cov = cov * norm
            cov = _tf.reduce_mean(cov, axis=-1)
            return cov


class Linear(_FunctionalLatentVariable):
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
            self._add_parameter(
                "weights",
                _gpr.RealParameter(
                    _np.ones([parent.size, self.size]),
                    _np.zeros([parent.size, self.size]) - 15,
                    _np.zeros([parent.size, self.size]) + 15
                )
            )

        # binary classification
        if (parent.size == 1) & (self.size == 2):
            self.parameters["weights"].set_value([[1, -1]])
            self.parameters["weights"].fix()

    def refresh(self, jitter=1e-9):
        weights = self.parameters["weights"].get_value()

        self.parent.refresh(jitter)

        if self.parent.inducing_points is not None:
            ip = self.parent.inducing_points
            ip_var = self.parent.inducing_points_variance

            ip = _tf.matmul(ip, weights)
            ip_var = _tf.matmul(ip_var, weights**2)

            self.inducing_points = ip
            self.inducing_points_variance = ip_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        weights = self.parameters["weights"].get_value()

        if n_sim > 0:
            mu, var, sims, exp_var, influence = \
                self.parent.predict(x, x_var, n_sim, seed)

            mu = _tf.einsum("xab,xy->yab", mu, weights)
            var = _tf.einsum("xa,xy->ya", var, weights ** 2)
            sims = _tf.einsum("xab,xy->yab", sims, weights)
            exp_var = _tf.einsum("xa,xy->ya", exp_var, weights ** 2)
            influence = _tf.einsum("xa,xy->ya", influence, weights ** 2)

            return mu, var, sims, exp_var, influence
        else:
            mu, var = self.parent.predict(x, x_var, n_sim, seed)
            mu = _tf.einsum("xab,xy->yab", mu, weights)
            var = _tf.einsum("xa,xy->ya", var, weights ** 2)
            return mu, var

    def predict_directions(self, x, dir_x, step=1e-3):
        mu, var, explained_var = self.parent.predict_directions(x, dir_x, step)

        weights = self.parameters["weights"].get_value()

        mu = _tf.einsum("xab,xy->yab", mu, weights)
        var = _tf.einsum("xa,xy->ya", var, weights ** 2)
        explained_var = _tf.einsum("xa,xy->ya", explained_var, weights ** 2)

        return mu, var, explained_var


class SelectInput(_FunctionalLatentVariable):
    def __init__(self, parent, columns):
        super().__init__(parent)
        self.columns = _tf.constant(columns)
        self._size = len(columns)

    def propagate(self, x, x_var=None):
        mean, var = self.parent.propagate(x, x_var)
        mean = _tf.gather(mean, self.columns, axis=1)
        var = _tf.gather(var, self.columns, axis=1)
        return mean, var

    def refresh(self, jitter=1e-9):
        self.parent.refresh(jitter)
        self.inducing_points = _tf.gather(
            self.parent.inducing_points,
            self.columns, axis=1)
        self.inducing_points_variance = _tf.gather(
            self.parent.inducing_points_variance,
            self.columns, axis=1)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        mu, var, sims, exp_var, influence = \
            self.parent.predict(x, x_var, n_sim, seed)
        mu = _tf.gather(mu, self.columns, axis=0)
        var = _tf.gather(var, self.columns, axis=0)

        if n_sim > 0:
            sims = _tf.gather(sims, self.columns, axis=0)
            exp_var = _tf.gather(exp_var, self.columns, axis=0)
            influence = _tf.gather(influence, self.columns, axis=0)

            return mu, var, sims, exp_var, influence
        else:
            return mu, var


class LinearCombination(_Operation):
    def __init__(self, *latent_variables, unit_variance=True):
        super().__init__(*latent_variables)
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("all parents must have the same size")

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

        if all([lat.inducing_points is not None for lat in self.parents]):
            weights = self.parameters["weights"].get_value()[:, None, None]
            ip = _tf.stack([lat.inducing_points for lat in self.parents],
                           axis=0)
            ip = _tf.reduce_sum(ip * weights, axis=0)

            ip_var = _tf.stack(
                [lat.inducing_points_variance for lat in self.parents], axis=0)
            ip_var = _tf.reduce_sum(ip_var * weights**2, axis=0)

            self.inducing_points = ip
            self.inducing_points_variance = ip_var

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []
        all_influence = []
        weights = self.parameters["weights"].get_value()

        for i, v in enumerate(self.parents):
            mu, var, sims, explained_var, influence = v.predict(
                x, x_var, n_sim, [seed[0] + i, seed[1]])
            all_mu.append(mu)
            all_var.append(var)
            all_sims.append(sims)
            all_explained_var.append(explained_var)
            all_influence.append(influence)

        all_mu = _tf.stack(all_mu, axis=-1)
        all_var = _tf.stack(all_var, axis=-1)
        all_sims = _tf.stack(all_sims, axis=-1)
        all_explained_var = _tf.stack(all_explained_var, axis=-1)
        all_influence = _tf.stack(all_influence, axis=-1)

        all_mu = _tf.reduce_sum(all_mu * weights, axis=-1)
        all_var = _tf.reduce_sum(all_var * weights**2, axis=-1)
        all_sims = _tf.reduce_sum(all_sims * weights, axis=-1)
        all_explained_var = _tf.reduce_sum(all_explained_var * weights**2,
                                           axis=-1)
        all_influence = _tf.reduce_sum(all_influence * weights ** 2, axis=-1)

        return all_mu, all_var, all_sims, all_explained_var, all_influence

    def predict_directions(self, x, dir_x, jitter=1e-9):
        all_mu = []
        all_var = []
        all_explained_var = []
        weights = self.parameters["weights"].get_value()

        for i, v in enumerate(self.parents):
            mu, var, explained_var = v.predict_directions(x, dir_x, jitter)
            all_mu.append(mu)
            all_var.append(var)
            all_explained_var.append(explained_var)

        all_mu = _tf.stack(all_mu, axis=-1)
        all_var = _tf.stack(all_var, axis=-1)
        all_explained_var = _tf.stack(all_explained_var, axis=-1)

        all_mu = _tf.reduce_sum(all_mu * weights, axis=-1)
        all_var = _tf.reduce_sum(all_var * weights ** 2, axis=-1)
        all_explained_var = _tf.reduce_sum(all_explained_var * weights**2,
                                           axis=-1)

        return all_mu, all_var, all_explained_var

    def propagate(self, x, x_var=None):
        mu, var, _, _ = self.predict(x, x_var, n_sim=1)
        mu = _tf.transpose(mu[:, :, 0])
        var = _tf.transpose(var)
        return mu, var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)
        # weights = self.parameters["weights"].get_value()
        # kl = _tf.reduce_sum(weights * _tf.math.log(weights * self.size))
        # return kl


class ProductOfExperts(_Operation):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("all parents must have the same size")

        self._size = sizes[0]

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []
        all_influence = []

        eff_n_sim = _np.maximum(n_sim, 1)

        for i, p in enumerate(self.parents):
            mu, var, sims, explained_var, influence = p.predict(
                x, x_var, eff_n_sim, [seed[0] + i, seed[1]])
            all_mu.append(mu)
            all_var.append(var)
            all_sims.append(sims)
            all_explained_var.append(explained_var)
            all_influence.append(influence)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_sims = _tf.stack(all_sims, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)
        all_influence = _tf.stack(all_influence, axis=0)

        weights = (all_explained_var / (all_var + 1e-6)) + 1e-6
        weights = weights / _tf.reduce_sum(weights, axis=0, keepdims=True)

        w_mu = _tf.reduce_sum(weights[:, :, :, None] * all_mu, axis=0)
        w_var = _tf.reduce_sum(weights * all_var, axis=0)
        w_sims = _tf.reduce_sum(weights[:, :, :, None] * all_sims, axis=0)
        w_explained_var = _tf.reduce_sum(
            weights * all_explained_var, axis=0)
        w_influence = _tf.reduce_sum(weights * all_influence, axis=0)

        if n_sim > 0:
            return w_mu, w_var, w_sims, w_explained_var, w_influence
        else:
            return w_mu, w_var

    def predict_directions(self, x, dir_x, step=1e-3):
        all_mu = []
        all_var = []
        all_explained_var = []

        for i, p in enumerate(self.parents):
            mu, var, explained_var = p.predict_directions(x, dir_x, step)
            all_mu.append(mu)
            all_var.append(var)
            all_explained_var.append(explained_var)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)

        weights = (all_explained_var / (all_var + 1e-6))
        weights = weights / _tf.reduce_sum(weights, axis=0, keepdims=True)

        w_mu = _tf.reduce_sum(weights[:, :, None] * all_mu, axis=0)
        w_var = _tf.reduce_sum(weights * all_var, axis=0)
        w_explained_var = _tf.reduce_sum(weights * all_explained_var, axis=0)

        return w_mu, w_var, w_explained_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class Exponentiation(_FunctionalLatentVariable):
    def __init__(self, parent):
        super().__init__(parent)
        self._add_parameter("amp_mean", _gpr.RealParameter(0, -5, 5))
        self._add_parameter(
            "amp_scale", _gpr.PositiveParameter(0.25, 0.01, 10))
        self._size = parent.size

    def refresh(self, jitter=1e-9):
        amp_mean = self.parameters["amp_mean"].get_value()
        amp_scale = self.parameters["amp_scale"].get_value()

        self.parent.refresh(jitter)

        if self.parent.inducing_points is not None:
            ip = self.parent.inducing_points
            ip_var = self.parent.inducing_points_variance

            ip = ip * _tf.sqrt(amp_scale) + amp_mean
            ip_var = ip_var * amp_scale

            amp_mu = _tf.exp(ip) * (1 + 0.5 * ip_var)
            amp_var = _tf.exp(2 * ip) * ip_var * (1 + ip_var)

            self.inducing_points = amp_mu
            self.inducing_points_variance = amp_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("exponentiation_prediction"):
            amp_mean = self.parameters["amp_mean"].get_value()
            amp_scale = self.parameters["amp_scale"].get_value()

            if n_sim > 0:
                mu, var, sims, explained_var, influence = self.parent.predict(
                    x, x_var, n_sim, seed)

                mu = mu * _tf.sqrt(amp_scale) + amp_mean
                var = var * amp_scale
                sims = sims * _tf.sqrt(amp_scale) + amp_mean
                explained_var = explained_var * amp_scale

                amp_mu = _tf.exp(mu) * (1 + 0.5 * var[:, :, None])
                amp_var = _tf.exp(2 * mu[:, :, 0]) * var * (1 + var)
                amp_explained_var = _tf.exp(2 * mu[:, :, 0]) \
                                    * (var + explained_var) \
                                    * (1 + var + explained_var) \
                                    - amp_var
                amp_sims = _tf.exp(sims)

                return amp_mu, amp_var, amp_sims, amp_explained_var, influence
            else:
                mu, var = self.parent.predict(x, x_var, n_sim=0)

                mu = mu * _tf.sqrt(amp_scale) + amp_mean
                var = var * amp_scale

                amp_mu = _tf.exp(mu) * (1 + 0.5 * var)
                amp_var = _tf.exp(2 * mu) * var * (1 + var)

                return amp_mu, amp_var

    def predict_directions(self, x, dir_x, step=1e-3):
        with _tf.name_scope("exponentiation_prediction"):
            amp_mean = self.parameters["amp_mean"].get_value()
            amp_scale = self.parameters["amp_scale"].get_value()

            mu, var, explained_var = self.parent.predict_directions(
                x, dir_x, step)

            mu = mu * _tf.sqrt(amp_scale) + amp_mean
            var = var * amp_scale
            explained_var = explained_var * amp_scale

            amp_mu = _tf.exp(mu) * (1 + 0.5 * var)
            amp_var = _tf.exp(2 * mu) * var * (1 + var)
            amp_explained_var = _tf.exp(2 * mu) \
                                * (var + explained_var) \
                                * (1 + var + explained_var) \
                                - amp_var

            return amp_mu, amp_var, amp_explained_var


class Multiply(_Operation):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("all parents must have the same size")

        self._size = sizes[0]

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []
        all_influence = []

        for i, v in enumerate(self.parents):
            mu, var, sims, explained_var, influence = v.predict(
                x, x_var, n_sim, [seed[0] + i, seed[1]])
            all_mu.append(mu)
            all_var.append(var)
            all_sims.append(sims)
            all_explained_var.append(explained_var)
            all_influence.append(influence)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_sims = _tf.stack(all_sims, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)
        all_influence = _tf.stack(all_influence, axis=0)

        pred_mu = _tf.reduce_prod(all_mu, axis=0)
        pred_var = _tf.reduce_prod(all_mu[:, :, :, 0] ** 2 + all_var, axis=0) \
                   - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0)
        pred_sims = _tf.reduce_prod(all_sims, axis=0)
        pred_influence = _tf.reduce_mean(all_influence, axis=0)

        pred_explained_var = \
            _tf.reduce_prod(
                all_mu[:, :, :, 0] ** 2 + all_var + all_explained_var,
                axis=0) \
            - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0) \
            - pred_var

        return pred_mu, pred_var, pred_sims, pred_explained_var, pred_influence

    def predict_directions(self, x, dir_x, jitter=1e-9):
        all_mu = []
        all_var = []
        all_explained_var = []

        for i, v in enumerate(self.parents):
            mu, var, explained_var = v.predict_directions(x, dir_x, jitter)
            all_mu.append(mu)
            all_var.append(var)
            all_explained_var.append(explained_var)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)

        pred_mu = _tf.reduce_prod(all_mu, axis=0)
        pred_var = _tf.reduce_prod(all_mu[:, :, :, 0] ** 2 + all_var, axis=0) \
                   - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0)

        pred_explained_var = \
            _tf.reduce_prod(
                all_mu[:, :, :, 0] ** 2 + all_var + all_explained_var,
                axis=0) \
            - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0) \
            - pred_var

        return pred_mu, pred_var, pred_explained_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class Add(_Operation):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("all parents must have the same size")

        self._size = sizes[0]

    def refresh(self, jitter=1e-9):
        for lat in self.parents:
            lat.refresh(jitter)

        # if all([lat.inducing_points is not None for lat in self.parents]):
        #     ip = _tf.stack([lat.inducing_points for lat in self.parents],
        #                    axis=0)
        #     ip = _tf.reduce_sum(ip, axis=0)
        #
        #     ip_var = _tf.stack(
        #         [lat.inducing_points_variance for lat in self.parents], axis=0)
        #     ip_var = _tf.reduce_sum(ip_var, axis=0)
        #
        #     self.inducing_points = ip
        #     self.inducing_points_variance = ip_var

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []
        all_influence = []

        if n_sim > 0:
            for i, v in enumerate(self.parents):
                mu, var, sims, explained_var, influence = v.predict(
                    x, x_var, n_sim, [seed[0] + i, seed[1]])
                all_mu.append(mu)
                all_var.append(var)
                all_sims.append(sims)
                all_explained_var.append(explained_var)
                all_influence.append(influence)

            all_mu = _tf.stack(all_mu, axis=-1)
            all_var = _tf.stack(all_var, axis=-1)
            all_sims = _tf.stack(all_sims, axis=-1)
            all_explained_var = _tf.stack(all_explained_var, axis=-1)
            all_influence = _tf.stack(all_influence, axis=-1)

            all_mu = _tf.reduce_sum(all_mu, axis=-1)
            all_var = _tf.reduce_sum(all_var, axis=-1)
            all_sims = _tf.reduce_sum(all_sims, axis=-1)
            all_explained_var = _tf.reduce_sum(all_explained_var, axis=-1)
            all_influence = _tf.reduce_mean(all_influence, axis=-1)

            return all_mu, all_var, all_sims, all_explained_var, all_influence

        else:
            for i, v in enumerate(self.parents):
                mu, var = v.predict(
                    x, x_var, n_sim, [seed[0] + i, seed[1]])
                all_mu.append(mu)
                all_var.append(var)

            all_mu = _tf.stack(all_mu, axis=-1)
            all_var = _tf.stack(all_var, axis=-1)

            all_mu = _tf.reduce_sum(all_mu, axis=-1)
            all_var = _tf.reduce_sum(all_var, axis=-1)

            return all_mu, all_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        all_mu = []
        all_var = []
        all_explained_var = []

        for i, v in enumerate(self.parents):
            mu, var, explained_var = v.predict_directions(x, dir_x, jitter)
            all_mu.append(mu)
            all_var.append(var)
            all_explained_var.append(explained_var)

        all_mu = _tf.stack(all_mu, axis=-1)
        all_var = _tf.stack(all_var, axis=-1)
        all_explained_var = _tf.stack(all_explained_var, axis=-1)

        all_mu = _tf.reduce_sum(all_mu, axis=-1)
        all_var = _tf.reduce_sum(all_var, axis=-1)
        all_explained_var = _tf.reduce_sum(all_explained_var, axis=-1)

        return all_mu, all_var, all_explained_var

    def propagate(self, x, x_var=None):
        mu, var = self.predict(x, x_var, n_sim=0)
        mu = _tf.transpose(mu[:, :, 0])
        var = _tf.transpose(var)
        return mu, var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class Bias(_FunctionalLatentVariable):
    def __init__(self, parent, scale=5):
        super().__init__(parent)
        self._size = parent.size

        self._add_parameter(
            "bias",
            _gpr.RealParameter(
                _np.zeros([self.size]),
                _np.zeros([self.size]) - scale,
                _np.zeros([self.size]) + scale
            )
        )

    def refresh(self, jitter=1e-9):
        bias = self.parameters["bias"].get_value()[None, :]

        self.parent.refresh(jitter)

        if self.parent.inducing_points is not None:
            self.inducing_points = self.parent.inducing_points + bias
            self.inducing_points_variance = \
                self.parent.inducing_points_variance

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        bias = self.parameters["bias"].get_value()

        if n_sim > 0:
            mu, var, sims, exp_var, influence = \
                self.parent.predict(x, x_var, n_sim, seed)

            mu = mu + bias[:, None, None]
            sims = sims + bias[:, None, None]

            return mu, var, sims, exp_var, influence
        else:
            mu, var = self.parent.predict(x, x_var, n_sim, seed)
            mu = mu + bias[:, None, None]
            return mu, var

    def predict_directions(self, x, dir_x, step=1e-3):
        return self.parent.predict_directions(x, dir_x, step)


class Scale(_FunctionalLatentVariable):
    def __init__(self, parent):
        super().__init__(parent)
        self._size = parent.size

        self._add_parameter(
            "scale",
            _gpr.PositiveParameter(
                _np.ones([self.size]),
                _np.ones([self.size]) / 100,
                _np.ones([self.size]) * 10
            )
        )

    def refresh(self, jitter=1e-9):
        scale = self.parameters["scale"].get_value()[None, :]

        self.parent.refresh(jitter)

        if self.parent.inducing_points is not None:
            self.inducing_points = self.parent.inducing_points \
                                   * _tf.sqrt(scale)
            self.inducing_points_variance = \
                self.parent.inducing_points_variance * scale

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        scale = self.parameters["scale"].get_value()

        if n_sim > 0:
            mu, var, sims, exp_var, influence = \
                self.parent.predict(x, x_var, n_sim, seed)

            mu = mu * _tf.sqrt(scale[:, None, None])
            sims = sims * _tf.sqrt(scale[:, None, None])
            var = var * scale[:, None]
            exp_var = exp_var * scale[:, None]
            influence = influence * scale[:, None]

            return mu, var, sims, exp_var, influence
        else:
            mu, var = self.parent.predict(x, x_var, n_sim, seed)
            mu = mu * _tf.sqrt(scale[:, None, None])
            var = var * scale[:, None]
            return mu, var

    def predict_directions(self, x, dir_x, step=1e-3):
        scale = self.parameters["scale"].get_value()

        mu, var, exp_var = self.parent.predict_directions(x, dir_x, step)

        mu = mu * _tf.sqrt(scale[:, None, None])
        var = var * scale[:, None]
        exp_var = exp_var * scale[:, None]

        return mu, var, exp_var


class CopyGP(_FunctionalLatentVariable):
    def __init__(self, parent, teacher_gp):
        super().__init__(parent)

        self.teacher_gp = teacher_gp
        self._size = teacher_gp.size

        teacher_gp.refresh(1e-6)

        self.teacher_inducing_points = _tf.constant(
            teacher_gp.parent.inducing_points, _tf.float64)
        self.teacher_inducing_points_variance = _tf.constant(
            teacher_gp.parent.inducing_points_variance, _tf.float64)
        self.teacher_cov_inv = _tf.constant(
            teacher_gp.cov_inv, _tf.float64)
        self.teacher_smooth_inv = _tf.constant(
            teacher_gp.cov_smooth_inv, _tf.float64)
        self.teacher_alpha = _tf.constant(teacher_gp.alpha, _tf.float64)
        self.teacher_chol_r = _tf.constant(teacher_gp.chol_r, _tf.float64)

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("copy_gp_refresh"):
            self.parent.refresh(jitter)

            cov = self.teacher_gp.covariance_matrix(
                self.parent.inducing_points,
                self.teacher_inducing_points,
                self.teacher_inducing_points_variance,
                self.teacher_inducing_points_variance)
            cov = _tf.tile(cov[None, :, :], [self.size, 1, 1])

            # inducing points
            pred_inputs = _tf.matmul(cov, self.teacher_alpha)
            self.inducing_points = _tf.transpose(pred_inputs[:, :, 0])
            pred_var = 1.0 - _tf.reduce_sum(
                    _tf.matmul(cov, self.teacher_smooth_inv) * cov,
                    axis=2, keepdims=False
                )
            self.inducing_points_variance = _tf.transpose(pred_var)

    def covariance_matrix(self, x, y, var_x=None, var_y=None):
        return self.teacher_gp.covariance_matrix(x, y, var_x, var_y)

    def covariance_matrix_d1(self, y, dir_y, step=1e-3):
        return self.teacher_gp.covariance_matrix_d1(y, dir_y, step)

    def point_variance_d2(self, x, dir_x, step=1e-3):
        return self.teacher_gp.point_variance_d2(x, dir_x, step)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("basic_prediction"):
            x, x_var = self.parent.propagate(x, x_var)

            # ranges = self.teacher_gp.parameters["ranges"].get_value()
            # total_ranges = _tf.sqrt(ranges ** 2 + x_var)
            # ip_ranges = _tf.sqrt(
            #     ranges ** 2 + self.teacher_inducing_points_variance)

            cov_cross = self.covariance_matrix(
                x, self.teacher_inducing_points,
                x_var, self.teacher_inducing_points_variance)
            cov_cross = _tf.tile(cov_cross[None, :, :], [self.size, 1, 1])

            mu = _tf.matmul(cov_cross, self.teacher_alpha)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.teacher_smooth_inv) * cov_cross,
                axis=2, keepdims=False)
            var = _tf.maximum(1.0 - explained_var, 0.0)

            influence = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.teacher_cov_inv) * cov_cross,
                axis=2, keepdims=False)

            if n_sim > 0:
                rnd = _tf.random.stateless_normal(
                    shape=[self.size, self.root.n_ip, n_sim],
                    seed=seed, dtype=_tf.float64
                )
                sims = _tf.matmul(
                    cov_cross, _tf.matmul(self.teacher_chol_r, rnd)) + mu

                return mu, var, sims, explained_var, influence

            else:
                return mu, var

    def predict_directions(self, x, dir_x, step=1e-3):
        with _tf.name_scope("basic_prediction_directions"):

            cov_cross = self.covariance_matrix_d1(x, dir_x, step)
            cov_cross = _tf.transpose(cov_cross)
            cov_cross = _tf.tile(cov_cross[None, :, :], [self.size, 1, 1])

            mu = _tf.matmul(cov_cross, self.teacher_alpha)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.teacher_smooth_inv) * cov_cross,
                axis=2, keepdims=False)

            point_var = self.point_variance_d2(x, dir_x, step)
            var = _tf.maximum(point_var - explained_var, 0.0)

            return mu, var, explained_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class GPWithGradient(_FunctionalLatentVariable):
    def __init__(self, parent, size=1, kernel=_kr.Gaussian()):
        super().__init__(parent)
        self._size = size
        self.kernel = self._register(kernel)

        self.cov = None
        self.cov_inv = None
        self.cov_chol = None
        self.cov_smooth = None
        self.cov_smooth_chol = None
        self.cov_smooth_inv = None
        self.chol_r = None
        self.alpha = None

        self.prior_cov = None
        self.prior_cov_inv = None
        self.prior_cov_chol = None

        n_ip = self.root.n_ip
        root_dims = self.root.size + 1
        self._add_parameter(
            "alpha_white",
            _gpr.RealParameter(
                _np.random.normal(
                    scale=1e-3,
                    size=[self.size, n_ip * root_dims, 1]
                ),
                _np.zeros([self.size, n_ip * root_dims, 1]) - 10,
                _np.zeros([self.size, n_ip * root_dims, 1]) + 10
            ))
        self._add_parameter(
            "delta",
            _gpr.PositiveParameter(
                _np.ones([self.size, n_ip]),
                _np.ones([self.size, n_ip]) * 1e-6,
                _np.ones([self.size, n_ip]) * 1e4
            ))

        self._add_parameter(
            "ranges",
            _gpr.PositiveParameter(
                _np.ones([1, 1, self.parent.size]),
                _np.ones([1, 1, self.parent.size]) * 1e-6,
                _np.ones([1, 1, self.parent.size]) * 10,
                fixed=True
            )
        )

    def covariance_matrix(self, x, y, var_x=None, var_y=None):
        with _tf.name_scope("basic_covariance_matrix"):
            ranges = self.parameters["ranges"].get_value()
            if var_x is None:
                var_x = _tf.zeros_like(x)
            if var_y is None:
                var_y = _tf.zeros_like(y)
            var_x = var_x[:, None, :]
            var_y = var_y[None, :, :]

            # [n_data, n_data, n_dim]
            dif = x[:, None, :] - y[None, :, :]

            total_var = ranges**2 + (var_x + var_y) / 2
            dist = _tf.sqrt(_tf.reduce_sum(dif ** 2 / total_var, axis=-1))
            cov = self.kernel.kernelize(dist)

            # normalization
            det_x = _tf.reduce_prod(var_x + ranges**2, axis=-1) ** (1 / 4)
            det_y = _tf.reduce_prod(var_y + ranges**2, axis=-1) ** (1 / 4)
            det_2 = _tf.sqrt(_tf.reduce_prod(total_var, axis=-1))

            norm = det_x * det_y / det_2
            # norm = _tf.reduce_prod(ranges) / det_2

            # output
            cov = cov * norm
            return cov

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("basic_refresh"):
            self.parent.refresh(jitter)

            # prior
            ip = self.parent.inducing_points
            ip_var = self.parent.inducing_points_variance

            ndim = _tf.shape(ip)[1]
            n_data = _tf.shape(ip)[0]
            base_dir = _tf.eye(ndim, dtype=_tf.float64)
            ip_dir = _tf.tile(base_dir, [1, n_data])
            ip_dir = _tf.reshape(ip_dir, [n_data * ndim, ndim])
            ip_2 = _tf.tile(ip, [ndim, 1])
            ip_var_2 = _tf.tile(ip_var, [ndim, 1])

            eye = _tf.eye(n_data * (ndim + 1), dtype=_tf.float64)

            base_cov = self.covariance_matrix(ip, ip, ip_var, ip_var)
            cov_d1 = self.covariance_matrix_d1(ip_2, ip_dir, ip_var_2)
            cov_d2 = self.covariance_matrix_d2(ip_2, ip_dir, ip_var_2)
            cov = _tf.concat([
                _tf.concat([base_cov, cov_d1], axis=1),
                _tf.concat([_tf.transpose(cov_d1), cov_d2], axis=1)
            ], axis=0)

            cov = cov + eye * jitter
            chol = _tf.linalg.cholesky(cov)
            cov_inv = _tf.linalg.cholesky_solve(chol, eye)

            self.cov = _tf.tile(cov[None, :, :], [self.size, 1, 1])
            self.cov_chol = _tf.tile(chol[None, :, :], [self.size, 1, 1])
            self.cov_inv = _tf.tile(cov_inv[None, :, :], [self.size, 1, 1])

            # posterior
            eye = _tf.tile(eye[None, :, :], [self.size, 1, 1])
            delta = self.parameters["delta"].get_value()
            delta = _tf.concat(
                [delta, _tf.zeros([self.size, ndim * n_data], _tf.float64)],
                axis=1)
            delta_diag = _tf.linalg.diag(delta)
            self.cov_smooth = self.cov + delta_diag
            self.cov_smooth_chol = _tf.linalg.cholesky(
                self.cov_smooth + eye * jitter)
            self.cov_smooth_inv = _tf.linalg.cholesky_solve(
                self.cov_smooth_chol, eye)
            self.chol_r = _tf.linalg.cholesky(
                self.cov_inv - self.cov_smooth_inv + eye * jitter)

            # inducing points
            alpha_white = self.parameters["alpha_white"].get_value()
            pred_inputs = _tf.matmul(self.cov_chol, alpha_white)
            self.inducing_points = _tf.transpose(pred_inputs[:, :n_data, 0])
            self.alpha = _tf.matmul(self.cov_inv, pred_inputs)
            pred_var = 1.0 - _tf.reduce_sum(
                    _tf.matmul(self.cov, self.cov_smooth_inv) * self.cov,
                    axis=2, keepdims=False
                )
            self.inducing_points_variance = _tf.transpose(pred_var[:, :n_data])

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("GPWithGradient_prediction"):
            x_pr, x_pr_var = self.parent.propagate(x, x_var)

            cov_1 = self.covariance_matrix(
                x_pr, self.parent.inducing_points,
                x_pr_var, self.parent.inducing_points_variance)
            cov_2 = self.covariance_matrix_d1_rev(x, x_var)
            cov_cross = _tf.concat([cov_1, cov_2], axis=1)
            cov_cross = _tf.tile(cov_cross[None, :, :], [self.size, 1, 1])

            mu = _tf.matmul(cov_cross, self.alpha)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)
            var = _tf.maximum(1.0 - explained_var, 0.0)

            influence = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_inv) * cov_cross,
                axis=2, keepdims=False)

            if n_sim > 0:
                rnd = _tf.random.stateless_normal(
                    shape=[self.size, self.root.n_ip * (self.root.size + 1),
                           n_sim],
                    seed=seed, dtype=_tf.float64
                )
                sims = _tf.matmul(cov_cross, _tf.matmul(self.chol_r, rnd)) + mu

                return mu, var, sims, explained_var, influence

            else:
                return mu, var

    def kl_divergence(self):
        with _tf.name_scope("basic_KL_divergence"):
            delta = self.parameters["delta"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()

            tr = _tf.reduce_sum(self.cov_smooth_inv * self.cov)
            fit = _tf.reduce_sum(alpha_white**2)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol)))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))
            kl = 0.5 * (- tr + fit + det_1 - det_2)

            return kl

    def covariance_matrix_d1(self, y, dir_y, y_var=None, step=1e-3):
        with _tf.name_scope("covariance_matrix_d1"):
            x_pr = self.parent.inducing_points
            x_var = self.parent.inducing_points_variance
            y_pr_plus, y_var_plus = self.parent.propagate(
                y + 0.5 * step * dir_y, y_var)
            y_pr_minus, y_var_minus = self.parent.propagate(
                y - 0.5 * step * dir_y, y_var)

            cov_1 = self.covariance_matrix(x_pr, y_pr_plus, x_var, y_var_plus)
            cov_2 = self.covariance_matrix(x_pr, y_pr_minus, x_var,
                                           y_var_minus)

            return (cov_1 - cov_2) / step

    def covariance_matrix_d2(self, y, dir_y, y_var=None, step=1e-3):
        """
        Direction-direction covariance.

        Covariance between a set of directions and inducing gradients.

        Parameters
        ----------
        y_var
        y
        dir_y
        step

        Returns
        -------

        """
        with _tf.name_scope("covariance_matrix_d2"):
            ndim = _tf.shape(self.parent.inducing_points)[1]
            n_data = _tf.shape(self.parent.inducing_points)[0]
            eye = _tf.eye(ndim, dtype=_tf.float64)
            ip_dir = _tf.tile(eye, [1, n_data])
            ip_dir = _tf.reshape(ip_dir, [n_data * ndim, ndim])
            ip = _tf.tile(self.parent.inducing_points, [ndim, 1])
            ip_var = _tf.tile(self.parent.inducing_points_variance, [ndim, 1])

            ip_plus, ip_var_plus = self.parent.propagate(
                ip + 0.5 * step * ip_dir, ip_var)
            ip_minus, ip_var_minus = self.parent.propagate(
                ip - 0.5 * step * ip_dir, ip_var)

            y_pr_plus, y_var_plus = self.parent.propagate(
                y + 0.5 * step * dir_y, y_var)
            y_pr_minus, y_var_minus = self.parent.propagate(
                y - 0.5 * step * dir_y, y_var)

            cov_1a = self.covariance_matrix(
                y_pr_plus, ip_plus, y_var_plus, ip_var_plus)
            cov_1b = self.covariance_matrix(
                y_pr_minus, ip_plus, y_var_minus, ip_var_plus)
            cov_1 = (cov_1a - cov_1b) / step

            cov_2a = self.covariance_matrix(
                y_pr_plus, ip_minus, y_var_plus, ip_var_minus)
            cov_2b = self.covariance_matrix(
                y_pr_minus, ip_minus, y_var_minus, ip_var_minus)
            cov_2 = (cov_2a - cov_2b) / step

            return (cov_1 - cov_2) / step

    def point_variance_d2(self, x, dir_x, step=1e-3):
        with _tf.name_scope("basic_point_variance_d2"):
            mu_1, var_1 = self.parent.propagate(x + 0.5 * dir_x * step)
            mu_2, var_2 = self.parent.propagate(x - 0.5 * dir_x * step)

            ranges = self.parameters["ranges"].get_value()[0, :, :]
            var_1 = var_1 + ranges ** 2
            var_2 = var_2 + ranges ** 2

            dif = mu_1 - mu_2
            avg_var = 0.5 * (var_1 + var_2)
            dist_sq = _tf.reduce_sum(dif ** 2 / avg_var, axis=1, keepdims=True)

            cov_step = self.kernel.kernelize(_tf.sqrt(dist_sq))

            det_avg = _tf.reduce_prod(avg_var, axis=1, keepdims=True) ** (1/2)
            det_1 = _tf.reduce_prod(var_1, axis=1, keepdims=True) ** (1/4)
            det_2 = _tf.reduce_prod(var_2, axis=1, keepdims=True) ** (1/4)

            # norm = _tf.reduce_prod(ranges) / det_avg
            norm = det_1 * det_2 / det_avg
            cov_step = cov_step * norm

            point_var = 2 * (1.0 - cov_step) / step ** 2
            point_var = _tf.tile(point_var, [1, self.size])
            point_var = _tf.transpose(point_var)

            return point_var

    def covariance_matrix_d1_rev(self, y, y_var=None, step=1e-3):
        """
        Point-direction covariance.

        Covariance between a set of coordinates and inducing gradients.

        Parameters
        ----------
        y_var
        y
        step

        Returns
        -------

        """
        with _tf.name_scope("covariance_matrix_d1_rev"):
            ndim = _tf.shape(self.parent.inducing_points)[1]
            n_data = _tf.shape(self.parent.inducing_points)[0]
            eye = _tf.eye(ndim, dtype=_tf.float64)
            ip_dir = _tf.tile(eye, [1, n_data])
            ip_dir = _tf.reshape(ip_dir, [n_data * ndim, ndim])
            ip = _tf.tile(self.parent.inducing_points, [ndim, 1])
            ip_var = _tf.tile(self.parent.inducing_points_variance, [ndim, 1])

            ip_plus, ip_var_plus = self.parent.propagate(
                ip + 0.5 * step * ip_dir, ip_var)
            ip_minus, ip_var_minus = self.parent.propagate(
                ip - 0.5 * step * ip_dir, ip_var)

            y, y_var = self.parent.propagate(y, y_var)

            cov_1 = self.covariance_matrix(
                y, ip_plus, y_var, ip_var_plus)
            cov_2 = self.covariance_matrix(
                y, ip_minus, y_var, ip_var_minus)

            return (cov_1 - cov_2) / step

    def predict_directions(self, x, dir_x, step=1e-3):
        with _tf.name_scope("basic_prediction_directions"):

            cov_1 = self.covariance_matrix_d1(x, dir_x, step=step)
            cov_1 = _tf.transpose(cov_1)
            cov_2 = self.covariance_matrix_d2(x, dir_x, step=step)
            cov_cross = _tf.concat([cov_1, cov_2], axis=1)
            cov_cross = _tf.tile(cov_cross[None, :, :], [self.size, 1, 1])

            mu = _tf.matmul(cov_cross, self.alpha)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)

            point_var = self.point_variance_d2(x, dir_x, step)
            var = _tf.maximum(point_var - explained_var, 0.0)

            return mu, var, explained_var


class RadialTrend(_FunctionalLatentVariable):
    def __init__(self, parent, size=1):
        super().__init__(parent)
        self._size = size

        self._add_parameter(
            "scale",
            _gpr.PositiveParameter(
                _np.ones([1, self.size]),
                _np.ones([1, self.size]) * 0.1,
                _np.ones([1, self.size]) * 10
            )
        )
        self._add_parameter(
            "center",
            _gpr.RealParameter(
                _np.zeros([self.parent.size, 1, self.size]),
                _np.zeros([self.parent.size, 1, self.size]) - 5,
                _np.zeros([self.parent.size, 1, self.size]) + 5
            )
        )

    def compute_trend(self, x):
        center = self.parameters["center"].get_value()
        scale = self.parameters["scale"].get_value()

        dif = x[:, :, None] - center
        dist = _tf.sqrt(_tf.reduce_sum(dif**2, axis=0) + 1e-12)  # [n_data, size]
        dist = dist / scale

        trend = _tf.where(
            _tf.greater(dist, 2.0),
            _tf.zeros_like(dist) - 1,
            _tf.where(
                _tf.less(dist, 1.0),
                1 - dist ** 2,
                dist**2 - 4*dist + 3
            )
        )

        return _tf.transpose(trend)

    def compute_trend_gradient(self, x):
        center = self.parameters["center"].get_value()
        scale = self.parameters["scale"].get_value()

        dif = x[:, :, None] - center
        dist = _tf.sqrt(_tf.reduce_sum(dif**2, axis=0) + 1e-12)  # [n_data, size]
        dist_sc = dist / scale

        trend = _tf.where(
            _tf.greater(dist_sc, 2.0),
            _tf.zeros_like(dist_sc),
            _tf.where(
                _tf.less(dist_sc, 1.0),
                - 2*dist_sc,
                2*dist_sc - 4
            )
        )

        trend = trend[:, :, None] / dist[:, :, None] * x[:, None, :]

        return _tf.transpose(trend, [1, 0, 2])

    def refresh(self, jitter=1e-9):
        self.parent.refresh(jitter)

        if self.parent.inducing_points is not None:
            ip = self.parent.inducing_points
            ip_var = self.parent.inducing_points_variance

            self.inducing_points = _tf.transpose(
                self.compute_trend(_tf.transpose(ip)))
            self.inducing_points_variance = _tf.zeros_like(self.inducing_points)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        if n_sim > 0:
            mu, var, sims, exp_var, influence = \
                self.parent.predict(x, x_var, n_sim, seed)

            mu = self.compute_trend(mu[:, :, 0])[:, :, None]
            var = _tf.zeros_like(mu[:, :, 0])
            sims = _tf.tile(mu, [1, 1, n_sim])
            exp_var = _tf.zeros_like(mu[:, :, 0])
            influence = _tf.zeros_like(mu[:, :, 0])

            return mu, var, sims, exp_var, influence
        else:
            mu, var = self.parent.predict(x, x_var, n_sim, seed)
            mu = self.compute_trend(mu[:, :, 0])[:, :, None]
            var = _tf.zeros_like(mu[:, :, 0])
            return mu, var

    def predict_directions(self, x, dir_x, step=1e-3):
        mu, var, explained_var = self.parent.predict_directions(x, dir_x, step)

        grad = self.compute_trend_gradient(mu)
        mu = _tf.reduce_sum(grad * dir_x[:, None, :], axis=2)
        var = _tf.zeros_like(mu[:, :, 0])
        explained_var = _tf.zeros_like(mu[:, :, 0])

        return mu, var, explained_var


class GPWalk(_FunctionalLatentVariable):
    def __init__(self, parent, step=0.01, n_steps=10):
        super().__init__(parent)

        if parent.size != parent.parent.size:
            raise ValueError("parent node must have the same size as "
                             "its own parent")

        self.walker = parent.parent
        self.field = parent
        self._size = parent.size

        self.step = step
        self.n_steps = n_steps

        self._add_parameter(
            "amp",
            _gpr.PositiveParameter(1, 0.01, 100)
        )

    def propagate(self, x, x_var=None):
        walker_mu, walker_var = self.walker.propagate(x, x_var)
        # field_mu, field_var = self.field.propagate(x, x_var)

        amp = self.parameters["amp"].get_value()

        for _ in range(self.n_steps):
            cov_cross = self.field.covariance_matrix(
                walker_mu, self.walker.inducing_points,
                walker_var, self.walker.inducing_points_variance)
            cov_cross = _tf.tile(cov_cross[None, :, :], [self.size, 1, 1])

            field_mu = _tf.matmul(cov_cross, self.field.alpha)[:, :, 0]
            field_mu = _tf.transpose(field_mu)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.field.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)
            field_var = _tf.maximum(1.0 - explained_var, 0.0)
            field_var = _tf.transpose(field_var)

            walker_mu = walker_mu + self.step * field_mu * amp
            walker_var = walker_var + self.step * field_var * amp ** 2

        return walker_mu, walker_var

    def refresh(self, jitter=1e-9):
        self.field.refresh(jitter)
        self.inducing_points, self.inducing_points_variance = self.propagate(
            *self.root.get_root_inducing_points()
        )

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

        # rng = self.field.parameters["ranges"].get_value()[0, :, :]
        # var = self.inducing_points_variance + rng**2
        # w_var = self.walker.inducing_points_variance + rng ** 2
        #
        # tr = _tf.reduce_sum(w_var / var)
        #
        # dist = _tf.reduce_sum(
        #     (self.walker.inducing_points - self.inducing_points)**2 / var)
        #
        # det = _tf.reduce_sum(_tf.math.log(var)) \
        #       - _tf.reduce_sum(_tf.math.log(w_var))
        #
        # return tr + dist + det - self.root.n_ip

    def compute_path(self, x, x_var=None):
        walker_mu, walker_var = self.walker.propagate(x, x_var)
        # field_mu, field_var = self.field.propagate(x, x_var)

        amp = self.parameters["amp"].get_value()

        all_mu = [walker_mu]
        all_var = [walker_var]
        for _ in range(self.n_steps):
            cov_cross = self.field.covariance_matrix(
                walker_mu, self.walker.inducing_points,
                walker_var, self.walker.inducing_points_variance)
            cov_cross = _tf.tile(cov_cross[None, :, :], [self.size, 1, 1])

            field_mu = _tf.matmul(cov_cross, self.field.alpha)[:, :, 0]
            field_mu = _tf.transpose(field_mu)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.field.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)
            field_var = _tf.maximum(1.0 - explained_var, 0.0)
            field_var = _tf.transpose(field_var)

            walker_mu = walker_mu + self.step * field_mu * amp
            walker_var = walker_var + self.step * field_var * amp ** 2

            all_mu.append(walker_mu)
            all_var.append(walker_var)
        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)

        return all_mu, all_var

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        mu, var = self.propagate(x, x_var)
        mu = _tf.transpose(mu)[:, :, None]
        var = _tf.transpose(var)

        if n_sim < 1:
            return mu, var

        explained_var = _tf.zeros_like(var)

        # samples are coherent among data points
        rnd = _tf.random.stateless_normal(
            shape=[self.size, 1, n_sim],
            seed=seed, dtype=_tf.float64
        )
        sims = mu + rnd * _tf.sqrt(var[:, :, None])

        influence = _tf.zeros_like(var)

        return mu, var, sims, explained_var, influence


class GaussianInput(_RootLatentVariable):
    def __init__(self, inducing_points, fix_inducing_points=True,
                 center=False):
        super().__init__()
        self._size = inducing_points.coordinates.shape[1]
        self.bounding_box = inducing_points.bounding_box

        self.n_ip = inducing_points.coordinates.shape[0]
        self._add_parameter(
            "inducing_points",
            _gpr.RealParameter(
                inducing_points.coordinates,
                _np.tile(self.bounding_box.min, [self.n_ip, 1]),
                _np.tile(self.bounding_box.max, [self.n_ip, 1]),
                fixed=fix_inducing_points
            ))
        self._add_parameter(
            "inducing_points_variance",
            _gpr.PositiveParameter(
                _np.ones_like(inducing_points.coordinates),
                _np.ones_like(inducing_points.coordinates) * 0.01,
                _np.ones_like(inducing_points.coordinates) * 10
            ))

        self.center = _np.zeros_like(self.bounding_box.max)
        if center:
            self.center = 0.5 * (self.bounding_box.min + self.bounding_box.max)

    def get_root_inducing_points(self):
        ip = self.parameters["inducing_points"].get_value()
        ip_var = self.parameters["inducing_points_variance"].get_value()
        return ip, ip_var

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("basic_input_refresh"):
            self.inducing_points = \
                self.parameters["inducing_points"].get_value() - self.center
            self.inducing_points_variance = \
                self.parameters["inducing_points_variance"].get_value()

    def propagate(self, x, x_var=None):
        return x - self.center, x_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        x = _tf.transpose(x - self.center)
        x_var = _tf.transpose(x_var)
        if n_sim > 0:
            sims = _tf.tile(x[:, :, None], [1, 1, n_sim])
            return x[:, :, None], x_var, sims, \
                   _tf.zeros_like(x_var), _tf.zeros_like(x_var)
        else:
            return x[:, :, None], x_var


class RefinerExperts(ProductOfExperts):
    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []
        all_influence = []

        eff_n_sim = _np.maximum(n_sim, 1)

        for i, p in enumerate(self.parents):
            mu, var, sims, explained_var, influence = p.predict(
                x, x_var, eff_n_sim, [seed[0] + i, seed[1]])
            all_mu.append(mu)
            all_var.append(var)
            all_sims.append(sims)
            all_explained_var.append(explained_var)
            all_influence.append(influence)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_sims = _tf.stack(all_sims, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)
        all_influence = _tf.stack(all_influence, axis=0)

        # cum_mu = _tf.math.cumsum(all_mu, axis=0)
        # all_sims = all_sims - all_mu + cum_mu

        all_sims = all_sims - all_mu

        weights = (all_explained_var / (all_var + 1e-6)) + 1e-6
        weights = weights / _tf.reduce_sum(weights, axis=0, keepdims=True)

        # prec = (1 / (all_var + 1e-6)) + 1e-6
        # weights = prec / _tf.reduce_sum(prec, axis=0, keepdims=True)

        w_mu = _tf.reduce_sum(all_mu, axis=0)
        # w_mu = _tf.reduce_sum(weights[:, :, :, None] * all_mu, axis=0)
        w_var = _tf.reduce_sum(weights * all_var, axis=0)
        # w_var = 1 / _tf.reduce_sum(prec, axis=0)
        w_sims = _tf.reduce_sum(weights[:, :, :, None] * all_sims, axis=0)
        w_explained_var = _tf.reduce_sum(
            weights * all_explained_var, axis=0)
        w_influence = _tf.reduce_sum(weights * all_influence, axis=0)

        w_sims = w_sims + w_mu

        if n_sim > 0:
            return w_mu, w_var, w_sims, w_explained_var, w_influence
        else:
            return w_mu, w_var

    def predict_directions(self, x, dir_x, step=1e-3):
        all_mu = []
        all_var = []
        all_explained_var = []

        for i, p in enumerate(self.parents):
            mu, var, explained_var = p.predict_directions(x, dir_x, step)
            all_mu.append(mu)
            all_var.append(var)
            all_explained_var.append(explained_var)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)

        weights = (all_explained_var / (all_var + 1e-6))
        weights = weights / _tf.reduce_sum(weights, axis=0, keepdims=True)

        # prec = (1 / (all_var + 1e-6)) + 1e-6
        # weights = prec / _tf.reduce_sum(prec, axis=0, keepdims=True)

        w_mu = _tf.reduce_sum(all_mu, axis=0)
        w_var = _tf.reduce_sum(weights * all_var, axis=0)
        # w_var = 1 / _tf.reduce_sum(prec, axis=0)
        w_explained_var = _tf.reduce_sum(weights * all_explained_var, axis=0)

        return w_mu, w_var, w_explained_var
