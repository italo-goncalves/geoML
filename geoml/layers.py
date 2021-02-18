# geoML - machine learning models for geospatial data
# Copyright (C) 2020  Ítalo Gomes Gonçalves
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

import geoml.parameter as _gpr
import geoml.tftools as _tftools
import geoml.kernels as _kr
import geoml.transform as _tr

import numpy as _np
import tensorflow as _tf


class _LatentVariableLayer:
    def __init__(self):
        self.parameters = {}
        self._all_parameters = []
        self._n_latent = 0

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        return s

    @property
    def all_parameters(self):
        return self._all_parameters

    @property
    def n_latent(self):
        return self._n_latent

    @staticmethod
    def reshape_chol(x, jitter):
        x = _tftools.reshape_lower_traingular(x)
        x = _tf.linalg.set_diag(
            x, _tf.nn.softplus(_tf.linalg.diag_part(x)) + jitter)
        return x

    @staticmethod
    def add_offset(x):
        ones = _tf.ones([_tf.shape(x)[0], 1], _tf.float64)
        return _tf.concat([ones, x], axis=1)

    @staticmethod
    def add_offset_grad(x):
        zeros = _tf.zeros([_tf.shape(x)[0], 1], _tf.float64)
        return _tf.concat([zeros, x], axis=1)

    def set_kernel_limits(self, data):
        pass

    def refresh(self, jitter=1e-9):
        raise NotImplementedError

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        raise NotImplementedError

    def predict_directions(self, x, dir_x, jitter=1e-9):
        raise NotImplementedError

    def kl_divergence(self, jitter=1e-9):
        raise NotImplementedError

    @staticmethod
    def modulate(mean, var):
        amp = _tf.exp(mean) * (1 + 0.5 * var)
        return amp


class Generic(_LatentVariableLayer):
    def __init__(self, kernels, inducing_points, fix_inducing_points=True):
        super().__init__()
        if not (isinstance(kernels, list) or isinstance(kernels, tuple)):
            kernels = [kernels]
        self.kernels = kernels
        self._n_latent = len(kernels)

        self.cov = None
        self.cov_inv = None
        self.cov_chol = None
        self.cov_smooth = None
        self.cov_smooth_chol = None
        self.cov_smooth_inv = None
        self.chol_r = None

        self.n_ps = inducing_points.coordinates.shape[0]
        box = inducing_points.bounding_box
        self.parameters.update({
            "inducing_points": _gpr.RealParameter(
                inducing_points.coordinates,
                _np.tile(box[0, :], [self.n_ps, 1]),
                _np.tile(box[1, :], [self.n_ps, 1]),
                fixed=fix_inducing_points
            ),
            # "alpha": _gpr.RealParameter(
            #     _np.zeros([self.n_latent, self.n_ps, 1]) + 0.1,
            #     _np.zeros([self.n_latent, self.n_ps, 1]) - 10,
            #     _np.zeros([self.n_latent, self.n_ps, 1]) + 10
            # ),
            "alpha_white": _gpr.RealParameter(
                _np.zeros([self.n_latent, self.n_ps, 1]) + 0.01,
                _np.zeros([self.n_latent, self.n_ps, 1]) - 10,
                _np.zeros([self.n_latent, self.n_ps, 1]) + 10
            ),
            "delta": _gpr.PositiveParameter(
                _np.ones([self.n_latent, self.n_ps]) * 1,
                _np.ones([self.n_latent, self.n_ps]) * 1e-6,
                _np.ones([self.n_latent, self.n_ps]) * 1e4
            )
        })

        self._all_parameters += [v for v in self.parameters.values()]
        for kernel in kernels:
            self._all_parameters += kernel.all_parameters

    def __repr__(self):
        s = self.__class__.__name__ + "\n\nInducing points: %d\n" % self.n_ps
        return s

    def set_kernel_limits(self, data):
        for kernel in self.kernels:
            kernel.set_limits(data)

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("latent_variable_refresh"):
            # prior
            ps = self.parameters["inducing_points"].get_value()

            cov_ps = []
            chol_k = []
            k_inv = []
            eye = _tf.eye(self.n_ps, dtype=_tf.float64)
            for kernel in self.kernels:
                cov_ps_i = kernel.self_covariance_matrix(ps)

                cov_ps_i = cov_ps_i + eye * jitter
                chol_k_i = _tf.linalg.cholesky(cov_ps_i)
                k_inv_i = _tf.linalg.cholesky_solve(chol_k_i, eye)

                cov_ps.append(cov_ps_i)
                chol_k.append(chol_k_i)
                k_inv.append(k_inv_i)

            self.cov = _tf.stack(cov_ps, axis=0)
            self.cov_chol = _tf.stack(chol_k, axis=0)
            self.cov_inv = _tf.stack(k_inv, axis=0)

            # posterior
            delta = self.parameters["delta"].get_value()
            delta_diag = _tf.linalg.diag(delta)
            self.cov_smooth = self.cov + delta_diag
            self.cov_smooth_chol = _tf.linalg.cholesky(
                self.cov_smooth + eye * jitter)
            self.cov_smooth_inv = _tf.linalg.cholesky_solve(
                self.cov_smooth_chol, eye)
            self.chol_r = _tf.linalg.cholesky(
                self.cov_inv - self.cov_smooth_inv + eye * jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction"):
            # alpha = self.parameters["alpha"].get_value()
            ps = self.parameters["inducing_points"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance(x)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data]

            cov_cross = _tf.stack(
                [kernel.covariance_matrix(x, ps)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            var = point_var - explained_var
            var = _tf.maximum(var, 0.0)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_ps, n_sim],
                seed=seed, dtype=_tf.float64
            )
            sims = _tf.matmul(cov_cross, _tf.matmul(self.chol_r, rnd)) + mu

            return mu, var, sims, explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction_directions"):
            # alpha = self.parameters["alpha"].get_value()
            ps = self.parameters["inducing_points"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance_d2(x, dir_x)
                 for kernel in self.kernels],
                axis=0)
            # scale = _tf.stack(
            #     [kernel.point_variance(x)
            #      for kernel in self.kernels],
            #     axis=0) / point_var
            cov_cross = _tf.stack(
                [_tf.transpose(kernel.covariance_matrix_d1(
                    ps, x, dir_x))
                    for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]
            # cov_cross = cov_cross * _tf.sqrt(scale[:, :, None])

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            var = point_var - explained_var
            # var = scale * point_var - explained_var
            var = _tf.maximum(var, 0.0)

            return mu, var, explained_var

    def kl_divergence(self, jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_KL_divergence"):
            # alpha = self.parameters["alpha"].get_value()
            delta = self.parameters["delta"].get_value()
            # mean = _tf.matmul(self.cov, alpha)
            alpha_white = self.parameters["alpha_white"].get_value()

            tr = _tf.reduce_sum(self.cov_smooth_inv * self.cov)
            # fit = _tf.reduce_sum(alpha * mean)
            fit = _tf.reduce_sum(alpha_white**2)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol)))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))
            kl = 0.5 * (- tr + fit + det_1 - det_2)

            return kl


class _GriddedLayer(_LatentVariableLayer):
    @staticmethod
    def rowwise_separable_matmul(mat_list, mat):
        # shape of matrices in mat_list: [batch, n_rows, grid_size_i]
        # shape of mat: [batch, prod(grid_size), n_cols]
        with _tf.name_scope("rowwise_separable_matmul"):
            dims = [_tf.shape(m)[2] for m in mat_list]
            mat_2 = _tf.transpose(mat, perm=[0, 2, 1])
            mat_2 = _tf.reshape(
                mat_2, [_tf.shape(mat_2)[0], _tf.shape(mat_2)[1]] + dims)

            op_str = ""
            idx = "abc"[:len(dims)]
            for i in range(len(dims)):
                op_str += "...i" + idx[i] + ","
            op_str += "...j" + idx + "->...ij"

            all_mats = mat_list + [mat_2]
            out = _tf.einsum(op_str, *all_mats)
            return out

    @staticmethod
    def covariance_matrix(x, y, ranges):
        # x, y and ranges must be 1d
        with _tf.name_scope("separable_covariance_matrix"):
            dist = x[:, None] - y[None, :]
            dist_sq = (dist[None, :, :] / ranges[:, None, None]) ** 2
            cov = _tf.exp(-3 * dist_sq)
            return cov

    def set_kernel_limits(self, data):
        pass


class Decoupled(_GriddedLayer):
    def __init__(self, n_latent, sparse_grid, dense_grid):
        super().__init__()
        if sparse_grid.__class__.__name__ not in ("Grid1D", "Grid2D", "Grid3D"):
            raise Exception("sparse_grid must be a grid object")
        if dense_grid.__class__.__name__ not in ("Grid1D", "Grid2D", "Grid3D"):
            raise Exception("dense_grid must be a grid object")
        if sparse_grid.n_dim != dense_grid.n_dim:
            raise ValueError("dimensions of sparse and dense grids"
                             " do not match")
        n_dim = sparse_grid.n_dim

        self._n_latent = n_latent
        self.cov = [None] * n_dim
        self.cov_inv = [None] * n_dim
        self.cov_chol = [None] * n_dim
        self.dense_cov = [None] * n_dim
        self.dense_cov_inv = [None] * n_dim
        self.cross_grid_cov = [None] * n_dim
        # self.cross_grid_weights = [None] * n_dim
        # self.mat_s = None
        # self.chol_s = None
        self.cov_smooth = None
        self.cov_smooth_chol = None
        self.cov_smooth_inv = None
        self.chol_r = None

        self.dense_pseudo_inputs = [_tf.constant(g, _tf.float64)
                                    for g in dense_grid.grid]
        self.sparse_pseudo_inputs = [_tf.constant(g, _tf.float64)
                                     for g in sparse_grid.grid]

        self.dense_n_ps = dense_grid.coordinates.shape[0]
        self.sparse_n_ps = sparse_grid.coordinates.shape[0]
        self.grid_size = sparse_grid.grid_size

        self.parameters.update({
            "alpha_dense": _gpr.RealParameter(
                _np.zeros([self.n_latent, self.dense_n_ps, 1]) + 0.01,
                _np.zeros([self.n_latent, self.dense_n_ps, 1]) - 10,
                _np.zeros([self.n_latent, self.dense_n_ps, 1]) + 10
            ),
            "alpha_sparse": _gpr.RealParameter(
                _np.zeros([self.n_latent, self.sparse_n_ps, 1]) - 0.01,
                _np.zeros([self.n_latent, self.sparse_n_ps, 1]) - 10,
                _np.zeros([self.n_latent, self.sparse_n_ps, 1]) + 10
            ),
            "delta": _gpr.PositiveParameter(
                _np.ones([self.n_latent, self.sparse_n_ps]),
                _np.ones([self.n_latent, self.sparse_n_ps]) * 1e-6,
                _np.ones([self.n_latent, self.sparse_n_ps]) * 10
            )
        })

        for d in range(n_dim):
            n_ps = dense_grid.grid_size[d]
            dif = dense_grid.grid[d][-1] - dense_grid.grid[d][0]
            self.parameters.update({
                "ranges_%d" % d: _gpr.PositiveParameter(
                    # _np.ones([self.n_latent]) * dif / 2,
                    dif / _np.arange(1, self.n_latent + 1),
                    _np.ones([self.n_latent]) * dif / n_ps,
                    _np.ones([self.n_latent]) * dif * 2
                )
            })

        self._all_parameters += [v for v in self.parameters.values()]

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("decoupled_layer_refresh"):
            n_dim = len(self.cov)

            # prior
            for d in range(n_dim):
                ps = self.sparse_pseudo_inputs[d]
                rng = self.parameters["ranges_%d" % d].get_value()

                eye = _tf.eye(self.grid_size[d], dtype=_tf.float64,
                              batch_shape=[self.n_latent])
                cov_ps = self.covariance_matrix(ps, ps, rng)
                chol_k = _tf.linalg.cholesky(cov_ps + eye * jitter)
                k_inv = _tf.linalg.cholesky_solve(chol_k, eye)

                self.cov[d] = cov_ps
                self.cov_chol[d] = chol_k
                self.cov_inv[d] = k_inv

                ps_dense = self.dense_pseudo_inputs[d]
                dense_cov = self.covariance_matrix(ps_dense, ps_dense, rng)
                self.dense_cov[d] = dense_cov

                cross_cov = self.covariance_matrix(ps, ps_dense, rng)
                self.cross_grid_cov[d] = cross_cov

            # posterior
            kron = _tf.linalg.LinearOperatorKronecker([
                _tf.linalg.LinearOperatorFullMatrix(mat) for mat in self.cov
            ])
            cov = kron.to_dense()
            kron_inv = _tf.linalg.LinearOperatorKronecker([
                _tf.linalg.LinearOperatorFullMatrix(mat) for mat in self.cov_inv
            ])
            cov_inv = kron_inv.to_dense()

            delta = self.parameters["delta"].get_value()
            delta_diag = _tf.linalg.diag(delta)

            eye = _tf.eye(self.sparse_n_ps, dtype=_tf.float64,
                          batch_shape=[self.n_latent])

            self.cov_smooth = cov + delta_diag
            self.cov_smooth_chol = _tf.linalg.cholesky(
                self.cov_smooth + eye * jitter)
            self.cov_smooth_inv = _tf.linalg.cholesky_solve(
                self.cov_smooth_chol, eye)
            self.chol_r = _tf.linalg.cholesky(
                cov_inv - self.cov_smooth_inv + eye * jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        self.refresh(jitter)

        with _tf.name_scope("decoupled_layer_prediction"):
            alpha_dense = self.parameters["alpha_dense"].get_value()
            alpha_sparse = self.parameters["alpha_sparse"].get_value()
            x_split = _tf.unstack(x, axis=1)
            n_dim = len(self.cov)

            # covariances
            point_var = _tf.ones([self.n_latent, _tf.shape(x)[0]],
                                 dtype=_tf.float64)

            cov_cross_dense = []
            for d in range(n_dim):
                rng = self.parameters["ranges_%d" % d].get_value()
                cov = self.covariance_matrix(
                    x_split[d], self.dense_pseudo_inputs[d], rng)
                cov_cross_dense.append(cov)  # [n_latent, n_data, n_ps_d]

            cov_cross_sparse = []
            for d in range(n_dim):
                rng = self.parameters["ranges_%d" % d].get_value()
                cov = self.covariance_matrix(
                    x_split[d], self.sparse_pseudo_inputs[d], rng)
                cov_cross_sparse.append(cov)

            cov_cross_full = self.rowwise_separable_matmul(
                cov_cross_sparse,
                _tf.eye(self.sparse_n_ps, dtype=_tf.float64,
                        batch_shape=[self.n_latent])
            )

            kron_correction = _tf.linalg.LinearOperatorKronecker([
                _tf.linalg.LinearOperatorFullMatrix(_tf.matmul(mat1, mat2))
                for mat1, mat2 in zip(self.cov_inv, self.cross_grid_cov)
            ])
            alpha_correction = kron_correction.matmul(alpha_dense)

            # latent prediction
            # [n_latent, dense_n_ps, 1]
            mu = self.rowwise_separable_matmul(cov_cross_dense, alpha_dense) \
                 + self.rowwise_separable_matmul(
                cov_cross_sparse, alpha_sparse - alpha_correction)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross_full, self.cov_smooth_inv)
                * cov_cross_full,
                axis=2, keepdims=False)  # [n_latent, n_data]
            var = point_var - explained_var
            var = _tf.maximum(var, 0.0)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_latent, self.sparse_n_ps, n_sim],
                seed=seed, dtype=_tf.float64
            )
            sims = _tf.matmul(cov_cross_full,
                              _tf.matmul(self.chol_r, rnd)) + mu

            return mu, var, sims, explained_var

    def kl_divergence(self, jitter=1e-9):
        self.refresh(jitter)

        with _tf.name_scope("decoupled_KL_divergence"):
            alpha_dense = self.parameters["alpha_dense"].get_value()
            alpha_sparse = self.parameters["alpha_sparse"].get_value()
            delta = self.parameters["delta"].get_value()

            kron_dense = _tf.linalg.LinearOperatorKronecker(
                [_tf.linalg.LinearOperatorFullMatrix(mat)
                 for mat in self.dense_cov]
            )
            pseudo_mean_dense = kron_dense.matmul(alpha_dense)

            kron_sparse = _tf.linalg.LinearOperatorKronecker(
                [_tf.linalg.LinearOperatorFullMatrix(mat)
                 for mat in self.cov]
            )
            pseudo_mean_sparse = kron_sparse.matmul(alpha_sparse)

            kron_cross = _tf.linalg.LinearOperatorKronecker(
                [_tf.linalg.LinearOperatorFullMatrix(
                    _tf.matmul(mat1, _tf.matmul(mat2, mat1), True, False))
                 for mat1, mat2 in zip(self.cross_grid_cov, self.cov_inv)]
            )
            pseudo_mean_correction = kron_cross.matmul(alpha_dense)

            fit = _tf.reduce_sum(alpha_dense * pseudo_mean_dense) \
                  + _tf.reduce_sum(alpha_sparse * pseudo_mean_sparse) \
                  - _tf.reduce_sum(alpha_dense * pseudo_mean_correction)

            cov_sparse = _tf.linalg.LinearOperatorKronecker(
                [_tf.linalg.LinearOperatorFullMatrix(mat)
                 for mat in self.cov]).to_dense()

            tr = _tf.reduce_sum(self.cov_smooth_inv * cov_sparse)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol)))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))
            kl = 0.5 * (- tr + fit + det_1 - det_2)

            return kl


class Separable(_GriddedLayer):
    def __init__(self, n_latent, inducing_points):
        super().__init__()
        if inducing_points.__class__.__name__ \
                not in ("Grid1D", "Grid2D", "Grid3D"):
            raise Exception("inducing_points must be a grid object")
        n_dim = inducing_points.n_dim

        self._n_latent = n_latent
        self.cov = [None] * n_dim
        self.cov_smooth = [None] * n_dim
        self.cov_smooth_chol = [None] * n_dim
        self.cov_smooth_inv = [None] * n_dim
        self.cov_inv = [None] * n_dim
        self.cov_chol = [None] * n_dim
        self.mat_r = [None] * n_dim
        self.chol_r = [None] * n_dim
        self.pseudo_inputs = [_tf.constant(g, _tf.float64)
                              for g in inducing_points.grid]

        self.n_ps = inducing_points.coordinates.shape[0]
        self.grid_size = inducing_points.grid_size
        self.parameters.update({
            "alpha_white": _gpr.RealParameter(
                _np.zeros([self.n_latent, self.n_ps, 1]) + 0.01,
                _np.zeros([self.n_latent, self.n_ps, 1]) - 10,
                _np.zeros([self.n_latent, self.n_ps, 1]) + 10
            )
        })

        for d in range(n_dim):
            n_ps = inducing_points.grid_size[d]
            step = inducing_points.step_size[d]
            dif = inducing_points.grid[d][-1] - inducing_points.grid[d][0]
            self.parameters.update({
                "delta_%d" % d: _gpr.PositiveParameter(
                    _np.ones([self.n_latent, n_ps]),
                    _np.ones([self.n_latent, n_ps]) * 0.001,
                    _np.ones([self.n_latent, n_ps]) + 10
                ),
                "ranges_%d" % d: _gpr.PositiveParameter(
                    # dif / _np.arange(1, self.n_latent + 1),
                    _np.ones([self.n_latent]) * step,
                    _np.ones([self.n_latent]) * step / 2,
                    _np.ones([self.n_latent]) * dif * 2,
                    fixed=False
                )
            })

        self._all_parameters += [v for v in self.parameters.values()]

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("separable_layer_refresh"):
            n_dim = len(self.cov)

            for d in range(n_dim):
                # prior
                ps = self.pseudo_inputs[d]
                rng = self.parameters["ranges_%d" % d].get_value()

                eye = _tf.eye(self.grid_size[d], dtype=_tf.float64,
                              batch_shape=[self.n_latent])
                cov_ps = self.covariance_matrix(ps, ps, rng)
                chol_k = _tf.linalg.cholesky(cov_ps + eye * jitter)
                k_inv = _tf.linalg.cholesky_solve(chol_k, eye)

                self.cov[d] = cov_ps
                self.cov_chol[d] = chol_k
                self.cov_inv[d] = k_inv

                # posterior
                delta = self.parameters["delta_%d" % d].get_value()
                delta_diag = _tf.linalg.diag(delta)
                self.cov_smooth[d] = self.cov[d] + delta_diag
                self.cov_smooth_chol[d] = _tf.linalg.cholesky(
                    self.cov_smooth[d] + eye * jitter)
                self.cov_smooth_inv[d] = _tf.linalg.cholesky_solve(
                    self.cov_smooth_chol[d], eye)
                self.chol_r[d] = _tf.linalg.cholesky(
                    self.cov_inv[d] - self.cov_smooth_inv[d] + eye * jitter)
                self.mat_r[d] = _tf.matmul(self.chol_r[d], self.chol_r[d],
                                           False, True)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        with _tf.name_scope("separable_layer_prediction"):
            self.refresh(jitter)

            x_split = _tf.unstack(x, axis=1)
            n_dim = len(self.cov)

            alpha_white = self.parameters["alpha_white"].get_value()
            kron_op_alpha = _tf.linalg.LinearOperatorKronecker(
                [_tf.linalg.LinearOperatorLowerTriangular(mat)
                 for mat in self.cov_chol]
            )
            alpha = kron_op_alpha.solve(alpha_white, adjoint=True)

            # covariances
            point_var = _tf.ones([self.n_latent, _tf.shape(x)[0]],
                                 dtype=_tf.float64)

            cov_cross = []
            for d in range(n_dim):
                rng = self.parameters["ranges_%d" % d].get_value()
                cov = self.covariance_matrix(
                    x_split[d], self.pseudo_inputs[d], rng)
                cov_cross.append(cov)  # [n_latent, n_data, n_ps_d]

            # latent prediction
            mu = self.rowwise_separable_matmul(
                cov_cross, alpha)  # [n_latent, n_ps, 1]

            explained_var = _tftools.prod_n(
                [_tf.reduce_sum(_tf.matmul(cov_cr, cov_inv) * cov_cr,
                                axis=2, keepdims=False)  # [n_latent, n_data]
                 for cov_cr, cov_inv in zip(cov_cross, self.cov_smooth_inv)])
            var = point_var - explained_var
            var = _tf.maximum(var, 0.0)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_ps, n_sim],
                seed=seed, dtype=_tf.float64
            )
            kron_op = _tf.linalg.LinearOperatorKronecker(
                [_tf.linalg.LinearOperatorLowerTriangular(mat)
                 for mat in self.chol_r]
            )
            sims = self.rowwise_separable_matmul(
                cov_cross, kron_op.matmul(rnd))
            sims = sims + mu

            return mu, var, sims, explained_var

    def kl_divergence(self, jitter=1e-9):
        with _tf.name_scope("separable_KL_divergence"):
            self.refresh(jitter)

            alpha_white = self.parameters["alpha_white"].get_value()

            tr = _tftools.prod_n(
                [_tf.reduce_sum(cov * mat_r)
                 for cov, mat_r in zip(self.cov, self.mat_r)])
            fit = _tf.reduce_sum(alpha_white**2)
            det_k = _tf.add_n(
                [2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(chol))) * self.n_ps / n
                 for chol, n in zip(self.cov_chol, self.grid_size)])
            det_r = _tf.add_n(
                [2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(chol))) * self.n_ps / n
                 for chol, n in zip(self.chol_r, self.grid_size)])
            const = _tf.constant(self.n_ps * self.n_latent, _tf.float64)
            kl = 0.5 * (tr + fit - const - det_k - det_r)

            return kl


class FullySeparable(_GriddedLayer):
    def __init__(self, n_latent, inducing_points, n_components=10):
        super().__init__()
        if inducing_points.__class__.__name__ \
                not in ("Grid1D", "Grid2D", "Grid3D"):
            raise Exception("inducing_points must be a grid object")
        n_dim = inducing_points.n_dim

        self._n_latent = n_latent
        self.cov = [None] * n_dim
        self.cov_smooth = [None] * n_dim
        self.cov_smooth_chol = [None] * n_dim
        self.cov_smooth_inv = [None] * n_dim
        self.cov_inv = [None] * n_dim
        self.cov_chol = [None] * n_dim
        self.mat_r = [None] * n_dim
        self.chol_r = [None] * n_dim
        self.pseudo_inputs = [_tf.constant(g, _tf.float64)
                              for g in inducing_points.grid]

        self.n_ps = inducing_points.coordinates.shape[0]
        self.grid_size = inducing_points.grid_size
        # self.parameters.update({
        #     "alpha_white": _gpr.RealParameter(
        #         _np.zeros([self.n_latent, self.n_ps, 1]) + 0.01,
        #         _np.zeros([self.n_latent, self.n_ps, 1]) - 10,
        #         _np.zeros([self.n_latent, self.n_ps, 1]) + 10
        #     )
        # })

        for d in range(n_dim):
            n_ps = inducing_points.grid_size[d]
            step = inducing_points.step_size[d]
            dif = inducing_points.grid[d][-1] - inducing_points.grid[d][0]
            self.parameters.update({
                "alpha_white_%d" % d: _gpr.RealParameter(
                    _np.random.normal(
                        scale=1e-3,
                        size=[self.n_latent, n_ps, n_components]
                    ),
                    # _np.zeros([self.n_latent, n_ps, n_components]) + 0.01,
                    _np.zeros([self.n_latent, n_ps, n_components]) - 10,
                    _np.zeros([self.n_latent, n_ps, n_components]) + 10,
                ),
                "delta_%d" % d: _gpr.PositiveParameter(
                    _np.ones([self.n_latent, n_ps]),
                    _np.ones([self.n_latent, n_ps]) * 0.001,
                    _np.ones([self.n_latent, n_ps]) + 10
                ),
                "ranges_%d" % d: _gpr.PositiveParameter(
                    # dif / _np.arange(1, self.n_latent + 1),
                    _np.ones([self.n_latent]) * step,
                    _np.ones([self.n_latent]) * step / 2,
                    _np.ones([self.n_latent]) * dif * 2,
                    fixed=False
                )
            })

        self._all_parameters += [v for v in self.parameters.values()]

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("separable_layer_refresh"):
            n_dim = len(self.cov)

            for d in range(n_dim):
                # prior
                ps = self.pseudo_inputs[d]
                rng = self.parameters["ranges_%d" % d].get_value()

                eye = _tf.eye(self.grid_size[d], dtype=_tf.float64,
                              batch_shape=[self.n_latent])
                cov_ps = self.covariance_matrix(ps, ps, rng)
                chol_k = _tf.linalg.cholesky(cov_ps + eye * jitter)
                k_inv = _tf.linalg.cholesky_solve(chol_k, eye)

                self.cov[d] = cov_ps
                self.cov_chol[d] = chol_k
                self.cov_inv[d] = k_inv

                # posterior
                delta = self.parameters["delta_%d" % d].get_value()
                delta_diag = _tf.linalg.diag(delta)
                self.cov_smooth[d] = self.cov[d] + delta_diag
                self.cov_smooth_chol[d] = _tf.linalg.cholesky(
                    self.cov_smooth[d] + eye * jitter)
                self.cov_smooth_inv[d] = _tf.linalg.cholesky_solve(
                    self.cov_smooth_chol[d], eye)
                self.chol_r[d] = _tf.linalg.cholesky(
                    self.cov_inv[d] - self.cov_smooth_inv[d] + eye * jitter)
                self.mat_r[d] = _tf.matmul(self.chol_r[d], self.chol_r[d],
                                           False, True)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        with _tf.name_scope("separable_layer_prediction"):
            self.refresh(jitter)

            x_split = _tf.unstack(x, axis=1)
            n_dim = len(self.cov)

            alphas = []
            for d in range(n_dim):
                alphas.append(_tf.linalg.solve(
                    self.cov_chol[d],
                    self.parameters["alpha_white_%d" % d].get_value()
                ))

            # covariances
            point_var = _tf.ones([self.n_latent, _tf.shape(x)[0]],
                                 dtype=_tf.float64)

            cov_cross = []
            for d in range(n_dim):
                rng = self.parameters["ranges_%d" % d].get_value()
                cov = self.covariance_matrix(
                    x_split[d], self.pseudo_inputs[d], rng)
                cov_cross.append(cov)  # [n_latent, n_data, n_ps_d]

            # latent prediction
            mu = _tftools.prod_n([_tf.matmul(cov, a)
                                  for cov, a in zip(cov_cross, alphas)])
            mu = _tf.reduce_sum(mu, axis=2, keepdims=True)

            explained_var = _tftools.prod_n(
                [_tf.reduce_sum(_tf.matmul(cov_cr, cov_inv) * cov_cr,
                                axis=2, keepdims=False)  # [n_latent, n_data]
                 for cov_cr, cov_inv in zip(cov_cross, self.cov_smooth_inv)])
            var = point_var - explained_var
            var = _tf.maximum(var, 0.0)

            rnd = [_tf.random.stateless_normal(
                shape=[self.n_latent, n, n_sim],
                seed=seed, dtype=_tf.float64
            ) for n in self.grid_size]
            sims = [_tf.matmul(cov, _tf.matmul(ch, r))
                    for cov, ch, r in zip(cov_cross, self.chol_r, rnd)]
            sims = _tftools.prod_n(sims) + mu

            return mu, var, sims, explained_var

    def kl_divergence(self, jitter=1e-9):
        with _tf.name_scope("separable_KL_divergence"):
            self.refresh(jitter)

            alpha_white = []
            for d in range(len(self.cov)):
                a = self.parameters["alpha_white_%d" % d].get_value()
                a = _tf.expand_dims(a, -1)  # [lat, ps, comp, 1]
                a = _tf.transpose(a, [0, 2, 1, 3])  # [lat, comp, ps, 1]
                alpha_white.append(a)
            alpha_white = _tf.linalg.LinearOperatorKronecker(
                [_tf.linalg.LinearOperatorFullMatrix(a) for a in alpha_white]
            )
            alpha_white = alpha_white.to_dense()
            alpha_white = _tf.reduce_sum(alpha_white, axis=1)  # [lat, ps, 1]

            tr = _tftools.prod_n(
                [_tf.reduce_sum(cov * mat_r)
                 for cov, mat_r in zip(self.cov, self.mat_r)])
            fit = _tf.reduce_sum(alpha_white ** 2)
            det_k = _tf.add_n(
                [2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(chol))) * self.n_ps / n
                 for chol, n in zip(self.cov_chol, self.grid_size)])
            det_r = _tf.add_n(
                [2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(chol))) * self.n_ps / n
                 for chol, n in zip(self.chol_r, self.grid_size)])
            const = _tf.constant(self.n_ps * self.n_latent, _tf.float64)
            kl = 0.5 * (tr + fit - const - det_k - det_r)

            return kl


class GenericWithTrend(Generic):
    def __init__(self, kernels, inducing_points,
                 fix_inducing_points=True):
        super().__init__(kernels, inducing_points, fix_inducing_points)

        self.trend_chol = None
        self.mat_a_inv = None
        self.inducing_trend = None
        self.beta = None
        self.n_trend = inducing_points.n_dim + 1

    def refresh(self, jitter=1e-9):
        super().refresh(jitter)

        with _tf.name_scope("trend_refresh"):
            ps = self.parameters["inducing_points"].get_value()
            ps_offset = self.add_offset(ps)

            self.inducing_trend = _tf.tile(
                ps_offset[None, :, :], [self.n_latent, 1, 1])
            mat_a = _tf.matmul(self.inducing_trend,
                               _tf.matmul(self.cov_inv, self.inducing_trend),
                               True)
            eye = _tf.eye(self.n_trend, dtype=_tf.float64)
            mat_a_inv = _tf.linalg.inv(mat_a + eye*jitter)
            self.mat_a_inv = mat_a_inv
            self.trend_chol = _tf.linalg.cholesky(mat_a_inv)

            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)
            self.beta = _tf.matmul(
                mat_a_inv, _tf.matmul(self.inducing_trend, alpha, True))

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction"):
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)
            ps = self.parameters["inducing_points"].get_value()

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance(x)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data]

            cov_cross = _tf.stack(
                [kernel.covariance_matrix(x, ps)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]

            # trend
            x_offset = self.add_offset(x)
            trend = _tf.tile(
                x_offset[None, :, :], [self.n_latent, 1, 1])

            trend_pred = trend - _tf.matmul(
                cov_cross, _tf.matmul(self.cov_inv, self.inducing_trend))

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]
            mu = mu + _tf.matmul(trend_pred, self.beta)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            trend_var = _tf.reduce_sum(
                _tf.matmul(trend_pred, self.mat_a_inv) * trend_pred,
                axis=2, keepdims=False
            )
            var = point_var - explained_var + trend_var
            var = _tf.maximum(var, 0.0)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_ps, n_sim],
                seed=seed, dtype=_tf.float64
            )
            sims = _tf.matmul(cov_cross, _tf.matmul(self.chol_r, rnd)) + mu

            rnd_2 = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_trend, n_sim],
                seed=[seed[0], seed[1] + 1], dtype=_tf.float64
            )
            sims = sims + _tf.matmul(
                trend_pred, _tf.matmul(self.trend_chol, rnd_2))

            return mu, var, sims, explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction_directions"):
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)
            ps = self.parameters["inducing_points"].get_value()

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance_d2(x, dir_x)
                 for kernel in self.kernels],
                axis=0)
            scale = _tf.stack(
                [kernel.point_variance(x)
                 for kernel in self.kernels],
                axis=0) / point_var
            cov_cross = _tf.stack(
                [_tf.transpose(kernel.covariance_matrix_d1(
                    ps, x, dir_x))
                    for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]
            cov_cross = cov_cross * _tf.sqrt(scale[:, :, None])

            # trend
            x_offset = self.add_offset_grad(dir_x)
            trend = _tf.tile(
                x_offset[None, :, :], [self.n_latent, 1, 1])
            trend = trend * _tf.sqrt(scale[:, :, None])

            trend_pred = trend - _tf.matmul(
                cov_cross, _tf.matmul(self.cov_inv, self.inducing_trend))

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]
            mu = mu + _tf.matmul(trend_pred, self.beta)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            trend_var = _tf.reduce_sum(
                _tf.matmul(trend_pred, self.mat_a_inv) * trend_pred,
                axis=2, keepdims=False
            )
            # var = point_var - explained_var + trend_var
            var = scale * point_var - explained_var + trend_var
            var = _tf.maximum(var, 0.0)

            return mu, var, explained_var


class GenericDirectional(Generic):
    def __init__(self, kernels, inducing_points, fix_inducing_points=True):
        super().__init__(kernels, inducing_points, fix_inducing_points)
        self.n_dim = inducing_points.n_dim

        self.scale = None

        self.parameters["alpha_white"] = _gpr.RealParameter(
                _np.zeros([self.n_latent,
                           self.n_ps * (1 + self.n_dim), 1]) + 0.1,
                _np.zeros([self.n_latent,
                           self.n_ps * (1 + self.n_dim), 1]) - 10,
                _np.zeros([self.n_latent,
                           self.n_ps * (1 + self.n_dim), 1]) + 10
            )
        self.parameters["delta"] = _gpr.PositiveParameter(
                _np.ones([self.n_latent, self.n_ps * (1 + self.n_dim)]) * 1e-3,
                _np.ones([self.n_latent, self.n_ps * (1 + self.n_dim)]) * 1e-6,
                _np.ones([self.n_latent, self.n_ps * (1 + self.n_dim)]) * 1e4
            )

        self._all_parameters = [v for v in self.parameters.values()]
        for kernel in self.kernels:
            self._all_parameters += kernel.all_parameters

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("latent_variable_refresh"):
            # prior
            ps = self.parameters["inducing_points"].get_value()

            cov_ps = []
            chol_k = []
            k_inv = []
            scale = []
            eye = _tf.eye(self.n_ps * (1 + self.n_dim), dtype=_tf.float64)
            for kernel in self.kernels:
                cov_ps_i = kernel.self_full_directional_covariance(ps)
                scale_i = _tf.sqrt(_tf.linalg.diag_part(cov_ps_i))
                scale_i = _tf.ones_like(scale_i)
                cov_ps_i = cov_ps_i / scale_i[:, None] / scale_i[None, :]

                cov_ps_i = cov_ps_i + eye * jitter
                chol_k_i = _tf.linalg.cholesky(cov_ps_i)
                k_inv_i = _tf.linalg.cholesky_solve(chol_k_i, eye)

                cov_ps.append(cov_ps_i)
                chol_k.append(chol_k_i)
                k_inv.append(k_inv_i)
                scale.append(scale_i)

            self.cov = _tf.stack(cov_ps, axis=0)
            self.cov_chol = _tf.stack(chol_k, axis=0)
            self.cov_inv = _tf.stack(k_inv, axis=0)
            self.scale = _tf.stack(scale, axis=0)

            # posterior
            delta = self.parameters["delta"].get_value()
            delta_diag = _tf.linalg.diag(delta)
            self.cov_smooth = self.cov + delta_diag
            self.cov_smooth_chol = _tf.linalg.cholesky(
                self.cov_smooth + eye * jitter)
            self.cov_smooth_inv = _tf.linalg.cholesky_solve(
                self.cov_smooth_chol, eye)
            self.chol_r = _tf.linalg.cholesky(
                self.cov_inv - self.cov_smooth_inv + eye * jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction"):
            ps = self.parameters["inducing_points"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance(x)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data]

            cov_cross = _tf.stack(
                [kernel.full_directional_covariance(x, ps)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]
            cov_cross = cov_cross / self.scale[:, None, :]

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            var = point_var - explained_var
            var = _tf.maximum(var, 0.0)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_ps * (1 + self.n_dim), n_sim],
                seed=seed, dtype=_tf.float64
            )
            sims = _tf.matmul(cov_cross, _tf.matmul(self.chol_r, rnd)) + mu

            return mu, var, sims, explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction_directions"):
            ps = self.parameters["inducing_points"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance_d2(x, dir_x)
                 for kernel in self.kernels],
                axis=0)
            # scale = _tf.stack(
            #     [kernel.point_variance(x)
            #      for kernel in self.kernels],
            #     axis=0) / point_var
            cov_cross = _tf.stack(
                [kernel.full_directional_covariance_d1(x, dir_x, ps)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]
            # cov_cross = cov_cross * _tf.sqrt(scale[:, :, None])
            cov_cross = cov_cross / self.scale[:, None, :]

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            var = point_var - explained_var
            # var = scale * point_var - explained_var
            var = _tf.maximum(var, 0.0)

            return mu, var, explained_var


class DirectionalWithTrend(GenericDirectional):
    def __init__(self, kernels, inducing_points, fix_inducing_points=True):
        super().__init__(kernels, inducing_points, fix_inducing_points)

        self.trend_chol = None
        self.mat_a_inv = None
        self.inducing_trend = None
        self.beta = None
        self.n_trend = self.n_dim + 1

    def refresh(self, jitter=1e-9):
        super().refresh(jitter)

        with _tf.name_scope("trend_refresh"):
            ps = self.parameters["inducing_points"].get_value()
            ps_offset = self.add_offset(ps)
            eye_dir = _tf.eye(self.n_dim, dtype=_tf.float64)
            ps_offset_grad = _tf.tile(eye_dir, [1, self.n_ps])
            ps_offset_grad = _tf.reshape(ps_offset_grad,
                                         [self.n_ps * self.n_dim, self.n_dim])
            ps_offset_grad = self.add_offset_grad(ps_offset_grad)
            ps_offset = _tf.concat([ps_offset, ps_offset_grad], axis=0)

            self.inducing_trend = _tf.tile(
                ps_offset[None, :, :], [self.n_latent, 1, 1])
            mat_a = _tf.matmul(self.inducing_trend,
                               _tf.matmul(self.cov_inv, self.inducing_trend),
                               True)
            eye = _tf.eye(self.n_trend, dtype=_tf.float64)
            mat_a_inv = _tf.linalg.inv(mat_a + eye*jitter)
            self.mat_a_inv = mat_a_inv
            self.trend_chol = _tf.linalg.cholesky(mat_a_inv)

            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)
            self.beta = _tf.matmul(
                mat_a_inv, _tf.matmul(self.inducing_trend, alpha, True))

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction"):
            ps = self.parameters["inducing_points"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance(x)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data]

            cov_cross = _tf.stack(
                [kernel.full_directional_covariance(x, ps)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]
            cov_cross = cov_cross / self.scale[:, None, :]

            # trend
            x_offset = self.add_offset(x)
            trend = _tf.tile(
                x_offset[None, :, :], [self.n_latent, 1, 1])

            trend_pred = trend - _tf.matmul(
                cov_cross, _tf.matmul(self.cov_inv, self.inducing_trend))

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]
            mu = mu + _tf.matmul(trend_pred, self.beta)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            trend_var = _tf.reduce_sum(
                _tf.matmul(trend_pred, self.mat_a_inv) * trend_pred,
                axis=2, keepdims=False
            )
            var = point_var - explained_var + trend_var
            var = _tf.maximum(var, 0.0)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_ps * (1 + self.n_dim), n_sim],
                seed=seed, dtype=_tf.float64
            )
            sims = _tf.matmul(cov_cross, _tf.matmul(self.chol_r, rnd)) + mu

            rnd_2 = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_trend, n_sim],
                seed=[seed[0], seed[1] + 1], dtype=_tf.float64
            )
            sims = sims + _tf.matmul(
                trend_pred, _tf.matmul(self.trend_chol, rnd_2))

            return mu, var, sims, explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("latent_layer_prediction_directions"):
            ps = self.parameters["inducing_points"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol, [0, 2, 1]), alpha_white)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance_d2(x, dir_x)
                 for kernel in self.kernels],
                axis=0)
            cov_cross = _tf.stack(
                [kernel.full_directional_covariance_d1(x, dir_x, ps)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]
            cov_cross = cov_cross / self.scale[:, None, :]

            # trend
            scale = _tf.stack(
                [kernel.point_variance(x)
                 for kernel in self.kernels],
                axis=0) / point_var
            x_offset = self.add_offset_grad(dir_x)
            trend = _tf.tile(
                x_offset[None, :, :], [self.n_latent, 1, 1])
            trend = trend * _tf.sqrt(scale[:, :, None])

            trend_pred = trend - _tf.matmul(
                cov_cross, _tf.matmul(self.cov_inv, self.inducing_trend))

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]
            mu = mu + _tf.matmul(trend_pred, self.beta)

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            trend_var = _tf.reduce_sum(
                _tf.matmul(trend_pred, self.mat_a_inv) * trend_pred,
                axis=2, keepdims=False
            )
            var = point_var - explained_var + trend_var
            var = _tf.maximum(var, 0.0)

            return mu, var, explained_var


class AutoRegressive(_LatentVariableLayer):
    def __init__(self, kernels, inducing_points, base_transform=None,
                 fix_inducing_points=True, extra_dims=1,
                 coordinates_kernel=_kr.Gaussian()):
        super().__init__()
        if not (isinstance(kernels, list) or isinstance(kernels, tuple)):
            kernels = [kernels]
        self.kernels = kernels
        self._n_latent = len(kernels)
        self.n_dim = inducing_points.n_dim
        self.extra_dims = extra_dims

        if base_transform is None:
            base_transform = _tr.AnisotropyARD(self.n_dim)
        self.base_transform = base_transform

        self.cov = [None] * (extra_dims + 1)
        self.cov_inv = [None] * (extra_dims + 1)
        self.cov_chol = [None] * (extra_dims + 1)
        self.cov_smooth = [None] * (extra_dims + 1)
        self.cov_smooth_chol = [None] * (extra_dims + 1)
        self.cov_smooth_inv = [None] * (extra_dims + 1)
        self.chol_r = None

        self.expanded_inducing_points = None
        self.expanded_std = None
        self.coordinates_alpha = [None] * extra_dims
        self.coordinates_kernel = coordinates_kernel

        self.n_ps = inducing_points.coordinates.shape[0]
        box = inducing_points.bounding_box
        self.parameters.update({
            "inducing_points": _gpr.RealParameter(
                inducing_points.coordinates,
                _np.tile(box[0, :], [self.n_ps, 1]),
                _np.tile(box[1, :], [self.n_ps, 1]),
                fixed=fix_inducing_points
            ),
            "alpha_white": _gpr.RealParameter(
                # _np.zeros([self.n_latent, self.n_ps, 1]) + 0.01,
                _np.random.normal(
                    scale=1e-3, size=[self.n_latent, self.n_ps, 1]),
                _np.zeros([self.n_latent, self.n_ps, 1]) - 10,
                _np.zeros([self.n_latent, self.n_ps, 1]) + 10
            ),
            "delta": _gpr.PositiveParameter(
                _np.ones([self.n_latent, self.n_ps]) * 1,
                _np.ones([self.n_latent, self.n_ps]) * 1e-6,
                _np.ones([self.n_latent, self.n_ps]) * 1e4
            ),
        })
        for i in range(extra_dims):
            self.parameters.update({
                "extra_alpha_white_%d" % i: _gpr.RealParameter(
                    _np.zeros([self.n_ps, 1]) + 0.001,
                    _np.zeros([self.n_ps, 1]) - 5,
                    _np.zeros([self.n_ps, 1]) + 5
                ),
                "extra_delta_%d" % i: _gpr.PositiveParameter(
                    _np.ones([self.n_ps]) * 10,
                    _np.ones([self.n_ps]) * 1e-6,
                    _np.ones([self.n_ps]) * 1e4
                ),
                "ranges_%d" % i: _gpr.PositiveParameter(
                    _np.ones([1, self.n_dim + i]),
                    _np.ones([1, self.n_dim + i]) * 1e-3,
                    _np.ones([1, self.n_dim + i]) * 10
                    # _np.ones([1, self.n_dim + i]),
                    # _np.array([[1e-3] * self.n_dim + [1]*i]),
                    # _np.array([[10] * self.n_dim + [1] * i])
                )
            })
        self.parameters.update({
            "ranges_%d" % extra_dims: _gpr.PositiveParameter(
                _np.ones([1, self.n_dim + extra_dims]),
                _np.ones([1, self.n_dim + extra_dims]) * 1e-3,
                _np.ones([1, self.n_dim + extra_dims]) * 10
                # _np.ones([1, self.n_dim + extra_dims]),
                # _np.array([[1e-3] * self.n_dim + [1] * extra_dims]),
                # _np.array([[10] * self.n_dim + [1] * extra_dims])
            )
        })
        self.parameters["ranges_0"].set_value(_np.ones([1, self.n_dim]))
        self.parameters["ranges_0"].fix()

        self._all_parameters += [v for v in self.parameters.values()]
        for kernel in kernels:
            self._all_parameters += kernel.all_parameters
        self._all_parameters += self.base_transform.all_parameters

    def set_kernel_limits(self, data):
        self.base_transform.set_limits(data)

    @staticmethod
    def covariance_matrix(x, y, rng_x, rng_y, kernel):
        with _tf.name_scope("autoregressive_covariance_matrix"):

            # [n_data, n_data, n_dim]
            dif = x[:, None, :] - y[None, :, :]

            avg_rng = 0.5 * (rng_x[:, None, :]**2 + rng_y[None, :, :]**2)

            dist = _tf.sqrt(_tf.reduce_sum(dif**2 / avg_rng, axis=-1))
            cov = kernel.kernelize(dist)

            # normalization
            det_avg = _tf.reduce_prod(avg_rng, axis=-1, keepdims=False)**(1/2)
            det_x = _tf.reduce_prod(rng_x**2, axis=-1, keepdims=True)**(1/4)
            det_y = _tf.reduce_prod(rng_y**2, axis=-1, keepdims=True)**(1/4)

            norm = det_x * _tf.transpose(det_y) / det_avg

            # output
            cov = cov * norm
            return cov

    def covariance_matrix_d1(self, x, y, dir_y, kernel, step=1e-3):
        with _tf.name_scope("autoregressive_covariance_matrix_d1"):
            x_exp, x_rng = self.predict_coordinates(x)
            y_exp_plus, y_rng_plus = self.predict_coordinates(
                y + 0.5*step*dir_y)
            y_exp_minus, y_rng_minus = self.predict_coordinates(
                y - 0.5 * step * dir_y)

            k1 = self.covariance_matrix(
                x_exp, y_exp_plus, x_rng, y_rng_plus, kernel
            )
            k2 = self.covariance_matrix(
                x_exp, y_exp_minus, x_rng, y_rng_minus, kernel
            )

            return (k1 - k2) / step

    def predict_coordinates(self, x):
        with _tf.name_scope("coordinate_prediction"):
            x = self.base_transform(x)
            x_std = _tf.zeros_like(x)

            for d in range(self.extra_dims):
                rng_d = self.parameters["ranges_%d" % d].get_value()

                total_rng_x = _tf.sqrt(x_std**2 + rng_d**2)
                total_rng_ps = _tf.sqrt(
                    self.expanded_std[:, :(self.n_dim + d)] ** 2 + rng_d ** 2)

                cov_cross = self.covariance_matrix(
                    x,
                    self.expanded_inducing_points[:, :(self.n_dim + d)],
                    total_rng_x,
                    total_rng_ps,
                    self.coordinates_kernel
                )

                mu = _tf.matmul(cov_cross, self.coordinates_alpha[d])

                explained_var = _tf.reduce_sum(
                    _tf.matmul(cov_cross, self.cov_smooth_inv[d]) * cov_cross,
                    axis=1, keepdims=True)
                var = 1 - explained_var
                var = _tf.maximum(var, 0.0)

                x = _tf.concat([x, mu], axis=1)
                x_std = _tf.concat([x_std, _tf.sqrt(var)], axis=1)

            rng_last = self.parameters[
                "ranges_%d" % self.extra_dims].get_value()
            x_std = _tf.sqrt(x_std ** 2 + rng_last ** 2)

            return x, x_std

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("autoregressive_refresh"):
            eye = _tf.eye(self.n_ps, dtype=_tf.float64)

            # coordinates
            ps = self.parameters["inducing_points"].get_value()

            ps = self.base_transform(ps)
            ps_std = _tf.zeros_like(ps)

            for d in range(self.extra_dims):
                rng_d = self.parameters["ranges_%d" % d].get_value()
                total_rng = _tf.sqrt(ps_std**2 + rng_d**2)

                self.cov[d] = self.covariance_matrix(
                    ps, ps, total_rng, total_rng, self.coordinates_kernel
                ) + eye * jitter
                self.cov_chol[d] = _tf.linalg.cholesky(self.cov[d])
                self.cov_inv[d] = _tf.linalg.cholesky_solve(
                    self.cov_chol[d], eye)

                alpha_white = self.parameters["extra_alpha_white_%d" % d]\
                    .get_value()
                extra_delta = self.parameters["extra_delta_%d" % d]\
                    .get_value()

                self.cov_smooth[d] = self.cov[d] + _tf.linalg.diag(extra_delta)
                self.cov_smooth_chol[d] = _tf.linalg.cholesky(
                    self.cov_smooth[d])
                self.cov_smooth_inv[d] = _tf.linalg.cholesky_solve(
                    self.cov_smooth_chol[d], eye)

                self.coordinates_alpha[d] = _tf.linalg.solve(
                    _tf.transpose(self.cov_chol[d]), alpha_white
                )
                new_coord = _tf.matmul(self.cov[d], self.coordinates_alpha[d])
                new_var = 1 - _tf.reduce_sum(
                    _tf.matmul(self.cov[d], self.cov_smooth_inv[d])
                    * self.cov[d],
                    axis=1, keepdims=True
                )

                ps = _tf.concat([ps, new_coord], axis=1)
                ps_std = _tf.concat([ps_std, _tf.sqrt(new_var)], axis=1)

            rng_last = self.parameters[
                "ranges_%d" % self.extra_dims].get_value()
            ps_std = _tf.sqrt(ps_std**2 + rng_last**2)

            self.expanded_inducing_points = ps
            self.expanded_std = ps_std

            # prior
            cov_ps = []
            chol_k = []
            k_inv = []
            for kernel in self.kernels:
                cov_ps_i = self.covariance_matrix(
                    ps, ps, ps_std, ps_std, kernel)

                cov_ps_i = cov_ps_i + eye * jitter
                chol_k_i = _tf.linalg.cholesky(cov_ps_i)
                k_inv_i = _tf.linalg.cholesky_solve(chol_k_i, eye)

                cov_ps.append(cov_ps_i)
                chol_k.append(chol_k_i)
                k_inv.append(k_inv_i)

            self.cov[-1] = _tf.stack(cov_ps, axis=0)
            self.cov_chol[-1] = _tf.stack(chol_k, axis=0)
            self.cov_inv[-1] = _tf.stack(k_inv, axis=0)

            # posterior
            delta = self.parameters["delta"].get_value()
            delta_diag = _tf.linalg.diag(delta)
            self.cov_smooth[-1] = self.cov[-1] + delta_diag
            self.cov_smooth_chol[-1] = _tf.linalg.cholesky(
                self.cov_smooth[-1] + eye * jitter)
            self.cov_smooth_inv[-1] = _tf.linalg.cholesky_solve(
                self.cov_smooth_chol[-1], eye)
            self.chol_r = _tf.linalg.cholesky(
                self.cov_inv[-1] - self.cov_smooth_inv[-1] + eye * jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("autoregressive_prediction"):
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol[-1], [0, 2, 1]), alpha_white)

            x, x_rng = self.predict_coordinates(x)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance(x)
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data]

            cov_cross = _tf.stack(
                [self.covariance_matrix(
                    x, self.expanded_inducing_points,
                    x_rng, self.expanded_std,
                    kernel)
                    for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]

            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv[-1]) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            var = point_var - explained_var
            var = _tf.maximum(var, 0.0)

            rnd = _tf.random.stateless_normal(
                shape=[self.n_latent, self.n_ps, n_sim],
                seed=seed, dtype=_tf.float64
            )
            sims = _tf.matmul(cov_cross, _tf.matmul(self.chol_r, rnd)) + mu

            return mu, var, sims, explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("autoregressive_prediction_directions"):
            ps = self.parameters["inducing_points"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()
            alpha = _tf.linalg.solve(
                _tf.transpose(self.cov_chol[-1], [0, 2, 1]), alpha_white)

            # covariances
            point_var = _tf.stack(
                [kernel.point_variance_d2(x, dir_x)
                 for kernel in self.kernels],
                axis=0)
            # scale = _tf.stack(
            #     [kernel.point_variance(x)
            #      for kernel in self.kernels],
            #     axis=0) / point_var
            cov_cross = _tf.stack(
                [_tf.transpose(self.covariance_matrix_d1(ps, x, dir_x, kernel))
                 for kernel in self.kernels],
                axis=0)  # [n_latent, n_data, n_ps]
            # cov_cross = cov_cross * _tf.sqrt(scale[:, :, None])

            # latent prediction
            mu = _tf.matmul(cov_cross, alpha)  # [n_latent, n_data, 1]

            # cov_inv = _tf.tile(self.cov_smooth_inv[0][None, :, :],
            #                    [self.n_latent, 1, 1])
            # explained_var = _tf.reduce_sum(
            #     _tf.matmul(cov_cross, cov_inv) * cov_cross,
            #     axis=2, keepdims=False)  # [n_latent, n_data]
            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_cross, self.cov_smooth_inv[-1]) * cov_cross,
                axis=2, keepdims=False)  # [n_latent, n_data]
            var = point_var - explained_var
            # var = scale * point_var - explained_var
            var = _tf.maximum(var, 0.0)

            return mu, var, explained_var

    def kl_divergence(self, jitter=1e-9):
        self.refresh(jitter)
        with _tf.name_scope("autoregressive_KL_divergence"):
            kl = _tf.constant(0.0, _tf.float64)

            for d in range(self.extra_dims):
                alpha_white = self.parameters["extra_alpha_white_%d" % d] \
                    .get_value()
                extra_delta = self.parameters["extra_delta_%d" % d] \
                    .get_value()

                tr = _tf.reduce_sum(self.cov_smooth_inv[d] * self.cov[d])
                fit = _tf.reduce_sum(alpha_white ** 2)
                det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(self.cov_smooth_chol[d])))
                det_2 = _tf.reduce_sum(_tf.math.log(extra_delta))
                kl_d = 0.5 * (- tr + fit + det_1 - det_2)
                kl = kl + kl_d

            delta = self.parameters["delta"].get_value()
            alpha_white = self.parameters["alpha_white"].get_value()

            tr = _tf.reduce_sum(self.cov_smooth_inv[-1] * self.cov[-1])
            fit = _tf.reduce_sum(alpha_white**2)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol[-1])))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))
            kl_last = 0.5 * (- tr + fit + det_1 - det_2)
            kl = kl + kl_last

            return kl


class Scale(_LatentVariableLayer):
    def __init__(self, latent_variable):
        super().__init__()
        self.latent_variable = latent_variable
        scale = _gpr.PositiveParameter(1, 1e-6, 10)
        self.parameters.update({"scale": scale})
        self._all_parameters.append(scale)
        self._all_parameters += latent_variable.all_parameters

    @property
    def n_latent(self):
        return self.latent_variable.n_latent

    def set_kernel_limits(self, data):
        self.latent_variable.set_kernel_limits(data)

    def refresh(self, jitter=1e-9):
        self.latent_variable.refresh(jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        mu, var, sims, explained_var = self.latent_variable.predict(
            x, n_sim, seed, jitter)

        scale = self.parameters["scale"].get_value()
        mu = mu * scale
        var = var * scale**2
        sims = sims * scale
        explained_var = explained_var * scale ** 2

        return mu, var, sims, explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        mu, var, explained_var = self.latent_variable.predict_directions(
            x, dir_x, jitter)

        scale = self.parameters["scale"].get_value()
        mu = mu * scale
        var = var * scale ** 2
        explained_var = explained_var * scale ** 2

        return mu, var, explained_var

    def kl_divergence(self, jitter=1e-9):
        return self.latent_variable.kl_divergence(jitter)


class Add(_LatentVariableLayer):
    def __init__(self, *latent_variables):
        super().__init__()
        self.latent_variables = latent_variables
        for v in self.latent_variables:
            self._all_parameters += v.all_parameters

    @property
    def n_latent(self):
        return self.latent_variables[0].n_latent

    def set_kernel_limits(self, data):
        for v in self.latent_variables:
            v.set_kernel_limits(data)

    def refresh(self, jitter=1e-9):
        for v in self.latent_variables:
            v.refresh(jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []

        for i, v in enumerate(self.latent_variables):
            mu, var, sims, explained_var = v.predict(
                x, n_sim, [seed[0] + i, seed[1]], jitter)
            all_mu.append(mu)
            all_var.append(var)
            all_sims.append(sims)
            all_explained_var.append(explained_var)

        all_mu = _tf.add_n(all_mu)
        all_var = _tf.add_n(all_var)
        all_sims = _tf.add_n(all_sims)
        all_explained_var = _tf.add_n(all_explained_var)

        return all_mu, all_var, all_sims, all_explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        all_mu = []
        all_var = []
        all_explained_var = []

        for i, v in enumerate(self.latent_variables):
            mu, var, explained_var = v.predict_directions(x, dir_x, jitter)
            all_mu.append(mu)
            all_var.append(var)
            all_explained_var.append(explained_var)

        all_mu = _tf.add_n(all_mu)
        all_var = _tf.add_n(all_var)
        all_explained_var = _tf.add_n(all_explained_var)

        return all_mu, all_var, all_explained_var

    def kl_divergence(self, jitter=1e-9):
        return _tf.add_n([v.kl_divergence(jitter)
                          for v in self.latent_variables])


class LinearCombination(Add):
    def __init__(self, *latent_variables):
        super().__init__(*latent_variables)
        weights = _gpr.CompositionalParameter(
            _np.ones(len(latent_variables)) / len(latent_variables))
        self.parameters.update({"weights": weights})
        self._all_parameters.append(weights)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []
        weights = self.parameters["weights"].get_value()

        for i, v in enumerate(self.latent_variables):
            mu, var, sims, explained_var = v.predict(
                x, n_sim, [seed[0] + i, seed[1]], jitter)
            all_mu.append(mu)
            all_var.append(var)
            all_sims.append(sims)
            all_explained_var.append(explained_var)

        all_mu = _tf.stack(all_mu, axis=-1)
        all_var = _tf.stack(all_var, axis=-1)
        all_sims = _tf.stack(all_sims, axis=-1)
        all_explained_var = _tf.stack(all_explained_var, axis=-1)

        all_mu = _tf.reduce_sum(all_mu * weights, axis=-1)
        all_var = _tf.reduce_sum(all_var * weights**2, axis=-1)
        all_sims = _tf.reduce_sum(all_sims * weights, axis=-1)
        all_explained_var = _tf.reduce_sum(all_explained_var * weights**2,
                                           axis=-1)

        return all_mu, all_var, all_sims, all_explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        all_mu = []
        all_var = []
        all_explained_var = []
        weights = self.parameters["weights"].get_value()

        for i, v in enumerate(self.latent_variables):
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


class Exponentiate(_LatentVariableLayer):
    def __init__(self, latent_variable):
        super().__init__()
        self.latent_variable = latent_variable
        # amp_mean = _gpr.RealParameter(0, -5, 5)
        # self.parameters.update({"amp_mean": amp_mean})
        # self._all_parameters.append(amp_mean)
        self._all_parameters += latent_variable.all_parameters

    @property
    def n_latent(self):
        return self.latent_variable.n_latent

    def set_kernel_limits(self, data):
        self.latent_variable.set_kernel_limits(data)

    def refresh(self, jitter=1e-9):
        self.latent_variable.refresh(jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        mu, var, sims, explained_var = self.latent_variable.predict(
            x, n_sim, seed, jitter)
        mu = mu[:, :, 0]  # + self.parameters["amp_mean"].get_value()

        with _tf.name_scope("exponential_prediction"):
            amp_mu = _tf.exp(mu) * (1 + 0.5 * var)
            amp_var = _tf.exp(2*mu) * var * (1 + var)
            amp_sims = _tf.exp(sims)
            amp_explained_var = _tf.exp(2 * mu) \
                                * (var + explained_var) \
                                * (1 + var + explained_var) \
                                - amp_var

            return amp_mu[:, :, None], amp_var, amp_sims, amp_explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        mu, var, explained_var = self.latent_variable.predict_directions(
            x, dir_x, jitter)
        mu = mu[:, :, 0]  # + self.parameters["amp_mean"].get_value()

        with _tf.name_scope("exponential_prediction_derivative"):
            amp_mu = _tf.exp(mu) * (1 + 0.5 * var)
            amp_var = _tf.exp(2 * mu) * var * (1 + var)
            amp_explained_var = _tf.exp(2 * mu) \
                                * (var + explained_var) \
                                * (1 + var + explained_var) \
                                - amp_var

            return amp_mu[:, :, None], amp_var, amp_explained_var

    def kl_divergence(self, jitter=1e-9):
        return self.latent_variable.kl_divergence(jitter)


class Multiply(_LatentVariableLayer):
    def __init__(self, *latent_variables):
        super().__init__()
        self.latent_variables = latent_variables
        for v in self.latent_variables:
            self._all_parameters += v.all_parameters

    @property
    def n_latent(self):
        return self.latent_variables[0].n_latent

    def set_kernel_limits(self, data):
        for v in self.latent_variables:
            v.set_kernel_limits(data)

    def refresh(self, jitter=1e-9):
        for v in self.latent_variables:
            v.refresh(jitter)

    def predict(self, x, n_sim=1, seed=(0, 0), jitter=1e-9):
        all_mu = []
        all_var = []
        all_sims = []
        all_explained_var = []

        for i, v in enumerate(self.latent_variables):
            mu, var, sims, explained_var = v.predict(
                x, n_sim, [seed[0] + i, seed[1]], jitter)
            all_mu.append(mu)
            all_var.append(var)
            all_sims.append(sims)
            all_explained_var.append(explained_var)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_sims = _tf.stack(all_sims, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)

        pred_mu = _tf.reduce_prod(all_mu, axis=0)
        pred_var = _tf.reduce_prod(all_mu[:, :, :, 0]**2 + all_var, axis=0) \
                   - _tf.reduce_prod(all_mu[:, :, :, 0]**2, axis=0)
        pred_sims = _tf.reduce_prod(all_sims, axis=0)

        pred_explained_var = \
            _tf.reduce_prod(all_mu[:, :, :, 0]**2 + all_var + all_explained_var,
                            axis=0) \
            - _tf.reduce_prod(all_mu[:, :, :, 0]**2, axis=0) \
            - pred_var

        return pred_mu, pred_var, pred_sims, pred_explained_var

    def predict_directions(self, x, dir_x, jitter=1e-9):
        all_mu = []
        all_var = []
        all_explained_var = []

        for i, v in enumerate(self.latent_variables):
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

    def kl_divergence(self, jitter=1e-9):
        return _tf.add_n([v.kl_divergence(jitter)
                          for v in self.latent_variables])
