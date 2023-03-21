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
# along with this program. If not, see <https://www.gnu.org/licenses/>.

__all__ = ["GP", "GPEnsemble", "StructuralField", "GPOptions",
           "VGPNetwork", "VGPNetworkEnsemble"]

import numpy as np

import geoml.data as _data
import geoml.parameter as _gpr
import geoml.likelihood as _lk
import geoml.warping as _warp
# import geoml.tftools as _tftools
import geoml

import numpy as _np
import tensorflow as _tf
import copy as _copy
import itertools as _iter
import warnings

import tensorflow_probability as _tfp
_tfd = _tfp.distributions


class _ModelOptions:
    def __init__(self, verbose=True, prediction_batch_size=20000,
                 training_batch_size=2000,
                 seed=1234):
        self.verbose = verbose
        self.training_batch_size = training_batch_size
        self.prediction_batch_size = prediction_batch_size
        self.seed = seed

    def batch_index(self, n_data, batch_size=None):
        if batch_size is None:
            batch_size = self.training_batch_size

        return _data.batch_index(n_data, batch_size)


class GPOptions(_ModelOptions):
    def __init__(self, verbose=True, prediction_batch_size=20000,
                 seed=1234, add_noise=False, jitter=1e-9,
                 training_batch_size=2000, training_samples=20):
        super().__init__(verbose, prediction_batch_size,
                         training_batch_size, seed)
        self.add_noise = add_noise
        self.jitter = jitter
        self.training_samples = training_samples


class _GPModel(_gpr.Parametric):
    def __init__(self, options=GPOptions()):
        super().__init__()
        self.options = options
        self._pre_computations = {}
        self._n_dim = None

    @property
    def n_dim(self):
        return self._n_dim


class GP(_GPModel):
    """
    Basic Gaussian process model.
    """
    def __init__(self, data, variable, covariance, warping=None,
                 directional_data=None, interpolation=False,
                 use_trend=False, options=GPOptions()):
        super().__init__(options)

        self.data = data
        self.variable = variable
        self.covariance = self._register(covariance)
        self.covariance.set_limits(data)

        if warping is None:
            warping = _warp.Identity()
        self.warping = self._register(warping)

        self.directional_data = directional_data
        self.use_trend = use_trend

        self._add_parameter("noise", _gpr.PositiveParameter(0.1, 1e-6, 10))
        if interpolation:
            self.parameters["noise"].set_value(1e-6)
            self.parameters["noise"].fix()

        self.training_log = []
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-2, 1, 0.999),
            amsgrad=True
        )

        self._pre_computations.update({
            "log_likelihood": _tf.Variable(_tf.constant(0.0, _tf.float64)),
        })

        self.cov = None
        self.cov_chol = None
        self.cov_inv = None
        self.scale = None
        self.alpha = None
        self.x = None
        self.y = None
        self.x_dir = None
        self.y_dir = None
        self.directions = None
        self.y_warped = None
        self.trend = None
        self.mat_a_inv = None
        self.trend_chol = None
        self.beta = None

    def __repr__(self):
        s = "Gaussian process model\n\n"
        s += "Variable: " + self.variable + "\n\n"
        s += "Kernel:\n"
        s += repr(self.covariance)
        s += "\nWarping:\n"
        s += repr(self.warping)
        return s

    def set_learning_rate(self, rate):
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(rate, 1, 0.999),
            amsgrad=True
        )

    def refresh(self, jitter=1e-9):
        keep = ~ _np.isnan(
            self.data.variables[self.variable].measurements.values)

        with _tf.name_scope("GP_refresh"):
            self.y = _tf.constant(self.data.variables[self.variable]
                                  .measurements.values[keep],
                                  _tf.float64)
            self.x = _tf.constant(self.data.coordinates[keep, :],
                                  _tf.float64)

            if self.directional_data is not None:
                self.x_dir = _tf.constant(self.directional_data.coordinates,
                                          _tf.float64)
                self.directions = _tf.constant(self.directional_data.directions,
                                               _tf.float64)

                cov = self.covariance.self_covariance_matrix(self.x)
                cov_d1 = self.covariance.covariance_matrix_d1(
                    self.x, self.x_dir, self.directions)
                cov_d2 = self.covariance.self_covariance_matrix_d2(
                    self.x_dir, self.directions)

                self.cov = _tf.concat([
                    _tf.concat([cov, cov_d1], axis=1),
                    _tf.concat([_tf.transpose(cov_d1), cov_d2], axis=1)
                ], axis=0)

                self.y_dir = _tf.constant(
                    self.directional_data.variables[self.variable]
                        .measurements.values,
                    _tf.float64
                )
                self.y_warped = _tf.concat([
                    self.warping.forward(self.y[:, None]),
                    self.y_dir[:, None]
                ], axis=0)

                eye = _tf.eye(_np.sum(keep) + self.directional_data.n_data,
                              dtype=_tf.float64)
                noise = _tf.concat([
                    _tf.ones([_np.sum(keep)], _tf.float64),
                    _tf.zeros([self.directional_data.n_data], _tf.float64)
                ], axis=0)
            else:
                self.cov = self.covariance.self_covariance_matrix(self.x)
                self.y_warped = self.warping.forward(self.y[:, None])

                eye = _tf.eye(_np.sum(keep), dtype=_tf.float64)
                noise = _tf.ones([_np.sum(keep)], _tf.float64)

            self.scale = _tf.sqrt(_tf.linalg.diag_part(self.cov))
            self.cov = self.cov / self.scale[:, None] / self.scale[None, :]
            noise = self.parameters["noise"].get_value() * noise
            noise = noise / self.scale**2

            self.cov_chol = _tf.linalg.cholesky(
                self.cov + _tf.linalg.diag(noise + jitter))
            self.cov_inv = _tf.linalg.cholesky_solve(self.cov_chol, eye)
            self.alpha = _tf.matmul(
                self.cov_inv, self.y_warped / self.scale[:, None])

            if self.use_trend:
                self.trend = _tf.concat([
                    _tf.ones([_np.sum(keep), 1], _tf.float64), self.x
                ], axis=1)

                if self.directional_data is not None:
                    trend_grad = _tf.concat([
                        _tf.zeros([self.directional_data.n_data, 1], _tf.float64),
                        self.directions
                    ], axis=1)
                    self.trend = _tf.concat([self.trend, trend_grad], axis=0)

                self.trend = self.trend / self.scale[:, None]
                mat_a = _tf.matmul(
                    self.trend, _tf.matmul(self.cov_inv, self.trend), True)
                eye = _tf.eye(self.data.n_dim + 1, dtype=_tf.float64)
                mat_a_inv = _tf.linalg.inv(mat_a + eye * jitter)
                self.mat_a_inv = mat_a_inv
                self.trend_chol = _tf.linalg.cholesky(mat_a_inv)
                self.beta = _tf.matmul(
                    mat_a_inv, _tf.matmul(self.trend, self.alpha, True))

    @_tf.function
    def log_likelihood(self, jitter=1e-9):
        self.refresh(jitter)

        with _tf.name_scope("GP_log_likelihood"):
            fit = -0.5 * _tf.reduce_sum(self.y_warped * self.alpha)
            det = - _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_chol))) \
                  - _tf.reduce_sum(_tf.math.log(self.scale))
            const = -0.5 * _tf.cast(_tf.shape(self.cov)[0], _tf.float64)\
                    * _np.log(2 * _np.pi)
            log_lik = fit + det + const

            y_derivative = self.warping.derivative(self.y)
            log_lik = log_lik + _tf.reduce_sum(_tf.math.log(y_derivative))

            if self.use_trend:
                det_2 = _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(self.trend_chol)))
                fit_2 = _tf.reduce_sum(
                    _tf.matmul(self.trend_chol,
                               _tf.matmul(self.trend, self.alpha, True),
                               True)**2)
                const_2 = 0.5 * _tf.constant(self.data.n_dim + 1, _tf.float64) \
                          * _np.log(2 * _np.pi)
                log_lik = log_lik + det_2 + fit_2 + const_2

            self._pre_computations["log_likelihood"].assign(log_lik)
            return log_lik

    @_tf.function
    def predict_raw(self, x_new, jitter=1e-9, quantiles=None,
                    probabilities=None):
        self.refresh(jitter)

        with _tf.name_scope("Prediction"):
            noise = self.parameters["noise"].get_value()

            # covariance
            cov_new = self.covariance.covariance_matrix(x_new, self.x)
            if self.directional_data is not None:
                cov_new_d1 = self.covariance.covariance_matrix_d1(
                    x_new, self.x_dir, self.directions)
                cov_new = _tf.concat([cov_new, cov_new_d1], axis=1)
            cov_new = cov_new / self.scale[None, :]

            # prediction
            mu = _tf.matmul(cov_new, self.alpha)

            point_var = self.covariance.point_variance(x_new)[:, None]
            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_new, self.cov_inv) * cov_new,
                axis=1, keepdims=True)
            var = _tf.maximum(point_var - explained_var, 0.0) + noise

            # trend
            if self.use_trend:
                trend_new = _tf.concat([
                    _tf.ones([_tf.shape(x_new)[0], 1], _tf.float64), x_new
                ], axis=1)

                trend_pred = trend_new - _tf.matmul(
                    cov_new, _tf.matmul(self.cov_inv, self.trend))
                mu = mu + _tf.matmul(trend_pred, self.beta)

                trend_var = _tf.reduce_sum(
                    _tf.matmul(trend_pred, self.mat_a_inv) * trend_pred,
                    axis=1, keepdims=True
                )
                var = var + trend_var

            # weights
            weights = (explained_var / (noise + 1e-6)) ** 2

            out = {"mean": _tf.squeeze(mu),
                   "variance": _tf.squeeze(var),
                   "weights": _tf.squeeze(weights)}

            # warping
            distribution = _tfd.Normal(mu, _tf.sqrt(var))

            def prob_fn(q):
                p = distribution.cdf(self.warping.forward(q))
                return p

            if quantiles is not None:
                prob = _tf.map_fn(prob_fn, quantiles)
                prob = _tf.transpose(prob)

                # single point case
                prob = _tf.cond(
                    _tf.less(_tf.rank(prob), 2),
                    lambda: _tf.expand_dims(prob, 0),
                    lambda: prob)

                out["probabilities"] = _tf.squeeze(prob)

            def quant_fn(p):
                q = self.warping.backward(distribution.quantile(p))
                return q

            if probabilities is not None:
                quant = _tf.map_fn(quant_fn, probabilities)
                quant = _tf.transpose(quant)

                # single point case
                quant = _tf.cond(
                    _tf.less(_tf.rank(quant), 2),
                    lambda: _tf.expand_dims(quant, 0),
                    lambda: quant)

                out["quantiles"] = _tf.squeeze(quant)

            return out

    def predict(self, newdata):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension.
            The object's variables are updated.
        """
        if self.data.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        if self.variable not in newdata.variables.keys():
            self.data.variables[self.variable].copy_to(newdata)
        prediction_input = self.data.variables[self.variable].prediction_input()

        # prediction in batches
        batch_id = self.options.batch_index(newdata.n_data,
                                            self.options.prediction_batch_size)
        n_batches = len(batch_id)

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(n_batches)), end="")

            output = self.predict_raw(
                _tf.constant(newdata.coordinates[batch], _tf.float64),
                jitter=self.options.jitter, **prediction_input)

            newdata.variables[self.variable].update(batch, **output)

        if self.options.verbose:
            print("\n")

    def train(self, max_iter=1000):
        model_variables = [pr.variable for pr in self._all_parameters
                           if not pr.fixed]

        def loss():
            return - self.log_likelihood(self.options.jitter)

        for i in range(max_iter):
            self.optimizer.minimize(loss, model_variables)

            for pr in self._all_parameters:
                pr.refresh()

            current_log_lik = self._pre_computations["log_likelihood"].numpy()
            self.training_log.append(current_log_lik)

            if self.options.verbose:
                print("\rIteration %s | Log-likelihood: %s" %
                      (str(i + 1), str(current_log_lik)), end="")

        if self.options.verbose:
            print("\n")


class VGPNetwork(_GPModel):
    """Vanilla VGP"""
    def __init__(self, data, variables, likelihoods,
                 latent_network,
                 directional_data=None,
                 options=GPOptions()):
        super().__init__(options=options)

        self.data = data
        self.latent_network = self._register(latent_network)

        if not (isinstance(likelihoods, (list, tuple))):
            likelihoods = [likelihoods]
        self.likelihoods = likelihoods
        self.lik_sizes = [lik.size for lik in likelihoods]
        for likelihood in likelihoods:
            self._register(likelihood)

        if not (isinstance(variables, (list, tuple))):
            variables = [variables]
        self.variables = variables
        self.var_lengths = [data.variables[v].length for v in variables]

        y, has_value = [], []
        for v in self.variables:
            y_v, h_v = data.variables[v].get_measurements()
            y.append(y_v)
            has_value.append(h_v)
        self.y = _np.concatenate(y, axis=1)
        self.has_value = _np.concatenate(has_value, axis=1)
        self.total_data = _np.sum(self.has_value)

        # initializing likelihoods
        for i, v in enumerate(self.variables):
            y, has_value = data.variables[v].get_measurements()
            has_value = _np.all(has_value == 1.0, axis=1)
            y = y[has_value, :]
            self.likelihoods[i].initialize(y)

        # directions
        self.directional_likelihood = _lk.GradientIndicator()
        # self.directional_likelihood = _lk.Gaussian()
        # self.directional_likelihood.parameters["noise"].set_value(1e-6)
        self.directional_data = directional_data
        self.total_data_dir = 0
        self.y_dir = None
        self.has_value_dir = None
        self.var_lengths_dir = None

        if directional_data is not None:
            if self.data.n_dim != directional_data.n_dim:
                raise ValueError("the directional data must have the"
                                 "same number of dimensions as the"
                                 "point data")

            self.var_lengths_dir = [1] * sum(self.var_lengths)

            y_dir, has_value_dir = [], []
            for v, s in zip(variables, self.var_lengths):
                if v in directional_data.variables.keys():
                    y_v, h_v = directional_data.variables[v].get_measurements()
                    y_dir.append(_np.tile(y_v, [1, s]))
                    has_value_dir.append(_np.tile(h_v, [1, s]))
                else:
                    y_dir.append(_np.zeros([directional_data.n_data, s]))
                    has_value_dir.append(_np.ones([directional_data.n_data, s]))
            y_dir = _np.concatenate(y_dir, axis=1)
            has_value_dir = _np.concatenate(has_value_dir, axis=1)

            self.y_dir = y_dir.copy()
            self.has_value_dir = has_value_dir.copy()
            self.total_data_dir = _np.sum(self.has_value_dir)

        # optimizer
        self.training_log = []
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-2, 1, 0.999),
            amsgrad=True
        )

        # intermediate tensors
        self.elbo = _tf.Variable(_tf.constant(0.0, _tf.float64))
        self.kl_div = _tf.Variable(_tf.constant(0.0, _tf.float64))

    def __repr__(self):
        s = "Variational Gaussian process model\n\n"
        s += "Variables:\n "
        for v, lik in zip(self.variables, self.likelihoods):
            s += "\t" + v + " (" + lik.__class__.__name__ + ")\n"
        s += "\nLatent layer:\n"
        s += repr(self.latent_network)
        return s

    def set_learning_rate(self, rate):
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(rate, 1, 0.999),
            amsgrad=True
        )

    @_tf.function
    def _training_elbo(self, x, y, has_value, training_inputs,
                       x_dir=None, directions=None, y_dir=None,
                       has_value_directions=None, x_var=None,
                       samples=20,
                       seed=0, jitter=1e-6):
        self.latent_network.refresh(jitter)

        # ELBO
        elbo = self._log_lik(x, y, has_value, training_inputs,
                             x_var=x_var, samples=samples, seed=seed)

        # ELBO for directions
        if x_dir is not None:
            elbo = elbo + self._log_lik_directions(
                x_dir, directions, y_dir, has_value_directions)

        # KL-divergence
        unique_nodes = self.latent_network.get_unique_parents()
        unique_nodes.append(self.latent_network)
        kl = _tf.add_n([node.kl_divergence() for node in unique_nodes])
        elbo = elbo - kl

        self.elbo.assign(elbo)
        self.kl_div.assign(kl)
        return elbo

    @_tf.function
    def _log_lik(self, x, y, has_value, training_inputs, x_var=None,
                 samples=20, seed=0):
        with _tf.name_scope("batched_elbo"):
            # prediction
            mu, var, sims, _, _ = self.latent_network.predict(
                x, x_var=x_var, n_sim=samples, seed=[seed, 0])

            mu = _tf.transpose(mu[:, :, 0])
            var = _tf.transpose(var)
            sims = _tf.transpose(sims, [1, 0, 2])

            # likelihood
            y_s = _tf.split(y, self.var_lengths, axis=1)
            mu = _tf.split(mu, self.lik_sizes, axis=1)
            var = _tf.split(var, self.lik_sizes, axis=1)
            hv = _tf.split(has_value, self.var_lengths, axis=1)
            sims = _tf.split(sims, self.lik_sizes, axis=1)

            elbo = _tf.constant(0.0, _tf.float64)
            for likelihood, mu_i, var_i, y_i, hv_i, sim_i, inp in zip(
                    self.likelihoods, mu, var, y_s,
                    hv, sims, training_inputs):
                elbo = elbo + likelihood.log_lik(
                    mu_i, var_i, y_i, hv_i, samples=sim_i, **inp)

            # batch weight
            batch_size = _tf.reduce_sum(has_value)
            elbo = elbo * self.total_data / batch_size

            return elbo

    @_tf.function
    def _log_lik_directions(self, x_dir, directions, y_dir, has_value):
        with _tf.name_scope("batched_elbo_directions"):
            # prediction
            mu, var, _ = self.latent_network.predict_directions(
                x_dir, directions)

            mu = _tf.transpose(mu[:, :, 0])
            var = _tf.transpose(var)

            # likelihood
            y_s = _tf.split(y_dir, self.var_lengths_dir, axis=1)
            mu = _tf.split(mu, self.var_lengths_dir, axis=1)
            var = _tf.split(var, self.var_lengths_dir, axis=1)
            hv = _tf.split(has_value, self.var_lengths_dir, axis=1)
            elbo = _tf.constant(0.0, _tf.float64)
            for mu_i, var_i, y_i, hv_i in zip(mu, var, y_s, hv):
                elbo = elbo + self.directional_likelihood.log_lik(
                    mu_i, var_i, y_i, hv_i)

            # batch weight
            batch_size = _tf.cast(_tf.shape(x_dir)[0], _tf.float64)
            elbo = elbo * self.total_data_dir / batch_size

            return elbo

    def train_full(self, max_iter=1000):
        training_inputs = [self.data.variables[v].training_input()
                           for v in self.variables]

        model_variables = self.get_unfixed_variables()

        def loss():
            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates, _tf.float64),
                    _tf.constant(self.y, _tf.float64),
                    _tf.constant(self.has_value, _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance(),
                                       _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed)
            else:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates, _tf.float64),
                    _tf.constant(self.y, _tf.float64),
                    _tf.constant(self.has_value, _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance(),
                                       _tf.float64),
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    y_dir=_tf.constant(self.y_dir, _tf.float64),
                    has_value_directions=_tf.constant(
                        self.has_value_dir, _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed)

        for i in range(max_iter):
            self.optimizer.minimize(loss, model_variables)

            for pr in self._all_parameters:
                pr.refresh()

            current_elbo = self.elbo.numpy()
            self.training_log.append(current_elbo)

            if self.options.verbose:
                print("\rIteration %s | ELBO: %s" %
                      (str(i+1), str(current_elbo)), end="")

        if self.options.verbose:
            print("\n")

    def train_svi(self, epochs=100):
        model_variables = self.get_unfixed_variables()

        def loss(idx):
            training_inputs = [
                self.data.variables[v].training_input(idx)
                for v in self.variables]

            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed
                )
            else:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    y_dir=_tf.constant(self.y_dir, _tf.float64),
                    has_value_directions=_tf.constant(
                        self.has_value_dir, _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed
                )

        _np.random.seed(self.options.seed)
        for i in range(epochs):
            current_elbo = []

            shuffled = _np.random.choice(
                self.data.n_data, self.data.n_data, replace=False)
            batches = self.options.batch_index(self.data.n_data)

            for batch in batches:
                self.optimizer.minimize(
                    lambda: loss(shuffled[batch]),
                    model_variables)

                for pr in self._all_parameters:
                    pr.refresh()

                current_elbo.append(self.elbo.numpy())
                self.training_log.append(current_elbo[-1])

            total_elbo = _np.mean(current_elbo)
            if self.options.verbose:
                print("\rEpoch %s | ELBO: %s" %
                      (str(i + 1), str(total_elbo)), end="")

        if self.options.verbose:
            print("\n")

    def train_window(self, start, step_size, n_steps, epochs=10):
        model_variables = self.get_unfixed_variables()

        if not isinstance(start, (list, tuple)):
            start = [start]
        if not isinstance(step_size, (list, tuple)):
            step_size = [step_size]
        if not isinstance(n_steps, (list, tuple)):
            n_steps = [n_steps]

        positions = [np.arange(i)*s for i, s in zip(n_steps, step_size)]
        combinations = _np.array(list(_iter.product(*positions)))
        combinations = combinations + _np.array(start)[None, :]

        # spatial index
        spatial_index = []
        for comb in combinations:
            box = _data.BoundingBox(comb, comb + _np.array(step_size))
            inside = box.contains_points(self.data.coordinates)
            if np.any(inside):
                spatial_index.append(_np.where(inside)[0])

        def loss(idx):
            training_inputs = [
                self.data.variables[v].training_input(idx)
                for v in self.variables]

            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed
                )
            else:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    y_dir=_tf.constant(self.y_dir, _tf.float64),
                    has_value_directions=_tf.constant(
                        self.has_value_dir, _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed
                )

        _np.random.seed(self.options.seed)
        for i in range(epochs):
            current_elbo = []

            for batch in spatial_index:
                self.optimizer.minimize(
                    lambda: loss(batch),
                    model_variables)

                for pr in self._all_parameters:
                    pr.refresh()

                current_elbo.append(self.elbo.numpy())
                self.training_log.append(current_elbo[-1])

            total_elbo = _np.mean(current_elbo)
            if self.options.verbose:
                print("\rEpoch %s | ELBO: %s" %
                      (str(i + 1), str(total_elbo)), end="")

        if self.options.verbose:
            print("\n")

    def train_svi_experts(self, global_epochs=10, epochs_per_expert=10):
        if not isinstance(self.latent_network, geoml.latent.Refine):
            raise Exception("the last network node must be a"
                            "Refine object")

        unique_params = set(self._all_parameters)
        network_params = set(self.latent_network.all_parameters)
        other_params = unique_params.difference(network_params)
        expert_params = []
        expert_variables = []
        for expert in self.latent_network.parents:
            expert_p = list(other_params.union(set(expert.all_parameters)))
            expert_variables.append([pr.variable for pr in expert_p
                                     if not pr.fixed])
            expert_params.append(expert_p)
        # model_variables = [pr.variable for pr in unique_params
        #                    if not pr.fixed]

        def loss(idx):
            training_inputs = [
                self.data.variables[v].training_input(idx)
                for v in self.variables]

            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed
                )
            else:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    y_dir=_tf.constant(self.y_dir, _tf.float64),
                    has_value_directions=_tf.constant(
                        self.has_value_dir, _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed
                )

        # spatial index
        spatial_index = []
        for expert in self.latent_network.parents:
            box = expert.root.bounding_box
            inside = box.contains_points(self.data.coordinates)
            spatial_index.append(_np.where(inside)[0])

        # main loop
        _np.random.seed(self.options.seed)
        for g in range(global_epochs):
            for i, expert in enumerate(self.latent_network.parents):
                n_data = len(spatial_index[i])

                for j in range(epochs_per_expert):
                    current_elbo = []

                    shuffled = _np.random.choice(n_data, n_data, replace=False)
                    batches = self.options.batch_index(n_data)

                    for batch in batches:
                        self.optimizer.minimize(
                            lambda: loss(spatial_index[i][shuffled[batch]]),
                            expert_variables[i])

                        for pr in expert_params[i]:
                            pr.refresh()

                        current_elbo.append(self.elbo.numpy())
                        self.training_log.append(current_elbo[-1])

                    total_elbo = float(_np.mean(current_elbo))
                    if self.options.verbose:
                        print("\rEpoch %d | Expert %d | "
                              "Expert epoch %d | ELBO: %f" %
                              (g + 1, i + 1, j + 1, total_elbo), end="")

        if self.options.verbose:
            print("\n")

    def train_random_search(self, max_iter=1000, step=0.01,
                            simultaneous_prop=0.01, memory=0.5,
                            patience=100, tol=0.1):
        training_inputs = [self.data.variables[v].training_input()
                           for v in self.variables]

        value, shape, position, min_val, max_val = self.get_parameter_values()
        amp = max_val - min_val
        n_val = value.size
        simultaneous_updates = int(n_val * simultaneous_prop)

        def loss():
            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates, _tf.float64),
                    _tf.constant(self.y, _tf.float64),
                    _tf.constant(self.has_value, _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance(),
                                       _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed)
            else:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates, _tf.float64),
                    _tf.constant(self.y, _tf.float64),
                    _tf.constant(self.has_value, _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance(),
                                       _tf.float64),
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    y_dir=_tf.constant(self.y_dir, _tf.float64),
                    has_value_directions=_tf.constant(
                        self.has_value_dir, _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed)

        loss()
        current_elbo = self.elbo.numpy()
        grad = _np.zeros_like(value)
        grad_memory = _np.zeros_like(value)
        base_value = value.copy()
        stagnation = 0
        num_resets = 0
        i = 0
        c = 0
        n_success = 1
        n_failure = 1
        while i < max_iter:
            c += 1
            order = self.options.batch_index(len(value), simultaneous_updates)
            for j in order:
                i += 1
                upd = _np.random.uniform(-step, step, len(j))
                grad[j] += upd + grad_memory[j]

                new_value = value.copy() + grad * amp
                new_value = _np.minimum(new_value, max_val)
                new_value = _np.maximum(new_value, min_val)
                self.update_parameters(new_value, shape, position)
                for pr in self._all_parameters:
                    pr.refresh()

                loss()
                new_elbo = self.elbo.numpy()

                dif = new_elbo - current_elbo
                if dif > 0:
                    if dif < tol:
                        stagnation += 1
                    else:
                        stagnation = 0
                        grad_memory[j] += grad[j] * memory * dif

                    current_elbo = new_elbo
                    value = new_value
                    n_success += 1
                else:
                    stagnation += 1
                    n_failure += 1
                    prop = n_success / (n_failure + n_success)
                    grad_memory[j] *= (1 - memory * prop)
                    # grad_memory[j] += grad[j] * memory * dif

                grad *= 0.0

                self.training_log.append(current_elbo)

                if stagnation > patience:
                    # step *= 0.5
                    patience *= 2
                    tol *= 0.25
                    simultaneous_prop *= 0.5
                    simultaneous_updates = int(
                        _np.ceil(n_val * simultaneous_prop))
                    stagnation = 0
                    num_resets += 1

                if self.options.verbose:
                    print(
                        "\rCycle %s | Iteration %s | ELBO: %s | Resets: %s     " %
                        (str(c + 1), str(i + 1), str(current_elbo), str(num_resets)),
                        end=""
                    )

        if self.options.verbose:
            print("\n")
            # print("Proportion: %s" % str(prop))

    @_tf.function
    def predict_raw(self, x_new, variable_inputs, x_var=None,
                    n_sim=1, seed=0, jitter=1e-6):
        self.latent_network.refresh(jitter)

        with _tf.name_scope("Prediction"):
            pred_mu, pred_var, pred_sim, pred_exp_var, _ = \
                self.latent_network.predict(
                    x_new, x_var=x_var, n_sim=n_sim, seed=[seed, 0]
                )

            pred_mu = _tf.transpose(pred_mu[:, :, 0])
            pred_var = _tf.transpose(pred_var)
            pred_sim = _tf.transpose(pred_sim, [1, 0, 2])
            pred_exp_var = _tf.transpose(pred_exp_var)

            pred_mu = _tf.split(pred_mu, self.lik_sizes, axis=1)
            pred_var = _tf.split(pred_var, self.lik_sizes, axis=1)
            pred_sim = _tf.split(pred_sim, self.lik_sizes, axis=1)
            pred_exp_var = _tf.split(pred_exp_var, self.lik_sizes, axis=1)

            output = []
            for mu, var, sim, exp_var, lik, v_inp in zip(
                    pred_mu, pred_var, pred_sim, pred_exp_var,
                    self.likelihoods, variable_inputs):
                output.append(lik.predict(mu, var, sim, exp_var, **v_inp))
            return output

    def predict(self, newdata, n_sim=20):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension.
            The object's variables are updated.
        n_sim : int
            Number of predictive samples to draw.
        """
        if self.data.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        # managing variables
        variable_inputs = []
        for v in self.variables:
            if v not in newdata.variables.keys():
                self.data.variables[v].copy_to(newdata)
            newdata.variables[v].allocate_simulations(n_sim)
            variable_inputs.append(self.data.variables[v].prediction_input())

        # prediction in batches
        batch_id = self.options.batch_index(
            newdata.n_data, batch_size=self.options.prediction_batch_size)
        n_batches = len(batch_id)

        # @_tf.function
        def batch_pred(x, x_var=None):
            out = self.predict_raw(
                x,
                variable_inputs,
                x_var=x_var,
                seed=self.options.seed,
                n_sim=n_sim,
                jitter=self.options.jitter
            )
            return out

        data_var = newdata.get_data_variance()

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(n_batches)), end="")

            output = batch_pred(
                _tf.constant(newdata.coordinates[batch], _tf.float64),
                _tf.constant(data_var[batch], _tf.float64))

            for v, upd in zip(self.variables, output):
                newdata.variables[v].update(batch, **upd)

        if self.options.verbose:
            print("\n")


class StructuralField(_GPModel):
    """Structural field modeling based on gradient data"""
    def __init__(self, tangents, covariance, normals=None, mean_vector=None,
                 options=GPOptions()):
        super().__init__(options=options)

        self.tangents = tangents
        self.normals = normals
        self.covariance = self._register(covariance)
        self.covariance.set_limits(self.tangents)

        if mean_vector is None:
            # initialized as vertical
            mean_vector = _np.zeros(self.tangents.n_dim)
            mean_vector[-1] = 1

        self.training_log = []
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-2, 1, 0.999),
            amsgrad=True
        )

        self._add_parameter(
            "mean_vector",
            _gpr.UnitColumnNormParameter(
                _np.array(mean_vector, ndmin=2).T,
                - _np.ones([self.tangents.n_dim, 1]),
                _np.ones([self.tangents.n_dim, 1]))
        )
        self._add_parameter("noise",
                            _gpr.PositiveParameter(1e-4, 1e-6, 10, fixed=True))

        if self.normals is None:
            # noise not used
            self.parameters["noise"].fix()

        # pre_computations
        self._pre_computations.update({
            "log_likelihood": _tf.Variable(_tf.constant(0.0, _tf.float64)),
        })

        self.cov = None
        self.cov_chol = None
        self.cov_inv = None
        self.scale = None
        self.alpha = None
        self.y = None
        self.all_coordinates = None
        self.all_directions = None

    def __repr__(self):
        s = "Gaussian process structural field model\n\n"
        s += "Kernel:\n"
        s += repr(self.covariance)
        return s

    def set_learning_rate(self, rate):
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(rate, 1, 0.999),
            amsgrad=True
        )

    def refresh(self, jitter=1e-9):
        with _tf.name_scope("structural_field_refresh"):
            mean_vector = self.parameters["mean_vector"].get_value()

            all_coordinates = self.tangents.coordinates
            all_directions = self.tangents.directions
            all_data = self.tangents.n_data

            noise = _tf.zeros([self.tangents.n_data], dtype=_tf.float64)
            y = _tf.zeros([self.tangents.n_data], dtype=_tf.float64)

            if self.normals is not None:
                all_coordinates = _np.concatenate([
                    all_coordinates, self.normals.coordinates
                ], axis=0)
                all_directions = _np.concatenate([
                    all_directions, self.normals.directions
                ], axis=0)
                all_data += self.normals.n_data

                noise = _tf.concat([
                    noise,
                    _tf.ones([self.normals.n_data], dtype=_tf.float64)
                ], axis=0) * self.parameters["noise"].get_value()

                y = _tf.concat([
                    y, _tf.ones([self.normals.n_data], dtype=_tf.float64)
                ], axis=0)

            self.all_coordinates = _tf.constant(all_coordinates, _tf.float64)
            self.all_directions = _tf.constant(all_directions, _tf.float64)

            self.cov = self.covariance.self_covariance_matrix_d2(
                self.all_coordinates, self.all_directions
            )
            self.scale = _tf.reduce_max(_tf.linalg.diag_part(self.cov))
            self.cov = self.cov / self.scale

            eye = _tf.eye(all_data, dtype=_tf.float64)
            noise = _tf.linalg.diag(noise + jitter)

            self.cov_chol = _tf.linalg.cholesky(self.cov + noise)
            self.cov_inv = _tf.linalg.cholesky_solve(self.cov_chol, eye)

            y = y[:, None] - _tf.matmul(all_directions, mean_vector)
            # y = y / _tf.sqrt(self.scale)
            self.alpha = _tf.matmul(self.cov_inv, y)
            self.y = y

    @_tf.function
    def log_likelihood(self, jitter=1e-9):
        self.refresh(jitter)

        with _tf.name_scope("structural_field_log_likelihood"):
            fit = -0.5 * _tf.reduce_sum(self.y * self.alpha)
            det = - _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_chol)))
            const = -0.5 * _tf.constant(
                self.tangents.n_data * self.tangents.n_dim * _np.log(2*_np.pi),
                _tf.float64)
            log_lik = fit + det + const

            self._pre_computations["log_likelihood"].assign(log_lik)
            return log_lik

    def train(self, max_iter=1000):
        model_variables = [pr.variable for pr in self._all_parameters
                           if not pr.fixed]

        def loss():
            return - self.log_likelihood(self.options.jitter)

        for i in range(max_iter):
            self.optimizer.minimize(loss, model_variables)

            for pr in self._all_parameters:
                pr.refresh()

            current_log_lik = self._pre_computations["log_likelihood"].numpy()
            self.training_log.append(current_log_lik)

            if self.options.verbose:
                print("\rIteration %s | Log-likelihood: %s" %
                      (str(i + 1), str(current_log_lik)), end="")

        if self.options.verbose:
            print("\n")

    @_tf.function
    def predict_raw(self, x_new, jitter=1e-9):
        self.refresh(jitter)

        with _tf.name_scope("Prediction"):
            mean_vector = self.parameters["mean_vector"].get_value()

            # mean of field
            cov_new = self.covariance.covariance_matrix_d1(
                x_new, self.all_coordinates, self.all_directions) / self.scale

            mu = _tf.matmul(cov_new, self.alpha)
            mu = mu + _tf.matmul(x_new, mean_vector)

            # variance of gradient along mean direction
            cov_new = self.covariance.covariance_matrix_d2(
                x_new, self.all_coordinates,
                _tf.transpose(mean_vector), self.all_directions
            ) / self.scale

            point_var = self.covariance.point_variance(x_new)[:, None]
            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_new, self.cov_inv) * cov_new,
                axis=1, keepdims=True)
            var = _tf.maximum(point_var - explained_var, 0.0)

            return mu, var

    @_tf.function
    def predict_raw_directions(self, x_new, x_new_dir, jitter=1e-9):
        self.refresh(jitter)

        with _tf.name_scope("Prediction"):
            mean_vector = self.parameters["mean_vector"].get_value()

            cov_new = self.covariance.covariance_matrix_d2(
                x_new, self.all_coordinates,
                x_new_dir, self.all_directions) / self.scale

            mu = _tf.matmul(cov_new, self.alpha)
            mu = mu + _tf.matmul(x_new_dir, mean_vector)

            point_var = self.covariance.point_variance(x_new)[:, None]
            explained_var = _tf.reduce_sum(
                _tf.matmul(cov_new, self.cov_inv) * cov_new,
                axis=1, keepdims=True)
            var = _tf.maximum(point_var - explained_var, 0.0)

            return mu, var

    def predict(self, newdata, variable):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension.
            The object's variables are updated.
        variable : str
            Name of output variable.
        """
        if self.tangents.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        # managing variables
        newdata.add_continuous_variable(variable)

        # prediction in batches
        batch_id = self.options.batch_index(newdata.n_data)
        n_batches = len(batch_id)

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(n_batches)), end="")

            if isinstance(newdata, _data.DirectionalData):
                mu, var = self.predict_raw_directions(
                    _tf.constant(newdata.coordinates[batch], _tf.float64),
                    _tf.constant(newdata.directions[batch], _tf.float64),
                    jitter=self.options.jitter
                )
            else:
                mu, var = self.predict_raw(
                    _tf.constant(newdata.coordinates[batch], _tf.float64),
                    jitter=self.options.jitter)
            output = {"mean": _tf.squeeze(mu),
                      "variance": _tf.squeeze(var)}

            newdata.variables[variable].update(batch, **output)

        if self.options.verbose:
            print("\n")


class _EnsembleModel(_GPModel):
    def __init__(self, options=GPOptions):
        super().__init__(options)
        self.models = None

    def set_learning_rate(self, rate):
        for model in self.models:
            model.set_learning_rate(rate)

    @staticmethod
    def combine(outputs):
        raise NotImplementedError


class GPEnsemble(_EnsembleModel):
    def __init__(self, data, variable, covariance, warping=None, tangents=None,
                 use_trend=False, options=GPOptions()):
        super().__init__(options)
        if not isinstance(data, (tuple, list)):
            raise ValueError("data must be a list or tuple containing"
                             "data objects")
        if tangents is None:
            tangents = [None for _ in data]
        elif not isinstance(tangents, (tuple, list)):
            raise ValueError("tangents must be a list or tuple containing"
                             "data objects or None")

        dims = set([d.n_dim for d in data])
        if len(dims) != 1:
            raise Exception("all data objects must have the same dimension")
        self._n_dim = list(dims)[0]

        self.models = [GP(
            data=d,
            variable=variable,
            covariance=_copy.deepcopy(covariance),
            warping=_copy.deepcopy(warping),
            directional_data=t,
            use_trend=use_trend,
            options=options)
            for d, t in zip(data, tangents)]
        for model in self.models:
            self._register(model)

        self.variable = variable

    def __repr__(self):
        s = "Gaussian process ensemble\n\n" \
            "Models: %d\n\n" % len(self.models)
        s += "Variable: " + self.variable + "\n"
        return s

    def train(self, max_iter=1000):
        for i, model in enumerate(self.models):
            if self.options.verbose:
                print("Training model %d of %d" % (i + 1, len(self.models)))

            model.train(max_iter=max_iter)

    def predict(self, newdata):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension.
            The object's variables are updated.
        """
        if self.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        if self.variable not in newdata.variables.keys():
            self.models[0].data.variables[self.variable].copy_to(newdata)
        prediction_input = self.models[0].data \
            .variables[self.variable].prediction_input()

        # prediction in batches
        batch_id = self.options.batch_index(newdata.n_data,
                                            self.options.prediction_batch_size)
        n_batches = len(batch_id)

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(n_batches)), end="")

            outputs = [model.predict_raw(
                _tf.constant(newdata.coordinates[batch], _tf.float64),
                jitter=self.options.jitter, **prediction_input)
                for model in self.models]

            output = self.combine(outputs)

            newdata.variables[self.variable].update(batch, **output)

        if self.options.verbose:
            print("\n")

    @staticmethod
    def combine(outputs):
        weights = _tf.stack([out["weights"] for out in outputs], axis=1)
        # weights = 1 / (1 - weights + 1e-6)
        weights = weights / _tf.reduce_sum(weights, axis=1, keepdims=True)

        mu = _tf.stack([out["mean"] for out in outputs], axis=1)
        mu = _tf.reduce_sum(weights * mu, axis=1)

        var = _tf.stack([out["variance"] for out in outputs], axis=1)
        var = _tf.reduce_sum(weights**2 * var, axis=1)

        combined = {"mean": mu, "variance": var}

        if "probabilities" in outputs[0].keys():
            prob = _tf.stack([out["probabilities"] for out in outputs], axis=2)
            prob = _tf.reduce_sum(weights[:, None, :] * prob, axis=2)
            combined["probabilities"] = prob

        if "quantiles" in outputs[0].keys():
            quant = _tf.stack([out["quantiles"] for out in outputs], axis=2)
            quant = _tf.reduce_sum(weights[:, None, :] * quant, axis=2)
            combined["quantiles"] = quant

        return combined


class VGPNetworkEnsemble(_EnsembleModel):
    def __init__(self, data, variables, likelihoods, latent_networks,
                 directional_data=None, options=GPOptions()):
        super().__init__(options)
        if not isinstance(data, (tuple, list)):
            raise ValueError("data must be a list or tuple containing"
                             "data objects")
        if not isinstance(latent_networks, (tuple, list)):
            raise ValueError("latent_trees must be a list or tuple containing"
                             "latent variable objects")
        if directional_data is None:
            directional_data = [None for _ in data]
        elif not isinstance(directional_data, (tuple, list)):
            raise ValueError("directional_data must be a list or tuple"
                             "containing data objects or None")

        dims = set([d.n_dim for d in data])
        if len(dims) != 1:
            raise Exception("all data objects must have the same dimension")
        self._n_dim = list(dims)[0]

        self.models = [VGPNetwork(
            data=d,
            variables=variables,
            # likelihoods=_copy.deepcopy(likelihoods),
            likelihoods=lik,
            latent_network=l,
            directional_data=dd,
            options=options)
            for d, l, dd, lik in zip(
                data, latent_networks, directional_data, likelihoods)]
        for model in self.models:
            self._register(model)

        if not (isinstance(variables, (list, tuple))):
            variables = [variables]
        self.variables = variables

    def __repr__(self):
        s = "Gaussian process ensemble\n\n" \
            "Models: %d\n\n" % len(self.models)
        s += "Variables:\n "
        for v, lik in zip(self.variables, self.models[0].likelihoods):
            s += "\t" + v + " (" + lik.__class__.__name__ + ")\n"
        return s

    # def train_full(self, cycles=10, max_iter_per_model=100):
    #     for c in range(cycles):
    #         for i, model in enumerate(self.models):
    #             if self.options.verbose:
    #                 print("Cycle %d of %d - training model %d of %d" %
    #                       (c + 1, cycles, i + 1, len(self.models)))
    #
    #             model.train_full(max_iter=max_iter_per_model)
    #
    # def train_svi(self, cycles=10, epochs_per_model=10):
    #     for c in range(cycles):
    #         for i, model in enumerate(self.models):
    #             if self.options.verbose:
    #                 print("Cycle %d of %d - training model %d of %d" %
    #                       (c + 1, cycles, i + 1, len(self.models)))
    #
    #             model.train_svi(epochs=epochs_per_model)

    def train_full(self, max_iter=1000):
        for model in self.models:
            model.train_full(max_iter)

    def train_svi(self, epochs=100):
        for model in self.models:
            model.train_svi(epochs)

    def predict(self, newdata, n_sim=20):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension.
            The object's variables are updated.
        n_sim : int
            Number of predictive samples to draw.
        """
        if self.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        # managing variables
        variable_inputs = []
        for v in self.variables:
            if v not in newdata.variables.keys():
                self.models[0].data.variables[v].copy_to(newdata)
            newdata.variables[v].allocate_simulations(n_sim)
            variable_inputs.append(
                self.models[0].data.variables[v].prediction_input())

        # prediction in batches
        batch_id = self.options.batch_index(
            newdata.n_data, batch_size=self.options.prediction_batch_size)
        n_batches = len(batch_id)

        def batch_pred(model, x):
            out = model.predict_raw(
                x,
                variable_inputs,
                seed=self.options.seed,
                n_sim=n_sim,
                jitter=self.options.jitter
            )
            return out

        @_tf.function
        def combined_pred(x):
            outputs = [batch_pred(model, x) for model in self.models]
            return self.combine(outputs)

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(n_batches)), end="")

            # outputs = [batch_pred(
            #     model,
            #     _tf.constant(newdata.coordinates[batch], _tf.float64))
            #     for model in self.models]
            #
            # output = self.combine(outputs)

            output = combined_pred(
                _tf.constant(newdata.coordinates[batch], _tf.float64))

            for v, upd in zip(self.variables, output):
                newdata.variables[v].update(batch, **upd)

        if self.options.verbose:
            print("\n")

    @_tf.function
    def combine(self, outputs):
        combined = [{} for _ in self.variables]
        for i, variable in enumerate(self.variables):
            var_keys = outputs[0][i].keys()

            weights = _tf.stack([out[i]["weights"] for out in outputs], axis=1)
            weights = weights + 1e-6
            weights = weights / _tf.reduce_sum(weights, axis=1, keepdims=True)

            for key in var_keys:
                if key != "weights":
                    tensor = _tf.stack([out[i][key] for out in outputs], axis=1)
                    if "variance" in key:
                        w = weights**2
                    else:
                        w = weights

                    w = _tf.cond(_tf.greater_equal(_tf.rank(tensor), 3),
                                 lambda: _tf.expand_dims(w, axis=-1),
                                 lambda: w)
                    w = _tf.cond(_tf.greater_equal(_tf.rank(tensor), 4),
                                 lambda: _tf.expand_dims(w, axis=-1),
                                 lambda: w)

                    tensor = _tf.reduce_sum(w * tensor, axis=1)
                    combined[i][key] = tensor

        return combined


class Normalizer(_GPModel):
    def __init__(self, warping, options=GPOptions()):
        super().__init__(options)
        self.warping = self._register(warping)

        self.training_log = []
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-1, 1, 0.99),
            amsgrad=True
        )

        self.objective = _tf.Variable(_tf.constant(0.0, _tf.float64))

    def normalize(self, x, max_iter=250):
        if len(self.all_parameters) == 0:
            warnings.warn("No trainable parameters.")
            return None

        model_variables = self.get_unfixed_variables()
        if len(model_variables) == 0:
            warnings.warn("All parameters are fixed. Unfix one or more"
                          "parameters to continue.")
            return None

        self.warping.initialize(x)

        def loss():
            x_warp = self.warping.forward(x)
            mean = _tf.reduce_mean(x_warp)
            var = _tf.math.reduce_variance(x_warp)
            std = _tf.sqrt(var)
            x_derivative = self.warping.derivative(x)
            log_derivative = _tf.math.log(x_derivative)

            kl = _tf.math.log(std) + (1 + mean ** 2) / (2 * var) - 0.5
            density = _tf.reduce_mean(-x_warp**2 + log_derivative)
            obj = density - kl

            self.objective.assign(obj)
            return -obj

        for i in range(max_iter):
            self.optimizer.minimize(loss, model_variables)

            for pr in self._all_parameters:
                pr.refresh()

            current_elbo = self.objective.numpy()
            self.training_log.append(current_elbo)

            if self.options.verbose:
                print("\rIteration %s | Objective: %s" %
                      (str(i + 1), str(current_elbo)), end="")


class VGPMultiscale(_GPModel):
    def __init__(self, data, variables, likelihoods,
                 latent_network_generators,  # list of functions
                 search_blocks,  # list of block objects
                 prediction_targets,  # list of data/grid objects
                 directional_data=None,
                 n_sims=50, minimum_data=10,
                 options=GPOptions()):
        super().__init__(options)
        self._n_dim = data.n_dim

        self.data = data
        self.directional_data = directional_data
        self.n_sims = n_sims
        self.minimum_data = minimum_data

        if not (isinstance(prediction_targets, (list, tuple))):
            prediction_targets = [prediction_targets]
        self.prediction_targets = prediction_targets

        if not (isinstance(likelihoods, (list, tuple))):
            likelihoods = [likelihoods]
        self.likelihoods = likelihoods
        self.lik_sizes = [lik.size for lik in likelihoods]

        if not (isinstance(variables, (list, tuple))):
            variables = [variables]
        self.variables = variables
        self.var_lengths = [data.variables[v].length for v in variables]

        y, has_value = [], []
        for v in self.variables:
            y_v, h_v = data.variables[v].get_measurements()
            y.append(y_v)
            has_value.append(h_v)
        self.y = _np.concatenate(y, axis=1)
        self.has_value = _np.concatenate(has_value, axis=1)
        self.total_data = _np.sum(self.has_value)

        self.latent_network_generators = latent_network_generators
        self.search_blocks = search_blocks

        size = _np.sum(self.lik_sizes)
        self.running_mean = _np.zeros([size, data.n_data, 1])
        self.running_variance = _np.ones([size, data.n_data])
        self.running_explained_variance = _np.zeros([size, data.n_data])
        self.running_sims = _np.zeros([size, data.n_data, n_sims])

        self.predicted_running_mean = [
            _np.zeros([size, t.n_data, 1])
            for t in prediction_targets
        ]
        self.predicted_running_variance = [
            _np.ones([size, t.n_data])
            for t in prediction_targets
        ]
        self.predicted_running_explained_variance = [
            _np.zeros([size, t.n_data])
            for t in prediction_targets
        ]
        self.predicted_running_sims = [
            _np.zeros([size, t.n_data, n_sims])
            for t in prediction_targets
        ]

        # intermediate tensors
        self.elbo = _tf.Variable(_tf.constant(0.0, _tf.float64))
        self.kl_div = _tf.Variable(_tf.constant(0.0, _tf.float64))
        self.total_kl_div = 0.0

        # training log
        self.training_log = [
            _np.empty(b.grid_size, _np.object)
            for b in self.search_blocks
        ]

    @staticmethod
    def refine(mu, var, sims, explained_var,
               prev_mu, prev_var, prev_sims, prev_explained_var):
        sims = sims - mu
        prev_sims = prev_sims - prev_mu

        w1 = explained_var / (var + 1e-6) + 1e-6
        w2 = prev_explained_var / (prev_var + 1e-6) + 1e-6

        new_mu = mu + prev_mu
        new_var = (w1 * var + w2 * prev_var) / (w1 + w2)
        new_exp_var = (w1 * explained_var + w2 * prev_explained_var) \
                      / (w1 + w2)

        w1, w2 = w1[:, :, None], w2[:, :, None]
        new_sims = (w1 * sims + w2 * prev_sims) / (w1 + w2)
        new_sims = new_sims + new_mu

        return new_mu, new_var, new_sims, new_exp_var

    @_tf.function
    def _log_lik(self, x, y, has_value, training_inputs,
                 prev_mu, prev_var, prev_sims, prev_exp_var, network,
                 x_var=None, seed=0):
        with _tf.name_scope("batched_elbo"):
            # prediction
            mu, var, sims, explained_var, influence = network.predict(
                x, x_var=x_var, n_sim=self.n_sims, seed=[seed, 0])

            mu, var, sims, _ = self.refine(
                mu, var, sims, explained_var,
                prev_mu, prev_var, prev_sims, prev_exp_var
            )

            mu = _tf.transpose(mu[:, :, 0])
            var = _tf.transpose(var)
            sims = _tf.transpose(sims, [1, 0, 2])

            # likelihood
            y_s = _tf.split(y, self.var_lengths, axis=1)
            mu = _tf.split(mu, self.lik_sizes, axis=1)
            var = _tf.split(var, self.lik_sizes, axis=1)
            hv = _tf.split(has_value, self.var_lengths, axis=1)
            sims = _tf.split(sims, self.lik_sizes, axis=1)

            elbo = _tf.constant(0.0, _tf.float64)
            for likelihood, mu_i, var_i, y_i, hv_i, sim_i, inp in zip(
                    self.likelihoods, mu, var, y_s,
                    hv, sims, training_inputs):
                elbo = elbo + likelihood.log_lik(
                    mu_i, var_i, y_i, hv_i, samples=sim_i, **inp)

            # batch weight
            batch_size = _tf.reduce_sum(has_value)
            elbo = elbo * self.total_data / batch_size

            return elbo

    @_tf.function
    def _training_elbo(self, x, y, has_value, training_inputs,
                       prev_mu, prev_var, prev_sims, prev_exp_var, network,
                       x_dir=None, directions=None, y_dir=None,
                       has_value_directions=None, x_var=None,
                       seed=0, jitter=1e-6):
        network.refresh(jitter)

        # ELBO
        elbo = self._log_lik(x, y, has_value, training_inputs,
                             prev_mu, prev_var, prev_sims, prev_exp_var,
                             network, x_var=x_var, seed=seed)

        # ELBO for directions
        # if x_dir is not None:
        #     elbo = elbo + self._log_lik_directions(
        #         x_dir, directions, y_dir, has_value_directions)

        # KL-divergence
        unique_nodes = network.get_unique_parents()
        unique_nodes.append(network)
        kl = _tf.add_n([node.kl_divergence() for node in unique_nodes])
        elbo = elbo - kl

        self.elbo.assign(elbo)
        self.kl_div.assign(kl)
        return elbo

    def _svi(self, data_index, network, epochs, learning_rate):
        optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, 1, 0.999),
            amsgrad=True
        )

        model_variables = network.get_unfixed_variables()

        def loss(idx):
            training_inputs = [
                self.data[data_index].variables[v].training_input(idx)
                for v in self.variables]

            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[data_index][idx],
                                 _tf.float64),
                    _tf.constant(self.y[data_index][idx], _tf.float64),
                    _tf.constant(self.has_value[data_index][idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(
                        self.data.get_data_variance()[data_index][idx],
                        _tf.float64),
                    jitter=self.options.jitter,
                    seed=self.options.seed,
                    network=network,
                    prev_mu=_tf.constant(
                        self.running_mean[:, data_index[idx], :],
                        _tf.float64),
                    prev_var=_tf.constant(
                        self.running_variance[:, data_index[idx]],
                        _tf.float64),
                    prev_sims=_tf.constant(
                        self.running_sims[:, data_index[idx], :],
                        _tf.float64),
                    prev_exp_var=_tf.constant(
                        self.running_explained_variance[:, data_index[idx]],
                        _tf.float64)
                )
            else:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    training_inputs,
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    # y_dir=_tf.constant(self.y_dir, _tf.float64),
                    # has_value_directions=_tf.constant(
                    #     self.has_value_dir, _tf.float64),
                    jitter=self.options.jitter,
                    seed=self.options.seed,
                    network=network,
                    prev_mu=_tf.constant(
                        self.running_mean[:, data_index[idx], :],
                        _tf.float64),
                    prev_var=_tf.constant(
                        self.running_variance[:, data_index[idx]],
                        _tf.float64),
                    prev_sims=_tf.constant(
                        self.running_sims[:, data_index[idx], :],
                        _tf.float64),
                    prev_exp_var=_tf.constant(
                        self.running_explained_variance[:, data_index[idx]],
                        _tf.float64)
                )

        _np.random.seed(self.options.seed)
        training_log = []
        for i in range(epochs):
            current_elbo = []

            shuffled = _np.random.choice(
                len(data_index), len(data_index), replace=False)
            batches = self.options.batch_index(len(data_index))

            for batch in batches:
                optimizer.minimize(
                    lambda: loss(shuffled[batch]),
                    model_variables)

                for pr in self._all_parameters:
                    pr.refresh()

                current_elbo.append(self.elbo.numpy())
                training_log.append(current_elbo[-1])

            total_elbo = _np.mean(current_elbo) - self.total_kl_div
            if self.options.verbose:
                print("\rEpoch %s | ELBO: %s" %
                      (str(i + 1), str(total_elbo)), end="")

        if self.options.verbose:
            print("\n")

        self.total_kl_div += self.kl_div.numpy()
        return _np.array(training_log)

    def update_training_latent(self, data_index, network):
        batches = self.options.batch_index(
            len(data_index), self.options.prediction_batch_size)

        for i, batch in enumerate(batches):
            if self.options.verbose:
                print("\rProcessing batch %s of %s             "
                      % (str(i + 1), str(len(batches))), end="")

            mu, var, sims, explained_var, _ = network.predict(
                self.data.coordinates[data_index][batch],
                n_sim=self.n_sims, seed=[self.options.seed, 0])

            mu, var, sims, exp_var = self.refine(
                mu, var, sims, explained_var,
                self.running_mean[:, data_index[batch], :],
                self.running_variance[:, data_index[batch]],
                self.running_sims[:, data_index[batch], :],
                self.running_explained_variance[:, data_index[batch]]
            )

            self.running_mean[:, data_index[batch], :] = mu.numpy()
            self.running_variance[:, data_index[batch]] = var.numpy()
            self.running_sims[:, data_index[batch], :] = sims.numpy()
            self.running_explained_variance[:, data_index[batch]] = \
                exp_var.numpy()

        if self.options.verbose:
            print("\n")

    def update_target_latent(self, data_index, network, target_idx):
        batches = self.options.batch_index(
            len(data_index), self.options.prediction_batch_size)

        for i, batch in enumerate(batches):
            if self.options.verbose:
                print("\rProcessing batch %s of %s             "
                      % (str(i + 1), str(len(batches))), end="")

            mu, var, sims, explained_var, _ = network.predict(
                self.prediction_targets[target_idx]
                    .coordinates[data_index][batch],
                n_sim=self.n_sims, seed=[self.options.seed, 0])

            mu, var, sims, exp_var = self.refine(
                mu, var, sims, explained_var,
                self.predicted_running_mean[target_idx]
                [:, data_index[batch], :],
                self.predicted_running_variance[target_idx]
                [:, data_index[batch]],
                self.predicted_running_sims[target_idx]
                [:, data_index[batch], :],
                self.predicted_running_explained_variance[target_idx]
                [:, data_index[batch]]
            )

            self.predicted_running_mean[target_idx] \
                [:, data_index[batch], :] = mu.numpy()
            self.predicted_running_variance[target_idx] \
                [:, data_index[batch]] = var.numpy()
            self.predicted_running_sims[target_idx] \
                [:, data_index[batch], :] = sims.numpy()
            self.predicted_running_explained_variance[target_idx] \
                [:, data_index[batch]] = exp_var.numpy()

        if self.options.verbose:
            print("\n")

    def predict_from_latent(self, target,
                            latent_mean, latent_variance,
                            latent_sims, latent_explained_variance):
        # managing variables
        variable_inputs = []
        for v in self.variables:
            if v not in target.variables.keys():
                self.data.variables[v].copy_to(target)
            target.variables[v].allocate_simulations(self.n_sims)
            variable_inputs.append(self.data.variables[v].prediction_input())

        batches = self.options.batch_index(
            target.n_data, self.options.prediction_batch_size)

        @_tf.function
        def predict_raw(pred_mu, pred_var, pred_sim, pred_exp_var):
            pred_mu = _tf.transpose(pred_mu[:, :, 0])
            pred_var = _tf.transpose(pred_var)
            pred_sim = _tf.transpose(pred_sim, [1, 0, 2])
            pred_exp_var = _tf.transpose(pred_exp_var)

            pred_mu = _tf.split(pred_mu, self.lik_sizes, axis=1)
            pred_var = _tf.split(pred_var, self.lik_sizes, axis=1)
            pred_sim = _tf.split(pred_sim, self.lik_sizes, axis=1)
            pred_exp_var = _tf.split(pred_exp_var, self.lik_sizes, axis=1)

            # likelihood
            out = []
            for mu, var, sim, exp_var, lik, v_inp in zip(
                    pred_mu, pred_var, pred_sim, pred_exp_var,
                    self.likelihoods, variable_inputs):
                out.append(lik.predict(mu, var, sim, exp_var, **v_inp))

            return out

        for i, batch in enumerate(batches):
            if self.options.verbose:
                print("\rProcessing batch %s of %s             "
                      % (str(i + 1), str(len(batches))), end="")

            output = predict_raw(
                _tf.constant(latent_mean[:, batch, :], _tf.float64),
                _tf.constant(latent_variance[:, batch], _tf.float64),
                _tf.constant(latent_sims[:, batch, :], _tf.float64),
                _tf.constant(latent_explained_variance[:, batch], _tf.float64)
            )

            # writing result
            for v, upd in zip(self.variables, output):
                target.variables[v].update(batch, **upd)

        if self.options.verbose:
            print("\n")

    def train_level(self, level, epochs, learning_rate):
        if self.options.verbose:
            print("Scale level %d/%d" % (level + 1, len(self.search_blocks)))

        train_block_index = self.search_blocks[level].index_data(self.data)
        target_block_index = [
            self.search_blocks[level].index_data(t)
            for t in self.prediction_targets
        ]

        grid_index = _np.array(
            list(_iter.product(
                *[_np.arange(n)
                  for n in self.search_blocks[level].grid_size[::-1]]
            ))
        )[:, ::-1]

        for combination in grid_index:
            s = "Looping through block " \
              + "-".join([str(i + 1) for i in combination]) \
              + " of " \
              + "-".join([str(i) for i in self.search_blocks[level].grid_size])
            if self.options.verbose:
                print(s)

            # selecting data
            # neighborhood_index = _np.array(
            #     list(_iter.product(
            #         *[[i - 1, i, i + 1]
            #           for i in combination[::-1]]
            #     ))
            # )[:, ::-1]
            # match = []
            # for block_idx in neighborhood_index:
            #     match_i = _np.stack([train_block_index[:, d] == block_idx[d]
            #                         for d in range(self.n_dim)], axis=1)
            #     match_i = _np.all(match_i, axis=1)
            #     match.append(match_i)
            # match = _np.stack(match, axis=-1)
            # match = _np.any(match, axis=-1)

            match = _np.stack([train_block_index[:, d] == combination[d]
                               for d in range(self.n_dim)], axis=1)
            match = _np.all(match, axis=1)
            match = _np.where(match)[0]

            if len(match) >= self.minimum_data:
                # inducing points and local network
                # inducing_points = self.search_blocks[level]\
                #     .discretized_coordinates(combination)
                # inducing_points = _data.PointData.from_array(inducing_points)
                inducing_points = self.search_blocks[level] \
                    .inducing_grid(combination)

                network = self.latent_network_generators[level](inducing_points)

                # training
                training_log = self._svi(match, network, epochs, learning_rate)
                self.training_log[level][_np.split(combination, self.n_dim)] \
                    = [training_log]
                network.refresh(self.options.jitter)

                # # update latent variables
                # if self.options.verbose:
                #     print("Updating training latent variables...")
                # self.update_training_latent(_np.arange(self.data.n_data), network)
                #
                # for t, target in enumerate(target_block_index):
                #     if self.options.verbose:
                #         print("Updating target latent variables...")
                #     self.update_target_latent(
                #         _np.arange(self.prediction_targets[t].n_data), network, t)

                # update latent variables in neighborhood of block
                neighborhood_index = _np.array(
                    list(_iter.product(
                        *[[i - 2, i - 1, i, i + 1, i + 2]
                          for i in combination[::-1]]
                    ))
                )[:, ::-1]

                for block_idx in neighborhood_index:
                    match = _np.stack([train_block_index[:, d] == block_idx[d]
                                       for d in range(self.n_dim)], axis=1)
                    match = _np.all(match, axis=1)
                    match = _np.where(match)[0]
                    if len(match) > 0:
                        if self.options.verbose:
                            print("Updating training latent variables...")
                        self.update_training_latent(match, network)

                    for t, target in enumerate(target_block_index):
                        match = _np.stack([target[:, d] == block_idx[d]
                                           for d in range(self.n_dim)], axis=1)
                        match = _np.all(match, axis=1)
                        match = _np.where(match)[0]
                        if len(match) > 0:
                            if self.options.verbose:
                                print("Updating target latent variables...")
                            self.update_target_latent(match, network, t)

    def train(self, epochs_per_pass=100, learning_rate=0.01):
        if self.options.verbose:
            print("Training...")
        for i, b in enumerate(self.search_blocks):
            self.train_level(i, epochs_per_pass, learning_rate)

        if self.options.verbose:
            print("Predicting on training data...")
        self.predict_from_latent(
            self.data, self.running_mean, self.running_variance,
            self.running_sims, self.running_explained_variance
        )

        if self.options.verbose:
            print("Predicting on targets...")
        for i, t in enumerate(self.prediction_targets):
            self.predict_from_latent(
                t,
                self.predicted_running_mean[i],
                self.predicted_running_variance[i],
                self.predicted_running_sims[i],
                self.predicted_running_explained_variance[i]
            )
