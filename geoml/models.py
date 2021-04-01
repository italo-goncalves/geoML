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
           "VGP", "VGPEnsemble"]

import geoml.data as _data
import geoml.parameter as _gpr
import geoml.likelihood as _lk
import geoml.warping as _warp

import numpy as _np
import tensorflow as _tf
import pickle as _pickle
import copy as _copy

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


class _GPModel:
    def __init__(self, options=GPOptions()):
        self.options = options
        self._pre_computations = {}
        self._all_parameters = []
        self.parameters = {}
        self._n_dim = None

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def all_parameters(self):
        return self._all_parameters

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

    def save_state(self, file):
        parameters = self.get_parameter_values(complete=True)
        with open(file, 'wb') as f:
            _pickle.dump(parameters, f)

    def load_state(self, file):
        with open(file, 'rb') as f:
            parameters = _pickle.load(f)

        value, shape, position, k_min_val, k_max_val = parameters
        self.update_parameters(value, shape, position)


class GP(_GPModel):
    """
    Basic Gaussian process model.
    """
    def __init__(self, data, variable, kernel, warping=None, tangents=None,
                 use_trend=False, options=GPOptions()):
        super().__init__(options)

        self.data = data
        self.variable = variable
        self.kernel = kernel
        self.kernel.set_limits(data)

        if warping is None:
            warping = _warp.Identity()
        self.warping = warping

        self.tangents = tangents
        self.use_trend = use_trend

        self._all_parameters.extend(kernel.all_parameters)
        self._all_parameters.extend(warping.all_parameters)
        self.parameters.update({
            "noise": _gpr.PositiveParameter(0.1, 1e-6, 10)
        })
        self._all_parameters.append(self.parameters["noise"])

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
        s += repr(self.kernel)
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

            if self.tangents is not None:
                self.x_dir = _tf.constant(self.tangents.coordinates,
                                          _tf.float64)
                self.directions = _tf.constant(self.tangents.directions,
                                               _tf.float64)

                cov = self.kernel.self_covariance_matrix(self.x)
                cov_d1 = self.kernel.covariance_matrix_d1(
                    self.x, self.x_dir, self.directions)
                cov_d2 = self.kernel.self_covariance_matrix_d2(
                    self.x_dir, self.directions)

                self.cov = _tf.concat([
                    _tf.concat([cov, cov_d1], axis=1),
                    _tf.concat([_tf.transpose(cov_d1), cov_d2], axis=1)
                ], axis=0)

                self.y_warped = _tf.concat([
                    self.warping.forward(self.y[:, None]),
                    _tf.zeros([self.tangents.n_data, 1], _tf.float64)
                ], axis=0)

                eye = _tf.eye(_np.sum(keep) + self.tangents.n_data,
                              dtype=_tf.float64)
                noise = _tf.concat([
                    _tf.ones([_np.sum(keep)], _tf.float64),
                    _tf.zeros([self.tangents.n_data], _tf.float64)
                ], axis=0)
            else:
                self.cov = self.kernel.self_covariance_matrix(self.x)
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

                if self.tangents is not None:
                    trend_grad = _tf.concat([
                        _tf.zeros([self.tangents.n_data, 1], _tf.float64),
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
            cov_new = self.kernel.covariance_matrix(x_new, self.x)
            if self.tangents is not None:
                cov_new_d1 = self.kernel.covariance_matrix_d1(
                    x_new, self.x_dir, self.directions)
                cov_new = _tf.concat([cov_new, cov_new_d1], axis=1)
            cov_new = cov_new / self.scale[None, :]

            # prediction
            mu = _tf.matmul(cov_new, self.alpha)

            point_var = self.kernel.point_variance(x_new)[:, None]
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


class VGP(_GPModel):
    """Vanilla VGP"""
    def __init__(self, data, variables, likelihoods,
                 latent_layer,
                 directional_data=None,
                 force_independence=False,
                 options=GPOptions()):
        super().__init__(options=options)

        self.data = data
        self.latent_layer = latent_layer

        if not (isinstance(likelihoods, list)
                or isinstance(likelihoods, tuple)):
            likelihoods = [likelihoods]
        self.likelihoods = likelihoods

        if not (isinstance(variables, (list, tuple))):
            variables = [variables]
        self.variables = variables
        self.var_lengths = [data.variables[v].length for v in variables]

        # mixing weights
        n_latent = self.latent_layer.n_latent
        _np.random.seed(self.options.seed)
        mix_start = _np.random.normal(
            size=[n_latent, sum(self.var_lengths) - 1])
        mix_start = mix_start / _np.sqrt(_np.sum(mix_start ** 2,
                                                 axis=0, keepdims=True))
        mix_start = _np.concatenate(
            [mix_start, - _np.sum(mix_start, axis=1, keepdims=True) + 1e-6],
            axis=1)
        self.parameters.update({
            "mixing_weights": _gpr.UnitColumnNormParameter(
                mix_start,
                _np.zeros([n_latent, sum(self.var_lengths)]) - 1,
                _np.zeros([n_latent, sum(self.var_lengths)]) + 1,
                name="mixing_weights",
            ),
        })

        if force_independence:
            if n_latent != sum(self.var_lengths):
                raise Exception("Cannot force independence: number of"
                                "latent variables do not match the number"
                                "of data variables.")
            self.parameters["mixing_weights"].set_value(_np.eye(n_latent))
            self.parameters["mixing_weights"].fix()

        # directions
        self.directional_likelihood = _lk.Gaussian()
        self.directional_likelihood.parameters["noise"].set_value(1e-6)
        self.directional_data = directional_data
        self.total_data_dir = 0

        if directional_data is not None:
            if self.data.n_dim != directional_data.n_dim:
                raise ValueError("the directional data must have the"
                                 "same number of dimensions as the"
                                 "point data")
            self.total_data_dir = directional_data.n_data

        # dealing with NaNs
        y = _np.concatenate([data.variables[v].get_measurements()
                             for v in self.variables], axis=1)
        y[_np.isnan(y)] = 0
        self.y = y
        self.has_value = (~ _np.isnan(y)) * 1.0
        self.total_data = _np.sum(self.has_value)

        # optimizer
        self.training_log = []
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-2, 1, 0.999),
            amsgrad=True
        )

        # setting trainable parameters
        self._all_parameters.extend(self.latent_layer.all_parameters)
        self.latent_layer.set_kernel_limits(self.data)
        self.latent_layer.refresh(self.options.jitter)
        for likelihood in likelihoods:
            self._all_parameters.extend(likelihood.all_parameters)
        self._all_parameters.append(self.parameters["mixing_weights"])

        # pre_computations
        self._pre_computations.update({
            "elbo": _tf.Variable(_tf.constant(0.0, _tf.float64)),
            "kl_div": _tf.Variable(_tf.constant(0.0, _tf.float64)),
        })

    def __repr__(self):
        s = "Variational Gaussian process model\n\n"
        s += "Variables:\n "
        for v, lik in zip(self.variables, self.likelihoods):
            s += "\t" + v + " (" + lik.__class__.__name__ + ")\n"
        s += "\nLatent layer:\n"
        s += repr(self.latent_layer)
        return s

    def set_learning_rate(self, rate):
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(rate, 1, 0.999),
            amsgrad=True
        )

    @_tf.function
    def _training_elbo(self, x, y, has_value, training_inputs,
                       x_dir=None, directions=None, samples=20,
                       seed=0, jitter=1e-6):

        elbo = self._log_lik(x, y, has_value, training_inputs,
                             samples=samples, seed=seed, jitter=jitter)

        if x_dir is not None:
            elbo = elbo + self._log_lik_directions(
                x_dir, directions, jitter=jitter)

        elbo = elbo - self._kl_divergence(jitter=jitter)

        self._pre_computations["elbo"].assign(elbo)
        return elbo

    @_tf.function
    def _log_lik(self, x, y, has_value, training_inputs,
                 samples=20, seed=0, jitter=1e-6):
        with _tf.name_scope("batched_elbo"):
            # prediction
            mu, var, sims, _ = self.latent_layer.predict(
                x, n_sim=samples, seed=[seed, 0], jitter=jitter)

            # mixing
            mixing_weights = self.parameters["mixing_weights"].get_value()

            mu = _tf.matmul(mu[:, :, 0], mixing_weights, True, False)
            var = _tf.matmul(var, mixing_weights ** 2, True, False)
            sims = _tf.einsum("lds,lv->dvs", sims, mixing_weights)

            # likelihood
            y_s = _tf.split(y, self.var_lengths, axis=1)
            mu = _tf.split(mu, self.var_lengths, axis=1)
            var = _tf.split(var, self.var_lengths, axis=1)
            hv = _tf.split(has_value, self.var_lengths, axis=1)
            sims = _tf.split(sims, self.var_lengths, axis=1)

            elbo = _tf.constant(0.0, _tf.float64)
            for likelihood, mu_i, var_i, y_i, hv_i, sim_i, inp in zip(
                    self.likelihoods, mu, var, y_s,
                    hv, sims, training_inputs):
                elbo = elbo + likelihood.log_lik(
                    mu_i, var_i, y_i, hv_i, samples=sim_i, **inp)

            # batch weight
            batch_size = _tf.reduce_sum(has_value)
            # elbo = elbo * self.data.n_data / batch_size
            elbo = elbo * self.total_data / batch_size

            return elbo

    @_tf.function
    def _log_lik_directions(self, x_dir, directions, jitter=1e-6):
        with _tf.name_scope("batched_elbo_directions"):
            # prediction
            mu, var, _ = self.latent_layer.predict_directions(
                x_dir, directions, jitter=jitter)
            has_value = _tf.ones_like(mu)

            # likelihood
            mu = _tf.split(_tf.transpose(mu[:, :, 0]),
                           self.latent_layer.n_latent, axis=1)
            var = _tf.split(_tf.transpose(var),
                            self.latent_layer.n_latent, axis=1)
            hv = _tf.split(has_value, self.latent_layer.n_latent, axis=1)
            elbo = _tf.constant(0.0, _tf.float64)
            for mu_i, var_i, hv_i in zip(mu, var, hv):
                elbo = elbo + self.directional_likelihood.log_lik(
                    mu_i, var_i, _tf.zeros_like(mu_i), hv_i)

            # batch weight
            batch_size = _tf.cast(_tf.shape(x_dir)[0], _tf.float64)
            elbo = elbo * self.total_data_dir / batch_size

            return elbo

    @_tf.function
    def _kl_divergence(self, jitter=1e-6):
        with _tf.name_scope("KL_divergence"):
            kl = self.latent_layer.kl_divergence(jitter)
            self._pre_computations["kl_div"].assign(kl)
            return kl

    # def elbo(self):
    #     """
    #     Outputs the model's ELBO, given the current parameters.
    #
    #     Returns
    #     -------
    #     elbo : double
    #         The model's expected lower bound on the log-likelihood.
    #     """
    #     return self._pre_computations["elbo"].numpy()

    def train_full(self, max_iter=1000):
        training_inputs = [self.data.variables[v].training_input()
                           for v in self.variables]

        model_variables = [pr.variable for pr in self._all_parameters
                           if not pr.fixed]

        def loss():
            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates, _tf.float64),
                    _tf.constant(self.y, _tf.float64),
                    _tf.constant(self.has_value, _tf.float64),
                    training_inputs,
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed)
            else:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates, _tf.float64),
                    _tf.constant(self.y, _tf.float64),
                    _tf.constant(self.has_value, _tf.float64),
                    training_inputs,
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed)

        for i in range(max_iter):
            self.optimizer.minimize(loss, model_variables)

            for pr in self._all_parameters:
                pr.refresh()

            current_elbo = self._pre_computations["elbo"].numpy()
            self.training_log.append(current_elbo)

            if self.options.verbose:
                print("\rIteration %s | ELBO: %s" %
                      (str(i+1), str(current_elbo)), end="")

        if self.options.verbose:
            print("\n")

    def train_svi(self, epochs=100):
        model_variables = [pr.variable for pr in self._all_parameters
                           if not pr.fixed]

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
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
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

                current_elbo.append(self._pre_computations["elbo"].numpy())
                self.training_log.append(current_elbo[-1])

            total_elbo = _np.mean(current_elbo)
            if self.options.verbose:
                print("\rEpoch %s | ELBO: %s" %
                      (str(i + 1), str(total_elbo)), end="")

        if self.options.verbose:
            print("\n")

    def train_batched(self, epochs=100):
        model_variables = [pr.variable for pr in self._all_parameters
                           if not pr.fixed]

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
                    x_dir=_tf.constant(
                        self.directional_data.coordinates,
                        _tf.float64),
                    directions=_tf.constant(
                        self.directional_data.directions, _tf.float64),
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed
                )

        _np.random.seed(self.options.seed)
        for i in range(epochs):
            current_elbo = []

            shuffled = _np.random.choice(
                self.data.n_data, self.data.n_data, replace=False)
            batches = self.options.batch_index(
                self.data.n_data, self.options.training_batch_size)

            all_grads = [[] for _ in model_variables]
            for batch in batches:
                with _tf.GradientTape() as g:
                    output = loss(shuffled[batch])
                grad = g.gradient(output, model_variables)

                for j, grad_j in enumerate(all_grads):
                    grad_j.append(grad[j])

                current_elbo.append(
                    self._pre_computations["elbo"].numpy())

            all_grads = [_tf.add_n(grad_j) for grad_j in all_grads]
            self.optimizer.apply_gradients(
                zip(all_grads, model_variables)
            )

            for pr in self._all_parameters:
                pr.refresh()

            total_elbo = _np.mean(current_elbo)
            self.training_log.append(total_elbo)

            if self.options.verbose:
                print("\rEpoch %s | ELBO: %s" %
                      (str(i + 1), str(total_elbo)), end="")

        if self.options.verbose:
            print("\n")

    @_tf.function
    def predict_raw(self, x_new, variable_inputs, n_sim=1, seed=0, jitter=1e-6):
        self.latent_layer.refresh(jitter)

        with _tf.name_scope("Prediction"):
            pred_mu, pred_var, pred_sim, pred_exp_var = \
                self.latent_layer.predict(
                    x_new, n_sim=n_sim, seed=[seed, 0], jitter=jitter
                )

            # mixing
            mixing_weights = self.parameters["mixing_weights"].get_value()
            pred_mu = _tf.matmul(pred_mu[:, :, 0], mixing_weights, True, False)
            pred_var = _tf.matmul(pred_var, mixing_weights ** 2, True, False)
            pred_sim = _tf.einsum("lds,lv->dvs", pred_sim, mixing_weights)
            pred_exp_var = _tf.matmul(pred_exp_var, mixing_weights ** 2,
                                      True, False)

            pred_mu = _tf.split(pred_mu, self.var_lengths, axis=1)
            pred_var = _tf.split(pred_var, self.var_lengths, axis=1)
            pred_sim = _tf.split(pred_sim, self.var_lengths, axis=1)
            pred_exp_var = _tf.split(pred_exp_var, self.var_lengths, axis=1)

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
        def batch_pred(x):
            out = self.predict_raw(
                x,
                variable_inputs,
                seed=self.options.seed,
                n_sim=n_sim,
                jitter=self.options.jitter
            )
            return out

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(n_batches)), end="")

            output = batch_pred(
                _tf.constant(newdata.coordinates[batch], _tf.float64))

            for v, upd in zip(self.variables, output):
                newdata.variables[v].update(batch, **upd)

        if self.options.verbose:
            print("\n")


class StructuralField(_GPModel):
    """Structural field modeling based on gradient data"""
    def __init__(self, tangents, kernel, normals=None, mean_vector=None,
                 options=GPOptions()):
        super().__init__(options=options)

        self.tangents = tangents
        self.normals = normals
        self.kernel = kernel
        self.kernel.set_limits(self.tangents)

        if mean_vector is None:
            # initialized as vertical
            mean_vector = _np.zeros(self.tangents.n_dim)
            mean_vector[-1] = 1

        self.training_log = []
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-2, 1, 0.999),
            amsgrad=True
        )

        self._all_parameters.extend(self.kernel.all_parameters)
        self.parameters.update({
            "mean_vector": _gpr.UnitColumnNormParameter(
                _np.array(mean_vector, ndmin=2).T,
                - _np.ones([self.tangents.n_dim, 1]),
                _np.ones([self.tangents.n_dim, 1])),
            "noise": _gpr.PositiveParameter(1e-4, 1e-6, 10, fixed=True)
        })
        self._all_parameters.append(self.parameters["mean_vector"])
        self._all_parameters.append(self.parameters["noise"])

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
        s += repr(self.kernel)
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

            self.cov = self.kernel.self_covariance_matrix_d2(
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
            cov_new = self.kernel.covariance_matrix_d1(
                x_new, self.all_coordinates, self.all_directions) / self.scale

            mu = _tf.matmul(cov_new, self.alpha)
            mu = mu + _tf.matmul(x_new, mean_vector)

            # variance of gradient along mean direction
            cov_new = self.kernel.covariance_matrix_d2(
                x_new, self.all_coordinates,
                _tf.transpose(mean_vector), self.all_directions
            ) / self.scale

            point_var = self.kernel.point_variance(x_new)[:, None]
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

            cov_new = self.kernel.covariance_matrix_d2(
                x_new, self.all_coordinates,
                x_new_dir, self.all_directions) / self.scale

            mu = _tf.matmul(cov_new, self.alpha)
            mu = mu + _tf.matmul(x_new_dir, mean_vector)

            point_var = self.kernel.point_variance(x_new)[:, None]
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
    def __init__(self, data, variable, kernel, warping=None, tangents=None,
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
            kernel=_copy.deepcopy(kernel),
            warping=_copy.deepcopy(warping),
            tangents=t,
            use_trend=use_trend,
            options=options)
            for d, t in zip(data, tangents)]

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


class VGPEnsemble(_EnsembleModel):
    def __init__(self, data, variables, likelihoods, latent_layers,
                 directional_data=None, force_independence=False,
                 options=GPOptions()):
        super().__init__(options)
        if not isinstance(data, (tuple, list)):
            raise ValueError("data must be a list or tuple containing"
                             "data objects")
        if not isinstance(latent_layers, (tuple, list)):
            raise ValueError("latent_layer must be a list or tuple containing"
                             "layer objects")
        if directional_data is None:
            directional_data = [None for _ in data]
        elif not isinstance(directional_data, (tuple, list)):
            raise ValueError("directional_data must be a list or tuple"
                             "containing data objects or None")

        dims = set([d.n_dim for d in data])
        if len(dims) != 1:
            raise Exception("all data objects must have the same dimension")
        self._n_dim = list(dims)[0]

        self.models = [VGP(
            data=d,
            variables=variables,
            likelihoods=_copy.deepcopy(likelihoods),
            latent_layer=l,
            directional_data=dd,
            force_independence=force_independence,
            options=options)
            for d, l, dd in zip(data, latent_layers, directional_data)]

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

    def train_full(self, max_iter=1000):
        for i, model in enumerate(self.models):
            if self.options.verbose:
                print("Training model %d of %d" % (i + 1, len(self.models)))

            model.train_full(max_iter=max_iter)

    def train_svi(self, epochs=100):
        for i, model in enumerate(self.models):
            if self.options.verbose:
                print("Training model %d of %d" % (i + 1, len(self.models)))

            model.train_svi(epochs=epochs)

    def train_batched(self, epochs=100):
        for i, model in enumerate(self.models):
            if self.options.verbose:
                print("Training model %d of %d" % (i + 1, len(self.models)))

            model.train_batched(epochs=epochs)

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
                if key is not "weights":
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
