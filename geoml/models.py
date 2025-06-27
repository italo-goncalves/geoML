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
import geoml.tftools as _tftools
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
                 seed=1234, jitter=1e-9,
                 training_batch_size=2000, training_samples=20):
        """
        Configuration of Gaussian process models.

        This object can be passed on to models based on the Gaussian process in order to
        control their behavior.

        Parameters
        ----------
        verbose
            Whether to show the training process on screen.
        prediction_batch_size : int
            Batch size for prediction/inference.
        seed : int
            Seed to control random number generation.
        jitter : float
            Small value added to covariance matrices for numerical stability.
        training_batch_size : int
            Number of data points per batch during training.
        training_samples : int
            Number of Monte Carlo samples to be drawn when training requires it.
        """
        super().__init__(verbose, prediction_batch_size,
                         training_batch_size, seed)
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

    def set_learning_rate(self, rate):
        """
        Resets the model's optimizer with the provided learning rate. Will erase the optimizer's memory.

        Parameters
        ----------
        rate : float
            The learning rate to use.
        """
        raise NotImplementedError


class GP(_GPModel):
    """
    Basic Gaussian process model.
    """
    def __init__(self, data, variable, covariance, warping=None,
                 directional_data=None, interpolation=False,
                 use_trend=False, options=GPOptions()):
        """
        Basic Gaussian process model.

        This model is based on the standard Gaussian process for a single output variable. It supports warping
        for non-Gaussian variables and directional data as gradients of the modelled field, but not both simultaneously.

        Parameters
        ----------
        data
            A `PointData` object from the ´data´ module.
        variable : str
            The name of the variable to be modelled. Must be a continuous variable.
        covariance
            The covariance function to build the covariance matrices.
        warping
            An object from the `warping` module. If None, the data is assumed to have zero mean and unit variance.
        directional_data
            A `DirectionalData` object from the ´data´ module. The corresponding variable will be used as the gradient
            of the modelled field.
        interpolation : bool
            If `True`, will assume that the data is noiseless and try to honor the data points.
        use_trend : bool
            If `True`, will model a linear trend in the data in addition to the GP.
        options : GPOptions
            Additional configurations.
        """
        super().__init__(options)

        self.data = data
        self.variable = variable
        self.covariance = self._register(covariance)
        self.covariance.set_limits(data)

        if warping is None:
            warping = _warp.Identity()
        self.warping = self._register(warping)

        keep = ~ _np.isnan(self.data.variables[self.variable].measurements.values)
        if _np.sum(keep) > 0:
            self.warping.initialize(self.data.variables[self.variable].measurements.values[keep])

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
        """
        Updates the model's internal state.

        If called within TensorFlow's eager mode, will allow inspection of the internal tensors.

        Parameters
        ----------
        jitter : float
            Small value added to the covariance matrices for numerical stability.
        """
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
        """
        Computes the model's log-likelihood with the current parameters.

        Parameters
        ----------
        jitter : float
            Small value added to the covariance matrices for numerical stability.
        """
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
            The object's variables will be updated.
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
        """
        Model training.

        The standard GP does not support batches of data, allways using the full data instead. This is feasible for
        up to a few thousand data points.

        Parameters
        ----------
        max_iter : int
            The number of iterations to train.
        """

        model_variables = [pr.variable for pr in self._all_parameters
                           if not pr.fixed]

        def loss():
            return - self.log_likelihood(self.options.jitter)

        for i in range(max_iter):
            # self.optimizer.minimize(loss, model_variables)
            _tftools.training_step(self.optimizer, loss, model_variables)

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
    """Variational Gaussian process network."""
    def __init__(self, data, variables, likelihoods,
                 latent_network,
                 directional_data=None,
                 options=GPOptions()):
        """
        Variational Gaussian process network.

        This is the heart of the `geoml` package, a generalization of the standard GP. It supports variables of any
        kind and flexible structures through the latent variable network.

        Parameters
        ----------
        data
            A `PointData` object from the ´data´ module.
        variables
            The name of a variable to model, or a list of names.
        likelihoods
            A `likelihood` object or a list matching the length of `variables`.
        latent_network
            An object from the `latent` module.
        directional_data
            A `DirectionalData` object from the ´data´ module. The corresponding variable will be used as the gradient
            of the modelled field.
        options : GPOptions
            Additional configurations.
        """
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
                       samples=20, seed=0, jitter=1e-6):
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

        self.elbo.assign(elbo - kl)
        self.kl_div.assign(kl)
        return elbo - kl

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
        """
        Model training.

        The VGP will be trained using all data at once. Only feasible for relatively small datasets, depending of
        te size of the latent network. For larger datasets, use the `train_svi` method.

        Parameters
        ----------
        max_iter : int
            The number of iterations to train.
        """
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
            # self.optimizer.minimize(loss, model_variables)
            _tftools.training_step(self.optimizer, loss, model_variables)

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
        """
        Stochastic Variational Inference training.

        The model will be trained in batches according to the `options` object.

        Parameters
        ----------
        epochs : int
            Number of epochs to train.
        """
        model_variables = self.get_unfixed_variables()

        if self.directional_data is not None:
            x_dir = _tf.constant(
                self.directional_data.coordinates,
                _tf.float64),
            directions = _tf.constant(
                self.directional_data.directions, _tf.float64),
            y_dir = _tf.constant(self.y_dir, _tf.float64),
            has_value_directions = _tf.constant(
                self.has_value_dir, _tf.float64),

        def loss(idx):
            # training_inputs = [
            #     self.data.variables[v].training_input(idx)
            #     for v in self.variables]

            if self.directional_data is None:
                return - self._training_elbo(
                    _tf.constant(self.data.coordinates[idx],
                                 _tf.float64),
                    _tf.constant(self.y[idx], _tf.float64),
                    _tf.constant(self.has_value[idx], _tf.float64),
                    # training_inputs,
                    [{} for v in self.variables],
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    # local_x, local_y, local_hv, {}, local_x_var,
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
                    # training_inputs,
                    [{} for v in self.variables],
                    x_var=_tf.constant(self.data.get_data_variance()[idx],
                                       _tf.float64),
                    # local_x, local_y, local_hv, {}, local_x_var,
                    x_dir=_tf.identity(x_dir),
                    # directions=_tf.constant(
                    #     self.directional_data.directions, _tf.float64),
                    # y_dir=_tf.constant(self.y_dir, _tf.float64),
                    # has_value_directions=_tf.constant(
                    #     self.has_value_dir, _tf.float64),
                    directions=_tf.identity(directions),
                    y_dir=_tf.identity(y_dir),
                    has_value_directions=_tf.identity(has_value_directions),
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
                # self.optimizer.minimize(
                #     # loss,
                #     lambda: loss(shuffled[batch]),
                #     model_variables)
                _tftools.training_step(self.optimizer, lambda: loss(shuffled[batch]), model_variables)

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

    @_tf.function
    def predict_raw(self, x_new, variable_inputs, x_var=None,
                    n_sim=1, seed=0, jitter=1e-6, include_noise=True):
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
                output.append(lik.predict(mu, var, sim, exp_var, include_noise=include_noise, **v_inp))
            return output

    def predict(self, newdata, n_sim=20, include_noise=True):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension.
            The object's variables will be updated.
        n_sim : int
            Number of predictive samples to draw.
        include_noise : bool
            If the likelihood should be sampled and added to the simulations.
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
                jitter=self.options.jitter,
                include_noise=include_noise
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
    """Structural field modeling based on gradient data."""
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
            # self.optimizer.minimize(loss, model_variables)
            _tftools.training_step(self.optimizer, loss, model_variables)

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
        self.models = []
        self.variable = None

    def set_learning_rate(self, rate):
        for model in self.models:
            model.set_learning_rate(rate)

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
        var = _tf.reduce_sum(weights ** 2 * var, axis=1)

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


class GPEnsemble(_EnsembleModel):
    """An ensemble of Gaussian processes."""
    def __init__(self, data, variable, covariance, warping=None, directional_data=None,
                 use_trend=False, options=GPOptions()):
        """
        An ensemble of Gaussian processes.

        This model combines independent GPs into a consolidated prediction using the Product of Experts approach.
        It is preferable to divide the data spatially instead of randomly, so that each expert can focus on a
        specific region of the space.

        Parameters
        ----------
        data
            A list or tuple of `PointData` objects.
        variable : str
            The name of the variable to be modelled. Must be present in all data objects.
        covariance
            The covariance function to build the covariance matrices.
        warping
            An object from the `warping` module. If None, the data is assumed to have zero mean and unit variance.
        directional_data
            A `DirectionalData` object from the ´data´ module. The corresponding variable will be used as the gradient
            of the modelled field.
        use_trend : bool
            If `True`, will model a linear trend in the data in addition to the GP.
        options : GPOptions
            Additional configurations.
        """
        super().__init__(options)
        if not isinstance(data, (tuple, list)):
            raise ValueError("data must be a list or tuple containing"
                             "data objects")
        if directional_data is None:
            directional_data = [None for _ in data]
        elif not isinstance(directional_data, (tuple, list)):
            raise ValueError("directional_data must be a list or tuple containing"
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
            directional_data=dd,
            use_trend=use_trend,
            options=options)
            for d, dd in zip(data, directional_data)]
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
            likelihoods=_copy.deepcopy(likelihoods),
            # likelihoods=lik,
            latent_network=l,
            directional_data=dd,
            options=options)
            # for d, l, dd, lik in zip(
            #     data, latent_networks, directional_data, likelihoods)
            for d, l, dd in zip(
                data, latent_networks, directional_data)
        ]
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
    """Trainable data normalizer."""
    def __init__(self, warping, options=GPOptions()):
        """
        Trainable data normalizer.

        This model will fit a `warping` object to a data vector, allowing its transformation to a Gaussian
        distribution with zero mean and unit variance.

        Parameters
        ----------
        warping
            An object from the `warping` module.
        options : GPOptions
            Additional configurations.
        """
        super().__init__(options)
        self.warping = self._register(warping)

        self.training_log = []
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-1, 1, 0.99),
            amsgrad=True
        )

        self.objective = _tf.Variable(_tf.constant(0.0, _tf.float64))

    def normalize(self, x, max_iter=250):
        """
        Model training.

        Parameters
        ----------
        x : array-like
            The data vector to train on.
        max_iter : int
            The number of iterations to run.
        """
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
            # self.optimizer.minimize(loss, model_variables)
            _tftools.training_step(self.optimizer, loss, model_variables)

            for pr in self._all_parameters:
                pr.refresh()

            current_elbo = self.objective.numpy()
            self.training_log.append(current_elbo)

            if self.options.verbose:
                print("\rIteration %s | Objective: %s" %
                      (str(i + 1), str(current_elbo)), end="")


class ProjectedVGP(VGPNetwork):
    @_tf.function
    def _log_lik(self, x, y, has_value, training_inputs, x_var=None,
                 samples=20, seed=0):
        with _tf.name_scope("batched_elbo"):
            # prediction
            sims = self.latent_network.predict(x, n_sim=samples, seed=[seed, 0])

            mu = sims[:, :, 0]  # dummy
            var = sims[:, :, 0]**2  # dummy

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
    def predict_raw(self, x_new, variable_inputs, x_var=None,
                    n_sim=1, seed=0, jitter=1e-6, include_noise=True):
        self.latent_network.refresh(jitter)

        with _tf.name_scope("Prediction"):
            pred_sim = self.latent_network.predict(x_new, n_sim=n_sim, seed=[seed, 0])

            pred_mu = pred_sim[:, :, 0]  # dummy
            pred_var = pred_sim[:, :, 0]**2  # dummy
            pred_exp_var = pred_var   # dummy

            pred_mu = _tf.split(pred_mu, self.lik_sizes, axis=1)
            pred_var = _tf.split(pred_var, self.lik_sizes, axis=1)
            pred_sim = _tf.split(pred_sim, self.lik_sizes, axis=1)
            pred_exp_var = _tf.split(pred_exp_var, self.lik_sizes, axis=1)

            output = []
            for mu, var, sim, exp_var, lik, v_inp in zip(
                    pred_mu, pred_var, pred_sim, pred_exp_var,
                    self.likelihoods, variable_inputs):
                output.append(lik.predict(mu, var, sim, exp_var, include_noise=include_noise, **v_inp))
            return output
