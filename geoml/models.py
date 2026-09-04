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

# ProjectedVGP stays out on purpose: importable for old saves, unadvertised
# until its latent side (geoml.latent.fourier) earns tests.
__all__ = ["GP", "GPEnsemble", "Normalizer", "StructuralField", "GPOptions",
           "VGPNetwork", "refine", "cross_validate", "conformalize",
           "ConformalCalibration"]

import numpy as np
from collections.abc import Sequence
from typing import Any as _Any, cast as _cast

import geoml._types as _types
import geoml.data as _data
import geoml.parameter as _gpr
import geoml.latent as _latent
import geoml.likelihood as _lk
import geoml.warping as _warp
import geoml.math.tf as _tftools
import geoml.metrics as _metrics
import geoml.persistence as _persistence
import geoml.stats.random as _srandom
import geoml

import numpy as _np
import pandas as _pd
import tensorflow as _tf
import copy as _copy
import itertools as _iter
import os as _os
import shutil as _shutil
import tempfile as _tempfile
import warnings

import tensorflow_probability as _tfp
_tfd = _tfp.distributions


class _ModelOptions:
    def __init__(self, verbose=True, prediction_batch_size=20000,
                 training_batch_size=2000):
        self.verbose = verbose
        self.training_batch_size = training_batch_size
        self.prediction_batch_size = prediction_batch_size
        # Drawn, not taken: `geoml.set_seed` is the one knob, so the same call
        # that fixes the initial parameters must fix training's Monte Carlo
        # draws and the simulation stream, which read this number through
        # stateless TensorFlow sampling. A saved model keeps the number it
        # drew (`persistence` restores the options `vars` wholesale, skipping
        # this constructor), so its simulations replay exactly on reload.
        self.seed = int(_srandom.rng().integers(2 ** 31 - 1))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(
            "%s=%r" % item for item in vars(self).items()))

    def batch_index(self, n_data, batch_size=None):
        if batch_size is None:
            batch_size = self.training_batch_size

        return _data.batch_index(n_data, batch_size)


class GPOptions(_ModelOptions):
    # Class defaults, so that a model saved before these options existed still
    # opens: `persistence` rebuilds options with `__new__` plus a `vars()`
    # update, never calling `__init__`, so the instance dict has no entry and
    # the lookup falls through to here.
    jit_predict = False
    qmc_simulations = False
    expert_propagation = "consensus"
    training_tolerance = None

    def __init__(self, verbose=True, prediction_batch_size=20000,
                 jitter=1e-9,
                 training_batch_size=2000, training_samples=20,
                 jit_predict=False, qmc_simulations=False,
                 expert_propagation="consensus", training_tolerance=None):
        """
        Configuration of Gaussian process models.

        This object can be passed on to models based on the Gaussian process in order to
        control their behavior.

        The `seed` that training and the simulations read is not a parameter:
        it is drawn from the package generator when the object is built, so
        `geoml.set_seed` before construction governs it the way it already
        governs parameter initialization, and a saved model keeps the number
        it drew.

        Parameters
        ----------
        verbose
            Whether to show the training process on screen.
        prediction_batch_size : int
            Batch size for prediction/inference.
        jitter : float
            Small value added to covariance matrices for numerical stability.
        training_batch_size : int
            Number of data points per batch during training.
        training_samples : int
            Number of Monte Carlo samples to be drawn when training requires it.
        jit_predict : bool
            Whether to compile the prediction graph with XLA. Worth 3-5x on a
            grid of any size, and more on a GPU, but XLA compiles per distinct
            batch shape (so a grid that `prediction_batch_size` does not divide
            pays that cost twice) and refuses to run anything it cannot
            compile, rather than falling back. Prediction only: the same
            treatment makes training both slower and unstable.
        qmc_simulations : bool
            Whether to draw the posterior simulations from a seeded-scramble
            Sobol sequence instead of pseudo-random normals, so the same
            number of them covers the predictive distribution evenly rather
            than by chance. Measured on the Walker Lake model at 16-256
            simulations: the ensemble mean lands 7-37x closer to the exact
            posterior mean, proportions below a cut-off and the outer
            quantiles about a quarter closer -- the accuracy of half again to
            twice the simulations -- while the ensemble's own spread and the
            correlation between locations gain nothing, and the cost is not
            measurable. Deterministic given `seed`, batch-invariant either
            way. The sequence covers at most 21201 dimensions (`size` times
            the inducing points of a node), beyond which scipy refuses.
        expert_propagation : str
            How a deep network's experts see each other's inducing sets,
            for every `BasicGP` in the network at once. `"consensus"` (the
            default, and the historical behavior) predicts every expert's
            set from every other and combines by precision weighting --
            O(K^2) in the expert count. `"independent"` lets each expert
            speak for its own set alone -- O(K) -- so duplicated points in
            overlapping sets may disagree, and the data-side weighting
            arbitrates. Measured (Walker deep model and a 3000-point
            synthetic, K = 5-40): training 1.6x to 6.3x faster and
            prediction up to 8x as K grows; quality within a few percent of
            consensus and sometimes ahead, the consensus coupling appearing
            to slow optimization at large K. Only deep (multi-layer)
            networks are affected: below a terminal node the propagation
            never runs.
        training_tolerance : float, optional
            When to stop training before its iteration count runs out, as a
            fraction. The bound is smoothed, and training stops once its
            gain over the last twenty iterations falls below this share of
            everything gained since the call began. `None` (the default)
            trains for exactly as long as it is told, which is what every
            version before 0.6.5 did. 0.01 is a reasonable setting.

            The count passed to `train_full`/`train_svi` remains the cap,
            and the criterion only ever ends training sooner. It is
            deliberately unable to stop a model that begins the call already
            converged: with nothing gained, there is nothing to take a
            fraction of, and the run goes to its cap. Training in phases is
            the pattern this protects -- a smaller learning rate makes
            progress the previous phase could not, and each phase is judged
            against its own starting point.
        """
        if expert_propagation not in ("consensus", "independent"):
            raise ValueError(
                "expert_propagation must be 'consensus' or 'independent', "
                "got %r" % (expert_propagation,))
        super().__init__(verbose, prediction_batch_size,
                         training_batch_size)
        self.jitter = jitter
        self.training_samples = training_samples
        self.jit_predict = jit_predict
        self.qmc_simulations = qmc_simulations
        self.expert_propagation = expert_propagation
        self.training_tolerance = training_tolerance


class _Convergence:
    """Has the bound stopped improving? The rule behind `training_tolerance`.

    Smooths the bound with an exponential moving average and compares its
    gain over the last window against the gain since training started,
    stopping when the first is a small enough fraction of the second.

    Two things about that ratio are deliberate. Measuring progress from the
    **start of this call** keeps the phased pattern working -- a model
    trained, given a smaller learning rate and trained again plateaus and
    then improves, and each phase is judged on its own terms. And measuring
    the recent gain against the total one rather than against the bound's
    value is the only scale-free choice available: an ELBO's magnitude means
    nothing on its own, since it grows with the number of data points and
    with whatever normalization the likelihood carries.

    The comparison is over a window rather than between consecutive
    iterations, which matters more than it sounds. For a bound approaching
    its limit with time constant `tau`, a per-iteration test fires once
    `exp(-t/tau) < tolerance * tau` -- so a slowly converging fit stops
    almost at once, and how early depends on `tau`, which nobody knows in
    advance. Over a window of `w`, it fires at
    `t > tau * log((exp(w/tau) - 1) / tolerance)`, which for `w` near `tau`
    is a few time constants whatever `tau` is.
    """

    # The window, in iterations (full batch) or epochs (SVI), used both as
    # the average's span and as how far back the comparison reaches.
    _WINDOW = 20

    def __init__(self, tolerance, window=None):
        self.tolerance = tolerance or 0.0
        self.window = self._WINDOW if window is None else int(window)
        self.start = None
        self.trail = []

    def stop(self, value):
        """Whether to stop, having seen `value` as the latest bound."""
        if self.tolerance <= 0.0:
            return False

        if self.start is None:
            self.start, smoothed = value, value
        else:
            weight = 2.0 / (self.window + 1.0)
            smoothed = weight * value + (1.0 - weight) * self.trail[-1]
        self.trail.append(smoothed)

        # nothing to look back at yet, which is the burn-in: a flat stretch
        # before any real progress has a small numerator and a small
        # denominator, and their ratio is not evidence of anything
        if len(self.trail) <= self.window:
            return False

        recent = abs(self.trail[-1] - self.trail[-1 - self.window])
        total = abs(self.trail[-1] - self.start)
        return recent < self.tolerance * total


class _GPModel(_gpr.Parametric):
    def __init__(self, options=None):
        super().__init__()
        # Not a default argument: one `GPOptions()` would then be built at
        # import time and shared by every model that did not bring its own,
        # so setting an option on one model would set it on all of them.
        self.options = GPOptions() if options is None else options
        self._pre_computations = {}
        self._n_dim = None
        self._compiled = {}

    def __str__(self):
        # a model is summarized, not written as a call: `pretty_print()` is
        # still there for the parameter tree
        return self.__repr__()

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

    def save(self, path: _types.PathLike) -> _types.PathLike:
        """Save the model's structure, parameters and training data.

        Parameters
        ----------
        path
            Directory to write to, overwritten if it already exists.

        Returns
        -------
        path
            The location written to.

        See Also
        --------
        geoml.persistence.save_model
        """
        import geoml.persistence as _persistence
        return _persistence.save_model(self, path)

    @classmethod
    def open(cls, path: _types.PathLike,
             data: "_data._SpatialData | None" = None) -> "_GPModel":
        """Load a model saved with :meth:`save`.

        The model comes back with its variables and their types, ready to
        predict or to be trained further.

        Parameters
        ----------
        path
            A directory written by :meth:`save`.
        data
            A data object to build the model around, in place of the one it
            was trained on. Used to refit the same structure on a subset,
            as cross-validation does.

        Returns
        -------
        _GPModel
            The reopened model.

        See Also
        --------
        geoml.persistence.load_model
        """
        import geoml.persistence as _persistence
        return _persistence.load_model(path, data=data)


class GP(_GPModel):
    """
    Basic Gaussian process model.
    """
    def __init__(self, data, variable, covariance, warping=None,
                 directional_data=None, interpolation=False,
                 use_trend=False, options=None):
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
            warping = _warp.ZScore(1)
        self.warping = self._register(warping)

        keep = ~ _np.isnan(self.data.variables[self.variable].measurements.values)
        if _np.sum(keep) > 0:
            self.warping.initialize(
                self.data.variables[self.variable]
                    .measurements.values.to_numpy()[keep, None])

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

        # Everything `refresh()` computes, cleared here so that a model whose
        # parameters changed cannot be predicted from a stale factorization.
        self.cov: _Any = None
        self.cov_chol: _Any = None
        self.cov_inv: _Any = None
        self.scale: _Any = None
        self.alpha: _Any = None
        self.x: _Any = None
        self.y: _Any = None
        self.x_dir: _Any = None
        self.y_dir: _Any = None
        self.directions: _Any = None
        self.y_warped: _Any = None
        self.log_derivative: _Any = None
        self.trend: _Any = None
        self.mat_a_inv: _Any = None
        self.trend_chol = None
        self.beta = None

    def __repr__(self):
        s = "Gaussian process model\n\n"
        s += "Variable: " + self.variable + "\n\n"
        s += "Kernel:\n"
        s += str(self.covariance)
        s += "\nWarping:\n"
        s += str(self.warping)
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
                                  .measurements.values.to_numpy()[keep, None],
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
                        .measurements.values.to_numpy(),
                    _tf.float64
                )
                self.y_warped = _tf.concat([
                    # self.warping.forward(self.y),
                    self.y,
                    self.y_dir[:, None]
                ], axis=0)
                self.log_derivative = _tf.constant(0.0, _tf.float64)

                eye = _tf.eye(_np.sum(keep) + self.directional_data.n_data,
                              dtype=_tf.float64)
                noise = _tf.concat([
                    _tf.ones([_np.sum(keep)], _tf.float64),
                    _tf.zeros([self.directional_data.n_data], _tf.float64)
                ], axis=0)
            else:
                self.cov = self.covariance.self_covariance_matrix(self.x)
                self.y_warped, self.log_derivative = self.warping.forward(self.y)

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

            # log_derivative = self.warping.log_derivative(self.y)
            log_lik = log_lik + _tf.reduce_sum(self.log_derivative)

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
    def predict_raw(self, x_new, jitter=1e-9, n_sim=50):
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

            # warping
            # distribution = _tfd.Normal(mu, _tf.sqrt(var))
            # sims = distribution.sample(50)
            rnd = _tf.random.stateless_normal([1, n_sim], seed=[0, 0], dtype=_tf.float64)
            sims = rnd * _tf.sqrt(var) + mu

            sims = _tf.transpose(sims)
            sims = _tf.map_fn(lambda x: self.warping.backward(x[:, None]), sims)
            sims = _tf.transpose(sims, [1, 2, 0])

            avg_sim = _tf.reduce_mean(sims, axis=2)

            out = {"mean": mu[:, 0],
                   "variance": var[:, 0],
                   "weights": _tf.squeeze(weights),
                   "simulations": sims[:, 0, :],
                   "average_sim": avg_sim[:, 0],
                   }

            #
            # def prob_fn(q):
            #     p = distribution.cdf(self.warping.forward(q))
            #     return p
            #
            # if quantiles is not None:
            #     prob = _tf.map_fn(prob_fn, quantiles)
            #     prob = _tf.transpose(prob)
            #
            #     # single point case
            #     prob = _tf.cond(
            #         _tf.less(_tf.rank(prob), 2),
            #         lambda: _tf.expand_dims(prob, 0),
            #         lambda: prob)
            #
            #     out["probabilities"] = _tf.squeeze(prob)
            #
            # def quant_fn(p):
            #     q = self.warping.backward(distribution.quantile(p))
            #     return q
            #
            # if probabilities is not None:
            #     quant = _tf.map_fn(quant_fn, probabilities)
            #     quant = _tf.transpose(quant)
            #
            #     # single point case
            #     quant = _tf.cond(
            #         _tf.less(_tf.rank(quant), 2),
            #         lambda: _tf.expand_dims(quant, 0),
            #         lambda: quant)
            #
            #     out["quantiles"] = _tf.squeeze(quant)

            return out

    def predict(self, newdata, n_sim=50):
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
        newdata.variables[self.variable].allocate_simulations(n_sim)
        batch_id = self.options.batch_index(newdata.n_data,
                                            self.options.prediction_batch_size)
        n_batches = len(batch_id)

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(n_batches)), end="")

            output = self.predict_raw(
                _tf.constant(newdata.coordinates[batch], _tf.float64),
                jitter=self.options.jitter, n_sim=n_sim, **prediction_input)

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
    """Variational Gaussian process network.

    A generalization of the standard Gaussian process: variables of any kind
    (continuous, categorical, compositional, directional) are modelled
    through a network of latent Gaussian processes, fitted by maximizing the
    evidence lower bound on inducing points.

    Parameters
    ----------
    data
        The training data, a container from the :mod:`geoml.data` module.
    variables
        The name of a variable in `data` to model, or a list of names.
    likelihoods
        A likelihood from :mod:`geoml.likelihood`, or a list of one per
        variable, in the same order as `variables`.
    latent_network
        The network's terminal node, from :mod:`geoml.latent`. Its size must
        match the likelihoods' sizes summed.
    directional_data
        Structural measurements, whose variable is taken as the gradient of
        the modelled field.
    options
        Training and prediction settings.

    Attributes
    ----------
    training_log : list of float
        The evidence lower bound at each iteration of the last training run.

    Examples
    --------
    >>> import geoml
    >>> geoml.set_seed(1234)
    >>> walker, grid = geoml.datasets.walker()
    >>> inducing = geoml.data.inducing.from_kmeans(walker, 100, seed=0)
    >>> gp = geoml.latent.BasicGP(
    ...     geoml.latent.BasicInput(inducing), size=1)
    >>> model = geoml.models.VGPNetwork(
    ...     walker, "V", geoml.likelihood.Gaussian(), gp)
    >>> model.train_full(max_iter=100)
    >>> model.predict(grid, n_sim=20)
    """

    def __init__(self, data: "_data.PointData",
                 variables: "str | Sequence[str]",
                 likelihoods: "_lk._Likelihood | Sequence[_lk._Likelihood]",
                 latent_network: "_latent.network._LatentVariable",
                 directional_data: "_data.DirectionalData | None" = None,
                 options: "GPOptions | None" = None):
        super().__init__(options=options)

        self.data = data
        self.latent_network = self._register(latent_network)

        if isinstance(likelihoods, _lk._Likelihood):
            likelihoods = [likelihoods]
        self.likelihoods: "list[_lk._Likelihood]" = list(likelihoods)
        self.lik_sizes = [lik.size for lik in self.likelihoods]
        for likelihood in self.likelihoods:
            self._register(likelihood)

        if isinstance(variables, str):
            variables = [variables]
        self.variables: list[str] = list(variables)
        self.var_lengths = [data.variables[v].length for v in self.variables]

        y, has_value = [], []
        for v in self.variables:
            y_v, h_v = data.variables[v].get_measurements()
            y.append(y_v)
            has_value.append(h_v)
        self.y = _np.concatenate(y, axis=1)
        self.has_value = _np.concatenate(has_value, axis=1)
        self.total_data = _np.sum(self.has_value)

        # initializing likelihoods -- declustered where the data carries
        # the column `container.decluster()` keeps, so the warpings start
        # on the field's distribution rather than the sampling's
        stored = data.metadata.get("declustering")
        stored = None if stored is None \
            else _np.asarray(stored.values, dtype=float).ravel()
        for i, v in enumerate(self.variables):
            y, has_value = data.variables[v].get_measurements()
            has_value = _np.all(has_value == 1.0, axis=1)
            y = y[has_value, :]
            self.likelihoods[i].initialize(
                y, weights=None if stored is None else stored[has_value])

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
        s += str(self.latent_network)
        return s

    def to_dot(self, legend=True, rankdir="BT"):
        """
        Writes the model as a Graphviz diagram: coordinates, latent network,
        warpings and output variables.

        See `geoml.graphviz.to_dot`.
        """
        # imported here because that module reads the modules this one needs
        import geoml.viz.graphviz as _gv
        return _gv.to_dot(self, legend=legend, rankdir=rankdir)

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

        # The MAP term: point-estimated parameters that declare a prior pay
        # its log-density here, making the objective a bound on
        # `log p(y, theta)`. Zero unless something declares one, so a model
        # without priors trains on exactly the objective it always did.
        log_prior = self.log_prior()

        self.elbo.assign(elbo - kl + log_prior)
        self.kl_div.assign(kl)
        return elbo - kl + log_prior

    @_tf.function
    def _log_lik(self, x, y, has_value, training_inputs, x_var=None,
                 samples=20, seed=0):
        with _tf.name_scope("batched_elbo"):
            # prediction
            mu, var, sims, _ = self.latent_network.predict(
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

    def _training_step(self, variables, training_inputs):
        """
        One traced training step: the ELBO, its gradient and the update.

        Keeping all three inside a single `tf.function` matters more than it
        looks. Run eagerly — as it is when the tape and `apply_gradients` sit
        outside a graph — Adam issues its update variable by variable, at a
        cost of roughly 1.5 ms each whatever the variable's size. A GP node
        holds three parameters *per expert*, so a network of 40 experts spends
        more time in the optimizer than in the model itself: 278 ms a step
        against 98 ms traced.

        Built once per call to `train_full`/`train_svi` and reused for every
        iteration; building it per iteration would retrace each time and cost
        far more than it saves.
        """
        directions = {}
        if self.directional_data is not None:
            directions = dict(
                x_dir=_tf.constant(
                    self.directional_data.coordinates, _tf.float64),
                directions=_tf.constant(
                    self.directional_data.directions, _tf.float64),
                y_dir=_tf.constant(self.y_dir, _tf.float64),
                has_value_directions=_tf.constant(
                    self.has_value_dir, _tf.float64))

        @_tf.function
        def step(x, y, has_value, x_var):
            with _tf.GradientTape() as tape:
                loss = - self._training_elbo(
                    x, y, has_value, training_inputs, x_var=x_var,
                    samples=self.options.training_samples,
                    jitter=self.options.jitter,
                    seed=self.options.seed,
                    **directions)
            self.optimizer.apply_gradients(
                zip(tape.gradient(loss, variables), variables))

        return step

    def train_full(self, max_iter: int = 1000) -> None:
        """Train on the whole data set at every iteration.

        Feasible while the data and the latent network fit in memory
        together; past that, use :meth:`train_svi`. The evidence lower bound
        of each iteration is appended to `training_log`.

        Parameters
        ----------
        max_iter
            Number of iterations, and a cap rather than a count when
            `options.training_tolerance` asks training to stop once the
            bound settles.

        See Also
        --------
        train_svi : minibatch training, for larger data sets.
        """
        training_inputs = [self.data.variables[v].training_input()
                           for v in self.variables]

        model_variables = self.get_unfixed_variables()
        step = self._training_step(model_variables, training_inputs)

        # the whole data set every iteration, so it is converted once
        x = _tf.constant(self.data.coordinates, _tf.float64)
        y = _tf.constant(self.y, _tf.float64)
        has_value = _tf.constant(self.has_value, _tf.float64)
        x_var = _tf.constant(self.data.get_batched_variance()[0], _tf.float64)

        converged = _Convergence(self.options.training_tolerance)

        # the propagation rule is read when the step traces (and re-traces),
        # which happens inside the loop
        with _latent.propagation_rule(self.options.expert_propagation):
            for i in range(max_iter):
                step(x, y, has_value, x_var)

                for pr in self._all_parameters:
                    pr.refresh()

                current_elbo = self.elbo.numpy()
                self.training_log.append(current_elbo)

                if self.options.verbose:
                    print("\rIteration %s | ELBO: %s" %
                          (str(i+1), str(current_elbo)), end="")

                if converged.stop(current_elbo):
                    if self.options.verbose:
                        print("\nStopped at iteration %s: the bound has "
                              "settled" % str(i + 1), end="")
                    break

        if self.options.verbose:
            print("\n")

    def train_svi(self, epochs: int = 100) -> None:
        """Train in minibatches, by stochastic variational inference.

        Each epoch visits the data once, in batches of
        `options.training_batch_size`, drawn in an order reproducible from
        the model's seed. The mean bound over an epoch's batches is appended
        to `training_log`.

        Parameters
        ----------
        epochs
            Number of passes over the data, and a cap rather than a count
            when `options.training_tolerance` asks training to stop once the
            bound settles. The criterion reads one value an epoch, the mean
            over its batches.

        See Also
        --------
        train_full : one gradient step per iteration on all the data.
        """
        model_variables = self.get_unfixed_variables()

        # The variables' own training inputs are not indexed by batch here, as
        # they are in `train_full` -- see the commented-out attempt below.
        step = self._training_step(
            model_variables, [{} for _ in self.variables])

        # judged once an epoch, on the mean over its batches: a single
        # batch's bound is an estimate with noise of its own, and smoothing
        # that would only measure how the batches were drawn
        converged = _Convergence(self.options.training_tolerance)

        # a generator of its own, so the batch order is reproducible from
        # options.seed without reaching any draw made outside training
        rng = _np.random.default_rng(self.options.seed)
        with _latent.propagation_rule(self.options.expert_propagation):
            for i in range(epochs):
                current_elbo = []

                shuffled = rng.choice(
                    self.data.n_data, self.data.n_data, replace=False)
                batches = self.options.batch_index(self.data.n_data)

                for batch in batches:
                    # training_inputs = [
                    #     self.data.variables[v].training_input(idx)
                    #     for v in self.variables]
                    idx = shuffled[batch]
                    step(_tf.constant(self.data.coordinates[idx], _tf.float64),
                         _tf.constant(self.y[idx], _tf.float64),
                         _tf.constant(self.has_value[idx], _tf.float64),
                         _tf.constant(self.data.get_batched_variance(idx)[0],
                                      _tf.float64))

                    for pr in self._all_parameters:
                        pr.refresh()

                    current_elbo.append(self.elbo.numpy())
                    self.training_log.append(current_elbo[-1])

                total_elbo = _np.mean(current_elbo)
                if self.options.verbose:
                    print("\rEpoch %s | ELBO: %s" %
                          (str(i + 1), str(total_elbo)), end="")

                if converged.stop(total_elbo):
                    if self.options.verbose:
                        print("\nStopped at epoch %s: the bound has settled"
                              % str(i + 1), end="")
                    break

        if self.options.verbose:
            print("\n")

    def predict_raw(self, *args, **kwargs):
        """`_predict_raw` in a graph, XLA-compiled when the options ask for it.

        `jit_compile` is settled when a `tf.function` is built, and the
        simulation draws are baked into the graph, so honouring either option
        means holding one function per combination of settings rather than a
        flag on a single one. Each is traced at most once per model, and
        `None` (rather than `False`) leaves the uncompiled path exactly as it
        was. The `simulation_rule`/`propagation_rule` contexts wrap the call
        rather than the trace because a retrace (a new batch shape, a new
        `n_sim`) can happen on any call, and has to see the flags the cache
        key promised. The propagation rule joins the key for the networks
        whose graphs refresh internally (`ProjectedVGP`); on this class the
        graph reads snapshotted state, and the extra key is merely unused.
        """
        jit = bool(self.options.jit_predict)
        qmc = bool(self.options.qmc_simulations)
        rule = self.options.expert_propagation
        traced = self._compiled.get((jit, qmc, rule))
        if traced is None:
            # `reduce_retracing` relaxes the batch shape, which is the one
            # thing that varies without meaning anything: a grid divides
            # into equal batches and a short last one, and every container
            # of a different size starts the count again. Measured on eight
            # predictions of differing size: eight traces and 7.6 s become
            # two and 2.2 s, bit-identical, and XLA agrees either way. What
            # remains keyed on the value -- `n_sim`, `include_noise` -- is
            # baked into the graph and has to retrace.
            traced = _tf.function(self._predict_raw, jit_compile=jit or None,
                                  reduce_retracing=True)
            self._compiled[(jit, qmc, rule)] = traced
        with _latent.simulation_rule(qmc), _latent.propagation_rule(rule):
            return traced(*args, **kwargs)

    def _predict_raw(self, x_new, variable_inputs, x_var=None,
                     n_sim=1, seed=0, include_noise=True, n_splits=None):
        # The posterior is refreshed once by `predict` and snapshotted into
        # Variables; this cached graph reads that state, so it is not recomputed
        # per batch.
        with _tf.name_scope("Prediction"):
            pred_mu, pred_var, pred_sim, pred_exp_var = \
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
                output.append(
                    lik.predict(
                        mu, var, sim, exp_var,
                        include_noise=include_noise,
                        n_splits=n_splits, **v_inp
                    )
                )
            return output

    def predict(self, newdata: "_data._SpatialData", n_sim: int = 20,
                include_noise: bool = True,
                where: _types.Where = None) -> None:
        """Predict at new locations, writing the answer into the container.

        The variables the model was trained on are created on `newdata` if
        absent and filled in place: prediction, latent moments, simulations,
        and the columns each variable kind adds to those.

        Parameters
        ----------
        newdata
            The locations to predict, of the same dimension as the training
            data. Modified in place.
        n_sim
            Number of realizations to draw per location.
        include_noise
            Whether to integrate the likelihood noise out of the answer. The
            prediction then reports the value the ground would show once
            measurement error and sub-resolution variability are averaged
            over -- a deterministic correction, and a large one on a skewed
            variable. Turn it off to see the latent field alone.
        where
            One boolean per location, the indices of the locations to visit,
            or `None` for all of them. Locations left out keep whatever they
            hold, including their simulations, and a location never visited
            stays missing.

        Notes
        -----
        A location's simulated values do not depend on what else is in its
        batch, so predicting a subset gives the same answer as predicting
        everything and reading that subset back.
        """
        if self.data.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        # managing variables
        variable_inputs = []
        for v in self.variables:
            fresh = v not in newdata.variables.keys()
            if fresh:
                self.data.variables[v].copy_to(newdata)
            # Allocate when the variable is new, whether or not only some
            # locations are being visited: what is not visited stays NaN,
            # which is what `unpredicted` reads and what the reporting layer
            # already skips. Never reallocate an existing one under `where` --
            # that would wipe the simulations the untouched locations hold,
            # which is the whole point of naming only some.
            if where is None or fresh:
                newdata.variables[v].allocate_simulations(n_sim)
            variable_inputs.append(self.data.variables[v].prediction_input())

        def batch_pred(x, x_var, n_splits):
            return self.predict_raw(
                x,
                variable_inputs,
                x_var=x_var,
                seed=self.options.seed,
                n_sim=n_sim,
                include_noise=include_noise,
                n_splits=n_splits
            )

        for batch, output in self._over_batches(newdata, batch_pred, where):
            for v, upd in zip(self.variables, output):
                newdata.variables[v].update(batch, **upd)

    def _over_batches(self, newdata, call, where=None):
        """Runs `call(coordinates, variance, n_splits)` over `newdata`.

        A discretized block fans out into several rows before it reaches the
        model, so the batch is measured in those rows: otherwise
        `prediction_batch_size` would mean `prod(discretization)` times as much
        work here as it does on a grid of points.

        Yields `(rows, result)`, the rows being indices into `newdata` so that
        a caller can write each result back to the location it came from.
        """
        batch_size = max(1, self.options.prediction_batch_size
                         // newdata.rows_per_location)
        rows = _np.arange(newdata.n_data)
        if where is not None:
            where = _np.asarray(where)
            rows = _np.flatnonzero(where) if where.dtype == bool else where
        batch_id = [rows[batch] for batch in
                    self.options.batch_index(len(rows), batch_size=batch_size)]

        # Refresh the posterior once (the parameters are fixed during
        # prediction) and snapshot each node's state into Variables, so the
        # cached `predict_raw` graph reads current values without recomputing the
        # posterior (Cholesky factorizations, etc.) on every batch. The refresh
        # itself is traced -- see `latent.refresh_cached`.
        with _latent.propagation_rule(self.options.expert_propagation):
            if hasattr(self.latent_network, "cache_prediction_state"):
                _latent.refresh_cached(self.latent_network,
                                       self.options.jitter)
            else:
                self.latent_network.refresh(self.options.jitter)

        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch %s of %s       "
                      % (str(i + 1), str(len(batch_id))), end="")

            data_coords, splits = newdata.get_batched_coordinates(batch)
            data_var, _ = newdata.get_batched_variance(batch)

            yield batch, call(_tf.constant(data_coords, _tf.float64),
                              _tf.constant(data_var, _tf.float64), splits)

        if self.options.verbose:
            print("\n")

    def predict_measurements(self, newdata: "_data._SpatialData",
                             n_sim: int = 20,
                             n_nodes: int = 32) -> "dict[str, _np.ndarray]":
        """Draw the predictive distribution of a measurement per location.

        :meth:`predict` reports the ground, with the likelihood noise
        integrated out, so its simulations describe a quantity no sample
        observes. This keeps the noise instead, giving `n_sim * n_nodes`
        equally likely readings per location -- the distribution to compare
        against measured data, as an accuracy plot or a cross-validation
        does.

        Nothing is stored: `newdata` is not modified.

        Parameters
        ----------
        newdata
            Locations to ask about, of the same dimension as the training
            data. Intended for the locations that carry measurements, not
            for a block model.
        n_sim
            Latent realizations per location.
        n_nodes
            Equal-share noise values per realization, so that the two axes
            pool into one sample.

        Returns
        -------
        dict of str to ndarray
            One `(n_data, size, n_sim * n_nodes)` array per variable whose
            likelihood carries a warping. Categorical variables are skipped:
            their noise lives in the probabilities, leaving no value for a
            measurement to scatter around.

        See Also
        --------
        predict : the ground, with the noise integrated out.
        """
        if newdata.rows_per_location != 1:
            raise ValueError(
                "a measurement is of a point, and %s fans each location out "
                "into %d rows; ask this of the data the model was trained "
                "from, or of a validation set"
                % (type(newdata).__name__, newdata.rows_per_location))
        if self.data.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        # a likelihood with no warping has no measurement to describe, so it
        # is passed over rather than asked
        wanted = [(v, lik) for v, lik in zip(self.variables, self.likelihoods)
                  if lik.warped]

        def batch_measure(x, x_var, n_splits):
            with _latent.simulation_rule(self.options.qmc_simulations):
                _, _, sims, _ = self.latent_network.predict(
                    x, x_var=x_var, n_sim=n_sim, seed=[self.options.seed, 0])
            sims = _tf.split(_tf.transpose(sims, [1, 0, 2]),
                             self.lik_sizes, axis=1)
            return [lik.measurement_samples(sim, n_nodes)
                    for sim, lik in zip(sims, self.likelihoods) if lik.warped]

        chunks = {v: [] for v, _ in wanted}
        for _, output in self._over_batches(newdata, batch_measure):
            for (v, _), values in zip(wanted, output):
                chunks[v].append(_np.asarray(values))
        # in the variable's own units: a composition's parts reach the model
        # as fractions of the whole, and what is handed back is compared
        # with assays
        return {v: self.data.variables[v].from_model_units(
                    _np.concatenate(parts, axis=0))
                for v, parts in chunks.items()}

    def responsibilities(self, newdata: "_data._SpatialData",
                         store: bool = True) -> "dict[str, _np.ndarray]":
        """Posterior probability that each measurement came from each noise
        component of a mixture likelihood.

        One answer per location: a :class:`~geoml.likelihood.Mixture` is a
        mixture over the row, so a measurement wrong in one component of a
        vector variable is a wrong measurement.

        Parameters
        ----------
        newdata
            Point data carrying the variables' measurements -- the training
            data, a validation set, or the out-of-fold container
            :func:`cross_validate` returns. Modified in place if `store`.
        store
            Whether to file the answer on each variable, as
            `<variable>/responsibilities/<component>`, as well as return it.

        Returns
        -------
        dict of str to ndarray
            One `(n_data, n_components)` array per variable whose likelihood
            is a mixture, rows summing to one, and missing at locations
            without a measurement.

        Raises
        ------
        ValueError
            If no variable has a mixture likelihood, if `newdata` lacks one
            of those variables, or if it fans each location into several
            rows, as a block model does.

        Notes
        -----
        Read out of fold. At a training location the model interpolates its
        own measurement, so an outlier is partly absorbed into the fit and
        its responsibility understates it.
        """
        if newdata.rows_per_location != 1:
            raise ValueError(
                "a measurement is of a point, and %s fans each location out "
                "into %d rows; ask this of the data the model was trained "
                "from, or of a validation set"
                % (type(newdata).__name__, newdata.rows_per_location))
        if self.data.n_dim != newdata.n_dim:
            raise ValueError("dimension of newdata is incompatible with model")

        wanted = [(v, lik) for v, lik in zip(self.variables, self.likelihoods)
                  if isinstance(lik, _lk.Mixture)]
        if not wanted:
            raise ValueError(
                "no variable in this model has a mixture likelihood, and "
                "responsibilities are a mixture's answer: with one noise "
                "mechanism every measurement came from it")
        absent = [v for v, _ in wanted if v not in newdata.variables.keys()]
        if absent:
            raise ValueError(
                "responsibilities are of measurements, and %s carries no %s"
                % (type(newdata).__name__, ", ".join(str(v) for v in absent)))

        def batch_moments(x, x_var, n_splits):
            mu, var, _, _ = self.latent_network.predict(
                x, x_var=x_var, n_sim=1, seed=[self.options.seed, 0])
            mu = _tf.split(_tf.transpose(mu[:, :, 0]), self.lik_sizes, axis=1)
            var = _tf.split(_tf.transpose(var), self.lik_sizes, axis=1)
            return [(m, v) for m, v, lik
                    in zip(mu, var, self.likelihoods)
                    if isinstance(lik, _lk.Mixture)]

        chunks = {v: ([], []) for v, _ in wanted}
        for _, output in self._over_batches(newdata, batch_moments):
            for (v, _), (mu, var) in zip(wanted, output):
                chunks[v][0].append(_np.asarray(mu))
                chunks[v][1].append(_np.asarray(var))

        out = {}
        for v, lik in wanted:
            mu = _np.concatenate(chunks[v][0], axis=0)
            var = _np.concatenate(chunks[v][1], axis=0)
            y, has_value = newdata.variables[v].get_measurements()
            answer = lik.responsibilities(mu, var, y)
            answer[~_np.all(has_value == 1.0, axis=1)] = _np.nan
            if store:
                newdata.variables[v].set_responsibilities(answer)
            out[v] = answer
        return out


def refine(model, blocks: "_data.BlockSet3D", n_sim: int = 20,
           split_on: "str | Sequence[str] | None" = None,
           tolerance: float = 0.05, include_noise: bool = True,
           where: _types.Where = None,
           meshes: "Sequence[_data.Mesh3D] | None" = None,
           verbose: bool = False) -> "_data.BlockSet3D":
    """Predict on a block model, cutting finer wherever it cannot decide.

    Predicts on the coarse blocks, splits the ones still in doubt, predicts
    only what the split created, and repeats. Three criteria mark a block for
    splitting: its realizations disagree about which side of a cut-off or a
    category boundary it falls on
    (:meth:`~geoml.data.BlockSet3D.needs_splitting`); a neighbour is more
    than one level finer than it (:meth:`~geoml.data.BlockSet3D.unbalanced`);
    or a mesh passes through it (:meth:`~geoml.data.BlockSet3D.crossed_by`).

    The loop ends by itself. Each pass takes the blocks it splits one level
    finer, and no criterion marks a block already at the lattice's
    `max_levels`, so within that many passes every block is either settled or
    as fine as the block model was built to go.

    Parameters
    ----------
    model
        A trained model with a `predict` method, normally a
        :class:`VGPNetwork`.
    blocks
        The coarse block model to start from. Not modified: each pass builds
        a new one.
    n_sim
        Realizations to draw at every pass.
    split_on
        Which variables have a say in the decision. All of them by default.
    tolerance
        The share of realizations that must find a block divided before it is
        cut.
    include_noise
        Passed to :meth:`VGPNetwork.predict`.
    where
        One boolean per block, the indices of the blocks worth modelling, or
        the name of a boolean metadata column holding the same. The rest are
        never predicted and never cut, and keep their missing values. Given
        once, against the blocks as they stand: the mask is carried across
        each split.
    meshes
        Surfaces and closed bodies whose crossings force a split. Costs a
        side test per sub-block of every splittable block, each pass.
    verbose
        Print what each pass cut.

    Returns
    -------
    BlockSet3D
        The refined model, predicted throughout.

    Notes
    -----
    What decides a split carries no noise: the likelihood noise is
    integrated out rather than drawn, so a block never straddles a cut-off on
    account of spread that splitting cannot resolve.

    Only the blocks a pass creates are predicted. A block that was not split
    is the same block on the same support, and its value still stands.

    Written out, the loop is three calls -- predict, ask, split -- which is
    the way to stop part way and inspect a pass. See
    `docs/variable-block-models.md`.
    """
    keep = None
    if where is not None:
        if isinstance(where, str):
            # a stored filter is an ordinary metadata column, which is what
            # the block model has instead of being subsettable
            keep = _np.asarray(
                blocks.get_metadata(where)).ravel().astype(bool)
        else:
            keep = _np.asarray(where)
            if keep.dtype != bool:
                index = _np.zeros(blocks.n_data, dtype=bool)
                index[keep] = True
                keep = index
        if keep.shape != (blocks.n_data,):
            raise ValueError(
                "`where` needs one value per block: got %d for %d"
                % (keep.size, blocks.n_data))

    model.predict(blocks, n_sim=n_sim, include_noise=include_noise, where=keep)

    step = 0
    while True:
        undecided = blocks.needs_splitting(split_on, tolerance=tolerance)
        uneven = blocks.unbalanced()
        crossed = _np.zeros(blocks.n_data, dtype=bool)
        for mesh in (meshes or []):
            crossed |= blocks.crossed_by(mesh)
        mask = undecided | uneven | crossed
        if keep is not None:
            # a block nobody asked for holds nothing to decide and draws no
            # surface, so cutting it would only make more of nothing
            mask = mask & keep
        if not _np.any(mask):
            return blocks

        step += 1
        # `split` keeps the unsplit blocks first, then each parent's children
        # in sub-block order, which is how the mask follows them across
        children = blocks.rows_per_location
        blocks = blocks.split(mask)
        visit = blocks.unpredicted()
        if keep is not None:
            keep = _np.concatenate(
                [keep[~mask], _np.repeat(keep[mask], children)])
            visit = visit & keep
        model.predict(blocks, n_sim=n_sim, include_noise=include_noise,
                      where=visit)
        if verbose:
            print("pass %d: cut %d block(s) (%d undecided, %d crossed by a "
                  "mesh, %d to level a jump), %d now"
                  % (step, int(_np.count_nonzero(mask)),
                     int(_np.count_nonzero(undecided)),
                     int(_np.count_nonzero(crossed & ~undecided)),
                     int(_np.count_nonzero(uneven & ~undecided & ~crossed)),
                     blocks.n_data))


# What encodes the data in a trained VGP, and how each piece starts over.
# Every node of the latent network registers these per inducing set (see
# `BasicGP._set_parameters`); everything else is a hyperparameter.
_VARIATIONAL_STATE = {
    "alpha_white_": lambda shape: _srandom.rng().normal(scale=1e-3,
                                                        size=shape),
    "delta_": _np.ones,
    "bias_": _np.zeros,
}


def _fresh_variational_state(model):
    """Freeze what one fold cannot change; forget what it can.

    The variational state -- `alpha_white_*`, `delta_*` and `bias_*` on every
    node of the latent network -- is where the data lives in a trained VGP, so
    a fold model gets it factory-fresh: re-initialized, it is structurally
    ignorant of the held-out rows, and no iterations are spent forgetting
    them. Everything else (kernels, warpings, likelihood noise) is fixed where
    it stands -- the concession kriging cross-validation makes when it keeps
    the fitted variogram, made once and said out loud in `cross_validate`'s
    docstring. The fresh values are drawn from the package generator, so
    `geoml.set_seed` makes the whole procedure reproducible.
    """
    for parameter in model._all_parameters:
        parameter.fix()

    network = model.latent_network
    for node in [network] + network.get_unique_parents():
        for name, parameter in node.parameters.items():
            for prefix, init in _VARIATIONAL_STATE.items():
                if name.startswith(prefix):
                    shape = _np.asarray(parameter.get_value()).shape
                    parameter.set_value(init(shape))
                    parameter.unfix()
                    break


def cross_validate(model: VGPNetwork, folds: str = "fold",
                   refit: str = "variational", iterations: int = 200,
                   n_sim: int = 20, n_nodes: int = 32,
                   path: _types.PathLike | None = None
                   ) -> "tuple[_data._SpatialData, _pd.DataFrame]":
    """Score a model on folds it never saw, with one short refit per fold.

    The trained model is saved once. Each fold gets a copy rebuilt around the
    data with that fold removed; under `refit="variational"` its variational
    state -- the part of a trained model that encodes the data -- is
    re-initialized and every other parameter frozen, so the fold model starts
    ignorant of the held-out rows, and only that state is refitted. The fold
    model then predicts its held-out rows, and only those, into one shared
    copy of the training data. Folds partition the data, so every location
    ends up predicted by a model that never saw it.

    Scores are of *measurements*: the held-out values are samples, so each
    fold model is asked through :meth:`VGPNetwork.predict_measurements`.
    Categorical variables get no rows -- subset the returned container by
    fold and use the variable's own `compute_metrics`.

    Parameters
    ----------
    model
        A trained model. It is saved and copied; the original is untouched.
    folds
        Name of the metadata column holding the fold labels, as
        :meth:`~geoml.data.PointData.spatial_k_fold` writes. Any labelling
        works: a hole-id column gives leave-one-hole-out.
    refit
        `"variational"` to refit the variational state alone, or `"all"` to
        warm-start every trainable parameter from its trained value and
        continue on the reduced data.
    iterations
        Training iterations per fold.
    n_sim
        Latent realizations, for the out-of-fold predictions and the
        measurement samples alike.
    n_nodes
        Noise values per realization in the measurement samples.
    path
        Where to keep the saved model and its fold copies. A temporary
        directory, removed at the end, unless one is given.

    Returns
    -------
    oof : container
        A copy of the training data carrying out-of-fold predictions and
        simulations, plus one metadata column per scored component
        (`pit_<variable>`, or `pit_<variable>_<component>`) holding where
        each measurement fell inside its own predictive distribution.
    scores : pandas.DataFrame
        One row per component and fold, plus a pooled `"all"` row, with
        `rmse`, `mae`, `bias`, `crps` and `goodness` against the held-out
        measurements.

    See Also
    --------
    conformalize : calibrate interval widths on the PIT columns.
    geoml.data.PointData.spatial_k_fold : folds that mimic a prediction task.

    Notes
    -----
    The hyperparameters and the warping were fitted on all the data,
    including each fold's -- the concession kriging cross-validation also
    makes when it keeps the variogram fixed. Design record and measurements:
    `docs/cross-validation.md`.
    """
    data = model.data
    labels = _np.asarray(data.get_metadata(folds))
    if _pd.isna(labels).any():
        raise ValueError(
            "every location needs a fold; column '%s' has missing entries"
            % folds)
    fold_names = _np.unique(labels)
    if fold_names.size < 2:
        raise ValueError(
            "cross-validation needs at least 2 folds; column '%s' holds %d"
            % (folds, fold_names.size))
    if refit not in ("variational", "all"):
        raise ValueError(
            "refit must be 'variational' or 'all', got %r" % (refit,))

    cleanup = path is None
    if cleanup:
        path = _tempfile.mkdtemp(prefix="geoml_cv_")
    saved = _os.path.join(path, "model")

    # one shared answer sheet: every fold writes only the rows it held out,
    # and the folds partition the data, so nothing stale survives the loop
    oof = data[_np.ones(data.n_data, dtype=bool)]
    for v in model.variables:
        oof.variables[v].allocate_simulations(n_sim)

    rows = []
    acc = {}
    pit = {}
    try:
        _persistence.save_model(model, saved)
        for fold in fold_names:
            held = labels == fold
            fold_model = _cast(VGPNetwork,
                               _persistence.load_model(saved, data=data[~held]))
            if refit == "variational":
                _fresh_variational_state(fold_model)
            fold_model.train_full(max_iter=iterations)
            fold_model.predict(oof, n_sim=n_sim, include_noise=True,
                               where=held)

            held_points = oof[held]
            samples = fold_model.predict_measurements(
                held_points, n_sim=n_sim, n_nodes=n_nodes)
            for v, sample in samples.items():
                y_true, has_value = \
                    held_points.variables[v].get_measurements()
                y_true = _np.asarray(y_true, dtype=float)
                has_value = _np.asarray(has_value)
                if y_true.ndim == 1:
                    y_true = y_true[:, None]
                if has_value.ndim == 1:
                    has_value = has_value[:, None]

                components = getattr(held_points.variables[v], "labels", None)
                components = [v] if components is None else list(components)
                for c, component in enumerate(components):
                    measured = has_value[:, min(c, has_value.shape[1] - 1)] \
                        == 1
                    if not measured.any():
                        continue
                    truth = y_true[measured, c]
                    draw = sample[measured, c, :]
                    point = draw.mean(axis=1)
                    nominal, observed = _metrics.coverage(truth, draw)
                    n = int(measured.sum())
                    score_crps = _metrics.crps(truth, draw)

                    # where each assay fell inside its own predictive
                    # distribution (mid-rank, so ties split evenly) --
                    # what `conformalize` calibrates on
                    u = (draw < truth[:, None]).mean(axis=1) \
                        + 0.5 * (draw == truth[:, None]).mean(axis=1)
                    column = pit.setdefault(
                        (v, component),
                        _np.full(data.n_data, _np.nan))
                    column[_np.flatnonzero(held)[measured]] = u
                    rows.append({
                        "variable": v, "component": component, "fold": fold,
                        "n": n,
                        "rmse": _metrics.rmse(truth, point),
                        "mae": _metrics.mae(truth, point),
                        "bias": _metrics.bias(truth, point),
                        "crps": score_crps,
                        "goodness": _metrics.goodness(nominal, observed),
                    })

                    # sufficient statistics, so the pooled row needs no
                    # second pass over the samples
                    a = acc.setdefault((v, component), {
                        "n": 0, "sse": 0.0, "sae": 0.0, "se": 0.0,
                        "crps": 0.0,
                        "observed": _np.zeros_like(observed),
                        "nominal": nominal,
                    })
                    a["n"] += n
                    a["sse"] += float(((point - truth) ** 2).sum())
                    a["sae"] += float(_np.abs(point - truth).sum())
                    a["se"] += float((point - truth).sum())
                    a["crps"] += score_crps * n
                    a["observed"] = a["observed"] + observed * n
    finally:
        if cleanup:
            _shutil.rmtree(path, ignore_errors=True)

    for (v, component), a in acc.items():
        n = a["n"]
        rows.append({
            "variable": v, "component": component, "fold": "all", "n": n,
            "rmse": float(_np.sqrt(a["sse"] / n)),
            "mae": a["sae"] / n,
            "bias": a["se"] / n,
            "crps": a["crps"] / n,
            "goodness": _metrics.goodness(a["nominal"], a["observed"] / n),
        })

    for (v, component), column in pit.items():
        oof.add_metadata(_pit_column(v, component), column)

    return oof, _pd.DataFrame(rows)


def _pit_column(name, component):
    """Where `cross_validate` keeps a component's out-of-fold PITs."""
    if component is None or component == name:
        return "pit_%s" % name
    return "pit_%s_%s" % (name, component)


class ConformalCalibration:
    """Interval coverage repaired from out-of-fold PIT values.

    Split conformal prediction on the score :math:`|u - 1/2|`, where `u` is
    where a held-out measurement fell inside its own predictive
    distribution. :meth:`nominal` answers: at what level must a central
    interval be cut so that it covers a given share of fresh measurements?
    For a calibrated model that is the share itself; an overconfident model
    is told to cut wider and a hedging one narrower. The conformal quantile
    carries the finite-sample guarantee -- coverage at least the share asked
    for -- under exchangeability of the calibration scores with the
    prediction's.

    Parameters
    ----------
    pit
        Probability integral transform values, one per calibration
        measurement, as :func:`cross_validate` stores them.

    Raises
    ------
    ValueError
        If no finite PIT values are given.

    Notes
    -----
    Spatial data is not exchangeable point by point, which is what the folds
    are for: built to mimic the prediction task, they draw the calibration
    scores from conditions like deployment's. The intervals are of
    measurements -- the ground is never observed.

    The repair is bounded by the ensemble: an interval cut from samples
    cannot reach past their range, so `nominal(q) == 1.0` means the model was
    too sure for its samples to say how much wider the interval should be.
    Raise `n_sim` or `n_nodes`, or reconsider the model.

    References
    ----------
    Vovk, V., Gammerman, A. and Shafer, G. (2005) *Algorithmic Learning in a
    Random World*. Springer.
    """

    def __init__(self, pit: _types.ArrayLike):
        pit = _np.asarray(pit, dtype=float).ravel()
        pit = pit[_np.isfinite(pit)]
        if pit.size == 0:
            raise ValueError("there are no PIT values to calibrate on")
        self._scores = _np.sort(_np.abs(pit - 0.5))

    def nominal(self, coverage: float) -> float:
        """The level to cut a central interval at, to cover `coverage`.

        Parameters
        ----------
        coverage
            The share of fresh measurements the interval should contain.

        Returns
        -------
        float
            The level to pass to :meth:`interval`, between 0 and 1. A value
            of 1 means the calibration data cannot say how much wider the
            interval must be.
        """
        n = self._scores.size
        k = int(_np.ceil((n + 1) * float(coverage)))
        if k > n:
            return 1.0
        return min(1.0, 2.0 * float(self._scores[k - 1]))

    def interval(self, samples: _types.ArrayLike, coverage: float = 0.9
                 ) -> "tuple[_types.FloatArray, _types.FloatArray]":
        """A calibrated central interval per row of measurement samples.

        Parameters
        ----------
        samples
            One component's measurement samples, of shape
            `(n_data, n_samples)`, as :meth:`VGPNetwork.predict_measurements`
            returns them once sliced to the component.
        coverage
            The share of fresh measurements the interval should cover.

        Returns
        -------
        lower, upper : ndarray
            One bound per location, of shape `(n_data,)`.
        """
        level = self.nominal(coverage)
        samples = _np.asarray(samples, dtype=float)
        lower = _np.quantile(samples, max(0.0, 0.5 - level / 2), axis=-1)
        upper = _np.quantile(samples, min(1.0, 0.5 + level / 2), axis=-1)
        return lower, upper


def conformalize(oof: "_data._SpatialData", name: str,
                 component: str | None = None) -> ConformalCalibration:
    """Build a conformal calibration from a cross-validation.

    Reads the out-of-fold PIT column :func:`cross_validate` left on its
    container.

    Parameters
    ----------
    oof
        The container :func:`cross_validate` returned.
    name
        The variable.
    component
        Which component, when the variable is a vector or compositional one.

    Returns
    -------
    ConformalCalibration
        Calibrated on that component's out-of-fold measurements.

    Examples
    --------
    >>> oof, scores = geoml.models.cross_validate(model)
    >>> calibration = geoml.models.conformalize(oof, "grade")
    >>> samples = model.predict_measurements(new_points)["grade"]
    >>> lower, upper = calibration.interval(samples[:, 0, :], coverage=0.9)
    """
    return ConformalCalibration(oof.get_metadata(_pit_column(name, component)))


class StructuralField(_GPModel):
    """Structural field modeling based on gradient data."""
    def __init__(self, tangents, covariance, normals=None, mean_vector=None,
                 options=None):
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

        # cleared here and filled by `refresh()`, as in `GP`
        self.cov: _Any = None
        self.cov_chol: _Any = None
        self.cov_inv: _Any = None
        self.scale: _Any = None
        self.alpha: _Any = None
        self.y: _Any = None
        self.all_coordinates: _Any = None
        self.all_directions: _Any = None

    def __repr__(self):
        s = "Gaussian process structural field model\n\n"
        s += "Kernel:\n"
        s += str(self.covariance)
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
            # `update` names the columns the way the likelihoods fill them:
            # `average_sim` is the prediction and `mean`/`variance` the
            # latent pair. This model has no likelihood and no warping to
            # separate them -- the potential field is the latent value --
            # so the prediction is the mean itself.
            output = {"average_sim": _tf.squeeze(mu),
                      "mean": _tf.squeeze(mu),
                      "variance": _tf.squeeze(var)}

            newdata.variables[variable].update(batch, **output)

        if self.options.verbose:
            print("\n")


class _EnsembleModel(_GPModel):
    def __init__(self, options=None):
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
                 use_trend=False, options=None):
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


# class VGPNetworkEnsemble(_EnsembleModel):
#     def __init__(self, data, variables, likelihoods, latent_networks,
#                  directional_data=None, options=GPOptions()):
#         super().__init__(options)
#         if not isinstance(data, (tuple, list)):
#             raise ValueError("data must be a list or tuple containing"
#                              "data objects")
#         if not isinstance(latent_networks, (tuple, list)):
#             raise ValueError("latent_trees must be a list or tuple containing"
#                              "latent variable objects")
#         if directional_data is None:
#             directional_data = [None for _ in data]
#         elif not isinstance(directional_data, (tuple, list)):
#             raise ValueError("directional_data must be a list or tuple"
#                              "containing data objects or None")
#
#         dims = set([d.n_dim for d in data])
#         if len(dims) != 1:
#             raise Exception("all data objects must have the same dimension")
#         self._n_dim = list(dims)[0]
#
#         self.models = [VGPNetwork(
#             data=d,
#             variables=variables,
#             likelihoods=_copy.deepcopy(likelihoods),
#             # likelihoods=lik,
#             latent_network=l,
#             directional_data=dd,
#             options=options)
#             # for d, l, dd, lik in zip(
#             #     data, latent_networks, directional_data, likelihoods)
#             for d, l, dd in zip(
#                 data, latent_networks, directional_data)
#         ]
#         for model in self.models:
#             self._register(model)
#
#         if not (isinstance(variables, (list, tuple))):
#             variables = [variables]
#         self.variables = variables
#
#     def __repr__(self):
#         s = "Gaussian process ensemble\n\n" \
#             "Models: %d\n\n" % len(self.models)
#         s += "Variables:\n "
#         for v, lik in zip(self.variables, self.models[0].likelihoods):
#             s += "\t" + v + " (" + lik.__class__.__name__ + ")\n"
#         return s
#
#     # def train_full(self, cycles=10, max_iter_per_model=100):
#     #     for c in range(cycles):
#     #         for i, model in enumerate(self.models):
#     #             if self.options.verbose:
#     #                 print("Cycle %d of %d - training model %d of %d" %
#     #                       (c + 1, cycles, i + 1, len(self.models)))
#     #
#     #             model.train_full(max_iter=max_iter_per_model)
#     #
#     # def train_svi(self, cycles=10, epochs_per_model=10):
#     #     for c in range(cycles):
#     #         for i, model in enumerate(self.models):
#     #             if self.options.verbose:
#     #                 print("Cycle %d of %d - training model %d of %d" %
#     #                       (c + 1, cycles, i + 1, len(self.models)))
#     #
#     #             model.train_svi(epochs=epochs_per_model)
#
#     def train_full(self, max_iter=1000):
#         for model in self.models:
#             model.train_full(max_iter)
#
#     def train_svi(self, epochs=100):
#         for model in self.models:
#             model.train_svi(epochs)
#
#     def predict(self, newdata, n_sim=20):
#         """
#         Makes a prediction on the specified coordinates.
#
#         Parameters
#         ----------
#         newdata :
#             A reference to a spatial points object of compatible dimension.
#             The object's variables are updated.
#         n_sim : int
#             Number of predictive samples to draw.
#         """
#         if self.n_dim != newdata.n_dim:
#             raise ValueError("dimension of newdata is incompatible with model")
#
#         # managing variables
#         variable_inputs = []
#         for v in self.variables:
#             if v not in newdata.variables.keys():
#                 self.models[0].data.variables[v].copy_to(newdata)
#             newdata.variables[v].allocate_simulations(n_sim)
#             variable_inputs.append(
#                 self.models[0].data.variables[v].prediction_input())
#
#         # prediction in batches
#         batch_id = self.options.batch_index(
#             newdata.n_data, batch_size=self.options.prediction_batch_size)
#         n_batches = len(batch_id)
#
#         def batch_pred(model, x):
#             out = model.predict_raw(
#                 x,
#                 variable_inputs,
#                 seed=self.options.seed,
#                 n_sim=n_sim,
#                 jitter=self.options.jitter
#             )
#             return out
#
#         @_tf.function
#         def combined_pred(x):
#             outputs = [batch_pred(model, x) for model in self.models]
#             return self.combine(outputs)
#
#         for i, batch in enumerate(batch_id):
#             if self.options.verbose:
#                 print("\rProcessing batch %s of %s       "
#                       % (str(i + 1), str(n_batches)), end="")
#
#             # outputs = [batch_pred(
#             #     model,
#             #     _tf.constant(newdata.coordinates[batch], _tf.float64))
#             #     for model in self.models]
#             #
#             # output = self.combine(outputs)
#
#             output = combined_pred(
#                 _tf.constant(newdata.coordinates[batch], _tf.float64))
#
#             for v, upd in zip(self.variables, output):
#                 newdata.variables[v].update(batch, **upd)
#
#         if self.options.verbose:
#             print("\n")
#
#     @_tf.function
#     def combine(self, outputs):
#         combined = [{} for _ in self.variables]
#         for i, variable in enumerate(self.variables):
#             var_keys = outputs[0][i].keys()
#
#             weights = _tf.stack([out[i]["weights"] for out in outputs], axis=1)
#             weights = weights + 1e-6
#             weights = weights / _tf.reduce_sum(weights, axis=1, keepdims=True)
#
#             for key in var_keys:
#                 if key != "weights":
#                     tensor = _tf.stack([out[i][key] for out in outputs], axis=1)
#                     if "variance" in key:
#                         w = weights**2
#                     else:
#                         w = weights
#
#                     w = _tf.cond(_tf.greater_equal(_tf.rank(tensor), 3),
#                                  lambda: _tf.expand_dims(w, axis=-1),
#                                  lambda: w)
#                     w = _tf.cond(_tf.greater_equal(_tf.rank(tensor), 4),
#                                  lambda: _tf.expand_dims(w, axis=-1),
#                                  lambda: w)
#
#                     tensor = _tf.reduce_sum(w * tensor, axis=1)
#                     combined[i][key] = tensor
#
#         return combined


class Normalizer(_GPModel):
    """Trainable data normalizer."""
    def __init__(self, warping, options=None):
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
        # self.optimizer = _tf.keras.optimizers.Nadam(
        #     _tf.keras.optimizers.schedules.ExponentialDecay(1e-1, 1, 0.99),
        # )
        self.optimizer = _tf.keras.optimizers.Adam(
            _tf.keras.optimizers.schedules.ExponentialDecay(1e-1, 1, 0.999),
            amsgrad=True#, clipvalue=0.001
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
            mean = _tf.reduce_mean(x_warp, axis=0, keepdims=True)
            var = _tf.math.reduce_variance(x_warp, axis=0, keepdims=True)
            std = _tf.sqrt(var)
            log_derivative = self.warping.log_derivative(x)

            # kl = _tf.reduce_sum(_tf.math.log(std) + (1 + mean ** 2) / (2 * var) - 0.5)
            kl = _tf.reduce_sum(var + mean**2 - 1 - _tf.math.log(var)) * 0.5
            density = _tf.reduce_mean(_tf.reduce_sum(-x_warp**2 / 2, axis=1, keepdims=True) + log_derivative)
            obj = density #- kl

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

    def _predict_raw(self, x_new, variable_inputs, x_var=None,
                     n_sim=1, seed=0, jitter=1e-6, include_noise=True):
        # `predict_raw` is inherited: it compiles this the way the options ask.
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


def search_throw(model: "VGPNetwork", fault: "_gpr.Parametric",
                 throws: _types.ArrayLike, iterations: int = 50,
                 learning_rate: float = 0.1, parameter: str = "throw",
                 refine: int = 0) -> "tuple[float, list]":
    """
    Chooses a fault's throw among candidates, on the bound.

    A fault's throw does not always train from zero: the bound is flat
    in it away from the right value, with a categorical likelihood or with
    several faults it was measured not to move at all. This is the search
    that gradient descent cannot do. For each candidate the model is
    restored to the state it started in, the throw set, a short burst of
    training run with a fresh optimizer, and the bound at its end recorded;
    the best candidate is set on the restored model, which is then trained
    as usual. A workflow rather than something a model is, like `refine`
    and `cross_validate`.

    Parameters
    ----------
    model
        The model, built and possibly trained; left in its starting state
        with the chosen throw set and a fresh optimizer at
        `learning_rate`.
    fault
        The `FaultDisplacement` (any transform holding the parameter).
    throws
        The candidate values.
    iterations
        Training iterations per candidate.
    learning_rate
        The rate of the bursts, and of the optimizer the model is left
        with.
    parameter
        The parameter searched, `"throw"` or `"strike_slip"`.
    refine
        Further passes, each over five candidates spanning one step of the
        previous pass either side of its best.

    Returns
    -------
    best : float
        The candidate chosen.
    passes : list of (candidates, scores)
        One pair of arrays per pass, the score being the bound each
        candidate reached, mean of its burst's last fifth.
    """
    candidates = _np.asarray(throws, dtype=float).ravel()
    value, shape, position, _, _ = model.get_parameter_values(complete=True)
    start = len(model.training_log)
    passes = []
    best = float(candidates[0])
    for level in range(refine + 1):
        scores = []
        for throw in candidates:
            model.update_parameters(value, shape, position)
            fault.parameters[parameter].set_value(float(throw))
            model.set_learning_rate(learning_rate)
            model.train_full(max_iter=iterations)
            burst = model.training_log[-max(1, iterations // 5):]
            scores.append(float(_np.mean(burst)))
        scores = _np.asarray(scores)
        passes.append((candidates, scores))
        best = float(candidates[int(_np.nanargmax(scores))])
        if level < refine:
            spacing = (candidates.max() - candidates.min()) \
                / max(len(candidates) - 1, 1)
            candidates = _np.linspace(best - spacing, best + spacing, 5)
    del model.training_log[start:]
    model.update_parameters(value, shape, position)
    fault.parameters[parameter].set_value(best)
    model.set_learning_rate(learning_rate)
    return best, passes
