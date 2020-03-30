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
# MERCHANTABILITY or FITNESS FOR matrix PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# __all__ = ["GP", "GPGrad", "GPMultiClassifier",
# "SparseGP", "SparseGPEnsemble"]

import geoml.swarm as _pso
import geoml.warping as _warp
import geoml.tftools as _tftools
import geoml.data as _data

import numpy as _np
import tensorflow as _tf
import pandas as _pd
import copy as _copy
import pickle as _pickle

import tensorflow_probability as tfp
tfd = tfp.distributions


class _ModelOptions:
    def __init__(self, verbose=True, prediction_batch_size=20000,
                 seed=1234, n_sim=100):
        self.verbose = verbose
        self.prediction_batch_size = prediction_batch_size
        self.seed = seed
        self.n_sim = n_sim

    def batch_id(self, n_data, batch_size=None):
        if batch_size is None:
            batch_size = self.prediction_batch_size

        n_batches = int(_np.ceil(n_data / batch_size))
        idx = [_np.arange(i * batch_size,
                          _np.minimum((i + 1) * batch_size,
                                      n_data))
               for i in range(n_batches)]
        return idx


class GPOptions(_ModelOptions):
    def __init__(self, verbose=True, prediction_batch_size=20000,
                 seed=1234, n_sim=100, add_noise=False, jitter=1e-9,
                 lanczos_iterations=100, training_batch_size=2000):
        super().__init__(verbose, prediction_batch_size, seed, n_sim)
        self.add_noise = add_noise
        self.jitter = jitter
        self.lanczos_iterations = lanczos_iterations
        self.training_batch_size = training_batch_size


class _Model:
    """Base model class"""

    def __init__(self):
        self._ndim = None
        self._pre_computations = {}

    @property
    def ndim(self):
        return self._ndim

    def train(self, **kwargs):
        """
        Calls the genetic algorithm to train the model.

        Parameters
        ----------
        kwargs :
            Arguments passed on to geoml.genetic.training_real() function.

        """
        pass

    def save_state(self, file):
        pass

    def load_state(self, file):
        pass

    def _refresh(self):
        """Updates the pre_computed values."""
        pass


class GP(_Model):
    """
    Base Gaussian Process model, not optimized for a large amount of data.
    It is expected to work well with up to ~5000 data points.

    Internally the model assumes unit range, unit variance and
    zero mean.

    Attributes
    ----------

    """

    def __init__(self, spatial_data, variable, kernel,
                 points_to_honor=None, options=GPOptions()):
        """
        Initializer for GP.

        Parameters
        ----------
        spatial_data :
            Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        kernel : Kernel
            A kernel object.
        """
        super().__init__()
        self.data = spatial_data
        self.y_name = variable
        self._ndim = spatial_data.ndim
        self.optimizer = None
        self.options = options
        self.kernel = kernel
        self.points_to_honor = points_to_honor

        # tidying up
        self.kernel.set_limits(spatial_data)
        self._initialize_pre_computed_variables()
        self._refresh()

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        s += "\nKernel:\n"
        s += repr(self.kernel)
        return s

    def _initialize_pre_computed_variables(self):
        not_nan = ~_np.isnan(self.data.data[self.y_name].values)
        n_data = self.data.coords.shape[0]
        y = _tf.constant(self.data.data[self.y_name].values[not_nan],
                         _tf.float64)

        if self.points_to_honor is not None:
            points_to_honor = _tf.constant(
                self.data.data[self.points_to_honor].values[not_nan],
                _tf.bool)
        else:
            points_to_honor = _tf.constant([False] * n_data, _tf.bool)

        self._pre_computations = {
            "y": _tf.expand_dims(y, 1),
            "coords": _tf.constant(self.data.coords[not_nan], _tf.float64),
            "alpha": _tf.Variable(_tf.zeros([n_data, 1], _tf.float64)),
            "chol": _tf.Variable(_tf.zeros([n_data, n_data], _tf.float64)),
            "log_lik": _tf.Variable(_tf.constant(0.0, _tf.float64)),
            "points_to_honor": points_to_honor,
        }

    def _refresh(self, only_training_variables=False):
        self._update_pre_computations(self.options.jitter)

    @_tf.function
    def _update_pre_computations(self, jitter):
        y = self._pre_computations["y"]
        n_data = _tf.shape(y)[0]
        points_to_honor = self._pre_computations["points_to_honor"]

        coords = self._pre_computations["coords"]

        cov_mat = self.kernel.self_covariance_matrix(coords, points_to_honor)
        cov_mat = cov_mat + _tf.eye(n_data, dtype=_tf.float64) * jitter

        chol = _tf.linalg.cholesky(cov_mat)
        alpha = _tf.linalg.cholesky_solve(chol, y)

        fit = - 0.5 * _tf.reduce_sum(alpha * y, name="fit")
        det = - _tf.reduce_sum(_tf.math.log(_tf.linalg.tensor_diag_part(
            chol)), name="det")
        const = - 0.5 * _tf.cast(n_data, _tf.float64) \
                * _tf.constant(_np.log(2 * _np.pi), _tf.float64)
        log_lik = fit + det + const

        self._pre_computations["alpha"].assign(alpha)
        self._pre_computations["chol"].assign(chol)
        self._pre_computations["log_lik"].assign(log_lik)

    @_tf.function
    def _predict(self, x_new, n_sim=0):
        with _tf.name_scope("Prediction"):
            coords = self._pre_computations["coords"]
            k_new = self.kernel.covariance_matrix(coords, x_new)
            pred_mu = _tf.matmul(k_new, self._pre_computations["alpha"],
                                 True, False)
            with _tf.name_scope("pred_var"):
                point_var = self.kernel.point_variance(x_new)
                info_gain = _tf.linalg.cholesky_solve(
                    self._pre_computations["chol"], k_new)
                info_gain = _tf.multiply(k_new, info_gain)
                pred_var = point_var - _tf.reduce_sum(info_gain, axis=0)
        return pred_mu, pred_var

    def log_lik(self):
        """
        Outputs the model's log-likelihood, given the current parameters.

        Returns
        -------
        log_lik : double
            The model's log-likelihood.
        """
        return self._pre_computations["log_lik"]

    def train(self, **kwargs):

        value, k_shape, k_position, \
            min_val, max_val = self.kernel.get_parameter_values()

        start = (value - min_val) / (max_val - min_val + 1e-6)

        def fitness(sol, finished=False):
            sol = sol * (max_val - min_val) + min_val

            self.kernel.update_parameters(sol, k_shape, k_position)

            self._refresh(only_training_variables=~finished)

            return self.log_lik().numpy()

        self.optimizer = _pso.ParticleSwarmOptimizer(len(start))
        self.optimizer.optimize(fitness, start=start, **kwargs)
        fitness(self.optimizer.best_position, finished=True)

    def predict(self, newdata, name=None):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            matrix reference to a spatial points object of compatible dimension.
            The prediction is written on the object's data attribute in-place.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # prediction in batches
        n_data = x_new.shape[0]
        batch_id = self.options.batch_id(n_data)
        n_batches = len(batch_id)

        mu = []
        var = []
        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            # TensorFlow
            mu_i, var_i = self._predict(x_new[batch])

            # update
            mu.append(mu_i)
            var.append(var_i)
        if self.options.verbose:
            print("\n")
        mu = _tf.squeeze(_tf.concat(mu, axis=0))
        var = _tf.squeeze(_tf.concat(var, axis=0))

        # output
        newdata.data[name + "_mean"] = mu.numpy()
        newdata.data[name + "_variance"] = var.numpy()

    def cross_validation(self, partition=None):
        raise NotImplementedError("to be implemented")

    def save_state(self, file):
        kernel_parameters = self.kernel.get_parameter_values(complete=True)
        with open(file, 'wb') as f:
            _pickle.dump(kernel_parameters, f)

    def load_state(self, file):
        with open(file, 'rb') as f:
            kernel_parameters = _pickle.load(f)

        k_value, k_shape, k_position, k_min_val, k_max_val = kernel_parameters
        self.kernel.update_parameters(k_value, k_shape, k_position)


class WarpedGP(GP):
    """
    Base Gaussian Process model, not optimized for a large amount of data.
    It is expected to work well with up to ~5000 data points.
    
    Internally the model assumes unit range, unit variance and
    zero mean.
    
    Attributes
    ----------

    """

    def __init__(self, spatial_data, variable, kernel,
                 warping=None, points_to_honor=None, options=GPOptions()):
        """
        Initializer for GP.

        Parameters
        ----------
        spatial_data :
            Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        kernel : Kernel
            A kernel object.
        warping : Warping
            A warping object.
        """
        self.warping = warping
        if warping is None:
            self.warping = _warp.Identity()

        super().__init__(spatial_data, variable, kernel,
                         points_to_honor=points_to_honor,
                         options=options)

    def __repr__(self):
        s = self.__class__.__name__ + "\n"
        s += "\nKernel:\n"
        s += repr(self.kernel)
        s += "\nWarping:\n"
        s += self.warping.__repr__()
        return s

    def _initialize_pre_computed_variables(self):
        not_nan = ~_np.isnan(self.data.data[self.y_name].values)
        n_data = self.data.coords.shape[0]
        y = _tf.constant(self.data.data[self.y_name].values[not_nan],
                         _tf.float64)
        self.warping.refresh(y)

        if self.points_to_honor is not None:
            points_to_honor = _tf.constant(
                self.data.data[self.points_to_honor].values[not_nan],
                _tf.bool)
        else:
            points_to_honor = _tf.constant([False]*n_data, _tf.bool)

        self._pre_computations = {
            "y": y,
            "y_warped": _tf.Variable(_tf.expand_dims(y, 1)),
            "y_derivative": _tf.Variable(y),
            "coords": _tf.constant(self.data.coords[not_nan], _tf.float64),
            "alpha": _tf.Variable(_tf.zeros([n_data, 1], _tf.float64)),
            "chol": _tf.Variable(_tf.zeros([n_data, n_data], _tf.float64)),
            "log_lik": _tf.Variable(_tf.constant(0.0, _tf.float64)),
            "points_to_honor": points_to_honor,
        }

    def _refresh(self, only_training_variables=False):
        y = self._pre_computations["y"]
        self.warping.refresh(y)
        y_warped = _tf.expand_dims(self.warping.forward(y), axis=1)
        self._pre_computations["y_warped"].assign(y_warped)
        y_derivative = self.warping.derivative(y)
        self._pre_computations["y_derivative"].assign(y_derivative)

        self._update_pre_computations(self.options.jitter)

    @_tf.function
    def _update_pre_computations(self, jitter):
        y_warped = self._pre_computations["y_warped"]
        n_data = _tf.shape(y_warped)[0]
        y_derivative = self._pre_computations["y_derivative"]
        points_to_honor = self._pre_computations["points_to_honor"]

        coords = self._pre_computations["coords"]

        cov_mat = self.kernel.self_covariance_matrix(coords, points_to_honor)
        cov_mat = cov_mat + _tf.eye(n_data, dtype=_tf.float64)*jitter

        chol = _tf.linalg.cholesky(cov_mat)
        alpha = _tf.linalg.cholesky_solve(chol, y_warped)

        fit = - 0.5 * _tf.reduce_sum(alpha * y_warped, name="fit")
        det = - _tf.reduce_sum(_tf.math.log(_tf.linalg.tensor_diag_part(
            chol)), name="det")
        const = - 0.5 * _tf.cast(n_data, _tf.float64) \
                * _tf.constant(_np.log(2 * _np.pi), _tf.float64)
        wp = _tf.reduce_sum(_tf.math.log(y_derivative),
                            name="warping_derivative")
        log_lik = fit + det + const + wp

        self._pre_computations["alpha"].assign(alpha)
        self._pre_computations["chol"].assign(chol)
        self._pre_computations["log_lik"].assign(log_lik)

    @_tf.function
    def _quantiles(self, perc, mu, var):
        mu = _tf.squeeze(mu)
        var = _tf.squeeze(var)
        distribution = tfd.Normal(mu, _tf.sqrt(var))

        def quant_fn(p):
            q = self.warping.backward(distribution.quantile(p))
            return q

        quantiles = _tf.map_fn(quant_fn, perc)
        quantiles = _tf.transpose(quantiles)
        return quantiles

    @_tf.function
    def _percentiles(self, quant, mu, var):
        mu = _tf.squeeze(mu)
        var = _tf.squeeze(var)
        distribution = tfd.Normal(mu, _tf.sqrt(var))

        def perc_fn(q):
            q = _tf.expand_dims(q, 0)
            p = distribution.cdf(self.warping.forward(q))
            return p

        percentiles = _tf.map_fn(perc_fn, quant)
        percentiles = _tf.transpose(percentiles)
        return percentiles

    def train(self, **kwargs):

        k_value, k_shape, k_position, \
            k_min_val, k_max_val = self.kernel.get_parameter_values()
        k_len = len(k_value)

        w_value, w_shape, w_position, \
            w_min_val, w_max_val = self.warping.get_parameter_values()
        w_len = len(w_value)

        min_val = _np.concatenate([k_min_val, w_min_val], axis=0)
        max_val = _np.concatenate([k_max_val, w_max_val], axis=0)
        value = _np.concatenate([k_value, w_value], axis=0)

        start = (value - min_val) / (max_val - min_val + 1e-6)

        def fitness(sol, finished=False):
            sol = sol * (max_val - min_val) + min_val
            k_params, w_params = _np.split(sol, _np.cumsum([k_len, w_len]))[:-1]

            self.kernel.update_parameters(k_params, k_shape, k_position)

            self.warping.update_parameters(w_params, w_shape, w_position)

            self._refresh(only_training_variables=~finished)

            return self.log_lik().numpy()

        self.optimizer = _pso.ParticleSwarmOptimizer(len(start))
        self.optimizer.optimize(fitness, start=start, **kwargs)
        fitness(self.optimizer.best_position, finished=True)

    def predict(self, newdata, name=None, perc=(0.025, 0.25, 0.5, 0.75, 0.975),
                quant=()):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            matrix reference to a spatial points object of compatible dimension.
            The prediction is written on the object's data attribute in-place.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        perc : tuple, list, or array
            The desired percentiles of the predictive distribution.
        quant : tuple, list, or array
            The values to calculate the predictive probability.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # prediction in batches
        n_data = x_new.shape[0]
        batch_id = self.options.batch_id(n_data)
        n_batches = len(batch_id)

        mu = []
        var = []
        quantiles = []
        percentiles = []
        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            # TensorFlow
            mu_i, var_i = self._predict(x_new[batch])

            # update
            mu.append(mu_i)
            var.append(var_i)
            if len(perc) > 0:
                quantiles.append(
                    self._quantiles(_tf.constant(perc, _tf.float64),
                                    mu_i, var_i)
                )
            if len(quant) > 0:
                percentiles.append(
                    self._percentiles(_tf.constant(quant, _tf.float64),
                                      mu_i, var_i)
                )
        if self.options.verbose:
            print("\n")
        mu = _tf.squeeze(_tf.concat(mu, axis=0))
        var = _tf.squeeze(_tf.concat(var, axis=0))

        # output
        newdata.data[name + "_mean"] = mu.numpy()
        newdata.data[name + "_variance"] = var.numpy()
        if len(perc) > 0:
            quantiles = _tf.concat(quantiles, axis=0)
            quantiles = quantiles.numpy()
            for col, p in enumerate(perc):
                newdata.data[name + "_p" + str(p)] = quantiles[:, col]
        if len(quant) > 0:
            percentiles = _tf.concat(percentiles, axis=0)
            percentiles = percentiles.numpy()
            for col, q in enumerate(quant):
                newdata.data[name + "_q" + str(q)] = percentiles[:, col]

    def save_state(self, file):
        kernel_parameters = self.kernel.get_parameter_values(complete=True)
        warping_parameters = self.warping.get_parameter_values(complete=True)
        with open(file, 'wb') as f:
            _pickle.dump((kernel_parameters, warping_parameters), f)

    def load_state(self, file):
        with open(file, 'rb') as f:
            kernel_parameters, warping_parameters = _pickle.load(f)

        k_value, k_shape, k_position, k_min_val, k_max_val = kernel_parameters
        self.kernel.update_parameters(k_value, k_shape, k_position)

        w_value, w_shape, w_position, w_min_val, w_max_val = warping_parameters
        self.warping.update_parameters(w_value, w_shape, w_position)


class GPGrad(GP):
    """
    Gaussian Process with support to directional data.

    This version does not support warping.
    """

    def __init__(self, spatial_data, variable, direction_data, kernel,
                 direction_variable=None, points_to_honor=None,
                 options=GPOptions()):
        """
        Initializer for GPGrad.

        Parameters
        ----------
        spatial_data :
            A Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        direction_data :
            A Directions1D, Directions2D, or Directions3D object.
        kernel : tuple or list
            A container with the desired kernels to model the data.
        direction_variable : str
            The name of the column with the derivatives of the dependent
            variable. If omitted, the derivatives are assumed to be 0.
        """
        self.x_dir = direction_data.coords
        self.direction_data = direction_data
        self.direction_variable = direction_variable
        super().__init__(spatial_data=spatial_data,
                         variable=variable,
                         kernel=kernel,
                         points_to_honor=points_to_honor,
                         options=options)

    def _initialize_pre_computed_variables(self):
        not_nan = ~_np.isnan(self.data.data[self.y_name].values)
        n_data = _np.sum(not_nan)
        y = _tf.constant(self.data.data[self.y_name].values[not_nan],
                         _tf.float64)

        if self.direction_variable is None:
            n_dir = self.direction_data.coords.shape[0]
            y_dir = _tf.zeros([n_dir], _tf.float64)
            dir_not_nan = _np.array([True]*n_dir)
        else:
            dir_not_nan = ~_np.isnan(
                self.direction_data.data[self.direction_variable].values)
            y_dir = _tf.constant(
                self.direction_data.data[self.direction_variable]
                    .values[not_nan],
                _tf.float64)
            n_dir = _np.sum(dir_not_nan)

        if self.points_to_honor is not None:
            points_to_honor = _tf.constant(
                self.data.data[self.points_to_honor].values[not_nan],
                _tf.bool)
        else:
            points_to_honor = _tf.constant([False]*n_data, _tf.bool)

        self._pre_computations = {
            "y": y,
            "coords": _tf.constant(self.data.coords[not_nan], _tf.float64),
            "alpha": _tf.Variable(_tf.zeros([n_data + n_dir, 1], _tf.float64)),
            "chol": _tf.Variable(_tf.zeros([n_data + n_dir, n_data + n_dir],
                                           _tf.float64)),
            "log_lik": _tf.Variable(_tf.constant(0.0, _tf.float64)),
            "dir_coords": _tf.constant(
                self.direction_data.coords[dir_not_nan],
                _tf.float64),
            "directions": _tf.constant(
                self.direction_data.directions[dir_not_nan],
                _tf.float64),
            "y_dir": y_dir,
            "points_to_honor": points_to_honor,
        }

    @_tf.function
    def _update_pre_computations(self, jitter):
        y = self._pre_computations["y"]
        y_dir = self._pre_computations["y_dir"]
        y_full = _tf.expand_dims(_tf.concat([y, y_dir], axis=0), axis=1)
        n_data = _tf.shape(y)[0]
        n_dir = _tf.shape(y_dir)[0]

        points_to_honor = self._pre_computations["points_to_honor"]

        coords = self._pre_computations["coords"]
        dir_coords = self._pre_computations["dir_coords"]
        directions = self._pre_computations["directions"]

        cov_coords = self.kernel.self_covariance_matrix(coords, points_to_honor)
        cov_cross = self.kernel.covariance_matrix_d1(
            coords, dir_coords, directions
        )
        cov_dir = self.kernel.self_covariance_matrix_d2(
            dir_coords, directions)

        cov_mat = _tf.concat([
            _tf.concat([cov_coords, cov_cross], axis=1),
            _tf.concat([_tf.transpose(cov_cross), cov_dir], axis=1)
        ], axis=0)
        cov_mat = cov_mat + _tf.eye(n_data + n_dir, dtype=_tf.float64)*jitter

        chol = _tf.linalg.cholesky(cov_mat)
        alpha = _tf.linalg.cholesky_solve(chol, y_full)

        fit = - 0.5 * _tf.reduce_sum(alpha * y_full, name="fit")
        det = - _tf.reduce_sum(_tf.math.log(_tf.linalg.tensor_diag_part(
            chol)), name="det")
        const = - 0.5 * _tf.cast(n_data + n_dir, _tf.float64) \
                * _tf.constant(_np.log(2 * _np.pi), _tf.float64)
        log_lik = fit + det + const

        self._pre_computations["alpha"].assign(alpha)
        self._pre_computations["chol"].assign(chol)
        self._pre_computations["log_lik"].assign(log_lik)

    def _predict(self, x_new, n_sim=0):
        with _tf.name_scope("Prediction"):
            coords = self._pre_computations["coords"]
            dir_coords = self._pre_computations["dir_coords"]
            directions = self._pre_computations["directions"]

            k_new_coords = self.kernel.covariance_matrix(x_new, coords)
            k_new_dir = self.kernel.covariance_matrix_d1(
                x_new, dir_coords, directions)
            k_new = _tf.concat([k_new_coords, k_new_dir], axis=1)
            pred_mu = _tf.matmul(k_new, self._pre_computations["alpha"])

            with _tf.name_scope("pred_var"):
                point_var = self.kernel.point_variance(x_new)
                info_gain = _tf.linalg.cholesky_solve(
                    self._pre_computations["chol"], _tf.transpose(k_new))
                info_gain = _tf.multiply(_tf.transpose(k_new), info_gain)
                pred_var = point_var - _tf.reduce_sum(info_gain, axis=0)
        return pred_mu, pred_var


class GPBinaryClassifier(GP):
    def __init__(self, spatial_data, var_1, var_2, positive_class, kernel,
                 options=GPOptions()):
        n_data = spatial_data.coords.shape[0]

        labels = _pd.api.types.union_categoricals(
            [_pd.Categorical(spatial_data.data[var_1]),
             _pd.Categorical(spatial_data.data[var_2])])
        labels = labels.categories.values

        # if len(labels) != 2:
        #     raise ValueError("the number of classes must be exactly 2")

        if positive_class not in labels:
            raise ValueError('label "' + positive_class + '" not present ' +
                             'in the class labels')

        latent_1 = _np.where(spatial_data.data[var_1].values == positive_class,
                             _np.ones(n_data), -_np.ones(n_data))
        latent_2 = _np.where(spatial_data.data[var_2].values == positive_class,
                             _np.ones(n_data), -_np.ones(n_data))
        latent_avg = (latent_1 + latent_2)/2

        tmpcoords = _pd.DataFrame(spatial_data.coords,
                                  columns=spatial_data.coords_label)
        tmpvals = _pd.DataFrame({positive_class: latent_avg,
                                 "is_boundary": latent_avg == 0})
        if spatial_data.ndim == 1:
            tmpdata = _data.Points1D(tmpcoords, tmpvals)
        elif spatial_data.ndim == 2:
            tmpdata = _data.Points2D(tmpcoords, tmpvals)
        elif spatial_data.ndim == 3:
            tmpdata = _data.Points3D(tmpcoords, tmpvals)
        else:
            raise ValueError("this model does not support the specified "
                             "data object")

        super().__init__(tmpdata, positive_class, kernel,
                         options=options,
                         points_to_honor="is_boundary")

    def predict(self, newdata, name=None):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            matrix reference to a spatial points object of compatible dimension.
            The prediction is written on the object's data attribute in-place.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # prediction in batches
        n_data = x_new.shape[0]
        batch_id = self.options.batch_id(n_data)
        n_batches = len(batch_id)

        mu = []
        var = []
        prob = []
        entropy = []
        uncertainty = []
        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            # TensorFlow
            mu_i, var_i = self._predict(x_new[batch])
            mu_i = _tf.squeeze(mu_i)
            var_i = _tf.squeeze(var_i)

            # update
            mu.append(mu_i)
            var.append(var_i)

            cdf = 0.5*(1 + _tf.math.erf(- mu_i/_tf.sqrt(2*var_i)))
            prob.append(1 - cdf)

            ent_i = -(cdf*_tf.math.log(cdf+1e-6)
                      + (1-cdf)*_tf.math.log(1-cdf+1e-6))
            log2 = _tf.math.log(_tf.constant(2.0, _tf.float64))
            ent_i = ent_i/log2
            ent_i = _tf.maximum(ent_i, 0.0)
            entropy.append(ent_i)

            uncertainty.append(_tf.sqrt(var_i*ent_i))

        if self.options.verbose:
            print("\n")
        mu = _tf.concat(mu, axis=0)
        var = _tf.concat(var, axis=0)
        prob = _tf.concat(prob, axis=0)
        entropy = _tf.concat(entropy, axis=0)
        uncertainty = _tf.concat(uncertainty, axis=0)

        # output
        newdata.data[name + "_mean"] = mu.numpy()
        newdata.data[name + "_variance"] = var.numpy()
        newdata.data[name + "_probability"] = prob.numpy()
        newdata.data[name + "_entropy"] = entropy.numpy()
        newdata.data[name + "_uncertainty"] = uncertainty.numpy()


class GPGradBinaryClassifier(GPGrad):
    def __init__(self, spatial_data, var_1, var_2, positive_class, kernel,
                 direction_data, options=GPOptions()):
        n_data = spatial_data.coords.shape[0]

        labels = _pd.api.types.union_categoricals(
            [_pd.Categorical(spatial_data.data[var_1]),
             _pd.Categorical(spatial_data.data[var_2])])
        labels = labels.categories.values

        # if len(labels) != 2:
        #     raise ValueError("the number of classes must be exactly 2")

        if positive_class not in labels:
            raise ValueError('label "' + positive_class + '" not present ' +
                             'in the class labels')

        latent_1 = _np.where(spatial_data.data[var_1].values == positive_class,
                             _np.ones(n_data), -_np.ones(n_data))
        latent_2 = _np.where(spatial_data.data[var_2].values == positive_class,
                             _np.ones(n_data), -_np.ones(n_data))
        latent_avg = (latent_1 + latent_2)/2

        tmpcoords = _pd.DataFrame(spatial_data.coords,
                                  columns=spatial_data.coords_label)
        tmpvals = _pd.DataFrame({positive_class: latent_avg,
                                 "is_boundary": latent_avg == 0})
        if spatial_data.ndim == 1:
            tmpdata = _data.Points1D(tmpcoords, tmpvals)
        elif spatial_data.ndim == 2:
            tmpdata = _data.Points2D(tmpcoords, tmpvals)
        elif spatial_data.ndim == 3:
            tmpdata = _data.Points3D(tmpcoords, tmpvals)
        else:
            raise ValueError("this model does not support the specified "
                             "data object")

        super().__init__(tmpdata, positive_class, direction_data, kernel,
                         options=options,
                         points_to_honor="is_boundary")

    def predict(self, newdata, name=None):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            matrix reference to a spatial points object of compatible dimension.
            The prediction is written on the object's data attribute in-place.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # prediction in batches
        n_data = x_new.shape[0]
        batch_id = self.options.batch_id(n_data)
        n_batches = len(batch_id)

        mu = []
        var = []
        prob = []
        entropy = []
        uncertainty = []
        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            # TensorFlow
            mu_i, var_i = self._predict(x_new[batch])
            mu_i = _tf.squeeze(mu_i)
            var_i = _tf.squeeze(var_i)

            # update
            mu.append(mu_i)
            var.append(var_i)

            cdf = 0.5*(1 + _tf.math.erf(- mu_i/_tf.sqrt(2*var_i)))
            prob.append(1 - cdf)

            ent_i = -(cdf*_tf.math.log(cdf+1e-6)
                      + (1-cdf)*_tf.math.log(1-cdf+1e-6))
            log2 = _tf.math.log(_tf.constant(2.0, _tf.float64))
            ent_i = ent_i/log2
            ent_i = _tf.maximum(ent_i, 0.0)
            entropy.append(ent_i)

            uncertainty.append(_tf.sqrt(var_i*ent_i))

        if self.options.verbose:
            print("\n")
        mu = _tf.concat(mu, axis=0)
        var = _tf.concat(var, axis=0)
        prob = _tf.concat(prob, axis=0)
        entropy = _tf.concat(entropy, axis=0)
        uncertainty = _tf.concat(uncertainty, axis=0)

        # output
        newdata.data[name + "_mean"] = mu.numpy()
        newdata.data[name + "_variance"] = var.numpy()
        newdata.data[name + "_probability"] = prob.numpy()
        newdata.data[name + "_entropy"] = entropy.numpy()
        newdata.data[name + "_uncertainty"] = uncertainty.numpy()


class GPMultiClassifier(_Model):
    """
    GP for classification. This model uses topological relations between the
    class labels to find a low-dimensional representation of the data. A
    separate GP model is trained on each latent variable.
    
    Attributes
    ----------
    x : ndarray
        The independent variables.
    latent : data frame
        The values of the latent variables used in the model.
    GPs : list
        The individual models that interpolate each latent variable.
    """
    def __repr__(self):
        s = ""
        for i, gp in enumerate(self.GPs):
            s += "Class: " + str(self.latent.columns[i]) + "\n"
            s += gp.__repr__() + "\n\n"
        return s

    def __init__(self, spatial_data, var_1, var_2, kernel,
                 directional_data=None, sparse=False, **kwargs):
        """
        Initializer for GPMultiClassifier.

        Parameters
        ----------
        spatial_data :
            matrix Points1D, Points2D, or Points3D object.
        var_1, var_2 : str
            The columns with the class labels to model.
        kernels : tuple or list
            matrix container with the desired kernel to model the data.
        directional_data :
            matrix Directions1D, Directions2D, or Directions3D object.
            It is assumed that these are directions with zero variation.
        sparse : bool
            Whether to use a sparse large scale model.
        kwargs :
            Passed on to the appropriate model.
        """
        super().__init__()
        self._ndim = spatial_data.ndim
        labels = _pd.api.types.union_categoricals(
            [_pd.Categorical(spatial_data.data[var_1]),
             _pd.Categorical(spatial_data.data[var_2])])
        labels = labels.categories.values

        # latent variables
        n_latent = len(labels)
        n_data = spatial_data.coords.shape[0]
        labels_dict = dict(zip(labels, _np.arange(n_latent)))
        lat1 = _np.ones([n_data, n_latent]) * -1
        lat2 = _np.ones([n_data, n_latent]) * -1
        idx1 = spatial_data.data[var_1].map(labels_dict).values
        idx2 = spatial_data.data[var_2].map(labels_dict).values
        for i in range(n_data):
            lat1[i, idx1[i]] = 1
            lat2[i, idx2[i]] = 1
        lat = 0.5 * lat1 + 0.5 * lat2
        interpolate = _np.apply_along_axis(
            lambda x: x == 0,
            axis=0, arr=lat)
        self.latent = _pd.DataFrame(lat, columns=labels)

        # latent variable models
        self.GPs = []
        n_latent = len(labels)
        temp_data = None
        if self.ndim == 1:
            temp_data = _data.Points1D(spatial_data.coords, _pd.DataFrame(
                self.latent.values, columns=labels))
        elif self.ndim == 2:
            temp_data = _data.Points2D(spatial_data.coords, _pd.DataFrame(
                self.latent.values, columns=labels))
        elif self.ndim == 3:
            temp_data = _data.Points3D(spatial_data.coords, _pd.DataFrame(
                self.latent.values, columns=labels))
        for i in range(n_latent):
            temp_data.data["boundary"] = interpolate[:, i]

            # full GP
            if not sparse:
                if directional_data is None:
                    gp = GP(spatial_data=temp_data,
                            variable=temp_data.data.columns[i],
                            kernel=_copy.deepcopy(kernel),
                            points_to_honor="boundary",
                            **kwargs)
                else:
                    gp = GPGrad(spatial_data=temp_data,
                                variable=temp_data.data.columns[i],
                                kernel=_copy.deepcopy(kernel),
                                points_to_honor="boundary",
                                direction_data=directional_data,
                                **kwargs)
            else:
                raise NotImplementedError()
            self.GPs.append(gp)

    def log_lik(self):
        """
        Outputs the model's log-likelihood, given the current parameters.

        Returns
        -------
        log_lik : double
            The model's log-likelihood.
        """
        return sum([gp.log_lik() for gp in self.GPs])

    def train(self, **kwargs):
        verb = True
        if "verbose" in kwargs:
            verb = kwargs["verbose"]
        for i, gp in enumerate(self.GPs):
            if verb:
                print("\nTraining latent variable " + str(i) + "\n")
            gp.train(**kwargs)

    def predict(self, newdata, name):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            matrix reference to a spatial points object of compatible dimension.
            The prediction is written on the object's data attribute in-place.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        temp_data = None
        if isinstance(newdata, _data.Points1D):
            temp_data = _data.Points1D(newdata.coords)
        elif isinstance(newdata, _data.Points2D):
            temp_data = _data.Points2D(newdata.coords)
        elif isinstance(newdata, _data.Points3D):
            temp_data = _data.Points3D(newdata.coords)
        n_latent = self.latent.shape[1]
        # n_classes = n_latent
        n_out = newdata.coords.shape[0]
        mu = _np.zeros([n_out, n_latent])
        var = _np.zeros([n_out, n_latent])

        # prediction of latent variables
        for i, gp in enumerate(self.GPs):
            if gp.options.verbose:
                print("Predicting latent variable " +
                      str(i + 1) + " of " + str(n_latent) + ":")
            gp.predict(temp_data, name="out")
            mu[:, i] = temp_data.data["out_mean"].values
            var[:, i] = temp_data.data["out_variance"].values

        # probability
        prob = []
        batches = self.GPs[0].options.batch_id(n_out, 5000)
        for i, batch in enumerate(batches):
            if self.GPs[0].options.verbose:
                print("\rComputing probabilities: batch " + str(i + 1)
                      + " of " + str(len(batches)), end="")

            prob_i = _tftools.highest_value_probability(
                _tf.constant(mu[batch], _tf.float32),
                _tf.constant(var[batch], _tf.float32),
                seed=_tf.constant(self.GPs[0].options.seed)
            )
            prob.append(prob_i.numpy())
        prob = _np.concatenate(prob, axis=0)

        # indicators
        ind = _np.log(prob)
        ind_skew = _np.zeros_like(ind)
        for i in range(n_latent):
            cols = list(range(n_latent))
            cols.pop(i)
            cols = _np.array(cols)
            base_ind = ind[:, i] - _np.max(ind[:, cols], axis=1)
            ind_skew[:, i] = base_ind

        # entropy
        entropy = - _np.sum(prob * _np.log(prob), axis=1) / _np.log(n_latent)
        mean_var = _np.exp(_np.mean(_np.log(var), axis=1))
        uncertainty = _np.sqrt(entropy * mean_var)

        # confidence
        # conf = 1 - _np.exp(_np.mean(_np.log(var), axis=1))

        # output
        idx = self.latent.columns
        labels = _np.array(idx[_np.argmax(ind, axis=1)])
        newdata.data[name] = labels
        for i in range(n_latent):
            newdata.data[name + "_" + str(idx[i]) + "_ind"] = ind_skew[:, i]
        for i in range(n_latent):
            newdata.data[name + "_" + str(idx[i]) + "_prob"] = prob[:, i]
        newdata.data[name + "_uncertainty"] = uncertainty
        newdata.data[name + "_entropy"] = entropy

    def __str__(self):
        s = ""
        for model in self.GPs:
            s += "\nModel for " + model.y_name + ":\n"
            s += str(model)
        return s

    def save_state(self, file):
        d = [gp.kernel.get_parameter_values(complete=True) for gp in self.GPs]
        with open(file, 'wb') as f:
            _pickle.dump(d, f)

    def load_state(self, file):
        with open(file, 'rb') as f:
            d = _pickle.load(f)
        for i, gp in enumerate(self.GPs):
            gp.kernel.update_parameters(d[i][0], d[i][1], d[i][2])


class SparseGP(WarpedGP):
    def __init__(self, spatial_data, variable, kernel,
                 pseudo_inputs, interpolating_kernel,
                 warping=None, points_to_honor=None, options=GPOptions()):
        """
        Initializer for SparseGP

        Parameters
        ----------
        spatial_data
        variable
        kernel
        pseudo_inputs : list containing spatial objects
        interpolating_kernel : a compact support kernel
        warping
        points_to_honor
        options
        """
        self.pseudo_inputs = pseudo_inputs
        self._n_pseudo = [ps.coords.shape[0] for ps in pseudo_inputs]
        self.interpolating_kernel = interpolating_kernel
        super().__init__(spatial_data=spatial_data,
                         variable=variable,
                         kernel=kernel,
                         warping=warping,
                         points_to_honor=points_to_honor,
                         options=options)

    # @_tf.function
    def _lanczos_0(self, coords, n_lanczos, jitter):
        with _tf.name_scope("lanczos_0"):
            # with _tf.device("GPU:0"):
            cov_mat = self.interpolating_kernel.sparse_covariance_matrix(
                coords, coords
            )

            mat_t, mat_q = _tftools.lanczos(
                lambda b: _tf.sparse.sparse_dense_matmul(cov_mat, b),
                _tf.ones([_tf.shape(coords)[0], 1], dtype=_tf.float64),
                m=n_lanczos
            )

            eigvals, eigvecs = _tf.linalg.eigh(mat_t)
            eigvals = _tf.maximum(eigvals, jitter)
            phi = _tf.linalg.diag(_tf.sqrt(1 / eigvals))
            phi = _tf.matmul(eigvecs, phi)
            phi = _tf.matmul(mat_q, phi)
        return phi

    @_tf.function
    def _lanczos_1(self, y, nugget, n_lanczos, jitter, seed):

        def matmul_fn(vec):
            out = vec * nugget

            for _sparse_cov, _phi_0, _cov in zip(
                    self._pre_computations["training_sparse_cov"],
                    self._pre_computations["phi_0"],
                    self._pre_computations["trainable_cov_mats"]):
                out_i = _tf.sparse.sparse_dense_matmul(_sparse_cov, vec, True)
                out_i = _tf.matmul(_phi_0, out_i, True, False)
                out_i = _tf.matmul(_phi_0, out_i)
                out_i = _tf.matmul(_cov, out_i)
                out_i = _tf.matmul(_phi_0, out_i, True, False)
                out_i = _tf.matmul(_phi_0, out_i)
                out_i = _tf.sparse.sparse_dense_matmul(_sparse_cov, out_i)
                out = out + out_i

            return out

        with _tf.device("GPU:0"):
            # alpha
            mat_t, mat_q = _tftools.lanczos(
                matmul_fn,
                q_0=y,
                m=n_lanczos)

            eigvals, eigvecs = _tf.linalg.eigh(mat_t)
            eigvals = _tf.maximum(eigvals, jitter)
            phi_1 = _tf.linalg.diag(_tf.sqrt(1 / eigvals))
            phi_1 = _tf.matmul(eigvecs, phi_1)
            phi_1 = _tf.matmul(mat_q, phi_1)

            alpha = _tf.matmul(phi_1, y, True, False)
            alpha = _tf.matmul(phi_1, alpha)

            alpha_pred = []
            for sparse_cov, phi_0, cov in zip(
                    self._pre_computations["training_sparse_cov"],
                    self._pre_computations["phi_0"],
                    self._pre_computations["trainable_cov_mats"]):
                ap_i = _tf.sparse.sparse_dense_matmul(sparse_cov, alpha, True)
                ap_i = _tf.matmul(phi_0, ap_i, True, False)
                ap_i = _tf.matmul(phi_0, ap_i)
                ap_i = _tf.matmul(cov, ap_i)
                ap_i = _tf.matmul(phi_0, ap_i, True, False)
                ap_i = _tf.matmul(phi_0, ap_i)
                alpha_pred.append(ap_i)
            alpha_pred = _tf.concat(alpha_pred, axis=0)

            # determinant
            lanczos_det = _tftools.determinant_lanczos(
                matmul_fn, _tf.shape(y)[0], n=5, m=n_lanczos,
                seed=seed)

        return phi_1, alpha, alpha_pred, lanczos_det

    @_tf.function
    def _lanczos_2(self, n_lanczos):
        phi_1 = self._pre_computations["phi_1"]

        def matmul_fn(vec):
            vec = _tf.split(vec, self._n_pseudo, axis=0)
            out = []

            for sparse_cov, phi_0, cov, vec_i in zip(
                    self._pre_computations["training_sparse_cov"],
                    self._pre_computations["phi_0"],
                    self._pre_computations["trainable_cov_mats"],
                    vec):

                out_i_1 = _tf.matmul(phi_0, vec_i, True, False)
                out_i_1 = _tf.matmul(phi_0, out_i_1)
                out_i_1 = _tf.matmul(cov, out_i_1)
                out_i_1 = _tf.matmul(phi_0, out_i_1, True, False)
                out_i_1 = _tf.matmul(phi_0, out_i_1)

                out_i_2 = _tf.sparse.sparse_dense_matmul(sparse_cov, out_i_1)
                out_i_2 = _tf.matmul(phi_1, out_i_2, True, False)
                out_i_2 = _tf.matmul(phi_1, out_i_2)
                out_i_2 = _tf.sparse.sparse_dense_matmul(
                    sparse_cov, out_i_2, True)
                out_i_2 = _tf.matmul(phi_0, out_i_2, True, False)
                out_i_2 = _tf.matmul(phi_0, out_i_2)
                out_i_2 = _tf.matmul(cov, out_i_2)
                out_i_2 = _tf.matmul(phi_0, out_i_2, True, False)
                out_i_2 = _tf.matmul(phi_0, out_i_2)

                out.append(out_i_1 - out_i_2)

            return _tf.concat(out, axis=0)

        with _tf.device("GPU:0"):
            mat_t, mat_q = _tftools.lanczos(
                matmul_fn,
                q_0=_tf.ones([sum(self._n_pseudo), 1], _tf.float64),
                m=n_lanczos)

            eigvals, eigvecs = _tf.linalg.eigh(mat_t)
            phi_2 = _tf.linalg.diag(_tf.sqrt(eigvals))
            phi_2 = _tf.matmul(eigvecs, phi_2)
            phi_2 = _tf.matmul(mat_q, phi_2)

        self._pre_computations["phi_2"].assign(phi_2)

    def _batched_sparse_covariance_matrix(self, batched_coords, coords):
        batch_id = self.options.batch_id(batched_coords.shape[0])
        with _tf.device("GPU:0"):
            coords = _tf.constant(coords, _tf.float64)
            cov_mats = [self.interpolating_kernel.sparse_covariance_matrix(
                _tf.constant(batched_coords[idx], _tf.float64), coords
            ) for idx in batch_id]
            full_mat = _tf.sparse.concat(axis=0, sp_inputs=cov_mats)
            return full_mat

    def _weight_by_group(self, cov_mats):
        with _tf.device("GPU:0"):
            avg_cov = _tf.concat(
                [_tf.sparse.reduce_sum(cov_mat, axis=1, keepdims=True)/n
                 for cov_mat, n in zip(cov_mats, self._n_pseudo)],
                axis=1
            )
            # avg_cov = _tf.concat(
            #     [_tf.sparse.sparse_dense_matmul(
            #         cov_mat, _tf.ones([n, 1], _tf.float64)/n)
            #      for cov_mat, n in zip(cov_mats, self._n_pseudo)],
            #     axis=1
            # )
            logits = avg_cov - _tf.reduce_mean(avg_cov, axis=1, keepdims=True)
            weights = _tf.sqrt(_tf.nn.softmax(logits, axis=1))
            # weights = _tf.nn.softmax(logits, axis=1)

            for i, cov_mat in enumerate(cov_mats):
                idx = cov_mat.indices[:, 0]
                w = _tf.gather(weights[:, i], idx)
                cov_mats[i] = _tf.sparse.SparseTensor(
                    indices=cov_mat.indices,
                    values=cov_mat.values * w,
                    dense_shape=cov_mat.dense_shape
                )

        return cov_mats

    def _initialize_pre_computed_variables(self):
        with _tf.device("GPU:0"):
            not_nan = ~_np.isnan(self.data.data[self.y_name].values)
            n_data = self.data.coords.shape[0]
            y = _tf.constant(self.data.data[self.y_name].values[not_nan],
                             _tf.float64)
            self.warping.refresh(y)

            if self.points_to_honor is not None:
                points_to_honor = _tf.constant(
                    self.data.data[self.points_to_honor].values[not_nan],
                    _tf.bool)
            else:
                points_to_honor = _tf.constant([False]*n_data, _tf.bool)

            n_lanczos = self.options.lanczos_iterations

            training_coords = _tf.constant(self.data.coords[not_nan],
                                           _tf.float64)

            # pseudo_inputs and interpolators
            phi_0 = [self._lanczos_0(_tf.constant(ps.coords, _tf.float64),
                                     self.options.lanczos_iterations,
                                     self.options.jitter)
                     for ps in self.pseudo_inputs]
            training_sparse_cov = [self._batched_sparse_covariance_matrix(
                self.data.coords[not_nan], ps.coords
            ) for ps in self.pseudo_inputs]
            training_sparse_cov = self._weight_by_group(training_sparse_cov)
            pseudo_input_coords = [_tf.constant(ps.coords, _tf.float64)
                                   for ps in self.pseudo_inputs]

            # trainable covariance matrices
            total_ps = sum(self._n_pseudo)
            trainable_cov_mats = [_tf.Variable(_tf.zeros([n, n], _tf.float64))
                                  for n in self._n_pseudo]

            self._pre_computations = {
                "y": y,
                "coords": training_coords,
                "pseudo_input_coords": pseudo_input_coords,
                "alpha": _tf.Variable(_tf.zeros([total_ps, 1], _tf.float64)),
                "log_lik": _tf.Variable(_tf.constant(0.0, _tf.float64)),
                "points_to_honor": points_to_honor,
                "phi_0": phi_0,
                "training_sparse_cov": training_sparse_cov,
                "trainable_cov_mats": trainable_cov_mats,
                "phi_1": _tf.Variable(
                    _tf.zeros([n_data, n_lanczos], _tf.float64),
                    validate_shape=False,
                    shape=_tf.TensorShape([n_data, None])),
                "phi_2": _tf.Variable(
                    _tf.zeros([total_ps, n_lanczos], _tf.float64),
                    validate_shape=False,
                    shape=_tf.TensorShape([total_ps, None])),
            }

    def _refresh(self, only_training_variables=False):
        y = self._pre_computations["y"]
        self.warping.refresh(y)
        y_warped = _tf.expand_dims(self.warping.forward(y), axis=1)
        n_data = _tf.shape(y_warped)[0]
        y_derivative = self.warping.derivative(y)
        points_to_honor = self._pre_computations["points_to_honor"]
        coords = self._pre_computations["coords"]

        # covariance matrices
        for i, ps_coords \
                in enumerate(self._pre_computations["pseudo_input_coords"]):
            self._pre_computations["trainable_cov_mats"][i].assign(
                self.kernel.covariance_matrix(ps_coords, ps_coords)
            )

        # 1st lanczos: alpha
        nugget_matmul = self.kernel.nugget_matmul(coords, points_to_honor)
        nugget = nugget_matmul(_tf.ones_like(y_warped))

        phi_1, alpha, alpha_pred, lanczos_det = self._lanczos_1(
            y_warped, nugget,
            n_lanczos=self.options.lanczos_iterations,
            jitter=self.options.jitter,
            seed=self.options.seed)

        # log-likelihood
        fit = - 0.5 * _tf.reduce_sum(alpha * y_warped, name="fit")
        det = - 0.5 * lanczos_det
        const = - 0.5 * _tf.cast(n_data, _tf.float64) \
                * _tf.constant(_np.log(2 * _np.pi), _tf.float64)
        wp = _tf.reduce_sum(_tf.math.log(y_derivative),
                            name="warping_derivative")
        log_lik = fit + det + const + wp

        # updates
        self._pre_computations["alpha"].assign(alpha_pred)
        self._pre_computations["log_lik"].assign(log_lik)
        self._pre_computations["phi_1"].assign(phi_1)

        # 2nd lanczos: predictive covariances
        if not only_training_variables:
            self._lanczos_2(self.options.lanczos_iterations)

    def _predict(self, x_new, n_sim=0):
        with _tf.name_scope("Prediction"):
            # weight matrix
            test_sparse_cov = [self._batched_sparse_covariance_matrix(
                x_new, _tf.constant(ps.coords, _tf.float64)
            ) for ps in self.pseudo_inputs]
            test_sparse_cov = self._weight_by_group(test_sparse_cov)
            w_test = _tf.sparse.concat(
                axis=1, sp_inputs=test_sparse_cov)

            # prediction
            pred_mu = _tf.sparse.sparse_dense_matmul(
                w_test, self._pre_computations["alpha"]
            )

            nugget_var = self.kernel.nugget_variance(x_new)

            pred_explained_cov = _tf.sparse.sparse_dense_matmul(
                w_test, self._pre_computations["phi_2"]
            )
            pred_var = _tf.reduce_sum(pred_explained_cov**2, axis=1) \
                       + nugget_var

            # simulation
            rnd = _tf.random.stateless_normal(
                shape=[_tf.shape(pred_explained_cov)[1], n_sim],
                seed=[self.options.seed, 0],
                dtype=_tf.float64
            )
            pred_sim = _tf.matmul(pred_explained_cov, rnd) + pred_mu

            if self.options.add_noise:
                noise = _tf.random.normal(
                    shape=_tf.shape(pred_sim),
                    seed=self.options.seed,
                    dtype=_tf.float64,
                    stddev=_tf.sqrt(_tf.expand_dims(nugget_var, axis=1))
                )
                pred_sim = pred_sim + noise

            # warping
            pred_sim = _tf.reshape(
                self.warping.backward(_tf.reshape(pred_sim, [-1])),
                _tf.shape(pred_sim)
            )

        return pred_mu, pred_var, pred_sim

    def predict(self, newdata, name=None, perc=(0.025, 0.25, 0.5, 0.75, 0.975),
                quant=(), n_sim=0):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            matrix reference to a spatial points object of compatible dimension.
            The prediction is written on the object's data attribute in-place.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        perc : tuple, list, or array
            The desired percentiles of the predictive distribution.
        quant : tuple, list, or array
            The values to calculate the predictive probability.
        n_sim : int
            The number of simulations to compute.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # prediction in batches
        n_data = x_new.shape[0]
        batch_id = self.options.batch_id(n_data)
        n_batches = len(batch_id)

        mu = []
        var = []
        quantiles = []
        percentiles = []
        sim = []
        for i, batch in enumerate(batch_id):
            if self.options.verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            # TensorFlow
            mu_i, var_i, sim_i = self._predict(x_new[batch], n_sim)

            # update
            mu.append(mu_i)
            var.append(var_i)
            sim.append(sim_i)
            if len(perc) > 0:
                quantiles.append(
                    self._quantiles(_tf.constant(perc, _tf.float64),
                                    mu_i, var_i)
                )
            if len(quant) > 0:
                percentiles.append(
                    self._percentiles(_tf.constant(quant, _tf.float64),
                                      mu_i, var_i)
                )
        if self.options.verbose:
            print("\n")
        mu = _tf.squeeze(_tf.concat(mu, axis=0))
        var = _tf.squeeze(_tf.concat(var, axis=0))
        sim = _tf.concat(sim, axis=0).numpy()

        # output
        newdata.data[name + "_mean"] = mu.numpy()
        newdata.data[name + "_variance"] = var.numpy()
        if len(perc) > 0:
            quantiles = _tf.concat(quantiles, axis=0)
            quantiles = quantiles.numpy()
            for col, p in enumerate(perc):
                newdata.data[name + "_p" + str(p)] = quantiles[:, col]
        if len(quant) > 0:
            percentiles = _tf.concat(percentiles, axis=0)
            percentiles = percentiles.numpy()
            for col, q in enumerate(quant):
                newdata.data[name + "_q" + str(q)] = percentiles[:, col]
        if n_sim > 0:
            n_digits = str(len(str(n_sim - 1)))
            for i in range(n_sim):
                col = name + "_sim_" + ("{:0>"+n_digits+"d}").format(i)
                newdata.data[col] = sim[:, i]
