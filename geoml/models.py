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

__all__ = ["GP", "GPGrad", "GPClassif", "SparseGP", "SparseGPEnsemble"]

import numpy as _np
import tensorflow as _tf
import scipy.stats as _st
import scipy.special as _ss
import scipy.sparse as _sp
import pandas as _pd
import copy as _copy
import pickle as _pickle
import itertools as _itertools

import geoml.genetic as _gen
import geoml.kernels as _ker
import geoml.warping as _warp
import geoml.tftools as _tftools
import geoml.data as _data
import geoml.transform as _transf
import geoml.interpolation as _int


class _Model:
    """Base model class"""

    def __init__(self):
        self._ndim = None

    @property
    def ndim(self):
        return self._ndim

    def __repr__(self):
        return self.__str__()

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


class GP(_Model):
    """
    Base Gaussian Process model, not optimized for a large amount of data.
    It is expected to work well with up to ~5000 data points.
    
    Internally the model assumes unit range, unit variance and
    zero mean.
    
    Attributes
    ----------
    x : ndarray
        The independent variables.
    y : ndarray
        The dependent variable.
    y_name : str
        Label for the dependent variable.
    cov_model :
        A covariance model object.
    interpolate : ndarray
        A boolean vector.
    training_log : dict
        The output of a genetic algorithm optimization.
    graph :
        A TensorFlow graph.
    """

    def __init__(self, sp_data, variable, kernels, warping=(),
                 interpolate=None):
        """
        Initializer for GP.

        Parameters
        ----------
        sp_data :
            A Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        kernels : tuple or list
            A container with the desired kernels to model the data.
        warping : tuple or list
            A container with the desired warping objects to model the data.
        interpolate : str
            The name of a column containing a Boolean variable, indicating which
            data points must be honored.
        """
        self.data = sp_data
        self.y = sp_data.data[variable].values
        self.y_name = variable
        self._ndim = sp_data.ndim
        self.cov_model = _ker.CovarianceModelRegression(
            kernels, warping)
        if interpolate is None:
            self.interpolate = _np.repeat(False, len(self.y))
        else:
            self.interpolate = sp_data.data[interpolate].values
        self.training_log = None

        # tidying up
        self.cov_model.auto_set_kernel_parameter_limits(sp_data)
        n_data = sp_data.coords.shape[0]

        # y variable initialization is currently outside TensorFlow
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)

        # TensorFlow
        self.graph = _tf.Graph()
        with self.graph.as_default():
            with _tf.name_scope("GP"):
                # constants
                # x = _tf.constant(patched_data.coords,
                #                  dtype=_tf.float64,
                #                  name="coords")
                interp = _tf.constant(self.interpolate,
                                      dtype=_tf.bool)

                # trainable parameters
                with _tf.name_scope("init_placeholders"):
                    self.cov_model.init_tf_placeholder()

                # placeholders
                y_tf = _tf.placeholder(_tf.float64, shape=(len(y), 1),
                                       name="y_tf")
                yd_tf = _tf.placeholder(_tf.float64, shape=(len(y), 1),
                                        name="yd_tf")
                jitter_tf = _tf.placeholder(_tf.float64, shape=[],
                                            name="jitter")
                x = _tf.placeholder_with_default(
                    _tf.constant(sp_data.coords, dtype=_tf.float64),
                    shape=[None, self.ndim],
                    name="x"
                )
                x_new = _tf.placeholder_with_default(
                    _tf.zeros([1, self.ndim], dtype=_tf.float64),
                    shape=[None, self.ndim],
                    name="x_new")
                n_sim = _tf.placeholder_with_default(
                    _tf.constant(0, _tf.int32), shape=[], name="n_sim"
                )
                sim_with_noise = _tf.placeholder_with_default(
                    False, shape=[], name="sim_with_noise")
                seed = _tf.placeholder_with_default(
                    _tf.constant(1234, _tf.int32), shape=[],
                    name="seed"
                )

                # covariance matrix
                with _tf.name_scope("covariance_matrix"):
                    # training points
                    nugget = self.cov_model.variance.tf_val[-1]
                    k_train = _tf.add(
                        self.cov_model.covariance_matrix(x, x),
                        self.cov_model.nugget.nugget_matrix(x, interp) * nugget)
                    # adding jitter to avoid Cholesky decomposition problems
                    sk = _tf.shape(k_train)
                    k_train = k_train + _tf.diag(
                        _tf.ones(sk[0], dtype=_tf.float64) * jitter_tf)

                    k_train_test = self.cov_model.covariance_matrix(x, x_new)

                # alpha
                with _tf.name_scope("alpha"):
                    chol_train = _tf.linalg.cholesky(k_train)
                    alpha = _tf.linalg.cholesky_solve(chol_train, y_tf)
                    chol_train = _tf.Variable(chol_train, validate_shape=False)
                    alpha = _tf.Variable(alpha, validate_shape=False)

                # log-likelihood
                with _tf.name_scope("log_lik"):
                    log_lik = \
                        - 0.5 * _tf.reduce_sum(alpha * y_tf, name="fit") \
                        - _tf.reduce_sum(_tf.math.log(_tf.diag_part(
                            chol_train)), name="det") \
                        - 0.5 * _tf.constant(n_data * _np.log(2 * _np.pi)) \
                        + _tf.reduce_sum(_tf.math.log(yd_tf),
                                         name="warping_derivative")

                # prediction
                with _tf.name_scope("Prediction"):
                    # k_new = self._covariance_matrix(x, interp, x_new)
                    pred_mu = _tf.squeeze(
                        _tf.matmul(k_train_test, alpha, transpose_a=True),
                        name="pred_mean")
                    with _tf.name_scope("pred_var"):
                        point_var = self.cov_model.point_variance(x_new)
                        # pred_var = _tf.linalg.cholesky_solve(
                        #     chol_train, k_train_test)
                        info_gain = _tf.linalg.solve(chol_train, k_train_test)
                        # pred_var = _tf.multiply(k_train_test, pred_var)
                        pred_var = point_var - _tf.reduce_sum(
                            info_gain * info_gain, axis=0)
                        # adding nugget
                        pred_var = pred_var + self.cov_model.variance.tf_val[-1]

                # simulation
                with _tf.name_scope("simulation"):
                    s_new = _tf.shape(x_new)
                    sim_shape_new = [s_new[0], n_sim]

                    k_test = self.cov_model.covariance_matrix(x_new, x_new)
                    # k_train_test = self._covariance_matrix(x, y=x_new)
                    # info_gain = _tf.linalg.solve(mat_chol, k_train_test)
                    k_cond = k_test - _tf.matmul(info_gain, info_gain,
                                                 True, False)
                    k_cond = k_cond + _tf.eye(
                        s_new[0], dtype=_tf.float64) * jitter_tf
                    chol_cond = _tf.linalg.cholesky(k_cond)
                    sim_x_new = _tf.random.stateless_normal(
                        sim_shape_new, dtype=_tf.float64, seed=[seed, 0])
                    y_sim = _tf.matmul(chol_cond, sim_x_new)
                    y_sim = y_sim + _tf.expand_dims(pred_mu, axis=1)

                    with _tf.name_scope("noise"):
                        def noise(mat):
                            rand = _tf.random.normal(
                                shape=_tf.shape(mat), dtype=_tf.float64,
                                stddev=_tf.sqrt(
                                    self.cov_model.variance.tf_val[-1]),
                                # seed=seed
                            )
                            return mat + rand

                        y_sim = _tf.cond(sim_with_noise,
                                         lambda: noise(y_sim),
                                         lambda: y_sim)

                self.tf_handles = {
                    "L": chol_train,
                    "x": x,
                    "alpha": alpha,
                    "y_tf": y_tf,
                    "yd_tf": yd_tf,
                    "k_train_test": k_train_test,
                    "log_lik": log_lik,
                    "x_new": x_new,
                    "pred_mu": pred_mu,
                    "pred_var": pred_var,
                    "jitter": jitter_tf,
                    "y_sim": y_sim,
                    "n_sim": n_sim,
                    "sim_with_noise": sim_with_noise,
                    "seed": seed,
                    "init": _tf.global_variables_initializer()}

        self.graph.finalize()

    def _covariance_matrix(self, x, interp=None, y=None, jitter=1e-9):
        with _tf.name_scope("covariance_matrix"):
            if y is None:
                nugget = self.cov_model.variance.tf_val[-1]
                k = _tf.add(
                    self.cov_model.covariance_matrix(x, y),
                    self.cov_model.nugget.nugget_matrix(x, interp) * nugget)
                # adding jitter to avoid Cholesky decomposition problems
                sk = _tf.shape(k)
                k = k + _tf.diag(_tf.ones(sk[0], dtype=_tf.float64) * jitter)
            else:
                k = self.cov_model.covariance_matrix(x, y)
        return k

    def _alpha(self, k, y):
        with _tf.name_scope("alpha"):
            chol = _tf.linalg.cholesky(k)
            alpha = _tf.linalg.cholesky_solve(chol, y)
        return chol, alpha

    def _log_lik(self, chol, alpha, y, yd):
        with _tf.name_scope("log_lik"):
            n_data = y.shape[0].value
            log_lik = - 0.5 * _tf.reduce_sum(alpha * y, name="fit") \
                      - _tf.reduce_sum(_tf.math.log(_tf.diag_part(chol)),
                                       name="det") \
                      - 0.5 * _tf.constant(n_data * _np.log(2 * _np.pi)) \
                      + _tf.reduce_sum(_tf.log(yd), name="warping_derivative")
        return log_lik

    def _predict(self, chol, alpha, x, x_new, interp):
        with _tf.name_scope("Prediction"):
            k_new = self._covariance_matrix(x, interp, x_new)
            pred_mu = _tf.squeeze(_tf.matmul(k_new, alpha, transpose_a=True),
                                  name="pred_mean")
            with _tf.name_scope("pred_var"):
                point_var = self.cov_model.point_variance(x_new)
                pred_var = _tf.linalg.cholesky_solve(chol, k_new)
                pred_var = _tf.multiply(k_new, pred_var)
                pred_var = point_var - _tf.reduce_sum(pred_var, axis=0)
                # adding nugget
                pred_var = pred_var + self.cov_model.variance.tf_val[-1]
        return pred_mu, pred_var

    def log_lik(self, session=None):
        """
        Outputs the model's log-likelihood, given the current parameters.

        Parameters
        ----------
        session :
            An active TensorFlow session. If omitted, one will be created.

        Returns
        -------
        log_lik : double
            The model's log-likelihood.
        """
        if session is None:
            with _tf.Session(graph=self.graph) as session:
                log_lik = self.log_lik(session)
            return log_lik

        # updating y
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)
        yd = self.cov_model.warp_derivative(self.y)

        # session
        feed = self.cov_model.feed_dict()
        feed.update({self.tf_handles["y_tf"]: _np.resize(y, (len(y), 1)),
                     self.tf_handles["yd_tf"]: _np.resize(yd, (len(y), 1)),
                     self.tf_handles["jitter"]: self.cov_model.jitter})
        run_opts = _tf.RunOptions(report_tensor_allocations_upon_oom=True)
        session.run(self.tf_handles["init"], feed_dict=feed)
        log_lik = session.run(
            self.tf_handles["log_lik"],
            feed_dict=feed,
            options=run_opts)
        return log_lik

    def cross_validation(self, partition=None, session=None):
        raise NotImplementedError("to be implemented")

    def train(self, **kwargs):

        pd = self.cov_model.params_dict()

        def fitness(z, sess):
            pd2 = pd.copy()
            pd2["param_val"] = z

            self.cov_model.update_params(pd2)
            return self.log_lik(sess)

        with _tf.Session(graph=self.graph) as session:
            opt = _gen.training_real(fitness=lambda z: fitness(z, session),
                                     minval=_np.array(pd["param_min"]),
                                     maxval=_np.array(pd["param_max"]),
                                     start=_np.array(pd["param_val"]),
                                     **kwargs)
            fitness(opt["best_sol"].tolist(), session)

        self.training_log = opt

    def predict(self, newdata, perc=(0.025, 0.25, 0.5, 0.75, 0.975),
                name=None, verbose=True, batch_size=20000,
                n_sim=0, add_noise=False, seed=1234):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension. The
            prediction is written on the object's data attribute in-place.
        perc : tuple, list, or array
            The desired percentiles of the predictive distribution.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        verbose : bool
            Whether to display progress info on console.
        batch_size : int
            Number of data points to process in each batch.
            If you get an out-of-memory
            TensorFlow error, try decreasing this value.
        n_sim : int
            The desired number of simulations.
        add_noise : bool
            Whether to include the model's noise in the simulations.
        seed : int
            A seed number to produce consistent simulations.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # updating y
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)

        # session and placeholders
        session = _tf.Session(graph=self.graph)
        feed = self.cov_model.feed_dict()
        feed.update({self.tf_handles["y_tf"]: _np.resize(y, (len(y), 1)),
                     self.tf_handles["jitter"]: self.cov_model.jitter,
                     self.tf_handles["n_sim"]: n_sim,
                     self.tf_handles["sim_with_noise"]: add_noise,
                     self.tf_handles["seed"]: seed})
        session.run(self.tf_handles["init"], feed_dict=feed)
        handles = [self.tf_handles["pred_mu"],
                   self.tf_handles["pred_var"]]
        if n_sim > 0:
            handles += [self.tf_handles["y_sim"]]

        # prediction in batches
        n_data = x_new.shape[0]
        n_batches = int(_np.ceil(n_data / batch_size))
        batch_id = [_np.arange(i * batch_size,
                               _np.minimum((i + 1) * batch_size, n_data))
                    for i in range(n_batches)]

        mu = _np.array([])
        var = _np.array([])
        y_sim = _np.empty([0, n_sim])
        for i in range(n_batches):
            if verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            feed.update({self.tf_handles["x_new"]: x_new[batch_id[i], :]})

            # TensorFlow
            out = session.run(handles, feed_dict=feed)

            # update
            mu = _np.concatenate([mu, out[0]])
            var = _np.concatenate([var, out[1]])
            if n_sim > 0:
                y_sim = _np.concatenate([y_sim, out[2]], axis=0)
        session.close()
        if verbose:
            print("\n")

        # output
        newdata.data[name + "_mean"] = mu
        newdata.data[name + "_variance"] = var
        for p in perc:
            newdata.data[name + "_p" + str(p)] = self.cov_model.warp_backward(
                _st.norm.ppf(p, loc=mu, scale=_np.sqrt(var)))

        # simulations
        n_digits = str(len(str(n_sim)) - 1)
        cols = [name + "_sim_" + ("{:0>" + n_digits + "d}").format(i)
                for i in range(n_sim)]
        for i in range(n_sim):
            newdata.data[cols[i]] = self.cov_model.warp_backward(
                y_sim[:, i])

    def __str__(self):
        s = "A " + self.__class__.__name__ + " object\n\n"
        s += "Covariance model: " + self.cov_model.__str__()
        return s

    def save_state(self, file):
        self.cov_model.save_state(file)

    def load_state(self, file):
        self.cov_model.load_state(file)

    def update_params(self, params):
        self.cov_model.update_params(params)

    def covariance_matrix(self, x, y):
        """
        Outputs the covariance matrix corresponding to this model at the
        specified coordinates.

        Attributes
        ----------
        x, y : ndarray
            The coordinates on which to calculate the covariance.

        Returns
        -------
        cov_mat : ndarray
            The covariance matrix.
        """
        if len(x.shape) < 2:
            x = _np.expand_dims(x, axis=1)
        if len(y.shape) < 2:
            y = _np.expand_dims(y, axis=1)

        if (self.ndim != x.shape[1]) | (self.ndim != y.shape[1]):
            raise ValueError("dimension of coordinates is incompatible "
                             "with model")

        # session and placeholders
        session = _tf.Session(graph=self.graph)
        feed = self.cov_model.feed_dict()
        feed.update({self.tf_handles["jitter"]: self.cov_model.jitter,
                     self.tf_handles["x"]: x,
                     self.tf_handles["x_new"]: y})
        cov_mat = session.run(self.tf_handles["k_train_test"],
                              feed_dict=feed)
        session.close()

        return cov_mat


class GPGrad(GP):
    """
    Gaussian Process with support to directional data.
    This version does not support warping.

    Attributes
    ----------
    x : ndarray
        The independent variables.
    x_dir : ndarray
        The coordinates of the directional data.
    directions : ndarray
        The directions (unit vectors).
    y : ndarray
        The dependent variable.
    y_dir : ndarray
        The derivative in the specified directions.
    y_name : str
        Label for the dependent variable.
    cov_model :
        A covariance model object.
    interpolate : ndarray
        A boolean vector.
    training_log : dict
        The output of a genetic algorithm optimization.
    graph :
        A TensorFlow graph.
    """

    def __init__(self, point_data, variable, dir_data,
                 kernels, dir_variable=None, interpolate=None):
        """
        Initializer for GPGrad.

        Parameters
        ----------
        point_data :
            A Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        dir_data :
            A Directions1D, Directions2D, or Directions3D object.
        kernels : tuple or list
            A container with the desired kernels to model the data.
        dir_variable : str
            The name of the column with the derivatives of the dependent
            variable. If omitted, the derivatives are assumed to be 0.
        interpolate : str
            The name of a column containing a Boolean variable, indicating
            which data points must be honored.
        """
        self.x = point_data.coords
        self.y = point_data.data[variable].values
        self.y_name = variable
        self._ndim = point_data.ndim
        self.cov_model = _ker.CovarianceModelRegression(
            kernels, warping=[])
        self.x_dir = dir_data.coords
        self.directions = dir_data.directions
        if dir_variable is None:
            self.y_dir = _np.zeros(self.x_dir.shape[0])
        else:
            self.y_dir = dir_data.data[dir_variable].values
        if interpolate is None:
            self.interpolate = _np.repeat(False, len(self.y))
        else:
            self.interpolate = point_data.data[interpolate].values
        self.training_log = None
        self.graph = _tf.Graph()

        # tidying up
        # joining x and x_dir for a composite bounding box
        n_dim = self.x.shape[1]
        if n_dim == 1:
            temp_data = _data.Points1D(_np.concatenate([self.x, self.x_dir],
                                                       axis=0))
        elif n_dim == 2:
            temp_data = _data.Points2D(_np.concatenate([self.x, self.x_dir],
                                                       axis=0))
        elif n_dim == 3:
            temp_data = _data.Points3D(_np.concatenate([self.x, self.x_dir],
                                                       axis=0))
        self.cov_model.auto_set_kernel_parameter_limits(temp_data)

        # TensorFlow
        self.graph = _tf.Graph()
        with self.graph.as_default():
            with _tf.name_scope("GP_Grad"):
                # constants
                x = _tf.constant(self.x,
                                 dtype=_tf.float64,
                                 name="coords")
                x_dir = _tf.constant(self.x_dir,
                                     dtype=_tf.float64,
                                     name="direction_coords")
                directions = _tf.constant(self.directions,
                                          dtype=_tf.float64,
                                          name="directions")
                interp = _tf.constant(self.interpolate,
                                      dtype=_tf.bool)

                # trainable parameters
                with _tf.name_scope("init_placeholders"):
                    self.cov_model.init_tf_placeholder()

                # placeholders
                y_tf = _tf.placeholder(_tf.float64, shape=(len(self.y), 1),
                                       name="y_tf")
                y_dir = _tf.placeholder(_tf.float64,
                                        shape=(len(self.y_dir), 1),
                                        name="y_dir")
                x_new_tf = _tf.placeholder_with_default(
                    _tf.zeros([1, self.x.shape[1]], dtype=_tf.float64),
                    shape=[None, self.x.shape[1]],
                    name="x_new_tf")
                jitter_tf = _tf.placeholder(_tf.float64, shape=[],
                                            name="jitter")

                # covariance matrix
                mat_k = self._covariance_matrix(x, interp, x_dir=x_dir,
                                                directions=directions,
                                                jitter=jitter_tf)

                # alpha
                y_join = _tf.concat([y_tf, y_dir], axis=0)
                mat_chol, alpha = self._alpha(mat_k, y_join)

                # log-likelihood
                yd = _tf.ones_like(y_join)
                log_lik = self._log_lik(mat_chol, alpha, y_join, yd)

                # prediction
                pred_mu, pred_var = self._predict(mat_chol, alpha, x, x_new_tf,
                                                  x_dir, directions,
                                                  interp)

                self.tf_handles = {"L": mat_chol,
                                   "alpha": alpha,
                                   "y_tf": y_tf,
                                   "y_dir": y_dir,
                                   "K": mat_k,
                                   "log_lik": log_lik,
                                   "x_new_tf": x_new_tf,
                                   "pred_mu": pred_mu,
                                   "pred_var": pred_var,
                                   "jitter": jitter_tf,
                                   "init": _tf.global_variables_initializer()}

        self.graph.finalize()

    def _covariance_matrix(self, x, interp, x_dir, directions, y=None,
                           jitter=1e-9):
        with _tf.name_scope("covariance_matrix"):
            if y is None:
                nugget = self.cov_model.variance.tf_val[-1]
                k_x = _tf.add(self.cov_model.covariance_matrix(x, y),
                              self.cov_model.nugget.nugget_matrix(x, interp)
                              * nugget)
                # adding jitter to avoid Cholesky decomposition problems
                sk = _tf.shape(k_x)
                k_x = k_x + _tf.diag(_tf.ones(
                    sk[0], dtype=_tf.float64) * jitter)

                k_x_dir = self.cov_model.covariance_matrix_d1(x, x_dir,
                                                              directions)
                k_dir = self.cov_model.covariance_matrix_d2(x_dir, x_dir,
                                                            directions,
                                                            directions)
                # jitter in directions
                k_dir = k_dir + _tf.eye(_tf.shape(k_dir)[0],
                                        dtype=_tf.float64) * jitter / 10
                k = _tf.concat([_tf.concat([k_x, k_x_dir], axis=1),
                                _tf.concat([_tf.transpose(k_x_dir), k_dir],
                                           axis=1)],
                               axis=0)
            else:
                k_y_x = self.cov_model.covariance_matrix(y, x)
                k_y_x_dir = self.cov_model.covariance_matrix_d1(y, x_dir,
                                                                directions)
                k = _tf.transpose(_tf.concat([k_y_x, k_y_x_dir], axis=1))
        return k

    def _predict(self, chol, alpha, x, x_new, x_dir, directions, interp):
        with _tf.name_scope("Prediction"):
            k_new = self._covariance_matrix(x, interp, x_dir, directions, x_new)
            pred_mu = _tf.squeeze(_tf.matmul(k_new, alpha, transpose_a=True),
                                  name="pred_mean")
            with _tf.name_scope("pred_var"):
                point_var = self.cov_model.point_variance(x_new)
                pred_var = _tf.linalg.cholesky_solve(chol, k_new)
                pred_var = _tf.multiply(k_new, pred_var)
                pred_var = point_var - _tf.reduce_sum(pred_var, axis=0)
                # adding nugget
                pred_var = pred_var + self.cov_model.variance.tf_val[-1]
        return pred_mu, pred_var

    def log_lik(self, session=None):
        if session is None:
            with _tf.Session(graph=self.graph) as session:
                log_lik = self.log_lik(session)
            return log_lik

        y = self.y
        y_dir = self.y_dir
        # session
        feed = self.cov_model.feed_dict()
        feed.update({
            self.tf_handles["y_tf"]: _np.resize(y, (len(y), 1)),
            self.tf_handles["y_dir"]: _np.resize(y_dir, (len(y_dir), 1)),
            self.tf_handles["jitter"]: self.cov_model.jitter})
        session.run(self.tf_handles["init"], feed_dict=feed)
        log_lik = session.run(self.tf_handles["log_lik"], feed_dict=feed)
        return log_lik

    def predict(self, newdata, perc=(0.025, 0.25, 0.5, 0.75, 0.975),
                name=None, verbose=True, batch_size=20000):
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        y = self.y
        y_dir = self.y_dir

        # prediction in batches
        n_data = x_new.shape[0]
        n_batches = int(_np.ceil(n_data / batch_size))
        batch_id = [_np.arange(i * batch_size,
                               _np.minimum((i + 1) * batch_size, n_data))
                    for i in range(n_batches)]

        mu = _np.array([])
        var = _np.array([])
        for i in range(n_batches):
            if verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            # placeholders
            feed = self.cov_model.feed_dict()
            feed.update({
                self.tf_handles["y_tf"]: _np.resize(y, (len(y), 1)),
                self.tf_handles["y_dir"]: _np.resize(y_dir, (len(y_dir), 1)),
                self.tf_handles["jitter"]: self.cov_model.jitter,
                self.tf_handles["x_new_tf"]: x_new[batch_id[i], :]})

            # TensorFlow
            with _tf.Session(graph=self.graph) as session:
                session.run(self.tf_handles["init"], feed_dict=feed)
                handles = [self.tf_handles["pred_mu"],
                           self.tf_handles["pred_var"]]
                mu_i, var_i = session.run(handles, feed_dict=feed)

            # update
            mu = _np.concatenate([mu, mu_i])
            var = _np.concatenate([var, var_i])
        if verbose:
            print("\n")

        # output
        newdata.data[name + "_mean"] = mu
        newdata.data[name + "_variance"] = var
        for p in perc:
            newdata.data[name + "_p" + str(p)] = self.cov_model.warp_backward(
                _st.norm.ppf(p, loc=mu, scale=_np.sqrt(var)))


class GPClassif(_Model):
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

    def __init__(self, sp_data, var_1, var_2, kernels, dir_data=None,
                 sparse=False, ensemble=False, **kwargs):
        """
        Initializer for GPClassif.

        Parameters
        ----------
        sp_data :
            A Points1D, Points2D, or Points3D object.
        var_1, var_2 : str
            The columns with the class labels to model.
        kernels : tuple or list
            A container with the desired kernels to model the data.
        dir_data :
            A Directions1D, Directions2D, or Directions3D object.
            It is assumed that these are directions with zero variation.
        sparse : bool
            Whether to use a sparse model.
        ensemble : bool
            If sparse=True, whether to use an ensemble model.
        kwargs :
            Passed on to the appropriate model.
        """
        self.x = sp_data.coords
        self._ndim = sp_data.ndim
        labels = _pd.api.types.union_categoricals(
            [_pd.Categorical(sp_data.data[var_1]),
             _pd.Categorical(sp_data.data[var_2])])
        labels = labels.categories.values

        # latent variables
        n_latent = len(labels)
        n_data = sp_data.coords.shape[0]
        labels_dict = dict(zip(labels, _np.arange(n_latent)))
        # lat1 = _np.ones([n_data, n_latent]) * (-1 / n_latent)
        # lat2 = _np.ones([n_data, n_latent]) * (-1 / n_latent)
        lat1 = _np.ones([n_data, n_latent]) * -1
        lat2 = _np.ones([n_data, n_latent]) * -1
        idx1 = sp_data.data[var_1].map(labels_dict).values
        idx2 = sp_data.data[var_2].map(labels_dict).values
        for i in range(n_data):
            lat1[i, idx1[i]] = 1
            lat2[i, idx2[i]] = 1
        lat = 0.5 * lat1 + 0.5 * lat2
        interpolate = _np.apply_along_axis(
            lambda x: x == 0,  # (1 - 1 / n_latent) / 2,
            axis=0, arr=lat)
        self.latent = _pd.DataFrame(lat, columns=labels)

        # latent variable models
        self.GPs = []
        n_latent = len(labels)
        temp_data = None
        if self.ndim == 1:
            temp_data = _data.Points1D(self.x, _pd.DataFrame(
                self.latent.values, columns=labels))
        elif self.ndim == 2:
            temp_data = _data.Points2D(self.x, _pd.DataFrame(
                self.latent.values, columns=labels))
        elif self.ndim == 3:
            temp_data = _data.Points3D(self.x, _pd.DataFrame(
                self.latent.values, columns=labels))
        for i in range(n_latent):
            temp_data.data["boundary"] = interpolate[:, i]

            # full GP
            if not sparse:
                if dir_data is None:
                    gp = GP(sp_data=temp_data,
                            variable=temp_data.data.columns[i],
                            kernels=_copy.deepcopy(kernels),
                            warping=[_warp.Identity()],
                            interpolate="boundary")
                else:
                    gp = GPGrad(point_data=temp_data,
                                variable=temp_data.data.columns[i],
                                kernels=_copy.deepcopy(kernels),
                                interpolate="boundary",
                                dir_data=dir_data)
            else:
                # sparse GP
                if not ensemble:
                    if dir_data is None:
                        gp = SparseGP(sp_data=temp_data,
                                      variable=temp_data.data.columns[i],
                                      kernels=_copy.deepcopy(kernels),
                                      warping=[_warp.Identity()],
                                      interpolate="boundary",
                                      **kwargs)
                    else:
                        raise NotImplementedError()
                # sparse GP ensemble
                else:
                    if dir_data is None:
                        gp = SparseGPEnsemble(
                            sp_data=temp_data,
                            variable=temp_data.data.columns[i],
                            kernels=_copy.deepcopy(kernels),
                            warping=[_warp.Identity()],
                            interpolate="boundary",
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
        for i in range(len(self.GPs)):
            if verb:
                print("\nTraining latent variable " + str(i) + "\n")
            self.GPs[i].train(**kwargs)

    def predict(self, newdata, name, verbose=True, batch_size=20000,
                n_samples=None, **kwargs):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension.
            The prediction is written on the object's data attribute in-place.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        verbose : bool
            Whether to display progress info on console.
        batch_size : int
            Number of data points to process in each batch.
            If you get an out-of-memory
            TensorFlow error, try decreasing this value.
        n_samples : int
            Number of sample to draw in order to estimate the classes'
            probabilities.
        kwargs :
            Passed on to the appropriate predict method.
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
        n_classes = n_latent
        n_out = newdata.coords.shape[0]
        mu = _np.zeros([n_out, n_latent])
        var = _np.zeros([n_out, n_latent])

        # prediction of latent variables
        for i in range(n_latent):
            if verbose:
                print("Predicting latent variable " +
                      str(i + 1) + " of " + str(n_latent) + ":")
            self.GPs[i].predict(temp_data, perc=[], name="out", verbose=verbose,
                                batch_size=batch_size, **kwargs)
            mu[:, i] = temp_data.data["out_mean"].values
            var[:, i] = temp_data.data["out_variance"].values
        std = _np.sqrt(var)

        # probabilities, indicators, and entropy
        if verbose:
            print("Calculating probabilities:")
        ind = mu / (std + 1e-3)
        top_2 = - _np.partition(-ind, 1, axis=1)[:, 0:2]
        mean_top_2 = _np.tile(_np.mean(top_2, axis=1, keepdims=True),
                              [1, n_latent])
        ind = ind - mean_top_2
        prob = _ss.softmax(ind, axis=1)
        entropy = - _np.sum(prob * _np.log(prob), axis=1) / _np.log(n_latent)
        uncertainty = _np.sqrt(entropy * _np.mean(var, axis=1))

        # output
        idx = self.latent.columns
        newdata.data[name] = idx[_np.argmax(prob, axis=1)]
        for i in range(n_classes):
            newdata.data[name + "_" + idx[i] + "_prob"] = prob[:, i]
        for i in range(n_classes):
            newdata.data[name + "_" + idx[i] + "_ind"] = ind[:, i]
        newdata.data[name + "_entropy"] = entropy
        newdata.data[name + "_uncertainty"] = uncertainty

    def __str__(self):
        s = ""
        for model in self.GPs:
            s += "\nModel for " + model.y_name + ":\n"
            s += str(model)
        return s

    def save_state(self, file):
        d = [gp.cov_model.params_dict(complete=True) for gp in self.GPs]
        with open(file, 'wb') as f:
            _pickle.dump(d, f)

    def load_state(self, file):
        with open(file, 'rb') as f:
            d = _pickle.load(f)
        for i in range(len(self.GPs)):
            self.GPs[i].update_params(d[i])


class SparseGP(GP):
    def __init__(self, sp_data, variable, kernels, warping=(),
                 interpolate=None, pseudo_inputs=None,
                 include_trace_penalty=False):
        """
        Initializer for SparseGP.

        Parameters
        ----------
        sp_data :
            A Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        kernels : tuple or list
            A container with the desired kernels to model the data.
        warping : tuple or list
            A container with the desired warping objects to model the data.
        interpolate : str
            The name of a column containing a Boolean variable, indicating which
            data points must be honored.
        pseudo_inputs :
            A spatial object with the same dimensionality of patched_data. Defaults
            to patched_data, but this risks incurring in an out-of-memory error if
            there are too many data points.
        include_trace_penalty : bool
            Whether to consider the trace penalty in the log-likelihood.
        """
        self.x = sp_data.coords
        self.y = sp_data.data[variable].values
        self.y_name = variable
        self._ndim = sp_data.ndim
        self.cov_model = _ker.CovarianceModelSparse(
            kernels, warping, pseudo_inputs)
        if interpolate is None:
            self.interpolate = _np.repeat(False, len(self.y))
        else:
            self.interpolate = sp_data.data[interpolate].values
        self.training_log = None

        # tidying up
        self.cov_model.auto_set_kernel_parameter_limits(sp_data)
        n_ps = pseudo_inputs.coords.shape[0]
        n_data = self.x.shape[0]

        # y variable initialization is currently outside TensorFlow
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)

        # TensorFlow
        self.graph = _tf.Graph()
        with self.graph.as_default():
            with _tf.name_scope("Sparse_GP"):
                # constants
                x = _tf.constant(self.x,
                                 dtype=_tf.float64,
                                 name="coords")
                interp = _tf.constant(self.interpolate,
                                      dtype=_tf.bool)

                # trainable parameters
                with _tf.name_scope("init_placeholders"):
                    self.cov_model.init_tf_placeholder()

                # placeholders
                y_tf = _tf.placeholder(_tf.float64, shape=(len(y), 1),
                                       name="y_tf")
                yd_tf = _tf.placeholder(_tf.float64, shape=(len(y), 1),
                                        name="yd_tf")
                jitter_tf = _tf.placeholder(_tf.float64, shape=[],
                                            name="jitter")
                x_new = _tf.placeholder_with_default(
                    _tf.zeros([1, self.x.shape[1]], dtype=_tf.float64),
                    shape=[None, self.x.shape[1]],
                    name="x_new")
                n_sim = _tf.placeholder_with_default(
                    _tf.constant(0, _tf.int32), shape=[], name="n_sim"
                )
                sim_with_noise = _tf.placeholder_with_default(
                    False, shape=[], name="sim_with_noise")
                seed = _tf.placeholder_with_default(
                    _tf.constant(1234, _tf.int32), shape=[],
                    name="seed"
                )

                # pseudo-inputs
                with _tf.name_scope("pseudo_inputs"):
                    ps_coords = self.cov_model.ps_coords.tf_val
                    ps_coords = _tf.reshape(ps_coords, [-1, self.ndim])

                # covariance matrices
                with _tf.name_scope("covariance_matrices"):
                    k_uu = self._covariance_matrix(
                        ps_coords,
                        interp=_tf.constant(False, _tf.bool, shape=[n_ps]),
                        jitter=jitter_tf)
                    chol_uu = _tf.linalg.cholesky(k_uu)
                    chol_uu = _tf.Variable(
                        lambda: chol_uu,
                        validate_shape=False,
                        name="chol_uu",
                        collections=[_tf.GraphKeys.GLOBAL_VARIABLES])
                    # k_uf = self._covariance_matrix(ps_coords, y=x)
                    k_uf_new = self._covariance_matrix(ps_coords, y=x_new)

                    # nugget vector
                    nugget = self.cov_model.variance.tf_val[-1]
                    nugget = _tf.where(interp,
                                       _tf.fill(_tf.shape(interp), jitter_tf),
                                       _tf.fill(_tf.shape(interp), nugget),
                                       name="nugget")

                # pre-computations
                with _tf.name_scope("pre_computations"):
                    # s_uf = _tf.shape(k_uf)
                    # nug = _tf.reshape(nugget, [1, s_uf[1]])
                    # nug = _tf.tile(nug, [s_uf[0], 1])
                    # sigma = _tf.matmul(k_uf, k_uf / nug, False, True) + k_uu
                    # sigma_chol = _tf.linalg.cholesky(sigma)
                    #
                    # y_nug = y_tf / _tf.expand_dims(nugget, axis=1)
                    # k_uf_y_nug = _tf.matmul(k_uf, y_nug)

                    # training data processed in batches to save memory
                    batch_size = 1000
                    n_batches = int(_np.ceil(n_data / batch_size))
                    batch_id = [_np.arange(
                        i * batch_size,
                        _np.minimum((i + 1) * batch_size, n_data))
                        for i in range(n_batches)]
                    sigma = k_uu
                    y_nug = y_tf / _tf.expand_dims(nugget, axis=1)
                    k_uf_y_nug = _tf.zeros([n_ps, 1], _tf.float64)

                    with _tf.name_scope("training_data_loop"):
                        for bid in batch_id:
                            point_i = _tf.gather(x, bid)
                            k = self._covariance_matrix(ps_coords, y=point_i)
                            nug = _tf.expand_dims(_tf.gather(nugget, bid),
                                                  axis=0)
                            nug = _tf.tile(nug, [n_ps, 1])
                            sigma = sigma + _tf.matmul(k, k / nug, False, True)
                            k_uf_y_nug = k_uf_y_nug \
                                         + _tf.matmul(k, _tf.gather(y_nug, bid))

                    # def pre_comp_loop(i, sigma_i, k_uf_y_i):
                    #     point_i = _tf.expand_dims(_tf.gather(x, i), axis=0)
                    #     k = self._covariance_matrix(ps_coords, y=point_i)
                    #     sigma_i = sigma_i \
                    #               + _tf.matmul(k, k, False, True) \
                    #               / _tf.gather(nugget, i)
                    #     k_uf_y_i = k_uf_y_i + k * _tf.gather(y_nug, i)
                    #     return i + 1, sigma_i, k_uf_y_i
                    #
                    # def pre_comp_cond(i, sigma_i, k_uf_y_i):
                    #     return _tf.less(i, x.shape[0])
                    #
                    # _, sigma, k_uf_y_nug = _tf.while_loop(
                    #     pre_comp_cond, pre_comp_loop,
                    #     [_tf.constant(0), sigma, k_uf_y_nug],
                    #     parallel_iterations=1000)
                    sigma_chol = _tf.linalg.cholesky(sigma)

                    # alpha
                    with _tf.name_scope("alpha"):
                        alpha = _tf.linalg.cholesky_solve(sigma_chol,
                                                          k_uf_y_nug)
                        alpha = _tf.Variable(lambda: alpha,
                                             validate_shape=False)

                    u_sqrt = _tf.linalg.solve(sigma_chol, k_uu)
                    # u_sqrt = _tf.Variable(
                    #     lambda: u_sqrt,
                    #     validate_shape=False,
                    #     name="u_sqrt",
                    #     collections=[_tf.GraphKeys.GLOBAL_VARIABLES])
                    u_mean = _tf.matmul(k_uu, alpha)
                    u_mean = _tf.Variable(
                        lambda: u_mean,
                        validate_shape=False,
                        name="u_mean",
                        collections=[_tf.GraphKeys.GLOBAL_VARIABLES])

                # log-likelihood
                with _tf.name_scope("log_lik"):
                    # var_full = self.cov_model.point_variance(x)
                    # var_approx = _tf.reduce_sum(
                    #     _tf.linalg.solve(chol_uu, k_uf) ** 2, axis=0)
                    # var_dif = var_full - var_approx

                    fit = -0.5 * (
                        _tf.reduce_sum(y_tf * y_nug)
                        - _tf.reduce_sum(k_uf_y_nug * alpha)
                    )
                    det = -0.5 * (
                        2 * _tf.reduce_sum(_tf.log(_tf.diag_part(sigma_chol)))
                        - 2 * _tf.reduce_sum(_tf.log(_tf.diag_part(chol_uu)))
                        + _tf.reduce_sum(_tf.log(nugget))
                    )
                    const = -0.5 * _tf.constant(
                        y.shape[0] * _np.log(2 * _np.pi), _tf.float64)
                    # trace = -0.5 * _tf.reduce_sum(var_dif / nugget)
                    warp = _tf.reduce_sum(_tf.log(yd_tf))
                    log_lik = fit + det + const + warp  # + trace
                    # if include_trace_penalty:
                    #     log_lik = log_lik + trace

                # prediction
                with _tf.name_scope("prediction"):
                    # mean
                    with _tf.name_scope("mean"):
                        pred_mu = _tf.squeeze(
                            _tf.matmul(k_uf_new, alpha, True, False))

                    # variance
                    with _tf.name_scope("variance"):
                        new_var_full = self.cov_model.point_variance(x_new)
                        new_var_approx = _tf.reduce_sum(
                            _tf.linalg.solve(chol_uu, k_uf_new) ** 2, axis=0)
                        pred_var_approx = _tf.reduce_sum(
                            _tf.linalg.solve(sigma_chol, k_uf_new) ** 2, axis=0)
                        pred_var_full = new_var_full \
                                        - new_var_approx \
                                        + pred_var_approx

                # simulation
                with _tf.name_scope("simulation"):
                    sim_shape = [n_ps, n_sim]

                    # stateless required to get the same results across batches
                    sim_normal = _tf.random.stateless_normal(
                        sim_shape, dtype=_tf.float64, seed=[seed, 0])

                    ps_sim = _tf.matmul(u_sqrt, sim_normal, True, False) \
                             + _tf.tile(u_mean, [1, n_sim])
                    ps_sim = _tf.linalg.cholesky_solve(chol_uu, ps_sim)
                    y_sim = _tf.matmul(k_uf_new, ps_sim, True, False)

                    with _tf.name_scope("noise"):
                        def noise(mat):
                            rand = _tf.random.normal(
                                shape=_tf.shape(mat), dtype=_tf.float64,
                                seed=seed
                            )
                            total_noise = pred_var_full \
                                          - pred_var_approx \
                                          + self.cov_model.variance.tf_val[-1]
                            total_noise = _tf.reshape(total_noise, [-1, 1])
                            total_noise = _tf.tile(total_noise, [1, n_sim])
                            return mat + rand * _tf.sqrt(total_noise)

                        y_sim = _tf.cond(sim_with_noise,
                                         lambda: noise(y_sim),
                                         lambda: y_sim)

                self.tf_handles = {
                    "alpha": alpha,
                    "y_tf": y_tf,
                    "yd_tf": yd_tf,
                    "log_lik": log_lik,
                    "x_new": x_new,
                    "pred_mu": pred_mu,
                    "pred_var": pred_var_full,
                    "pred_var_approx": pred_var_approx,
                    "jitter": jitter_tf,
                    "n_sim": n_sim,
                    "sim_with_noise": sim_with_noise,
                    "y_sim": y_sim,
                    "seed": seed,
                    "init": _tf.global_variables_initializer()}

        self.graph.finalize()

    def predict(self, newdata, perc=(0.025, 0.25, 0.5, 0.75, 0.975),
                name=None, verbose=True, batch_size=20000, n_sim=100,
                add_noise=False, seed=1234):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension. The
            prediction is written on the object's data attribute in-place.
        perc : tuple, list, or array
            The desired percentiles of the predictive distribution.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        verbose : bool
            Whether to display progress info on console.
        batch_size : int
            Number of data points to process in each batch.
            If you get an out-of-memory
            TensorFlow error, try decreasing this value.
        n_sim : int
            The desired number of simulations.
        add_noise : bool
            Whether to include the model's noise in the simulations.
        seed : int
            A seed number to produce consistent simulations.
        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # updating y
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)

        # session and placeholders
        session = _tf.Session(graph=self.graph)
        feed = self.cov_model.feed_dict()
        feed.update({self.tf_handles["y_tf"]: _np.resize(y, (len(y), 1)),
                     self.tf_handles["jitter"]: self.cov_model.jitter,
                     self.tf_handles["n_sim"]: n_sim,
                     self.tf_handles["sim_with_noise"]: add_noise,
                     self.tf_handles["seed"]: seed
                     })
        session.run(self.tf_handles["init"], feed_dict=feed)
        handles = [self.tf_handles["pred_mu"],
                   self.tf_handles["pred_var"],
                   self.tf_handles["pred_var_approx"],
                   self.tf_handles["y_sim"]]

        # prediction in batches
        n_data = x_new.shape[0]
        n_batches = int(_np.ceil(n_data / batch_size))
        batch_id = [_np.arange(i * batch_size,
                               _np.minimum((i + 1) * batch_size, n_data))
                    for i in range(n_batches)]

        mu = _np.array([])
        var = _np.array([])
        var_approx = _np.array([])
        y_sim = _np.empty([0, n_sim])
        for i in range(n_batches):
            if verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            feed.update({self.tf_handles["x_new"]: x_new[batch_id[i], :]})

            # TensorFlow
            mu_i, var_i, var_ap_i, sim_i = session.run(handles, feed_dict=feed)

            # update
            mu = _np.concatenate([mu, mu_i])
            var = _np.concatenate([var, var_i])
            var_approx = _np.concatenate([var_approx, var_ap_i])
            y_sim = _np.concatenate([y_sim, sim_i], axis=0)
        session.close()
        if verbose:
            print("\n")

        # output
        newdata.data[name + "_mean"] = mu
        newdata.data[name + "_variance"] = var
        newdata.data[name + "_variance_approximated"] = var_approx
        for p in perc:
            newdata.data[name + "_p" + str(p)] = self.cov_model.warp_backward(
                _st.norm.ppf(p, loc=mu, scale=_np.sqrt(var)))

        # simulations
        n_digits = str(len(str(n_sim)) - 1)
        cols = ["sim_" + ("{:0>" + n_digits + "d}").format(i)
                for i in range(n_sim)]
        for i in range(n_sim):
            newdata.data[cols[i]] = self.cov_model.warp_backward(y_sim[:, i])


class SparseGPEnsemble(_Model):
    def __init__(self, sp_data, variable, kernels, sparse_pseudo_inputs,
                 dense_pseudo_inputs, region, warping=(),
                 interpolate=None, seed=None,
                 include_trace_penalty=False):
        """
        Initializer for SparseGPEnsemble.

        Parameters
        ----------
        sp_data :
            A Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        kernels : tuple or list
            A container with the desired kernels to model the data.
        warping : tuple or list
            A container with the desired warping objects to model the data.
        interpolate : str
            The name of a column containing a Boolean variable, indicating which
            data points must be honored.

        seed : int
            A seed number to produce consistent simulations.
        include_trace_penalty : bool
            Whether to consider the trace penalty in the log-likelihood.
        """
        super().__init__()
        if dense_pseudo_inputs.ndim != sparse_pseudo_inputs.ndim:
            raise ValueError("pseudo_inputs dimensions mismatch")
        self._ndim = dense_pseudo_inputs.ndim

        if seed is None:
            seed = 1234
        region_id = _pd.unique(dense_pseudo_inputs.data[region])
        coords_sparse = sparse_pseudo_inputs.coords

        self.GPs = []
        for i in range(len(region_id)):
            print("\rInitializing model", str(i+1), "of", str(len(region_id)),
                  end="")
            keep = dense_pseudo_inputs.data[region] == region_id[i]
            coords_i = dense_pseudo_inputs.coords[keep, :]
            df = _pd.DataFrame(_np.concatenate([coords_sparse, coords_i],
                                               axis=0))
            coords_i = df.drop_duplicates().values
            if self.ndim == 1:
                coords_i = _data.Points1D(coords_i)
            if self.ndim == 2:
                coords_i = _data.Points2D(coords_i)
            if self.ndim == 3:
                coords_i = _data.Points3D(coords_i)
            gp = SparseGP(sp_data, variable, _copy.deepcopy(kernels),
                          warping=_copy.deepcopy(warping),
                          interpolate=interpolate, seed=seed + i,
                          pseudo_inputs=coords_i)
            self.GPs.append(gp)
        print("\n")
        self.base_gp = SparseGP(sp_data, variable, _copy.deepcopy(kernels),
                                warping=_copy.deepcopy(warping),
                                interpolate=interpolate,
                                pseudo_inputs=sparse_pseudo_inputs,
                                include_trace_penalty=include_trace_penalty)

    def train(self, **kwargs):
        self.base_gp.train(**kwargs)
        params = self.base_gp.cov_model.params_dict(True)
        df = _pd.DataFrame(params)
        params = df.loc[df["param_type"] != "pseudo_inputs", :].to_dict("list")
        for model in self.GPs:
            model.cov_model.update_params(params)

    def update_params(self, params):
        df = _pd.DataFrame(params)
        params = df.loc[df["param_type"] != "pseudo_inputs", :].to_dict("list")
        self.base_gp.update_params(params)
        for model in self.GPs:
            model.update_params(params)

    def save_state(self, file):
        self.base_gp.save_state(file)

    def load_state(self, file):
        self.base_gp.load_state(file)
        params = self.base_gp.cov_model.params_dict(True)
        df = _pd.DataFrame(params)
        params = df.loc[df["param_type"] != "pseudo_inputs", :].to_dict("list")
        for model in self.GPs:
            model.cov_model.update_params(params)

    def predict(self, newdata, perc=(0.025, 0.25, 0.5, 0.75, 0.975),
                name=None, verbose=True, batch_size=20000, n_sim=100,
                add_noise=False):
        """
        Makes a prediction on the specified coordinates.

        Parameters
        ----------
        newdata :
            A reference to a spatial points object of compatible dimension. The
            prediction is written on the object's data attribute in-place.
        perc : tuple, list, or array
            The desired percentiles of the predictive distribution.
        name : str
            Name of the predicted variable, used as a prefix in the output
            columns. Defaults to self.y_name.
        verbose : bool
            Whether to display progress info on console.
        batch_size : int
            Number of data points to process in each batch.
            If you get an out-of-memory
            TensorFlow error, try decreasing this value.
        n_sim : int
            The desired number of simulations.
        add_noise : bool
            Whether to include the model's noise in the simulations.
        """
        n_data = newdata.coords.shape[0]
        mu = _np.zeros([n_data])
        var = _np.zeros([n_data])
        var_ap = _np.zeros([n_data])
        weights = _np.zeros([n_data])
        back_tr = _np.zeros([n_data, len(perc)])
        y_sim = _np.zeros([n_data, n_sim])

        tmpdata = None
        if self.ndim == 1:
            tmpdata = _data.Points1D(newdata.coords)
        if self.ndim == 2:
            tmpdata = _data.Points2D(newdata.coords)
        if self.ndim == 3:
            tmpdata = _data.Points3D(newdata.coords)

        for i in range(len(self.GPs)):
            if verbose:
                print("Processing model", str(i+1), "of", str(len(self.GPs)))

            model = self.GPs[i]
            model.predict(tmpdata, perc, name, verbose, batch_size,
                          n_sim, add_noise)
            mu_i = tmpdata.data[name + "_mean"].values
            var_i = tmpdata.data[name + "_variance"].values
            var_ap_i = tmpdata.data[name + "_variance_approximated"].values
            back_tr_i = _np.array([model.cov_model.warp_backward(_st.norm.ppf(
                p, loc=mu_i, scale=_np.sqrt(var_i))) for p in perc])
            back_tr_i = back_tr_i.transpose()
            sim_i = tmpdata.data.filter(like="sim").values
            weights_i = 1 / (var_i - var_ap_i)

            weights += weights_i
            var += weights_i * var_i
            var_ap += weights_i * var_ap_i
            mu += weights_i * mu_i
            back_tr += _np.expand_dims(weights_i, axis=1) * back_tr_i
            y_sim += _np.expand_dims(weights_i, axis=1) * sim_i

        mu = mu / weights
        var = var / weights
        var_ap = var_ap / weights
        back_tr = back_tr / _np.expand_dims(weights, axis=1)
        y_sim = y_sim / _np.expand_dims(weights, axis=1)

        # output
        newdata.data[name + "_mean"] = mu
        newdata.data[name + "_variance"] = var
        newdata.data[name + "_variance_approximated"] = var_ap
        for i in range(len(perc)):
            newdata.data[name + "_p" + str(perc[i])] = back_tr[:, i]

        # simulations
        n_digits = str(len(str(n_sim)))
        cols = ["sim_" + ("{:0>" + n_digits + "d}").format(i)
                for i in range(n_sim)]
        for i in range(n_sim):
            newdata.data[cols[i]] = y_sim[:, i]


class SPICE(GP):
    """
    SPatial Interpolation on Circulant Embedding Gaussian Process.
    """

    def __init__(self, sp_data, variable, kernels, circulant_grid, warping=(),
                 interpolate=None):
        """
        Initializer for SPICE.

        Parameters
        ----------
        sp_data :
            A Points1D, Points2D, or Points3D object.
        variable : str
            The name of the column with the data to model.
        kernels : tuple or list
            A container with the desired kernels to model the data.
        circulant_grid :
            A CirculantGrid1D, CirculantGrid2D, or CirculantGrid3D object.
            Its bounding box must contain all the data points (train and test)
            and should have a margin of at least
            20% of empty space in each direction.
        warping : tuple or list
            A container with the desired warping objects to model the data.
        interpolate : str
            The name of a column containing a Boolean variable, indicating which
            data points must be honored.
        """
        self.data = sp_data
        self.y = sp_data.data[variable].values
        self.y_name = variable
        self._ndim = sp_data.ndim
        self.circulant_grid = circulant_grid
        self.cov_model = _ker.CovarianceModelRegression(
            kernels, warping)
        if interpolate is None:
            self.interpolate = _np.repeat(False, len(self.y))
        else:
            self.interpolate = sp_data.data[interpolate].values
        self.training_log = None

        if sp_data.ndim != circulant_grid.ndim:
            raise ValueError("Data and circulant grid must have matching "
                             "dimensions")

        # tidying up
        self.cov_model.auto_set_kernel_parameter_limits(sp_data)
        n_data = sp_data.coords.shape[0]
        n_circ = circulant_grid.coords.shape[0]

        # y variable initialization is currently outside TensorFlow
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)
        yd = self.cov_model.warp_derivative(self.y)

        # interpolation weights
        w_mat = circulant_grid.interpolation_weights(sp_data)

        # TensorFlow
        self.graph = _tf.Graph()
        with self.graph.as_default():
            with _tf.name_scope("SPICE"):
                # constants
                interp = _tf.constant(self.interpolate,
                                      dtype=_tf.bool)
                circulant_coords = _tf.constant(circulant_grid.coords,
                                                dtype=_tf.float64)
                point_zero = _tf.constant(circulant_grid.point_zero,
                                          _tf.float64)

                # trainable parameters
                with _tf.name_scope("init_placeholders"):
                    self.cov_model.init_tf_placeholder()

                # placeholders
                y_tf = _tf.compat.v1.placeholder(_tf.float64, shape=(None, 1),
                                       name="y_tf")
                yd_tf = _tf.compat.v1.placeholder(_tf.float64, shape=(None, 1),
                                        name="yd_tf")
                jitter_tf = _tf.compat.v1.placeholder(_tf.float64, shape=[],
                                            name="jitter")
                x = _tf.placeholder_with_default(
                    _tf.constant(sp_data.coords, dtype=_tf.float64),
                    shape=[None, self.ndim],
                    name="x"
                )
                x_new = _tf.placeholder_with_default(
                    point_zero,
                    shape=[None, self.ndim],
                    name="x_new")
                n_sim = _tf.placeholder_with_default(
                    _tf.constant(50, _tf.int32), shape=[], name="n_sim"
                )
                sim_with_noise = _tf.placeholder_with_default(
                    False, shape=[], name="sim_with_noise")
                seed = _tf.placeholder_with_default(
                    _tf.constant(1234, _tf.int32), shape=[],
                    name="seed"
                )
                w_row = _tf.placeholder_with_default(
                    _tf.constant(w_mat.row, _tf.int64), shape=[None],
                    name="w_row")
                w_col = _tf.placeholder_with_default(
                    _tf.constant(w_mat.col, _tf.int64), shape=[None],
                    name="w_col")
                w_data = _tf.placeholder_with_default(
                    _tf.constant(w_mat.data, _tf.float64), shape=[None],
                    name="w_data")
                w_row_new = _tf.placeholder_with_default(
                    _tf.constant([0], _tf.int64), shape=[None],
                    name="w_row_new")
                w_col_new = _tf.placeholder_with_default(
                    _tf.constant([0], _tf.int64), shape=[None],
                    name="w_col_new")
                w_data_new = _tf.placeholder_with_default(
                    _tf.constant([1], _tf.float64), shape=[None],
                    name="w_data_new")

                # covariance matrix
                with _tf.name_scope("covariance_matrix"):
                    quadrants = list(
                        _itertools.product([-3, -2, -1, 0, 1, 2, 3],
                                           repeat=self.ndim))

                    # stacking covariance by quadrant
                    cov_circ = _tf.zeros([n_circ, 1], _tf.float64)
                    for quad in quadrants:
                        point = point_zero + _np.array(quad) \
                                * circulant_grid.grid_size \
                                * circulant_grid.step_size
                        k_tmp = self.cov_model.covariance_matrix(
                            circulant_coords, point)
                        cov_circ = cov_circ + k_tmp
                    cov_circ = SPICE.reshape_vec2grid(
                        _tf.squeeze(cov_circ), circulant_grid.grid_size)

                    # rolling
                    for d in range(self.ndim):
                        n_roll = (circulant_grid.grid_size[d] + 1) // 2
                        cov_circ = _tf.roll(cov_circ, -n_roll, axis=d)

                    # nugget
                    nugget = self.cov_model.variance.tf_val[-1]
                    total_var = 1 - nugget

                # spectrum
                with _tf.name_scope("spectrum"):
                    cov_comp = _tf.cast(cov_circ, _tf.complex128)

                    if self.ndim == 1:
                        eigvals = _tf.math.abs(_tf.signal.fft(cov_comp))
                    elif self.ndim == 2:
                        eigvals = _tf.math.abs(_tf.signal.fft2d(cov_comp))
                    elif self.ndim == 3:
                        eigvals = _tf.math.abs(_tf.signal.fft3d(cov_comp))
                    eigvals = eigvals \
                              * n_circ \
                              / _tf.reduce_sum(eigvals) \
                              * total_var + jitter_tf
                    eigvals = _tf.expand_dims(eigvals, axis=0)
                    eigvals_comp = _tf.cast(eigvals, _tf.complex128)

                # interpolation weights
                with _tf.name_scope("interpolation_weights"):
                    w_n_cols = _np.prod(circulant_grid.grid_size)
                    w_train = _tf.sparse.SparseTensor(
                        _tf.stack([w_row, w_col], axis=1), w_data,
                        dense_shape=[_tf.shape(x)[0], w_n_cols]
                    )
                    w_test = _tf.sparse.SparseTensor(
                        _tf.stack([w_row_new, w_col_new], axis=1), w_data_new,
                        dense_shape=[_tf.shape(x_new)[0], w_n_cols]
                    )

                # unconditional simulation
                with _tf.name_scope("unconditional_simulation"):
                    sim_shape = [n_circ, n_sim]

                    # iid simulation
                    sim_real = _tf.random.stateless_normal(
                        sim_shape, [seed, 0]
                    )
                    sim_imag = _tf.random.stateless_normal(
                        sim_shape, [seed, 1]
                    )
                    sim_comp = _tf.cast(sim_real, _tf.complex128) \
                               + _tf.cast(sim_imag, _tf.complex128) * 1.0j

                    sim_unc = SPICE.circular_sqrt(eigvals_comp, sim_comp,
                                                  circulant_grid.grid_size)
                    sim_unc = _tf.concat([
                        _tf.math.real(sim_unc),
                        _tf.math.imag(sim_unc)
                    ], axis=1)

                # alpha
                with _tf.name_scope("alpha"):
                    def matmul_fn(w1, w2, nug, b):
                        out_1 = _tf.sparse.sparse_dense_matmul(w2, b, True, False)
                        out_1 = self.circular_matmul(eigvals_comp, out_1,
                                                     circulant_grid.grid_size)
                        out_1 = _tf.sparse.sparse_dense_matmul(w1, out_1)
                        out_2 = nug * b
                        return out_1 + out_2

                    alpha = _tftools.conjugate_gradient(
                        lambda b: matmul_fn(w_train, w_train, nugget, b),
                        y_tf, max_iter=100
                    )
                    alpha_pred = _tf.sparse.sparse_dense_matmul(w_train, alpha,
                                                     True, False)
                    alpha_pred = SPICE.circular_matmul(eigvals_comp, alpha_pred,
                                                        circulant_grid.grid_size)
                    alpha_pred = _tf.Variable(lambda: alpha_pred,
                                              validate_shape=False)

                    alpha_sim = _tftools.conjugate_gradient_block(
                        lambda b: matmul_fn(w_train, w_train, nugget, b),
                        _tf.sparse.sparse_dense_matmul(w_train, sim_unc),
                        jitter=jitter_tf
                    )
                    alpha_sim = _tf.sparse.sparse_dense_matmul(w_train, alpha_sim,
                                                     True, False)
                    alpha_sim = SPICE.circular_matmul(eigvals_comp, alpha_sim,
                                                        circulant_grid.grid_size)

                # log-likelihood
                with _tf.name_scope("log_lik"):
                    fit = -0.5 * _tf.reduce_sum(alpha * y_tf, name="fit")

                    # Lanczos
                    # with _tf.device('/CPU:0'):
                    #     det = -0.5 * _tftools.determinant_lanczos(
                    #         lambda b: matmul_fn(w_train, w_train, nugget, b),
                    #         n_data, m=100, seed=seed, n=5
                    #     )

                    # samples
                    det1 = _tf.math.log(nugget) * n_data

                    prod = _tf.sparse.sparse_dense_matmul(w_train, sim_unc)
                    prod = _tf.matmul(prod, prod, True, False)
                    chol = _tf.linalg.cholesky(
                        prod + _tf.eye(2*n_sim, dtype=_tf.float64))
                    det2 = 2 * _tf.reduce_sum(
                        _tf.math.log(_tf.diag_part(chol)))
                    det = -0.5 * (det1 + det2)

                    const = - 0.5 * _tf.constant(n_data * _np.log(2 * _np.pi),
                                                 dtype=_tf.float64)
                    warp = _tf.reduce_sum(_tf.math.log(yd_tf))

                    log_lik = fit + det + const + warp

                # prediction
                with _tf.name_scope("Prediction"):

                    pred_mu = _tf.sparse.sparse_dense_matmul(w_test, alpha_pred,
                                                name="pred_mean")

                # conditional simulation
                with _tf.name_scope("conditional_simulation"):

                    y_sim = pred_mu + _tf.sparse.sparse_dense_matmul(
                        w_test, sim_unc - alpha_sim)

                    with _tf.name_scope("approximate_variance"):
                        sample_var = _tf.reduce_mean((y_sim - pred_mu)**2,
                                                     axis=1)
                        pred_var = sample_var + nugget

                    with _tf.name_scope("noise"):
                        def noise(mat):
                            rand = _tf.random.normal(
                                shape=_tf.shape(mat), dtype=_tf.float64,
                                stddev=_tf.sqrt(nugget)
                            )
                            return mat + rand

                        y_sim = _tf.cond(sim_with_noise,
                                         lambda: noise(y_sim),
                                         lambda: y_sim)

                self.tf_handles = {
                    "x": x,
                    # "alpha": alpha,
                    "y_tf": y_tf,
                    "yd_tf": yd_tf,
                    # "w_test": w_test,
                    "log_lik": log_lik,
                    "x_new": x_new,
                    "pred_mu": _tf.squeeze(pred_mu),
                    "pred_var": _tf.squeeze(pred_var),
                    "jitter": jitter_tf,
                    "y_sim": y_sim,
                    "n_sim": n_sim,
                    "seed": seed,
                    "sim_with_noise": sim_with_noise,
                    "w_row_new": w_row_new,
                    "w_col_new": w_col_new,
                    "w_data_new": w_data_new,
                    "w_row": w_row,
                    "w_col": w_col,
                    "w_data": w_data,
                    "init": _tf.compat.v1.global_variables_initializer()}

        self.graph.finalize()

    def covariance_matrix(self, x, y):
        raise NotImplementedError()

    @staticmethod
    def reshape_vec2grid(vec, grid_size):
        with _tf.name_scope("reshape_vec2grid"):
            if len(grid_size) == 1:
                return _tf.reshape(vec, grid_size)
            if len(grid_size) == 2:
                return _tf.transpose(
                    _tf.reshape(vec, _np.flip(grid_size))
                )
            if len(grid_size) == 3:
                return _tf.transpose(
                    _tf.reshape(vec, [grid_size[2], grid_size[1], grid_size[0]]),
                    perm=[2, 1, 0]
                )

    @staticmethod
    def reshape_mat2grid(mat, grid_size):
        with _tf.name_scope("reshape_mat2grid"):
            if len(grid_size) == 1:
                return _tf.reshape(mat, grid_size + [-1])
            if len(grid_size) == 2:
                return _tf.transpose(
                    _tf.reshape(mat, _np.flip(grid_size).tolist() + [-1]),
                    perm=[1, 0, 2]
                )
            if len(grid_size) == 3:
                return _tf.transpose(_tf.reshape(
                    mat, [grid_size[2], grid_size[1], grid_size[0], -1]),
                    perm=[2, 1, 0, 3]
                )

    @staticmethod
    def circular_matmul(spectrum, mat, grid_size):
        with _tf.name_scope("circular_matmul"):
            grid = SPICE.reshape_mat2grid(mat, grid_size)
            grid = _tf.cast(grid, _tf.complex128)
            last_dim = _tf.shape(grid)[-1]
            if len(grid_size) == 1:
                grid = _tf.transpose(grid, perm=[1, 0])
                grid = _tf.signal.fft(grid)
                grid = spectrum * grid
                grid = _tf.signal.ifft(grid)
                grid = _tf.transpose(grid, perm=[1, 0])
            if len(grid_size) == 2:
                grid = _tf.transpose(grid, perm=[2, 0, 1])
                grid = _tf.signal.fft2d(grid)
                grid = spectrum * grid
                grid = _tf.signal.ifft2d(grid)
                grid = _tf.transpose(grid, perm=[2, 1, 0])
            if len(grid_size) == 3:
                grid = _tf.transpose(grid, perm=[3, 0, 1, 2])
                grid = _tf.signal.fft3d(grid)
                grid = spectrum * grid
                grid = _tf.signal.ifft3d(grid)
                grid = _tf.transpose(grid, perm=[3, 2, 1, 0])
            grid = _tf.cast(grid, _tf.float64)
            return _tf.reshape(grid, [-1, last_dim])

    @staticmethod
    def circular_sqrt(spectrum, mat, grid_size):
        # mat must be of complex type
        n_circ = _np.prod(grid_size)
        with _tf.name_scope("circular_sqrt"):
            grid = SPICE.reshape_mat2grid(mat, grid_size)
            last_dim = _tf.shape(grid)[-1]
            if len(grid_size) == 1:
                grid = _tf.transpose(grid, perm=[1, 0])
                grid = grid * _tf.sqrt(spectrum * n_circ)
                grid = _tf.signal.ifft(grid)
                grid = _tf.transpose(grid, perm=[1, 0])
            elif len(grid_size) == 2:
                grid = _tf.transpose(grid, perm=[2, 0, 1])
                grid = grid * _tf.sqrt(spectrum * n_circ)
                grid = _tf.signal.ifft2d(grid)
                grid = _tf.transpose(grid, perm=[2, 1, 0])
            elif len(grid_size) == 3:
                grid = _tf.transpose(grid, perm=[3, 0, 1, 2])
                grid = grid * _tf.sqrt(spectrum * n_circ)
                grid = _tf.signal.ifft3d(grid)
                grid = _tf.transpose(grid, perm=[3, 2, 1, 0])
            grid = _tf.reshape(grid, [-1, last_dim])
        return grid

    def predict(self, newdata, perc=(0.025, 0.25, 0.5, 0.75, 0.975),
                name=None, verbose=True, batch_size=20000,
                n_sim=100, add_noise=False, seed=1234):

        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

        # tidying up
        if name is None:
            name = self.y_name
        x_new = newdata.coords

        # updating y
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)

        # interpolation matrix
        w_mat = self.circulant_grid.interpolation_weights(newdata)
        w_mat = _sp.csr_matrix(w_mat)

        # session and placeholders
        session = _tf.Session(graph=self.graph)
        feed = self.cov_model.feed_dict()
        feed.update({self.tf_handles["y_tf"]: _np.resize(y, (len(y), 1)),
                     self.tf_handles["jitter"]: self.cov_model.jitter,
                     self.tf_handles["n_sim"]: n_sim,
                     self.tf_handles["sim_with_noise"]: add_noise,
                     self.tf_handles["seed"]: seed})
        session.run(self.tf_handles["init"], feed_dict=feed)
        handles = [self.tf_handles["pred_mu"],
                   self.tf_handles["pred_var"]]
        if n_sim > 0:
            handles += [self.tf_handles["y_sim"]]

        # prediction in batches
        n_data = x_new.shape[0]
        n_batches = int(_np.ceil(n_data / batch_size))
        batch_id = [_np.arange(i * batch_size,
                               _np.minimum((i + 1) * batch_size, n_data))
                    for i in range(n_batches)]

        mu = _np.array([])
        var = _np.array([])
        y_sim = _np.empty([0, n_sim*2])
        for i in range(n_batches):
            if verbose:
                print("\rProcessing batch " + str(i + 1) + " of "
                      + str(n_batches) + "       ", sep="", end="")

            sub_mat = _sp.coo_matrix(w_mat[batch_id[i], :])
            feed.update({self.tf_handles["x_new"]: x_new[batch_id[i], :],
                         self.tf_handles["w_row_new"]: sub_mat.row,
                         self.tf_handles["w_col_new"]: sub_mat.col,
                         self.tf_handles["w_data_new"]: sub_mat.data})

            # TensorFlow
            out = session.run(handles, feed_dict=feed)

            # update
            mu = _np.concatenate([mu, out[0]])
            var = _np.concatenate([var, out[1]])
            if n_sim > 0:
                y_sim = _np.concatenate([y_sim, out[2]], axis=0)
        session.close()
        if verbose:
            print("\n")

        # output
        newdata.data[name + "_mean"] = mu
        newdata.data[name + "_variance"] = var
        for p in perc:
            newdata.data[name + "_p" + str(p)] = self.cov_model.warp_backward(
                _st.norm.ppf(p, loc=mu, scale=_np.sqrt(var)))

        # simulations
        n_digits = str(len(str(n_sim*2 - 1)))
        cols = [name + "_sim_" + ("{:0>" + n_digits + "d}").format(i)
                for i in range(n_sim*2)]
        for i in range(n_sim*2):
            newdata.data[cols[i]] = self.cov_model.warp_backward(
                y_sim[:, i])

    def cross_validation(self, partition=None, n_sim=10, seed=1234,
                         add_noise=False,
                         session=None):
        """
        Computes a cross-validation, given a partition and the current
        parameters.

        Parameters
        ----------
        partition : ndarray
            A vector of ints to partition the data. If none is provided, the
            data is partitioned randomly into 5 parts.
        n_sim : int
            Number of simulations to perform.
        seed : int
            Seed number for simulations.
        add_noise : bool
            Whether to include noise in the simulations.
        session :
            An active TensorFlow session. If omitted, one will be created.

        Returns
        -------
        y_pred : ndarray
            The model's predictions.
        """
        if session is None:
            with _tf.Session(graph=self.graph) as session:
                y_pred = self.cross_validation(partition, n_sim, seed,
                                               add_noise, session)
            return y_pred

        x = self.data.coords

        # updating y
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)
        y = _np.expand_dims(y, axis=1)
        yd = self.cov_model.warp_derivative(self.y)
        yd = _np.expand_dims(yd, axis=1)
        y_pred = _np.zeros([x.shape[0], n_sim*2])

        # interpolation matrix
        w_mat = self.circulant_grid.interpolation_weights(self.data)
        w_mat = _sp.csr_matrix(w_mat)

        # partition
        if partition is None:
            partition = _np.random.choice(_np.arange(5), x.shape[0])

        # session
        feed = self.cov_model.feed_dict()
        feed.update({self.tf_handles["jitter"]: self.cov_model.jitter,
                     self.tf_handles["n_sim"]: n_sim,
                     self.tf_handles["sim_with_noise"]: add_noise,
                     self.tf_handles["seed"]: seed
                     })
        run_opts = _tf.RunOptions(report_tensor_allocations_upon_oom=True)

        for idx in _np.unique(partition):
            keep = partition == idx

            w_train = _sp.coo_matrix(w_mat[~keep, :])
            w_test = _sp.coo_matrix(w_mat[keep, :])

            feed.update({
                self.tf_handles["y_tf"]: y[~keep],
                self.tf_handles["yd_tf"]: yd[~keep],
                self.tf_handles["x"]: x[~keep],
                self.tf_handles["x_new"]: x[keep],
                self.tf_handles["w_row"]: w_train.row,
                self.tf_handles["w_col"]: w_train.col,
                self.tf_handles["w_data"]: w_train.data,
                self.tf_handles["w_row_new"]: w_test.row,
                self.tf_handles["w_col_new"]: w_test.col,
                self.tf_handles["w_data_new"]: w_test.data
            })
            session.run(self.tf_handles["init"], feed_dict=feed)
            y_sim = session.run(
                self.tf_handles["y_sim"],
                feed_dict=feed,
                options=run_opts)
            y_pred[keep, :] = y_sim

        # output
        return _np.squeeze(y), y_pred
