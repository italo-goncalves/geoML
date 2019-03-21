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

__all__ = ["GP", "GPGrad", "GPClassif"]

import numpy as _np
import tensorflow as _tf
import scipy.stats as _st
import pandas as _pd
import copy as _copy

import geoml.genetic as _gen
import geoml.kernels as _ker
import geoml.warping as _warp
# import geoml.tftools as tftools
import geoml.data as _data


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
    def __init__(self, sp_data, variable, kernels, warping=(), interpolate=None):
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
        self.x = sp_data.coords
        self.y = sp_data.data[variable].values
        self.y_name = variable
        self._ndim = sp_data.ndim
        self.cov_model = _ker.CovarianceModelRegression(kernels, warping)
        if interpolate is None:
            self.interpolate = _np.repeat(False, len(self.y))
        else:
            self.interpolate = sp_data.data[interpolate].values
        self.training_log = None
        self.graph = _tf.Graph()

        # tidying up
        self.cov_model.auto_set_kernel_parameter_limits(sp_data)

        # y variable initialization is currently outside TensorFlow
        self.cov_model.warp_refresh(self.y)
        y = self.cov_model.warp_forward(self.y)

        # TensorFlow
        self.graph = _tf.Graph()
        with self.graph.as_default():
            with _tf.name_scope("GP"):
                # constants
                x = _tf.constant(self.x,
                                 dtype=_tf.float64,
                                 name="coords")
                interp = _tf.constant(self.interpolate,
                                      dtype=_tf.bool)

                # trainable parameters
                with _tf.name_scope("init_placeholders"):
                    self.cov_model.init_tf_placeholder()

                # pre-computations
                y_tf = _tf.placeholder(_tf.float64, shape=(len(y), 1),
                                       name="y_tf")
                yd_tf = _tf.placeholder(_tf.float64, shape=(len(y), 1),
                                        name="yd_tf")
                x_new_tf = _tf.placeholder_with_default(
                    _tf.zeros([1, self.x.shape[1]], dtype=_tf.float64),
                    shape=[None, self.x.shape[1]],
                    name="x_new_tf")

                # covariance matrix
                mat_k = self._covariance_matrix(x, interp)

                # alpha
                mat_chol, alpha = self._alpha(mat_k, y_tf)

                # log-likelihood
                log_lik = self._log_lik(mat_chol, alpha, y_tf, yd_tf)

                # prediction
                pred_mu, pred_var = self._predict(mat_chol, alpha, x, x_new_tf,
                                                  interp)

                self.tf_handles = {"L": mat_chol,
                                   "alpha": alpha,
                                   "y_tf": y_tf,
                                   "yd_tf": yd_tf,
                                   "K": mat_k,
                                   "log_lik": log_lik,
                                   "x_new_tf": x_new_tf,
                                   "pred_mu": pred_mu,
                                   "pred_var": pred_var,
                                   "init": _tf.global_variables_initializer()}

    def _covariance_matrix(self, x, interp, y=None):
        with _tf.name_scope("covariance_matrix"):
            if y is None:
                nugget = self.cov_model.variance.tf_val[-1]
                k = _tf.add(
                    self.cov_model.covariance_matrix(x, y),
                    self.cov_model.nugget.nugget_matrix(x, interp) * nugget)
            else:
                k = self.cov_model.covariance_matrix(x, y)
        return k

    @staticmethod
    def _alpha(k, y):
        with _tf.name_scope("alpha"):
            chol = _tf.linalg.cholesky(k)
            alpha = _tf.linalg.cholesky_solve(chol, y)
        return chol, alpha

    @staticmethod
    def _log_lik(chol, alpha, y, yd):
        with _tf.name_scope("log_lik"):
            n_data = y.shape[0].value
            log_lik = - 0.5 * _tf.reduce_sum(alpha * y, name="fit") \
                      - _tf.reduce_sum(_tf.log(_tf.diag_part(chol)),
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
                     self.tf_handles["yd_tf"]: _np.resize(yd, (len(y), 1))})
        session.run(self.tf_handles["init"], feed_dict=feed)
        log_lik = session.run(self.tf_handles["log_lik"], feed_dict=feed)
        return log_lik

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
                name=None, verbose=True, batch_size=20000):
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
            feed.update({self.tf_handles["y_tf"]: _np.resize(y, (len(y), 1)),
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

    def __str__(self):
        s = "A " + self.__class__.__name__ + " object\n\n"
        s += "Covariance model: " + self.cov_model.__str__()
        return s


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
        self.cov_model = _ker.CovarianceModelRegression(kernels, warping=[])
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

                # pre-computations
                y_tf = _tf.placeholder(_tf.float64, shape=(len(self.y), 1),
                                       name="y_tf")
                y_dir = _tf.placeholder(_tf.float64,
                                        shape=(len(self.y_dir), 1),
                                        name="y_dir")
                x_new_tf = _tf.placeholder_with_default(
                    _tf.zeros([1, self.x.shape[1]], dtype=_tf.float64),
                    shape=[None, self.x.shape[1]],
                    name="x_new_tf")

                # covariance matrix
                mat_k = self._covariance_matrix(x, interp, x_dir=x_dir,
                                                directions=directions)

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
                                   "init": _tf.global_variables_initializer()}

    #        with tf.Session(graph = self.graph) as session:
    #            self._refresh(session)

    def _covariance_matrix(self, x, interp, x_dir, directions, y=None):
        with _tf.name_scope("covariance_matrix"):
            if y is None:
                nugget = self.cov_model.variance.tf_val[-1]
                k_x = _tf.add(self.cov_model.covariance_matrix(x, y),
                              self.cov_model.nugget.nugget_matrix(x, interp)
                              * nugget)
                k_x_dir = self.cov_model.covariance_matrix_d1(x, x_dir,
                                                              directions)
                k_dir = self.cov_model.covariance_matrix_d2(x_dir, x_dir,
                                                            directions,
                                                            directions)
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
            self.tf_handles["y_dir"]: _np.resize(y_dir, (len(y_dir), 1))})
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

#    def _get_cov_mat(self):
#        with tf.Session(graph = self.graph) as session:
#            y = self.y
#            y_dir = self.y_dir
#            # session
#            feed = self.cov_model.feed_dict()
#            feed.update({self.tf_handles["y_tf"] : np.resize(y, (len(y), 1)),
#                         self.tf_handles["y_dir"] : np.resize(y_dir, (len(y_dir), 1))})
#            session.run(self.tf_handles["init"], feed_dict = feed)
#            K = session.run(self.tf_handles["K"], feed_dict = feed)
#        return K


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
    def __init__(self, sp_data, var_1, var_2, kernels, dir_data=None):
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
        lat1 = _np.ones([n_data, n_latent]) * (-1 / n_latent)
        lat2 = _np.ones([n_data, n_latent]) * (-1 / n_latent)
        idx1 = sp_data.data[var_1].map(labels_dict).values
        idx2 = sp_data.data[var_2].map(labels_dict).values
        for i in range(n_data):
            lat1[i, idx1[i]] = 1
            lat2[i, idx2[i]] = 1
        lat = 0.5 * lat1 + 0.5 * lat2
        interpolate = _np.apply_along_axis(
            lambda x: x == (1 - 1 / n_latent) / 2,
            axis=0, arr=lat)
        self.latent = _pd.DataFrame(lat, columns=labels)

        # latent variable models
        self.GPs = []
        n_dim = self.x.shape[1]
        if n_dim == 1:
            temp_data = _data.Points1D(self.x, _pd.DataFrame(
                self.latent.values, columns=labels))
        elif n_dim == 2:
            temp_data = _data.Points2D(self.x, _pd.DataFrame(
                self.latent.values, columns=labels))
        elif n_dim == 3:
            temp_data = _data.Points3D(self.x, _pd.DataFrame(
                self.latent.values, columns=labels))
        for i in range(n_latent):
            temp_data.data["boundary"] = interpolate[:, i]
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
                n_samples=1000):
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

        """
        if self.ndim != newdata.ndim:
            raise ValueError("dimension of newdata is incompatible with model")

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
                                batch_size=batch_size)
            mu[:, i] = temp_data.data["out_mean"].values
            var[:, i] = temp_data.data["out_variance"].values
        std = _np.sqrt(var)

        # probabilities, indicators, and entropy
        if verbose:
            print("Calculating probabilities:")
        prob, ind, entropy = self._calc_prob(mu, std, n_samples)

        # output
        idx = self.latent.columns
        newdata.data[name] = idx[_np.argmax(prob, axis=1)]
        for i in range(n_classes):
            newdata.data[name + "_" + idx[i] + "_prob"] = prob[:, i]
        for i in range(n_classes):
            newdata.data[name + "_" + idx[i] + "_ind"] = ind[:, i]
        newdata.data[name + "_entropy"] = entropy

    def __str__(self):
        s = ""
        for model in self.GPs:
            s += "\nModel for " + model.y_name + ":\n"
            s += str(model)
        return s

    def _calc_prob(self, mu, std, n_samples=1000):
        g = _tf.Graph()
        with g.as_default():
            mu = _tf.constant(mu, dtype=_tf.float64)
            std = _tf.constant(std, dtype=_tf.float64)
            self_lat = _tf.constant(self.latent.values, dtype=_tf.float64)
            s = _tf.shape(self_lat)  # n_data, n_lat
            max_entropy = _tf.log(_tf.cast(s[1], _tf.float64))

            # same random numbers across test points for smooth plotting
            rnd = _tf.random.normal(shape=[n_samples, s[1]], dtype=_tf.float64)

            # loop
            def loop_prob(x):
                mu_i = _tf.reshape(x[0], [1, s[1]])
                mu_i = _tf.tile(mu_i, [n_samples, 1])
                std_i = _tf.reshape(x[1], [1, s[1]])
                std_i = _tf.tile(std_i, [n_samples, 1])
                lat = rnd * std_i + mu_i
                idx = _tf.argmax(lat, axis=1, output_type=_tf.int32)
                count = _tf.bincount(idx, minlength=s[1], dtype=_tf.float64) \
                        + 1.0
                return count / _tf.reduce_sum(count)

            prob = _tf.map_fn(loop_prob, [mu, std], dtype=_tf.float64,
                              parallel_iterations=1000)

            # indicators
            def loop_ind(x):
                x = _tf.log(x)
                x_sort = _tf.contrib.framework.sort(x, direction="DESCENDING")
                x = x - _tf.reduce_mean(x_sort[0:2])
                return x

            ind = _tf.map_fn(loop_ind, prob, dtype=_tf.float64,
                             parallel_iterations=1000)

            # entropy
            entropy = - _tf.reduce_sum(prob * _tf.log(prob), axis=1)
            entropy = entropy / max_entropy

        with _tf.Session(graph=g) as session:
            session.run(_tf.global_variables_initializer())
            prob, ind, entropy = session.run([prob, ind, entropy])

        return prob, ind, entropy
