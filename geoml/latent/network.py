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
import numpy as np

import geoml.parameter as _gpr
import geoml.math.tf as _tftools
import geoml.kernels as _kr
import geoml.transform as _tr
import geoml.math.interpolate as _gint
import geoml.data as _data
import geoml.stats.random as _rnd

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp
import contextlib as _contextlib
import warnings as _warnings
from scipy import special as _special

_tfd = _tfp.distributions


def _gamma_mode_one(concentration, mode=1.0):
    """A Gamma prior peaking at `mode`, for a MAP penalty on a range.

    MAP pulls toward the density's *mode*, not its mean, so the mode is what
    gets centered: `Gamma(c, (c - 1) / mode)` peaks exactly at `mode`, its
    log-density falls to minus infinity as the value approaches zero (a range
    collapsing to nothing is the failure this discourages most), and decays
    linearly on the long side, where a large range merely says the field is
    smooth. `concentration` must exceed one or the peak sits at zero.
    """
    if concentration <= 1.0:
        raise ValueError(
            "concentration must be greater than 1 for the prior to peak "
            "away from zero, got %r" % (concentration,))
    return _tfd.Gamma(
        concentration=_tf.constant(concentration, _tf.float64),
        rate=_tf.constant((concentration - 1.0) / mode, _tf.float64))


class _ColumnwiseDirichlet:
    """A Dirichlet over each column of a `[n_parents, size]` weight matrix.

    `UnitColumnSumParameter` keeps every *column* on the simplex, while
    `tfd.Dirichlet` reads the *last* axis as the event -- so the value is
    transposed on the way in, giving one log-density per column, which
    `RealParameter.log_prior` then sums.
    """

    def __init__(self, concentration):
        self._dirichlet = _tfd.Dirichlet(concentration)

    def log_prob(self, value):
        return self._dirichlet.log_prob(_tf.transpose(value))


# Which rule draws the posterior simulations. Set through `simulation_rule`
# by the model around its prediction call, never directly: the choice lives
# in `GPOptions`, and threading it through every node's `predict` signature
# would touch each of them to serve three draw sites.
_QMC_SIMULATIONS = False


@_contextlib.contextmanager
def simulation_rule(qmc):
    """Chooses how the posterior simulations are drawn while active."""
    global _QMC_SIMULATIONS
    previous = _QMC_SIMULATIONS
    _QMC_SIMULATIONS = bool(qmc)
    try:
        yield
    finally:
        _QMC_SIMULATIONS = previous


# How a deep network's experts see each other's inducing sets. Set through
# `propagation_rule` by the model around training and prediction, never
# directly: the choice lives in `GPOptions.expert_propagation`, and it is
# read at trace time by `BasicGP.refresh`.
_EXPERT_PROPAGATION = "consensus"


@_contextlib.contextmanager
def propagation_rule(rule):
    """Chooses how experts propagate their inducing sets while active."""
    global _EXPERT_PROPAGATION
    previous = _EXPERT_PROPAGATION
    _EXPERT_PROPAGATION = rule
    try:
        yield
    finally:
        _EXPERT_PROPAGATION = previous


def _simulation_normals(shape, seed):
    """Standard normals shaped `[size, n, n_sim]` for the posterior draws.

    Monte Carlo is a stateless draw. Under `simulation_rule(True)` the same
    numbers come instead from a seeded-scramble Sobol sequence pushed through
    the normal quantile: each simulation is one point of a `size * n`-
    dimensional sequence, so the ensemble covers the posterior evenly rather
    than by chance. `shape` and `seed` are Python values at trace time, which
    is what lets the points be computed once and embedded as a constant.
    Either way the numbers are fixed by the seed, so a value does not depend
    on the batch that computed it.
    """
    if not _QMC_SIMULATIONS:
        return _tf.random.stateless_normal(
            shape=shape, seed=seed, dtype=_tf.float64)

    size, n, n_sim = (int(s) for s in shape)
    rng = _np.random.default_rng([abs(int(s)) for s in seed])
    with _warnings.catch_warnings():
        # scipy warns unless n_sim is a power of two; the balance it asks
        # for helps but is not required
        _warnings.simplefilter("ignore")
        points = _rnd.sobol_engine(size * n, rng).random(n_sim)
    normals = _special.ndtri(_np.clip(points, 1e-6, 1 - 1e-6))
    return _tf.constant(
        normals.reshape([n_sim, size, n]).transpose([1, 2, 0]), _tf.float64)


def _graph_state(node):
    """
    The attributes a node holds as tensors, which a traced refresh must return.

    An attribute written while a `tf.function` is tracing keeps a symbolic
    tensor, which is unusable once the trace is over. Reading them off the node
    rather than listing them per class means a new node needs nothing new here.
    """
    state = {}
    for name, value in vars(node).items():
        if name.startswith("_"):
            continue
        if isinstance(value, _tf.Tensor):
            state[name] = value
        elif (isinstance(value, (tuple, list)) and len(value) > 0
                and all(isinstance(v, _tf.Tensor) for v in value)):
            state[name] = tuple(value)
    return state


def refresh_cached(network, jitter=1e-6):
    """
    Refreshes a network once and snapshots it for prediction.

    `refresh` is pure arithmetic over parameters that do not move during
    prediction, but running it eagerly pays Python overhead for each of the
    K x K covariance blocks a multi-expert network builds -- at 32 experts that
    is most of a `predict` call. Tracing it collapses those into one graph
    call. The trace is kept on the network, so predicting again does not
    rebuild it, and it reads the parameters live, so it also follows further
    training.

    Parameters
    ----------
    network
        The output node of a latent network.
    jitter : float
        Small value added to the covariance matrices for numerical stability.
    """
    # the propagation rule is a Python-level branch inside `refresh`, so it
    # is baked into the trace and must key the cache with the jitter
    key = (jitter, _EXPERT_PROPAGATION)
    cached = network._refresh_graph
    if cached is None or cached[0] != key:
        # fixed once, so that the values coming back keep lining up with the
        # nodes they belong to
        nodes = list(set(network.get_unique_parents()) | {network})

        def traced():
            network.refresh(jitter)
            return [_graph_state(node) for node in nodes]

        cached = (key, _tf.function(traced), nodes)
        network._refresh_graph = cached

    _, traced_refresh, nodes = cached
    for node, state in zip(nodes, traced_refresh()):
        for name, value in state.items():
            setattr(node, name, value)

    for node in nodes:
        node.cache_prediction_state()


class NodeIncompatibilityError(Exception):
    """Exception raised for incompatibilities between a node and its parents/children."""
    pass


class BrokenPropagationError(NodeIncompatibilityError):
    """Exception raised when inducing points can't be propagated through nodes."""
    pass


class SizeIncompatibilityError(NodeIncompatibilityError):
    """Exception raised for incompatibilities in the number of latent variables in nodes."""
    pass


class _LatentVariable(_gpr.Parametric):
    def __init__(self):
        super().__init__()
        self._size = 0

        # These attributes must be defined by subclasses. The `root` is a reference to the object's root
        # traced along the tree. Nodes whose parents have different inducing point sets do not have a
        # traceable root.
        self.children = []
        self.root = None
        self.propagates_inducing_points = None

        # Filled in by `_set_name`, once the node is wired to its neighbors.
        self.name = None

        # The traced refresh built by `refresh_cached`, kept so that repeated
        # predictions reuse it instead of tracing again. Holds (jitter, fn).
        self._refresh_graph = None

        # These are TensorFlow attributes, defined at graph execution time
        self.inducing_points = None
        self.inducing_points_variance = None

        # Non-trainable Variables holding a snapshot of the prediction state, so
        # a cached (tf.function) prediction graph reads current values instead of
        # baking them in at tracing time. Keyed by name, created on first use.
        self._state_vars = {}

    def _summary_line(self):
        name = self.name or self.__class__.__name__
        if not name.startswith(self.__class__.__name__):
            # a name of the user's choosing says nothing about the node's type
            name = "%s '%s'" % (self.__class__.__name__, name)
        return "%s (size %d)" % (name, self.size)

    def __repr__(self):
        return _gpr.describe(self, size=self.size)

    def _connected_nodes(self):
        """
        Every other node reachable from this one, in either direction.

        Walking upwards alone is not enough to name a node: a new node's
        siblings are not among its ancestors. They are reachable through the
        `children` list every node keeps of the nodes built on top of it, which
        is what this follows in the other direction.
        """
        found, stack = {}, [self]
        while len(stack) > 0:
            node = stack.pop()
            if id(node) in found:
                continue
            found[id(node)] = node
            stack.extend(node.children)
            stack.extend(node.get_unique_parents())
        return [node for node in found.values() if node is not self]

    def _set_name(self, name):
        """
        Names the node. Called once its parents and children are wired.

        A node left unnamed takes the first `Class_k` that no node it is
        connected to is using, so the branches of a network can be told apart
        without the user naming anything. Two subnetworks built separately and
        joined only later are the exception — while they are being numbered
        they cannot see each other, so they may repeat a name, which `get_node`
        reports if it is ever asked for one.
        """
        if name is None:
            taken = {node.name for node in self._connected_nodes()}
            index = 1
            while "%s_%d" % (self.__class__.__name__, index) in taken:
                index += 1
            name = "%s_%d" % (self.__class__.__name__, index)
        self.name = name

    def to_dot(self, legend=True, rankdir="BT"):
        """
        Writes this node and everything feeding it as a Graphviz diagram.

        See `geoml.graphviz.to_dot`, which draws a whole model when given one.
        """
        # imported here because that module reads this one
        import geoml.viz.graphviz as _gv
        return _gv.to_dot(self, legend=legend, rankdir=rankdir)

    def get_node(self, name):
        """
        Finds a node by name, among this node and everything feeding it.

        Parameters
        ----------
        name : str
            The node's name, as it appears in `str(network)`.

        Returns
        -------
        node
            The node with that name.
        """
        nodes = [self] + self.get_unique_parents()
        found = [node for node in nodes if node.name == name]

        if len(found) > 1:
            raise KeyError(
                "%d nodes are named %r; name them explicitly to tell them "
                "apart" % (len(found), name))
        if len(found) == 0:
            raise KeyError(
                "no node named %r; found %s"
                % (name, ", ".join(sorted(node.name for node in nodes))))
        return found[0]

    @property
    def size(self):
        return self._size

    # @property
    # def is_deterministic(self):
    #     return self._is_deterministic

    def set_parameter_limits(self, data):
        pass

    def refresh(self, jitter=1e-6):
        """
        Updates the model's internal state.

        If called within TensorFlow's eager mode, will allow inspection of the internal tensors.

        Parameters
        ----------
        jitter : float
            Small value added to the covariance matrices for numerical stability.
        """
        pass

    def _state_var(self, name, value):
        """
        Store `value` in a non-trainable tf.Variable and return it.

        The Variable is created on first use (the shape is unknown when the node
        is built) and reassigned afterwards. A cached prediction graph that reads
        the returned Variable sees the value written by the latest call, so the
        posterior can be refreshed once per prediction rather than per batch.

        The Variable takes the shape of the first value it receives. It must not
        be left shapeless: a graph reading a shapeless Variable gets a tensor of
        unknown rank, which spreads through the whole prediction and breaks any
        operation that needs a static rank (`tf.nn.softmax` on a given axis, for
        one). These values are sized by the network's structure -- the number of
        inducing points and the node's size -- so they do not change from one
        prediction to the next.
        """
        var = self._state_vars.get(name)
        if var is None:
            var = _tf.Variable(value, dtype=_tf.float64, trainable=False)
            self._state_vars[name] = var
        else:
            var.assign(value)
        return var

    def _cache_tuple(self, name, values):
        return tuple(self._state_var(name + "_" + str(i), v)
                     for i, v in enumerate(values))

    def cache_prediction_state(self):
        """
        Snapshot the propagated state into Variables (see `_state_var`).

        Called once per prediction (after `refresh`) for every node in the
        network. Subclasses holding additional prediction state extend this.
        """
        if self.inducing_points is not None:
            self.inducing_points = self._cache_tuple(
                "inducing_points", self.inducing_points)
        if self.inducing_points_variance is not None:
            self.inducing_points_variance = self._cache_tuple(
                "inducing_points_variance", self.inducing_points_variance)

    def get_unique_parents(self):
        raise NotImplementedError

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        """
        Prediction on the previous node's latent variables. If `n_sim=0` only the mean and variance are returned.

        Parameters
        ----------
        x : Tensor
            Mean of the input.
        x_var : Tensor
            Variance of the input.
        n_sim : int
            Number of simulations to draw.
        seed : tuple
            A set of two seeds for the random number generator.

        Returns
        -------
        mu
            Mean of the output.
        var
            Variance of the output.
        sims
            A set of simulations generated from the predictive distribution.
        explained_var
            Amount of variance "explained away" by conditioning on the inducing points.
        influence
            Fraction of the full variance that the model is able to sustain at a given position. Increases closer to
            the inducing points.
        """
        raise NotImplementedError

    def predict_directions(self, x, dir_x, step=1e-3):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError

    def propagate(self, x, x_var=None):
        """
        Propagates mean and variance to the next node.

        Parameters
        ----------
        x : Tensor
            Mean of the input.
        x_var : Tensor
            Variance of the input.

        Returns
        -------
        mu
            Mean of the output.
        var
            Variance of the output.
        """
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
    """
    Root latent variable.

    A root latent variable node processes an input, passing it along to other nodes as a Gaussian random variable.
    """
    def __init__(self, name=None):
        super().__init__()
        self.root = self
        self.propagates_inducing_points = True
        self._n_experts = None
        self._set_name(name)

    def get_unique_parents(self):
        return []

    def get_root_inducing_points(self):
        return NotImplementedError

    @property
    def n_experts(self):
        return self._n_experts


class _FunctionalLatentVariable(_LatentVariable):
    """
    Functional latent variable.

    A functional latent variable node applies a function to its input, returning a new random variable
    that may or not be Gaussian.

    """
    def __init__(self, parent, name=None):
        """
        Initializer for _FunctionalLatentVariable.

        Parameters
        ----------
        parent
            Parent node.
        name : str
            A name for this node.
        """
        super().__init__()
        self.parent = self._register(parent)
        parent.children.append(self)
        self.root = parent.root
        self.propagates_inducing_points = self.parent.propagates_inducing_points
        self._set_name(name)

    def get_unique_parents(self):
        return [self.parent] + self.parent.get_unique_parents()

    def propagate(self, x, x_var=None):
        mu, var = self.predict(x, x_var, n_sim=0)
        mu = _tf.transpose(mu[:, :, 0])
        var = _tf.transpose(var)
        return mu, var

    def set_parameter_limits(self, data):
        self.parent.set_parameter_limits(data)

    def refresh(self, jitter=1e-6):
        self.parent.refresh(jitter)


class _Operation(_LatentVariable):
    """
    Operation node.

    An operation node combines multiple latent variables in some form (sum, linear combination, concatenation, etc.).

    """
    def __init__(self, *latent_variables, name=None):
        super().__init__()
        self.parents = list(latent_variables)

        self.same_root = all(node.root is latent_variables[0].root for node in latent_variables)
        if self.same_root:
            self.root = latent_variables[0].root

        for node in latent_variables:
            self._register(node)
            node.children.append(self)

        self._set_name(name)

    def get_unique_parents(self):
        all_parents = self.parents.copy()
        for p in self.parents:
            all_parents.extend(p.get_unique_parents())
        return list(set(all_parents))

    def _common_size(self):
        """The parents' size, which the combining nodes require to be shared."""
        sizes = [p.size for p in self.parents]
        if not all(s == sizes[0] for s in sizes):
            raise SizeIncompatibilityError(
                "%s: all parents must have the same size. Found %s."
                % (self.name, ", ".join("%s (size %d)" % (p.name, p.size)
                                        for p in self.parents)))
        return sizes[0]

    def set_parameter_limits(self, data):
        for p in self.parents:
            p.set_parameter_limits(data)


class _GPNode(_FunctionalLatentVariable):
    def __init__(self, parent, name=None):
        super().__init__(parent, name=name)
        if not self.propagates_inducing_points:
            raise BrokenPropagationError(
                '%s: GP nodes require their parent to propagate inducing '
                'points, and %s does not.' % (self.name, parent.name))

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        """
        Prediction on the previous node's latent variables. If `n_sim=0` only the mean and variance are returned.

        Parameters
        ----------
        x : Tensor
            Mean of the input.
        x_var : Tensor
            Variance of the input.
        n_sim : int
            Number of simulations to draw.
        seed : tuple
            A set of two seeds for the random number generator.

        Returns
        -------
        mu
            Mean of the output.
        var
            Variance of the output.
        sims
            A set of simulations generated from the predictive distribution.
        explained_var
            Amount of variance "explained away" by conditioning on the inducing points.
        influence
            Fraction of the full variance that the model is able to sustain at a given position. Increases closer to
            the inducing points.
        """
        with _tf.name_scope("gp_prediction"):
            x, x_var = self.parent.propagate(x, x_var)

            if n_sim > 0:
                return self.interpolate(x, x_var, n_sim, seed)

            else:
                return self.interpolate(x, x_var, n_sim=0)

    def interpolate(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        return NotImplementedError

    @staticmethod
    def get_expert_weights(variances):
        # variances is [n_experts, ...]
        explained_var = 1 - variances

        weights = (explained_var / (variances + 1e-6)) + 1e-6
        weights = weights / _tf.reduce_sum(weights, axis=0, keepdims=True)

        return weights


class BasicInput(_RootLatentVariable):
    """
    Basic input node.

    Converts a deterministic input (usually spatial coordinates) into Gaussian latent variables with zero variance,
    after applying a transform for normalization. Also defines the inducing points that will be propagated to other
    nodes.

    """
    def __init__(self, inducing_points, transform=_tr.Identity(),
                 fix_transform=False,
                 center=False, name=None):
        """
        Initializer for BasicInput.

        Parameters
        ----------
        inducing_points
            A `PointData` object, or a list of these objects.
        transform
            An object from the `transform` module for normalization.
        fix_transform : bool
            Whether to fix the transform parameters to prevent them from changing during training.
        center : bool
            Whether to center the data, based on the inducing points' bounding box.
        name : str
            A name for this node, shown in the printed network and accepted by
            `get_node`. Numbered automatically if omitted.
        """
        super().__init__(name=name)

        if not isinstance(inducing_points, (list, tuple)):
            inducing_points = (inducing_points, )
        self._n_experts = len(inducing_points)

        test_point = _np.ones([1, inducing_points[0].n_dim], dtype=_np.float64)
        test_point = transform(test_point)
        self._size = test_point.shape[1]

        all_coords = _np.concatenate([ip.coordinates for ip in inducing_points])

        self.bounding_box = _data.BoundingBox.from_array(_data.bounding_box(all_coords)[0])

        self.transform = self._register(transform)
        if fix_transform:
            for p in self.transform.all_parameters:
                p.fix()
        self.transform.set_limits(_data.PointData.from_array(all_coords))

        self.n_ip = tuple(ip.coordinates.shape[0] for ip in inducing_points)

        self.base_inducing_points = tuple(_tf.constant(ip.coordinates, dtype=_tf.float64) for ip in inducing_points)
        self.inducing_points_variance = tuple(_tf.zeros([n, self.size], _tf.float64) for n in self.n_ip)

        # self.center = _np.zeros_like(transform(self.bounding_box.max.astype(_np.float64)))
        self.center = _np.zeros_like(self.bounding_box.max.astype(_np.float64))
        if center:
            self.center = 0.5 * (self.bounding_box.min + self.bounding_box.max)

    def get_root_inducing_points(self):
        return self.base_inducing_points, self.inducing_points_variance

    def refresh(self, jitter=1e-6):
        with _tf.name_scope("basic_input_refresh"):
            self.transform.refresh()
            self.inducing_points = tuple(self.transform(ip - self.center) for ip in self.base_inducing_points)

    def propagate(self, x, x_var=None):
        x_tr = self.transform(x - self.center)
        return x_tr, _tf.zeros_like(x_tr)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def set_parameter_limits(self, data):
        self.transform.set_limits(data)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        x_tr = _tf.transpose(self.transform(x - self.center))
        var = _tf.zeros_like(x_tr)
        if n_sim > 0:
            sims = _tf.tile(x_tr[:, :, None], [1, 1, n_sim])
            return x_tr[:, :, None], var, sims, \
                   _tf.zeros_like(var), _tf.zeros_like(var)
        else:
            return x_tr[:, :, None], var

    def predict_directions(self, x, dir_x, step=1e-3):
        x_plus = self.transform(x - self.center + dir_x*step/2)
        x_minus = self.transform(x - self.center - dir_x * step / 2)

        mu = _tf.transpose((x_plus - x_minus) / step)
        return mu[:, :, None], _tf.zeros_like(mu), _tf.zeros_like(mu)


class Stack(_Operation):
    """
    Latent variable stacking.

    Consolidates a list of latent variables into a single object.
    """
    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        self._size = sum([p.size for p in self.parents])

    def propagate(self, x, x_var=None):
        means, variances = [], []
        for lat in self.parents:
            m, v = lat.propagate(x, x_var)
            means.append(m)
            variances.append(v)

        mean = _tf.concat(means, axis=1)
        var = _tf.concat(variances, axis=1)
        return mean, var

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)

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


class Concatenate(Stack):
    """
    Latent variable concatenation.

    Consolidates a list of latent variables into a single object. This operation requires all its parent nodes to
    be able to propagate inducing points.
    """
    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        if self.same_root:
            self.propagates_inducing_points = True

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)
        self.inducing_points = tuple(_tf.concat(
            [lat.inducing_points[i] for lat in self.parents],
            axis=1) for i in range(self.root.n_experts))
        self.inducing_points_variance = tuple(_tf.concat(
            [lat.inducing_points_variance[i] for lat in self.parents],
            axis=1) for i in range(self.root.n_experts))


class BasicGP(_GPNode):
    """
    Standard Gaussian process node.

    In this module, GP nodes are able to work with inputs that may be
    Gaussian, having an associated variance. This variance is integrated by considering it as a squared range and
    applying the non-stationary covariance.
    """
    def __init__(self, parent, size=1, kernel=_kr.Gaussian(),
                 fix_range=False, isotropic=False, range_prior=2.0,
                 name=None):
        """
        Initializer for BasicGP.

        Parameters
        ----------
        parent
            Parent node.
        size
            Number of output latent variables
        kernel
            The kernel to use for the covariance matrices.
        fix_range : bool
            Whether to force a unit range for all input dimensions.
        isotropic : bool
            If `True`, forces the same range for all input dimensions.
        range_prior : float, optional
            Strength of the Gamma prior that regularizes the ranges, which
            stay point estimates -- the prior's log-density joins the
            training objective. It peaks at 1, the natural scale of the
            whitened space every node works in, falls hard as a range
            collapses toward zero and gently as it grows. Larger values
            hold on tighter; `None` removes it, leaving the ranges to the
            data alone as in versions before 0.6.5.
        name : str
            A name for this node, shown in the printed network and accepted by
            `get_node`. Numbered automatically if omitted.
        """
        super().__init__(parent, name=name)
        self._size = size
        self.kernel = self._register(kernel)
        self.range_prior = range_prior

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

        self.fix_range = fix_range
        self.isotropic = isotropic
        self._set_parameters()

    def _set_parameters(self):
        for i, n in enumerate(self.root.n_ip):
            self._add_parameter(
                f"alpha_white_{i}",
                _gpr.RealParameter(
                    _rnd.rng().normal(
                        scale=1e-3,
                        size=[self.size, n, 1]
                    ),
                    _np.zeros([self.size, n, 1]) - 10,
                    _np.zeros([self.size, n, 1]) + 10
                ))
            self._add_parameter(
                f"delta_{i}",
                _gpr.PositiveParameter(
                    _np.ones([self.size, n]),
                    _np.ones([self.size, n]) * 1e-6,
                    _np.ones([self.size, n]) * 1e2
                ))
            self._add_parameter(
                f"bias_{i}",
                # A point estimate, deliberately: no KL prices it and none is
                # needed -- one bounded scalar per expert, the level the data
                # sets. `cross_validate` still re-initializes it along with
                # the variational state, because it encodes the data.
                _gpr.RealParameter(0, -5, 5))

        if self.isotropic:
            self._add_parameter(
                "ranges",
                _gpr.PositiveParameter(
                    _np.ones([1, 1, 1]),
                    _np.ones([1, 1, 1]) * 1e-6,
                    _np.ones([1, 1, 1]) * 10,
                    fixed=self.fix_range
                )
            )
        else:
            self._add_parameter(
                "ranges",
                _gpr.PositiveParameter(
                    _np.ones([1, 1, self.parent.size]),
                    _np.ones([1, 1, self.parent.size]) * 1e-6,
                    _np.ones([1, 1, self.parent.size]) * 10,
                    fixed=self.fix_range
                )
            )
        if self.range_prior is not None:
            self.parameters["ranges"].prior = _gamma_mode_one(self.range_prior)

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

            # output
            cov = cov * norm
            return cov

    def refresh(self, jitter=1e-6):
        with _tf.name_scope("basic_refresh"):
            self.parent.refresh(jitter)

            # prior
            # ip = self.parent.inducing_points
            # ip_var = self.parent.inducing_points_variance

            eye = tuple(_tf.eye(n, dtype=_tf.float64) for n in self.root.n_ip)

            cov = tuple(
                    self.covariance_matrix(ip, ip, ip_var, ip_var) + e * jitter
                    for ip, ip_var, e in zip(self.parent.inducing_points, self.parent.inducing_points_variance, eye)
            )
            chol = tuple(_tf.linalg.cholesky(mat) for mat in cov)
            cov_inv = tuple(_tf.linalg.cholesky_solve(mat, e) for mat, e in zip(chol, eye))

            self.cov = cov
            self.cov_chol = chol
            self.cov_inv = cov_inv

            # posterior
            eye = tuple(_tf.tile(e[None, :, :], [self.size, 1, 1]) for e in eye)
            delta = tuple(self.parameters[f"delta_{i}"].get_value() for i in range(self.root.n_experts))
            delta_diag = tuple(_tf.linalg.diag(d) for d in delta)
            self.cov_smooth = tuple(mat[None, :, :] + d for mat, d in zip(self.cov, delta_diag))
            self.cov_smooth_chol = tuple(
                _tf.linalg.cholesky(mat + e * jitter)
                for mat, e in zip(self.cov_smooth, eye)
            )
            self.cov_smooth_inv = tuple(
                _tf.linalg.cholesky_solve(mat, e)
                for mat, e in zip(self.cov_smooth_chol, eye)
            )
            self.chol_r = tuple(
                _tf.linalg.cholesky(m1[None, :, :] - m2 + e * jitter)
                for m1, m2, e in zip(self.cov_inv, self.cov_smooth_inv, eye)
            )

            # inducing points
            alpha_white = tuple(self.parameters[f"alpha_white_{i}"].get_value() for i in range(self.root.n_experts))
            means = tuple(
                _tf.einsum("ab,sbc->sac", mat, vec)
                for mat, vec in zip(self.cov_chol, alpha_white)
            )
            self.alpha = tuple(
                _tf.einsum("ab,sbc->sac", mat, vec)
                for mat, vec in zip(self.cov_inv, means)
            )

            # inducing points, for whatever is built on top of this node.
            # Under the default rule every expert's set is predicted from
            # every other and combined by precision weighting -- the one
            # quadratic step in the network; with a terminal node it is pure
            # waste, since nothing ever reads the result. Under
            # `GPOptions(expert_propagation="independent")` each expert
            # speaks for its own set alone: duplicated points in overlapping
            # sets are then free to disagree (measured at several latent
            # standard deviations), and the data-side weighting in
            # `interpolate` arbitrates. That trades the consensus for O(K)
            # cost -- measured 6.3x training and 8x prediction at 40 experts,
            # with quality within a few percent either way.
            if len(self.children) > 0:
                bias = [self.parameters[f'bias_{i}'].get_value() for i in range(self.root.n_experts)]

                self.inducing_points = []
                self.inducing_points_variance = []
                if _EXPERT_PROPAGATION == "independent":
                    for i in range(self.root.n_experts):
                        ip_i = self.parent.inducing_points[i]
                        ipv_i = self.parent.inducing_points_variance[i]
                        cov = self.covariance_matrix(ip_i, ip_i, ipv_i, ipv_i)
                        mean = _tf.einsum(
                            "ab,sbc->sac", cov, self.alpha[i]) + bias[i]
                        pred_var = 1.0 - _tf.reduce_sum(
                            _tf.einsum("ab,sbc->sac", cov,
                                       self.cov_smooth_inv[i])
                            * cov[None, :, :],
                            axis=2, keepdims=False
                        )
                        self.inducing_points.append(
                            _tf.transpose(mean[:, :, 0]))
                        self.inducing_points_variance.append(
                            _tf.transpose(pred_var))
                else:
                    for i in range(self.root.n_experts):
                        ip_i = self.parent.inducing_points[i]
                        ipv_i = self.parent.inducing_points_variance[i]
                        means = []
                        pred_vars = []
                        for j in range(self.root.n_experts):
                            ip_j = self.parent.inducing_points[j]
                            ipv_j = self.parent.inducing_points_variance[j]
                            cov = self.covariance_matrix(ip_i, ip_j, ipv_i, ipv_j)
                            means.append(_tf.einsum("ab,sbc->sac", cov, self.alpha[j]) + bias[j])
                            pred_vars.append(
                                1.0 - _tf.reduce_sum(
                                    _tf.einsum("ab,sbc->sac", cov, self.cov_smooth_inv[j]) * cov[None, :, :],
                                    axis=2, keepdims=False
                                )
                            )
                        means = _tf.stack(means, axis=0)  # [n_experts, n_latent, n_data, 1]
                        pred_vars = _tf.stack(pred_vars, axis=0)  # [n_experts, n_latent, n_data]
                        weights = _GPNode.get_expert_weights(pred_vars)
                        self.inducing_points.append(
                            _tf.transpose(_tf.reduce_sum(means[:, :, :, 0] * weights, axis=0))
                        )
                        self.inducing_points_variance.append(
                            _tf.transpose(_tf.reduce_sum(pred_vars * weights, axis=0))
                        )

    def cache_prediction_state(self):
        super().cache_prediction_state()
        self.alpha = self._cache_tuple("alpha", self.alpha)
        self.cov_inv = self._cache_tuple("cov_inv", self.cov_inv)
        self.cov_smooth_inv = self._cache_tuple(
            "cov_smooth_inv", self.cov_smooth_inv)
        self.chol_r = self._cache_tuple("chol_r", self.chol_r)

    def interpolate(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("basic_interpolation"):
            cov_cross = [
                self.covariance_matrix(x, ip, x_var, ip_var)
                for ip, ip_var in zip(self.parent.inducing_points, self.parent.inducing_points_variance)
            ]

            bias = [self.parameters[f'bias_{i}'].get_value() for i in range(self.root.n_experts)]
            mu = [
                _tf.einsum("ab,sbc->sac", mat, vec) + b
                for mat, vec, b in zip(cov_cross, self.alpha, bias)
            ]

            explained_var = [
                _tf.reduce_sum(
                    _tf.einsum("ab,sbc->sac", m1, m2) * m1[None, :, :],
                    axis=2, keepdims=False
                )
                for m1, m2 in zip(cov_cross, self.cov_smooth_inv)
            ]
            var = _tf.stack([_tf.maximum(1.0 - v, 0.0) for v in explained_var], axis=0)

            influence = [
                _tf.reduce_sum(_tf.matmul(m1, m2) * m1, axis=1, keepdims=False)
                for m1, m2 in zip(cov_cross, self.cov_inv)
            ]
            influence = [_tf.tile(i[None, :], [self.size, 1]) for i in influence]

            weights = _GPNode.get_expert_weights(var)

            w_mu = _tf.reduce_sum(_tf.stack(mu, axis=0) * weights[:, :, :, None], axis=0)
            w_var = _tf.reduce_sum(_tf.stack(var, axis=0) * weights, axis=0)
            w_exp_var = _tf.reduce_sum(_tf.stack(explained_var, axis=0) * weights, axis=0)
            w_inf = _tf.reduce_sum(_tf.stack(explained_var, axis=0) * influence, axis=0)

            if n_sim > 0:
                rnd = [
                    _simulation_normals([self.size, n, n_sim], seed)
                    for n in self.root.n_ip
                ]
                sims = [
                    _tf.einsum("ab,sbc->sac", a, _tf.matmul(b, c)) + d
                    for a, b, c, d in zip(cov_cross, self.chol_r, rnd, mu)
                ]

                w_sims = _tf.reduce_sum(_tf.stack(sims, axis=0) * weights[:, :, :, None], axis=0)

                return w_mu, w_var, w_sims, w_exp_var, w_inf

            else:
                return w_mu, w_var

    def kl_divergence(self):
        with _tf.name_scope("basic_KL_divergence"):
            all_kl = []
            for i in range(self.root.n_experts):
                delta = self.parameters[f"delta_{i}"].get_value()
                alpha_white = self.parameters[f"alpha_white_{i}"].get_value()

                tr = _tf.reduce_sum(self.cov_smooth_inv[i] * self.cov[i][None, :, :])
                fit = _tf.reduce_sum(alpha_white**2)
                det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(self.cov_smooth_chol[i])))
                det_2 = _tf.reduce_sum(_tf.math.log(delta))
                kl = 0.5 * (- tr + fit + det_1 - det_2)

                all_kl.append(kl)

            return _tf.add_n(all_kl)

    # def covariance_matrix_d1(self, y, dir_y, step=1e-3):
    #     with _tf.name_scope("basic_covariance_matrix_d1"):
    #         x_pr = self.parent.inducing_points
    #         x_var = self.parent.inducing_points_variance
    #         y_pr_plus, y_var_plus = self.parent.propagate(
    #             y + 0.5 * step * dir_y)
    #         y_pr_minus, y_var_minus = self.parent.propagate(
    #             y - 0.5 * step * dir_y)
    #
    #         cov_1 = self.covariance_matrix(x_pr, y_pr_plus, x_var, y_var_plus)
    #         cov_2 = self.covariance_matrix(x_pr, y_pr_minus, x_var,
    #                                        y_var_minus)
    #
    #         return (cov_1 - cov_2) / step
    #
    # def point_variance_d2(self, x, dir_x, step=1e-3):
    #     with _tf.name_scope("basic_point_variance_d2"):
    #         mu_1, var_1 = self.parent.propagate(x + 0.5 * dir_x * step)
    #         mu_2, var_2 = self.parent.propagate(x - 0.5 * dir_x * step)
    #
    #         ranges = self.parameters["ranges"].get_value()[0, :, :]
    #         var_1 = var_1 + ranges ** 2
    #         var_2 = var_2 + ranges ** 2
    #
    #         dif = mu_1 - mu_2
    #         avg_var = 0.5 * (var_1 + var_2)
    #         dist_sq = _tf.reduce_sum(dif ** 2 / avg_var, axis=1, keepdims=True)
    #
    #         cov_step = self.kernel.kernelize(_tf.sqrt(dist_sq))
    #
    #         det_avg = _tf.reduce_prod(avg_var, axis=1, keepdims=True) ** (1/2)
    #         det_1 = _tf.reduce_prod(var_1, axis=1, keepdims=True) ** (1/4)
    #         det_2 = _tf.reduce_prod(var_2, axis=1, keepdims=True) ** (1/4)
    #
    #         # norm = _tf.reduce_prod(ranges) / det_avg
    #         norm = det_1 * det_2 / det_avg
    #         cov_step = cov_step * norm
    #
    #         point_var = 2 * (1.0 - cov_step) / step ** 2
    #         point_var = _tf.tile(point_var, [1, self.size])
    #         point_var = _tf.transpose(point_var)
    #
    #         return point_var
    #
    # def predict_directions(self, x, dir_x, step=1e-3):
    #     with _tf.name_scope("basic_prediction_directions"):
    #
    #         cov_cross = self.covariance_matrix_d1(x, dir_x, step)
    #         cov_cross = _tf.transpose(cov_cross)
    #
    #         mu = _tf.einsum("ab,sbc->sac", cov_cross, self.alpha)
    #
    #         explained_var = _tf.reduce_sum(
    #             _tf.einsum("ab,sbc->sac", cov_cross, self.cov_smooth_inv)
    #             * cov_cross[None, :, :],
    #             axis=2, keepdims=False)
    #
    #         point_var = self.point_variance_d2(x, dir_x, step)
    #         var = _tf.maximum(point_var - explained_var, 0.0)
    #
    #         return mu, var, explained_var


class AdditiveGP(BasicGP):
    """
    Additive GP node.

    This node is similar to the `BasicGP`, with the difference that is covariance matrices are computed separately for
    each input dimension and then averaged. It makes more sense to use it on high-dimensional non-spatial inputs.
    """
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
    """
    Linear node.

    This node outputs one or more linear combinations of the inputs. Its role in a network depends on its position.
    Close to a root node it induces rotation in the coordinates. At the end it induces correlations between the
    outputs, and in the middle it can serve as an information bottleneck.
    """
    def __init__(self, parent, size=1, unit_norm=True, weight_prior=1.0,
                 name=None):
        """
        Initializer for Linear.

        Parameters
        ----------
        parent
            Parent node
        size
            Number of output latent variables.
        unit_norm : bool
            Whether the weights should form a unit norm vector. If `False`,
            the weights are free and regularized by `weight_prior`.
        weight_prior : float, optional
            Standard deviation of the zero-mean Gaussian prior on the free
            weights (`unit_norm=False` only -- the unit norm is constraint
            enough on its own). The weights stay point estimates; the
            prior's log-density joins the training objective, so a weight
            grows only while the data pays for it, which matters because
            this is the parameter whose count scales with the network
            (`parent.size` times `size`) and no KL prices it. The standard
            deviation of 1 matches the whitened scale the network works in.
            `None` removes the prior and restores the hard [-1, 1] walls of
            versions before 0.6.5.
        name : str
            A name for this node.
        """
        super().__init__(parent, name=name)
        self._size = size

        if unit_norm:
            rnd = _rnd.rng().normal(size=(parent.size, self.size))
            rnd = rnd / _np.sqrt(_np.sum(rnd ** 2, axis=0, keepdims=True))
            self._add_parameter(
                "weights",
                _gpr.UnitColumnNormParameter(
                    rnd, - _np.ones_like(rnd), _np.ones_like(rnd)
                )
            )
        else:
            rnd = _rnd.rng().normal(size=(parent.size, self.size), scale=1e-4)
            # with a prior the walls step back to a safety net: the prior is
            # what holds the weights now, and it can be out-argued by the
            # data where a wall cannot
            wall = 1.0 if weight_prior is None else 10.0
            self._add_parameter(
                "weights",
                _gpr.RealParameter(
                    _np.zeros([parent.size, self.size]) + rnd + 1/parent.size,
                    _np.zeros([parent.size, self.size]) - wall,
                    _np.zeros([parent.size, self.size]) + wall
                )
            )
            if weight_prior is not None:
                self.parameters["weights"].prior = _tfd.Normal(
                    _tf.constant(0.0, _tf.float64),
                    _tf.constant(float(weight_prior), _tf.float64))

        # binary classification
        if (parent.size == 1) & (self.size == 2):
            self.parameters["weights"].set_value([[1, -1]])
            self.parameters["weights"].fix()

    def refresh(self, jitter=1e-6):
        weights = self.parameters["weights"].get_value()

        self.parent.refresh(jitter)

        if self.propagates_inducing_points:
            self.inducing_points = tuple(
                _tf.matmul(ip, weights)
                for ip in self.parent.inducing_points
            )
            self.inducing_points_variance = tuple(
                _tf.matmul(ip_var, weights**2)
                for ip_var in self.parent.inducing_points
            )

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

    # def predict_directions(self, x, dir_x, step=1e-3):
    #     mu, var, explained_var = self.parent.predict_directions(x, dir_x, step)
    #
    #     weights = self.parameters["weights"].get_value()
    #
    #     mu = _tf.einsum("xab,xy->yab", mu, weights)
    #     var = _tf.einsum("xa,xy->ya", var, weights ** 2)
    #     explained_var = _tf.einsum("xa,xy->ya", explained_var, weights ** 2)
    #
    #     return mu, var, explained_var


class SelectInput(_FunctionalLatentVariable):
    """
    Variable selection.

    Returns the specified columns of the input, discarding the others.
    """
    def __init__(self, parent, columns, name=None):
        """
        Initializer for SelectInput.

        Parameters
        ----------
        parent
            Parent node.
        columns : list
            List of indices to retain.
        name : str
            A name for this node.
        """
        super().__init__(parent, name=name)
        self.columns = _tf.constant(columns)
        self._size = len(columns)

    def propagate(self, x, x_var=None):
        mean, var = self.parent.propagate(x, x_var)
        mean = _tf.gather(mean, self.columns, axis=1)
        var = _tf.gather(var, self.columns, axis=1)
        return mean, var

    def refresh(self, jitter=1e-6):
        self.parent.refresh(jitter)

        if self.propagates_inducing_points:
            self.inducing_points = tuple(
                _tf.gather(ip, self.columns, axis=1)
                for ip in self.parent.inducing_points
            )
            self.inducing_points_variance = tuple(
                _tf.gather(ip_var, self.columns, axis=1)
                for ip_var in self.parent.inducing_points_variance
            )

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
    """
    Linear combination.

    This node combines the inputs linearly with positive weights.
    """
    def __init__(self, *latent_variables, unit_variance=True,
                 per_component=False, weight_concentration=2.0, name=None):
        """
        Initializer for LinearCombination.

        Parameters
        ----------
        latent_variables
            Nodes to combine. They must all have the same number of variables.
        unit_variance : bool
            If `True`, constrains the weights to unit sum to control the variance of the output.
        per_component : bool
            One set of mixing weights per output component instead of one
            for the whole node, so each component takes its own share of
            each parent -- one element can lean on a trend that another
            ignores. Requires `unit_variance`, and multiplies the weight
            count by `size`, which is why the prior below comes with it.
        weight_concentration : float, optional
            Concentration of the symmetric Dirichlet prior on each
            component's weights (`per_component=True` only -- the shared
            weights are few enough to need none). The weights stay point
            estimates; the prior's log-density joins the training
            objective, holding each component's shares near equal until its
            data argues otherwise. Must exceed 1 for the pull to point at
            equal shares; `None` removes it.
        name : str
            A name for this node.
        """
        super().__init__(*latent_variables, name=name)
        self._size = self._common_size()
        self.propagates_inducing_points = self.same_root and all([p.propagates_inducing_points for p in self.parents])
        self.per_component = per_component

        n_parents = len(latent_variables)
        if per_component:
            if not unit_variance:
                raise ValueError(
                    "per_component weights are compositional; they require "
                    "unit_variance=True")
            self._add_parameter(
                "weights",
                _gpr.UnitColumnSumParameter(
                    _np.ones([n_parents, self._size]) / n_parents)
            )
            if weight_concentration is not None:
                if weight_concentration <= 1.0:
                    raise ValueError(
                        "weight_concentration must be greater than 1 for "
                        "the prior to peak at equal shares, got %r"
                        % (weight_concentration,))
                self.parameters["weights"].prior = _ColumnwiseDirichlet(
                    _tf.constant(
                        _np.full(n_parents, float(weight_concentration)),
                        _tf.float64))
        elif unit_variance:
            self._add_parameter(
                "weights",
                _gpr.CompositionalParameter(
                    _np.ones(n_parents) / n_parents)
            )
        else:
            self._add_parameter(
                "weights",
                _gpr.PositiveParameter(
                    _np.ones(n_parents) / n_parents,
                    _np.ones(n_parents) * 0.01,
                    _np.ones(n_parents) * 100
                )
            )

    def _weights_for(self, stacked):
        """The weights, broadcast-ready for one `[size, ..., n_parents]`
        stack. Shared weights ride the trailing axis at any rank; the
        per-component ones need their `size` axis leading and ones between,
        and the stacks do not agree on rank (a mean carries a simulation
        axis, a variance does not), so the shape is read off each stack."""
        weights = self.parameters["weights"].get_value()
        if not self.per_component:
            return weights
        shape = [self.size] + [1] * (len(stacked.shape) - 2) \
            + [len(self.parents)]
        return _tf.reshape(_tf.transpose(weights), shape)

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)

        if self.propagates_inducing_points:
            weights = self.parameters["weights"].get_value()
            if self.per_component:
                # against the [n_parents, n_ip, size] stacking below
                weights = weights[:, None, :]
            else:
                weights = weights[:, None, None]

            all_ip, all_ip_var = [], []
            for i in range(self.root.n_experts):
                ip = _tf.stack([lat.inducing_points[i] for lat in self.parents], axis=0)
                ip = _tf.reduce_sum(ip * weights, axis=0)
                all_ip.append(ip)

                ip_var = _tf.stack([lat.inducing_points_variance[i] for lat in self.parents], axis=0)
                ip_var = _tf.reduce_sum(ip_var * weights**2, axis=0)
                all_ip_var.append(ip_var)

            self.inducing_points = tuple(all_ip)
            self.inducing_points_variance = tuple(all_ip_var)

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

        all_mu = _tf.stack(all_mu, axis=-1)
        all_var = _tf.stack(all_var, axis=-1)
        all_sims = _tf.stack(all_sims, axis=-1)
        all_explained_var = _tf.stack(all_explained_var, axis=-1)
        all_influence = _tf.stack(all_influence, axis=-1)

        all_mu = _tf.reduce_sum(
            all_mu * self._weights_for(all_mu), axis=-1)
        all_var = _tf.reduce_sum(
            all_var * self._weights_for(all_var) ** 2, axis=-1)
        all_sims = _tf.reduce_sum(
            all_sims * self._weights_for(all_sims), axis=-1)
        all_explained_var = _tf.reduce_sum(
            all_explained_var * self._weights_for(all_explained_var) ** 2,
            axis=-1)
        all_influence = _tf.reduce_sum(
            all_influence * self._weights_for(all_influence) ** 2, axis=-1)

        return all_mu, all_var, all_sims, all_explained_var, all_influence

    def predict_directions(self, x, dir_x, jitter=1e-6):
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

        all_mu = _tf.reduce_sum(
            all_mu * self._weights_for(all_mu), axis=-1)
        all_var = _tf.reduce_sum(
            all_var * self._weights_for(all_var) ** 2, axis=-1)
        all_explained_var = _tf.reduce_sum(
            all_explained_var * self._weights_for(all_explained_var) ** 2,
            axis=-1)

        return all_mu, all_var, all_explained_var

    def propagate(self, x, x_var=None):
        mu, var, _, _, _ = self.predict(x, x_var, n_sim=1)
        mu = _tf.transpose(mu[:, :, 0])
        var = _tf.transpose(var)
        return mu, var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)
        # weights = self.parameters["weights"].get_value()
        # kl = _tf.reduce_sum(weights * _tf.math.log(weights * self.size))
        # return kl


class ProductOfExperts(_Operation):
    """
    Product of Experts.

    The Product of Experts combines latent variables from different nodes with weights inversely proportional to
    the local variance. It is more useful when combining the outputs of smaller networks with different set of
    inducing points, allowing each one to focus on a region of space.

    This node treats its parents independently. Means and variances will be "stiched" smoothly, but individual
    simulations may exhibit artifacts.

    This node is not capable of propagating inducing points.
    """
    def __init__(self, *latent_variables, name=None):
        """
        Initializer for ProductOfExperts.

        Parameters
        ----------
        latent_variables
            Parent nodes to combine.
        name : str
            A name for this node.
        """
        super().__init__(*latent_variables, name=name)
        self._size = self._common_size()
        self.propagates_inducing_points = False

    def refresh(self, jitter=1e-6):
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

        w_mu = _tf.reduce_sum(weights[:, :, :, None] * all_mu, axis=0)
        w_var = _tf.reduce_sum(weights * all_var, axis=0)
        w_explained_var = _tf.reduce_sum(weights * all_explained_var, axis=0)

        return w_mu, w_var, w_explained_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class Exponentiation(_FunctionalLatentVariable):
    def __init__(self, parent, name=None):
        super().__init__(parent, name=name)
        self._add_parameter("amp_mean", _gpr.RealParameter(0, -5, 5))
        self._add_parameter(
            "amp_scale", _gpr.PositiveParameter(0.25, 0.01, 10))
        self._size = parent.size
        self.propagates_inducing_points = False

    # def refresh(self, jitter=1e-6):
        # amp_mean = self.parameters["amp_mean"].get_value()
        # amp_scale = self.parameters["amp_scale"].get_value()

        # self.parent.refresh(jitter)

        # if self.parent.inducing_points is not None:
        #     ip = self.parent.inducing_points
        #     ip_var = self.parent.inducing_points_variance
        #
        #     ip = ip * _tf.sqrt(amp_scale) + amp_mean
        #     ip_var = ip_var * amp_scale
        #
        #     amp_mu = _tf.exp(ip) * (1 + 0.5 * ip_var)
        #     amp_var = _tf.exp(2 * ip) * ip_var * (1 + ip_var)
        #
        #     self.inducing_points = amp_mu
        #     self.inducing_points_variance = amp_var

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


class Multiply(_Operation):
    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        self._size = self._common_size()
        self.propagates_inducing_points = False

    def refresh(self, jitter=1e-6):
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

    # def predict_directions(self, x, dir_x, jitter=1e-6):
    #     all_mu = []
    #     all_var = []
    #     all_explained_var = []
    #
    #     for i, v in enumerate(self.parents):
    #         mu, var, explained_var = v.predict_directions(x, dir_x, jitter)
    #         all_mu.append(mu)
    #         all_var.append(var)
    #         all_explained_var.append(explained_var)
    #
    #     all_mu = _tf.stack(all_mu, axis=0)
    #     all_var = _tf.stack(all_var, axis=0)
    #
    #     pred_mu = _tf.reduce_prod(all_mu, axis=0)
    #     pred_var = _tf.reduce_prod(all_mu[:, :, :, 0] ** 2 + all_var, axis=0) \
    #                - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0)
    #
    #     pred_explained_var = \
    #         _tf.reduce_prod(
    #             all_mu[:, :, :, 0] ** 2 + all_var + all_explained_var,
    #             axis=0) \
    #         - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0) \
    #         - pred_var
    #
    #     return pred_mu, pred_var, pred_explained_var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class Add(_Operation):
    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        self._size = self._common_size()
        self.propagates_inducing_points = self.same_root and all([p.propagates_inducing_points for p in self.parents])

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)

        if self.propagates_inducing_points:
            all_ip, all_ip_var = [], []
            for i in range(self.root.n_experts):
                ip = _tf.stack([lat.inducing_points[i] for lat in self.parents], axis=0)
                ip = _tf.reduce_sum(ip, axis=0)
                all_ip.append(ip)

                ip_var = _tf.stack([lat.inducing_points_variance[i] for lat in self.parents], axis=0)
                ip_var = _tf.reduce_sum(ip_var, axis=0)
                all_ip_var.append(ip_var)

            self.inducing_points = tuple(all_ip)
            self.inducing_points_variance = tuple(all_ip_var)

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

    # def predict_directions(self, x, dir_x, jitter=1e-6):
    #     all_mu = []
    #     all_var = []
    #     all_explained_var = []
    #
    #     for i, v in enumerate(self.parents):
    #         mu, var, explained_var = v.predict_directions(x, dir_x, jitter)
    #         all_mu.append(mu)
    #         all_var.append(var)
    #         all_explained_var.append(explained_var)
    #
    #     all_mu = _tf.stack(all_mu, axis=-1)
    #     all_var = _tf.stack(all_var, axis=-1)
    #     all_explained_var = _tf.stack(all_explained_var, axis=-1)
    #
    #     all_mu = _tf.reduce_sum(all_mu, axis=-1)
    #     all_var = _tf.reduce_sum(all_var, axis=-1)
    #     all_explained_var = _tf.reduce_sum(all_explained_var, axis=-1)
    #
    #     return all_mu, all_var, all_explained_var

    def propagate(self, x, x_var=None):
        mu, var = self.predict(x, x_var, n_sim=0)
        mu = _tf.transpose(mu[:, :, 0])
        var = _tf.transpose(var)
        return mu, var

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class Bias(_FunctionalLatentVariable):
    """
    Bias

    Adds a deterministic constant to its input.
    """
    def __init__(self, parent, scale=5, name=None):
        super().__init__(parent, name=name)
        self._size = parent.size

        self._add_parameter(
            "bias",
            _gpr.RealParameter(
                _np.zeros([self.size]),
                _np.zeros([self.size]) - scale,
                _np.zeros([self.size]) + scale
            )
        )

    def refresh(self, jitter=1e-6):
        bias = self.parameters["bias"].get_value()[None, :]

        self.parent.refresh(jitter)

        if self.propagates_inducing_points:
            self.inducing_points = tuple(ip + bias for ip in self.parent.inducing_points)
            self.inducing_points_variance = self.parent.inducing_points_variance

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

    # def predict_directions(self, x, dir_x, step=1e-3):
    #     return self.parent.predict_directions(x, dir_x, step)


class Scale(_FunctionalLatentVariable):
    """
    Scale.

    Multiplies its input by a constant. The variance is multiplied by the square of the same value.
    """
    def __init__(self, parent, name=None):
        super().__init__(parent, name=name)
        self._size = parent.size

        self._add_parameter(
            "scale",
            _gpr.PositiveParameter(
                _np.ones([self.size]),
                _np.ones([self.size]) / 100,
                _np.ones([self.size]) * 10
            )
        )

    def refresh(self, jitter=1e-6):
        scale = self.parameters["scale"].get_value()[None, :]

        self.parent.refresh(jitter)

        if self.propagates_inducing_points:
            self.inducing_points = tuple(ip * _tf.sqrt(scale) for ip in self.parent.inducing_points)
            self.inducing_points_variance = tuple(ip_var * scale for ip_var in self.parent.inducing_points_variance)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        scale = self.parameters["scale"].get_value()

        if n_sim > 0:
            mu, var, sims, exp_var, influence = self.parent.predict(x, x_var, n_sim, seed)

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


class RadialTrend(_FunctionalLatentVariable):
    """
    Radial trend.

    This node outputs a (hyper)spherical deterministic function, positive on the inside and negative on the outside.
    It can be made ellipsoidal or with a more complex shape depending on its parent nodes. Its main use is for
    implicit geological modelling.

    It will ignore the variance of its inputs.
    """
    def __init__(self, parent, size=1, name=None):
        """
        Initializer for RadialTrend.

        Parameters
        ----------
        parent
            Parent node.
        size : int
            Number of output functions to generate.
        name : str
            A name for this node.
        """
        super().__init__(parent, name=name)
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

    def refresh(self, jitter=1e-6):
        self.parent.refresh(jitter)

        if self.propagates_inducing_points:
            self.inducing_points = tuple(
                _tf.transpose(self.compute_trend(_tf.transpose(ip)))
                for ip in self.parent.inducing_points
            )
            self.inducing_points_variance = tuple(
                _tf.zeros_like(ip_var)
                for ip_var in self.parent.inducing_points_variance
            )

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
    """
    Stochastic Differential Equation.

    This node uses the vector field defined by its parent to move points in space. After each step the field is
    reevaluated and the point's mean, variance, and direction is updated. It is very effective to learn non-stationary
    patterns, but it is computationally expensive.

    The node's parent (a GP) defines the vector field and the parent's parent contains the coordinates that will be
    moved. Both must have the same size.
    """
    def __init__(self, parent, step=0.01, n_steps=10, name=None):
        """
        Initializer for GPWalk.

        In principle the `step` argument does not need to be changed, as the underlying GP tends to adjust its
        amplitude to take larger or smaller steps in practice. A higher `n_steps` allows the model to have finer control
        of the points' trajectories at a higher computational cost.  `n_steps=5` seems to be the minimum possible
        for practical purposes.

        Parameters
        ----------
        parent
            Parent node. Must be a GP variant.
        step : float
            Size of the step at each iteration.
        n_steps : int
            Number of steps.
        name : str
            A name for this node.
        """
        super().__init__(parent, name=name)

        if parent.size != parent.parent.size:
            raise SizeIncompatibilityError(
                f"{self.name}: the parent node must have the same size as its own parent. "
                f"Found {parent.name} (size {parent.size}) and "
                f"{parent.parent.name} (size {parent.parent.size})."
            )

        self.walker = parent.parent
        self.field = parent
        self._size = parent.size

        self.step = step
        self.n_steps = n_steps

        self._add_parameter(
            "amp",
            _gpr.PositiveParameter(1, 0.01, 100)
        )
        self._add_parameter(
            "precision",
            _gpr.PositiveParameter(0.1, 0.01, 100)
        )

    def propagate(self, x, x_var=None):
        walker_mu, walker_var = self.walker.propagate(x, x_var)

        amp = self.parameters["amp"].get_value()
        prec = self.parameters["precision"].get_value()

        for _ in range(self.n_steps):
            field_mu, field_var = self.field.interpolate(
                walker_mu, walker_var, n_sim=0)
            field_mu = _tf.transpose(field_mu[:, :, 0])
            field_var = _tf.transpose(field_var)

            walker_mu = walker_mu + self.step * field_mu * amp
            walker_var = walker_var + self.step * field_var * amp ** 2

            # Kalman filtering
            walker_var = walker_var / (prec + 1)

        return walker_mu, walker_var

    def refresh(self, jitter=1e-6):
        self.field.refresh(jitter)
        # self.inducing_points, self.inducing_points_variance = self.propagate(
        #     *self.root.get_root_inducing_points()
        # )

        root_ip, root_var = self.root.get_root_inducing_points()
        all_ip, all_ip_var = [], []
        for i in range(self.root.n_experts):
            ip, ip_var = self.propagate(root_ip[i], root_var[i])
            all_ip.append(ip)
            all_ip_var.append(ip_var)
        self.inducing_points = tuple(all_ip)
        self.inducing_points_variance = tuple(all_ip_var)

    def kl_divergence(self):
        # return _tf.constant(0.0, _tf.float64)
        # mu_1 = self.parent.parent.inducing_points
        # var_1 = self.parent.parent.inducing_points_variance + 0.01

        # mu_2 = self.inducing_points
        # var_2 = self.inducing_points_variance

        # kl = 0.5 * _tf.reduce_sum(
        #     var_2 / var_1
        #     - self.root.n_ip
        #     + (mu_2 - mu_1)**2 / var_1
        #     + _tf.math.log(var_1 / var_2)
        # )
        # kl = 0.5 * _tf.reduce_sum((mu_2 - mu_1) ** 2 / var_2)

        kl = _tf.add_n([
            0.5 * _tf.reduce_sum((mu_2 - mu_1) ** 2 / var_2)
            for mu_1, mu_2, var_2 in zip(
                self.parent.parent.inducing_points, self.inducing_points, self.inducing_points_variance
            )
        ])
        return kl

    def compute_path(self, x, x_var=None):
        walker_mu, walker_var = self.walker.propagate(x, x_var)

        amp = self.parameters["amp"].get_value()
        prec = self.parameters["precision"].get_value()

        all_mu = [walker_mu]
        all_var = [walker_var]
        for _ in range(self.n_steps):
            field_mu, field_var = self.field.interpolate(
                walker_mu, walker_var, n_sim=0)
            field_mu = _tf.transpose(field_mu[:, :, 0])
            field_var = _tf.transpose(field_var)

            walker_mu = walker_mu + self.step * field_mu * amp
            walker_var = walker_var + self.step * field_var * amp ** 2
            walker_var = walker_var / (prec + 1)

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
        rnd = _simulation_normals([self.size, 1, n_sim], seed)
        sims = mu + rnd * _tf.sqrt(var[:, :, None])

        influence = _tf.zeros_like(var)

        return mu, var, sims, explained_var, influence


# class GaussianInput(_RootLatentVariable):
#     def __init__(self, inducing_points, fix_inducing_points=True,
#                  center=False):
#         super().__init__()
#         self._size = inducing_points.coordinates.shape[1]
#         self.bounding_box = inducing_points.bounding_box
#
#         self.n_ip = inducing_points.coordinates.shape[0]
#         self._add_parameter(
#             "inducing_points",
#             _gpr.RealParameter(
#                 inducing_points.coordinates,
#                 _np.tile(self.bounding_box.min, [self.n_ip, 1]),
#                 _np.tile(self.bounding_box.max, [self.n_ip, 1]),
#                 fixed=fix_inducing_points
#             ))
#         self._add_parameter(
#             "inducing_points_variance",
#             _gpr.PositiveParameter(
#                 _np.ones_like(inducing_points.coordinates),
#                 _np.ones_like(inducing_points.coordinates) * 0.01,
#                 _np.ones_like(inducing_points.coordinates) * 10
#             ))
#
#         self.center = _np.zeros_like(self.bounding_box.max)
#         if center:
#             self.center = 0.5 * (self.bounding_box.min + self.bounding_box.max)
#
#     def get_root_inducing_points(self):
#         ip = self.parameters["inducing_points"].get_value()
#         ip_var = self.parameters["inducing_points_variance"].get_value()
#         return ip, ip_var
#
#     def refresh(self, jitter=1e-6):
#         with _tf.name_scope("basic_input_refresh"):
#             self.inducing_points = \
#                 self.parameters["inducing_points"].get_value() - self.center
#             self.inducing_points_variance = \
#                 self.parameters["inducing_points_variance"].get_value()
#
#     def propagate(self, x, x_var=None):
#         return x - self.center, x_var
#
#     def kl_divergence(self):
#         return _tf.constant(0.0, _tf.float64)
#
#     def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
#         x = _tf.transpose(x - self.center)
#         x_var = _tf.transpose(x_var)
#         if n_sim > 0:
#             sims = _tf.tile(x[:, :, None], [1, 1, n_sim])
#             return x[:, :, None], x_var, sims, \
#                    _tf.zeros_like(x_var), _tf.zeros_like(x_var)
#         else:
#             return x[:, :, None], x_var


class MultiStructureGP(BasicGP):
    """
    Gaussian process with multiple structures.

    A linear combination of multiple kernels with (possibly) different ranges. The difference between using this node
    and applying a linear combination externally is that here the combination is at the kernel level instead of the
    latent variable level.
    """
    def __init__(self, parent, size=1, kernel=_kr.Gaussian(), fix_range=False,
                 n_structures=2, weight_concentration="staircase",
                 range_prior=2.0, name=None):
        """
        Initializer for MultiStructureGP.

        Parameters
        ----------
        parent
            Parent node.
        size : int
            Number of output functions.
        kernel
            The kernel to use for the covariance matrices.
        fix_range : bool
            Whether to force a unit range for all input dimensions.
        n_structures : int
            Number of kernels to combine (minimum 2).
        weight_concentration : str, float, or None
            The Dirichlet prior on the structure weights, which stay point
            estimates -- the prior's log-density joins the training
            objective. `"staircase"` (the default) aligns the prior with
            the ranges: structure `n` starts with range `1 / (n + 1)`, and
            its weight's share of the prior's peak follows the same
            ordering, so mass sits on the long-range structure until the
            data moves it to the short ones. The weights themselves still
            start uniform -- initializing them on the staircase was
            measured and rejected, since training never left that basin. A
            number gives a symmetric Dirichlet peaking at equal shares (it
            must exceed 1); `None` removes the prior, as in versions
            before 0.6.5.
        range_prior : float, optional
            Strength of the Gamma priors on the ranges, one per structure,
            each peaking at that structure's own starting range rather than
            at a common value -- a shared peak would fight the staircase the
            structures exist for. `None` removes them.
        name : str
            A name for this node.
        """
        self.n_structures = n_structures
        self.weight_concentration = weight_concentration
        super().__init__(parent, size, kernel, fix_range,
                         range_prior=range_prior, name=name)

    def _set_parameters(self):
        for i, n in enumerate(self.root.n_ip):
            self._add_parameter(
                f"alpha_white_{i}",
                _gpr.RealParameter(
                    _rnd.rng().normal(
                        scale=1e-3,
                        size=[self.size, n, 1]
                    ),
                    _np.zeros([self.size, n, 1]) - 10,
                    _np.zeros([self.size, n, 1]) + 10
                ))
            self._add_parameter(
                f"delta_{i}",
                _gpr.PositiveParameter(
                    _np.ones([self.size, n]),
                    _np.ones([self.size, n]) * 1e-6,
                    _np.ones([self.size, n]) * 1e2
                ))
            self._add_parameter(
                f"bias_{i}",
                _gpr.RealParameter(0, -5, 5))

        concentration = self.weight_concentration
        if concentration == "staircase":
            # concentrations 1 + 2/(n+1): the prior's peak puts shares in
            # proportion to each structure's starting range. The weights
            # still START uniform -- initializing them on the staircase was
            # measured on Walker Lake against the exhaustive truth and
            # rejected: training never leaves that basin ([0.83, 0.10,
            # 0.07] against the [0.72, 0.13, 0.15] a uniform start finds
            # with or without the prior), and every truth-facing score is
            # worse. The prior alone improved all of them, on every seed.
            alpha = 1.0 + 2.0 / (_np.arange(self.n_structures) + 1.0)
        elif isinstance(concentration, str):
            # any other string would reach the comparison below and fail
            # there, on a TypeError naming neither the argument nor its
            # choices
            raise ValueError(
                "weight_concentration must be 'staircase', a number greater "
                "than 1, or None; got %r" % (concentration,))
        elif concentration is not None:
            if concentration <= 1.0:
                raise ValueError(
                    "weight_concentration must be greater than 1 for the "
                    "prior to peak at equal shares, got %r" % (concentration,))
            alpha = _np.full(self.n_structures, float(concentration))
        else:
            alpha = None

        self._add_parameter(
            "weights", _gpr.CompositionalParameter(
                _np.ones(self.n_structures) / self.n_structures))
        if alpha is not None:
            self.parameters["weights"].prior = _tfd.Dirichlet(
                _tf.constant(alpha, _tf.float64))

        for n in range(self.n_structures):
            self._add_parameter(
                f"ranges_{n}",
                _gpr.PositiveParameter(
                    _np.ones([1, 1, self.parent.size]) / (n + 1),
                    _np.ones([1, 1, self.parent.size]) * 1e-2,
                    _np.ones([1, 1, self.parent.size]) * 10,
                    fixed=self.fix_range
                )
            )
            if self.range_prior is not None:
                # each structure's prior peaks at its own starting range:
                # a common peak at 1 would fight the staircase
                self.parameters[f"ranges_{n}"].prior = _gamma_mode_one(
                    self.range_prior, mode=1.0 / (n + 1))

    def covariance_matrix(self, x, y, var_x=None, var_y=None):
        with _tf.name_scope("basic_covariance_matrix"):
            weights = self.parameters["weights"].get_value()
            cov_mats = []

            if var_x is None:
                var_x = _tf.zeros_like(x)
            if var_y is None:
                var_y = _tf.zeros_like(y)
            var_x = var_x[:, None, :]
            var_y = var_y[None, :, :]

            # [n_data, n_data, n_dim]
            dif = x[:, None, :] - y[None, :, :]

            for n in range(self.n_structures):
                ranges = self.parameters[f"ranges_{n}"].get_value()

                total_var = ranges**2 + (var_x + var_y) / 2
                dist = _tf.sqrt(_tf.reduce_sum(dif ** 2 / total_var, axis=-1))
                cov = self.kernel.kernelize(dist)

                # normalization
                det_x = _tf.reduce_prod(var_x + ranges**2, axis=-1) ** (1 / 4)
                det_y = _tf.reduce_prod(var_y + ranges**2, axis=-1) ** (1 / 4)
                det_2 = _tf.sqrt(_tf.reduce_prod(total_var, axis=-1))

                norm = det_x * det_y / det_2

                # output
                cov = cov * norm * weights[n]
                cov_mats.append(cov)

            cov = _tf.add_n(cov_mats)
            return cov


class GradientConstrainedInput(_RootLatentVariable):
    """
    Inputs constrained by structural data.

    This node uses a set of directional data to constrain the output's gradient. The output GP is considered to have
    zero gradient in the specified directions, flowing only in the orthogonal direction.
    """
    def __init__(self, inducing_points, directional_data,
                 covariance, size=1, fix_covariance=False, name=None):
        """
        Initializer for GradientConstrainedInput.

        The locations of the provided `directional_data` will be added to the inducing points set to better constrain
        the output.

        Parameters
        ----------
        inducing_points
            A `PointData` object, or a list of these objects.
        directional_data
            A `DirectionalData` object, or a list of these objects.
        covariance
            A covariance object, containing a kernel and transform.
        size : int
            Number of output variables.
        fix_covariance : bool
            Whether to fix the covariance's parameters during training.
        name : str
            A name for this node.
        """
        super().__init__(name=name)

        self._size = size
        # self.root_size = inducing_points.n_dim

        if not isinstance(inducing_points, (list, tuple)):
            inducing_points = (inducing_points, )
        self._n_experts = len(inducing_points)

        if not isinstance(directional_data, (list, tuple)):
            directional_data = (directional_data, )

        all_coords = _np.concatenate(
            [ip.coordinates for ip in inducing_points]
            + [ip.coordinates for ip in directional_data]
        )
        self.bounding_box = _data.BoundingBox.from_array(_data.bounding_box(all_coords)[0])

        self.covariance = self._register(covariance)
        self.covariance.set_limits(_data.PointData.from_array(all_coords))
        if fix_covariance:
            for p in self.covariance.all_parameters:
                p.fix()

        self.n_dir = tuple(d.n_data for d in directional_data)
        self.directional_data = directional_data

        self.base_inducing_points = tuple(
            _tf.constant(_np.unique(_np.concatenate([ip.coordinates, d.coordinates]), axis=0),
                         dtype=_tf.float64
                         )
            for ip, d in zip(inducing_points, directional_data)
        )

        # The inducing set is the deduplicated union of the user inducing points
        # and the directional-data locations, so n_ip must be derived from that
        # base set (refresh sizes the covariance blocks off it).
        self.n_ip = tuple(int(ip.shape[0]) for ip in self.base_inducing_points)

        self.inducing_points_variance = tuple(_tf.zeros([n, self.size], _tf.float64) for n in self.n_ip)

        # GP setup
        self.scale = None
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

        self._set_parameters()

    def _set_parameters(self):
        for i, n in enumerate(self.root.n_ip):
            self._add_parameter(
                f"alpha_white_{i}",
                _gpr.RealParameter(
                    _rnd.rng().normal(
                        scale=1e-3,
                        size=[self.size, n, 1]
                    ),
                    _np.zeros([self.size, n, 1]) - 10,
                    _np.zeros([self.size, n, 1]) + 10
                ))
            self._add_parameter(
                f"delta_{i}",
                _gpr.PositiveParameter(
                    _np.ones([self.size, n]),
                    _np.ones([self.size, n]) * 1e-6,
                    _np.ones([self.size, n]) * 1e2
                ))
            self._add_parameter(
                f"bias_{i}",
                _gpr.RealParameter(0, -5, 5))

    def get_root_inducing_points(self):
        return self.base_inducing_points, self.inducing_points_variance

    def refresh(self, jitter=1e-6):
        with _tf.name_scope("constrained_input_refresh"):
            # constrained prior
            dir_coords = [_tf.constant(d.coordinates, _tf.float64)
                          for d in self.directional_data]
            dirs = [_tf.constant(d.directions, _tf.float64)
                          for d in self.directional_data]

            base_cov = tuple(
                self.covariance.self_covariance_matrix(ip)
                for ip in self.base_inducing_points
            )
            cross_cov = tuple(
                self.covariance.covariance_matrix_d1(ip, dc, d)
                for ip, dc, d in zip(self.base_inducing_points, dir_coords, dirs)
            )
            dir_cov = tuple(
                self.covariance.self_covariance_matrix_d2(dc, d)
                for dc, d in zip(dir_coords, dirs)
            )
            full_cov = tuple(
                _tf.concat([
                    _tf.concat([dc, _tf.transpose(cc)], axis=1),
                    _tf.concat([cc, bc], axis=1),
                ], axis=0)
                for bc, cc, dc in zip(base_cov, cross_cov, dir_cov)
            )

            self.scale = tuple(_tf.sqrt(_tf.linalg.diag_part(mat)) for mat in full_cov)
            full_cov = tuple(
                mat / sc[:, None] / sc[None, :]
                for mat, sc  in zip(full_cov, self.scale)
            )

            eye = tuple(_tf.eye(n + d, dtype=_tf.float64) for n, d in zip(self.n_ip, self.n_dir))
            chol = tuple(_tf.linalg.cholesky(mat) for mat in full_cov)
            cov_inv = tuple(_tf.linalg.cholesky_solve(mat, e) for mat, e in zip(chol, eye))

            self.cov = full_cov
            self.cov_chol = chol
            self.cov_inv = cov_inv

            # posterior
            eye = tuple(_tf.tile(e[None, :, :], [self.size, 1, 1]) for e in eye)
            delta = tuple(self.parameters[f"delta_{i}"].get_value() for i in range(self.n_experts))
            delta = tuple(
                _tf.concat([_tf.zeros([self.size, n], dtype=_tf.float64), d], axis=1)
                for n, d in zip(self.n_dir, delta)
            )
            delta_diag = tuple(_tf.linalg.diag(d) for d in delta)
            self.cov_smooth = tuple(mat[None, :, :] + d for mat, d in zip(self.cov, delta_diag))
            self.cov_smooth_chol = tuple(
                _tf.linalg.cholesky(mat + e * jitter)
                for mat, e in zip(self.cov_smooth, eye)
            )
            self.cov_smooth_inv = tuple(
                _tf.linalg.cholesky_solve(mat, e)
                for mat, e in zip(self.cov_smooth_chol, eye)
            )
            self.chol_r = tuple(
                _tf.linalg.cholesky(m1[None, :, :] - m2 + e * jitter)
                for m1, m2, e in zip(self.cov_inv, self.cov_smooth_inv, eye)
            )

            # inducing points
            alpha_white = tuple(self.parameters[f"alpha_white_{i}"].get_value() for i in range(self.n_experts))
            alpha_white = tuple(
                _tf.concat([_tf.zeros([self.size, n, 1], dtype=_tf.float64), a], axis=1)
                for n, a in zip(self.n_dir, alpha_white)
            )
            means = tuple(
                _tf.einsum("ab,sbc->sac", mat, vec)
                for mat, vec in zip(self.cov_chol, alpha_white)
            )
            self.alpha = tuple(
                _tf.einsum("ab,sbc->sac", mat, vec)
                for mat, vec in zip(self.cov_inv, means)
            )

            bias = [self.parameters[f'bias_{i}'].get_value() for i in range(self.n_experts)]

            self.inducing_points = []
            self.inducing_points_variance = []
            for i in range(self.n_experts):
                ip_i = self.base_inducing_points[i]
                # ipv_i = self.parent.inducing_points_variance[i]
                means = []
                pred_vars = []
                for j in range(self.n_experts):
                    ip_j = self.base_inducing_points[j]
                    cov_aa = self.covariance.covariance_matrix_d2(
                        dir_coords[i], dir_coords[j], dirs[i], dirs[j]
                    )
                    cov_ab = _tf.transpose(self.covariance.covariance_matrix_d1(ip_j, dir_coords[i], dirs[i]))
                    cov_ba = self.covariance.covariance_matrix_d1(ip_i, dir_coords[j], dirs[j])
                    cov_bb = self.covariance.covariance_matrix(ip_i, ip_j)
                    cov = _tf.concat([
                        _tf.concat([cov_aa, cov_ab], axis=1),
                        _tf.concat([cov_ba, cov_bb], axis=1)
                    ], axis=0)
                    means.append(_tf.einsum("ab,sbc->sac", cov, self.alpha[j]) + bias[j])
                    pred_vars.append(
                        1.0 - _tf.reduce_sum(
                            _tf.einsum("ab,sbc->sac", cov, self.cov_smooth_inv[j]) * cov[None, :, :],
                            axis=2, keepdims=False
                        )
                    )
                means = _tf.stack(means, axis=0)  # [n_experts, n_latent, n_data, 1]
                pred_vars = _tf.stack(pred_vars, axis=0)  # [n_experts, n_latent, n_data]
                weights = _GPNode.get_expert_weights(pred_vars)
                self.inducing_points.append(
                    _tf.transpose(_tf.reduce_sum(means[:, :, :, 0] * weights, axis=0))
                )
                self.inducing_points_variance.append(
                    _tf.transpose(_tf.reduce_sum(pred_vars * weights, axis=0))
                )


    def cache_prediction_state(self):
        super().cache_prediction_state()
        self.scale = self._cache_tuple("scale", self.scale)
        self.alpha = self._cache_tuple("alpha", self.alpha)
        self.cov_inv = self._cache_tuple("cov_inv", self.cov_inv)
        self.cov_smooth_inv = self._cache_tuple(
            "cov_smooth_inv", self.cov_smooth_inv)
        self.chol_r = self._cache_tuple("chol_r", self.chol_r)

    def propagate(self, x, x_var=None):
        mu, var = self.predict(x, x_var, n_sim=0)
        mu = _tf.transpose(mu[:, :, 0])
        var = _tf.transpose(var)
        return mu, var

    def kl_divergence(self):
        with _tf.name_scope("constrained_KL_divergence"):
            all_kl = []
            for i in range(self.root.n_experts):
                delta = self.parameters[f"delta_{i}"].get_value()
                alpha_white = self.parameters[f"alpha_white_{i}"].get_value()

                tr = _tf.reduce_sum(self.cov_smooth_inv[i] * self.cov[i][None, :, :])
                fit = _tf.reduce_sum(alpha_white ** 2)
                det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(self.cov_smooth_chol[i])))
                det_2 = _tf.reduce_sum(_tf.math.log(delta))
                kl = 0.5 * (- tr + fit + det_1 - det_2)

                all_kl.append(kl)

            return _tf.add_n(all_kl)

    def set_parameter_limits(self, data):
        self.covariance.set_limits(data)

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        with _tf.name_scope("constrained_root_prediction"):
            cov_cross = [
                _tf.concat([
                    self.covariance.covariance_matrix_d1(x, coord.coordinates, dir.directions),
                    self.covariance.covariance_matrix(x, ip)
                ], axis=1)
                for ip, coord, dir in zip(
                    self.base_inducing_points,
                    self.directional_data,
                    self.directional_data
                )
            ]
            cov_cross = [mat / sc[None, :] for mat, sc in zip(cov_cross, self.scale)]

            bias = [self.parameters[f'bias_{i}'].get_value() for i in range(self.root.n_experts)]
            mu = [
                _tf.einsum("ab,sbc->sac", mat, vec) + b
                for mat, vec, b in zip(cov_cross, self.alpha, bias)
            ]

            explained_var = [
                _tf.reduce_sum(
                    _tf.einsum("ab,sbc->sac", m1, m2) * m1[None, :, :],
                    axis=2, keepdims=False
                )
                for m1, m2 in zip(cov_cross, self.cov_smooth_inv)
            ]
            var = _tf.stack([_tf.maximum(1.0 - v, 0.0) for v in explained_var], axis=0)

            influence = [
                _tf.reduce_sum(_tf.matmul(m1, m2) * m1, axis=1, keepdims=False)
                for m1, m2 in zip(cov_cross, self.cov_inv)
            ]
            influence = [_tf.tile(i[None, :], [self.size, 1]) for i in influence]

            weights = _GPNode.get_expert_weights(var)

            w_mu = _tf.reduce_sum(_tf.stack(mu, axis=0) * weights[:, :, :, None], axis=0)
            w_var = _tf.reduce_sum(_tf.stack(var, axis=0) * weights, axis=0)
            w_exp_var = _tf.reduce_sum(_tf.stack(explained_var, axis=0) * weights, axis=0)
            w_inf = _tf.reduce_sum(_tf.stack(explained_var, axis=0) * influence, axis=0)

            if n_sim > 0:
                rnd = [
                    _simulation_normals([self.size, n + d, n_sim], seed)
                    for n, d in zip(self.n_ip, self.n_dir)
                ]
                sims = [
                    _tf.einsum("ab,sbc->sac", a, _tf.matmul(b, c)) + d
                    for a, b, c, d in zip(cov_cross, self.chol_r, rnd, mu)
                ]

                w_sims = _tf.reduce_sum(_tf.stack(sims, axis=0) * weights[:, :, :, None], axis=0)

                return w_mu, w_var, w_sims, w_exp_var, w_inf

            else:
                return w_mu, w_var
