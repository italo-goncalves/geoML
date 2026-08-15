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
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# __all__ = ["RealParameter",
#            "PositiveParameter",
#            "CompositionalParameter",
#            "CircularParameter"]

# import geoml.tftools as _tftools

import tensorflow as _tf
import numpy as _np
import pickle as _pickle
import functools as _functools

import geoml.stats.random as _rnd


def describe(obj, depth=1, **extra):
    """
    Writes an object as the call that would build it: `Class(arg, kw=value)`.

    Reads the arguments recorded by `Parametric.__init_subclass__`, so it works
    for any subclass without one of its own. Values are shortened — an array
    becomes its shape, a data container its size — and nesting stops at `depth`,
    so this stays cheap no matter how large the objects behind the arguments
    are. Anything given in `extra` is added unless the object was built with an
    argument of that name.

    A node's parents are left out: they belong to the composition, which `str`
    lays out, and nesting one node's arguments inside the next one's says the
    same thing far less clearly.

    An object holding a few small parameters is written with their *current*
    values instead — `Isotropic(range=111.0)`, not the range it was built with,
    which training has long since moved. Objects whose parameters are many or
    large (a GP node's inducing values) fall back to their arguments, since
    those values belong in `str`, not on one line.
    """
    values = _current_values(obj)
    if values is not None:
        return "%s(%s)" % (obj.__class__.__name__, ", ".join(values))

    skip = set()
    parent = getattr(obj, "parent", None)
    if parent is not None:
        skip.add(id(parent))
    for node in getattr(obj, "parents", None) or ():
        skip.add(id(node))

    kwargs = dict(getattr(obj, "_init_kwargs", {}))
    for name, value in extra.items():
        kwargs.setdefault(name, value)

    parts = [_brief(value, depth)
             for value in getattr(obj, "_init_args", ())
             if id(value) not in skip]
    parts += ["%s=%s" % (name, _brief(value, depth))
              for name, value in kwargs.items() if id(value) not in skip]
    return "%s(%s)" % (obj.__class__.__name__, ", ".join(parts))


def _current_values(obj):
    """The object's own parameters, if few and small enough to fit on a line.

    The shapes are read from `RealParameter.shape`, so a large parameter is
    ruled out without touching its value.
    """
    parameters = getattr(obj, "parameters", None)
    if not parameters or len(parameters) > 4:
        return None
    if any(int(_np.prod(p.shape)) > 4 for p in parameters.values()):
        return None
    return ["%s=%s" % (name, _short_value(parameter))
            for name, parameter in parameters.items()]


def _brief(value, depth):
    """A constructor argument, short enough to sit on one line."""
    if isinstance(value, Parametric):
        if depth <= 0:
            return value.__class__.__name__ + "(...)"
        return describe(value, depth - 1)
    if isinstance(value, _np.ndarray):
        if value.size <= 4:
            return _np.array2string(value, precision=4, separator=", ")
        return "array(shape=%s)" % (value.shape,)
    if isinstance(value, (list, tuple)):
        if len(value) > 3:
            return "[%d items]" % len(value)
        inner = ", ".join(_brief(item, depth) for item in value)
        return "[%s]" % inner if isinstance(value, list) else "(%s)" % inner
    if isinstance(value, float):
        return "%g" % value
    if hasattr(value, "n_data"):
        return "%s(n_data=%s)" % (value.__class__.__name__, value.n_data)
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _short_value(parameter):
    """A parameter's value: in full when small, summarized when not."""
    value = parameter.get_value().numpy()
    if value.ndim == 0:
        return "%g" % value
    if value.size <= 6:
        return _np.array2string(value, precision=4, separator=", ")
    return "shape %s in [%.4g, %.4g]" % (
        value.shape, value.min(), value.max())


class Parametric(object):
    """An abstract class for objects with trainable parameters"""

    def __init_subclass__(cls, **kwargs):
        """Records the arguments each subclass is constructed with.

        A trained model is saved by writing down how it was built and rebuilding
        it with the same calls (see the `persistence` module), so every object
        must remember its own arguments. Doing it here keeps that bookkeeping
        out of the constructors themselves.
        """
        super().__init_subclass__(**kwargs)

        init = cls.__dict__.get("__init__")
        if init is None:
            # this class inherits an already wrapped initializer
            return

        @_functools.wraps(init)
        def wrapped_init(self, *args, **kwargs):
            # A subclass initializer runs before the `super().__init__()` it
            # calls, so the first one to arrive is the one that was actually
            # invoked; the inner calls must not overwrite it.
            if "_init_args" not in self.__dict__:
                self._init_args = args
                self._init_kwargs = kwargs
            init(self, *args, **kwargs)

        cls.__init__ = wrapped_init

    def __init__(self):
        self.parameters = {}
        self._all_parameters = []
        self._param_ids = set()  # identities already in _all_parameters
        self._children = []      # registered sub-objects, in the order given

    def _summary_line(self):
        """The object's own line in `pretty_print`. Subclasses add to it."""
        return self.__class__.__name__

    def pretty_print(self, depth=0):
        """The object's parameters and everything registered inside it."""
        s = "  " * depth + self._summary_line() + "\n"
        for name, parameter in self.parameters.items():
            s += "  " * (depth + 1) + name + ": " + _short_value(parameter)
            if parameter.fixed:
                s += " (fixed)"
            s += "\n"
        for child in self._children:
            s += child.pretty_print(depth + 1)
        return s

    def __str__(self):
        return self.pretty_print()

    def __repr__(self):
        return describe(self)

    @property
    def all_parameters(self):
        return self._all_parameters

    def _append_unique(self, parameter):
        # A parameter shared between two children (e.g. a transform reused by
        # several kernels) would otherwise appear more than once in the flat
        # list, inflating get_parameter_values / save_state and the reported
        # degrees of freedom. Keep each identity once.
        if id(parameter) not in self._param_ids:
            self._param_ids.add(id(parameter))
            self._all_parameters.append(parameter)

    def _add_parameter(self, name, parameter):
        self.parameters[name] = parameter
        self._append_unique(parameter)

    def _register(self, parametric):
        for parameter in parametric.all_parameters:
            self._append_unique(parameter)
        if not any(child is parametric for child in self._children):
            self._children.append(parametric)
        return parametric

    def _set_parameters(self):
        pass

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

    def get_unfixed_variables(self):
        unique_params = list(set(self._all_parameters))
        model_variables = [pr.variable for pr in unique_params
                           if not pr.fixed]
        return model_variables

    def log_prior(self):
        """The summed prior log-density of every unfixed parameter here.

        The MAP half of the objective: parameters that declare a prior stay
        point estimates, and this term joins the ELBO next to the KL, making
        the whole a lower bound on `log p(y, theta)` rather than
        `log p(y | theta)`. Almost every parameter declares none and
        contributes nothing, so a model without priors trains on exactly the
        objective it always did. A fixed parameter is left out: its term
        would be a constant with no gradient, polluting the reported value
        of the objective without informing anything.

        `_all_parameters` is already unique by identity (`_append_unique`)
        and its order is the registration order, so the sum is deterministic.
        """
        terms = [parameter.log_prior() for parameter in self._all_parameters
                 if not parameter.fixed and parameter.prior is not None]
        if not terms:
            return _tf.constant(0.0, _tf.float64)
        return _tf.add_n(terms)


class RealParameter(object):
    """
    Trainable model parameter. Can be a vector, matrix, or scalar.

    The `fixed` property applies to the array as a whole.
    """
    def __init__(self, value, min_val, max_val, fixed=False,
                 name="Parameter", prior=None):
        self.name = name
        self.fixed = fixed

        # A prior distribution over the parameter's value (any object with a
        # `log_prob`, in practice a float64 tensorflow-probability
        # distribution), or None. The parameter stays a point estimate: the
        # prior's log-density joins the training objective next to the KL
        # (see `Parametric.log_prior`), which is MAP estimation rather than
        # variational inference, and costs one differentiable term. Read
        # when the training graph traces, so it must be in place before the
        # first call to train.
        self.prior = prior

        value = _np.array(value)
        min_val = _np.array(min_val)
        max_val = _np.array(max_val)

        if not max_val.shape == value.shape:
            raise ValueError(
                "Shape of max_val do not match shape of value: expected %s "
                "and found %s" % (str(value.shape),
                                  str(max_val.shape)))

        if not min_val.shape == value.shape:
            raise ValueError(
                "Shape of min_val do not match shape of value: expected %s "
                "and found %s" % (str(value.shape),
                                  str(min_val.shape)))

        self.shape = value.shape

        self.variable = _tf.Variable(self._transform(value),
                                     dtype=_tf.float64, name=name)

        self.max_transformed = _tf.Variable(
            self._transform(max_val), dtype=_tf.float64)
        self.min_transformed = _tf.Variable(
            self._transform(min_val), dtype=_tf.float64)

        self.refresh()

    def __repr__(self):
        s = "%s(%s" % (self.__class__.__name__, _short_value(self))
        if self.name != "Parameter":
            s += ", name=%r" % self.name
        if self.fixed:
            s += ", fixed=True"
        return s + ")"

    def _transform(self, x):
        return x

    def _back_transform(self, x):
        return x

    def fix(self):
        self.fixed = True

    def unfix(self):
        self.fixed = False

    def log_prior(self):
        """The prior's log-density at the current value, as one number.

        Zero when the parameter carries no prior, which is the default
        everywhere. Differentiable through `get_value`, so the gradient of
        the objective pulls the parameter toward the prior's mass exactly as
        hard as the density says to.
        """
        if self.prior is None:
            return _tf.constant(0.0, _tf.float64)
        return _tf.reduce_sum(self.prior.log_prob(self.get_value()))

    def set_limits(self, min_val=None, max_val=None):
        if min_val is not None:
            self.min_transformed.assign(self._transform(min_val))

        if max_val is not None:
            self.max_transformed.assign(self._transform(max_val))
        self.refresh()

    def set_value(self, value, transformed=False):
        if transformed:
            self.variable.assign(value)
        else:
            self.variable.assign(self._transform(value))
        self.refresh()

    def get_value(self):
        return self._back_transform(self.variable)

    def refresh(self):
        value = _tf.maximum(self.min_transformed,
                            _tf.minimum(self.max_transformed, self.variable))
        self.variable.assign(value)

    def randomize(self):
        val = (self.variable - self.min_transformed) \
              / (self.max_transformed - self.min_transformed)
        val = val + _rnd.rng().uniform(size=self.shape, low=-0.05, high=0.05)
        val = _tf.maximum(0, _tf.minimum(1, val))
        val = val * (self.max_transformed - self.min_transformed) \
              + self.min_transformed
        self.variable.assign(val)


class PositiveParameter(RealParameter):
    """Parameter in log scale"""

    def _transform(self, x):
        return _tf.math.log(_tf.cast(x, _tf.float64))

    def _back_transform(self, x):
        return _tf.math.exp(x)


class CompositionalParameter(RealParameter):
    """
    A vector parameter in logit coordinates
    """
    def __init__(self, value, fixed=False, name="Parameter"):
        super().__init__(value, value, value, fixed, name=name)
        self.min_transformed.assign(-10 * _tf.ones_like(self.variable))
        self.max_transformed.assign(10 * _tf.ones_like(self.variable))
        self.variable.assign(self._transform(value))

    def _transform(self, x):
        x_tr = _tf.math.log(_tf.cast(x, _tf.float64))
        return x_tr - _tf.reduce_mean(x_tr)

    def _back_transform(self, x):
        return _tf.nn.softmax(x)


class CircularParameter(RealParameter):
    def refresh(self):
        amp = self.max_transformed - self.min_transformed
        n_laps = _tf.floor((self.variable - self.min_transformed) / amp)
        value = self.variable - n_laps * amp
        self.variable.assign(value)


class UnitColumnNormParameter(RealParameter):
    def __init__(self, value, min_val, max_val, fixed=False, name="Parameter"):
        value = _np.array(value)
        if len(value.shape) != 2:
            raise ValueError("value must be rank 2")
        super().__init__(value, min_val, max_val, fixed, name)

    def refresh(self):
        value = self.get_value()
        normalized = value / (_tf.math.reduce_euclidean_norm(
            value, axis=0, keepdims=True) + 1e-6)
        self.variable.assign(normalized)


class CenteredUnitColumnNormParameter(RealParameter):
    def __init__(self, value, min_val, max_val, fixed=False, name="Parameter"):
        value = _np.array(value)
        if len(value.shape) != 2:
            raise ValueError("value must be rank 2")
        super().__init__(value, min_val, max_val, fixed, name)

    def refresh(self):
        value = self.get_value()
        value = value - _tf.reduce_mean(value, axis=1, keepdims=True)
        normalized = value / (_tf.math.reduce_euclidean_norm(
            value, axis=0, keepdims=True) + 1e-6)
        self.variable.assign(normalized)


class UnitColumnSumParameter(RealParameter):
    def __init__(self, value, fixed=False, name="Parameter"):
        value = _np.array(value)
        if len(value.shape) != 2:
            raise ValueError("value must be rank 2")
        super().__init__(value, value, value, fixed, name)
        self.min_transformed.assign(-100 * _tf.ones_like(self.variable))
        self.max_transformed.assign(100 * _tf.ones_like(self.variable))
        self.variable.assign(self._transform(value))

    def _transform(self, x):
        x_tr = _tf.math.log(_tf.cast(x, _tf.float64))
        return x_tr - _tf.reduce_mean(x_tr, axis=0, keepdims=True)

    def _back_transform(self, x):
        return _tf.nn.softmax(x, axis=0)


class OrthonormalMatrix(RealParameter):
    def __init__(self, rows, cols, batch_shape=(),
                 fixed=False, name="Parameter"):
        if cols > rows:
            raise ValueError("cols cannot be higher than rows")
        # rnd = _tf.random.stateless_normal(batch_shape + (rows, cols),
        #                                   seed=[rows, cols])
        rnd = _rnd.rng().normal(size=batch_shape + (rows, cols))
        q, _ = _tf.linalg.qr(rnd)
        value = q.numpy()
        min_val = -1.1 * _np.ones_like(value)
        max_val = 1.1 * _np.ones_like(value)
        super().__init__(value, min_val, max_val, fixed, name)

    def refresh(self):
        value = self.get_value()
        norm = _tf.math.reduce_euclidean_norm(value, axis=0, keepdims=True)
        value = value / (norm + 1e-6)
        q, _ = _tf.linalg.qr(value)
        # q = q * norm
        self.variable.assign(q)


class CenteredOrthonormalMatrix(RealParameter):
    def __init__(self, rows, cols, batch_shape=(),
                 fixed=False, name="Parameter"):
        if cols > rows:
            raise ValueError("cols cannot be higher than rows")
        rnd = _rnd.rng().normal(size=batch_shape + (rows, cols))
        rnd = rnd - _tf.reduce_mean(rnd, axis=-2, keepdims=True)
        q, _ = _tf.linalg.qr(rnd)
        value = q.numpy()
        min_val = -1.1 * _np.ones_like(value)
        max_val = 1.1 * _np.ones_like(value)
        super().__init__(value, min_val, max_val, fixed, name)

    def refresh(self):
        value = self.get_value()
        value = value - _tf.reduce_mean(value, axis=-2, keepdims=True)
        norm = _tf.math.reduce_euclidean_norm(value, axis=0, keepdims=True)
        value = value / (norm + 1e-6)
        q, _ = _tf.linalg.qr(value)
        # q = q * norm
        self.variable.assign(q)
