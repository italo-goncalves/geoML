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
import collections as _collections
import contextlib as _contextlib
import warnings as _warnings
import zlib as _zlib
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

# Whether a GP node reads an uncertain input through the expected kernel,
# its parent handing on the covariance between locations, or through the
# marginal rule of the versions before 0.9.0. Set with the rule above, from
# `GPOptions.propagation`, and read at trace time like it.
_JOINT_PROPAGATION = False


@_contextlib.contextmanager
def propagation_rule(rule, joint=False):
    """Chooses how experts propagate their inducing sets, and whether
    uncertainty travels with its covariance between locations, while
    active."""
    global _EXPERT_PROPAGATION, _JOINT_PROPAGATION
    previous = _EXPERT_PROPAGATION, _JOINT_PROPAGATION
    _EXPERT_PROPAGATION = rule
    _JOINT_PROPAGATION = bool(joint)
    try:
        yield
    finally:
        _EXPERT_PROPAGATION, _JOINT_PROPAGATION = previous


# Which experts the GP nodes compute, or None for all of them. Set through
# `expert_subset` by the model's expert-by-expert training and prediction,
# never directly; read at trace time, so it keys every traced function that
# reads it, as the propagation rule does. A tuple applies to every input; a
# dict maps an input's id to its own, for a network on several inputs.
_EXPERT_SUBSET = None


@_contextlib.contextmanager
def expert_subset(experts):
    """Computes only the given experts, by index, while active: a sequence
    for every input, or a mapping from an input node to its experts."""
    global _EXPERT_SUBSET
    previous = _EXPERT_SUBSET
    if experts is None:
        _EXPERT_SUBSET = None
    elif isinstance(experts, dict):
        _EXPERT_SUBSET = {id(root): tuple(sorted(int(e) for e in subset))
                          for root, subset in experts.items()}
    else:
        _EXPERT_SUBSET = tuple(sorted(int(e) for e in experts))
    try:
        yield
    finally:
        _EXPERT_SUBSET = previous


def _subset_of(root):
    """The experts `root`'s GP nodes compute under a subset, or None."""
    if _EXPERT_SUBSET is None or root is None:
        return None
    if isinstance(_EXPERT_SUBSET, dict):
        return _EXPERT_SUBSET.get(id(root))
    return _EXPERT_SUBSET


def _subset_key():
    """What the subset adds to a trace's key."""
    if isinstance(_EXPERT_SUBSET, dict):
        return tuple(sorted(_EXPERT_SUBSET.items()))
    return _EXPERT_SUBSET


def _active_experts(root):
    """The indices of the experts `root`'s GP nodes compute: all, or the
    subset."""
    subset = _subset_of(root)
    return tuple(range(root.n_experts)) if subset is None else subset


class _Slots:
    """The experts an expert-by-expert pass computes, as a fixed number of
    slots, so that one traced function serves every set of experts.

    `ids` holds the expert in each slot -- the padding expert, numbered
    `n_experts`, in a slot left empty -- and `mask` one where a slot holds
    a real expert: tensors when they are a traced step's arguments,
    Variables when a prediction's cached traces read them. `stacks`, when
    given, maps each GP node's id to the raw values of its own
    parameters (`alpha_white`, `delta`, `bias`) stacked over the experts
    and padded, which training steps in place of the parameters; otherwise
    they are stacked from the parameters."""

    def __init__(self, size, ids, mask, stacks=None):
        self.size = int(size)
        self.ids = ids
        self.mask = mask
        self.stacks = stacks


# The slots an expert-by-expert pass computes, or None. Set through
# `expert_slots` by the model, never directly. Read at trace time like the
# subset, but the experts in the slots are tensors, so only the number of
# slots keys a trace. One `_Slots` applies to every input; a dict maps an
# input's id to its own.
_EXPERT_SLOTS = None


@_contextlib.contextmanager
def expert_slots(slots):
    """Computes only the experts in `slots` while active: a `_Slots` for
    every input, or a mapping from an input node to its own."""
    global _EXPERT_SLOTS
    previous = _EXPERT_SLOTS
    _EXPERT_SLOTS = {id(root): value for root, value in slots.items()} \
        if isinstance(slots, dict) else slots
    try:
        yield
    finally:
        _EXPERT_SLOTS = previous


def _slots_of(root):
    """The slots `root`'s nodes compute under, or None."""
    if _EXPERT_SLOTS is None or root is None:
        return None
    if isinstance(_EXPERT_SLOTS, dict):
        return _EXPERT_SLOTS.get(id(root))
    return _EXPERT_SLOTS


def _slots_key():
    """What the active slots add to a trace's key: their number, per
    input."""
    if _EXPERT_SLOTS is None:
        return None
    if isinstance(_EXPERT_SLOTS, dict):
        return ("slots",) + tuple(sorted(
            (key, value.size) for key, value in _EXPERT_SLOTS.items()))
    return ("slots", _EXPERT_SLOTS.size)


def padded_inducing_points(root):
    """An input's inducing points stacked over its experts and padded to the
    largest set, with one padding expert after the last:
    `[n_experts + 1, m, d]`, and the mask of the real points
    `[n_experts + 1, m]`. A padded entry repeats its expert's first point,
    so the transform only ever sees places it has seen."""
    cached = root.__dict__.get("_padded_points")
    if cached is None:
        base = [_np.asarray(p) for p in root.base_inducing_points]
        m = max(len(p) for p in base)
        points = _np.zeros([len(base) + 1, m, base[0].shape[1]])
        mask = _np.zeros([len(base) + 1, m])
        for i, p in enumerate(base):
            points[i, :len(p)] = p
            points[i, len(p):] = p[0]
            mask[i, :len(p)] = 1.0
        points[-1] = base[0][0]
        cached = (points, mask)
        root._padded_points = cached
    return cached


def _slot_inputs(root):
    """The input's own points for its active slots, before the transform,
    flattened over the slots: `[slots * m, d]`."""
    points, _ = padded_inducing_points(root)
    base = _tf.gather(_tf.constant(points, _tf.float64),
                      _slots_of(root).ids)
    return _tf.reshape(base, [-1, points.shape[2]])


def _slot_mask(root):
    """Which of the active slots' inducing points are real: `[slots, m]`."""
    _, mask = padded_inducing_points(root)
    return _tf.gather(_tf.constant(mask, _tf.float64), _slots_of(root).ids)


def _slot_points(node):
    """The inducing points `node` hands on under slots, `[slots, m, d]`, and
    their variances. Under slots every node holds them as one tensor
    flattened over the slots -- a one-element tuple -- so that a node
    working row by row (a `Linear`, a `Bias`, a `SelectInput`) needs nothing
    of its own."""
    m = padded_inducing_points(node.root)[0].shape[1]
    size = _slots_of(node.root).size
    return (_tf.reshape(node.inducing_points[0], [size, m, -1]),
            _tf.reshape(node.inducing_points_variance[0], [size, m, -1]))


def _points_of(node, ids):
    """The inducing points `node` hands on for the experts `ids`, in that
    order, and their variances. A node holds every expert's -- the input,
    and whatever is computed from the input alone -- or, refreshed under a
    subset downstream of a GP node, the active experts' only, in their
    order; the number held tells which. (Read off the node rather than
    stamped on it: a cached refresh replays its trace without running this
    code.)"""
    if len(node.inducing_points) == node.root.n_experts:
        positions = ids
    else:
        subset = _subset_of(node.root)
        positions = [subset.index(i) for i in ids]
    return ([node.inducing_points[p] for p in positions],
            [node.inducing_points_variance[p] for p in positions])


def _aligned_points(parents, root):
    """The parents' inducing points, expert by expert: one row per expert
    the node will hold -- every one, the active ones under a subset, or
    under slots the one tensor flattened over them -- and in each row one
    `(points, variances)` per parent."""
    if _slots_of(root) is not None:
        return [[(p.inducing_points[0], p.inducing_points_variance[0])
                 for p in parents]]
    ids = _active_experts(root)
    held = [_points_of(p, ids) for p in parents]
    return [[(h[0][k], h[1][k]) for h in held] for k in range(len(ids))]


# --------------------------------------------------------------------------- #
# what a node hands its children
# --------------------------------------------------------------------------- #
class _Joint(_collections.namedtuple("_Joint", "mean variance covariance")):
    """One expert's chain at the data, under the expected kernel.

    `mean` is `[n, size]`; `variance` the same, or None where the node is
    certain; `covariance` `[n, m, size]`, between the node's outputs at the
    data and at the expert's `m` inducing points, or None where there is
    none. Under slots a leading slot axis comes first -- of length one
    where every slot holds the same, as an input's output does -- and the
    inducing points are the slot's, padded.

    Each output is taken as independent of the others: a node mixing its
    parent's outputs (`Linear`, `LinearCombination`) gives each output the
    covariance its rule gives a variance, and drops the covariance between
    outputs it creates.
    """


class _Second(_collections.namedtuple("_Second", "left jitter")):
    """What the second moment adds to an expert's moments: `left`, the
    variance its inducing points leave, `1 - explained` -- what the experts
    are weighted by, the variance less the spread of the mean -- and
    `jitter`, the latent variance its realizations leave out, or None where
    it is not asked for."""


class _Moments(tuple):
    """What `propagate` hands on: ``(mean, variance)``, each `[n, size]` and
    blended over the experts, unpacking as the pair it always was -- and
    `experts`, one `_Joint` per active expert (one for all of them under
    slots), where a GP node under the expected kernel reads this node's
    output, and None elsewhere."""

    def __new__(cls, mean, variance, experts=None):
        moments = super().__new__(cls, (mean, variance))
        moments.experts = experts
        return moments

    @property
    def mean(self):
        return self[0]

    @property
    def variance(self):
        return self[1]


class _Predicted(tuple):
    """What `predict` returns: ``(mu, var, sims, explained_var)``,
    unpacking as the four it always was, and `jitter`, `[size, n]` or
    None: the latent variance the realizations leave out because a GP
    node's input is uncertain (see `BasicGP._mixture_moments`), which a
    likelihood integrates out beside its noise."""

    def __new__(cls, mu, var, sims, explained_var, jitter=None):
        predicted = super().__new__(cls, (mu, var, sims, explained_var))
        predicted.jitter = jitter
        return predicted


def _jitters(parents):
    """The parents' latent jitters, `[size, n]` each, zeros for a parent
    with none -- or None where no parent has any."""
    held = [p._input_jitter for p in parents]
    if all(j is None for j in held):
        return None
    return [_tf.zeros_like(p._explained_var) if j is None else j
            for p, j in zip(parents, held)]


def _feeds_gp(node):
    """Whether a GP node reads this node's output: directly, or through
    nodes that pass the inducing points on."""
    for child in node.children:
        if isinstance(child, _GPNode):
            return True
        if child.propagates_inducing_points and _feeds_gp(child):
            return True
    return False


def _wants_joint(node):
    """Whether `node` hands its children the expected kernel's chain."""
    return _JOINT_PROPAGATION and _feeds_gp(node)


def _certain(experts):
    """Whether a chain carries no uncertainty at all, as an input's does:
    a GP node above it takes the plain kernel."""
    return all(e.variance is None and e.covariance is None for e in experts)


def _held_by_all(node, mean, variance=None):
    """The chain of an output that no expert gives -- an input's: the same
    for every expert, with no covariance with the inducing points."""
    if _slots_of(node.root) is not None:
        return (_Joint(mean[None], None if variance is None
                       else variance[None], None),)
    return tuple(_Joint(mean, variance, None)
                 for _ in _active_experts(node.root))


def _map_chain(experts, mean, spread):
    """Each expert's chain through a node acting output by output: `mean`
    applied to the means, `spread` to the variances and the covariances."""
    if experts is None:
        return None
    return tuple(_Joint(mean(e.mean),
                        None if e.variance is None else spread(e.variance),
                        None if e.covariance is None
                        else spread(e.covariance))
                 for e in experts)


def _chain_of(moments):
    """The chain a parent's `propagate` handed on, or None."""
    return getattr(moments, "experts", None)


def _added(values, weights=None, power=1):
    """Tensors added up, each times its weight to `power` where weights
    are given (one per value, broadcasting over the last axis); a None is
    zero, and the sum is None where every value is."""
    kept = [v if weights is None else v * weights[i] ** power
            for i, v in enumerate(values) if v is not None]
    if not kept:
        return None
    return _tf.add_n(_broadcast_all(kept))


def _side_by_side(values, sizes):
    """Tensors joined along their last axis, one per parent of the sizes
    given; a None is zero, shaped as the others but for its own size, and
    the result is None where every value is."""
    if all(v is None for v in values):
        return None
    shape = _tf.shape(next(v for v in values if v is not None))
    filled = [_tf.zeros(_tf.concat([shape[:-1], [size]], 0), _tf.float64)
              if v is None else v for v, size in zip(values, sizes)]
    return _tf.concat(_broadcast_leading(filled), axis=-1)


def _combined_covariances(rows, combine):
    """A node's covariances at the inducing points from its parents', one
    row per expert, `combine` applied to each: None where no parent holds
    any."""
    if all(c is None for row in rows for c in row):
        return None
    return tuple(combine(row) for row in rows)


def _sum_chains(chains, weights=None):
    """The chains of several parents added, expert by expert -- each times
    its weight and its spreads times the square, where weights are given --
    the parents taken as independent, as their variances are."""
    if any(c is None for c in chains):
        return None
    return tuple(_Joint(_added([p.mean for p in parts], weights, 1),
                        _added([p.variance for p in parts], weights, 2),
                        _added([p.covariance for p in parts], weights, 2))
                 for parts in zip(*chains))


def _broadcast_all(values):
    """Tensors brought to one shape, for adding up."""
    if len(values) < 2:
        return values
    shape = _tf.shape(values[0])
    for v in values[1:]:
        shape = _tf.broadcast_dynamic_shape(shape, _tf.shape(v))
    return [_tf.broadcast_to(v, shape) for v in values]


def _joined_chains(chains, sizes):
    """The chains of several parents side by side, output after output,
    as `Concatenate` joins them; a part one parent lacks is zero."""
    if any(c is None for c in chains):
        return None
    return tuple(_Joint(
        _tf.concat(_broadcast_leading([p.mean for p in parts]), axis=-1),
        _side_by_side([p.variance for p in parts], sizes),
        _side_by_side([p.covariance for p in parts], sizes))
        for parts in zip(*chains))


def _broadcast_leading(values):
    """Tensors brought to one shape on every axis but the last, which is
    what they are to be joined along."""
    if len(values) < 2:
        return values
    shape = _tf.shape(values[0])[:-1]
    for v in values[1:]:
        shape = _tf.broadcast_dynamic_shape(shape, _tf.shape(v)[:-1])
    return [_tf.broadcast_to(v, _tf.concat([shape, _tf.shape(v)[-1:]], 0))
            for v in values]


def _covariances_of(node, ids):
    """The covariances between `node`'s outputs at the inducing points of
    the experts `ids`, `[m, m, size]` each, or None for each where the node
    holds none (see `_points_of`)."""
    held = node.inducing_points_covariance
    if held is None:
        return [None] * len(ids)
    if len(held) == node.root.n_experts:
        positions = ids
    else:
        subset = _subset_of(node.root)
        positions = [subset.index(i) for i in ids]
    return [held[p] for p in positions]


def _aligned_covariances(parents, root):
    """`_aligned_points` for the covariances: one row per expert the node
    will hold, one entry per parent, None where a parent holds none."""
    if _slots_of(root) is not None:
        return [[None if p.inducing_points_covariance is None
                 else p.inducing_points_covariance[0] for p in parents]]
    ids = _active_experts(root)
    held = [_covariances_of(p, ids) for p in parents]
    return [[h[k] for h in held] for k in range(len(ids))]


def _slot_covariance(node):
    """The covariances `node` hands on under slots, `[slots, m, m, size]`,
    or None (see `_slot_points`)."""
    if node.inducing_points_covariance is None:
        return None
    m = padded_inducing_points(node.root)[0].shape[1]
    size = _slots_of(node.root).size
    return _tf.reshape(node.inducing_points_covariance[0], [size, m, m, -1])


def _node_seed(seed, key):
    """The seed for one node's draw: the sweep's seed with the node's name
    folded into its second entry.

    Every draw in a sweep is handed the same seed, so without the fold two
    GP nodes of one size on one root drew the same numbers -- measured
    2026-09-08: the latent realizations of Jura's rock and metal leaves
    correlated at 0.995, component for component, a coupling nobody
    modelled. A node's name is numbered within its tree and replayed by a
    save, so the fold is stable across a reload; a CRC rather than `hash`,
    which Python salts per process. A node with no name yet (one drawn
    outside a model) keeps the bare seed.
    """
    if key is None:
        return seed
    digest = _zlib.crc32(str(key).encode("utf-8")) & 0x7FFFFFFF
    return [seed[0], seed[1] + digest]


def _simulation_normals(shape, seed, key=None):
    """Standard normals shaped `[size, n, n_sim]` for the posterior draws.

    Monte Carlo is a stateless draw. Under `simulation_rule(True)` the same
    numbers come instead from a seeded-scramble Sobol sequence pushed through
    the normal quantile: each simulation is one point of a `size * n`-
    dimensional sequence, so the ensemble covers the posterior evenly rather
    than by chance. `shape` and `seed` are Python values at trace time, which
    is what lets the points be computed once and embedded as a constant.
    Either way the numbers are fixed by the seed, so a value does not depend
    on the batch that computed it. `key` -- the drawing node's name -- is
    folded into the seed (`_node_seed`), so two nodes handed one seed draw
    different numbers. Under the Sobol rule a different scramble is not
    enough: SciPy's is a linear matrix scramble, whose leading bit stays a
    linear function of the base digits, so two scrambles of one sequence
    keep their points paired (measured: the leaves' latent realizations
    still correlated at 0.37). So the realizations are also put in an order
    of the node's own, drawn from the same seed: each node keeps its evenly
    spread set, and realization k of one node no longer sits beside
    realization k of another.
    """
    seed = _node_seed(seed, key)
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
    if key is not None:
        normals = normals[rng.permutation(n_sim)]
    return _tf.constant(
        normals.reshape([n_sim, size, n]).transpose([1, 2, 0]), _tf.float64)


# --------------------------------------------------------------------------- #
# the expected kernel
# --------------------------------------------------------------------------- #
# A GP node whose input is another node's uncertain output takes, under
# `GPOptions(propagation="joint")`, the covariance E[k(h(x), h(y))] over the
# input's joint distribution -- which needs the variance of the difference
# h(x) - h(y), so the covariance between the two locations as well as their
# variances. For the Gaussian kernel the expectation is closed. The kernels
# that are scale mixtures of Gaussians, k(d) = E_w[exp(-w d^2)], take it
# component by component, over a fixed set of components with positive
# weights: a positive sum of expected Gaussian kernels is a covariance
# whatever its nodes, so the inducing points' matrix stays positive definite,
# where nodes placed pair by pair (measured first: Laplace-Hermite about each
# pair's tilted measure) err by 7e-3 on the exponential and guarantee
# nothing. Derivation and measurements: `docs/expected-kernel.md`.

# the Matern family read through 8 fitted Gaussians each, sum_q w_q
# exp(-r_q d^2) over the distance in ranges, the weights summing to one so
# that a covariance's diagonal stays one: fitted by
# `docs/benchmarks/kernel_mixtures.py`, the largest error over [0, 6] ranges
# 5.1e-4 (exponential), 9.8e-6 (Matern32) and 9.1e-6 (Matern52) -- which
# bounds the expected kernel's at any range and any uncertainty, the
# expectation being linear in the mixture. Measured against the trapezoid
# over each kernel's own mixing measure they replace (52, 32 and 28
# components, 1e-7): 3.5 to 6.5 times fewer evaluations
_KERNEL_MIXTURES = {
    _kr.Exponential: (
        (0.99353925727621273, 3.3888193300513523, 12.918190801234061,
         57.032592822643089, 305.13453091903932, 2169.502626707987,
         24917.497245652659, 921566.41220288316),
        (0.10583563086085193, 0.30159085568923139, 0.278218876296772,
         0.16919913539835968, 0.086494459885879116, 0.039068092644517549,
         0.015136513506592337, 0.0044564357177960439)),
    _kr.Matern32: (
        (1.0880466624657978, 2.2363121733574021, 4.7231391881645335,
         10.67412121993307, 26.587831672090051, 76.258839280320188,
         275.93676370152588, 1671.1712412119864),
        (0.036588330279395669, 0.23342241322629709, 0.35523711836715716,
         0.24168940845966788, 0.09897435763757878, 0.027991020429147327,
         0.0054965623174983106, 0.00060078928325784677)),
    _kr.Matern52: (
        (1.2580597454939826, 2.4331415607782336, 4.8849517861830112,
         10.770000331335869, 27.79264531658427, 67.822283244787457,
         113.55171525815763, 14964.867586466944),
        (0.054071122242525964, 0.32679209607345494, 0.40374906109805481,
         0.1782403835619305, 0.034268505964559853, 0.0010506800096522499,
         0.0018281490507311728, 1.9990905547959023e-09)),
}
# the rational quadratic's components, on a grid that follows its trained
# `scale` (narrow and far up when it is large, long when it is small):
# 4e-5 or better for scales from 1e-3 to 100
_RQ_NODES = 48


def _expected_kernel_supported(kernel):
    """Whether a GP node can read an uncertain input with `kernel`: the
    Gaussian, and the scale mixtures of Gaussians."""
    return type(kernel) in (_kr.Gaussian, _kr.RationalQuadratic) \
        or type(kernel) in _KERNEL_MIXTURES


def _kernel_components(kernel):
    """`kernel` as a positive mixture of Gaussians in the distance in
    ranges, `constant + sum_q w_q exp(-r_q d^2)`: the rates and the weights
    as tensors, and the constant part -- one component for the Gaussian, the
    fitted table for the Matern family, the rational quadratic's grid."""
    if type(kernel) is _kr.Gaussian:
        return (_tf.constant([3.0], _tf.float64),
                _tf.constant([1.0], _tf.float64), 0.0)
    if type(kernel) is _kr.RationalQuadratic:
        omega, weights, below, _ = _rq_components(
            kernel.parameters["scale"].get_value())
        return omega, weights, below
    if type(kernel) in _KERNEL_MIXTURES:
        rates, weights = _KERNEL_MIXTURES[type(kernel)]
        return (_tf.constant(rates, _tf.float64),
                _tf.constant(weights, _tf.float64), 0.0)
    raise NotImplementedError(
        "the expected kernel takes the Gaussian kernel or a scale mixture of "
        "Gaussians (Exponential, Matern32, Matern52, RationalQuadratic); %s "
        "is neither" % type(kernel).__name__)


def _rq_components(alpha):
    """The rational quadratic's components at `scale` alpha, as tensors:
    w = 3 g / alpha, g ~ Gamma(alpha, 1), on a trapezoid in log g placed
    about the mode and wide enough for the tail a small alpha has."""
    q = _RQ_NODES
    centre = _tf.math.log(alpha)
    width = 8.0 / _tf.sqrt(alpha)
    lo = _tf.maximum(_tf.math.log(1e-7 * alpha / 3.0), centre - width)
    hi = _tf.minimum(_tf.math.log(1e9 * alpha / 3.0),
                     centre + _tf.maximum(4.0, width))
    step = (hi - lo) / (q - 1)
    y = lo + step * _tf.range(q, dtype=_tf.float64)
    ends = _tf.constant([0.5] + [1.0] * (q - 2) + [0.5], _tf.float64)
    weights = ends * step * _tf.exp(
        alpha * y - _tf.exp(y) - _tf.math.lgamma(alpha))
    below = _tf.math.igamma(alpha, _tf.exp(lo))
    above = _tf.math.igammac(alpha, _tf.exp(hi))
    weights = weights * (1.0 - below - above) / _tf.reduce_sum(weights)
    return 3.0 * _tf.exp(y) / alpha, weights, below, above


def _expected_kernel(kernel, ranges, mean_x, var_x, mean_y, var_y, cov=None,
                     gradient=False):
    """E[k(h(x), h(y))] over jointly Gaussian inputs.

    `mean_x` is `[..., n, d]` and `mean_y` `[..., m, d]`; the variances are
    shaped alike, or None for none; `cov` is the covariance between the two,
    `[..., n, m, d]`, or None for independent inputs. Each input dimension
    is taken as independent of the others. Returns `[..., n, m]`, and with
    `gradient` also its derivative in `mean_x` at fixed variances,
    `[..., n, m, d]` -- in closed form, component by component.
    """
    with _tf.name_scope("expected_kernel"):
        r2 = ranges ** 2
        dif = mean_x[..., :, None, :] - mean_y[..., None, :, :]
        dif2 = dif ** 2 / r2
        # the variance of the difference, in squared ranges
        v = _tf.zeros_like(dif2)
        if var_x is not None:
            v = v + var_x[..., :, None, :]
        if var_y is not None:
            v = v + var_y[..., None, :, :]
        if cov is not None:
            v = v - 2.0 * cov
        v = _tf.maximum(v / r2, 0.0)

        omega, weights, constant = _kernel_components(kernel)
        total = constant + _tf.zeros_like(dif2[..., 0])
        slope = _tf.zeros_like(dif2) if gradient else None
        for i in range(int(omega.shape[0])):
            w = omega[i]
            inflated = 1.0 + 2.0 * w * v
            term = weights[i] * _tf.exp(
                -0.5 * _tf.reduce_sum(_tf.math.log1p(2.0 * w * v), -1)
                - w * _tf.reduce_sum(dif2 / inflated, -1))
            total = total + term
            if gradient:
                slope = slope - term[..., None] * 2.0 * w * dif \
                    / (r2 * inflated)
        # a pair at one place with nothing uncertain between them, where
        # every component reads one, is one exactly, as the diagonal of the
        # inducing points' matrix must be, rather than the weights' sum to
        # rounding
        same = _tf.logical_and(_tf.equal(_tf.reduce_sum(dif2, -1), 0.0),
                               _tf.equal(_tf.reduce_sum(v, -1), 0.0))
        total = _tf.where(same, _tf.ones_like(total), total)
        return (total, slope) if gradient else total


def _kernel_items(kernel, ranges, weight=1.0):
    """`kernel` at `ranges` as the Gaussians the second moment pairs up:
    `(weight, rates)` per component, `weight * w_q exp(-sum rates d²)` over
    the offsets in the input's own units, `rates` `[d]` (or `[1]`)."""
    omega, weights, constant = _kernel_components(kernel)
    if not isinstance(constant, float) or constant != 0.0:
        raise NotImplementedError(
            "the second moment takes the Gaussian kernel and the Matern "
            "family's tables, not %s" % type(kernel).__name__)
    r2 = _tf.reshape(ranges, [-1]) ** 2
    return [(weight * weights[q], omega[q] / r2)
            for q in range(int(omega.shape[0]))]


def _second_moment_supported(kernel):
    """Whether a GP node reading an uncertain input with `kernel` takes the
    second moment: every kernel the expected kernel takes -- in closed form
    for the Gaussian, by quadrature over the input for the others."""
    return _expected_kernel_supported(kernel)


# the scale mixtures' second moment by quadrature over the input: in closed
# form a table of eight Gaussians pairs into 36 terms, each an [n, m, m]
# array, measured 100 times a first-moment training iteration at 100
# inducing points and out of 45 GB at 300 (Matern32, 1000 locations). On
# Walker Lake 32 nodes came within 4.2% of the mixture and 64 within 1.8%
# for the exponential and the Matern32 at every input variance tried
_QUADRATURE_NODES = 64
_QUADRATURE_SEED = 20261009
_QUADRATURE = {}


def _plain_distance(x, y, ranges):
    """The distance in ranges between certain points, `[..., n, m]`, through
    the expansion `|x|² + |y|² - 2 x yᵀ` -- one product of matrices, where
    the differences make an array with the dimensions on it, measured 3.7
    times slower with its gradient. The points are taken about the mean of
    `y`, so that coordinates far from the origin (a mine grid's) do not
    cancel away the digits a short distance needs."""
    scale = _tf.reshape(ranges, [-1])
    ys = y / scale
    origin = _tf.reduce_mean(ys, axis=-2, keepdims=True)
    xs, ys = x / scale - origin, ys - origin
    square = _tf.reduce_sum(xs ** 2, -1)[..., :, None] \
        + _tf.reduce_sum(ys ** 2, -1)[..., None, :] \
        - 2.0 * _tf.matmul(xs, ys, transpose_b=True)
    return _tf.sqrt(_tf.maximum(square, 1e-30))


def _input_nodes(dimension):
    """Standard normal points for an input of `dimension` coordinates,
    `[q, dimension]`: half of them scrambled Sobol through the normal
    quantile, the other half their negatives, the set whitened to unit
    covariance -- so that its first two moments are the Gaussian's exactly
    and the rule is exact for a quadratic (raw Sobol missed by 4.5% at small
    input variances, through a mean that is not quite zero); fixed by a
    seed of their own, so that a moment depends on nothing else."""
    if dimension not in _QUADRATURE:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            points = _rnd.sobol_engine(dimension, _QUADRATURE_SEED).random(
                _QUADRATURE_NODES // 2)
        half = _special.ndtri(_np.clip(points, 1e-6, 1 - 1e-6))
        nodes = _np.concatenate([half, -half])
        root = _np.linalg.cholesky(nodes.T @ nodes / len(nodes))
        _QUADRATURE[dimension] = _np.linalg.solve(root, nodes.T).T
    return _QUADRATURE[dimension]


def _second_moment(items, mean, var, points, weights):
    """`sum_ij W_kij E[k(x, z_i) k(x, z_j)]` over `x ~ N(mean, var)` and
    certain points `z`, the kernel given as `_kernel_items`.

    `mean` and `var` are `[..., n, d]`, `points` `[..., m, d]` and `weights`
    a list of stacks `[..., k, m, m]`, each matrix symmetric. Returns one
    `[..., k, n]` per stack -- separate contractions sharing each pair's
    exponential, so that a stack nothing reads costs nothing in a graph.

    Each pair of components `(a, b)` gives, per input dimension,
    `exp(-ab/(a+b) (z_i - z_j)²)` times the expectation of
    `exp(-(a+b) (x - z_ij)²)` about their weighted midpoint `z_ij`, which is
    `(1 + 2(a+b)var)^(-1/2) exp(-(a e_i + b e_j)² / ((a+b)(1 + 2(a+b)var)))`
    in the offsets `e = mean - z`. Expanded, the part tying `i`, `j` and the
    location together is one bilinear form in the offsets, so the whole
    exponent is one product of matrices, `[..., n, m, m]`, and never an
    array with the dimensions on it as well.
    """
    with _tf.name_scope("second_moment"):
        e = mean[..., :, None, :] - points[..., None, :, :]
        gaps = (points[..., :, None, :] - points[..., None, :, :]) ** 2
        ones = _tf.ones_like(e[..., :1])
        totals = [None] * len(weights)
        for p in range(len(items)):
            for q in range(p, len(items)):
                (wa, a), (wb, b) = items[p], items[q]
                c = a + b
                inflated = 1.0 + 2.0 * c * var                 # [..., n, d]
                scaled = e / (c * inflated)[..., :, None, :]
                # the bilinear form with the squares and the normalization
                # folded in as two more coordinates: left [2ab e/(c s), u + h,
                # 1], right [e, 1, v], so one product gives every term
                own = _tf.reduce_sum(a ** 2 * e * scaled, -1, keepdims=True) \
                    + 0.5 * _tf.reduce_sum(_tf.math.log(inflated), -1)[
                        ..., :, None, None]
                other = _tf.reduce_sum(b ** 2 * e * scaled, -1, keepdims=True)
                left = _tf.concat([2.0 * a * b * scaled, own, ones], -1)
                right = _tf.concat([e, ones, other], -1)
                exponent = _tf.matmul(left, right, transpose_b=True) \
                    + _tf.reduce_sum(a * b / c * gaps, -1)[..., None, :, :]
                moment = _tf.exp(-exponent)
                # the pair (b, a) is the transpose of (a, b), and the weights
                # are symmetric
                factor = wa * wb * (1.0 if p == q else 2.0)
                for k, w in enumerate(weights):
                    term = _tf.einsum("...nij,...kij->...kn", moment, w) \
                        * factor
                    totals[k] = term if totals[k] is None \
                        else totals[k] + term
        return totals


def _graph_state(node):
    """
    The attributes a node holds as tensors, which a traced refresh must return.

    An attribute written while a `tf.function` is tracing keeps a symbolic
    tensor, which is unusable once the trace is over. Reading them off the node
    rather than listing them per class means a new node needs nothing new here.
    """
    graph = _tf.compat.v1.get_default_graph()

    def usable(value):
        # eager, or written by this trace: a symbolic tensor another trace
        # left behind -- a training step's slots after a prediction of the
        # whole model, say -- belongs to a graph that is gone
        return isinstance(value, _tf.Tensor) and (
            not _tf.is_symbolic_tensor(value) or value.graph is graph)

    state = {}
    for name, value in vars(node).items():
        if name.startswith("_"):
            continue
        if usable(value):
            state[name] = value
        elif (isinstance(value, (tuple, list)) and len(value) > 0
                and all(usable(v) for v in value)):
            state[name] = tuple(value)
    return state


def refresh_cached(network, jitter=1e-6, owner=None):
    """
    Refreshes a network once and snapshots it for prediction.

    `refresh` is pure arithmetic over parameters that do not move during
    prediction, but running it eagerly pays Python overhead for each of the
    K x K covariance blocks a multi-expert network builds -- at 32 experts that
    is most of a `predict` call. Tracing it collapses those into one graph
    call. The trace is kept -- on `owner`, or on the node -- so predicting
    again does not rebuild it, and it reads the parameters live, so it also
    follows further training.

    Parameters
    ----------
    network
        The output node of a latent network, or a list of its leaves -- a
        model with one leaf per likelihood, whose leaves may share parents
        or sit on separate trees. Every node is refreshed and snapshotted
        once whichever way it is reached.
    jitter : float
        Small value added to the covariance matrices for numerical stability.
    owner
        Where to keep the trace. A list of leaves has no single node to hang
        it on, so the model passes itself.
    """
    leaves = list(network) if isinstance(network, (list, tuple)) \
        else [network]
    holder = owner if owner is not None else leaves[0]

    # the propagation rule and the expert subset are Python-level branches
    # inside `refresh`, so they are baked into the trace and key the cache
    key = (jitter, _EXPERT_PROPAGATION, _JOINT_PROPAGATION, _subset_key(),
           _slots_key())
    # every expert: one trace, replaced when the key changes; a subset of
    # them, or a number of slots: one trace each, kept, since a prediction
    # by expert visits several in turn and comes back to them
    subsets = None
    if _EXPERT_SUBSET is not None or _EXPERT_SLOTS is not None:
        subsets = holder.__dict__.setdefault("_subset_refresh_graphs", {})
        cached = subsets.get(key)
    else:
        cached = holder._refresh_graph
    if cached is None or cached[0] != key:
        # fixed once, so that the values coming back keep lining up with the
        # nodes they belong to; a parent two leaves share is listed once
        seen, nodes = set(), []
        for leaf in leaves:
            for node in [leaf] + leaf.get_unique_parents():
                if id(node) not in seen:
                    seen.add(id(node))
                    nodes.append(node)

        def traced():
            for leaf in leaves:
                leaf.refresh(jitter)
            return [_graph_state(node) for node in nodes]

        cached = (key, _tf.function(traced), nodes)
        if subsets is not None:
            subsets[key] = cached
        else:
            holder._refresh_graph = cached

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
        # under the expected kernel, the covariance between the outputs at
        # an expert's inducing points, `[m, m, size]` per expert (a tuple
        # like the points), or None where the outputs there are certain
        self.inducing_points_covariance = None

        # Non-trainable Variables holding a snapshot of the prediction state, so
        # a cached (tf.function) prediction graph reads current values instead of
        # baking them in at tracing time. Keyed by name, created on first use.
        self._state_vars = {}

        # The sweep state: `propagate` stamps, `simulate` draws, `predict`
        # pairs them. These are per-batch tensors alive only within one
        # traced call -- underscore-named on purpose, so `_graph_state`
        # never snapshots them across traces. Internal moment queries at
        # other locations (`interpolate`, a refresh) must never stamp.
        self._sim_state = None
        self._explained_var = None
        # the latent variance the realizations leave out where a GP node's
        # input is uncertain, `[size, n]`, carried up by the nodes acting
        # linearly and dropped by the others; None elsewhere
        self._input_jitter = None

    # Whether the node's output is Gaussian: True, False, or "parents",
    # Gaussian exactly when every parent is. The training quadrature reads
    # a leaf's mean and variance as a Gaussian's, so a model trains the
    # likelihood of a leaf that is not on its realizations instead.
    _GAUSSIAN = "parents"

    # Whether `simulate` hands parent i the seed `[seed[0] + i, seed[1]]`
    # rather than its own. `VGPNetwork.predict_node` replays the shifts
    # along the path from a leaf, so that a node's realization s is the
    # one the leaf's realization s was built from.
    _SHIFTS_PARENT_SEEDS = False

    @property
    def gaussian(self):
        """Whether the node's output is a Gaussian random variable."""
        if self._GAUSSIAN != "parents":
            return self._GAUSSIAN
        parents = getattr(self, "parents", None)
        if parents is None:
            parents = [self.parent]
        return all(p.gaussian for p in parents)

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

        See `geoml.viz.graphviz.to_dot`, which draws a whole model when given one.
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
        # by position, or, where the node holds a subset's active experts
        # only, by expert: each Variable keeps the one shape its expert
        # gives it
        root = getattr(self, "root", None)
        subset = _subset_of(root)
        ids = range(len(values)) if subset is None \
            or len(values) == root.n_experts else subset
        return tuple(self._state_var(name + "_" + str(i), v)
                     for i, v in zip(ids, values))

    def _cache_slot_state(self):
        """Under slots, snapshots the `slots_*` state and the inducing points
        handed on (one tensor flattened over the slots), named by the number
        of slots: their shapes are fixed by that number, so one prediction
        trace reads them whichever experts fill the slots."""
        prefix = "slots%d_" % _slots_of(self.root).size
        for name, value in list(vars(self).items()):
            if name.startswith("slots_") and isinstance(value, _tf.Tensor):
                setattr(self, name, self._state_var(prefix + name, value))
        for name in ("inducing_points", "inducing_points_variance",
                     "inducing_points_covariance"):
            value = getattr(self, name, None)
            if isinstance(value, tuple) and len(value) == 1 \
                    and isinstance(value[0], _tf.Tensor):
                setattr(self, name,
                        (self._state_var(prefix + name, value[0]),))

    def cache_prediction_state(self):
        """
        Snapshot the propagated state into Variables (see `_state_var`).

        Called once per prediction (after `refresh`) for every node in the
        network. Subclasses holding additional prediction state extend this.
        """
        if _slots_of(getattr(self, "root", None)) is not None:
            self._cache_slot_state()
            return
        if self.inducing_points is not None:
            self.inducing_points = self._cache_tuple(
                "inducing_points", self.inducing_points)
        if self.inducing_points_variance is not None:
            self.inducing_points_variance = self._cache_tuple(
                "inducing_points_variance", self.inducing_points_variance)
        if self.inducing_points_covariance is not None:
            self.inducing_points_covariance = self._cache_tuple(
                "inducing_points_covariance", self.inducing_points_covariance)

    def get_unique_parents(self):
        raise NotImplementedError

    def predict(self, x, x_var=None, n_sim=1, seed=(0, 0)):
        """
        Prediction on this node's latent variables.

        The one composer of the node protocol: `propagate` carries the
        moments (and stamps whatever its node's `simulate` will need),
        `simulate` draws from that state, and this method pairs them.

        Parameters
        ----------
        x : Tensor
            Mean of the input.
        x_var : Tensor
            Variance of the input.
        n_sim : int
            Number of simulations to draw. At least 1.
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
        """
        if n_sim < 1:
            raise ValueError(
                "n_sim must be at least 1: the moments alone come from "
                "propagate(), and simulations are what predict adds to them")
        mu, var = self.propagate(x, x_var)
        sims = self.simulate(n_sim, seed)
        mu = _tf.transpose(mu)[:, :, None]
        var = _tf.transpose(var)
        return _Predicted(mu, var, sims, self._explained_var,
                          self._input_jitter)

    def predict_directions(self, x, dir_x, step=1e-3):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError

    def propagate(self, x, x_var=None):
        """
        Propagates mean and variance to the next node.

        Also stamps the node's sweep state: `_explained_var` always, and
        `_sim_state` wherever the node's own `simulate` draws rather than
        transforms -- simulations originate at GP nodes and are carried
        pathwise by the operation nodes above them, so an operation node's
        `simulate` calls its parents' instead of reading a stash.

        Parameters
        ----------
        x : Tensor
            Mean of the input.
        x_var : Tensor
            Variance of the input.

        Returns
        -------
        _Moments
            The mean and the variance of the output, `[n, size]` each, which
            unpack as a pair; and, under the expected kernel and where a GP
            node reads the output, each expert's chain in `experts`: the
            output's moments as that expert alone gives them, and its
            covariance with the expert's inducing points.
        """
        raise NotImplementedError

    def simulate(self, n_sim, seed=(0, 0)):
        """
        Draws from the state the same sweep's `propagate` stamped.

        Parameters
        ----------
        n_sim : int
            Number of simulations to draw.
        seed : tuple
            A set of two seeds for the random number generator.

        Returns
        -------
        sims
            Simulations of shape `[size, n_data, n_sim]`.
        """
        raise NotImplementedError

    def _swept(self):
        """The stamped sim state, refusing to draw from a sweep that never
        ran. A stale stash from another trace fails on its own -- TensorFlow
        refuses tensors across graphs -- so the guard's job is the None."""
        if self._sim_state is None:
            raise RuntimeError(
                "%s.simulate() before propagate(): predict() pairs them -- "
                "simulate draws from the state the same sweep's propagate "
                "wrote" % self.name)
        return self._sim_state

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
    _GAUSSIAN = True

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
        if len(latent_variables) == 0:
            raise ValueError("%s needs at least one parent"
                             % type(self).__name__)
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
        # by identity, each kept where it was first met: a set of nodes
        # iterates by memory address, so the tree came out in a different
        # order in every process -- and that order is what the KL is summed
        # in (`VGPNetwork._nodes`)
        unique, seen = [], set()
        for node in all_parents:
            if id(node) not in seen:
                seen.add(id(node))
                unique.append(node)
        return unique

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
    _GAUSSIAN = True

    # Whether the node reads its parent's chain under the expected kernel.
    # `UncertainInputGP` integrates over its input's marginal itself.
    _READS_CHAIN = True

    def __init__(self, parent, name=None):
        super().__init__(parent, name=name)
        if not self.propagates_inducing_points:
            raise BrokenPropagationError(
                '%s: GP nodes require their parent to propagate inducing '
                'points, and %s does not.' % (self.name, parent.name))

    def _moments(self, x, x_var=None):
        """The conditional moments at already-propagated locations.

        Returns the per-expert internals (`cov_cross`, `mu`, the expert
        `weights`) alongside the weighted moments: the simulation draws
        combine the internals, not the weighted outputs, which is why
        `propagate` stamps them rather than its return values.
        """
        raise NotImplementedError

    def propagate(self, x, x_var=None):
        with _tf.name_scope("gp_prediction"):
            parent = self.parent.propagate(x, x_var)
            x, x_var = parent
            jitter = None
            if not self._READS_CHAIN:
                cov_cross, mu, weights, w_mu, w_var, w_exp_var = \
                    self._moments(x, x_var)
                experts = None
            else:
                # the parent's chain, where it carries any uncertainty;
                # otherwise the plain kernel, as before the expected kernel
                chain = _chain_of(parent) if _JOINT_PROPAGATION else None
                if chain is not None and _certain(chain):
                    chain = None
                cov_cross, mu, var, explained, second = \
                    self._expert_moments(x, x_var, chain, with_second=True)
                weights, w_mu, w_var, w_exp_var = self._blend(
                    mu, var, explained,
                    None if second is None else second.left)
                experts = self._chain(cov_cross, mu, var) \
                    if _wants_joint(self) else None
                if second is not None:
                    # blended as the variances are; nothing reads it in
                    # training, so the graph there prunes it
                    jitter = _tf.reduce_sum(
                        (_tf.stack(second.jitter, axis=0)
                         if isinstance(second.jitter, list)
                         else second.jitter) * weights, axis=0)
            self._sim_state = (cov_cross, mu, weights)
            self._explained_var = w_exp_var
            self._input_jitter = jitter
            return _Moments(_tf.transpose(w_mu[:, :, 0]),
                            _tf.transpose(w_var), experts)

    def simulate(self, n_sim, seed=(0, 0)):
        if _slots_of(self.root) is not None:
            return self._slot_simulate(n_sim, seed)
        cov_cross, mu, weights = self._swept()
        with _tf.name_scope("gp_simulation"):
            rnd = [
                _simulation_normals([self.size, self.root.n_ip[i], n_sim],
                                    seed, key=self.name)
                for i in _active_experts(self.root)
            ]
            sims = [
                _tf.einsum("ab,sbc->sac", a, _tf.matmul(b, c)) + d
                for a, b, c, d in zip(cov_cross, self.chol_r, rnd, mu)
            ]
            return _tf.reduce_sum(
                _tf.stack(sims, axis=0) * weights[:, :, :, None], axis=0)

    def interpolate(self, x, x_var=None):
        """Stash-free moments at arbitrary already-propagated locations.

        The door for internal queries -- `GPWalk`'s stepping asks the field
        at the walked coordinates -- which must never stamp the sweep state
        a shared node's `simulate` will read.
        """
        _, _, _, w_mu, w_var, _ = self._moments(x, x_var)
        return w_mu, w_var

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
    def __init__(self, inducing_points, transform=None,
                 fix_transform=False,
                 center=False, name=None):
        """
        Initializer for BasicInput.

        Parameters
        ----------
        inducing_points
            A `PointData` object, or a list of these objects.
        transform
            An object from the `transform` module for normalization; the
            identity if left out.
        fix_transform : bool
            Whether to fix the transform parameters to prevent them from changing during training.
        center : bool
            Whether to center the data, based on the inducing points' bounding box.
        name : str
            A name for this node, shown in the printed network and accepted by
            `get_node`. Numbered automatically if omitted.
        """
        super().__init__(name=name)
        if transform is None:
            transform = _tr.Identity()

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
        # kept, made once and eagerly: under slots the variances are swapped
        # for the slots' own, and a refresh puts these back -- never fresh
        # zeros, which a trace would leave behind as symbolic tensors
        self._zero_variances = self.inducing_points_variance

        # self.center = _np.zeros_like(transform(self.bounding_box.max.astype(_np.float64)))
        self.center = _np.zeros_like(self.bounding_box.max.astype(_np.float64))
        if center:
            self.center = 0.5 * (self.bounding_box.min + self.bounding_box.max)

    def get_root_inducing_points(self):
        # under slots the active slots' points, flattened over them
        if _slots_of(self) is not None:
            base = _slot_inputs(self)
            return (base,), (_tf.zeros([base.shape[0], self.size],
                                       _tf.float64),)
        return self.base_inducing_points, self._zero_variances

    def refresh(self, jitter=1e-6):
        with _tf.name_scope("basic_input_refresh"):
            self.transform.refresh()
            base, variance = self.get_root_inducing_points()
            self.inducing_points = tuple(self.transform(ip - self.center)
                                         for ip in base)
            self.inducing_points_variance = variance

    def propagate(self, x, x_var=None):
        x_tr = self.transform(x - self.center)
        self._sim_state = (x_tr,)
        self._explained_var = _tf.zeros_like(_tf.transpose(x_tr))
        return _Moments(x_tr, _tf.zeros_like(x_tr),
                        _held_by_all(self, x_tr) if _wants_joint(self)
                        else None)

    def simulate(self, n_sim, seed=(0, 0)):
        (x_tr,) = self._swept()
        x_t = _tf.transpose(x_tr)
        return _tf.tile(x_t[:, :, None], [1, 1, n_sim])

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def set_parameter_limits(self, data):
        self.transform.set_limits(data)

    def predict_directions(self, x, dir_x, step=1e-3):
        x_plus = self.transform(x - self.center + dir_x*step/2)
        x_minus = self.transform(x - self.center - dir_x * step / 2)

        mu = _tf.transpose((x_plus - x_minus) / step)
        return mu[:, :, None], _tf.zeros_like(mu), _tf.zeros_like(mu)


def _input_jacobian_squared(transform, x):
    """`(d x_tr_j / d x_i)^2` at every row of `x`, as [n, n_dim, size].

    One forward-mode pass per input coordinate -- never nested, which is the
    combination that crashes (see the kernels row of CLAUDE.md) -- squared,
    so that a per-coordinate input variance maps to the transformed space as
    the diagonal of `J diag(var) J^T`.
    """
    n_dim = x.shape[1]
    columns = []
    for i in range(n_dim):
        tangent = _tf.ones_like(x) * _tf.one_hot(i, n_dim, dtype=_tf.float64)
        with _tf.autodiff.ForwardAccumulator(x, tangent) as acc:
            x_tr = transform(x)
        columns.append(_tf.square(acc.jvp(x_tr)))
    return _tf.stack(columns, axis=1)


class GaussianInput(BasicInput):
    """
    Input node for uncertain inputs.

    Takes each input as a Gaussian -- a mean and a variance per coordinate,
    which is what a `GaussianData` container holds -- and hands the network
    both, where `BasicInput` hands it the mean alone. The case it is built
    for is a high-dimensional input with missing entries: an entry that is
    not known is given the mean and variance it could have, and the expected
    kernel the GP nodes carry integrates over it, so the row is used for what
    it says rather than dropped or imputed. An uncertain location in space is
    the same mechanism in three coordinates.

    The variance is carried through the transform as the diagonal of
    ``J diag(var) J^T``: exactly for an affine transform (the ellipsoids,
    projections, ARD, selections -- `transform.linear`), where the Jacobian
    is read off one probe point and applied as one matrix product, and to
    first order through a nonlinear one (`Periodic`, the faults), where it is
    measured at every point. Inducing points are exact. Given no variance the
    node is `BasicInput` to the last bit.

    Parameters
    ----------
    inducing_points
        A `PointData` object, or a list of these objects.
    transform
        An object from the `transform` module for normalization.
    fix_transform
        Whether to fix the transform parameters to prevent them from
        changing during training.
    center
        Whether to center the data, based on the inducing points' bounding
        box.
    name
        A name for this node, shown in the printed network and accepted by
        `get_node`. Numbered automatically if omitted.

    See Also
    --------
    BasicInput : the deterministic input.
    geoml.data.GaussianData : the container carrying a variance per
        coordinate.
    """
    def propagate(self, x, x_var=None):
        x_c = x - self.center
        x_tr = self.transform(x_c)
        if x_var is None:
            var_tr = _tf.zeros_like(x_tr)
        elif self.transform.linear:
            # an affine map has one Jacobian everywhere: read it off a
            # single probe point and the variance maps in one product. Read
            # here rather than stashed by `refresh`: a tensor made inside the
            # traced refresh cannot be used from another graph, and the
            # probe costs `n_dim` passes on one row
            probe = _tf.zeros([1, x.shape[1]], _tf.float64)
            jacobian_sq = _input_jacobian_squared(self.transform, probe)[0]
            var_tr = _tf.matmul(x_var, jacobian_sq)
        else:
            var_tr = _tf.reduce_sum(
                _input_jacobian_squared(self.transform, x_c)
                * x_var[:, :, None], axis=1)
        self._sim_state = (x_tr,)
        self._explained_var = _tf.zeros_like(_tf.transpose(x_tr))
        experts = None
        if _wants_joint(self):
            # each location's own uncertainty, unrelated to any other's
            # and to the inducing points, which are exact
            experts = _held_by_all(self, x_tr,
                                   None if x_var is None else var_tr)
        return _Moments(x_tr, var_tr, experts)


class Stack(_Operation):
    """
    Latent variable stacking.

    Consolidates a list of latent variables into a single object.
    """
    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        self._size = sum([p.size for p in self.parents])

    def propagate(self, x, x_var=None):
        means, variances, exp_vars, chains = [], [], [], []
        for lat in self.parents:
            moments = lat.propagate(x, x_var)
            m, v = moments
            means.append(m)
            variances.append(v)
            exp_vars.append(lat._explained_var)
            chains.append(_chain_of(moments))

        mean = _tf.concat(means, axis=1)
        var = _tf.concat(variances, axis=1)
        self._explained_var = _tf.concat(exp_vars, axis=0)
        jitters = _jitters(self.parents)
        self._input_jitter = None if jitters is None \
            else _tf.concat(jitters, axis=0)
        # a `Concatenate` hands its parents' chains on side by side
        experts = None
        if self.propagates_inducing_points and _wants_joint(self):
            experts = _joined_chains(chains, [p.size for p in self.parents])
        return _Moments(mean, var, experts)

    def simulate(self, n_sim, seed=(0, 0)):
        return _tf.concat([lat.simulate(n_sim, seed)
                           for lat in self.parents], axis=0)

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)


class Concatenate(Stack):
    """
    Latent variable concatenation.

    Consolidates a list of latent variables into a single object. This operation requires all its parent nodes to
    be able to propagate inducing points.
    """
    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        # the parents' inducing points are concatenated, so each must have
        # some to give, as `Add` and `LinearCombination` ask
        self.propagates_inducing_points = self.same_root and all(
            p.propagates_inducing_points for p in self.parents)

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)
        # every expert, the active ones under a subset, or the slots as one
        rows = _aligned_points(self.parents, self.root)
        self.inducing_points = tuple(
            _tf.concat([p for p, _ in row], axis=1) for row in rows)
        self.inducing_points_variance = tuple(
            _tf.concat([v for _, v in row], axis=1) for row in rows)
        sizes = [p.size for p in self.parents]
        self.inducing_points_covariance = _combined_covariances(
            _aligned_covariances(self.parents, self.root),
            lambda row: _side_by_side(row, sizes))


class BasicGP(_GPNode):
    """
    Standard Gaussian process node.

    In this module, GP nodes are able to work with inputs that may be
    Gaussian, having an associated variance. This variance is integrated by considering it as a squared range and
    applying the non-stationary covariance.
    """
    def __init__(self, parent, size=1, kernel=None,
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
            The kernel to use for the covariance matrices. A fresh
            `Gaussian` if omitted.
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
        # each node gets its own kernel object -- a shared default would
        # couple every node built without one the day a kernel gains a
        # trainable parameter
        if kernel is None:
            kernel = _kr.Gaussian()
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
        self.joint_gain = None

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
            # leading axes broadcast, so the slots' sets of inducing
            # points go through at once; on matrices this is the same as
            # ever
            var_x = var_x[..., :, None, :]
            var_y = var_y[..., None, :, :]

            # [..., n_data, n_data, n_dim]
            dif = x[..., :, None, :] - y[..., None, :, :]

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

    @staticmethod
    def _whitened_root(chol, delta, eye):
        """`L^-T chol(W)`, `W = (I + L^T D^-1 L)^-1`, per output: a square
        root of `K^-1 - (K+D)^-1` that never forms the difference."""
        scaled = _tf.transpose(chol)[None, :, :] / delta[:, None, :]
        inner = eye + _tf.matmul(scaled, chol[None, :, :])
        w = _tf.linalg.cholesky_solve(_tf.linalg.cholesky(inner), eye)
        root = _tf.linalg.cholesky(w)
        return _tf.linalg.triangular_solve(chol[None, :, :], root,
                                           lower=True, adjoint=True)

    def refresh(self, jitter=1e-6):
        with _tf.name_scope("basic_refresh"):
            self.parent.refresh(jitter)
            if _slots_of(self.root) is not None:
                self._slot_refresh(jitter)
                return

            # prior
            # ip = self.parent.inducing_points
            # ip_var = self.parent.inducing_points_variance

            # every expert, or the subset an expert-by-expert pass asks for;
            # the tuples below hold the active experts in that order
            ids = _active_experts(self.root)
            ips, ipvs = self._parent_points(ids)

            eye = tuple(_tf.eye(self.root.n_ip[i], dtype=_tf.float64)
                        for i in ids)

            # under the expected kernel, where the parent's outputs at the
            # inducing points are uncertain, their covariance is what the
            # kernel is averaged over
            ipcs = _covariances_of(self.parent, ids)
            raw = None
            if self._READS_CHAIN and _JOINT_PROPAGATION \
                    and any(c is not None for c in ipcs):
                raw = tuple(
                    self.expected_covariance(ip, ip_var, ip, ip_var, c)
                    for ip, ip_var, c in zip(ips, ipvs, ipcs))
                cov = tuple(r + e * jitter for r, e in zip(raw, eye))
            else:
                cov = tuple(
                        self.covariance_matrix(ip, ip, ip_var, ip_var) + e * jitter
                        for ip, ip_var, e in zip(ips, ipvs, eye)
                )
            chol = tuple(_tf.linalg.cholesky(mat) for mat in cov)
            cov_inv = tuple(_tf.linalg.cholesky_solve(mat, e) for mat, e in zip(chol, eye))

            self.cov = cov
            self.cov_chol = chol
            self.cov_inv = cov_inv

            # posterior
            eye = tuple(_tf.tile(e[None, :, :], [self.size, 1, 1]) for e in eye)
            delta = tuple(self.parameters[f"delta_{i}"].get_value() for i in ids)
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
            # the square root the simulations draw through, with
            # chol_r chol_r^T = K^-1 - (K+D)^-1, in the whitened form
            # L^-T chol(W) with W = (I + L^T D^-1 L)^-1 rather than as the
            # Cholesky of the difference itself: that difference cancels
            # catastrophically once K is ill-conditioned -- inducing points
            # 0.03 apart behind a fault displacement put K^-1 at 1e9 against
            # (K+D)^-1 at 1e3, and the Cholesky came back NaN in graph mode
            # while it passed eagerly, so every simulation and the prediction
            # built on them was NaN -- where W has its eigenvalues in (0, 1]
            # whatever K does
            self.chol_r = tuple(
                self._whitened_root(chol, d, e)
                for chol, d, e in zip(self.cov_chol, delta, eye)
            )
            # the factor that turns a covariance with the inducing points
            # into the posterior's: `(K + D)^-1 D`, per output
            self.joint_gain = tuple(
                inv * d[:, None, :]
                for inv, d in zip(self.cov_smooth_inv, delta)
            ) if _wants_joint(self) else None

            # inducing points
            alpha_white = tuple(self.parameters[f"alpha_white_{i}"].get_value() for i in ids)
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
            # with quality within a few percent either way. Under an expert
            # subset the sets are those of the active experts, in their
            # order, and only the active experts are consulted.
            self.inducing_points_covariance = None
            if len(self.children) > 0:
                bias = [self.parameters[f'bias_{i}'].get_value() for i in ids]

                self.inducing_points = []
                self.inducing_points_variance = []
                if _JOINT_PROPAGATION:
                    # each expert speaks for its own set, as under the
                    # independent rule, and where a GP node reads the
                    # result, hands on the covariance between its outputs
                    # there -- `K (K + D)^-1 D`, made symmetric -- whose
                    # diagonal is the variance, exactly
                    covariances = []
                    for p in range(len(ids)):
                        k = raw[p] if raw is not None else \
                            self.covariance_matrix(ips[p], ips[p], ipvs[p],
                                                   ipvs[p])
                        mean = _tf.einsum(
                            "ab,sbc->sac", k, self.alpha[p]) + bias[p]
                        self.inducing_points.append(
                            _tf.transpose(mean[:, :, 0]))
                        if self.joint_gain is None:
                            pred_var = 1.0 - _tf.reduce_sum(
                                _tf.einsum("ab,sbc->sac", k,
                                           self.cov_smooth_inv[p])
                                * k[None, :, :], axis=2)
                            self.inducing_points_variance.append(
                                _tf.transpose(pred_var))
                            continue
                        c = _tf.einsum("ab,sbc->sac", k, self.joint_gain[p])
                        c = 0.5 * (c + _tf.transpose(c, [0, 2, 1]))
                        self.inducing_points_variance.append(
                            _tf.transpose(_tf.linalg.diag_part(c)))
                        covariances.append(_tf.transpose(c, [1, 2, 0]))
                    if covariances:
                        self.inducing_points_covariance = tuple(covariances)
                elif _EXPERT_PROPAGATION == "independent":
                    for p in range(len(ids)):
                        ip_i = ips[p]
                        ipv_i = ipvs[p]
                        cov = self.covariance_matrix(ip_i, ip_i, ipv_i, ipv_i)
                        mean = _tf.einsum(
                            "ab,sbc->sac", cov, self.alpha[p]) + bias[p]
                        pred_var = 1.0 - _tf.reduce_sum(
                            _tf.einsum("ab,sbc->sac", cov,
                                       self.cov_smooth_inv[p])
                            * cov[None, :, :],
                            axis=2, keepdims=False
                        )
                        self.inducing_points.append(
                            _tf.transpose(mean[:, :, 0]))
                        self.inducing_points_variance.append(
                            _tf.transpose(pred_var))
                else:
                    for p in range(len(ids)):
                        ip_i = ips[p]
                        ipv_i = ipvs[p]
                        means = []
                        pred_vars = []
                        for q in range(len(ids)):
                            ip_j = ips[q]
                            ipv_j = ipvs[q]
                            cov = self.covariance_matrix(ip_i, ip_j, ipv_i, ipv_j)
                            means.append(_tf.einsum("ab,sbc->sac", cov, self.alpha[q]) + bias[q])
                            pred_vars.append(
                                1.0 - _tf.reduce_sum(
                                    _tf.einsum("ab,sbc->sac", cov, self.cov_smooth_inv[q]) * cov[None, :, :],
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

    def _parent_points(self, ids):
        """The parent's inducing points and their variances for the experts
        `ids`, in that order (see `_points_of`)."""
        return _points_of(self.parent, ids)

    def _slot_locals(self):
        """The active slots' own parameters -- `alpha_white` `[slots, size,
        m, 1]`, `delta` `[slots, size, m]`, `bias` `[slots]` -- with alpha
        zero and delta one where an expert has fewer points than `m`."""
        slots = _slots_of(self.root)
        if slots.stacks is not None:
            alpha, delta, bias = slots.stacks[id(self)]
        else:
            m = max(self.root.n_ip)
            n = self.root.n_ip
            alpha = _tf.stack(
                [_tf.pad(self.parameters["alpha_white_%d" % i].variable,
                         [[0, 0], [0, m - n[i]], [0, 0]])
                 for i in range(self.root.n_experts)]
                + [_tf.zeros([self.size, m, 1], _tf.float64)])
            delta = _tf.stack(
                [_tf.pad(self.parameters["delta_%d" % i].variable,
                         [[0, 0], [0, m - n[i]]])
                 for i in range(self.root.n_experts)]
                + [_tf.zeros([self.size, m], _tf.float64)])
            bias = _tf.stack(
                [self.parameters["bias_%d" % i].variable
                 for i in range(self.root.n_experts)]
                + [_tf.zeros([], _tf.float64)])
        # the raw values pad with zeros, which each of the three parameters
        # reads back as the padding wants: alpha zero, delta one, bias zero
        return (self.parameters["alpha_white_0"]._back_transform(
                    _tf.gather(alpha, slots.ids)),
                self.parameters["delta_0"]._back_transform(
                    _tf.gather(delta, slots.ids)),
                self.parameters["bias_0"]._back_transform(
                    _tf.gather(bias, slots.ids)))

    def _slot_refresh(self, jitter):
        """`refresh` under slots: the active experts' matrices at once,
        batched over the slots. A padded point's row and column of the
        covariance are the identity's and its alpha is zero, so it adds
        nothing to any expert, and an empty slot is masked out of every
        blend."""
        slots = _slots_of(self.root)
        ips, ipvs = _slot_points(self.parent)
        pmask = _slot_mask(self.root)
        alpha_white, delta, bias = self._slot_locals()
        m = pmask.shape[1]
        eye = _tf.eye(m, dtype=_tf.float64)
        outer = pmask[:, :, None] * pmask[:, None, :]
        ipcs = _slot_covariance(self.parent)
        if self._READS_CHAIN and _JOINT_PROPAGATION and ipcs is not None:
            raw = self.expected_covariance(ips, ipvs, ips, ipvs, ipcs) \
                * outer
        else:
            raw = self.covariance_matrix(ips, ips, ipvs, ipvs) * outer
        cov = raw + eye[None] * (1.0 - pmask)[:, None, :] \
            + eye[None] * jitter
        chol = _tf.linalg.cholesky(cov)
        cov_inv = _tf.linalg.cholesky_solve(
            chol, _tf.broadcast_to(eye, _tf.shape(cov)))
        eye_s = _tf.broadcast_to(eye, [slots.size, self.size, m, m])
        cov_smooth = cov[:, None, :, :] + _tf.linalg.diag(delta)
        smooth_chol = _tf.linalg.cholesky(cov_smooth + eye_s * jitter)
        smooth_inv = _tf.linalg.cholesky_solve(smooth_chol, eye_s)
        # the whitened root of `_whitened_root`, per slot and output
        chol_s = _tf.broadcast_to(chol[:, None, :, :],
                                  [slots.size, self.size, m, m])
        scaled = _tf.linalg.matrix_transpose(chol_s) / delta[:, :, None, :]
        inner = eye_s + _tf.matmul(scaled, chol_s)
        w = _tf.linalg.cholesky_solve(_tf.linalg.cholesky(inner), eye_s)
        chol_r = _tf.linalg.triangular_solve(
            chol_s, _tf.linalg.cholesky(w), lower=True, adjoint=True)
        means = _tf.einsum("pab,psbc->psac", chol, alpha_white)
        alpha = _tf.einsum("pab,psbc->psac", cov_inv, means)

        self.slots_cov = cov
        self.slots_cov_smooth_chol = smooth_chol
        self.slots_cov_smooth_inv = smooth_inv
        self.slots_chol_r = chol_r
        self.slots_alpha = alpha
        self.slots_bias = bias
        self.slots_alpha_white = alpha_white
        self.slots_delta = delta
        self.slots_input_mask = pmask
        # `(K + D)^-1 D`, as `refresh` keeps it per expert
        self.slots_joint_gain = smooth_inv * delta[:, :, None, :] \
            if _wants_joint(self) else None

        self.inducing_points_covariance = None
        if len(self.children) == 0:
            return
        if _JOINT_PROPAGATION:
            # each slot speaks for its own set, and hands on the covariance
            # of its outputs there where a GP node reads them, as `refresh`
            mean = _tf.einsum("pab,psbc->psac", raw, alpha) \
                + bias[:, None, None, None]
            points = _tf.transpose(mean[:, :, :, 0], [0, 2, 1])
            if self.slots_joint_gain is None:
                pred_var = 1.0 - _tf.reduce_sum(
                    _tf.einsum("pab,psbc->psac", raw, smooth_inv)
                    * raw[:, None, :, :], axis=3)
            else:
                c = _tf.einsum("pab,psbc->psac", raw, self.slots_joint_gain)
                c = 0.5 * (c + _tf.transpose(c, [0, 1, 3, 2]))
                pred_var = _tf.linalg.diag_part(c)
                self.inducing_points_covariance = (_tf.reshape(
                    _tf.transpose(c, [0, 2, 3, 1]), [-1, m, self.size]),)
            points_var = _tf.transpose(pred_var, [0, 2, 1])
        elif _EXPERT_PROPAGATION == "independent":
            mean = _tf.einsum("pab,psbc->psac", raw, alpha) \
                + bias[:, None, None, None]
            pred_var = 1.0 - _tf.reduce_sum(
                _tf.einsum("pab,psbc->psac", raw, smooth_inv)
                * raw[:, None, :, :], axis=3)
            points = _tf.transpose(mean[:, :, :, 0], [0, 2, 1])
            points_var = _tf.transpose(pred_var, [0, 2, 1])
        else:
            # every active expert's set predicted from every other's and
            # blended by precision, as `refresh` blends them; an empty slot
            # is masked out of each blend
            cov_pq = self.covariance_matrix(
                ips[:, None], ips[None], ipvs[:, None], ipvs[None]) \
                * (pmask[:, None, :, None] * pmask[None, :, None, :])
            means = _tf.einsum("pqab,qsbc->pqsac", cov_pq, alpha)[..., 0] \
                + bias[None, :, None, None]
            pred_vars = 1.0 - _tf.reduce_sum(
                _tf.einsum("pqab,qsbc->pqsac", cov_pq, smooth_inv)
                * cov_pq[:, :, None, :, :], axis=-1)
            raw_w = ((1.0 - pred_vars) / (pred_vars + 1e-6) + 1e-6) \
                * slots.mask[None, :, None, None]
            weights = raw_w / _tf.reduce_sum(raw_w, axis=1, keepdims=True)
            points = _tf.transpose(
                _tf.reduce_sum(means * weights, axis=1), [0, 2, 1])
            points_var = _tf.transpose(
                _tf.reduce_sum(pred_vars * weights, axis=1), [0, 2, 1])
        # handed on as every node hands them on under slots: one tensor,
        # flattened over the slots
        self.inducing_points = (_tf.reshape(points, [-1, self.size]),)
        self.inducing_points_variance = (
            _tf.reshape(points_var, [-1, self.size]),)

    def _slot_expert_moments(self, x, x_var=None, chain=None,
                             with_second=False):
        """`_expert_moments` under slots, batched over them; `_blend`
        masks an empty slot out."""
        ips, ipvs = _slot_points(self.parent)
        pmask = _slot_mask(self.root)
        if chain is None:
            cov_cross = self.covariance_matrix(x, ips, x_var, ipvs)
        else:
            (e,) = chain
            cov_cross = self.expected_covariance(e.mean, e.variance, ips,
                                                 ipvs, e.covariance)
        cov_cross = cov_cross * pmask[:, None, :]
        mu = _tf.einsum("pab,psbc->psac", cov_cross, self.slots_alpha) \
            + self.slots_bias[:, None, None, None]
        if chain is not None and self._takes_second_moment(
                chain, [_slot_covariance(self.parent)]):
            # a padded point reaches nothing: its row and column of every
            # matrix the moments are read through are zero
            outer = pmask[:, None, :, None] * pmask[:, None, None, :]
            var, explained_var, jitter = self._mixture_moments(
                chain[0], ips, self.slots_cov_smooth_inv * outer,
                self.slots_alpha, mu[..., 0]
                - self.slots_bias[:, None, None], cov_cross,
                self.slots_chol_r * outer if with_second else None)
            second = _Second(_tf.maximum(1.0 - explained_var, 0.0), jitter)
            return (cov_cross, mu, var, explained_var) \
                + ((second,) if with_second else ())
        explained_var = _tf.reduce_sum(
            _tf.einsum("pab,psbc->psac", cov_cross, self.slots_cov_smooth_inv)
            * cov_cross[:, None, :, :], axis=3)
        var = _tf.maximum(1.0 - explained_var, 0.0)
        return (cov_cross, mu, var, explained_var) \
            + ((None,) if with_second else ())

    def _slot_simulate(self, n_sim, seed):
        """`simulate` under slots: the normals each expert draws in
        `simulate`, gathered into the slots."""
        slots = _slots_of(self.root)
        cov_cross, mu, weights = self._swept()
        with _tf.name_scope("gp_simulation"):
            m = cov_cross.shape[-1]
            # each expert's normals drawn at its own size, as `simulate`
            # draws them, and padded: a draw fills its array in order, so
            # one draw at the padded size gives a smaller expert's second
            # output other numbers. A padded point's normals reach nothing,
            # its covariance with every location being masked to zero
            drawn = [_tf.pad(_simulation_normals([self.size, n, n_sim], seed,
                                                 key=self.name),
                             [[0, 0], [0, m - n], [0, 0]])
                     for n in self.root.n_ip]
            drawn.append(_tf.zeros([self.size, m, n_sim], _tf.float64))
            rnd = _tf.gather(_tf.stack(drawn), slots.ids)
            sims = _tf.einsum(
                "pab,psbc->psac", cov_cross,
                _tf.matmul(self.slots_chol_r, rnd)) + mu
            return _tf.reduce_sum(sims * weights[:, :, :, None], axis=0)

    def cache_prediction_state(self):
        if _slots_of(self.root) is not None:
            self._cache_slot_state()
            return
        if _subset_of(self.root) is None:
            super().cache_prediction_state()
            self.alpha = self._cache_tuple("alpha", self.alpha)
            self.cov_inv = self._cache_tuple("cov_inv", self.cov_inv)
            self.cov_smooth_inv = self._cache_tuple(
                "cov_smooth_inv", self.cov_smooth_inv)
            self.chol_r = self._cache_tuple("chol_r", self.chol_r)
            if self.joint_gain is not None:
                self.joint_gain = self._cache_tuple("joint_gain",
                                                    self.joint_gain)
            return
        # under a subset the tuples hold the active experts only, so each
        # snapshot is named after its expert rather than its position, and
        # a Variable keeps the one shape its expert gives it
        ids = _active_experts(self.root)

        def by_expert(name, values):
            return tuple(self._state_var("%s_%d" % (name, i), v)
                         for i, v in zip(ids, values))

        if self.inducing_points is not None:
            self.inducing_points = by_expert(
                "inducing_points", self.inducing_points)
        if self.inducing_points_variance is not None:
            self.inducing_points_variance = by_expert(
                "inducing_points_variance", self.inducing_points_variance)
        if self.inducing_points_covariance is not None:
            self.inducing_points_covariance = by_expert(
                "inducing_points_covariance", self.inducing_points_covariance)
        if self.joint_gain is not None:
            self.joint_gain = by_expert("joint_gain", self.joint_gain)
        self.alpha = by_expert("alpha", self.alpha)
        self.cov_inv = by_expert("cov_inv", self.cov_inv)
        self.cov_smooth_inv = by_expert("cov_smooth_inv", self.cov_smooth_inv)
        self.chol_r = by_expert("chol_r", self.chol_r)

    def _moments(self, x, x_var=None):
        with _tf.name_scope("basic_interpolation"):
            cov_cross, mu, var, explained = self._expert_moments(x, x_var)
            weights, w_mu, w_var, w_exp_var = self._blend(mu, var, explained)
            return cov_cross, mu, weights, w_mu, w_var, w_exp_var

    def expected_covariance(self, mean_x, var_x, mean_y, var_y, cov=None,
                            gradient=False):
        """The covariance between two sets of uncertain inputs under the
        expected kernel: `mean_x` `[..., n, d]`, `mean_y` `[..., m, d]`,
        their variances alike or None, and their covariance
        `[..., n, m, d]` or None. Returns `[..., n, m]`, and with
        `gradient` its derivative in `mean_x`, `[..., n, m, d]`."""
        return _expected_kernel(self.kernel,
                                self.parameters["ranges"].get_value(),
                                mean_x, var_x, mean_y, var_y, cov, gradient)

    def _expert_moments(self, x, x_var=None, chain=None, with_second=False):
        """Each active expert's moments at already-propagated locations:
        the covariance with its inducing points, its mean `[size, n, 1]`,
        and its variance and the variance it explains `[size, n]` -- lists
        over the experts, the variances stacked, or under slots tensors
        with a leading slot axis. `chain`, the parent's chain under the
        expected kernel, takes the place of `x` and `x_var`. With
        `with_second`, also a `_Second` where the second moment is taken
        (the variances in it stacked or batched as the variance), and None
        where it is not."""
        with _tf.name_scope("basic_interpolation"):
            if _slots_of(self.root) is not None:
                return self._slot_expert_moments(x, x_var, chain,
                                                 with_second)
            ids = _active_experts(self.root)
            ips, ipvs = self._parent_points(ids)
            if chain is None:
                cov_cross = [
                    self.covariance_matrix(x, ip, x_var, ip_var)
                    for ip, ip_var in zip(ips, ipvs)
                ]
            else:
                cov_cross = [
                    self.expected_covariance(e.mean, e.variance, ip, ip_var,
                                             e.covariance)
                    for e, ip, ip_var in zip(chain, ips, ipvs)
                ]

            bias = [self.parameters[f'bias_{i}'].get_value() for i in ids]
            mu = [
                _tf.einsum("ab,sbc->sac", mat, vec) + b
                for mat, vec, b in zip(cov_cross, self.alpha, bias)
            ]

            if chain is not None and self._takes_second_moment(
                    chain, _covariances_of(self.parent, ids)):
                moments = [
                    self._mixture_moments(e, ip, inv, a, m[..., 0] - b, l,
                                          r if with_second else None)
                    for e, ip, inv, a, m, b, l, r in zip(
                        chain, ips, self.cov_smooth_inv, self.alpha, mu, bias,
                        cov_cross, self.chol_r)]
                var = _tf.stack([v for v, _, _ in moments], axis=0)
                explained_var = [x for _, x, _ in moments]
                second = _Second(_tf.maximum(
                    1.0 - _tf.stack(explained_var, axis=0), 0.0),
                    [j for _, _, j in moments] if with_second else None)
                return (cov_cross, mu, var, explained_var) \
                    + ((second,) if with_second else ())

            explained_var = [
                _tf.reduce_sum(
                    _tf.einsum("ab,sbc->sac", m1, m2) * m1[None, :, :],
                    axis=2, keepdims=False
                )
                for m1, m2 in zip(cov_cross, self.cov_smooth_inv)
            ]
            var = _tf.stack([_tf.maximum(1.0 - v, 0.0) for v in explained_var], axis=0)
            return (cov_cross, mu, var, explained_var) \
                + ((None,) if with_second else ())

    def _blend(self, mu, var, explained_var, left=None):
        """The experts' moments blended by their weights: the weights, the
        mean `[size, n, 1]`, the variance and the explained variance.

        An expert is weighted by the variance its inducing points leave:
        the variance itself, or `left` where the second moment adds to the
        variance the spread of the mean over an uncertain input, which says
        nothing of how well the expert knows the ground there."""
        weigh = var if left is None else left
        slots = _slots_of(self.root)
        if slots is not None:
            raw_w = ((1.0 - weigh) / (weigh + 1e-6) + 1e-6) \
                * slots.mask[:, None, None]
            weights = raw_w / _tf.reduce_sum(raw_w, axis=0, keepdims=True)
            w_mu = _tf.reduce_sum(mu * weights[:, :, :, None], axis=0)
            w_var = _tf.reduce_sum(var * weights, axis=0)
            w_exp_var = _tf.reduce_sum(explained_var * weights, axis=0)
            return weights, w_mu, w_var, w_exp_var

        weights = _GPNode.get_expert_weights(weigh)

        w_mu = _tf.reduce_sum(_tf.stack(mu, axis=0) * weights[:, :, :, None], axis=0)
        w_var = _tf.reduce_sum(_tf.stack(var, axis=0) * weights, axis=0)
        w_exp_var = _tf.reduce_sum(_tf.stack(explained_var, axis=0) * weights, axis=0)
        return weights, w_mu, w_var, w_exp_var

    def _takes_second_moment(self, chain, covariances):
        """Whether the moments at an uncertain input are completed by the
        second moment: under the expected kernel, where the input is
        uncertain and its inducing points are not -- an input's own output,
        through nodes acting row by row -- and the kernel is one the second
        moment takes."""
        return (self._READS_CHAIN
                and all(e.covariance is None for e in chain)
                and all(c is None for c in covariances)
                and _second_moment_supported(self.kernel))

    def _kernel_items(self):
        """The kernel as the Gaussians `_second_moment` pairs up."""
        return _kernel_items(self.kernel,
                             self.parameters["ranges"].get_value())

    def _closed_second_moment(self):
        """Whether the second moment is taken in closed form -- the Gaussian
        kernel, one pair of components -- rather than by quadrature over
        the input."""
        return type(self.kernel) is _kr.Gaussian

    def _plain_covariance(self, x, y):
        """`covariance_matrix` between certain points, `[..., n, m]`: the
        kernel at the plain distance, the uncertain-input normalization
        being exactly one there (`_plain_distance`)."""
        return self.kernel.kernelize(_plain_distance(
            x, y, self.parameters["ranges"].get_value()))

    def _at_input_nodes(self, mean, var, points):
        """The node's own kernel between `points` and each location's input
        at the quadrature nodes, `[..., n, q, m]`."""
        nodes = _tf.constant(_input_nodes(int(mean.shape[-1])), _tf.float64)
        # the standard deviation with a finite gradient at zero, both
        # branches of a `where` being differentiated
        positive = var > 0.0
        sd = _tf.where(positive, _tf.sqrt(_tf.where(positive, var, 1.0)),
                       _tf.zeros_like(var))
        draws = mean[..., :, None, :] + sd[..., :, None, :] * nodes
        shape = _tf.shape(draws)
        flat = _tf.reshape(draws, _tf.concat(
            [shape[:-3], [shape[-3] * shape[-2]], shape[-1:]], 0))
        k = self._plain_covariance(flat, points)
        return _tf.reshape(k, _tf.concat(
            [_tf.shape(k)[:-2], shape[-3:-1], _tf.shape(k)[-1:]], 0))

    def _second_moments(self, mean, var, points, weights):
        """`sum_ij W_kij E[k(x, z_i) k(x, z_j)]`, `[..., k, n]`, for each
        stack `W` in the list `weights`: see `_second_moment`."""
        return _second_moment(self._kernel_items(), mean, var, points,
                              weights)

    def _mixture_moments(self, chain, points, smooth_inv, alpha, offset,
                         cov_cross=None, root_r=None):
        """One expert's variance, explained variance and latent jitter at
        an uncertain input over certain inducing points: the mixture's,
        exactly.

        With `L = E[k(x, z) k(x, z)ᵀ]` and `l = E[k(x, z)]` (`cov_cross`),
        the posterior's variance averaged over the input is
        `1 - tr((K + D)^-1 L)`, the variance of its mean `alphaᵀ L alpha -
        (l alpha)²` (Girard's second moment), and the variance explained is
        the first trace. `offset` is the expected kernel's mean less the
        bias, `l alpha`, `[..., size, n]`. The realizations, `l (alpha + R
        eps)`, carry `l R Rᵀ lᵀ` where the mixture's carry `tr(R Rᵀ L)` and
        the spread of the mean: the difference, `tr((R Rᵀ + alpha
        alphaᵀ)(L - l lᵀ))`, never negative, is the jitter -- computed
        where `root_r`, `R`, is given, and None otherwise. In closed form
        for the Gaussian kernel (`_second_moment`); for the others `L` and
        `l` are averages of the node's own kernel over 64 scrambled Sobol
        points of the input (`_input_nodes`), so that the spread and the
        jitter are variances over them."""
        var = chain.variance
        if var is None:
            var = _tf.zeros_like(chain.mean)
        if not self._closed_second_moment():
            # the same quantities over the quadrature nodes, each a variance
            # over them where it is one, so never negative
            k = self._at_input_nodes(chain.mean, var, points)
            # L = l lᵀ + Cov k: the first part from the expected kernel,
            # exact, the quadrature asked only for the covariance, which is
            # small where the input variance is
            spread_k = k - _tf.reduce_mean(k, axis=-2, keepdims=True)
            explained = _tf.einsum(
                "...ni,...sij,...nj->...sn", cov_cross, smooth_inv, cov_cross)                 + _tf.reduce_mean(_tf.einsum(
                    "...nqi,...sij,...nqj->...snq", spread_k, smooth_inv,
                    spread_k), axis=-1)
            f = _tf.einsum("...nqm,...sm->...snq", k, alpha[..., 0])
            spread = _tf.reduce_mean(
                (f - _tf.reduce_mean(f, -1, keepdims=True)) ** 2, axis=-1)
            jitter = None
            if root_r is not None:
                g = _tf.einsum("...nqm,...smj->...snqj", k, root_r)
                jitter = _tf.reduce_mean(_tf.reduce_sum(
                    (g - _tf.reduce_mean(g, -2, keepdims=True)) ** 2, -1),
                    -1) + spread
            return _tf.maximum(1.0 - explained, 0.0) + spread, explained, \
                jitter
        weights = [smooth_inv, alpha * _tf.linalg.matrix_transpose(alpha)]
        if root_r is not None:
            weights.append(_tf.matmul(root_r, root_r, transpose_b=True))
        moments = self._second_moments(chain.mean, var, points, weights)
        explained = moments[0]
        spread = _tf.maximum(moments[1] - offset ** 2, 0.0)
        jitter = None
        if root_r is not None:
            seen = _tf.reduce_sum(
                _tf.einsum("...nm,...smj->...snj", cov_cross, root_r) ** 2,
                axis=-1)
            jitter = _tf.maximum(moments[2] - seen, 0.0) + spread
        return _tf.maximum(1.0 - explained, 0.0) + spread, explained, jitter

    def _chain(self, cov_cross, mu, var):
        """What this node hands a GP node above it under the expected
        kernel: each expert's mean and variance at the data, and the
        covariance between its outputs there and at its inducing points --
        the posterior's own, `k_x (K + D)^-1 D`."""
        if _slots_of(self.root) is not None:
            return (_Joint(_tf.transpose(mu[:, :, :, 0], [0, 2, 1]),
                           _tf.transpose(var, [0, 2, 1]),
                           _tf.einsum("pnb,psbc->pncs", cov_cross,
                                      self.slots_joint_gain)),)
        return tuple(
            _Joint(_tf.transpose(mu[p][:, :, 0]), _tf.transpose(var[p]),
                   _tf.einsum("nb,sbc->ncs", cov_cross[p],
                              self.joint_gain[p]))
            for p in range(len(mu)))

    def kl_divergence(self):
        with _tf.name_scope("basic_KL_divergence"):
            if _slots_of(self.root) is not None:
                return _tf.reduce_sum(self.expert_kl_terms()
                                      * _slots_of(self.root).mask)
            return _tf.add_n(self.expert_kl_terms())

    def expert_kl_terms(self):
        """Each active expert's KL divergence, in the order of the active
        experts -- what `kl_divergence` adds up. Under slots a tensor over
        the slots, to which a padded point adds nothing."""
        if _slots_of(self.root) is not None:
            pmask = self.slots_input_mask
            outer = pmask[:, :, None] * pmask[:, None, :]
            tr = _tf.reduce_sum(self.slots_cov_smooth_inv
                                * (self.slots_cov * outer)[:, None, :, :],
                                axis=[1, 2, 3])
            fit = _tf.reduce_sum(self.slots_alpha_white ** 2, axis=[1, 2, 3])
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(_tf.linalg.diag_part(
                self.slots_cov_smooth_chol)) * pmask[:, None, :], axis=[1, 2])
            det_2 = _tf.reduce_sum(_tf.math.log(self.slots_delta)
                                   * pmask[:, None, :], axis=[1, 2])
            return 0.5 * (- tr + fit + det_1 - det_2)
        all_kl = []
        for p, i in enumerate(_active_experts(self.root)):
            delta = self.parameters[f"delta_{i}"].get_value()
            alpha_white = self.parameters[f"alpha_white_{i}"].get_value()

            tr = _tf.reduce_sum(self.cov_smooth_inv[p] * self.cov[p][None, :, :])
            fit = _tf.reduce_sum(alpha_white**2)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol[p])))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))
            kl = 0.5 * (- tr + fit + det_1 - det_2)

            all_kl.append(kl)
        return all_kl

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
            # a distance, so never negative: the Matern kernels read a
            # negative one as a growing exponential
            dist = _tf.abs(dif) / _tf.sqrt(total_var)
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

    def expected_covariance(self, mean_x, var_x, mean_y, var_y, cov=None,
                            gradient=False):
        # one dimension at a time, as the covariance is built
        ranges = self.parameters["ranges"].get_value() \
            * _tf.ones([1, 1, self.parent.size], _tf.float64)

        def column(t, d):
            return None if t is None else t[..., d:d + 1]

        parts = [_expected_kernel(self.kernel, ranges[..., d:d + 1],
                                  mean_x[..., d:d + 1], column(var_x, d),
                                  mean_y[..., d:d + 1], column(var_y, d),
                                  column(cov, d), gradient)
                 for d in range(self.parent.size)]
        if not gradient:
            return _tf.reduce_mean(_tf.stack(parts, axis=-1), axis=-1)
        # each dimension's kernel moves with its own coordinate only
        return (_tf.reduce_mean(_tf.stack([p[0] for p in parts], -1), -1),
                _tf.concat([p[1] for p in parts], axis=-1) / self.parent.size)

    def _plain_covariance(self, x, y):
        ranges = _tf.reshape(self.parameters["ranges"].get_value(), [-1]) \
            * _tf.ones([self.parent.size], _tf.float64)
        dist = _tf.abs(x[..., :, None, :] - y[..., None, :, :]) / ranges
        return _tf.reduce_mean(self.kernel.kernelize(dist), axis=-1)

    def _second_moments(self, mean, var, points, weights):
        # the mean of one kernel per dimension: two dimensions' kernels are
        # independent over an input whose dimensions are, so a pair of them
        # is the product of their first moments, and a dimension with
        # itself its own second moment --
        # L = (s sᵀ - sum_d l_d l_dᵀ + sum_d L_d) / D², s = sum_d l_d
        size = self.parent.size
        ranges = self.parameters["ranges"].get_value() \
            * _tf.ones([1, 1, size], _tf.float64)
        own, firsts = [], []
        for d in range(size):
            column = slice(d, d + 1)
            own.append(_second_moment(
                _kernel_items(self.kernel, ranges[..., column]),
                mean[..., column], var[..., column], points[..., column],
                weights))
            firsts.append(_expected_kernel(
                self.kernel, ranges[..., column], mean[..., column],
                var[..., column], points[..., column], None))

        def form(first, w):                    # first [..., n, m]
            return _tf.einsum("...ni,...kij,...nj->...kn", first, w, first)

        total = _tf.add_n(firsts)
        return [(form(total, w) - _tf.add_n([form(f, w) for f in firsts])
                 + _tf.add_n([o[k] for o in own])) / size ** 2
                for k, w in enumerate(weights)]


class UncertainInputGP(BasicGP):
    """
    GP node that integrates over the uncertainty of its input by
    quadrature. Deprecated.

    Deprecated since 0.9.0, and to be removed: under the expected kernel
    (`GPOptions(propagation="joint")`, the default for a new model) a
    `BasicGP` on an uncertain input takes the mixture's moments in closed
    form, more closely than this node's quadrature, and the likelihood
    integrates what its realizations leave out. Under the old rule
    `BasicGP` reads an uncertain input through an inflated covariance --
    Paciorek's nonstationary form -- and takes its moments as if the input
    were one point under that kernel, which understates the predictive
    variance several times over and misplaces the mean once the input
    variance reaches a tenth of the squared range. This node computes the
    mixture instead: `n_nodes`
    scrambled Sobol points of the input's Gaussian, the deterministic
    posterior at each of them, and the mixture's moments out -- the mean of
    the means, the mean of the variances plus the variance of the means.
    Each realization is drawn at a node of its own, so the simulations
    carry the mixture as well. Nothing depends on the kernel.

    Parameters
    ----------
    parent
        The node whose output is the input. Its variance is what is
        integrated over: a `GaussianInput` root, or any GP node above.
    size
        Number of latent variables.
    kernel
        A kernel object from `geoml.kernels`.
    fix_range
        Whether to fix the range parameters.
    isotropic
        Whether to use a single range for every input dimension.
    range_prior
        As in `BasicGP`.
    n_nodes
        Quadrature points per input. A power of two keeps the Sobol
        sequence balanced.
    name
        A name for this node, shown in the printed network and accepted by
        `get_node`. Numbered automatically if omitted.

    See Also
    --------
    BasicGP : the node this extends.
    GaussianInput : the root that supplies an input variance.

    Notes
    -----
    The cost is `n_nodes` cross-covariances per prediction or training step
    where `BasicGP` computes one, and the same factor in memory for them.
    Given an input with no variance the node is `BasicGP`.
    """
    _READS_CHAIN = False

    def __init__(self, parent, size=1, kernel=None, fix_range=False,
                 isotropic=False, range_prior=2.0, n_nodes=32, name=None):
        _warnings.warn(
            "UncertainInputGP is deprecated since 0.9.0 and will be removed: "
            "a BasicGP takes the mixture's moments at an uncertain input in "
            "closed form under the expected kernel "
            "(GPOptions(propagation='joint'), the default)", FutureWarning,
            stacklevel=2)
        super().__init__(parent, size=size, kernel=kernel,
                         fix_range=fix_range, isotropic=isotropic,
                         range_prior=range_prior, name=name)
        self.n_nodes = int(n_nodes)
        # the input's Gaussian sampled once: scrambled Sobol through the
        # normal quantile, the scramble drawn from the package RNG so that
        # `set_seed` fixes it, and kept as a fixed parameter so that a saved
        # model replays the same nodes
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            points = _rnd.sobol_engine(self.parent.size, _rnd.rng()) \
                .random(self.n_nodes)
        nodes = _special.ndtri(points)
        self._add_parameter("nodes", _gpr.RealParameter(
            nodes, _np.full_like(nodes, -10.0), _np.full_like(nodes, 10.0),
            fixed=True))

    def _moments(self, x, x_var=None):
        with _tf.name_scope("uncertain_input_moments"):
            nodes = self.parameters["nodes"].get_value()       # [q, d]
            q = self.n_nodes
            if x_var is None:
                x_var = _tf.zeros_like(x)
            # the standard deviation, with a finite gradient at zero: the
            # root's derivative is infinite there and, multiplied by a zero
            # from an exact input, turns every gradient NaN; both branches
            # of a `where` are differentiated, so the unselected one must be
            # finite too
            positive = x_var > 0.0
            sd = _tf.where(positive, _tf.sqrt(_tf.where(positive, x_var, 1.0)),
                           _tf.zeros_like(x_var))
            # every node of every point, stacked along the batch axis, and
            # the deterministic posterior at all of them in one pass
            at_nodes = x[None, :, :] + sd[None, :, :] * nodes[:, None, :]  # [q, n, d]
            stacked = _tf.reshape(at_nodes, [-1, x.shape[1]])
            cov_cross, mu, weights, w_mu, w_var, w_exp_var = \
                super()._moments(stacked, None)

            # the mixture's moments over the nodes
            mean_q = _tf.reshape(w_mu[:, :, 0], [self.size, q, -1])
            var_q = _tf.reshape(w_var, [self.size, q, -1])
            exp_q = _tf.reshape(w_exp_var, [self.size, q, -1])
            mean = _tf.reduce_mean(mean_q, axis=1)
            spread = _tf.reduce_mean((mean_q - mean[:, None, :]) ** 2, axis=1)
            var = _tf.reduce_mean(var_q, axis=1) + spread
            exp_var = _tf.reduce_mean(exp_q, axis=1)
            return cov_cross, mu, weights, mean[:, :, None], var, exp_var

    def simulate(self, n_sim, seed=(0, 0)):
        cov_cross, mu, weights = self._swept()
        q = self.n_nodes
        with _tf.name_scope("uncertain_input_simulation"):
            # each realization sits at a node of its own, so the ensemble
            # samples the input's uncertainty as well as the posterior's
            which = _tf.range(n_sim) % q
            rnd = [
                _simulation_normals([self.size, m, n_sim], seed,
                                    key=self.name)
                for m in self.root.n_ip
            ]
            sims = []
            for a, b, c, d, m in zip(cov_cross, self.chol_r, rnd, mu,
                                     self.root.n_ip):
                a_sel = _tf.gather(_tf.reshape(a, [q, -1, m]), which)
                d_sel = _tf.gather(
                    _tf.reshape(d[:, :, 0], [self.size, q, -1]), which, axis=1)
                draws = _tf.matmul(b, c)                            # [size, m, n_sim]
                sims.append(_tf.einsum("snm,zms->zns", a_sel, draws)
                            + _tf.transpose(d_sel, [0, 2, 1]))
            w_sel = _tf.gather(
                _tf.reshape(weights, [self.root.n_experts, self.size, q, -1]),
                which, axis=2)
            w_sel = _tf.transpose(w_sel, [0, 1, 3, 2])              # [E, size, n, n_sim]
            return _tf.reduce_sum(_tf.stack(sims, axis=0) * w_sel, axis=0)


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
                for ip_var in self.parent.inducing_points_variance
            )
            held = self.parent.inducing_points_covariance
            self.inducing_points_covariance = None if held is None \
                else tuple(_tf.einsum("...s,st->...t", c, weights ** 2)
                           for c in held)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def propagate(self, x, x_var=None):
        weights = self.parameters["weights"].get_value()

        parent = self.parent.propagate(x, x_var)
        mean, var = parent
        mu = _tf.einsum("xab,xy->yab", _tf.transpose(mean)[:, :, None],
                        weights)
        var = _tf.einsum("xa,xy->ya", _tf.transpose(var), weights ** 2)
        self._explained_var = _tf.einsum(
            "xa,xy->ya", self.parent._explained_var, weights ** 2)
        jitter = self.parent._input_jitter
        self._input_jitter = None if jitter is None \
            else _tf.einsum("xa,xy->ya", jitter, weights ** 2)
        experts = _map_chain(
            _chain_of(parent),
            lambda m: _tf.einsum("...s,st->...t", m, weights),
            lambda v: _tf.einsum("...s,st->...t", v, weights ** 2)) \
            if _wants_joint(self) else None
        return _Moments(_tf.transpose(mu[:, :, 0]), _tf.transpose(var),
                        experts)

    def simulate(self, n_sim, seed=(0, 0)):
        weights = self.parameters["weights"].get_value()
        sims = self.parent.simulate(n_sim, seed)
        return _tf.einsum("xab,xy->yab", sims, weights)

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
        parent = self.parent.propagate(x, x_var)
        mean, var = parent
        mean = _tf.gather(mean, self.columns, axis=1)
        var = _tf.gather(var, self.columns, axis=1)
        self._explained_var = _tf.gather(
            self.parent._explained_var, self.columns, axis=0)
        jitter = self.parent._input_jitter
        self._input_jitter = None if jitter is None \
            else _tf.gather(jitter, self.columns, axis=0)

        def pick(t):
            return _tf.gather(t, self.columns, axis=-1)

        experts = _map_chain(_chain_of(parent), pick, pick) \
            if _wants_joint(self) else None
        return _Moments(mean, var, experts)

    def simulate(self, n_sim, seed=(0, 0)):
        return _tf.gather(self.parent.simulate(n_sim, seed),
                          self.columns, axis=0)

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
            held = self.parent.inducing_points_covariance
            self.inducing_points_covariance = None if held is None \
                else tuple(_tf.gather(c, self.columns, axis=-1) for c in held)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

class LinearCombination(_Operation):
    """
    Linear combination.

    This node combines the inputs linearly with positive weights.
    """
    _SHIFTS_PARENT_SEEDS = True

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
            # every expert, the active ones under a subset, or the slots as
            # one
            for row in _aligned_points(self.parents, self.root):
                ip = _tf.stack([p for p, _ in row], axis=0)
                ip = _tf.reduce_sum(ip * weights, axis=0)
                all_ip.append(ip)

                ip_var = _tf.stack([v for _, v in row], axis=0)
                ip_var = _tf.reduce_sum(ip_var * weights**2, axis=0)
                all_ip_var.append(ip_var)

            self.inducing_points = tuple(all_ip)
            self.inducing_points_variance = tuple(all_ip_var)
            per_parent = self._parent_weights()
            self.inducing_points_covariance = _combined_covariances(
                _aligned_covariances(self.parents, self.root),
                lambda row: _added(row, per_parent, 2))

    def _parent_weights(self):
        """Each parent's weight, as a scalar or one per output, to act on
        a last axis of outputs."""
        weights = self.parameters["weights"].get_value()
        return [weights[i] for i in range(len(self.parents))]

    def propagate(self, x, x_var=None):
        all_mu = []
        all_var = []
        all_explained_var = []
        chains = []

        for v in self.parents:
            moments = v.propagate(x, x_var)
            mean, var = moments
            chains.append(_chain_of(moments))
            all_mu.append(_tf.transpose(mean)[:, :, None])
            all_var.append(_tf.transpose(var))
            all_explained_var.append(v._explained_var)

        all_mu = _tf.stack(all_mu, axis=-1)
        all_var = _tf.stack(all_var, axis=-1)
        all_explained_var = _tf.stack(all_explained_var, axis=-1)

        all_mu = _tf.reduce_sum(
            all_mu * self._weights_for(all_mu), axis=-1)
        all_var = _tf.reduce_sum(
            all_var * self._weights_for(all_var) ** 2, axis=-1)
        self._explained_var = _tf.reduce_sum(
            all_explained_var * self._weights_for(all_explained_var) ** 2,
            axis=-1)
        jitters = _jitters(self.parents)
        if jitters is None:
            self._input_jitter = None
        else:
            jitters = _tf.stack(jitters, axis=-1)
            self._input_jitter = _tf.reduce_sum(
                jitters * self._weights_for(jitters) ** 2, axis=-1)

        experts = _sum_chains(chains, self._parent_weights()) \
            if self.propagates_inducing_points and _wants_joint(self) \
            else None
        return _Moments(_tf.transpose(all_mu[:, :, 0]),
                        _tf.transpose(all_var), experts)

    def simulate(self, n_sim, seed=(0, 0)):
        all_sims = _tf.stack(
            [v.simulate(n_sim, [seed[0] + i, seed[1]])
             for i, v in enumerate(self.parents)], axis=-1)
        return _tf.reduce_sum(
            all_sims * self._weights_for(all_sims), axis=-1)

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
    _SHIFTS_PARENT_SEEDS = True

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

    def propagate(self, x, x_var=None):
        all_mu = []
        all_var = []
        all_explained_var = []

        for p in self.parents:
            mean, var = p.propagate(x, x_var)
            all_mu.append(_tf.transpose(mean)[:, :, None])
            all_var.append(_tf.transpose(var))
            all_explained_var.append(p._explained_var)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)

        weights = (all_explained_var / (all_var + 1e-6)) + 1e-6
        weights = weights / _tf.reduce_sum(weights, axis=0, keepdims=True)
        self._sim_state = (weights,)

        w_mu = _tf.reduce_sum(weights[:, :, :, None] * all_mu, axis=0)
        w_var = _tf.reduce_sum(weights * all_var, axis=0)
        self._explained_var = _tf.reduce_sum(
            weights * all_explained_var, axis=0)

        return _tf.transpose(w_mu[:, :, 0]), _tf.transpose(w_var)

    def simulate(self, n_sim, seed=(0, 0)):
        (weights,) = self._swept()
        all_sims = _tf.stack(
            [p.simulate(n_sim, [seed[0] + i, seed[1]])
             for i, p in enumerate(self.parents)], axis=0)
        return _tf.reduce_sum(weights[:, :, :, None] * all_sims, axis=0)

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
    """
    The exponential of a latent variable: a field that is always positive.

    Each output is ``exp(sqrt(amp_scale) * f + amp_mean)`` of its parent's
    output ``f``, the two parameters trained, and its moments are those of
    the log-normal that makes. Meant as an amplitude multiplied into another
    branch, so a field's variability can change from place to place. The
    output is no longer Gaussian, so no inducing points pass through it and
    nothing that needs them can sit above it.

    Parameters
    ----------
    parent
        The latent variable to exponentiate; the output has its size.
    name
        The node's name, numbered within the tree.
    """
    _GAUSSIAN = False

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

    def propagate(self, x, x_var=None):
        with _tf.name_scope("exponentiation_prediction"):
            amp_mean = self.parameters["amp_mean"].get_value()
            amp_scale = self.parameters["amp_scale"].get_value()

            mean, var = self.parent.propagate(x, x_var)
            mu = _tf.transpose(mean)[:, :, None]
            var = _tf.transpose(var)
            explained_var = self.parent._explained_var

            mu = mu * _tf.sqrt(amp_scale) + amp_mean
            var = var * amp_scale
            explained_var = explained_var * amp_scale

            amp_mu = _tf.exp(mu) * (1 + 0.5 * var[:, :, None])
            amp_var = _tf.exp(2 * mu[:, :, 0]) * var * (1 + var)
            self._explained_var = _tf.exp(2 * mu[:, :, 0]) \
                                  * (var + explained_var) \
                                  * (1 + var + explained_var) \
                                  - amp_var

            return _tf.transpose(amp_mu[:, :, 0]), _tf.transpose(amp_var)

    def simulate(self, n_sim, seed=(0, 0)):
        amp_mean = self.parameters["amp_mean"].get_value()
        amp_scale = self.parameters["amp_scale"].get_value()
        sims = self.parent.simulate(n_sim, seed)
        return _tf.exp(sims * _tf.sqrt(amp_scale) + amp_mean)


class Multiply(_Operation):
    """
    The product of latent variables of one size, output by output.

    The mean and variance are those of a product of independent variables,
    and each realization is the product of the parents' realizations. The
    usual use is an amplitude times a field, the amplitude an
    `Exponentiation`. A product of Gaussians is not Gaussian, so no inducing
    points pass through it.

    Parameters
    ----------
    latent_variables
        The latent variables to multiply, all of the same size.
    name
        The node's name, numbered within the tree.
    """
    _GAUSSIAN = False
    _SHIFTS_PARENT_SEEDS = True

    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        self._size = self._common_size()
        self.propagates_inducing_points = False

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)

    def propagate(self, x, x_var=None):
        all_mu = []
        all_var = []
        all_explained_var = []

        for v in self.parents:
            mean, var = v.propagate(x, x_var)
            all_mu.append(_tf.transpose(mean)[:, :, None])
            all_var.append(_tf.transpose(var))
            all_explained_var.append(v._explained_var)

        all_mu = _tf.stack(all_mu, axis=0)
        all_var = _tf.stack(all_var, axis=0)
        all_explained_var = _tf.stack(all_explained_var, axis=0)

        pred_mu = _tf.reduce_prod(all_mu, axis=0)
        pred_var = _tf.reduce_prod(all_mu[:, :, :, 0] ** 2 + all_var, axis=0) \
                   - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0)

        self._explained_var = \
            _tf.reduce_prod(
                all_mu[:, :, :, 0] ** 2 + all_var + all_explained_var,
                axis=0) \
            - _tf.reduce_prod(all_mu[:, :, :, 0] ** 2, axis=0) \
            - pred_var

        return _tf.transpose(pred_mu[:, :, 0]), _tf.transpose(pred_var)

    def simulate(self, n_sim, seed=(0, 0)):
        all_sims = _tf.stack(
            [v.simulate(n_sim, [seed[0] + i, seed[1]])
             for i, v in enumerate(self.parents)], axis=0)
        return _tf.reduce_prod(all_sims, axis=0)

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


class GaussianMixture(_Operation):
    """
    A mixture of latent variables, weighted by the softmax of others.

    `weights` holds one latent variable per component, in the components'
    order: the first latent variable of `weights` weighs the first
    component, the second the second, and so on. At every location the
    softmax of the weights, scaled by a trained amplitude, gives each
    component its share, and the output is the components' sum under those
    shares -- a field that follows one component where its weight
    dominates and passes smoothly to another where the weights change
    places. The amplitude sets how sharp the passage is: a large one makes
    each realization nearly one component at a time, a small one blends
    them.

    The mixture is computed realization by realization, from the
    realizations of the weights and of the components, so it keeps
    whatever those share through common parents. The output is not
    Gaussian: a model trains the likelihood of a leaf above this node on
    its realizations, and no inducing points pass through it.

    Parameters
    ----------
    weights
        A node of one latent variable per component, in the order of
        `components`.
    components
        Two or more nodes of one common size, which is the output's size.
        Their order is the order of the weights' latent variables.
    n_nodes
        Quadrature points over the weights for the moments. A power of
        two keeps the Sobol sequence balanced.
    name
        A name for this node, shown in the printed network and accepted by
        `get_node`. Numbered automatically if omitted.

    Raises
    ------
    SizeIncompatibilityError
        If the weights do not have one latent variable per component, or
        the components differ in size.

    See Also
    --------
    ProductOfExperts : components weighted by their own variances.
    geoml.likelihood.Mixture : a mixture of noise laws, not of fields.

    Notes
    -----
    A component as flexible as the field it blends can fit every regime on
    its own, and then the weights never switch: give the components a
    smoother structure than the passage between regimes (a longer range).

    The mean and variance take the weights as independent of the
    components, and each weight as independent of the others: the softmax
    is averaged over `n_nodes` scrambled Sobol points of the weights'
    marginal Gaussians. Where the weights and the components share a
    parent the moments miss that correlation; the realizations do not.
    """
    _GAUSSIAN = False

    def __init__(self, weights, components, n_nodes=64, name=None):
        components = list(components)
        if len(components) < 2:
            raise ValueError("a mixture needs at least two components")
        super().__init__(weights, *components, name=name)
        self.weights = weights
        self.components = components
        if weights.size != len(components):
            raise SizeIncompatibilityError(
                "%s: one weight per component, but %s has size %d for %d "
                "components" % (self.name, weights.name, weights.size,
                                len(components)))
        sizes = [c.size for c in components]
        if not all(s == sizes[0] for s in sizes):
            raise SizeIncompatibilityError(
                "%s: all components must have the same size. Found %s."
                % (self.name, ", ".join("%s (size %d)" % (c.name, c.size)
                                        for c in components)))
        self._size = sizes[0]
        self.propagates_inducing_points = False

        # the weights' Gaussian sampled once, as `UncertainInputGP` samples
        # its input: scrambled Sobol through the normal quantile, the
        # scramble drawn from the package RNG, kept fixed so that a save
        # replays the same nodes
        self.n_nodes = int(n_nodes)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            points = _rnd.sobol_engine(len(components), _rnd.rng()) \
                .random(self.n_nodes)
        nodes = _special.ndtri(points)
        self._add_parameter("nodes", _gpr.RealParameter(
            nodes, _np.full_like(nodes, -10.0), _np.full_like(nodes, 10.0),
            fixed=True))
        # the weights' variance multiplier: a GP node's prior variance is
        # one, which caps how sharp the softmax of its realizations can be
        self._add_parameter("amplitude", _gpr.PositiveParameter(1.0, 0.01, 100.0))

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def _mixture_moments(self, w_mu, w_var, c_mu, c_var):
        """Mean and variance of the mixture, `[n, size]` each, from the
        weights' moments `[n, K]` and the components' `[K, n, size]`."""
        nodes = self.parameters["nodes"].get_value()            # [q, K]
        # the softmax at every node; the small constant keeps the root's
        # derivative finite where a variance is exactly zero
        latent = w_mu[None, :, :] \
            + _tf.sqrt(w_var + 1e-12)[None, :, :] * nodes[:, None, :]
        shares = _tf.nn.softmax(latent, axis=2)                 # [q, n, K]
        first = _tf.reduce_mean(shares, axis=0)                 # [n, K]
        second = _tf.reduce_mean(
            shares[:, :, :, None] * shares[:, :, None, :], axis=0)  # [n, K, K]

        mean = _tf.einsum("nk,knp->np", first, c_mu)
        raw = _tf.einsum("nkl,knp,lnp->np", second, c_mu, c_mu) \
            + _tf.einsum("nkk,knp->np", second, c_var)
        return mean, _tf.maximum(raw - mean ** 2, 0.0)

    def propagate(self, x, x_var=None):
        with _tf.name_scope("gaussian_mixture_prediction"):
            amplitude = self.parameters["amplitude"].get_value()
            w_mu, w_var = self.weights.propagate(x, x_var)
            w_mu = w_mu * _tf.sqrt(amplitude)
            w_var = w_var * amplitude
            w_exp = _tf.transpose(self.weights._explained_var) * amplitude
            c_mu, c_var, c_exp = [], [], []
            for c in self.components:
                mean, var = c.propagate(x, x_var)
                c_mu.append(mean)
                c_var.append(var)
                c_exp.append(_tf.transpose(c._explained_var))
            c_mu = _tf.stack(c_mu, axis=0)
            c_var = _tf.stack(c_var, axis=0)
            c_exp = _tf.stack(c_exp, axis=0)

            mean, var = self._mixture_moments(w_mu, w_var, c_mu, c_var)
            # what conditioning explained away: the variance with the
            # explained parts put back, less the variance without them
            _, total = self._mixture_moments(
                w_mu, w_var + w_exp, c_mu, c_var + c_exp)
            self._explained_var = _tf.transpose(_tf.maximum(total - var, 0.0))
            return mean, var

    def simulate(self, n_sim, seed=(0, 0)):
        # one seed for every parent, as `Stack` does: a node the weights and
        # a component share then draws the same realizations on both paths
        amplitude = self.parameters["amplitude"].get_value()
        shares = _tf.nn.softmax(
            self.weights.simulate(n_sim, seed) * _tf.sqrt(amplitude), axis=0)
        sims = _tf.stack([c.simulate(n_sim, seed) for c in self.components],
                         axis=0)                        # [K, size, n, n_sim]
        return _tf.reduce_sum(shares[:, None, :, :] * sims, axis=0)


class Add(_Operation):
    """
    The sum of latent variables of one size, output by output.

    Means, variances and realizations add up, the parents taken as
    independent. Inducing points pass through, summed, when every parent
    passes its own on and all of them grow from one input, so a GP can sit
    on the sum -- a trend plus a residual, or structures at several scales.

    Parameters
    ----------
    latent_variables
        The latent variables to add, all of the same size.
    name
        The node's name, numbered within the tree.
    """
    _SHIFTS_PARENT_SEEDS = True

    def __init__(self, *latent_variables, name=None):
        super().__init__(*latent_variables, name=name)
        self._size = self._common_size()
        self.propagates_inducing_points = self.same_root and all([p.propagates_inducing_points for p in self.parents])

    def refresh(self, jitter=1e-6):
        for lat in self.parents:
            lat.refresh(jitter)

        if self.propagates_inducing_points:
            all_ip, all_ip_var = [], []
            # every expert, the active ones under a subset, or the slots as
            # one
            for row in _aligned_points(self.parents, self.root):
                ip = _tf.stack([p for p, _ in row], axis=0)
                ip = _tf.reduce_sum(ip, axis=0)
                all_ip.append(ip)

                ip_var = _tf.stack([v for _, v in row], axis=0)
                ip_var = _tf.reduce_sum(ip_var, axis=0)
                all_ip_var.append(ip_var)

            self.inducing_points = tuple(all_ip)
            self.inducing_points_variance = tuple(all_ip_var)
            self.inducing_points_covariance = _combined_covariances(
                _aligned_covariances(self.parents, self.root), _added)

    def propagate(self, x, x_var=None):
        all_mu = []
        all_var = []
        all_explained_var = []
        chains = []

        for v in self.parents:
            moments = v.propagate(x, x_var)
            mean, var = moments
            chains.append(_chain_of(moments))
            all_mu.append(_tf.transpose(mean)[:, :, None])
            all_var.append(_tf.transpose(var))
            all_explained_var.append(v._explained_var)

        all_mu = _tf.stack(all_mu, axis=-1)
        all_var = _tf.stack(all_var, axis=-1)
        all_explained_var = _tf.stack(all_explained_var, axis=-1)

        all_mu = _tf.reduce_sum(all_mu, axis=-1)
        all_var = _tf.reduce_sum(all_var, axis=-1)
        self._explained_var = _tf.reduce_sum(all_explained_var, axis=-1)
        jitters = _jitters(self.parents)
        self._input_jitter = None if jitters is None else _tf.add_n(jitters)

        experts = _sum_chains(chains) \
            if self.propagates_inducing_points and _wants_joint(self) \
            else None
        return _Moments(_tf.transpose(all_mu[:, :, 0]),
                        _tf.transpose(all_var), experts)

    def simulate(self, n_sim, seed=(0, 0)):
        all_sims = _tf.stack(
            [v.simulate(n_sim, [seed[0] + i, seed[1]])
             for i, v in enumerate(self.parents)], axis=-1)
        return _tf.reduce_sum(all_sims, axis=-1)

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
            self.inducing_points_covariance = \
                self.parent.inducing_points_covariance

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def propagate(self, x, x_var=None):
        bias = self.parameters["bias"].get_value()
        parent = self.parent.propagate(x, x_var)
        mean, var = parent
        self._explained_var = self.parent._explained_var
        self._input_jitter = self.parent._input_jitter
        experts = _map_chain(_chain_of(parent), lambda m: m + bias,
                             lambda v: v) if _wants_joint(self) else None
        return _Moments(mean + bias[None, :], var, experts)

    def simulate(self, n_sim, seed=(0, 0)):
        bias = self.parameters["bias"].get_value()
        return self.parent.simulate(n_sim, seed) + bias[:, None, None]

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
            held = self.parent.inducing_points_covariance
            self.inducing_points_covariance = None if held is None \
                else tuple(c * scale[0] for c in held)

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def propagate(self, x, x_var=None):
        scale = self.parameters["scale"].get_value()
        parent = self.parent.propagate(x, x_var)
        mean, var = parent
        self._explained_var = self.parent._explained_var * scale[:, None]
        jitter = self.parent._input_jitter
        self._input_jitter = None if jitter is None \
            else jitter * scale[:, None]
        experts = _map_chain(_chain_of(parent),
                             lambda m: m * _tf.sqrt(scale),
                             lambda v: v * scale) \
            if _wants_joint(self) else None
        return _Moments(mean * _tf.sqrt(scale[None, :]), var * scale[None, :],
                        experts)

    def simulate(self, n_sim, seed=(0, 0)):
        scale = self.parameters["scale"].get_value()
        return self.parent.simulate(n_sim, seed) \
            * _tf.sqrt(scale[:, None, None])

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
            # deterministic, one column per output
            self.inducing_points_variance = tuple(
                _tf.zeros_like(ip) for ip in self.inducing_points
            )

    def kl_divergence(self):
        return _tf.constant(0.0, _tf.float64)

    def propagate(self, x, x_var=None):
        mean, _ = self.parent.propagate(x, x_var)
        trend = self.compute_trend(_tf.transpose(mean))
        self._sim_state = (trend,)
        self._explained_var = _tf.zeros_like(trend)
        # certain, on a certain input: a random one is refused under the
        # expected kernel, its variance being dropped here
        experts = _held_by_all(self, _tf.transpose(trend)) \
            if _wants_joint(self) else None
        return _Moments(_tf.transpose(trend),
                        _tf.zeros_like(_tf.transpose(trend)), experts)

    def simulate(self, n_sim, seed=(0, 0)):
        (trend,) = self._swept()
        return _tf.tile(trend[:, :, None], [1, 1, n_sim])

    def predict_directions(self, x, dir_x, step=1e-3):
        mu, var, explained_var = self.parent.predict_directions(x, dir_x, step)

        grad = self.compute_trend_gradient(mu)
        mu = _tf.reduce_sum(grad * dir_x[:, None, :], axis=2)
        var = _tf.zeros_like(mu[:, :, 0])
        explained_var = _tf.zeros_like(mu[:, :, 0])

        return mu, var, explained_var


class GPWalk(_FunctionalLatentVariable):
    """
    A walk along an uncertain vector field.

    This node uses the vector field defined by its parent to move points in
    space: `n_steps` steps of `step` times the field, scaled by a trained
    amplitude, the field read again where each step lands. It learns
    non-stationary patterns -- a GP above it reads coordinates that the
    field has stretched and folded -- at the cost of the steps.

    The node's parent (a GP) defines the vector field and the parent's
    parent contains the coordinates that will be moved. Both must have the
    same size.

    Under the expected kernel (`GPOptions(propagation="joint")`, the default
    since 0.9.0) the field is one random field, the same at every step: a
    point carries its uncertainty and its covariance with every other point
    along, the field is read under the expected kernel with the uncertainty
    accumulated so far -- so an uncertain walker reads a weaker field and
    slows down -- and more steps refine the path rather than adding noise.
    `step * n_steps * amp` is the walk's reach. The field's variance that
    its inducing points leave unexplained -- the whole prior far from them --
    moves each point on its own, so far from the data a walk is uncertain.
    Each realization walks a realization of the field. The walk adds no
    random variable of its own: the field's KL prices the deformation.
    Under the marginal rule of the versions before, a KL term prices the
    inducing points' displacement against the walk's spread instead, and
    `precision` shrinks the variance at each step; both are deprecated, and
    ignored here.

    Build the GP that reads the walk with `isotropic=True`. The walk
    already bends the space, so a range per dimension in its reader is a
    second way to say the same thing, and training settles the trade on a
    reader stretched along one axis over a near-certain walk -- intervals
    too narrow on new data. The anisotropy the model starts from belongs in
    the input's transform, where it carries what is known beforehand, and
    the reader's one range is relative to it.
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

        # the field is read by interpolating a GP (`_GPNode.interpolate`)
        if not isinstance(parent, _GPNode):
            raise NodeIncompatibilityError(
                "%s: the parent must be a GP node, whose field moves the "
                "points; found %s, a %s"
                % (self.name, parent.name, type(parent).__name__))
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
        # under the expected kernel, the walked inducing points'
        # sensitivities to their starts and to the field (see `_joint_walk`)
        self.walk_a = None
        self.walk_h = None

    def cache_prediction_state(self):
        super().cache_prediction_state()
        if _slots_of(self.root) is None and self.walk_a is not None:
            self.walk_a = self._cache_tuple("walk_a", self.walk_a)
            self.walk_h = self._cache_tuple("walk_h", self.walk_h)

    def _walk(self, x, x_var=None):
        """The moment stepping, stash-free: `refresh` walks the inducing
        points through here, and a stamp from that walk must not shadow the
        one a prediction's own sweep writes."""
        walker_mu, walker_var = self.walker.propagate(x, x_var)

        amp = self.parameters["amp"].get_value()
        prec = self.parameters["precision"].get_value()

        for _ in range(self.n_steps):
            field_mu, field_var = self.field.interpolate(
                walker_mu, walker_var)
            field_mu = _tf.transpose(field_mu[:, :, 0])
            field_var = _tf.transpose(field_var)

            walker_mu = walker_mu + self.step * field_mu * amp
            walker_var = walker_var + self.step * field_var * amp ** 2

            # Kalman filtering
            walker_var = walker_var / (prec + 1)

        return walker_mu, walker_var

    # ------------------------------------------------------------------ #
    # under the expected kernel
    # ------------------------------------------------------------------ #
    # The field is a GP whose realizations are `k(u, U) (alpha + R eta) + b`
    # -- `R` its `chol_r`, `eta` standard normals, `U` its inducing inputs,
    # the walker's outputs there -- one random field, the same at every
    # step. A point's deviation from its mean path is carried, linearized,
    # as its sensitivity to its own start (`a`, `[n, d, d]`) and to `eta`
    # (`h`, `[n, d, d, m]`): its variance and its covariance with any other
    # point, the walked inducing points included, follow in closed form. The
    # field is read at each step under the expected kernel with the
    # uncertainty accumulated so far, and its slope taken as the expected
    # gradient (Stein's lemma), so an uncertain walker reads a weaker,
    # smoother field. The covariance between a point's deviation and the
    # field values it meets is left out of the mean (second order).

    def _fields(self):
        """The field's state for each expert the walk computes: its inducing
        inputs and their variance, `alpha`, `chol_r`, `(K + D)^-1`, the bias
        and, under slots, the mask of real points -- one tuple per active
        expert, or one with a leading slot axis."""
        field = self.field
        if _slots_of(self.root) is not None:
            u, u_var = _slot_points(self.walker)
            return [(u, u_var, field.slots_alpha, field.slots_chol_r,
                     field.slots_cov_smooth_inv,
                     field.slots_bias[:, None, None], _slot_mask(self.root))]
        ids = _active_experts(self.root)
        us, u_vars = _points_of(self.walker, ids)
        return [(us[p], u_vars[p], field.alpha[p], field.chol_r[p],
                 field.cov_smooth_inv[p],
                 field.parameters["bias_%d" % i].get_value(), None)
                for p, i in enumerate(ids)]

    def _joint_walk(self, mean, var0, cov0, field):
        """Walks points from `mean` `[..., n, d]`, their variance `var0` and
        their covariance with the field's inducing inputs `cov0`
        `[..., n, m, d]` (None for none), along one expert's field. Returns
        the end `mean`, the sensitivities `a` and `h`, and the field's
        variance at the start `[..., d, n]`, which weighs the experts."""
        u, u_var, alpha, root_r, smooth_inv, bias, mask = field
        step = self.step * self.parameters["amp"].get_value()
        size = self.size
        eye = _tf.eye(size, dtype=_tf.float64)
        mean = _tf.broadcast_to(mean, _tf.concat(
            [_tf.shape(u)[:-2], _tf.shape(mean)[-2:]], 0)) \
            if mask is not None else mean
        lead = _tf.shape(mean)[:-1]
        a = _tf.broadcast_to(eye, _tf.concat([lead, [size, size]], 0))
        h = _tf.zeros(_tf.concat([lead, [size, size, _tf.shape(u)[-2]]], 0),
                      _tf.float64)
        # the field's variance its inducing points do not explain, 1 - k K^-1
        # k, small near them and the whole prior far from them: carried as
        # each point's sensitivity to normals of its own, the same along its
        # path and shared with no other point
        r = _tf.zeros(_tf.concat([lead, [size, size]], 0), _tf.float64)
        start_var = None
        for k in range(self.n_steps):
            v = _tf.reduce_sum(h ** 2, axis=[-2, -1]) \
                + _tf.reduce_sum(r ** 2, axis=-1)
            if var0 is not None:
                v = v + _tf.einsum("...de,...e->...d", a ** 2, var0)
            c = None if cov0 is None else \
                cov0 * _tf.linalg.diag_part(a)[..., :, None, :]
            # the field at the walkers, and its expected slope there, the
            # derivative in the walkers' mean in closed form
            cov, grad = self.field.expected_covariance(
                mean, v, u, u_var, c, gradient=True)
            if mask is not None:
                cov = cov * mask[..., None, :]
                grad = grad * mask[..., None, :, None]
            f = _tf.einsum("...nm,...smo->...ns", cov, alpha) + bias
            jac = _tf.einsum("...nmd,...smo->...nsd", grad, alpha)
            # a walker the field's uncertainty has pushed is correlated with
            # the field it meets: E[k(w, U) R eta] = E[grad k] Cov(w, eta) R
            # (Stein's lemma), zero at the first step
            if k > 0:
                f = f + _tf.einsum("...njd,...sjl,...ndsl->...ns",
                                   grad, root_r, h)
            if k == 0:
                explained = _tf.reduce_sum(
                    _tf.einsum("...nm,...sml->...snl", cov, smooth_inv)
                    * cov[..., None, :, :], axis=-1)
                start_var = _tf.maximum(1.0 - explained, 0.0)
            g = _tf.einsum("...nm,...sml->...snl", cov, root_r)
            mean = mean + step * f
            a = a + step * _tf.einsum("...ij,...jk->...ik", jac, a)
            h = h + step * (_tf.einsum("...ij,...jsq->...isq", jac, h)
                            + _tf.einsum("...snq,ds->...ndsq", g, eye))
            # (K + D)^-1 and R R^T = K^-1 - (K + D)^-1 together give
            # k K^-1 k, and what is left of the prior is unexplained
            smooth = _tf.reduce_sum(
                _tf.einsum("...nm,...sml->...snl", cov, smooth_inv)
                * cov[..., None, :, :], axis=-1)
            left = _tf.maximum(
                1.0 - smooth - _tf.reduce_sum(g ** 2, axis=-1), 0.0)
            positive = left > 0.0
            sd = _tf.where(positive, _tf.sqrt(_tf.where(positive, left, 1.0)),
                           _tf.zeros_like(left))
            r = r + step * (_tf.einsum("...ij,...js->...is", jac, r)
                            + _tf.einsum("...sn,ds->...nds", sd, eye))
        return mean, a, h, r, start_var

    @staticmethod
    def _walked_covariance(a_x, h_x, a_y, h_y, cov0):
        """The covariance between the ends of two sets of walks, `[..., n,
        m, d]`: through the field, and through their starts' covariance
        `cov0` where they had one."""
        cov = _tf.einsum("...ndsq,...jdsq->...njd", h_x, h_y)
        if cov0 is not None:
            cov = cov + _tf.einsum("...nde,...jde,...nje->...njd",
                                   a_x, a_y, cov0)
        return cov

    @staticmethod
    def _walked_variance(a, h, r, var0):
        v = _tf.reduce_sum(h ** 2, axis=[-2, -1]) \
            + _tf.reduce_sum(r ** 2, axis=-1)
        if var0 is not None:
            v = v + _tf.einsum("...de,...e->...d", a ** 2, var0)
        return v

    def _joint_propagate(self, x, x_var):
        walker = self.walker.propagate(x, x_var)
        chain = _chain_of(walker)
        if chain is None:
            chain = _held_by_all(self, walker[0])
        slots = _slots_of(self.root)
        fields = self._fields()
        held_a = (self.slots_walk_a,) if slots is not None else self.walk_a
        held_h = (self.slots_walk_h,) if slots is not None else self.walk_h
        means, variances, covariances, start = [], [], [], []
        for e, field, a_z, h_z in zip(chain, fields, held_a, held_h):
            mean, a, h, r, start_var = self._joint_walk(
                e.mean, e.variance, e.covariance, field)
            means.append(mean)
            variances.append(self._walked_variance(a, h, r, e.variance))
            if _wants_joint(self):
                covariances.append(self._walked_covariance(
                    a, h, a_z, h_z, e.covariance))
            start.append(start_var)
        if slots is not None:
            raw_w = ((1.0 - start[0]) / (start[0] + 1e-6) + 1e-6) \
                * slots.mask[:, None, None]
            weights = raw_w / _tf.reduce_sum(raw_w, axis=0, keepdims=True)
            stacked_mean = _tf.transpose(means[0], [0, 2, 1])
            stacked_var = _tf.transpose(variances[0], [0, 2, 1])
        else:
            weights = _GPNode.get_expert_weights(_tf.stack(start, axis=0))
            stacked_mean = _tf.stack([_tf.transpose(m) for m in means], 0)
            stacked_var = _tf.stack([_tf.transpose(v) for v in variances], 0)
        w_mean = _tf.reduce_sum(stacked_mean * weights, axis=0)
        w_var = _tf.reduce_sum(stacked_var * weights, axis=0)
        experts = tuple(_Joint(m, v, c) for m, v, c in
                        zip(means, variances, covariances)) \
            if _wants_joint(self) else None
        self._sim_state = ("joint", weights)
        self._explained_var = _tf.zeros_like(w_var)
        return _Moments(_tf.transpose(w_mean), _tf.transpose(w_var), experts)

    def _joint_refresh(self):
        """The inducing points walked along each expert's field: their means,
        variances and, where a GP node reads them, covariances, with the
        sensitivities `propagate` needs for the data's covariance with
        them."""
        slots = _slots_of(self.root)
        if slots is not None:
            u, u_var = _slot_points(self.walker)
            starts = [(u, u_var, _slot_covariance(self.walker))]
        else:
            ids = _active_experts(self.root)
            us, u_vars = _points_of(self.walker, ids)
            starts = list(zip(us, u_vars, _covariances_of(self.walker, ids)))
        points, variances, covariances, a_all, h_all = [], [], [], [], []
        for (u, u_var, cov0), field in zip(starts, self._fields()):
            mean, a, h, r, _ = self._joint_walk(u, u_var, cov0, field)
            cov = self._walked_covariance(a, h, a, h, cov0)
            cov = 0.5 * (cov + _tf.einsum("...njd->...jnd", cov))
            # each point's own share of the unexplained variance, on the
            # diagonal only
            own = _tf.reduce_sum(r ** 2, axis=-1)                # [..., m, d]
            m_z = _tf.shape(own)[-2]
            cov = cov + _tf.eye(m_z, dtype=_tf.float64)[:, :, None] \
                * own[..., :, None, :]
            var = _tf.linalg.matrix_transpose(_tf.linalg.diag_part(
                _tf.einsum("...njd->...dnj", cov)))
            if slots is not None:
                m = u.shape[-2]
                points.append(_tf.reshape(mean, [-1, self.size]))
                variances.append(_tf.reshape(var, [-1, self.size]))
                covariances.append(_tf.reshape(cov, [-1, m, self.size]))
            else:
                points.append(mean)
                variances.append(var)
                covariances.append(cov)
            a_all.append(a)
            h_all.append(h)
        self.inducing_points = tuple(points)
        self.inducing_points_variance = tuple(variances)
        self.inducing_points_covariance = tuple(covariances) \
            if _wants_joint(self) else None
        if slots is not None:
            self.slots_walk_a, self.slots_walk_h = a_all[0], h_all[0]
            self.walk_a = self.walk_h = None
        else:
            self.walk_a, self.walk_h = tuple(a_all), tuple(h_all)

    def _joint_simulate(self, n_sim, seed):
        """Each realization walks a realization of the field -- the field's
        own normals, so that realization s of the walk rides realization s
        of the field -- from the walker's realization, and the experts are
        blended as the moments are."""
        _, weights = self._swept()
        step = self.step * self.parameters["amp"].get_value()
        field = self.field
        starts = _tf.transpose(self.walker.simulate(n_sim, seed), [2, 1, 0])
        slots = _slots_of(self.root)
        if slots is not None:
            u, _ = _slot_points(self.walker)
            mask = _slot_mask(self.root)
            m = u.shape[-2]
            drawn = [_tf.pad(_simulation_normals([self.size, n, n_sim], seed,
                                                 key=field.name),
                             [[0, 0], [0, m - n], [0, 0]])
                     for n in self.root.n_ip]
            drawn.append(_tf.zeros([self.size, m, n_sim], _tf.float64))
            rnd = _tf.gather(_tf.stack(drawn), slots.ids)
            coef = field.slots_alpha + _tf.matmul(field.slots_chol_r, rnd)
            groups = [(u, _tf.transpose(coef, [3, 0, 1, 2]),
                       field.slots_bias[:, None, None], mask)]
        else:
            ids = _active_experts(self.root)
            us, _ = _points_of(self.walker, ids)
            groups = []
            for p, i in enumerate(ids):
                rnd = _simulation_normals(
                    [self.size, self.root.n_ip[i], n_sim], seed,
                    key=field.name)
                coef = field.alpha[p] + _tf.matmul(field.chol_r[p], rnd)
                groups.append((us[p], _tf.transpose(coef, [2, 0, 1]),
                               field.parameters["bias_%d" % i].get_value(),
                               None))
        ends = []
        for u, coef, bias, mask in groups:
            def walk(args, u=u, bias=bias, mask=mask):
                position, c = args
                if mask is not None:
                    position = _tf.broadcast_to(
                        position, _tf.concat([_tf.shape(u)[:1],
                                              _tf.shape(position)], 0))
                for _ in range(self.n_steps):
                    k = field.covariance_matrix(position, u)
                    if mask is not None:
                        k = k * mask[:, None, :]
                    position = position + step * (
                        _tf.einsum("...nm,...sm->...ns", k, c) + bias)
                return position
            ends.append(_tf.map_fn(walk, (starts, coef),
                                   fn_output_signature=_tf.float64))
        if slots is not None:
            # [n_sim, slots, n, d] -> [slots, d, n, n_sim]
            stacked = _tf.transpose(ends[0], [1, 3, 2, 0])
        else:
            stacked = _tf.stack([_tf.transpose(e, [2, 1, 0]) for e in ends],
                                axis=0)
        return _tf.reduce_sum(stacked * weights[..., None], axis=0)

    def propagate(self, x, x_var=None):
        if _JOINT_PROPAGATION:
            return self._joint_propagate(x, x_var)
        walker_mu, walker_var = self._walk(x, x_var)
        self._sim_state = (walker_mu, walker_var)
        self._explained_var = _tf.zeros_like(_tf.transpose(walker_var))
        return walker_mu, walker_var

    def refresh(self, jitter=1e-6):
        self.field.refresh(jitter)
        if _JOINT_PROPAGATION:
            self._joint_refresh()
            return
        self.inducing_points_covariance = None
        self.walk_a = self.walk_h = None
        # self.inducing_points, self.inducing_points_variance = self.propagate(
        #     *self.root.get_root_inducing_points()
        # )

        root_ip, root_var = self.root.get_root_inducing_points()
        # every expert, the active ones under a subset, or the slots as one
        positions = [0] if _slots_of(self.root) is not None \
            else _active_experts(self.root)
        all_ip, all_ip_var = [], []
        for i in positions:
            ip, ip_var = self._walk(root_ip[i], root_var[i])
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

        slots = _slots_of(self.root)
        if slots is not None:
            total = _tf.reduce_sum(self.expert_kl_terms() * slots.mask)
        else:
            total = _tf.add_n(self.expert_kl_terms())
        if _JOINT_PROPAGATION:
            # `precision` is ignored under the expected kernel, and given a
            # zero gradient rather than none, which the optimizer would warn
            # about at every trace
            total = total + 0.0 * _tf.reduce_sum(
                self.parameters["precision"].get_value())
        return total

    def expert_kl_terms(self):
        """Each active expert's term of the walk's KL, in the order of the
        active experts; under slots a tensor over the slots, to which a
        padded point adds nothing.

        Under the marginal rule, the inducing points' displacement against
        the walk's own spread where they land. Under the expected kernel the
        walk adds no random variable of its own, so nothing: the field's KL
        prices the deformation. The displacement term did that job by a
        heuristic -- calibration 1.97 to 1.26 on the folded section's first
        seed, 2.24 on another -- and an isotropic GP reading the walk scored
        -6.1 a new hole on three seeds against its -10.7."""
        if _JOINT_PROPAGATION:
            if _slots_of(self.root) is not None:
                return _tf.zeros([_slots_of(self.root).size], _tf.float64)
            return [_tf.constant(0.0, _tf.float64)
                    for _ in _active_experts(self.root)]
        if _slots_of(self.root) is not None:
            mu_1, _ = _slot_points(self.walker)
            mu_2, var_2 = _slot_points(self)
            terms = 0.5 * _tf.reduce_sum((mu_2 - mu_1) ** 2 / var_2, axis=2)
            return _tf.reduce_sum(terms * _slot_mask(self.root), axis=1)
        ids = _active_experts(self.root)
        mu_1, _ = _points_of(self.walker, ids)
        mu_2, var_2 = _points_of(self, ids)
        return [0.5 * _tf.reduce_sum((b - a) ** 2 / v)
                for a, b, v in zip(mu_1, mu_2, var_2)]

    def compute_path(self, x, x_var=None):
        walker_mu, walker_var = self.walker.propagate(x, x_var)

        amp = self.parameters["amp"].get_value()
        prec = self.parameters["precision"].get_value()

        all_mu = [walker_mu]
        all_var = [walker_var]
        for _ in range(self.n_steps):
            field_mu, field_var = self.field.interpolate(
                walker_mu, walker_var)
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

    def simulate(self, n_sim, seed=(0, 0)):
        if isinstance(self._swept()[0], str):
            return self._joint_simulate(n_sim, seed)
        walker_mu, walker_var = self._swept()
        mu = _tf.transpose(walker_mu)[:, :, None]
        var = _tf.transpose(walker_var)

        # samples are coherent among data points
        rnd = _simulation_normals([self.size, 1, n_sim], seed,
                                  key=self.name)
        return mu + rnd * _tf.sqrt(var[:, :, None])


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
    def __init__(self, parent, size=1, kernel=None, fix_range=False,
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
            The kernel to use for the covariance matrices. A fresh
            `Gaussian` if omitted.
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
        if n_structures < 2:
            raise ValueError("a MultiStructureGP combines at least 2 "
                             "structures, got %r" % (n_structures,))
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
            # leading axes broadcast, as in `BasicGP.covariance_matrix`
            var_x = var_x[..., :, None, :]
            var_y = var_y[..., None, :, :]

            # [..., n_data, n_data, n_dim]
            dif = x[..., :, None, :] - y[..., None, :, :]

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

    def expected_covariance(self, mean_x, var_x, mean_y, var_y, cov=None,
                            gradient=False):
        weights = self.parameters["weights"].get_value()
        parts = [_expected_kernel(self.kernel,
                                  self.parameters[f"ranges_{n}"].get_value(),
                                  mean_x, var_x, mean_y, var_y, cov, gradient)
                 for n in range(self.n_structures)]
        if not gradient:
            return _tf.add_n([p * weights[n] for n, p in enumerate(parts)])
        return (_tf.add_n([p[0] * weights[n] for n, p in enumerate(parts)]),
                _tf.add_n([p[1] * weights[n] for n, p in enumerate(parts)]))

    def _plain_covariance(self, x, y):
        weights = self.parameters["weights"].get_value()
        return _tf.add_n([
            self.kernel.kernelize(_plain_distance(
                x, y, self.parameters[f"ranges_{n}"].get_value())) * weights[n]
            for n in range(self.n_structures)])

    def _kernel_items(self):
        # every structure's components, each weighted by its structure: a
        # pair across two structures pairs Gaussians of two ranges
        weights = self.parameters["weights"].get_value()
        return [item for n in range(self.n_structures)
                for item in _kernel_items(
                    self.kernel, self.parameters[f"ranges_{n}"].get_value(),
                    weights[n])]


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

    def propagate(self, x, x_var=None):
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

            weights = _GPNode.get_expert_weights(var)

            w_mu = _tf.reduce_sum(_tf.stack(mu, axis=0) * weights[:, :, :, None], axis=0)
            w_var = _tf.reduce_sum(_tf.stack(var, axis=0) * weights, axis=0)

            self._sim_state = (cov_cross, mu, weights)
            self._explained_var = _tf.reduce_sum(
                _tf.stack(explained_var, axis=0) * weights, axis=0)

            return _tf.transpose(w_mu[:, :, 0]), _tf.transpose(w_var)

    def simulate(self, n_sim, seed=(0, 0)):
        cov_cross, mu, weights = self._swept()
        with _tf.name_scope("constrained_root_simulation"):
            rnd = [
                _simulation_normals([self.size, n + d, n_sim], seed,
                                    key=self.name)
                for n, d in zip(self.n_ip, self.n_dir)
            ]
            sims = [
                _tf.einsum("ab,sbc->sac", a, _tf.matmul(b, c)) + d
                for a, b, c, d in zip(cov_cross, self.chol_r, rnd, mu)
            ]
            return _tf.reduce_sum(
                _tf.stack(sims, axis=0) * weights[:, :, :, None], axis=0)


# --------------------------------------------------------------------------- #
# the catalogue
# --------------------------------------------------------------------------- #
# What `geoml.catalogue` cannot read off a node: its category, its parents,
# how its output size follows from its arguments, and whether inducing points
# pass through it -- "parents" where they do exactly when every parent passes
# them on and all share one root. The arguments' types are declared too, the
# constructors here carrying no annotations. `test_catalogue.py` builds every
# node against these claims.
_NAME = {"type": "str"}
_SIZE = {"type": "int", "size_param": True, "constraints": {"min": 1}}
# `parents` is a list of slots, one per constructor parameter that takes
# parents -- empty for an input, two for `GaussianMixture`
_NO_PARENTS = []
_ONE_PARENT = [{"param": "parent", "min": 1, "max": 1}]
_PARENTS = [{"param": "latent_variables", "min": 1, "max": None}]


def _declared(category, parents, sizing, propagates, requires=False,
              label=None, stability=None, **params):
    entry = {"category": category, "parents": parents, "size": sizing,
             "propagates_inducing": propagates,
             "requires_propagation": requires,
             "params": dict(params, name=_NAME)}
    if label is not None:
        entry["label"] = label
    if stability is not None:
        entry["stability"] = stability
    return entry


# a prior's strength, None switching the prior off
_PRIOR = {"type": "float", "nullable": True,
          "constraints": {"exclusive_min": 0}}
_GP = dict(parent={"type": "node"}, size=_SIZE, kernel={"type": "ref:kernel"},
           fix_range={"type": "bool"}, isotropic={"type": "bool"},
           range_prior=_PRIOR)
_INPUT = dict(inducing_points={"type": "data:PointData"},
              transform={"type": "ref:transform"},
              fix_transform={"type": "bool"}, center={"type": "bool"})

BasicInput._catalogue = _declared(
    "input", _NO_PARENTS, {"rule": "input_dimension"}, True, label="Input",
    **_INPUT)
GaussianInput._catalogue = _declared(
    "input", _NO_PARENTS, {"rule": "input_dimension"}, True,
    label="Uncertain input", stability="experimental", **_INPUT)
GradientConstrainedInput._catalogue = _declared(
    "input", _NO_PARENTS, {"rule": "param", "param": "size"}, True,
    label="Gradient-constrained input",
    inducing_points={"type": "data:PointData"},
    directional_data={"type": "data:DirectionalData"},
    covariance={"type": "ref:covariance"}, size=_SIZE,
    fix_covariance={"type": "bool"})

BasicGP._catalogue = _declared(
    "latent", _ONE_PARENT, {"rule": "param", "param": "size"}, "parents",
    requires=True, label="GP", **_GP)
AdditiveGP._catalogue = _declared(
    "latent", _ONE_PARENT, {"rule": "param", "param": "size"}, "parents",
    requires=True, label="Additive GP", **_GP)
UncertainInputGP._catalogue = _declared(
    "latent", _ONE_PARENT, {"rule": "param", "param": "size"}, "parents",
    requires=True, label="Uncertain-input GP", stability="experimental",
    n_nodes={"type": "int", "constraints": {"min": 1}}, **_GP)
MultiStructureGP._catalogue = _declared(
    "latent", _ONE_PARENT, {"rule": "param", "param": "size"}, "parents",
    requires=True, label="Multi-structure GP",
    n_structures={"type": "int", "constraints": {"min": 2}},
    weight_concentration={"type": "json"},
    **{k: v for k, v in _GP.items() if k != "isotropic"})

Linear._catalogue = _declared(
    "function", _ONE_PARENT, {"rule": "param", "param": "size"}, "parents",
    parent={"type": "node"}, size=_SIZE, unit_norm={"type": "bool"},
    weight_prior=_PRIOR)
SelectInput._catalogue = _declared(
    "function", _ONE_PARENT, {"rule": "len", "param": "columns"}, "parents",
    label="Select", parent={"type": "node"}, columns={"type": "int[]"})
Bias._catalogue = _declared(
    "function", _ONE_PARENT, {"rule": "same_as_parent"}, "parents",
    parent={"type": "node"},
    scale={"type": "float", "constraints": {"exclusive_min": 0}})
Scale._catalogue = _declared(
    "function", _ONE_PARENT, {"rule": "same_as_parent"}, "parents",
    parent={"type": "node"})
RadialTrend._catalogue = _declared(
    "function", _ONE_PARENT, {"rule": "param", "param": "size"}, "parents",
    label="Radial trend", parent={"type": "node"}, size=_SIZE)
GPWalk._catalogue = _declared(
    "function", [dict(_ONE_PARENT[0], category=["latent"])],
    {"rule": "same_as_parent"}, "parents", label="GP walk",
    parent={"type": "node"},
    step={"type": "float", "constraints": {"exclusive_min": 0}},
    n_steps={"type": "int", "constraints": {"min": 1}})
Exponentiation._catalogue = _declared(
    "function", _ONE_PARENT, {"rule": "same_as_parent"}, False,
    label="Exp", parent={"type": "node"})

Stack._catalogue = _declared(
    "operation", _PARENTS, {"rule": "sum"}, False,
    latent_variables={"type": "node[]"})
Concatenate._catalogue = _declared(
    "operation", _PARENTS, {"rule": "sum"}, "parents",
    latent_variables={"type": "node[]"})
LinearCombination._catalogue = _declared(
    "operation", _PARENTS, {"rule": "common"}, "parents",
    label="Linear combination", latent_variables={"type": "node[]"},
    unit_variance={"type": "bool"}, per_component={"type": "bool"},
    weight_concentration={"type": "float", "nullable": True,
                          "constraints": {"exclusive_min": 1}})
Add._catalogue = _declared(
    "operation", _PARENTS, {"rule": "common"}, "parents",
    latent_variables={"type": "node[]"})
ProductOfExperts._catalogue = _declared(
    "operation", _PARENTS, {"rule": "common"}, False,
    label="Product of experts", latent_variables={"type": "node[]"})
Multiply._catalogue = _declared(
    "operation", _PARENTS, {"rule": "common"}, False,
    latent_variables={"type": "node[]"})
GaussianMixture._catalogue = _declared(
    "operation",
    [{"param": "weights", "min": 1, "max": 1,
      "size": {"rule": "len", "param": "components"}},
     {"param": "components", "min": 2, "max": None}],
    {"rule": "common", "param": "components"}, False,
    label="Gaussian mixture", stability="internal",
    weights={"type": "node"}, components={"type": "node[]"},
    n_nodes={"type": "int", "constraints": {"min": 1}})
