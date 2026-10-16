# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
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

"""Doubly stochastic propagation through a geoML network -- a prototype.

Not part of the package. The leaf GP node reads draws of its parent
instead of the expected kernel over the parent's moments (Salimbeni &
Deisenroth 2017). For each draw `h_s` of the parent at the data points the
leaf is evaluated with a certain input, and

  - the data term is (1/S) sum_s E_q(f | h_s)[log p(y | f)], the last
    layer's expectation in closed form by the likelihood's quadrature;
  - the predictive is the mixture of the S conditionals.

The leaf's inducing inputs stay where `refresh` puts them under the joint
rule: the parent's prediction at the root's inducing points, uncertain,
with their covariance with the data points carried by the chain. Two ways
to read them at a draw:

  coupled      the inducing inputs conditioned on the draw through the
               chain's covariance -- the same model the expected kernel
               trains; averaged over draws the mean is the moment model's
               mean exactly, and sampling changes only the spread.
  independent  the inducing inputs' marginal blur, independent of the
               draw -- f at a fixed point beside f at random points, a
               valid joint prior and Salimbeni's bound as written.

Networks: a leaf BasicGP over any parent (here a BasicGP, or a
Concatenate of one with the input), one expert.
"""
import numpy as np
import tensorflow as tf

from geoml.latent import network as net

COUPLED, INDEPENDENT = "coupled", "independent"


def _seed(seed, k):
    return tf.stack([tf.cast(seed, tf.int32), tf.constant(k, tf.int32)])


def _parent_moments(leaf, x):
    """The leaf's parent at `x`: mean and variance `[n, d]` and its chain
    for the one expert (mean, variance, covariance `[n, m, d]` or None)."""
    moments = leaf.parent.propagate(x)
    mean, var = moments[0], moments[1]
    chain = net._chain_of(moments)
    cov = None if chain is None else chain[0].covariance
    return mean, var, cov


def conditional_kernel(leaf, h, mean, var, cov, coupling):
    """The leaf's covariance between certain inputs `h` `[S, n, d]` and its
    inducing inputs, each a Gaussian per dimension: `[S, n, m]`."""
    ip = leaf._parent_points(net._active_experts(leaf.root))
    ip, ipv = ip[0][0], ip[1][0]
    if ipv is None:
        ipv = tf.zeros_like(ip)
    r2 = leaf.parameters["ranges"].get_value()[0] ** 2              # [1, d]
    if coupling == COUPLED and cov is not None:
        gain = tf.math.divide_no_nan(cov, var[:, None, :])          # [n,m,d]
        mz = ip[None, None] + gain[None] * (h - mean[None])[:, :, None, :]
        vz = tf.maximum(ipv[None] - gain * cov, 0.0)[None]          # [1,n,m,d]
    else:
        mz = ip[None, None]
        vz = ipv[None, None]
    dif2 = (h[:, :, None, :] - mz) ** 2
    omega, weights, constant = net._kernel_components(leaf.kernel)
    total = constant
    for i in range(int(omega.shape[0])):
        den = r2 + 2.0 * omega[i] * vz
        total = total + weights[i] * tf.reduce_prod(
            tf.sqrt(r2 / den), axis=-1) * tf.exp(
            -omega[i] * tf.reduce_sum(dif2 / den, axis=-1))
    return total


def leaf_conditionals(leaf, kx):
    """The leaf's conditional mean, variance and the cross-covariance's
    posterior-explained part, given `kx` `[S, n, m]`: `[S, n, size]`."""
    i = net._active_experts(leaf.root)[0]
    alpha = leaf.alpha[0][:, :, 0]
    bias = leaf.parameters[f"bias_{i}"].get_value()
    mu = tf.einsum("snm,am->sna", kx, alpha) + bias
    explained = tf.einsum("snm,amk,snk->sna", kx, leaf.cov_smooth_inv[0], kx)
    return mu, tf.maximum(1.0 - explained, 0.0)


def leaf_moments(model, x, n_draws, seed, coupling=COUPLED):
    """The leaf's conditional moments at S pointwise draws of its parent,
    `[S, n, size]` twice. Call inside `model._propagation()` after
    `model._refresh`."""
    leaf = model.leaves[0]
    mean, var, cov = _parent_moments(leaf, x)
    eps = tf.random.stateless_normal(
        tf.concat([[n_draws], tf.shape(mean)], 0), _seed(seed, 11),
        dtype=tf.float64)
    h = mean[None] + tf.sqrt(var)[None] * eps
    kx = conditional_kernel(leaf, h, mean, var, cov, coupling)
    return leaf_conditionals(leaf, kx)


# --------------------------------------------------------------------------- #
# the analytic bound with the full second moment, by quadrature
# --------------------------------------------------------------------------- #
def uncertain_dims(node):
    """Which output dimensions of `node` are uncertain: a GP node's all,
    an input's none, a concatenation's side by side."""
    if isinstance(node, net.BasicInput):
        return [False] * node.size
    if isinstance(node, net.Stack):
        return sum((uncertain_dims(p) for p in node.parents), [])
    return [True] * node.size


def hermite_grid(k, mask):
    """A tensor Gauss-Hermite grid of k nodes in each uncertain dimension
    of a standard normal, zero in the certain ones: nodes `[Q, d]` and
    weights `[Q]` summing to one."""
    z, w = np.polynomial.hermite_e.hermegauss(k)
    w = w / w.sum()
    d_unc = int(sum(mask))
    grids = np.meshgrid(*([z] * d_unc), indexing="ij")
    pts = np.stack([g.ravel() for g in grids], -1)
    wts = np.prod(np.stack(np.meshgrid(*([w] * d_unc), indexing="ij"),
                           -1).reshape(-1, d_unc), -1)
    full = np.zeros([pts.shape[0], len(mask)])
    full[:, np.flatnonzero(mask)] = pts
    return tf.constant(full, tf.float64), tf.constant(wts, tf.float64)


def quadrature_moments(model, x, k):
    """The leaf's conditional moments at the grid's nodes over its parent
    and the nodes' weights: `[Q, n, size]` twice and `[Q]`."""
    leaf = model.leaves[0]
    nodes, weights = hermite_grid(k, uncertain_dims(leaf.parent))
    mean, var, cov = _parent_moments(leaf, x)
    h = mean[None] + tf.sqrt(var)[None] * nodes[:, None, :]
    kx = conditional_kernel(leaf, h, mean, var, cov, COUPLED)
    mu, v = leaf_conditionals(leaf, kx)
    return mu, v, weights


def quadrature_data_term(model, x, y, has_value, training_inputs, k):
    """sum_q w_q E_q(f | h_q)[log p(y | f)]: every node's rows stacked,
    each weighed through `has_value` (which multiplies the log-density and
    the warping's log-derivative alike)."""
    mu, var, weights = quadrature_moments(model, x, k)
    q = int(weights.shape[0])
    size = int(mu.shape[-1])
    n = tf.shape(y)[0]
    hv = tf.reshape(has_value[None] * weights[:, None, None], [-1, size])
    likelihood = model.likelihoods[0]
    dummy = tf.zeros([q * n, size, 1], tf.float64)
    return likelihood.log_lik(tf.reshape(mu, [-1, size]),
                              tf.reshape(var, [-1, size]),
                              tf.tile(y, [q, 1]), hv, samples=dummy,
                              **training_inputs[0])


# --------------------------------------------------------------------------- #
# field realizations
# --------------------------------------------------------------------------- #
def _residual(node):
    """The variance a node's field realizations leave out at the locations
    its last `propagate` saw, `[n, size]`: 1 - k K^-1 k for a GP node,
    nothing for an input, side by side for a concatenation."""
    if isinstance(node, net.BasicInput):
        (x_tr,) = node._sim_state
        return tf.zeros_like(x_tr)
    if isinstance(node, net.Stack):
        return tf.concat([_residual(p) for p in node.parents], axis=-1)
    cov_cross, _, _ = node._sim_state
    k = cov_cross[0]
    left = 1.0 - tf.reduce_sum((k @ node.cov_inv[0]) * k, axis=1)
    return tf.tile(tf.maximum(left, 0.0)[:, None], [1, node.size])


def leaf_fields(model, x, n_draws, seed, kind="explained",
                coupling=COUPLED):
    """Field realizations of the leaf, `[S, n, size]`: the parent's own
    realizations (geoML's, the part its inducing points explain; with
    `kind="full"` plus the rest as independent noise), and at each the
    leaf drawn the same way."""
    leaf = model.leaves[0]
    mean, var, cov = _parent_moments(leaf, x)
    sims = leaf.parent.simulate(n_draws, [int(seed), 0])            # [d,n,S]
    h = tf.transpose(sims, [2, 1, 0])
    if kind == "full":
        eps = tf.random.stateless_normal(tf.shape(h), _seed(seed, 17),
                                         dtype=tf.float64)
        h = h + tf.sqrt(_residual(leaf.parent))[None] * eps
    kx = conditional_kernel(leaf, h, mean, var, cov, coupling)
    mu, _ = leaf_conditionals(leaf, kx)
    m = int(kx.shape[-1])
    rnd = tf.random.stateless_normal([n_draws, leaf.size, m], _seed(seed, 13),
                                     dtype=tf.float64)
    field = mu + tf.einsum("snm,amk,sak->sna", kx, leaf.chol_r[0], rnd)
    if kind == "full":
        left = tf.maximum(
            1.0 - tf.einsum("snm,mk,snk->sn", kx, leaf.cov_inv[0], kx), 0.0)
        eps = tf.random.stateless_normal(tf.shape(field), _seed(seed, 19),
                                         dtype=tf.float64)
        field = field + tf.sqrt(left)[..., None] * eps
    return field


# --------------------------------------------------------------------------- #
# training on the doubly stochastic bound
# --------------------------------------------------------------------------- #
def ds_data_term(model, x, y, has_value, training_inputs, n_draws, seed,
                 coupling):
    """(1/S) sum_s E_q(f | h_s)[log p(y | f)]: every draw's rows stacked
    as one data set through the likelihood's quadrature, divided by S."""
    mu, var = leaf_moments(model, x, n_draws, seed, coupling)
    size = int(mu.shape[-1])
    mu = tf.reshape(mu, [-1, size])
    var = tf.reshape(var, [-1, size])
    likelihood = model.likelihoods[0]
    dummy = tf.zeros([tf.shape(mu)[0], size, 1], tf.float64)
    total = likelihood.log_lik(mu, var, tf.tile(y, [n_draws, 1]),
                               tf.tile(has_value, [n_draws, 1]),
                               samples=dummy, **training_inputs[0])
    return total / n_draws


def train_ds(model, iterations, n_draws, coupling=COUPLED, fresh=True,
             hermite=None):
    """Adam on the doubly stochastic bound, with the model's optimizer, KL
    and priors; fresh draws each step unless `fresh` is False."""
    training_inputs = [model.data.variables[v].training_input()
                       for v in model.variables]
    variables = model.get_unfixed_variables()
    x = tf.constant(model.data.coordinates, tf.float64)
    y = tf.constant(model.y, tf.float64)
    hv = tf.constant(model.has_value, tf.float64)
    counter = tf.Variable(1, dtype=tf.int64)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            model._refresh(model.options.jitter)
            seed = counter if fresh else tf.constant(1, tf.int64)
            if hermite is not None:
                data = quadrature_data_term(model, x, y, hv,
                                            training_inputs, hermite)
            else:
                data = ds_data_term(model, x, y, hv, training_inputs,
                                    n_draws, seed, coupling)
            kl = tf.add_n([node.kl_divergence() for node in model._nodes()])
            elbo = data - kl + model.log_prior()
            loss = -elbo
        model.optimizer.apply_gradients(
            zip(tape.gradient(loss, variables), variables))
        counter.assign_add(1)
        return elbo

    log, began = [], None
    with model._propagation():
        for i in range(iterations):
            if i == 10:
                began = __import__("time").perf_counter()
            log.append(float(step()))
            for p in model._all_parameters:
                p.refresh()
    model.training_log.extend(log)
    per_iter = (__import__("time").perf_counter() - began) \
        / (iterations - 10) if began is not None else None
    return log, per_iter
