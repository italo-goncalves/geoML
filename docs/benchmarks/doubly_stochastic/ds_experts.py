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

"""The doubly stochastic prototype with several experts.

Under the joint rule each expert of the leaf reads its parent's same
expert: that expert's chain (mean, variance, covariance with its own
inducing points). So each expert draws the parent from its own chain,
conditions its own inducing inputs on the draw, and gives its own
conditional moments; the experts are then blended draw by draw with the
package's weights (the variance each leaves, `_GPNode.get_expert_weights`),
as `_blend` does with moments. The dense path only: every point reads
every expert.
"""
import numpy as np
import tensorflow as tf

from geoml.latent import network as net
import ds_prototype as ds


def _kernel(leaf, h, e, ip, ipv):
    """`ds.conditional_kernel` with one expert's chain and points."""
    r2 = leaf.parameters["ranges"].get_value()[0] ** 2
    if ipv is None:
        ipv = tf.zeros_like(ip)
    if e.covariance is not None:
        gain = tf.math.divide_no_nan(e.covariance, e.variance[:, None, :])
        mz = ip[None, None] + gain[None] * (h - e.mean[None])[:, :, None, :]
        vz = tf.maximum(ipv[None] - gain * e.covariance, 0.0)[None]
    else:
        mz, vz = ip[None, None], ipv[None, None]
    dif2 = (h[:, :, None, :] - mz) ** 2
    omega, weights, constant = net._kernel_components(leaf.kernel)
    total = constant
    for i in range(int(omega.shape[0])):
        den = r2 + 2.0 * omega[i] * vz
        total = total + weights[i] * tf.reduce_prod(
            tf.sqrt(r2 / den), axis=-1) * tf.exp(
            -omega[i] * tf.reduce_sum(dif2 / den, axis=-1))
    return total


def expert_moments(model, x, nodes):
    """The leaf's blended conditional moments at `nodes` `[Q, d]` standard
    normal offsets of each expert's own parent marginal: `[Q, n, size]`
    twice. `nodes` are draws (equal weight) or a grid (weights apart)."""
    leaf = model.leaves[0]
    moments = leaf.parent.propagate(x)
    chain = net._chain_of(moments)
    ids = net._active_experts(leaf.root)
    ips, ipvs = leaf._parent_points(ids)
    mus, vars_ = [], []
    for p, i in enumerate(ids):
        e = chain[p]
        var = e.variance if e.variance is not None \
            else tf.zeros_like(e.mean)
        offsets = nodes[:, None, :] if len(nodes.shape) == 2 else nodes
        h = e.mean[None] + tf.sqrt(var)[None] * offsets
        e = net._Joint(e.mean, var, e.covariance)
        kx = _kernel(leaf, h, e, ips[p], ipvs[p])
        alpha = leaf.alpha[p][:, :, 0]
        bias = leaf.parameters[f"bias_{i}"].get_value()
        mus.append(tf.einsum("snm,am->sna", kx, alpha) + bias)
        explained = tf.einsum("snm,amk,snk->sna", kx,
                              leaf.cov_smooth_inv[p], kx)
        vars_.append(tf.maximum(1.0 - explained, 0.0))
    var = tf.stack(vars_, 0)                                  # [E,Q,n,size]
    w = net._GPNode.get_expert_weights(var)
    return (tf.reduce_sum(tf.stack(mus, 0) * w, 0),
            tf.reduce_sum(var * w, 0))


def data_term(model, x, y, has_value, inputs, nodes, weights):
    mu, var = expert_moments(model, x, nodes)
    q = int(nodes.shape[0])
    size = int(mu.shape[-1])
    n = tf.shape(y)[0]
    hv = tf.reshape(has_value[None] * weights[:, None, None], [-1, size])
    dummy = tf.zeros([q * n, size, 1], tf.float64)
    return model.likelihoods[0].log_lik(
        tf.reshape(mu, [-1, size]), tf.reshape(var, [-1, size]),
        tf.tile(y, [q, 1]), hv, samples=dummy, **inputs[0])


def train(model, iterations, draws=None, hermite=None):
    """Adam on the sampled bound (`draws` fresh offsets a step) or on the
    second-moment bound (`hermite` nodes a dimension)."""
    inputs = [model.data.variables[v].training_input()
              for v in model.variables]
    variables = model.get_unfixed_variables()
    x = tf.constant(model.data.coordinates, tf.float64)
    y = tf.constant(model.y, tf.float64)
    hv = tf.constant(model.has_value, tf.float64)
    counter = tf.Variable(1, dtype=tf.int64)
    mask = ds.uncertain_dims(model.leaves[0].parent)
    if hermite is not None:
        grid, gw = ds.hermite_grid(hermite, mask)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            model._refresh(model.options.jitter)
            if hermite is not None:
                nodes, w = grid, gw
            else:
                # each point its own draws, as in the single-expert runs
                nodes = tf.random.stateless_normal(
                    tf.stack([draws, tf.shape(x)[0], len(mask)]),
                    ds._seed(counter, 5), dtype=tf.float64) * tf.constant(
                    np.array(mask, float), tf.float64)
                w = tf.fill([draws], tf.constant(1.0 / draws, tf.float64))
            data = data_term(model, x, y, hv, inputs, nodes, w)
            kl = tf.add_n([node.kl_divergence() for node in model._nodes()])
            loss = -(data - kl + model.log_prior())
        model.optimizer.apply_gradients(
            zip(tape.gradient(loss, variables), variables))
        counter.assign_add(1)
        return -loss

    import time
    log, began = [], None
    with model._propagation():
        for i in range(iterations):
            if i == 10:
                began = time.perf_counter()
            log.append(float(step()))
            for p in model._all_parameters:
                p.refresh()
    model.training_log.extend(log)
    return (time.perf_counter() - began) / (iterations - 10) \
        if began is not None else None
