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
"""The expected kernel, E[k(h(x), h(y))] over jointly Gaussian inputs."""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from scipy.stats import norm, qmc

import geoml
from geoml.latent import network as _net

KERNELS = {
    "Gaussian": geoml.kernels.Gaussian,
    "Exponential": geoml.kernels.Exponential,
    "Matern32": geoml.kernels.Matern32,
    "Matern52": geoml.kernels.Matern52,
    "RationalQuadratic": lambda: geoml.kernels.RationalQuadratic(0.7),
}


def _joint_inputs(n=12, dim=2, seed=3):
    """Means, variances and covariances of a random Gaussian input at `n`
    locations, one independent field per dimension, correlated between
    locations: `[n, dim]`, `[n, dim]`, `[n, n, dim]`."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(-0.8, 0.8, [n, dim])
    covs = []
    for _ in range(dim):
        features = rng.normal(size=[n, 3]) * 0.3
        sites = rng.uniform(0, 1, [n, 1])
        smooth = 0.15 * np.exp(-(sites - sites.T) ** 2 / 0.1)
        covs.append(features @ features.T + smooth + 1e-9 * np.eye(n))
    cov = np.stack(covs, axis=-1)
    var = np.stack([np.diag(c) for c in covs], axis=-1)
    return mean, var, cov


def _kernel_values(kernel, d):
    return np.asarray(kernel.kernelize(tf.constant(d, tf.float64)))


def _monte_carlo(kernel, ranges, mean, cov, draws=400_000, seed=11):
    """The average of k over draws of the joint input, and its standard
    error, `[n, n]` each."""
    rng = np.random.default_rng(seed)
    n, dim = mean.shape
    roots = [np.linalg.cholesky(cov[:, :, s]) for s in range(dim)]
    total = np.zeros([n, n])
    square = np.zeros([n, n])
    done = 0
    while done < draws:
        size = min(20_000, draws - done)
        h = np.stack([mean[:, s][None, :]
                      + rng.normal(size=[size, n]) @ roots[s].T
                      for s in range(dim)], axis=-1)       # [size, n, dim]
        d = np.sqrt(np.sum(((h[:, :, None, :] - h[:, None, :, :])
                            / ranges) ** 2, axis=-1))
        k = _kernel_values(kernel, d)
        total += k.sum(0)
        square += (k ** 2).sum(0)
        done += size
    avg = total / draws
    se = np.sqrt(np.maximum(square / draws - avg ** 2, 0.0) / draws)
    return avg, se


def _expected(kernel, ranges, mean, var, cov):
    r = tf.constant(np.asarray(ranges, float).reshape([1, 1, -1]), tf.float64)
    m = tf.constant(mean, tf.float64)
    v = None if var is None else tf.constant(var, tf.float64)
    c = None if cov is None else tf.constant(cov, tf.float64)
    return np.asarray(_net._expected_kernel(kernel, r, m, v, m, v, c))


@pytest.mark.parametrize("name", list(KERNELS))
def test_no_uncertainty_is_the_kernel(name):
    kernel = KERNELS[name]()
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, [40, 2])
    ranges = np.array([0.6, 1.3])
    d = np.sqrt(np.sum(((x[:, None] - x[None]) / ranges) ** 2, -1))
    got = _expected(kernel, ranges, x, None, None)
    # the Matern family through its fitted table of Gaussians
    tolerance = 1e-12 if name == "Gaussian" else 1e-3
    assert np.abs(got - _kernel_values(kernel, d)).max() < tolerance
    # one place, no uncertainty: one, exactly
    assert np.all(np.diag(got) == 1.0)


@pytest.mark.parametrize("name", ["Exponential", "Matern32", "Matern52"])
def test_the_tables_are_the_kernels(name):
    # a positive mixture whose weights sum to one, within 1e-3 of the
    # kernel wherever it reaches -- which bounds the expected kernel's error
    rates, weights = _net._KERNEL_MIXTURES[type(KERNELS[name]())]
    assert np.all(np.asarray(rates) > 0) and np.all(np.asarray(weights) >= 0)
    assert abs(sum(weights) - 1.0) < 1e-12
    d = np.concatenate([np.geomspace(1e-6, 0.1, 200), np.linspace(0.1, 8, 800)])
    table = np.exp(-np.outer(d ** 2, rates)) @ np.asarray(weights)
    assert np.abs(table - _kernel_values(KERNELS[name](), d)).max() < 1e-3


@pytest.mark.parametrize("name", list(KERNELS))
def test_the_gradient_is_the_derivative(name):
    kernel = KERNELS[name]()
    mean, var, cov = _joint_inputs(n=8)
    m = tf.Variable(mean)
    r = tf.constant(np.array([[[0.7, 1.1]]]))
    v = tf.constant(var)
    c = tf.constant(cov)
    with tf.GradientTape() as tape:
        k = _net._expected_kernel(kernel, r, m, v, tf.constant(mean), v, c)
        weights = tf.constant(np.random.default_rng(1).normal(size=[8, 8]))
        loss = tf.reduce_sum(k * weights)
    automatic = np.asarray(tape.gradient(loss, m))
    _, slope = _net._expected_kernel(kernel, r, m, v, tf.constant(mean), v,
                                     c, gradient=True)
    closed = np.einsum("nmd,nm->nd", np.asarray(slope), np.asarray(weights))
    np.testing.assert_allclose(closed, automatic, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", list(KERNELS))
def test_against_monte_carlo(name):
    kernel = KERNELS[name]()
    mean, var, cov = _joint_inputs()
    ranges = np.array([0.7, 1.1])
    got = _expected(kernel, ranges, mean, var, cov)
    avg, se = _monte_carlo(kernel, ranges, mean, cov)
    off = ~np.eye(len(mean), dtype=bool)
    assert np.abs(got - avg)[off].max() < 0.005
    assert (np.abs(got - avg)[off] / se[off]).max() < 5.0


def test_the_covariance_between_locations_matters():
    # the research's finding: dropping the cross terms costs ~60x the error
    kernel = geoml.kernels.Gaussian()
    mean, var, cov = _joint_inputs()
    ranges = np.array([0.7, 1.1])
    avg, _ = _monte_carlo(kernel, ranges, mean, cov, draws=100_000)
    off = ~np.eye(len(mean), dtype=bool)
    joint = np.abs(_expected(kernel, ranges, mean, var, cov) - avg)[off].max()
    alone = np.abs(_expected(kernel, ranges, mean, var, None) - avg)[off].max()
    assert alone > 10 * joint


@pytest.mark.parametrize("name", list(KERNELS))
def test_positive_definite(name):
    kernel = KERNELS[name]()
    mean, var, cov = _joint_inputs(n=40, dim=3, seed=5)
    got = _expected(kernel, [0.4, 0.9, 2.0], mean, var, cov)
    assert np.allclose(got, got.T, atol=1e-14)
    assert np.linalg.eigvalsh(got).min() > -1e-10


@pytest.mark.parametrize("name", list(KERNELS))
def test_gradients_are_finite_on_the_diagonal(name):
    kernel = KERNELS[name]()
    mean, var, cov = _joint_inputs(n=8)
    m = tf.Variable(mean)
    c = tf.Variable(cov)
    r = tf.Variable(np.array([[[0.7, 1.1]]]))
    with tf.GradientTape() as tape:
        v = tf.transpose(tf.linalg.diag_part(tf.transpose(c, [2, 0, 1])))
        k = _net._expected_kernel(kernel, r, m, v, m, v, c)
        loss = tf.reduce_sum(k ** 2)
    grads = tape.gradient(loss, [m, c, r] + list(
        p.variable for p in kernel.all_parameters))
    for g in grads:
        assert g is not None
        assert np.all(np.isfinite(np.asarray(g)))


def _network(build, n_experts=1, kernel=None, seed=2, propagation="joint"):
    """An input, the nodes `build(root)` makes on it, and a leaf GP on top.

    `build` returns the node the leaf reads, the GP nodes on the input
    below it, and a function from draws of those GPs (`{id(gp): [c, p,
    size]}`) and the input's locations (`[p, d]`) to what the leaf reads
    (`[c, p, d']`). The GPs' posteriors are made informative by hand."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", np.sin(x[:, 0] / 20))
    ip = geoml.data.inducing.from_kmeans(data, 30, seed=0)
    if n_experts > 1:
        ip = geoml.data.inducing.experts(ip, n_experts, seed=0)
    root = geoml.latent.BasicInput(ip, geoml.transform.Isotropic(40))
    below, hidden, mapping = build(root)
    leaf = geoml.latent.BasicGP(below, size=1, kernel=kernel)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(
            verbose=False, propagation=propagation,
            expert_propagation="independent" if propagation == "joint"
            else "consensus"))
    for gp in hidden:
        for i in range(root.n_experts):
            gp.parameters["alpha_white_%d" % i].set_value(
                rng.normal(size=[gp.size, root.n_ip[i], 1]))
            gp.parameters["delta_%d" % i].set_value(
                np.full([gp.size, root.n_ip[i]], 0.05))
    leaf.parameters["ranges"].set_value(np.full([1, 1, below.size], 0.6))
    return model, leaf, hidden, mapping, x


def _two_layers(root):
    hidden = geoml.latent.BasicGP(root, size=2)
    return hidden, [hidden], lambda h, r: h[id(hidden)]


def _hidden_posterior(hidden, root, x, e):
    """A GP on the input, expert `e`: its joint posterior at the data and
    at the expert's inducing points, built from the kernel directly --
    means `[p, s]` and covariances `[s, p, p]`, `p` the data then the
    inducing points."""
    x_tr = np.asarray(root.propagate(tf.constant(x))[0])
    z = np.asarray(root.inducing_points[e])
    points = tf.constant(np.concatenate([x_tr, z]))
    k_pp = np.asarray(hidden.covariance_matrix(points, points))
    k_pz = k_pp[:, len(x):]
    alpha = np.asarray(hidden.alpha[e])[:, :, 0]
    bias = float(hidden.parameters["bias_%d" % e].get_value())
    inv = np.asarray(hidden.cov_smooth_inv[e])
    mean = (k_pz @ alpha.T) + bias
    cov = np.stack([k_pp - k_pz @ inv[s] @ k_pz.T for s in range(len(inv))])
    return mean, cov, np.concatenate([x_tr, z])


def _leaf_against_draws(model, leaf, hidden, mapping, x, draws=200_000):
    """The largest difference between the leaf's covariances with each
    expert's inducing points -- with the data and among themselves -- and
    their average over draws of the GPs' posteriors pushed through the
    nodes between."""
    root = leaf.root
    rng = np.random.default_rng(7)
    x = x[:20]
    n = len(x)
    worst = 0.0
    with model._propagation():
        model._refresh(model.options.jitter)
        parent = leaf.parent.propagate(tf.constant(x))
        cov_cross = leaf._expert_moments(parent[0], parent[1],
                                         parent.experts)[0]
        ranges = np.asarray(leaf.parameters["ranges"].get_value())[0, 0]
        for e in range(root.n_experts):
            posteriors = {}
            for gp in hidden:
                mean, cov, points = _hidden_posterior(gp, root, x, e)
                roots = [np.linalg.cholesky(c + 1e-10 * np.eye(len(c)))
                         for c in cov]
                posteriors[id(gp)] = (mean, roots)
            p = len(points)
            total = np.zeros([p, p])
            for _ in range(draws // 10_000):
                h = {key: np.stack([m[:, s] + rng.normal(size=[10_000, p])
                                    @ r[s].T for s in range(len(r))], -1)
                     for key, (m, r) in posteriors.items()}
                g = mapping(h, points)
                d = np.sqrt(np.sum(((g[:, :, None] - g[:, None]) / ranges)
                                   ** 2, -1))
                total += _kernel_values(leaf.kernel, d).sum(0)
            avg = total / draws
            k_xz = np.asarray(cov_cross[e])
            k_zz = np.asarray(leaf.cov[e]) - model.options.jitter * np.eye(
                p - n)
            worst = max(worst, np.abs(k_xz - avg[:n, n:]).max(),
                        np.abs(k_zz - avg[n:, n:]).max())
    return worst


@pytest.mark.parametrize("n_experts", [1, 3])
@pytest.mark.parametrize("name", ["Gaussian", "Matern52"])
def test_a_leaf_reads_its_parents_posterior(n_experts, name):
    args = _network(_two_layers, n_experts, KERNELS[name]())
    assert _leaf_against_draws(*args) < 0.005


def _one_at_a_time(root):
    # a linear map to one output, scaled, shifted, joined to the
    # coordinates and one coordinate dropped: each output's covariance is
    # exact through all of them
    hidden = geoml.latent.BasicGP(root, size=2)
    linear = geoml.latent.Linear(hidden, size=1, unit_norm=False)
    linear.parameters["weights"].set_value([[0.8], [-0.5]])
    scale = geoml.latent.Scale(linear)
    scale.parameters["scale"].set_value([2.0])
    bias = geoml.latent.Bias(scale)
    bias.parameters["bias"].set_value([0.3])
    below = geoml.latent.SelectInput(geoml.latent.Concatenate(bias, root),
                                     [0, 2])

    def mapping(h, r):
        g = 0.3 + np.sqrt(2.0) * (h[id(hidden)] @ np.array([0.8, -0.5]))
        both = np.concatenate(
            [g[..., None], np.broadcast_to(r, g.shape + (2,))], -1)
        return both[..., [0, 2]]

    return below, [hidden], mapping


def _independent_parents(root):
    # sums of GPs that share nothing: the covariances add
    a, b, c = (geoml.latent.BasicGP(root, size=1) for _ in range(3))
    combination = geoml.latent.LinearCombination(geoml.latent.Add(a, b), c)
    combination.parameters["weights"].set_value([0.7, 0.3])
    below = geoml.latent.Concatenate(combination, root)

    def mapping(h, r):
        g = 0.7 * (h[id(a)] + h[id(b)]) + 0.3 * h[id(c)]
        return np.concatenate([g, np.broadcast_to(r, g.shape[:2] + (2,))],
                              -1)

    return below, [a, b, c], mapping


@pytest.mark.parametrize("build", [_one_at_a_time, _independent_parents])
def test_the_operations_carry_the_covariance(build):
    assert _leaf_against_draws(*_network(build)) < 0.005


def _three_layers(root):
    first = geoml.latent.BasicGP(root, size=2)
    second = geoml.latent.BasicGP(first, size=2)
    return second, [first], None


@pytest.mark.parametrize("n_experts", [1, 3])
def test_three_layers_chain_the_posteriors(n_experts):
    """A GP on a GP on a GP: the middle one's posterior, rebuilt here from
    the first one's under the expected kernel, is what the leaf reads."""
    model, leaf, (first,), _, x = _network(_three_layers, n_experts)
    second, root = leaf.parent, leaf.root
    rng = np.random.default_rng(3)
    for i in range(root.n_experts):
        second.parameters["alpha_white_%d" % i].set_value(
            rng.normal(size=[2, root.n_ip[i], 1]))
        second.parameters["delta_%d" % i].set_value(
            np.full([2, root.n_ip[i]], 0.05))
    x = x[:20]
    n = len(x)
    with model._propagation():
        model._refresh(model.options.jitter)
        parent = second.propagate(tf.constant(x))
        cov_cross = leaf._expert_moments(parent[0], parent[1],
                                         parent.experts)[0]
        for e in range(root.n_experts):
            mean, cov, _ = _hidden_posterior(first, root, x, e)
            var = np.stack([np.diag(c) for c in cov], -1)
            k2 = np.asarray(second.expected_covariance(
                tf.constant(mean), tf.constant(var), tf.constant(mean),
                tf.constant(var), tf.constant(np.transpose(cov, [1, 2, 0]))))
            k2_zz = k2[n:, n:] + model.options.jitter * np.eye(len(k2) - n)
            alpha = np.linalg.solve(k2_zz, np.linalg.cholesky(k2_zz)
                                    @ np.asarray(second.parameters[
                                        "alpha_white_%d" % e].get_value()
                                    )[:, :, 0].T)
            bias = float(second.parameters["bias_%d" % e].get_value())
            delta = np.asarray(second.parameters["delta_%d" % e].get_value())
            m2 = k2[:, n:] @ alpha + bias
            c2 = np.stack([k2 - k2[:, n:] @ np.linalg.inv(
                k2_zz + np.diag(delta[s])) @ k2[n:, :] for s in range(2)], -1)
            v2 = np.stack([np.diag(c2[:, :, s]) for s in range(2)], -1)
            k3 = np.asarray(leaf.expected_covariance(
                tf.constant(m2[:n]), tf.constant(v2[:n]),
                tf.constant(m2[n:]), tf.constant(v2[n:]),
                tf.constant(c2[:n, n:])))
            np.testing.assert_allclose(np.asarray(cov_cross[e]), k3,
                                       atol=2e-6)


def test_the_marginal_rule_is_far_off_on_the_same_network():
    # the same check under the rule before 0.9.0 -- what the gate is for
    args = _network(_two_layers, propagation="marginal")
    assert _leaf_against_draws(*args, draws=40_000) > 0.05


def test_a_wide_input_returns_to_the_constant_a_kernel_tends_to():
    # a very uncertain input makes two locations unrelated: the Matern
    # family falls to zero, the rational quadratic to its constant part
    mean = np.zeros([2, 1])
    var = np.full([2, 1], 1e6)
    for name in ("Gaussian", "Exponential", "Matern32", "Matern52"):
        got = _expected(KERNELS[name](), [1.0], mean, var, None)
        assert got[0, 1] < 1e-2


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def _walker_model(depth, propagation, kernel=None, n_experts=2, seed=4):
    geoml.set_seed(seed)
    point, _ = geoml.datasets.walker()
    ip = geoml.data.inducing.experts(
        geoml.data.inducing.from_kmeans(point, 20 * n_experts, seed=0),
        n_experts, seed=0)
    root = geoml.latent.BasicInput(ip, geoml.transform.Isotropic(50))
    node = root
    for _ in range(depth - 1):
        node = geoml.latent.Concatenate(geoml.latent.BasicGP(node, size=1),
                                        root)
    leaf = geoml.latent.BasicGP(node, size=1, kernel=kernel)
    options = geoml.models.GPOptions(
        verbose=False, propagation=propagation,
        expert_propagation="independent")
    return geoml.models.VGPNetwork(point, "V", geoml.likelihood.Gaussian(),
                                   leaf, options=options)


def _grid():
    return geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[10, 10])


def _predicted(model, n_sim=3):
    grid = _grid()
    model.predict(grid, n_sim=n_sim)
    v = grid.variables["V"]
    return (np.asarray(v.latent_mean.values), np.asarray(v.latent_variance.values),
            np.asarray(v.simulations))


def test_a_single_layer_model_is_the_same_under_both_rules():
    # an input is certain, so the expected kernel is the kernel itself
    out = {}
    for rule in ("joint", "marginal"):
        model = _walker_model(1, rule)
        model.train_full(5)
        out[rule] = (np.array(model.training_log),) + _predicted(model)
    for a, b in zip(out["joint"], out["marginal"]):
        assert np.array_equal(a, b)


def test_a_deep_model_differs_between_the_rules():
    out = {}
    for rule in ("joint", "marginal"):
        model = _walker_model(2, rule)
        model.train_full(5)
        out[rule] = _predicted(model)[0]
    assert not np.allclose(out["joint"], out["marginal"])


def test_a_deep_model_is_batch_invariant_and_finite():
    model = _walker_model(3, "joint")
    model.train_full(5)
    whole = _predicted(model)
    model.options.prediction_batch_size = 17
    parts = _predicted(model)
    for a, b in zip(whole, parts):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)


def test_the_options_refuse_the_consensus():
    with pytest.raises(ValueError, match="independent"):
        geoml.models.GPOptions(propagation="joint",
                               expert_propagation="consensus")
    model = _walker_model(2, "joint")
    model.options.expert_propagation = "consensus"
    with pytest.raises(ValueError, match="independent"):
        model.train_full(1)


@pytest.mark.parametrize("kernel", [geoml.kernels.Spherical,
                                    geoml.kernels.Cubic])
def test_a_kernel_that_is_no_mixture_is_refused_on_an_uncertain_input(kernel):
    with pytest.raises(ValueError, match="scale mixture"):
        _walker_model(2, "joint", kernel=kernel())
    # on the input itself it is the kernel it always was
    _walker_model(1, "joint", kernel=kernel())
    # and under the marginal rule nothing is refused
    _walker_model(2, "marginal", kernel=kernel())


def test_cosine_is_refused_in_a_network():
    with pytest.raises(ValueError, match="Cosine"):
        _walker_model(1, "joint", kernel=geoml.kernels.Cosine())


def test_nodes_without_an_expected_kernel_are_refused():
    point, _ = geoml.datasets.walker()
    ip = geoml.data.inducing.from_kmeans(point, 20, seed=0)

    def build(make):
        root = geoml.latent.BasicInput(ip, geoml.transform.Isotropic(50))
        return geoml.models.VGPNetwork(
            point, "V", geoml.likelihood.Gaussian(), make(root),
            options=geoml.models.GPOptions(verbose=False))

    with pytest.raises(ValueError, match="RadialTrend"):
        build(lambda r: geoml.latent.BasicGP(geoml.latent.RadialTrend(
            geoml.latent.BasicGP(r, size=2), size=1)))
    with pytest.raises(ValueError, match="UncertainInputGP"):
        build(lambda r: geoml.latent.BasicGP(
            geoml.latent.UncertainInputGP(r, size=1)))
    # a walk reads its field at uncertain positions
    with pytest.raises(ValueError, match="GPWalk"):
        build(lambda r: geoml.latent.BasicGP(geoml.latent.GPWalk(
            geoml.latent.BasicGP(r, size=2, kernel=geoml.kernels.Cubic()))))
    # on a certain input, read by nothing, they are what they were
    build(lambda r: geoml.latent.UncertainInputGP(r, size=1))
    build(lambda r: geoml.latent.BasicGP(geoml.latent.RadialTrend(r)))


def test_the_rule_is_saved_and_an_older_save_keeps_the_marginal_one(tmp_path):
    import zarr
    model = _walker_model(2, "joint")
    model.train_full(3)
    before = _predicted(model)
    path = str(tmp_path / "joint.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)
    assert loaded.options.propagation == "joint"
    for a, b in zip(before, _predicted(loaded)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)

    # a save from before 0.9.0 names neither option: it opens under the
    # rules it was trained with, and predicts as that model does
    group = zarr.open_group(path, mode="r+")
    meta = dict(group.attrs["geoml_model"])

    def strip(node):
        if isinstance(node, dict):
            if node.get("$") == "options":
                node["values"].pop("propagation", None)
                node["values"]["expert_propagation"] = "consensus"
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    strip(meta)
    group.attrs["geoml_model"] = meta
    old = geoml.models.VGPNetwork.open(path)
    assert old.options.propagation == "marginal"
    model.options.propagation = "marginal"
    model.options.expert_propagation = "consensus"
    for a, b in zip(_predicted(model), _predicted(old)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #
def _walk_model(n_experts, amp=8.0, delta=0.3, seed=6):
    """An input, a field of two outputs on it, a walk, and a GP on the
    walk; the field's posterior made uncertain by hand."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", np.sin(x[:, 0] / 20))
    ip = geoml.data.inducing.from_kmeans(data, 30, seed=0)
    if n_experts > 1:
        ip = geoml.data.inducing.experts(ip, n_experts, seed=0)
    root = geoml.latent.BasicInput(ip, geoml.transform.Isotropic(40))
    field = geoml.latent.BasicGP(root, size=2)
    walk = geoml.latent.GPWalk(field)
    leaf = geoml.latent.BasicGP(walk, size=1)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False))
    for i in range(root.n_experts):
        field.parameters["alpha_white_%d" % i].set_value(
            rng.normal(size=[2, root.n_ip[i], 1]))
        field.parameters["delta_%d" % i].set_value(
            np.full([2, root.n_ip[i]], delta))
    walk.parameters["amp"].set_value(amp)
    return model, walk, x


def _walk_against_draws(model, walk, x, draws=10_000, seed=8):
    """The walk's moments and covariances, expert by expert, against walks
    along `draws` realizations of each expert's field: the largest mean
    error in the draws' standard deviations, the variance ratios' range,
    and the largest covariance error on the scale of a correlation."""
    root, field = walk.root, walk.field
    rng = np.random.default_rng(seed)
    x = x[:20]
    n = len(x)
    step = walk.step * float(walk.parameters["amp"].get_value())
    worst_mean, ratios, worst_cov = 0.0, [], 0.0
    with model._propagation():
        model._refresh(model.options.jitter)
        moments = walk.propagate(tf.constant(x))
        for e in range(root.n_experts):
            chain = moments.experts[e]
            z = np.asarray(root.inducing_points[e])
            starts = np.concatenate(
                [np.asarray(root.propagate(tf.constant(x))[0]), z])
            alpha = np.asarray(field.alpha[e])[:, :, 0]
            root_r = np.asarray(field.chol_r[e])
            bias = float(field.parameters["bias_%d" % e].get_value())
            eta = rng.normal(size=[draws, 2, len(z)])
            coef = alpha[None] + np.einsum("sml,rsl->rsm", root_r, eta)
            # the variance the inducing points leave unexplained, each
            # walker's own normals, the same along its path
            own = rng.normal(size=[draws, len(starts), 2])
            k_inv = np.asarray(field.cov_inv[e])
            p = np.broadcast_to(starts, (draws,) + starts.shape).copy()
            for _ in range(walk.n_steps):
                k = np.asarray(field.covariance_matrix(
                    tf.constant(p), tf.constant(z)))
                left = np.maximum(1.0 - np.einsum(
                    "rnm,ml,rnl->rn", k, k_inv, k), 0.0)
                p = p + step * (np.einsum("rnm,rsm->rns", k, coef) + bias
                                + np.sqrt(left)[..., None] * own)
            mean, var = p.mean(0), p.var(0)
            dev = p - mean
            cov_xz = np.einsum("rnd,rjd->njd", dev[:, :n], dev[:, n:]) / draws
            cov_zz = np.einsum("rnd,rjd->njd", dev[:, n:], dev[:, n:]) / draws
            got_mean = np.concatenate(
                [np.asarray(chain.mean), np.asarray(walk.inducing_points[e])])
            got_var = np.concatenate(
                [np.asarray(chain.variance),
                 np.asarray(walk.inducing_points_variance[e])])
            worst_mean = max(worst_mean,
                             (np.abs(got_mean - mean) / np.sqrt(var)).max())
            ratios.append(got_var / var)
            scale_xz = np.sqrt(var[:n, None, :] * var[None, n:, :])
            scale_zz = np.sqrt(var[n:, None, :] * var[None, n:, :])
            worst_cov = max(
                worst_cov,
                (np.abs(np.asarray(chain.covariance) - cov_xz)
                 / scale_xz).max(),
                (np.abs(np.asarray(walk.inducing_points_covariance[e])
                        - cov_zz) / scale_zz).max())
    ratios = np.concatenate([r.ravel() for r in ratios])
    return worst_mean, (ratios.min(), ratios.max()), worst_cov


@pytest.mark.parametrize("n_experts", [1, 3])
def test_the_walk_against_walks_along_sampled_fields(n_experts):
    # a reach of a tenth of the field's range: accurate to twice it, and
    # past that the linearized spread errs both ways (the roadmap)
    worst_mean, (low, high), worst_cov = _walk_against_draws(
        *_walk_model(n_experts, amp=1.0), draws=40_000)
    assert worst_mean < 0.2
    assert 0.8 < low and high < 1.25
    assert worst_cov < 0.05


def test_a_model_on_a_walk_trains_and_predicts():
    model, walk, x = _walk_model(2)
    model.train_full(5)
    assert np.all(np.isfinite(model.training_log))
    whole = _predicted_v(model)
    model.options.prediction_batch_size = 13
    for a, b in zip(whole, _predicted_v(model)):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)


def _walk_network(propagation):
    geoml.set_seed(3)
    x = np.random.default_rng(3).uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", np.sin(x[:, 0] / 20))
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 30, seed=0),
        geoml.transform.Isotropic(40))
    walk = geoml.latent.GPWalk(geoml.latent.BasicGP(root, size=2))
    leaf = geoml.latent.BasicGP(walk, size=1, isotropic=True)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(
            verbose=False, propagation=propagation,
            expert_propagation="independent" if propagation == "joint"
            else "consensus"))
    return model, walk, leaf


def test_the_walk_adds_no_kl_under_the_expected_kernel():
    for propagation, nothing in (("joint", True), ("marginal", False)):
        model, walk, _ = _walk_network(propagation)
        walk.parameters["amp"].set_value(5.0)
        with model._propagation():
            model._refresh(model.options.jitter)
            kl = float(walk.kl_divergence())
        assert (kl == 0.0) == nothing, (propagation, kl)


def _predicted_v(model):
    grid = geoml.data.Grid2D(start=[0, 0], end=[100, 100], n=[8, 8])
    model.predict(grid, n_sim=3)
    v = grid.variables["v"]
    return (np.asarray(v.latent_mean.values),
            np.asarray(v.latent_variance.values), np.asarray(v.simulations))


def test_every_batch_runs_under_the_model_s_rule():
    # the batches of a prediction, of the measurement samples, of the PIT
    # check and of the responsibilities read the state the refresh made
    # under the model's rule, and must propagate under it too
    model = _walker_model(2, "joint")
    seen = []

    def call(x, x_var, n_splits):
        seen.append((_net._JOINT_PROPAGATION, _net._EXPERT_PROPAGATION))
        return None

    for _ in model._over_batches(_grid(), call):
        pass
    assert seen and all(rule == (True, "independent") for rule in seen)
    samples = model.predict_measurements(_grid(), n_sim=2)
    assert np.all(np.isfinite(samples["V"]))


# --------------------------------------------------------------------------- #
# the second moment
# --------------------------------------------------------------------------- #
NODES = {"basic": geoml.latent.BasicGP,
         "multi": geoml.latent.MultiStructureGP,
         "additive": geoml.latent.AdditiveGP}


def _uncertain_data(x, var=None):
    data = geoml.data.PointData.from_array(x, ["X", "Y"]) if var is None \
        else geoml.data.GaussianData(
            pd.DataFrame({"X": x[:, 0], "Y": x[:, 1], "VX": var[:, 0],
                          "VY": var[:, 1]}), ["X", "Y"], ["VX", "VY"])
    data.add_continuous_variable("v", np.sin(x[:, 0] / 20))
    data.add_continuous_variable("w", np.cos(x[:, 1] / 15))
    return data


def _uncertain_model(kernel=None, node="basic", n_experts=1, seed=5,
                     var=None, likelihood=geoml.likelihood.Gaussian):
    """A GP of two outputs on an uncertain input, its posterior made
    informative by hand."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    data = _uncertain_data(rng.uniform(0, 100, [80, 2]), var)
    ip = geoml.data.inducing.from_kmeans(
        data, 30 if n_experts == 1 else 20 * n_experts, seed=0)
    if n_experts > 1:
        ip = geoml.data.inducing.experts(ip, n_experts, seed=0)
    root = geoml.latent.GaussianInput(ip, geoml.transform.Isotropic(40))
    leaf = NODES[node](root, size=2, kernel=kernel)
    model = geoml.models.VGPNetwork(
        data, {"v": likelihood(), "w": likelihood()}, latent_network=leaf,
        options=geoml.models.GPOptions(verbose=False))
    for i in range(root.n_experts):
        leaf.parameters["alpha_white_%d" % i].set_value(
            rng.normal(size=[2, root.n_ip[i], 1]))
        leaf.parameters["delta_%d" % i].set_value(
            np.full([2, root.n_ip[i]], 0.05))
        leaf.parameters["bias_%d" % i].set_value(0.3)
    if node == "multi":
        leaf.parameters["ranges_0"].set_value(np.full([1, 1, 2], 0.9))
        leaf.parameters["ranges_1"].set_value(np.full([1, 1, 2], 0.35))
        leaf.parameters["weights"].set_value(np.array([0.6, 0.4]))
    else:
        leaf.parameters["ranges"].set_value(np.array([[[0.5, 0.8]]]))
    return model, leaf


def _queries(leaf, level, n=15, seed=9):
    """Locations, and variances `level` times the squared ranges -- the
    longest structure's -- in the input's own units."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(10, 90, [n, 2])
    name = "ranges_0" if "ranges_0" in leaf.parameters else "ranges"
    ranges = np.asarray(leaf.parameters[name].get_value()).ravel() * 40
    return x, np.broadcast_to(level * ranges ** 2, x.shape).copy()


def _node_moments(model, leaf, x, var):
    """The node's mean and variance at the uncertain locations, `[n,
    size]`, and the locations in the transformed space."""
    with model._propagation():
        model._refresh(model.options.jitter)
        mean, variance = leaf.propagate(tf.constant(x), tf.constant(var))
        x_tr, var_tr = leaf.root.propagate(tf.constant(x), tf.constant(var))
    return (np.asarray(mean), np.asarray(variance), np.asarray(x_tr),
            np.asarray(var_tr))


def _mixture(model, leaf, x_tr, var_tr, points=4096):
    """The exact mixture's mean and variance, `[n, size]`: the posterior at
    known points, over scrambled Sobol points of each input's Gaussian."""
    nodes = norm.ppf(qmc.Sobol(2, scramble=True, seed=0).random(points))
    draws = x_tr[:, None, :] + np.sqrt(var_tr)[:, None, :] * nodes[None]
    with model._propagation():
        model._refresh(model.options.jitter)
        mu, v = leaf.interpolate(tf.constant(draws.reshape(-1, 2)), None)
    mu = np.asarray(mu)[:, :, 0].reshape(2, len(x_tr), points)
    v = np.asarray(v).reshape(2, len(x_tr), points)
    return mu.mean(-1).T, (v.mean(-1) + mu.var(-1)).T


def test_the_second_moment_is_girard_s_for_the_gaussian_kernel():
    model, leaf = _uncertain_model()
    for level in (0.05, 0.5, 2.0):
        x, var = _queries(leaf, level)
        mean, variance, u, s = _node_moments(model, leaf, x, var)
        # Girard's closed form, written out: geoML's Gaussian kernel is
        # exp(-3 d² / w²), a Gaussian of squared width w² / 6
        z = np.asarray(leaf.parent.inducing_points[0])
        w = np.asarray(leaf.parameters["ranges"].get_value()).ravel()
        s2 = w ** 2 / 6.0
        c1 = np.prod((1 + s / s2) ** -0.5, axis=1)
        ell = c1[:, None] * np.exp(-((u[:, None] - z[None]) ** 2
                                     / (2 * (s2 + s)[:, None])).sum(-1))
        c2 = np.prod((1 + 2 * s / s2) ** -0.5, axis=1)
        mid = 0.5 * (z[:, None] + z[None])
        pair = np.exp(-((z[:, None] - z[None]) ** 2 / (4 * s2)).sum(-1))
        bias = float(leaf.parameters["bias_0"].get_value())
        for out in range(2):
            alpha = np.asarray(leaf.alpha[0])[out, :, 0]
            inv = np.asarray(leaf.cov_smooth_inv[0])[out]
            expected = []
            for i in range(len(u)):
                big = c2[i] * pair * np.exp(-((u[i] - mid) ** 2
                                              / (s2 + 2 * s[i])).sum(-1))
                expected.append(1.0 - np.sum(inv * big) + alpha @ big @ alpha
                                - (ell[i] @ alpha) ** 2)
            np.testing.assert_allclose(mean[:, out], ell @ alpha + bias,
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(variance[:, out], expected,
                                       rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("node", list(NODES))
def test_the_moments_are_the_mixture_s(node, name):
    # closed form for the Gaussian kernel (measured below 0.1%); 64 points
    # of quadrature for the others, on a field rougher than a trained one
    # (2-6% measured; Walker Lake within 1.8%, `second_moment.py walker`)
    tolerance = 0.02 if name == "Gaussian" else 0.08
    model, leaf = _uncertain_model(KERNELS[name](), node)
    for level in (0.05, 0.3, 1.0):
        x, var = _queries(leaf, level)
        mean, variance, u, s = _node_moments(model, leaf, x, var)
        exact_mean, exact_var = _mixture(model, leaf, u, s)
        for got, want in ((mean, exact_mean), (variance, exact_var)):
            error = np.mean(np.abs(got - want)) / np.mean(np.abs(want))
            assert error < tolerance, (level, error)


def test_no_input_variance_takes_no_second_moment():
    # zero variance: the mixture is one point, and the moments are the
    # kernel's; no variance at all: the plain kernel, as always
    model, leaf = _uncertain_model()
    x, var = _queries(leaf, 0.0)
    with_zeros = _node_moments(model, leaf, x, var)
    with model._propagation():
        model._refresh(model.options.jitter)
        mean, variance = leaf.propagate(tf.constant(x), None)
    np.testing.assert_allclose(with_zeros[0], mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(with_zeros[1], variance, rtol=1e-10,
                               atol=1e-12)


@pytest.mark.parametrize("name", ["Exponential", "Matern32",
                                  "RationalQuadratic"])
@pytest.mark.parametrize("node", list(NODES))
@pytest.mark.parametrize("offset", [0.0, 2.5e4])
def test_the_plain_covariance_is_the_covariance(node, name, offset):
    # what the quadrature reads its kernel through: the node's covariance
    # between certain points, the distance by the expansion -- about a
    # common origin, so a mine grid's coordinates (here 1e6 m at a range of
    # 40 m) keep the digits a short distance needs
    _, leaf = _uncertain_model(KERNELS[name](), node)
    rng = np.random.default_rng(1)
    x = tf.constant(offset + rng.uniform(0, 3, [50, 2]))
    y = tf.constant(offset + rng.uniform(0, 3, [20, 2]))
    np.testing.assert_allclose(np.asarray(leaf._plain_covariance(x, y)),
                               np.asarray(leaf.covariance_matrix(x, y)),
                               rtol=1e-9, atol=1e-11)


def test_only_the_gaussian_kernel_takes_it_in_closed_form():
    # the scale mixtures by quadrature over the input: in closed form a
    # table pairs into 36 arrays of [n, m, m]
    for name, closed in (("Gaussian", True), ("Matern32", False),
                         ("RationalQuadratic", False)):
        model, leaf = _uncertain_model(KERNELS[name]())
        x, var = _queries(leaf, 1.0)
        with model._propagation():
            model._refresh(model.options.jitter)
            parent = leaf.parent.propagate(tf.constant(x), tf.constant(var))
            assert leaf._takes_second_moment(parent.experts, [None])
            assert leaf._closed_second_moment() == closed


def _uncertain_targets(n=40, seed=12, spread=400.0):
    rng = np.random.default_rng(seed)
    return _uncertain_data(rng.uniform(0, 100, [n, 2]),
                           rng.uniform(0, spread, [n, 2]))


def test_the_slots_take_the_second_moment_as_the_sets_do():
    model, _ = _uncertain_model(n_experts=3)
    a, b = _uncertain_targets(), _uncertain_targets()
    model.predict(a, n_sim=4)
    info = model.predict_by_expert(b, n_sim=4, coverage=1.0)
    assert info["slots"] == 3
    for path in ("v/latent_mean", "v/latent_variance", "w/latent_variance"):
        np.testing.assert_allclose(a.values(path), b.values(path),
                                   rtol=1e-9, atol=1e-12)


def test_a_model_trains_on_uncertain_locations():
    # some locations exact, the rest uncertain
    var = np.random.default_rng(3).choice([0.0, 25.0, 400.0], [80, 2])
    model, _ = _uncertain_model(KERNELS["Matern32"](), n_experts=2, var=var)
    model.train_full(10)
    assert np.all(np.isfinite(model.training_log))
    targets = _uncertain_targets()
    model.predict(targets, n_sim=3)
    assert np.all(np.isfinite(targets.values("v/latent_variance")))


# --------------------------------------------------------------------------- #
# the input's spread, integrated like noise
# --------------------------------------------------------------------------- #
def _warped(model):
    """The model's likelihoods given a warping that bends: a sinh-arcsinh
    set by hand, heavy on one side."""
    for lik in model.likelihoods:
        lik.warping.parameters["skewness"].set_value(np.array([0.8]))
        lik.warping.parameters["tailweight"].set_value(np.array([0.6]))
    return model


def _warped_model(kernel=None):
    model, leaf = _uncertain_model(
        kernel, likelihood=lambda: geoml.likelihood.Gaussian(
            geoml.warping.SinhArcsinh(1)))
    return _warped(model), leaf


def _realizations(model, leaf, x, var, n_real=4000, seed=21):
    """Realizations at the uncertain locations, `[n, size, n_real]`: the
    expected kernel's, `l (alpha + R eta_s) + b`, and at drawn inputs,
    `k(x_s) (alpha + R eta_s) + b` with the same `eta_s` -- the exact
    mixture's -- with the node's jitter `[n, size]` and, from the same
    draws of the input, `tr((R Rᵀ + alpha alphaᵀ)(L - l lᵀ))` by
    quadrature."""
    rng = np.random.default_rng(seed)
    with model._propagation():
        model._refresh(model.options.jitter)
        predicted = leaf.predict(tf.constant(x), x_var=tf.constant(var),
                                 n_sim=1)
        u, s = (np.asarray(t) for t in leaf.root.propagate(
            tf.constant(x), tf.constant(var)))
        z = leaf.parent.inducing_points[0]
        alpha = np.asarray(leaf.alpha[0])[:, :, 0]
        root_r = np.asarray(leaf.chol_r[0])
        bias = float(leaf.parameters["bias_0"].get_value())
        l = np.asarray(leaf.expected_covariance(
            tf.constant(u), tf.constant(s), z, tf.zeros_like(z)))
        nodes = norm.ppf(qmc.Sobol(2, scramble=True, seed=seed)
                         .random(n_real))
        draws = u[:, None, :] + np.sqrt(s)[:, None, :] * nodes[None]
        k = np.asarray(leaf.covariance_matrix(
            tf.constant(draws.reshape(-1, 2)), z)).reshape(len(u), n_real, -1)
    eta = rng.normal(size=[alpha.shape[0], alpha.shape[1], n_real])
    coef = alpha[:, :, None] + np.einsum("smj,sjr->smr", root_r, eta)
    ek = np.einsum("nm,smr->nsr", l, coef) + bias
    exact = np.einsum("nrm,smr->nsr", k, coef) + bias
    # the jitter by quadrature over the same input draws
    mean_k = k.mean(1)
    cov_k = np.einsum("nri,nrj->nij", k, k) / n_real \
        - mean_k[:, :, None] * mean_k[:, None, :]
    weights = np.einsum("smj,slj->sml", root_r, root_r) \
        + alpha[:, :, None] * alpha[:, None, :]
    reference = np.einsum("sij,nij->ns", weights, cov_k)
    return ek, exact, np.asarray(predicted.jitter).T, reference


@pytest.mark.parametrize("name", ["Gaussian", "Matern52"])
def test_the_jitter_is_what_the_realizations_leave_out(name):
    # by quadrature a variance over 64 points of the input, on a rough
    # field: 9% at the widest input measured
    tolerance = 0.02 if name == "Gaussian" else 0.12
    model, leaf = _uncertain_model(KERNELS[name]())
    for level in (0.05, 0.3, 1.0):
        x, var = _queries(leaf, level)
        _, _, jitter, reference = _realizations(model, leaf, x, var)
        assert np.all(jitter >= 0.0)
        error = np.mean(np.abs(jitter - reference)) / np.mean(reference)
        assert error < tolerance, (level, error)


def _quantiles(samples):
    return np.quantile(samples, [0.05, 0.95], axis=-1)


def test_the_prediction_and_its_quantiles_are_the_mixture_s():
    # through a warping that bends, against realizations at drawn inputs:
    # the value integrated over the jitter, and a measurement drawing it
    model, leaf = _warped_model()
    lik = model.likelihoods[0]
    x, var = _queries(leaf, 0.5)
    ek, exact, jitter, _ = _realizations(model, leaf, x, var)
    ek, exact, jitter = (tf.constant(t[:, :1]) for t in (ek, exact, jitter))
    rng = np.random.default_rng(4)
    shift = tf.constant(rng.uniform(size=[len(x), 1, ek.shape[2]]))
    jitter_shift = tf.constant(rng.uniform(size=[len(x), 1, ek.shape[2]]))

    def prediction(sims, j=None):
        return np.asarray(lik.integrated_backward(sims, j)[0]).mean(-1)

    def measured(sims, j=None):
        return _quantiles(np.asarray(lik.measurement_samples(
            sims, 32, shift, j, None if j is None else jitter_shift)))

    truth = prediction(exact)
    width = np.mean(np.abs(truth))
    with_jitter = np.mean(np.abs(prediction(ek, jitter) - truth)) / width
    without = np.mean(np.abs(prediction(ek) - truth)) / width
    # measured 0.011 against 0.26 without
    assert with_jitter < 0.1 * without and with_jitter < 0.02, \
        (with_jitter, without)

    # the quantiles of a measurement, against the interval's width: 0.045
    # against 0.32 without -- what is left is the mixture not being
    # Gaussian, the jitter's variance being the mixture's to 0.3%
    truth = measured(exact)
    width = np.mean(truth[1] - truth[0])
    with_jitter = np.mean(np.abs(measured(ek, jitter) - truth)) / width
    without = np.mean(np.abs(measured(ek) - truth)) / width
    assert with_jitter < 0.25 * without and with_jitter < 0.06, \
        (with_jitter, without)


def test_a_certain_input_carries_no_jitter():
    model, leaf = _uncertain_model()
    x, _ = _queries(leaf, 0.0)
    with model._propagation():
        model._refresh(model.options.jitter)
        assert leaf.predict(tf.constant(x), n_sim=2).jitter is None
        jitter = leaf.predict(tf.constant(x), x_var=tf.zeros([len(x), 2],
                                                            tf.float64),
                              n_sim=2).jitter
    assert np.max(np.abs(np.asarray(jitter))) < 1e-10


def test_the_operations_carry_the_jitter():
    model, leaf = _uncertain_model()
    x, var = _queries(leaf, 0.5)
    nodes = {
        "linear": geoml.latent.Linear(leaf, 2),
        "select": geoml.latent.SelectInput(leaf, [1]),
        "scale": geoml.latent.Scale(leaf),
        "bias": geoml.latent.Bias(leaf),
    }
    with model._propagation():
        model._refresh(model.options.jitter)
        base = np.asarray(leaf.predict(tf.constant(x), tf.constant(var),
                                       n_sim=1).jitter)
        for name, node in nodes.items():
            got = np.asarray(node.predict(tf.constant(x), tf.constant(var),
                                          n_sim=1).jitter)
            if name == "linear":
                w = np.asarray(node.parameters["weights"].get_value())
                want = (w ** 2).T @ base
            elif name == "select":
                want = base[[1]]
            elif name == "scale":
                want = base * np.asarray(
                    node.parameters["scale"].get_value())[:, None]
            else:
                want = base
            np.testing.assert_allclose(got, want, rtol=1e-12, err_msg=name)


def test_a_prediction_at_uncertain_locations_integrates_the_jitter():
    model, _ = _warped_model()
    exact, uncertain = _uncertain_targets(spread=0.0), _uncertain_targets()
    model.predict(exact, n_sim=8)
    model.predict(uncertain, n_sim=8)
    # the spread of a measurement takes the jitter in
    assert np.all(uncertain.values("v/noise_variance")
                  >= exact.values("v/noise_variance") - 1e-9)
    assert np.mean(uncertain.values("v/noise_variance")) \
        > 1.5 * np.mean(exact.values("v/noise_variance"))
    # one location's answer does not depend on its batch
    again = _uncertain_targets()
    model.options.prediction_batch_size = 7
    model.predict(again, n_sim=8)
    for path in ("v/prediction", "v/noise_variance"):
        np.testing.assert_allclose(uncertain.values(path), again.values(path),
                                   rtol=1e-9, atol=1e-12)
    samples = model.predict_measurements(_uncertain_targets(), n_sim=4)
    assert np.all(np.isfinite(samples["v"]))
