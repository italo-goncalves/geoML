"""`latent.UncertainInputGP`: the mixture over an uncertain input.

`BasicGP` reads an input variance through Paciorek's inflated covariance and
takes the moments at one point under it. Measured on Walker Lake against the
exact mixture over the input's Gaussian (2026-09-03), that understates the
predictive variance four- to tenfold and misplaces the mean by tens of
percent from a tenth of the squared range upward, for every kernel.
`UncertainInputGP` computes the mixture by Sobol quadrature over the input
and draws each realization at a node of its own.

Pinned here: with no variance the node is `BasicGP`; the moments against
Girard's closed form for the Gaussian kernel and against Monte Carlo for a
Matérn; the simulations carrying the mixture; training, batch invariance
and reload on a `GaussianData`; the node as a child of another GP; the
diagram.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.persistence as persistence
import geoml.transform as tr


def _points(n, n_dim, seed, low=0.0, high=100.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, (n, n_dim))


def _training_data(n=60, n_dim=2, seed=1234):
    coords = _points(n, n_dim, seed)
    labels = ["c%d" % i for i in range(n_dim)]
    point = geoml.data.PointData.from_array(coords, labels)
    point.add_continuous_variable(
        "v", np.sin(coords[:, 0] / 20.0) + 0.5 * np.cos(coords[:, 1] / 30.0))
    inducing = geoml.data.PointData.from_array(_points(25, n_dim, seed + 1),
                                               labels)
    return point, inducing, labels


def _model(container, root, node=geoml.latent.UncertainInputGP, seed=1234,
           max_iter=3, kernel=None, **node_kwargs):
    geoml.set_seed(seed)
    gp = node(root, size=1, kernel=kernel or geoml.kernels.Gaussian(),
              **node_kwargs)
    options = geoml.models.GPOptions(verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        container, "v", geoml.likelihood.Gaussian(), gp, options=options)
    if max_iter:
        model.train_full(max_iter=max_iter)
    model.latent_network.refresh()          # eager state to probe
    return model, gp


def _gaussian_query(labels, n=30, scale=1.0, seed=6):
    coords = _points(n, 2, seed)
    return geoml.data.GaussianData.from_array(
        coords, np.full_like(coords, scale), labels)


def _girard(gp, u, var):
    """Girard's exact mixture moments for geoML's Gaussian kernel
    `exp(-3 r^2 / w^2)`, a Gaussian of squared width `w^2 / 6`."""
    z = gp.parent.inducing_points[0].numpy()
    alpha = gp.alpha[0].numpy()[0, :, 0]
    A = gp.cov_smooth_inv[0].numpy()[0]
    bias = float(np.asarray(gp.parameters["bias_0"].get_value()).ravel()[0])
    w = np.broadcast_to(np.asarray(gp.parameters["ranges"].get_value()).ravel(),
                        (z.shape[1],))
    s2 = w ** 2 / 6.0
    c1 = np.prod((1 + var / s2) ** -0.5, axis=1)
    q1 = ((u[:, None, :] - z[None, :, :]) ** 2
          / (2 * (s2 + var)[:, None, :])).sum(-1)
    ell = c1[:, None] * np.exp(-q1)
    mean = ell @ alpha + bias
    c2 = np.prod((1 + 2 * var / s2) ** -0.5, axis=1)
    zbar = 0.5 * (z[:, None, :] + z[None, :, :])
    pair = np.exp(-((z[:, None, :] - z[None, :, :]) ** 2 / (4 * s2)).sum(-1))
    variance = np.empty(u.shape[0])
    for i in range(u.shape[0]):
        qi = ((u[i, None, None, :] - zbar) ** 2
              / (s2 + 2 * var[i])[None, None, :]).sum(-1)
        L = c2[i] * pair * np.exp(-qi)
        variance[i] = 1.0 - np.sum(A * L) + alpha @ L @ alpha \
            - (ell[i] @ alpha) ** 2
    return mean, variance


def _transformed(model, container):
    root = model.latent_network.root
    x, v = root.propagate(tf.constant(np.asarray(container.coordinates, float)),
                          tf.constant(container.get_batched_variance()[0]))
    return x.numpy(), v.numpy()


# --------------------------------------------------------------------------- #
def test_no_variance_is_basic_gp():
    point, inducing, labels = _training_data()
    query = geoml.data.PointData.from_array(_points(20, 2, seed=5), labels)
    results = []
    for node in (geoml.latent.BasicGP, geoml.latent.UncertainInputGP):
        model, _ = _model(point, geoml.latent.GaussianInput(
            inducing, transform=tr.Isotropic(40.0)), node=node)
        model.predict(query, n_sim=6)
        results.append((
            np.asarray(query.values("v/prediction"), dtype=float),
            np.asarray(query.values("v/latent_variance"), dtype=float),
            np.asarray(query.get("v").get_simulations(), dtype=float),
            float(model.training_log[-1])))
    for a, b in zip(results[0], results[1]):
        assert np.allclose(a, b, rtol=0.0, atol=1e-10)


def test_moments_match_the_exact_mixture_for_the_gaussian_kernel():
    point, inducing, labels = _training_data()
    model, gp = _model(point, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)), max_iter=30, n_nodes=256)
    w2 = float(np.asarray(gp.parameters["ranges"].get_value()).ravel()[0]) ** 2
    # a variance of a third of the squared range, in the transformed space
    r2 = float(np.asarray(model.latent_network.root.transform
                          .parameters["range"].get_value())) ** 2
    query = _gaussian_query(labels, scale=0.3 * w2 * r2)
    u, var = _transformed(model, query)
    mu, v = gp.interpolate(tf.constant(u), tf.constant(var))
    mean_exact, var_exact = _girard(gp, u, var)
    # the means sit near zero, so the tolerance is of their spread
    assert np.allclose(mu.numpy()[0, :, 0], mean_exact,
                       atol=0.05 * np.std(mean_exact))
    assert np.allclose(v.numpy()[0], var_exact, rtol=0.05)


def test_moments_match_monte_carlo_for_a_matern_kernel():
    point, inducing, labels = _training_data()
    model, gp = _model(point, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)), max_iter=30,
        kernel=geoml.kernels.Matern32())
    w2 = float(np.asarray(gp.parameters["ranges"].get_value()).ravel()[0]) ** 2
    r2 = float(np.asarray(model.latent_network.root.transform
                          .parameters["range"].get_value())) ** 2
    query = _gaussian_query(labels, n=20, scale=0.5 * w2 * r2)
    u, var = _transformed(model, query)
    mu, v = gp.interpolate(tf.constant(u), tf.constant(var))

    rng = np.random.default_rng(0)
    draws = u[:, None, :] + np.sqrt(var)[:, None, :] * rng.normal(
        size=(u.shape[0], 4000, 2))
    m, s = geoml.latent.BasicGP.interpolate(
        gp, tf.constant(draws.reshape(-1, 2)), None)
    m = m.numpy()[0, :, 0].reshape(u.shape[0], -1)
    s = s.numpy()[0].reshape(u.shape[0], -1)
    # the quadrature's error scales with how much the mean varies over the
    # input's spread, and the Monte Carlo reference has an error of its own
    within = m.std(1).mean()
    assert np.allclose(mu.numpy()[0, :, 0], m.mean(1),
                       atol=0.1 * within + 3 * within / np.sqrt(m.shape[1]))
    assert np.allclose(v.numpy()[0], s.mean(1) + m.var(1), rtol=0.1)


def test_simulations_carry_the_mixture():
    """Each realization sits at a node of its own, so the ensemble at an
    uncertain point is the ensemble of the deterministic model pooled over
    the node locations: same mean, same spread."""
    point, inducing, labels = _training_data()
    model, gp = _model(point, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)), max_iter=30)
    w2 = float(np.asarray(gp.parameters["ranges"].get_value()).ravel()[0]) ** 2
    r2 = float(np.asarray(model.latent_network.root.transform
                          .parameters["range"].get_value())) ** 2
    scale = 1.0 * w2 * r2
    query = _gaussian_query(labels, scale=scale)
    model.predict(query, n_sim=256)
    sims = np.asarray(query.get("v").get_simulations(), dtype=float)
    mu = np.asarray(query.values("v/prediction"), float).ravel()
    assert np.allclose(sims.mean(axis=1), mu, atol=0.1 * np.std(mu))

    # the reference: the node locations as exact points -- the transform is
    # isotropic, so a node in the transformed space is the same step in
    # the original one -- pooled per query point
    nodes = np.asarray(gp.parameters["nodes"].get_value())
    coords = np.asarray(query.coordinates, dtype=float)
    stacked = (coords[None, :, :] + np.sqrt(scale) * nodes[:, None, :])         .reshape(-1, 2)
    reference = geoml.data.PointData.from_array(stacked, labels)
    model.predict(reference, n_sim=256)
    pooled = np.asarray(reference.get("v").get_simulations(), dtype=float)         .reshape(nodes.shape[0], coords.shape[0], -1)
    pooled_var = pooled.var(axis=2).mean(axis=0) + pooled.mean(axis=2).var(axis=0)
    # 256 draws per point estimate a variance to ~10%: the level is checked
    # on the average over points, and every point within a few of those
    own_var = sims.var(axis=1)
    assert abs(own_var.mean() / pooled_var.mean() - 1.0) < 0.1
    assert np.allclose(own_var, pooled_var, rtol=0.4)


def test_training_prediction_and_reload_on_gaussian_data(tmp_path):
    point, inducing, labels = _training_data()
    coords = np.asarray(point.coordinates, dtype=float)
    rng = np.random.default_rng(4)
    gauss = geoml.data.GaussianData.from_array(
        coords, rng.uniform(1.0, 100.0, coords.shape), labels)
    gauss.add_continuous_variable("v", np.asarray(point.values("v/measurements")))

    model, _ = _model(gauss, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)))
    assert np.isfinite(model.training_log).all()
    model.train_svi(epochs=1)
    assert np.isfinite(model.training_log).all()

    query = _gaussian_query(labels, n=17, scale=50.0)
    model.predict(query, n_sim=5)
    whole = np.asarray(query.values("v/prediction"), dtype=float).copy()
    assert np.isfinite(whole).all()
    model.options.prediction_batch_size = 4
    model.predict(query, n_sim=5)
    assert np.allclose(whole, np.asarray(query.values("v/prediction")))

    persistence.save_model(model, tmp_path / "model")
    restored = persistence.load_model(tmp_path / "model")
    gp = restored.latent_network
    assert isinstance(gp, geoml.latent.UncertainInputGP) and gp.n_nodes == 32
    restored.predict(query, n_sim=5)
    assert np.allclose(whole, np.asarray(query.values("v/prediction")))


def test_as_a_child_of_another_gp():
    """A deep use: the parent's posterior variance is what is integrated."""
    point, inducing, labels = _training_data()
    geoml.set_seed(1234)
    root = geoml.latent.BasicInput(inducing, transform=tr.Isotropic(40.0))
    inner = geoml.latent.BasicGP(root, size=2, kernel=geoml.kernels.Gaussian())
    outer = geoml.latent.UncertainInputGP(inner, size=1, n_nodes=8)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), outer,
        options=geoml.models.GPOptions(verbose=False, training_samples=8))
    model.train_full(max_iter=3)
    assert np.isfinite(model.training_log).all()
    query = geoml.data.PointData.from_array(_points(12, 2, seed=5), labels)
    model.predict(query, n_sim=4)
    assert np.isfinite(np.asarray(query.values("v/prediction"), float)).all()
    assert np.isfinite(np.asarray(query.get("v").get_simulations(), float)).all()


def test_the_diagram_names_the_node():
    point, inducing, _ = _training_data()
    model, _ = _model(point, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)), max_iter=0)
    assert "UncertainInputGP" in model.to_dot()
