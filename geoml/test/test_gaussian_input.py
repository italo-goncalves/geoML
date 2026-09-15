"""`latent.GaussianInput`: an input variance that reaches the network.

`BasicInput` hands the network the coordinates alone -- deterministic by
design -- and the variance a `GaussianData` container carries was reaching
the root only to be zeroed there. `GaussianInput` maps it through the
transform (exactly when the transform is affine, to first order otherwise)
and the expected kernel the GP nodes already carry integrates over it.

What is pinned here: the affine declarations against a numerical Jacobian;
the node with no variance being `BasicInput` to the last bit; the mapping
through the affine family by hand and through a nonlinear transform against
Monte Carlo; the fault transforms surviving forward-mode differentiation
through their Newton steps; the prediction returning to the prior as the
input variance grows; training and prediction on a `GaussianData`,
batch-invariant; the
high-dimensional missing-entry use it is built for; save/load; the diagram.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import geoml
import geoml.persistence as persistence
import geoml.transform as tr

from test_faults import _plane


def _points(n, n_dim, seed, low=0.0, high=100.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, (n, n_dim))


def _model(container, root, seed=1234, max_iter=3):
    geoml.set_seed(seed)
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        container, "v", geoml.likelihood.Gaussian(), gp, options=options)
    if max_iter:
        model.train_full(max_iter=max_iter)
    return model


def _training_data(n=40, n_dim=2, seed=1234):
    coords = _points(n, n_dim, seed)
    labels = ["c%d" % i for i in range(n_dim)]
    point = geoml.data.PointData.from_array(coords, labels)
    point.add_continuous_variable("v", np.sin(coords[:, 0] / 25.0))
    inducing = geoml.data.PointData.from_array(_points(6, n_dim, seed + 1),
                                               labels)
    return point, inducing, labels


def _jacobians(transform, x):
    x = tf.constant(x, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = transform(x)
    return tape.batch_jacobian(y, x).numpy()


# --------------------------------------------------------------------------- #
# the declarations
# --------------------------------------------------------------------------- #
_BOX = geoml.data.BoundingBox(np.zeros(3), np.full(3, 100.0))


@pytest.mark.parametrize("transform, n_dim", [
    (tr.Identity(), 3),
    (tr.Isotropic(40.0), 3),
    (tr.Anisotropy2D(maxrange=40.0, minrange_fct=0.5, azimuth=30.0), 2),
    (tr.Anisotropy3D(maxrange=40.0, midrange_fct=0.7, minrange_fct=0.4,
                     azimuth=30.0, dip=20.0, rake=10.0), 3),
    (tr.ProjectionTo1D(3), 3),
    (tr.AnisotropyARD(3), 3),
    (tr.SelectVariables([0, 2]), 3),
    (tr.NormalizeWithBoundingBox(_BOX), 3),
    (tr.RandomProjections(3, 4), 3),
    (tr.Concatenate(tr.Isotropic(40.0), tr.AnisotropyARD(3)), 3),
    (tr.ChainedTransform(tr.Isotropic(40.0), tr.SelectVariables([1])), 3),
    (tr.Periodic(), 3),
    (tr.ChainedTransform(tr.Isotropic(40.0), tr.Periodic()), 3),
])
def test_declared_linearity_matches_a_numerical_jacobian(transform, n_dim):
    """A transform says whether it is affine; the Jacobian at two random
    points agrees exactly when it is."""
    jac = _jacobians(transform, _points(2, n_dim, seed=3))
    constant = np.allclose(jac[0], jac[1], atol=1e-10)
    assert constant == transform.linear


def test_the_faults_are_declared_nonlinear():
    points, normals = _plane(axis=0, position=50.0)
    fault = tr.FaultDisplacement(points, normals, throw=20.0, reach=1000.0,
                                 width=0.5)
    assert not fault.linear
    assert not tr.ImplicitFault(points, normals).linear
    assert not tr.ChainedTransform(tr.Isotropic(40.0), fault).linear


# --------------------------------------------------------------------------- #
# the node
# --------------------------------------------------------------------------- #
def test_no_variance_is_basic_input_to_the_last_bit():
    point, inducing, _ = _training_data()
    query = geoml.data.PointData.from_array(_points(20, 2, seed=5), ["c0", "c1"])

    results = []
    for node in (geoml.latent.BasicInput, geoml.latent.GaussianInput):
        model = _model(point, node(inducing, transform=tr.Isotropic(40.0)))
        model.predict(query, n_sim=4)
        results.append((
            np.asarray(query.values("v/prediction"), dtype=float),
            np.asarray(query.values("v/latent_variance"), dtype=float),
            np.asarray(query.get("v").get_simulations(), dtype=float),
            float(model.training_log[-1])))
    for a, b in zip(results[0], results[1]):
        assert np.allclose(a, b, rtol=0.0, atol=1e-12)


def test_variance_maps_exactly_through_the_affine_family():
    """`Isotropic(r)` divides by `r^2`; an ellipsoid maps as `var @ A^2`
    with `A` the matrix the transform multiplies by."""
    _, inducing, _ = _training_data()
    x = tf.constant(_points(5, 2, seed=8), tf.float64)
    var = tf.constant(_points(5, 2, seed=9, low=1.0, high=9.0), tf.float64)

    root = geoml.latent.GaussianInput(inducing, transform=tr.Isotropic(40.0))
    root.refresh()
    _, var_tr = root.propagate(x, var)
    # `set_limits` may have moved the range off the argument at construction
    r = float(np.asarray(root.transform.parameters["range"].get_value()))
    assert np.allclose(var_tr.numpy(), var.numpy() / r ** 2, atol=1e-12)

    ellipsoid = tr.Anisotropy2D(maxrange=40.0, minrange_fct=0.5, azimuth=30.0)
    root = geoml.latent.GaussianInput(inducing, transform=ellipsoid)
    root.refresh()
    _, var_tr = root.propagate(x, var)
    matrix = ellipsoid._anis_inv.numpy()
    assert np.allclose(var_tr.numpy(), var.numpy() @ matrix ** 2, atol=1e-12)

    # and no variance at all is no variance out
    _, zero = root.propagate(x, None)
    assert not np.any(zero.numpy())


def test_first_order_through_a_nonlinear_transform_against_monte_carlo():
    _, inducing, _ = _training_data()
    transform = tr.ChainedTransform(tr.Isotropic(40.0), tr.Periodic())
    root = geoml.latent.GaussianInput(inducing, transform=transform)
    root.refresh()
    x = _points(5, 2, seed=8)
    var = np.full_like(x, 1.0)
    _, var_tr = root.propagate(tf.constant(x), tf.constant(var))

    rng = np.random.default_rng(0)
    draws = x[:, None, :] + np.sqrt(var)[:, None, :] * rng.normal(
        size=(5, 200000, 2))
    through = transform(tf.constant(draws.reshape(-1, 2))).numpy() \
        .reshape(5, 200000, -1)
    # first order: where the derivative vanishes only the second order is
    # left, hence the absolute floor beside the relative tolerance
    assert np.allclose(var_tr.numpy(), through.var(axis=1), rtol=0.05,
                       atol=2e-5)


def test_forward_mode_survives_a_fault_transform():
    """`FaultDisplacement` runs Newton steps inside `__call__`; the
    per-point Jacobian must come through them. Away from the fault the
    displacement is a rigid shift, so the variance passes unchanged."""
    points, normals = _plane(axis=0, position=50.0)
    fault = tr.FaultDisplacement(points, normals, throw=20.0, reach=1000.0,
                                 width=0.5)
    inducing = geoml.data.PointData.from_array(_points(6, 3, seed=1))
    root = geoml.latent.GaussianInput(inducing, transform=fault)
    root.refresh()
    x = _points(8, 3, seed=2, low=10.0, high=90.0)
    x[:, 0] = np.where(x[:, 0] < 50.0, 20.0, 80.0)      # 30 away from it
    var = _points(8, 3, seed=3, low=1.0, high=4.0)
    _, var_tr = root.propagate(tf.constant(x), tf.constant(var))
    assert np.all(np.isfinite(var_tr.numpy()))
    assert np.allclose(var_tr.numpy(), var, rtol=1e-2)


def test_a_wide_input_variance_returns_the_prediction_to_the_prior():
    """The expected kernel's normalization shrinks the cross-covariance as
    the input variance grows, so a location known only to within a wide
    error is predicted as the prior: the mean at the bias, the latent
    variance at one. Measured 2026-09-03 and worth knowing: up to a
    variance of the range's own order the moment path *narrows* (mean
    latent variance 0.515 -> 0.468 at `var = range^2` on this fixture)
    rather than widening -- the expected kernel smooths the cross-covariance
    faster than its normalization shrinks it, and the path carries no term
    for the variance of the mean over the input (Girard's second-moment
    correction would; see the board) -- so what is pinned is the wide
    limit and the direction from zero to a hundred ranges of doubt."""
    point, inducing, labels = _training_data()
    model = _model(point, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)))
    coords = np.asarray(point.coordinates, dtype=float)
    r = float(np.asarray(model.latent_network.root.transform
                         .parameters["range"].get_value()))

    spreads, means = [], []
    for scale in (0.0, 100.0 * r ** 2, 1e12):
        gauss = geoml.data.GaussianData.from_array(
            coords, np.full_like(coords, scale), labels)
        model.predict(gauss, n_sim=2)
        spreads.append(np.asarray(gauss.values("v/latent_variance"),
                                  dtype=float).ravel())
        means.append(np.asarray(gauss.values("v/prediction"),
                                dtype=float).ravel())
    assert spreads[0].mean() < 0.9                    # the data are seen
    assert spreads[1].mean() > spreads[0].mean()      # on the way to the prior
    assert np.allclose(spreads[2], 1.0, atol=1e-6)    # the prior
    assert np.ptp(means[2]) < 1e-2 * np.ptp(means[0])  # the bias everywhere


def test_training_and_prediction_on_gaussian_data_are_batch_invariant():
    point, inducing, labels = _training_data()
    coords = np.asarray(point.coordinates, dtype=float)
    rng = np.random.default_rng(4)
    gauss = geoml.data.GaussianData.from_array(
        coords, rng.uniform(1.0, 50.0, coords.shape), labels)
    gauss.add_continuous_variable("v", np.asarray(point.values("v/measurements")))

    model = _model(gauss, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)))
    assert np.isfinite(model.training_log).all()
    model.train_svi(epochs=1)
    assert np.isfinite(model.training_log).all()

    query = geoml.data.GaussianData.from_array(
        _points(15, 2, seed=6), rng.uniform(1.0, 50.0, (15, 2)), labels)
    model.predict(query, n_sim=3)
    whole = np.asarray(query.values("v/prediction"), dtype=float).copy()
    model.options.prediction_batch_size = 4
    model.predict(query, n_sim=3)
    assert np.allclose(whole, np.asarray(query.values("v/prediction")))
    assert np.isfinite(whole).all()


def test_high_dimensional_inputs_with_missing_entries():
    """The case the node is built for: a missing entry given its column's
    mean and variance, the row kept. Rows without a missing entry predict
    exactly as through `BasicInput`; rows with one are read differently."""
    n, n_dim = 80, 12
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 100.0, (n, n_dim))
    y = np.sin(x[:, 0] / 25.0) + 0.5 * np.cos(x[:, 1] / 30.0)
    missing = rng.uniform(size=x.shape) < 0.25
    mean, var = x.mean(axis=0), x.var(axis=0)
    filled = np.where(missing, mean, x)
    variance = np.where(missing, var, 0.0)
    labels = ["x%d" % i for i in range(n_dim)]

    train = geoml.data.GaussianData.from_array(filled[:60], variance[:60], labels)
    train.add_continuous_variable("v", y[:60])
    inducing = geoml.data.PointData.from_array(filled[:12], labels)
    transform = tr.AnisotropyARD(n_dim)
    model = _model(train, geoml.latent.GaussianInput(inducing, transform=transform))
    # twelve dimensions put every point far from every other: a range wide
    # enough for the inducing points to be seen at all
    transform.parameters["ranges"].set_value(np.full(n_dim, 150.0))

    uncertain = geoml.data.GaussianData.from_array(filled[60:], variance[60:], labels)
    certain = geoml.data.PointData.from_array(filled[60:], labels)
    model.predict(uncertain, n_sim=2)
    model.predict(certain, n_sim=2)
    p_unc = np.asarray(uncertain.values("v/prediction"), dtype=float).ravel()
    p_cer = np.asarray(certain.values("v/prediction"), dtype=float).ravel()
    v_cer = np.asarray(certain.values("v/latent_variance"), dtype=float).ravel()
    holes = missing[60:].any(axis=1)
    assert np.isfinite(p_unc).all()
    assert v_cer.mean() < 0.99                       # not the prior everywhere
    assert np.allclose(p_unc[~holes], p_cer[~holes], atol=1e-12)
    assert not np.allclose(p_unc[holes], p_cer[holes], atol=1e-6)


def test_save_and_load(tmp_path):
    point, inducing, labels = _training_data()
    model = _model(point, geoml.latent.GaussianInput(
        inducing, transform=tr.Anisotropy2D(maxrange=40.0, minrange_fct=0.5)))
    coords = _points(10, 2, seed=6)
    query = geoml.data.GaussianData.from_array(
        coords, np.full_like(coords, 9.0), labels)
    model.predict(query, n_sim=3)
    before = np.asarray(query.values("v/prediction"), dtype=float).copy()

    persistence.save_model(model, tmp_path / "model")
    restored = persistence.load_model(tmp_path / "model")
    assert isinstance(restored.latent_network.root, geoml.latent.GaussianInput)
    restored.predict(query, n_sim=3)
    assert np.allclose(before, np.asarray(query.values("v/prediction")))


def test_the_diagram_names_the_node():
    point, inducing, _ = _training_data()
    model = _model(point, geoml.latent.GaussianInput(
        inducing, transform=tr.Isotropic(40.0)), max_iter=0)
    assert "GaussianInput" in model.to_dot()
