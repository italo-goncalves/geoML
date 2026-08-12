"""The mixture likelihood: two noise mechanisms, one latent field.

Covers the constructor's guarantees, the split the contamination flag makes
between training (full mixture) and the ground (genuine components only),
the bisected mixture quantile behind `measurement_samples`, the
responsibilities diagnostic, end-to-end training on contaminated data, and
the persistence round trip.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from scipy import optimize, special, stats

import geoml


def _mixture(w=(0.9, 0.1), s1=0.5, s2=3.0):
    """A two-Gaussian mixture with hand-set parameters and an identity warp."""
    mix = geoml.likelihood.Mixture(
        [geoml.likelihood.Gaussian(), geoml.likelihood.Gaussian()],
        geoml.warping.ZScore(1))
    mix.parameters["weights"].set_value(np.array(w))
    mix.components[0].parameters["noise"].set_value(
        np.full([1, 1, 1], s1 ** 2))
    mix.components[1].parameters["noise"].set_value(
        np.full([1, 1, 1], s2 ** 2))
    # ZScore starts at mean 0, sd 1: warped space and value space coincide
    return mix


def _contaminated(seed=1234, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    f = np.sin(2.2 * x) + 0.6 * np.sin(0.7 * x + 1.0)
    eps = rng.normal(0.0, 0.3, n)
    is_out = rng.uniform(size=n) < 0.05
    eps[is_out] = rng.normal(0.0, 3.0, is_out.sum())
    return x, f + eps, is_out


def _model(likelihood, x, y, seed=1234, max_iter=200):
    geoml.set_seed(seed)
    tf.random.set_seed(seed)
    point = geoml.data.PointData(pd.DataFrame({"c0": x}), ["c0"])
    point.add_continuous_variable("v", y)
    inducing = geoml.data.Grid1D(start=0.0, n=40, step=10.0 / 39)
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(1.5))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "v", likelihood, gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=max_iter)
    return model, point


# --------------------------------------------------------------------------- #
# constructor guarantees
# --------------------------------------------------------------------------- #
def test_a_mixture_needs_two_components():
    with pytest.raises(ValueError, match="two components"):
        geoml.likelihood.Mixture([geoml.likelihood.Gaussian()],
                                 geoml.warping.ZScore(1))


def test_something_must_describe_the_ground():
    with pytest.raises(ValueError, match="ground"):
        geoml.likelihood.Mixture(
            [geoml.likelihood.Gaussian(), geoml.likelihood.Gaussian()],
            geoml.warping.ZScore(1), contamination=[True, True])


def test_scalar_components_serve_a_wider_mixture():
    """The mixture's size is the warping's, and a component's parameters
    broadcast over the columns: scalar components need not be rebuilt as
    sized twins to serve a multivariate mixture."""
    mix = geoml.likelihood.Mixture(
        [geoml.likelihood.Gaussian(), geoml.likelihood.Gaussian()],
        geoml.warping.ZScore(2))
    mix.parameters["weights"].set_value(np.array([0.9, 0.1]))
    mix.components[0].parameters["noise"].set_value(np.full([1, 1, 1], 0.25))
    mix.components[1].parameters["noise"].set_value(np.full([1, 1, 1], 9.0))

    sims = tf.zeros([5, 2, 3], tf.float64)
    value, noise_var = mix.integrated_backward(sims)
    assert value.shape == (5, 2, 3)
    assert np.allclose(value.numpy(), 0.0, atol=1e-9)
    # the shared inlier noise reaches both columns
    assert np.allclose(noise_var.numpy(), 0.25, rtol=1e-6)


def test_component_warpings_are_inert():
    """The mixture's warping is the one applied; a component's own must not
    leave trainable parameters drifting in the ELBO."""
    mix = _mixture()
    for component in mix.components:
        for parameter in component.warping._all_parameters:
            assert parameter.fixed
    # the mixture's own warping still trains
    assert any(not p.fixed for p in mix.warping._all_parameters)


def test_components_may_differ_in_kind():
    mix = geoml.likelihood.Mixture(
        [geoml.likelihood.Gaussian(), geoml.likelihood.StudentT()],
        geoml.warping.ZScore(1))
    assert mix.size == 1


def test_a_robust_zscore_does_not_let_contamination_set_the_scale():
    """`ZScore(size, robust=True)` initializes on a winsorized copy: a
    handful of gross outliers must not inflate the scale everything is
    normalized by, where the plain fit keeps the moments as they come. On
    clean data the two fits agree."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, 300)
    clean = y.copy()
    y[:3] = 500.0

    def fitted_std(warping, values):
        warping.initialize(values[:, None])
        return float(warping.parameters["std"].get_value().numpy()[0])

    assert fitted_std(geoml.warping.ZScore(1, robust=True), y) < 2.0
    assert fitted_std(geoml.warping.ZScore(1), y) > 20.0
    assert abs(fitted_std(geoml.warping.ZScore(1, robust=True), clean)
               - fitted_std(geoml.warping.ZScore(1), clean)) < 0.05


# --------------------------------------------------------------------------- #
# the ground excludes the contamination; a measurement includes it
# --------------------------------------------------------------------------- #
def test_the_ground_is_averaged_over_the_genuine_component_alone():
    """`integrated_backward` on a zero field: the value is the noise mean
    (zero) and `noise_variance` the *inlier* variance, the wide component
    having no say in what the ground holds."""
    mix = _mixture(w=(0.9, 0.1), s1=0.5, s2=3.0)
    sims = tf.zeros([7, 1, 4], tf.float64)
    value, noise_var = mix.integrated_backward(sims)

    assert np.allclose(value.numpy(), 0.0, atol=1e-9)
    # Gauss-Hermite is exact for a quadratic: the inlier variance, exactly
    assert np.allclose(noise_var.numpy(), 0.25, rtol=1e-6)


def test_a_measurement_keeps_the_full_mixture():
    """`measurement_samples` describes a fresh assay, which can be a bad
    one: its spread is the whole mixture's, not the inlier's."""
    mix = _mixture(w=(0.9, 0.1), s1=0.5, s2=3.0)
    sims = tf.zeros([5, 1, 1], tf.float64)
    samples = mix.measurement_samples(sims, n_nodes=400).numpy()

    total = 0.9 * 0.25 + 0.1 * 9.0
    spread = samples.var()
    # equal-share nodes truncate the extreme tails, so the spread sits a
    # little under the exact mixture variance -- but far above the inlier's
    assert spread > 0.6 * total
    assert abs(spread - total) / total < 0.25
    assert spread > 4 * 0.25


def test_measurement_nodes_follow_the_mixture_quantile():
    """The bisection must land on the true quantile of the mixture CDF."""
    w, s1, s2 = (0.9, 0.1), 0.5, 3.0
    mix = _mixture(w=w, s1=s1, s2=s2)
    nodes = mix._measurement_values(10).numpy()[:, 0, 0]

    u = (np.arange(10) + 0.5) / 10
    for node, p in zip(nodes, u):
        exact = optimize.brentq(
            lambda t: w[0] * stats.norm.cdf(t, scale=s1)
            + w[1] * stats.norm.cdf(t, scale=s2) - p, -40, 40)
        assert abs(node - exact) < 1e-6


def test_contamination_flags_choose_what_the_ground_keeps():
    """Marking nothing as contamination turns the same parameters into a
    plain (heavy-tailed) noise model: the full variance comes back."""
    mix = geoml.likelihood.Mixture(
        [geoml.likelihood.Gaussian(), geoml.likelihood.Gaussian()],
        geoml.warping.ZScore(1), contamination=[False, False])
    mix.parameters["weights"].set_value(np.array([0.9, 0.1]))
    mix.components[0].parameters["noise"].set_value(np.full([1, 1, 1], 0.25))
    mix.components[1].parameters["noise"].set_value(np.full([1, 1, 1], 9.0))

    sims = tf.zeros([3, 1, 2], tf.float64)
    _, noise_var = mix.integrated_backward(sims)
    assert np.allclose(noise_var.numpy(), 0.9 * 0.25 + 0.1 * 9.0, rtol=1e-6)


# --------------------------------------------------------------------------- #
# end to end on contaminated data
# --------------------------------------------------------------------------- #
def test_training_finds_the_contamination():
    x, y, is_out = _contaminated()
    mix = geoml.likelihood.Mixture(
        [geoml.likelihood.Gaussian(), geoml.likelihood.Gaussian()],
        geoml.warping.ZScore(1))
    # what the docstring asks for: the contamination component starts wide,
    # so it is the cheaper explanation for a tail value from the first step
    mix.components[1].parameters["noise"].set_value(np.full([1, 1, 1], 9.0))
    model, point = _model(mix, x, y)

    weight = float(mix.parameters["weights"].get_value()[1])
    assert 0.01 < weight < 0.2

    model.predict(point, n_sim=4)
    resp = mix.responsibilities(
        np.asarray(point.variables["v"].latent_mean.values),
        np.asarray(point.variables["v"].latent_variance.values),
        y)
    assert resp.shape == (len(y), 2)
    assert np.allclose(resp.sum(axis=1), 1.0)
    # the flagged points are overwhelmingly the planted ones
    assert resp[is_out, 1].mean() > 5 * resp[~is_out, 1].mean()


def test_prediction_runs_end_to_end():
    x, y, _ = _contaminated()
    mix = geoml.likelihood.Mixture(
        [geoml.likelihood.Gaussian(), geoml.likelihood.Gaussian()],
        geoml.warping.ZScore(1))
    model, _ = _model(mix, x, y, max_iter=50)

    grid = geoml.data.Grid1D(start=0.0, n=50, step=10.0 / 49)
    model.predict(grid, n_sim=8)
    for column in ("prediction", "latent_mean", "noise_variance",
                   "dispersion"):
        values = grid.values("v/%s" % column)
        assert values.shape == (50,)
    assert np.all(np.isfinite(grid.values("v/prediction")))


def test_a_saved_mixture_model_reopens(tmp_path):
    x, y, _ = _contaminated()
    mix = geoml.likelihood.Mixture(
        [geoml.likelihood.Gaussian(), geoml.likelihood.StudentT()],
        geoml.warping.ZScore(1))
    model, _ = _model(mix, x, y, max_iter=30)

    grid = geoml.data.Grid1D(start=0.0, n=30, step=10.0 / 29)
    model.predict(grid, n_sim=4)
    before = np.array(grid.values("v/prediction"), copy=True)

    path = str(tmp_path / "mixture_model")
    model.save(path)
    reopened = geoml.models.VGPNetwork.open(path)

    lik = reopened.likelihoods[0]
    assert isinstance(lik, geoml.likelihood.Mixture)
    assert lik.contamination == [False, True]
    assert np.allclose(lik.parameters["weights"].get_value().numpy(),
                       mix.parameters["weights"].get_value().numpy())

    grid2 = geoml.data.Grid1D(start=0.0, n=30, step=10.0 / 29)
    reopened.predict(grid2, n_sim=4)
    assert np.allclose(np.array(grid2.values("v/prediction")), before,
                       atol=1e-8)
