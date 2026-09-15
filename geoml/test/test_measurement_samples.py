"""The measurement samples: rotated equal-share nodes.

A measurement sample used to be built on the strata's midpoints, one fixed
set of noise values for every location. That is right for reading one
location and wrong twice over: the midpoints carry less than the noise
variance (0.96 of a Gaussian's at 32 nodes, 0.89 of a Laplace's, in warped
space; through a spline warping in data units, under half on Jura's
heavy-tailed metals), and every location in a column carried the same
value, so anything read across locations -- a variogram, a regional mean
-- saw noise that cancelled. The nodes are now rotated modulo one by a
uniform per location, component and realization, drawn from the model's
seed: the lattice keeps one point per stratum, each marginally uniform,
so every finite moment is unbiased, the tails reached, locations
independent, and a location's sample the same whatever batch computed it.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import geoml


def _reference_variance(likelihood, n=200_000):
    """The noise law's variance by a fine quantile sweep."""
    dist = likelihood._make_distribution(tf.constant(0.0, tf.float64))
    u = tf.constant((np.arange(n) + 0.5) / n, tf.float64)
    q = np.asarray(dist.quantile(u[:, None, None]), dtype=float).ravel()
    return float(np.mean(q ** 2) - np.mean(q) ** 2)


def _pooled_variance(likelihood, n_nodes, shift, n=2000):
    """The sample variance of the noise alone: zero latent, identity
    warping, every value pooled."""
    sims = tf.zeros([n, 1, 1], tf.float64)
    samples = np.asarray(likelihood.measurement_samples(
        sims, n_nodes=n_nodes, shift=shift), dtype=float).ravel()
    return float(np.var(samples))


def _epsilon_insensitive():
    # at its defaults (epsilon 0.001) this is a Laplace with a hair of a
    # flat top; a visible epsilon makes it the third law it is meant to be
    likelihood = geoml.likelihood.EpsilonInsensitive(
        warping=geoml.warping.Identity(1))
    likelihood.parameters["epsilon"].set_value(np.full((1, 1, 1), 1.0))
    return likelihood


@pytest.mark.parametrize("make", [
    lambda: geoml.likelihood.Gaussian(warping=geoml.warping.Identity(1)),
    lambda: geoml.likelihood.Laplace(warping=geoml.warping.Identity(1)),
    _epsilon_insensitive,
])
def test_the_rotated_sample_carries_the_noise_variance(make):
    """Rotated, the pooled sample's variance is the law's; the midpoints'
    falls short, and further the heavier the tails."""
    likelihood = make()
    reference = _reference_variance(likelihood)
    rng = np.random.default_rng(0)
    shift = tf.constant(rng.random((2000, 1, 1)), tf.float64)

    rotated = _pooled_variance(likelihood, 32, shift) / reference
    midpoints = _pooled_variance(likelihood, 32, None) / reference

    assert 0.95 < rotated < 1.05
    assert midpoints < 0.97
    assert midpoints < rotated


def test_the_rotation_keeps_one_node_per_stratum():
    likelihood = geoml.likelihood.Gaussian(warping=geoml.warping.Identity(1))
    rng = np.random.default_rng(1)
    shift = rng.random((5, 1, 3))

    u = np.asarray(likelihood._measurement_nodes(8, shift))

    assert u.shape == (8, 5, 1, 3)
    strata = np.floor(u * 8).astype(int)
    for row in range(5):
        for r in range(3):
            assert sorted(strata[:, row, 0, r]) == list(range(8))
    # different rotations at different locations, and at different
    # realizations of one location
    assert not np.allclose(u[:, 0, 0, 0], u[:, 1, 0, 0])
    assert not np.allclose(u[:, 0, 0, 0], u[:, 0, 0, 1])
    # without a shift, the midpoints as before
    plain = np.asarray(likelihood._measurement_nodes(8))
    assert np.allclose(plain[:, 0], (np.arange(8) + 0.5) / 8)


def test_two_locations_never_share_a_noise_value():
    likelihood = geoml.likelihood.Gaussian(warping=geoml.warping.Identity(1))
    sims = tf.zeros([2, 1, 3], tf.float64)
    rng = np.random.default_rng(2)

    shifted = np.asarray(likelihood.measurement_samples(
        sims, n_nodes=4, shift=tf.constant(rng.random((2, 1, 3)))))
    plain = np.asarray(likelihood.measurement_samples(sims, n_nodes=4))

    assert shifted.shape == (2, 1, 12)
    assert not np.allclose(shifted[0], shifted[1])
    assert np.allclose(plain[0], plain[1])


def test_the_mixture_quantile_holds_under_rotation():
    """The bisected mixture quantile answers a rotated `u` of any shape:
    the mixture's CDF at the value returns the `u` it was asked for."""
    mixture = geoml.likelihood.Mixture(geoml.warping.ZScore(1),
                                       n_components=2)
    rng = np.random.default_rng(3)
    shift = tf.constant(rng.random((3, 1, 2)), tf.float64)

    values = mixture._measurement_values(4, shift)
    u = mixture._measurement_nodes(4, shift)
    w = np.asarray(mixture.parameters["weights"].get_value()).ravel()
    cdf = sum(wk * np.asarray(d.cdf(values))
              for wk, d in zip(w, mixture._component_distributions()))

    assert values.shape == (4, 3, 1, 2)
    assert np.allclose(cdf, np.asarray(u), atol=1e-6)


def _walker_model(seed):
    geoml.set_seed(seed)
    walker, _ = geoml.datasets.walker()
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(walker, 30, seed=0),
        transform=geoml.transform.Isotropic(50))
    model = geoml.models.VGPNetwork(
        walker, "V", geoml.likelihood.Gaussian(), geoml.latent.BasicGP(root),
        options=geoml.models.GPOptions(verbose=False, training_samples=4))
    model.train_full(max_iter=5)
    return model, walker


def test_the_door_is_seeded_and_batch_invariant():
    """The rotation comes from the model's seed and is sliced by row, so
    the samples repeat call to call, do not depend on the batch size, and
    change with the seed."""
    model, walker = _walker_model(1234)
    first = model.predict_measurements(walker, n_sim=2, n_nodes=4)["V"]
    again = model.predict_measurements(walker, n_sim=2, n_nodes=4)["V"]
    assert np.array_equal(first, again)

    model.options.prediction_batch_size = 97
    batched = model.predict_measurements(walker, n_sim=2, n_nodes=4)["V"]
    assert np.allclose(batched, first, atol=1e-10)

    other, walker_2 = _walker_model(4321)
    different = other.predict_measurements(walker_2, n_sim=2, n_nodes=4)["V"]
    assert not np.allclose(different, first)


def test_the_noise_is_independent_between_locations():
    """Read across locations, a column of samples is an independent draw
    of the noise at each location, so it carries no correlation along the
    rows. (Within one row the columns are one rotated lattice and so
    dependent on each other, which is what a stratified sample is.)
    Before the rotation every location in a column carried the same
    noise value, so every row's deviation series was the same up to the
    warping and the mean pairwise correlation read one."""
    model, walker = _walker_model(1234)
    # one latent realization, so the columns of a row differ by the noise
    # alone
    samples = model.predict_measurements(walker, n_sim=1, n_nodes=32)["V"]
    deviations = samples[:, 0, :] - samples[:, 0, :].mean(axis=1, keepdims=True)
    # the old code's signature: identical deviation series row to row
    correlations = np.corrcoef(deviations[:80])
    assert abs(correlations[np.triu_indices(80, k=1)].mean()) < 0.05
    # the property meant: one column, read down the rows, is independent
    for k in (0, 7, 31):
        column = deviations[:, k]
        lag_one = np.corrcoef(column[:-1], column[1:])[0, 1]
        assert abs(lag_one) < 0.15


def test_a_shift_of_the_wrong_width_is_refused():
    likelihood = geoml.likelihood.MultivariateGaussian(
        3, warping=geoml.warping.Identity(3))
    with pytest.raises(ValueError, match="size 3"):
        likelihood._measurement_nodes(4, np.random.default_rng(0).random((5, 1, 2)))
