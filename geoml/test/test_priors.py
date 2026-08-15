"""MAP priors on the point-estimated parameters.

A parameter that declares a `prior` stays a point estimate; the prior's
log-density joins the training objective next to the KL, so the whole is a
bound on `log p(y, theta)` rather than `log p(y | theta)`. Nothing is
integrated and nothing is sampled: the term is one differentiable number,
and the gradient does the rest. Almost every parameter declares no prior,
and the objective is then exactly what it always was.
"""
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import geoml
import geoml.parameter as gpr

tfd = tfp.distributions


def _f64(value):
    return tf.constant(value, tf.float64)


def _model(seed=1234):
    """Deliberately prior-free (`range_prior=None`), so these tests see the
    mechanism alone rather than the canonical defaults, which have their own
    section below."""
    geoml.set_seed(seed)
    walker_point, _ = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[8, 8], step=[33, 37])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Cubic(),
                              range_prior=None)
    options = geoml.models.GPOptions(verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        walker_point, "V", geoml.likelihood.Gaussian(), gp, options=options)
    return model, gp


# --------------------------------------------------------------------------- #
# the term itself
# --------------------------------------------------------------------------- #
def test_a_parameter_without_a_prior_contributes_zero():
    parameter = gpr.RealParameter(1.5, -5.0, 5.0)
    assert float(parameter.log_prior()) == 0.0

    model, _ = _model()
    assert float(model.log_prior()) == 0.0


def test_the_log_prior_is_the_distributions_log_density():
    parameter = gpr.RealParameter(
        np.array([0.5, -1.0, 2.0]), np.full(3, -5.0), np.full(3, 5.0),
        prior=tfd.Normal(_f64(0.0), _f64(1.0)))

    value = parameter.get_value().numpy()
    expected = np.sum(-0.5 * value ** 2 - 0.5 * np.log(2 * np.pi))
    assert float(parameter.log_prior()) == pytest.approx(expected)


def test_the_models_term_is_the_sum_over_its_parameters():
    model, gp = _model()
    gp.parameters["ranges"].prior = tfd.Gamma(_f64(2.0), _f64(2.0))
    likelihood = model.likelihoods[0]
    noise = likelihood.parameters["noise"]
    noise.prior = tfd.Gamma(_f64(2.0), _f64(2.0))

    expected = float(gp.parameters["ranges"].log_prior()) \
        + float(noise.log_prior())
    assert float(model.log_prior()) == pytest.approx(expected)


def test_a_fixed_parameter_is_left_out():
    """Its term would be a constant with no gradient: noise in the reported
    objective, information in nothing."""
    model, gp = _model()
    gp.parameters["ranges"].prior = tfd.Gamma(_f64(2.0), _f64(2.0))
    gp.parameters["ranges"].fix()
    assert float(model.log_prior()) == 0.0


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #
def test_the_prior_pulls_the_parameter():
    """The mechanism end to end: the same model, the same seed, and a tight
    prior well away from where unpriored training goes. The priored run must
    land nearer the prior's mean, and its reported objective must carry the
    term."""
    plain, gp_plain = _model()
    plain.train_full(max_iter=60)
    free_range = gp_plain.parameters["ranges"].get_value().numpy().ravel()

    pulled, gp_pulled = _model()
    # mean 5, standard deviation ~0.35: a strong opinion, far from anywhere
    # the unpriored run settles
    gp_pulled.parameters["ranges"].prior = tfd.Gamma(_f64(200.0), _f64(40.0))
    pulled.train_full(max_iter=60)
    pulled_range = gp_pulled.parameters["ranges"].get_value().numpy().ravel()

    assert np.abs(pulled_range - 5.0).mean() < np.abs(free_range - 5.0).mean()
    assert not np.allclose(pulled_range, free_range)
    assert not np.allclose(plain.training_log, pulled.training_log)


def test_without_priors_the_objective_is_untouched():
    """Two identically seeded models, one built before anyone thought about
    priors and one after, are the same model."""
    first, _ = _model()
    first.train_full(max_iter=10)
    second, _ = _model()
    second.train_full(max_iter=10)
    assert np.allclose(first.training_log, second.training_log)


# --------------------------------------------------------------------------- #
# the canonical priors
# --------------------------------------------------------------------------- #
def _root(n=6):
    geoml.set_seed(1234)
    walker_point, _ = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[n, n], step=[44, 50])
    return geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50)), walker_point


def test_a_basic_gps_ranges_carry_the_gamma_by_default():
    root, _ = _root()
    gp = geoml.latent.BasicGP(root, size=1)
    ranges = gp.parameters["ranges"]
    assert ranges.prior is not None
    # peaking at 1, the whitened scale: the log-density's gradient is zero
    # there, negative above, positive below
    with tf.GradientTape() as tape:
        tape.watch(ranges.variable)
        density = ranges.log_prior()
    gradient = tape.gradient(density, ranges.variable)
    assert gradient is not None

    bare = geoml.latent.BasicGP(root, size=1, range_prior=None)
    assert bare.parameters["ranges"].prior is None


def test_a_free_linears_weights_carry_the_standard_gaussian():
    root, _ = _root()
    free = geoml.latent.Linear(root, size=3, unit_norm=False)
    weights = free.parameters["weights"]
    assert weights.prior is not None
    # the walls step back to a safety net when the prior holds the weights
    assert np.allclose(weights.min_transformed.numpy(), -10.0)

    bare = geoml.latent.Linear(root, size=3, unit_norm=False,
                               weight_prior=None)
    assert bare.parameters["weights"].prior is None
    assert np.allclose(bare.parameters["weights"].min_transformed.numpy(),
                       -1.0)

    # the unit norm is constraint enough on its own
    normed = geoml.latent.Linear(root, size=3, unit_norm=True)
    assert normed.parameters["weights"].prior is None


def test_per_component_weights_share_one_vote_per_column():
    root, _ = _root()
    first = geoml.latent.BasicGP(root, size=2)
    second = geoml.latent.BasicGP(root, size=2)

    combined = geoml.latent.LinearCombination(first, second,
                                              per_component=True)
    weights = combined.parameters["weights"].get_value().numpy()
    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=0), 1.0)
    assert combined.parameters["weights"].prior is not None
    assert np.isfinite(float(combined.parameters["weights"].log_prior()))

    shared = geoml.latent.LinearCombination(first, second)
    assert shared.parameters["weights"].get_value().numpy().shape == (2,)
    assert shared.parameters["weights"].prior is None

    with pytest.raises(ValueError, match="unit_variance"):
        geoml.latent.LinearCombination(first, second, per_component=True,
                                       unit_variance=False)
    with pytest.raises(ValueError, match="greater than 1"):
        geoml.latent.LinearCombination(first, second, per_component=True,
                                       weight_concentration=1.0)

    bare = geoml.latent.LinearCombination(first, second, per_component=True,
                                          weight_concentration=None)
    assert bare.parameters["weights"].prior is None


def test_the_multi_structure_weights_start_uniform_under_the_staircase():
    """The prior's peak is the staircase; the start is deliberately not.
    Initializing on the staircase was measured on Walker Lake against the
    exhaustive truth and rejected -- training never left that basin -- so a
    uniform start with the prior's pull is the default, and its Dirichlet
    concentrations are the decreasing ones."""
    root, _ = _root()
    gp = geoml.latent.MultiStructureGP(root, size=1, n_structures=3)
    weights = gp.parameters["weights"].get_value().numpy()
    assert np.allclose(weights, 1.0 / 3.0)
    assert gp.parameters["weights"].prior is not None
    concentration = gp.parameters[
        "weights"].prior.concentration.numpy().ravel()
    assert np.all(np.diff(concentration) < 0)
    for n in range(3):
        assert gp.parameters[f"ranges_{n}"].prior is not None

    symmetric = geoml.latent.MultiStructureGP(root, size=1, n_structures=3,
                                              weight_concentration=2.0)
    assert np.allclose(symmetric.parameters["weights"].get_value().numpy(),
                       1.0 / 3.0)
    assert symmetric.parameters["weights"].prior is not None

    with pytest.raises(ValueError, match="'staircase'"):
        geoml.latent.MultiStructureGP(root, size=1, n_structures=3,
                                      weight_concentration="uniform")

    bare = geoml.latent.MultiStructureGP(
        root, size=1, n_structures=3,
        weight_concentration=None, range_prior=None)
    assert np.allclose(bare.parameters["weights"].get_value().numpy(),
                       1.0 / 3.0)
    assert bare.parameters["weights"].prior is None
    for n in range(3):
        assert bare.parameters[f"ranges_{n}"].prior is None


def test_a_per_component_combination_predicts_end_to_end():
    """The broadcasting in `predict`, `refresh` and the inducing-point
    propagation, exercised live rather than argued about."""
    root, walker_point = _root()
    first = geoml.latent.BasicGP(root, size=1,
                                 kernel=geoml.kernels.Cubic())
    second = geoml.latent.BasicGP(root, size=1,
                                  kernel=geoml.kernels.Spherical())
    combined = geoml.latent.LinearCombination(first, second,
                                              per_component=True)

    model = geoml.models.VGPNetwork(
        walker_point, "V", geoml.likelihood.Gaussian(), combined,
        options=geoml.models.GPOptions(verbose=False, training_samples=4))
    model.train_full(max_iter=5)
    assert np.all(np.isfinite(model.training_log))

    grid = geoml.data.Grid2D(start=[0, 0], end=[260, 300], n=[12, 12])
    model.predict(grid, n_sim=4)
    assert np.all(np.isfinite(grid.values("V/prediction")))


def test_a_model_with_default_priors_round_trips(tmp_path):
    """Persistence replays constructors, so the priors come back with the
    model rather than being pickled."""
    root, walker_point = _root()
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Cubic())
    model = geoml.models.VGPNetwork(
        walker_point, "V", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=4))
    model.train_full(max_iter=3)

    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    ranges = loaded.latent_network.parameters["ranges"]
    assert ranges.prior is not None
    loaded.train_full(max_iter=3)
    assert np.all(np.isfinite(loaded.training_log))
