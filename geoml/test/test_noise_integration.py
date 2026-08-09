"""How the likelihood noise is integrated out of a prediction.

A prediction reports ``E[g(z + eps)]`` -- the value the ground would show once
the measurement error and the variability below the model's resolution are
averaged over -- rather than ``g(z + eps)`` for some drawn ``eps``. Nothing is
drawn, so there is no seed and no dependence on how a prediction was batched,
and the correction is a deterministic shift: on a convex back-transform,
leaving it out biases the answer low.

The integral is a quadrature over the unit cube reached through the
likelihood's quantile function, and the only thing that distinguishes the
univariate from the multivariate case is which array of nodes is used. That
choice rests on `_Warping.elementwise`, so the first test here is that no
warping claims to work component by component when it does not.
"""
import numpy as np
import pytest

import geoml
import geoml.warping as wp


def _cases():
    """Every warping, with data it can be initialized on."""
    rng = np.random.default_rng(0)
    positive = rng.lognormal(size=[200, 3])
    normal = rng.normal(size=[200, 3])
    unit = rng.uniform(0.05, 0.95, size=[200, 3])
    composition = positive / positive.sum(axis=1, keepdims=True)
    return [
        (wp.Identity(3), normal),
        (wp.Spline(3), normal),
        (wp.ZScore(3), normal),
        (wp.Center(3), normal),
        (wp.Softplus(3), positive),
        (wp.Log(3), positive),
        (wp.Scale(3), positive),
        (wp.Sigmoid(3), unit),
        (wp.PCA(3), normal),
        (wp.RobustPCA(3), normal),
        (wp.CenteredLogRatio(3), composition),
        (wp.Rotation(3), normal),
        (wp.ScaledSimplex(3), composition),
        (wp.ContinuousNormalizingFlow(3, inducing_points=10), normal),
    ]


def _crosstalk(warping):
    """How far the other components move when the first one is nudged."""
    import tensorflow as tf
    rng = np.random.default_rng(1)
    base = rng.normal(size=[64, 3]) * 0.5
    bump = np.zeros([1, 3])
    bump[0, 0] = 1e-4

    out = np.asarray(warping.backward(tf.constant(base)))
    nudged = np.asarray(warping.backward(tf.constant(base + bump)))
    moved = np.abs(nudged - out).max(axis=0)
    return moved[1:].max() / max(np.abs(out).max(), 1e-12)


@pytest.mark.parametrize("warping,data", _cases(),
                         ids=lambda x: type(x).__name__ if hasattr(x, "backward")
                         else "")
def test_a_warping_that_claims_to_be_elementwise_is(warping, data):
    """The declaration decides how the noise is integrated, so it has to be
    true. Claiming to mix when you do not merely costs nodes; claiming not to
    when you do integrates the wrong thing."""
    warping.initialize(data)
    if warping.elementwise:
        assert _crosstalk(warping) < 1e-9


def test_the_mixing_warpings_really_mix():
    """The other direction, for the ones where it can be seen at once. The
    flow is left out: freshly initialized it is nearly the identity, which is
    its own item on the to-do list."""
    for warping, data in _cases():
        if type(warping).__name__ == "ContinuousNormalizingFlow":
            continue
        warping.initialize(data)
        if not warping.elementwise:
            assert _crosstalk(warping) > 1e-9


def test_one_mixing_link_carries_the_whole_chain():
    """Everything applied after a mix sees the mixture."""
    plain = wp.ChainedWarping(wp.Softplus(3), wp.ZScore(3), wp.Spline(3))
    mixed = wp.ChainedWarping(wp.Softplus(3), wp.ZScore(3), wp.Rotation(3),
                              wp.Spline(3))

    assert plain.elementwise
    assert not mixed.elementwise


def test_a_single_component_has_nothing_to_mix():
    """A one-by-one rotation is a sign and a one-component PCA is a scale."""
    assert wp.Rotation(1).elementwise
    assert wp.PCA(1).elementwise
    assert not wp.Rotation(2).elementwise


# --------------------------------------------------------------------------- #
# the integral
# --------------------------------------------------------------------------- #
def _likelihood(warping=None, noise=0.1, likelihood=None):
    """A likelihood whose back-transform bends, so integrating does something."""
    if warping is None:
        warping = wp.ChainedWarping(wp.Softplus(1), wp.ZScore(1))
    if likelihood is None:
        likelihood = geoml.likelihood.Gaussian(warping)
        likelihood.parameters["noise"].set_value(np.full([1, 1, 1], noise))
    likelihood.initialize(
        np.exp(np.random.default_rng(0).normal(size=[500, 1])))
    return likelihood


def _integrated(likelihood, z, part=0):
    import tensorflow as tf
    return np.asarray(likelihood.integrated_backward(
        tf.constant(np.reshape(z, [-1, 1, 1])))[part])[:, 0, 0]


def _plain(likelihood, z):
    import tensorflow as tf
    return np.asarray(likelihood.warping.backward(
        tf.constant(np.reshape(z, [-1, 1]))))[:, 0]


def test_an_affine_back_transform_has_nothing_to_integrate():
    """`E[a(z + eps) + b] = az + b`. The default warping is a `ZScore`, so on
    an unwarped model the correction is exactly zero."""
    likelihood = _likelihood(wp.ZScore(1))
    z = np.linspace(-3, 3, 25)
    assert np.allclose(_integrated(likelihood, z), _plain(likelihood, z),
                       atol=1e-10)


def test_a_measurement_still_scatters_under_an_affine_warping():
    """The value is unchanged, but a sample of it is not the value: `a eps`
    is still there. The two moments answer different questions."""
    likelihood = _likelihood(wp.ZScore(1), noise=0.3)
    scale = np.asarray(
        likelihood.warping.parameters["std"].get_value()).ravel()[0]

    variance = _integrated(likelihood, np.linspace(-3, 3, 25), part=1)
    assert np.allclose(variance, 0.3 * scale ** 2, rtol=1e-6)


def test_the_measurement_variance_matches_a_fine_reference():
    likelihood = _likelihood(noise=0.2)
    z = np.linspace(-2.5, 2.5, 21)

    t = np.linspace(-9, 9, 10001)
    weight = np.exp(-0.5 * t ** 2)
    weight = weight / weight.sum()
    reference = []
    for zi in z:
        values = _plain(likelihood, zi + np.sqrt(0.2) * t)
        mean = np.sum(values * weight)
        reference.append(np.sum((values - mean) ** 2 * weight))

    assert np.allclose(_integrated(likelihood, z, part=1), reference,
                       rtol=1e-4, atol=1e-8)


def test_the_measurement_variance_is_never_negative():
    """Second moment minus the square of the first, so rounding can take it
    below zero where the two are equal."""
    likelihood = _likelihood(noise=1e-6)
    assert np.all(_integrated(likelihood, np.linspace(-3, 3, 41), part=1) >= 0)


def test_the_integral_matches_a_fine_reference():
    """Eight Gauss-Hermite nodes against ten thousand evenly spaced ones.

    They agree to five figures, and the tolerance is set by the *reference*:
    a Riemann sum over a uniform grid converges as the square of its spacing,
    which is what the last digit is doing.
    """
    likelihood = _likelihood(noise=0.2)
    z = np.linspace(-2.5, 2.5, 21)

    t = np.linspace(-9, 9, 10001)
    weight = np.exp(-0.5 * t ** 2)
    weight = weight / weight.sum()
    reference = np.array(
        [np.sum(_plain(likelihood, zi + np.sqrt(0.2) * t) * weight)
         for zi in z])

    assert np.allclose(_integrated(likelihood, z), reference,
                       rtol=1e-5, atol=1e-8)


def test_integrating_raises_a_skewed_variable():
    """Jensen: the back-transform is convex, so the noise pushes the value up
    and leaving it out biases the answer low."""
    likelihood = _likelihood(noise=0.5)
    z = np.linspace(-2, 2, 41)
    assert np.all(_integrated(likelihood, z) > _plain(likelihood, z))


def test_the_noise_adds_no_randomness():
    """It moves every value along one monotone curve, so the order of the
    simulations is untouched. Drawing the noise would scramble it."""
    likelihood = _likelihood(noise=0.5)
    z = np.random.default_rng(1).normal(size=200)
    assert np.array_equal(np.argsort(_integrated(likelihood, z)),
                          np.argsort(_plain(likelihood, z)))


def test_the_likelihood_decides_the_noise_law():
    """The nodes live in the unit cube and reach the noise through the
    likelihood's own quantile function, so two laws of the same width give
    two different answers."""
    gaussian = _likelihood(noise=0.5)
    laplace = _likelihood(likelihood=geoml.likelihood.Laplace(
        wp.ChainedWarping(wp.Softplus(1), wp.ZScore(1))))

    z = np.linspace(-2, 2, 21)
    assert not np.allclose(_integrated(gaussian, z), _integrated(laplace, z))


# --------------------------------------------------------------------------- #
# the nodes
# --------------------------------------------------------------------------- #
def _nodes(warping):
    likelihood = geoml.likelihood.MultivariateGaussian(
        warping.size_out, warping)
    return likelihood._noise_nodes()


def test_an_elementwise_chain_gets_one_node_per_component():
    """Gauss-Hermite, the same node applied to every component -- which is
    why the cost does not grow with the number of them."""
    u, weights = _nodes(wp.ChainedWarping(wp.Softplus(3), wp.ZScore(3)))

    assert u.shape == (8, 3)
    assert np.allclose(np.asarray(weights).sum(), 1.0)
    # one node, repeated across the components
    assert np.allclose(np.asarray(u), np.asarray(u)[:, :1])


def test_a_mixing_chain_gets_a_rule_over_every_component_at_once():
    u, weights = _nodes(wp.ChainedWarping(wp.Softplus(3), wp.Rotation(3)))

    assert u.shape == (64, 3)
    assert np.allclose(np.asarray(weights).sum(), 1.0)
    # a genuinely multivariate point set: the components differ
    assert not np.allclose(np.asarray(u), np.asarray(u)[:, :1])


def test_the_rule_is_fixed_rather_than_random():
    """The scramble is seeded, so a prediction does not depend on when it was
    made, nor on how it was batched."""
    chain = wp.ChainedWarping(wp.Softplus(3), wp.Rotation(3))
    assert np.array_equal(np.asarray(_nodes(chain)[0]),
                          np.asarray(_nodes(chain)[0]))
