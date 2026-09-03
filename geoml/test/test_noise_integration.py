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
        (wp.Spline(3, backbone="rq"), normal),
        (wp.ZScore(3), normal),
        (wp.Center(3), normal),
        (wp.Softplus(3), positive),
        (wp.Log(3), positive),
        (wp.Scale(3), positive),
        (wp.Sigmoid(3), unit),
        (wp.BoxCox(3), positive),
        (wp.YeoJohnson(3), normal),
        (wp.Arcsinh(3), normal),
        (wp.SinhArcsinh(3), normal),
        (wp.PCA(3), normal),
        (wp.RobustPCA(3), normal),
        (wp.CenteredLogRatio(3), composition),
        (wp.Rotation(3), normal),
        (wp.ScaledSimplex(3), composition),
        (wp.ContinuousNormalizingFlow(3, inducing_points=10), normal),
        (wp.TensorProductFlow(3, grid=7, rank=3), normal),
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
    flow included since 0.6.9: it initializes near the identity, but near
    is not at, and a nudge must couple the components even before training
    moves it."""
    for warping, data in _cases():
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
# the log-determinant
# --------------------------------------------------------------------------- #
def _jacobian(warping, row, step=1e-6):
    """The forward map's Jacobian at one row, by central differences."""
    import tensorflow as tf
    row = np.asarray(row, dtype=float)[None, :]
    out = np.zeros([row.shape[1], row.shape[1]])
    for column in range(row.shape[1]):
        bump = np.zeros_like(row)
        bump[0, column] = step
        plus = np.asarray(warping.forward(tf.constant(row + bump))[0])
        minus = np.asarray(warping.forward(tf.constant(row - bump))[0])
        out[:, column] = (plus - minus)[0] / (2 * step)
    return out


def _numeric_log_det(warping, values, step=1e-6):
    """`log|det J|` of the forward map, by central differences."""
    return np.asarray([
        np.log(np.abs(np.linalg.det(_jacobian(warping, row, step))))
        for row in np.atleast_2d(values)])


def _log_det_cases():
    """The warpings whose forward map has a Jacobian determinant of its own.

    `CenteredLogRatio` is left out because its Jacobian between arrays of
    `n_dim` columns is singular rather than merely constant, so it is
    checked on the hyperplane it actually maps onto, in its own test below.
    The flow is left out because its log-determinant comes from the
    integration it performs rather than from a closed form.
    """
    rng = np.random.default_rng(3)
    positive = rng.lognormal(size=[8, 3])
    normal = rng.normal(size=[8, 3])
    unit = rng.uniform(0.05, 0.95, size=[8, 3])
    # a covariance estimate needs rows, and a robust one needs more of them
    spread = rng.normal(size=[40, 3]) @ np.array([[2.0, 0.3, 0.0],
                                                  [0.0, 1.0, 0.5],
                                                  [0.1, 0.0, 3.0]])
    parts = rng.dirichlet(np.ones(3), size=40)
    return [
        (wp.Identity(3), normal),
        (wp.Spline(3), normal),
        (wp.Spline(3, backbone="rq"), normal),
        (wp.ZScore(3), normal),
        (wp.Center(3), normal),
        (wp.Scale(3), positive),
        (wp.Softplus(3), positive),
        (wp.Log(3), positive),
        (wp.Sigmoid(3), unit),
        (wp.BoxCox(3), positive),
        (wp.YeoJohnson(3), normal),
        (wp.Arcsinh(3), normal),
        (wp.SinhArcsinh(3), normal),
        (wp.Rotation(3), normal),
        (wp.PCA(3), spread),
        (wp.RobustPCA(3), spread),
        (wp.ScaledSimplex(3), parts),
    ]


@pytest.mark.parametrize("warping,data", _log_det_cases(),
                         ids=lambda x: type(x).__name__
                         if hasattr(x, "backward") else "")
def test_the_reported_log_determinant_is_the_log_of_the_jacobians(warping,
                                                                  data):
    """`forward` returns the *log* of the Jacobian determinant, not the
    determinant.

    `Log` returned `sum(1 / (x + shift))` until 0.6.5, where the truth is
    `-sum(log(x + shift))`. It went unnoticed because `Log` carries no
    trainable parameter, so as the first link of a chain its term is a
    constant that no gradient sees; put anything trainable in front of it
    and the model optimizes the wrong objective. `PCA`, `RobustPCA` and
    `ScaledSimplex` reported zero for the same length of time, each where
    the truth is a constant settled at initialization.
    """
    import tensorflow as tf
    warping.initialize(data)
    # every row is checked against differences, but only a few of them:
    # fitting a covariance takes more rows than the check needs
    rows = data[:8]
    reported = np.asarray(warping.forward(tf.constant(rows))[1])
    assert np.allclose(reported, _numeric_log_det(warping, rows),
                       atol=1e-5, rtol=1e-4)


def _tangent_basis(n, rotation=None):
    """An orthonormal basis of `{u : sum(u) = 0}` -- the simplex's tangent
    space, and the space the centered log-ratio maps onto. `rotation` turns
    it into a different basis of the same space."""
    centred = np.eye(n) - np.ones([n, n]) / n
    basis = np.linalg.svd(centred)[0][:, :n - 1]
    return basis if rotation is None else basis @ rotation


def _helmert_basis(n):
    """The classical balance matrix, each row contrasting one part against
    the average of those before it."""
    rows = []
    for i in range(1, n):
        row = np.zeros(n)
        row[:i] = 1.0 / i
        row[i] = -1.0
        rows.append(row * np.sqrt(i / (i + 1.0)))
    return np.asarray(rows).T


@pytest.mark.parametrize("n", [3, 5])
def test_the_log_ratios_determinant_is_the_one_on_the_hyperplane(n):
    """Between arrays of `n` columns the centered log-ratio has no
    determinant: it ignores a rescaling of the whole row, so its Jacobian is
    singular. What it does have is a volume factor for the map it really is,
    from the simplex onto the hyperplane where the components sum to zero,
    and that is what `forward` reports.
    """
    import tensorflow as tf
    rng = np.random.default_rng(20 + n)
    data = rng.dirichlet(np.ones(n), size=6)

    warping = wp.CenteredLogRatio(n)
    reported = np.asarray(warping.forward(tf.constant(data))[1])

    basis = _tangent_basis(n)
    for i, row in enumerate(data):
        jacobian = _jacobian(warping, row)
        # singular, and read as a rank rather than as a determinant: a row
        # with a small part makes the entries huge, and a determinant that
        # is zero next to them is not a number close to zero
        singular = np.linalg.svd(jacobian, compute_uv=False)
        assert singular[-1] < 1e-8 * singular[0]

        restricted = basis.T @ jacobian @ basis
        assert np.log(abs(np.linalg.det(restricted))) == \
            pytest.approx(reported[i], abs=1e-5)


@pytest.mark.parametrize("n", [3, 5])
def test_the_log_ratios_determinant_needs_no_balance_matrix(n):
    """Two orthonormal bases of the same hyperplane differ by a rotation,
    whose determinant is one either way, so the volume factor does not
    depend on which is used. Which is why the reported value is a closed
    form with no basis in it, and why no balance matrix has to be chosen.
    """
    rng = np.random.default_rng(30 + n)
    data = rng.dirichlet(np.ones(n), size=3)

    turn = np.linalg.qr(rng.normal(size=[n - 1, n - 1]))[0]
    bases = [_tangent_basis(n), _helmert_basis(n), _tangent_basis(n, turn)]
    for basis in bases:
        assert np.allclose(basis.T @ basis, np.eye(n - 1))
        assert np.allclose(basis.sum(axis=0), 0.0)

    warping = wp.CenteredLogRatio(n)
    for row in data:
        jacobian = _jacobian(warping, row)
        measured = [np.log(abs(np.linalg.det(basis.T @ jacobian @ basis)))
                    for basis in bases]
        assert measured == pytest.approx([measured[0]] * len(bases), abs=1e-6)


def test_a_projection_reports_no_determinant():
    """With fewer components than variables the map is not square, so there
    is no determinant to report and nothing to correct a density by: what
    the likelihood sees is the projection."""
    import tensorflow as tf
    data = np.random.default_rng(9).normal(size=[40, 3]) \
        @ np.array([[2.0, 0.3, 0.0], [0.0, 1.0, 0.5], [0.1, 0.0, 3.0]])

    warping = wp.PCA(3, n_components=2)
    warping.initialize(data)
    reported = np.asarray(warping.forward(tf.constant(data))[1])
    assert np.allclose(reported, 0.0)


def test_a_chain_adds_its_links_log_determinants():
    """Which is the only composition rule a log-determinant obeys, and the
    reason the accumulator has to start at zero. It started at the number of
    columns until 0.6.5, offsetting every chained warping by its own width.
    """
    import tensorflow as tf
    data = np.random.default_rng(4).lognormal(size=[8, 3])

    links = [wp.Log(3), wp.ZScore(3), wp.Spline(3)]
    chain = wp.ChainedWarping(*links)
    chain.initialize(data)

    expected = np.zeros(data.shape[0])
    passed = tf.constant(data)
    for link in links:
        passed, log_det = link.forward(passed)
        expected = expected + np.asarray(log_det)

    reported = np.asarray(chain.forward(tf.constant(data))[1])
    assert np.allclose(reported, expected)


def test_a_chain_of_identities_leaves_the_log_determinant_at_zero():
    """The width used to leak in through the accumulator, so a chain of
    identities came back at 3 instead of 0."""
    import tensorflow as tf
    data = np.random.default_rng(5).normal(size=[8, 3])

    chain = wp.ChainedWarping(wp.Identity(3), wp.Identity(3))
    chain.initialize(data)

    assert np.allclose(np.asarray(chain.forward(tf.constant(data))[1]), 0.0)


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


def test_a_width_changing_warping_comes_back_at_the_data_width():
    """A chain holding a PCA takes the latent columns back to *more* data
    components than it was handed -- the Macpass composition maps 3 to 4.
    The vectorized back-transform used to reshape by the input width and
    crash there; the width is the warping's to declare, not the input's."""
    import tensorflow as tf

    rng = np.random.default_rng(2)
    positive = rng.lognormal(size=[200, 4])
    composition = positive / positive.sum(axis=1, keepdims=True)

    chain = wp.ChainedWarping(wp.CenteredLogRatio(4), wp.PCA(4, 3))
    likelihood = geoml.likelihood.MultivariateGaussian(3, chain)
    likelihood.initialize(composition)

    sims = tf.constant(rng.normal(size=[30, 3, 7]) * 0.3)

    values = np.asarray(likelihood._back_transform(sims))
    assert values.shape == (30, 4, 7)
    # exact against the one-realization-at-a-time path it replaced
    for r in range(7):
        alone = np.asarray(chain.backward(sims[:, :, r]))
        assert np.allclose(values[:, :, r], alone)

    mean, variance = likelihood.integrated_backward(sims)
    assert tuple(mean.shape) == (30, 4, 7)
    assert tuple(variance.shape) == (30, 4, 7)
    # every integrated value is a composition again: positive and closed
    assert np.all(np.asarray(mean) > 0)
    assert np.allclose(np.asarray(mean).sum(axis=1), 1.0)


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
