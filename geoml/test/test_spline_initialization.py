"""`Spline.initialize`: the knots fitted to the data's normal-score transform.

The spline's input knots are fixed on a regular grid over [-5, 5] and its
*output* positions are what train, carried by two compositional parameters
per dimension. That construction pins the map at `(-5, -5)`, `(0, 0)` and
`(5, 5)` -- a compositional vector sums to one -- and makes it monotone
whatever the parameters hold. The initializer chooses the shape between the
anchors, and nothing else.

Before this existed the knots kept their uniform partition, which reads out
as exactly the input grid: the identity. A chain of `Rotation` and `Spline`
pairs therefore rotated repeatedly with nothing gaussianizing in between,
which is measured below as the thing that changed.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml.warping as wp


def _f64(value):
    return tf.constant(np.asarray(value, dtype=float), tf.float64)


def _skewed(n=600, size=2, seed=0):
    """Lognormal columns: strongly skewed, which is what the transform is
    for, and standardized, which is what the knot grid assumes."""
    rng = np.random.default_rng(seed)
    raw = np.exp(rng.normal(size=(n, size)) * 1.2)
    return (raw - raw.mean(0)) / raw.std(0)


def _departure(x):
    """Skewness and excess kurtosis of each column, as one number."""
    x = np.asarray(x, dtype=float)
    z = (x - x.mean(0)) / x.std(0)
    return (np.mean(z ** 3, axis=0) ** 2
            + (np.mean(z ** 4, axis=0) - 3.0) ** 2 / 4)


def _projection_index(x, seed=0):
    """The same departure, along many random directions rather than along
    the axes -- which is the only fair way to compare across a chain that
    contains a rotation, since a rotation moves the axes."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(x.shape[1], 200))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    return float(np.mean(_departure(x @ directions)))


# --------------------------------------------------------------------------- #
# what the construction guarantees, before and after
# --------------------------------------------------------------------------- #
def test_an_uninitialized_spline_is_the_identity():
    """The baseline the initializer improves on, pinned so that a change in
    the default partition cannot pass unnoticed."""
    spline = wp.Spline(2, knots_per_arm=5)
    x = _f64(np.linspace(-4, 4, 17)[:, None] * np.ones((1, 2)))
    warped, log_det = spline.forward(x)

    assert np.allclose(warped, x, atol=1e-12)
    # the interpolator's derivative is computed rather than exact, so the
    # log-determinant of an identity map is a few parts per million from
    # zero rather than zero
    assert np.allclose(log_det, 0.0, atol=1e-4)


@pytest.mark.parametrize("knots_per_arm", [3, 5, 10])
def test_the_anchors_survive_initialization(knots_per_arm):
    """`(-5, -5)`, `(0, 0)` and `(5, 5)` are structural: the compositional
    parameters cannot express anything else, and the initializer must not
    be written as though it could."""
    spline = wp.Spline(2, knots_per_arm=knots_per_arm)
    spline.initialize(_f64(_skewed(size=2)))

    for dim in range(2):
        coordinates = spline._get_warped_coordinates(dim).numpy()
        assert coordinates[0] == pytest.approx(-5.0)
        assert coordinates[knots_per_arm] == pytest.approx(0.0)
        assert coordinates[-1] == pytest.approx(5.0)


@pytest.mark.parametrize("knots_per_arm", [3, 5, 10])
def test_the_map_stays_monotone(knots_per_arm):
    spline = wp.Spline(3, knots_per_arm=knots_per_arm)
    spline.initialize(_f64(_skewed(size=3, seed=1)))

    for dim in range(3):
        coordinates = spline._get_warped_coordinates(dim).numpy()
        assert np.all(np.diff(coordinates) > 0)

    # and the map itself, densely, not just at the knots
    grid = _f64(np.linspace(-4.9, 4.9, 400)[:, None] * np.ones((1, 3)))
    warped, log_det = spline.forward(grid)
    assert np.all(np.diff(warped.numpy(), axis=0) > 0)
    assert np.all(np.isfinite(log_det.numpy()))


# --------------------------------------------------------------------------- #
# that it does the thing it exists for
# --------------------------------------------------------------------------- #
def test_it_gaussianizes_a_skewed_column():
    """Lognormal columns start at a departure of several hundred. What is
    left afterwards is set by the anchors -- each arm is rescaled to reach
    them, so the shape of the normal-score transform survives and its scale
    does not -- and by how coarse the empirical CDF is at the knots out in
    the tail."""
    data = _skewed(size=2, seed=2)
    before = _departure(data)
    assert np.all(before > 100)

    spline = wp.Spline(2, knots_per_arm=10)
    after = _departure(spline.initialize(_f64(data)))

    assert np.all(after < before / 50)
    assert np.all(after < 3.0)


def test_initialize_returns_what_it_warped():
    """A chain initializes each link on the previous link's output, so the
    return value is load-bearing rather than a convenience."""
    data = _skewed(size=2, seed=3)
    spline = wp.Spline(2, knots_per_arm=5)
    returned = spline.initialize(_f64(data))

    assert np.allclose(returned, spline.forward(_f64(data))[0])


def test_a_constant_column_is_left_alone():
    """No spread, no CDF worth inverting. The column keeps the uniform
    partition rather than producing a degenerate one -- the parameters are
    compositional, so a zero share would be minus infinity in logit
    coordinates."""
    data = _skewed(size=2, seed=4)
    data[:, 1] = 3.0

    spline = wp.Spline(2, knots_per_arm=5)
    spline.initialize(_f64(data))

    untouched = spline._get_warped_coordinates(1).numpy()
    assert np.allclose(untouched, np.linspace(-5, 5, 11), atol=1e-10)
    fitted = spline._get_warped_coordinates(0).numpy()
    assert not np.allclose(fitted, np.linspace(-5, 5, 11), atol=1e-3)
    assert np.all(np.isfinite(
        spline.forward(_f64(data))[1].numpy()))


def test_the_round_trip_comes_back():
    """An initialized spline is curved from the first iteration, which is
    what used to make `backward` -- then a second spline through the
    swapped knots -- inexact by a tenth of a standard deviation. It solves
    the forward polynomial now, so this comes back. `test_spline_inversion`
    owns the detail, including why the tolerance is not machine precision.
    """
    data = _skewed(size=2, seed=5)
    inside = np.clip(data, -4.5, 4.5)

    for knots_per_arm in (5, 20):
        spline = wp.Spline(2, knots_per_arm=knots_per_arm)
        spline.initialize(_f64(data))
        back = spline.backward(spline.forward(_f64(inside))[0]).numpy()
        assert np.all(np.isfinite(back))
        assert np.allclose(back, inside, atol=1e-9)


# --------------------------------------------------------------------------- #
# the knots the data does not reach
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("knots_per_arm", [3, 5, 10, 20])
def test_the_outermost_segment_keeps_its_share(knots_per_arm):
    """The knots span [-5, 5] and data rarely fills it, so the outer knots
    of each arm read the same empirical quantile and ask to sit on top of
    one another. The two that touch the anchors are held apart; the ones
    between them are free to collapse, and do."""
    data = _skewed(n=259, size=3, seed=8)
    spline = wp.Spline(3, knots_per_arm=knots_per_arm)
    spline.initialize(_f64(data))

    uniform = 5.0 / knots_per_arm
    for dim in range(3):
        gaps = np.diff(spline._get_warped_coordinates(dim).numpy())
        assert gaps[0] >= wp._OUTERMOST_SHARE * uniform * (1 - 1e-9)
        assert gaps[-1] >= wp._OUTERMOST_SHARE * uniform * (1 - 1e-9)


@pytest.mark.parametrize("knots_per_arm", [5, 10, 20])
def test_backward_stays_on_scale_past_the_knots(knots_per_arm):
    """What the user saw: a simulated value a little beyond the knot range
    came back at 1e5, and the `Log` below it in the chain overflowed to
    infinity. Out there the spline extrapolates along the end segment, so
    the inverse crosses `1 / slope` per unit -- and holding that slope at
    `_OUTERMOST_SHARE` bounds the answer, whatever the knot count.
    """
    data = _skewed(n=259, size=2, seed=9)
    spline = wp.Spline(2, knots_per_arm=knots_per_arm)
    spline.initialize(_f64(data))

    outside = np.linspace(-9, 9, 37)
    back = spline.backward(_f64(outside[:, None] * np.ones((1, 2)))).numpy()

    assert np.all(np.isfinite(back))
    assert np.all(np.diff(back, axis=0) > 0)
    bound = 5.0 + np.maximum(np.abs(outside) - 5.0, 0.0) / wp._OUTERMOST_SHARE
    assert np.all(np.abs(back) <= bound[:, None] * (1 + 1e-9))


def test_a_repeated_value_does_not_break_the_partition():
    """A mass point ties the empirical CDF at several knots. Inside the
    arm that is allowed -- `backward` clips into the bracket it located, so
    a flat segment there costs accuracy over one interval and cannot leave
    it -- but every share must stay strictly positive, the partitions being
    compositional."""
    rng = np.random.default_rng(10)
    column = np.concatenate([rng.normal(size=200), np.zeros(300)])
    data = np.stack([column, rng.normal(size=500)], axis=1)

    spline = wp.Spline(2, knots_per_arm=5)
    spline.initialize(_f64(data))

    coordinates = spline._get_warped_coordinates(0).numpy()
    assert np.all(np.diff(coordinates) > 0)
    assert np.all(np.isfinite(spline.forward(_f64(data))[1].numpy()))


def test_a_chain_ending_in_an_exponential_stays_finite():
    """The reported failure in miniature: `Log` at the bottom of the chain
    means `backward` ends in an exponential, so a spline inverse that leaves
    the scale does not merely read oddly -- it overflows to infinity, and
    the quantiles taken over the simulations come back NaN."""
    rng = np.random.default_rng(11)
    values = np.exp(rng.normal(size=(259, 4)) * 0.8) * np.array([1.0, 10.0, 0.1, 3.0])

    chain = wp.ChainedWarping(wp.Log(4), wp.RobustPCA(4, 4),
                              wp.Spline(4, knots_per_arm=5), wp.ZScore(4))
    chain.initialize(_f64(values))

    latent = _f64(rng.normal(size=(5000, 4)) * 1.5)
    assert np.all(np.isfinite(chain.backward(latent).numpy()))


# --------------------------------------------------------------------------- #
# the reason it was written: a chain initializes as a projection-pursuit sweep
# --------------------------------------------------------------------------- #
def test_a_chain_gaussianizes_between_its_rotations():
    """`ChainedWarping.initialize` threads the data link by link, so with a
    `Spline` that fits, `Rotation -> ZScore -> Spline` repeats become the
    greedy sweep of a projection-pursuit transform: each rotation runs its
    ICA on data the previous spline has already gaussianized.
    """
    rng = np.random.default_rng(6)
    source = np.stack([rng.exponential(size=500),
                       rng.uniform(size=500) ** 3,
                       rng.normal(size=500)], axis=1)
    mixed = source @ rng.normal(size=(3, 3))

    chain = wp.ChainedWarping(
        wp.ZScore(3),
        wp.Rotation(3), wp.ZScore(3), wp.Spline(3, knots_per_arm=10),
        wp.Rotation(3), wp.ZScore(3), wp.Spline(3, knots_per_arm=10))
    out = chain.initialize(_f64(mixed))

    assert _projection_index(out) < _projection_index(mixed) / 4
    assert np.all(np.isfinite(np.asarray(out)))
    # the second rotation had something left to find, which is only true
    # because a spline gaussianized before it
    second = chain.warpings[4].parameters["rotation"].get_value().numpy()
    assert np.mean(np.max(np.abs(second), axis=1)) < 0.99
