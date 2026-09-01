"""The rational-quadratic backbone of the `Spline` warping.

Same knots, same Steffen slopes, same linear extrapolation as the
monotonic cubic -- so the two agree exactly outside the knots and at them
-- with a closed-form inverse between them. The tests pin what the
drop-in promises: the knots interpolated, monotonicity, the reported
derivative, agreement with the cubic wherever both are linear, a round
trip at machine precision (the cubic's is bounded by its iteration), and
that the default backbone is untouched.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.math.interpolate as interp
import geoml.warping as wp


def _knots(seed=0, n=11, columns=3):
    """Monotone knots shaped like a normal-score fit: some segments
    nearly flat, some steep."""
    rng = np.random.default_rng(seed)
    x = np.tile(np.linspace(-5, 5, n)[:, None], [1, columns])
    steps = rng.lognormal(mean=0.0, sigma=1.5, size=[n - 1, columns])
    steps[1, :] = 1e-4  # a nearly flat interior segment
    y = np.concatenate([np.zeros([1, columns]), np.cumsum(steps, axis=0)])
    y = -5 + 10 * y / y[-1:]
    return tf.constant(x), tf.constant(y)


def test_the_knots_are_interpolated_and_the_map_is_monotone():
    x, y = _knots()
    spline = interp.MonotonicRationalQuadraticSpline()

    at_knots = spline.interpolate(x, y, x)
    assert np.allclose(np.asarray(at_knots), np.asarray(y), atol=1e-12)

    dense = tf.constant(np.tile(np.linspace(-7, 7, 2001)[:, None], [1, 3]))
    values = np.asarray(spline.interpolate(x, y, dense))
    assert np.all(np.diff(values, axis=0) >= -1e-12)


def test_the_reported_derivative_matches_differences():
    """Inside the knots, against central differences at the tolerance
    `test_noise_integration` holds every warping to. Outside them the
    comparison is exact instead: the padded extrapolation segment spans
    1e6 units, so a value there carries ~1e-10 of cancellation -- nothing
    for the value, but a central difference divides it by the step and
    reads it as 1e-4 in the derivative. What extrapolation actually
    promises is the end slope, so that is what is asserted there."""
    x, y = _knots(seed=1)
    spline = interp.MonotonicRationalQuadraticSpline()
    inside = tf.constant(np.random.default_rng(2).uniform(-4.9, 4.9, [200, 3]))

    reported = np.asarray(spline.interpolate(x, y, inside, grad=True))
    step = 1e-6
    plus = np.asarray(spline.interpolate(x, y, inside + step))
    minus = np.asarray(spline.interpolate(x, y, inside - step))
    assert np.allclose(reported, (plus - minus) / (2 * step),
                       rtol=1e-4, atol=1e-6)

    outside = tf.constant(np.tile(np.array([-8.0, -5.5, 5.5, 8.0])[:, None],
                                  [1, 3]))
    slopes = np.asarray(spline._get_derivative(x, y))
    expected = np.stack([slopes[0], slopes[0], slopes[-1], slopes[-1]])
    assert np.allclose(np.asarray(spline.interpolate(x, y, outside, grad=True)),
                       expected)


def test_it_agrees_with_the_cubic_where_both_are_linear():
    """Beyond the knots both extrapolate along the end slopes, so the two
    backbones coincide there -- and at the knots themselves."""
    x, y = _knots(seed=3)
    cubic = interp.MonotonicCubicSpline()
    rational = interp.MonotonicRationalQuadraticSpline()
    outside = tf.constant(np.tile(np.array([-9.0, -6.5, 6.5, 9.0])[:, None],
                                  [1, 3]))

    assert np.allclose(np.asarray(cubic.interpolate(x, y, outside)),
                       np.asarray(rational.interpolate(x, y, outside)))
    assert np.allclose(np.asarray(cubic.interpolate(x, y, x)),
                       np.asarray(rational.interpolate(x, y, x)))


def test_the_round_trip_is_at_machine_precision():
    x, y = _knots(seed=4)
    spline = interp.MonotonicRationalQuadraticSpline()
    query = tf.constant(np.random.default_rng(5).uniform(-6, 6, [500, 3]))

    forward = spline.interpolate(x, y, query)
    back = np.asarray(spline.invert(x, y, forward))

    assert np.max(np.abs(back - np.asarray(query))) < 1e-9

    # and the other way round, which is the direction the likelihood
    # takes: a latent value in, its data value out, and back
    latent = tf.constant(np.random.default_rng(6).uniform(-4.5, 4.5, [500, 3]))
    inverted = spline.invert(x, y, latent)
    assert np.max(np.abs(np.asarray(spline.interpolate(x, y, inverted))
                         - np.asarray(latent))) < 1e-9


def test_the_warping_round_trips_and_keeps_its_default():
    geoml.set_seed(11)
    data = np.random.default_rng(11).lognormal(size=[300, 3])

    plain = wp.Spline(3)
    assert isinstance(plain.spline, interp.MonotonicCubicSpline)
    assert not isinstance(plain.spline, interp.MonotonicRationalQuadraticSpline)

    rational = wp.Spline(3, backbone="rq")
    rational.initialize(data)
    z = tf.constant(np.random.default_rng(12).normal(size=[200, 3]))
    warped, _ = rational.forward(z)
    back = np.asarray(rational.backward(warped))
    assert np.max(np.abs(back - np.asarray(z))) < 1e-9

    with pytest.raises(ValueError, match="backbone"):
        wp.Spline(3, backbone="quintic")
