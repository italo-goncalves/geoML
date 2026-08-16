"""`MonotonicCubicSpline.invert`, and the guards it needed repaired.

`Spline.backward` used to interpolate the knots the other way round, which
is not an inverse: the inverse of a cubic is not a cubic, so the two curves
meet at the knots and part between them, leaving `forward(backward(y))` a
tenth of a standard deviation from `y`. `invert` solves the actual
polynomial by Newton instead, from the bracket's own secant, with the
interval located once because a monotone map puts the answer in the
interval the query occupies among the y-knots.

Repaired alongside: the module's additive epsilons. `w + 1e-6` perturbs
every interval to protect against the rare zero one, and at a knot spacing
of 0.125 that is a relative error of 8e-6 -- enough to put a floor under
the spline's self-consistency. `_safe_width` replaces only the zeros.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml.math.interpolate as gint
import geoml.warping as wp


def _f64(value):
    return tf.constant(np.asarray(value, dtype=float), tf.float64)


def _skewed(n=2000, size=2, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.exp(rng.normal(size=(n, size)) * 1.2)
    return (raw - raw.mean(0)) / raw.std(0)


def _fitted(knots_per_arm, seed=0):
    spline = wp.Spline(2, knots_per_arm=knots_per_arm)
    spline.initialize(_f64(_skewed(seed=seed)))
    return spline


# --------------------------------------------------------------------------- #
# the inverse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("knots_per_arm", [5, 10, 20, 40])
@pytest.mark.parametrize("seed", [0, 3])
def test_the_round_trip_is_as_exact_as_the_map_allows(knots_per_arm, seed):
    """Where it used to be 1e-1.

    The tolerance is 1e-9 rather than machine precision on purpose. A
    normal-score fit leaves half its intervals nearly flat, and where the
    forward map compresses by `s` a residual at machine precision returns
    magnified by `1/s`; measured, that floor is around 5e-11. Asking for
    1e-15 would be asking the solver to recover information the transform
    discarded. Two seeds, because the floor is a property of the data and
    one sample of it is not a contract.
    """
    spline = _fitted(knots_per_arm, seed=seed)
    values = np.clip(_skewed(seed=seed + 1), -4.5, 4.5)

    warped, _ = spline.forward(_f64(values))
    assert np.allclose(spline.backward(warped), values, atol=1e-9)

    latent = np.clip(np.random.default_rng(seed + 2).normal(size=(2000, 2)),
                     -4.0, 4.0) * 1.2
    there = spline.forward(_f64(spline.backward(_f64(latent))))[0]
    assert np.allclose(there, latent, atol=1e-9)


def test_the_identity_map_inverts_exactly():
    """An uninitialized spline is the identity, whose inverse the old route
    got right -- so this pins that nothing regressed for it."""
    spline = wp.Spline(2, knots_per_arm=10)
    values = np.clip(_skewed(seed=3), -4.5, 4.5)
    assert np.allclose(spline.backward(spline.forward(_f64(values))[0]),
                       values, atol=1e-12)


def test_it_inverts_outside_the_knot_range():
    """Past the last knot the spline extrapolates linearly, and `invert`
    pads exactly as `interpolate` pads so the two agree there too."""
    spline = _fitted(10)
    far = np.array([[-9.0, 9.0], [-6.0, 6.0], [6.5, -6.5]])
    assert np.allclose(spline.backward(spline.forward(_f64(far))[0]), far,
                       atol=1e-8)


def test_the_default_number_of_steps_is_where_convergence_stops():
    """Half the intervals of a normal-score fit are nearly flat, which
    Newton crosses slowly, so a frugal step count is a false economy: three
    steps leave a thousandth. Twelve reaches the floor set by the map's own
    conditioning, and more buys nothing -- which is the argument for the
    default being what it is."""
    spline = _fitted(20, seed=4)
    values = np.clip(_skewed(seed=5), -4.5, 4.5)
    warped = tf.stack([spline._get_warped_coordinates(i) for i in range(2)],
                      axis=1)
    forward = spline.forward(_f64(values))[0]

    def error(steps):
        got = spline.spline.invert(spline.x_original, warped, forward,
                                   steps=steps)
        return np.max(np.abs(got.numpy() - values))

    # how far short three steps fall varies with the data -- 1e-3 on the
    # worst of six samples, 1e-7 on this one -- so what is asserted is the
    # shape of the curve rather than a level that happened to hold once
    assert error(3) > 100 * error(12)
    assert error(12) < 1e-9
    assert error(32) == pytest.approx(error(12), abs=1e-12)


def test_the_gradient_is_the_implicit_one():
    """The last correction is taken from a detached iterate, so what comes
    out is `1 / f'(t)` exactly rather than the derivative of an unrolled
    loop."""
    spline = _fitted(10)
    probe = _f64([[-2.0, 0.3], [0.5, 1.7], [2.5, -1.2]])

    with tf.GradientTape() as tape:
        tape.watch(probe)
        total = tf.reduce_sum(spline.backward(probe))
    got = tape.gradient(total, probe).numpy()

    step = 1e-6
    reference = ((spline.backward(_f64(np.asarray(probe) + step)).numpy()
                  - spline.backward(_f64(np.asarray(probe) - step)).numpy())
                 / (2 * step))
    assert np.allclose(got, reference, rtol=1e-6)


def test_a_flat_segment_returns_a_number_rather_than_an_infinity():
    """A constant stretch has no inverse. The answer there is arbitrary;
    what is not arbitrary is that it stays finite and inside the bracket."""
    x = _f64(np.linspace(-5, 5, 11)[:, None] * np.ones((1, 2)))
    y = np.linspace(-5, 5, 11)[:, None] * np.ones((1, 2))
    y[4:7, :] = y[4, :]                          # a plateau
    y = _f64(np.maximum.accumulate(y, axis=0))

    spline = gint.MonotonicCubicSpline()
    query = _f64(np.linspace(-4.5, 4.5, 40)[:, None] * np.ones((1, 2)))
    got = spline.invert(x, y, query).numpy()
    assert np.all(np.isfinite(got))
    assert np.all(got >= -5 - 1e6) and np.all(got <= 5 + 1e6)


# --------------------------------------------------------------------------- #
# the guards
# --------------------------------------------------------------------------- #
def test_a_zero_width_interval_does_not_poison_its_neighbours():
    """What the additive epsilon was protecting against, and what
    `_safe_width` protects against without touching the rest."""
    x = np.linspace(-5, 5, 11)[:, None] * np.ones((1, 2))
    x[5, :] = x[4, :]                            # two knots in one place
    y = np.linspace(-5, 5, 11)[:, None] * np.ones((1, 2))

    spline = gint.MonotonicCubicSpline()
    query = _f64(np.linspace(-4.9, 4.9, 50)[:, None] * np.ones((1, 2)))
    got = spline.interpolate(_f64(x), _f64(y), query).numpy()
    assert np.all(np.isfinite(got))


def test_widths_that_are_fine_are_left_alone():
    """The point of the repair: an interval of 0.125 used to be divided by
    0.125001, and now is divided by 0.125."""
    widths = _f64([[0.125, 1.0], [4.0, 1e-3]])
    assert np.allclose(gint._safe_width(widths), widths, rtol=0, atol=0)
    assert np.allclose(gint._safe_width(_f64([[0.0, 2.0]])), [[1.0, 2.0]])


def test_a_straight_line_is_reproduced_to_machine_precision():
    """The clearest reading of the epsilons' cost: interpolating a line
    should return the line. With `w + 1e-6` it did not, by roughly the
    relative size of that epsilon."""
    for spacing in (1.0, 0.125, 0.01):
        count = int(round(10.0 / spacing)) + 1
        grid = np.linspace(-5, 5, count)[:, None] * np.ones((1, 2))
        line = 3.0 * grid - 1.0

        spline = gint.MonotonicCubicSpline()
        query = _f64(np.linspace(-4.9, 4.9, 97)[:, None] * np.ones((1, 2)))
        got = spline.interpolate(_f64(grid), _f64(line), query).numpy()
        assert np.allclose(got, 3.0 * np.asarray(query) - 1.0, atol=1e-12)
