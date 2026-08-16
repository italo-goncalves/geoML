"""Directional covariances -- the derivatives directional data are built on.

`covariance_matrix_d1` is the covariance between a point and a *direction* at
another point, `covariance_matrix_d2` between two directions, and both used
to be central differences over a step of 1e-3. They are now exact: the
kernel, a one-dimensional elementwise function, is differentiated by
autodiff, and the spatial part is carried by the chain rule

    K = f(r),  r = |h|,  h_ij = T(x)_i - T(y)_j
    dK/dy . u  = -phi A,   d2K/dx dy = -(psi A B + phi C)

with `phi = f'(r)/r`, `psi = (f''(r) - phi)/r**2`, and A, B, C the pieces
that carry the directions. The point of doing it this way rather than
differentiating the covariance matrix itself is the origin: `|h|` is not
twice differentiable there, and every d2 call has coincident points on its
diagonal.

The closed forms below are written out by hand from each `kernelize`, so
they check the implementation against the mathematics rather than against
another run of the same code.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.kernels as kr
import geoml.transform as tr


def _f64(value):
    return tf.constant(np.asarray(value, dtype=float), tf.float64)


# f'(r) and f''(r) for each kernel, and -f''(0), which is the variance of the
# field's derivative along a unit direction when the range is 1
DERIVATIVES = {
    "Cubic": (lambda r: -14*r + 105/4*r**2 - 35/2*r**4 + 21/4*r**6,
              lambda r: -14 + 105/2*r - 70*r**3 + 63/2*r**5,
              14.0),
    "Gaussian": (lambda r: -6*r*np.exp(-3*r**2),
                 lambda r: (36*r**2 - 6)*np.exp(-3*r**2),
                 6.0),
    "Matern32": (lambda r: -25*r*np.exp(-5*r),
                 lambda r: (-25 + 125*r)*np.exp(-5*r),
                 25.0),
    "Matern52": (lambda r: -(12*r + 72*r**2)*np.exp(-6*r),
                 lambda r: (-12 - 72*r + 432*r**2)*np.exp(-6*r),
                 12.0),
    "Cosine": (lambda r: -2*np.pi*np.sin(2*np.pi*r),
               lambda r: -(2*np.pi)**2*np.cos(2*np.pi*r),
               (2*np.pi)**2),
}


def _closed_form(x, y, dir_x, dir_y, kernel_name, ranges):
    """The two directional covariances, written out for an isotropic or
    per-axis scaling `T(x) = x / ranges`."""
    first, second, _ = DERIVATIVES[kernel_name]
    ranges = np.asarray(ranges, dtype=float)
    h = (x[:, None, :] - y[None, :, :]) / ranges
    dist = np.linalg.norm(h, axis=2)
    # directions pushed through the (diagonal) Jacobian of the transform
    u = dir_y / ranges
    v = dir_x / ranges
    hu = np.einsum("ijk,jk->ij", h, u)
    hv = np.einsum("ijk,ik->ij", h, v)
    uv = np.einsum("jk,ik->ij", u, v)
    d1 = -first(dist) * hu / dist
    d2 = -(second(dist) * hu * hv / dist**2
           + first(dist) / dist * (uv - hu * hv / dist**2))
    return d1, d2


def _numeric_derivatives(covariance, x, y, dir_x, dir_y, step=0.1):
    """Central differences on a covariance, Richardson-extrapolated.

    Finite differences belong in a test, as the reference — this one knows
    nothing about how the answer is assembled, which is what makes it worth
    comparing against. Good for about eight digits.
    """
    def value(a, b):
        return covariance.covariance_matrix(_f64(a), _f64(b)).numpy()

    def first(h):
        return (value(x, y + 0.5 * h * dir_y)
                - value(x, y - 0.5 * h * dir_y)) / h

    def second(h):
        return (value(x + 0.5*h*dir_x, y + 0.5*h*dir_y)
                - value(x + 0.5*h*dir_x, y - 0.5*h*dir_y)
                - value(x - 0.5*h*dir_x, y + 0.5*h*dir_y)
                + value(x - 0.5*h*dir_x, y - 0.5*h*dir_y)) / h ** 2

    return ((4 * first(step / 2) - first(step)) / 3,
            (4 * second(step / 2) - second(step)) / 3)


def _sample(n_x=7, n_y=5, dim=2, spread=30.0, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, spread, size=(n_x, dim)) + offset
    y = rng.uniform(0, spread, size=(n_y, dim)) + offset
    dir_x = rng.normal(size=(n_x, dim))
    dir_y = rng.normal(size=(n_y, dim))
    dir_x /= np.linalg.norm(dir_x, axis=1, keepdims=True)
    dir_y /= np.linalg.norm(dir_y, axis=1, keepdims=True)
    return x, y, dir_x, dir_y


# --------------------------------------------------------------------------- #
# exactness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(DERIVATIVES))
def test_the_directional_covariances_match_the_closed_form(name):
    x, y, dir_x, dir_y = _sample()
    covariance = kr.Covariance(getattr(kr, name)(), tr.Isotropic(50.0))
    want_d1, want_d2 = _closed_form(x, y, dir_x, dir_y, name, 50.0)

    got_d1 = covariance.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y))
    got_d2 = covariance.covariance_matrix_d2(
        _f64(x), _f64(y), _f64(dir_x), _f64(dir_y))

    assert np.allclose(got_d1, want_d1, rtol=1e-12, atol=1e-14)
    assert np.allclose(got_d2, want_d2, rtol=1e-12, atol=1e-14)


def test_a_transform_carries_the_direction_through_its_jacobian():
    """A direction is given in the untransformed space, so it reaches the
    kernel through the transform's Jacobian -- an anisotropy stretches it,
    and the two axes must be stretched differently for this to be a test."""
    x, y, dir_x, dir_y = _sample(dim=2, seed=3)
    ranges = np.array([40.0, 12.0])
    covariance = kr.Covariance(kr.Gaussian(), tr.AnisotropyARD(2))
    covariance.transform.parameters["ranges"].set_value(ranges)
    want_d1, want_d2 = _closed_form(x, y, dir_x, dir_y, "Gaussian", ranges)

    assert np.allclose(
        covariance.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)),
        want_d1, rtol=1e-12, atol=1e-14)
    assert np.allclose(
        covariance.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)),
        want_d2, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name", sorted(DERIVATIVES))
@pytest.mark.parametrize("offset", [0.0, 557000.0])
def test_a_point_against_itself_gives_the_derivative_variance(name, offset):
    """The diagonal of the self-covariance is `-f''(0)` over the range
    squared -- a limit the old central differences could only approach, and
    the reason the distance is snapped to zero where it is indistinguishable
    from it. The offset repeats the check at mine-grid coordinates, where
    the squared distance of a point from itself is a cancellation."""
    x, _, dir_x, _ = _sample(offset=offset, seed=5)
    covariance = kr.Covariance(getattr(kr, name)(), tr.Isotropic(50.0))
    want = DERIVATIVES[name][2] / 50.0 ** 2

    diagonal = np.diag(covariance.self_covariance_matrix_d2(
        _f64(x), _f64(dir_x)).numpy())
    assert np.allclose(diagonal, want, rtol=1e-12)
    # the cheap route to the same numbers
    assert np.allclose(covariance.point_variance_d2(_f64(x), _f64(dir_x)),
                       want, rtol=1e-12)


def test_the_derivative_block_is_positive_definite_at_a_long_range():
    """What the exactness is for. `GradientConstrainedInput` factorizes the
    assembled covariance with no jitter at all, so the directional block has
    to be positive definite on its own merits. Central differences divide by
    a step of 1e-3, which magnifies rounding by a million while the answer
    itself shrinks as 1/range**2; past a long enough range the block goes
    indefinite and the Cholesky returns NaN. Measured: at range 5000 over a
    100-unit domain the old code failed here and this passes."""
    rng = np.random.default_rng(11)
    locations = rng.uniform(0, 100, size=(40, 2))
    directions = rng.normal(size=(40, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    grid = geoml.data.Grid2D(start=[0, 0], n=[21, 21], end=[100, 100])

    covariance = kr.Covariance(kr.Cubic(), tr.Isotropic(5000.0))
    base = covariance.self_covariance_matrix(_f64(grid.coordinates))
    cross = covariance.covariance_matrix_d1(
        _f64(grid.coordinates), _f64(locations), _f64(directions))
    block = covariance.self_covariance_matrix_d2(
        _f64(locations), _f64(directions))
    full = tf.concat([
        tf.concat([block, tf.transpose(cross)], axis=1),
        tf.concat([cross, base], axis=1)], axis=0)
    scale = tf.sqrt(tf.linalg.diag_part(full))
    full = full / scale[:, None] / scale[None, :]

    assert np.all(np.isfinite(tf.linalg.cholesky(full).numpy()))
    symmetric = (full + tf.transpose(full)) / 2
    assert np.min(np.linalg.eigvalsh(symmetric.numpy())) > 0


# --------------------------------------------------------------------------- #
# the pieces around the covariance
# --------------------------------------------------------------------------- #
def test_scaling_reaches_every_derivative():
    x, y, dir_x, dir_y = _sample(seed=7)
    plain = kr.Covariance(kr.Gaussian(), tr.Isotropic(50.0))
    scaled = kr.Scale(kr.Covariance(kr.Gaussian(), tr.Isotropic(50.0)))
    scaled.parameters["amplitude"].set_value(2.5)

    assert np.allclose(
        scaled.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)),
        2.5 * plain.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)))
    assert np.allclose(
        scaled.covariance_matrix_d2(_f64(x), _f64(y), _f64(dir_x), _f64(dir_y)),
        2.5 * plain.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)))
    assert np.allclose(
        scaled.point_variance_d2(_f64(x), _f64(dir_x)),
        2.5 * plain.point_variance_d2(_f64(x), _f64(dir_x)))


def test_a_linear_covariance_differentiates_to_the_other_point():
    """`Linear` has no kernel and no distance, so it takes the base class's
    generic route -- forward-mode autodiff over the covariance matrix. For
    `K(x, y) = x.y` the derivative along `u` at `y` is just `x.u`, which is
    also the case the old central differences got wrong: they measured the
    step about a shifted origin, which is only harmless when the covariance
    depends on differences alone."""
    x, y, dir_x, dir_y = _sample(seed=13)
    covariance = kr.Linear()

    assert np.allclose(
        covariance.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)),
        x @ dir_y.T)
    assert np.allclose(
        covariance.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)),
        dir_x @ dir_y.T)
    assert np.allclose(covariance.point_variance_d2(_f64(x), _f64(dir_x)),
                       np.sum(dir_x * dir_x, axis=1))


def test_a_sum_adds_its_components_derivatives():
    x, y, dir_x, dir_y = _sample(seed=17)
    one = kr.Covariance(kr.Gaussian(), tr.Isotropic(30.0))
    two = kr.Covariance(kr.Cubic(), tr.Isotropic(70.0))
    total = kr.Sum(kr.Covariance(kr.Gaussian(), tr.Isotropic(30.0)),
                   kr.Covariance(kr.Cubic(), tr.Isotropic(70.0)))
    weights = total.parameters["variance"].get_value().numpy()

    want = (weights[0] * one.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y))
            + weights[1] * two.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)))
    assert np.allclose(
        total.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)), want)


def test_a_product_of_gaussians_is_the_gaussian_it_equals():
    """A product of Gaussian kernels is itself Gaussian, with
    `1/c**2 = 1/a**2 + 1/b**2` — so the product's derivatives have an exact
    reference in the single-covariance path checked above. Applying the
    operation to the components' derivatives, as a sum may, gives the
    product of the derivatives instead of the product rule: measured 98%
    wrong on this very case."""
    x, y, dir_x, dir_y = _sample(seed=37)
    a, b = 60.0, 35.0
    product = kr.Product(kr.Covariance(kr.Gaussian(), tr.Isotropic(a)),
                         kr.Covariance(kr.Gaussian(), tr.Isotropic(b)))
    single = kr.Covariance(kr.Gaussian(),
                           tr.Isotropic(1 / np.sqrt(1/a**2 + 1/b**2)))

    assert np.allclose(product.covariance_matrix(_f64(x), _f64(y)),
                       single.covariance_matrix(_f64(x), _f64(y)))
    assert np.allclose(
        product.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)),
        single.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)))
    assert np.allclose(
        product.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)),
        single.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)))
    assert np.allclose(
        product.self_covariance_matrix_d2(_f64(x), _f64(dir_x)),
        single.self_covariance_matrix_d2(_f64(x), _f64(dir_x)))
    assert np.allclose(product.point_variance_d2(_f64(x), _f64(dir_x)),
                       single.point_variance_d2(_f64(x), _f64(dir_x)))


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_a_product_follows_the_product_rule(n_components):
    """Against central differences on the product covariance itself, which
    know nothing about any product rule. The derivative in the first
    argument that the rule needs is not in the interface and does not have
    to be: a covariance is symmetric, so it is the derivative in the second
    with the arguments swapped and the answer transposed."""
    x, y, dir_x, dir_y = _sample(seed=41)
    parts = [kr.Covariance(kr.Cubic(), tr.Isotropic(60.0)),
             kr.Covariance(kr.Gaussian(), tr.Isotropic(35.0)),
             kr.Covariance(kr.Matern52(), tr.Isotropic(80.0))]
    product = kr.Product(*parts[:n_components])
    want_d1, want_d2 = _numeric_derivatives(product, x, y, dir_x, dir_y)

    assert np.allclose(
        product.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)),
        want_d1, rtol=1e-6, atol=1e-9)
    assert np.allclose(
        product.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)),
        want_d2, rtol=1e-6, atol=1e-9)


def test_a_product_carries_each_components_own_anisotropy():
    """Two differently oriented ellipsoids: the transpose identity has to
    hold under each component's own Jacobian, not a shared one."""
    x, y, dir_x, dir_y = _sample(seed=43)
    product = kr.Product(
        kr.Covariance(kr.Gaussian(), tr.Anisotropy2D(60.0, 0.4, 30.0)),
        kr.Covariance(kr.Cubic(), tr.Anisotropy2D(40.0, 0.7, -20.0)))
    want_d1, want_d2 = _numeric_derivatives(product, x, y, dir_x, dir_y)

    assert np.allclose(
        product.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)),
        want_d1, rtol=1e-6, atol=1e-9)
    assert np.allclose(
        product.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)),
        want_d2, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# what training and prediction need of them
# --------------------------------------------------------------------------- #
def test_the_training_gradient_reaches_the_kernel_parameters():
    """The derivatives are themselves built out of derivatives, so training
    differentiates a second time -- through a nested `GradientTape` over the
    kernel and a `ForwardAccumulator` over the transform. Checked against a
    finite difference of the loss, which is where finite differences belong."""
    x, y, dir_x, dir_y = _sample(seed=19)
    covariance = kr.Covariance(kr.Cubic(), tr.Isotropic(50.0))
    variable = covariance.transform.parameters["range"].variable

    def loss():
        return tf.reduce_sum(covariance.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)) ** 2)

    with tf.GradientTape() as tape:
        value = loss()
    gradient = float(tape.gradient(value, variable).numpy().ravel()[0])

    start = variable.numpy().ravel()[0]
    step = 1e-6
    variable.assign(tf.reshape(_f64(start + step), variable.shape))
    up = float(loss().numpy())
    variable.assign(tf.reshape(_f64(start - step), variable.shape))
    down = float(loss().numpy())
    variable.assign(tf.reshape(_f64(start), variable.shape))

    assert gradient == pytest.approx((up - down) / (2 * step), rel=1e-5)
    assert abs(gradient) > 1e-8              # and it is not trivially zero


def test_the_derivatives_survive_tracing_and_a_change_of_shape():
    """Prediction batches vary in size, so these are traced once and called
    with whatever comes. (Nesting two forward accumulators through
    `math.tf.pairwise_dist` -- a `tf.function` with a shape-polymorphic
    signature -- reuses one shape's graph for another and fails; the
    implementation avoids nesting for exactly this reason.)"""
    covariance = kr.Covariance(kr.Cubic(), tr.Isotropic(50.0))

    @tf.function
    def traced(x, y, dir_x, dir_y):
        return covariance.covariance_matrix_d2(x, y, dir_x, dir_y)

    for n_y in (5, 9, 5):
        x, y, dir_x, dir_y = _sample(n_y=n_y, seed=n_y)
        got = traced(_f64(x), _f64(y), _f64(dir_x), _f64(dir_y)).numpy()
        assert got.shape == (7, n_y)
        assert np.all(np.isfinite(got))


def test_one_direction_serves_every_point():
    """`StructuralField.predict_raw` asks for the covariance along a single
    mean vector at every prediction point, and passes it as one row. The
    arithmetic this replaced broadcast it; an accumulator wants one tangent
    per row, so it has to be spelled out."""
    x, y, _, dir_y = _sample(n_x=6, n_y=4, seed=29)
    covariance = kr.Covariance(kr.Gaussian(), tr.Isotropic(50.0))
    one = np.array([[0.6, 0.8]])

    got = covariance.covariance_matrix_d2(
        _f64(x), _f64(y), _f64(one), _f64(dir_y))
    want = covariance.covariance_matrix_d2(
        _f64(x), _f64(y), _f64(np.tile(one, [6, 1])), _f64(dir_y))
    assert got.shape == (6, 4)
    assert np.allclose(got, want)


def test_a_constant_kernel_has_no_derivative():
    """`Constant.kernelize` ignores the distance, so the tape returns no
    gradient at all rather than a zero one -- a `None` the implementation
    has to read as zero."""
    x, y, dir_x, dir_y = _sample(seed=23)
    covariance = kr.Covariance(kr.Constant(), tr.Isotropic(50.0))

    assert np.allclose(
        covariance.covariance_matrix_d1(_f64(x), _f64(y), _f64(dir_y)), 0.0)
    assert np.allclose(
        covariance.covariance_matrix_d2(
            _f64(x), _f64(y), _f64(dir_x), _f64(dir_y)), 0.0)
