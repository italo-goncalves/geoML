"""The parametric links: `BoxCox`, `YeoJohnson`, `Arcsinh`, `SinhArcsinh`.

Each is elementwise, closed form both ways, and starts its parameters at
the values that make a column most Gaussian (Box and Cox's profile
likelihood, on declustering weights where given). Pinned here: the round
trips, the special values every family contains (the logarithm, the
identity), the reported log-Jacobian against finite differences, the
start reducing skewness and landing where the data say, the declustered
start, a draw past a bounded support coming back finite, gradients
reaching the parameters, and a model that trains, saves and reloads with
them. The declarations against a numerical Jacobian live in
`test_noise_integration.py`.
"""
import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

import geoml
import geoml.persistence as persistence
import geoml.warping as wp


def _lognormal(n=400, k=2, seed=0):
    return np.random.default_rng(seed).lognormal(0.5, 0.8, size=[n, k])


def _skewed_centred(n=400, k=2, seed=1):
    x = np.random.default_rng(seed).gamma(2.0, size=[n, k])
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _set(warping, **values):
    for name, value in values.items():
        warping.parameters[name].set_value(np.full([warping.size_in], value))
    return warping


def _round_trip(warping, x, atol=1e-9):
    y, _ = warping.forward(tf.constant(x))
    back = np.asarray(warping.backward(y))
    assert np.allclose(back, x, atol=atol, rtol=1e-9)
    again, _ = warping.forward(tf.constant(back))
    assert np.allclose(np.asarray(again), np.asarray(y), atol=atol, rtol=1e-9)


def _numerical_log_det(warping, x, step=1e-6):
    """The log of the product of the elementwise derivatives."""
    total = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        bump = np.zeros_like(x)
        bump[:, i] = step
        up, _ = warping.forward(tf.constant(x + bump))
        down, _ = warping.forward(tf.constant(x - bump))
        total += np.log((np.asarray(up)[:, i] - np.asarray(down)[:, i]) / (2 * step))
    return total


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exponent", [0.0, 1e-10, 0.5, 1.0, 1.7, 2.0])
def test_box_cox_round_trips_at_every_exponent(exponent):
    _round_trip(_set(wp.BoxCox(2), exponent=exponent), _lognormal())


@pytest.mark.parametrize("exponent", [0.0, 0.3, 1.0, 1.7, 2.0])
def test_yeo_johnson_round_trips_at_every_exponent(exponent):
    _round_trip(_set(wp.YeoJohnson(2), exponent=exponent), _skewed_centred())


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_arcsinh_round_trips(scale):
    _round_trip(_set(wp.Arcsinh(2), scale=scale), _skewed_centred())


@pytest.mark.parametrize("skewness,tailweight", [(0.0, 1.0), (-1.5, 0.5), (1.0, 2.5)])
def test_sinh_arcsinh_round_trips(skewness, tailweight):
    _round_trip(_set(wp.SinhArcsinh(2), skewness=skewness, tailweight=tailweight),
                _skewed_centred())


def test_the_families_contain_the_classical_maps():
    x = _lognormal()
    # Box-Cox at zero is the logarithm
    log_y, log_det = _set(wp.BoxCox(2, shift=1e-6), exponent=0.0).forward(tf.constant(x))
    ref, ref_det = wp.Log(2, shift=1e-6).forward(tf.constant(x))
    assert np.allclose(np.asarray(log_y), np.asarray(ref), atol=1e-12)
    assert np.allclose(np.asarray(log_det), np.asarray(ref_det), atol=1e-12)
    # Yeo-Johnson at one is the identity, and so is sinh-arcsinh at (0, 1)
    z = _skewed_centred()
    y, det = _set(wp.YeoJohnson(2), exponent=1.0).forward(tf.constant(z))
    assert np.allclose(np.asarray(y), z, atol=1e-12) and np.allclose(det, 0.0, atol=1e-12)
    y, det = wp.SinhArcsinh(2).forward(tf.constant(z))
    assert np.allclose(np.asarray(y), z, atol=1e-12) and np.allclose(det, 0.0, atol=1e-12)
    # arcsinh is linear well inside its scale
    small = z * 1e-3
    y, _ = _set(wp.Arcsinh(2), scale=1.0).forward(tf.constant(small))
    assert np.allclose(np.asarray(y), small, rtol=1e-5)


@pytest.mark.parametrize("warping,data", [
    (_set(wp.BoxCox(2), exponent=0.4), _lognormal()),
    (_set(wp.YeoJohnson(2), exponent=0.3), _skewed_centred()),
    (_set(wp.Arcsinh(2), scale=0.7), _skewed_centred()),
    (_set(wp.SinhArcsinh(2), skewness=0.8, tailweight=0.6), _skewed_centred()),
])
def test_the_log_determinant_matches_finite_differences(warping, data):
    _, log_det = warping.forward(tf.constant(data))
    assert np.allclose(np.asarray(log_det), _numerical_log_det(warping, data),
                       atol=1e-6)


def _heavy_tailed(n=400, k=2, seed=2):
    return np.random.default_rng(seed).standard_t(3.0, size=[n, k])


@pytest.mark.parametrize("factory,data,statistic", [
    (lambda: wp.BoxCox(2), _lognormal(), stats.skew),
    (lambda: wp.YeoJohnson(2), _lognormal(), stats.skew),
    # an odd map cannot unskew; what arcsinh tames is the tails
    (lambda: wp.Arcsinh(2), _heavy_tailed(), stats.kurtosis),
    (lambda: wp.SinhArcsinh(2), _skewed_centred(), stats.skew),
    (lambda: wp.SinhArcsinh(2), _heavy_tailed(), stats.kurtosis),
])
def test_the_start_makes_the_column_more_gaussian(factory, data, statistic):
    warping = factory()
    before = np.abs(statistic(data, axis=0))
    warped = np.asarray(warping.initialize(data))
    after = np.abs(statistic(warped, axis=0))
    assert np.all(after < 0.5 * before)


def test_the_start_lands_where_the_data_say():
    # lognormal data: the Gaussianizing power is the logarithm
    warping = wp.BoxCox(2)
    warping.initialize(_lognormal())
    assert np.all(np.abs(np.asarray(warping.parameters["exponent"].get_value())) < 0.3)
    # data that are already Gaussian leave every family near its identity
    gaussian = np.random.default_rng(3).normal(size=[600, 2])
    warping = wp.YeoJohnson(2)
    warping.initialize(gaussian)
    assert np.all(np.abs(np.asarray(warping.parameters["exponent"].get_value()) - 1.0) < 0.3)
    warping = wp.SinhArcsinh(2)
    warping.initialize(gaussian)
    assert np.all(np.abs(np.asarray(warping.parameters["skewness"].get_value())) < 0.3)
    assert np.all(np.abs(np.log(np.asarray(warping.parameters["tailweight"].get_value()))) < 0.3)


def test_the_start_takes_declustering_weights():
    """Unit weights are the plain start; halving the weight of duplicated
    rows is the start on the unduplicated data."""
    data = _lognormal(n=200)
    plain, weighted = wp.BoxCox(2), wp.BoxCox(2)
    plain.initialize(data)
    weighted.initialize(data, weights=np.ones(200))
    assert np.allclose(np.asarray(plain.parameters["exponent"].get_value()),
                       np.asarray(weighted.parameters["exponent"].get_value()))

    doubled = np.concatenate([data, data[:100]], axis=0)
    weights = np.concatenate([np.full(100, 0.5), np.ones(100), np.full(100, 0.5)])
    declustered = wp.BoxCox(2)
    declustered.initialize(doubled, weights=weights)
    assert np.allclose(np.asarray(plain.parameters["exponent"].get_value()),
                       np.asarray(declustered.parameters["exponent"].get_value()))


def test_a_draw_past_a_bounded_support_comes_back_finite():
    far = np.array([[-50.0, 50.0], [-5.0, 5.0], [0.0, 0.0]])
    for warping in (_set(wp.BoxCox(2), exponent=0.5),
                    _set(wp.BoxCox(2), exponent=1.7),
                    _set(wp.YeoJohnson(2), exponent=0.3),
                    _set(wp.YeoJohnson(2), exponent=1.8)):
        back = np.asarray(warping.backward(tf.constant(far)))
        assert np.all(np.isfinite(back))
    # a Box-Cox draw below the bound lands on the floor: a zero grade
    floor = np.asarray(_set(wp.BoxCox(2), exponent=0.5).backward(tf.constant(far)))
    assert np.allclose(floor[0, 0], -1e-6, atol=1e-9)
    # and an exponent outside [0, 2] is pulled back by the parameter's limits
    exponent = wp.BoxCox(2).parameters["exponent"]
    exponent.set_value(np.full(2, -1.0))
    exponent.refresh()
    assert np.all(np.asarray(exponent.get_value()) >= 0.0)
    # and the exponential comparison: the logarithm's inverse at 50 is 1e21
    assert np.asarray(wp.Log(2).backward(tf.constant(far)))[0, 1] > 1e20


@pytest.mark.parametrize("warping,data,name", [
    (wp.BoxCox(2), _lognormal(), "exponent"),
    (wp.YeoJohnson(2), _skewed_centred(), "exponent"),
    (wp.Arcsinh(2), _skewed_centred(), "scale"),
    (wp.SinhArcsinh(2), _skewed_centred(), "tailweight"),
])
def test_gradients_reach_the_parameters(warping, data, name):
    variable = warping.parameters[name].variable
    with tf.GradientTape() as tape:
        y, log_det = warping.forward(tf.constant(data))
        loss = tf.reduce_sum(y ** 2) + tf.reduce_sum(log_det)
    grad = tape.gradient(loss, variable)
    assert grad is not None and np.all(np.isfinite(grad.numpy()))
    assert np.any(np.asarray(grad) != 0.0)


def test_a_model_trains_saves_and_reloads_with_the_links(tmp_path):
    rng = np.random.default_rng(5)
    coords = rng.uniform(0.0, 100.0, (60, 2))
    values = np.exp(np.sin(coords[:, 0] / 20.0) + 0.3 * rng.normal(size=60))
    point = geoml.data.PointData.from_array(coords, ["x", "y"])
    point.add_continuous_variable("v", values)
    inducing = geoml.data.PointData.from_array(rng.uniform(0.0, 100.0, (12, 2)), ["x", "y"])

    geoml.set_seed(1234)
    chain = wp.ChainedWarping(wp.BoxCox(1), wp.ZScore(1), wp.SinhArcsinh(1), wp.ZScore(1))
    gp = geoml.latent.BasicGP(
        geoml.latent.BasicInput(inducing, transform=geoml.transform.Isotropic(40.0)),
        size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(chain), gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=8))
    model.train_full(max_iter=5)
    assert np.isfinite(model.training_log).all()
    assert not chain.warpings[0].parameters["exponent"].fixed

    query = geoml.data.PointData.from_array(rng.uniform(0.0, 100.0, (15, 2)), ["x", "y"])
    model.predict(query, n_sim=4)
    before = np.asarray(query.values("v/prediction"), dtype=float).copy()
    assert np.isfinite(before).all()

    persistence.save_model(model, tmp_path / "model")
    restored = persistence.load_model(tmp_path / "model")
    restored.predict(query, n_sim=4)
    assert np.allclose(before, np.asarray(query.values("v/prediction")))


def test_every_link_is_elementwise():
    for warping in (wp.BoxCox(3), wp.YeoJohnson(3), wp.Arcsinh(3), wp.SinhArcsinh(3)):
        assert warping.elementwise
