"""A mixture of whole likelihoods, each reading its own latent columns.

`LikelihoodMixture` mixes densities rather than values: a measurement comes
from one component, and a realization belongs to one component everywhere.
The gates on synthetic data and on Tom East are benchmarks
(`docs/benchmarks/gaussian_mixture_gates.py likelihood`,
`docs/benchmarks/tom_east_mixture.py`); these tests pin the plumbing.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.warping as wp


def _rows(n=30, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(n, 1))
    var = rng.uniform(0.1, 0.5, size=(n, 1))
    y = rng.normal(size=(n, 1))
    return (tf.constant(mu), tf.constant(var), tf.constant(y),
            tf.ones([n, 1], tf.float64))


def test_two_identical_components_are_the_component():
    """The bound of a mixture whose components agree is the component's:
    log Σ π exp(t) = t when every t is the same."""
    warping = wp.ZScore(1)
    single = lk.Gaussian(warping)
    mixture = lk.LikelihoodMixture([lk.Gaussian(warping),
                                    lk.Gaussian(warping)])
    for component in mixture.components:
        component.parameters["noise"].set_value(
            single.parameters["noise"].get_value().numpy())
    mu, var, y, has = _rows()
    expected = single.log_lik(mu, var, y, has)
    got = mixture.log_lik(tf.concat([mu, mu], 1), tf.concat([var, var], 1),
                          y, has)
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-10)


def test_a_component_s_columns_are_its_own():
    """Each component reads its own slice: moving the second slice's mean
    changes nothing a component-one-only bound would see."""
    mixture = lk.LikelihoodMixture([lk.Gaussian(wp.ZScore(1)),
                                    lk.Gaussian(wp.ZScore(1))],
                                   weights=[1.0 - 1e-9, 1e-9])
    mu, var, y, has = _rows()
    near = mixture.log_lik(tf.concat([mu, mu], 1), tf.concat([var, var], 1),
                           y, has)
    far = mixture.log_lik(tf.concat([mu, mu + 50], 1),
                          tf.concat([var, var], 1), y, has)
    np.testing.assert_allclose(near.numpy(), far.numpy(), rtol=1e-6)


def test_what_cannot_be_mixed_is_refused():
    with pytest.raises(ValueError, match="two components"):
        lk.LikelihoodMixture([lk.Gaussian()])
    with pytest.raises(ValueError, match="columns"):
        lk.LikelihoodMixture([lk.Gaussian(wp.ZScore(1)),
                              lk.Gaussian(wp.ZScore(2))])
    with pytest.raises(TypeError, match="continuous"):
        lk.LikelihoodMixture([lk.Gaussian(), lk.Bernoulli()])
    with pytest.raises(ValueError, match="one weight"):
        lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                             weights=[1.0])


@pytest.mark.parametrize("shares, n_sim", [((0.7, 0.3), 20),
                                           ((0.5, 0.5), 7),
                                           ((0.2, 0.3, 0.5), 40)])
def test_the_realizations_are_shared_out_in_proportion(shares, n_sim):
    mixture = lk.LikelihoodMixture([lk.Gaussian() for _ in shares],
                                   weights=shares)
    labels = mixture.labels(n_sim)
    counts = np.bincount(labels, minlength=len(shares))
    assert counts.sum() == n_sim
    assert np.all(np.abs(counts - np.asarray(shares) * n_sim) <= 1)
    # interleaved: the first few already hold more than one component
    assert len(set(labels[:4])) > 1


def test_each_realization_comes_from_its_own_component():
    """Latent values far apart in the two slices: every realization of the
    prediction is the one of its own component's slice."""
    mixture = lk.LikelihoodMixture([lk.Gaussian(wp.ZScore(1)),
                                    lk.Gaussian(wp.ZScore(1))],
                                   weights=[0.6, 0.4])
    n, n_sim = 5, 10
    sims = tf.concat([tf.fill([n, 1, n_sim], tf.constant(-3.0, tf.float64)),
                      tf.fill([n, 1, n_sim], tf.constant(3.0, tf.float64))],
                     axis=1)
    mu = tf.zeros([n, 2], tf.float64)
    out = mixture.predict(mu, tf.ones([n, 2], tf.float64), sims, None,
                          include_noise=False)
    values = out["simulations"].numpy()[:, 0, :]
    labels = mixture.labels(n_sim)
    assert np.all(values[:, labels == 0] < 0)
    assert np.all(values[:, labels == 1] > 0)
    assert out["population_prediction"].shape == (n, 1, 2)

    samples = mixture.measurement_samples(sims, n_nodes=4).numpy()[:, 0, :]
    tiled = np.tile(labels, 4)
    assert np.all(samples[:, tiled == 0] < 0)
    assert np.all(samples[:, tiled == 1] > 0)


def _model(n=80, seed=0):
    geoml.set_seed(1234)
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    which = rng.uniform(size=n) < 0.5
    y = np.where(which, 3 + np.sin(x), np.exp(0.3 * np.cos(x)))
    data = geoml.data.PointData.from_array(x[:, None], ["X"])
    data.add_continuous_variable("v", y)
    root = latent.BasicInput(geoml.data.Grid1D(start=0, n=21, step=0.5),
                             transform=geoml.transform.Isotropic(2.0))
    model = geoml.models.VGPNetwork(
        data, "v", lk.LikelihoodMixture([
            lk.Gaussian(wp.ChainedWarping(wp.BoxCox(1), wp.ZScore(1))),
            lk.Gaussian(wp.ZScore(1))]),
        latent.BasicGP(root, size=2),
        options=geoml.models.GPOptions(verbose=False, training_samples=10))
    return model, data, which


def test_the_components_start_apart():
    """Started from the data clustered in two, each component's warping on
    its own group: latent zero means a different value in each."""
    model, data, _ = _model()
    lik = model.likelihoods[0]
    zero = tf.zeros([1, 1], tf.float64)
    low, high = (float(np.ravel(c.warping.backward(zero))[0])
                 for c in lik.components)
    assert low < 2.0 < high
    shares = lik.parameters["weights"].get_value().numpy()
    assert np.all(shares > 0.3)


def test_the_start_repeats_under_one_seed():
    """The clustering behind the start must not tie: on one column, normal
    scores are symmetric whatever the data, and the two splits either side
    of the median tied and were broken differently from run to run."""
    starts = []
    for _ in range(3):
        model, _, _ = _model()
        starts.append(np.concatenate([np.ravel(p.get_value())
                                      for p in model.all_parameters]))
    for start in starts[1:]:
        np.testing.assert_array_equal(start, starts[0])


def test_a_model_trains_predicts_and_finds_its_populations(tmp_path):
    model, data, which = _model()
    model.train_full(150)
    assert model.training_log[-1] > model.training_log[0]

    target = geoml.data.PointData.from_array(
        np.linspace(0, 10, 50)[:, None], ["X"])
    model.predict(target, n_sim=20)
    assert np.all(np.isfinite(target.values("v/prediction")))

    answer = model.responsibilities(data)["v"]
    np.testing.assert_allclose(answer.sum(axis=1), 1.0)
    first = answer[:, 0] > 0.5
    assert max(np.mean(first == which), np.mean(first != which)) > 0.8

    draws = model.predict_measurements(target, n_sim=4)["v"]
    assert np.all(np.isfinite(draws))

    path = str(tmp_path / "mixture")
    model.save(path)
    again = geoml.models.VGPNetwork.open(path)
    np.testing.assert_allclose(
        again.likelihoods[0].parameters["weights"].get_value().numpy(),
        model.likelihoods[0].parameters["weights"].get_value().numpy())
    reloaded = geoml.data.PointData.from_array(
        np.linspace(0, 10, 50)[:, None], ["X"])
    again.predict(reloaded, n_sim=20)
    np.testing.assert_allclose(reloaded.values("v/prediction"),
                               target.values("v/prediction"), rtol=1e-8)


# --------------------------------------------------------------------------- #
# shares that change from place to place
# --------------------------------------------------------------------------- #
def test_latent_shares_take_a_column_per_population():
    mixture = lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                                   shares="latent")
    assert mixture.size == 4
    assert "weights" not in mixture.parameters
    with pytest.raises(ValueError, match="fixed shares"):
        lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                             weights=[0.5, 0.5], shares="latent")
    with pytest.raises(ValueError, match="'fixed' or 'latent'"):
        lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                             shares="spatial")
    with pytest.raises(ValueError, match="changes from place"):
        lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                             shares="latent").labels(10)


def test_the_amplitude_sharpens_the_shares():
    """The share columns are scaled by the square root of the amplitude, a
    variance, before the softmax: at one the shares are the plain softmax,
    at four the columns count double."""
    mixture = lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                                   shares="latent")
    columns = tf.constant([[0.0, 0.0, -0.5, 0.5]], tf.float64)[:, :, None]
    plain = 1 / (1 + np.exp(-1.0))
    assert float(mixture._share_values(columns)[0, 1, 0]) \
        == pytest.approx(plain)
    mixture.parameters["amplitude"].set_value(4.0)
    assert float(mixture._share_values(columns)[0, 1, 0]) \
        == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_the_bias_sets_the_shares_where_the_columns_are_zero():
    """Away from the data the share columns return to zero, and the shares
    to the softmax of the bias; `initialize` starts the bias on the sizes
    of the groups it finds."""
    mixture = lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                                   shares="latent")
    zero = tf.zeros([1, 4, 1], tf.float64)
    np.testing.assert_allclose(mixture._share_values(zero)[0, :, 0],
                               [0.5, 0.5])
    mixture.parameters["bias"].set_value(np.log([0.3, 0.7]))
    np.testing.assert_allclose(mixture._share_values(zero)[0, :, 0],
                               [0.3, 0.7])

    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 30), rng.normal(20, 1, 70)])
    geoml.set_seed(1234)
    mixture.initialize(y)
    np.testing.assert_allclose(mixture._share_values(zero)[0, :, 0],
                               [0.3, 0.7])


def test_a_realization_follows_the_shares_from_place_to_place():
    """Share columns favouring the first population at the first rows and
    the second at the last: every realization changes population between
    them, and at each row the realizations split as the shares do."""
    mixture = lk.LikelihoodMixture([lk.Gaussian(), lk.Gaussian()],
                                   shares="latent")
    n_sim = 40
    lean = np.linspace(-4, 4, 9)
    share_columns = np.stack([-lean, lean], axis=1)[:, :, None] \
        * np.ones([1, 1, n_sim])
    sims = tf.constant(np.concatenate(
        [np.zeros([9, 2, n_sim]), share_columns], axis=1))
    labels = mixture._location_labels(sims).numpy()
    assert np.all(labels[0] == 0) and np.all(labels[-1] == 1)
    second = 1 / (1 + np.exp(2 * lean))
    np.testing.assert_allclose(labels.mean(axis=1), 1 - second, atol=1 / n_sim)


def _latent_model(n=80, seed=0):
    geoml.set_seed(1234)
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    y = np.where(x < 5, np.sin(x), 3 + np.sin(x))
    data = geoml.data.PointData.from_array(x[:, None], ["X"])
    data.add_continuous_variable("v", y)
    root = latent.BasicInput(geoml.data.Grid1D(start=0, n=21, step=0.5),
                             transform=geoml.transform.Isotropic(2.0))
    model = geoml.models.VGPNetwork(
        data, "v", lk.LikelihoodMixture([lk.Gaussian(wp.ZScore(1)),
                                         lk.Gaussian(wp.ZScore(1))],
                                        shares="latent"),
        latent.BasicGP(root, size=4),
        options=geoml.models.GPOptions(verbose=False, training_samples=10))
    return model, data


def test_a_prediction_stores_the_populations(tmp_path):
    model, _ = _latent_model()
    model.train_full(100)
    target = geoml.data.PointData.from_array(
        np.linspace(0, 10, 30)[:, None], ["X"])
    model.predict(target, n_sim=12)
    variable = target.variables["v"]
    labels = np.asarray(variable.population)
    assert labels.shape == (30, 12) and set(np.unique(labels)) <= {0, 1}
    shares = np.stack([variable.responsibilities[k].values.to_numpy()
                       for k in (0, 1)], axis=1)
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    assert set(variable.population_prediction) == {0, 1}
    np.testing.assert_array_equal(target.values("v/population/3"),
                                  labels[:, 3])

    path = str(tmp_path / "populations.zarr")
    target.to_zarr(path)
    opened = geoml.data.PointData.open(path)
    np.testing.assert_array_equal(
        np.asarray(opened.variables["v"].population), labels)
    np.testing.assert_array_equal(
        np.asarray(target[np.arange(30) < 10].variables["v"].population),
        labels[:10])


def test_a_variable_no_mixture_models_has_no_population():
    point = geoml.data.PointData.from_array(np.arange(5.0)[:, None], ["X"])
    point.add_continuous_variable("v", np.arange(5.0))
    assert point.variables["v"].population is None
    assert "v/population" not in [str(p) for p, _ in point.addressable()]


def test_the_warped_pairs_refuse_a_mixture():
    """Each population is warped its own way, so there is no one space the
    measurements are seen in; the figure says so rather than drawing one
    population's."""
    import geoml.plots.prepare as prepare
    model, _ = _latent_model()
    with pytest.raises(TypeError, match="no one warped space"):
        prepare.warped_values(model, "v")
