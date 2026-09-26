"""The Gaussian mixture node, and training a leaf that is not Gaussian.

`GaussianMixture` blends its components by the softmax of its weights,
realization by realization, so its output is not Gaussian; a model whose
leaf is not Gaussian trains the likelihood on the realizations instead of
reading the leaf's mean and variance as a Gaussian's. Pinned here: the
constructor's refusals, the moments against the realizations they describe,
nothing drawn on the moment path, a location's draws independent of its
batch, gradients reaching the weights and every component, the samples
branch taken for a leaf that is not Gaussian and never for one that is, the
warning where a likelihood has no samples branch, save and load, and the
gate -- two sinusoids split at a known boundary, where the mixture must
beat one GP against the truth and put its switch at the boundary.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.latent as latent
import geoml.latent.network as network
import geoml.likelihood as lk
import geoml.metrics as metrics
import geoml.persistence as persistence
import geoml.transform as tr

PERIOD = 4.0
BOUNDARY = 5.0


def _grid():
    return geoml.data.Grid1D(start=-0.5, n=45, step=0.25)


def _container(x, y=None):
    container = geoml.data.PointData.from_array(np.asarray(x)[:, None], ["X"])
    if y is not None:
        container.add_continuous_variable("v", y)
    return container


def _truth(x):
    """One sinusoid below the boundary and the same in opposite phase
    above it."""
    a = np.sin(2 * np.pi * x / PERIOD)
    return np.where(x >= BOUNDARY, -a, a)


def _split_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 10.0, n))
    return _container(x, _truth(x) + rng.normal(0.0, 0.1, n))


def _mixture(k=2, size=1):
    """Components smoother than the jump, and weights of a longer range
    still: the configuration the gate found to switch."""
    geoml.set_seed(1234)
    weights_root = latent.BasicInput(_grid(), transform=tr.Isotropic(6.0))
    component_root = latent.BasicInput(_grid(), transform=tr.Isotropic(4.0))
    weights = latent.BasicGP(weights_root, size=k)
    return latent.GaussianMixture(
        weights, [latent.BasicGP(component_root, size=size)
                  for _ in range(k)])


def _model(data, leaf, likelihood=None, variable="v"):
    return geoml.models.VGPNetwork(
        data, variable, likelihood or lk.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False))


@pytest.fixture(scope="module")
def trained():
    model = _model(_split_data(), _mixture())
    model.train_full(max_iter=200)
    return model


def _locations(n=40):
    return tf.constant(np.linspace(0.0, 10.0, n)[:, None], tf.float64)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_one_component_is_refused():
    root = latent.BasicInput(_grid(), transform=tr.Isotropic(2.0))
    with pytest.raises(ValueError, match="two components"):
        latent.GaussianMixture(latent.BasicGP(root, size=1),
                               [latent.BasicGP(root, size=1)])


def test_the_weights_are_one_per_component():
    root = latent.BasicInput(_grid(), transform=tr.Isotropic(2.0))
    with pytest.raises(latent.SizeIncompatibilityError, match="one weight"):
        latent.GaussianMixture(latent.BasicGP(root, size=3),
                               [latent.BasicGP(root, size=1),
                                latent.BasicGP(root, size=1)])


def test_the_components_share_a_size():
    root = latent.BasicInput(_grid(), transform=tr.Isotropic(2.0))
    with pytest.raises(latent.SizeIncompatibilityError, match="same size"):
        latent.GaussianMixture(latent.BasicGP(root, size=2),
                               [latent.BasicGP(root, size=1),
                                latent.BasicGP(root, size=2)])


def test_the_output_is_the_components_size_and_not_gaussian():
    node = _mixture(k=3, size=2)
    assert node.size == 2
    assert node.gaussian is False
    assert node.propagates_inducing_points is False


# --------------------------------------------------------------------------- #
# the protocol
# --------------------------------------------------------------------------- #
def test_the_moments_are_those_of_the_mixture_of_its_parents(trained):
    """Against brute force over the parents' marginals. Not against the
    realizations: a GP node's realizations carry only the variance its
    inducing points explain, while its moments carry all of it."""
    leaf = trained.leaves[0]
    trained._refresh(1e-6)
    x = _locations()
    mean, var = (np.asarray(t) for t in leaf.propagate(x))
    w_mu, w_var = (np.asarray(t) for t in leaf.weights.propagate(x))
    amplitude = float(np.ravel(leaf.parameters["amplitude"].get_value())[0])
    w_mu, w_var = w_mu * np.sqrt(amplitude), w_var * amplitude
    parts = [[np.asarray(t) for t in c.propagate(x)]
             for c in leaf.components]

    rng = np.random.default_rng(0)
    n = 100_000
    w = w_mu[None] + np.sqrt(w_var)[None] * rng.standard_normal(
        (n,) + w_mu.shape)
    shares = np.exp(w - w.max(axis=2, keepdims=True))
    shares /= shares.sum(axis=2, keepdims=True)
    draws = sum(shares[:, :, k, None]
                * (c_mu[None] + np.sqrt(c_var)[None]
                   * rng.standard_normal((n,) + c_mu.shape))
                for k, (c_mu, c_var) in enumerate(parts))

    np.testing.assert_allclose(mean, draws.mean(axis=0), atol=0.01)
    np.testing.assert_allclose(np.sqrt(var), draws.std(axis=0),
                               rtol=0.03, atol=0.005)
    explained = np.asarray(leaf._explained_var)
    assert np.all(np.isfinite(explained)) and np.all(explained >= 0.0)


def test_the_moment_path_draws_nothing(trained, monkeypatch):
    leaf = trained.leaves[0]
    trained._refresh(1e-6)
    calls = []
    original = network._simulation_normals

    def counting(shape, seed, key=None):
        calls.append(tuple(shape))
        return original(shape, seed, key=key)

    monkeypatch.setattr(network, "_simulation_normals", counting)
    leaf.propagate(_locations())
    assert calls == []
    leaf.predict(_locations(), n_sim=3)
    assert len(calls) > 0


def test_a_location_draws_the_same_whatever_the_batch(trained):
    leaf = trained.leaves[0]
    trained._refresh(1e-6)
    whole = np.asarray(leaf.predict(_locations(), n_sim=5, seed=[3, 0])[2])
    part = np.asarray(leaf.predict(_locations()[:7], n_sim=5,
                                   seed=[3, 0])[2])
    np.testing.assert_allclose(part, whole[:, :7], rtol=0, atol=1e-10)


def test_training_moves_the_weights_and_every_component():
    leaf = _mixture()
    model = _model(_split_data(), leaf)
    watched = [leaf, leaf.weights] + leaf.components

    def own(node):
        return [np.array(p.get_value()) for p in node.parameters.values()
                if not p.fixed]

    before = [own(node) for node in watched]
    model.train_full(max_iter=3)
    for node, start in zip(watched, before):
        assert any(not np.array_equal(a, b)
                   for a, b in zip(start, own(node))), node.name


# --------------------------------------------------------------------------- #
# training a leaf that is not Gaussian
# --------------------------------------------------------------------------- #
def _recorded(model):
    likelihood = model.likelihoods[0]
    seen = []
    original = likelihood.log_lik

    def recording(*args, **kwargs):
        seen.append(kwargs.get("latent_gaussian", True))
        return original(*args, **kwargs)

    likelihood.log_lik = recording
    model.train_full(max_iter=1)
    return seen


def test_a_leaf_that_is_not_gaussian_trains_on_its_realizations():
    assert _recorded(_model(_split_data(), _mixture())) == [False]


def test_a_product_leaf_trains_on_its_realizations_too():
    geoml.set_seed(1234)
    root = latent.BasicInput(_grid(), transform=tr.Isotropic(2.0))
    product = latent.Multiply(latent.BasicGP(root, size=1),
                              latent.Exponentiation(
                                  latent.BasicGP(root, size=1)))
    assert _recorded(_model(_split_data(), product)) == [False]


def test_a_gaussian_leaf_trains_as_it_did():
    geoml.set_seed(1234)
    root = latent.BasicInput(_grid(), transform=tr.Isotropic(2.0))
    assert _recorded(_model(_split_data(), latent.BasicGP(root, size=1))) \
        == [True]


def test_a_likelihood_without_samples_warns():
    data = _split_data()
    x = np.asarray(data.coordinates)[:, 0]
    data.add_binary_variable(
        "b", measurements=np.where(x > 5.0, "in", "out"))
    geoml.set_seed(1234)
    root = latent.BasicInput(_grid(), transform=tr.Isotropic(2.0))
    leaf = latent.Exponentiation(latent.BasicGP(root, size=1))
    with pytest.warns(UserWarning, match="not Gaussian"):
        model = _model(data, leaf, lk.Bernoulli(), variable="b")
    model.train_full(max_iter=1)
    assert np.isfinite(model.training_log[-1])


def test_a_saved_mixture_predicts_the_same(trained, tmp_path):
    target = _container(np.linspace(0.0, 10.0, 30))
    trained.predict(target, n_sim=8)
    before = np.asarray(target.values("v/prediction"), dtype=float)

    persistence.save_model(trained, tmp_path / "model")
    restored = persistence.load_model(tmp_path / "model")
    assert isinstance(restored.leaves[0], latent.GaussianMixture)
    again = _container(np.linspace(0.0, 10.0, 30))
    restored.predict(again, n_sim=8)
    np.testing.assert_allclose(
        np.asarray(again.values("v/prediction"), dtype=float), before)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_the_mixture_finds_the_regimes():
    """Against the truth on a dense line: lower RMSE and CRPS than one GP,
    and the weights' switch at the planted boundary."""
    data = _split_data()
    x = np.linspace(0.0, 10.0, 400)
    truth = _truth(x)

    geoml.set_seed(1234)
    root = latent.BasicInput(_grid(), transform=tr.Isotropic(2.0))
    scores = {}
    for name, leaf in (("gp", latent.BasicGP(root, size=1)),
                       ("mixture", _mixture())):
        model = _model(data, leaf)
        model.train_full(max_iter=1000)
        target = _container(x)
        model.predict(target, n_sim=200)
        prediction = np.asarray(target.values("v/prediction"), dtype=float)
        sims = np.asarray(target.variables["v"].get_simulations())
        scores[name] = (np.sqrt(np.mean((prediction - truth) ** 2)),
                        metrics.crps(truth, sims))
    assert scores["mixture"][0] < scores["gp"][0]
    assert scores["mixture"][1] < scores["gp"][1]

    # the component that dominates changes within a quarter period of the
    # boundary
    leaf = model.leaves[0]
    model._refresh(1e-6)
    mu, _ = leaf.weights.propagate(tf.constant(x[:, None], tf.float64))
    first = np.asarray(tf.nn.softmax(mu, axis=1))[:, 0]
    changes = x[1:][np.diff(first > 0.5) != 0]
    assert np.min(np.abs(changes - BOUNDARY)) < PERIOD / 4
