"""End-to-end runs of the real modelling cases, on the bundled sample data.

These mirror the working scripts kept outside the package (Jura land use and
elements, the Arctic lake composition, sunspot numbers, Walker Lake), scaled
down to a few iterations and small grids. They are not about accuracy: they
exercise the variable types, latent networks, warpings and likelihoods that
those cases actually use, end to end, which unit tests on synthetic data do not
reach.

The Jura land-use case is here for a specific reason: caching the prediction
posterior in shapeless Variables made every prediction tensor lose its static
rank, which only broke the paths needing one (``tf.nn.softmax`` on a given
axis). Categorical prediction was dead while every other test passed.
"""
import numpy as np
import pytest

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr
import geoml.warping as wp


def _seed(seed=1234):
    """Pins the initial parameters, the synthetic data, and the noise draws."""
    import tensorflow as tf
    geoml.set_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _options(**kwargs):
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("training_samples", 8)
    return geoml.models.GPOptions(**kwargs)


# --------------------------------------------------------------------------- #
# Jura -- categorical land use
# --------------------------------------------------------------------------- #
def test_jura_categorical():
    _seed()
    jura_train, _ = geoml.datasets.jura()
    labels = list(jura_train.variables["Landuse"].labels)

    network_input = gl.BasicInput(
        inducing_points=jura_train, transform=tr.Isotropic(0.5))
    network_output = gl.BasicGP(
        parent=network_input, size=len(labels), kernel=kr.Matern32(),
        fix_range=True)

    model = geoml.models.VGPNetwork(
        data=jura_train, variables="Landuse", latent_network=network_output,
        likelihoods=lk.CategoricalGaussianIndicator(n_components=len(labels)),
        options=_options())
    model.train_full(5)

    grid = geoml.data.Grid2D(start=[0, 0], end=[6, 6], n=[21, 21])
    model.predict(grid)

    variable = grid.variables["Landuse"]
    assert type(variable) is geoml.data.CategoricalVariable
    assert list(variable.labels) == labels

    predicted = variable.predicted.as_image()
    uncertainty = variable.uncertainty.as_image()
    assert predicted.shape == (21, 21)
    assert set(np.unique(predicted)).issubset(set(labels))
    assert np.all(np.isfinite(uncertainty))
    assert np.all((uncertainty >= 0) & (uncertainty <= 1))

    # the class probabilities must be a partition of unity
    probabilities = np.stack(
        [np.asarray(variable.components[lb].probability.values)
         for lb in labels], axis=1)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert variable.components[labels[0]].probability.as_image().shape == (21, 21)

    # and the model scores itself on the data it was trained on
    model.predict(jura_train)
    metrics = jura_train.variables["Landuse"].compute_metrics()
    assert metrics is not None


def test_jura_categorical_prediction_is_batch_invariant():
    """Guards the cached prediction state: batching must not change results."""
    _seed()
    jura_train, _ = geoml.datasets.jura()
    labels = list(jura_train.variables["Landuse"].labels)

    root = gl.BasicInput(inducing_points=jura_train,
                         transform=tr.Isotropic(0.5))
    net = gl.BasicGP(parent=root, size=len(labels), kernel=kr.Matern32(),
                     fix_range=True)
    model = geoml.models.VGPNetwork(
        data=jura_train, variables="Landuse", latent_network=net,
        likelihoods=lk.CategoricalGaussianIndicator(n_components=len(labels)),
        options=_options())
    model.train_full(5)

    grid = geoml.data.Grid2D(start=[0, 0], end=[6, 6], n=[15, 15])
    model.options.prediction_batch_size = 10 ** 9
    model.predict(grid)
    one = np.asarray(grid.variables["Landuse"].components[labels[0]]
                     .probability.values).copy()

    model.options.prediction_batch_size = 40
    model.predict(grid)
    many = np.asarray(grid.variables["Landuse"].components[labels[0]]
                      .probability.values)

    assert np.allclose(one, many, atol=1e-8)


# --------------------------------------------------------------------------- #
# Jura -- the seven elements as a vector variable
# --------------------------------------------------------------------------- #
def test_jura_elements():
    _seed()
    jura_train, jura_validation = geoml.datasets.jura()
    elements = list(jura_train.variables["Elements"].labels)
    n_el = len(elements)

    inducing = geoml.data.Grid2D(start=[0, 0], n=[9, 9], end=[6, 6])
    net_input = gl.BasicInput(inducing, tr.Isotropic(1), fix_transform=False)
    net_out = gl.MultiStructureGP(net_input, size=n_el, kernel=kr.Spherical())

    warping = wp.ChainedWarping(
        wp.Scale(n_el),
        wp.Softplus(n_el),
        wp.ZScore(n_el),
        wp.RobustPCA(n_el, n_el),
        wp.Spline(n_el, 10),
        wp.ZScore(n_el),
        wp.Rotation(n_el, fixed=True),
    )
    likelihood = lk.MultivariateEpsilonInsensitive(n_el, warping, sharpness=1)

    model = geoml.models.VGPNetwork(
        data=jura_train, variables="Elements", likelihoods=likelihood,
        latent_network=net_out, options=_options(prediction_batch_size=2000))
    model.set_learning_rate(2e-2)
    model.train_full(5)

    grid = geoml.data.Grid2D(start=[0, 0], n=[21, 21], end=[6, 6])
    model.predict(grid, n_sim=10, include_noise="monte_carlo")
    grid.variables["Elements"].reset_quantiles([0.025, 0.5, 0.975])

    component = grid.variables["Elements"].components[elements[0]]
    median = component.quantiles[0.5].as_image()
    low = component.quantiles[0.025].as_image()
    high = component.quantiles[0.975].as_image()
    assert median.shape == (21, 21)
    assert np.all(np.isfinite(median))
    assert np.all(high >= low)                      # ordered quantiles

    # one realization, reshaped like any other attribute
    assert component.simulation(0).as_image().shape == (21, 21)
    assert component.n_sim == 10

    simulations = grid.variables["Elements"].get_simulations()
    assert simulations.shape == (grid.n_data, 10, n_el)

    # scoring against held-out data
    model.predict(jura_validation, n_sim=10, include_noise="monte_carlo")
    jura_validation.variables["Elements"].reset_quantiles([0.025, 0.5, 0.975])
    metrics = jura_validation.variables["Elements"].compute_metrics()
    assert metrics is not None

    scores, _ = warping.forward(
        jura_train.variables["Elements"].get_measurements()[0])
    assert np.all(np.isfinite(np.asarray(scores)))


def test_jura_mixed_categorical_and_elements():
    """Two variables at once, through a deep network with an input walk."""
    _seed()
    jura_train, _ = geoml.datasets.jura()
    elements = list(jura_train.variables["Elements"].labels)
    n_el = len(elements)
    n_rock = len(jura_train.variables["Rock"].labels)

    inducing = geoml.data.Grid2D(start=[0, 0], n=[7, 7], end=[6, 6])
    net_input = gl.BasicInput(inducing, tr.Isotropic(0.5))

    field = gl.MultiStructureGP(net_input, size=2, kernel=kr.Cubic())
    coords = gl.GPWalk(field)
    categorical = gl.MultiStructureGP(coords, size=n_rock, kernel=kr.Cubic())

    bottleneck = gl.Linear(categorical, size=2, unit_norm=False)
    trend = gl.Linear(bottleneck, size=n_el, unit_norm=False)
    numeric = gl.MultiStructureGP(net_input, size=n_el, kernel=kr.Spherical())
    net_out = gl.Concatenate(categorical, gl.Add(trend, numeric))

    warping = wp.ChainedWarping(
        wp.Scale(n_el), wp.Softplus(n_el), wp.ZScore(n_el))

    model = geoml.models.VGPNetwork(
        data=jura_train, variables=["Rock", "Elements"],
        likelihoods=[lk.CategoricalGaussianIndicator(n_rock),
                     lk.MultivariateEpsilonInsensitive(n_el, warping,
                                                       sharpness=1)],
        latent_network=net_out,
        options=_options(prediction_batch_size=1000, jitter=1e-6))
    model.train_full(3)

    grid = geoml.data.Grid2D(start=[0, 0], n=[15, 15], end=[6, 6])
    model.predict(grid, n_sim=5, include_noise=True)
    grid.variables["Elements"].reset_quantiles([0.025, 0.5, 0.975])

    rock = grid.variables["Rock"]
    assert rock.predicted.as_image().shape == (15, 15)
    assert np.all(np.isfinite(rock.uncertainty.as_image()))

    probabilities = np.stack(
        [np.asarray(rock.components[lb].probability.values)
         for lb in rock.labels], axis=1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    median = grid.variables["Elements"].components[elements[0]].quantiles[0.5]
    assert np.all(np.isfinite(median.as_image()))


# --------------------------------------------------------------------------- #
# Arctic lake -- a composition along a single axis
# --------------------------------------------------------------------------- #
def test_arctic_lake_composition():
    _seed()
    arctic_lake = geoml.datasets.arctic_lake()
    parts = list(arctic_lake.variables["comp"].labels)

    warping = wp.ChainedWarping(
        wp.CenteredLogRatio(3),
        wp.RobustPCA(3, 2),
        wp.ZScore(2),
        wp.Spline(2, 10),
        wp.ZScore(2),
    )
    composition = arctic_lake.variables["comp"].get_measurements()[0]
    scores = warping.initialize(composition)
    assert np.asarray(scores).shape == (arctic_lake.n_data, 2)

    likelihood = lk.MultivariateEpsilonInsensitive(
        n_components=3, warping=warping)

    net_input = gl.BasicInput(
        inducing_points=geoml.data.Grid1D(10, n=40, end=105),
        transform=tr.Isotropic(2.5))
    extra_latent = gl.BasicGP(net_input, size=1, fix_range=True)
    concat = gl.Concatenate(net_input, extra_latent)
    net_out = gl.BasicGP(concat, size=2, kernel=kr.Matern32())

    model = geoml.models.VGPNetwork(
        data=arctic_lake, variables="comp", likelihoods=likelihood,
        latent_network=net_out,
        options=_options(training_samples=20, jitter=1e-4))
    model.train_full(5)

    grid = geoml.data.Grid1D(10, n=96, step=1.0)
    model.predict(grid, n_sim=20)

    # the predicted parts must still form a composition
    predicted = np.stack(
        [grid.variables["comp"].components[p].prediction.as_series(sigma=3)
         for p in parts], axis=1)
    assert predicted.shape == (96, 3)
    assert np.all(np.isfinite(predicted))
    assert np.allclose(predicted.sum(axis=1), 1.0, atol=1e-6)

    simulations = grid.variables["comp"].get_simulations()
    assert simulations.shape == (96, 20, 3)
    assert np.allclose(simulations.sum(axis=2), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Sunspots -- periodic transforms, an input walk and stochastic training
# --------------------------------------------------------------------------- #
def test_sunspot_deep_model():
    _seed()
    sunspots = geoml.datasets.sunspot_number()
    monthly = sunspots["points"]["monthly"]

    transform = tr.Concatenate(
        tr.Isotropic(20 * 12),
        tr.ChainedTransform(tr.Isotropic(2 * 12), tr.Periodic()),
        tr.ChainedTransform(tr.Isotropic(10 * 12), tr.Periodic()),
    )

    inducing_points = geoml.data.Grid1D(start=0, n=60, step=64)
    net_input = gl.BasicInput(inducing_points, transform)
    time = gl.SelectInput(net_input, [0])
    field = gl.BasicGP(net_input, size=5)
    walk = gl.GPWalk(field)
    sines = gl.SelectInput(walk, [1, 2, 3, 4])
    output_layer = gl.BasicGP(gl.Concatenate(time, sines))

    model = geoml.models.VGPNetwork(
        data=monthly, variables="sn", latent_network=output_layer,
        likelihoods=lk.Gaussian(wp.ChainedWarping(
            wp.Softplus(1), wp.ZScore(1), wp.Spline(1, 3))),
        options=_options(training_batch_size=500, prediction_batch_size=2000))
    model.train_svi(2)
    assert len(model.training_log) > 0
    assert np.all(np.isfinite(model.training_log))

    grid = geoml.data.Grid1D(start=0, n=200, step=18)
    model.predict(grid, n_sim=20)
    grid.variables["sn"].reset_quantiles([0.025, 0.5, 0.975])

    quantiles = grid.variables["sn"].quantiles
    median = quantiles[0.5].as_series(sigma=2)
    assert median.shape == (200,)
    assert np.all(np.isfinite(median))
    assert np.all(np.asarray(quantiles[0.975].values)
                  >= np.asarray(quantiles[0.025].values))


# --------------------------------------------------------------------------- #
# Walker Lake -- the closed-form GP
# --------------------------------------------------------------------------- #
def test_walker_lake_legacy_gp():
    _seed()
    walker, walker_grid = geoml.datasets.walker()
    max_v = 1700.0

    warping = wp.ChainedWarping(
        wp.Scale(1, max_v),
        wp.Softplus(1),
        wp.ZScore(1),
        wp.Spline(1, knots_per_arm=2),
    )
    covariance = kr.Covariance(kernel=kr.Spherical(),
                               transform=tr.Isotropic(100))

    model = geoml.models.GP(
        data=walker, variable="V", covariance=covariance, warping=warping,
        options=_options())
    model.train(5)
    assert len(model.training_log) == 5
    assert np.all(np.isfinite(model.training_log))

    model.predict(walker_grid, n_sim=20)
    walker_grid.variables["V"].reset_quantiles([0.025, 0.5, 0.975])

    quantiles = walker_grid.variables["V"].quantiles
    median = quantiles[0.5].as_image()
    assert median.shape == tuple(walker_grid.grid_size)[::-1]
    assert np.all(np.isfinite(median))
    assert np.all(np.asarray(quantiles[0.975].values)
                  >= np.asarray(quantiles[0.025].values))

    # the warping must be usable on its own, and monotonic
    normal = np.linspace(-4, 4, 101)[:, None]
    back = np.asarray(model.warping.backward(normal))
    assert np.all(np.isfinite(back))
    assert np.all(np.diff(back[:, 0]) >= -1e-8)
