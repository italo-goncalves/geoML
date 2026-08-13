"""The mixture likelihood: several noise scales, one latent field.

Covers the constructor's guarantees (one family, components separated at
construction, nothing taken for contamination unless declared), the mixture
being over the *row* rather than over the columns, the split the
contamination flag makes between what the ground holds and what a
measurement would read, the bisected mixture quantile behind
`measurement_samples`, the responsibilities diagnostic and where it is
filed, end-to-end training on contaminated data, and the persistence round
trip.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from scipy import optimize, stats

import geoml


def _mixture(w=(0.9, 0.1), s1=0.5, s2=3.0, contamination=None):
    """A two-Gaussian mixture with hand-set widths and an identity warping."""
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1),
                                   contamination=contamination)
    mix.parameters["weights"].set_value(np.array(w))
    mix.components[0].parameters["noise"].set_value(
        np.full([1, 1, 1], s1 ** 2))
    mix.components[1].parameters["noise"].set_value(
        np.full([1, 1, 1], s2 ** 2))
    # ZScore starts at mean 0, sd 1: warped space and value space coincide
    return mix


def _contaminated(seed=1234, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    f = np.sin(2.2 * x) + 0.6 * np.sin(0.7 * x + 1.0)
    eps = rng.normal(0.0, 0.3, n)
    is_out = rng.uniform(size=n) < 0.05
    eps[is_out] = rng.normal(0.0, 3.0, is_out.sum())
    return x, f + eps, is_out


def _model(likelihood, x, y, seed=1234, max_iter=200):
    geoml.set_seed(seed)
    tf.random.set_seed(seed)
    point = geoml.data.PointData(pd.DataFrame({"c0": x}), ["c0"])
    point.add_continuous_variable("v", y)
    inducing = geoml.data.Grid1D(start=0.0, n=40, step=10.0 / 39)
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(1.5))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "v", likelihood, gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=max_iter)
    return model, point


# --------------------------------------------------------------------------- #
# constructor guarantees
# --------------------------------------------------------------------------- #
def test_a_mixture_needs_two_components():
    with pytest.raises(ValueError, match="two components"):
        geoml.likelihood.Mixture(geoml.warping.ZScore(1), n_components=1)


def test_something_must_describe_the_ground():
    with pytest.raises(ValueError, match="ground"):
        geoml.likelihood.Mixture(geoml.warping.ZScore(1),
                                 contamination=[True, True])


def test_an_unknown_family_names_the_ones_that_work():
    with pytest.raises(ValueError, match="unknown mixture family"):
        geoml.likelihood.Mixture(geoml.warping.ZScore(1), family="gamma")


def test_the_components_are_separated_at_construction():
    """Components of equal width never pull apart in training, so the family's
    width parameters are spread at construction -- each by the exponent that
    parameter carries the width with."""
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1), n_components=3,
                                   separation=3.0)
    noise = [float(c.parameters["noise"].get_value().numpy().ravel()[0])
             for c in mix.components]
    # `noise` is a variance, so a width factor of 3 is a factor of 9 in it
    assert np.allclose(noise[1] / noise[0], 9.0)
    assert np.allclose(noise[2] / noise[1], 9.0)


def test_a_rate_is_separated_the_other_way():
    """`c_rate` is a rate: wider means smaller, and `epsilon`, in the data's
    own units, moves with the width."""
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1), n_components=2,
                                   family="epsiloninsensitive",
                                   separation=2.0)
    rate = [float(c.parameters["c_rate"].get_value().numpy().ravel()[0])
            for c in mix.components]
    eps = [float(c.parameters["epsilon"].get_value().numpy().ravel()[0])
           for c in mix.components]
    assert np.allclose(rate[1] / rate[0], 0.5)
    assert np.allclose(eps[1] / eps[0], 2.0)


def test_the_separation_is_not_clamped_by_the_family_ceiling():
    """A bound chosen for one noise has no say over the component built to be
    the wide one; a value clamped back would leave the components equal."""
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1), n_components=4,
                                   separation=3.0)
    noise = [float(c.parameters["noise"].get_value().numpy().ravel()[0])
             for c in mix.components]
    assert noise[3] > 10.0            # past `Gaussian`'s own maximum
    assert np.allclose(noise[3] / noise[2], 9.0)


def test_nothing_is_contamination_unless_it_is_declared():
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1), n_components=3)
    assert mix.contamination == [False, False, False]


def test_the_components_are_sized_by_the_warping():
    """The mixture's size is the warping's, and the components are built at
    it, so the wide scale can be wide only in the column that needs it."""
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(2))
    assert mix.size == 2
    for component in mix.components:
        assert component.parameters["noise"].get_value().shape == (1, 2, 1)


def test_component_warpings_are_inert():
    """The mixture's warping is the one applied; a component's own must not
    leave trainable parameters drifting in the ELBO."""
    mix = _mixture()
    for component in mix.components:
        for parameter in component.warping._all_parameters:
            assert parameter.fixed
    # the mixture's own warping still trains
    assert any(not p.fixed for p in mix.warping._all_parameters)


# --------------------------------------------------------------------------- #
# the mixture is over the row
# --------------------------------------------------------------------------- #
def _row_and_cell_reference(y, w, sd):
    """The two candidate log-likelihoods for a zero field, by hand."""
    density = np.stack([stats.norm.pdf(y, 0.0, s) for s in sd], axis=0)
    rowwise = np.log(sum(w[k] * density[k].prod(axis=1)
                         for k in range(len(w)))).sum()
    cellwise = np.log(sum(w[k] * density[k] for k in range(len(w)))).sum()
    return rowwise, cellwise


def test_a_vector_mixture_is_over_the_row_not_the_cell():
    """One component explains a whole measurement: the columns' densities are
    multiplied before the components are weighted. Taking it the other way
    round is cellwise contamination, a different model, and not what a vector
    variable describes -- its columns are one observation in sample space."""
    w, sd = (0.9, 0.1), (0.5, 3.0)
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(2))
    mix.parameters["weights"].set_value(np.array(w))
    for component, s in zip(mix.components, sd):
        component.parameters["noise"].set_value(np.full([1, 2, 1], s ** 2))

    y = np.array([[0.5, 6.0], [0.2, -0.3], [7.0, 6.5]])
    zero = np.zeros_like(y)
    tiny = np.full_like(y, 1e-12)
    # a vector mixture takes the expectation over the joint samples; with a
    # vanishing posterior variance they all sit on the mean
    samples = tf.constant(zero[:, :, None], tf.float64)

    got = float(mix.log_lik(tf.constant(zero), tf.constant(tiny),
                            tf.constant(y), tf.constant(np.ones_like(y)),
                            samples=samples).numpy())
    rowwise, cellwise = _row_and_cell_reference(y, w, sd)
    assert abs(got - rowwise) < 1e-6
    assert abs(got - cellwise) > 1.0


def test_the_diagnostic_speaks_the_same_language_as_the_training():
    """`responsibilities` is the posterior of the same row-level mixture the
    likelihood fits -- the two disagreed for a vector variable until the
    density was taken over the row."""
    w, sd = (0.9, 0.1), (0.5, 3.0)
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(2))
    mix.parameters["weights"].set_value(np.array(w))
    for component, s in zip(mix.components, sd):
        component.parameters["noise"].set_value(np.full([1, 2, 1], s ** 2))

    y = np.array([[0.5, 6.0], [0.2, -0.3]])
    got = mix.responsibilities(np.zeros_like(y), np.full_like(y, 1e-12), y)

    density = np.stack([stats.norm.pdf(y, 0.0, s) for s in sd], axis=0)
    reference = np.stack([w[k] * density[k].prod(axis=1) for k in range(2)],
                         axis=1)
    reference = reference / reference.sum(axis=1, keepdims=True)
    assert np.allclose(got, reference, atol=1e-6)
    # the row with one wild column is bad as a row
    assert got[0, 1] > 0.99


def test_one_column_keeps_the_cheaper_quadrature():
    """A scalar mixture factorizes trivially, so it stays on Gauss-Hermite;
    a vector one has to integrate over the joint posterior samples."""
    assert geoml.likelihood.Mixture(
        geoml.warping.ZScore(1))._column_quadrature()
    assert not geoml.likelihood.Mixture(
        geoml.warping.ZScore(3))._column_quadrature()


# --------------------------------------------------------------------------- #
# the ground, and what a measurement would read
# --------------------------------------------------------------------------- #
def test_by_default_every_scale_describes_the_ground():
    """Nothing is excluded unless it is declared: the value is averaged over
    the whole mixture and the spread beside it is the mixture's own."""
    mix = _mixture(w=(0.9, 0.1), s1=0.5, s2=3.0)
    sims = tf.zeros([7, 1, 4], tf.float64)
    value, noise_var = mix.integrated_backward(sims)

    assert np.allclose(value.numpy(), 0.0, atol=1e-9)
    # Gauss-Hermite is exact for a quadratic
    assert np.allclose(noise_var.numpy(), 0.9 * 0.25 + 0.1 * 9.0, rtol=1e-6)


def test_declared_contamination_leaves_the_ground_alone():
    """A contaminated reading replaces a measurement rather than reporting
    one, so it has no place in what the ground holds. On a nonlinear warping
    that is a real difference -- integrating a wide component into the value
    biases it the way the warping bends."""
    warped = geoml.warping.ChainedWarping(geoml.warping.Softplus(1),
                                          geoml.warping.ZScore(1))

    def built(contamination):
        mix = geoml.likelihood.Mixture(warped, contamination=contamination)
        mix.parameters["weights"].set_value(np.array([0.9, 0.1]))
        mix.components[0].parameters["noise"].set_value(np.full([1, 1, 1], .25))
        mix.components[1].parameters["noise"].set_value(np.full([1, 1, 1], 9.))
        return mix.integrated_backward(tf.zeros([4, 1, 2], tf.float64))

    ground, ground_spread = built([False, True])
    everything, everything_spread = built(None)

    # the convex warping turns the wide component into inflation of the value
    assert np.all(ground.numpy() < everything.numpy())
    # ... which the spread of a *measurement* keeps, contamination and all
    assert np.all(ground_spread.numpy() > everything_spread.numpy())


def test_the_spread_is_taken_about_the_reported_value():
    """What `spread_check` compares a residual with is `E[(y - prediction)^2]`,
    so the second moment is about the value reported, not about a mean of its
    own: the two differ by exactly the shift the contamination makes."""
    warped = geoml.warping.ChainedWarping(geoml.warping.Softplus(1),
                                          geoml.warping.ZScore(1))

    def built(contamination):
        mix = geoml.likelihood.Mixture(warped, contamination=contamination)
        mix.parameters["weights"].set_value(np.array([0.9, 0.1]))
        mix.components[0].parameters["noise"].set_value(np.full([1, 1, 1], .25))
        mix.components[1].parameters["noise"].set_value(np.full([1, 1, 1], 9.))
        return [x.numpy() for x in
                mix.integrated_backward(tf.zeros([4, 1, 2], tf.float64))]

    ground, ground_spread = built([False, True])
    everything, everything_spread = built(None)

    assert np.allclose(ground_spread,
                       everything_spread + (everything - ground) ** 2,
                       rtol=1e-8)


def test_a_measurement_keeps_the_full_mixture():
    """`measurement_samples` describes a fresh assay, which can be a bad
    one: its spread is the whole mixture's."""
    mix = _mixture(w=(0.9, 0.1), s1=0.5, s2=3.0)
    sims = tf.zeros([5, 1, 1], tf.float64)
    samples = mix.measurement_samples(sims, n_nodes=400).numpy()

    total = 0.9 * 0.25 + 0.1 * 9.0
    spread = samples.var()
    # equal-share nodes truncate the extreme tails, so the spread sits a
    # little under the exact mixture variance -- but far above the inlier's
    assert spread > 0.6 * total
    assert abs(spread - total) / total < 0.25
    assert spread > 4 * 0.25


def test_measurement_nodes_follow_the_mixture_quantile():
    """The bisection must land on the true quantile of the mixture CDF."""
    w, s1, s2 = (0.9, 0.1), 0.5, 3.0
    mix = _mixture(w=w, s1=s1, s2=s2)
    nodes = mix._measurement_values(10).numpy()[:, 0, 0]

    u = (np.arange(10) + 0.5) / 10
    for node, p in zip(nodes, u):
        exact = optimize.brentq(
            lambda t: w[0] * stats.norm.cdf(t, scale=s1)
            + w[1] * stats.norm.cdf(t, scale=s2) - p, -40, 40)
        assert abs(node - exact) < 1e-6


def test_a_robust_zscore_does_not_let_contamination_set_the_scale():
    """`ZScore(size, robust=True)` initializes on a winsorized copy: a
    handful of gross outliers must not inflate the scale everything is
    normalized by, where the plain fit keeps the moments as they come. On
    clean data the two fits agree."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, 300)
    clean = y.copy()
    y[:3] = 500.0

    def fitted_std(warping, values):
        warping.initialize(values[:, None])
        return float(warping.parameters["std"].get_value().numpy()[0])

    assert fitted_std(geoml.warping.ZScore(1, robust=True), y) < 2.0
    assert fitted_std(geoml.warping.ZScore(1), y) > 20.0
    assert abs(fitted_std(geoml.warping.ZScore(1, robust=True), clean)
               - fitted_std(geoml.warping.ZScore(1), clean)) < 0.05


# --------------------------------------------------------------------------- #
# end to end on contaminated data
# --------------------------------------------------------------------------- #
def test_training_finds_the_contamination():
    """Nothing is set by hand: the construction-time separation has to be
    enough for the wide component to claim the outliers."""
    x, y, is_out = _contaminated()
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1, robust=True))
    model, point = _model(mix, x, y)

    weight = float(mix.parameters["weights"].get_value()[1])
    assert 0.01 < weight < 0.25

    resp = model.responsibilities(point)["v"]
    assert resp.shape == (len(y), 2)
    assert np.allclose(resp.sum(axis=1), 1.0)
    # the flagged points are overwhelmingly the planted ones
    assert resp[is_out, 1].mean() > 5 * resp[~is_out, 1].mean()


def test_prediction_runs_end_to_end():
    x, y, _ = _contaminated()
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1))
    model, _ = _model(mix, x, y, max_iter=50)

    grid = geoml.data.Grid1D(start=0.0, n=50, step=10.0 / 49)
    model.predict(grid, n_sim=8)
    for column in ("prediction", "latent_mean", "noise_variance",
                   "dispersion"):
        values = grid.values("v/%s" % column)
        assert values.shape == (50,)
    assert np.all(np.isfinite(grid.values("v/prediction")))


# --------------------------------------------------------------------------- #
# where the responsibilities are filed
# --------------------------------------------------------------------------- #
def test_responsibilities_land_on_the_variable(tmp_path):
    x, y, _ = _contaminated(n=120)
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1))
    model, point = _model(mix, x, y, max_iter=30)

    answer = model.responsibilities(point)
    assert set(answer) == {"v"}

    # one column per component, addressed by its position, and every consumer
    # of the tree has it for nothing
    stored = np.stack([point.values("v/responsibilities/%d" % k)
                       for k in range(2)], axis=1)
    assert np.allclose(stored, answer["v"])
    found = [str(path) for path in point.select("**/responsibilities/*")]
    assert found == ["v/responsibilities/0.0", "v/responsibilities/1.0"]
    assert "v_responsibilities_1.0" in point.as_data_frame().columns

    # and it survives the Zarr round trip, off the same declaration
    path = str(tmp_path / "responsibilities.zarr")
    point.to_zarr(path)
    reopened = geoml.data.PointData.open(path)
    assert np.allclose(reopened.values("v/responsibilities/1"),
                       answer["v"][:, 1])


def test_responsibilities_can_be_read_without_being_stored():
    x, y, _ = _contaminated(n=80)
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1))
    model, point = _model(mix, x, y, max_iter=20)

    model.responsibilities(point, store=False)
    assert len(point.variables["v"].responsibilities) == 0


def test_a_row_without_a_measurement_gets_no_responsibility():
    x, y, _ = _contaminated(n=80)
    y[5] = np.nan
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1))
    model, point = _model(mix, x, y, max_iter=20)

    answer = model.responsibilities(point)["v"]
    assert np.all(np.isnan(answer[5]))
    assert np.all(np.isfinite(answer[6]))


def test_a_vector_variable_gets_one_answer_per_location():
    """The mixture is over the row, so the responsibilities belong to the
    variable rather than to its components -- and a vector mixture trains
    through the joint-sample branch on the way there."""
    rng = np.random.default_rng(7)
    n = 100
    x = rng.uniform(0.0, 10.0, n)
    values = np.stack([np.sin(x), np.cos(x)], axis=1) \
        + rng.normal(0.0, 0.2, (n, 2))

    geoml.set_seed(1234)
    point = geoml.data.PointData(pd.DataFrame({"c0": x}), ["c0"])
    point.add_vector_variable("v", ["a", "b"], values)
    inducing = geoml.data.Grid1D(start=0.0, n=20, step=10.0 / 19)
    gp = geoml.latent.BasicGP(
        geoml.latent.BasicInput(inducing,
                                transform=geoml.transform.Isotropic(1.5)),
        size=2, kernel=geoml.kernels.Gaussian())
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(2), n_components=3)
    model = geoml.models.VGPNetwork(
        point, "v", mix, gp, options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=20)

    answer = model.responsibilities(point)["v"]
    assert answer.shape == (n, 3)          # one row, three components
    assert np.allclose(answer.sum(axis=1), 1.0)
    assert len(point.variables["v"].responsibilities) == 3
    for label in ("a", "b"):
        assert len(point.variables["v"].components[label].responsibilities) == 0


def test_responsibilities_need_a_mixture():
    x, y, _ = _contaminated(n=60)
    model, point = _model(geoml.likelihood.Gaussian(), x, y, max_iter=20)
    with pytest.raises(ValueError, match="mixture"):
        model.responsibilities(point)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_a_saved_mixture_model_reopens(tmp_path):
    x, y, _ = _contaminated()
    mix = geoml.likelihood.Mixture(geoml.warping.ZScore(1), n_components=3,
                                   family="studentt",
                                   contamination=[False, False, True])
    model, _ = _model(mix, x, y, max_iter=30)

    grid = geoml.data.Grid1D(start=0.0, n=30, step=10.0 / 29)
    model.predict(grid, n_sim=4)
    before = np.array(grid.values("v/prediction"), copy=True)

    path = str(tmp_path / "mixture_model")
    model.save(path)
    reopened = geoml.models.VGPNetwork.open(path)

    lik = reopened.likelihoods[0]
    assert isinstance(lik, geoml.likelihood.Mixture)
    assert lik.contamination == [False, False, True]
    assert len(lik.components) == 3
    assert np.allclose(lik.parameters["weights"].get_value().numpy(),
                       mix.parameters["weights"].get_value().numpy())

    grid2 = geoml.data.Grid1D(start=0.0, n=30, step=10.0 / 29)
    reopened.predict(grid2, n_sim=4)
    assert np.allclose(np.array(grid2.values("v/prediction")), before,
                       atol=1e-8)
