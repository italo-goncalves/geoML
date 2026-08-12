"""Exploratory figures.

A picture is hard to assert on, so the arithmetic lives in `plots.prepare` and
is tested against numbers here: which components a share of the variance asks
for, what a composition looks like once it is opened up, which rows count as
measured. The figures themselves are checked for the things that are still
structural -- how many panels, which ones are drawn, what a bad request says --
and for the one promise the style makes, that drawing a geoML figure does not
change how anyone else's figures look.
"""
import copy
import re

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.plots.prepare as prepare


COLS = ["X", "Y"]


@pytest.fixture(autouse=True)
def _close_figures():
    """Every figure here is inspected and dropped; pyplot keeps them all until
    told otherwise, and warns once there are twenty of them."""
    yield
    matplotlib.pyplot.close("all")


@pytest.fixture(scope="module")
def jura():
    train, _ = geoml.datasets.jura()
    return train


def _points(n=60, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    return geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS), rng


@pytest.fixture(scope="module")
def trained():
    """A small trained model, and data carrying its predictions."""
    geoml.set_seed(1234)
    point, rng = _points(n=50)
    values = np.column_stack([
        np.sin(point.coordinates[:, 0] / 30.0),
        np.cos(point.coordinates[:, 1] / 30.0),
        rng.normal(size=50) * 0.1])
    point.add_vector_variable("v", ["a", "b", "c"], values)

    inducing = geoml.data.Grid2D(start=[0, 0], n=[5, 5], step=[25, 25])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(40))
    network = geoml.latent.BasicGP(root, size=3,
                                   kernel=geoml.kernels.Gaussian())

    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.MultivariateGaussian(3), network,
        options=geoml.models.GPOptions(verbose=False, training_samples=5))
    model.train_full(max_iter=6)
    model.predict(point, n_sim=20)
    return model, point


# --------------------------------------------------------------------------- #
# the numbers
# --------------------------------------------------------------------------- #
def test_the_components_kept_reach_the_share_that_was_asked_for():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 2))
    values = np.column_stack([base[:, 0], base[:, 0] * 0.5 + base[:, 1] * 0.1,
                              base[:, 1], rng.normal(size=200) * 0.01])

    analysis = prepare.principal_components(values, explained=0.95)
    cumulative = np.cumsum(analysis["ratio"])

    assert np.isclose(np.sum(analysis["ratio"]), 1.0)
    assert cumulative[analysis["n_components"] - 1] >= 0.95
    assert cumulative[analysis["n_components"] - 2] < 0.95


def test_two_components_are_kept_even_when_one_would_do():
    """One is a line, not a plot. A variable whose variance is essentially one
    direction still gets a second axis to be drawn against."""
    rng = np.random.default_rng(2)
    line = rng.normal(size=(200, 1)) * np.array([[3.0, 1.0, 0.5]])
    values = line + rng.normal(size=(200, 3)) * 1e-4

    analysis = prepare.principal_components(values, explained=0.9)
    assert analysis["ratio"][0] > 0.99
    assert analysis["n_components"] == 2
    assert analysis["scores"].shape == (200, 2)


def test_the_number_of_components_never_exceeds_the_columns():
    rng = np.random.default_rng(3)
    analysis = prepare.principal_components(rng.normal(size=(50, 2)),
                                            explained=1.0)
    assert analysis["n_components"] == 2


def test_a_log_ratio_removes_the_constant_sum():
    rng = np.random.default_rng(4)
    composition = rng.uniform(0.1, 1, (30, 4))
    composition = composition / composition.sum(axis=1, keepdims=True)

    opened = prepare.centred_log_ratio(composition)
    assert np.allclose(opened.sum(axis=1), 0.0)
    # the closure is what a PCA of the raw proportions would spend a component
    # describing: their covariance is singular, the log-ratios' is not
    assert np.linalg.matrix_rank(np.cov(composition.T)) == 3


def test_a_composition_is_opened_up_before_its_pca():
    point, rng = _points()
    composition = rng.uniform(0.1, 1, (60, 3))
    composition = composition / composition.sum(axis=1, keepdims=True)
    point.add_compositional_variable("assay", ["a", "b", "c"], composition)

    through_variable = prepare.component_analysis(point.variables["assay"])
    by_hand = prepare.principal_components(
        prepare.centred_log_ratio(composition))

    assert np.allclose(through_variable["scores"], by_hand["scores"])


def test_a_pca_can_be_taken_of_the_logarithms(jura):
    values, measured, _ = prepare.numeric_values(jura.variables["Elements"])

    logged = prepare.component_analysis(jura.variables["Elements"], log=True)
    by_hand = prepare.principal_components(np.log(values[measured]))

    assert np.allclose(logged["scores"], by_hand["scores"])
    assert logged["labels"][0] == "log(Cd)"
    # and it is a different analysis from the one on the measurements
    plain = prepare.component_analysis(jura.variables["Elements"])
    assert not np.allclose(plain["ratio"], logged["ratio"])
    assert plain["labels"][0] == "Cd"


def test_a_composition_is_opened_up_whether_or_not_logs_were_asked_for():
    """There is no useful PCA of proportions to fall back on: the constant sum
    makes their covariance singular, so the first component would go on
    describing the closure."""
    point, parts = _composition()
    variable = point.variables["assay"]

    without = prepare.component_analysis(variable)
    with_log = prepare.component_analysis(variable, log=True)

    assert np.allclose(without["scores"], with_log["scores"])
    assert without["labels"] == with_log["labels"] == ["clr(a)", "clr(b)",
                                                       "clr(c)"]


def test_a_vector_is_measured_only_where_every_component_is(jura):
    values, measured, labels = prepare.numeric_values(
        jura.variables["Elements"])

    assert values.shape == (jura.n_data, 7)
    assert labels == ["Cd", "Co", "Cr", "Cu", "Ni", "Pb", "Zn"]
    assert measured.shape == (jura.n_data,)
    assert measured.all()


def test_a_category_reads_back_as_its_label(jura):
    values, measured, labels = prepare.category_values(jura.variables["Rock"])

    assert set(labels) <= set(jura.variables["Rock"].labels)
    assert measured.all()
    assert set(values[measured]) == set(labels)
    # every location belongs to exactly one of the groups
    counted = sum(int(mask.sum()) for _, mask
                  in prepare.groups(values, measured, labels))
    assert counted == jura.n_data


def test_an_unmeasured_category_is_not_a_group():
    point, _ = _points(n=6)
    point.add_categorical_variable(
        "rock", labels=["basalt", "granite"],
        measurements=np.array(["granite", None, "basalt", None, None, None]))

    values, measured, labels = prepare.category_values(point.variables["rock"])
    assert list(measured) == [True, False, True, False, False, False]
    assert sorted(labels) == ["basalt", "granite"]


def test_asking_for_a_variable_that_is_not_there_says_what_is(jura):
    with pytest.raises(KeyError, match="Elements"):
        prepare.variable(jura, "Gold")


def test_the_panel_grid_is_as_square_as_it_goes():
    assert prepare.grid_shape(1) == (1, 1)
    assert prepare.grid_shape(4) == (2, 2)
    assert prepare.grid_shape(7) == (3, 3)
    assert prepare.grid_shape(12) == (3, 4)


# --------------------------------------------------------------------------- #
# the figures
# --------------------------------------------------------------------------- #
def _visible(figure):
    return [axes for axes in figure.axes if axes.get_visible()]


def test_a_histogram_has_one_panel_per_component(jura):
    figure = geoml.plots.Explorer(jura, continuous="Elements").histogram()
    assert len(_visible(figure)) == 7


def test_a_pairs_plot_leaves_out_the_half_that_repeats(jura):
    figure = geoml.plots.Explorer(
        jura, continuous="Elements", categorical="Rock").pairs()

    # 7 x 7 panels drawn, but only the diagonal and below are kept
    assert len(figure.axes) == 49
    assert len(_visible(figure)) == 7 * 8 // 2


def test_the_components_can_be_drawn_on_the_datas_own_axes(jura):
    """The reverse of `pca`: instead of the data on the components' axes, the
    components on the data's."""
    explorer = geoml.plots.Explorer(jura, continuous="Elements")

    def lines(figure):
        """The double-headed arrows; `annotate` files them under texts."""
        return sum(1 for panel in figure.axes for artist in panel.texts
                   if isinstance(artist, matplotlib.text.Annotation))

    off_diagonal = 7 * 8 // 2 - 7
    assert lines(explorer.pairs()) == 0
    assert lines(explorer.pairs(principal_components=2)) == 2 * off_diagonal
    assert lines(explorer.pairs(principal_components=3)) == 3 * off_diagonal


def test_a_component_line_is_one_standard_deviation_either_way():
    """Its length is in the data's own units, so a long line means that
    direction carries much of the spread -- no rescaling to the panel."""
    rng = np.random.default_rng(0)
    along = rng.normal(size=200)
    values = np.column_stack([10 + 3 * along, 5 - 3 * along]) \
        + rng.normal(size=(200, 2)) * 1e-3

    analysis = prepare.principal_components(values, explained=1.0)

    assert np.allclose(analysis["mean"], [10, 5], atol=0.5)
    # the spread lies along (1, -1)/sqrt(2), with sd 3*sqrt(2)
    assert np.isclose(np.sqrt(analysis["eigenvalues"][0]), 3 * np.sqrt(2),
                      rtol=0.1)
    assert np.isclose(abs(analysis["loadings"][0, 0]), 1 / np.sqrt(2),
                      rtol=0.05)


def _composition(n=60):
    point, rng = _points(n=n)
    parts = rng.uniform(0.1, 1, (n, 3))
    parts = parts / parts.sum(axis=1, keepdims=True)
    point.add_compositional_variable("assay", ["a", "b", "c"], parts)
    return point, parts


def test_a_compositions_components_are_not_directions_in_its_proportions():
    """They belong to the log-ratios, and drawing them over proportions would
    be a wrong line that nobody would catch."""
    point, _ = _composition()

    with pytest.raises(TypeError, match="log=True"):
        geoml.plots.Explorer(point, continuous="assay").pairs(
            principal_components=2)


def test_on_a_log_scale_a_composition_can_carry_its_components():
    """Drawn as log-ratios, the axes are the space the components live in."""
    point, _ = _composition()
    figure = geoml.plots.Explorer(point, continuous="assay").pairs(
        log=True, principal_components=2)

    assert figure.axes[3].get_xlabel() in ("", "clr(a)")
    assert "clr(a)" in [panel.get_ylabel() for panel in figure.axes]


def test_a_log_scale_says_so_on_the_axes(jura):
    figure = geoml.plots.Explorer(jura, continuous="Elements").pairs(log=True)
    assert "log(Cd)" in [panel.get_ylabel() for panel in figure.axes]


def test_the_log_of_a_composition_is_the_log_ratio():
    point, parts = _composition()
    values, measured, _ = prepare.numeric_values(point.variables["assay"])

    logged = prepare.logarithm(values[measured], compositional=True)
    assert np.allclose(logged, prepare.centred_log_ratio(parts))
    assert np.allclose(logged.sum(axis=1), 0.0)

    plain = prepare.logarithm(values[measured], compositional=False)
    assert np.allclose(plain, np.log(parts))


def test_a_zero_cannot_be_logged_and_says_which_column():
    point, rng = _points(n=20)
    values = rng.uniform(1, 2, (20, 3))
    values[5, 1] = 0.0
    point.add_vector_variable("v", ["a", "b", "c"], values)

    with pytest.raises(ValueError, match="columns 1"):
        geoml.plots.Explorer(point, continuous="v").pairs(log=True)


def test_the_upper_triangle_can_be_filled_in(jura):
    explorer = geoml.plots.Explorer(jura, continuous="Elements",
                                    categorical="Rock")

    assert len([a for a in explorer.pairs().axes if a.get_visible()]) \
        == 7 * 8 // 2
    for filling in ("hist2d", "density", "correlation"):
        figure = explorer.pairs(upper=filling, bins=10)
        assert len([a for a in figure.axes if a.get_visible()]) == 49


def test_the_upper_triangle_pools_the_categories(jura):
    """The lower half says which category a point is; the upper says where the
    data sits, all of it, which a colour-split scatter hides behind whichever
    category was drawn last."""
    explorer = geoml.plots.Explorer(jura, continuous="Elements",
                                    categorical="Rock")
    figure = explorer.pairs(upper="hist2d", bins=10)

    upper_panel = figure.axes[1]                      # row 0, column 1
    assert len(upper_panel.collections) == 1          # one surface, not five


def test_an_unknown_upper_says_what_it_takes(jura):
    with pytest.raises(ValueError, match="hist2d"):
        geoml.plots.Explorer(jura, continuous="Elements").pairs(upper="hexbin")


def test_the_pca_draws_only_the_components_it_kept(jura):
    explorer = geoml.plots.Explorer(jura, continuous="Elements")
    kept = prepare.component_analysis(
        jura.variables["Elements"], explained=0.9)["n_components"]

    figure = explorer.pca(explained=0.9)
    assert len(figure.axes) == kept ** 2


def test_asking_for_more_variance_draws_more_components(jura):
    explorer = geoml.plots.Explorer(jura, continuous="Elements")
    assert len(explorer.pca(explained=0.99).axes) \
        > len(explorer.pca(explained=0.9).axes)


def test_pairing_needs_something_to_pair(jura):
    point, rng = _points()
    point.add_continuous_variable("v", rng.normal(size=60))

    with pytest.raises(TypeError, match="vector variable"):
        geoml.plots.Explorer(point, continuous="v").pairs()


def test_a_variable_with_nothing_measured_says_so():
    point, _ = _points()
    point.add_continuous_variable("v")

    with pytest.raises(ValueError, match="no measurements"):
        geoml.plots.Explorer(point, continuous="v").histogram()


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
def test_a_scene_can_be_coloured_by_one_component(jura):
    figure = geoml.plots.Explorer(jura, continuous="Elements").scene(
        color="Zn")
    # the style puts titles on the left, which is where they are read from
    assert figure.axes[0].get_title(loc="left") =="Zn"


def test_a_vector_is_not_one_colour_scale(jura):
    """Colouring by seven numbers at once cannot mean anything, so the
    categorical variable is drawn instead of the first component quietly
    standing in for the whole variable."""
    figure = geoml.plots.Explorer(
        jura, continuous="Elements", categorical="Rock").scene()
    # the style puts titles on the left, which is where they are read from
    assert figure.axes[0].get_title(loc="left") =="Rock"


def test_a_vector_with_nothing_to_fall_back_on_asks_for_a_component(jura):
    with pytest.raises(ValueError, match="not one colour scale"):
        geoml.plots.Explorer(jura, continuous="Elements").scene()


def test_an_unknown_colour_lists_what_there_is(jura):
    with pytest.raises(KeyError, match="Zn"):
        geoml.plots.Explorer(jura, continuous="Elements").scene(color="Gold")


def test_a_scene_needs_coordinates_it_can_draw():
    rng = np.random.default_rng(5)
    columns = ["a", "b", "c", "d"]
    frame = pd.DataFrame(rng.uniform(size=(20, 4)), columns=columns)
    point = geoml.data.PointData(frame, columns)
    point.add_continuous_variable("v", rng.normal(size=20))

    with pytest.raises(ValueError, match="1, 2 or 3 coordinates"):
        geoml.plots.Explorer(point, continuous="v").scene()


# --------------------------------------------------------------------------- #
# what the model adds
# --------------------------------------------------------------------------- #
def test_the_data_goes_through_the_models_own_warping(trained):
    """Not a warping of the plot's own making: the fitted one the model
    trained with, whose parameters have moved since it was built."""
    model, point = trained
    warped, measured, labels = prepare.warped_values(model, "v")

    values, _, _ = prepare.numeric_values(point.variables["v"])
    expected, _ = model.likelihoods[0].warping.forward(values[measured])

    assert np.allclose(warped, np.asarray(expected))
    # numbered, not named after what was measured: a warping may rotate or
    # bend the data, so a column is generally a mixture of the measurements
    # rather than any one of them, and calling this one "a" would be wrong in
    # a way nobody would catch
    assert labels == ["Variable 1", "Variable 2", "Variable 3"]


def test_only_measured_rows_are_warped(trained):
    """`get_measurements` pads the gaps with 1.0 to keep the array
    rectangular, and a warping owes nothing sensible to a value never taken."""
    model, point = trained
    warped, measured, _ = prepare.warped_values(model, "v")

    assert len(warped) == int(np.sum(measured))
    assert np.all(np.isfinite(warped))


def test_a_likelihood_with_no_warping_has_nothing_to_transform():
    point, _ = _points(n=20)
    point.add_categorical_variable("rock", labels=["a", "b"],
                                   measurements=np.array(["a", "b"] * 10))
    inducing = geoml.data.Grid2D(start=[0, 0], n=[3, 3], step=[50, 50])
    network = geoml.latent.BasicGP(
        geoml.latent.BasicInput(inducing), size=2)
    model = geoml.models.VGPNetwork(
        point, "rock", geoml.likelihood.CategoricalGaussianIndicator(2),
        network, options=geoml.models.GPOptions(verbose=False))

    with pytest.raises(TypeError, match="no warping"):
        prepare.warped_values(model, "rock")


def test_a_variable_the_model_does_not_hold_says_what_it_does(trained):
    model, _ = trained
    with pytest.raises(KeyError, match="it holds v"):
        prepare.warped_values(model, "gold")


def test_a_running_mean_is_drawn_against_its_own_positions():
    position, smoothed = prepare.moving_average(np.arange(10.0), window=3)

    assert len(position) == len(smoothed) == 8
    assert position[0] == 3           # the first mean exists at the third point
    assert np.isclose(smoothed[0], 1.0)


def test_a_window_longer_than_the_log_leaves_it_alone():
    position, smoothed = prepare.moving_average(np.arange(4.0), window=10)
    assert np.array_equal(smoothed, np.arange(4.0))
    assert np.array_equal(position, [1, 2, 3, 4])


def test_a_curve_needs_a_model_that_has_trained():
    point, _ = _points(n=10)
    point.add_continuous_variable("v", np.zeros(10))
    inducing = geoml.data.Grid2D(start=[0, 0], n=[3, 3], step=[50, 50])
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(),
        geoml.latent.BasicGP(geoml.latent.BasicInput(inducing), size=1),
        options=geoml.models.GPOptions(verbose=False))

    with pytest.raises(ValueError, match="not been trained"):
        prepare.training_curve(model)


def test_the_model_figures_say_when_there_is_no_model(jura):
    explorer = geoml.plots.Explorer(jura, continuous="Elements")
    with pytest.raises(ValueError, match="needs a model"):
        explorer.training_curve()
    with pytest.raises(ValueError, match="needs a model"):
        explorer.transformed_pairs()


def test_the_transformed_pairs_keep_the_lower_triangle(trained):
    model, point = trained
    figure = geoml.plots.Explorer(
        point, continuous="v", model=model).transformed_pairs()

    assert len(figure.axes) == 9
    assert len(_visible(figure)) == 3 * 4 // 2


def test_the_transformed_pairs_can_fill_the_upper_triangle_too(trained):
    """Same filling as `pairs`, and here `density` is the pointed one: a pair
    the warping has settled has round, centred contours."""
    model, point = trained
    explorer = geoml.plots.Explorer(point, continuous="v", model=model)

    assert len(_visible(explorer.transformed_pairs())) == 3 * 4 // 2
    for filling in ("hist2d", "density", "correlation"):
        figure = explorer.transformed_pairs(upper=filling, bins=8)
        assert len(_visible(figure)) == 9
        # the upper panels borrow the scales written along the edges
        assert figure.axes[1].get_xticklabels() == [] \
            or all(not label.get_text()
                   for label in figure.axes[1].get_xticklabels())


def test_the_training_curve_overlays_a_running_mean(trained):
    model, _ = trained
    figure = geoml.plots.Explorer(model.data, model=model).training_curve(
        window=2)

    # the log itself, and the mean over it
    assert len(figure.axes[0].lines) == 2


def test_a_short_log_gets_no_second_line(trained):
    model, _ = trained
    figure = geoml.plots.Explorer(model.data, model=model).training_curve(
        window=len(model.training_log) + 5)
    assert len(figure.axes[0].lines) == 1


# --------------------------------------------------------------------------- #
# measurements against simulations
# --------------------------------------------------------------------------- #
def test_the_simulations_are_thinned_by_one_stride_for_every_component(
        trained):
    """The pairs have to stay pairs. Thinning each component on its own would
    put the value simulated at one location beside the value simulated at
    another, and the joint shape -- the whole point of the figure -- would be
    an artefact of the thinning."""
    _, point = trained
    variable = point.variables["v"]

    sample = prepare.simulation_sample(variable, most=200)
    full = np.stack([np.asarray(variable.components[label].get_simulations())
                     .reshape(-1) for label in variable.labels], axis=1)
    stride = max(1, len(full) // 200)

    assert np.allclose(sample, full[::stride])
    assert sample.shape[1] == 3


def test_the_sample_is_about_as_large_as_it_was_asked_to_be(trained):
    _, point = trained
    variable = point.variables["v"]
    # a vector variable reports n_sim 0 -- the simulations belong to its
    # components -- so the count comes from one of those
    total = point.n_data * variable.components["a"].n_sim

    assert len(prepare.simulation_sample(variable, most=100)) <= total
    assert len(prepare.simulation_sample(variable, most=100)) >= 100
    # asking for more than there is gives everything, not an error
    assert len(prepare.simulation_sample(variable, most=10 ** 9)) == total


def test_a_variable_with_no_simulations_says_so():
    point, rng = _points(n=20)
    point.add_vector_variable("v", ["a", "b"], rng.normal(size=(20, 2)))

    with pytest.raises(ValueError, match="no simulations"):
        prepare.simulation_sample(point.variables["v"])


def test_the_comparison_fills_the_whole_matrix(trained):
    _, point = trained
    figure = geoml.plots.Explorer(point,
                                  continuous="v").simulation_pairs(most=500)

    assert len(_visible(figure)) == 9
    # counted into cells by default: simulations are too many to draw one by one
    assert len(figure.axes[1].collections) == 1


def test_both_halves_of_the_comparison_share_their_scales(trained):
    """A panel and its mirror have to be on the same axes, or the two halves
    cannot be read against each other at all."""
    _, point = trained
    figure = geoml.plots.Explorer(point,
                                  continuous="v").simulation_pairs(most=500)

    below = figure.axes[3]        # row 1, column 0
    above = figure.axes[1]        # row 0, column 1
    assert np.allclose(below.get_xlim(), above.get_ylim())
    assert np.allclose(below.get_ylim(), above.get_xlim())


def test_the_diagonal_carries_the_measured_bars_and_the_simulated_line(
        trained):
    _, point = trained
    figure = geoml.plots.Explorer(point,
                                  continuous="v").simulation_pairs(most=500)
    diagonal = figure.axes[0]

    assert len(diagonal.patches) > 0        # the measured histogram
    assert len(diagonal.lines) == 1         # the simulated density
    assert [line.get_label() for line in diagonal.lines] == ["simulated"]


def test_the_window_opens_the_measured_range_by_a_share_of_itself():
    values = np.column_stack([np.array([0.0, 10.0, 5.0]),
                              np.array([2.0, 2.0, 2.0])])
    (first, second) = prepare.padded_range(values, margin=0.1)

    assert first == (-1.0, 11.0)
    # a column with no span to open gets room anyway, not a window of nothing
    assert second == (1.0, 3.0)


def test_simulations_beyond_the_measured_range_are_left_out(trained):
    """A few realizations reaching far past anything measured would set the
    scale for every panel and squeeze the comparison into a corner."""
    _, point = trained
    variable = point.variables["v"]

    # push one simulated value of component "a" far away
    simulations = np.asarray(variable.components["a"].get_simulations())
    simulations[0, 0] = 1000.0
    variable.components["a"].simulations[:, :] = simulations

    figure = geoml.plots.Explorer(point, continuous="v").simulation_pairs(
        most=500, margin=0.1)

    measured, _, _ = prepare.numeric_values(variable)
    low, high = prepare.padded_range(measured[:, [0]], 0.1)[0]
    assert figure.axes[0].get_xlim()[1] < 1000.0
    assert np.isclose(figure.axes[0].get_xlim()[1], high, rtol=0.05)
    assert np.isclose(figure.axes[0].get_xlim()[0], low, rtol=0.05)


def test_a_runaway_in_one_component_keeps_the_others(trained):
    """The window is applied panel by panel, so a realization that runs away
    in one component still has something to say about the rest."""
    _, point = trained
    sample = prepare.simulation_sample(point.variables["v"], most=500)
    limits = prepare.padded_range(
        prepare.numeric_values(point.variables["v"])[0], 0.1)

    inside = np.column_stack([(sample[:, i] >= low) & (sample[:, i] <= high)
                              for i, (low, high) in enumerate(limits)])
    both = inside[:, 0] & inside[:, 1]
    every = np.all(inside, axis=1)

    assert both.sum() >= every.sum()


def test_a_sparse_half_is_not_binned_like_a_dense_one():
    """A few hundred measurements against a hundred thousand simulated values:
    binning both alike leaves the sparse half as scattered single counts."""
    assert prepare.cells(259, 60) < prepare.cells(100000, 60)
    assert prepare.cells(100000, 60) == 60      # capped by the request
    assert prepare.cells(4, 60) == 10           # and floored


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #
def test_a_single_component_is_drawn_with_its_margins(trained):
    _, point = trained
    figure = geoml.plots.Explorer(point, continuous="v").prediction_scatter(
        component="b")

    # the scatter, and a histogram on each side
    assert len(figure.axes) == 3


def test_several_components_are_drawn_as_panels(trained):
    _, point = trained
    figure = geoml.plots.Explorer(point, continuous="v").prediction_scatter()
    assert len(_visible(figure)) == 3


def test_asking_for_a_component_that_is_not_there(trained):
    _, point = trained
    with pytest.raises(KeyError, match="found a, b, c"):
        geoml.plots.Explorer(point, continuous="v").prediction_scatter(
            component="z")


def test_a_prediction_is_needed_before_it_can_be_drawn():
    """A variable carries an empty prediction column from the moment it is
    built, so this is what forgetting to run the model looks like."""
    point, rng = _points(n=20)
    point.add_vector_variable("v", ["a", "b"], rng.normal(size=(20, 2)))

    with pytest.raises(ValueError, match="run the model over this data"):
        geoml.plots.Explorer(point, continuous="v").prediction_scatter()


def test_the_accuracy_plot_draws_the_reference_and_a_line_per_component(
        trained):
    model, point = trained
    figure = geoml.plots.Explorer(point, continuous="v",
                                  model=model).accuracy()

    assert len(figure.axes[0].lines) == 1 + 3
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert labels[0] == "perfect"
    assert all("G = " in label for label in labels[1:])


@pytest.fixture(scope="module")
def trained_composition():
    """A trained model on a *composition*.

    Worth having beside `trained`: a composition's parts are `_Component`,
    which is a different class from the components a vector variable holds,
    with an `update` of its own. Anything a prediction writes has to be
    carried through `CompositionalVariable.update` by hand, and forgetting is
    silent -- the column is simply never filled.
    """
    geoml.set_seed(1234)
    point, rng = _points(n=40)
    point.add_compositional_variable(
        "assay", ["a", "b", "c"], rng.dirichlet([4.0, 2.0, 1.0], size=40))

    warping = geoml.warping.ChainedWarping(
        geoml.warping.CenteredLogRatio(3),
        geoml.warping.RobustPCA(3, 2),
        geoml.warping.ZScore(2))
    inducing = geoml.data.Grid2D(start=[0, 0], n=[4, 4], step=[30, 30])
    network = geoml.latent.BasicGP(
        geoml.latent.BasicInput(inducing,
                                transform=geoml.transform.Isotropic(40)),
        size=2, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "assay", geoml.likelihood.MultivariateGaussian(3, warping),
        network,
        options=geoml.models.GPOptions(verbose=False, training_samples=5))
    model.train_full(max_iter=5)
    model.predict(point, n_sim=10)
    return model, point


def test_a_compositions_parts_each_carry_a_noise_variance(trained_composition):
    _, point = trained_composition

    for label in point.variables["assay"].labels:
        part = point.variables["assay"].components[label]
        assert part.noise_variance._has_content()
        assert np.all(part.noise_variance.values.to_numpy() > 0)


def test_a_compositions_parts_each_carry_a_dispersion(trained_composition):
    """And only where the container discretizes: a point has no interior, so
    there the answer stays missing rather than becoming zero."""
    model, point = trained_composition
    blocks = geoml.data.Blocks2D(start=[0, 0], n=[3, 3], step=[30, 30],
                                 discretization=[2, 2])
    model.predict(blocks, n_sim=6)

    for label in blocks.variables["assay"].labels:
        part = blocks.variables["assay"].components[label]
        assert part.dispersion._has_content()
        assert np.all(part.dispersion.values.to_numpy() >= 0)

    on_points = point.variables["assay"].components["a"]
    assert not on_points.dispersion._has_content()


def test_subsetting_a_composition_carries_every_column(trained_composition):
    """A held-out set is a subset, and a column left at its old length only
    goes wrong later, in whatever reads it against the new one."""
    _, point = trained_composition
    kept = point[np.arange(0, point.n_data, 2)]

    for label in kept.variables["assay"].labels:
        part = kept.variables["assay"].components[label]
        for role in part._ZARR_ATTRS:
            column = getattr(part, role)
            # a part has no latent space of its own, and says so with a None
            if column is not None:
                assert len(column.values) == kept.n_data, role
        assert part.noise_variance._has_content()


def test_the_spread_check_works_on_a_composition(trained_composition):
    _, point = trained_composition
    figure = geoml.plots.Explorer(point, continuous="assay").spread_check(
        bins=3)

    assert len([ax for ax in figure.axes if ax.lines]) == 3


def test_a_per_bin_value_is_drawn_as_a_step():
    """One number per bin is not a curve, and a line through the centres says
    it is."""
    x, y = prepare.step_path([0, 1], [1, 2], [10, 20])
    assert list(x) == [0, 1, 1, 2]
    assert list(y) == [10, 10, 20, 20]


def test_a_gap_between_bins_is_a_break_rather_than_a_line_across_it():
    x, y = prepare.step_path([0, 3], [1, 4], [10, 20])
    assert np.isnan(x[2]) and np.isnan(y[2])


def test_the_spread_check_bins_hold_the_same_count(trained):
    """A count asks for equal-count bins: a predicted grade is skewed, and
    equal width would leave the top bins with a sample each."""
    _, point = trained
    panels = prepare.spread_check(point, "v", bins=5)

    assert [panel["label"] for panel in panels] == ["a", "b", "c"]
    for panel in panels:
        assert panel["count"].sum() == point.n_data
        assert panel["count"].max() - panel["count"].min() <= 1


def test_the_spread_check_takes_the_bin_positions(trained):
    _, point = trained
    predicted = point.variables["v"].components["a"] \
        .prediction.values.to_numpy()
    edges = [predicted.min(), float(np.median(predicted)), predicted.max()]

    panel = prepare.spread_check(point, "v", bins=edges)[0]
    assert np.allclose(panel["lo"], edges[:2])
    assert np.allclose(panel["hi"], edges[1:])


def test_the_spread_check_measures_the_residuals_it_bins(trained):
    _, point = trained
    panel = prepare.spread_check(point, "v", bins=2)[0]
    part = point.variables["v"].components["a"]
    measured = part.measurements.values.to_numpy()
    predicted = part.prediction.values.to_numpy()

    inside = (predicted >= panel["lo"][0]) & (predicted < panel["hi"][0])
    expected = np.sqrt(np.mean((measured[inside] - predicted[inside]) ** 2))
    assert panel["observed"][0] == pytest.approx(expected)
    assert panel["observed_error"][0] == pytest.approx(
        expected / np.sqrt(2 * panel["count"][0]))


def test_the_claimed_total_is_never_below_its_noise_part(trained):
    """It is the noise and the model's own uncertainty added in quadrature,
    so the band can only sit under the line."""
    _, point = trained
    for panel in prepare.spread_check(point, "v", bins=4):
        assert np.all(panel["total"] >= panel["noise"] - 1e-12)


def test_the_spread_check_needs_the_noise_to_have_been_integrated(trained):
    model, point = trained
    latent = geoml.data.PointData.from_array(np.asarray(point.coordinates))
    model.predict(latent, n_sim=5, include_noise=False)

    with pytest.raises(ValueError, match="carries no noise variance"):
        prepare.spread_check(latent, "v")


def test_the_spread_check_draws_a_panel_per_component(trained):
    """And needs no model: everything it reads is on the container."""
    _, point = trained
    figure = geoml.plots.Explorer(point, continuous="v").spread_check(bins=4)

    drawn = [ax for ax in figure.axes if ax.lines]
    assert len(drawn) == 3
    for ax in drawn:
        assert len(ax.lines) >= 2            # the claim, and the observed
        assert len(ax.collections) >= 1      # the noise band


def test_the_spread_check_puts_its_key_on_an_axes_of_its_own(trained):
    """Three entries explaining a band, a line and a set of points cover the
    curve they explain if they sit inside a panel."""
    _, point = trained
    figure = geoml.plots.Explorer(point, continuous="v").spread_check(bins=4)

    key = figure.axes[0]
    assert not key.axison                    # nothing but the legend on it
    assert not key.lines
    assert [text.get_text() for text in key.get_legend().get_texts()] == [
        "claimed: measurement noise", "claimed: noise and uncertainty",
        "observed: rms residual"]
    assert all(ax.get_legend() is None for ax in figure.axes[1:])


def test_the_accuracy_plot_needs_the_model(trained):
    """Its intervals are of a measurement, and only the model can say what a
    measurement would read -- the container holds the ground."""
    _, point = trained
    with pytest.raises(ValueError, match="accuracy needs a model"):
        geoml.plots.Explorer(point, continuous="v").accuracy()


def test_the_accuracy_intervals_are_the_models_not_the_containers(trained):
    """The stored simulations have the noise integrated out, so scoring them
    against measured values reads as wild over-confidence. Asking the model
    what a sample would read is the same question the plot always meant."""
    model, point = trained
    figure = geoml.plots.Explorer(point, continuous="v",
                                  model=model).accuracy()
    from_model = [float(line.get_label().split("G = ")[1].rstrip(")"))
                  for line in figure.axes[0].lines[1:]]

    stored = []
    for label in point.variables["v"].labels:
        part = point.variables["v"].components[label]
        measured = part.measurements.values.to_numpy().astype(float)
        nominal, observed = geoml.metrics.coverage(
            measured, np.asarray(part.get_simulations(), dtype=float))
        stored.append(geoml.metrics.goodness(nominal, observed))

    assert all(a > b for a, b in zip(from_model, stored))


def test_the_measurement_samples_are_built_once(trained):
    """Every component wants the same array, and a figure redrawn wants it
    again; nothing of it reaches the container."""
    model, point = trained
    explorer = geoml.plots.Explorer(point, continuous="v", model=model)

    calls = []
    original = type(model).predict_measurements

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    type(model).predict_measurements = spy
    try:
        explorer.accuracy()
        explorer.accuracy()
    finally:
        type(model).predict_measurements = original

    assert len(calls) == 1


def test_the_points_can_be_counted_into_cells_instead(trained):
    """A block model has more points than a scatter can show, whatever its
    transparency."""
    _, point = trained
    figure = geoml.plots.Explorer(point, continuous="v").prediction_scatter(
        component="b", kind="hist2d", bins=10)

    main = figure.axes[0]
    assert len(main.collections) == 1        # the mesh, not a cloud of points
    assert main.collections[0].get_cmap().name == geoml.plots.SEQUENTIAL


def test_transparency_is_the_users_to_set(trained):
    _, point = trained
    figure = geoml.plots.Explorer(point, continuous="v").prediction_scatter(
        component="b", alpha=0.15)
    assert figure.axes[0].collections[0].get_alpha() == 0.15


def test_only_the_two_kinds_are_accepted(trained):
    _, point = trained
    with pytest.raises(ValueError, match="scatter"):
        geoml.plots.Explorer(point, continuous="v").prediction_scatter(
            kind="hexbin")


# --------------------------------------------------------------------------- #
# grade and tonnage
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def blocks():
    """A small block model carrying simulations of a grade and a density."""
    geoml.set_seed(7)
    grid = geoml.data.Grid2D(start=[0, 0], n=[10, 10], step=[2.0, 5.0])
    rng = np.random.default_rng(3)

    grade = grid.coordinates[:, 0] + rng.normal(size=grid.n_data)
    grid.add_continuous_variable("grade", grade)
    grid.variables["grade"].allocate_simulations(4)
    grid.variables["grade"].simulations[:, :] = \
        grade[:, None] + rng.normal(size=(grid.n_data, 4)) * 0.1

    density = np.full(grid.n_data, 2.0)
    grid.add_continuous_variable("rho", density)
    grid.variables["rho"].allocate_simulations(4)
    grid.variables["rho"].simulations[:, :] = \
        density[:, None] + rng.normal(size=(grid.n_data, 4)) * 0.01

    grid.add_metadata("fixed_density", np.full(grid.n_data, 3.0))
    return grid


def test_without_a_density_the_curve_is_in_volume(blocks):
    curves = prepare.grade_tonnage(blocks, "grade", cutoffs=5)

    # a two-dimensional grid gives an area, and the axis has to say so
    assert curves["unit"] == "area"
    cell = 2.0 * 5.0
    assert np.allclose(curves["tonnage"][0], blocks.n_data * cell)


def test_a_density_turns_the_volume_into_a_mass(blocks):
    volume = prepare.grade_tonnage(blocks, "grade", cutoffs=5)
    mass = prepare.grade_tonnage(blocks, "grade", density="fixed_density",
                                 cutoffs=5)

    assert mass["unit"] == "mass"
    assert np.allclose(mass["tonnage"], 3.0 * volume["tonnage"])


def test_a_number_serves_as_a_density(blocks):
    by_name = prepare.grade_tonnage(blocks, "grade", density="fixed_density",
                                    cutoffs=5)
    by_value = prepare.grade_tonnage(blocks, "grade", density=3.0,
                                     cutoffs=5)
    assert np.allclose(by_name["tonnage"], by_value["tonnage"])


def test_an_uncertain_density_is_matched_realization_by_realization(blocks):
    """Simulation `i` of the density belongs with simulation `i` of the grade;
    pairing them any other way invents a correlation nobody modelled."""
    curves = prepare.grade_tonnage(blocks, "grade", density="rho",
                                   cutoffs=4)

    grade = np.asarray(blocks.variables["grade"].get_simulations())
    density = np.asarray(blocks.variables["rho"].get_simulations())
    cell = 2.0 * 5.0

    for i, cutoff in enumerate(curves["cutoff"]):
        above = grade >= cutoff
        expected = np.sum(np.where(above, cell * density, 0.0), axis=0)
        assert np.allclose(curves["tonnage"][i], expected)


def test_a_density_with_the_wrong_number_of_realizations_is_refused(blocks):
    blocks.add_continuous_variable("rho2", np.full(blocks.n_data, 2.0))
    blocks.variables["rho2"].allocate_simulations(3)
    blocks.variables["rho2"].simulations[:, :] = 2.0

    with pytest.raises(ValueError, match="matched one to one"):
        prepare.grade_tonnage(blocks, "grade", density="rho2", cutoffs=3)


def test_the_grade_above_a_cutoff_is_weighted_by_what_carries_it(blocks):
    curves = prepare.grade_tonnage(blocks, "grade", cutoffs=4)
    grade = np.asarray(blocks.variables["grade"].get_simulations())

    for i, cutoff in enumerate(curves["cutoff"]):
        above = grade[:, 0] >= cutoff
        if above.sum() == 0:
            continue
        assert np.isclose(curves["grade"][i, 0], grade[above, 0].mean())


def test_the_tonnage_only_falls_as_the_cutoff_climbs(blocks):
    curves = prepare.grade_tonnage(blocks, "grade", density="rho")
    assert np.all(np.diff(curves["tonnage"], axis=0) <= 1e-9)


def test_a_number_of_cutoffs_spreads_them_over_the_range(blocks):
    grade = np.asarray(blocks.variables["grade"].get_simulations())
    curves = prepare.grade_tonnage(blocks, "grade", cutoffs=7)

    assert len(curves["cutoff"]) == 7
    assert np.isclose(curves["cutoff"][0], np.min(grade))
    assert np.isclose(curves["cutoff"][-1], np.max(grade))
    assert np.allclose(np.diff(curves["cutoff"]), np.diff(curves["cutoff"])[0])


def test_the_cutoffs_can_be_given_outright(blocks):
    curves = prepare.grade_tonnage(blocks, "grade", cutoffs=[1.0, 5.0, 9.0])
    assert list(curves["cutoff"]) == [1.0, 5.0, 9.0]


def test_the_tonnage_can_be_read_on_a_log_scale(blocks):
    """Most of a deposit clears the low cut-offs, so the high ones are a flat
    line along the bottom until the axis is logarithmic. The grade axis spans
    one order of magnitude at most and is left alone."""
    explorer = geoml.plots.Explorer(blocks, continuous="grade")

    linear, _ = explorer.grade_tonnage(density="rho").axes
    tonnage_axes, grade_axes = explorer.grade_tonnage(
        density="rho", log_mass=True).axes

    assert linear.get_yscale() == "linear"
    assert tonnage_axes.get_yscale() == "log"
    assert grade_axes.get_yscale() == "linear"


def test_each_scale_gets_its_own_gridlines(blocks):
    """One grid on a twin axis belongs to the left scale without saying so,
    and a reader following the grade curve to a line reads a tonnage."""
    figure = geoml.plots.Explorer(blocks, continuous="grade").grade_tonnage(
        density="rho")
    tonnage_axes, grade_axes = figure.axes

    tonnage_color = matplotlib.colors.to_rgb(geoml.plots.PALETTE[0])
    grade_color = matplotlib.colors.to_rgb(geoml.plots.PALETTE[1])

    assert matplotlib.colors.to_rgb(
        tonnage_axes.yaxis.get_gridlines()[0].get_color()) == tonnage_color
    assert matplotlib.colors.to_rgb(
        grade_axes.yaxis.get_gridlines()[0].get_color()) == grade_color
    # the cut-off is shared by both, so its lines belong to neither
    assert matplotlib.colors.to_rgb(
        tonnage_axes.xaxis.get_gridlines()[0].get_color()) \
        not in (tonnage_color, grade_color)


def test_a_grade_tonnage_needs_blocks_to_measure():
    """Points have no size, so there is no tonnage to be had from them --
    whether the sizes come one per model or one per block."""
    point, rng = _points(n=20)
    point.add_continuous_variable("v", rng.normal(size=20))

    with pytest.raises(TypeError, match="how big a block is"):
        prepare.grade_tonnage(point, "v")


def test_a_cutoff_applies_to_one_grade(trained):
    _, point = trained
    with pytest.raises(ValueError, match="name one with component="):
        geoml.plots.Explorer(point, continuous="v").grade_tonnage()


def test_the_curves_are_drawn_on_two_scales(blocks):
    figure = geoml.plots.Explorer(blocks, continuous="grade").grade_tonnage(
        density="rho")

    assert len(figure.axes) == 2          # tonnage left, grade right
    # the four realizations, and the median drawn over them, on each scale
    assert len(figure.axes[0].lines) == 4 + 1
    assert len(figure.axes[1].lines) == 4 + 1


def test_an_unknown_density_says_where_it_looked(blocks):
    with pytest.raises(KeyError, match="fixed_density"):
        prepare.grade_tonnage(blocks, "grade", density="not_there")


def test_the_cells_can_be_coloured_by_the_log_of_the_count(jura):
    """A skewed variable puts most of its data in a few cells, and on a linear
    scale those take the whole colour range while the rest reads as empty."""
    explorer = geoml.plots.Explorer(jura, continuous="Elements")

    linear = explorer.pairs(kind="hist2d", bins=8)
    logged = explorer.pairs(kind="hist2d", bins=8, log_counts=True)

    assert not isinstance(linear.axes[7].collections[0].norm,
                          matplotlib.colors.LogNorm)
    assert isinstance(logged.axes[7].collections[0].norm,
                      matplotlib.colors.LogNorm)


def test_counting_into_cells_pools_the_categories(jura):
    """One surface cannot show five rock types, so it shows all of them and
    says nothing about which is which."""
    explorer = geoml.plots.Explorer(jura, continuous="Elements",
                                    categorical="Rock")

    assert len(explorer.pairs(kind="hist2d", bins=8).legends) == 0
    assert len(explorer.pairs().legends) == 1


def test_every_matrix_takes_the_same_options(jura, trained):
    model, point = trained
    assert geoml.plots.Explorer(jura, continuous="Elements").pca(
        kind="hist2d", bins=8) is not None
    assert geoml.plots.Explorer(
        point, continuous="v", model=model).transformed_pairs(
            kind="hist2d", bins=8) is not None


def test_the_variance_share_is_named_to_a_decimal(jura):
    figure = geoml.plots.Explorer(jura, continuous="Elements").pca()
    assert "%" in figure.axes[0].get_ylabel()
    assert re.search(r"PC1 \(\d+\.\d%\)", figure.axes[0].get_ylabel())


# --------------------------------------------------------------------------- #
# containers that carry no measurements
# --------------------------------------------------------------------------- #
def test_a_grid_predicted_onto_has_nothing_to_check_against(trained):
    """The case this guards: a block model carries simulations everywhere and
    observations nowhere."""
    model, _ = trained
    grid = geoml.data.Grid2D(start=[0, 0], n=[6, 6], step=[20, 20])
    model.predict(grid, n_sim=5)

    explorer = geoml.plots.Explorer(grid, continuous="v", model=model)
    with pytest.raises(ValueError, match="no measurements here"):
        explorer.accuracy()
    with pytest.raises(ValueError, match="no measurements here"):
        explorer.histogram()
    with pytest.raises(ValueError, match="no measurements here"):
        explorer.pairs()


def test_transformed_pairs_needs_something_measured(trained):
    model, _ = trained
    grid = geoml.data.Grid2D(start=[0, 0], n=[6, 6], step=[20, 20])
    model.predict(grid, n_sim=5)

    with pytest.raises(ValueError, match="no measurements here"):
        geoml.plots.Explorer(grid, continuous="v",
                             model=model).transformed_pairs()


# --------------------------------------------------------------------------- #
# leaving out what the model is unsure of
# --------------------------------------------------------------------------- #
def test_blocks_can_be_left_out_by_their_uncertainty(blocks):
    doubt = np.linspace(0.0, 1.0, blocks.n_data)
    blocks.add_metadata("doubt", doubt)

    everything = prepare.grade_tonnage(blocks, "grade", cutoffs=4)
    filtered = prepare.grade_tonnage(blocks, "grade", cutoffs=4,
                                     uncertainty="doubt", max_uncertainty=0.5)

    assert everything["kept"] == everything["total"] == blocks.n_data
    assert filtered["kept"] == int(np.sum(doubt <= 0.5))
    assert filtered["total"] == blocks.n_data
    # less material once the doubtful blocks stop counting
    assert np.all(filtered["tonnage"][0] < everything["tonnage"][0])


def test_the_uncertainty_may_be_an_attribute_of_the_variable(blocks):
    """`latent_variance` on a continuous variable, `uncertainty` on a vector
    one -- which it is depends on what was modelled."""
    variance = np.linspace(0.0, 2.0, blocks.n_data)
    blocks.variables["grade"].latent_variance.values[:] = variance

    curves = prepare.grade_tonnage(blocks, "grade", cutoffs=3,
                                   uncertainty="latent_variance",
                                   max_uncertainty=1.0)
    assert curves["kept"] == int(np.sum(variance <= 1.0))


def test_the_uncertainty_may_belong_to_the_variable_that_contains_the_grade(
        trained):
    """The ordinary case for a vector variable: a component has
    `latent_variance` set to None and no uncertainty of its own, so the column
    that exists is the parent's."""
    model, point = trained
    grid = geoml.data.Grid2D(start=[0, 0], n=[8, 8], step=[12, 12])
    model.predict(grid, n_sim=6)

    doubt = np.linspace(0.0, 1.0, grid.n_data)
    grid.variables["v"].uncertainty.values[:] = doubt

    # the component carries a latent_variance of its own, but nothing wrote
    # one, and an empty column must not stop the search short of the parent
    component = grid.variables["v"].components["a"]
    assert np.all(np.isnan(component.latent_variance.values.to_numpy()))

    curves = prepare.grade_tonnage(grid, "a", cutoffs=3,
                                   uncertainty="uncertainty",
                                   max_uncertainty=0.5)
    assert curves["kept"] == int(np.sum(doubt <= 0.5))


def test_a_path_says_which_variable_to_read_it_from(trained):
    model, point = trained
    grid = geoml.data.Grid2D(start=[0, 0], n=[8, 8], step=[12, 12])
    model.predict(grid, n_sim=6)
    grid.variables["v"].uncertainty.values[:] = \
        np.linspace(0.0, 1.0, grid.n_data)

    by_search = prepare.grade_tonnage(grid, "a", cutoffs=3,
                                      uncertainty="uncertainty",
                                      max_uncertainty=0.5)
    by_name = prepare.grade_tonnage(grid, "a", cutoffs=3,
                                    uncertainty="v/uncertainty",
                                    max_uncertainty=0.5)
    assert by_name["kept"] == by_search["kept"]

    # a wrong path is answered with what is there
    with pytest.raises(KeyError, match="'v' holds"):
        prepare.grade_tonnage(grid, "a", uncertainty="v/nonsense",
                              max_uncertainty=0.5)

    # the dotted form is refused with the replacement spelled out, never
    # silently reinterpreted: a `.` inside a label would make a wrong guess
    # look like a working one
    with pytest.raises(KeyError, match="use the path 'v/uncertainty'"):
        prepare.grade_tonnage(grid, "a", uncertainty="v.uncertainty",
                              max_uncertainty=0.5)


def test_the_values_themselves_can_be_handed_over(blocks):
    """The way out of every case a name does not reach."""
    doubt = np.linspace(0.0, 1.0, blocks.n_data)
    curves = prepare.grade_tonnage(blocks, "grade", cutoffs=3,
                                   uncertainty=doubt, max_uncertainty=0.25)

    assert curves["kept"] == int(np.sum(doubt <= 0.25))

    with pytest.raises(ValueError, match="one value per location"):
        prepare.grade_tonnage(blocks, "grade", uncertainty=doubt[:5],
                              max_uncertainty=0.25)


def test_a_threshold_with_no_column_to_read_says_so(blocks):
    with pytest.raises(ValueError, match="no uncertainty column"):
        prepare.grade_tonnage(blocks, "grade", max_uncertainty=0.5)


def test_a_threshold_nothing_clears_says_what_the_smallest_was(blocks):
    blocks.add_metadata("doubt2", np.full(blocks.n_data, 5.0))
    with pytest.raises(ValueError, match="smallest"):
        prepare.grade_tonnage(blocks, "grade", uncertainty="doubt2",
                              max_uncertainty=1.0)


def test_the_explorer_carries_the_uncertainty_column(blocks):
    blocks.add_metadata("doubt3", np.linspace(0.0, 1.0, blocks.n_data))
    explorer = geoml.plots.Explorer(blocks, continuous="grade",
                                    uncertainty="doubt3")

    figure = explorer.grade_tonnage(max_uncertainty=0.5)
    assert "doubt3 <= 0.5" in figure.axes[0].get_title(loc="left")
    # and without a threshold nothing is left out, so nothing is said
    assert "doubt3" not in explorer.grade_tonnage().axes[0].get_title(
        loc="left")


# --------------------------------------------------------------------------- #
# reading the simulations a band at a time
# --------------------------------------------------------------------------- #
def _chunk(variable, tmp_path, name, rows):
    """Put a variable's simulations on disk in chunks of `rows`.

    A block model's simulations run to hundreds of gigabytes and arrive
    chunked; a fixture small enough to stay in RAM as one piece is the one
    case that would not catch a band being read wrong.
    """
    values = np.asarray(variable.simulations)
    variable.simulations = geoml.storage.ArrayStore.allocate(
        values.shape, backend="zarr", store=str(tmp_path / name),
        chunks=(rows, values.shape[1]))
    variable.simulations[:] = values
    return variable


@pytest.fixture
def streamed():
    """A block model to re-chunk -- so, unlike `blocks`, one per test.

    Some blocks are left unvalued: a cut-off has to place those nowhere, and
    the reduction that finds the range of the data has to step over them. Two
    whole bands are emptied as well, one by having no values in it and one by
    the uncertainty filter -- on a real block model a band of blocks outside
    the domain is the ordinary case, not the corner one.
    """
    geoml.set_seed(11)
    grid = geoml.data.Grid2D(start=[0, 0], n=[10, 10], step=[2.0, 5.0])
    rng = np.random.default_rng(5)

    grade = grid.coordinates[:, 0] + rng.normal(size=grid.n_data)
    grid.add_continuous_variable("grade", grade)
    grid.variables["grade"].allocate_simulations(4)
    simulated = grade[:, None] + rng.normal(size=(grid.n_data, 4)) * 0.1
    simulated[rng.random(simulated.shape) < 0.05] = np.nan
    simulated[91:98] = np.nan                       # a band with nothing in it
    grid.variables["grade"].simulations[:, :] = simulated

    density = np.full(grid.n_data, 2.0)
    grid.add_continuous_variable("rho", density)
    grid.variables["rho"].allocate_simulations(4)
    grid.variables["rho"].simulations[:, :] = \
        density[:, None] + rng.normal(size=(grid.n_data, 4)) * 0.01

    doubt = rng.uniform(0.0, 0.6, grid.n_data)
    doubt[70:77] = 1.0                     # a band the filter empties entirely
    grid.add_metadata("doubt", doubt)
    return grid


def test_the_curves_do_not_depend_on_how_the_simulations_are_chunked(
        streamed, tmp_path):
    """Every block is placed at the highest cut-off it clears and the bands are
    summed afterwards, so where the chunk boundaries fall must not reach the
    answer -- including the boundaries of the density, which multiplies the
    grade band for band and is chunked differently here."""
    asked = dict(density="rho", cutoffs=6, uncertainty="doubt",
                 max_uncertainty=0.7)
    whole = prepare.grade_tonnage(streamed, "grade", **asked)

    _chunk(streamed.variables["grade"], tmp_path, "g.zarr", rows=7)
    _chunk(streamed.variables["rho"], tmp_path, "r.zarr", rows=3)
    assert len(streamed.variables["grade"].simulations.row_bands()) > 1
    assert len(streamed.variables["rho"].simulations.row_bands()) > 1

    banded = prepare.grade_tonnage(streamed, "grade", **asked)
    for key in ("cutoff", "tonnage", "grade", "metal"):
        assert np.allclose(banded[key], whole[key], equal_nan=True)
    assert banded["kept"] == whole["kept"] < streamed.n_data


def test_the_simulations_are_never_held_whole(streamed, tmp_path, monkeypatch):
    """The failure this replaced: a 280 GB block model died on the `asarray`
    that used to open the function."""
    _chunk(streamed.variables["grade"], tmp_path, "g.zarr", rows=7)
    read = []

    materialize = geoml.storage.ArrayStore.__array__
    monkeypatch.setattr(
        geoml.storage.ArrayStore, "__array__",
        lambda self, *args, **kwargs: (read.append(self.shape),
                                       materialize(self, *args, **kwargs))[1])

    prepare.grade_tonnage(streamed, "grade", cutoffs=6,
                          uncertainty="doubt", max_uncertainty=0.5)

    # the column of doubt is one number per block and is read whole, which is
    # also what says the watch is wired up
    assert (streamed.n_data,) in read
    assert streamed.variables["grade"].simulations.shape not in read


def test_a_thinned_sample_is_the_same_however_it_is_chunked(trained, tmp_path):
    """`_strided` picks up where the last band left off, so which values are
    taken follows from the stride and not from where the chunks happen to
    end -- otherwise two components chunked differently would be thinned to
    different locations and the pairs would stop being pairs."""
    _, point = trained
    variable = copy.deepcopy(point.variables["v"])

    whole = prepare.simulation_sample(variable, most=200)
    for i, label in enumerate(variable.labels):
        _chunk(variable.components[label], tmp_path, "c%d.zarr" % i,
               rows=3 + i)

    assert np.allclose(prepare.simulation_sample(variable, most=200), whole)


def test_a_grade_with_no_simulations_falls_back_on_its_prediction(streamed):
    """A variable that was predicted but not simulated still has one
    realization, and it is the prediction."""
    predicted = streamed.variables["rho"]
    predicted.simulations = None
    predicted.prediction.values[:] = np.linspace(1.0, 5.0, streamed.n_data)

    store = prepare.realization_store(predicted)
    assert store.shape == (streamed.n_data, 1)
    assert store.row_bands() == [slice(0, streamed.n_data)]
    assert np.allclose(np.asarray(store)[:, 0],
                       predicted.prediction.values.to_numpy())


# --------------------------------------------------------------------------- #
# the style
# --------------------------------------------------------------------------- #
def test_drawing_does_not_change_anyone_elses_figures(jura):
    """The settings are applied to the figure being drawn, through
    `rc_context`, and importing geoML settles nothing globally."""
    before = dict(matplotlib.rcParams)

    geoml.plots.Explorer(jura, continuous="Elements",
                         categorical="Rock").histogram()

    assert matplotlib.rcParams["axes.titlelocation"] == \
        before["axes.titlelocation"]
    assert matplotlib.rcParams["axes.prop_cycle"] == before["axes.prop_cycle"]


def test_a_palette_of_the_users_own_is_taken_in_order(jura):
    explorer = geoml.plots.Explorer(
        jura, continuous="Elements", categorical="Rock",
        palette=["#ff0000", "#00ff00", "#0000ff"])

    figure = explorer.scene()
    drawn = figure.axes[0].collections
    assert tuple(drawn[0].get_facecolor()[0][:3]) == (1.0, 0.0, 0.0)
    assert tuple(drawn[1].get_facecolor()[0][:3]) == (0.0, 1.0, 0.0)
    # five rock types against three colours: it comes round again
    assert tuple(drawn[3].get_facecolor()[0][:3]) == (1.0, 0.0, 0.0)


def test_a_palette_can_name_the_categories(jura):
    """What a mapping convention needs: this rock is always this colour,
    wherever it lands in the order."""
    explorer = geoml.plots.Explorer(
        jura, continuous="Elements", categorical="Rock",
        palette={"Sequanian": "#ff0000"})

    _, measured, labels = prepare.category_values(jura.variables["Rock"])
    figure = explorer.scene()
    drawn = figure.axes[0].collections[labels.index("Sequanian")]

    assert tuple(drawn.get_facecolor()[0][:3]) == (1.0, 0.0, 0.0)
    # the ones it says nothing about keep the package's colours
    first = figure.axes[0].collections[0]
    assert matplotlib.colors.to_hex(first.get_facecolor()[0]) == \
        geoml.plots.PALETTE[0]


def test_the_default_colormap_is_cividis(jura):
    assert geoml.plots.SEQUENTIAL == "cividis"
    figure = geoml.plots.Explorer(jura, continuous="Elements").scene(
        color="Zn")
    assert figure.axes[0].collections[0].get_cmap().name == "cividis"


def test_a_colormap_of_the_users_own_is_used(jura):
    figure = geoml.plots.Explorer(
        jura, continuous="Elements", cmap="viridis").scene(color="Zn")
    assert figure.axes[0].collections[0].get_cmap().name == "viridis"


def test_the_palette_is_one_list_for_the_whole_package():
    assert geoml.plots.PALETTE[0] == geoml.plots.style.color(0)
    # it cycles rather than running out
    assert geoml.plots.style.color(len(geoml.plots.PALETTE)) == \
        geoml.plots.PALETTE[0]


# --------------------------------------------------------------------------- #
# the variogram
# --------------------------------------------------------------------------- #
def _line_points(values):
    coords = np.stack([np.arange(len(values), dtype=float),
                       np.zeros(len(values))], axis=1)
    point = geoml.data.PointData.from_array(coords)
    point.add_continuous_variable("z", np.asarray(values, dtype=float))
    return point


def test_the_variogram_on_a_worked_example():
    """Three points on a line, every pair by hand."""
    point = _line_points([0.0, 1.0, 4.0])
    panel = prepare.variogram(point, "z", n_lags=1, max_lag=3.0)[0]

    assert panel["count"][0] == 3
    assert panel["data"][0] == pytest.approx(0.5 * (1 + 9 + 16) / 3)
    assert panel["sill"] == pytest.approx(np.var([0.0, 1.0, 4.0]))
    assert panel["realizations"] is None


def test_iid_values_lie_flat_and_a_smooth_field_rises():
    """The two shapes the figure exists to tell apart: pure nugget sits at
    the sill from the first lag, a continuous field climbs to it."""
    rng = np.random.default_rng(5)
    n = 400
    coords = rng.uniform(0, 100, size=[n, 2])
    point = geoml.data.PointData.from_array(coords)
    point.add_continuous_variable("iid", rng.normal(size=n))
    point.add_continuous_variable(
        "smooth", np.sin(coords[:, 0] / 20.0) + np.cos(coords[:, 1] / 25.0))

    flat = prepare.variogram(point, "iid", n_lags=8)[0]
    busy = flat["count"] > 30
    assert np.all(np.abs(flat["data"][busy] - flat["sill"])
                  < 0.35 * flat["sill"])

    rising = prepare.variogram(point, "smooth", n_lags=8)[0]
    assert rising["data"][0] < 0.25 * rising["sill"]
    assert np.nanmax(rising["data"]) > 0.75 * rising["sill"]


def test_a_directional_variogram_keeps_the_pairs_along_its_direction():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    point = geoml.data.PointData.from_array(coords)
    point.add_continuous_variable("z", coords[:, 0])

    panel = prepare.variogram(point, "z", n_lags=1, max_lag=2.0,
                              direction=[1, 0], tolerance=10.0)[0]
    # the two horizontal pairs; the vertical and diagonal ones are out
    assert panel["count"][0] == 2
    assert panel["data"][0] == pytest.approx(0.5)


def test_the_fan_is_the_variogram_of_each_realization(trained):
    _, point = trained
    panel = prepare.variogram(point, "v", n_lags=5)[0]
    fan = panel["realizations"]
    part = point.variables["v"].components["a"]
    assert fan.shape == (part.simulations.shape[1], 5)

    column = np.asarray(part.simulations[:, 7]).ravel()
    check = geoml.data.PointData.from_array(np.asarray(point.coordinates))
    check.add_continuous_variable("r", column)
    alone = prepare.variogram(check, "r", n_lags=5)[0]["data"]
    assert np.allclose(fan[7], alone, equal_nan=True)


def test_the_residual_variogram_reads_the_prediction_and_drops_the_fan(
        trained):
    _, point = trained
    panel = prepare.variogram(point, "v", n_lags=4, residuals=True)[0]
    assert panel["realizations"] is None

    part = point.variables["v"].components["a"]
    residual = part.measurements.values.to_numpy() \
        - part.prediction.values.to_numpy()
    assert panel["sill"] == pytest.approx(np.var(residual))


def test_the_variogram_reads_realizations_one_column_at_a_time(
        trained, monkeypatch):
    """A block model's store does not fit in memory whole; a column does."""
    _, point = trained
    read = []
    materialize = geoml.storage.ArrayStore.__array__
    monkeypatch.setattr(
        geoml.storage.ArrayStore, "__array__",
        lambda self, *args, **kwargs: (read.append(self.shape),
                                       materialize(self, *args, **kwargs))[1])

    prepare.variogram(point, "v", n_lags=4)
    assert point.variables["v"].components["a"].simulations.shape not in read


def test_the_variogram_figure_draws_a_panel_per_component_with_its_fan(
        trained):
    _, point = trained
    figure = geoml.plots.Explorer(point, continuous="v").variogram(n_lags=5)

    drawn = [ax for ax in figure.axes if ax.lines]
    assert len(drawn) == 3
    n_sim = point.variables["v"].components["a"].simulations.shape[1]
    for ax in drawn:
        # the fan, the sill and the data curve
        assert len(ax.lines) == n_sim + 2
