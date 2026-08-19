"""The plotly figures, and the dashboard that links them.

A figure is still hard to assert on, but a plotly one is a document rather than
a picture, so rather more of it can be read back: how many traces there are,
which subplot each landed in, what the axes were called. The arithmetic behind
them is `plots.prepare`, tested in `test_plots.py` and not repeated here.

What is tested hardest is the one promise these figures make that the
matplotlib ones do not: that a trace drawn from locations carries, in
`customdata`, the row of the container each of its points came from. That is
the whole of what the dashboard's linked selection stands on -- get it wrong
and a brush over the map lights up the wrong samples everywhere else, quietly
and plausibly.
"""
import matplotlib
# a few of these compare the two backends, and importing geoml.plots brings
# matplotlib with it; without this it goes looking for a window to open, and
# under WSL it finds one and takes the interpreter down with it
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.plots.prepare as prepare


@pytest.fixture(autouse=True)
def _close_figures():
    """pyplot holds every figure until told otherwise, and warns at twenty."""
    yield
    matplotlib.pyplot.close("all")


COLS = ["X", "Y"]


@pytest.fixture(scope="module")
def jura():
    train, _ = geoml.datasets.jura()
    return train


@pytest.fixture(scope="module")
def eda(jura):
    return geoml.plots.Interactive(jura, continuous="Elements",
                                   categorical="Rock")


@pytest.fixture(scope="module")
def trained():
    """A small trained model, and data carrying its predictions."""
    geoml.set_seed(1234)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, (50, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
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


@pytest.fixture(scope="module")
def blocks():
    """A block model carrying simulations of a grade."""
    geoml.set_seed(7)
    grid = geoml.data.Grid2D(start=[0, 0], n=[10, 10], step=[2.0, 5.0])
    rng = np.random.default_rng(3)

    grade = grid.coordinates[:, 0] + rng.normal(size=grid.n_data)
    grid.add_continuous_variable("grade", grade)
    grid.variables["grade"].allocate_simulations(4)
    grid.variables["grade"].simulations[:, :] = \
        grade[:, None] + rng.normal(size=(grid.n_data, 4)) * 0.1
    grid.add_metadata("doubt", rng.uniform(0, 1, grid.n_data))
    return grid


def traces_with_rows(figure):
    """Every trace that says which locations it was drawn from."""
    return [trace for trace in figure.data
            if getattr(trace, "customdata", None) is not None]


# --------------------------------------------------------------------------- #
# the numbers the plotly figures added
# --------------------------------------------------------------------------- #
def test_an_empty_cell_is_a_gap_and_not_a_zero():
    """Painted, an empty cell takes the bottom of the colour scale and fills
    the panel with a background that looks like data."""
    counted = prepare.counts_2d([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], bins=2)
    assert np.isnan(counted["z"][1, 0])
    assert np.isnan(counted["z"][0, 1])
    assert counted["z"][0, 0] == 2
    assert counted["count"][0, 0] == 2


def test_the_counts_are_indexed_the_way_an_image_is():
    x = [0.1, 0.1, 0.1, 0.9]
    y = [0.1, 0.1, 0.9, 0.9]
    counted = prepare.counts_2d(x, y, bins=2)
    # two points at the bottom left, one top left, one top right
    assert counted["z"][0, 0] == 2
    assert counted["z"][1, 0] == 1
    assert counted["z"][1, 1] == 1
    assert len(counted["x"]) == 2 and len(counted["y"]) == 2


def test_the_colour_can_be_taken_from_the_logarithm_of_the_count():
    x = np.concatenate([np.zeros(100), np.ones(1)])
    counted = prepare.counts_2d(x, x, bins=2, log=True)
    assert counted["z"][0, 0] == pytest.approx(2.0)      # log10(100)
    assert counted["z"][1, 1] == pytest.approx(0.0)      # log10(1)
    # the count itself is kept, so a hover can still say how many
    assert counted["count"][0, 0] == 100


def test_a_density_needs_something_to_smooth():
    assert prepare.density_grid([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert prepare.density_grid([1.0], [1.0]) is None
    assert prepare.density_curve(np.ones(50), (0.0, 2.0)) is None


def test_a_density_grid_is_square_and_covers_the_data():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=200), rng.normal(size=200)
    x_axis, y_axis, density = prepare.density_grid(x, y, grid=25)
    assert density.shape == (25, 25)
    assert x_axis[0] == pytest.approx(np.min(x))
    assert y_axis[-1] == pytest.approx(np.max(y))
    assert np.all(density >= 0)


def test_the_normal_drawn_is_the_one_the_data_has():
    values = np.random.default_rng(1).normal(loc=3.0, scale=2.0, size=5000)
    x, density = prepare.normal_curve(values, -8.0, 14.0)
    assert x[np.argmax(density)] == pytest.approx(3.0, abs=0.3)
    # a density integrates to one over a window wide enough to hold it
    assert np.trapezoid(density, x) == pytest.approx(1.0, abs=0.01)
    assert prepare.normal_curve(np.ones(10), 0.0, 2.0) is None


def test_the_rows_a_prediction_was_drawn_from_index_the_container(trained):
    _, point = trained
    true, predicted, labels, rows = prepare.prediction_values(point, "v")
    assert len(rows) == len(true)
    measured = point.variables["v"].components["a"].measurements.values \
        .to_numpy()
    assert np.allclose(measured[rows], true[:, 0])


# --------------------------------------------------------------------------- #
# the figures
# --------------------------------------------------------------------------- #
def test_a_histogram_has_a_panel_per_component_and_a_trace_per_category(eda):
    figure = eda.histogram()
    # seven elements, five rock types
    assert len(figure.data) == 7 * 5
    assert all(trace.type == "histogram" for trace in figure.data)
    assert figure.layout.title.text == "Elements"


def test_a_category_is_named_in_the_legend_once(eda):
    figure = eda.histogram()
    named = [trace.name for trace in figure.data if trace.showlegend]
    assert len(named) == 5
    assert len(set(named)) == 5
    # and the rest join it, so hiding one hides it everywhere
    assert all(trace.legendgroup in set(named) for trace in figure.data)


def test_a_pairs_plot_leaves_out_the_half_that_repeats(eda):
    figure = eda.pairs()
    # seven components: 21 pairs below the diagonal and 7 on it
    assert len(figure.data) == 28 * 5


def test_the_upper_triangle_can_be_filled_in(eda):
    filled = eda.pairs(upper="hist2d")
    assert len(filled.data) == 28 * 5 + 21
    assert sum(t.type == "heatmap" for t in filled.data) == 21


def test_the_upper_triangle_pools_the_categories(eda):
    """Whichever category were drawn last would hide the rest, so up here the
    question is where the data sits, all of it together."""
    figure = eda.pairs(upper="density")
    contours = [t for t in figure.data if t.type == "contour"]
    assert len(contours) == 21


def test_a_correlation_panel_carries_something_to_hang_the_number_on(eda):
    """An axis with no trace on it is dropped on the way to the drawn figure,
    and an annotation pinned to that axis goes with it."""
    figure = eda.pairs(upper="correlation")
    assert len(figure.layout.annotations) == 21
    anchors = [t for t in figure.data
               if t.type == "scatter" and len(t.x) == 1 and t.x[0] is None]
    assert len(anchors) == 21
    assert {a.xref for a in figure.layout.annotations} == \
        {"x%d domain" % ((r) * 7 + c + 1)
         for r in range(7) for c in range(7) if c > r}


def test_counting_into_cells_pools_the_categories(eda):
    figure = eda.pairs(kind="hist2d")
    assert sum(t.type == "heatmap" for t in figure.data) == 21
    assert sum(t.type == "histogram" for t in figure.data) == 7


def test_the_components_can_be_drawn_on_the_datas_own_axes(eda):
    plain = eda.pairs()
    with_pc = eda.pairs(principal_components=2)
    # two lines added to each of the 21 panels below the diagonal
    assert len(with_pc.data) - len(plain.data) == 42
    lines = [t for t in with_pc.data
             if t.type == "scatter" and t.mode == "lines+text"]
    assert {tuple(t.text)[1] for t in lines} == {"PC1", "PC2"}


def test_a_compositions_components_are_not_directions_in_its_proportions():
    point = geoml.data.PointData(
        pd.DataFrame(np.random.default_rng(0).uniform(0, 10, (30, 2)),
                     columns=COLS), COLS)
    parts = np.random.default_rng(1).dirichlet([2, 3, 4], 30)
    point.add_compositional_variable("c", ["a", "b", "d"], parts)
    board = geoml.plots.Interactive(point, continuous="c")

    with pytest.raises(TypeError, match="log-ratios"):
        board.pairs(principal_components=2)
    assert board.pairs(log=True, principal_components=2) is not None


def test_a_log_scale_says_so_on_the_axes(eda):
    figure = eda.pairs(log=True)
    assert figure.layout.xaxis43.title.text == "log(Cd)"


def test_the_pca_draws_only_the_components_it_kept(eda):
    two = eda.pca(explained=0.9)
    assert len(two.data) == 3 * 5                 # 2x2 matrix, minus a corner
    assert two.layout.xaxis3.title.text.startswith("PC1 (")
    more = eda.pca(explained=0.99)
    assert len(more.data) > len(two.data)


def test_the_variance_share_is_named_to_a_decimal(eda):
    label = eda.pca().layout.xaxis3.title.text
    assert label.split("(")[1].split("%")[0].count(".") == 1


def test_the_loadings_are_drawn_as_arrows_on_the_right_panel(eda):
    figure = eda.pca(explained=0.9)
    arrows = [a for a in figure.layout.annotations if a.showarrow]
    assert len(arrows) == 7                       # one per element
    # a 2x2 matrix has one panel below the diagonal, the third of the four
    assert {a.xref for a in arrows} == {"x3"}
    # the tail has to be pinned to the same axes, or it hangs off panel one
    assert {a.axref for a in arrows} == {"x3"}
    assert all(a.ax == 0 and a.ay == 0 for a in arrows)


def test_a_loading_is_named_at_its_head_and_not_at_the_origin(eda):
    """Plotly draws an annotation's text at the tail of its arrow, so one
    annotation carrying both would pile all seven names on the origin."""
    figure = eda.pca(explained=0.9)
    named = [a for a in figure.layout.annotations if not a.showarrow]
    arrows = [a for a in figure.layout.annotations if a.showarrow]

    assert sorted(a.text for a in named) == sorted(
        prepare.component_analysis(eda.continuous, 0.9)["labels"])
    assert all(a.text == "" for a in arrows)
    # every name sits where an arrow ends, and none of them at the origin
    heads = {(a.x, a.y) for a in arrows}
    assert {(a.x, a.y) for a in named} == heads
    assert (0, 0) not in heads


def test_a_scene_is_drawn_in_as_many_dimensions_as_there_are():
    for n, kind in ((1, "scatter"), (2, "scatter"), (3, "scatter3d")):
        grid = geoml.data.Grid1D(start=0, n=8, step=1) if n == 1 else (
            geoml.data.Grid2D(start=[0, 0], n=[4, 4], step=[1, 1]) if n == 2
            else geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3],
                                   step=[1, 1, 1]))
        grid.add_continuous_variable("z", np.arange(float(grid.n_data)))
        figure = geoml.plots.Interactive(grid, continuous="z").scene()
        assert figure.data[0].type == kind


def test_a_map_keeps_the_two_axes_on_one_scale(eda):
    figure = eda.scene(color="Cd")
    assert figure.layout.yaxis.scaleanchor == "x"


def test_a_scene_of_a_categorical_variable_draws_one_trace_each(eda):
    figure = eda.scene()
    assert len(figure.data) == 5
    assert figure.layout.title.text == "Rock"


def test_a_vector_is_not_one_colour_scale(jura):
    board = geoml.plots.Interactive(jura, continuous="Elements")
    with pytest.raises(ValueError, match="not one colour scale") as raised:
        board.scene()
    # and the message points at the menu, which is the answer to it
    assert "['Elements']" in str(raised.value)


# --------------------------------------------------------------------------- #
# a scene with a menu of variables
# --------------------------------------------------------------------------- #
def test_a_list_of_variables_puts_a_menu_on_the_scene(eda):
    figure = eda.scene(color=["Cd", "Zn", "Ni"])
    menus = figure.layout.updatemenus
    assert len(menus) == 1
    assert [b.label for b in menus[0].buttons] == ["Cd", "Zn", "Ni"]
    assert figure.layout.title.text == "Cd"


def test_a_vector_variable_stands_for_all_of_its_components(eda, jura):
    figure = eda.scene(color=["Elements"])
    labels = [str(label) for label in jura.variables["Elements"].labels]
    assert [b.label for b in figure.layout.updatemenus[0].buttons] == labels


def test_the_ground_is_drawn_once_and_the_values_swapped_over_it(eda, jura):
    """A trace per choice would carry a copy of the coordinates with it, and
    on a block model the coordinates are the bulk of the figure."""
    figure = eda.scene(color=["Cd", "Zn"])
    assert len(figure.data) == 1

    values, rows, _ = prepare.color_choices(jura, ["Cd", "Zn"])
    assert np.allclose(figure.data[0].marker.color, values[:, 0])
    for i, button in enumerate(figure.layout.updatemenus[0].buttons):
        assert button.method == "update"
        assert np.allclose(button.args[0]["marker.color"][0], values[:, i])
        assert button.args[1]["title"]["text"] == ["Cd", "Zn"][i]
    # and the points still say which locations they are, so they still link
    assert np.array_equal(np.asarray(figure.data[0].customdata), rows)


def test_in_one_dimension_the_menu_moves_the_points_rather_than_painting():
    line = geoml.data.Grid1D(start=0, n=12, step=1)
    line.add_continuous_variable("a", np.arange(12.0))
    line.add_continuous_variable("b", np.arange(12.0) ** 2)
    figure = geoml.plots.Interactive(line, continuous="a").scene(
        color=["a", "b"])

    second = figure.layout.updatemenus[0].buttons[1]
    assert "marker.color" not in second.args[0]
    assert np.allclose(second.args[0]["y"][0], np.arange(12.0) ** 2)
    assert second.args[1]["yaxis"]["title"]["text"] == "b"


def test_a_menu_holds_one_set_of_points_for_every_choice_on_it():
    """Only what is measured in all of them: a cloud that gained and lost
    points as the menu changed would be two things changing at once."""
    point = geoml.data.PointData(
        pd.DataFrame(np.random.default_rng(0).uniform(0, 10, (6, 2)),
                     columns=COLS), COLS)
    point.add_continuous_variable("a", np.array([1.0, 2, 3, 4, 5, 6]))
    point.add_continuous_variable("b", np.array([1.0, np.nan, 3,
                                                 np.nan, 5, 6]))
    values, rows, labels = prepare.color_choices(point, ["a", "b"])
    assert list(rows) == [0, 2, 4, 5]
    assert values.shape == (4, 2)
    assert labels == ["a", "b"]


def test_a_menu_needs_a_location_the_choices_share():
    point = geoml.data.PointData(
        pd.DataFrame(np.random.default_rng(0).uniform(0, 10, (2, 2)),
                     columns=COLS), COLS)
    point.add_continuous_variable("a", np.array([1.0, np.nan]))
    point.add_continuous_variable("b", np.array([np.nan, 2.0]))
    with pytest.raises(ValueError, match="no location carries all"):
        prepare.color_choices(point, ["a", "b"])


def test_a_categorical_variable_is_not_a_choice_on_a_scale(eda):
    with pytest.raises(TypeError, match="categorical"):
        eda.scene(color=["Cd", "Rock"])


# --------------------------------------------------------------------------- #
# clipping the colour scale
# --------------------------------------------------------------------------- #
def test_an_outlier_does_not_have_to_take_the_whole_colour_scale():
    values = np.concatenate([np.arange(100.0), [10000.0]])
    assert prepare.color_limits(values, None) is None

    low, high = prepare.color_limits(values, [0, 0.99])
    assert low == 0.0
    assert high < 200          # the outlier is off the scale, not on it
    assert high == pytest.approx(np.quantile(values, 0.99))

    both = prepare.color_limits(values, [0.01, 0.99])
    assert both[0] > low


def test_the_quantiles_have_to_be_a_pair_between_zero_and_one():
    values = np.arange(10.0)
    for bad in ([0, 99], [0.5], [0.9, 0.1], [-0.1, 0.9], [0.5, 0.5]):
        with pytest.raises(ValueError, match="two quantiles"):
            prepare.color_limits(values, bad)


def test_a_scale_of_no_width_is_no_scale():
    """Every value the same, or a window narrow enough to catch only one."""
    assert prepare.color_limits(np.ones(50), [0, 0.99]) is None
    assert prepare.color_limits(np.array([np.nan, np.nan]), [0, 1]) is None


def test_a_scene_bounds_its_colours_where_it_was_told_to(eda, jura):
    plain = eda.scene(color="Cd").data[0].marker
    clipped = eda.scene(color="Cd", clip=[0, 0.95]).data[0].marker

    assert plain.cmin is None and plain.cmax is None
    values = jura.variables["Elements"].components["Cd"] \
        .measurements.values.to_numpy()
    assert clipped.cmax == pytest.approx(np.nanquantile(values, 0.95))
    # the values themselves are untouched, so a hover still tells the truth
    assert np.nanmax(np.asarray(clipped.color)) > clipped.cmax


def test_each_choice_on_a_menu_is_clipped_by_its_own_quantiles(eda, jura):
    figure = eda.scene(color=["Cd", "Zn"], clip=[0, 0.95])
    values, _, _ = prepare.color_choices(jura, ["Cd", "Zn"])

    assert figure.data[0].marker.cmax == pytest.approx(
        np.quantile(values[:, 0], 0.95))
    for i, button in enumerate(figure.layout.updatemenus[0].buttons):
        assert button.args[0]["marker.cmax"] == pytest.approx(
            np.quantile(values[:, i], 0.95))
        assert button.args[0]["marker.cauto"] is False


def test_a_menu_left_unclipped_lets_plotly_set_the_ends(eda):
    figure = eda.scene(color=["Cd", "Zn"])
    for button in figure.layout.updatemenus[0].buttons:
        assert button.args[0]["marker.cauto"] is True
        assert button.args[0]["marker.cmin"] is None


def test_a_printed_scene_clips_the_same_way(jura):
    board = geoml.plots.Explorer(jura, continuous="Elements")
    panel = board.scene(color="Cd", clip=[0, 0.95]).axes[0]
    values = jura.variables["Elements"].components["Cd"] \
        .measurements.values.to_numpy()
    assert panel.collections[0].get_clim()[1] == pytest.approx(
        np.nanquantile(values, 0.95))


# --------------------------------------------------------------------------- #
# a figure with nothing on it
# --------------------------------------------------------------------------- #
def test_a_category_measured_nowhere_says_so_rather_than_drawing_nothing():
    """Every series is drawn per category, so no categories means no traces
    at all -- a blank figure and no complaint."""
    point = geoml.data.PointData(
        pd.DataFrame(np.random.default_rng(0).uniform(0, 10, (8, 2)),
                     columns=COLS), COLS)
    point.add_continuous_variable("z", np.arange(8.0))
    # the values went in where the categories belong, which is the way to
    # arrive here by accident
    point.add_categorical_variable("rock", list("ABABABAB"))

    for kind in (geoml.plots.Interactive, geoml.plots.Explorer):
        with pytest.raises(ValueError, match="no category measured"):
            kind(point, continuous="z", categorical="rock")


def test_a_printed_figure_has_no_menu_to_offer(jura):
    board = geoml.plots.Explorer(jura, continuous="Elements")
    with pytest.raises(TypeError, match="only `Interactive`"):
        board.scene(color=["Cd", "Zn"])


# --------------------------------------------------------------------------- #
# what the dashboard stands on
# --------------------------------------------------------------------------- #
def test_every_point_says_which_location_it_came_from(eda, jura):
    """The one promise these figures make that the printed ones do not."""
    figure = eda.pairs()
    values, measured, _ = prepare.numeric_values(jura.variables["Elements"])

    for trace in traces_with_rows(figure):
        rows = np.asarray(trace.customdata)
        assert rows.max() < jura.n_data
        assert np.all(measured[rows])

    # and the values drawn are the values at those rows: the first panel
    # below the diagonal is Co against Cd
    below = [t for t in figure.data if t.type == "scatter"][0]
    rows = np.asarray(below.customdata)
    assert np.allclose(np.asarray(below.x), values[rows, 0])
    assert np.allclose(np.asarray(below.y), values[rows, 1])


def test_the_rows_of_a_category_are_that_categorys_rows(eda, jura):
    figure = eda.histogram()
    labels, measured, _ = prepare.category_values(jura.variables["Rock"])
    for trace in figure.data[:5]:
        rows = np.asarray(trace.customdata)
        assert set(labels[rows]) == {trace.name}


def test_the_pca_scores_point_back_at_the_measured_rows(eda, jura):
    figure = eda.pca()
    measured = prepare.numeric_values(jura.variables["Elements"])[1]
    for trace in traces_with_rows(figure):
        assert np.all(measured[np.asarray(trace.customdata)])


def test_a_counted_cell_is_not_a_location(eda):
    """A cell holds a number of points, not one of them, so a matrix drawn
    this way takes no part in a linked selection."""
    figure = eda.pairs(kind="hist2d")
    assert all(t.customdata is None
               for t in figure.data if t.type == "heatmap")


def test_a_three_dimensional_scene_promises_no_link():
    """Plotly has no box to drag over a 3D scatter, and no per-point selection
    on one to show a brush made elsewhere."""
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3], step=[1, 1, 1])
    grid.add_continuous_variable("z", np.arange(27.0))
    figure = geoml.plots.Interactive(grid, continuous="z").scene()
    assert figure.data[0].customdata is None


def test_a_simulated_value_is_a_realization_and_not_a_place(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).simulation_pairs(
        kind="scatter")
    upper = [t for t in figure.data if t.type == "scatter"
             and t.mode == "markers" and t.name == "simulated"]
    assert len(upper) == 3
    assert all(t.customdata is None for t in upper)
    # the measured half does say where its points came from
    measured = [t for t in figure.data if t.type == "scatter"
                and t.mode == "markers" and t.name == "measured"]
    assert len(measured) == 3
    assert all(t.customdata is not None for t in measured)


# --------------------------------------------------------------------------- #
# the model figures
# --------------------------------------------------------------------------- #
def test_the_training_curve_overlays_a_running_mean(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).training_curve(window=2)
    assert len(figure.data) == 2
    assert figure.data[1].name == "mean of 2"


def test_a_short_log_gets_no_second_line(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).training_curve(window=999)
    assert len(figure.data) == 1


def test_the_transformed_pairs_carry_the_normal_they_aim_at(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).transformed_pairs()
    # three warped columns: 3 histograms, 3 scatters, 3 normal curves
    assert sum(t.type == "histogram" for t in figure.data) == 3
    assert sum(t.mode == "lines" for t in figure.data
               if t.type == "scatter") == 3
    assert len(figure.layout.annotations) == 3
    assert all(a.text.startswith("r = ") for a in figure.layout.annotations)


def test_the_transformed_columns_are_numbered_not_named(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).transformed_pairs()
    assert figure.layout.xaxis7.title.text == "Variable 1"


def test_the_comparison_fills_the_whole_matrix(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).simulation_pairs()
    # three measured halves, three simulated, three histograms, three curves
    assert sum(t.type == "heatmap" for t in figure.data) == 6
    assert sum(t.type == "histogram" for t in figure.data) == 3


def test_both_halves_of_the_comparison_share_their_scales(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).simulation_pairs()
    # panel (1,2) and panel (2,1) are the same pair, told the other way
    assert figure.layout.xaxis2.range == figure.layout.yaxis4.range
    assert figure.layout.yaxis2.range == figure.layout.xaxis4.range


def test_a_single_component_is_drawn_with_its_margins(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).prediction_scatter(
        component="a")
    assert figure.layout.title.text == "a"
    assert sum(t.type == "histogram" for t in figure.data) == 2
    assert figure.layout.xaxis3.title.text == "measured"


def test_several_components_are_drawn_as_panels(trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).prediction_scatter()
    # a cloud and a 1:1 line each
    assert len(figure.data) == 6
    assert [a.text for a in figure.layout.annotations] == ["a", "b", "c"]


def test_the_confusion_matrix_is_one_heatmap_reading_top_down():
    rng = np.random.default_rng(11)
    coords = rng.uniform(0, 100, (8, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    point.add_categorical_variable(
        "rock", measurements=np.array(["granite", "basalt"] * 4))
    point.variables["rock"].predicted.values[:] = \
        np.array([1, 0, 1, 0, 0, 0, 1, -1])

    figure = geoml.plots.Interactive(point, categorical="rock") \
        .confusion_matrix()

    assert len(figure.data) == 1
    trace = figure.data[0]
    assert trace.type == "heatmap"
    assert np.array_equal(np.asarray(trace.text, dtype=int),
                          [[3, 0], [1, 3]])
    assert list(trace.x) == ["basalt", "granite"]
    # measured reads top-down, as the printed matrix does
    assert figure.layout.yaxis.autorange == "reversed"
    assert "agreement" in figure.layout.title.text
    # cells are not locations: nothing here for a dashboard to link on
    assert traces_with_rows(figure) == []


def test_the_spread_check_draws_three_traces_per_component(trained):
    """The noise band, the whole claim, and what the errors actually did."""
    _, point = trained
    figure = geoml.plots.Interactive(point, continuous="v").spread_check(
        bins=4)

    assert len(figure.data) == 9
    assert figure.data[2].error_x is not None      # the bins the points span
    assert sum(t.showlegend is True for t in figure.data) == 3


def test_the_accuracy_plot_draws_the_reference_and_a_line_per_component(
        trained):
    model, point = trained
    figure = geoml.plots.Interactive(point, continuous="v",
                                     model=model).accuracy()
    assert len(figure.data) == 4
    assert figure.data[0].name == "perfect"
    assert all("G = " in t.name for t in figure.data[1:])
    assert figure.layout.yaxis.scaleanchor == "x"


def test_the_grade_tonnage_curves_are_drawn_on_two_scales(blocks):
    figure = geoml.plots.Interactive(blocks, continuous="grade") \
        .grade_tonnage(cutoffs=10)
    assert {t.yaxis for t in figure.data} == {"y", "y2"}
    assert figure.layout.yaxis2.overlaying == "y"
    assert figure.layout.yaxis2.side == "right"
    assert figure.layout.yaxis.type == "linear"


def test_only_the_tonnage_takes_the_log_scale(blocks):
    """The grade axis spans one order of magnitude at most, so a log scale
    there would say nothing."""
    figure = geoml.plots.Interactive(blocks, continuous="grade") \
        .grade_tonnage(cutoffs=10, log_mass=True)
    assert figure.layout.yaxis.type == "log"
    assert figure.layout.yaxis2.type != "log"


def test_each_scale_gets_gridlines_in_the_colour_of_its_curve(blocks):
    """Two scales share one frame, and a reader following the grade curve to a
    grey line would otherwise read a tonnage off it."""
    figure = geoml.plots.Interactive(blocks, continuous="grade") \
        .grade_tonnage(cutoffs=5)
    assert figure.layout.yaxis.gridcolor != figure.layout.yaxis2.gridcolor
    assert figure.layout.yaxis.gridcolor.startswith("rgba(")


def test_a_family_of_realizations_is_one_trace_cut_by_gaps(blocks):
    """A hundred traces of a hundred points is a hundred legend entries and a
    far heavier page."""
    figure = geoml.plots.Interactive(blocks, continuous="grade") \
        .grade_tonnage(cutoffs=6)
    family = figure.data[0]
    assert len(family.x) == 4 * 7                 # four realizations, 6 + a gap
    assert np.isnan(np.asarray(family.y, dtype=float)).sum() == 4


def test_the_uncertainty_filter_is_named_in_the_title(blocks):
    figure = geoml.plots.Interactive(blocks, continuous="grade",
                                     uncertainty="doubt") \
        .grade_tonnage(cutoffs=5, max_uncertainty=0.5)
    assert "doubt" in figure.layout.title.text
    assert "of 100 blocks" in figure.layout.title.text


# --------------------------------------------------------------------------- #
# what a bad request says
# --------------------------------------------------------------------------- #
def test_the_model_figures_say_when_there_is_no_model(jura):
    board = geoml.plots.Interactive(jura, continuous="Elements")
    for name in ("training_curve", "transformed_pairs"):
        with pytest.raises(ValueError, match="needs a model"):
            getattr(board, name)()


def test_the_message_names_the_class_that_was_built(jura):
    board = geoml.plots.Interactive(jura, categorical="Rock")
    with pytest.raises(ValueError, match="the Interactive was built"):
        board.histogram()


def test_pairing_needs_something_to_pair():
    point = geoml.data.PointData(
        pd.DataFrame(np.random.default_rng(0).uniform(0, 10, (20, 2)),
                     columns=COLS), COLS)
    point.add_continuous_variable("z", np.arange(20.0))
    with pytest.raises(TypeError, match="vector variable"):
        geoml.plots.Interactive(point, continuous="z").pairs()


def test_only_the_two_kinds_are_accepted(eda):
    with pytest.raises(ValueError, match="scatter"):
        eda.pairs(kind="hexbin")


def test_an_unknown_upper_says_what_it_takes(eda):
    with pytest.raises(ValueError, match="hist2d"):
        eda.pairs(upper="violin")


def test_a_grid_predicted_onto_has_nothing_to_check_against(trained):
    model, _ = trained
    grid = geoml.data.Grid2D(start=[0, 0], n=[6, 6], step=[20, 20])
    model.predict(grid, n_sim=5)
    board = geoml.plots.Interactive(grid, continuous="v", model=model)
    with pytest.raises(ValueError, match="no measurements here"):
        board.accuracy()


# --------------------------------------------------------------------------- #
# the colours
# --------------------------------------------------------------------------- #
def test_a_palette_of_the_users_own_is_taken_in_order(jura):
    board = geoml.plots.Interactive(jura, continuous="Elements",
                                    categorical="Rock",
                                    palette=["#111111", "#222222"])
    colours = [t.marker.color for t in board.histogram().data[:4]]
    assert colours == ["#111111", "#222222", "#111111", "#222222"]


def test_a_palette_can_name_the_categories(jura):
    board = geoml.plots.Interactive(
        jura, continuous="Elements", categorical="Rock",
        palette={"Kimmeridgian": "#abcdef"})
    figure = board.histogram()
    named = {t.name: t.marker.color for t in figure.data[:5]}
    assert named["Kimmeridgian"] == "#abcdef"
    # the ones it left out fall back to the package palette
    assert named["Argovian"] == geoml.plots.PALETTE[0]


def test_the_default_colormap_is_cividis(eda):
    assert geoml.plots.SEQUENTIAL == "cividis"
    scale = eda.scene(color="Cd").data[0].marker.colorscale
    assert scale[0][1] == "#00224e" and scale[-1][1] == "#fee838"


def test_a_colormap_of_the_users_own_is_used(jura):
    figure = geoml.plots.Interactive(jura, continuous="Elements",
                                     cmap="magma").pairs(kind="hist2d")
    heatmap = [t for t in figure.data if t.type == "heatmap"][0]
    assert heatmap.colorscale[0][1] != "#00224e"


def test_the_template_carries_the_package_palette():
    assert geoml.plots.TEMPLATE["layout"]["colorway"] is geoml.plots.PALETTE


# --------------------------------------------------------------------------- #
# the dashboard
# --------------------------------------------------------------------------- #
def test_a_dashboard_writes_every_figure_and_the_script_that_links_them(eda):
    board = geoml.plots.Dashboard([eda.scene(), eda.histogram()],
                                  title="Jura", plotlyjs="cdn")
    page = board.to_html()
    ids = ["geoml-%s-%d" % (board.id, i) for i in range(2)]
    for div_id in ids:
        assert page.count(div_id) >= 2        # the div, and the script's list
    assert "plotly_selected" in page
    assert "plotly_deselect" in page
    assert "<h1>Jura</h1>" in page


def test_the_page_carries_plotly_or_fetches_it(eda):
    """Nothing here searches *inside* the carried library, and that is
    deliberate. It is four megabytes of minified javascript containing very
    nearly every short string one might think to look for -- `cdn.plot.ly`
    among them, as the default source of topojson for maps geoML never draws.
    A failing `in` over a string that size sends pytest through difflib to
    explain itself and the run stops for minutes, so the claims here are made
    with `startswith` and with lengths, whose failures are cheap to report."""
    figure = eda.histogram()
    fetched = geoml.plots.Dashboard([figure], plotlyjs="cdn")
    carried = geoml.plots.Dashboard([figure], plotlyjs="embed")

    assert fetched._library().startswith('<script src="https://cdn.plot.ly/')
    # nothing is fetched to draw the carried page, which is what makes it a
    # report rather than a link
    assert carried._library().startswith("<script>")
    # and the library is some megabytes of it, which is what that costs
    assert len(carried.to_html()) > len(fetched.to_html()) + 1_000_000


def test_an_unknown_way_of_loading_plotly_says_what_there_is(eda):
    with pytest.raises(ValueError, match="'embed'"):
        geoml.plots.Dashboard([eda.scene()], plotlyjs="local")


def test_a_dashboard_needs_a_figure(eda):
    with pytest.raises(ValueError, match="at least one"):
        geoml.plots.Dashboard([])


def test_a_caption_replaces_the_figures_own_title(eda):
    board = geoml.plots.Dashboard([("Where the samples are", eda.scene())],
                                  plotlyjs="cdn")
    assert board.figures[0].layout.title.text == "Where the samples are"


def test_the_figure_handed_over_is_left_as_it_was(eda):
    """It belongs to whoever built it, and may be on screen already."""
    figure = eda.histogram()
    before = (figure.layout.width, figure.layout.height,
              figure.layout.title.text)
    geoml.plots.Dashboard([("renamed", figure)], plotlyjs="cdn").to_html()
    assert (figure.layout.width, figure.layout.height,
            figure.layout.title.text) == before


def test_a_panel_keeps_the_proportions_it_was_drawn_at(eda):
    """Pouring a scatter matrix into half a page turns every panel into a
    slot; handing the shape over as an aspect ratio makes it smaller
    instead."""
    board = geoml.plots.Dashboard([eda.pairs()], plotlyjs="cdn")
    width, height = board.shapes[0]
    assert "aspect-ratio: %g / %g" % (width, height) in board.to_html()
    # and the figure itself is freed to fill whatever it lands in
    assert board.figures[0].layout.width is None
    assert board.figures[0].layout.height is None
    assert board.figures[0].layout.autosize


def test_a_panel_that_can_be_brushed_opens_with_the_box_tool(eda):
    board = geoml.plots.Dashboard([eda.scene(), eda.histogram()],
                                  plotlyjs="cdn")
    assert all(f.layout.dragmode == "select" for f in board.figures)


def test_a_scene_that_cannot_be_brushed_keeps_its_rotation():
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3], step=[1, 1, 1])
    grid.add_continuous_variable("z", np.arange(27.0))
    figure = geoml.plots.Interactive(grid, continuous="z").scene()
    board = geoml.plots.Dashboard([figure], plotlyjs="cdn")
    assert board.figures[0].layout.dragmode is None


def test_a_notebook_gets_a_frame_that_runs_its_own_script(eda):
    """A notebook is not obliged to run a script it is handed, and JupyterLab
    does not; an iframe is a document in its own right and runs its own."""
    board = geoml.plots.Dashboard([eda.histogram()], plotlyjs="cdn")
    frame = board._repr_html_()
    assert frame.startswith("<iframe srcdoc=")
    assert "&lt;script&gt;" in frame or "&lt;/script&gt;" in frame
    assert 'height="%d"' % board.height() in frame


def test_a_nested_entry_is_a_row_of_its_own(eda):
    board = geoml.plots.Dashboard(
        [eda.scene(),
         [eda.histogram(), eda.pca()],
         eda.pairs()], plotlyjs="cdn")
    assert board.rows == [[0], [1, 2], [3]]
    assert len(board.figures) == 4
    # each row asks the page for as many columns as it holds
    page = board.to_html()
    assert "repeat(1, minmax(0, 1fr))" in page
    assert "repeat(2, minmax(0, 1fr))" in page


def test_without_a_nested_entry_the_figures_are_dealt_out_by_columns(eda):
    figures = [eda.scene(), eda.histogram(), eda.pca()]
    assert geoml.plots.Dashboard(figures, columns=2,
                                 plotlyjs="cdn").rows == [[0, 1], [2]]
    assert geoml.plots.Dashboard(figures, columns=3,
                                 plotlyjs="cdn").rows == [[0, 1, 2]]
    assert geoml.plots.Dashboard(figures, columns=1,
                                 plotlyjs="cdn").rows == [[0], [1], [2]]


def test_a_caption_pair_is_not_mistaken_for_a_row(eda):
    """Both arrive as a tuple; what tells them apart is that a caption is a
    name and a figure is not."""
    board = geoml.plots.Dashboard(
        [("Where the samples are", eda.scene()), eda.histogram()],
        columns=2, plotlyjs="cdn")
    assert board.rows == [[0, 1]]
    assert board.figures[0].layout.title.text == "Where the samples are"

    # and a captioned figure inside a row keeps its caption
    nested = geoml.plots.Dashboard(
        [[("A map", eda.scene()), ("A histogram", eda.histogram())],
         eda.pca()], plotlyjs="cdn")
    assert nested.rows == [[0, 1], [2]]
    assert [f.layout.title.text for f in nested.figures[:2]] == \
        ["A map", "A histogram"]


def test_a_row_of_one_is_taller_than_a_row_of_two(eda):
    """A panel given the whole width is drawn at the whole width's height."""
    figures = [eda.histogram(), eda.pca()]
    wide = geoml.plots.Dashboard([[figures[0]], [figures[1]]],
                                 plotlyjs="cdn")
    side = geoml.plots.Dashboard([figures], plotlyjs="cdn")
    assert wide.height() > side.height()


def test_the_scenes_are_told_to_turn_together_only_when_there_are_several():
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3], step=[1, 1, 1])
    grid.add_continuous_variable("z", np.arange(27.0))
    grid.add_continuous_variable("w", np.arange(27.0)[::-1].copy())
    space = geoml.plots.Interactive(grid, continuous="z")

    alone = geoml.plots.Dashboard([space.scene()], plotlyjs="cdn")
    several = geoml.plots.Dashboard(
        [space.scene(), space.scene(color="w")], plotlyjs="cdn")
    assert several._scenes() == 2
    assert "Turning one 3D scene" in several.hint
    assert "Turning one 3D scene" not in alone.hint
    # the camera is copied by the same script that copies selections
    assert "plotly_relayout" in several.to_html()
    assert ".camera" in several.to_html()


def test_the_page_grows_with_what_is_on_it(eda):
    one = geoml.plots.Dashboard([eda.histogram()], plotlyjs="cdn")
    two = geoml.plots.Dashboard([eda.histogram(), eda.pairs()], columns=1,
                                plotlyjs="cdn")
    assert two.height() > one.height()
    # side by side, the same two take one row instead of two
    assert geoml.plots.Dashboard(
        [eda.histogram(), eda.pairs()], columns=2,
        plotlyjs="cdn").height() < two.height()


def test_a_fragment_carries_no_head_to_load_plotly_into(eda):
    board = geoml.plots.Dashboard([eda.histogram()], plotlyjs="cdn")
    piece = board.to_html(full=False)
    assert "<!DOCTYPE html>" not in piece
    assert "cdn.plot.ly" not in piece
    assert "geoml-grid" in piece


def test_the_short_way_draws_the_figures_it_was_named(eda):
    board = eda.dashboard(figures=("histogram", "scene"), plotlyjs="cdn")
    assert len(board.figures) == 2
    assert board.figures[0].data[0].type == "histogram"
    assert "Elements" in board.title


def test_a_dashboard_written_to_a_file_is_the_page(tmp_path, eda):
    path = eda.dashboard(figures=("scene",), plotlyjs="cdn").write_html(
        str(tmp_path / "board.html"))
    with open(path, encoding="utf-8") as file:
        assert file.read().startswith("<!DOCTYPE html>")


def test_the_variogram_twin_carries_fan_sill_and_data(trained):
    _, point = trained
    figure = geoml.plots.Interactive(point, continuous="v").variogram(
        n_lags=4)

    assert {trace.name for trace in figure.data} == \
        {"realizations", "data variance", "data"}
    n_sim = point.variables["v"].components["a"].simulations.shape[1]
    assert len(figure.data) == 3 * (n_sim + 2)
    # one legend entry per meaning, on the first panel alone
    assert sum(bool(trace.showlegend) for trace in figure.data) == 3


def test_the_residual_variogram_twin_drops_the_fan(trained):
    _, point = trained
    figure = geoml.plots.Interactive(point, continuous="v").variogram(
        n_lags=4, residuals=True)
    assert {trace.name for trace in figure.data} == {"data variance", "data"}
