# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`MassFractionVariable`: the parts of a whole, each its exact fraction.

Pinned here: the preparation (zeros kept, negatives refused, a partial row
keeping its parts, the rest missing with any listed part and zero where the
parts make up the whole, rows over the whole scaled or marked missing, a
clean table stored to the bit); what the model is handed (per part, and the
complete rows alone under a likelihood that weighs rows whole); Zarr;
aggregation; the drillhole conversion; and the power coordinates the
exploratory figures open a composition into.
"""
import warnings

import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.plots.prepare as prep
import geoml.warping as wp
from geoml.data import DrillholeData


def _point(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return geoml.data.PointData.from_array(
        rng.uniform(0.0, 100.0, (n, 2)), ["x", "y"])


def _added(values, labels=("a", "b", "c"), **kwargs):
    point = _point(len(values))
    point.add_mass_fraction_variable("m", list(labels), np.asarray(values),
                                     **kwargs)
    return point.variables["m"]


def _stored(variable):
    return np.stack([variable.components[label].measurements.values
                     for label in variable.labels], axis=1).astype(float)


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #
def test_zeros_are_kept_and_a_clean_table_is_stored_to_the_bit():
    values = np.array([[0.2, 0.3, 0.5], [0.0, 0.4, 0.6], [0.1, 0.0, 0.9]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        variable = _added(values)

    assert isinstance(variable, geoml.data.CompositionalVariable)
    np.testing.assert_array_equal(_stored(variable), values)
    fractions, has_value = variable.get_measurements()
    np.testing.assert_array_equal(fractions, values)
    assert (has_value == 1.0).all()


def test_negative_values_are_refused_by_part():
    values = np.array([[0.2, 0.3, 0.5], [-0.01, 0.4, 0.61], [0.1, -0.1, 1.0]])
    with pytest.raises(ValueError, match="a 1, b 1"):
        _added(values)


def test_a_partial_row_keeps_the_parts_it_has():
    values = np.array([[0.2, 0.3, 0.5], [0.1, np.nan, 0.6]])
    variable = _added(values)

    np.testing.assert_array_equal(_stored(variable), values)
    fractions, has_value = variable.get_measurements()
    np.testing.assert_array_equal(has_value, [[1, 1, 1], [1, 0, 1]])
    np.testing.assert_array_equal(fractions[1], [0.1, 1.0, 0.6])


def test_a_partial_row_over_the_whole_is_marked_missing():
    values = np.array([[0.2, 0.3, 0.5], [0.7, np.nan, 0.6]])
    with pytest.warns(UserWarning, match="1 sample"):
        variable = _added(values)

    assert np.isnan(_stored(variable)[1]).all()
    assert (variable.get_measurements()[1][1] == 0).all()


def test_without_a_rest_the_complete_rows_are_closed_and_the_others_not():
    values = np.array([[2.0, 3.0, 5.0], [0.1, np.nan, 0.3]])
    variable = _added(values)

    np.testing.assert_allclose(_stored(variable)[0], [0.2, 0.3, 0.5])
    np.testing.assert_array_equal(_stored(variable)[1], values[1])


def test_a_row_holding_nothing_is_marked_missing():
    values = np.array([[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]])
    with pytest.warns(UserWarning, match="zero in every part"):
        variable = _added(values)
    assert np.isnan(_stored(variable)[1]).all()


def test_the_rest_is_missing_with_any_part_and_zero_at_the_whole():
    values = np.array([[0.2, 0.3], [0.4, 0.6], [0.1, np.nan],
                       [0.3, 0.7 - 1e-12]])
    variable = _added(values, labels=("a", "b"), rest=True)

    rest = _stored(variable)[:, 2]
    np.testing.assert_allclose(rest[0], 0.5)
    assert rest[1] == 0.0 and rest[3] == 0.0
    assert np.isnan(rest[2])
    fractions, has_value = variable.get_measurements()
    np.testing.assert_array_equal(has_value[2], [1, 0, 0])
    np.testing.assert_allclose(fractions[has_value.all(axis=1)].sum(axis=1),
                               1.0, atol=1e-10)


def test_a_complete_row_over_the_whole_is_scaled_down_and_its_rest_is_zero():
    values = np.array([[0.2, 0.3], [0.8, 0.4]])
    with pytest.warns(UserWarning, match="scaled down"):
        variable = _added(values, labels=("a", "b"), rest=True)

    np.testing.assert_allclose(_stored(variable)[1], [2 / 3, 1 / 3, 0.0])


def test_the_parts_keep_their_units():
    values = np.array([[20.0, 30_000.0], [0.0, 10_000.0], [10.0, np.nan]])
    variable = _added(values, labels=("pb", "zn"), units=("%", "ppm"),
                      rest=True)

    np.testing.assert_array_equal(_stored(variable)[:, :2], values)
    fractions, has_value = variable.get_measurements()
    np.testing.assert_allclose(fractions[0], [0.2, 0.03, 0.77])
    np.testing.assert_allclose(fractions[1], [0.0, 0.01, 0.99])
    np.testing.assert_array_equal(has_value[2], [1, 0, 0])


# --------------------------------------------------------------------------- #
# persistence and aggregation
# --------------------------------------------------------------------------- #
def test_a_store_keeps_the_class_the_zeros_the_gaps_and_the_units(tmp_path):
    point = _point(3)
    point.add_mass_fraction_variable(
        "m", ["pb", "zn"], np.array([[20.0, 30_000.0], [0.0, 10_000.0],
                                     [10.0, np.nan]]),
        units=("%", "ppm"), rest=True)
    point.to_zarr(str(tmp_path / "p.zarr"))
    back = geoml.data.PointData.open(str(tmp_path / "p.zarr"))

    variable = back.variables["m"]
    assert type(variable) is geoml.data.MassFractionVariable
    assert [variable.components[c].unit for c in ("pb", "zn")] \
        == ["%", "ppm"]
    np.testing.assert_array_equal(_stored(variable),
                                  _stored(point.variables["m"]))


def test_aggregation_averages_the_complete_rows_in_fractions():
    point = geoml.data.PointData.from_array(
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [15.0, 15.0]]),
        ["x", "y"])
    point.add_mass_fraction_variable(
        "m", ["pb", "zn"], np.array([[20.0, 0.0], [0.0, 10_000.0],
                                     [10.0, np.nan], [5.0, 5_000.0]]),
        units=("%", "ppm"), rest=True)
    grid = geoml.data.Grid2D(start=[0, 0], n=[2, 2], step=[10, 10])
    grid.aggregate(point)

    variable = grid.variables["m"]
    assert type(variable) is geoml.data.MassFractionVariable
    cell = grid._cell_of(point)[0]
    fractions, has_value = variable.get_measurements()
    # the first two rows only: the third is missing a part
    np.testing.assert_allclose(fractions[cell], [0.1, 0.005, 0.895])
    np.testing.assert_allclose(_stored(variable)[cell, :2], [10.0, 5_000.0])


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def _model(warping, n=40, seed=3):
    rng = np.random.default_rng(seed)
    point = _point(n, seed)
    values = rng.dirichlet([2.0, 3.0, 4.0], n)
    values[:6, 0] = 0.0
    values[:6] /= values[:6].sum(axis=1, keepdims=True)
    values[6:9, 1] = np.nan
    point.add_mass_fraction_variable("m", ["a", "b", "c"], values)
    inducing = geoml.data.PointData.from_array(
        rng.uniform(0.0, 100.0, (10, 2)), ["x", "y"])
    geoml.set_seed(1234)
    gp = geoml.latent.BasicGP(geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(40.0)), size=3)
    return point, lambda: geoml.models.VGPNetwork(
        point, "m", geoml.likelihood.Gaussian(warping), gp,
        options=geoml.models.GPOptions(verbose=False))


def test_the_model_trains_on_the_complete_rows_and_says_how_many_it_left():
    point, build = _model(wp.ChainedWarping(wp.ScaledSimplex(3),
                                            wp.ZScore(3)))
    with pytest.warns(UserWarning, match="3 row"):
        model = build()

    assert (model.has_value[6:9] == 0).all()
    assert (model.has_value[:6] == 1).all()
    model.train_full(max_iter=5)
    assert np.isfinite(model.training_log).all()

    query = geoml.data.PointData.from_array(
        np.random.default_rng(0).uniform(0.0, 100.0, (10, 2)), ["x", "y"])
    model.predict(query, n_sim=4)
    parts = np.stack([query.values("m/%s/prediction" % c)
                      for c in ("a", "b", "c")], axis=1)
    np.testing.assert_allclose(parts.sum(axis=1), 1.0, atol=1e-8)
    assert type(query.variables["m"]) is geoml.data.MassFractionVariable


def test_a_log_ratio_warping_names_the_variable_its_zeros_are_in():
    _, build = _model(wp.CenteredLogRatio(3))
    with pytest.raises(wp.DomainError, match="Variable 'm'"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build()


# --------------------------------------------------------------------------- #
# drillholes
# --------------------------------------------------------------------------- #
def _holes(values):
    collar = pd.DataFrame({"HoleID": ["H1"], "X": [0.0], "Y": [0.0],
                           "Z": [100.0], "Length": [10.0], "Dip": [90.0],
                           "Azimuth": [0.0]})
    holes = DrillholeData(collar, hole="HoleID", x="X", y="Y", z="Z",
                          length="Length", dip="Dip", azimuth="Azimuth")
    frame = pd.DataFrame(values)
    frame["HoleID"] = "H1"
    frame["From"] = 2.0 * np.arange(len(frame))
    frame["To"] = frame["From"] + 2.0
    holes.add_intervals("assay", frame, hole="HoleID", fr="From", to="To",
                        units={"pb": "%", "ag": "ppm"})
    return holes


def test_a_drillhole_group_becomes_a_mass_fraction_variable():
    holes = _holes({"pb": [1.0, 0.0, 3.0], "ag": [10.0, 20.0, np.nan]})
    point = holes.as_point_data(
        mass_fraction={"metals": {"columns": ["pb", "ag"], "rest": True}})

    metals = point.variables["metals"]
    assert type(metals) is geoml.data.MassFractionVariable
    assert "pb" not in point.variables and "ag" not in point.variables
    assert [metals.components[c].unit for c in ("pb", "ag")] == ["%", "ppm"]
    stored = _stored(metals)
    np.testing.assert_array_equal(stored[:, 0], [1.0, 0.0, 3.0])
    assert np.isnan(stored[2, 1]) and np.isnan(stored[2, 2])
    np.testing.assert_allclose(stored[1, 2], 1 - 20e-6)


# --------------------------------------------------------------------------- #
# the figures' power coordinates
# --------------------------------------------------------------------------- #
def _compositions(n=50, seed=1):
    return np.random.default_rng(seed).dirichlet([2.0, 3.0, 4.0, 1.0], n)


def test_the_power_coordinates_at_zero_are_the_log_ratios_moved():
    values = _compositions()
    opened = prep.power_coordinates(values, 0.0)
    moved = opened - prep.centred_log_ratio(values)
    np.testing.assert_allclose(moved, moved[:1].repeat(len(values), axis=0),
                               atol=1e-12)


def test_the_power_coordinates_at_one_are_the_parts_over_their_means():
    values = _compositions()
    ratio = values / values.mean(axis=0)
    np.testing.assert_allclose(prep.power_coordinates(values, 1.0),
                               ratio - ratio.mean(axis=1, keepdims=True),
                               atol=1e-12)


def test_the_power_coordinates_ignore_the_units_and_draw_zeros():
    values = _compositions()
    values[:5, 0] = 0.0
    scaled = values * np.array([100.0, 1e6, 1.0, 1.0])
    np.testing.assert_allclose(prep.power_coordinates(values, 0.5),
                               prep.power_coordinates(scaled, 0.5),
                               atol=1e-12)
    assert np.isfinite(prep.power_coordinates(values, 0.5)).all()
    with pytest.raises(ValueError, match="hold zeros"):
        prep.power_coordinates(values, 0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        prep.power_coordinates(values, 1.5)


def test_a_power_is_for_a_composition():
    with pytest.raises(TypeError, match="composition"):
        prep.opened_up(_compositions(), list("abcd"), False, power=0.5)


@pytest.fixture(scope="module")
def zeros():
    point = _point(60, 2)
    values = _compositions(60, 3)
    values[:8, 0] = 0.0
    values[:8] /= values[:8].sum(axis=1, keepdims=True)
    point.add_mass_fraction_variable("m", list("abcd"), values)
    point.add_compositional_variable("c", list("abcd"), _compositions(60, 4))
    return point


def test_a_mass_fraction_is_analysed_as_it_is_unless_a_power_is_given(zeros):
    plain = prep.component_analysis(zeros.variables["m"])
    assert plain["labels"][0] == "a"
    powered = prep.component_analysis(zeros.variables["m"], power=0.5)
    assert powered["labels"][0] == "pow0.5(a)"
    assert np.isfinite(powered["scores"]).all()
    # a composition takes the log-ratios unless a power is given
    assert prep.component_analysis(zeros.variables["c"])["labels"][0] \
        == "clr(a)"
    assert prep.component_analysis(zeros.variables["c"], power=0.0)[
        "labels"][0] == "pow0(a)"


@pytest.mark.parametrize("backend", ["Explorer", "Interactive"])
def test_both_backends_draw_a_power(zeros, backend):
    eda = getattr(geoml.plots, backend)(zeros, continuous="m")
    eda.pairs(power=0.5, principal_components=2)
    eda.pca(power=0.5)
    eda.pca()
    with pytest.raises(ValueError, match="hold zeros"):
        eda.pairs(power=0.0)
    matplotlib.pyplot.close("all")
