"""Tests for the drillhole subsystem: desurveying, interval tables,
compositing and conversion to point data.

The geometry tests use holes whose answer can be worked out by hand, so a
regression in the minimum curvature code shows up as a number rather than as a
vague shape change. The compositing tests check the two properties that matter
for a resource model: a composite is the length- (or mass-) weighted mean of
what it covers, and no composite crosses a geological boundary when a domain is
given.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.drillhole as drillhole
from geoml.drillhole import (DrillholeData, IntervalTable, HOLE, FROM, TO,
                             LENGTH)


def _collar(holes=("H1",), x=0.0, y=0.0, z=100.0, length=100.0,
            dip=None, azimuth=None):
    frame = pd.DataFrame({
        "HoleID": list(holes),
        "X": np.full(len(holes), x, dtype=float),
        "Y": np.full(len(holes), y, dtype=float),
        "Z": np.full(len(holes), z, dtype=float),
        "Length": np.full(len(holes), length, dtype=float)})
    if dip is not None:
        frame["Dip"] = dip
    if azimuth is not None:
        frame["Azimuth"] = azimuth
    return frame


def _drillholes(**kwargs):
    kwargs.setdefault("dip", "Dip")
    kwargs.setdefault("azimuth", "Azimuth")
    collar = kwargs.pop("collar")
    return DrillholeData(collar, hole="HoleID", x="X", y="Y", z="Z",
                         length="Length", **kwargs)


# --------------------------------------------------------------------------- #
# desurveying
# --------------------------------------------------------------------------- #
def test_vertical_hole_goes_straight_down():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0))
    got = holes.coordinates_at(["H1", "H1"], [0.0, 50.0])

    assert np.allclose(got[0], [0.0, 0.0, 100.0])
    assert np.allclose(got[1], [0.0, 0.0, 50.0])


def test_inclined_hole_matches_trigonometry():
    """A 45 degree hole due east drops and steps out by the same amount."""
    holes = _drillholes(collar=_collar(dip=45.0, azimuth=90.0))
    got = holes.coordinates_at(["H1"], [100.0])[0]

    step = 100.0 * np.sqrt(0.5)
    assert np.allclose(got, [step, 0.0, 100.0 - step])


def test_azimuth_is_measured_clockwise_from_north():
    holes = _drillholes(collar=_collar(dip=0.0, azimuth=0.0))
    north = holes.coordinates_at(["H1"], [10.0])[0]
    assert np.allclose(north, [0.0, 10.0, 100.0])

    holes = _drillholes(collar=_collar(dip=0.0, azimuth=90.0))
    east = holes.coordinates_at(["H1"], [10.0])[0]
    assert np.allclose(east, [10.0, 0.0, 100.0])


def test_dip_sign_convention_can_be_reversed():
    """Databases recording a downward hole as negative need the flag."""
    down = _drillholes(collar=_collar(dip=90.0, azimuth=0.0))
    flipped = DrillholeData(
        _collar(dip=-90.0, azimuth=0.0), hole="HoleID", x="X", y="Y", z="Z",
        length="Length", dip="Dip", azimuth="Azimuth",
        dip_positive_down=False)

    assert np.allclose(down.coordinates_at(["H1"], [40.0]),
                       flipped.coordinates_at(["H1"], [40.0]))


def test_upward_hole_gains_elevation():
    holes = _drillholes(collar=_collar(dip=-90.0, azimuth=0.0))
    got = holes.coordinates_at(["H1"], [30.0])[0]
    assert np.allclose(got, [0.0, 0.0, 130.0])


def test_survey_overrides_the_collar_attitude():
    survey = pd.DataFrame({"HoleID": ["H1", "H1"], "Depth": [0.0, 100.0],
                           "Dip": [90.0, 90.0], "Azimuth": [0.0, 0.0]})
    holes = _drillholes(collar=_collar(dip=0.0, azimuth=45.0), survey=survey,
                        depth="Depth")

    # the survey says vertical, the collar says horizontal
    assert np.allclose(holes.coordinates_at(["H1"], [50.0])[0],
                       [0.0, 0.0, 50.0])


def test_deviated_hole_keeps_its_length():
    """Minimum curvature follows an arc, so along-hole distance is preserved."""
    depth = np.arange(0.0, 201.0, 10.0)
    survey = pd.DataFrame({
        "HoleID": "H1", "Depth": depth,
        # the hole flattens and swings from north to east as it goes
        "Dip": np.linspace(90.0, 45.0, len(depth)),
        "Azimuth": np.linspace(0.0, 90.0, len(depth))})
    holes = _drillholes(collar=_collar(length=200.0), survey=survey,
                        depth="Depth")

    path = holes.coordinates_at(["H1"] * len(depth), depth)
    walked = np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))

    # the chords of a curve are slightly shorter than the arc
    assert walked == pytest.approx(200.0, rel=1e-3)
    assert path[-1, 2] < path[0, 2]          # it went down
    assert path[-1, 0] > 0 and path[-1, 1] > 0   # and to the north-east


def test_hole_without_survey_or_dip_is_taken_as_vertical():
    collar = _collar()                      # no dip, no azimuth column
    with pytest.warns(UserWarning, match="vertical"):
        holes = DrillholeData(collar, hole="HoleID", x="X", y="Y", z="Z",
                              length="Length")
    assert np.allclose(holes.coordinates_at(["H1"], [10.0])[0],
                       [0.0, 0.0, 90.0])


def test_depth_beyond_the_last_station_continues_straight():
    survey = pd.DataFrame({"HoleID": ["H1"], "Depth": [0.0], "Dip": [90.0],
                           "Azimuth": [0.0]})
    holes = _drillholes(collar=_collar(), survey=survey, depth="Depth")
    assert np.allclose(holes.coordinates_at(["H1"], [500.0])[0],
                       [0.0, 0.0, -400.0])


# --------------------------------------------------------------------------- #
# interval tables
# --------------------------------------------------------------------------- #
def _assay(hole="H1", edges=(0, 2, 4, 6), grade=(1.0, 2.0, 3.0),
           density=None, rock=None):
    frame = pd.DataFrame({"HoleID": hole,
                          "From": edges[:-1], "To": edges[1:],
                          "grade": grade})
    if density is not None:
        frame["density"] = density
    if rock is not None:
        frame["rock"] = rock
    return frame


def test_roles_default_by_dtype_and_can_be_redeclared():
    table = IntervalTable(_assay(rock=["a", "b", "b"]), hole="HoleID",
                          fr="From", to="To")

    assert table.roles["grade"] == "grade"        # numeric
    assert table.roles["rock"] == "ignore"        # text, until declared

    table.set_role("rock", "categorical")
    assert table.columns_with_role("categorical") == ["rock"]

    with pytest.raises(ValueError, match="unknown role"):
        table.set_role("grade", "nonsense")


def test_reserved_columns_cannot_take_a_role():
    table = IntervalTable(_assay(), hole="HoleID", fr="From", to="To")
    with pytest.raises(ValueError):
        table.set_role(FROM, "grade")


def test_validate_finds_overlaps_gaps_and_bad_lengths():
    frame = pd.DataFrame({
        "HoleID": ["H1"] * 4,
        "From": [0.0, 1.0, 6.0, 10.0],     # 1-2 overlaps 0-2; gap before 6
        "To": [2.0, 3.0, 6.0, 12.0],       # 6-6 has no length
        "grade": [1.0, 2.0, 3.0, 4.0]})
    table = IntervalTable(frame, hole="HoleID", fr="From", to="To",
                          name="assay")

    report = table.validate(on_error="ignore")
    issues = set(report["issue"])

    assert "overlaps the previous interval" in issues
    assert "gap after the previous interval" in issues
    assert "non-positive length" in issues
    assert set(report.loc[report["issue"] == "non-positive length",
                          "severity"]) == {"error"}
    assert set(report.loc[report["issue"].str.startswith("gap"),
                          "severity"]) == {"warning"}


def test_validate_raises_only_on_errors():
    overlapping = pd.DataFrame({"HoleID": ["H1", "H1"], "From": [0.0, 1.0],
                                "To": [2.0, 3.0], "grade": [1.0, 2.0]})
    with pytest.raises(ValueError, match="overlaps"):
        IntervalTable(overlapping, hole="HoleID", fr="From",
                      to="To").validate(on_error="raise")

    gapped = pd.DataFrame({"HoleID": ["H1", "H1"], "From": [0.0, 5.0],
                           "To": [2.0, 7.0], "grade": [1.0, 2.0]})
    with pytest.warns(UserWarning, match="gap"):
        # a gap is reported but must not stop the work
        IntervalTable(gapped, hole="HoleID", fr="From",
                      to="To").validate(on_error="raise")


def test_the_combined_report_passes_over_clean_tables():
    """A clean table's empty report must not decide the combined dtypes."""
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=10.0))
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0], "To": [10.0],
                      "rock": ["ore"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")
    with pytest.warns(UserWarning, match="gap"):
        holes.add_intervals(
            "assay",
            pd.DataFrame({"HoleID": ["H1", "H1"], "From": [0.0, 8.0],
                          "To": [4.0, 10.0], "grade": [1.0, 2.0]}),
            hole="HoleID", fr="From", to="To")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        report = holes.validate(on_error="ignore")

    assert set(report["table"]) == {"assay"}
    assert list(report["issue"]) == ["gap after the previous interval"]


def test_intervals_of_unknown_holes_are_dropped():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0))
    frame = _assay(hole=["H1", "H1", "GHOST"])

    with pytest.warns(UserWarning, match="no collar"):
        holes.add_intervals("assay", frame, hole="HoleID", fr="From", to="To")

    assert list(holes.intervals["assay"].holes) == ["H1"]


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #
def _simple_holes(**kwargs):
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=6.0))
    holes.add_intervals("assay", _assay(**kwargs), hole="HoleID", fr="From",
                        to="To")
    return holes


def test_composite_is_the_length_weighted_mean():
    # 0-2 at 1.0, 2-4 at 2.0, 4-6 at 3.0, composited to 3 m
    holes = _simple_holes()
    got = holes.composite_fixed(3.0).intervals["assay"].data

    assert list(got[FROM]) == [0.0, 3.0]
    # 0-3 covers 2 m of grade 1 and 1 m of grade 2
    assert got["grade"][0] == pytest.approx((2 * 1.0 + 1 * 2.0) / 3)
    assert got["grade"][1] == pytest.approx((1 * 2.0 + 2 * 3.0) / 3)


def test_density_weights_the_grades_but_not_itself():
    holes = _simple_holes(density=[1.0, 4.0, 1.0])
    got = holes.composite_fixed(3.0).intervals["assay"].data
    holes.intervals["assay"].set_role("density", "density")
    weighted = holes.composite_fixed(3.0).intervals["assay"].data

    # without the density role both columns are plain length-weighted means
    assert got["grade"][0] == pytest.approx((2 * 1.0 + 1 * 2.0) / 3)
    # with it, the grade is weighted by length x density
    assert weighted["grade"][0] == pytest.approx(
        (2 * 1.0 * 1.0 + 1 * 4.0 * 2.0) / (2 * 1.0 + 1 * 4.0))
    # the density itself stays length-weighted
    assert weighted["density"][0] == pytest.approx((2 * 1.0 + 1 * 4.0) / 3)


def test_missing_values_are_skipped_not_propagated():
    holes = _simple_holes(grade=[1.0, np.nan, 3.0])
    got = holes.composite_fixed(3.0).intervals["assay"].data

    # the composite is the mean of what is actually there
    assert got["grade"][0] == pytest.approx(1.0)
    assert got["grade"][1] == pytest.approx(3.0)


def test_a_composite_with_no_data_is_missing():
    frame = pd.DataFrame({"HoleID": "H1", "From": [0.0, 8.0], "To": [2.0, 10.0],
                          "grade": [1.0, 2.0]})
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=10.0))
    with pytest.warns(UserWarning, match="gap"):
        holes.add_intervals("assay", frame, hole="HoleID", fr="From", to="To")

    got = holes.composite_fixed(2.0).intervals["assay"].data
    assert np.isnan(got["grade"][2])          # the 4-6 m run sits in the gap


def test_composite_to_reproduces_the_source():
    holes = _simple_holes()
    got = holes.composite_to("assay").intervals["assay"].data
    np.testing.assert_allclose(got["grade"].values.astype(float),
                               [1.0, 2.0, 3.0])


def test_residual_is_merged_into_the_last_run():
    holes = _simple_holes(edges=(0, 2, 4, 5), grade=(1.0, 2.0, 3.0))
    got = holes.composite_fixed(2.0).intervals["assay"].data

    # 5 m at 2 m gives two runs, not two plus a 1 m stub
    assert len(got) == 2
    assert list(got[FROM]) == [0.0, 2.0]
    assert list(got[TO]) == [2.0, 5.0]


def test_a_zone_shorter_than_the_composite_is_kept_whole():
    holes = _simple_holes(edges=(0, 1), grade=(5.0,))
    got = holes.composite_fixed(10.0).intervals["assay"].data

    assert len(got) == 1
    assert got[TO][0] == pytest.approx(1.0)
    assert got["grade"][0] == pytest.approx(5.0)


def test_composites_do_not_cross_a_domain_boundary():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=10.0))
    holes.add_intervals(
        "assay", _assay(edges=(0, 5, 10), grade=(1.0, 2.0)),
        hole="HoleID", fr="From", to="To")
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 5.0], "To": [5.0, 10.0],
                      "rock": ["ore", "waste"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")

    free = holes.composite_fixed(3.0).intervals["assay"].data
    honoured = holes.composite(3.0, domain="litho").intervals["assay"].data

    # ignoring geology, a run straddles the contact at 5 m and mixes the two
    assert np.any((free[FROM] < 5.0) & (free[TO] > 5.0))
    # honouring it, every run stops at the contact
    assert not np.any((honoured[FROM] < 5.0) & (honoured[TO] > 5.0))
    assert 5.0 in set(honoured[TO])


def test_domain_categories_survive_compositing():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=6.0))
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 4.0], "To": [4.0, 6.0],
                      "rock": ["ore", "waste"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")

    got = holes.composite(2.0, domain="litho").intervals["litho"].data
    assert list(got["rock"]) == ["ore", "ore", "waste"]


def test_the_longest_category_wins_a_mixed_composite():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=4.0))
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 1.0], "To": [1.0, 4.0],
                      "rock": ["ore", "waste"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")

    got = holes.composite_fixed(4.0).intervals["litho"].data
    assert len(got) == 1
    assert got["rock"][0] == "waste"          # 3 m against 1 m


def test_compositing_leaves_all_tables_on_one_support():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=6.0))
    holes.add_intervals("assay", _assay(), hole="HoleID", fr="From", to="To")
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 3.0], "To": [3.0, 6.0],
                      "rock": ["ore", "waste"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")

    composited = holes.composite_fixed(2.0)
    assay = composited.intervals["assay"].data
    litho = composited.intervals["litho"].data

    np.testing.assert_allclose(assay[FROM].values, litho[FROM].values)
    np.testing.assert_allclose(assay[TO].values, litho[TO].values)


# --------------------------------------------------------------------------- #
# conversion to point data
# --------------------------------------------------------------------------- #
def test_as_point_data_puts_points_at_the_interval_centre():
    holes = _simple_holes()
    point = holes.composite_to("assay").as_point_data()

    coordinates = np.asarray(point.coordinates)
    assert point.n_data == 3
    # a vertical hole from z = 100, intervals 0-2, 2-4, 4-6
    np.testing.assert_allclose(coordinates[:, 2], [99.0, 97.0, 95.0])
    np.testing.assert_allclose(
        np.asarray(point.variables["grade"].measurements.values),
        [1.0, 2.0, 3.0])


def test_as_point_data_position_can_be_moved():
    holes = _simple_holes()
    composited = holes.composite_to("assay")

    top = np.asarray(composited.as_point_data(position=0.0).coordinates)
    bottom = np.asarray(composited.as_point_data(position=1.0).coordinates)

    np.testing.assert_allclose(top[:, 2], [100.0, 98.0, 96.0])
    np.testing.assert_allclose(bottom[:, 2], [98.0, 96.0, 94.0])


def test_as_point_data_refuses_mismatched_supports():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=6.0))
    holes.add_intervals("assay", _assay(), hole="HoleID", fr="From", to="To")
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0], "To": [6.0],
                      "rock": ["ore"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")

    with pytest.raises(ValueError, match="different supports"):
        holes.as_point_data()

    # compositing is what brings them together
    assert holes.composite_fixed(2.0).as_point_data().n_data == 3


def test_as_point_data_builds_the_right_variable_types():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=6.0))
    holes.add_intervals(
        "assay", _assay(density=[2.5, 2.6, 2.7], rock=["a", "a", "b"]),
        hole="HoleID", fr="From", to="To", density="density",
        categorical="rock")

    point = holes.composite_to("assay").as_point_data()

    assert isinstance(point.variables["grade"], geoml.data.ContinuousVariable)
    assert isinstance(point.variables["density"], geoml.data.ContinuousVariable)
    assert isinstance(point.variables["rock"], geoml.data.CategoricalVariable)


def _two_holes():
    """Two holes with intervals of three different lengths."""
    holes = _drillholes(collar=_collar(holes=("H1", "H2"), dip=90.0,
                                       azimuth=0.0, length=6.0))
    holes.add_intervals(
        "assay",
        pd.DataFrame({"HoleID": ["H1", "H1", "H2"],
                      "From": [0.0, 2.0, 0.0], "To": [2.0, 6.0, 3.0],
                      "grade": [1.0, 2.0, 3.0]}),
        hole="HoleID", fr="From", to="To")
    return holes


def test_as_point_data_carries_the_hole_and_the_length():
    point = _two_holes().as_point_data()

    assert list(point.get_metadata(HOLE)) == ["H1", "H1", "H2"]
    np.testing.assert_allclose(point.get_metadata(LENGTH), [2.0, 4.0, 3.0])

    # metadata, not variables: the model must not see either of them
    assert HOLE not in point.variables
    assert LENGTH not in point.variables


def test_the_hole_survives_the_conversion_and_a_subset():
    point = _two_holes().as_point_data()
    kept = point[np.asarray(point.get_metadata(HOLE)) == "H1"]

    assert list(kept.get_metadata(HOLE)) == ["H1", "H1"]
    np.testing.assert_allclose(kept.get_metadata(LENGTH), [2.0, 4.0])


def test_empty_composites_are_dropped():
    frame = pd.DataFrame({"HoleID": "H1", "From": [0.0, 8.0], "To": [2.0, 10.0],
                          "grade": [1.0, 2.0]})
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=10.0))
    with pytest.warns(UserWarning, match="gap"):
        holes.add_intervals("assay", frame, hole="HoleID", fr="From", to="To")
    composited = holes.composite_fixed(2.0)

    assert composited.as_point_data(drop_missing=True).n_data < \
           composited.as_point_data(drop_missing=False).n_data


def test_composition_is_closed_without_a_rest():
    names, parts = _composition(pd.DataFrame({"pb": [1.0], "zn": [3.0]}))

    assert names == ["pb", "zn"]
    np.testing.assert_allclose(parts[0], [0.25, 0.75])


def _composition(frame, units=None, **group):
    """Runs a small table through the composition machinery.

    Values are read as plain fractions unless `units` says otherwise, so the
    expected results below can be written down directly.
    """
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0,
                                       length=2.0 * len(frame)))
    frame = frame.copy()
    frame["HoleID"] = "H1"
    frame["From"] = 2.0 * np.arange(len(frame))
    frame["To"] = frame["From"] + 2.0
    holes.add_intervals("assay", frame, hole="HoleID", fr="From", to="To")

    if units is None:
        units = {c: "fraction" for c in frame.columns
                 if c not in ("HoleID", "From", "To")}
    group.setdefault("columns", units)
    point = holes.composite_to("assay").as_point_data(
        compositional={"metals": group}, drop_missing=False)
    metals = point.variables["metals"]
    return list(metals.components.keys()), np.stack(
        [np.asarray(c.measurements.values)
         for c in metals.components.values()], axis=1)


def test_zeros_are_replaced_by_half_the_smallest_positive_of_their_column():
    names, parts = _composition(
        pd.DataFrame({"pb": [0.0, 4.0, 6.0], "zn": [1.0, 0.0, 3.0]}))

    # pb: smallest positive is 4 -> 2; zn: smallest positive is 1 -> 0.5
    raw = np.array([[2.0, 1.0], [4.0, 0.5], [6.0, 3.0]])
    np.testing.assert_allclose(parts, raw / raw.sum(axis=1, keepdims=True))


def test_units_are_converted_so_the_parts_can_be_added_up():
    names, parts = _composition(
        pd.DataFrame({"pb": [1.0], "ag": [10000.0]}),
        units={"pb": "%", "ag": "ppm"}, rest=True)

    # 1 % and 10000 ppm are the same fraction
    np.testing.assert_allclose(parts[0], [0.01, 0.01, 0.98])


def test_a_unit_may_be_given_as_a_number():
    names, parts = _composition(
        pd.DataFrame({"pb": [5.0]}), units={"pb": 100.0}, rest=True)
    np.testing.assert_allclose(parts[0], [0.05, 0.95])


def test_an_unknown_unit_is_rejected():
    with pytest.raises(ValueError, match="unknown unit"):
        _composition(pd.DataFrame({"pb": [1.0]}), units={"pb": "carats"})


def test_units_are_required():
    with pytest.raises(ValueError, match="mapping from column name to unit"):
        _composition(pd.DataFrame({"pb": [1.0], "zn": [1.0]}),
                     units=["pb", "zn"])


def test_the_rest_varies_with_how_much_was_measured():
    names, parts = _composition(
        pd.DataFrame({"pb": [1.0, 10.0], "zn": [1.0, 10.0]}),
        units={"pb": "%", "zn": "%"}, rest=True)

    assert names == ["pb", "zn", "rest"]
    # a barren sample is nearly all rest; a rich one much less so
    np.testing.assert_allclose(parts[:, 2], [0.98, 0.80])
    np.testing.assert_allclose(parts.sum(axis=1), 1.0)
    assert np.all(parts > 0)


def test_the_rest_falls_back_to_the_minimum_when_there_is_no_room():
    with pytest.warns(UserWarning, match="no room for a rest"):
        names, parts = _composition(
            pd.DataFrame({"pb": [1.0, 60.0], "zn": [1.0, 60.0]}),
            units={"pb": "%", "zn": "%"}, rest=True)

    # the second sample's parts sum to 1.2, so they are scaled to make room
    # for a rest of 0.01, the smallest part anywhere
    np.testing.assert_allclose(parts[0], [0.01, 0.01, 0.98])
    np.testing.assert_allclose(parts[1], [0.495, 0.495, 0.01])
    np.testing.assert_allclose(parts.sum(axis=1), 1.0)
    assert np.all(parts > 0)


def test_zeros_are_replaced_before_the_rest_is_worked_out():
    """The rest must account for the replacement, not the original zero."""
    names, parts = _composition(
        pd.DataFrame({"pb": [0.0, 0.4], "zn": [0.1, 0.2]}), rest=True)

    # pb becomes half of 0.4, so the first rest is 1 - 0.2 - 0.1, not 1 - 0.1
    np.testing.assert_allclose(parts[0], [0.2, 0.1, 0.7])
    np.testing.assert_allclose(parts.sum(axis=1), 1.0)


def test_a_column_with_no_positive_value_is_left_alone():
    with pytest.warns(UserWarning, match="no scale"):
        names, parts = _composition(
            pd.DataFrame({"pb": [0.0, 0.0], "zn": [1.0, 3.0]}))
    np.testing.assert_allclose(parts[:, 0], [0.0, 0.0])


def test_one_missing_part_makes_the_whole_row_missing():
    with pytest.warns(UserWarning, match="missing some but not all"):
        names, parts = _composition(
            pd.DataFrame({"pb": [1.0, np.nan, 3.0], "zn": [1.0, 2.0, 3.0]}))

    assert np.all(np.isnan(parts[1]))            # both parts, not just pb
    assert np.all(np.isfinite(parts[[0, 2]]))


def test_a_missing_row_stays_missing_through_rest_and_closure():
    with pytest.warns(UserWarning, match="missing some but not all"):
        names, parts = _composition(
            pd.DataFrame({"pb": [1.0, np.nan], "zn": [1.0, 2.0]}), rest=True)

    assert names == ["pb", "zn", "rest"]
    assert np.all(np.isnan(parts[1]))            # the rest is missing too
    np.testing.assert_allclose(parts[0].sum(), 1.0)


def test_a_missing_row_does_not_set_the_replacement_scale():
    """The dropped values must not count towards their column's minimum."""
    with pytest.warns(UserWarning, match="missing some but not all"):
        names, parts = _composition(
            pd.DataFrame({"pb": [0.0, 1.0, 8.0], "zn": [5.0, np.nan, 4.0]}))

    # pb's 1.0 belongs to a row that is missing, so the smallest positive
    # value left is 8 and the zero becomes 4
    raw = np.array([4.0, 5.0])
    np.testing.assert_allclose(parts[0], raw / raw.sum())


def test_the_same_grade_in_different_units_gives_the_same_composition():
    as_percent = _composition(pd.DataFrame({"pb": [1.0, 10.0]}),
                              units={"pb": "%"}, rest=True)[1]
    as_ppm = _composition(pd.DataFrame({"pb": [10000.0, 100000.0]}),
                          units={"pb": "ppm"}, rest=True)[1]
    np.testing.assert_allclose(as_percent, as_ppm)


# --------------------------------------------------------------------------- #
# contacts and classification input
# --------------------------------------------------------------------------- #
def _two_rock_holes():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=10.0))
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 2.0, 6.0],
                      "To": [2.0, 6.0, 10.0],
                      "rock": ["ore", "ore", "waste"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")
    return holes


def test_merge_domains_joins_touching_runs_of_one_category():
    merged = _two_rock_holes().merge_domains("litho").data

    assert len(merged) == 2                  # the two "ore" intervals join up
    assert list(merged[FROM]) == [0.0, 6.0]
    assert list(merged[TO]) == [6.0, 10.0]


def test_contacts_sit_at_the_boundary_and_carry_both_categories():
    contacts = _two_rock_holes().get_contacts("litho")

    assert contacts.n_data == 1
    # the contact is at 6 m down a vertical hole collared at z = 100
    np.testing.assert_allclose(np.asarray(contacts.coordinates)[0],
                               [0.0, 0.0, 94.0])
    rock = contacts.variables["rock"]
    assert rock.measurements_a.to_numpy()[0] == "ore"
    assert rock.measurements_b.to_numpy()[0] == "waste"


def test_classification_input_mixes_interior_points_with_contacts():
    point = _two_rock_holes().as_classification_input("litho", length=2.0)

    rock = point.variables["rock"]
    first = rock.measurements_a.to_numpy()
    second = rock.measurements_b.to_numpy()

    boundary = first != second
    assert boundary.sum() == 1               # one contact, at 6 m
    # interior points know only their own rock type
    assert set(first[~boundary]) == {"ore", "waste"}
    assert isinstance(point.variables["rock"], geoml.data.RockTypeVariable)


def test_contacts_carry_the_hole_but_no_length():
    contacts = _two_rock_holes().get_contacts("litho")

    assert list(contacts.get_metadata(HOLE)) == ["H1"]
    assert LENGTH not in contacts.metadata


def test_classification_input_gives_the_contacts_zero_length():
    point = _two_rock_holes().as_classification_input("litho", length=2.0)

    rock = point.variables["rock"]
    boundary = rock.measurements_a.to_numpy() != rock.measurements_b.to_numpy()

    length = point.get_metadata(LENGTH)
    assert set(point.get_metadata(HOLE)) == {"H1"}
    np.testing.assert_allclose(length[boundary], 0.0)
    np.testing.assert_allclose(length[~boundary], 2.0)


def test_classification_input_accepts_an_order():
    point = _two_rock_holes().as_classification_input(
        "litho", length=2.0, label_order=["waste", "ore"])
    assert list(point.variables["rock"].labels) == ["waste", "ore"]


def test_contacts_need_a_change_of_category():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=4.0))
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 2.0], "To": [2.0, 4.0],
                      "rock": ["ore", "ore"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")

    with pytest.raises(geoml.data.NoDataError, match="no contacts"):
        holes.get_contacts("litho")


# --------------------------------------------------------------------------- #
# domain resolution and reporting
# --------------------------------------------------------------------------- #
def test_a_domain_can_be_named_in_three_ways():
    holes = _two_rock_holes()
    by_table = holes.merge_domains("litho").data
    by_column = holes.merge_domains("rock").data
    by_pair = holes.merge_domains(("litho", "rock")).data

    assert by_table.equals(by_column) and by_table.equals(by_pair)


def test_an_ambiguous_domain_must_be_named_in_full():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=4.0))
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": "H1", "From": [0.0], "To": [4.0],
                      "rock": ["ore"], "alteration": ["strong"]}),
        hole="HoleID", fr="From", to="To",
        categorical=["rock", "alteration"])

    with pytest.raises(ValueError, match="name one explicitly"):
        holes.merge_domains("litho")


def test_repeated_holes_in_the_collar_are_rejected():
    with pytest.raises(ValueError, match="repeated holes"):
        _drillholes(collar=_collar(holes=("H1", "H1"), dip=90.0, azimuth=0.0))


def _assay_holes():
    """One hole with a grade and a lithology column, both named badly."""
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=10.0))
    holes.add_intervals(
        "assay",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 5.0], "To": [5.0, 10.0],
                      "Pb_pct_ICP": [1.0, 3.0], "rock_code": ["ore", "waste"]}),
        hole="HoleID", fr="From", to="To", categorical="rock_code")
    return holes


def test_rename_carries_the_roles_with_the_columns():
    """Renaming the frame directly leaves the roles behind, keyed by old name."""
    holes = _assay_holes()
    holes.rename("assay", {"Pb_pct_ICP": "Pb", "rock_code": "litho"})
    table = holes.intervals["assay"]

    assert table.value_columns == ["Pb", "litho"]
    assert table.columns_with_role("grade") == ["Pb"]
    assert table.columns_with_role("categorical") == ["litho"]
    assert "Pb_pct_ICP" not in table.roles


def test_a_renamed_table_still_composites_and_converts():
    holes = _assay_holes()
    holes.rename("assay", {"Pb_pct_ICP": "Pb"})

    composited = holes.composite(5.0)
    assert "Pb" in composited.intervals["assay"].data.columns
    assert list(composited.intervals["assay"].data["Pb"]) == [1.0, 3.0]

    points = composited.as_point_data()
    assert "Pb" in points.variables


def test_renaming_only_some_columns_leaves_the_rest_alone():
    holes = _assay_holes()
    holes.rename("assay", {"Pb_pct_ICP": "Pb"})
    table = holes.intervals["assay"]

    assert table.value_columns == ["Pb", "rock_code"]
    assert table.roles["rock_code"] == "categorical"


def test_rename_reports_columns_that_are_not_there():
    holes = _assay_holes()
    with pytest.raises(ValueError, match="not in table"):
        holes.rename("assay", {"Zn": "zinc"})


def test_the_hole_and_depth_columns_cannot_be_renamed():
    holes = _assay_holes()
    with pytest.raises(ValueError, match="reserved"):
        holes.rename("assay", {FROM: "depth"})
    with pytest.raises(ValueError, match="reserved"):
        holes.rename("assay", {"Pb_pct_ICP": HOLE})


def test_a_rename_cannot_shadow_another_column():
    holes = _assay_holes()
    with pytest.raises(ValueError, match="collide"):
        holes.rename("assay", {"Pb_pct_ICP": "rock_code"})
    with pytest.raises(ValueError, match="collide"):
        holes.rename("assay", {"Pb_pct_ICP": "x", "rock_code": "x"})


def test_renaming_an_unknown_table_is_an_error():
    holes = _assay_holes()
    with pytest.raises(ValueError, match="no table named"):
        holes.rename("litho", {"Pb_pct_ICP": "Pb"})


def test_the_table_can_be_renamed_on_its_own():
    """`IntervalTable.rename` chains, as `set_role` does."""
    table = _assay_holes().intervals["assay"]
    assert table.rename({"Pb_pct_ICP": "Pb"}) is table
    assert "Pb: grade" in str(table)


def test_str_reports_the_tables_and_their_roles():
    holes = _two_rock_holes()

    assert "1 drillholes" in str(holes)
    assert "litho" in str(holes)
    assert "rock: categorical" in str(holes.intervals["litho"])
    # a container summarizes itself when named on its own, as those in `data` do
    assert repr(holes) == str(holes)


# --------------------------------------------------------------------------- #
# subsetting and grouping
# --------------------------------------------------------------------------- #
def _field():
    """Three vertical holes 100 m apart, each with a lithology log."""
    collar = pd.DataFrame({
        "HoleID": ["H1", "H2", "H3"],
        "X": [0.0, 100.0, 200.0], "Y": [0.0, 0.0, 0.0],
        "Z": [100.0, 100.0, 100.0], "Length": [10.0, 10.0, 10.0],
        "Dip": 90.0, "Azimuth": 0.0})
    holes = _drillholes(collar=collar)
    holes.add_intervals(
        "litho",
        pd.DataFrame({"HoleID": ["H1", "H1", "H2", "H2", "H3"],
                      "From": [0.0, 6.0, 0.0, 5.0, 0.0],
                      "To": [6.0, 10.0, 5.0, 10.0, 10.0],
                      "rock": ["ore", "waste", "waste", "waste", "ore"]}),
        hole="HoleID", fr="From", to="To", categorical="rock")
    return holes


def test_subset_holes_keeps_the_named_holes_whole():
    subset = _field().subset_holes(["H1", "H3"])

    assert list(subset.collar.index) == ["H1", "H3"]
    assert subset.n_holes == 2
    # H1 keeps both of its intervals, not only the matching one
    assert list(subset.intervals["litho"].holes) == ["H1", "H3"]
    assert len(subset.intervals["litho"]) == 3


def test_subsetting_leaves_the_original_alone():
    holes = _field()
    holes.subset_holes("H1")

    assert holes.n_holes == 3
    assert len(holes.intervals["litho"]) == 5


def test_unknown_holes_are_reported_and_skipped():
    with pytest.warns(UserWarning, match="not in the collar"):
        subset = _field().subset_holes(["H1", "GHOST"])
    assert list(subset.collar.index) == ["H1"]

    with pytest.raises(geoml.data.NoDataError, match="none of the given"):
        _field().subset_holes("GHOST")


def test_subset_region_selects_collars_in_plan_view():
    subset = _field().subset_region([-1.0, -1.0], [150.0, 1.0])

    assert list(subset.collar.index) == ["H1", "H2"]
    # the bounding box follows the holes that were kept
    assert subset.bounding_box.max[0, 0] == pytest.approx(100.0)


def test_subset_region_accepts_a_bounding_box_and_three_dimensions():
    holes = _field()
    whole = holes.subset_region(holes.bounding_box.min, holes.bounding_box.max)
    assert whole.n_holes == 3

    with pytest.raises(geoml.data.NoDataError, match="no collar"):
        holes.subset_region([1000.0, 1000.0], [2000.0, 2000.0])


def test_subset_region_checks_its_arguments():
    with pytest.raises(ValueError, match="same length"):
        _field().subset_region([0.0], [1.0])


def test_a_subset_can_still_be_composited():
    point = _field().subset_holes(["H1", "H3"]) \
        .composite(2.0, domain="litho").as_point_data()

    assert point.n_data > 0
    assert np.all(np.isfinite(np.asarray(point.coordinates)))


def test_filter_intervals_keeps_the_matching_rows():
    holes = _field().filter_intervals("litho", "rock", "ore")
    data = holes.intervals["litho"].data

    assert set(data["rock"]) == {"ore"}
    assert list(data[HOLE]) == ["H1", "H3"]
    # the collars are untouched: only the intervals were cut
    assert holes.n_holes == 3


def test_filter_intervals_can_keep_whole_holes():
    holes = _field().filter_intervals("litho", "rock", "ore",
                                      whole_holes=True)

    assert list(holes.collar.index) == ["H1", "H3"]
    # H1 comes through entire, waste included
    assert set(holes.intervals["litho"].data["rock"]) == {"ore", "waste"}


def test_filter_intervals_needs_a_match():
    with pytest.raises(geoml.data.NoDataError, match="no interval"):
        _field().filter_intervals("litho", "rock", "gneiss")

    with pytest.raises(ValueError, match="not in table"):
        _field().filter_intervals("litho", "nonsense", "ore")


def test_category_legend_measures_each_label():
    legend = _field().category_legend("rock")

    assert list(legend["label"]) == ["ore", "waste"]      # sorted by length
    assert list(legend["length"]) == [16.0, 14.0]
    assert list(legend["n_intervals"]) == [2, 3]
    # the group column starts as a copy, ready to be edited
    assert list(legend["group"]) == ["ore", "waste"]


def test_category_legend_accounts_for_unlogged_intervals():
    holes = _field()
    data = holes.intervals["litho"].data
    data.loc[0, "rock"] = None                  # 6 m of H1 loses its code

    legend = holes.category_legend("rock")
    missing = legend["label"].isna()

    assert missing.sum() == 1
    assert legend.loc[missing, "length"].iloc[0] == 6.0
    # every interval of the table is now accounted for
    assert legend["length"].sum() == pytest.approx(
        holes.intervals["litho"].length.sum())


def test_group_categories_lumps_labels_together():
    holes = _field().group_categories(
        "rock", {"mineralised": ["ore"]}, other="barren")

    assert set(holes.intervals["litho"].data["rock"]) \
           == {"mineralised", "barren"}
    assert holes.intervals["litho"].roles["rock"] == "categorical"


def test_group_categories_can_keep_the_original_column():
    holes = _field().group_categories(
        "rock", {"mineralised": "ore", "barren": "waste"},
        new_column="domain")
    data = holes.intervals["litho"].data

    assert list(data["rock"]) == ["ore", "waste", "waste", "waste", "ore"]
    assert list(data["domain"]) == ["mineralised", "barren", "barren",
                                    "barren", "mineralised"]


def test_labels_left_out_of_the_groups_are_reported():
    with pytest.warns(UserWarning, match="left out"):
        holes = _field().group_categories("rock", {"mineralised": "ore"})

    # they keep their own label rather than disappearing
    assert set(holes.intervals["litho"].data["rock"]) \
           == {"mineralised", "waste"}


def test_groups_naming_a_label_that_is_not_there_are_reported():
    with pytest.warns(UserWarning, match="do not appear"):
        _field().group_categories(
            "rock", {"mineralised": ["ore", "gneiss"], "barren": "waste"})


def test_group_categories_accepts_an_edited_legend():
    holes = _field()
    legend = holes.category_legend("rock")
    legend["group"] = ["mineralised", "barren"]      # as edited in a spreadsheet

    grouped = holes.group_categories("rock", legend)
    assert set(grouped.intervals["litho"].data["rock"]) \
           == {"mineralised", "barren"}

    with pytest.raises(ValueError, match="category_legend"):
        holes.group_categories("rock", legend[["label"]])


def test_a_label_cannot_be_in_two_groups():
    with pytest.raises(ValueError, match="placed in both"):
        _field().group_categories("rock", {"a": ["ore"], "b": ["ore"]})


def test_grouping_leaves_the_original_object_alone():
    holes = _field()
    holes.group_categories("rock", {"mineralised": "ore"}, other="barren")

    assert set(holes.intervals["litho"].data["rock"]) == {"ore", "waste"}


def test_a_grouped_column_can_be_used_as_a_domain():
    holes = _field().group_categories(
        "rock", {"mineralised": "ore", "barren": "waste"})
    composited = holes.composite(2.0, domain="litho")

    data = composited.intervals["litho"].data
    # H1's contact at 6 m is still honoured after grouping
    assert 6.0 in list(data.loc[data[HOLE] == "H1", TO])


def _ore_only():
    """Three 10 m holes in which only the ore was logged.

    H1 has ore at 2-4 m, H2 at 0-3 m and an interval with no code at 5-6 m,
    and H3 was never logged at all.
    """
    collar = pd.DataFrame({
        "HoleID": ["H1", "H2", "H3"],
        "X": [0.0, 100.0, 200.0], "Y": [0.0, 0.0, 0.0],
        "Z": [100.0, 100.0, 100.0], "Length": [10.0, 10.0, 10.0],
        "Dip": 90.0, "Azimuth": 0.0})
    holes = _drillholes(collar=collar)
    with pytest.warns(UserWarning, match="gap"):
        holes.add_intervals(
            "litho",
            pd.DataFrame({"HoleID": ["H1", "H2", "H2"],
                          "From": [2.0, 0.0, 5.0], "To": [4.0, 3.0, 6.0],
                          "rock": ["ore", "ore", None]}),
            hole="HoleID", fr="From", to="To", categorical="rock")
    return holes


def test_fill_unlogged_covers_every_hole_end_to_end():
    with pytest.warns(UserWarning, match="no interval"):
        holes = _ore_only().fill_unlogged("litho", "waste")
    data = holes.intervals["litho"].data

    h1 = data.loc[data[HOLE] == "H1"]
    assert list(h1[FROM]) == [0.0, 2.0, 4.0]
    assert list(h1[TO]) == [2.0, 4.0, 10.0]
    assert list(h1["rock"]) == ["waste", "ore", "waste"]

    # the hole that was never logged comes out waste from end to end
    h3 = data.loc[data[HOLE] == "H3"]
    assert list(h3[FROM]) == [0.0] and list(h3[TO]) == [10.0]
    assert list(h3["rock"]) == ["waste"]


def test_fill_unlogged_labels_intervals_that_carry_no_value():
    with pytest.warns(UserWarning):
        holes = _ore_only().fill_unlogged("litho", "waste")
    data = holes.intervals["litho"].data

    # H2's 5-6 m interval existed but had no code
    kept = data.loc[(data[HOLE] == "H2") & (data[FROM] == 5.0)]
    assert list(kept["rock"]) == ["waste"]
    assert not data["rock"].isna().any()


def test_filling_leaves_no_gaps_behind():
    with pytest.warns(UserWarning):
        holes = _ore_only().fill_unlogged("litho", "waste")

    report = holes.validate(on_error="ignore")
    assert len(report) == 0

    # 3 holes of 10 m, all of it now logged
    legend = holes.category_legend("litho")
    assert legend["length"].sum() == pytest.approx(30.0)
    assert not legend["label"].isna().any()


def test_fill_unlogged_can_leave_the_ends_alone():
    holes = _ore_only().fill_unlogged("litho", "waste", ends=False)
    data = holes.intervals["litho"].data

    h1 = data.loc[data[HOLE] == "H1"]
    assert list(h1[FROM]) == [2.0] and list(h1[TO]) == [4.0]
    # only H2's interior gap, 3-5 m, was filled
    assert list(data.loc[data["rock"] == "waste", FROM]) == [3.0, 5.0]
    assert "H3" not in set(data[HOLE])


def test_fill_unlogged_reports_holes_of_unknown_length():
    holes = _ore_only()
    holes.collar.loc["H1", "LENGTH"] = np.nan

    with pytest.warns(UserWarning, match="no recorded length"):
        filled = holes.fill_unlogged("litho", "waste")

    data = filled.intervals["litho"].data
    h1 = data.loc[data[HOLE] == "H1"]
    assert list(h1[TO]) == [2.0, 4.0]        # nothing added past 4 m


def test_filling_leaves_the_original_alone():
    holes = _ore_only()
    with pytest.warns(UserWarning):
        holes.fill_unlogged("litho", "waste")

    assert len(holes.intervals["litho"]) == 3
    assert holes.intervals["litho"].data["rock"].isna().any()


def test_filled_holes_can_be_used_as_classification_input():
    with pytest.warns(UserWarning):
        holes = _ore_only().fill_unlogged("litho", "waste")

    point = holes.as_classification_input("litho", length=2.0)
    rock = point.variables["rock"]
    first = rock.measurements_a.to_numpy()
    second = rock.measurements_b.to_numpy()

    assert set(first) == {"ore", "waste"}
    # the ore/waste boundaries are now explicit contacts
    assert (first != second).sum() > 0


# --------------------------------------------------------------------------- #
# the bundled dataset
# --------------------------------------------------------------------------- #
def test_ararangua_loads_and_converts():
    import geoml.datasets as datasets

    holes = datasets.ararangua()
    assert holes.n_holes == 13
    assert len(holes.intervals["lito"]) == 507

    point = holes.as_classification_input(("lito", "Lito"), length=10.0)
    assert point.n_data > 0
    assert np.all(np.isfinite(np.asarray(point.coordinates)))


# --------------------------------------------------------------------------- #
# renaming and dropping tables
# --------------------------------------------------------------------------- #
def test_a_table_can_be_renamed_in_place():
    holes = _assay_holes()
    holes.add_intervals(
        "extra",
        pd.DataFrame({"HoleID": "H1", "From": [0.0], "To": [10.0],
                      "d": [2.5]}),
        hole="HoleID", fr="From", to="To")

    holes.rename_table("assay", "chemistry")

    # same table, new name, same position -- the order is what
    # `as_point_data` merges by default
    assert list(holes.intervals) == ["chemistry", "extra"]
    assert holes.intervals["chemistry"].name == "chemistry"
    assert holes.intervals["chemistry"].value_columns == \
        ["Pb_pct_ICP", "rock_code"]

    points = holes.composite(5.0).as_point_data("chemistry")
    assert "Pb_pct_ICP" in points.variables


def test_renaming_refuses_what_is_not_there_and_what_already_is():
    holes = _assay_holes()
    with pytest.raises(ValueError, match="no table named"):
        holes.rename_table("nope", "other")

    holes.add_intervals(
        "extra",
        pd.DataFrame({"HoleID": "H1", "From": [0.0], "To": [10.0],
                      "d": [2.5]}),
        hole="HoleID", fr="From", to="To")
    with pytest.raises(ValueError, match="already a table named"):
        holes.rename_table("assay", "extra")
    # renaming to itself is a no-op, not a collision
    holes.rename_table("assay", "assay")


def test_a_table_can_be_dropped():
    holes = _assay_holes()
    holes.add_intervals(
        "extra",
        pd.DataFrame({"HoleID": "H1", "From": [0.0], "To": [10.0],
                      "d": [2.5]}),
        hole="HoleID", fr="From", to="To")

    holes.drop_table("extra")
    assert list(holes.intervals) == ["assay"]

    with pytest.raises(ValueError, match="no table named"):
        holes.drop_table("extra")


# --------------------------------------------------------------------------- #
# the recovery role
# --------------------------------------------------------------------------- #
def _recovery_holes():
    holes = _drillholes(collar=_collar(dip=90.0, azimuth=0.0, length=10.0))
    holes.add_intervals(
        "assay",
        pd.DataFrame({"HoleID": "H1", "From": [0.0, 5.0], "To": [5.0, 10.0],
                      "Pb": [1.0, 3.0], "rec": [1.0, 0.5]}),
        hole="HoleID", fr="From", to="To", recovery="rec")
    return holes


def test_recovery_is_a_role_of_its_own():
    table = _recovery_holes().intervals["assay"]

    assert table.columns_with_role("recovery") == ["rec"]
    assert table.columns_with_role("grade") == ["Pb"]   # not swept up
    assert "recovery" in str(table)


def test_recovery_composites_by_length_and_never_weights_the_grade():
    """A fraction of a length is exactly what length-weighting averages; how
    much a poorly recovered assay should count is a modelling decision, not a
    compositing one."""
    holes = _recovery_holes()
    table = holes.composite(10.0).intervals["assay"]

    assert np.allclose(table.data["rec"], [0.75])       # (1.0 + 0.5) / 2
    assert np.allclose(table.data["Pb"], [2.0])         # unweighted by rec


def test_recovery_reaches_the_points_as_metadata():
    """It describes the sample rather than the ground, so it rides beside
    HOLEID and LENGTH where the models never see it."""
    points = _recovery_holes().as_point_data()

    assert "rec" not in points.variables
    assert np.allclose(points.get_metadata("rec"), [1.0, 0.5])


def test_recovery_survives_a_column_rename():
    holes = _recovery_holes()
    holes.rename("assay", {"rec": "core_recovery"})

    assert holes.intervals["assay"].columns_with_role("recovery") == \
        ["core_recovery"]
