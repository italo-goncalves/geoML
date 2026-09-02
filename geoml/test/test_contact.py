"""Contact analysis on the data alone: a grade against its signed distance
down the hole to the nearest contact between two domains.

The holes are vertical and the profiles planted, so every number can be
worked out by hand: a step stays a step, a ramp stays a ramp, the sign runs
the way the pair says, a hole without the contact is left out and counted,
and a third domain beyond the far one is kept off the profile only when the
samples say which domain they are in.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.plots.prepare as prep
from geoml.data.drillhole import DrillholeData, HOLE, DEPTH


def _database(logs, grade):
    """Vertical holes logged as `{hole: [(from, to, rock), ...]}`, with
    one-metre assays of `grade(hole, depth)` down each logged length."""
    names = list(logs)
    collar = pd.DataFrame({
        "HoleID": names, "X": np.zeros(len(names)),
        "Y": 10.0 * np.arange(len(names)), "Z": np.full(len(names), 100.0),
        "Length": np.full(len(names), 100.0), "Dip": 90.0, "Azimuth": 0.0})
    holes = DrillholeData(collar, hole="HoleID", x="X", y="Y", z="Z",
                          length="Length", dip="Dip", azimuth="Azimuth")
    lito = pd.DataFrame([(h, a, b, r) for h, runs in logs.items()
                         for a, b, r in runs],
                        columns=["HoleID", "FROM", "TO", "rock"])
    holes.add_intervals("lito", lito, hole="HoleID", fr="FROM", to="TO",
                        categorical=["rock"])
    rows = [(h, float(k), float(k + 1), grade(h, k + 0.5))
            for h, runs in logs.items()
            for k in range(int(max(b for _, b, _ in runs)))]
    assay = pd.DataFrame(rows, columns=["HoleID", "FROM", "TO", "Au"])
    holes.add_intervals("assay", assay, hole="HoleID", fr="FROM", to="TO")
    return holes


CONTACT = {"H1": 50.0, "H2": 40.0}


def _step(hole, depth):
    return 1.0 if depth < CONTACT[hole] else 3.0


def _ramp(hole, depth):
    return 1.0 + 2.0 * np.clip((depth - CONTACT[hole] + 10.0) / 20.0, 0, 1)


def _two_holes(grade):
    holes = _database({"H1": [(0, 50, "A"), (50, 100, "B")],
                       "H2": [(0, 40, "A"), (40, 100, "B")]}, grade)
    return holes.as_point_data("assay"), holes.get_contacts(("lito", "rock"))


def test_a_planted_step_stays_a_step():
    data, contacts = _two_holes(_step)
    result = prep.contact(data, "Au", contacts, ("A", "B"), bins=5,
                          max_distance=50.0)
    assert np.allclose(result["lo"], np.arange(-50, 50, 10))
    negative, positive = result["centre"] < 0, result["centre"] > 0
    assert np.allclose(result["mean"][negative], 1.0)
    assert np.allclose(result["mean"][positive], 3.0)
    assert result["count"].sum() == 200 - 10  # H2's last ten metres lie past 50
    assert result["n_unplaced"] == 0 and result["weighted"]
    assert len(result["rows"]) == 200

    # the sample straddling the contact from above sits half a metre before it
    hole = np.asarray(data.get_metadata(HOLE))[result["rows"]]
    depth = np.asarray(data.get_metadata(DEPTH), dtype=float)[result["rows"]]
    assert result["distance"][(hole == "H1") & (depth == 49.5)] == \
        pytest.approx(-0.5)
    assert result["distance"][(hole == "H2") & (depth == 40.5)] == \
        pytest.approx(0.5)


def test_the_pair_order_sets_the_sign():
    data, contacts = _two_holes(_step)
    ab = prep.contact(data, "Au", contacts, ("A", "B"), bins=5,
                      max_distance=50.0)
    ba = prep.contact(data, "Au", contacts, ("B", "A"), bins=5,
                      max_distance=50.0)
    assert np.allclose(ba["distance"], -ab["distance"])
    assert np.allclose(ba["mean"], ab["mean"][::-1], equal_nan=True)


def test_a_planted_ramp_stays_a_ramp():
    data, contacts = _two_holes(_ramp)
    result = prep.contact(data, "Au", contacts, ("A", "B"), bins=5,
                          max_distance=50.0)
    assert result["mean"][0] == pytest.approx(1.0)
    assert result["mean"][-1] == pytest.approx(3.0)
    assert np.all(np.diff(result["mean"]) >= -1e-9)
    central = np.abs(result["centre"]) < 10
    assert np.all(np.diff(result["mean"][central]) > 0.1)
    # the band brackets the mean wherever the ramp varies inside a bin
    assert np.all(result["band_lo"] <= result["mean"] + 1e-9)
    assert np.all(result["band_hi"] >= result["mean"] - 1e-9)


def test_a_hole_without_the_contact_is_left_out_and_counted():
    holes = _database({"H1": [(0, 50, "A"), (50, 100, "B")],
                       "H3": [(0, 100, "A")]},
                      lambda hole, depth: _step("H1", depth))
    data = holes.as_point_data("assay")
    contacts = holes.get_contacts(("lito", "rock"))
    result = prep.contact(data, "Au", contacts, ("A", "B"))
    assert result["n_unplaced"] == 100
    hole = np.asarray(data.get_metadata(HOLE))
    assert set(hole[result["rows"]]) == {"H1"}


def test_a_third_domain_stays_off_the_profile_when_the_samples_say_so():
    grade = {"A": 1.0, "B": 3.0, "C": 9.0}
    logs = {"H4": [(0, 30, "A"), (30, 60, "B"), (60, 100, "C")]}

    def by_rock(hole, depth):
        return grade[next(r for a, b, r in logs[hole] if a <= depth < b)]

    holes = _database(logs, by_rock)
    contacts = holes.get_contacts(("lito", "rock"))
    # two contacts logged, one of the pair; without the samples' own domain
    # the C samples read as far B
    plain = prep.contact(holes.as_point_data("assay"), "Au", contacts,
                         ("A", "B"))
    assert plain["distance"].max() > 60.0
    assert plain["n_outside"] == 0

    composited = holes.composite_to("assay").as_point_data()
    aware = prep.contact(composited, "Au", contacts, ("A", "B"),
                         domain="rock")
    assert aware["n_outside"] == 40
    assert aware["distance"].max() < 30.5
    assert aware["value"].max() == pytest.approx(3.0)


def test_edges_can_be_given_and_the_wrong_inputs_are_refused():
    data, contacts = _two_holes(_step)
    result = prep.contact(data, "Au", contacts, ("A", "B"),
                          bins=[-20, -10, 0, 10, 20])
    assert np.allclose(result["lo"], [-20, -10, 0, 10])
    assert result["count"].sum() == 2 * 40

    with pytest.raises(ValueError, match="no contact"):
        prep.contact(data, "Au", contacts, ("A", "C"))
    frame = pd.DataFrame(np.asarray(data.coordinates), columns=["X", "Y", "Z"])
    bare = geoml.data.PointData(frame, ["X", "Y", "Z"])
    bare.add_continuous_variable("Au", np.asarray(data.values("Au/measurements")))
    with pytest.raises(ValueError, match="DEPTH|HOLEID"):
        prep.contact(bare, "Au", contacts, ("A", "B"))


def test_both_backends_draw_the_profile():
    import matplotlib
    matplotlib.use("Agg")
    data, contacts = _two_holes(_ramp)

    figure = geoml.plots.Explorer(data, continuous="Au").contact(
        contacts, ("A", "B"), bins=5, max_distance=50.0)
    assert len(figure.axes) == 2
    matplotlib.pyplot.close(figure)

    figure = geoml.plots.Interactive(data, continuous="Au").contact(
        contacts, ("A", "B"), bins=5, max_distance=50.0)
    names = [trace.name for trace in figure.data]
    assert "samples" in names and "mean by length" in names
    samples = figure.data[names.index("samples")]
    # the samples are locations: they carry the row index the dashboard links on
    assert len(samples.customdata) == 200
    assert len(figure.data) == 5
