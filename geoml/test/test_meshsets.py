"""Mesh sets: every contour of a column, at every cut-off, as one set.

A radial field -- fifty less the distance to the model's centre -- makes
every shell a sphere of known radius, and realizations planted as that
field plus a constant move each sphere by exactly the constant, so volumes,
bands, spacings and the dispersion across realizations all have answers
worked out by hand. Categories are three slabs across the model, their
draws the distance to each slab's middle, so the bodies meet along known
planes.
"""
import gc
import os
import warnings

import ezdxf
import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np
import pytest

import geoml
import geoml.data.meshsets as msm
import geoml.math.geometry as gmt
import geoml.plots.prepare as prep
from geoml.data import DTM3D, MeshSet, Solid3D

CENTRE = 35.0
OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def _sphere(radius):
    return 4.0 / 3.0 * np.pi * radius ** 3


def _radial(cutoffs=(20.0, 30.0, 40.0), offsets=OFFSETS):
    """Blocks of 5 m over an 80 m box, `g` fifty less the distance to the
    centre, each realization shifted by one of `offsets`."""
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2), max_levels=2)
    blocks = blocks.split(np.arange(blocks.n_data))
    distance = np.linalg.norm(np.asarray(blocks.coordinates) - CENTRE, axis=1)
    blocks.add_continuous_variable("g", 50.0 - distance)
    var = blocks.variables["g"]
    var.prediction.values[:] = 50.0 - distance
    if offsets is not None:
        var.allocate_simulations(len(offsets))
        var.simulations[:, :] = (50.0 - distance)[:, None] + offsets[None, :]
    if cutoffs is not None:
        var.set_cutoffs(list(cutoffs))
    return blocks


def _terrain(z):
    """A flat terrain at height `z`, reaching past the model on every side."""
    points = np.array([[-20, -20, z], [90, -20, z], [-20, 90, z],
                       [90, 90, z]], dtype=float)
    triangles = np.array([[0, 1, 3], [0, 3, 2]])
    return DTM3D(points, triangles, gmt.vertex_normals(points, triangles))


def _box(low, high):
    corners = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1]
                        for i in range(8)], dtype=float)
    return msm._box_body(np.asarray(low) + corners * (np.asarray(high)
                                                      - np.asarray(low)))


def _slabs(shifts=(-1.0, 0.0, 1.0)):
    """Three categories in slabs across x, meeting at x = 20 and 50, each
    realization moving A and C's draws apart by a planted amount."""
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2), max_levels=1)
    x = np.asarray(blocks.coordinates)[:, 0]
    blocks.add_rock_type_variable("Rock", labels=["A", "B", "C"])
    rock = blocks.variables["Rock"]
    draws = np.stack([15.0 - np.abs(x - middle)
                      for middle in (5.0, 35.0, 65.0)], axis=1)
    skew = msm._category_fields(draws, "largest")
    for j, name in enumerate("ABC"):
        rock.components[name].indicator_predicted.values[:] = skew[:, j]
        rock.components[name].allocate_simulations(len(shifts))
        rock.components[name].simulations[:, :] = \
            draws[:, j:j + 1] + np.asarray(shifts)[None, :] * (1 - j)
    rock.predicted.values[:] = np.argmax(draws, axis=1)
    return blocks


@pytest.fixture(scope="module")
def radial():
    return _radial()


@pytest.fixture(scope="module")
def shells(radial):
    return MeshSet(radial, "g", workers=2)


# --------------------------------------------------------------------------- #
# the mapping
# --------------------------------------------------------------------------- #
def test_a_set_is_keyed_by_the_variables_own_cutoffs(shells):
    assert list(shells) == [20.0, 30.0, 40.0]
    assert len(shells) == 3
    assert 30 in shells and 30.0 in shells and 35.0 not in shells
    # an int names the same cut-off as the float it was declared as
    assert shells[30] is shells[30.0]
    with pytest.raises(KeyError, match="20, 30, 40"):
        shells[35.0]
    assert all(isinstance(mesh, Solid3D) for mesh in shells.values())


def test_every_shell_is_the_sphere_its_field_describes(shells):
    for cutoff in shells:
        assert shells[cutoff].volume == pytest.approx(
            _sphere(50.0 - cutoff), rel=0.05)
    # one field, so the shells nest, measured exactly
    assert np.allclose(shells.check()["volume"], 0.0)


def test_a_column_with_no_cutoffs_declared_is_refused_until_given_some():
    blocks = _radial(cutoffs=None, offsets=None)
    with pytest.raises(ValueError, match="set_cutoffs"):
        MeshSet(blocks, "g")
    given = MeshSet(blocks, "g", cutoffs=[30.0, 20.0, 30.0])
    assert list(given) == [20.0, 30.0]


def test_a_level_the_field_never_reaches_is_an_empty_body():
    blocks = _radial(cutoffs=(30.0, 60.0))
    shells = MeshSet(blocks, "g", workers=1)
    assert shells[60.0].n_data == 0 and shells[60.0].volume == 0.0
    assert shells.simulations[4][60.0].n_data == 0


def test_the_bands_add_up_to_the_outermost_shell(shells):
    spans = list(shells.bands)
    assert spans == [(20.0, 30.0), (30.0, 40.0), (40.0, np.inf)]
    # to Manifold's welding of the vertices, a millionth of a unit
    total = sum(shells.bands[span].volume for span in spans)
    assert total == pytest.approx(shells[20.0].volume, rel=1e-7)
    # a band is the shell at its lower cut-off less the one at its upper
    assert shells.bands[(20.0, 30.0)].volume == pytest.approx(
        shells[20.0].volume - shells[30.0].volume, rel=1e-7)
    with pytest.raises(KeyError, match="the bands are"):
        shells.bands[(20.0, 40.0)]


def test_closed_below_a_set_nests_the_other_way(radial):
    below = MeshSet(radial, "g", close="below", simulations=False)
    assert below[40.0].volume > below[30.0].volume > below[20.0].volume
    assert np.allclose(below.check()["volume"], 0.0)
    spans = list(below.bands)
    assert spans[0] == (-np.inf, 20.0)
    assert sum(below.bands[s].volume for s in spans) == pytest.approx(
        below[40.0].volume, rel=1e-7)


def test_a_set_of_sheets_is_refused(radial):
    with pytest.raises(ValueError, match="every contour closes"):
        MeshSet(radial, "g", close=False)


# --------------------------------------------------------------------------- #
# limits
# --------------------------------------------------------------------------- #
def test_a_terrain_keeps_what_lies_below_it_and_says_what_it_took(radial):
    through_the_centre = _terrain(CENTRE)
    below = MeshSet(radial, "g", limits={"topography": through_the_centre},
                    simulations=False)
    above = MeshSet(radial, "g", exclude={"topography": through_the_centre},
                    simulations=False)
    table = below.table()
    for cutoff in below:
        raw = table.loc[cutoff, "raw_volume"]
        assert below[cutoff].volume == pytest.approx(raw / 2, rel=1e-3)
        assert table.loc[cutoff, "removed: topography"] == pytest.approx(
            raw - below[cutoff].volume)
        # the two halves make the whole, to the welding of the vertices
        assert below[cutoff].volume + above[cutoff].volume == pytest.approx(
            raw, rel=1e-7)
        assert below[cutoff].bounding_box.max[0, 2] <= CENTRE + 1e-6
        assert above[cutoff].bounding_box.min[0, 2] >= CENTRE - 1e-6


def test_a_body_limit_keeps_its_inside_and_every_realization_is_cut(radial):
    half = _box([-10.0, -10.0, -10.0], [CENTRE, 90.0, 90.0])
    cut = MeshSet(radial, "g", limits={"west": half}, workers=2)
    for cutoff in cut:
        assert cut[cutoff].bounding_box.max[0, 0] <= CENTRE + 1e-6
        whole = cut.table().loc[cutoff, "raw_volume"]
        assert cut[cutoff].volume == pytest.approx(whole / 2, rel=1e-3)
    realization = cut.simulations[4]
    assert realization[30.0].bounding_box.max[0, 0] <= CENTRE + 1e-6
    assert realization[30.0].volume == pytest.approx(_sphere(22.0) / 2,
                                                     rel=0.05)


def test_a_mesh_that_is_neither_sheet_nor_body_is_refused(radial):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    loose = geoml.data.Mesh3D(points, np.array([[0, 1, 2]]),
                              np.zeros((3, 3)))
    with pytest.raises(geoml.data.MeshTypeError, match="no side to keep"):
        MeshSet(radial, "g", limits={"odd": loose}, simulations=False)


# --------------------------------------------------------------------------- #
# realizations
# --------------------------------------------------------------------------- #
def test_every_realization_is_a_set_of_its_own(shells):
    ensemble = shells.simulations
    assert len(ensemble) == len(OFFSETS)
    assert ensemble.numbers == list(range(len(OFFSETS)))
    fourth = ensemble[4]
    assert isinstance(fourth, MeshSet) and fourth.realization == 4
    assert list(fourth) == list(shells)
    # realization 4 lifts the field by two, so each sphere grows by two
    for cutoff in fourth:
        assert fourth[cutoff].volume == pytest.approx(
            _sphere(52.0 - cutoff), rel=0.05)
    assert fourth[30.0].provenance["realization"] == 4
    assert fourth.simulations is None
    assert isinstance(ensemble[1:3], type(ensemble))
    assert [s.realization for s in ensemble[1:3]] == [1, 2]


def test_what_was_measured_is_what_the_stored_meshes_hold(shells):
    volumes = shells.realization_volumes()
    for number in (0, 2, 4):
        loaded = shells.simulations[number]
        for cutoff in shells:
            assert loaded[cutoff].volume == pytest.approx(
                volumes.loc[number, cutoff], rel=1e-9)


def test_workers_and_one_process_make_the_same_set(radial, shells):
    alone = MeshSet(radial, "g", workers=1)
    assert np.array_equal(alone.realization_volumes().to_numpy(),
                          shells.realization_volumes().to_numpy())


def test_some_realizations_can_be_asked_for_by_number(radial):
    some = MeshSet(radial, "g", simulations=[1, 3], workers=1)
    assert some.simulations.numbers == [1, 3]
    assert some.simulations[1].realization == 3
    first = MeshSet(radial, "g", simulations=2, workers=1)
    assert first.simulations.numbers == [0, 1]
    with pytest.raises(ValueError, match="not among"):
        MeshSet(radial, "g", simulations=[7])


def test_a_realization_that_fails_is_recorded_not_fatal(radial, monkeypatch):
    original = msm._shell

    def flaky(data, values, level, *args):
        # realization 4 is the only one reaching above 47
        if level == 30.0 and np.nanmax(values) > 47.0:
            raise RuntimeError("planted")
        return original(data, values, level, *args)

    monkeypatch.setattr(msm, "_shell", flaky)
    with pytest.warns(UserWarning, match="could not be made"):
        set_ = MeshSet(radial, "g", workers=1)
    assert set_.failures == [{"realization": 4, "key": 30.0,
                              "error": "RuntimeError: planted"}]
    assert np.isnan(set_.realization_volumes().loc[4, 30.0])
    assert np.isfinite(set_.realization_volumes().loc[3, 30.0])


def test_the_dispersion_places_the_prediction_among_its_realizations(shells):
    table = shells.volume_dispersion()
    # the offsets are symmetric about the prediction's, so it is the
    # median realization: two smaller, one the same, two larger
    assert np.allclose(table["rank"], 0.5)
    for cutoff in shells:
        volumes = [_sphere(50.0 + o - cutoff) for o in OFFSETS]
        assert table.loc[cutoff, "p50"] == pytest.approx(
            np.median(volumes), rel=0.05)
        # a sphere grown by two gains the shell between its radii and loses
        # nothing; one shrunk loses it; the mean over the five is both
        own = _sphere(50.0 - cutoff)
        gained = np.mean([max(v - own, 0.0) for v in volumes]) / own
        assert table.loc[cutoff, "gained"] == pytest.approx(gained, rel=0.1)
        lost = np.mean([max(own - v, 0.0) for v in volumes]) / own
        assert table.loc[cutoff, "lost"] == pytest.approx(lost, rel=0.1)


def test_a_set_without_realizations_has_no_dispersion(radial):
    alone = MeshSet(radial, "g", simulations=False)
    assert alone.simulations is None
    with pytest.raises(ValueError, match="no realizations"):
        alone.volume_dispersion()


# --------------------------------------------------------------------------- #
# categories
# --------------------------------------------------------------------------- #
def test_a_categorical_set_is_keyed_by_name():
    blocks = _slabs()
    bodies = MeshSet(blocks, "Rock", workers=2)
    assert list(bodies) == ["A", "B", "C"]
    assert bodies.kind == "category"
    with pytest.raises(KeyError, match="A, B, C"):
        bodies["D"]
    # the slabs meet at x = 20 and x = 50
    assert bodies["A"].bounding_box.max[0, 0] == pytest.approx(20.0)
    assert bodies["B"].bounding_box.min[0, 0] == pytest.approx(20.0)
    assert bodies["B"].bounding_box.max[0, 0] == pytest.approx(50.0)
    checked = bodies.check()
    assert np.allclose(checked[checked["kind"] == "overlap"]["volume"], 0.0,
                       atol=1e-6)
    # what is left is the model's own edges, rounded by the closing caps
    gap = checked[checked["kind"] == "gap"]["share"].iloc[0]
    assert 0.0 < gap < 0.06
    with pytest.raises(TypeError, match="no bands"):
        bodies.bands


def test_a_categorical_realization_moves_its_contacts():
    blocks = _slabs(shifts=(-4.0, 0.0, 4.0))
    bodies = MeshSet(blocks, "Rock", workers=1)
    # A's draw lifted by 4 against B's moves their contact 2 m into B
    lifted = bodies.simulations[2]
    assert lifted["A"].bounding_box.max[0, 0] == pytest.approx(22.0, abs=0.5)
    lowered = bodies.simulations[0]
    assert lowered["A"].bounding_box.max[0, 0] == pytest.approx(18.0, abs=0.5)
    dispersion = bodies.volume_dispersion()
    assert dispersion.loc["A", "p90"] > dispersion.loc["A", "prediction"] \
        > dispersion.loc["A", "p10"]


def test_the_priority_rule_lets_a_later_category_override():
    draws = np.array([[2.0, 1.0, -1.0],   # A and B positive: B wins
                      [3.0, -1.0, -2.0],  # only A positive: A
                      [0.5, 0.2, 0.1]])   # all positive: C, the last
    fields = msm._category_fields(draws, "priority")
    assert np.argmax(fields, axis=1).tolist() == [1, 0, 2]
    assert (fields > 0).sum(axis=1).tolist() == [1, 1, 1]
    largest = msm._category_fields(draws, "largest")
    assert np.argmax(largest, axis=1).tolist() == [0, 0, 0]
    # a tie is a contact: both categories read zero
    tied = msm._category_fields(np.array([[1.0, 1.0, -3.0]]), "largest")
    assert tied[0, :2].tolist() == [0.0, 0.0]


def test_a_categorical_set_refuses_what_it_has_no_use_for():
    blocks = _slabs()
    with pytest.raises(ValueError, match="no use for cutoffs"):
        MeshSet(blocks, "Rock", cutoffs=[0.0])
    with pytest.raises(ValueError, match="no\\s+other side"):
        MeshSet(blocks, "Rock", close="below")
    with pytest.raises(TypeError, match="one category"):
        MeshSet(blocks, "Rock/A")
    with pytest.raises(ValueError, match="rule"):
        MeshSet(blocks, "Rock", rule="loudest")


# --------------------------------------------------------------------------- #
# check and repair
# --------------------------------------------------------------------------- #
def test_a_crossing_is_measured_exactly_and_repaired(shells):
    sticking_out = _box([60.0, 60.0, 60.0], [80.0, 80.0, 80.0])
    crossed = shells._derived({20.0: shells[20.0], 30.0: sticking_out,
                               40.0: shells[40.0]})
    checked = crossed.check()
    outside = checked.set_index("first").loc[30.0, "volume"]
    # all of the box but the corner of it the outer shell reaches
    expected = sticking_out.volume \
        - sticking_out.intersection(shells[20.0]).volume
    assert outside == pytest.approx(expected, rel=1e-6)
    # the 40 shell sits wholly outside the box standing in at 30
    assert checked.set_index("first").loc[40.0, "volume"] == pytest.approx(
        shells[40.0].volume, rel=1e-6)
    fixed = crossed.repair()
    assert np.allclose(fixed.check()["volume"], 0.0, atol=1e-6)
    assert fixed.repairs.loc[30.0] == pytest.approx(outside, rel=1e-6)
    assert fixed.repairs.loc[20.0] == 0.0
    # the set repaired from is left as it was
    assert crossed.check().set_index("first").loc[30.0, "volume"] > 0


def test_categories_overlapping_are_resolved_in_priority_order():
    blocks = _slabs()
    bodies = MeshSet(blocks, "Rock", simulations=False)
    wide = _box([-5.0, -5.0, -5.0], [30.0, 75.0, 75.0])
    overlapping = bodies._derived({"A": wide, "B": bodies["B"],
                                   "C": bodies["C"]})
    shared = overlapping.check().set_index(["first", "second"]).loc[
        ("A", "B"), "volume"]
    # a ten-metre strip of B, less B's own rounded edges along it
    assert shared == pytest.approx(wide.intersection(bodies["B"]).volume,
                                   rel=1e-6)
    assert 0.9 * 10.0 * 80.0 * 80.0 < shared < 10.0 * 80.0 * 80.0
    kept_b = overlapping.repair(priority=["B", "A", "C"])
    assert kept_b.repairs.loc["A"] == pytest.approx(shared, rel=1e-6)
    assert kept_b.repairs.loc["B"] == 0.0
    kept_a = overlapping.repair()
    assert kept_a.repairs.loc["B"] == pytest.approx(shared, rel=1e-6)
    with pytest.raises(ValueError, match="every category once"):
        overlapping.repair(priority=["A", "B"])


def test_repair_at_construction_leaves_every_set_consistent(radial):
    repaired = MeshSet(radial, "g", repair=True, workers=1)
    assert repaired.provenance["repaired"] is True
    assert np.allclose(repaired.repairs.to_numpy(), 0.0)
    assert np.allclose(repaired.simulations[3].check()["volume"], 0.0)


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
def test_the_table_measures_the_bands_against_the_blocks(radial, shells):
    table = shells.table(grade="g", density=2.5)
    values = radial.variables["g"].prediction.values.to_numpy()
    volume = radial.block_volume
    for (low, high), cutoff in zip(shells.bands, shells):
        band = shells.bands[(low, high)]
        share = msm._sub_block_shares(radial, msm._within_body(band))
        share = np.nan_to_num(share)
        weight = volume * share * 2.5
        assert table.loc[cutoff, "tonnage"] == pytest.approx(weight.sum(),
                                                             rel=1e-9)
        assert table.loc[cutoff, "mean"] == pytest.approx(
            np.sum(weight * values) / weight.sum(), rel=1e-9)
        # a block's grade here sits between the band's two cut-offs
        assert low - 2.0 <= table.loc[cutoff, "mean"] <= min(high, 50.0)
        sims = radial.variables["g"].simulations[:, :]
        metals = (weight[:, None] * sims).sum(axis=0)
        assert table.loc[cutoff, "metal_p50"] == pytest.approx(
            np.percentile(metals, 50), rel=1e-9)
    # the blocks clearing each cut-off, against the raw contour
    assert table.loc[30.0, "blocks_volume"] == pytest.approx(
        volume[values >= 30.0].sum())
    assert table.loc[30.0, "blocks_volume"] == pytest.approx(
        table.loc[30.0, "raw_volume"], rel=0.1)


def test_each_realization_s_own_bands_hold_its_own_metal(radial, shells):
    table = shells.realization_table(grade="g", density=2.0,
                                     simulations=[0, 4])
    assert list(table.index.get_level_values(0).unique()) == [0, 4]
    volume = radial.block_volume
    sims = radial.variables["g"].simulations[:, :]
    for number in (0, 4):
        own = shells.simulations[number]
        for (low, high), cutoff in zip(own.bands, own):
            band = own.bands[(low, high)]
            row = table.loc[(number, cutoff)]
            assert row["volume"] == pytest.approx(band.volume)
            share = np.nan_to_num(msm._sub_block_shares(
                radial, msm._within_body(band)))
            weight = volume * share * 2.0
            assert row["tonnage"] == pytest.approx(weight.sum(), rel=1e-9)
            assert row["metal"] == pytest.approx(
                np.sum(weight * sims[:, number]), rel=1e-9)
    # a realization lifted by two holds more metal at the top
    assert table.loc[(4, 40.0), "metal"] > table.loc[(0, 40.0), "metal"]


def test_a_block_s_share_is_exact_from_its_sub_blocks(radial, shells):
    body = shells.bands[(20.0, 30.0)]
    quick = msm._block_shares(radial, body)
    every = np.nan_to_num(msm._sub_block_shares(radial,
                                                msm._within_body(body)))
    assert np.array_equal(quick, every)


def test_probability_shells_nest_by_level(radial):
    likely = MeshSet.probability(radial, "g", 30.0, levels=(0.1, 0.5, 0.9))
    assert list(likely) == [0.1, 0.5, 0.9]
    assert likely[0.1].volume > likely[0.5].volume > likely[0.9].volume
    # three realizations of five clear 30 inside a radius of 20
    assert likely[0.5].volume == pytest.approx(_sphere(20.0), rel=0.1)
    assert likely.provenance["probability_of"] == "g/prediction"
    assert np.allclose(likely.check()["volume"], 0.0)
    unsure = MeshSet.probability(radial, "g", 30.0, levels=[0.5],
                                 side="below")
    assert unsure[0.5].volume > likely[0.5].volume


def test_connectivity_counts_the_pods_and_drop_pieces_removes_them():
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2), max_levels=1)
    blocks = blocks.split(np.arange(blocks.n_data))
    points = np.asarray(blocks.coordinates)
    field = np.maximum(30.0 - np.linalg.norm(points - [20, 35, 35], axis=1),
                       12.0 - np.linalg.norm(points - [60, 35, 35], axis=1))
    blocks.add_continuous_variable("h", field)
    blocks.variables["h"].prediction.values[:] = field
    pods = MeshSet(blocks, "h", cutoffs=[5.0])
    table = pods.connectivity()
    assert table.loc[5.0, "pieces"] == 2
    small = _sphere(7.0)
    assert table.loc[5.0, "largest"] == pytest.approx(
        _sphere(25.0) / (_sphere(25.0) + small), rel=0.02)
    kept = pods.drop_pieces(4 * small)
    assert kept.connectivity().loc[5.0, "pieces"] == 1
    assert kept[5.0].volume == pytest.approx(pods[5.0].volume - small,
                                             rel=0.1)


def test_consecutive_shells_sit_as_far_apart_as_their_radii(shells):
    spacing = shells.spacing()
    assert spacing["inner"].tolist() == [30.0, 40.0]
    assert np.allclose(spacing["p50"], 10.0, atol=0.2)


def test_compare_measures_the_move_both_ways(shells):
    moved = shells.compare(shells.simulations[4])
    for cutoff in shells:
        assert moved.loc[cutoff, "moved_p50"] == pytest.approx(2.0, abs=0.2)
        assert moved.loc[cutoff, "gained"] == pytest.approx(
            _sphere(52.0 - cutoff) - _sphere(50.0 - cutoff), rel=0.1)
        assert moved.loc[cutoff, "lost"] == pytest.approx(0.0, abs=1.0)


def test_a_section_draws_each_shell_where_it_crosses(shells):
    lines = shells.section("Z", CENTRE)
    for cutoff, drawn in lines.items():
        assert len(drawn) == 1
        radius = np.linalg.norm(drawn[0][:, :2] - CENTRE, axis=1)
        assert np.median(radius) == pytest.approx(50.0 - cutoff, abs=0.3)
        assert np.allclose(drawn[0][:, 2], CENTRE)
    assert shells.section(2, 200.0) == {20.0: [], 30.0: [], 40.0: []}


def test_simplifying_keeps_the_shells_nested(shells):
    simple = shells.simplify(1.0)
    for cutoff in shells:
        assert len(simple[cutoff].triangles) < len(shells[cutoff].triangles)
        assert simple[cutoff].volume == pytest.approx(shells[cutoff].volume,
                                                      rel=0.05)
    assert np.allclose(simple.check()["volume"], 0.0, atol=1e-6)
    assert simple.provenance["simplify"] == 1.0
    assert simple.simulations is None


def test_clip_and_exclude_return_new_sets(shells):
    through_the_centre = _terrain(CENTRE)
    below = shells.clip(through_the_centre, "topography")
    assert list(below.limits) == ["topography"] and not shells.limits
    for cutoff in shells:
        assert below[cutoff].volume == pytest.approx(shells[cutoff].volume
                                                     / 2, rel=1e-3)
    both = below.exclude(_box([CENTRE, -10, -10], [90, 90, 90]), "east")
    table = both.table()
    assert list(table.columns[2:4]) == ["removed: topography",
                                        "removed: east"]
    assert both[20.0].volume == pytest.approx(shells[20.0].volume / 4,
                                              rel=1e-3)
    with pytest.raises(ValueError, match="already names"):
        both.clip(through_the_centre, "topography")


def test_assign_writes_the_band_every_location_falls_in(shells, radial):
    points = geoml.data.PointData.from_array(np.array(
        [[CENTRE, CENTRE, CENTRE + r] for r in (0.0, 15.0, 27.0, 36.0)]))
    shells.assign(points, "band")
    assert points.get_metadata("band").tolist() == ["≥ 40", "30–40",
                                                    "20–30", "< 20"]
    blocks = _radial(offsets=None)
    shells.assign(blocks, "band", fraction="share")
    shares = np.stack([blocks.get_metadata("share %s" % label)
                       for label in ("20–30", "30–40", "≥ 40")], axis=1)
    total = shares.sum(axis=1)
    radius = np.linalg.norm(np.asarray(blocks.coordinates) - CENTRE, axis=1)
    # the bands share out whatever of a block the outer shell holds: all of
    # a block well inside it, none of one well outside
    assert np.all((total >= 0.0) & (total <= 1.0 + 1e-12))
    assert np.allclose(total[radius < 25.0], 1.0)
    assert np.allclose(total[radius > 35.0], 0.0)


def test_crossed_by_marks_every_block_a_shell_passes_through(shells, radial):
    crossed = shells.crossed_by(radial)
    alone = np.zeros(radial.n_data, dtype=bool)
    for mesh in shells.values():
        alone |= radial.crossed_by(mesh)
    assert np.array_equal(crossed, alone) and crossed.any()


# --------------------------------------------------------------------------- #
# persistence and exports
# --------------------------------------------------------------------------- #
def test_a_set_reads_back_whole_from_its_store(shells, radial, tmp_path):
    path = str(tmp_path / "shells.zarr")
    shells.to_zarr(path)
    back = MeshSet.open(path, data=radial)
    assert list(back) == list(shells)
    for cutoff in shells:
        assert back[cutoff].volume == pytest.approx(shells[cutoff].volume,
                                                    rel=1e-12)
    assert back.simulations[4][30.0].volume == pytest.approx(
        shells.simulations[4][30.0].volume, rel=1e-12)
    assert back.volume_dispersion().equals(shells.volume_dispersion())
    assert back.table()["blocks_volume"].equals(
        shells.table()["blocks_volume"])
    # every mesh in the store is a container of its own
    alone = Solid3D.open(os.path.join(path, "simulations", "4", "30.0"))
    assert alone.provenance["realization"] == 4
    assert alone.provenance["source"] == "g/prediction"
    without = MeshSet.open(path)
    assert "blocks_volume" not in without.table()


def test_a_store_given_at_construction_is_the_set_s_own(radial, tmp_path):
    path = str(tmp_path / "given.zarr")
    made = MeshSet(radial, "g", workers=1, store=path)
    assert MeshSet.open(path).realization_volumes().equals(
        made.realization_volumes())
    del made
    gc.collect()
    assert os.path.isdir(path)


def test_the_temporary_store_goes_with_the_set(radial):
    made = MeshSet(radial, "g", workers=1)
    store = made._store
    view = made.simulations[2]
    del made
    gc.collect()
    # a realization's set keeps the store alive as long as it is held
    assert os.path.isdir(store) and view[30.0].n_data > 0
    del view
    gc.collect()
    assert not os.path.exists(store)


def test_limits_travel_with_the_store(radial, tmp_path):
    cut = MeshSet(radial, "g", limits={"topography": _terrain(CENTRE)},
                  simulations=False)
    path = str(tmp_path / "cut.zarr")
    cut.to_zarr(path)
    back = MeshSet.open(path)
    assert list(back.limits) == ["topography"]
    assert isinstance(back.limits["topography"], DTM3D)
    assert back.table()["removed: topography"].equals(
        cut.table()["removed: topography"])


def test_a_contour_records_where_it_came_from(radial, tmp_path):
    shell = radial.get_contour("g", 30.0, close="above", simplify=0.5)
    assert shell.provenance == {"source": "g/prediction", "value": 30.0,
                                "close": "above", "supersample": 0,
                                "simplify": 0.5}
    path = str(tmp_path / "shell.zarr")
    shell.to_zarr(path)
    assert Solid3D.open(path).provenance == shell.provenance


def test_one_dxf_holds_a_layer_per_cutoff(shells, tmp_path):
    path = str(tmp_path / "shells.dxf")
    shells.export_dxf(path)
    document = ezdxf.readfile(path)
    layers = {layer.dxf.name for layer in document.layers}
    assert {"g ge 20", "g ge 30", "g ge 40"} <= layers
    meshes = document.modelspace().query("MESH")
    assert len(meshes) == 3
    assert sorted(m.dxf.layer for m in meshes) == ["g ge 20", "g ge 30",
                                                   "g ge 40"]


def test_the_set_writes_into_a_geoh5_workspace(shells, tmp_path):
    pytest.importorskip("geoh5py")
    path = str(tmp_path / "shells.geoh5")
    shells.to_geoh5(path, folder="Shells", simulations=[4])
    listed = geoml.data.geoh5.contents(path)
    assert {"Shells/g >= 20", "Shells/g >= 30", "Shells/g >= 40",
            "Shells/simulations/4/g >= 30"} <= set(listed)


def test_a_scene_and_a_multiblock_name_every_mesh(shells):
    blocks = shells.as_pyvista()
    assert blocks.keys() == ["g >= 20", "g >= 30", "g >= 40"]
    class Recorder:
        # stands in for a pyvista Plotter, so nothing asks for OpenGL
        def __init__(self):
            self.added = []

        def add_mesh(self, mesh, **kwargs):
            self.added.append(kwargs["label"])

    recorder = shells.plot(Recorder())
    assert recorder.added == ["g >= 20", "g >= 30", "g >= 40"]


def test_a_grid_makes_a_set_too():
    grid = geoml.data.Grid3D([0, 0, 0], [20, 20, 20], [3.0, 3.0, 3.0])
    distance = np.linalg.norm(np.asarray(grid.coordinates) - 28.5, axis=1)
    grid.add_continuous_variable("g", 50.0 - distance)
    grid.variables["g"].prediction.values[:] = 50.0 - distance
    shells = MeshSet(grid, "g", cutoffs=[30.0, 40.0])
    assert shells[30.0].volume == pytest.approx(_sphere(20.0), rel=0.03)
    assert shells[40.0].volume < shells[30.0].volume


def test_anything_but_a_block_model_or_grid_is_refused():
    points = geoml.data.PointData.from_array(np.zeros((3, 3)))
    with pytest.raises(TypeError, match="block model or a three"):
        MeshSet(points, "g", cutoffs=[1.0])


# --------------------------------------------------------------------------- #
# the figures
# --------------------------------------------------------------------------- #
def test_the_figures_read_what_the_set_measured(shells, radial):
    panel = prep.volume_dispersion(shells)
    assert panel["labels"] == ["20", "30", "40"]
    assert [len(v) for v in panel["values"]] == [5, 5, 5]
    relative = prep.volume_dispersion(shells, relative=True)
    assert np.allclose(relative["prediction"], 1.0)
    connected = prep.connectivity(shells)
    assert np.allclose(connected["largest"], 1.0)
    assert connected["band"] is not None
    section = prep.mesh_section(shells, "Z", CENTRE,
                                variable=radial.variables["g"])
    assert set(section["lines"]) == {"20", "30", "40"}
    assert section["axes"] == ("X", "Y")
    image = section["image"]["values"]
    assert np.nanmax(image) == pytest.approx(50.0, abs=5.0)


@pytest.mark.parametrize("kind", ["box", "violin", "jitter"])
def test_both_backends_draw_the_volume_dispersion(shells, radial, kind):
    import matplotlib.pyplot as plt
    figure = geoml.plots.Explorer(radial).volume_dispersion(shells, kind=kind)
    assert figure.axes[0].get_ylim()[0] == 0.0
    plt.close(figure)
    drawn = geoml.plots.Interactive(radial).volume_dispersion(shells,
                                                              kind=kind)
    names = {trace.name for trace in drawn.data}
    assert names == {"realizations", "prediction"}


def test_both_backends_draw_connectivity_and_a_section(shells, radial):
    import matplotlib.pyplot as plt
    explorer = geoml.plots.Explorer(radial, continuous="g")
    for figure in (explorer.connectivity(shells),
                   explorer.section(shells, "Z", CENTRE)):
        plt.close(figure)
    interactive = geoml.plots.Interactive(radial, continuous="g")
    connected = interactive.connectivity(shells)
    assert len(connected.data) == 5
    section = interactive.section(shells, "Z", CENTRE)
    assert [trace.type for trace in section.data] == ["heatmap", "scatter",
                                                      "scatter", "scatter"]
