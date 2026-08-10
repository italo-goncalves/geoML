"""Addressing a piece of data inside a container by path.

A container holds variables, a variable holds components or attributes, and an
attribute holds one array per location. `VariablePath` names a place in that
tree, `walk`/`leaves` enumerate it once, and everything that reads a container
whole is a fold over that one traversal rather than a list of columns written
out per class. Design and reasoning in `docs/variable-paths.md`.
"""
import warnings

import numpy as np
import pytest

import geoml
from geoml.data import VariablePath, render, render_all


def _points(n=6):
    coords = np.stack([np.arange(float(n))] * 3, axis=1)
    point = geoml.data.PointData.from_array(coords)
    point.add_continuous_variable("au", np.arange(float(n)))
    point.add_vector_variable("vec", ["p", "q"],
                              np.arange(2.0 * n).reshape(n, 2))
    point.add_compositional_variable(
        "assay", ["a", "b"], np.tile([0.6, 0.4], (n, 1)))
    point.add_categorical_variable(
        "rock", measurements=np.array(["ore", "waste"] * (n // 2)))
    point.add_metadata("fold", np.arange(n) % 3)
    return point


# --------------------------------------------------------------------------- #
# the path itself
# --------------------------------------------------------------------------- #
def test_a_path_is_built_from_a_string_parts_or_another_path():
    assert VariablePath("assay/Zn/prediction").parts == \
        ("assay", "Zn", "prediction")
    assert VariablePath(["assay", "Zn"]) == VariablePath("assay/Zn")
    assert VariablePath(VariablePath("a/b")) == VariablePath("a/b")
    assert str(VariablePath("a/b")) == "a/b"


def test_the_separator_composes_paths():
    assert VariablePath("assay") / "Zn" / "prediction" == \
        VariablePath("assay/Zn/prediction")
    assert VariablePath("assay") / ("Zn", "prediction") == \
        VariablePath("assay/Zn/prediction")


def test_empty_segments_are_dropped_so_the_root_is_empty():
    assert len(VariablePath("")) == 0
    assert len(VariablePath("/")) == 0
    assert VariablePath("/assay//Zn/") == VariablePath("assay/Zn")


def test_a_name_may_not_contain_the_separator():
    with pytest.raises(ValueError, match="separator"):
        VariablePath(["Qtz/Fsp", "prediction"])


def test_the_name_and_the_parent():
    path = VariablePath("assay/Zn/prediction")
    assert path.name == "prediction"
    assert path.parent == VariablePath("assay/Zn")
    assert VariablePath("").name == ""


def test_a_path_is_a_dictionary_key():
    seen = {VariablePath("a/b"): 1}
    assert seen[VariablePath("a/b")] == 1
    assert VariablePath("a/b") == "a/b"


# --------------------------------------------------------------------------- #
# walking the tree
# --------------------------------------------------------------------------- #
def test_the_container_is_the_root_of_the_walk():
    point = _points()
    paths = [str(p) for p, _ in point.walk()]

    assert paths[0] == ""
    assert "assay" in paths and "assay/a" in paths and "rock/ore" in paths


def test_a_variable_walked_alone_roots_at_its_own_name():
    point = _points()
    paths = [str(p) for p, _ in point.variables["assay"].walk()]

    assert paths == ["assay", "assay/a", "assay/b"]


def test_every_column_appears_once_with_its_full_path():
    point = _points()
    paths = [str(p) for p, _ in point.leaves()]

    assert len(paths) == len(set(paths))
    assert "au/measurements" in paths
    assert "assay/a/prediction" in paths
    assert "rock/ore/probability" in paths
    assert "_metadata/fold" in paths


def test_a_component_has_no_latent_space_and_so_no_such_leaf():
    point = _points()
    paths = [str(p) for p, _ in point.leaves()]

    assert "au/latent_mean" in paths
    assert "assay/a/latent_mean" not in paths      # None, skipped
    assert "vec/p/latent_mean" in paths            # a plain ContinuousVariable


def test_a_cutoff_family_is_addressed_by_its_key():
    point = _points()
    grade = point.variables["au"]
    grade.set_cutoffs([1.5])
    grade.proportions[1.5] = geoml.data._Attribute(point, np.ones(6))

    paths = [str(p) for p, _ in point.leaves()]
    assert "au/proportions/1.5" in paths


# --------------------------------------------------------------------------- #
# reaching a leaf
# --------------------------------------------------------------------------- #
def test_a_column_is_reached_by_path():
    point = _points()
    assert point.get("au/measurements") is point.variables["au"].measurements
    assert point.get("assay/a/prediction") is \
        point.variables["assay"].components["a"].prediction
    assert np.allclose(point.values("au/measurements"), np.arange(6.0))


def test_a_node_is_reached_by_path_and_the_root_is_the_container():
    point = _points()
    assert point.get("assay") is point.variables["assay"]
    assert point.get("assay/a") is point.variables["assay"].components["a"]
    assert point.get("") is point
    assert point.get("assay").get("a/prediction") is \
        point.variables["assay"].components["a"].prediction


def test_metadata_lives_under_a_root_of_its_own():
    point = _points()
    assert point.get("_metadata/fold") is point.metadata["fold"]
    assert np.array_equal(point.values("_metadata/fold"),
                          np.arange(6) % 3)


def test_a_coded_column_comes_back_decoded():
    point = _points()
    assert point.values("rock/predicted").dtype == object


def test_a_cutoff_key_is_read_however_it_is_written():
    point = _points()
    grade = point.variables["au"]
    grade.quantiles[1.5] = geoml.data._Attribute(point, np.ones(6))

    wanted = grade.quantiles[1.5]
    assert point.get("au/quantiles/1.5") is wanted
    assert point.get("au/quantiles/1.50") is wanted     # same number
    assert point.get("au/quantiles", 1.5) is wanted     # the two-argument form


def test_a_missing_path_says_what_is_there():
    """Reported from as deep as the path reached: told that `au` holds no
    `nope`, the reader knows where to look. A list of the container's
    variables would only say the first segment was fine."""
    point = _points()
    with pytest.raises(KeyError, match="nothing at 'au/nope'"):
        point.get("au/nope")
    with pytest.raises(KeyError, match="'au' holds .*measurements"):
        point.get("au/nope")
    with pytest.raises(KeyError, match="'a' holds .*prediction"):
        point.get("assay/a/nope")
    with pytest.raises(KeyError, match="the container holds"):
        point.get("nope")


def test_subsetting_never_holds_every_realization_at_once(monkeypatch):
    """`np.asarray(store)[mask]` reads every realization of every location
    before throwing most of them away; on a block model that is the end of
    the session."""
    import geoml.storage as storage
    point = _points()
    grade = point.variables["au"]
    grade.allocate_simulations(4)
    grade.simulations[:, :] = np.arange(24.0).reshape(6, 4)

    def explode(self, *args, **kwargs):
        if len(self.shape) == 2:
            raise AssertionError("the whole simulations store was read")
        return np.asarray(self._array)

    monkeypatch.setattr(storage.ArrayStore, "__array__", explode)
    kept = point[np.array([True, False, True, False, True, False])]

    assert np.allclose(np.asarray(kept.variables["au"].simulations._array),
                       np.arange(24.0).reshape(6, 4)[::2])


def test_simulations_are_a_leaf_of_their_own_shape():
    point = _points()
    grade = point.variables["au"]
    grade.allocate_simulations(4)
    grade.simulations[:, :] = np.arange(24.0).reshape(6, 4)

    assert point.get("au/simulations").shape == (6, 4)
    assert np.allclose(point.values("au/simulations/2"),
                       np.arange(24.0).reshape(6, 4)[:, 2])
    # not swept up by the column walk: it carries a realization axis
    assert "au/simulations" not in [str(p) for p, _ in point.leaves()]


# --------------------------------------------------------------------------- #
# the node's own facts
# --------------------------------------------------------------------------- #
def test_a_node_reports_the_facts_a_rebuild_has_to_carry():
    point = _points()
    point.variables["au"].set_cutoffs([1.0, 2.0])

    assert point.variables["au"].node_attrs() == {"cutoffs": [1.0, 2.0]}
    assert point.variables["assay"].components["a"].node_attrs() == \
        {"cutoffs": None}


# --------------------------------------------------------------------------- #
# selecting
# --------------------------------------------------------------------------- #
def test_one_star_stops_at_a_segment_boundary():
    """One segment down, whatever is there — a component or a column of the
    variable's own, since both are addressed the same way."""
    point = _points()
    chosen = [str(p) for p in point.select("assay/*")]
    assert sorted(chosen) == ["assay/a", "assay/b", "assay/uncertainty"]

    assert [str(p) for p in point.select("assay/*/prediction")] == \
        ["assay/a/prediction", "assay/b/prediction"]


def test_two_stars_cross_any_number_of_segments():
    point = _points()
    chosen = [str(p) for p in point.select("**/prediction")]

    assert "au/prediction" in chosen
    assert "assay/a/prediction" in chosen
    assert "vec/p/prediction" in chosen
    assert all(p.endswith("/prediction") for p in chosen)


def test_two_stars_match_nothing_as_well_as_something():
    point = _points()
    chosen = [str(p) for p in point.select("assay/**/prediction")]
    assert "assay/a/prediction" in chosen        # one segment between
    assert "au/prediction" not in chosen         # a different variable


def test_a_subtree_comes_whole():
    point = _points()
    chosen = [str(p) for p in point.select("assay/**")]

    assert "assay" in chosen                      # the node itself
    assert "assay/a" in chosen                    # its component
    assert "assay/a/prediction" in chosen         # and the columns
    assert not any(p.startswith("au") for p in chosen)


def test_filled_keeps_columns_only_and_says_which():
    point = _points()
    filled = point.select("au/**", filled=True)
    empty = point.select("au/**", filled=False)

    assert [str(p) for p in filled] == ["au/measurements"]
    assert "au/prediction" in [str(p) for p in empty]
    # a node is neither filled nor empty, so it is left out of both
    assert "au" not in [str(p) for p in filled] + [str(p) for p in empty]


def test_a_bare_double_star_leaves_the_realizations_alone():
    """The store is one name; unrolling it into a hundred is something you
    have to ask for, or a default export of a simulated model emits a hundred
    arrays per variable."""
    point = _points()
    point.variables["au"].allocate_simulations(3)

    everything = [str(p) for p in point.select("**")]
    assert "au/simulations" in everything
    assert "au/simulations/0" not in everything

    unrolled = [str(p) for p in point.select("**/simulations/*")]
    assert unrolled == ["au/simulations/0", "au/simulations/1",
                        "au/simulations/2"]


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_the_three_styles_are_the_segments_joined():
    path = "assay/Zn/noise_variance"

    assert render(path, "path") == "assay/Zn/noise_variance"
    assert render(path, "flat") == "assay_Zn_noise_variance"
    assert render(path, "pretty") == "assay - Zn - noise_variance"


def test_an_unknown_style_says_what_there_is():
    with pytest.raises(ValueError, match="'path', 'flat' or 'pretty'"):
        render("a/b", "fancy")


def test_a_namespace_with_no_collision_renders_straight_through():
    names = render_all(["au/prediction", "assay/a/prediction"], "flat")

    assert list(names.values()) == ["au_prediction", "assay_a_prediction"]


def test_a_flat_collision_is_resolved_by_path_order_and_warned_about():
    clash = ["noise_variance/prediction", "noise/variance/prediction"]

    with pytest.warns(UserWarning, match="render to the column name"):
        names = render_all(clash, "flat")

    # sorted by path, so the answer does not depend on the order asked for
    assert names[VariablePath("noise/variance/prediction")] == \
        "noise_variance_prediction"
    assert names[VariablePath("noise_variance/prediction")] == \
        "noise_variance_prediction_2"
    with pytest.warns(UserWarning):
        # same answer whichever order they were handed over in
        assert dict(render_all(clash[::-1], "flat")) == dict(names)


def test_the_path_style_cannot_collide():
    names = render_all(
        ["noise_variance/prediction", "noise/variance/prediction"], "path")
    assert len(set(names.values())) == 2


# --------------------------------------------------------------------------- #
# printing the tree
# --------------------------------------------------------------------------- #
def test_the_tree_shows_the_shape_of_the_container():
    point = _points()
    text = point.tree()

    assert text.startswith("PointData - 6 locations")
    assert "|-- au" in text
    assert "|   |-- measurements" in text          # a column of that variable
    assert "|   |-- a " in text or "|   `-- a " in text   # a component
    assert "_metadata/fold" in text               # one line, not a subtree


def test_the_tree_says_which_columns_hold_something():
    """The question worth asking of a prediction: a column allocated and never
    written is the shape of most of the bugs this addressing was built to
    stop."""
    point = _points()
    empty_before = point.tree().count("empty")

    assert "|-- measurements           float64" in point.tree()
    assert "|-- prediction             empty" in point.tree()

    point.variables["au"].prediction.values[:] = np.arange(6.0)
    assert "|-- prediction             float64" in point.tree()
    assert point.tree().count("empty") == empty_before - 1


def test_the_structure_can_be_printed_without_reading_the_columns(monkeypatch):
    """`status=True` reads each column once, which on a disk-backed block
    model is a pass over the lot; `status=False` is the structure alone."""
    import geoml.storage as storage
    point = _points()

    def explode(self, *args, **kwargs):
        raise AssertionError("the backing array was read")

    monkeypatch.setattr(storage.ArrayStore, "__array__", explode)
    text = point.tree(status=False)
    assert "au" in text and "empty" not in text


def test_a_node_prints_its_own_facts():
    point = _points()
    point.variables["au"].set_cutoffs([1.5])

    assert "cutoffs=[1.5]" in point.tree()


def test_a_variable_prints_its_own_tree():
    point = _points()
    text = point.variables["assay"].tree()

    assert text.startswith("CompositionalVariable 'assay'")
    assert "|-- a" in text or "`-- a" in text


def test_a_containers_own_namespace_renders_without_a_warning():
    point = _points()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        names = render_all([p for p, _ in point.leaves()], "flat")
    assert len(set(names.values())) == len(names)
