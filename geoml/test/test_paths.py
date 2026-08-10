"""Addressing a piece of data inside a container by path.

A container holds variables, a variable holds components or attributes, and an
attribute holds one array per location. `VariablePath` names a place in that
tree, `walk`/`leaves` enumerate it once, and everything that reads a container
whole is a fold over that one traversal rather than a list of columns written
out per class. Design and reasoning in `docs/variable-paths.md`.
"""
import numpy as np
import pytest

import geoml
from geoml.data import VariablePath


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
