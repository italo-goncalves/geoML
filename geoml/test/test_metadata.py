"""Point-wise metadata on the data containers.

Metadata is information known per location that the models never see — an
air/solid code, a cross-validation fold, a sample weight. It is stored the way
a variable's attributes are (one `ArrayStore` per column, spilling to Zarr when
large), so these tests cover the column semantics, what happens on subsetting
and export, and the Zarr round-trip.
"""
import numpy as np
import pandas as pd
import pytest

import geoml


def _points(n=12):
    coords = np.stack([np.arange(n, dtype=float),
                       np.arange(n, dtype=float) * 2,
                       np.zeros(n)], axis=1)
    df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    return geoml.data.PointData(df, ["X", "Y", "Z"])


def test_a_container_starts_with_no_metadata():
    point = _points()
    assert point.metadata == {}
    assert geoml.data.Grid2D(start=[0, 0], n=[4, 3], step=[1, 1]).metadata == {}


def test_metadata_columns_are_the_attributes_variables_are_built_from():
    """`_Attribute` is a module-level class; `_Variable` keeps an alias."""
    point = _points()
    point.add_metadata("fold", np.arange(12))
    point.add_continuous_variable("v", np.arange(12, dtype=float))

    assert geoml.data._Variable._Attribute is geoml.data._Attribute
    assert isinstance(point.metadata["fold"], geoml.data._Attribute)
    assert type(point.metadata["fold"]) \
        is type(point.variables["v"].measurements)
    # a plain attribute is not coded
    assert point.variables["v"].measurements.labels is None


def test_a_numeric_column_keeps_its_values_and_dtype():
    point = _points()
    folds = np.arange(12) % 5
    point.add_metadata("fold", folds)

    assert np.array_equal(point.get_metadata("fold"), folds)
    assert point.metadata["fold"].values.dtype == folds.dtype
    assert point.metadata["fold"].labels is None


def test_a_boolean_column_stays_boolean():
    point = _points()
    point.add_metadata("solid", np.arange(12) > 5)
    assert point.metadata["solid"].values.dtype == np.dtype(bool)
    assert point.get_metadata("solid").sum() == 6


def test_a_text_column_is_stored_as_codes_and_read_back_as_labels():
    point = _points()
    rock = np.array(["ore", "waste"] * 6)
    point.add_metadata("rock", rock)

    column = point.metadata["rock"]
    assert column.labels == ["ore", "waste"]
    # the codes are the compact part: an object column can neither spill to
    # disk nor shrink
    assert column.values.dtype == np.dtype(np.int8)
    assert np.array_equal(np.asarray(column.values), [0, 1] * 6)
    assert np.array_equal(point.get_metadata("rock"), rock)


def test_codes_can_be_given_directly_with_their_labels():
    point = _points()
    point.add_metadata("rock", np.zeros(12, dtype=int), labels=["ore", "waste"])
    assert np.all(point.get_metadata("rock") == "ore")


def test_a_column_of_the_wrong_length_is_refused():
    point = _points()
    with pytest.raises(ValueError, match="size mismatch"):
        point.add_metadata("fold", np.arange(5))


def test_asking_for_a_column_that_is_not_there_lists_the_ones_that_are():
    point = _points()
    point.add_metadata("fold", np.zeros(12))
    with pytest.raises(ValueError, match="no metadata column named weight"):
        point.get_metadata("weight")


def test_adding_a_column_twice_replaces_it():
    point = _points()
    point.add_metadata("fold", np.zeros(12))
    point.add_metadata("fold", np.ones(12))
    assert list(point.metadata.keys()) == ["fold"]
    assert np.all(point.get_metadata("fold") == 1)


def test_metadata_leads_the_exported_data_frame():
    point = _points()
    point.add_metadata("fold", np.arange(12) % 3)
    point.add_metadata("rock", np.array(["ore", "waste"] * 6))
    point.add_continuous_variable("v", np.arange(12, dtype=float))

    df = point.as_data_frame()
    assert list(df.columns[:2]) == ["fold", "rock"]
    assert df["rock"].tolist() == ["ore", "waste"] * 6

    assert "fold" not in point.as_data_frame(metadata=False).columns


def test_an_empty_metadata_set_does_not_disturb_the_data_frame():
    point = _points()
    point.add_continuous_variable("v", np.arange(12, dtype=float))
    df = point.as_data_frame()
    assert list(df.columns[:3]) == ["X", "Y", "Z"]
    assert len(df) == 12


def test_subsetting_points_carries_the_metadata():
    point = _points()
    point.add_metadata("fold", np.arange(12))
    point.add_metadata("rock", np.array(["ore", "waste"] * 6))

    subset = point[np.array([0, 3, 7])]
    assert np.array_equal(subset.get_metadata("fold"), [0, 3, 7])
    assert subset.get_metadata("rock").tolist() == ["ore", "waste", "waste"]
    assert subset.metadata["fold"].coordinates is subset


def test_subsetting_a_region_carries_the_metadata():
    point = _points()
    point.add_metadata("fold", np.arange(12))

    # the upper bound is excluded by default, so this keeps X = 0 to 3
    subset = point.subset_region([0, 0, -1], [4, 100, 1])
    assert np.array_equal(subset.get_metadata("fold"), np.arange(4))


def test_subsetting_a_grid_region_carries_the_metadata():
    grid = geoml.data.Grid2D(start=[0, 0], n=[4, 3], step=[1, 1])
    grid.add_metadata("fold", np.arange(grid.n_data))

    subset = grid.subset_region([0, 0], [2, 3])
    assert subset.n_data == len(subset.get_metadata("fold"))
    assert np.array_equal(subset.get_metadata("fold"),
                          np.arange(grid.n_data)[
                              np.asarray(grid.coordinates)[:, 0] < 2])


def test_a_grid_column_reshapes_like_a_variable_attribute():
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[4, 3, 2], step=[1, 1, 1])
    grid.add_metadata("above_ground",
                      np.asarray(grid.coordinates)[:, 2] > 0.5)

    cube = grid.metadata["above_ground"].as_cube()
    assert cube.shape == (4, 3, 2)
    assert not cube[:, :, 0].any()
    assert cube[:, :, 1].all()


def test_metadata_survives_the_zarr_round_trip(tmp_path):
    point = _points()
    point.add_metadata("fold", np.arange(12) % 3)
    point.add_metadata("rock", np.array(["ore", "waste"] * 6))
    point.add_continuous_variable("v", np.arange(12, dtype=float))

    path = str(tmp_path / "points.zarr")
    point.to_zarr(path)
    reopened = geoml.data.PointData.open(path)

    assert list(reopened.metadata.keys()) == ["fold", "rock"]
    assert np.array_equal(reopened.get_metadata("fold"), np.arange(12) % 3)
    assert reopened.get_metadata("rock").tolist() == ["ore", "waste"] * 6
    # left on disk, as the variables are
    assert reopened.metadata["fold"].values.backend == "zarr"


def test_grid_metadata_survives_the_zarr_round_trip(tmp_path):
    grid = geoml.data.Grid2D(start=[0, 0], n=[4, 3], step=[1, 1])
    grid.add_metadata("fold", np.arange(grid.n_data))

    path = str(tmp_path / "grid.zarr")
    grid.to_zarr(path)
    reopened = geoml.data.Grid2D.open(path)

    assert np.array_equal(reopened.get_metadata("fold"),
                          np.arange(grid.n_data))


def test_a_store_without_metadata_still_opens(tmp_path):
    """Stores written before metadata was persisted have no such section."""
    import zarr

    point = _points()
    path = str(tmp_path / "points.zarr")
    point.to_zarr(path)

    group = zarr.open_group(path, mode="r+")
    meta = dict(group.attrs["geoml"])
    del meta["metadata"]
    group.attrs["geoml"] = meta

    assert geoml.data.PointData.open(path).metadata == {}


def test_a_large_column_spills_to_disk(monkeypatch):
    monkeypatch.setattr(geoml.storage, "DEFAULT_THRESHOLD", 8)
    point = _points()
    point.add_metadata("fold", np.arange(12))
    assert point.metadata["fold"].values.backend == "zarr"
