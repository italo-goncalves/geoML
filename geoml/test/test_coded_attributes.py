"""The categorical variables keep their text as codes.

`RockTypeVariable`, `CategoricalVariable`, `OrderedRockType`, `BinaryVariable`
and `AnomalyVariable` used to hold one Python string per data location in
`predicted` and in their measurement columns. They now hold integer codes plus
a label list, which is the only form `ArrayStore` can spill to disk. These
tests pin the two things that could go wrong quietly: what a code means, and
where the categories come from.
"""
import numpy as np
import pandas as pd
import pytest

import geoml


N = 6


def _point():
    df = pd.DataFrame({"c0": np.arange(N, dtype=float),
                       "c1": np.arange(N, dtype=float) * 2})
    return geoml.data.PointData(df, ["c0", "c1"])


def _rock():
    point = _point()
    point.add_categorical_variable(
        "rock", labels=["basalt", "granite"],
        measurements=np.array(["granite", "basalt"] * 3))
    return point.variables["rock"]


def test_measurements_are_stored_as_codes():
    rock = _rock()
    stored = np.asarray(rock.measurements_a.values)

    assert stored.dtype == np.int8
    assert rock.measurements_a.labels == ["basalt", "granite"]
    assert np.array_equal(stored, [1, 0] * 3)
    assert np.array_equal(rock.measurements_a.to_numpy(),
                          ["granite", "basalt"] * 3)


def test_an_unmeasured_variable_is_missing_everywhere():
    point = _point()
    point.add_categorical_variable("rock", labels=["basalt", "granite"])
    rock = point.variables["rock"]

    assert np.all(np.asarray(rock.measurements_a.values) == -1)
    # missing decodes to the empty string, which is what it used to be stored as
    assert np.all(rock.measurements_a.to_numpy() == "")
    assert np.all(rock.predicted.to_numpy() == "")


def test_a_missing_column_is_not_exported():
    """`_has_content` has to read -1 as nothing, not as a category."""
    point = _point()
    point.add_categorical_variable("rock", labels=["basalt", "granite"])
    rock = point.variables["rock"]

    class _Sink:
        point_data = {}

    sink = _Sink()
    rock.predicted.fill_pyvista_points(sink, "predicted")
    assert sink.point_data == {}

    rock.predicted.values[0] = 1
    rock.predicted.fill_pyvista_points(sink, "predicted")
    assert sink.point_data["predicted"][0] == "granite"


def test_a_prediction_is_written_as_the_winning_position():
    rock = _rock()
    rock.allocate_simulations(1)
    import tensorflow as tf

    probability = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4],
                               [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]],
                              dtype=tf.float64)
    rock.update(np.arange(N),
                entropy=tf.zeros([N], dtype=tf.float64),
                uncertainty=tf.zeros([N], dtype=tf.float64),
                probability=probability,
                mean=tf.zeros([N, 2], dtype=tf.float64),
                variance=tf.ones([N, 2], dtype=tf.float64),
                indicators=tf.zeros([N, 2], dtype=tf.float64),
                simulations=tf.zeros([N, 2, 1], dtype=tf.float64))

    assert np.array_equal(np.asarray(rock.predicted.values),
                          [0, 1, 0, 1, 0, 1])
    assert np.array_equal(
        rock.predicted.to_numpy(),
        ["basalt", "granite", "basalt", "granite", "basalt", "granite"])


def test_measurements_keep_a_value_the_variable_has_no_label_for():
    """An `AnomalyVariable` labels the other class `_dummy`; the measurement
    still says what it actually was."""
    point = _point()
    point.add_anomaly_variable(
        "anom", label="hit", measurements=np.array(["hit", "none"] * 3))
    var = point.variables["anom"]

    assert list(var.labels) == ["hit", "_dummy"]
    assert var.measurements.labels == ["hit", "none"]
    assert np.array_equal(var.measurements.to_numpy(), ["hit", "none"] * 3)
    # the indicator still reads the anomaly off the raw measurements
    assert np.array_equal(np.asarray(var.indicator.values),
                          [1.0, np.nan] * 3, equal_nan=True)


def test_a_measurement_outside_the_labels_survives_the_round_trip(tmp_path):
    point = _point()
    point.add_anomaly_variable(
        "anom", label="hit", measurements=np.array(["hit", "none"] * 3))

    path = str(tmp_path / "anom.zarr")
    point.to_zarr(path)
    reopened = geoml.data.PointData.open(path)

    var = reopened.variables["anom"]
    assert var.measurements.labels == ["hit", "none"]
    assert np.array_equal(var.measurements.to_numpy(), ["hit", "none"] * 3)


def test_an_older_store_holding_strings_is_re_encoded(tmp_path):
    """Attributes were written as fixed-length text before 0.5.3."""
    import zarr

    point = _point()
    point.add_categorical_variable(
        "rock", labels=["basalt", "granite"],
        measurements=np.array(["granite", "basalt"] * 3))
    path = str(tmp_path / "rock.zarr")
    point.to_zarr(path)

    # rewrite `predicted` the way the previous version stored it
    group = zarr.open_group(path, mode="r+")
    meta = dict(group.attrs["geoml"])
    info = meta["variables"]["rock"]["attrs"]["predicted"]
    text = np.asarray(["granite", "basalt"] * 3).astype("<U7")
    del group["rock/predicted"]
    array = group.create_array(name="rock/predicted", shape=text.shape,
                               chunks=text.shape, dtype=text.dtype)
    array[:] = text
    info["encoding"] = "str"
    info.pop("labels", None)
    group.attrs["geoml"] = meta

    reopened = geoml.data.PointData.open(path)
    rock = reopened.variables["rock"]
    assert np.asarray(rock.predicted.values).dtype == np.int8
    assert np.array_equal(rock.predicted.to_numpy(), ["granite", "basalt"] * 3)


def test_slicing_drops_the_categories_that_are_gone():
    """Slicing is pre-processing: a category with no data left is dropped, and
    the components and the prediction codes follow it."""
    rock = _rock()
    rock.predicted.values[:] = np.asarray(rock.measurements_a.values)
    sliced = rock[np.array([0, 2, 4])]                # granite only

    assert np.array_equal(sliced.measurements_a.to_numpy(), ["granite"] * 3)
    assert list(sliced.labels) == ["granite"]
    assert sliced.length == 1
    assert list(sliced.components.keys()) == ["granite"]
    # the code that meant granite was 1 in the parent and is 0 here
    assert np.array_equal(np.asarray(sliced.predicted.values), [0, 0, 0])
    assert np.array_equal(sliced.predicted.to_numpy(), ["granite"] * 3)
    # a measurement column keeps its own dictionary, which is harmless
    assert sliced.measurements_a.labels == ["basalt", "granite"]


def test_slicing_an_unmeasured_variable_keeps_its_categories():
    """A prediction target has no measurements to drop categories by."""
    point = _point()
    point.add_categorical_variable("rock", labels=["basalt", "granite"])
    sliced = point.variables["rock"][np.array([0, 2, 4])]

    assert list(sliced.labels) == ["basalt", "granite"]
    assert list(sliced.components.keys()) == ["basalt", "granite"]
    assert np.all(np.asarray(sliced.predicted.values) == -1)


def test_slicing_an_ordered_variable_rescales_its_implicit_values():
    """They are positions in the label sequence, so dropping a category shifts
    every one after it: a slice must give what building on those rows gives."""
    point = _point()
    a = np.array(["low", "low", "mid", "mid", "high", "high"])
    b = np.array(["low", "mid", "mid", "high", "high", "high"])
    point.add_rock_type_variable("seq", labels=["low", "mid", "high"],
                                 measurements_a=a, measurements_b=b,
                                 ordered=True)
    keep = np.array([2, 3, 4, 5])                     # no "low" left
    sliced = point.variables["seq"][keep]

    rebuilt = _point()[keep]
    rebuilt.add_rock_type_variable("seq", labels=["mid", "high"],
                                   measurements_a=a[keep],
                                   measurements_b=b[keep], ordered=True)

    assert list(sliced.labels) == ["mid", "high"]
    assert np.array_equal(
        np.asarray(sliced.implicit_values.values),
        np.asarray(rebuilt.variables["seq"].implicit_values.values))


def test_slicing_an_unmeasured_ordered_variable_keeps_it_missing():
    point = _point()
    point.add_rock_type_variable("seq", labels=["low", "mid", "high"],
                                 ordered=True)
    sliced = point.variables["seq"][np.array([0, 2, 4])]

    assert np.all(np.isnan(np.asarray(sliced.implicit_values.values)))


def test_categories_that_are_not_text_are_left_alone():
    """Stringifying the derived labels made the measurements disagree with the
    variable's own labels, and every metric came out at chance."""
    point = _point()
    point.add_categorical_variable("rock", measurements=np.array([1, 2] * 3))
    rock = point.variables["rock"]

    assert list(rock.measurements_a.labels) == [1, 2]
    assert list(rock.measurements_a.to_numpy()) == [1, 2] * 3

    rock.predicted.values[:] = np.asarray(rock.measurements_a.values)
    metrics = rock.compute_metrics()
    assert list(metrics.loc["Balanced accuracy"]) == [1.0, 1.0]


def test_a_variable_with_integer_categories_can_be_saved(tmp_path):
    """The component names and dict keys are JSON, and a category can be a
    NumPy integer — which used to raise on the way out."""
    point = _point()
    point.add_categorical_variable("rock", measurements=np.array([1, 2] * 3))

    path = str(tmp_path / "rock.zarr")
    point.to_zarr(path)
    rock = geoml.data.PointData.open(path).variables["rock"]

    # categories come back as their names, as the variable's labels always have
    assert list(rock.labels) == ["1", "2"]
    assert list(rock.measurements_a.labels) == ["1", "2"]
    assert list(rock.measurements_a.to_numpy()) == ["1", "2"] * 3


def test_the_exported_frame_reads_as_text():
    rock = _rock()
    df = rock.as_data_frame()

    assert df["rock_a"].tolist() == ["granite", "basalt"] * 3
    assert df["rock_predicted"].tolist() == [""] * N
