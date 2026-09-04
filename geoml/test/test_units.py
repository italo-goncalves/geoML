"""Units on variables, and compositions in the units they were assayed in.

A unit is a fact of the variable: it rides the tree, the Zarr store and
every rebuild off `_NODE_ATTRS`, and on a variable the model reads directly
it changes no number. On a part of a composition it is also the divisor, and
the parts are stored, predicted and simulated in their own units --
percent, ppm, g/t -- becoming fractions of the whole only at the two doors
where the model reads and writes them, since parts in different units cannot
be added up.

The gate is `test_a_declared_unit_changes_nothing_but_the_scale`: the same
data as percentages and as fractions must train to the same bound and report
a factor of a hundred apart, the variances by its square.
"""
import numpy as np
import pytest

import geoml
import geoml.warping as wp
from geoml.data.variables import UNITS, _divisor

LABELS = ["a", "b", "rest"]


def _points(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return geoml.data.PointData.from_array(
        rng.uniform(0.0, 100.0, (n, 2)), ["x", "y"])


def _composition(units=None, rest=True, values=None, n=12):
    """A two-part composition plus a rest, in whatever units are given."""
    point = _points(n)
    if values is None:
        values = np.tile([[1.0, 10000.0]], (n, 1))
    point.add_compositional_variable(
        "metals", ["a", "b"], values, units=units, rest=rest)
    return point


# --------------------------------------------------------------------------- #
# the fact
# --------------------------------------------------------------------------- #
def test_a_continuous_variable_carries_its_unit():
    point = _points()
    point.add_continuous_variable("au", np.arange(12.0), unit="g/t")

    assert point.get("au").unit == "g/t"
    assert point.get("au").node_attrs()["unit"] == "g/t"
    assert point.get("au").divisor() == 1e6


def test_a_unit_is_a_label_on_a_variable_the_model_reads():
    """Nothing is divided: the values reach the likelihood as they stand."""
    values = np.linspace(1.0, 5.0, 12)
    plain, labelled = _points(), _points()
    plain.add_continuous_variable("au", values)
    labelled.add_continuous_variable("au", values, unit="g/t")

    np.testing.assert_array_equal(plain.get("au").get_measurements()[0],
                                  labelled.get("au").get_measurements()[0])
    assert labelled.get("au").prediction_input() == {}
    labelled.get("au").set_cutoffs([2.0])
    assert labelled.get("au").prediction_input() == {"cutoffs": [2.0]}


def test_a_vector_variable_takes_one_unit_per_component():
    point = _points()
    point.add_vector_variable(
        "assay", ["pb", "zn"], np.ones((12, 2)), units={"pb": "%", "zn": "ppm"})

    assert [point.get("assay").components[c].unit for c in ("pb", "zn")] \
        == ["%", "ppm"]
    # a label the model reads directly: nothing is converted
    np.testing.assert_array_equal(point.get("assay").get_measurements()[0],
                                  np.ones((12, 2)))


def test_units_may_be_given_in_the_labels_own_order():
    point = _points()
    point.add_vector_variable("assay", ["pb", "zn"], np.ones((12, 2)),
                              units=["%", "ppm"])
    assert [point.get("assay").components[c].unit for c in ("pb", "zn")] \
        == ["%", "ppm"]


def test_a_unit_for_a_label_that_is_not_there_is_refused():
    point = _points()
    with pytest.raises(ValueError, match="not among the labels"):
        point.add_vector_variable("assay", ["pb"], np.ones((12, 1)),
                                  units={"cu": "%"})


def test_the_wrong_number_of_units_is_refused():
    point = _points()
    with pytest.raises(ValueError, match="unit"):
        point.add_vector_variable("assay", ["pb", "zn"], np.ones((12, 2)),
                                  units=["%"])


def test_a_composition_insists_on_a_unit_it_can_divide_by():
    """A part's unit is the divisor, so it has to be one the package knows.
    A plain variable's is a label and takes anything."""
    point = _points()
    with pytest.raises(ValueError, match="unknown unit"):
        point.add_compositional_variable(
            "metals", ["a", "b"], np.ones((12, 2)), units={"a": "carats"})

    point.add_continuous_variable("note", np.arange(12.0), unit="carats")
    assert point.get("note").unit == "carats"


def test_the_tree_shows_what_a_variable_is_measured_in():
    point = _points()
    point.add_continuous_variable("au", np.arange(12.0), unit="g/t")
    assert "unit=g/t" in point.tree()


def test_the_container_lists_its_units_by_path():
    point = _composition(units={"a": "%", "b": "ppm"})
    point.add_continuous_variable("au", np.arange(12.0), unit="g/t")

    assert point.units() == {"au": "g/t", "metals/a": "%", "metals/b": "ppm"}


def test_a_unit_survives_zarr(tmp_path):
    point = _composition(units={"a": "%", "b": "ppm"})
    point.add_continuous_variable("au", np.arange(12.0), unit="g/t")
    point.to_zarr(tmp_path / "c.zarr")

    back = geoml.data.PointData.open(tmp_path / "c.zarr")
    assert back.units() == point.units()


def test_a_store_written_before_units_existed_loads_undeclared(tmp_path):
    """`unit` is a node fact, and a fact absent from the store keeps the
    constructor's default -- so an old container opens without one."""
    point = _points()
    point.add_continuous_variable("au", np.arange(12.0), unit="g/t")
    point.to_zarr(tmp_path / "c.zarr")

    import zarr
    store = zarr.open_group(str(tmp_path / "c.zarr"), mode="a")
    meta = dict(store.attrs["geoml"])
    del meta["variables"]["au"]["node_attrs"]["unit"]
    store.attrs["geoml"] = meta

    assert geoml.data.PointData.open(tmp_path / "c.zarr").get("au").unit is None


@pytest.mark.parametrize("method", ["copy_to", "carry_to", "subset"])
def test_a_unit_survives_a_rebuild(method):
    point = _composition(units={"a": "%", "b": "ppm"})
    point.add_continuous_variable("au", np.arange(12.0), unit="g/t")

    if method == "subset":
        new = point[np.arange(5)]
    else:
        new = geoml.data.PointData.from_array(np.zeros((12, 2)), ["x", "y"])
        for variable in point.variables.values():
            if method == "copy_to":
                variable.copy_to(new)
            else:
                new.variables[variable.name] = variable.carry_to(
                    new, np.ones(12, dtype=bool), 0)

    assert new.units() == point.units()


def test_a_derived_variable_can_be_told_its_unit():
    point = _points()
    point.add_continuous_variable("au", np.arange(1.0, 13.0), unit="g/t")
    point.get("au").allocate_simulations(3)
    point.get("au").simulations[:, :] = np.ones((12, 3))

    point.derive("value", lambda grade: grade * 2.0, ["au"], units="USD")
    assert point.get("value").unit == "USD"


# --------------------------------------------------------------------------- #
# the composition: stored in its own units, converted at the doors
# --------------------------------------------------------------------------- #
def test_the_parts_keep_the_units_they_were_given_in():
    metals = _composition(units={"a": "%", "b": "ppm"}).get("metals")

    assert metals.labels == LABELS
    np.testing.assert_allclose(
        np.asarray(metals.components["a"].measurements.values)[:3], 1.0)
    np.testing.assert_allclose(
        np.asarray(metals.components["b"].measurements.values)[:3], 10000.0)
    np.testing.assert_allclose(
        np.asarray(metals.components["rest"].measurements.values)[:3], 0.98)


def test_the_model_is_given_one_whole():
    metals = _composition(units={"a": "%", "b": "ppm"}).get("metals")
    values, has_value = metals.get_measurements()

    np.testing.assert_allclose(values[0], [0.01, 0.01, 0.98])
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
    assert np.all(has_value == 1.0)
    np.testing.assert_allclose(metals.divisors(), [100.0, 1e6, 1.0])


def test_a_cut_off_is_declared_in_the_parts_own_unit():
    metals = _composition(units={"a": "%", "b": "ppm"}).get("metals")
    metals.components["a"].set_cutoffs([2.0])          # two percent

    sent = metals.prediction_input()["cutoffs"]
    assert sent[0][0] == pytest.approx(0.02)           # a fraction, to the model
    assert np.isinf(sent[1][0])                        # b declared none


def test_model_units_come_back_in_the_parts_own_unit():
    metals = _composition(units={"a": "%", "b": "ppm"}).get("metals")
    fractions = np.tile([[0.01, 0.02, 0.97]], (12, 1))

    np.testing.assert_allclose(metals.from_model_units(fractions)[0],
                               [1.0, 20000.0, 0.97])


def test_a_variable_the_model_reads_directly_returns_what_it_was_given():
    point = _points()
    point.add_continuous_variable("au", np.arange(12.0), unit="g/t")
    values = np.arange(12.0).reshape(12, 1)
    assert point.get("au").from_model_units(values) is values


# --------------------------------------------------------------------------- #
# what the closure does, and does not do
# --------------------------------------------------------------------------- #
def test_data_that_arrives_closed_is_left_exactly_alone():
    """The pre-units spelling: fractions summing to one, no units declared.
    Nothing may touch them, down to the last bit."""
    point = _points()
    fractions = np.array([[0.25, 0.25, 0.5]] * 12)
    point.add_compositional_variable("c", ["p", "q", "r"], fractions)

    stored = np.stack([np.asarray(point.get("c").components[k].measurements.values)
                       for k in ("p", "q", "r")], axis=1)
    assert np.array_equal(stored, fractions)


def test_closing_without_a_rest_keeps_each_part_in_its_own_unit():
    point = _points()
    # percentages that do not quite add up, as an assay's rarely do
    values = np.tile([[30.0, 30.0, 60.0]], (12, 1))
    point.add_compositional_variable(
        "c", ["p", "q", "r"], values,
        units={"p": "%", "q": "%", "r": "%"})

    stored = np.stack([np.asarray(point.get("c").components[k].measurements.values)
                       for k in ("p", "q", "r")], axis=1)
    np.testing.assert_allclose(stored[0], [25.0, 25.0, 50.0])
    np.testing.assert_allclose(point.get("c").get_measurements()[0].sum(axis=1),
                               1.0)


def test_zeros_are_replaced_in_the_columns_own_unit():
    point = _points(n=3)
    values = np.array([[0.0, 4.0], [4.0, 0.0], [6.0, 3.0]])
    point.add_compositional_variable(
        "c", ["p", "q"], values, units={"p": "%", "q": "%"}, rest=True)

    stored = np.stack([np.asarray(point.get("c").components[k].measurements.values)
                       for k in ("p", "q")], axis=1)
    np.testing.assert_allclose(stored[0, 0], 2.0)     # half of 4 %, in percent
    np.testing.assert_allclose(stored[1, 1], 1.5)     # half of 3 %


def test_the_measurements_must_match_the_labels():
    point = _points()
    with pytest.raises(ValueError, match="part"):
        point.add_compositional_variable(
            "c", ["p", "q", "r"], np.ones((12, 2)), rest=True)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def _trained(values, units, cutoff, iterations=15):
    """A composition trained on the same numbers, told a different unit."""
    geoml.set_seed(1234)
    data = geoml.data.PointData.from_array(
        np.linspace(0.0, 100.0, 30)[:, None], ["depth"])
    data.add_compositional_variable("c", ["p", "q", "r"], values, units=units)
    data.get("c/p").set_cutoffs([cutoff])

    chain = wp.ChainedWarping(wp.CenteredLogRatio(3), wp.ZScore(3))
    gp = geoml.latent.BasicGP(
        geoml.latent.BasicInput(
            geoml.data.Grid1D(0.0, n=8, step=14.0),
            transform=geoml.transform.Isotropic(20.0)),
        size=3)
    model = geoml.models.VGPNetwork(
        data, "c", geoml.likelihood.MultivariateGaussian(3, chain), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=iterations)

    target = geoml.data.Grid1D(0.0, n=12, step=8.0)
    model.predict(target, n_sim=6)
    samples = model.predict_measurements(target, n_sim=6, n_nodes=4)["c"]
    variable = target.get("c")
    variable.reset_quantiles([0.5])
    parts = ["p", "q", "r"]
    return {
        "log": np.asarray(model.training_log),
        "prediction": np.stack([np.asarray(variable.components[k].prediction.values)
                                for k in parts], axis=1),
        "simulations": np.stack([np.asarray(variable.components[k].simulations)
                                 for k in parts], axis=1),
        "dispersion": np.stack([np.asarray(variable.components[k].dispersion.values)
                                for k in parts], axis=1),
        "noise": np.stack([np.asarray(variable.components[k].noise_variance.values)
                           for k in parts], axis=1),
        "median": np.stack([np.asarray(variable.components[k].quantiles[0.5].values)
                            for k in parts], axis=1),
        "shares": np.asarray(
            variable.components["p"].proportions[cutoff].values),
        "measurements": samples,
    }


@pytest.fixture(scope="module")
def gate():
    rng = np.random.default_rng(3)
    as_percent = rng.dirichlet([6.0, 3.0, 11.0], size=30) * 100.0
    # the same numbers on both sides: the percent table divided by its unit
    # is what the undeclared one is handed, so the model sees one array
    percent = _trained(as_percent, {"p": "%", "q": "%", "r": "%"}, 30.0)
    plain = _trained(as_percent / 100.0, None, 0.30)
    return percent, plain


def test_a_declared_unit_changes_nothing_but_the_scale(gate):
    """The gate. The same composition told it is in percent trains to the
    same bound as one told nothing, and everything it reports comes back a
    hundred times larger -- ten thousand for a variance."""
    percent, plain = gate

    np.testing.assert_array_equal(percent["log"], plain["log"])
    for key, factor in (("prediction", 100.0), ("simulations", 100.0),
                        ("median", 100.0), ("measurements", 100.0),
                        ("dispersion", 100.0 ** 2), ("noise", 100.0 ** 2)):
        np.testing.assert_allclose(percent[key], plain[key] * factor,
                                   rtol=1e-10, atol=0.0,
                                   err_msg="%s did not scale by %g"
                                           % (key, factor))


def test_a_cut_off_names_the_same_ground_in_either_unit(gate):
    """30 % and 0.30 are one cut-off, so the share of each block above it
    is the same number."""
    percent, plain = gate
    np.testing.assert_allclose(percent["shares"], plain["shares"])


# --------------------------------------------------------------------------- #
# the unit table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("unit,divisor", [
    ("%", 100.0), ("pct", 100.0), ("ppm", 1e6), ("g/t", 1e6), ("ppb", 1e9),
    ("fraction", 1.0), (None, 1.0), (250.0, 250.0), (" PPM ", 1e6)])
def test_the_divisor_of_a_unit(unit, divisor):
    assert _divisor(unit) == divisor


def test_a_divisor_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        _divisor(-1.0)


def test_the_table_is_shared_with_the_drillhole_module():
    import geoml.data.drillhole as dh
    assert dh.UNITS is UNITS
