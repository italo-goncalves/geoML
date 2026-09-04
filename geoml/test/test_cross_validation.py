"""Cross-validation: the folds, the driver, and the calibration.

Design record and the E1 refit measurements: docs/cross-validation.md.

A random fold on spatial data is answered by its neighbours, so the folds
here are built to mimic the prediction task instead: held-out points should
sit as far from the training data as the prediction locations do
(`spatial_k_fold`), a whole drill hole stands or falls together, and the
scores are read on data the model never saw.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import spatial, stats

import geoml


def _clustered_points(seed=0, n_holes=12, per_hole=30, spread=2.0):
    """Tightly clustered samples inside a much wider box -- the drillhole
    geometry that makes random folds lie."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(10, 90, size=[n_holes, 2])
    xy = (centers[:, None, :]
          + rng.normal(scale=spread, size=[n_holes, per_hole, 2]))
    df = pd.DataFrame(xy.reshape(-1, 2), columns=["X", "Y"])
    points = geoml.data.PointData(df, ["X", "Y"])
    points.add_metadata("hole", np.repeat(np.arange(n_holes), per_hole))
    return points


def _grid():
    return geoml.data.Grid2D(start=[0, 0], n=[50, 50], step=[2, 2])


def _w_of(points, fold, target):
    """The criterion, recomputed independently of the method under test."""
    coords = np.asarray(points.coordinates)
    pooled = np.concatenate([
        spatial.cKDTree(coords[fold != f]).query(
            coords[fold == f], k=1)[0]
        for f in np.unique(fold)])
    return stats.wasserstein_distance(pooled, target)


# --------------------------------------------------------------------------- #
# the folds
# --------------------------------------------------------------------------- #
def test_the_chosen_folds_mimic_the_task_better_than_random_ones():
    points = _clustered_points()
    w, target, pooled = points.spatial_k_fold(_grid(), k=4, seed=0)

    rng = np.random.default_rng(1)
    random_w = [
        _w_of(points, rng.integers(0, 4, points.n_data), target)
        for _ in range(5)]

    assert w < min(random_w)
    assert np.isclose(w, _w_of(points, points.get_metadata("fold"), target))


def test_a_group_stands_or_falls_together():
    points = _clustered_points()
    points.spatial_k_fold(_grid(), k=4, groups="hole")

    fold = points.get_metadata("fold")
    hole = points.get_metadata("hole")
    for h in np.unique(hole):
        assert np.unique(fold[hole == h]).size == 1


def test_the_fold_column_is_written_with_every_fold_present():
    points = _clustered_points()
    points.spatial_k_fold(_grid(), k=4, groups="hole")

    fold = points.get_metadata("fold")
    assert set(np.unique(fold)) == {0, 1, 2, 3}


def test_the_fold_column_takes_the_name_it_is_given():
    points = _clustered_points()
    points.spatial_k_fold(_grid(), k=4, groups="hole", name="cv_scheme")

    assert "cv_scheme" in points.metadata
    assert "fold" not in points.metadata


def test_grouped_folds_are_deterministic():
    a = _clustered_points()
    b = _clustered_points()
    a.spatial_k_fold(_grid(), k=4, groups="hole")
    b.spatial_k_fold(_grid(), k=4, groups="hole")

    assert np.array_equal(a.get_metadata("fold"), b.get_metadata("fold"))


def test_too_few_groups_refuse():
    points = _clustered_points(n_holes=3)
    with pytest.raises(ValueError, match="3 groups"):
        points.spatial_k_fold(_grid(), k=5, groups="hole")


def test_the_match_is_reported_honestly():
    """The returned distances are what the criterion was computed on: the
    target from the prediction grid, the pooled ones from the folds."""
    points = _clustered_points()
    w, target, pooled = points.spatial_k_fold(_grid(), k=4, seed=0)

    assert target.size == _grid().n_data
    assert pooled.size == points.n_data
    assert w >= 0


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
def _walker_model():
    geoml.set_seed(1234)
    walker, grid = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[10, 10], step=[28, 30])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        walker, "V", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False))
    return model, walker, grid


@pytest.fixture(scope="module")
def walker_cv():
    """A small trained Walker model, spatial folds, and one cross-validation
    run shared by the assertions below."""
    model, walker, grid = _walker_model()
    model.train_full(max_iter=100)
    walker.spatial_k_fold(grid, k=3, seed=0)
    oof, scores = geoml.models.cross_validate(
        model, iterations=30, n_sim=8, n_nodes=8)
    return model, walker, oof, scores


def test_every_location_gets_an_out_of_fold_prediction(walker_cv):
    """The folds partition the data, so after the loop every row holds an
    answer from the one model that never saw it."""
    _, walker, oof, _ = walker_cv
    prediction = np.asarray(oof.variables["V"].prediction.values)
    simulations = np.asarray(oof.variables["V"].simulations)
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(simulations))
    assert oof.n_data == walker.n_data


def test_the_score_table_covers_every_fold_and_pools_them(walker_cv):
    _, walker, _, scores = walker_cv
    assert set(scores["fold"]) == {0, 1, 2, "all"}
    pooled = scores[scores["fold"] == "all"]
    assert pooled["n"].item() == walker.n_data
    per_fold = scores[scores["fold"] != "all"]
    assert per_fold["n"].sum() == walker.n_data
    assert np.all(np.isfinite(per_fold[
        ["rmse", "mae", "bias", "crps", "goodness"]].to_numpy(dtype=float)))


def test_held_out_scores_are_worse_than_in_sample_ones(walker_cv):
    """The point of the whole exercise: a model scored on its own training
    data flatters itself, and the out-of-fold score should show it."""
    model, walker, _, scores = walker_cv
    samples = model.predict_measurements(walker, n_sim=8, n_nodes=8)["V"]
    y_true, has_value = walker.variables["V"].get_measurements()
    measured = np.asarray(has_value)[:, 0] == 1
    in_sample = geoml.metrics.rmse(
        np.asarray(y_true)[measured, 0], samples[measured, 0, :].mean(axis=1))

    oof_rmse = scores[scores["fold"] == "all"]["rmse"].item()
    assert oof_rmse > in_sample


def test_the_original_model_is_left_untouched(walker_cv):
    """The fold models are copies: nothing here may fix or overwrite the
    parameters of the model handed in."""
    model, _, _, _ = walker_cv
    assert not model.likelihoods[0].parameters["noise"].fixed
    assert not model.latent_network.parameters["alpha_white_0"].fixed


def test_the_fresh_state_is_ignorant_and_everything_else_is_frozen():
    model, _, _ = _walker_model()
    trained_noise = np.asarray(
        model.likelihoods[0].parameters["noise"].get_value())

    geoml.models._fresh_variational_state(model)

    gp = model.latent_network
    assert np.abs(np.asarray(
        gp.parameters["alpha_white_0"].get_value())).max() < 0.1
    assert np.allclose(np.asarray(gp.parameters["delta_0"].get_value()), 1.0)
    assert np.allclose(np.asarray(gp.parameters["bias_0"].get_value()), 0.0)

    unfixed = model.get_unfixed_variables()
    assert len(unfixed) == 3  # alpha, delta, bias -- nothing else trains
    assert model.likelihoods[0].parameters["noise"].fixed
    assert np.allclose(np.asarray(
        model.likelihoods[0].parameters["noise"].get_value()), trained_noise)


def test_bad_arguments_are_refused_before_any_work(walker_cv):
    model, _, _, _ = walker_cv
    with pytest.raises(ValueError, match="no metadata column"):
        geoml.models.cross_validate(model, folds="nope")
    with pytest.raises(ValueError, match="refit"):
        geoml.models.cross_validate(model, refit="bogus")
    with pytest.raises(ValueError, match="method"):
        geoml.models.cross_validate(model, method="minibatch")


# --------------------------------------------------------------------------- #
# the measurement samples are held whole, so their size is guarded
# --------------------------------------------------------------------------- #
def test_measurement_samples_cost_what_the_shapes_say(walker_cv):
    """The guard's arithmetic, against the array it is guarding: one value
    per realization per noise node per column, in float64."""
    model, walker, _, _ = walker_cv
    samples = model.predict_measurements(walker, n_sim=4, n_nodes=8)["V"]

    assert samples.shape == (walker.n_data, 1, 4 * 8)
    assert samples.nbytes == walker.n_data * 1 * 4 * 8 * 8


def test_a_request_too_large_to_hold_is_refused_before_any_work(walker_cv):
    """It returns the whole answer as one array, so a request for tens of
    gigabytes has to be refused rather than attempted -- the OOM killer
    ends the session instead of raising."""
    model, walker, _, _ = walker_cv

    # 470 rows is small; the node counts are what make it enormous
    huge = int(np.ceil(
        geoml.models.MEASUREMENT_LIMIT / (walker.n_data * 8))) + 1
    with pytest.raises(MemoryError, match="past the"):
        model.predict_measurements(walker, n_sim=huge, n_nodes=1)

    # and the message names the knobs that get you under it
    with pytest.raises(MemoryError, match="n_sim or n_nodes"):
        model.predict_measurements(walker, n_sim=1, n_nodes=huge)


def test_a_request_that_fits_is_left_alone(walker_cv):
    model, walker, _, _ = walker_cv
    just_under = int(
        geoml.models.MEASUREMENT_LIMIT / (walker.n_data * 8) * 0.5)
    assert just_under > 1
    samples = model.predict_measurements(walker, n_sim=2, n_nodes=2)
    assert samples["V"].shape == (walker.n_data, 1, 4)


# --------------------------------------------------------------------------- #
# refitting each fold in minibatches
# --------------------------------------------------------------------------- #
def test_the_folds_can_be_refitted_by_svi(walker_cv, monkeypatch):
    """`method="svi"` refits each fold in batches of the size the model's
    own options carry, and answers the same questions the full-batch run
    does: every row predicted out of fold, and the same score table."""
    model, walker, full_oof, full_scores = walker_cv
    # the fixture's model is shared, so the batch size is put back after
    monkeypatch.setattr(model.options, "training_batch_size", 200)

    oof, scores = geoml.models.cross_validate(
        model, method="svi", epochs=3, n_sim=8, n_nodes=8)

    assert np.all(np.isfinite(np.asarray(oof.values("V/prediction"))))
    assert list(scores.columns) == list(full_scores.columns)
    assert set(scores["fold"]) == set(full_scores["fold"])
    assert scores[scores["fold"] == "all"]["n"].item() == walker.n_data
    # a different route to the same kind of answer, not the same numbers
    assert np.all(np.isfinite(scores[
        ["rmse", "mae", "bias", "crps", "goodness"]].to_numpy(dtype=float)))


def test_the_epochs_are_what_svi_counts(walker_cv, monkeypatch):
    """`iterations` counts gradient steps and `epochs` counts passes over
    the data, so the two arguments are separate and only one is read."""
    model, walker, _, _ = walker_cv
    monkeypatch.setattr(model.options, "training_batch_size", 200)

    seen = []
    original = geoml.models.VGPNetwork.train_svi

    def spy(self, epochs=100):
        seen.append(epochs)
        return original(self, epochs=epochs)

    monkeypatch.setattr(geoml.models.VGPNetwork, "train_svi", spy)
    geoml.models.cross_validate(
        model, method="svi", epochs=2, iterations=999, n_sim=4, n_nodes=4)

    # one refit per fold, each given the epochs and not the iterations
    assert seen == [2, 2, 2]


# --------------------------------------------------------------------------- #
# the calibration
# --------------------------------------------------------------------------- #
def _pit_of(truth, samples):
    return (samples < truth[:, None]).mean(axis=1) \
        + 0.5 * (samples == truth[:, None]).mean(axis=1)


def test_conformal_repairs_a_planted_overconfidence():
    """Intervals narrower than they should be, told to cut wider -- and the
    repaired coverage lands on the nominal on data the calibration never
    saw. The plant is bounded by what samples can express: an interval cut
    from an ensemble can never reach past the ensemble's own range, so a far
    harsher overconfidence saturates at `nominal == 1` instead of repairing.
    """
    rng = np.random.default_rng(7)
    truth = rng.normal(size=2000)
    narrow = rng.normal(size=(2000, 1000)) * 0.7

    calibration = geoml.models.ConformalCalibration(
        _pit_of(truth[:1000], narrow[:1000]))
    assert calibration.nominal(0.9) > 0.95

    lower, upper = calibration.interval(narrow[1000:], coverage=0.9)
    covered = np.mean((truth[1000:] >= lower) & (truth[1000:] <= upper))
    assert covered == pytest.approx(0.9, abs=0.03)


def test_conformal_narrows_a_planted_hedge():
    rng = np.random.default_rng(8)
    truth = rng.normal(size=2000)
    wide = rng.normal(size=(2000, 400)) * 3.0

    calibration = geoml.models.ConformalCalibration(
        _pit_of(truth[:1000], wide[:1000]))
    assert calibration.nominal(0.9) < 0.6

    lower, upper = calibration.interval(wide[1000:], coverage=0.9)
    covered = np.mean((truth[1000:] >= lower) & (truth[1000:] <= upper))
    assert covered == pytest.approx(0.9, abs=0.03)


def test_a_calibrated_forecast_is_left_almost_alone():
    rng = np.random.default_rng(9)
    truth = rng.normal(size=3000)
    honest = rng.normal(size=(3000, 400))

    calibration = geoml.models.ConformalCalibration(_pit_of(truth, honest))
    assert calibration.nominal(0.9) == pytest.approx(0.9, abs=0.05)
    assert calibration.nominal(0.5) == pytest.approx(0.5, abs=0.05)


def test_the_map_is_monotone_and_saturates():
    rng = np.random.default_rng(10)
    truth = rng.normal(size=500)
    samples = rng.normal(size=(500, 200))
    calibration = geoml.models.ConformalCalibration(_pit_of(truth, samples))

    levels = [calibration.nominal(q) for q in (0.3, 0.6, 0.9, 0.999)]
    assert all(a <= b for a, b in zip(levels, levels[1:]))
    assert calibration.nominal(0.9999) == 1.0


def test_cross_validate_leaves_the_pits_behind(walker_cv):
    """The OOF container carries the column `conformalize` reads, filled at
    every measured location."""
    _, _, oof, _ = walker_cv
    pit = oof.get_metadata("pit_V")
    assert np.all(np.isfinite(pit))
    assert np.all((pit >= 0) & (pit <= 1))

    calibration = geoml.models.conformalize(oof, "V")
    assert 0 < calibration.nominal(0.9) <= 1.0


def test_conformalize_names_the_missing_column(walker_cv):
    _, _, oof, _ = walker_cv
    with pytest.raises(ValueError, match="no metadata column"):
        geoml.models.conformalize(oof, "V", component="nope")
