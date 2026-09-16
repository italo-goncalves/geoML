"""What a long call reports, and what a cancelled one leaves behind.

`geoml.progress(callback)` is one context manager over every long call, and
the callback raising is the cancel. Two things have to hold for that to be
worth anything: the count must be of units *finished*, so that what a cancel
leaves matches what was last reported, and the work that finished must stay
finished. The gate is the resume -- predicting what `unpredicted()` names
after a cancel gives, to the last bit, what predicting the lot would have.
"""
import numpy as np
import pytest

import geoml


class Stop(Exception):
    """A cancel raised by a callback. Any exception does; this one names it."""


def _model(batch=50):
    geoml.set_seed(1234)
    point, _ = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[5, 5], step=[55, 62])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    return geoml.models.VGPNetwork(
        point, "V", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False,
                                       prediction_batch_size=batch))


def _grid(n=10):
    return geoml.data.Grid2D(start=[10, 10], n=[n, n], step=[20, 20])


@pytest.fixture(scope="module")
def trained():
    model = _model()
    model.train_full(max_iter=3)
    return model


# --------------------------------------------------------------------------- #
# what each site reports
# --------------------------------------------------------------------------- #
def test_training_reports_every_iteration_with_the_bound():
    model = _model()
    seen = []
    with geoml.progress(seen.append):
        model.train_full(max_iter=4)

    assert [e.done for e in seen] == [1, 2, 3, 4]
    assert all(e.task == "train" and e.unit == "iteration" and e.total == 4
               and e.within == () for e in seen)
    # the bound reported is the one the log kept, and the report comes after
    # the log, so a cancel leaves the model where the caller was told it was
    assert [e.bound for e in seen] == \
        [float(v) for v in model.training_log[-4:]]


def test_minibatch_training_reports_every_batch():
    """An epoch is many gradient steps, and a caller watching a long run
    wants to hear from it oftener than once a pass."""
    model = _model()
    seen = []
    with geoml.progress(seen.append):
        model.train_svi(epochs=2)

    assert all(e.task == "train" and e.unit == "batch" for e in seen)
    assert [e.done for e in seen] == list(range(1, len(seen) + 1))
    assert seen[-1].done == seen[-1].total


def test_prediction_reports_batches_that_are_written(trained):
    grid = _grid()
    seen = []
    with geoml.progress(seen.append):
        trained.predict(grid, n_sim=3)

    assert all(e.task == "predict" and e.unit == "batch" for e in seen)
    # from nothing done to everything done: the consumer writes each batch
    # before asking for the next, so `done` is what a cancel would leave
    assert [e.done for e in seen] == list(range(seen[-1].total + 1))
    assert seen[-1].total == 2      # 100 locations in batches of 50


def test_a_refinement_names_itself_over_the_predictions_it_makes():
    """A block model is three-dimensional, so this one needs a model of its
    own rather than Walker's two."""
    geoml.set_seed(7)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 80.0, (60, 3))
    point = geoml.data.PointData.from_array(coords)
    point.add_continuous_variable("V", coords[:, 2] / 40.0 - 1.0)
    point.variables["V"].set_cutoffs([0.0])
    root = geoml.latent.BasicInput(
        geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3], step=[40, 40, 40]),
        transform=geoml.transform.Isotropic(40))
    model = geoml.models.VGPNetwork(
        point, "V", geoml.likelihood.Gaussian(),
        geoml.latent.BasicGP(root, size=1),
        options=geoml.models.GPOptions(verbose=False,
                                       prediction_batch_size=200))
    model.train_full(max_iter=3)

    blocks = geoml.data.BlockSet3D([0, 0, 0], [4, 4, 4], [20.0, 20.0, 20.0],
                                   discretization=(2, 2, 2), max_levels=2)
    seen = []
    with geoml.progress(seen.append):
        geoml.models.refine(model, blocks, n_sim=4)

    passes = [e for e in seen if e.task == "refine"]
    predictions = [e for e in seen if e.task == "predict"]
    assert passes and predictions
    assert all(e.unit == "pass" and e.within == () for e in passes)
    # which is how a caller tells a refinement's predictions from a bare one
    assert all(e.within == ("refine",) for e in predictions)
    assert [e.done for e in passes] == list(range(len(passes)))
    # a pass is not bounded by `max_levels`: a block still at level 0 can be
    # marked by any later pass, as the field sharpens around its neighbours
    assert all(e.total is None for e in passes)
    assert len(passes) - 1 > blocks.max_levels


def test_cross_validation_reports_folds_and_names_itself():
    model = _model()
    model.train_full(max_iter=2)
    # the folds are read off the model's own data, not a fresh copy of it
    model.data.add_metadata("fold", np.arange(model.data.n_data) % 3)
    seen = []
    with geoml.progress(seen.append):
        geoml.models.cross_validate(model, folds="fold", iterations=2,
                                    n_sim=3)

    folds = [e for e in seen if e.task == "cross_validate"]
    assert [e.done for e in folds] == [0, 1, 2, 3]
    assert all(e.unit == "fold" and e.total == 3 for e in folds)
    assert {e.within for e in seen if e.task != "cross_validate"} \
        == {("cross_validate",)}


# --------------------------------------------------------------------------- #
# listening changes nothing
# --------------------------------------------------------------------------- #
def test_reporting_changes_no_number(trained):
    watched, plain = _grid(), _grid()
    with geoml.progress(lambda event: None):
        trained.predict(watched, n_sim=3)
    trained.predict(plain, n_sim=3)

    for path in ("V/prediction", "V/latent_variance"):
        assert np.array_equal(np.asarray(watched.get(path).values),
                              np.asarray(plain.get(path).values))


def test_nothing_is_reported_outside_the_block(trained):
    seen = []
    with geoml.progress(seen.append):
        pass
    trained.predict(_grid(), n_sim=3)
    assert seen == []


def test_an_inner_block_can_silence_an_outer_one(trained):
    seen = []
    with geoml.progress(seen.append):
        with geoml.progress(None):
            trained.predict(_grid(), n_sim=3)
        assert seen == []
        trained.predict(_grid(), n_sim=3)
    assert seen


# --------------------------------------------------------------------------- #
# a cancel, and the resume it makes possible
# --------------------------------------------------------------------------- #
def test_a_cancelled_prediction_keeps_the_batches_that_finished(trained):
    grid = _grid()

    def stop_after_one(event):
        if event.done >= 1:
            raise Stop()

    with pytest.raises(Stop):
        with geoml.progress(stop_after_one):
            trained.predict(grid, n_sim=3)

    written = ~np.isnan(np.asarray(grid.get("V/prediction").values))
    assert written.sum() == 50            # one batch of the two
    assert np.array_equal(grid.unpredicted(), ~written)


def test_resuming_a_cancelled_prediction_gives_the_whole_answer(trained):
    """The gate. A location's values do not depend on what else is in its
    batch, so finishing what `unpredicted()` names must give what predicting
    the lot gives -- otherwise a cancel is a corruption, not a pause."""
    whole = _grid()
    trained.predict(whole, n_sim=3)

    resumed = _grid()

    def stop_after_one(event):
        if event.done >= 1:
            raise Stop()

    with pytest.raises(Stop):
        with geoml.progress(stop_after_one):
            trained.predict(resumed, n_sim=3)
    left = resumed.unpredicted()
    assert left.any()
    trained.predict(resumed, n_sim=3, where=left)

    assert not resumed.unpredicted().any()
    for path in ("V/prediction", "V/latent_variance", "V/noise_variance"):
        assert np.array_equal(np.asarray(whole.get(path).values),
                              np.asarray(resumed.get(path).values)), path
    assert np.array_equal(
        np.asarray(whole.get("V").simulations),
        np.asarray(resumed.get("V").simulations))


def test_training_cancelled_leaves_a_model_that_trains_on():
    model = _model()

    def stop_after_two(event):
        if event.done >= 2:
            raise Stop()

    with pytest.raises(Stop):
        with geoml.progress(stop_after_two):
            model.train_full(max_iter=10)

    assert len(model.training_log) == 2
    model.train_full(max_iter=2)
    assert len(model.training_log) == 4
    assert np.all(np.isfinite(model.training_log))
