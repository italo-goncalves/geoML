"""Stopping training once the bound has settled (`training_tolerance`).

The rule itself is arithmetic over a sequence, so most of it is tested on
sequences written by hand -- a curve that flattens, one that keeps climbing,
one that diverges -- where the right answer is known rather than measured.
The rest checks that both trainers are actually wired to it, and that the
option is absent unless asked for.
"""
import numpy as np

import geoml
import geoml.models as models


def _flattening(n=200, limit=100.0, tau=15.0):
    """A bound approaching a limit, the shape a training curve has."""
    t = np.arange(n)
    return limit - 50.0 * np.exp(-t / tau)


def _climbing(n=200, slope=0.5):
    """A bound still going up at a steady rate when the run ends."""
    return slope * np.arange(n)


def _stops_at(rule, curve):
    for i, value in enumerate(curve):
        if rule.stop(value):
            return i + 1
    return None


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_no_tolerance_never_stops():
    """The default, and what every version before 0.6.5 did."""
    for tolerance in (None, 0.0):
        rule = models._Convergence(tolerance)
        assert _stops_at(rule, _flattening()) is None


def test_a_settled_bound_stops_and_a_climbing_one_does_not():
    curve = _flattening()
    at = _stops_at(models._Convergence(0.01), curve)
    assert at is not None

    # and it stopped where the curve had nothing left to give
    total = curve[-1] - curve[0]
    assert (curve[at - 1] - curve[0]) / total > 0.95

    assert _stops_at(models._Convergence(0.01), _climbing()) is None


def test_nothing_stops_before_the_window_has_filled():
    """The burn-in. A flat stretch before any real progress has a small gain
    over the window and a small gain overall, and their ratio is not evidence
    of anything -- so the rule refuses to look until it can see a window
    back."""
    rule = models._Convergence(0.5, window=20)
    assert _stops_at(rule, np.zeros(15) + 3.0) is None

    rule = models._Convergence(0.5, window=20)
    assert _stops_at(rule, _flattening(tau=2.0)) > 20


def test_a_diverged_run_is_not_called_converged():
    """A bound that goes to NaN fails every comparison, so training runs to
    its cap and leaves the NaNs in the log where they can be seen, rather
    than stopping and reporting success."""
    curve = _flattening(200, tau=5.0)
    would_stop = _stops_at(models._Convergence(0.05), curve)
    assert would_stop is not None

    poisoned = curve.copy()
    poisoned[would_stop - 3:] = np.nan
    assert _stops_at(models._Convergence(0.05), poisoned) is None


def test_progress_is_measured_from_the_start_of_this_call():
    """Which is what keeps phased training working: a model that plateaus,
    is given a smaller learning rate and improves again is judged on what
    the new phase gained, not on what the old one did."""
    first = _flattening(120)
    rule = models._Convergence(0.01)
    assert _stops_at(rule, first) is not None

    # the second phase starts where the first ended and climbs slowly: small
    # steps, but every one of them a real share of what this phase has won
    second = first[-1] + _climbing(120, slope=0.005)
    assert _stops_at(models._Convergence(0.01), second) is None

    # ...and a rule that had carried the first phase's baseline would see a
    # huge denominator and stop at once
    carried = models._Convergence(0.01)
    carried.stop(first[0])
    assert _stops_at(carried, second) is not None


def test_a_wider_window_looks_further_back():
    curve = _flattening(400, tau=40.0)
    narrow = _stops_at(models._Convergence(0.01, window=5), curve)
    wide = _stops_at(models._Convergence(0.01, window=40), curve)
    assert narrow < wide


# --------------------------------------------------------------------------- #
# the trainers
# --------------------------------------------------------------------------- #
def _model(tolerance=None, batch_size=2000, seed=1234):
    geoml.set_seed(seed)
    walker_point, _ = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[8, 8], step=[33, 37])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Cubic())
    options = geoml.models.GPOptions(
        verbose=False, training_samples=8, training_batch_size=batch_size,
        training_tolerance=tolerance)
    return geoml.models.VGPNetwork(
        walker_point, "V", geoml.likelihood.Gaussian(), gp, options=options)


def test_train_full_runs_its_full_count_without_a_tolerance():
    model = _model()
    model.train_full(max_iter=40)
    assert len(model.training_log) == 40


def test_train_full_stops_once_the_bound_settles():
    model = _model(tolerance=0.05)
    model.train_full(max_iter=400)
    assert 20 < len(model.training_log) < 400


def test_train_svi_stops_on_whole_epochs():
    """The criterion reads one value an epoch -- the mean over its batches --
    so training ends at an epoch boundary, never part way through one."""
    model = _model(tolerance=0.1, batch_size=100)
    model.train_svi(epochs=60)

    per_epoch = len(model.options.batch_index(model.data.n_data))
    assert len(model.training_log) % per_epoch == 0
    assert 0 < len(model.training_log) < 60 * per_epoch


def test_a_second_phase_is_judged_from_its_own_start():
    """The pattern deep models are trained with: train, lower the learning
    rate, train again. The second call has its own baseline, so it is not
    stopped at once by everything the first one achieved."""
    model = _model(tolerance=0.05)
    model.train_full(max_iter=200)
    first = len(model.training_log)

    model.set_learning_rate(2e-3)
    model.train_full(max_iter=120)
    second = len(model.training_log) - first
    assert second > 20


# --------------------------------------------------------------------------- #
# the option
# --------------------------------------------------------------------------- #
def test_the_criterion_is_off_unless_asked_for():
    assert geoml.models.GPOptions().training_tolerance is None


def test_options_saved_before_the_criterion_still_open():
    """`persistence` rebuilds options with __new__ plus a vars() update, so an
    option added later is absent from an old file."""
    old = geoml.models.GPOptions.__new__(geoml.models.GPOptions)
    vars(old).update({"verbose": False, "prediction_batch_size": 20000,
                      "training_batch_size": 2000, "seed": 1234,
                      "jitter": 1e-9, "training_samples": 20})

    assert "training_tolerance" not in vars(old)
    assert old.training_tolerance is None
