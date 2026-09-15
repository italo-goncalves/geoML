"""Scoring metrics.

`coverage` and `goodness` are the numbers behind an accuracy plot, and they are
here rather than in the plotting code so that they can be had without drawing
anything -- and tested against distributions whose answer is known in advance.

The categorical scores at the end are the table a categorical variable's
`compute_metrics` reports, one column per category, checked on worked
examples small enough to count by hand.
"""
import numpy as np
import pandas as pd
import pytest
import sklearn.metrics

import geoml
import geoml.math.geometry as geometry
import geoml.metrics as metrics


def test_coverage_finds_the_share_that_is_actually_inside():
    """Truth and simulations drawn from the same distribution: an interval
    holding `p` of the simulations should hold `p` of the truth."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=4000)
    simulations = rng.normal(size=(4000, 200))

    nominal, observed = metrics.coverage(y_true, simulations)
    assert np.allclose(observed, nominal, atol=0.03)


def test_coverage_catches_intervals_that_are_too_narrow():
    """The failure that matters: a model sure of itself and wrong."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=2000)
    simulations = rng.normal(size=(2000, 200)) * 0.2

    _, observed = metrics.coverage(y_true, simulations)
    assert np.all(observed < np.linspace(0.05, 0.95, 19))


def test_coverage_takes_the_probabilities_it_is_given():
    rng = np.random.default_rng(2)
    nominal, observed = metrics.coverage(
        rng.normal(size=200), rng.normal(size=(200, 50)),
        probabilities=[0.5, 0.9])

    assert list(nominal) == [0.5, 0.9]
    assert len(observed) == 2


def test_goodness_is_one_when_the_line_is_met():
    nominal = np.linspace(0.05, 0.95, 19)
    assert metrics.goodness(nominal, nominal) == 1.0


def test_goodness_counts_optimism_twice_as_heavily():
    """Claiming a precision the model does not have is the worse mistake: it
    is the one someone acts on."""
    nominal = np.linspace(0.05, 0.95, 19)

    hedging = metrics.goodness(nominal, np.clip(nominal + 0.1, 0, 1))
    optimistic = metrics.goodness(nominal, np.clip(nominal - 0.1, 0, 1))

    assert hedging > optimistic
    assert np.isclose(1 - optimistic, 2 * (1 - hedging), rtol=0.05)


def test_the_point_errors_on_a_worked_example():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([1.0, 1.0, 0.0])

    assert np.isclose(metrics.rmse(y_true, y_pred), np.sqrt(5 / 3))
    assert np.isclose(metrics.mae(y_true, y_pred), 1.0)
    assert np.isclose(metrics.bias(y_true, y_pred), -1 / 3)


def test_crps_matches_the_gaussian_closed_form():
    """`sigma * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / sqrt(pi))` for a
    Gaussian forecast -- the sample estimator should land on it."""
    from scipy import stats

    rng = np.random.default_rng(3)
    z = np.array([-2.0, -0.5, 0.0, 1.0, 2.5])
    samples = rng.normal(size=[z.size, 40000])

    exact = z * (2 * stats.norm.cdf(z) - 1) \
        + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    for i, (zi, ei) in enumerate(zip(z, exact)):
        assert np.isclose(metrics.crps(np.array([zi]), samples[i:i + 1]), ei,
                          atol=0.01)


def test_crps_reduces_to_the_absolute_error_for_one_sample():
    y_true = np.array([0.0, 1.0, -2.0])
    y_pred = np.array([[1.5], [1.0], [0.5]])
    assert np.isclose(metrics.crps(y_true, y_pred),
                      metrics.mae(y_true, y_pred.ravel()))


def test_crps_is_proper():
    """The truth's own distribution beats both the hedged and the
    overconfident forecast of it -- the property that makes the score worth
    reporting."""
    rng = np.random.default_rng(4)
    y_true = rng.normal(size=3000)

    honest = rng.normal(size=(3000, 400))
    hedged = rng.normal(size=(3000, 400)) * 3.0
    overconfident = rng.normal(size=(3000, 400)) * 0.2

    score = metrics.crps(y_true, honest)
    assert score < metrics.crps(y_true, hedged)
    assert score < metrics.crps(y_true, overconfident)


def test_variogram_score_on_a_worked_example():
    """Three locations, one realization, every pair by hand at p=1."""
    y_true = np.array([0.0, 1.0, 3.0])
    y_pred = np.array([[0.0], [2.0], [3.0]])

    truth = np.array([1.0, 3.0, 2.0])
    ensemble = np.array([2.0, 3.0, 1.0])
    expected = np.mean((truth - ensemble) ** 2)
    assert np.isclose(metrics.variogram_score(y_true, y_pred, p=1.0),
                      expected)


def test_variogram_score_punishes_broken_spatial_structure():
    """Same values at every location, dependence destroyed: `crps` cannot
    tell the two ensembles apart, and this score is the number that can."""
    rng = np.random.default_rng(6)
    n, m = 60, 200
    honest = np.cumsum(rng.normal(size=(n, m)), axis=0)
    y_true = np.cumsum(rng.normal(size=n))

    shuffled = honest.copy()
    for i in range(n):
        shuffled[i] = shuffled[i, rng.permutation(m)]

    assert metrics.variogram_score(y_true, honest) < \
        metrics.variogram_score(y_true, shuffled)


def test_variogram_score_weights_each_pair_by_its_ends():
    """With coordinates the pairs are declustered, and the number is the
    weighted average the weights say it is."""
    rng = np.random.default_rng(11)
    n, m = 40, 30

    # a sparse spread with one crowded knot in it, so the weights are not flat
    coordinates = np.vstack([rng.uniform(0, 100, size=(n, 2)),
                             rng.normal([20, 20], 1.0, size=(n, 2))])
    y_true = coordinates[:, 0] * 0.05 + rng.normal(size=2 * n)
    y_pred = y_true[:, None] + rng.normal(size=(2 * n, m))

    raw = metrics.variogram_score(y_true, y_pred, coordinates=coordinates,
                                  decluster=False)
    assert np.isclose(raw, metrics.variogram_score(y_true, y_pred))

    weights = geometry.declustering_weights(coordinates, y_true)[0]
    i_idx, j_idx = np.triu_indices(2 * n, k=1)
    truth = np.abs(y_true[i_idx] - y_true[j_idx]) ** 0.5
    ensemble = np.mean(
        np.abs(y_pred[i_idx, :] - y_pred[j_idx, :]) ** 0.5, axis=1)
    share = weights[i_idx] * weights[j_idx]
    expected = (share * (truth - ensemble) ** 2).sum() / share.sum()

    declustered = metrics.variogram_score(y_true, y_pred,
                                          coordinates=coordinates)
    assert np.isclose(declustered, expected)
    assert not np.isclose(declustered, raw)


def test_the_score_takes_stored_weights_directly():
    """`weights=` hands the column `container.decluster()` keeps straight
    in, taking precedence over anything computed from coordinates."""
    rng = np.random.default_rng(12)
    n, m = 30, 10
    coordinates = rng.uniform(0, 50, size=(n, 2))
    y_true = rng.normal(size=n)
    y_pred = y_true[:, None] + rng.normal(size=(n, m))

    weights = geometry.declustering_weights(coordinates, y_true)[0]
    given = metrics.variogram_score(y_true, y_pred, weights=weights)
    computed = metrics.variogram_score(y_true, y_pred,
                                       coordinates=coordinates)
    assert np.isclose(given, computed)

    # even weights are the raw score, whatever the coordinates would say
    even = metrics.variogram_score(y_true, y_pred, weights=np.ones(n),
                                   coordinates=coordinates)
    raw = metrics.variogram_score(y_true, y_pred)
    assert np.isclose(even, raw)


# --------------------------------------------------------------------------- #
# the categorical scores
# --------------------------------------------------------------------------- #
def _called(measured, called, probabilities=None, labels=("a", "b", "c")):
    """A categorical variable measured as `measured` and predicted as
    `called`, each category's probability written in as a prediction leaves
    it. None in either list is a location missing that side."""
    n = len(measured)
    point = geoml.data.PointData.from_array(
        np.column_stack([np.arange(n, dtype=float), np.zeros(n)]))
    point.add_categorical_variable(
        "rock", labels=list(labels),
        measurements=np.array(measured, dtype=object))
    rock = point.variables["rock"]
    rock.predicted.values[:] = [-1 if c is None else labels.index(c)
                                for c in called]
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        for k, label in enumerate(labels):
            rock.components[label].probability.values[:] = probabilities[:, k]
    return point, rock


# eight locations: a is under-called and b over-called, so that precision and
# recall part ways
MEASURED = ["a", "a", "a", "a", "b", "b", "c", "c"]
CALLED = ["a", "a", "a", "b", "b", "b", "b", "c"]


def test_a_location_missing_either_side_is_not_scored():
    """As the confusion matrix leaves them out. Counted, a missing
    measurement was a wrong call for the category predicted there, and a
    missing prediction a miss of the one measured."""
    _, rock = _called(["a", "b", "a", None, "b"], ["a", "b", "a", "a", None],
                      labels=("a", "b"))
    scores = rock.compute_metrics()

    assert list(scores.loc["Balanced accuracy"]) == [1.0, 1.0]
    assert list(scores.loc["Cohen's kappa"]) == [1.0, 1.0]


def test_a_variable_predicted_nowhere_is_refused():
    _, rock = _called(["a", "b"], [None, None], labels=("a", "b"))
    with pytest.raises(ValueError, match="predict"):
        rock.compute_metrics()


def test_each_category_gets_the_kappa_of_itself_against_the_rest():
    """Cohen's chance-corrected agreement on the two-way question "is it
    this category?", worked out from its definition."""
    scores = _called(MEASURED, CALLED)[1].compute_metrics()

    measured, called = np.array(MEASURED), np.array(CALLED)
    for label in "abc":
        truth, call = measured == label, called == label
        agreement = np.mean(truth == call)
        chance = truth.mean() * call.mean() \
            + (1 - truth.mean()) * (1 - call.mean())
        assert np.isclose(scores.loc["Cohen's kappa", label],
                          (agreement - chance) / (1 - chance))


def test_with_two_categories_each_kappa_is_the_whole_classifications():
    """Ore against waste is one question asked twice, so both columns hold
    the kappa of the classification as a whole."""
    measured = ["ore", "ore", "waste", "waste", "waste", "ore"]
    called = ["ore", "waste", "waste", "waste", "ore", "ore"]
    scores = _called(measured, called,
                     labels=("ore", "waste"))[1].compute_metrics()

    whole = sklearn.metrics.cohen_kappa_score(measured, called)
    assert np.allclose(scores.loc["Cohen's kappa"], whole)


def test_precision_and_recall_keep_the_two_errors_apart():
    """a is under-called (every call right, a quarter of it missed), b
    over-called (all of it found, half the calls wrong)."""
    scores = _called(MEASURED, CALLED)[1].compute_metrics()

    assert np.allclose(scores.loc["Precision"], [1.0, 0.5, 1.0])
    assert np.allclose(scores.loc["Recall"], [0.75, 1.0, 0.5])
    assert np.allclose(scores.loc["F1 score"], [6 / 7, 2 / 3, 2 / 3])


def test_a_category_never_called_has_no_precision():
    """Undefined, and said so, rather than a zero that reads as a score."""
    scores = _called(["a", "b", "c"], ["a", "b", "b"])[1].compute_metrics()

    assert np.isnan(scores.loc["Precision", "c"])
    assert scores.loc["Recall", "c"] == 0.0


def test_the_disagreement_splits_into_quantity_and_allocation():
    """Pontius and Millones: an error is a wrong proportion or a wrong
    place."""
    # every error here is a wrong proportion: nothing is called a that is
    # not a, so no exchange of labels between locations mends anything
    scores = _called(MEASURED, CALLED)[1].compute_metrics()
    assert np.allclose(scores.loc["Quantity disagreement"],
                       [1 / 8, 2 / 8, 1 / 8])
    assert np.allclose(scores.loc["Allocation disagreement"], 0.0)

    # and here the proportions are right and the places wrong
    scores = _called(["a", "b", "a", "b"], ["b", "a", "a", "b"],
                     labels=("a", "b"))[1].compute_metrics()
    assert np.allclose(scores.loc["Quantity disagreement"], 0.0)
    assert np.allclose(scores.loc["Allocation disagreement"], [0.5, 0.5])


def test_the_two_parts_add_up_to_everything_the_map_gets_wrong():
    """Summed over the categories and halved, one minus the accuracy."""
    rng = np.random.default_rng(7)
    measured = rng.choice(["a", "b", "c"], 300, p=[0.6, 0.3, 0.1])
    called = np.where(rng.uniform(size=300) < 0.7, measured,
                      rng.choice(["a", "b", "c"], 300))
    scores = _called(list(measured), list(called))[1].compute_metrics()

    parts = scores.loc["Quantity disagreement"].sum() / 2 \
        + scores.loc["Allocation disagreement"].sum() / 2
    assert np.isclose(parts, np.mean(measured != called))


def test_the_probability_scores_on_a_worked_example():
    """Each category's probability scored as a forecast of "is it this
    one?"."""
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3],
                              [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])
    measured = ["a", "b", "c", "b"]
    scores = _called(measured, ["a", "b", "c", "a"],
                     probabilities)[1].compute_metrics()

    hit = np.array([[m == label for label in "abc"] for m in measured],
                   dtype=float)
    assert np.allclose(scores.loc["Brier score"],
                       np.mean((probabilities - hit) ** 2, axis=0))
    assert np.allclose(scores.loc["Log score"], -np.mean(
        hit * np.log(probabilities) + (1 - hit) * np.log(1 - probabilities),
        axis=0))


def test_the_probability_scores_reward_the_honest_claim():
    """A probability that is the frequency it claims beats the same one
    hedged toward even odds or pushed toward certainty, where the label
    scores cannot tell them apart: the calls are the same."""
    rng = np.random.default_rng(8)
    honest = rng.uniform(0.05, 0.95, 4000)
    measured = list(np.where(rng.uniform(size=4000) < honest, "a", "b"))

    def scored(claim):
        called = list(np.where(claim >= 0.5, "a", "b"))
        return _called(measured, called, np.column_stack([claim, 1 - claim]),
                       labels=("a", "b"))[1].compute_metrics()["a"]

    truthful = scored(honest)
    hedged = scored(0.5 + 0.3 * (honest - 0.5))
    certain = scored(np.clip(0.5 + 2.0 * (honest - 0.5), 0.01, 0.99))
    for score in ("Brier score", "Log score"):
        assert truthful[score] < hedged[score]
        assert truthful[score] < certain[score]
    assert truthful["Cohen's kappa"] == hedged["Cohen's kappa"] \
        == certain["Cohen's kappa"]


def test_declustering_is_an_option_off_by_default():
    """A location of weight two scores as that location measured twice,
    and only when asked: the stored column alone changes nothing."""
    measured = ["a", None, "a", "b", "b", "c"]
    called = ["a", "b", "b", "b", "c", "c"]
    probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3],
                              [0.4, 0.5, 0.1], [0.1, 0.8, 0.1],
                              [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
    point, rock = _called(measured, called, probabilities)
    # the unmeasured location's weight has to leave with it
    point.add_metadata("declustering",
                       np.array([2.0, 5.0, 1.0, 1.0, 1.0, 1.0]))

    plain = _called(measured, called, probabilities)[1].compute_metrics()
    pd.testing.assert_frame_equal(rock.compute_metrics(), plain)

    twice = _called(["a"] + measured, ["a"] + called,
                    np.vstack([probabilities[:1], probabilities]))[1]
    pd.testing.assert_frame_equal(rock.compute_metrics(decluster=True),
                                  twice.compute_metrics())


def test_declustering_needs_the_weights_stored_first():
    rock = _called(["a", "b"], ["a", "b"], labels=("a", "b"))[1]
    with pytest.raises(ValueError, match="decluster"):
        rock.compute_metrics(decluster=True)
