"""A trained bias per category on the Gaussian indicator likelihoods.

Away from the data a category's latent value returns to zero, so without a
bias every category is as likely as the next there; the bias is what the
categories return to instead. Optional, since a save stores its parameters
by position.
"""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.transform as tr


@pytest.mark.parametrize("cls", [lk.CategoricalGaussianIndicator,
                                 lk.HierarchicalGaussianIndicator])
def test_no_bias_unless_asked(cls):
    assert "bias" not in cls(3).parameters
    assert len(cls(3, bias=True).all_parameters) \
        == len(cls(3).all_parameters) + 1


@pytest.mark.parametrize("cls", [lk.CategoricalGaussianIndicator,
                                 lk.HierarchicalGaussianIndicator])
def test_the_bias_moves_the_latent_before_anything_reads_it(cls):
    """With a bias the prediction is the unbiased prediction of the latent
    moved by it, and so are the realizations; at a latent of zero the plain
    indicator's categories are even (the hierarchical one's are not, the
    higher priority winning a tie)."""
    n = 4
    zero = tf.zeros([n, 2], tf.float64)
    ones = tf.ones([n, 2], tf.float64)
    sims = tf.zeros([n, 2, 5], tf.float64)

    biased = cls(2, bias=True)
    biased.parameters["bias"].set_value([1.0, -1.0])
    out = biased.predict(zero, ones, sims, ones)
    moved = tf.constant([[1.0, -1.0]] * n, tf.float64)
    reference = cls(2).predict(moved, ones, sims + moved[:, :, None], ones)
    np.testing.assert_allclose(out["probability"], reference["probability"])
    np.testing.assert_allclose(out["simulations"], reference["simulations"])
    if cls is lk.CategoricalGaussianIndicator:
        plain = cls(2).predict(zero, ones, sims, ones)
        np.testing.assert_allclose(plain["probability"], 0.5)
        assert np.all(np.asarray(out["probability"])[:, 0] > 0.5)

    y = tf.constant([[1.0, 0.0]] * n, tf.float64)
    assert float(biased.log_lik(zero, ones, y, ones)) == pytest.approx(
        float(cls(2).log_lik(moved, ones, y, ones)))


def _model(bias):
    geoml.set_seed(1234)
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 60))
    labels = np.where(rng.uniform(size=60) < 0.8, "granite", "basalt")
    data = geoml.data.PointData.from_array(x[:, None], ["X"])
    data.add_categorical_variable("rock", ["basalt", "granite"], labels)
    root = latent.BasicInput(geoml.data.Grid1D(start=-1, n=25, step=0.5),
                             transform=tr.Isotropic(0.3))
    model = geoml.models.VGPNetwork(
        data, "rock", lk.CategoricalGaussianIndicator(2, bias=bias),
        latent.BasicGP(root, size=2),
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=300)
    return model


def test_a_trained_bias_returns_to_the_majority_away_from_the_data(tmp_path):
    """Four samples in five granite, the categories' ranges short: far from
    every sample the plain likelihood says even odds and the biased one
    says granite -- and a reloaded model says the same."""
    far = geoml.data.PointData.from_array(np.array([[40.0], [60.0]]), ["X"])
    plain = _model(False)
    plain.predict(far, n_sim=10)
    granite = far.variables["rock"].components["granite"]
    np.testing.assert_allclose(granite.probability.values, 0.5, atol=0.02)

    model = _model(True)
    target = geoml.data.PointData.from_array(np.array([[40.0], [60.0]]),
                                             ["X"])
    model.predict(target, n_sim=10)
    granite = target.variables["rock"].components["granite"]
    assert np.all(np.asarray(granite.probability.values) > 0.6)

    model.save(str(tmp_path / "m"))
    again = geoml.persistence.load_model(str(tmp_path / "m"))
    other = geoml.data.PointData.from_array(np.array([[40.0], [60.0]]),
                                            ["X"])
    again.predict(other, n_sim=10)
    np.testing.assert_allclose(
        other.variables["rock"].components["granite"].probability.values,
        granite.probability.values)
