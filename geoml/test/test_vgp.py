"""Integration tests for the core VGP path (build / train / predict).

These exercise the ``VGPNetwork`` model end to end on the bundled Walker Lake
dataset. They are deliberately small so they run in a few seconds on CPU, while
still covering the invariants that the efficiency refactors must preserve:
reproducibility, loss reduction, finite outputs, and prediction results that do
not depend on the batching used to compute them.
"""
import numpy as np
import geoml


def build_model(n_ip=12, samples=10, seed=1234):
    """A minimal single-GP network on the Walker Lake ``V`` variable.

    ``BasicGP`` draws its initial inducing-point values from the global NumPy
    RNG, so we seed it here to make construction (and therefore the whole run)
    reproducible.
    """
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    walker_point, walker_grid = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[n_ip, n_ip], step=[22, 25])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    likelihood = geoml.likelihood.Gaussian()
    options = geoml.models.GPOptions(
        verbose=False, seed=seed, training_samples=samples)
    model = geoml.models.VGPNetwork(
        walker_point, "V", likelihood, gp, options=options)
    return model, walker_grid


def test_train_is_reproducible():
    """Same seed must give the same ELBO trajectory."""
    m1, _ = build_model()
    m1.train_full(max_iter=15)
    m2, _ = build_model()
    m2.train_full(max_iter=15)
    assert np.allclose(m1.training_log, m2.training_log)


def test_train_reduces_loss():
    """The ELBO should improve over training."""
    model, _ = build_model()
    model.train_full(max_iter=30)
    assert model.training_log[-1] > model.training_log[0]


def test_outputs_are_finite():
    model, grid = build_model()
    model.train_full(max_iter=10)
    model.predict(grid, n_sim=8)
    latent_mean = np.asarray(grid.variables["V"].latent_mean.values)
    prediction = np.asarray(grid.variables["V"].get_predictions())
    assert np.all(np.isfinite(latent_mean))
    assert np.all(np.isfinite(prediction))


def test_prediction_is_batch_invariant():
    """Prediction must not depend on the prediction batch size.

    ``latent_mean`` is the deterministic posterior mean (no Monte Carlo), so it
    must be identical whether the grid is processed in one batch or many. This
    guards the batching logic in ``VGPNetwork.predict``.
    """
    model, grid = build_model()
    model.train_full(max_iter=10)

    model.options.prediction_batch_size = 10 ** 9  # single batch
    model.predict(grid, n_sim=4)
    one_batch = np.array(grid.variables["V"].latent_mean.values, copy=True)

    model.options.prediction_batch_size = 5000     # ~16 batches
    model.predict(grid, n_sim=4)
    many_batches = np.array(grid.variables["V"].latent_mean.values, copy=True)

    assert np.allclose(one_batch, many_batches, atol=1e-8)


def test_prediction_reflects_retraining():
    """A second prediction must reflect parameters updated by further training.

    The posterior is refreshed once per ``predict`` call, so predicting again
    after more training has to pick up the new parameters rather than reuse a
    stale cached state.
    """
    model, grid = build_model()
    model.train_full(max_iter=10)
    model.predict(grid, n_sim=4)
    before = np.array(grid.variables["V"].latent_mean.values, copy=True)

    model.train_full(max_iter=20)
    model.predict(grid, n_sim=4)
    after = np.array(grid.variables["V"].latent_mean.values, copy=True)

    assert not np.allclose(before, after)
