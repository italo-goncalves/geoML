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

    Initial parameter values are drawn when the objects are built, so the seed
    goes in before that. TensorFlow's global generator is seeded as well: the
    simulation noise in ``likelihood.white_noise`` still draws from it.
    """
    geoml.set_seed(seed)
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


def test_prediction_uses_zarr_backend(monkeypatch):
    """Forcing the on-disk (Zarr) backend must give finite, batch-invariant
    predictions identical to the in-RAM (NumPy) path.

    Walker Lake is small enough to stay on NumPy, so we drop the size threshold
    to route every attribute onto Zarr. This exercises the on-disk batched-write
    path: contiguous-batch region writes into the ``prediction``/``latent_mean``
    stores and the (n_data, n_sim) simulation store.
    """
    import geoml.storage as storage

    # NumPy-backed reference (default threshold).
    ref_model, ref_grid = build_model()
    ref_model.train_full(max_iter=10)
    ref_model.predict(ref_grid, n_sim=4)
    numpy_mean = np.array(ref_grid.variables["V"].latent_mean.values, copy=True)

    # Same model/data, but every attribute forced onto Zarr, predicted in
    # several batches.
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    model, grid = build_model()
    model.train_full(max_iter=10)
    model.options.prediction_batch_size = 500
    model.predict(grid, n_sim=4)

    var = grid.variables["V"]
    assert var.latent_mean.values.backend == "zarr"
    assert var.simulations.backend == "zarr"

    zarr_mean = np.array(var.latent_mean.values, copy=True)
    assert np.all(np.isfinite(zarr_mean))
    assert np.allclose(numpy_mean, zarr_mean, atol=1e-8)


def test_zarr_scratch_is_consolidated_and_quantiles_lazy(monkeypatch):
    """With the on-disk backend, all of a container's working arrays must share
    one scratch store, and reset_quantiles must match the eager computation
    without materializing the simulations."""
    import geoml.storage as storage
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)

    model, grid = build_model()
    model.train_full(max_iter=10)
    model.predict(grid, n_sim=8)
    var = grid.variables["V"]

    paths = {var.latent_mean.values.store_path,
             var.latent_variance.values.store_path,
             var.prediction.values.store_path,
             var.simulations.store_path}
    assert len(paths) == 1                          # one store per container

    reference = np.quantile(
        np.asarray(var.simulations), [0.25, 0.75], axis=1).T
    var.reset_quantiles([0.25, 0.75])
    got = np.stack([np.asarray(var.quantiles[p].values)
                    for p in (0.25, 0.75)], axis=1)
    assert np.allclose(got, reference)
    assert var.quantiles[0.25].values.backend == "zarr"   # results spill too

    # reset_probabilities is the inverse of reset_quantiles: a cutoff in the
    # variable's units maps to the fraction of simulations at or below it.
    sims = np.asarray(var.simulations)
    cutoff = float(np.median(sims))
    var.reset_probabilities([cutoff])
    prob = np.asarray(var.probabilities[cutoff].values)
    assert np.allclose(prob, np.mean(sims <= cutoff, axis=1))
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def build_gradient_constrained_model(seed=1234):
    """A network rooted at a GradientConstrainedInput (gradient/implicit path).

    The inducing grid deliberately does not contain the directional-data
    locations; the node merges them into its (deduplicated) base set. This
    construction used to crash until n_ip was derived from that base set.
    """
    import tensorflow as tf
    geoml.set_seed(seed)
    tf.random.set_seed(seed)
    point, dirs, _ = geoml.datasets.example_fold()
    ip = geoml.data.Grid2D(start=[10, 10], n=[8, 8], step=[11, 11])
    covariance = geoml.kernels.Covariance(
        geoml.kernels.Gaussian(), geoml.transform.Isotropic(30))
    root = geoml.latent.GradientConstrainedInput(ip, dirs, covariance, size=1)
    options = geoml.models.GPOptions(
        verbose=False, seed=seed, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "rock_num", geoml.likelihood.Gaussian(), root, options=options)
    grid = geoml.data.Grid2D(start=[5, 5], n=[40, 40], step=[2, 2])
    return model, grid


def test_gradient_constrained_input_batch_invariant():
    model, grid = build_gradient_constrained_model()
    model.train_full(max_iter=8)

    model.options.prediction_batch_size = 10 ** 9
    model.predict(grid, n_sim=4)
    one = np.array(grid.variables["rock_num"].latent_mean.values, copy=True)
    model.options.prediction_batch_size = 300
    model.predict(grid, n_sim=4)
    many = np.array(grid.variables["rock_num"].latent_mean.values, copy=True)

    assert np.all(np.isfinite(one))
    assert np.allclose(one, many, atol=1e-8)


def test_gradient_constrained_input_reflects_retraining():
    """Guards the GradientConstrainedInput.cache_prediction_state override: its
    own posterior state (scale, alpha, cov_*) must be refreshed between calls."""
    model, grid = build_gradient_constrained_model()
    model.train_full(max_iter=8)
    model.predict(grid, n_sim=4)
    before = np.array(grid.variables["rock_num"].latent_mean.values, copy=True)

    model.train_full(max_iter=12)
    model.predict(grid, n_sim=4)
    after = np.array(grid.variables["rock_num"].latent_mean.values, copy=True)

    assert not np.allclose(before, after)
