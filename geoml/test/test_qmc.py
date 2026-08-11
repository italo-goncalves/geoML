"""The quasi-Monte Carlo option for the posterior simulations.

``GPOptions(qmc_simulations=True)`` draws the simulations from a
seeded-scramble Sobol sequence instead of pseudo-random normals. These tests
pin the contract: off by default and for options saved before it existed,
deterministic and batch-invariant when on, honoured when the flag is flipped
on a live model, unchanged by XLA, and actually worth turning on -- the same
number of simulations must land closer to the posterior the model reports.
"""
import numpy as np
import tensorflow as tf

import geoml
from test_vgp import build_model


def _sims(grid):
    return np.array(np.asarray(grid.variables["V"].simulations), copy=True)


def test_the_default_is_monte_carlo():
    assert geoml.models.GPOptions().qmc_simulations is False


def test_options_saved_before_qmc_simulations_still_open():
    """`persistence` rebuilds options with __new__ plus a vars() update, so an
    option added later is absent from an old file; the class default covers."""
    old = geoml.models.GPOptions.__new__(geoml.models.GPOptions)
    vars(old).update({"verbose": False, "prediction_batch_size": 20000,
                      "training_batch_size": 2000, "seed": 1234,
                      "jitter": 1e-9, "training_samples": 20})

    assert "qmc_simulations" not in vars(old)
    assert old.qmc_simulations is False


def test_qmc_prediction_is_deterministic_and_batch_invariant():
    """The Sobol points are fixed by the seed and drawn over the inducing
    values, not the data, so a location's simulation cannot depend on the
    batch that computed it -- the same contract Monte Carlo already keeps."""
    model, grid = build_model()
    model.train_full(max_iter=10)
    model.options.qmc_simulations = True

    model.options.prediction_batch_size = 10 ** 9  # single batch
    model.predict(grid, n_sim=8)
    one_batch = _sims(grid)

    model.options.prediction_batch_size = 5000     # ~16 batches
    model.predict(grid, n_sim=8)
    many_batches = _sims(grid)

    assert np.all(np.isfinite(one_batch))
    assert np.allclose(one_batch, many_batches, atol=1e-8)


def test_the_flag_flips_on_a_live_model():
    """Each simulation rule is baked into its trace, so flipping the option
    has to reach for a different traced function -- and back again."""
    model, grid = build_model()
    model.train_full(max_iter=10)

    model.predict(grid, n_sim=8)
    monte_carlo = _sims(grid)
    mean_mc = np.array(grid.variables["V"].latent_mean.values, copy=True)
    first = model._compiled[(False, False)]

    model.options.qmc_simulations = True
    model.predict(grid, n_sim=8)
    quasi = _sims(grid)
    mean_qmc = np.array(grid.variables["V"].latent_mean.values, copy=True)

    assert set(model._compiled) == {(False, False), (False, True)}
    assert not np.allclose(monte_carlo, quasi)
    # the deterministic outputs do not depend on how the ensemble is drawn
    assert np.allclose(mean_mc, mean_qmc, atol=1e-8)

    model.options.qmc_simulations = False
    model.predict(grid, n_sim=8)
    assert model._compiled[(False, False)] is first  # reused, not rebuilt
    assert np.allclose(_sims(grid), monte_carlo, atol=1e-8)


def test_qmc_is_unchanged_by_xla():
    """The Sobol points are constants in the graph, so XLA compiles them like
    anything else and must not change the answer."""
    model, grid = build_model()
    model.train_full(max_iter=10)
    model.options.qmc_simulations = True

    model.predict(grid, n_sim=8)
    plain = _sims(grid)

    model.options.jit_predict = True
    model.predict(grid, n_sim=8)
    compiled = _sims(grid)

    assert (False, True) in model._compiled
    assert (True, True) in model._compiled
    assert np.allclose(plain, compiled, atol=1e-8)


def test_qmc_lands_closer_to_the_posterior():
    """The point of the option: at the same ``n_sim``, the ensemble's mean
    must sit far closer to the exact posterior the same call reports (the
    benchmark measured 7-37x over 16-256 simulations), while the ensemble's
    own spread -- where QMC gains nothing -- must not be paid for it.
    Everything is fixed by the seeds, so this is a pinned comparison, not a
    statistical one."""
    model, grid = build_model()
    model.train_full(max_iter=10)
    geoml.latent.refresh_cached(model.latent_network, model.options.jitter)

    x = tf.constant(grid.coordinates[:500], tf.float64)
    with geoml.latent.simulation_rule(False):
        mu, var, sims_mc, _, _ = model.latent_network.predict(
            x, n_sim=64, seed=[1234, 0])
    with geoml.latent.simulation_rule(True):
        _, _, sims_qmc, _, _ = model.latent_network.predict(
            x, n_sim=64, seed=[1234, 0])

    mu = mu.numpy()[:, :, 0]
    sd = np.sqrt(var.numpy())

    def rms(a):
        return np.sqrt(np.mean(a ** 2))

    mean_error_mc = rms(np.mean(sims_mc.numpy(), axis=-1) - mu)
    mean_error_qmc = rms(np.mean(sims_qmc.numpy(), axis=-1) - mu)
    sd_error_mc = rms(np.std(sims_mc.numpy(), axis=-1) - sd)
    sd_error_qmc = rms(np.std(sims_qmc.numpy(), axis=-1) - sd)

    assert mean_error_qmc < 0.5 * mean_error_mc
    assert sd_error_qmc < 1.5 * sd_error_mc
