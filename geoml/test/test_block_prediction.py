"""Prediction on blocks with discretization: batching, fan-out, noise.

A block fans out into ``prod(discretization)`` sub-blocks that the model
evaluates before they are averaged back to one value per block. Three things
have to hold: the batch is measured in the rows the model actually sees, the
fan-out is the same as taking each block's sub-grid in turn, and the noise is
drawn per sub-block position — independent within a block, the same pattern in
every block — so that averaging integrates it instead of adding fresh
randomness, and the result does not depend on how the prediction was batched.
"""
import numpy as np
import pandas as pd
import pytest

import geoml


def _model(n_dim=2, seed=1234, n_train=30, n_ip=5, max_iter=3):
    import tensorflow as tf
    geoml.set_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cols = [f"c{i}" for i in range(n_dim)]
    coords = np.random.uniform(0, 100, (n_train, n_dim))
    df = pd.DataFrame(coords, columns=cols)
    point = geoml.data.PointData(df, cols)
    point.add_continuous_variable("v", np.sin(coords[:, 0] / 25.0))

    ind = pd.DataFrame(np.random.uniform(0, 100, (n_ip, n_dim)), columns=cols)
    root = geoml.latent.BasicInput(
        geoml.data.PointData(ind, cols),
        transform=geoml.transform.Isotropic(40))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(
        verbose=False, seed=seed, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), gp, options=options)
    model.train_full(max_iter=max_iter)
    return model


def _blocks(discretization=(3, 3)):
    return geoml.data.Blocks2D(
        start=[0, 0], n=[4, 4], step=[20, 20],
        discretization=list(discretization))


# --------------------------------------------------------------------------- #
# the fan-out
# --------------------------------------------------------------------------- #
def test_the_fan_out_matches_block_by_block_discretization():
    blocks = _blocks()
    coords, splits = blocks.get_batched_coordinates(np.arange(blocks.n_data))

    k = int(np.prod(blocks.discretization))
    assert splits == blocks.n_data
    assert coords.shape == (blocks.n_data * k, blocks.n_dim)

    # block-major: block b owns rows b*k to (b+1)*k, which is what the
    # aggregation assumes
    centers = np.asarray(blocks.coordinates)
    for b in (0, 5, blocks.n_data - 1):
        expected = blocks.sub_grid + centers[b][None, :]
        assert np.allclose(coords[b * k:(b + 1) * k], expected)


def test_a_batch_covers_only_its_own_blocks():
    blocks = _blocks()
    k = int(np.prod(blocks.discretization))
    coords, splits = blocks.get_batched_coordinates(np.array([2, 3]))
    assert splits == 2
    assert coords.shape == (2 * k, blocks.n_dim)
    centers = np.asarray(blocks.coordinates)
    assert np.allclose(coords[:k], blocks.sub_grid + centers[2][None, :])


def test_without_discretization_nothing_fans_out():
    blocks = _blocks(discretization=(1, 1))
    coords, splits = blocks.get_batched_coordinates(np.arange(blocks.n_data))
    assert splits is None
    assert coords.shape == (blocks.n_data, blocks.n_dim)
    assert blocks.rows_per_location == 1


# --------------------------------------------------------------------------- #
# the batch counts the rows the model sees
# --------------------------------------------------------------------------- #
def test_rows_per_location_is_the_discretization():
    assert _blocks((3, 3)).rows_per_location == 9
    assert _blocks((1, 1)).rows_per_location == 1

    point = geoml.data.PointData(
        pd.DataFrame(np.zeros((3, 2)), columns=["c0", "c1"]), ["c0", "c1"])
    assert point.rows_per_location == 1


def test_the_batch_is_measured_in_sub_blocks(monkeypatch):
    """20000 blocks of [5,5,5] used to hand the model 2.5M rows at once."""
    model = _model()
    blocks = _blocks()
    k = int(np.prod(blocks.discretization))
    model.options.prediction_batch_size = 2 * k      # two blocks' worth

    seen = []
    original = type(blocks).get_batched_coordinates

    def spy(self, index):
        coords, splits = original(self, index)
        seen.append(coords.shape[0])
        return coords, splits

    monkeypatch.setattr(type(blocks), "get_batched_coordinates", spy)
    model.predict(blocks, n_sim=4)

    assert max(seen) <= model.options.prediction_batch_size
    assert sum(seen) == blocks.n_data * k


# --------------------------------------------------------------------------- #
# the noise
# --------------------------------------------------------------------------- #
def _noise(likelihood, n_blocks, k, n_sim=4, seed=1234):
    import tensorflow as tf
    zeros = tf.zeros([n_blocks * k, 1, n_sim], dtype=tf.float64)
    return np.asarray(likelihood._add_noise(zeros, seed=seed,
                                            n_splits=n_blocks))


def test_the_sub_blocks_of_a_block_get_independent_noise():
    """Averaging independent draws is what integrates the noise over a block."""
    noise = _noise(geoml.likelihood.Gaussian(), n_blocks=3, k=9)
    within = noise.reshape(3, 9, -1)[0, :, 0]
    assert len(np.unique(within)) == 9


def test_every_block_gets_the_same_pattern():
    """So the noise shifts the map instead of roughening it."""
    noise = _noise(geoml.likelihood.Gaussian(), n_blocks=3, k=9)
    grouped = noise.reshape(3, 9, 1, -1)
    assert np.allclose(grouped[0], grouped[1])
    assert np.allclose(grouped[0], grouped[2])


def test_the_noise_shape_does_not_depend_on_the_batch():
    """The draw is indexed by sub-block, so batching cannot move it."""
    likelihood = geoml.likelihood.Gaussian()
    two = _noise(likelihood, n_blocks=2, k=9).reshape(2, 9, 1, -1)
    five = _noise(likelihood, n_blocks=5, k=9).reshape(5, 9, 1, -1)
    assert np.allclose(two[0], five[0])


def test_the_noise_follows_the_seed():
    likelihood = geoml.likelihood.Gaussian()
    same = _noise(likelihood, 2, 9, seed=7)
    again = _noise(likelihood, 2, 9, seed=7)
    other = _noise(likelihood, 2, 9, seed=8)
    assert np.allclose(same, again)
    assert not np.allclose(same, other)


def test_without_discretization_every_row_gets_its_own_noise():
    likelihood = geoml.likelihood.Gaussian()
    import tensorflow as tf
    zeros = tf.zeros([20, 1, 3], dtype=tf.float64)
    noise = np.asarray(likelihood._add_noise(zeros, seed=1234, n_splits=None))
    assert noise.shape == (20, 1, 3)
    assert len(np.unique(noise[:, 0, 0])) == 20


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def _simulations(model, batch_size, n_sim=6, seed=None):
    blocks = _blocks()
    if seed is not None:
        model.options.seed = seed
    model.options.prediction_batch_size = batch_size
    model.predict(blocks, n_sim=n_sim)
    return np.asarray(blocks.variables["v"].simulations)


def test_the_simulations_do_not_depend_on_the_batching():
    """The whole point: one block per batch, or all of them, same answer."""
    model = _model()
    one_batch = _simulations(model, batch_size=10 ** 9)
    many = _simulations(model, batch_size=9)          # a block at a time
    assert np.allclose(one_batch, many, atol=1e-10)


def test_the_simulations_follow_the_model_seed():
    """`options.seed` used to reach the latent draws but not the noise."""
    model = _model()
    first = _simulations(model, batch_size=10 ** 9, seed=1234)
    again = _simulations(model, batch_size=10 ** 9, seed=1234)
    other = _simulations(model, batch_size=10 ** 9, seed=99)
    assert np.allclose(first, again)
    assert not np.allclose(first, other)


def test_the_blocks_still_average_their_sub_blocks():
    """Aggregation is unchanged: a block is the mean of its sub-blocks."""
    model = _model()
    blocks = _blocks()
    model.options.prediction_batch_size = 10 ** 9
    model.predict(blocks, n_sim=4)
    mean = np.asarray(blocks.variables["v"].latent_mean.values)

    points = geoml.data.PointData.from_array(
        blocks.get_batched_coordinates(np.arange(blocks.n_data))[0])
    model.predict(points, n_sim=4)
    sub = np.asarray(points.variables["v"].latent_mean.values)

    k = int(np.prod(blocks.discretization))
    assert np.allclose(mean, sub.reshape(blocks.n_data, k).mean(axis=1),
                       atol=1e-8)


# --------------------------------------------------------------------------- #
# within-block dispersion
# --------------------------------------------------------------------------- #
def _dispersed(model, n_sim=8, discretization=(3, 3)):
    """A block model predicted noise-free, and its sub-blocks as points.

    Noise-free on both sides deliberately. With `monte_carlo` the block path
    draws one noise value per sub-block *position* and shares the pattern
    across blocks, while a point path draws one per row, so the two would not
    be looking at the same field and nothing could be checked against
    anything.
    """
    blocks = _blocks(discretization)
    model.options.prediction_batch_size = 10 ** 9
    model.predict(blocks, n_sim=n_sim, include_noise=None)

    points = geoml.data.PointData.from_array(
        blocks.get_batched_coordinates(np.arange(blocks.n_data))[0])
    model.predict(points, n_sim=n_sim, include_noise=None)

    k = int(np.prod(blocks.discretization))
    sub = np.asarray(points.variables["v"].simulations)
    return blocks, sub.reshape(blocks.n_data, k, n_sim)


def test_the_dispersion_is_the_spread_of_the_sub_blocks():
    """What the column claims: the variance over a block's own sub-blocks,
    averaged over realizations."""
    model = _model()
    blocks, grouped = _dispersed(model)

    # the block value being the mean of its sub-blocks is what says both
    # predictions saw the same field; without it the comparison below proves
    # nothing
    assert np.allclose(grouped.mean(axis=1),
                       np.asarray(blocks.variables["v"].simulations),
                       atol=1e-10)

    stored = blocks.variables["v"].dispersion.values.to_numpy()
    assert np.allclose(stored, grouped.var(axis=1).mean(axis=1), atol=1e-12)
    assert np.all(stored >= 0.0)


def test_the_dispersion_is_not_the_model_uncertainty():
    """Two different questions -- how much the block varies inside itself,
    against how sure the model is of the block."""
    model = _model()
    blocks, _ = _dispersed(model)

    stored = blocks.variables["v"].dispersion.values.to_numpy()
    latent = blocks.variables["v"].latent_variance.values.to_numpy()
    assert not np.allclose(stored, latent)


def test_a_finer_discretization_sees_more_of_the_block():
    """A block cut more finely reaches nearer its corners, so it can only
    find at least as much spread as a coarse cut did."""
    model = _model()
    coarse, _ = _dispersed(model, discretization=(2, 2))
    fine, _ = _dispersed(model, discretization=(6, 6))

    assert np.mean(fine.variables["v"].dispersion.values.to_numpy()) > \
        np.mean(coarse.variables["v"].dispersion.values.to_numpy())


def test_without_a_discretization_there_is_no_dispersion():
    """A location has no interior, and a block the model treats as its own
    centre has one it knows nothing about. Neither is zero, which would read
    as 'uniform inside' -- a claim nobody made."""
    model = _model()
    points = geoml.data.Grid2D(start=[0, 0], n=[4, 4], step=[20, 20])
    model.predict(points, n_sim=4)

    dispersion = points.variables["v"].dispersion
    assert np.all(np.isnan(dispersion.values.to_numpy()))
    assert not dispersion._has_content()
    # and so it stays out of the data frame entirely
    assert not any("dispersion" in c for c in points.as_data_frame().columns)

    blocks = _blocks(discretization=(1, 1))
    model.predict(blocks, n_sim=4)
    assert np.all(np.isnan(
        blocks.variables["v"].dispersion.values.to_numpy()))


def test_the_dispersion_is_carried_around_like_any_other_column(tmp_path):
    model = _model()
    blocks, _ = _dispersed(model)
    stored = blocks.variables["v"].dispersion.values.to_numpy()

    assert any("dispersion" in c for c in blocks.as_data_frame().columns)

    blocks.to_zarr(str(tmp_path / "b.zarr"))
    reopened = geoml.data.Blocks2D.open(str(tmp_path / "b.zarr"))
    assert np.allclose(
        reopened.variables["v"].dispersion.values.to_numpy(), stored)
