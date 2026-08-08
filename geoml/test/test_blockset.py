"""Blocks of several sizes on one integer lattice.

`BlockSet3D` carries fine blocks where they are wanted and coarse ones
everywhere else. Three things have to hold, and the integer lattice is what
makes each of them checkable rather than approximate: the blocks tile their
box exactly at every stage, a block of any size fans out into the same number
of rows as any other (so nothing downstream has to know that levels exist),
and an unrefined set is the same object a `Blocks3D` would have been.
"""
import numpy as np
import pytest

import geoml


START, N, STEP = [0, 0, 0], [4, 4, 2], [40.0, 40.0, 20.0]


def _blockset(discretization=(2, 2, 2), max_levels=2):
    return geoml.data.BlockSet3D(START, N, STEP,
                                 discretization=discretization,
                                 max_levels=max_levels)


def _model(seed=1234):
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 160, size=[200, 3])
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable(
        "y", np.sin(xyz[:, 0] / 40) + xyz[:, 2] / 100)

    ip = geoml.inducing.from_kmeans(point, 60, seed=0)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(50.0))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "y", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False, seed=seed,
                                       training_samples=8))
    model.train_full(max_iter=4)
    return model


# --------------------------------------------------------------------------- #
# the lattice
# --------------------------------------------------------------------------- #
def test_it_starts_as_a_full_uniform_grid():
    blocks = _blockset()
    assert blocks.n_data == int(np.prod(N))
    assert np.all(blocks.level == 0)
    assert np.allclose(np.unique(blocks.block_size, axis=0), [STEP])
    assert blocks.is_full()
    assert np.isclose(blocks.block_volume.sum(),
                      np.prod(np.array(N) * np.array(STEP)))


def test_the_base_cell_follows_max_levels():
    """`max_levels` is what fixes the lattice everything else is counted in."""
    for levels in (0, 1, 3):
        blocks = _blockset(max_levels=levels)
        assert np.allclose(blocks.base_step,
                           np.array(STEP) / 2 ** levels)
        assert np.all(blocks.lattice_shape
                      == np.array(N) * 2 ** levels)


def test_splitting_keeps_the_box_full_and_the_volume_exact():
    blocks = _blockset()
    mask = np.asarray(blocks.coordinates)[:, 0] < 60.0
    fine = blocks.split(mask)

    assert fine.n_data == (~mask).sum() + mask.sum() * 8
    assert fine.is_full()
    assert np.isclose(fine.block_volume.sum(), blocks.block_volume.sum())
    assert sorted(np.unique(fine.level).tolist()) == [0, 1]


def test_splitting_twice_still_tiles():
    blocks = _blockset()
    once = blocks.split(np.asarray(blocks.coordinates)[:, 0] < 60.0)
    twice = once.split(once.level == 1)

    assert twice.is_full()
    assert np.isclose(twice.block_volume.sum(), blocks.block_volume.sum())
    assert sorted(np.unique(twice.level).tolist()) == [0, 2]


def test_a_block_cannot_be_split_past_the_finest_level():
    blocks = _blockset(max_levels=1)
    fine = blocks.split(np.ones(blocks.n_data, dtype=bool))

    assert np.all(fine.level == fine.max_levels)
    with pytest.raises(ValueError, match="finest level"):
        fine.split(np.ones(fine.n_data, dtype=bool))


def test_the_mask_may_be_indices_and_has_to_fit():
    blocks = _blockset()
    by_index = blocks.split([0, 3])
    by_mask = np.zeros(blocks.n_data, dtype=bool)
    by_mask[[0, 3]] = True

    assert by_index.n_data == blocks.split(by_mask).n_data
    with pytest.raises(ValueError, match="one value per block"):
        blocks.split(np.ones(blocks.n_data + 1, dtype=bool))


def test_splitting_does_not_carry_the_predictions_over():
    """A value belongs to the support it was predicted on. Handing a parent's
    to its children would manufacture eight blocks that agree exactly, which
    is what refining is meant to find out rather than assume."""
    blocks = _blockset()
    blocks.add_continuous_variable("y", np.zeros(blocks.n_data))
    fine = blocks.split([0])

    assert list(blocks.variables) == ["y"]
    assert list(fine.variables) == []


# --------------------------------------------------------------------------- #
# what the model sees
# --------------------------------------------------------------------------- #
def test_every_block_fans_out_to_the_same_number_of_rows():
    """The reason nothing downstream needs to know about levels: `_aggregate`
    reshapes rather than splits, so a batch has to be rectangular."""
    blocks = _blockset()
    fine = blocks.split([0])

    assert fine.rows_per_location == 8
    coords, splits = fine.get_batched_coordinates(np.arange(fine.n_data))
    assert splits == fine.n_data
    assert coords.shape == (fine.n_data * 8, 3)

    # a batch mixing levels is still rectangular
    mixed = np.array([0, fine.n_data - 1])
    coords, splits = fine.get_batched_coordinates(mixed)
    assert splits == 2 and coords.shape == (16, 3)


def test_the_sub_blocks_scale_with_each_block():
    blocks = _blockset()
    fine = blocks.split([0])
    coords, _ = fine.get_batched_coordinates(np.arange(fine.n_data))
    coords = coords.reshape(fine.n_data, 8, 3)

    # a sub-block's offset from its centre is half its block, halved again
    # because 2x2x2 puts them at the quarter points
    spread = coords.max(axis=1) - coords.min(axis=1)
    assert np.allclose(spread, fine.block_size / 2)


def test_an_unrefined_set_matches_the_equivalent_blocks3d():
    """Nothing new happens until something is refined."""
    blocks = _blockset()
    reference = geoml.data.Blocks3D(start=START, n=N, step=STEP,
                                    discretization=[2, 2, 2])
    order = np.lexsort(np.asarray(blocks.coordinates).T)
    ref_order = np.lexsort(np.asarray(reference.coordinates).T)

    assert np.allclose(np.asarray(blocks.coordinates)[order],
                       np.asarray(reference.coordinates)[ref_order])
    assert np.allclose(
        blocks.get_batched_coordinates(order[:4])[0],
        reference.get_batched_coordinates(ref_order[:4])[0])


def test_prediction_matches_a_blocks3d_and_ignores_the_batching():
    model = _model()
    blocks = _blockset()
    reference = geoml.data.Blocks3D(start=START, n=N, step=STEP,
                                    discretization=[2, 2, 2])
    order = np.lexsort(np.asarray(blocks.coordinates).T)
    ref_order = np.lexsort(np.asarray(reference.coordinates).T)

    model.predict(blocks, n_sim=6)
    model.predict(reference, n_sim=6)
    assert np.allclose(
        blocks.variables["y"].prediction.values.to_numpy()[order],
        reference.variables["y"].prediction.values.to_numpy()[ref_order])

    fine = blocks.split(np.asarray(blocks.coordinates)[:, 0] < 60.0)
    model.options.prediction_batch_size = 7
    model.predict(fine, n_sim=6)
    small = fine.variables["y"].prediction.values.to_numpy().copy()
    model.options.prediction_batch_size = 10 ** 9
    model.predict(fine, n_sim=6)
    assert np.allclose(small,
                       fine.variables["y"].prediction.values.to_numpy())


def test_a_refined_block_disperses_less_than_its_parent():
    """A smaller block holds less of the field, so there is less of it to
    vary -- Krige's relation, which is why a coarse answer is a different
    answer and not a worse one."""
    model = _model()
    blocks = _blockset()
    model.predict(blocks, n_sim=12, include_noise=None)
    coarse = np.nanmean(blocks.variables["y"].dispersion.values.to_numpy())

    fine = blocks.split(np.ones(blocks.n_data, dtype=bool))
    model.predict(fine, n_sim=12, include_noise=None)
    assert np.nanmean(
        fine.variables["y"].dispersion.values.to_numpy()) < coarse


# --------------------------------------------------------------------------- #
# carrying it around
# --------------------------------------------------------------------------- #
def test_the_block_size_reaches_the_data_frame():
    blocks = _blockset().split([0])
    df = blocks.as_data_frame()
    assert [c for c in df.columns if c.startswith("_")] == ["_X", "_Y", "_Z"]
    assert np.allclose(df[["_X", "_Y", "_Z"]].values, blocks.block_size)


def test_it_survives_a_zarr_round_trip(tmp_path):
    model = _model()
    blocks = _blockset().split([0, 5])
    model.predict(blocks, n_sim=5)
    blocks.add_metadata("domain", np.arange(blocks.n_data) % 3)

    path = str(tmp_path / "bs.zarr")
    blocks.to_zarr(path)
    back = geoml.data.BlockSet3D.open(path)

    assert isinstance(back, geoml.data.BlockSet3D)
    assert back.n_data == blocks.n_data
    assert back.max_levels == blocks.max_levels
    assert back.discretization == blocks.discretization
    assert np.allclose(back.block_size, blocks.block_size)
    assert np.allclose(np.asarray(back.coordinates),
                       np.asarray(blocks.coordinates))
    assert np.all(back.level == blocks.level)
    assert back.is_full()
    assert np.allclose(back.variables["y"].prediction.values.to_numpy(),
                       blocks.variables["y"].prediction.values.to_numpy())
    assert np.all(back.get_metadata("domain")
                  == blocks.get_metadata("domain"))
    # and it can still be predicted into and refined afterwards
    assert back.split([0]).is_full()


def test_it_refuses_a_lattice_it_cannot_count_in():
    with pytest.raises(ValueError, match="max_levels"):
        geoml.data.BlockSet3D(START, N, STEP, max_levels=-1)
    with pytest.raises(ValueError, match="three positive numbers"):
        geoml.data.BlockSet3D(START, N, STEP, discretization=(2, 2))
    with pytest.raises(ValueError, match="three numbers each"):
        geoml.data.BlockSet3D([0, 0], N, STEP)
