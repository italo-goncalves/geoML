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


def test_the_base_cell_follows_the_discretization_not_a_factor_of_two():
    """The discretization *is* the refinement ratio: a block splits into its
    own sub-blocks. So the base cell divides by it, per axis."""
    for discretization in ((2, 2, 2), (3, 3, 3), (4, 4, 2), (2, 2, 1)):
        for levels in (0, 1, 2):
            blocks = _blockset(discretization, max_levels=levels)
            ratio = np.array(discretization) ** levels
            assert np.allclose(blocks.base_step, np.array(STEP) / ratio)
            assert np.all(blocks.lattice_shape == np.array(N) * ratio)


def test_a_split_makes_one_child_per_sub_block():
    for discretization in ((2, 2, 2), (3, 3, 3), (4, 4, 2), (2, 2, 1)):
        blocks = _blockset(discretization, max_levels=1)
        fine = blocks.split([0])

        k = int(np.prod(discretization))
        assert fine.n_data == blocks.n_data - 1 + k
        assert fine.is_full()
        assert np.isclose(fine.block_volume.sum(), blocks.block_volume.sum())
        # the child is the parent divided by the discretization, per axis
        child = fine.block_size[fine.level == 1][0]
        assert np.allclose(child, np.array(STEP) / np.array(discretization))


def test_an_axis_given_one_sub_block_is_never_refined():
    """`[2, 2, 1]` refines in plan and leaves the bench height alone."""
    blocks = _blockset((2, 2, 1), max_levels=2)
    fine = blocks.split(np.ones(blocks.n_data, dtype=bool))

    assert np.allclose(fine.block_size[:, 2], STEP[2])
    assert np.allclose(fine.block_size[:, 0], STEP[0] / 2)
    assert fine.is_full()


def test_the_children_land_where_the_sub_blocks_were():
    """Sub-block j and child j are the same corner of the parent, which is
    what lets a coarse prediction speak about the blocks a split would make."""
    blocks = _blockset((2, 2, 2), max_levels=1)
    sub, _ = blocks.get_batched_coordinates([0])

    fine = blocks.split([0])
    children = np.asarray(fine.coordinates)[fine.level == 1]
    assert np.allclose(sub, children)


def test_a_discretization_that_cannot_refine_is_refused():
    with pytest.raises(ValueError, match="cannot refine"):
        geoml.data.BlockSet3D(START, N, STEP, discretization=(1, 1, 1),
                              max_levels=2)
    # ... unless it is not being asked to
    flat = geoml.data.BlockSet3D(START, N, STEP, discretization=(1, 1, 1),
                                 max_levels=0)
    assert flat.rows_per_location == 1
    assert flat.get_batched_coordinates([0])[1] is None


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


def test_a_split_keeps_what_the_unsplit_blocks_already_held():
    """A block that was not split is the same block on the same support, so
    its value still stands. The children start missing -- handing a parent's
    value down would manufacture children agreeing exactly, which is what
    refining is meant to find out rather than assume."""
    model = _model()
    blocks = _blockset()
    model.predict(blocks, n_sim=6)
    before = blocks.variables["y"].prediction.values.to_numpy().copy()
    before_sims = np.asarray(blocks.variables["y"].simulations).copy()

    mask = np.asarray(blocks.coordinates)[:, 0] < 60.0
    fine = blocks.split(mask)
    kept = int((~mask).sum())

    carried = fine.variables["y"].prediction.values.to_numpy()
    assert np.allclose(carried[:kept], before[~mask])
    assert np.allclose(np.asarray(fine.variables["y"].simulations)[:kept],
                       before_sims[~mask])
    assert np.all(np.isnan(carried[kept:]))
    assert fine.unpredicted("y").sum() == fine.n_data - kept


def test_metadata_is_inherited_by_the_children():
    """Unlike a prediction, metadata describes the ground, and a child sits on
    its parent's ground."""
    blocks = _blockset()
    blocks.add_metadata("domain", np.arange(blocks.n_data) % 3)
    mask = np.zeros(blocks.n_data, dtype=bool)
    mask[[1, 4]] = True
    fine = blocks.split(mask)

    kept = int((~mask).sum())
    assert np.all(fine.get_metadata("domain")[:kept]
                  == blocks.get_metadata("domain")[~mask])
    assert np.all(fine.get_metadata("domain")[kept:]
                  == np.repeat(blocks.get_metadata("domain")[mask], 8))


def test_predicting_only_the_new_blocks_gives_the_same_answer():
    """The point of carrying anything over. A location's simulated value does
    not depend on what else is in the batch, so filling in the children alone
    lands exactly where predicting the lot would have."""
    model = _model()
    blocks = _blockset()
    model.predict(blocks, n_sim=6)
    mask = np.asarray(blocks.coordinates)[:, 0] < 60.0

    partial = blocks.split(mask)
    model.predict(partial, n_sim=6, where=partial.unpredicted("y"))

    whole = blocks.split(mask, carry=False)
    model.predict(whole, n_sim=6)

    for role in ("prediction", "dispersion", "latent_mean"):
        assert np.allclose(
            getattr(partial.variables["y"], role).values.to_numpy(),
            getattr(whole.variables["y"], role).values.to_numpy(),
            equal_nan=True)
    assert np.allclose(np.asarray(partial.variables["y"].simulations),
                       np.asarray(whole.variables["y"].simulations))
    assert not np.any(np.isnan(
        partial.variables["y"].prediction.values.to_numpy()))


def test_carry_can_be_turned_off():
    blocks = _blockset()
    blocks.add_continuous_variable("y", np.zeros(blocks.n_data))
    assert list(blocks.split([0], carry=False).variables) == []
    assert list(blocks.split([0]).variables) == ["y"]


def test_predicting_a_subset_needs_the_variable_to_be_there_already():
    model = _model()
    blocks = _blockset()
    with pytest.raises(ValueError, match="on the object already"):
        model.predict(blocks, n_sim=4, where=np.arange(3))


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


# --------------------------------------------------------------------------- #
# reporting on blocks of more than one size
# --------------------------------------------------------------------------- #
def _graded(blocks, values):
    blocks.add_continuous_variable("g")
    blocks.variables["g"].prediction.values[:] = values
    blocks.variables["g"].allocate_simulations(1)
    blocks.variables["g"].simulations[:, :] = np.asarray(values)[:, None]
    return blocks


def test_the_tonnage_counts_every_block_at_its_own_size():
    """Splitting a block into eight and giving each an eighth of the volume
    has to leave the curve where it was."""
    from geoml.plots import prepare

    blocks = _blockset()
    grade = np.linspace(0.2, 4.0, blocks.n_data)
    _graded(blocks, grade)
    cutoffs = np.linspace(0.3, 3.8, 8)
    before = prepare.grade_tonnage(blocks, "g", cutoffs=cutoffs, density=2.7)

    fine = blocks.split(np.ones(blocks.n_data, dtype=bool), carry=False)
    _graded(fine, np.repeat(grade, 8))
    after = prepare.grade_tonnage(fine, "g", cutoffs=cutoffs, density=2.7)

    assert np.allclose(after["tonnage"], before["tonnage"])
    assert np.allclose(after["grade"], before["grade"], equal_nan=True)
    assert after["unit"] == "mass"


def test_a_half_refined_model_reports_the_same_tonnage():
    from geoml.plots import prepare

    blocks = _blockset()
    grade = np.linspace(0.2, 4.0, blocks.n_data)
    _graded(blocks, grade)
    cutoffs = np.linspace(0.3, 3.8, 8)
    before = prepare.grade_tonnage(blocks, "g", cutoffs=cutoffs)

    mask = np.asarray(blocks.coordinates)[:, 0] < 60.0
    mixed = blocks.split(mask, carry=False)
    _graded(mixed, np.concatenate([grade[~mask], np.repeat(grade[mask], 8)]))

    after = prepare.grade_tonnage(mixed, "g", cutoffs=cutoffs)
    assert np.allclose(after["tonnage"], before["tonnage"])
    assert after["unit"] == "volume"


def test_the_export_is_one_welded_hexahedron_per_block():
    blocks = _blockset().split([0])
    _graded(blocks, np.arange(blocks.n_data, dtype=float))
    blocks.add_metadata("domain", np.arange(blocks.n_data) % 2)
    mesh = blocks.as_pyvista()

    assert mesh.n_cells == blocks.n_data
    # welded: eight corners a block would be 8 * n_data points
    assert mesh.n_points < 8 * blocks.n_data
    assert "g - prediction" in mesh.cell_data
    assert "domain" in mesh.cell_data
    assert np.allclose(mesh.cell_data["g - prediction"],
                       np.arange(blocks.n_data))
    assert np.isclose(mesh.volume, blocks.block_volume.sum())


def _sphere(blocks, centre=(80.0, 80.0, 80.0), radius=60.0):
    return 60.0 - np.linalg.norm(
        np.asarray(blocks.coordinates) - np.array(centre), axis=1)


def test_a_contour_over_mixed_blocks_agrees_with_marching_cubes():
    start, n, step = [0, 0, 0], [16, 16, 16], [10.0, 10.0, 10.0]
    blocks = geoml.data.BlockSet3D(start, n, step, max_levels=2)
    _graded(blocks, _sphere(blocks))
    surface = blocks.get_contour("g", 0.0)

    grid = geoml.data.Grid3D(start=start, n=n, step=step)
    grid.add_continuous_variable("f")
    grid.variables["f"].prediction.values[:] = 60.0 - np.linalg.norm(
        np.asarray(grid.coordinates) - np.array([80.0, 80.0, 80.0]), axis=1)
    reference = grid.variables["f"].prediction.get_contour(0.0)

    assert isinstance(surface, geoml.data.Solid3D)
    assert surface.closed
    # two different algorithms over the same field, so close rather than equal
    assert abs(surface.area / reference.area - 1) < 0.05


def test_refining_away_from_a_surface_leaves_it_alone():
    """The blocks the surface passes through are unchanged, so the surface is
    too -- which is the whole argument for carrying the rest coarsely."""
    start, n, step = [0, 0, 0], [16, 16, 16], [10.0, 10.0, 10.0]
    blocks = geoml.data.BlockSet3D(start, n, step, max_levels=2)
    _graded(blocks, _sphere(blocks))
    before = blocks.get_contour("g", 0.0)

    far = np.abs(_sphere(blocks)) > 50.0
    fine = blocks.split(far, carry=False)
    _graded(fine, _sphere(fine))
    after = fine.get_contour("g", 0.0)

    assert fine.n_data > blocks.n_data
    assert np.isclose(after.area, before.area)


def test_a_contour_says_when_there_is_nothing_to_draw():
    blocks = _blockset()
    _graded(blocks, np.linspace(0.0, 1.0, blocks.n_data))

    with pytest.raises(ValueError, match="no surface at"):
        blocks.get_contour("g", 99.0)
    with pytest.raises(ValueError, match="no variable named"):
        blocks.get_contour("nope", 0.5)
    with pytest.raises(ValueError, match="nothing under"):
        blocks.get_contour("g", 0.5, attribute="measurements")


def test_it_refuses_a_lattice_it_cannot_count_in():
    with pytest.raises(ValueError, match="max_levels"):
        geoml.data.BlockSet3D(START, N, STEP, max_levels=-1)
    with pytest.raises(ValueError, match="three positive numbers"):
        geoml.data.BlockSet3D(START, N, STEP, discretization=(2, 2))
    with pytest.raises(ValueError, match="three numbers each"):
        geoml.data.BlockSet3D([0, 0], N, STEP)
