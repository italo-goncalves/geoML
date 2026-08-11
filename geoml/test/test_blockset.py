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


def test_predicting_a_subset_of_a_fresh_container_leaves_the_rest_missing():
    """`where` on a container that does not carry the variable yet used to be
    refused. It is how a filter works: the variable is created, the named
    locations are predicted, and the rest keep `nan` -- which is what
    `unpredicted` reads, so nothing downstream mistakes them for answers."""
    model = _model()
    blocks = _blockset()
    model.predict(blocks, n_sim=4, where=np.arange(3))

    values = np.asarray(blocks.variables["y"].prediction.values)
    assert np.all(np.isfinite(values[:3]))
    assert np.all(np.isnan(values[3:]))
    assert np.all(blocks.unpredicted("y")[3:])
    assert np.asarray(
        blocks.variables["y"].simulations)[:3].shape == (3, 4)


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
    model.predict(blocks, n_sim=12, include_noise=False)
    coarse = np.nanmean(blocks.variables["y"].dispersion.values.to_numpy())

    fine = blocks.split(np.ones(blocks.n_data, dtype=bool))
    model.predict(fine, n_sim=12, include_noise=False)
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
# deciding what to split
# --------------------------------------------------------------------------- #
def _ore_model(seed=1234):
    """A compact ore body in a mostly barren domain, with a cut-off declared
    on the data -- the shape of problem refinement exists for."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 160, size=[500, 3])
    radius = np.linalg.norm(xyz - np.array([80.0, 80.0, 80.0]), axis=1)

    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable(
        "au", 4.0 * np.exp(-(radius / 28.0) ** 2) + 0.02)
    point.variables["au"].set_cutoffs([1.0])

    ip = geoml.inducing.from_kmeans(point, 150, seed=0)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(22.0))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "au", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False, seed=seed,
                                       training_samples=10))
    model.train_full(max_iter=60)
    return model


def _ore_blocks():
    return geoml.data.BlockSet3D([0, 0, 0], [8, 8, 4], [20.0, 20.0, 40.0],
                                 discretization=(2, 2, 2), max_levels=2)


def test_the_cutoffs_travel_with_the_variable():
    """Declared on the data, and every model predicted from it knows them
    without being told a second time."""
    model = _ore_model()
    blocks = _ore_blocks()
    model.predict(blocks, n_sim=8)

    assert blocks.variables["au"].cutoffs == [1.0]
    assert list(blocks.variables["au"].proportions) == [1.0]
    assert list(blocks.variables["au"].divided) == [1.0]


def test_a_share_of_a_block_is_between_none_and_all_of_it():
    model = _ore_model()
    blocks = _ore_blocks()
    model.predict(blocks, n_sim=8)

    share = blocks.variables["au"].proportions[1.0].values.to_numpy()
    assert np.all((share >= 0.0) & (share <= 1.0))
    # and some blocks are wholly on one side, or the model has resolved
    # nothing and the rest of these tests mean little
    assert np.any(share >= 1.0 - 1e-12)


def test_the_criterion_and_the_prediction_see_one_field():
    """The noise being integrated out rather than drawn, there is no longer a
    noisy field and a clean one to choose between: a block's cut-off shares,
    its dispersion and its simulations are all read from the same numbers, and
    none of them carries a spread that cutting the block could not resolve.

    This model's warping is the default `ZScore`, which is affine, so the
    integral has nothing to correct and turning it off changes nothing at
    all -- `test_noise_integration.py` is where a bending one is measured."""
    model = _ore_model()

    integrated, latent = _ore_blocks(), _ore_blocks()
    model.predict(integrated, n_sim=8)
    model.predict(latent, n_sim=8, include_noise=False)

    for role in ("proportions", "divided"):
        assert np.allclose(
            getattr(integrated.variables["au"], role)[1.0].values.to_numpy(),
            getattr(latent.variables["au"], role)[1.0].values.to_numpy())
    assert np.allclose(integrated.variables["au"].dispersion.values.to_numpy(),
                       latent.variables["au"].dispersion.values.to_numpy())
    assert np.allclose(np.asarray(integrated.variables["au"].simulations),
                       np.asarray(latent.variables["au"].simulations))


def test_being_unsure_is_not_a_reason_to_split():
    """The distinction the whole criterion turns on. Realizations either side
    of a cut-off are the model not knowing, which cutting cannot mend;
    sub-blocks either side *within* one realization are two answers in one
    block, which is exactly what cutting separates."""
    model = _ore_model()
    blocks = _ore_blocks()
    model.predict(blocks, n_sim=8)

    var = blocks.variables["au"]
    share = var.proportions[1.0].values.to_numpy()
    divided = var.divided[1.0].values.to_numpy()

    # The implication runs one way only. A block divided in some realization
    # must have sub-blocks either side of the cut-off, so its share cannot be
    # 0 or 1; but a block can be partly above the cut-off across its
    # realizations while no single realization finds it divided, and that is
    # the model being unsure rather than the block holding two answers.
    straddling = (share > 0.0) & (share < 1.0)
    assert np.all(straddling[divided > 0.0])
    assert np.count_nonzero(divided > 0.0) <= np.count_nonzero(straddling)
    assert np.all(divided[share <= 0.0] == 0.0)
    assert np.all(divided[share >= 1.0] == 0.0)


def test_refinement_goes_where_the_cutoff_is():
    model = _ore_model()
    blocks = _ore_blocks()

    refined = geoml.models.refine(model, blocks, n_sim=8)
    assert refined.is_full()
    assert np.isclose(refined.block_volume.sum(), blocks.block_volume.sum())
    assert not np.any(np.isnan(
        refined.variables["au"].prediction.values.to_numpy()))

    # a good deal smaller than refining the lot
    assert refined.n_data < blocks.n_data * 8 ** 2 / 4

    # and the coarse blocks are the ones far from the cut-off
    grade = refined.variables["au"].prediction.values.to_numpy()
    coarse = refined.level == 0
    assert np.nanmax(np.abs(grade[coarse] - 1.0)) > \
        np.nanmin(np.abs(grade[~coarse] - 1.0))


def test_refining_stops_on_its_own():
    """It is told how many passes by the lattice, not by an argument. Every
    pass takes what it splits one level finer and the criterion never marks a
    block already at `max_levels`, so it runs out within `max_levels` passes
    whatever it is given."""
    model = _ore_model()

    # a model whose variable declares no cut-off has nothing to decide, so
    # the first pass is also the last
    quiet_model, quiet = _model(), _blockset()
    quiet_model.predict(quiet, n_sim=6)
    assert quiet.variables["y"].cutoffs is None
    assert not np.any(quiet.needs_splitting())
    assert geoml.models.refine(
        quiet_model, quiet, n_sim=6).n_data == quiet.n_data

    # and where there is something to cut, it stops at the finest level the
    # lattice allows rather than running on
    blocks = _ore_blocks()
    refined = geoml.models.refine(model, blocks, n_sim=8)
    assert refined.level.max() <= refined.max_levels
    assert not np.any(refined.needs_splitting())


def _vector_model(seed=1234):
    """Two grades modelled together, so the components carry cut-offs of
    their own -- the shape `block_shares` used to guess wrong about."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 160, size=[300, 3])
    radius = np.linalg.norm(xyz - np.array([80.0, 80.0, 80.0]), axis=1)
    core = np.exp(-(radius / 30.0) ** 2)

    point = geoml.data.PointData.from_array(xyz)
    point.add_vector_variable(
        "metals", ["zn", "pb"],
        np.stack([4.0 * core + 0.02, 1.5 * core + 0.01], axis=1))
    point.variables["metals"].components["zn"].set_cutoffs([1.0])
    point.variables["metals"].components["pb"].set_cutoffs([0.5])

    ip = geoml.inducing.from_kmeans(point, 100, seed=0)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(30.0))
    gp = geoml.latent.BasicGP(root, size=2, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "metals", geoml.likelihood.MultivariateGaussian(2), gp,
        options=geoml.models.GPOptions(verbose=False, seed=seed,
                                       training_samples=10))
    model.train_full(max_iter=30)
    return model


def test_a_vector_variables_components_each_bring_their_own_cutoffs():
    """`divided` is an `_Attribute` on a category and an `OrderedDict` on a
    graded variable, so reaching in for it from outside found the wrong thing
    on a vector variable's components. Each variable reports its own now."""
    model = _vector_model()
    blocks = _ore_blocks()
    model.predict(blocks, n_sim=8)

    shares = blocks.block_shares()
    assert sorted(shares) == ["metals pb @ 0.5", "metals zn @ 1"]
    for values in shares.values():
        assert values.shape == (blocks.n_data,)
        assert np.all((values >= 0.0) & (values <= 1.0))

    # and the whole workflow runs on it, doing exactly what the criterion
    # asked -- which is the invariant, where whether this particular field
    # happens to straddle a cut-off is a property of the fixture
    wanted = blocks.needs_splitting()
    refined = geoml.models.refine(model, blocks, n_sim=8)
    assert refined.is_full()
    assert (refined.n_data > blocks.n_data) == bool(np.any(wanted))


def test_only_the_named_variables_get_a_say():
    model = _ore_model()
    blocks = _ore_blocks()
    model.predict(blocks, n_sim=8)

    assert list(blocks.block_shares()) == ["au @ 1"]
    assert blocks.block_shares(split_on="au") == blocks.block_shares()
    with pytest.raises(ValueError, match="no variable named"):
        blocks.needs_splitting(split_on="nope")


def test_a_category_is_scored_against_the_best_of_the_others():
    """The rival is the best of the *others*: the runner-up for whoever wins,
    the winner for everybody else. Two categories that tie are each other's
    rival, so both come out at zero -- which is what puts the contact on the
    zero level set instead of a hair to one side of it."""
    prob = np.array([[0.5, 0.3, 0.2],       # one winner
                     [0.4, 0.4, 0.2],       # two tied
                     [1 / 3, 1 / 3, 1 / 3]])  # all tied
    var = np.ones_like(prob)
    skew = geoml.likelihood.CategoricalGaussianIndicator \
        .entropy_and_indicators(prob, var, var)[2].numpy()

    log_p = np.log(prob + 1e-6)
    expected = np.array(
        [[log_p[i, j] - max(log_p[i, k] for k in range(3) if k != j)
          for j in range(3)] for i in range(3)])
    assert np.allclose(skew, expected)

    assert skew[0, 0] > 0 and np.all(skew[0, 1:] < 0)
    assert np.all(skew[1, :2] == 0.0) and skew[1, 2] < 0
    assert np.all(skew[2] == 0.0)


def _rock_model(seed=1234):
    """Two rock types either side of a plane, so there is a contact to find."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 160, size=[350, 3])
    rock = np.where(xyz[:, 0] + 0.4 * xyz[:, 1] < 110, "granite", "schist")

    point = geoml.data.PointData.from_array(xyz)
    point.add_categorical_variable("rock", measurements=rock)

    ip = geoml.inducing.from_kmeans(point, 100, seed=0)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(40.0))
    gp = geoml.latent.BasicGP(root, size=2, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "rock", geoml.likelihood.CategoricalGaussianIndicator(2), gp,
        options=geoml.models.GPOptions(verbose=False, seed=seed,
                                       training_samples=10))
    model.train_full(max_iter=30)
    return model


def test_a_category_splits_a_block_where_its_boundary_runs():
    """`ind_skew` is a category's log-odds against its best rival, so the
    contact is its zero level set and the criterion is the same one a grade
    gets -- with the cut-off fixed at zero and nothing to declare."""
    model = _rock_model()
    blocks = _ore_blocks()
    model.predict(blocks, n_sim=8)

    variable = blocks.variables["rock"]
    # the labels stay bare -- the zero is an artefact of the log-odds, not a
    # number anyone declared, so `rock granite @ 0` would only read as noise.
    # Pinned here so a later tidy-up does not "make it consistent".
    assert sorted(blocks.block_shares()) == ["rock granite", "rock schist"]

    for component in variable.components.values():
        # the same dict a grade keeps, keyed by the one cut-off a category
        # has: zero on `ind_skew`, whose zero level set the contact is
        share = component.proportions[0.0].values.to_numpy()
        assert np.all((share >= 0.0) & (share <= 1.0))
        assert np.any(share >= 1.0 - 1e-12)      # blocks wholly one rock
    # the shares of the two categories account for the whole block
    total = sum(c.proportions[0.0].values.to_numpy()
                for c in variable.components.values())
    assert np.allclose(total, 1.0)

    assert np.any(blocks.needs_splitting())


def test_a_variable_with_components_survives_being_carried_over():
    """`from_variable` builds the components, so carrying has to fill those
    rather than build a second set that would go nowhere."""
    model = _rock_model()
    blocks = _ore_blocks()
    model.predict(blocks, n_sim=8)

    mask = blocks.needs_splitting()
    fine = blocks.split(mask)
    kept = int((~mask).sum())

    assert sorted(fine.variables["rock"].components) == ["granite", "schist"]
    for label, component in fine.variables["rock"].components.items():
        before = blocks.variables["rock"].components[label]
        assert np.allclose(
            component.probability.values.to_numpy()[:kept],
            before.probability.values.to_numpy()[~mask])

    refined = geoml.models.refine(model, blocks, n_sim=8)
    assert refined.is_full()
    assert refined.n_data < blocks.n_data * 8 ** 2 / 4


def test_the_criterion_never_marks_the_finest_blocks():
    model = _ore_model()
    blocks = geoml.data.BlockSet3D([0, 0, 0], [4, 4, 2], [20.0, 20.0, 40.0],
                                   discretization=(2, 2, 2), max_levels=1)
    refined = geoml.models.refine(model, blocks, n_sim=8)

    assert np.all(refined.level <= refined.max_levels)
    assert not np.any(refined.needs_splitting()[
        refined.level >= refined.max_levels])


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


def test_a_contour_over_blocks_refined_at_the_surface_is_closed():
    """The case the whole feature produces: `refine` cuts exactly the blocks
    that straddle, so the coarse/fine interfaces sit right where the surface
    runs. Contoured naively that tears the surface open along every one of
    them -- a third of the area went missing before the mesh was cut down."""
    start, n, step = [0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0]
    blocks = geoml.data.BlockSet3D(start, n, step, max_levels=2)
    field = _sphere(blocks, centre=(80.0, 80.0, 80.0), radius=55.0)
    _graded(blocks, field)

    # cut precisely what the surface passes through, and nothing more
    straddle = np.abs(field) < np.linalg.norm(blocks.block_size, axis=1) / 2
    for _ in range(2):
        blocks = blocks.split(straddle & (blocks.level < blocks.max_levels),
                              carry=False)
        _graded(blocks, _sphere(blocks, radius=55.0))
        field = _sphere(blocks, radius=55.0)
        straddle = np.abs(field) < np.linalg.norm(blocks.block_size, axis=1) / 2

    assert len(np.unique(blocks.level)) > 1, "no mixed sizes, nothing to test"
    surface = blocks.get_contour("g", 0.0)

    assert isinstance(surface, geoml.data.Solid3D)
    assert surface.closed

    # and it is the surface the finest size throughout would have drawn
    fine = geoml.data.BlockSet3D(start, [k * 4 for k in n],
                                 [s / 4 for s in step], max_levels=0)
    _graded(fine, _sphere(fine, radius=55.0))
    assert abs(surface.area / fine.get_contour("g", 0.0).area - 1) < 0.02


def test_cutting_the_mesh_for_a_contour_keeps_each_block_estimate():
    """A child is its parent's value plus what the corners say about the shape
    across it, and that correction cancels over the children -- so the mesh a
    contour is drawn on holds the same metal as the model it came from."""
    blocks = _blockset(max_levels=2)
    rng = np.random.default_rng(7)
    grade = rng.uniform(0.5, 4.0, blocks.n_data)
    _graded(blocks, grade)

    for supersample in (0, 1):
        origin, size, step, values = blocks._cut_to_contour(
            grade, 2.0, supersample=supersample)
        assert origin is not None

        volume = np.prod(size * step, axis=1)
        assert np.isclose((values * volume).sum(),
                          (grade * blocks.block_volume).sum())
        assert np.isclose(volume.sum(), blocks.block_volume.sum())


def test_supersampling_cuts_the_mesh_below_the_model_but_not_the_model():
    """The lattice a contour is drawn on may go finer than any block that was
    ever predicted -- the reconstruction gets rounder, and it costs nothing but
    mesh."""
    blocks = _blockset(max_levels=1)
    _graded(blocks, _sphere(blocks, centre=(80.0, 80.0, 40.0), radius=45.0))
    finest = blocks.block_size.min()

    plain = blocks._cut_to_contour(
        np.asarray(blocks.variables["g"].prediction.values), 0.0,
        supersample=0)
    deeper = blocks._cut_to_contour(
        np.asarray(blocks.variables["g"].prediction.values), 0.0,
        supersample=1)

    assert np.isclose(np.prod(plain[1] * plain[2], axis=1).min() ** (1 / 3),
                      finest, rtol=0.5)
    # the finer lattice reaches below anything the model carries
    assert (deeper[1] * deeper[2]).min() < finest
    assert len(deeper[0]) > len(plain[0])
    # and the model itself is untouched by either
    assert blocks.n_data == int(np.prod(N))
    assert blocks.get_contour("g", 0.0, supersample=2).area > 0


def _model_blocks(max_levels=2):
    """Eight 20 m blocks an axis, so the box runs -10 to 150."""
    return geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0],
                                 max_levels=max_levels)


def _ramp(z0=40.0, slope=0.4, lo=-30.0, hi=170.0):
    """A tilted plane reaching across the model, so it cuts many blocks."""
    xy = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
    points = np.column_stack([xy, z0 + slope * (xy[:, 0] - lo)])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    return geoml.data.Surface3D(
        points, triangles, geoml.geometry.vertex_normals(points, triangles))


def _body(bounds=(38.0, 102.0, 38.0, 102.0, 38.0, 102.0)):
    """A watertight box whose walls fall *between* sub-block centres. On a
    block boundary nothing would straddle, and `crossed_by` would rightly mark
    nothing at all."""
    import pyvista as pv
    mesh = pv.Box(bounds=bounds).triangulate()
    points = np.asarray(mesh.points, dtype=float)
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    return geoml.data.Solid3D(
        points, triangles, geoml.geometry.vertex_normals(points, triangles))


def test_a_sheet_marks_the_blocks_it_runs_through():
    """The question `needs_splitting` asks of a cut-off, asked of geometry: a
    block the surface passes through holds two answers whatever is predicted
    into it."""
    blocks = _model_blocks()
    sheet = _ramp()
    crossed = blocks.crossed_by(sheet)

    assert crossed.any() and not crossed.all()

    # what it marked really does straddle, and what it left alone does not
    blocks.assign_from_surface(sheet, "side", fraction="below")
    share = np.asarray(blocks.get_metadata("below"))
    assert np.all((share[crossed] > 0) & (share[crossed] < 1))
    assert np.all((share[~crossed] == 0) | (share[~crossed] == 1))

    # and cutting them leaves the model tiling its box exactly
    cut = blocks.split(crossed)
    assert cut.n_data > blocks.n_data
    assert cut.is_full()
    assert np.isclose(cut.block_volume.sum(), blocks.block_volume.sum())


def test_a_body_marks_the_blocks_on_its_wall():
    blocks = _model_blocks(max_levels=1)
    body = _body()
    crossed = blocks.crossed_by(body)

    blocks.assign_from_solid(body, "domain", fraction="inside")
    share = np.asarray(blocks.get_metadata("inside"))
    assert crossed.any()
    assert np.all((share[crossed] > 0) & (share[crossed] < 1))
    # a block wholly inside the body is not on its wall and is left alone
    assert np.any(share == 1.0) and not np.any(crossed & (share == 1.0))


def test_a_mesh_with_no_sides_is_refused():
    blocks = _model_blocks()
    body = _body()
    neither = geoml.data.Mesh3D(np.asarray(body.coordinates),
                                np.asarray(body.triangles),
                                np.asarray(body.normals))
    with pytest.raises(geoml.data.MeshTypeError, match="no sides"):
        blocks.crossed_by(neither)


def test_refining_at_a_mesh_needs_no_model():
    """`crossed_by` is geometry, so the loop against `split` refines a block
    model before anything has been predicted into it."""
    blocks = _model_blocks()
    sheet = _ramp()
    coarse = blocks.n_data

    while True:
        mask = blocks.crossed_by(sheet)
        if not np.any(mask):
            break
        blocks = blocks.split(mask, carry=False)

    assert blocks.n_data > coarse
    assert blocks.is_full()
    assert blocks.level.max() == blocks.max_levels
    # ground away from the sheet is still carried coarsely, which is the point
    assert np.any(blocks.level == 0)
    assert blocks.n_data < (8 * 4) ** 3


def test_refine_never_visits_the_ground_it_was_not_asked_about():
    """A filter is what excludes ground from a model that cannot be subsetted.
    Blocks left out are never predicted at any pass, and never cut either --
    they hold nothing to decide and draw no surface."""
    model = _model()
    model.data.variables["y"].set_cutoffs([0.5])
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0],
                                   max_levels=2)
    wanted = np.asarray(blocks.coordinates)[:, 0] < 80.0

    refined = geoml.models.refine(model, blocks, n_sim=6, where=wanted)
    values = np.asarray(refined.variables["y"].prediction.values)
    inside = np.asarray(refined.coordinates)[:, 0] < 80.0

    assert np.all(np.isnan(values[~inside])), "predicted what it was not asked"
    assert np.any(np.isfinite(values[inside]))
    # the excluded half was never cut, so it is still the coarse blocks it was
    assert np.all(refined.level[~inside] == 0)
    assert refined.is_full()

    # and a filtered model is cheaper than the same model unfiltered
    whole = geoml.models.refine(
        model,
        geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0],
                              max_levels=2),
        n_sim=6)
    assert refined.n_data < whole.n_data


def test_a_location_is_found_in_the_block_that_holds_it():
    """Point location on the lattice: one searchsorted a level, and the
    coordinates of the block found must contain the point."""
    blocks = _model_blocks().split([0, 5, 17])
    blocks = blocks.split(np.flatnonzero(blocks.level == 1)[:4])

    rng = np.random.default_rng(0)
    xyz = rng.uniform(-9.0, 149.0, size=[300, 3])
    found = blocks.index_data(geoml.data.PointData.from_array(xyz))

    assert np.all(found >= 0), "every point is inside the box"
    low = np.asarray(blocks.coordinates) - blocks.block_size / 2
    high = low + blocks.block_size
    assert np.all(xyz >= low[found]) and np.all(xyz < high[found])

    # and outside the box is -1 rather than a wrong answer
    outside = geoml.data.PointData.from_array(
        np.array([[-50.0, 0, 0], [0, 500.0, 0], [0, 0, -200.0]]))
    assert np.all(blocks.index_data(outside) == -1)


def test_measurements_aggregate_into_the_blocks_that_hold_them():
    blocks = _model_blocks().split([0])
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-9.0, 149.0, size=[400, 3])
    points = geoml.data.PointData.from_array(xyz)
    points.add_continuous_variable("g", xyz[:, 0])
    points.add_categorical_variable(
        "rock", np.where(xyz[:, 2] > 70, "waste", "ore"))

    blocks.aggregate(points)

    held = blocks.index_data(points)
    mean = np.asarray(blocks.variables["g"].measurements.values)
    for b in np.unique(held):
        assert np.isclose(mean[b], xyz[held == b, 0].mean())
    # a block nothing fell in has nothing to report
    empty = np.setdiff1d(np.arange(blocks.n_data), held)
    assert np.all(np.isnan(mean[empty]))
    assert "rock" in blocks.variables


def test_group_undoes_split_exactly():
    blocks = _model_blocks()
    _graded(blocks, np.arange(blocks.n_data, dtype=float))
    blocks.add_metadata("domain", np.arange(blocks.n_data) % 3)
    before = blocks.n_data

    cut = blocks.split([0, 1, 2])
    back = cut.group(np.flatnonzero(cut.level == 1))

    assert back.n_data == before
    assert back.is_full()
    assert np.isclose(back.block_volume.sum(), blocks.block_volume.sum())
    assert np.array_equal(np.sort(back.level), np.sort(blocks.level))

    # the three regrouped blocks are on new ground and hold nothing; the rest
    # never moved, so what they held still stands
    values = np.asarray(back.variables["g"].prediction.values)
    assert np.count_nonzero(back.unpredicted()) == 3
    assert np.all(np.isnan(values[back.unpredicted()]))
    assert np.all(np.isfinite(values[~back.unpredicted()]))
    # metadata describes the ground, so it survives the change of support
    assert not np.any(np.isnan(np.asarray(back.get_metadata("domain"),
                                          dtype=float)))


def test_group_refuses_a_family_that_is_not_all_there():
    """A partial family averages over children that are not there, which is
    the mass-conservation error the lattice exists to prevent."""
    cut = _model_blocks().split([0, 1])
    children = np.flatnonzero(cut.level == 1)

    with pytest.raises(ValueError, match="whole families"):
        cut.group(children[:5])
    with pytest.raises(ValueError, match="coarsest level"):
        cut.group(np.flatnonzero(cut.level == 0)[:1])


def test_a_stored_column_can_be_the_filter():
    model = _model()
    blocks = _model_blocks()
    blocks.add_metadata(
        "wanted", np.asarray(blocks.coordinates)[:, 0] < 80.0)

    refined = geoml.models.refine(model, blocks, n_sim=4, where="wanted")
    values = np.asarray(refined.variables["y"].prediction.values)
    outside = np.asarray(refined.coordinates)[:, 0] >= 80.0
    assert np.all(np.isnan(values[outside]))


def test_a_block_model_refuses_to_be_subsetted():
    """It is structurally complete by design. `PointData`'s subsetting would
    hand back a plain `PointData` -- no origin, no level, no size -- which
    looks usable and cannot report a tonnage."""
    blocks = _blockset().split([0])
    _graded(blocks, np.arange(blocks.n_data, dtype=float))

    for call in (lambda: blocks[np.arange(3)],
                 lambda: blocks[blocks.level == 0],
                 lambda: blocks.subset_region([0, 0, 0], [80, 80, 40])):
        with pytest.raises(TypeError, match="structurally complete"):
            call()

    # the ways out the message names all still work
    assert "_X" in blocks.as_data_frame().columns
    assert blocks.as_pyvista().n_cells == blocks.n_data


def _brute_unbalanced(blocks, gap=1):
    """Every base cell painted with its block's level, then a look just outside
    each face. Exact, and unaffordable on anything real -- which is the whole
    reason `unbalanced` does not work this way."""
    lvl = np.full(tuple(blocks.lattice_shape), -1, dtype=np.int16)
    for o, s, l in zip(blocks._origin, blocks._size, blocks.level):
        lvl[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = l
    out = np.zeros(blocks.n_data, dtype=bool)
    for i, (o, s, l) in enumerate(zip(blocks._origin, blocks._size,
                                      blocks.level)):
        best = -1
        for axis in range(3):
            for side in (-1, 1):
                j = o[axis] - 1 if side < 0 else o[axis] + s[axis]
                if j < 0 or j >= blocks.lattice_shape[axis]:
                    continue
                cut = [slice(o[a], o[a] + s[a]) for a in range(3)]
                cut[axis] = slice(j, j + 1)
                best = max(best, int(lvl[tuple(cut)].max()))
        out[i] = (best - l) > gap
    return out & (blocks.level < blocks.max_levels)


@pytest.mark.parametrize("n,discretization,max_levels", [
    ([8, 8, 8], (2, 2, 2), 2),
    ([6, 6, 6], (2, 2, 2), 3),
    ([8, 8, 6], (2, 2, 1), 2),
    ([6, 6, 6], (3, 3, 3), 2),
    ([6, 6, 4], (4, 2, 2), 2),
])
def test_an_uneven_jump_is_found_without_painting_the_lattice(
        n, discretization, max_levels):
    """`unbalanced` asks from the fine side and reads sorted tables, never an
    array the shape of the base lattice. It must find exactly what painting
    that lattice would have found."""
    rng = np.random.default_rng(3)
    blocks = geoml.data.BlockSet3D([0, 0, 0], n, [20.0, 20.0, 20.0],
                                   discretization=discretization,
                                   max_levels=max_levels)
    # refine unevenly, so jumps of every size occur
    for _ in range(max_levels):
        pick = ((np.abs(_sphere(blocks, centre=(60.0, 60.0, 60.0))) < 25.0)
                & (blocks.level < blocks.max_levels)
                & (rng.random(blocks.n_data) < 0.7))
        if not np.any(pick):
            break
        blocks = blocks.split(pick, carry=False)

    assert np.array_equal(blocks.unbalanced(), _brute_unbalanced(blocks))


def test_refining_levels_the_jumps_it_makes():
    """A block whose own sub-blocks agree is never marked by the criterion,
    but a neighbour cut twice over says the field turns sharply nearby."""
    model = _model()
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0],
                                   max_levels=2)
    model.data.variables["y"].set_cutoffs([0.5])
    refined = geoml.models.refine(model, blocks, n_sim=6)

    assert not np.any(refined.unbalanced())
    assert refined.is_full()


def test_a_component_of_a_vector_variable_can_be_contoured():
    """Only the components of a composition hold a grade, so naming one has to
    work -- the variable itself has nothing to draw a surface through."""
    model = _vector_model()
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0],
                                   discretization=(2, 2, 2), max_levels=1)
    model.predict(blocks, n_sim=6)

    grade = blocks.variables["metals"].components["zn"] \
        .prediction.values.to_numpy()
    middle = float(np.nanmean(grade))

    surface = blocks.get_contour("zn", middle)
    assert isinstance(surface, geoml.data.Mesh3D)
    assert len(surface.triangles) > 0

    # and naming the variable it belongs to says which parts to ask for
    with pytest.raises(ValueError, match="made of components"):
        blocks.get_contour("metals", middle)
    with pytest.raises(ValueError, match="no variable or component named"):
        blocks.get_contour("nickel", middle)


def test_a_contour_is_named_by_its_path():
    """The tree path names the column outright -- `metals/zn` the component,
    `metals/zn/prediction` the column itself -- and the bare component name
    stays as sugar over the same tree."""
    model = _vector_model()
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0],
                                   discretization=(2, 2, 2), max_levels=1)
    model.predict(blocks, n_sim=6)
    middle = float(np.nanmean(blocks.values("metals/zn/prediction")))

    by_name = blocks.get_contour("zn", middle)
    by_path = blocks.get_contour("metals/zn", middle)
    by_column = blocks.get_contour("metals/zn/prediction", middle)

    assert np.allclose(by_path.coordinates, by_name.coordinates)
    assert np.allclose(by_column.coordinates, by_name.coordinates)


def test_a_name_in_two_places_needs_its_path():
    """The bare-name sugar only works while the name is unique; a component
    held by two variables has to be reached by its full path."""
    blocks = _blockset()
    blocks.add_vector_variable("a", labels=["zn", "pb"])
    blocks.add_vector_variable("b", labels=["zn", "cu"])

    with pytest.raises(ValueError, match="more than one place"):
        blocks.get_contour("zn", 0.5)
    # the full path never is ambiguous; nothing is written yet, but the
    # complaint must be about that, not about the name
    with pytest.raises(ValueError) as err:
        blocks.get_contour("a/zn/prediction", 0.5)
    assert "more than one place" not in str(err.value)


def test_a_contour_says_when_there_is_nothing_to_draw():
    blocks = _blockset()
    _graded(blocks, np.linspace(0.0, 1.0, blocks.n_data))

    with pytest.raises(ValueError, match="no surface at"):
        blocks.get_contour("g", 99.0)
    with pytest.raises(ValueError, match="no variable or component named"):
        blocks.get_contour("nope", 0.5)
    with pytest.raises(ValueError, match="nothing under"):
        blocks.get_contour("g/measurements", 0.5)


def test_it_refuses_a_lattice_it_cannot_count_in():
    with pytest.raises(ValueError, match="max_levels"):
        geoml.data.BlockSet3D(START, N, STEP, max_levels=-1)
    with pytest.raises(ValueError, match="three positive numbers"):
        geoml.data.BlockSet3D(START, N, STEP, discretization=(2, 2))
    with pytest.raises(ValueError, match="three numbers each"):
        geoml.data.BlockSet3D([0, 0], N, STEP)


def _ball_blocks(centre):
    """A cubic model graded by distance to `centre`, negated so 'above' is
    the inside of a ball."""
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [20.0, 20.0, 20.0],
                                   discretization=(2, 2, 2), max_levels=1)
    radius = np.linalg.norm(np.asarray(blocks.coordinates)
                            - np.asarray([centre]), axis=1)
    return _graded(blocks, -radius)


def test_a_contour_running_out_of_the_model_can_close():
    """A shell reaching the box edge is open there; `close=` mirrors ghost
    cells beyond the boundary so it shuts against the box, as on a grid."""
    blocks = _ball_blocks([10.0, 80.0, 80.0])

    open_shell = blocks.get_contour("g", -50.0)
    closed = blocks.get_contour("g", -50.0, close="above")

    assert not open_shell.closed
    assert isinstance(closed, geoml.data.Solid3D)
    assert closed.volume > 0


def test_closing_leaves_an_interior_shell_alone():
    blocks = _ball_blocks([80.0, 80.0, 80.0])

    plain = blocks.get_contour("g", -50.0)
    closed = blocks.get_contour("g", -50.0, close="above")

    assert isinstance(plain, geoml.data.Solid3D)
    assert np.isclose(closed.volume, plain.volume, rtol=0.01)


def test_a_closed_contour_survives_a_refined_boundary():
    """The ghosts mirror whatever sizes the boundary blocks ended at, so a
    split along the box face must not tear the closing cap."""
    centre = np.array([10.0, 80.0, 80.0])
    blocks = _ball_blocks(centre)

    crossed = blocks.variables["g"].prediction.values.to_numpy() > -50.0
    touched = np.zeros(blocks.n_data, dtype=bool)
    touched[np.where(crossed)[0][:6]] = True
    fine = blocks.split(touched)
    radius = np.linalg.norm(np.asarray(fine.coordinates) - centre[None],
                            axis=1)
    fine.variables["g"].prediction.values[:] = -radius

    closed = fine.get_contour("g", -50.0, close="above")
    assert isinstance(closed, geoml.data.Solid3D)


def test_close_wants_a_side():
    blocks = _blockset()
    _graded(blocks, np.linspace(0.0, 1.0, blocks.n_data))
    with pytest.raises(ValueError, match="close takes"):
        blocks.get_contour("g", 0.5, close="outside")
