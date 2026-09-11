"""A block set at its coarsest level, as a regular block model.

`BlockSet3D.as_blocks3d` hands a refined model to anything that reads a
regular grid. A block that was never split is the same block on the same
support and keeps every column exactly. One gathered from finer blocks takes
the volume-weighted mean of the columns that are means over a block's
sub-blocks, and of every realization index by index; it reads the quantiles,
the dispersion and the predicted category again off those, and leaves the
rest missing. Conservation is what the conversion answers to, so most of
what follows is a sum taken two ways.
"""
import numpy as np
import pytest

import geoml


START, N, STEP = [0, 0, 0], [4, 4, 2], [40.0, 40.0, 20.0]


def _model(seed=1234):
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 160, size=[200, 3])
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable(
        "y", np.sin(xyz[:, 0] / 40) + xyz[:, 2] / 100)
    point.variables["y"].set_cutoffs([0.5])

    ip = geoml.data.inducing.from_kmeans(point, 60, seed=0)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(50.0))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "y", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=8))
    model.train_full(max_iter=4)
    return model


def _rock_model(seed=1234):
    """Two rock types either side of a plane, so blocks gather a contact."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 160, size=[350, 3])
    rock = np.where(xyz[:, 0] + 0.4 * xyz[:, 1] < 110, "granite", "schist")

    point = geoml.data.PointData.from_array(xyz)
    point.add_categorical_variable("rock", measurements=rock)

    ip = geoml.data.inducing.from_kmeans(point, 100, seed=0)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(40.0))
    gp = geoml.latent.BasicGP(root, size=2, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "rock", geoml.likelihood.CategoricalGaussianIndicator(2), gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=10))
    model.train_full(max_iter=30)
    return model


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.fixture(scope="module")
def rock_model():
    return _rock_model()


def _refined(cls=geoml.data.BlockSet3D, **angles):
    """Every level present: two coarse blocks split, one child split again."""
    blocks = cls(START, N, STEP, discretization=(2, 2, 2), max_levels=2,
                 **angles)
    fine = blocks.split([0, 5])
    return fine.split(np.flatnonzero(fine.level == 1)[:1])


def _cells(fine, regular):
    """The coarse block each of the set's blocks falls in, found by where it
    sits rather than by the lattice arithmetic under test."""
    return regular._cell_of(
        geoml.data.PointData.from_array(np.asarray(fine.coordinates)))


def _kept(fine, regular):
    """For the coarse blocks that are one of the set's blocks whole: which
    block that is, and the mask of them."""
    at = fine.index_data(
        geoml.data.PointData.from_array(np.asarray(regular.coordinates)))
    return at, fine.level[at] == 0


def _share(fine, regular):
    return fine.block_volume / np.prod(regular.step_size)


# --------------------------------------------------------------------------- #
# what never moved
# --------------------------------------------------------------------------- #
def test_an_unrefined_set_is_the_equivalent_blocks3d(model):
    blocks = geoml.data.BlockSet3D(START, N, STEP, discretization=(2, 2, 2),
                                   max_levels=2)
    model.predict(blocks, n_sim=6)
    regular = blocks.as_blocks3d()

    reference = geoml.data.Blocks3D(start=START, n=N, step=STEP,
                                    discretization=[2, 2, 2])
    assert type(regular) is geoml.data.Blocks3D
    assert regular.discretization == [2, 2, 2]
    assert np.allclose(np.asarray(regular.coordinates),
                       np.asarray(reference.coordinates))

    # every column is the one the set held, only put in the grid's order
    at, whole = _kept(blocks, regular)
    assert np.all(whole)
    for column in ("prediction", "latent_variance", "dispersion",
                   "noise_variance", "proportions/0.5", "divided/0.5"):
        assert np.array_equal(regular.values("y/" + column),
                              blocks.values("y/" + column)[at])
    assert np.array_equal(np.asarray(regular.variables["y"].simulations),
                          np.asarray(blocks.variables["y"].simulations)[at])
    # and it is what a model predicting onto the grid directly says
    model.predict(reference, n_sim=6)
    assert np.allclose(reference.values("y/prediction"),
                       regular.values("y/prediction"))


def test_a_block_never_split_keeps_every_column_exactly(model):
    fine = _refined()
    model.predict(fine, n_sim=6)
    fine.variables["y"].reset_quantiles([0.1, 0.9])
    fine.variables["y"].reset_probabilities([0.5])
    regular = fine.as_blocks3d()

    at, whole = _kept(fine, regular)
    assert 0 < np.count_nonzero(~whole) < regular.n_data
    for column in ("prediction", "latent_mean", "latent_variance",
                   "dispersion", "noise_variance", "quantiles/0.1",
                   "probabilities/0.5", "proportions/0.5", "divided/0.5"):
        assert np.array_equal(regular.values("y/" + column)[whole],
                              fine.values("y/" + column)[at[whole]])
    assert np.array_equal(
        np.asarray(regular.variables["y"].simulations)[whole],
        np.asarray(fine.variables["y"].simulations)[at[whole]])


# --------------------------------------------------------------------------- #
# what is gathered
# --------------------------------------------------------------------------- #
def test_the_prediction_and_every_realization_keep_their_mass(model):
    fine = _refined()
    model.predict(fine, n_sim=6)
    regular = fine.as_blocks3d()
    volume = np.prod(regular.step_size)

    assert np.isclose((regular.values("y/prediction") * volume).sum(),
                      (fine.values("y/prediction") * fine.block_volume).sum(),
                      rtol=1e-12)
    coarse = np.asarray(regular.variables["y"].simulations).sum(axis=0)
    source = (np.asarray(fine.variables["y"].simulations)
              * fine.block_volume[:, None]).sum(axis=0)
    assert np.allclose(coarse * volume, source, rtol=1e-12)
    # a share of a block below the cut-off is a volume, and keeps it too
    assert np.isclose(
        (regular.values("y/proportions/0.5") * volume).sum(),
        (fine.values("y/proportions/0.5") * fine.block_volume).sum(),
        rtol=1e-12)


def test_a_gathered_block_is_the_volume_weighted_mean_of_its_parts(model):
    fine = _refined()
    model.predict(fine, n_sim=6)
    regular = fine.as_blocks3d()

    cell, share = _cells(fine, regular), _share(fine, regular)
    _, whole = _kept(fine, regular)
    sims = np.asarray(fine.variables["y"].simulations)
    for c in np.flatnonzero(~whole):
        parts = cell == c
        assert np.isclose(share[parts].sum(), 1.0)
        for column in ("prediction", "latent_mean", "latent_variance",
                       "noise_variance", "proportions/0.5"):
            assert np.isclose(
                regular.values("y/" + column)[c],
                share[parts] @ fine.values("y/" + column)[parts])
        # realization i of the block is realization i of its parts
        assert np.allclose(
            np.asarray(regular.variables["y"].simulations)[c],
            share[parts] @ sims[parts])


def test_the_dispersion_is_the_variance_of_the_mixture():
    """A block's interior varies as much as its parts do inside themselves,
    plus as much as they differ from one another: the variance of a mixture,
    checked here against the sub-blocks themselves."""
    fine = _refined()
    rng = np.random.default_rng(0)
    k, n_sim = 8, 5
    # the levels sit apart, so the spread between the parts is not nothing
    values = rng.normal(size=(fine.n_data, k, n_sim)) \
        + 2.0 * fine.level[:, None, None]
    fine.add_continuous_variable("g")
    g = fine.variables["g"]
    g.allocate_simulations(n_sim)
    g.simulations[:, :] = values.mean(axis=1)
    g.prediction.values[:] = values.mean(axis=(1, 2))
    g.dispersion.values[:] = values.var(axis=1).mean(axis=1)

    regular = fine.as_blocks3d()
    cell, share = _cells(fine, regular), _share(fine, regular)
    _, whole = _kept(fine, regular)
    for c in np.flatnonzero(~whole):
        parts = cell == c
        weight = np.repeat(share[parts] / k, k)
        inside = values[parts].reshape(-1, n_sim)
        mean = weight @ inside
        expected = np.mean(weight @ (inside - mean) ** 2)
        assert np.isclose(regular.values("g/dispersion")[c], expected,
                          rtol=1e-12)


def test_the_quantiles_are_read_off_the_averaged_realizations(model):
    fine = _refined()
    model.predict(fine, n_sim=6)
    fine.variables["y"].reset_quantiles([0.1, 0.9])
    fine.variables["y"].reset_probabilities([0.5])
    regular = fine.as_blocks3d()

    _, whole = _kept(fine, regular)
    sims = np.asarray(regular.variables["y"].simulations)[~whole]
    assert np.allclose(regular.values("y/quantiles/0.9")[~whole],
                       np.quantile(sims, 0.9, axis=1))
    assert np.allclose(regular.values("y/probabilities/0.5")[~whole],
                       np.mean(sims <= 0.5, axis=1))


def test_what_does_not_follow_from_the_parts_is_missing(model):
    fine = _refined()
    model.predict(fine, n_sim=6)
    # measurements averaged into the blocks are a mean over samples, and the
    # parts do not say how many each held
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-20, 140, size=(400, 3)) * [1.0, 1.0, 0.25]
    samples = geoml.data.PointData.from_array(xyz)
    samples.add_continuous_variable("assay", xyz[:, 0])
    fine.aggregate(samples, metadata=False)
    regular = fine.as_blocks3d()

    at, whole = _kept(fine, regular)
    for path in ("y/divided/0.5", "assay/measurements"):
        assert np.all(np.isnan(regular.values(path)[~whole]))
        assert np.array_equal(regular.values(path)[whole],
                              fine.values(path)[at[whole]], equal_nan=True)


def test_a_block_with_a_part_left_unpredicted_holds_nothing(model):
    fine = _refined()
    finest = fine.level == 2
    model.predict(fine, n_sim=6, where=~finest)
    regular = fine.as_blocks3d()

    cell = _cells(fine, regular)
    empty = np.zeros(regular.n_data, dtype=bool)
    empty[np.unique(cell[finest])] = True
    for path in ("y/prediction", "y/dispersion", "y/proportions/0.5"):
        values = regular.values(path)
        assert np.all(np.isnan(values[empty]))
        assert np.all(np.isfinite(values[~empty]))
    sims = np.asarray(regular.variables["y"].simulations)
    assert np.all(np.isnan(sims[empty]))
    assert np.all(np.isfinite(sims[~empty]))


# --------------------------------------------------------------------------- #
# the other kinds
# --------------------------------------------------------------------------- #
def test_a_category_is_the_winner_of_the_averaged_probabilities(rock_model):
    fine = _refined()
    rock_model.predict(fine, n_sim=4)
    regular = fine.as_blocks3d()

    cell, share = _cells(fine, regular), _share(fine, regular)
    at, whole = _kept(fine, regular)
    labels = list(regular.variables["rock"].labels)
    probability = np.stack(
        [regular.values("rock/%s/probability" % label) for label in labels],
        axis=1)
    source = np.stack(
        [fine.values("rock/%s/probability" % label) for label in labels],
        axis=1)
    for c in np.flatnonzero(~whole):
        parts = cell == c
        assert np.allclose(probability[c], share[parts] @ source[parts])
    assert np.allclose(probability.sum(axis=1), 1.0)

    predicted = regular.values("rock/predicted")
    assert list(predicted[~whole]) == \
        [labels[i] for i in np.argmax(probability[~whole], axis=1)]
    assert np.array_equal(predicted[whole],
                          fine.values("rock/predicted")[at[whole]])
    # each block's ground is still all accounted for between the rocks
    total = sum(regular.values("rock/%s/proportions/0.0" % label)
                for label in labels)
    assert np.allclose(total, 1.0)


def test_a_category_with_a_part_left_unpredicted_is_no_category(rock_model):
    """A category's probability reads 0 where nothing was predicted, so the
    label is what tells the parts that were apart."""
    fine = _refined()
    finest = fine.level == 2
    rock_model.predict(fine, n_sim=4, where=~finest)
    regular = fine.as_blocks3d()

    empty = np.unique(_cells(fine, regular)[finest])
    assert np.all(regular.values("rock/predicted")[empty] == "")
    for label in regular.variables["rock"].labels:
        assert np.all(np.isnan(
            regular.values("rock/%s/probability" % label)[empty]))


def test_a_binary_variable_names_its_class_again_and_drops_its_entropy():
    fine = _refined()
    rng = np.random.default_rng(2)
    probability = rng.uniform(size=fine.n_data)
    fine.add_binary_variable("ore", labels=["ore", "waste"])
    ore = fine.variables["ore"]
    ore.probability.values[:] = probability
    ore.predicted.values[:] = np.where(probability < 0.5, 1, 0)
    ore.entropy.values[:] = rng.uniform(size=fine.n_data)
    ore.allocate_simulations(3)
    ore.simulations[:, :] = rng.uniform(size=(fine.n_data, 3))

    regular = fine.as_blocks3d()
    cell, share = _cells(fine, regular), _share(fine, regular)
    at, whole = _kept(fine, regular)
    coarse = regular.values("ore/probability")
    for c in np.flatnonzero(~whole):
        assert np.isclose(coarse[c], share[cell == c] @ probability[cell == c])
    assert list(regular.values("ore/predicted")[~whole]) == \
        ["waste" if p < 0.5 else "ore" for p in coarse[~whole]]
    # a function of the block's own probability, not a mean over it
    assert np.all(np.isnan(regular.values("ore/entropy")[~whole]))
    assert np.array_equal(regular.values("ore/entropy")[whole],
                          fine.values("ore/entropy")[at[whole]])


def test_the_parts_of_a_composition_still_close():
    fine = _refined()
    rng = np.random.default_rng(3)
    parts = rng.dirichlet([2.0, 3.0, 5.0], size=fine.n_data) * 100.0
    fine.add_compositional_variable("assay", ["a", "b", "c"],
                                    units={"a": "%", "b": "%", "c": "%"})
    assay = fine.variables["assay"]
    for i, label in enumerate(["a", "b", "c"]):
        assay.components[label].prediction.values[:] = parts[:, i]

    regular = fine.as_blocks3d()
    total = sum(regular.values("assay/%s/prediction" % label)
                for label in ["a", "b", "c"])
    assert np.allclose(total, 100.0)
    assert regular.variables["assay"].components["a"].unit == "%"


def test_metadata_is_gathered_by_volume():
    fine = _refined()
    xyz = np.asarray(fine.coordinates)
    fine.add_metadata("x", xyz[:, 0])
    # the first coarse block is halved between two labels by its parts'
    # heights, which is a tie; the one split once leans to "a", six to two
    first = (np.abs(xyz[:, 0]) < 20) & (np.abs(xyz[:, 1]) < 20) \
        & (np.abs(xyz[:, 2]) < 10)
    side = np.where(xyz[:, 2] < 0, "a", "b")
    lean = ((xyz[:, 0] % 40) < 20) | ((xyz[:, 1] % 40) < 20)
    fine.add_metadata("side", np.where(first, side,
                                       np.where(lean, "a", "b")))
    fine.add_metadata("low", xyz[:, 2] < 0)

    regular = fine.as_blocks3d()
    centres = np.asarray(regular.coordinates)
    # parts tiling a block have the block's centre for their mean position
    assert np.allclose(regular.get_metadata("x"), centres[:, 0])
    at, whole = _kept(fine, regular)
    side = regular.get_metadata("side")
    cell = _cells(fine, regular)
    assert side[cell[first][0]] == ""
    others = np.setdiff1d(np.flatnonzero(~whole), cell[first])
    assert list(side[others]) == ["a"] * len(others)
    assert np.array_equal(side[whole], fine.get_metadata("side")[at[whole]])
    # a flag holds where it held throughout, so a block half low is not low
    low = regular.metadata["low"].values.to_numpy()
    assert low.dtype == bool
    assert not np.any(low[~whole])
    assert np.array_equal(low[whole], (xyz[:, 2] < 0)[at[whole]])


# --------------------------------------------------------------------------- #
# turned
# --------------------------------------------------------------------------- #
ANGLES = dict(azimuth=30.0, dip=10.0, rake=5.0)


def test_a_rotated_set_comes_back_turned_the_same_way(model):
    fine = _refined(geoml.data.RotatedBlockSet3D, **ANGLES)
    model.predict(fine, n_sim=6)
    regular = fine.as_blocks3d()

    assert type(regular) is geoml.data.RotatedBlocks3D
    assert (regular.azimuth, regular.dip, regular.rake) == (30.0, 10.0, 5.0)
    # each coarse block sits where the set's own coarsest block does
    coarse = geoml.data.RotatedBlockSet3D(START, N, STEP, discretization=(
        2, 2, 2), max_levels=2, **ANGLES)
    at = coarse.index_data(
        geoml.data.PointData.from_array(np.asarray(regular.coordinates)))
    assert np.allclose(np.asarray(regular.coordinates),
                       np.asarray(coarse.coordinates)[at])

    at, whole = _kept(fine, regular)
    assert np.array_equal(regular.values("y/prediction")[whole],
                          fine.values("y/prediction")[at[whole]])
    volume = np.prod(regular.step_size)
    assert np.isclose((regular.values("y/prediction") * volume).sum(),
                      (fine.values("y/prediction") * fine.block_volume).sum(),
                      rtol=1e-12)


def test_the_rotated_model_predicts_as_the_unrefined_set_does(model):
    """Same blocks, same support, same sub-blocks turned the same way."""
    turned = geoml.data.RotatedBlockSet3D(START, N, STEP, discretization=(
        2, 2, 2), max_levels=0, **ANGLES)
    regular = geoml.data.RotatedBlocks3D(START, N, STEP, discretization=[
        2, 2, 2], **ANGLES)
    model.predict(turned, n_sim=6)
    model.predict(regular, n_sim=6)

    at = turned.index_data(
        geoml.data.PointData.from_array(np.asarray(regular.coordinates)))
    assert np.allclose(regular.values("y/prediction"),
                       turned.values("y/prediction")[at])
    assert np.allclose(regular.values("y/dispersion"),
                       turned.values("y/dispersion")[at])
