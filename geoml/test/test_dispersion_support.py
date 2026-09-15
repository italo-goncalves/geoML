"""The dispersion by block size: blocks merged into their parents.

A `BlockSet3D` holds its blocks at several sizes, and a block's dispersion is
how much the ground varies inside it. Merging blocks into their parents gives
the same number at every size the lattice has, and the merge has to be exact:
a parent's dispersion is the spread of every value its blocks were read at,
which the tests plant by hand and work out directly.
"""
import numpy as np
import pytest

import geoml
import geoml.plots.prepare as prep


def _family(discretization=(2, 2, 2)):
    """One coarsest block, split, and its first child split again: blocks of
    three sizes, with one parent to put together at each coarser one."""
    blocks = geoml.data.BlockSet3D([0, 0, 0], [1, 1, 1], [40.0, 40.0, 20.0],
                                   discretization=discretization,
                                   max_levels=2)
    return blocks.split([0]).split([0])


def _plant(part, points):
    """A block variable's simulations and dispersion, from values planted at
    each block's sub-blocks -- `(n_blocks, n_sub_blocks, n_sim)` -- exactly as
    the likelihood works them out."""
    part.allocate_simulations(points.shape[2])
    part.simulations[:, :] = points.mean(axis=1)
    part.dispersion.values[:] = points.var(axis=1).mean(axis=1)
    part.prediction.values[:] = points.mean(axis=(1, 2))


def _planted(blocks, n_sim=5, seed=0):
    """A variable `g` on `blocks`, planted; the points come back for the
    reference to be worked out from."""
    rng = np.random.default_rng(seed)
    points = 10.0 + rng.normal(size=[
        blocks.n_data, int(np.prod(blocks.discretization)), n_sim]) \
        * (1.0 + blocks.level)[:, None, None]
    blocks.add_continuous_variable("g", np.full(blocks.n_data, np.nan))
    _plant(blocks.variables["g"], points)
    return points


def _reference(blocks, points, members):
    """The spread of every planted value inside a parent, each weighted by
    its block's volume, in each realization and then averaged."""
    weight = blocks.block_volume[members] / blocks.block_volume[members].sum()
    n_sub = points.shape[1]
    values = points[members].reshape(-1, points.shape[2])
    w = np.repeat(weight / n_sub, n_sub)[:, None]
    mean = np.sum(w * values, axis=0)
    return float(np.mean(np.sum(w * (values - mean) ** 2, axis=0)))


@pytest.mark.parametrize("discretization", [(2, 2, 2), (2, 2, 1)])
def test_a_parent_holds_exactly_the_spread_of_every_value_inside_it(
        discretization):
    blocks = _family(discretization)
    points = _planted(blocks)
    (panel,) = prep.dispersion_by_support(blocks, "g")
    finest, middle, coarsest = panel["sizes"]

    # the coarsest size is the one block the set started as, put together
    # from all of its descendants whatever their size
    assert coarsest["count"] == 1
    assert np.isclose(coarsest["deviation"][0] ** 2,
                      _reference(blocks, points, np.arange(blocks.n_data)))

    # one size down, the child that was split sits among its siblings
    split = middle["depth"] == 1
    assert np.count_nonzero(split) == 1
    assert np.isclose(
        middle["deviation"][split][0] ** 2,
        _reference(blocks, points, np.flatnonzero(blocks.level == 2)))

    # and the siblings the refinement left whole keep what they were given
    stored = blocks.variables["g"].dispersion.values.to_numpy()
    assert np.allclose(np.sort(middle["deviation"][~split] ** 2),
                       np.sort(stored[blocks.level == 1]))
    assert np.allclose(np.sort(finest["deviation"] ** 2),
                       np.sort(stored[blocks.level == 2]))


def test_every_size_carries_its_blocks_their_depth_and_their_share():
    blocks = _family()
    _planted(blocks)
    (panel,) = prep.dispersion_by_support(blocks, "g")
    sizes = panel["sizes"]

    assert [entry["level"] for entry in sizes] == [2, 1, 0]
    assert np.allclose(sizes[0]["size"], [10.0, 10.0, 5.0])
    assert np.allclose(sizes[2]["size"], [40.0, 40.0, 20.0])
    assert [entry["count"] for entry in sizes] == [8, 8, 1]
    assert [sorted(entry["depth"].tolist()) for entry in sizes] == \
        [[0] * 8, [0] * 7 + [1], [2]]
    # the finest blocks only cover the ground that was cut that far
    assert np.allclose([entry["share"] for entry in sizes], [1 / 8, 1, 1])
    for entry in sizes:
        assert entry["left_out"] == 0
        assert np.isclose(entry["rms"],
                          np.sqrt(np.mean(entry["deviation"] ** 2)))


def test_ground_never_predicted_leaves_its_parents_out_at_every_size_above():
    """A partial family would be a parent put together from blocks that are
    not there, which `group` refuses and this does too."""
    blocks = _family()
    _planted(blocks)
    missing = np.flatnonzero(blocks.level == 2)[3]
    blocks.variables["g"].dispersion.values[missing] = np.nan
    blocks.variables["g"].simulations[missing, :] = np.nan

    (panel,) = prep.dispersion_by_support(blocks, "g")
    finest, middle, coarsest = panel["sizes"]
    assert (finest["count"], finest["left_out"]) == (7, 1)
    assert (middle["count"], middle["left_out"]) == (7, 1)
    assert (coarsest["count"], coarsest["left_out"]) == (0, 1)
    assert np.isnan(coarsest["rms"])
    assert np.all(middle["depth"] == 0)


def test_the_simulations_are_read_a_band_and_a_slice_at_a_time(tmp_path,
                                                               monkeypatch):
    blocks = _family()
    _planted(blocks, n_sim=6)
    (expected,) = prep.dispersion_by_support(blocks, "g")

    var = blocks.variables["g"]
    sims = np.asarray(var.simulations)
    var.simulations = geoml.storage.ArrayStore.allocate(
        sims.shape, backend="zarr", store=str(tmp_path / "sims"),
        chunks=(4, sims.shape[1]))
    var.simulations[:] = sims

    # a block model's simulations are the one thing a container holds that
    # will not fit in memory, so reading them whole is refused outright
    class Refusing(geoml.storage.ArrayStore):
        def __array__(self, dtype=None, copy=None):
            raise AssertionError("the simulations were read whole")
    var.simulations.__class__ = Refusing

    # and with room for two realizations of the two parents at a time, the
    # six are taken in three passes
    monkeypatch.setattr(geoml.storage, "DEFAULT_THRESHOLD", 8 * 2 * 2)
    (panel,) = prep.dispersion_by_support(blocks, "g")
    for got, want in zip(panel["sizes"], expected["sizes"]):
        assert np.allclose(got["deviation"], want["deviation"])


def test_one_component_can_be_asked_for_by_name():
    blocks = _family()
    rng = np.random.default_rng(1)
    blocks.add_vector_variable("v", ["a", "b"],
                               np.full((blocks.n_data, 2), np.nan))
    for label in ("a", "b"):
        _plant(blocks.variables["v"].components[label],
               rng.normal(size=[blocks.n_data, 8, 4]))

    (panel,) = prep.dispersion_by_support(blocks, "v", component="b")
    assert panel["name"] == "b"
    assert len(prep.dispersion_by_support(blocks, "v")) == 2
    with pytest.raises(KeyError, match="found a, b"):
        prep.dispersion_by_support(blocks, "v", component="z")


def test_it_needs_a_block_set_and_a_dispersion():
    point = geoml.data.PointData.from_array(np.zeros([3, 3]))
    point.add_continuous_variable("g", np.ones(3))
    with pytest.raises(ValueError, match="BlockSet3D"):
        prep.dispersion_by_support(point, "g")

    # a variable nothing was predicted onto has no dispersion to merge
    blocks = _family()
    blocks.add_continuous_variable("g", np.full(blocks.n_data, np.nan))
    with pytest.raises(ValueError, match="carries no dispersion"):
        prep.dispersion_by_support(blocks, "g")


def test_the_labels_say_what_each_size_is():
    assert prep.split_label(0) == "left whole"
    assert prep.split_label(2) == "split twice"
    assert prep.split_label(3) == "split 3 times"
    entry = {"size": np.array([10.0, 10.0, 5.0]), "count": 8, "share": 0.125,
             "left_out": 0}
    assert prep.support_tick(entry) == ["10 × 10 × 5", "8 blocks, 12%"]
    entry["left_out"] = 2
    assert prep.support_tick(entry)[-1] == "2 left out"
    entry["count"], entry["share"] = 1, 1.0
    assert prep.support_tick(entry)[1] == "1 block, 100%"


# --------------------------------------------------------------------------- #
# a block set a model predicted onto
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def predicted():
    """A small model predicted onto a block set refined in one corner: four
    coarsest blocks split once, and one family of their children again."""
    geoml.set_seed(1234)
    rng = np.random.default_rng(1234)
    xyz = rng.uniform(0, 160, size=[200, 3])
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable(
        "y", np.sin(xyz[:, 0] / 40) + xyz[:, 2] / 100)

    ip = geoml.data.inducing.from_kmeans(point, 60, seed=0)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(50.0))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        point, "y", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=8))
    model.train_full(max_iter=4)

    blocks = geoml.data.BlockSet3D([0, 0, 0], [4, 4, 2], [40.0, 40.0, 20.0],
                                   max_levels=2)
    blocks = blocks.split(np.arange(4))
    blocks = blocks.split(np.flatnonzero(blocks.level == 1)[:8])
    model.predict(blocks, n_sim=8)
    return blocks


def test_a_predicted_set_reads_at_every_size(predicted):
    (panel,) = prep.dispersion_by_support(predicted, "y")
    sizes = panel["sizes"]
    assert [entry["count"] for entry in sizes] == [64, 32, 32]
    assert np.isclose(sizes[-1]["share"], 1.0)
    assert sorted(set(sizes[-1]["depth"].tolist())) == [0, 1, 2]
    for entry in sizes:
        assert np.all(np.isfinite(entry["deviation"]))
        assert np.all(entry["deviation"] >= 0)


@pytest.mark.parametrize("kind", ["box", "violin", "jitter"])
def test_both_backends_draw_every_kind(predicted, kind):
    import matplotlib
    matplotlib.use("Agg")
    figure = geoml.plots.Explorer(
        predicted, continuous="y").dispersion_by_support(kind=kind)
    axes = figure.axes[0]
    assert len(axes.get_xticks()) == 3
    assert axes.get_ylabel() == "within-block standard deviation"
    # from zero, so the change with size reads at its scale
    assert axes.get_ylim()[0] == 0.0
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert "root mean square" in labels
    split = {"left whole", "split once", "split twice"}
    # coloured by the depth of splitting as a strip, pooled otherwise
    assert (split <= set(labels)) == (kind == "jitter")
    matplotlib.pyplot.close(figure)

    figure = geoml.plots.Interactive(
        predicted, continuous="y").dispersion_by_support(kind=kind)
    if kind == "jitter":
        # one strip per depth of splitting, and an empty key for each; the
        # line is a scatter too, but of lines
        markers = [trace for trace in figure.data
                   if trace.type == "scatter" and trace.mode == "markers"]
        assert sum(trace.showlegend is False for trace in markers) == 3
        assert sum(trace.showlegend is not False for trace in markers) == 3
    else:
        # one per size
        assert sum(trace.type == kind for trace in figure.data) == 3
    names = {trace.name for trace in figure.data
             if trace.showlegend is not False}
    assert (split <= names) == (kind == "jitter")
    assert figure.layout.yaxis.rangemode == "tozero"
    # a parent is a row of no container, so nothing links to one
    assert all(getattr(trace, "customdata", None) is None
               for trace in figure.data)


def test_a_strip_draws_at_most_about_most_blocks_of_each_size(predicted):
    import matplotlib
    matplotlib.use("Agg")
    figure = geoml.plots.Explorer(
        predicted, continuous="y").dispersion_by_support(kind="jitter",
                                                         most=2)
    points = sum(len(collection.get_offsets())
                 for collection in figure.axes[0].collections)
    assert points <= 3 * 2
    matplotlib.pyplot.close(figure)


def test_the_strip_is_faint_by_default_and_its_key_is_not(predicted):
    import matplotlib
    matplotlib.use("Agg")
    explorer = geoml.plots.Explorer(predicted, continuous="y")
    axes = explorer.dispersion_by_support(kind="jitter").axes[0]
    assert {collection.get_alpha() for collection in axes.collections} == \
        {0.2}
    assert all(handle.get_alpha() == 1.0
               for handle in axes.get_legend().legend_handles)
    axes = explorer.dispersion_by_support(kind="jitter", alpha=0.5).axes[0]
    assert {collection.get_alpha() for collection in axes.collections} == \
        {0.5}
    matplotlib.pyplot.close("all")

    figure = geoml.plots.Interactive(
        predicted, continuous="y").dispersion_by_support(kind="jitter")
    markers = [trace for trace in figure.data
               if trace.type == "scatter" and trace.mode == "markers"]
    assert {trace.marker.opacity for trace in markers
            if trace.showlegend is False} == {0.2}
    assert all(trace.marker.opacity is None for trace in markers
               if trace.showlegend is not False)


def test_only_the_three_kinds_are_accepted(predicted):
    with pytest.raises(ValueError, match="'box' or 'violin' or 'jitter'"):
        geoml.plots.Explorer(
            predicted, continuous="y").dispersion_by_support(kind="hist2d")
