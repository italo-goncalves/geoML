"""Latent nodes identify themselves by name.

A network used to print as `Add (size 1) / BasicGP (size 1) / BasicGP (size 1)`,
with no way to tell the two branches apart or to refer to either of them. Every
node now carries a name: the one it was built with, or `Class_k` numbered
against the nodes it is connected to. These tests pin where a number comes from,
since the numbering is what the user never states and so never checks.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.latent as latent


COLS = ["X", "Y"]


def _inducing(n=8, seed=99):
    rng = np.random.default_rng(seed)
    return geoml.data.PointData(
        pd.DataFrame(rng.uniform(0, 100, (n, 2)), columns=COLS), COLS)


def _root():
    geoml.set_seed(1234)
    return latent.BasicInput(_inducing(), transform=geoml.transform.Isotropic(40))


def _network():
    """Two branches on one shared input, one of them named."""
    root = _root()
    return latent.Add(
        latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian()),
        latent.BasicGP(root, size=1, kernel=geoml.kernels.Cubic(),
                       name="rough"))


# --------------------------------------------------------------------------- #
# where a name comes from
# --------------------------------------------------------------------------- #
def test_siblings_left_unnamed_are_numbered_apart():
    """The whole point: a node's siblings are not among its ancestors, so this
    only works because the parents keep a list of what was built on them."""
    root = _root()
    first = latent.BasicGP(root, size=1)
    second = latent.BasicGP(root, size=1)

    assert first.name == "BasicGP_1"
    assert second.name == "BasicGP_2"
    assert root.name == "BasicInput_1"


def test_a_name_that_was_given_is_kept():
    network = _network()
    assert network.parents[1].name == "rough"
    assert network.parents[0].name == "BasicGP_1"


def test_the_numbering_is_per_class():
    root = _root()
    gp = latent.BasicGP(root, size=1)
    assert latent.Add(gp, latent.BasicGP(root, size=1)).name == "Add_1"
    assert latent.Bias(gp).name == "Bias_1"


def test_the_numbering_does_not_depend_on_what_was_built_before():
    """Names come from the network, not from a counter that runs all session:
    the same network built twice must read the same both times."""
    first = [node.name for node in _network().get_unique_parents()]
    second = [node.name for node in _network().get_unique_parents()]
    assert sorted(first) == sorted(second)


def test_a_shared_node_is_one_node_under_one_name():
    network = _network()
    text = str(network)

    # the shared input prints under each branch, but it is a single node
    assert text.count("BasicInput_1") == 2
    assert network.get_node("BasicInput_1") is network.parents[0].parent
    assert network.parents[0].parent is network.parents[1].parent


def test_the_tree_names_every_node():
    text = str(_network())
    assert "Add_1 (size 1)" in text
    assert "BasicGP_1 (size 1)" in text
    assert "BasicInput_1 (size 2)" in text
    # a name of the user's own says nothing about the type, so the class stays
    assert "BasicGP 'rough' (size 1)" in text


def test_an_automatic_name_stays_out_of_the_repr():
    """It is already on the node's line in the tree; repeating it there is noise.
    A name the user chose is worth showing -- it is how they built the node."""
    network = _network()
    assert repr(network.parents[0]) == "BasicGP(size=1, kernel=Gaussian())"
    assert repr(network.parents[1]) == \
        "BasicGP(size=1, kernel=Cubic(), name='rough')"


# --------------------------------------------------------------------------- #
# lookup
# --------------------------------------------------------------------------- #
def test_a_node_is_found_by_either_kind_of_name():
    network = _network()
    assert network.get_node("rough") is network.parents[1]
    assert network.get_node("BasicGP_1") is network.parents[0]
    assert network.get_node("Add_1") is network


def test_an_unknown_name_lists_the_ones_there_are():
    with pytest.raises(KeyError, match="BasicInput_1"):
        _network().get_node("smooth")


def test_two_nodes_under_one_name_are_reported():
    """Two subnetworks built apart cannot see each other while they are being
    numbered; joining them is where that shows, so that is where it is said."""
    first, second = _root(), _root()
    assert first.name == second.name == "BasicInput_1"

    joined = latent.Concatenate(first, second)
    with pytest.raises(KeyError, match="name them explicitly"):
        joined.get_node("BasicInput_1")


# --------------------------------------------------------------------------- #
# error messages
# --------------------------------------------------------------------------- #
def test_a_size_mismatch_says_which_nodes():
    root = _root()
    with pytest.raises(latent.SizeIncompatibilityError) as error:
        latent.Add(latent.BasicGP(root, size=1),
                   latent.BasicGP(root, size=2, name="wide"))

    message = str(error.value)
    assert "BasicGP_1 (size 1)" in message
    assert "wide (size 2)" in message
    assert message.startswith("Add_1:")


def test_a_gp_on_a_node_that_cannot_propagate_says_which():
    root = _root()
    product = latent.Multiply(latent.BasicGP(root, size=1),
                              latent.BasicGP(root, size=1))

    with pytest.raises(latent.BrokenPropagationError) as error:
        latent.BasicGP(product, size=1)

    assert "Multiply_1" in str(error.value)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def _model():
    rng = np.random.default_rng(1234)
    coords = rng.uniform(0, 100, (30, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    point.add_continuous_variable("v", np.sin(coords[:, 0] / 25.0))

    return geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), _network(),
        options=geoml.models.GPOptions(verbose=False))


def test_the_names_come_back_from_a_saved_model(tmp_path):
    """An automatic name is not a constructor argument, so nothing writes it
    down: it has to be regenerated by the replay, in the same order."""
    model = _model()
    before = sorted(node.name for node in
                    model.latent_network.get_unique_parents())

    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    after = sorted(node.name for node in
                   loaded.latent_network.get_unique_parents())
    assert after == before
    assert loaded.latent_network.get_node("rough").size == 1
