"""Models and latent networks as Graphviz diagrams.

`str` lays a network out as an indented tree, which repeats a node shared by
several branches and cannot show that it is one node. The diagram can, so these
tests check the graph the DOT describes -- which boxes exist, what colour they
are and which way the arrows run -- rather than the text itself. Graphviz is
not needed to write a diagram and so is not needed to test one; the one test
that renders skips when it is missing.
"""
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.latent as latent
import geoml.warping as warping


COLS = ["X", "Y", "Z"]


def _boxes(dot):
    """label -> (id, attributes), for every box outside the legend.

    A circle is keyed by both its glyph and the name beside it.
    """
    found = {}
    for line in dot.splitlines():
        match = re.match(r'\s+(n\d+) \[(.*)\];', line)
        if match is None:
            continue
        for key in re.findall(r'x?label="(.*?)"', match.group(2)):
            found[key] = (match.group(1), match.group(2))
    return found


def _arrows(dot):
    """(tail, head, label) for every arrow."""
    return [(m.group(1), m.group(2), m.group(3)) for m in
            re.finditer(r'\s+(n\d+) -> (n\d+) \[label=" (.*?)".*?\];', dot)]


def _double(dot):
    """The (tail, head) pairs drawn with a head at both ends."""
    return {(m.group(1), m.group(2)) for m in
            re.finditer(r'\s+(n\d+) -> (n\d+) \[.*?dir=both.*?\];', dot)}


def _network():
    """One input feeding two branches, joined again -- a diamond."""
    geoml.set_seed(1234)
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3], step=[50, 50, 50])
    root = latent.BasicInput(grid, transform=geoml.transform.Isotropic(50))
    return latent.Concatenate(
        latent.BasicGP(root, size=1, kernel=geoml.kernels.Matern32()),
        latent.BasicGP(root, size=2, kernel=geoml.kernels.Spherical(),
                       name="rough"))


def _points():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, (30, 3))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    composition = rng.uniform(0.1, 1, (30, 4))
    composition = composition / composition.sum(axis=1, keepdims=True)
    point.add_compositional_variable(
        "assay", ["Ag", "Pb", "Zn", "Other"], composition)
    return point


def _model(network=None):
    point = _points()
    likelihood = geoml.likelihood.MultivariateGaussian(
        4, warping=warping.ChainedWarping(warping.CenteredLogRatio(4),
                                          warping.PCA(4, 3)))
    return geoml.models.VGPNetwork(
        point, "assay", likelihood, network or _network(),
        options=geoml.models.GPOptions(verbose=False))


# --------------------------------------------------------------------------- #
# the latent network
# --------------------------------------------------------------------------- #
def test_every_node_becomes_a_box():
    boxes = _boxes(_network().to_dot())
    assert "BasicGP_1\\nMatern32" in boxes
    assert "rough\\nBasicGP (Spherical)" in boxes
    assert "Concatenate_1" in boxes


def test_a_shared_node_is_drawn_once():
    """What the printed tree cannot say: both branches read the same input."""
    network = _network()
    dot = network.to_dot()

    root_id = _boxes(dot)["BasicInput_1"][0]
    assert str(network).count("BasicInput_1") == 2        # printed twice
    assert len(re.findall(r"\b%s \[label" % root_id, dot)) == 1
    assert len([a for a in _arrows(dot) if a[0] == root_id]) == 2


def test_an_arrow_carries_the_number_of_variables():
    dot = _network().to_dot()
    boxes = _boxes(dot)
    sizes = {(a[0], a[1]): a[2] for a in _arrows(dot)}

    matern = boxes["BasicGP_1\\nMatern32"][0]
    rough = boxes["rough\\nBasicGP (Spherical)"][0]
    joined = boxes["Concatenate_1"][0]
    assert sizes[(matern, joined)] == "1"
    assert sizes[(rough, joined)] == "2"


def test_a_concatenation_is_a_circle_not_a_box():
    dot = _network().to_dot()
    assert '[label="+", shape=circle' in dot


def test_the_root_is_the_input_colour_and_names_the_coordinates():
    dot = _model().to_dot()
    box = _boxes(dot)["BasicInput_1\\nX, Y, Z"]
    assert geoml.viz.graphviz.PALETTE["input"] in box[1]


def test_a_line_break_is_written_the_way_dot_reads_it():
    """A label built with a literal backslash-n prints as one, and the box
    comes out on a single line."""
    dot = _model().to_dot()
    assert "\\\\n" not in dot
    assert "\\n" in dot


def test_a_name_of_the_users_own_is_escaped():
    geoml.set_seed(1234)
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[3, 3, 3], step=[50, 50, 50])
    root = latent.BasicInput(grid)
    node = latent.BasicGP(root, size=1, name='the "good" one')

    assert r'the \"good\" one' in node.to_dot()


# --------------------------------------------------------------------------- #
# the model around it
# --------------------------------------------------------------------------- #
def test_the_warpings_are_drawn_the_way_the_model_generates():
    """Forward, a warping reads data into the latent space; the diagram runs
    the other way, so the chain is reversed and the sizes go with it."""
    dot = _model().to_dot()
    boxes, arrows = _boxes(dot), _arrows(dot)
    sizes = {(a[0], a[1]): a[2] for a in arrows}

    joined = boxes["Concatenate_1"][0]
    pca = boxes["PCA"][0]
    clr = boxes["CenteredLogRatio"][0]
    variable = boxes["assay\\nMultivariateGaussian"][0]

    assert sizes[(joined, pca)] == "3"          # what the network puts out
    assert sizes[(pca, clr)] == "4"             # PCA opens back up to 4 parts
    assert sizes[(clr, variable)] == "4"
    assert geoml.viz.graphviz.PALETTE["warping"] in boxes["PCA"][1]
    assert geoml.viz.graphviz.PALETTE["output"] in boxes[
        "assay\\nMultivariateGaussian"][1]


def test_a_warping_is_drawn_as_a_two_way_step():
    """It reads a value into the latent space and generates one back out, so
    both ends carry a head -- but the step out of the latent network does not,
    since only that direction generates."""
    dot = _model().to_dot()
    boxes, double = _boxes(dot), _double(dot)

    joined = boxes["Concatenate_1"][0]
    pca = boxes["PCA"][0]
    clr = boxes["CenteredLogRatio"][0]
    variable = boxes["assay\\nMultivariateGaussian"][0]

    assert (pca, clr) in double
    assert (clr, variable) in double
    assert (joined, pca) not in double


def test_a_likelihood_with_no_warping_keeps_a_single_arrow():
    """A categorical likelihood carries no warping at all, unlike a continuous
    one, which defaults to a `ZScore` and so does get a box of its own."""
    point = _points()
    point.add_categorical_variable("rock", labels=["a", "b", "c"])
    model = geoml.models.VGPNetwork(
        point, "rock", geoml.likelihood.CategoricalGaussianIndicator(3),
        _network(), options=geoml.models.GPOptions(verbose=False))

    dot = model.to_dot()
    assert "rock\\nCategoricalGaussianIndicator" in _boxes(dot)
    assert _double(dot) == set()


def test_a_warping_that_does_nothing_is_left_out():
    network = _network()
    model = geoml.models.VGPNetwork(
        _model(network).data, "assay",
        geoml.likelihood.MultivariateGaussian(
            4, warping=warping.ChainedWarping(warping.Identity(4),
                                              warping.PCA(4, 3))),
        network, options=geoml.models.GPOptions(verbose=False))

    boxes = _boxes(model.to_dot())
    assert "PCA" in boxes
    assert "Identity" not in boxes


def test_a_network_can_be_drawn_without_a_model():
    """Then there is nothing to say about outputs, so nothing is said."""
    boxes = _boxes(_network().to_dot())
    assert not any(geoml.viz.graphviz.PALETTE["output"] in attributes
                   for _, attributes in boxes.values())


def test_the_legend_can_be_left_out():
    assert "cluster_legend" in _network().to_dot()
    assert "cluster_legend" not in _network().to_dot(legend=False)


# --------------------------------------------------------------------------- #
# the real thing
# --------------------------------------------------------------------------- #
def test_graphviz_accepts_the_result(tmp_path):
    """Everything above reads the DOT the way this package writes it; only
    Graphviz can say whether Graphviz agrees."""
    if shutil.which("dot") is None:
        pytest.skip("graphviz is not installed")

    path = tmp_path / "model.dot"
    path.write_text(_model().to_dot(), encoding="utf-8")
    subprocess.run(["dot", "-Tsvg", "-o", str(tmp_path / "model.svg"),
                    str(path)], check=True, capture_output=True)
    assert (tmp_path / "model.svg").exists()
