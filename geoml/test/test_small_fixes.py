"""Three small bugs found by reading, each pinned where it was.

Directional data ignored under a variable named as a string; a 1-D
projection that could only point into the positive orthant; a GP node that
failed to build left among its parent's children.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.latent as latent
import geoml.transform as tr


def _points(n=30, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0, 100, (n, 2)), columns=["c0", "c1"])
    point = geoml.data.PointData(df, ["c0", "c1"])
    point.add_continuous_variable("v", np.sin(df["c0"].values / 25.0))
    return point


def test_directional_data_reach_a_variable_named_as_a_string():
    """`variables="v"` used to walk the letters of the name, so the
    directional measurements of "v" were looked up under nothing they could
    match whenever the name was longer than one letter."""
    point = _points()
    point.add_continuous_variable("grade", point.get("v").measurements.values)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"c0": rng.uniform(0, 100, 10),
                       "c1": rng.uniform(0, 100, 10),
                       "dx": np.ones(10), "dy": np.zeros(10)})
    dirs = geoml.data.DirectionalData(df, ["c0", "c1"], ["dx", "dy"])
    dirs.add_continuous_variable("grade", np.full(10, 0.5))

    def build(variables):
        geoml.set_seed(1234)
        root = latent.BasicInput(_points(6, seed=2),
                                 transform=tr.Isotropic(40))
        gp = latent.BasicGP(root, size=1)
        return geoml.models.VGPNetwork(
            point, variables, geoml.likelihood.Gaussian(), gp,
            directional_data=dirs,
            options=geoml.models.GPOptions(verbose=False))

    as_string, as_list = build("grade"), build(["grade"])
    np.testing.assert_array_equal(as_string.y_dir, as_list.y_dir)
    np.testing.assert_array_equal(as_string.y_dir, np.full([10, 1], 0.5))


def test_a_projection_to_1d_can_point_into_any_quadrant():
    projection = tr.ProjectionTo1D(2)
    projection.parameters["directions"].set_value(
        np.array([[-1.0], [1.0]]) / np.sqrt(2))
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        projection(x).numpy(), [[-1 / np.sqrt(2)], [1 / np.sqrt(2)]])


def test_a_gp_node_that_fails_to_build_is_not_left_a_child():
    geoml.set_seed(1234)
    root = latent.BasicInput(_points(6), transform=tr.Isotropic(40))
    product = latent.Multiply(latent.BasicGP(root, size=1),
                              latent.BasicGP(root, size=1))
    with pytest.raises(latent.BrokenPropagationError):
        latent.BasicGP(product, size=1)
    assert product.children == []
