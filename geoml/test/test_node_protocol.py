"""The node protocol: propagate stamps, simulate draws, predict pairs them.

Simulations originate at GP nodes and are carried pathwise by the operation
nodes above them, so the moment path must generate no random draws at all --
a `LinearCombination` used to draw one simulation per parent inside its
`propagate` and throw it away, which is the regression the counting test
exists to catch. The other tests pin the protocol's guards: `simulate`
without its paired `propagate` refuses, and `predict` no longer has an
`n_sim=0` shape.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import geoml
import geoml.latent as latent
import geoml.latent.network as network


COLS = ["X", "Y"]


def _points(n=10, seed=42):
    rng = np.random.default_rng(seed)
    return geoml.data.PointData(
        pd.DataFrame(rng.uniform(0, 100, (n, 2)), columns=COLS), COLS)


def _network():
    """A GP whose moments propagate through a LinearCombination."""
    geoml.set_seed(1234)
    root = latent.BasicInput(_points(),
                             transform=geoml.transform.Isotropic(40))
    combined = latent.LinearCombination(
        latent.BasicGP(root, size=2), latent.BasicGP(root, size=2))
    top = latent.BasicGP(combined, size=1)
    top.refresh(1e-6)
    return top


def _locations(n=7):
    rng = np.random.default_rng(0)
    return tf.constant(rng.uniform(0, 100, (n, 2)), tf.float64)


def test_moment_path_draws_no_normals(monkeypatch):
    top = _network()

    calls = []
    original = network._simulation_normals

    def counting(shape, seed):
        calls.append(tuple(shape))
        return original(shape, seed)

    monkeypatch.setattr(network, "_simulation_normals", counting)

    top.propagate(_locations())
    assert calls == []

    top.predict(_locations(), n_sim=3)
    assert len(calls) > 0


def test_simulate_before_propagate_raises():
    geoml.set_seed(1234)
    root = latent.BasicInput(_points(),
                             transform=geoml.transform.Isotropic(40))
    gp = latent.BasicGP(root, size=1)
    gp.refresh(1e-6)

    with pytest.raises(RuntimeError, match="propagate"):
        gp.simulate(3)


def test_zero_simulations_are_refused():
    top = _network()
    with pytest.raises(ValueError, match="n_sim"):
        top.predict(_locations(), n_sim=0)


def test_predict_returns_four_values():
    top = _network()
    out = top.predict(_locations(5), n_sim=2)

    assert len(out) == 4
    mu, var, sims, exp_var = out
    assert tuple(sims.shape) == (1, 5, 2)
    assert tuple(mu.shape) == (1, 5, 1)
    assert tuple(var.shape) == (1, 5)
    assert tuple(exp_var.shape) == (1, 5)
