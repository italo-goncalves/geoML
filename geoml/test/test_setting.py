# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The setting a traced graph reads (`network._Setting`) keys every cache
of traced functions: a graph traced under one setting never answers under
another, and coming back to a setting reuses what it traced."""
import dataclasses

import numpy as np
import pytest

import geoml
import geoml.latent.network as _net

# how each part of the setting is changed: an option of the model's, or a
# switch of the module's; the consensus rule takes the marginal one
BASE = {"expert_propagation": {"propagation": "marginal"}}
FLIPS = {
    "qmc": ("option", "qmc_simulations", True),
    "expert_propagation": ("option", "expert_propagation", "consensus"),
    "joint": ("option", "propagation", "marginal"),
    "full_variance": ("option", "full_variance", False),
    "memory": ("option", "quadrature_memory", 1e9),
    "precision": ("option", "quadrature_precision", "float32"),
    "second_moment": ("module", "_UNCERTAIN_POINTS_SECOND_MOMENT", False),
    "pathwise": ("module", "_PATHWISE_REALIZATIONS", False),
    "rule": ("module", "_QUADRATURE_RULE", "sobol"),
    "hermite_points": ("module", "_HERMITE_POINTS", 5),
    "quadrature_nodes": ("module", "_QUADRATURE_NODES", 32),
    "xla": ("module", "_QUADRATURE_XLA", False),
}
# the caches whose calls draw realizations; the rest run with the
# simulation rule of whatever is in force, training's being Monte Carlo
DRAWING = {"predict", "node", "measure"}


def _model(**options):
    """A GP reading the coordinates and the GP below them, two experts."""
    geoml.set_seed(1234)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (120, 2))
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 15) + np.cos(x[:, 1] / 20))
    ip = geoml.data.inducing.from_kmeans(data, 32, seed=0)
    root = geoml.latent.BasicInput(geoml.data.inducing.experts(ip, 2, seed=0),
                                   transform=geoml.transform.Isotropic(20.0))
    first = geoml.latent.BasicGP(root, size=1)
    leaf = geoml.latent.BasicGP(geoml.latent.Concatenate(root, first), size=1)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), leaf,
        options=geoml.models.GPOptions(
            **dict(dict(verbose=False, jit_predict=False,
                        propagation="joint", expert_propagation="independent",
                        quadrature_memory=2e9, quadrature_precision="float64"),
                   **options)))
    return model, data, first


def _handles(model, data, node, target):
    """What every cache holds after its door has been used once under the
    setting in force: the keys of the traced functions it keeps, or the
    trace it holds."""
    model.predict(target, n_sim=2)
    list(model.measurement_batches(data[np.arange(5)], n_sim=2, n_nodes=2))
    model.predict_node(node, target, n_sim=2)
    model.expert_weights(target)
    qmc = bool(model.options.qmc_simulations)
    with model._propagation():
        moments = model._moments_function(False)

    def kinds(kind):
        return frozenset(k for k in model._compiled
                         if isinstance(k, tuple) and k[0] == kind)

    return {
        "predict": frozenset(k for k in model._compiled
                             if isinstance(k[0], bool)),
        "node": kinds("node"),
        "measure": kinds("measure"),
        "moments": moments,
        "refresh": model._refresh_graph[0],
        "weights": frozenset(model._weight_sweeps),
        "step": model._step_rule(),
        "qmc": qmc,
    }


def _flip(model, monkeypatch, field):
    where, name, value = FLIPS[field]
    if where == "option":
        old = model.options.__dict__[name]
        model.options.__dict__[name] = value
        return lambda: model.options.__dict__.__setitem__(name, old)
    old = getattr(_net, name)
    monkeypatch.setattr(_net, name, value)
    return lambda: monkeypatch.setattr(_net, name, old)


def test_every_part_of_the_setting_has_a_flip():
    fields = {f.name for f in dataclasses.fields(_net._Setting)}
    assert set(FLIPS) == fields - {"rows"}


@pytest.mark.parametrize("field", sorted(FLIPS))
def test_every_cache_retraces_when_the_setting_changes(field, monkeypatch):
    model, data, node = _model(**BASE.get(field, {}))
    target = geoml.data.PointData.from_array(
        np.random.default_rng(1).uniform(0, 100, (6, 2)), ["X", "Y"])
    before = _handles(model, data, node, target)

    restore = _flip(model, monkeypatch, field)
    flipped = _handles(model, data, node, target)
    caches = DRAWING if field == "qmc" else set(before) - {"qmc"}
    for cache in caches:
        assert flipped[cache] != before[cache], cache

    # back to the first setting: every cache keeping several traces hands
    # back the one it traced then, tracing nothing new
    restore()
    again = _handles(model, data, node, target)
    for cache in ("predict", "node", "measure", "weights"):
        assert again[cache] == flipped[cache], cache
    assert again["step"] == before["step"]


def test_the_rows_a_call_carries_key_nothing():
    current = _net._current()
    assert dataclasses.replace(current, rows=current.rows + 1) == current
    assert hash(dataclasses.replace(current, rows=7)) == hash(current)
