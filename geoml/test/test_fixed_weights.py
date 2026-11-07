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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Experts weighted by a fixed reference coverage
(`GPOptions(expert_weights="coverage")`), tempered weights and the
balanced blend: the weights a function of the location alone, the same
under any set of experts holding a location's own, the blends' moments,
and a prediction computing each group of locations with its own experts
giving what computing every expert everywhere gives."""
import dataclasses

import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.latent.network as _net


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, (n, 2))
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 15) + np.cos(x[:, 1] / 20)
        + 0.1 * rng.normal(size=n))
    return data


def _model(deep=False, n_experts=4, data=None, **options):
    """A GP on four experts, or a GP reading the coordinates and a GP."""
    geoml.set_seed(1234)
    data = _data() if data is None else data
    ip = geoml.data.inducing.from_kmeans(data, 80, seed=0)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.experts(ip, n_experts, seed=0),
        transform=geoml.transform.Isotropic(20.0))
    if deep:
        first = geoml.latent.BasicGP(root, size=1)
        leaf = geoml.latent.BasicGP(geoml.latent.Concatenate(root, first),
                                    size=1)
    else:
        leaf = geoml.latent.BasicGP(root, size=1)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), leaf,
        options=geoml.models.GPOptions(
            **dict(dict(verbose=False, jit_predict=False,
                        quadrature_precision="float64"), **options)))
    return model, data


def _grid(n=31):
    return geoml.data.Grid2D(start=[0, 0], end=[100, 100], n=[n, n])


def _weights(root, coords, ids, **setting):
    value = dataclasses.replace(_net._current(), **setting)
    with _net.setting(value):
        return np.asarray(_net._fixed_weights(
            root, tf.constant(coords, tf.float64), ids))


def test_a_model_saved_before_keeps_the_variance_rule():
    # a save replays the options' dict over the class defaults
    options = geoml.models.GPOptions.__new__(geoml.models.GPOptions)
    assert options.expert_weights == "variance"
    assert options.expert_temperature == 1.0
    assert options.expert_blend == "moments"


def test_a_new_model_takes_fixed_weights_wherever_it_can():
    assert geoml.models.GPOptions().expert_weights is None
    model, _ = _model(deep=True)
    assert model._expert_weights() == "coverage"
    model.options.__dict__["propagation"] = "marginal"
    model.options.__dict__["expert_propagation"] = "consensus"
    assert model._expert_weights() == "variance"
    data = _data()
    ip = geoml.data.inducing.from_kmeans(data, 40, seed=0)
    root = geoml.latent.BasicInput(geoml.data.inducing.experts(ip, 2, seed=0))
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)),
        geoml.latent.UncertainInputGP(root),
        options=geoml.models.GPOptions(verbose=False))
    assert model._expert_weights() == "variance"
    model.options.__dict__["expert_blend"] = "balanced"
    with pytest.raises(ValueError, match="balanced"):
        model._expert_weights()


@pytest.mark.parametrize("blend", ["moments", "balanced"])
def test_the_weights_normalize_as_the_blend_asks(blend):
    model, _ = _model()
    root = model.leaves[0].root
    coords = np.asarray(_grid().coordinates)
    w = _weights(root, coords, tuple(range(4)), expert_weights="coverage",
                 expert_blend=blend)
    assert w.shape == (4, len(coords))
    assert np.all(w >= 0)
    total = np.sum(w ** 2 if blend == "balanced" else w, axis=0)
    np.testing.assert_allclose(total, 1.0, rtol=1e-12)
    # most locations need fewer experts than there are
    assert np.mean(w == 0) > 0.3


def test_a_location_is_weighted_alike_by_any_set_holding_its_experts():
    model, _ = _model()
    root = model.leaves[0].root
    coords = np.asarray(_grid().coordinates)
    full = _weights(root, coords, tuple(range(4)), expert_weights="coverage")
    for k in range(len(coords)):
        own = np.flatnonzero(full[:, k] > 0)
        others = [e for e in range(4) if e not in own]
        for extra in ([], others[:1], others):
            ids = tuple(sorted(list(own) + list(extra)))
            part = _weights(root, coords[k:k + 1], ids,
                            expert_weights="coverage")[:, 0]
            np.testing.assert_allclose(part, full[list(ids), k], rtol=1e-12,
                                       atol=0)


def test_far_from_every_expert_the_nearest_takes_the_weight():
    model, _ = _model()
    root = model.leaves[0].root
    far = np.array([[1e5, 1e5], [-1e5, 50.0]])
    w = _weights(root, far, tuple(range(4)), expert_weights="coverage")
    np.testing.assert_array_equal(np.sort(w, axis=0)[-1], 1.0)
    np.testing.assert_array_equal(np.sum(w > 0, axis=0), 1)


def test_the_temperature_sharpens_the_weights():
    model, _ = _model()
    root = model.leaves[0].root
    coords = np.asarray(_grid().coordinates)
    soft = _weights(root, coords, tuple(range(4)), expert_weights="coverage")
    sharp = _weights(root, coords, tuple(range(4)), expert_weights="coverage",
                     expert_temperature=2.0)
    assert np.mean(np.max(sharp, axis=0)) > np.mean(np.max(soft, axis=0))
    # under the variance rule, the raw weights squared
    v = tf.constant(np.random.default_rng(0).uniform(0.01, 0.99, (3, 1, 5)))
    with _net.setting(dataclasses.replace(_net._current(),
                                          expert_temperature=2.0)):
        w = np.asarray(_net._GPNode.get_expert_weights(v))
    raw = ((1 - v.numpy()) / (v.numpy() + 1e-6) + 1e-6) ** 2
    np.testing.assert_allclose(w, raw / raw.sum(axis=0), rtol=1e-12)


def test_fixed_weights_do_not_move_with_training():
    model, data = _model(expert_weights="coverage")
    before = model.expert_weights(data)
    model.train_full(10)
    np.testing.assert_array_equal(model.expert_weights(data), before)
    # the variance rule's do
    model, data = _model(expert_weights="variance")
    before = model.expert_weights(data)
    model.train_full(10)
    assert np.max(np.abs(model.expert_weights(data) - before)) > 1e-3


@pytest.mark.parametrize("blend", ["moments", "balanced"])
def test_the_blend_reports_its_moments(blend):
    model, _ = _model(expert_weights="coverage", expert_blend=blend)
    model.train_full(5)
    leaf = model.leaves[0]
    coords = tf.constant(np.asarray(_grid(11).coordinates), tf.float64)
    with model._propagation():
        _net.refresh_cached(model.leaves, model.options.jitter, owner=model)
        mean, var = leaf.propagate(coords)[:2]
        x, x_var = leaf.parent.propagate(coords)[:2]
        _, mu, v, _ = leaf._expert_moments(x, x_var)
        w = np.asarray(_net._fixed_weights(leaf.root, coords,
                                           tuple(range(4))))
    mu = np.stack([np.asarray(m)[0, :, 0] for m in mu])
    v = np.asarray(v)[:, 0, :]
    np.testing.assert_allclose(np.asarray(mean)[:, 0],
                               np.sum(w * mu, axis=0), rtol=1e-10, atol=1e-12)
    held = w ** 2 if blend == "balanced" else w
    np.testing.assert_allclose(np.asarray(var)[:, 0], np.sum(held * v, axis=0),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("options", [
    dict(expert_weights="coverage"),
    dict(expert_weights="coverage", expert_temperature=2.0, deep=True),
    dict(expert_weights="coverage", expert_blend="balanced")])
def test_a_prediction_does_not_depend_on_its_batches(options):
    model, _ = _model(**options)
    model.train_full(5)
    whole, parts = _grid(), _grid()
    model.predict(whole, n_sim=4)
    model.options.prediction_batch_size = 97
    model.predict(parts, n_sim=4)
    for column in ("prediction", "latent_mean", "latent_variance"):
        np.testing.assert_allclose(
            np.asarray(parts.values("v/" + column)),
            np.asarray(whole.values("v/" + column)), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(parts.get("v").simulations[:, :]),
        np.asarray(whole.get("v").simulations[:, :]), rtol=1e-8, atol=1e-10)


def test_the_balanced_realizations_carry_the_reported_variance():
    model, _ = _model(expert_weights="coverage", expert_blend="balanced")
    model.train_full(20)
    coords = tf.constant(np.asarray(_grid(11).coordinates), tf.float64)
    with model._propagation():
        _net.refresh_cached(model.leaves, model.options.jitter, owner=model)
        with _net.once_per_pass():
            _, var, sims, _ = model.leaves[0].predict(coords, n_sim=4000,
                                                      seed=[5, 0])
    ratio = np.var(np.asarray(sims)[0], axis=1) / np.asarray(var)[0]
    assert abs(np.median(ratio) - 1) < 0.05


def test_the_options_survive_a_save(tmp_path):
    model, _ = _model(expert_weights="coverage", expert_temperature=2.0,
                      expert_blend="balanced")
    model.train_full(5)
    a = _grid(11)
    model.predict(a, n_sim=3)
    model.save(str(tmp_path / "m.zarr"))
    loaded = geoml.models.VGPNetwork.open(str(tmp_path / "m.zarr"))
    assert loaded.options.expert_weights == "coverage"
    assert loaded.options.expert_temperature == 2.0
    assert loaded.options.expert_blend == "balanced"
    b = _grid(11)
    loaded.predict(b, n_sim=3)
    np.testing.assert_allclose(np.asarray(b.values("v/prediction")),
                               np.asarray(a.values("v/prediction")),
                               rtol=1e-10)


def test_what_fixed_weights_cannot_weight_is_refused():
    with pytest.raises(ValueError, match="balanced"):
        _model(deep=True, expert_weights="coverage", expert_blend="balanced")
    with pytest.raises(ValueError, match="fixed weights"):
        geoml.models.GPOptions(expert_weights="variance",
                               expert_blend="balanced")
    with pytest.raises(ValueError, match="joint"):
        geoml.models.GPOptions(expert_weights="coverage",
                               propagation="marginal",
                               expert_propagation="consensus")
    with pytest.raises(ValueError, match="positive"):
        geoml.models.GPOptions(expert_temperature=0.0)
    data = _data()
    ip = geoml.data.inducing.from_kmeans(data, 40, seed=0)
    root = geoml.latent.BasicInput(geoml.data.inducing.experts(ip, 2, seed=0))
    with pytest.raises(ValueError, match="variance"):
        geoml.models.VGPNetwork(
            data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)),
            geoml.latent.UncertainInputGP(root),
            options=geoml.models.GPOptions(verbose=False,
                                           expert_weights="coverage"))


def test_training_by_expert_reads_the_fixed_weights(monkeypatch):
    model, data = _model(expert_weights="coverage")

    def no_sweep(*args, **kwargs):
        raise AssertionError("fixed weights need no sweep")

    monkeypatch.setattr(model, "_weight_sweep", no_sweep)
    table = model.expert_weights()
    np.testing.assert_allclose(table, model._fixed_table(data), rtol=0)
    model.train_by_expert(1)
    assert np.isfinite(model.training_log[-1])


@pytest.mark.parametrize("options", [
    dict(expert_weights="coverage"),
    dict(expert_weights="coverage", expert_temperature=2.0, deep=True),
    dict(expert_weights="coverage", expert_blend="balanced")])
def test_a_prediction_by_group_is_the_prediction_by_every_expert(options):
    model, _ = _model(**options)
    model.train_full(5)
    grouped, everywhere = _grid(), _grid()
    info = model.predict_by_expert(grouped, n_sim=4)
    assert np.nanmax(info["left_out"]) < 1e-12
    # a location needs fewer experts than there are
    assert np.mean(np.sum(model.expert_weights(grouped) > 0, axis=1)) < 3
    model._predict(everywhere, 4, True, None)
    for column in ("prediction", "latent_mean", "latent_variance"):
        np.testing.assert_allclose(
            np.asarray(grouped.values("v/" + column)),
            np.asarray(everywhere.values("v/" + column)), rtol=1e-9,
            atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(grouped.get("v").simulations[:, :]),
        np.asarray(everywhere.get("v").simulations[:, :]), rtol=1e-8,
        atol=1e-10)


def test_predict_computes_each_group_with_its_own_experts_where_dear(
        monkeypatch):
    model, _ = _model(deep=True, expert_weights="coverage")
    model.train_full(5)
    calls = []
    original = model._predict_by_expert

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_predict_by_expert", spy)
    grid = _grid()
    model.predict(grid, n_sim=3)
    assert calls
    assert np.all(np.isfinite(np.asarray(grid.values("v/prediction"))))
    # one expert has nothing to group
    for quiet in (_model(deep=True, n_experts=1, expert_weights="coverage"),
                  _model(expert_weights="coverage")):
        monkeypatch.setattr(quiet[0], "_predict_by_expert", spy)
        calls.clear()
        quiet[0].predict(_grid(), n_sim=3)
        assert not calls


@pytest.mark.parametrize("sampling", ["partition", "assignment"])
def test_an_epoch_by_expert_adds_up_to_the_bound(sampling):
    # at fixed parameters each row counts once and each expert's KL once,
    # the experts of zero weight on a row never reached; a partition fills
    # an expert's quota from rows it has no weight for where it runs short
    model, data = _model(expert_weights="coverage")
    model.options.training_batch_size = 100
    model.train_full(5)
    for parameter in model._all_parameters:
        parameter.fix()
    rows = np.arange(data.n_data)
    with model._propagation():
        model._refresh(model.options.jitter)
        log_lik = float(model._data_log_lik(
            tf.constant(data.coordinates, tf.float64),
            tf.constant(model.y, tf.float64),
            tf.constant(model.has_value, tf.float64),
            [{} for _ in model.variables],
            x_var=tf.constant(data.get_batched_variance(rows)[0], tf.float64),
            samples=model.options.training_samples, seed=model.options.seed))
        kl = float(tf.add_n([n.kl_divergence() for n in model._nodes()]))
    full = log_lik - kl + float(model.log_prior())
    record = model.train_by_expert(1, coverage=1.0, sampling=sampling)
    np.testing.assert_allclose(record["bound"][0], full, rtol=1e-9)


def test_measurements_and_nodes_by_expert_are_the_plain_ones():
    model, data = _model(deep=True, expert_weights="coverage")
    model.train_full(5)
    target = data[np.arange(60)]
    plain = model.predict_measurements(target, n_sim=4, n_nodes=4)["v"]
    grouped = model.predict_measurements(target, n_sim=4, n_nodes=4,
                                         by_expert=True)["v"]
    np.testing.assert_allclose(grouped, plain, rtol=1e-8, atol=1e-10)
    node = model.leaves[0].parent.parents[1]
    a, b = _grid(11), _grid(11)
    model.predict_node(node, a, n_sim=3, name="n")
    model.predict_node(node, b, n_sim=3, name="n", by_expert=True)

    def sims(container):
        return np.stack([np.asarray(part.simulations) for part
                         in container.variables["n"].components.values()])
    np.testing.assert_allclose(b.values("n/0/latent_mean"),
                               a.values("n/0/latent_mean"), rtol=1e-9,
                               atol=1e-12)
    np.testing.assert_allclose(sims(b), sims(a), rtol=1e-8, atol=1e-10)
