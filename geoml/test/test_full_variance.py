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

"""`GPOptions(full_variance=)`: on for a new model, off for an older save.

On, a deep model's realizations carry the variance its moments report; off
they carry what 0.9.3's did, the explained part, and every expert draws
under the node's own name. Bit-for-bit agreement with 0.9.3 itself needs
the 0.9.3 tree beside this one and is checked by
`docs/benchmarks/deep_second_moment/gate_off.py`.
"""
import numpy as np

import geoml
import geoml.latent.network as _net


def _deep_model(full_variance=True):
    geoml.set_seed(7)
    x = np.random.default_rng(3).uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 20) + 0.3 * np.cos(x[:, 1] / 15))
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 24, seed=0),
        geoml.transform.Isotropic(40))
    leaf = geoml.latent.BasicGP(geoml.latent.BasicGP(root, size=2), size=1)
    return geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.Identity(1)),
        leaf,
        options=geoml.models.GPOptions(
            verbose=False, propagation="joint",
            expert_propagation="independent", full_variance=full_variance))


def _variance_ratio(model, n_sim=1000):
    """The realizations' variance over the reported latent variance, the
    median over 30 locations."""
    x = np.random.default_rng(5).uniform(0, 100, [30, 2])
    target = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(target, n_sim=n_sim)
    sims = np.asarray(target.variables["v"].get_simulations(), dtype=float)
    reported = np.asarray(target.values("v/latent_variance"), dtype=float)
    return float(np.median(np.var(sims, axis=1) / reported.ravel()))


def test_a_new_model_takes_it_and_an_older_save_does_not(tmp_path):
    model = _deep_model()
    assert model.options.full_variance and model._full_variance()
    # a save made before 0.9.4 records no such option
    del model.options.__dict__["full_variance"]
    model.save(tmp_path / "old")
    reopened = geoml.persistence.load_model(tmp_path / "old")
    assert not reopened._full_variance()
    assert reopened._propagation_key() != _deep_model()._propagation_key()


def test_off_every_expert_draws_under_the_node_name():
    with _net.propagation_rule("independent", joint=True, full=False):
        assert _net._expert_key("gp", 2) == "gp"
    with _net.propagation_rule("independent", joint=True, full=True):
        assert _net._expert_key("gp", 2) == "gp/expert2"
        assert _net._expert_key("gp", 0) == "gp"


def test_the_realizations_carry_the_reported_variance_only_when_on():
    on, off = _deep_model(True), _deep_model(False)
    for model in (on, off):
        model.train_full(30)
    ratio_on, ratio_off = _variance_ratio(on), _variance_ratio(off)
    print("variance ratio: on %.3f, off %.3f" % (ratio_on, ratio_off))
    assert abs(ratio_on - 1.0) < 0.1
    assert ratio_off < 0.8


def test_a_deep_model_sizes_its_prediction_batches_to_the_budget():
    model = _deep_model(True)
    assert _deep_model(False)._prediction_rows() == 20000
    model.options.quadrature_memory = 1e7
    # four realizations of the reader's pathwise draw: 24 inducing points
    # in two dimensions, 1024 features and the constant, 48 bytes each
    rows = model._prediction_rows()
    assert rows == int(1e7 // (4 * 48 * (24 * 2 + 1025)))
    x = np.random.default_rng(5).uniform(0, 100, [120, 2])
    small = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(small, n_sim=4)
    model.options.quadrature_memory = 1e12
    whole = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(whole, n_sim=4)
    for path in ("v/prediction", "v/latent_variance"):
        np.testing.assert_allclose(small.values(path), whole.values(path),
                                   rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        small.variables["v"].get_simulations(),
        whole.variables["v"].get_simulations(), rtol=1e-8, atol=1e-10)
