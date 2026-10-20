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

"""A GP node's prior mean: zero unless `bias=True`.

The bias is fixed at zero by default and stays there through training;
`bias=True` trains it; and a model saved with a free bias -- every model
saved before 0.9.4 -- comes back with it free and at the value it had,
since a save restores the fixed flags it was made with.
"""
import numpy as np
import pandas as pd

import geoml

COLS = ["X", "Y"]


def _model(bias=None, nodes=False):
    geoml.set_seed(7)
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 100, (40, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    # a level well away from zero, which a free bias takes up
    point.add_continuous_variable("v", 3.0 + np.sin(coords[:, 0] / 25.0))
    inducing = geoml.data.PointData(
        pd.DataFrame(rng.uniform(0, 100, (8, 2)), columns=COLS), COLS)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.experts(inducing, 2, seed=0) if nodes
        else inducing, transform=geoml.transform.Isotropic(40))
    kwargs = {} if bias is None else dict(bias=bias)
    gp = geoml.latent.BasicGP(root, size=1, **kwargs)
    warping = geoml.warping.Identity(1)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(warping), gp,
        options=geoml.models.GPOptions(verbose=False))
    return model, gp


def _biases(gp):
    return [float(np.asarray(gp.parameters["bias_%d" % i].get_value()))
            for i in range(gp.root.n_experts)]


def test_the_bias_is_fixed_at_zero_by_default():
    model, gp = _model(nodes=True)
    assert all(gp.parameters["bias_%d" % i].fixed
               for i in range(gp.root.n_experts))
    model.train_full(20)
    assert _biases(gp) == [0.0, 0.0]


def test_bias_true_trains_it():
    model, gp = _model(bias=True, nodes=True)
    assert not any(gp.parameters["bias_%d" % i].fixed
                   for i in range(gp.root.n_experts))
    model.train_full(20)
    assert all(b > 0.05 for b in _biases(gp)), _biases(gp)


def test_a_save_made_with_a_free_bias_reloads_free(tmp_path):
    # as a model saved before 0.9.4: built without the argument, its bias
    # free and trained
    model, gp = _model()
    gp.parameters["bias_0"].unfix()
    model.train_full(20)
    trained = _biases(gp)[0]
    assert trained != 0.0
    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)
    node = loaded.leaves[0]
    assert not node.parameters["bias_0"].fixed
    assert _biases(node)[0] == trained
