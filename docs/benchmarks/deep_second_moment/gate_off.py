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

"""Step 4's gate: with `full_variance` off the branch is 0.9.3.

Four small models -- a shallow GP over two experts, a GP on a GP over two
experts, a GP reading a `GPWalk`, and a `GaussianMixture` (which trains on
realizations) -- each built, trained 20 steps, predicted with
realizations, sampled as measurements and saved. Run under one tree, it
writes an `.npz` per model and the saves; `load` mode opens another tree's
saves and predicts again. Run once on 0.9.3 (`build`), then on the branch
(`build`, which passes `full_variance=False` and `bias=True` where the
tree takes them, and `load` on 0.9.3's saves): every array should be the
same bit for bit. Run single-threaded, the order of a sum otherwise
changing between processes.

Usage: python gate_off.py TREE OUT build|load [SAVES]
"""
import inspect
import os
import sys

import numpy as np

tree, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tree)
import geoml  # noqa: E402

latent, tr = geoml.latent, geoml.transform
NEW = "bias" in inspect.signature(latent.BasicGP.__init__).parameters


def gp(parent, **kwargs):
    if NEW:
        kwargs["bias"] = True
    return latent.BasicGP(parent, **kwargs)


def options(**kwargs):
    if NEW:
        kwargs["full_variance"] = False
    return geoml.models.GPOptions(verbose=False, **kwargs)


def data():
    x = np.random.default_rng(3).uniform(0, 100, [60, 2])
    d = geoml.data.PointData.from_array(x, ["X", "Y"])
    d.add_continuous_variable(
        "v", np.sin(x[:, 0] / 20) + 0.3 * np.cos(x[:, 1] / 15)
        + 0.1 * np.random.default_rng(4).standard_normal(60))
    return d


def build(name):
    geoml.set_seed(7)
    d = data()
    ip = geoml.data.inducing.from_kmeans(d, 24, seed=0)
    joint = dict(propagation="joint", expert_propagation="independent")
    if name == "shallow":
        root = latent.BasicInput(geoml.data.inducing.experts(ip, 2, seed=0),
                                 tr.Isotropic(40))
        return geoml.models.VGPNetwork(
            d, "v", geoml.likelihood.Gaussian(), gp(root, size=1),
            options=options())
    if name == "dgp":
        root = latent.BasicInput(geoml.data.inducing.experts(ip, 2, seed=0),
                                 tr.Isotropic(40))
        leaf = gp(gp(root, size=2), size=1)
        return geoml.models.VGPNetwork(
            d, "v", geoml.likelihood.Gaussian(), leaf,
            options=options(**joint))
    if name == "walk":
        root = latent.BasicInput(ip, tr.Isotropic(40))
        leaf = gp(latent.GPWalk(gp(root, size=2)), size=1, isotropic=True)
        return geoml.models.VGPNetwork(
            d, "v", geoml.likelihood.Gaussian(), leaf,
            options=options(**joint))
    if name == "mixture":
        weights = gp(latent.BasicInput(ip, tr.Isotropic(60)), size=2)
        components = [gp(latent.BasicInput(ip, tr.Isotropic(30)), size=1)
                      for _ in range(2)]
        return geoml.models.VGPNetwork(
            d, "v", geoml.likelihood.Gaussian(),
            latent.GaussianMixture(weights, components), options=options())
    raise ValueError(name)


def answers(model):
    x = np.random.default_rng(5).uniform(0, 100, [40, 2])
    target = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(target, n_sim=8)
    found = {}
    for k in ("prediction", "variance", "latent_mean", "latent_variance",
              "noise_variance"):
        try:
            found[k] = np.asarray(target.values("v/" + k), dtype=float)
        except (KeyError, ValueError):
            pass
    found["simulations"] = np.asarray(
        target.variables["v"].get_simulations(), dtype=float)
    measured = data()
    found["measurements"] = np.asarray(
        model.predict_measurements(measured, n_sim=4)["v"], dtype=float)
    return found


os.makedirs(out, exist_ok=True)
for name in ("shallow", "dgp", "walk", "mixture"):
    if mode == "build":
        model = build(name)
        model.train_full(20)
        found = answers(model)
        found["log"] = np.asarray(model.training_log, dtype=float)
        model.save(os.path.join(out, name + ".geoml"))
    else:
        model = geoml.persistence.load_model(
            os.path.join(sys.argv[4], name + ".geoml"))
        found = answers(model)
    np.savez(os.path.join(out, name + ".npz"), **found)
    print(name, "done", flush=True)
