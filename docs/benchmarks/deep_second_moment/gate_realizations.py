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

"""Gate 2a of the deep second moment: the realizations hold the variance
the moments report.

Three networks -- the folded section's GP on a GP and GP beside the
coordinates, and a walk (a field, a `GPWalk`, an isotropic reader) -- one
expert, trained briefly. At points of new holes, 2000 realizations of the
leaf (ten calls of 200, each its own seed), drawn through the parent's
(`_PATHWISE_REALIZATIONS`) and the 0.9.3 way, against the leaf's reported
mean and variance: the share of points whose realizations' variance falls
outside the 95% chi-square interval about the reported one, the share
whose mean falls outside 1.96 standard errors, and the median ratio of
the variances. Writes `gate_realizations.json` and `.png`.

Usage: python gate_realizations.py [ITERATIONS]
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402

N_SIM, CALLS = 200, 10


def walk_model(data):
    geoml.set_seed(gm.ek.SEED)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, gm.ek.N_IND, seed=gm.ek.SEED))
    field = geoml.latent.BasicGP(root, size=2)
    walk = geoml.latent.GPWalk(field)
    leaf = geoml.latent.BasicGP(walk, size=1, isotropic=True)
    model = geoml.models.VGPNetwork(
        data, "V", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False, propagation="joint",
                                       expert_propagation="independent"))
    for p in model.likelihoods[0].warping.all_parameters:
        p.fix()
    walk.parameters["amp"].set_value(3.0)
    return model


def leaf_draws(model, x, pathwise):
    net._PATHWISE_REALIZATIONS = pathwise
    try:
        leaf = model.leaves[0]
        sims = []
        with model._propagation():
            model._refresh(model.options.jitter)
            for c in range(CALLS):
                with net.once_per_pass():
                    mu, var, s, _ = leaf.predict(tf.constant(x), n_sim=N_SIM,
                                                 seed=[11, c])
                sims.append(np.asarray(s)[0])
        return (np.asarray(mu)[0, :, 0], np.asarray(var)[0],
                np.concatenate(sims, axis=1))
    finally:
        net._PATHWISE_REALIZATIONS = True


def main(iterations):
    nodes = int(os.environ.get("NODES", "64"))
    net._QUADRATURE_NODES = nodes
    net._QUADRATURE.clear()
    suffix = "" if nodes == 64 else "_%d" % nodes
    experts = int(os.environ.get("N_EXPERTS", "1"))
    suffix += "" if experts == 1 else "_experts%d" % experts
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    new, _ = gm.ek.drillholes(gm.ek.N_NEW, gm.ek.NEW_SEED)
    pick = np.random.default_rng(3).choice(new.n_data, gm.N_POINTS,
                                           replace=False)
    x = np.asarray(new.coordinates, dtype=float)[pick]
    n = N_SIM * CALLS
    lo, hi = stats.chi2.ppf([0.025, 0.975], n - 1) / (n - 1)
    rows, panels = [], []
    builds = (("dgp", lambda: gm.build(data, "dgp", kr.Gaussian)),
              ("concat", lambda: gm.build(data, "concat", kr.Gaussian)),
              ("walk", lambda: walk_model(data)))
    for network, build in builds:
        model = build()
        model.train_full(iterations)
        row = dict(network=network, iterations=iterations, nodes=nodes)
        for pathwise in (False, True):
            mean, var, sims = leaf_draws(model, x, pathwise)
            s_var = sims.var(axis=1, ddof=1)
            ratio = s_var / var
            z = (sims.mean(axis=1) - mean) / np.sqrt(var / n)
            key = "pathwise" if pathwise else "0.9.3"
            row[key] = dict(
                var_ratio_median=float(np.median(ratio)),
                var_outside_95=float(np.mean((ratio < lo) | (ratio > hi))),
                mean_outside_95=float(np.mean(np.abs(z) > 1.96)))
            panels.append((network, key, var, s_var))
        print(json.dumps(row), flush=True)
        rows.append(row)
    with open(os.path.join(gm.FIGURES, "gate_realizations%s.json" % suffix), "w") as f:
        json.dump(rows, f, indent=1)
    figure(panels, lo, hi, suffix)


def figure(panels, lo, hi, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    networks = ["dgp", "concat", "walk"]
    labels = {"dgp": "GP on a GP", "concat": "beside the coordinates",
              "walk": "GPWalk, isotropic reader"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, network in zip(axes, networks):
        top = 0.0
        for net_, key, var, s_var in panels:
            if net_ != network:
                continue
            ax.scatter(var, s_var / var, s=6, alpha=0.6,
                       color="#999999" if key == "0.9.3" else "#1f6fb4",
                       label="0.9.3 realizations" if key == "0.9.3"
                       else "drawn through the parent's")
            top = max(top, (s_var / var).max())
        ax.axhline(1.0, color="black", lw=0.8)
        ax.axhspan(lo, hi, color="#1f6fb4", alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_ylim(0, max(1.3, top * 1.05))
        ax.set_title(labels[network], fontsize=10)
        ax.set_xlabel("variance the leaf reports")
    axes[0].set_ylabel("realizations' variance / reported variance")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.savefig(os.path.join(gm.FIGURES, "gate_realizations%s.png" % suffix), dpi=130)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
