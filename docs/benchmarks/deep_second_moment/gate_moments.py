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

"""Gate 1a of the deep second moment: the leaf's analytic moments against
draws of the deep model.

The folded section's two networks (a GP on a GP, and a GP on a GP beside
the coordinates), one expert, trained briefly on the 0.9.3 moment bound;
at points of the new holes, the leaf's mean and variance as the package
reports them -- with the conditioned second moment and without it -- set
against 4000 coupled draws of the hidden layer
(`doubly_stochastic/run_folded.exact_moments`, the leaf's inducing inputs
conditioned on each draw). Writes `gate_moments.json` and
`gate_moments.png` beside the other figures.

Usage: python gate_moments.py [ITERATIONS]
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "doubly_stochastic"))
import expected_kernel as ek  # noqa: E402
import run_folded as rf  # noqa: E402
import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402

FIGURES = os.path.join(os.path.dirname(HERE), "figures", "deep_second_moment")
os.makedirs(FIGURES, exist_ok=True)
N_POINTS = 300


def build(data, network, kernel):
    """`run_folded.build` with the leaf's kernel chosen."""
    geoml.set_seed(ek.SEED)
    ip = geoml.data.inducing.from_kmeans(data, ek.N_IND, seed=ek.SEED)
    n_experts = int(os.environ.get("N_EXPERTS", "1"))
    if n_experts > 1:
        ip = geoml.data.inducing.experts(ip, n_experts, seed=0)
    root = geoml.latent.BasicInput(ip)
    hidden = geoml.latent.BasicGP(root, size=2)
    below = hidden if network == "dgp" \
        else geoml.latent.Concatenate(hidden, root)
    leaf = geoml.latent.BasicGP(below, size=1, kernel=kernel())
    options = geoml.models.GPOptions(verbose=False, propagation="joint",
                                     expert_propagation="independent")
    model = geoml.models.VGPNetwork(data, "V", geoml.likelihood.Gaussian(),
                                    leaf, options=options)
    for p in model.likelihoods[0].warping.all_parameters:
        p.fix()
    return model


def analytic(model, x, conditioned):
    net._UNCERTAIN_POINTS_SECOND_MOMENT = conditioned
    try:
        with model._propagation():
            model._refresh(model.options.jitter)
            mean, var = model.leaves[0].propagate(tf.constant(x))
        return np.asarray(mean)[:, 0], np.asarray(var)[:, 0]
    finally:
        net._UNCERTAIN_POINTS_SECOND_MOMENT = True


def main(iterations):
    data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
    new, _ = ek.drillholes(ek.N_NEW, ek.NEW_SEED)
    pick = np.random.default_rng(3).choice(new.n_data, N_POINTS,
                                           replace=False)
    x = np.asarray(new.coordinates, dtype=float)[pick]
    rows, panels = [], []
    for network in ("dgp", "concat"):
        for name, kernel in (("Gaussian", kr.Gaussian),
                             ("Matern52", kr.Matern52)):
            model = build(data, network, kernel)
            # trained as models are today, on the 0.9.3 moment bound
            net._UNCERTAIN_POINTS_SECOND_MOMENT = False
            model.train_full(iterations)
            net._UNCERTAIN_POINTS_SECOND_MOMENT = True
            ex_mean, ex_evar, ex_vmu = rf.exact_moments(model, x, 4000)
            ex_var = ex_evar + ex_vmu
            row = dict(network=network, kernel=name, iterations=iterations,
                       exact_var_median=float(np.median(ex_var)),
                       spread_share=float(np.median(ex_vmu / ex_var)))
            for conditioned in (False, True):
                mean, var = analytic(model, x, conditioned)
                key = "conditioned" if conditioned else "0.9.3"
                ratio = var / ex_var
                row[key] = dict(
                    mean_error_sd=float(np.max(np.abs(mean - ex_mean)
                                               / np.sqrt(ex_var))),
                    var_ratio_median=float(np.median(ratio)),
                    var_ratio_5=float(np.percentile(ratio, 5)),
                    var_ratio_95=float(np.percentile(ratio, 95)))
                panels.append((network, name, key, ex_var, var))
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(os.path.join(FIGURES, "gate_moments.json"), "w") as f:
        json.dump(rows, f, indent=1)
    figure(panels)


def figure(panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combos = sorted({(p[0], p[1]) for p in panels},
                    key=lambda c: (c[0] != "dgp", c[1]))
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 4),
                             constrained_layout=True)
    labels = {"dgp": "GP on a GP", "concat": "beside the coordinates"}
    for ax, combo in zip(axes, combos):
        lo, hi = np.inf, 0.0
        for network, name, key, ex_var, var in panels:
            if (network, name) != combo:
                continue
            ax.scatter(ex_var, var, s=6, alpha=0.6,
                       label="0.9.3" if key == "0.9.3"
                       else "conditioned second moment",
                       color="#999999" if key == "0.9.3" else "#1f6fb4")
            lo, hi = min(lo, ex_var.min(), var.min()), \
                max(hi, ex_var.max(), var.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("%s, %s leaf" % (labels[combo[0]], combo[1]),
                     fontsize=10)
        ax.set_xlabel("variance over 4000 draws of the hidden layer")
    axes[0].set_ylabel("variance the leaf reports")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIGURES, "gate_moments.png"), dpi=130)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
