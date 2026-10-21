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

"""The random-feature residual against the exact covariance.

A shallow GP on the folded section, one expert, trained briefly; on the
true field's grid, the covariance its realizations hold -- 0.9.3's
(`_PATHWISE_REALIZATIONS` off: the inducing values' draw alone) and with
the feature residual -- against the variational posterior's own,
`k(x, x') - k_x K^-1 k_x'ᵀ + k_x R Rᵀ k_x'ᵀ`, exact: the variance at each
point, the semivariance along the section, and the largest error of the
correlation matrix. Gaussian and Matérn 5/2 kernels. Writes
`gate_residual.json` and `.png`.

Usage: python gate_residual.py [ITERATIONS]
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402

N_SIM, CALLS = 250, 8


def build(data, kernel):
    geoml.set_seed(gm.ek.SEED)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, gm.ek.N_IND, seed=gm.ek.SEED))
    leaf = geoml.latent.BasicGP(root, size=1, kernel=kernel())
    model = geoml.models.VGPNetwork(
        data, "V", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False, propagation="joint",
                                       expert_propagation="independent"))
    return model


def exact_covariance(model, x):
    leaf = model.leaves[0]
    with model._propagation():
        model._refresh(model.options.jitter)
        x_tr = leaf.parent.propagate(tf.constant(x))[0]
        z = leaf.parent.inducing_points[0]
        kxx = np.asarray(leaf.covariance_matrix(x_tr, x_tr))
        kxz = np.asarray(leaf.covariance_matrix(x_tr, z))
        k_inv = np.asarray(leaf.cov_inv[0])
        root = np.asarray(leaf.chol_r[0])[0]
    g = kxz @ root
    return kxx - kxz @ k_inv @ kxz.T + g @ g.T


def realizations(model, x, pathwise):
    net._PATHWISE_REALIZATIONS = pathwise
    leaf = model.leaves[0]
    try:
        out = []
        with model._propagation():
            model._refresh(model.options.jitter)
            for c in range(CALLS):
                with net.once_per_pass():
                    leaf.propagate(tf.constant(x))
                    out.append(np.asarray(leaf.simulate(N_SIM, [7, c]))[0])
        return np.concatenate(out, axis=1)
    finally:
        net._PATHWISE_REALIZATIONS = True


def semivariance_from_cov(cov, shape, lags):
    var = np.diag(cov).reshape(shape)
    out = []
    for lag in lags:
        idx = np.arange(np.prod(shape)).reshape(shape)
        a, b = idx[:, :-lag].ravel(), idx[:, lag:].ravel()
        out.append(np.mean(0.5 * (np.diag(cov)[a] + np.diag(cov)[b])
                           - cov[a, b]))
    return np.array(out)


def main(iterations):
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    points, _ = gm.rf.grid()
    shape = (31, 47)
    lags = np.arange(1, 24)
    rows, curves = [], {}
    for name, kernel in (("Gaussian", kr.Gaussian),
                         ("Matern52", kr.Matern52)):
        model = build(data, kernel)
        model.train_full(iterations)
        exact = exact_covariance(model, points)
        sd = np.sqrt(np.diag(exact))
        row = dict(kernel=name, iterations=iterations)
        curves[(name, "exact")] = semivariance_from_cov(exact, shape, lags)
        for pathwise in (False, True):
            sims = realizations(model, points, pathwise)
            cov = np.cov(sims)
            key = "residual" if pathwise else "0.9.3"
            corr_err = np.abs(cov / np.outer(np.sqrt(np.diag(cov)),
                                             np.sqrt(np.diag(cov)))
                              - exact / np.outer(sd, sd))
            row[key] = dict(
                var_ratio_median=float(np.median(np.diag(cov)
                                                 / np.diag(exact))),
                corr_error_median=float(np.median(corr_err)),
                corr_error_95=float(np.percentile(corr_err, 95)))
            curves[(name, key)] = semivariance_from_cov(cov, shape, lags)
        print(json.dumps(row), flush=True)
        rows.append(row)
    with open(os.path.join(gm.FIGURES, "gate_residual.json"), "w") as f:
        json.dump(rows, f, indent=1)
    figure(curves, lags * 4.6 / 46)


def figure(curves, lags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, name in zip(axes, ("Gaussian", "Matern52")):
        ax.plot(lags, curves[(name, "exact")], color="black", lw=2,
                label="exact posterior covariance")
        ax.plot(lags, curves[(name, "0.9.3")], color="#999999", lw=1.5,
                label="0.9.3 realizations")
        ax.plot(lags, curves[(name, "residual")], color="#1f6fb4", lw=1.5,
                ls="--", label="with the feature residual")
        ax.set_title("shallow GP, %s kernel" % name, fontsize=10)
        ax.set_xlabel("lag along the section")
    axes[0].set_ylabel("semivariance (latent)")
    axes[0].legend(fontsize=8)
    fig.savefig(os.path.join(gm.FIGURES, "gate_residual.png"), dpi=130)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
