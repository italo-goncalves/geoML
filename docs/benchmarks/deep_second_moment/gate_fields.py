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

"""Gates 2b and 2c of the deep second moment: the realizations against
the folded section's true field.

The folded section's two networks, trained on the 0.9.3 moment bound
(three seeds), predicted at the true field's grid through `model.predict`,
64 realizations, drawn through the parent's and the 0.9.3 way: the share
of the grid whose truth falls inside the ensemble's 90% band, its CRPS and
spread (`doubly_stochastic/run_folded.ensemble_scores`), and the
realizations' variogram along the section against the true field's.
Writes `gate_fields.json` and `.png`.

Usage: python gate_fields.py [ITERATIONS]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402

SEEDS = (11, 12, 13)
N_SIM = 64


def realizations(model, points, pathwise):
    net._PATHWISE_REALIZATIONS = pathwise
    try:
        target = geoml.data.PointData.from_array(points, ["X", "Y"])
        model.predict(target, n_sim=N_SIM)
        return np.asarray(target.variables["V"].get_simulations()).T
    finally:
        net._PATHWISE_REALIZATIONS = True


def variogram(field, shape, lags):
    """Semivariance along the section's first axis, averaged over rows and
    realizations: `field` `[S, rows * cols]`."""
    f = field.reshape((field.shape[0],) + shape)
    return np.array([0.5 * np.mean((f[:, :, lag:] - f[:, :, :-lag]) ** 2)
                     for lag in lags])


def main(iterations):
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    points, truth = gm.rf.grid()
    shape = (31, 47)
    lags = np.arange(1, 24)
    true_vario = variogram(truth[None], shape, lags)
    rows, curves = [], {}
    for network in ("dgp", "concat"):
        for seed in SEEDS:
            gm.ek.SEED, keep = seed, gm.ek.SEED
            try:
                model = gm.build(data, network, kr.Gaussian)
            finally:
                gm.ek.SEED = keep
            net._UNCERTAIN_POINTS_SECOND_MOMENT = False
            model.train_full(iterations)
            net._UNCERTAIN_POINTS_SECOND_MOMENT = True
            row = dict(network=network, seed=seed, iterations=iterations)
            for pathwise in (False, True):
                sims = realizations(model, points, pathwise)
                key = "pathwise" if pathwise else "0.9.3"
                row[key] = gm.rf.ensemble_scores(sims, truth)
                curves.setdefault((network, key), []).append(
                    variogram(sims, shape, lags))
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(os.path.join(gm.FIGURES, "gate_fields.json"), "w") as f:
        json.dump(rows, f, indent=1)
    figure(curves, true_vario, lags * 4.6 / 46)


def figure(curves, true_vario, lags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"dgp": "GP on a GP", "concat": "beside the coordinates"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, network in zip(axes, ("dgp", "concat")):
        ax.plot(lags, true_vario, color="black", lw=2, label="true field")
        for key, color in (("0.9.3", "#999999"), ("pathwise", "#1f6fb4")):
            for i, curve in enumerate(curves[(network, key)]):
                ax.plot(lags, curve, color=color, lw=1,
                        label=None if i else
                        ("0.9.3 realizations" if key == "0.9.3"
                         else "drawn through the parent's"))
        ax.set_title(labels[network], fontsize=10)
        ax.set_xlabel("lag along the section")
    axes[0].set_ylabel("semivariance (realizations, data units)")
    axes[0].legend(fontsize=8)
    fig.savefig(os.path.join(gm.FIGURES, "gate_fields.png"), dpi=130)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
