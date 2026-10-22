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

"""The fold ablation's maps, one panel an arm.

Reads the JSON lines `fold_ablation.py` printed and the maps it saved, and
draws each arm's ore probability with its 0.5 contour, the three holes and
their ore intervals; the title names the arm, the ore bodies, P on the
fold's two limbs, the walk's trained amplitude and the bound.

Usage: python fold_ablation_maps.py RESULTS.jsonl MAPS_DIR OUT.png NAME...
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "ref_cubic_marginal": "Cubic, marginal (reference)",
    "m32_marginal": "Matern32, marginal",
    "gauss_marginal": "Gaussian, marginal",
    "m32_joint": "Matern32, joint (0.9.4 default)",
    "m32_joint_amp10": "Matern32, joint, amp starts at 10",
    "m32_joint_noprior": "Matern32, joint, no range prior",
    "m32_joint_full0": "Matern32, joint, full_variance off",
    "gauss_joint": "Gaussian, joint",
    "m32_joint_long": "Matern32, joint, 1000 it",
    "m32_joint_dfield001": "joint, delta 0.01 on the field",
    "m32_joint_dfield01": "joint, delta 0.1 on the field",
    "m32_joint_dall001": "joint, delta 0.01 on every GP",
    "m32_joint_dall01": "joint, delta 0.1 on every GP",
    "m32_joint_dall001_long": "joint, delta 0.01 on every GP, 1000 it",
    "long_default_r1": "joint, default delta, tolerance 0.01",
    "long_conf_r1": "joint, delta 0.01, tolerance 0.01",
    "conf_r1_3000": "joint, delta 0.01, 3000 steps",
}


def main(results, maps, out, names):
    rows = {}
    for line in open(results, encoding="utf-8"):
        if line.startswith("{"):
            row = json.loads(line)
            rows[row["name"]] = row          # the last run of a name wins
    z1 = np.linspace(0, -20, 101)
    # the right hole is the left one's log reversed, its ore at -8.5 to -5.5
    ore = {-5: [(-14.5, -11.5)], 5: [(-8.5, -5.5)],
           0: [(-5.5, -2.5), (-11.5, -8.5), (-17.5, -14.5)]}
    figure, axes = plt.subplots(2, 4, figsize=(15, 10.5), sharex=True,
                                sharey=True)
    for ax, name in zip(axes.ravel(), names):
        row = rows[name]
        p = np.load(os.path.join(maps, "%s_s%d.npz" % (name, row["seed"])))["p"]
        image = ax.imshow(p, origin="lower", extent=[-8, 8, -20, 0],
                          cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
        ax.contour(np.linspace(-8, 8, 161), np.linspace(-20, 0, 201), p,
                   levels=[0.5], colors="k", linewidths=1.0)
        for x, intervals in ore.items():
            ax.plot([x, x], [0, -20], color="k", lw=1.2)
            for low, high in intervals:
                ax.plot([x, x], [low, high], color="w", lw=2.2)
        ax.set_title("%s\n%d %s, limbs %.2f / %.2f\namp %.2f, bound %.1f"
                     % (LABELS.get(name, name), row["bodies"],
                        "body" if row["bodies"] == 1 else "bodies",
                        row["limbs"][0], row["limbs"][1], row["amp"],
                        row["bound"]), fontsize=8.5)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    figure.colorbar(image, ax=axes, shrink=0.6, label="P(ore)")
    figure.suptitle(os.environ.get(
        "FOLD_TITLE", "Folded ore on 0.9.4, one factor changed at a time "
        "(seed 1, 200 iterations)"))
    figure.savefig(out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
