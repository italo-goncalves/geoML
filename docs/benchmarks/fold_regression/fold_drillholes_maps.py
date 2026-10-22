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

"""The drilled vein's maps: the truth and its holes, then each run's P(ore).

Reads the JSON lines `fold_drillholes.py` printed and the maps it saved.
One row a seed: the truth with the holes (ore samples in red), the
stationary model and the deep one (`FOLD_NAMES`, default
"stationary,deep"), each with its 0.5 contour in black and
the truth's outline dashed; the titles carry the scores against the truth.

Usage: python fold_drillholes_maps.py RESULTS MAPS_DIR OUT.png SEED...
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXTENT = [-8, 8, -20, 0]
X, Y = np.linspace(-8, 8, 161), np.linspace(-20, 0, 201)


def main(results, maps, out, seeds):
    rows = {}
    for line in open(results, encoding="utf-8"):
        if line.startswith("{"):
            row = json.loads(line)
            rows[(row["name"], row["seed"])] = row
    here = os.path.dirname(os.path.abspath(__file__))
    truth = np.load(os.path.join(here, "figures", "vein_truth.npz"))["ore"]
    # the two runs' names, stationary first
    names = os.environ.get("FOLD_NAMES", "stationary,deep").split(",")
    figure, axes = plt.subplots(len(seeds), 3, figsize=(12.5, 5.2 * len(seeds)),
                                sharex=True, sharey=True, squeeze=False)
    for line, seed in zip(axes, seeds):
        saved = np.load(os.path.join(maps, "%s_s%d.npz" % (names[0], seed)))
        coords, ore = saved["coords"], saved["ore"]
        line[0].imshow(truth, origin="lower", extent=EXTENT, cmap="Greys",
                       vmin=0, vmax=1.6)
        line[0].set_title("truth and the holes\n%d samples, %d ore\n"
                          % (len(ore), ore.sum()), fontsize=9)
        for name, ax in zip(names, line[1:]):
            row = rows[(name, seed)]
            p = np.load(os.path.join(maps, "%s_s%d.npz" % (name, seed)))["p"]
            image = ax.imshow(p, origin="lower", extent=EXTENT,
                              cmap="RdBu_r", vmin=0, vmax=1)
            ax.contour(X, Y, p, levels=[0.5], colors="k", linewidths=1.0)
            ax.set_title("%s, seed %d: %d %s\naccuracy %.3f, IoU %.3f\n"
                         "Brier %.3f, bound %.1f"
                         % (name, seed, row["bodies"],
                            "body" if row["bodies"] == 1 else "bodies",
                            row["accuracy"], row["iou"], row["brier"],
                            row["bound"]), fontsize=9)
        for ax in line:
            ax.contour(X, Y, truth.astype(float), levels=[0.5],
                       colors="0.35", linewidths=0.9, linestyles="--")
            ax.plot(coords[:, 0], coords[:, 1], ".", color="k", ms=1)
            ax.plot(coords[ore, 0], coords[ore, 1], ".", color="#d62728",
                    ms=2.5)
            ax.set_xlim(-8, 8)
            ax.set_ylim(-20, 0)
    figure.colorbar(image, ax=axes, shrink=0.5, label="P(ore)")
    figure.suptitle(os.environ.get(
        "FOLD_TITLE", "The folded vein drilled by five holes: 500 steps, "
        "one optimizer"))
    figure.savefig(out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4:]])
