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

"""Maps of step 3's models, from the `maps/*.npz` the runs save: on
Walker, per network and seed, a row per arm -- the truth, the prediction,
three realizations of the ground and one draw of a measurement, on one
colour scale; on Jura, per seed, a row per arm -- the rock's prediction
and three realizations, then one metal's. Draws whatever maps exist.

Usage: python phase3_maps.py [METAL]
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(os.path.dirname(HERE), "figures", "deep_second_moment")
MAPS = os.path.join(FIGURES, "maps")
ARMS = {"moment": "0.9.3 bound", "second": "second moment",
        "warmup": "warm-up, then second"}


def walker():
    found = {}
    for path in glob.glob(os.path.join(MAPS, "walker_*.npz")):
        _, network, arm, seed = os.path.basename(path)[:-4].split("_")
        found.setdefault((network, int(seed)), {})[arm] = np.load(path)
    for (network, seed), arms in sorted(found.items()):
        order = [a for a in ARMS if a in arms]
        first = arms[order[0]]
        shape = tuple(first["shape"])
        top = np.percentile(first["truth"], 99)
        fig, axes = plt.subplots(len(order), 6, figsize=(17, 2.9 * len(order)),
                                 constrained_layout=True, squeeze=False)
        for row, arm in enumerate(order):
            m = arms[arm]
            panels = [("truth", m["truth"]), ("prediction", m["prediction"])] \
                + [("realization %d (ground)" % (k + 1), m["realizations"][:, k])
                   for k in range(3)] \
                + [("a measurement", m["measurement"])]
            for col, (title, values) in enumerate(panels):
                ax = axes[row, col]
                image = ax.imshow(values.reshape(shape), origin="lower",
                                  cmap="cividis", vmin=0, vmax=top)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(title, fontsize=9)
            axes[row, 0].set_ylabel(ARMS[arm], fontsize=10)
        fig.colorbar(image, ax=axes, shrink=0.6, label="V")
        fig.suptitle("Walker Lake, %s, seed %d" % (
            {"dgp": "GP on a GP", "concat": "GP beside the coordinates"}
            [network], seed), fontsize=11)
        fig.savefig(os.path.join(FIGURES, "phase3_walker_maps_%s_%d.png"
                                 % (network, seed)), dpi=110)
        plt.close(fig)


def jura(metal):
    found = {}
    for path in glob.glob(os.path.join(MAPS, "jura_*.npz")):
        _, arm, seed = os.path.basename(path)[:-4].split("_")
        found.setdefault(int(seed), {})[arm] = np.load(path)
    for seed, arms in sorted(found.items()):
        order = [a for a in ARMS if a in arms]
        first = arms[order[0]]
        shape = tuple(first["shape"])
        rocks = list(first["rocks"])
        elements = list(first["elements"])
        j = elements.index(metal)
        palette = ListedColormap(plt.get_cmap("tab10").colors[:len(rocks)])
        top = np.percentile(first["metal_realizations"][..., j], 99)
        fig, axes = plt.subplots(len(order), 8, figsize=(21, 3.0 * len(order)),
                                 constrained_layout=True, squeeze=False)
        for row, arm in enumerate(order):
            m = arms[arm]
            panels = [("rock, predicted", m["rock_predicted"], "rock"),
                      ("rock, entropy", m["rock_entropy"], "entropy")] \
                + [("rock, realization %d" % (k + 1),
                    m["rock_realizations"][:, k], "rock") for k in range(3)] \
                + [("%s, prediction" % metal, m["metal_prediction"][:, j],
                    "metal")] \
                + [("%s, realization %d" % (metal, k + 1),
                    m["metal_realizations"][:, k, j], "metal")
                   for k in range(2)]
            for col, (title, values, kind) in enumerate(panels):
                ax = axes[row, col]
                if kind == "rock":
                    image = ax.imshow(values.reshape(shape), origin="lower",
                                      cmap=palette, vmin=-0.5,
                                      vmax=len(rocks) - 0.5,
                                      interpolation="nearest")
                    rock_image = image
                elif kind == "entropy":
                    ax.imshow(values.reshape(shape), origin="lower",
                              cmap="magma")
                else:
                    metal_image = ax.imshow(values.reshape(shape),
                                            origin="lower", cmap="cividis",
                                            vmin=0, vmax=top)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(title, fontsize=9)
            axes[row, 0].set_ylabel(ARMS[arm], fontsize=10)
        bar = fig.colorbar(rock_image, ax=axes[:, :5], shrink=0.6,
                           ticks=range(len(rocks)))
        bar.ax.set_yticklabels(rocks, fontsize=8)
        fig.colorbar(metal_image, ax=axes[:, 5:], shrink=0.6, label=metal)
        fig.suptitle("Chapter 16 (Jura), seed %d" % seed, fontsize=11)
        fig.savefig(os.path.join(FIGURES, "phase3_jura_maps_%d.png" % seed),
                    dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    walker()
    jura(sys.argv[1] if len(sys.argv) > 1 else "Cd")
    print("drawn")
