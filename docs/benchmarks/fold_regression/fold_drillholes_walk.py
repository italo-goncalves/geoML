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

"""Where the drilled vein's deep models move the section.

Loads the deep models `fold_drillholes.py` saved and predicts two nodes of
each on an 81 x 101 grid with `predict_node`: the walk's start (the
transformed coordinates it reads) and the walk itself. One row a model: the
displacement's size over the section with its arrows; the truth carried to
the walked coordinates, a cell an ore or waste dot, with the section's
grid lines (every metre) drawn where the walk puts them; and the samples
carried the same way, each hole a line. Everything after the first panel
is in the reader's space, where the GP above the walk is stationary.

Usage: python fold_drillholes_walk.py MAPS_DIR OUT.png NAME_SEED...
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import geoml  # noqa: E402
import geoml.latent as gl  # noqa: E402
import geoml.persistence as gp  # noqa: E402


def walked(model, coordinates):
    """The walk's start and end at `coordinates`, each `(n, 2)`."""
    walk = next(node for node in model._nodes()
                if isinstance(node, gl.GPWalk))
    target = geoml.data.PointData.from_array(coordinates)
    ends = []
    for node in (walk.walker, walk):
        variable = model.predict_node(node, target, n_sim=1, name=node.name)
        ends.append(variable.get_predictions())
    return ends[0], ends[1], float(np.asarray(walk.parameters["amp"]
                                              .get_value()))


def main(maps, out, stems):
    here = os.path.dirname(os.path.abspath(__file__))
    truth = np.load(os.path.join(here, "figures", "vein_truth.npz"))["ore"]
    xs, ys = np.linspace(-8, 8, 81), np.linspace(-20, 0, 101)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    ore_cell = truth[np.round((grid[:, 1] + 20) / 0.1).astype(int),
                     np.round((grid[:, 0] + 8) / 0.1).astype(int)]
    figure, axes = plt.subplots(len(stems), 3,
                                figsize=(15, 5.6 * len(stems)), squeeze=False)
    for line, stem in zip(axes, stems):
        model = gp.load_model(os.path.join(maps, stem + "_model"))
        saved = np.load(os.path.join(maps, stem + ".npz"))
        samples, ore = saved["coords"], saved["ore"]
        start, end, amp = walked(model, grid)
        s_start, s_end, _ = walked(model, samples)
        shift = np.linalg.norm(end - start, axis=1)

        image = line[0].imshow(shift.reshape(101, 81), origin="lower",
                               extent=[-8, 8, -20, 0], cmap="viridis")
        every = (slice(None, None, 5), slice(None, None, 5))
        line[0].quiver(gx[every], gy[every],
                       (end - start)[:, 0].reshape(101, 81)[every],
                       (end - start)[:, 1].reshape(101, 81)[every],
                       color="w", angles="xy", scale_units="xy", scale=0.25)
        line[0].contour(np.linspace(-8, 8, 161), np.linspace(-20, 0, 201),
                        truth.astype(float), levels=[0.5], colors="r",
                        linewidths=0.8)
        figure.colorbar(image, ax=line[0], shrink=0.8,
                        label="displacement, transformed units")
        line[0].set_title("%s: the walk's displacement\n(arrows x4, amp %.2f;"
                          " max %.2f, mean %.2f)"
                          % (stem, amp, shift.max(), shift.mean()),
                          fontsize=9)
        line[0].set_aspect("equal")

        for ax in line[1:]:
            # the section's grid lines, every metre, where the walk puts them
            for j in range(0, 81, 5):
                ax.plot(end.reshape(101, 81, 2)[:, j, 0],
                        end.reshape(101, 81, 2)[:, j, 1], color="0.75",
                        lw=0.5)
            for i in range(0, 101, 5):
                ax.plot(end.reshape(101, 81, 2)[i, :, 0],
                        end.reshape(101, 81, 2)[i, :, 1], color="0.75",
                        lw=0.5)
            ax.set_aspect("equal")
        line[1].scatter(end[~ore_cell, 0], end[~ore_cell, 1], s=1.5,
                        color="#9ecae1", lw=0)
        line[1].scatter(end[ore_cell, 0], end[ore_cell, 1], s=1.5,
                        color="#d62728", lw=0)
        line[1].set_title("the truth in the walked coordinates\n(ore red, "
                          "waste blue; grey: the section's metre grid)",
                          fontsize=9)
        breaks = np.flatnonzero(np.linalg.norm(np.diff(samples, axis=0),
                                               axis=1) > 0.5) + 1
        for hole in np.split(np.arange(len(samples)), breaks):
            line[2].plot(s_end[hole, 0], s_end[hole, 1], color="k", lw=0.8)
        line[2].scatter(s_end[ore, 0], s_end[ore, 1], s=6, color="#d62728",
                        zorder=3)
        line[2].set_title("the samples in the walked coordinates\n(ore "
                          "red, each hole a line)", fontsize=9)
        line[2].set_xlim(line[1].get_xlim())
        line[2].set_ylim(line[1].get_ylim())
    figure.suptitle(os.environ.get(
        "FOLD_TITLE", "The deep models' walk: the section before and after"))
    figure.savefig(out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
