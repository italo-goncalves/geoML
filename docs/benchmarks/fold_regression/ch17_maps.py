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

"""Chapter 17's 0.8.0 models on each version: a vertical section (north
15 875) and a plan (RL 1450) through the Vein indicator, the vein where it
is positive, a column per version, a row per model and view."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["v0.8.0", "v0.8.4", "v0.8.8", "v0.9.0", "v0.9.3", "branch"]
rows = {}
for line in open(os.path.join(HERE, "ch17_results.jsonl")):
    if line.startswith("{"):
        r = json.loads(line)
        rows[(r["name"], r["model"])] = r
versions = [v for v in ORDER
            if os.path.exists(os.path.join(HERE, "ch17",
                                           "ch17_%s_deep_s1234.npz" % v))]
fig, axes = plt.subplots(4, len(versions), figsize=(2.9 * len(versions), 10),
                         squeeze=False, constrained_layout=True)
for col, version in enumerate(versions):
    for k, kind in enumerate(("stationary", "deep")):
        path = os.path.join(HERE, "ch17", "ch17_%s_%s_s1234.npz"
                            % (version, kind))
        if not os.path.exists(path):
            continue
        m = np.load(path)
        r = rows.get((version, kind), {})
        for j, (view, extent, label) in enumerate((
                ("section", [24850, 25150, 1300, 1600], "E-RL"),
                ("plan", [24850, 25150, 15700, 16050], "E-N"))):
            ax = axes[2 * k + j, col]
            value = m[view]
            ax.imshow(value > 0, origin="lower", extent=extent,
                      cmap="cividis", aspect="auto", vmin=0, vmax=1)
            ax.contour(np.linspace(extent[0], extent[1], value.shape[1]),
                       np.linspace(extent[2], extent[3], value.shape[0]),
                       value, levels=[0.0], colors="white", linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title("%s %s\n%s pieces, agreement %.3f" % (
                    version, kind, r.get("pieces", "?"),
                    r.get("agreement", float("nan"))), fontsize=8)
            if col == 0:
                ax.set_ylabel("%s, %s" % (kind, label), fontsize=8)
fig.savefig(os.path.join(HERE, "ch17_maps.png"), dpi=110)
print("drawn", len(versions))
