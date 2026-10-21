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

"""The ore probability of every fold-test run, one panel each, in the order
of `results.jsonl`, the drillholes over them. Usage: python fold_maps.py
[FILTER] [OUT]"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
key = sys.argv[1] if len(sys.argv) > 1 else ""
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fold_maps.png")
rows, seen = [], set()
for line in open(os.path.join(HERE, "results.jsonl")):
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    tag = "%s_r%g_s%d_%s" % (r["name"], r["range"], r["seed"],
                             r.get("propagation", "marginal"))
    if key in tag and tag not in seen:
        seen.add(tag)
        rows.append(r)
n = len(rows)
cols = min(n, 6)
fig, axes = plt.subplots(-(-n // cols), cols, figsize=(2.6 * cols,
                                                       3.3 * -(-n // cols)),
                         squeeze=False, constrained_layout=True)
z = np.linspace(0, -20, 101)
for ax, r in zip(axes.ravel(), rows):
    p = np.load(os.path.join(HERE, "maps", "%s_r%g_s%d.npz" % (
        r["name"], r["range"], r["seed"])))["p"]
    ax.imshow(p, origin="lower", extent=[-8, 8, -20, 0], vmin=0, vmax=1,
              cmap="RdBu_r")
    for x in (-5, 0, 5):
        ax.plot([x, x], [0, -20], color="black", lw=0.6)
    ax.set_title("%s r%g s%d\n%d bodies, bound %.1f" % (
        r["name"], r["range"], r["seed"], r["bodies"], r["bound"] or 0),
        fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.savefig(out, dpi=110)
print("drawn", n)
