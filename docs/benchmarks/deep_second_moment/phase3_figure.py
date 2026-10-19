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

"""The figure of step 3 on the folded section, from `phase3_folded.jsonl`:
the log score per new hole and the realizations' CRPS against the true
field, by arm, a point per seed."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(os.path.dirname(HERE), "figures", "deep_second_moment")
rows = [json.loads(line) for line in open(
    os.path.join(FIGURES, "phase3_folded.jsonl"))]
arms = ["moment", "second", "warmup"]
labels = {"moment": "0.9.3 bound", "second": "second moment",
          "warmup": "sampled warm-up,\nthen second moment"}
networks = {"dgp": "GP on a GP", "concat": "beside the coordinates"}
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
for col, network in enumerate(networks):
    for row, (metric, title) in enumerate(
            (("score", "log score per new hole (higher is better)"),
             ("crps", "field CRPS of the realizations (lower is better)"))):
        ax = axes[row, col]
        for i, arm in enumerate(arms):
            values = [r["moments"]["score_per_hole"] if metric == "score"
                      else r["field"]["crps"]
                      for r in rows if r["network"] == network
                      and r["arm"] == arm]
            ax.scatter(np.full(len(values), i), values, s=30,
                       color=["#999999", "#1f6fb4", "#d9822b"][i])
            if values:
                ax.hlines(np.mean(values), i - 0.25, i + 0.25, color="black")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([labels[a] for a in arms], fontsize=8)
        ax.set_xlim(-0.5, len(arms) - 0.5)
        if row == 0:
            ax.set_title(networks[network], fontsize=10)
        if col == 0:
            ax.set_ylabel(title, fontsize=9)
fig.savefig(os.path.join(FIGURES, "phase3_folded.png"), dpi=130)
print("drawn")
