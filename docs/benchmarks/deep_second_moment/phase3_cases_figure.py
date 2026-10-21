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

"""The figures of step 3 on Walker Lake and chapter 16, from
`phase3_walker.jsonl` and `phase3_jura.jsonl`, by arm, a point per seed,
the mean a bar; whatever rows are there so far."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(os.path.dirname(HERE), "figures", "deep_second_moment")
COLOURS = {"moment": "#999999", "second": "#1f6fb4", "warmup": "#d9822b"}
LABELS = {"moment": "0.9.3 bound", "second": "second\nmoment",
          "warmup": "warm-up,\nthen second"}


def rows(name):
    path = os.path.join(FIGURES, name)
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def panel(ax, data, arms, value, title):
    for i, arm in enumerate(arms):
        values = [value(r) for r in data if r["arm"] == arm]
        ax.scatter(np.full(len(values), i), values, s=30,
                   color=COLOURS[arm])
        if values:
            ax.hlines(np.mean(values), i - 0.25, i + 0.25, color="black")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABELS[a] for a in arms], fontsize=8)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_title(title, fontsize=9)


def walker():
    data = rows("phase3_walker.jsonl")
    arms = ["moment", "second", "warmup"]
    metrics = (
        (lambda r: r["moments"]["log_score"],
         "moments: log score a point,\nwarped (higher better)"),
        (lambda r: r["moments"]["cover90"], "moments: share inside 90%"),
        (lambda r: r["field"]["rmse"], "prediction RMSE, V\n(lower better)"),
        (lambda r: r["field"]["measurement"]["crps"],
         "measurement samples: CRPS, V\n(lower better)"),
        (lambda r: r["field"]["measurement"]["cover90"],
         "measurement samples: 90%\ncoverage"),
        (lambda r: r["field"]["ground"]["cover90"],
         "ground realizations: 90%\ncoverage (nugget left out)"))
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.2),
                             constrained_layout=True)
    for row, (network, name) in enumerate((("dgp", "GP on a GP"),
                                           ("concat", "beside the coords"))):
        subset = [r for r in data if r["network"] == network]
        for col, (value, title) in enumerate(metrics):
            panel(axes[row, col], subset, arms, value,
                  title if row == 0 else "")
            if col in (1, 4, 5):
                axes[row, col].axhline(0.9, color="black", lw=0.6, ls=":")
        axes[row, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("Walker Lake, two-layer networks: %d runs so far" % len(data),
                 fontsize=10)
    fig.savefig(os.path.join(FIGURES, "phase3_walker.png"), dpi=130)


def jura():
    data = rows("phase3_jura.jsonl")
    arms = ["moment", "second"]
    metrics = (
        (lambda r: r["metals_crps_sd"], "metals CRPS / sd (lower better)"),
        (lambda r: r["metals_rmse_sd"], "metals RMSE / sd (lower better)"),
        (lambda r: r["metals_goodness"], "metals goodness"),
        (lambda r: r["rock_log"], "rock log score (lower better)"),
        (lambda r: r["rock_accuracy"], "rock accuracy (higher better)"),
        (lambda r: r["seconds"], "training, seconds"))
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.6), constrained_layout=True)
    for ax, (value, title) in zip(axes, metrics):
        panel(ax, data, arms, value, title)
    fig.suptitle("Chapter 16 (Jura), held-out sites: %d runs so far"
                 % len(data), fontsize=10)
    fig.savefig(os.path.join(FIGURES, "phase3_jura.png"), dpi=130)


if __name__ == "__main__":
    walker()
    jura()
    print("drawn")
