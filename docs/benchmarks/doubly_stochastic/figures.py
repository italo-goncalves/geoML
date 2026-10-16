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

"""Figures for the doubly stochastic experiment, from the saved maps and
results. Usage: python figures.py [SEED]"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import expected_kernel as ek  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "doubly_stochastic")
OUT = HERE
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
ARMS = [("moment", "moment bound\n(today)", None, 2000),
        ("ds-coupled", "sampled,\n64 draws/step", 64, 2000),
        ("quad5", "second moment,\n5x5 Hermite", 16, 2000),
        ("quad3", "second moment,\n3x3 Hermite", 16, 2000),
        ("ds-coupled", "sampled,\n1 draw/step", 1, 10000)]
LABEL = {(a, s, i): l for a, l, s, i in ARMS}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 9.5})


def rows():
    out = []
    for line in open(os.path.join(HERE, "results.jsonl")):
        out.append(json.loads(line))
    return out


def maps(net, arm, s, seed, it):
    s_name = s if arm.startswith("ds") else 16
    path = os.path.join(HERE, "maps", "%s_%s_S%s_seed%d_it%d.npz"
                        % (net, arm, s_name, seed, it))
    return np.load(path) if os.path.exists(path) else None


def holes():
    data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
    return np.asarray(data.coordinates)


def grid():
    gx, gd = np.meshgrid(np.linspace(0, 5, 101), np.linspace(-0.2, 3.2, 69))
    return gx, gd


def style_map(ax, h):
    ax.plot(h[:, 0], h[:, 1], ".", color=INK, markersize=1.6)
    ax.set_xlim(0, 5)
    ax.set_ylim(3.2, -0.2)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_maps(net, seed):
    """Per training: the predictive mean, the exact predictive sd, the
    hidden layer's first output, and one realization."""
    gx, gd = grid()
    h = holes()
    arms = [a for a in ARMS if a[0] != "quad3"]
    fig, axes = plt.subplots(len(arms) + 1, 4, figsize=(11, 2.15 * (len(arms) + 1)),
                             layout="constrained")
    truth = ek.folded(gx, gd)
    flat = maps("vgp", "vgp", None, 2026, 2000)
    axes[0, 0].pcolormesh(gx, gd, truth, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axes[0, 0].set_title("truth, and the 16 holes")
    if flat is not None:
        shp = tuple(flat["shape"])
        axes[0, 1].pcolormesh(gx, gd, flat["moment_mean"].reshape(shp),
                              cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        axes[0, 1].set_title("flat GP: mean")
        axes[0, 2].pcolormesh(gx, gd, flat["moment_sd"].reshape(shp),
                              cmap="Blues", vmin=0, vmax=1)
        axes[0, 2].set_title("flat GP: sd")
        axes[0, 3].pcolormesh(gx, gd, flat["today_fields"][0].reshape(shp),
                              cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        axes[0, 3].set_title("flat GP: a realization")
    for r, (arm, label, s, it) in enumerate(arms, start=1):
        m = maps(net, arm, s, seed, it)
        name = label.replace("\n", " ")
        if m is None:
            for c in range(4):
                axes[r, c].axis("off")
            continue
        shp = tuple(m["shape"])
        im_f = axes[r, 0].pcolormesh(gx, gd, m["exact_mean"].reshape(shp),
                                     cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        axes[r, 0].set_title("%s: mean" % name)
        im_s = axes[r, 1].pcolormesh(gx, gd, m["exact_sd"].reshape(shp),
                                     cmap="Blues", vmin=0, vmax=1)
        axes[r, 1].set_title("exact predictive sd")
        hm = m["hidden_mean"][:, 0].reshape(shp)
        axes[r, 2].pcolormesh(gx, gd, hm, cmap="PuOr_r",
                              vmin=-np.abs(hm).max(), vmax=np.abs(hm).max())
        axes[r, 2].contour(gx, gd, hm, levels=9, colors=INK,
                           linewidths=0.4)
        axes[r, 2].set_title("hidden layer, output 1 (mean)")
        axes[r, 3].pcolormesh(gx, gd, m["ds_fields"][0].reshape(shp),
                              cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        axes[r, 3].set_title("a realization (through the hidden one)")
    for ax in axes.ravel():
        if ax.has_data():
            style_map(ax, h)
    fig.colorbar(im_f, ax=axes[:, 0], location="bottom", shrink=0.8,
                 label="value")
    fig.colorbar(im_s, ax=axes[:, 1], location="bottom", shrink=0.8,
                 label="standard deviation")
    fig.suptitle("Folded section, %s network, seed %d: each training's "
                 "fitted model" % ("GP on the GP beside the coordinates"
                                   if net == "dgpc" else "GP on a GP", seed))
    path = os.path.join(OUT, "maps_%s_seed%d.png" % (net, seed))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_missing_variance(seed):
    """The moment path's predictive sd against the exact one, and the
    ratio, for the moment-trained models -- the second moment's absence
    made visible."""
    gx, gd = grid()
    h = holes()
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.6), layout="constrained")
    for r, net in enumerate(("dgp", "dgpc")):
        m = maps(net, "moment", None, seed, 2000)
        if m is None:
            continue
        shp = tuple(m["shape"])
        a = axes[r, 0].pcolormesh(gx, gd, m["moment_sd"].reshape(shp),
                                  cmap="Blues", vmin=0, vmax=1.2)
        axes[r, 0].set_title("%s: sd by the moment path" % net)
        axes[r, 1].pcolormesh(gx, gd, m["exact_sd"].reshape(shp),
                              cmap="Blues", vmin=0, vmax=1.2)
        axes[r, 1].set_title("exact sd (same model, sampled)")
        ratio = (m["moment_sd"] / m["exact_sd"]).reshape(shp)
        b = axes[r, 2].pcolormesh(gx, gd, ratio, cmap="Greys_r", vmin=0,
                                  vmax=1.2)
        axes[r, 2].set_title("ratio: moment sd / exact sd")
        for ax in axes[r]:
            style_map(ax, h)
    fig.colorbar(a, ax=axes[:, :2], location="right", shrink=0.8,
                 label="latent sd, data units")
    fig.colorbar(b, ax=axes[:, 2], location="right", shrink=0.8,
                 label="ratio")
    fig.suptitle("Trained on the moment bound: the variance the moment path "
                 "reports against the model's exact variance (seed %d)" % seed)
    path = os.path.join(OUT, "missing_variance_seed%d.png" % seed)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def served(row):
    """The score a user of that arm would get: moments for the moment
    bound, the mixture for the sampled and quadrature arms."""
    key = "moment" if row["arm"] == "moment" else "mixture"
    return row[key]["score_per_hole"]


def fig_scores(all_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True,
                             layout="constrained")
    flat = [r for r in all_rows if r["arm"] == "vgp"]
    for ax, net in zip(axes, ("dgp", "dgpc")):
        for k, (arm, label, s, it) in enumerate(ARMS):
            sel = [r for r in all_rows if r["network"] == net
                   and r["arm"] == arm and r["iterations"] == it
                   and (not arm.startswith("ds") or r["s_train"] == s)]
            vals = [served(r) for r in sel]
            colour = SERIES[min(k, 3)] if k < 4 else MUTED
            ax.scatter([k] * len(vals), vals, s=36, color=colour,
                       edgecolor="white", linewidth=1.2, zorder=3)
            if vals:
                ax.plot([k - 0.25, k + 0.25], [np.mean(vals)] * 2,
                        color=INK, linewidth=2, zorder=4)
                ax.annotate("%.1f" % np.mean(vals), (k + 0.28, np.mean(vals)),
                            va="center", fontsize=8, color=INK)
        if flat:
            ax.axhline(served(flat[0]) if flat[0]["arm"] != "vgp"
                       else flat[0]["moment"]["score_per_hole"],
                       color=MUTED, linestyle="--", linewidth=1)
            ax.annotate("flat GP", (len(ARMS) - 0.6,
                                    flat[0]["moment"]["score_per_hole"]),
                        va="bottom", fontsize=8, color=MUTED)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([a[1] for a in ARMS], fontsize=8)
        ax.set_title({"dgp": "GP on a GP",
                      "dgpc": "GP on the GP beside the coordinates"}[net])
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("log score a new hole (higher is better)")
    fig.suptitle("Folded section, 200 new drillholes: three seeds per arm "
                 "(dots), their mean (bar)")
    path = os.path.join(OUT, "scores.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_cost():
    lines = [json.loads(l) for l in open(os.path.join(HERE, "scale.jsonl"))]
    sizes = sorted({(d.get("n", d.get("n_holes", 0) * 12), d["m"])
                    for d in lines})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for k, (n, m) in enumerate(sizes):
        sel = [d for d in lines if d.get("n", d.get("n_holes", 0) * 12) == n
               and d["m"] == m]
        mom = [d for d in sel if d["arm"] == "moment" and "failed" not in d]
        dsl = sorted([d for d in sel if d["arm"] == "ds" and "failed" not in d],
                     key=lambda d: d["s"])
        colour = SERIES[k]
        label = "%d points, %d inducing" % (n, m)
        for ax, key in ((axes[0], "ms_per_iter"), (axes[1], "total_peak_mb")):
            xs = [d["s"] for d in dsl]
            ys = [d[key] / (1000 if key == "total_peak_mb" else 1) for d in dsl]
            ax.plot(xs, ys, "-o", color=colour, linewidth=2, markersize=6,
                    label=label)
            if mom:
                ax.plot([1], [mom[0][key] / (1000 if key == "total_peak_mb"
                                              else 1)],
                        marker="s", color=colour, markersize=9,
                        markerfacecolor="white", markeredgewidth=2)
            if any("failed" in d for d in sel):
                ax.annotate("out of memory\nat 64 draws", (64, ys[-1]),
                            fontsize=7.5, color=colour, va="bottom",
                            ha="right")
    for ax, ylabel in ((axes[0], "ms per training step"),
                       (axes[1], "peak resident memory, GB")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_xlabel("draws of the hidden layer per step")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("time (open square: the moment bound)")
    axes[1].set_title("memory (open square: the moment bound)")
    fig.suptitle("Cost of a training step, GP on a GP, at larger sizes")
    path = os.path.join(OUT, "cost.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_training(net, seed):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    for k, (arm, label, s, it) in enumerate(ARMS):
        m = maps(net, arm, s, seed, it)
        if m is None:
            continue
        log = m["log"]
        x = np.linspace(0, 1, len(log))
        smooth = np.convolve(log, np.ones(25) / 25, mode="valid")
        ax.plot(x[:len(smooth)], smooth, color=SERIES[min(k, 3)]
                if k < 4 else MUTED, linewidth=2,
                label=label.replace("\n", " ")
                + ("" if it == 2000 else " (10 000 steps)"))
    ax.set_ylim(np.percentile(m["log"], 5) - 50, None)
    ax.set_xlabel("share of training")
    ax.set_ylabel("bound (25-step mean)")
    ax.grid(color=GRID)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("%s, seed %d: each arm's own bound" % (net, seed))
    path = os.path.join(OUT, "training_%s_seed%d.png" % (net, seed))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    all_rows = rows()
    for p in (fig_maps("dgpc", seed), fig_maps("dgp", seed),
              fig_missing_variance(seed), fig_scores(all_rows), fig_cost(),
              fig_training("dgpc", seed)):
        print(p)
