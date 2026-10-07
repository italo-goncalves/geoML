"""Figures of the batched-experts experiments, from results/*.jsonl.

Writes into experiments/batched_experts/figures/ and copies each figure to
the research project's figures/ folder.
"""
import glob
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = "experiments/batched_experts"
OUT = os.path.join(HERE, "figures")
RESEARCH = glob.glob("/mnt/c/Users/*talo/OneDrive/Claude/Research/"
                     "Batched experts/figures")
INK, INK2, MUTED, GRID, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                   "#e1e0d9", "#fcfcfb")
METHODS = {"svi": ("every expert each step (train_svi)", "#2a78d6", "o", "-"),
           "batch": ("by expert, shared stepped per batch", "#eb6834", "s",
                     "--"),
           "epoch": ("by expert, shared stepped per epoch", "#1baf7a", "^",
                     ":"),
           "epoch4": ("by expert, per epoch, 4 visits", "#eda100", "D",
                      "-."),
           "epoch16": ("by expert, per epoch, 16 visits", "#e87ba4", "v",
                       (0, (5, 1)))}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
    "lines.markersize": 7, "legend.frameon": False})


def load(name):
    path = os.path.join(HERE, "results", name)
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path)]
    # the last run of each configuration wins
    keep = {}
    for r in rows:
        keep[(r["case"], r["J"], r.get("method"))] = r
    return list(keep.values())


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    for target in RESEARCH:
        shutil.copy(path, os.path.join(target, name))
    plt.close(fig)
    print("wrote", path)


def by(rows, case):
    out = {}
    for r in rows:
        if r["case"] == case:
            out.setdefault(r["method"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["J"])
    return out


def legend_of_all(fig, axes):
    """One legend for every series drawn on any panel, above the panels."""
    seen = {}
    for ax in np.ravel(axes):
        for h, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, h)
    fig.legend(list(seen.values()), list(seen), loc="upper center",
               ncol=min(3, len(seen)), fontsize=8,
               bbox_to_anchor=(0.5, 1.06))


def scaling_figures(rows, case="shallow"):
    groups = by(rows, case)
    if not groups:
        return
    # training memory and time
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, (label, colour, marker, style) in METHODS.items():
        rs = groups.get(method, [])
        if not rs:
            continue
        J = [r["J"] for r in rs]
        axes[0].plot(J, [r["train_peak_mb"] for r in rs], color=colour,
                     marker=marker, linestyle=style, label=label)
        axes[1].plot(J, [r["train_seconds"] / r["epochs"] for r in rs],
                     color=colour, marker=marker, linestyle=style,
                     label=label)
    for ax, title in zip(axes, ("Peak device memory while training, MB",
                                "Seconds per epoch (tracing included)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("experts J (ground and data grow with J)")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "training_memory_time_%s.png" % case)

    # held-out scores against epochs, one column per J
    Js = sorted({r["J"] for rs in groups.values() for r in rs})
    fig, axes = plt.subplots(2, len(Js), figsize=(3.6 * len(Js), 5.6),
                             squeeze=False)
    for c, J in enumerate(Js):
        for method, (label, colour, marker, style) in METHODS.items():
            r = [r for r in groups.get(method, []) if r["J"] == J]
            if not r:
                continue
            curve = r[0]["curve"]
            e = [p["epoch"] for p in curve]
            for row, key in enumerate(("rmse", "crps")):
                axes[row, c].plot(e, [p[key] for p in curve], color=colour,
                                  marker=marker, linestyle=style,
                                  markersize=4, label=label)
        axes[0, c].set_title("J = %d" % J, loc="left", fontsize=10)
        axes[1, c].set_xlabel("epoch")
    axes[0, 0].set_ylabel("held-out rmse (noiseless field)")
    axes[1, 0].set_ylabel("held-out CRPS (measurements)")
    legend_of_all(fig, axes)
    save(fig, "heldout_curves_%s.png" % case)

    # the same against the seconds spent training
    fig, axes = plt.subplots(1, len(Js), figsize=(3.6 * len(Js), 3.2),
                             squeeze=False)
    for c, J in enumerate(Js):
        for method, (label, colour, marker, style) in METHODS.items():
            r = [r for r in groups.get(method, []) if r["J"] == J]
            if not r:
                continue
            curve = r[0]["curve"]
            axes[0, c].plot([p["seconds"] for p in curve],
                            [p["rmse"] for p in curve], color=colour,
                            marker=marker, linestyle=style, markersize=4,
                            label=label)
        axes[0, c].set_title("J = %d" % J, loc="left", fontsize=10)
        axes[0, c].set_xlabel("seconds training (tracing included)")
    axes[0, 0].set_ylabel("held-out rmse")
    legend_of_all(fig, axes)
    save(fig, "heldout_by_time_%s.png" % case)

    # prediction: memory, time and the truncation's cost
    predictions = [r for r in load("prediction.jsonl")
                   if r["case"] == case]
    predictions.sort(key=lambda r: r["J"])
    for target in ("points", "blocks"):
        rs = [r for r in predictions if target in r]
        if not rs:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        J = [r["J"] for r in rs]
        axes[0].plot(J, [r[target]["full_peak_mb"] for r in rs],
                     color="#2a78d6", marker="o", label="every expert")
        axes[1].plot(J, [r[target]["full_seconds"] for r in rs],
                     color="#2a78d6", marker="o", label="every expert")
        for g, colour, marker, style in (("home", "#eb6834", "s", "--"),
                                         ("exact", "#1baf7a", "^", ":")):
            sub = [r for r in rs if g in r[target]]
            if not sub:
                continue
            Jg = [r["J"] for r in sub]
            label = "by expert, %s grouping" % g
            axes[0].plot(Jg, [r[target][g]["part_peak_mb"] for r in sub],
                         color=colour, marker=marker, linestyle=style,
                         label=label)
            axes[1].plot(Jg, [r[target][g]["part_seconds"] for r in sub],
                         color=colour, marker=marker, linestyle=style,
                         label=label)
            axes[2].plot(Jg, [r[target][g]["diff_max"] for r in sub],
                         color=colour, marker=marker, linestyle=style,
                         label="largest |difference|, %s" % g)
            axes[2].plot(Jg, [r[target][g]["diff_mean"] for r in sub],
                         color=colour, marker=marker, linestyle=style,
                         alpha=0.45, label="mean |difference|, %s" % g)
        titles = ("Peak device memory, MB", "Seconds (tracing included)",
                  "By expert against every expert")
        for ax, title in zip(axes, titles):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("experts J")
            ax.set_title(title, loc="left", fontsize=10)
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=7)
        fig.suptitle("Prediction on %s, coverage 0.99 (model trained by "
                     "train_svi; at most 1%% of a location's weight left "
                     "out)" % target, x=0.01, ha="left", fontsize=10)
        save(fig, "prediction_%s_%s.png" % (target, case))

    # how many experts hold a point's weight
    rs = groups.get("svi", [])
    if rs:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        J = [r["J"] for r in rs]
        for level, colour, marker, style in ((0.9, "#2a78d6", "o", "-"),
                                             (0.99, "#eb6834", "s", "--"),
                                             (0.999, "#1baf7a", "^", ":")):
            key = "experts_for_%g" % level
            ax.plot(J, [r[key][0] for r in rs], color=colour, marker=marker,
                    linestyle=style, label="mean, %g of the weight" % level)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("experts J")
        ax.set_title("Experts holding a data point's weight", loc="left",
                     fontsize=10)
        ax.legend(fontsize=8)
        save(fig, "concentration_%s.png" % case)


def tom_figures(rows):
    groups = by(rows, "tom")
    if not groups:
        return
    Js = sorted({r["J"] for rs in groups.values() for r in rs})
    fig, axes = plt.subplots(2, len(Js), figsize=(3.8 * len(Js), 5.6),
                             squeeze=False)
    for c, J in enumerate(Js):
        for method, (label, colour, marker, style) in METHODS.items():
            r = [r for r in groups.get(method, []) if r["J"] == J]
            if not r:
                continue
            curve = r[0]["curve"]
            e = [p["epoch"] for p in curve]
            axes[0, c].plot(e, [p["balanced"] for p in curve], color=colour,
                            marker=marker, linestyle=style, markersize=4,
                            label=label)
            axes[1, c].plot(e, [p["brier"] for p in curve], color=colour,
                            marker=marker, linestyle=style, markersize=4,
                            label=label)
        axes[0, c].set_title("Tom, J = %d" % J, loc="left", fontsize=10)
        axes[1, c].set_xlabel("epoch")
    axes[0, 0].set_ylabel("held-out balanced accuracy")
    axes[1, 0].set_ylabel("held-out Brier score")
    legend_of_all(fig, axes)
    save(fig, "tom_heldout.png")


if __name__ == "__main__":
    rows = load("scaling.jsonl")
    scaling_figures(rows, "shallow")
    scaling_figures(rows, "deep")
    tom_figures(load("tom.jsonl"))
