"""Figures of the second round, from results/v2.jsonl and scaling.jsonl:
held-out scores against training time at 64 experts, every expert against
by expert in slots under the two schedules, seed 1, each run alone on the
GPU. Same palette and helpers as figures.py."""
import json
import os

import figures as f
import matplotlib.pyplot as plt

R = os.path.join(f.HERE, "results")
runs = [json.loads(line) for line in open(os.path.join(R, "v2.jsonl"))]
old = [json.loads(line) for line in open(os.path.join(R, "scaling.jsonl"))]


def pick(case, J, method, epochs, decay, clock):
    hits = [r for r in runs if r["case"] == case and r["J"] == J
            and r["method"] == method and r["epochs"] == epochs
            and r["seed"] == 20261006 and r.get("decay") == decay
            and r.get("clock") == clock and r["train_peak_mb"] > 0]
    return hits[-1]


svi = [r for r in old if r["J"] == 64 and r["method"] == "svi"][-1]
lines = [
    (svi, "every expert each step (train_svi)", "#2a78d6", "o", "-"),
    (pick("shallow", 64, "epoch4", 60, None, None),
     "by expert in slots, steps counted", "#eda100", "D", "-."),
    (pick("shallow", 64, "epoch4", 60, None, "shared"),
     "by expert in slots, both on train_svi's clock", "#1baf7a", "^", ":"),
]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, key, label in zip(axes, ("rmse", "crps"),
                          ("held-out rmse", "held-out CRPS")):
    for r, name, color, marker, style in lines:
        xs = [c["seconds"] for c in r["curve"]]
        ys = [c[key] for c in r["curve"]]
        ax.plot(xs, ys, color=color, marker=marker, linestyle=style,
                label=name)
    ax.set_xlabel("training seconds")
    ax.set_ylabel(label)
    ax.set_ylim(top=0.2 if key == "rmse" else 0.18)
axes[0].legend(loc="upper right", fontsize=8.5)
fig.suptitle("64 experts, seed 1: device peak 1101 MB every expert, "
             "264-333 MB by expert", color=f.INK2, fontsize=10)
f.save(fig, "round2_heldout_by_time_64.png")
