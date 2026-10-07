"""The open items' tables, from results/v2.jsonl, tom2.jsonl,
prediction2.jsonl and consensus.jsonl: the replicated comparisons as mean
and spread over seeds, and the single runs as they stand."""
import json

import numpy as np

R = "experiments/batched_experts/results/"


def rows(name):
    try:
        return [json.loads(line) for line in open(R + name)]
    except FileNotFoundError:
        return []


def at(curve, seconds):
    """The score on a curve at the last point no later than `seconds`."""
    past = [c for c in curve if c["seconds"] <= seconds * 1.02]
    return past[-1] if past else None


def summary(label, runs, keys):
    if not runs:
        return
    seeds = sorted({r["seed"] for r in runs})
    out = ["%-38s n=%d" % (label, len(runs))]
    for k in keys:
        v = np.array([k(r) for r in runs], dtype=float)
        out.append("%s %.3f +- %.3f" % (k.__name__, v.mean(), v.std(ddof=1)
                                        if len(v) > 1 else 0.0))
    print("  ".join(out), "seeds", seeds)


def rmse(r):
    """Of the last prediction with every expert, which scores training."""
    return r["curve"][-1]["rmse"]


def crps(r):
    return r["curve"][-1]["crps"]


def seconds(r):
    return r["train_seconds"]


def schedule(r):
    """How a run's learning rates decayed. A run made before the option
    existed has no `decay` and counted steps; `decay` None or `"epochs"`
    before `clock="shared"` was recorded put the experts alone on
    train_svi's clock, after it both."""
    if r.get("decay", "steps") == "steps":
        return "steps"
    return "both clocks" if r.get("clock") == "shared" else "experts' clock"


def by_default(r):
    """Trained by expert on the final defaults: slots, steps counted."""
    return r["method"] != "svi" and r.get("slots", True) \
        and schedule(r) == "steps"


def latest(runs):
    """The last run of each seed: a seed run again on the same schedule
    (re-formed sets, say) replaces its earlier run."""
    return list({r["seed"]: r for r in runs}.values())


v2 = rows("v2.jsonl")
# the seed-1 run with every expert at 64 experts is the earlier one: the
# path is unchanged, and 15 minutes are not spent repeating it
for r in rows("scaling.jsonl"):
    if r["case"] == "shallow" and r["J"] == 64 and r["method"] == "svi":
        v2.append(dict(r, seed=20261006))
print("== synthetic, shallow")
for J, epochs in ((16, 60), (64, 20), (64, 60)):
    summary("J=%d every expert, %d epochs" % (J, epochs),
            [r for r in v2 if r["case"] == "shallow" and r["J"] == J
             and r["method"] == "svi" and r["epochs"] == epochs],
            [rmse, crps, seconds])
    for kind in ("steps", "experts' clock", "both clocks"):
        summary("J=%d by expert epoch4, %d epochs, %s" % (J, epochs, kind),
                latest([r for r in v2 if r["case"] == "shallow"
                        and r["J"] == J and r["method"] == "epoch4"
                        and r["epochs"] == epochs and r.get("batch") == 100
                        and r.get("slots", True) and schedule(r) == kind]),
                [rmse, crps, seconds])

t2 = rows("tom2.jsonl")


def auc(r):
    return r["final_by_expert"]["auc"]


def brier(r):
    return r["final_by_expert"]["brier"]


def balanced(r):
    return r["final_by_expert"]["balanced"]


def at_share(r):
    return r["final_by_expert"]["balanced_at_share"]


def mean_p(r):
    return r["final_by_expert"]["mean_p"]


print("== Tom")
for J in (10, 40):
    for method in ("svi", "epoch4"):
        kinds = ("",) if method == "svi" else             ("steps", "experts' clock", "both clocks")
        for kind in kinds:
            summary("Tom J=%d %s %s" % (J, method, kind),
                    latest([r for r in t2 if r["J"] == J
                            and r["method"] == method
                            and (method == "svi" or schedule(r) == kind)]),
                    [auc, brier, balanced, at_share, mean_p, seconds])

print("== prediction on the saved every-expert models")
for r in rows("prediction2.jsonl"):
    for target in ("points", "blocks"):
        t = r[target]
        print("J=%d %s every expert %.1f s %.0f MB" % (
            r["J"], target, t["full_seconds"], t["full_peak_mb"]))
        for label in ("packed", "unpacked", "ten_slots"):
            if label in t:
                p = t[label]
                print("   %-9s %4d groups %2d slots %6.1f s %5.0f MB "
                      "over coverage %.3f diff max %.3f mean %.4f" % (
                          label, p["groups"], p["slots"], p["part_seconds"],
                          p["part_peak_mb"], p["over_coverage"],
                          p["diff_max"], p["diff_mean"]))

print("== consensus under subsets")
for r in rows("consensus.jsonl"):
    print(r)

print("== Tom, a fixed total of inducing points split among the experts")
# the last run of each configuration (a Tom run that ignored the
# partition, before tom2.py passed it on, repeated one exactly)
split = {(r["total"], r["J"], r["method"], r.get("sampling", "replacement")):
         r for r in t2 if "total" in r and r["seed"] == 20261006
         and r["train_peak_mb"] > 0}   # timed alone on the GPU
for key in sorted(split):
    r = split[key]
    f = r["final_by_expert"]
    b = r.get("blocks", {})
    print("total %4d J=%2d (%3d each) %-7s %-11s auc %.3f brier %.3f "
          "balanced %.3f at share %.3f | train %4.0f s %5.0f MB | blocks "
          "%5.1f s %5.0f MB by expert %5.1f s %5.0f MB, %3s groups, %2s "
          "slots" % (
              r["total"], r["J"], r["total"] // r["J"], r["method"],
              r.get("sampling", "replacement"),
              f["auc"], f["brier"], f["balanced"], f["balanced_at_share"],
              r["train_seconds"], r["train_peak_mb"],
              b.get("full_seconds", float("nan")),
              b.get("full_peak_mb", float("nan")),
              b.get("part_seconds", float("nan")),
              b.get("part_peak_mb", float("nan")), b.get("groups"),
              b.get("slots")))
