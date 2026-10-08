"""Steps 1, 2, 4, 5 and 6 on the synthetic case, one run per process.

Usage: python scaling.py <n_experts> <svi|batch|epoch> <epochs> [every]

`svi` is the baseline, every expert at every step (`train_svi`); `batch`
and `epoch` are `train_by_expert` with the shared parameters stepped per
batch or once per epoch. Both use batches of 400 rows, and an epoch is
n_experts steps either way (the data hold 400 rows per expert). Every
`every` epochs the test points are predicted with every expert and
scored. After training: how concentrated the expert weights are, and
prediction with every expert against prediction by expert, on a grid of
points and on blocks. One JSON line is appended to results/scaling.jsonl.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

J, method, epochs = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
# "batch8" is the per-batch rule with eight visits per expert per epoch,
# each a batch of 400 / 8 rows
update = method.rstrip("0123456789")
visits = int(method[len(update):] or 1)
every = int(sys.argv[4]) if len(sys.argv) > 4 else 5
case = os.environ.get("BE_CASE", "shallow")


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


data, test, yt, truth = common.dataset(J)
m = common.model(data, J, deep=case == "deep",
                 propagation="independent" if case == "deep"
                 else "consensus")
out = dict(case=case, J=J, method=method, epochs=epochs, n=data.n_data,
           inducing=int(sum(m.leaves[0].root.n_ip)), curve=[])
rss0 = rss_mb()
train_seconds, train_peak = 0.0, 0.0
record = None
for chunk in range(epochs // every):
    common.gpu_peak_reset()
    start = time.perf_counter()
    if method == "svi":
        m.train_svi(every)
    else:
        record = m.train_by_expert(every, batch_size=400 // visits,
                                   global_update=update, visits=visits)
    train_seconds += time.perf_counter() - start
    train_peak = max(train_peak, common.gpu_peak_mb())
    t = geoml.data.PointData.from_array(np.asarray(test.coordinates),
                                        ["X", "Y"])
    m.predict(t, n_sim=50)
    out["curve"].append(dict(epoch=(chunk + 1) * every,
                             seconds=train_seconds, **common.scores(t, yt,
                                                                    truth)))
    print(out["curve"][-1], flush=True)
out.update(train_seconds=train_seconds, train_peak_mb=train_peak,
           rss_growth_mb=rss_mb() - rss0)
if record is not None:
    out.update(mean_subset=float(np.mean([len(s) for s in
                                          record["subsets"]])),
               max_subset=int(max(len(s) for s in record["subsets"])),
               distinct_subsets=len(set(record["subsets"])),
               dropped_mean=float(np.mean(record["dropped"])),
               dropped_max=float(np.max(record["dropped"])),
               traced_steps=len(m._by_expert["steps"]))

# step 2: how many experts hold a point's weight
start = time.perf_counter()
table = m.expert_weights()
out["table_seconds"] = time.perf_counter() - start
ordered = -np.sort(-table, axis=1)
for level in (0.9, 0.99, 0.999):
    k = (np.cumsum(ordered, axis=1) < level).sum(axis=1) + 1
    out["experts_for_%g" % level] = [float(k.mean()), int(k.max())]
print("concentration", {k: v for k, v in out.items()
                        if k.startswith("experts_for")}, flush=True)

# step 6: prediction with every expert against prediction by expert
L = common.side(J)
targets = dict(
    points=lambda: geoml.data.Grid2D(start=[1.25, 1.25],
                                     n=[int(L / 2.5)] * 2, step=[2.5, 2.5]),
    blocks=lambda: geoml.data.Blocks2D(start=[5.0, 5.0], n=[int(L / 10)] * 2,
                                       step=[10.0, 10.0],
                                       discretization=[3, 3]))
for name, build in targets.items():
    full = build()
    common.gpu_peak_reset()
    start = time.perf_counter()
    m.predict(full, n_sim=20)
    t_full, p_full = time.perf_counter() - start, common.gpu_peak_mb()
    a = np.asarray(full.values("v/prediction"))
    sa = full.variables["v"].get_simulations()
    out[name] = dict(locations=full.n_data, full_seconds=t_full,
                     full_peak_mb=p_full)
    for grouping in ("home", "exact") if J <= 16 else ("home",):
        part = build()
        common.gpu_peak_reset()
        start = time.perf_counter()
        info = m.predict_by_expert(part, n_sim=20, coverage=0.99,
                                   grouping=grouping)
        t_part, p_part = time.perf_counter() - start, common.gpu_peak_mb()
        b = np.asarray(part.values("v/prediction"))
        sb = part.variables["v"].get_simulations()
        out[name][grouping] = dict(
            part_seconds=t_part, part_peak_mb=p_part,
            groups=len(info["subsets"]),
            mean_set=float(np.average([len(s) for s in info["subsets"]],
                                      weights=info["sizes"])),
            left_out_max=float(np.max(info["left_out"])),
            left_out_mean=float(np.mean(info["left_out"])),
            diff_max=float(np.max(np.abs(a - b))),
            diff_mean=float(np.mean(np.abs(a - b))),
            sims_diff_max=float(np.max(np.abs(sa - sb))))
    print(name, out[name], flush=True)

# the test points by expert too, scored
t = geoml.data.PointData.from_array(np.asarray(test.coordinates), ["X", "Y"])
m.predict_by_expert(t, n_sim=50, coverage=0.99)
out["final_by_expert"] = common.scores(t, yt, truth)
out["rss_end_mb"] = rss_mb()
os.makedirs("experiments/batched_experts/results", exist_ok=True)
with open("experiments/batched_experts/results/scaling.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("done", json.dumps({k: v for k, v in out.items() if k != "curve"}),
      flush=True)
