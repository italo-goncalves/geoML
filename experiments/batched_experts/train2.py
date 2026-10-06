"""The open items, on the synthetic case: one trace per run, the round
update, steps per expert, deep trees through the coordinates, replication.

Usage: python train2.py <case> <n_experts> <method> <epochs> <every> [seed]

`case` is `shallow`, `concat` (a GP on the coordinates and the first GP)
or `deep` (a GP on the first GP alone). `method` is `svi` (every expert at
every step, batches of 400) or `<update><visits>` -- `round4`, `epoch4`,
`batch1` -- trained by expert in slots, `-sets` appended for a trace per
set. By expert, a batch is 400 / visits rows unless `BE_BATCH` says
otherwise. Every `every` epochs the test points are predicted with every
expert and scored. With `BE_PREDICT=1` the trained model then predicts a
grid of points and blocks by expert (slots, packed) against every expert.
One JSON line to results/v2.jsonl.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

case, J, method = sys.argv[1], int(sys.argv[2]), sys.argv[3]
epochs, every = int(sys.argv[4]), int(sys.argv[5])
seed = int(sys.argv[6]) if len(sys.argv) > 6 else common.SEED
slots = not method.endswith("-sets")
core = method[:-len("-sets")] if not slots else method
update = core.rstrip("0123456789")
visits = int(core[len(update):] or 1)
batch = int(os.environ.get("BE_BATCH", 400 // visits))
decay = os.environ.get("BE_DECAY") or "steps"


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


data, test, yt, truth = common.dataset(J, seed=seed)
deep = {"shallow": False, "concat": "concat", "deep": True}[case]
m = common.model(data, J, deep=deep, seed=seed,
                 propagation="consensus" if case == "shallow"
                 else "independent")
out = dict(case=case, J=J, method=method, epochs=epochs, seed=seed, decay=decay,
           clock="shared",
           batch=batch if method != "svi" else 400, slots=slots,
           n=data.n_data, curve=[])


def ranges():
    """The trained ranges of every GP node, first to last."""
    nodes = [n for n in m._nodes()
             if isinstance(n, geoml.latent.network.BasicGP)]
    return [np.ravel(n.parameters["ranges"].get_value()).tolist()
            for n in nodes]


rss0 = rss_mb()
train_seconds, train_peak, traces = 0.0, 0.0, 0
record = None
for chunk in range(epochs // every):
    common.gpu_peak_reset()
    start = time.perf_counter()
    if method == "svi":
        m.train_svi(every)
    else:
        record = m.train_by_expert(every, batch_size=batch,
                                   global_update=update, visits=visits,
                                   slots=slots, decay=decay)
        traces = max(traces, record["traces"])
    train_seconds += time.perf_counter() - start
    train_peak = max(train_peak, common.gpu_peak_mb())
    t = geoml.data.PointData.from_array(np.asarray(test.coordinates),
                                        ["X", "Y"])
    m.predict(t, n_sim=50)
    out["curve"].append(dict(epoch=(chunk + 1) * every,
                             seconds=train_seconds, ranges=ranges(),
                             **common.scores(t, yt, truth)))
    print("epoch", {k: v for k, v in out["curve"][-1].items()
                    if k != "ranges"}, flush=True)
out.update(train_seconds=train_seconds, train_peak_mb=train_peak,
           rss_growth_mb=rss_mb() - rss0, traces=traces, ranges=ranges())
# the shared parameters at the end: kernel, transform, noise, warping
def named(obj, prefix):
    return [[prefix + k, np.ravel(v.get_value()).tolist()]
            for k, v in obj.parameters.items()
            if not k.startswith(("alpha_white", "delta_", "bias_"))
            and np.size(v.get_value()) <= 4]


lik = m.likelihoods[0]
out["shared"] = (named(lik, "likelihood/")
                 + named(lik.warping, "warping/")
                 + named(m.leaves[0].root.transform, "transform/")
                 + [x for n in m._nodes()
                    if isinstance(n, geoml.latent.network.BasicGP)
                    for x in named(n, n.name + "/")])
if record is not None:
    out.update(mean_subset=float(np.mean([len(s) for s in
                                          record["subsets"]])),
               max_subset=int(max(len(s) for s in record["subsets"])),
               dropped_mean=float(np.mean(record["dropped"])),
               dropped_max=float(np.max(record["dropped"])))

table = m.expert_weights()
ordered = -np.sort(-table, axis=1)
for level in (0.9, 0.99):
    k = (np.cumsum(ordered, axis=1) < level).sum(axis=1) + 1
    out["experts_for_%g" % level] = [float(k.mean()), int(k.max())]
print("concentration", {k: v for k, v in out.items()
                        if k.startswith("experts_for")}, flush=True)

if os.environ.get("BE_PREDICT") == "1":
    L = common.side(J)
    targets = dict(
        points=lambda: geoml.data.Grid2D(start=[1.25, 1.25],
                                         n=[int(L / 2.5)] * 2,
                                         step=[2.5, 2.5]),
        blocks=lambda: geoml.data.Blocks2D(start=[5.0, 5.0],
                                           n=[int(L / 10)] * 2,
                                           step=[10.0, 10.0],
                                           discretization=[3, 3]))
    for name, build in targets.items():
        full = build()
        common.gpu_peak_reset()
        start = time.perf_counter()
        m.predict(full, n_sim=20)
        t_full, p_full = time.perf_counter() - start, common.gpu_peak_mb()
        a = np.asarray(full.values("v/prediction"))
        part = build()
        common.gpu_peak_reset()
        start = time.perf_counter()
        info = m.predict_by_expert(part, n_sim=20, coverage=0.99)
        t_part, p_part = time.perf_counter() - start, common.gpu_peak_mb()
        b = np.asarray(part.values("v/prediction"))
        out[name] = dict(locations=full.n_data, full_seconds=t_full,
                         full_peak_mb=p_full, part_seconds=t_part,
                         part_peak_mb=p_part, groups=len(info["subsets"]),
                         slots=info["slots"],
                         left_out_max=float(np.nanmax(info["left_out"])),
                         diff_max=float(np.max(np.abs(a - b))),
                         diff_mean=float(np.mean(np.abs(a - b))))
        print(name, out[name], flush=True)

t = geoml.data.PointData.from_array(np.asarray(test.coordinates), ["X", "Y"])
m.predict_by_expert(t, n_sim=50, coverage=0.99)
out["final_by_expert"] = common.scores(t, yt, truth)
out["rss_end_mb"] = rss_mb()
os.makedirs("experiments/batched_experts/results", exist_ok=True)
with open("experiments/batched_experts/results/v2.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("done", json.dumps({k: v for k, v in out.items()
                          if k not in ("curve", "ranges")}), flush=True)
