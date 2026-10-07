"""The real case: Tom's rock type (ore against waste) on 20 126 composites.

Usage: python tom.py <n_experts> <svi|batch|epoch> <epochs> [every]

A fifth of the holes are held out (chosen by seed). One BasicGP of two
outputs on an isotropic input, 150 inducing points per expert from
k-means of the training composites, `CategoricalGaussianIndicator(2)`.
Scores on the held-out composites: accuracy, balanced accuracy and the
Brier score of the probability of ore. Then prediction with every expert
against prediction by expert on blocks of 50 m with 2x2x2 sub-blocks over
the deposit. One JSON line is appended to results/tom.jsonl.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402
import geoml.latent as latent  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.transform as tr  # noqa: E402

J, method, epochs = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
update = method.rstrip("0123456789")
visits = int(method[len(update):] or 1)
every = int(sys.argv[4]) if len(sys.argv) > 4 else 5

npz = np.load("local_data/tom_composites.npz")
coords, rock, holes = npz["coords"], npz["rock"], npz["holes"]
rng = np.random.default_rng(common.SEED)
names = np.unique(holes)
held_holes = rng.choice(names, len(names) // 5, replace=False)
held = np.isin(holes, held_holes)


def container(rows):
    c = geoml.data.PointData.from_array(coords[rows], ["X", "Y", "Z"])
    c.add_categorical_variable("Rock", labels=["Waste", "Ore"],
                               measurements=rock[rows])
    return c


data = container(~held)
truth = rock[held] == "Ore"

geoml.set_seed(common.SEED)
ip = geoml.data.inducing.from_kmeans(data, 150 * J, seed=0)
experts = geoml.data.inducing.experts(ip, J, overlap=0.1, seed=0)
root = latent.BasicInput(experts, transform=tr.Isotropic(50.0))
leaf = latent.BasicGP(root, size=2, kernel=geoml.kernels.Matern32())
m = geoml.models.VGPNetwork(
    data, "Rock", lk.CategoricalGaussianIndicator(2), leaf,
    options=geoml.models.GPOptions(verbose=False, training_batch_size=400,
                                   prediction_batch_size=5000))


def scores():
    t = geoml.data.PointData.from_array(coords[held], ["X", "Y", "Z"])
    m.predict(t, n_sim=20)
    p = np.asarray(t.variables["Rock"].components["Ore"].probability.values)
    call = p > 0.5
    recall = [np.mean(call[truth]), np.mean(~call[~truth])]
    return dict(accuracy=float(np.mean(call == truth)),
                balanced=float(np.mean(recall)),
                brier=float(np.mean((p - truth) ** 2)))


out = dict(case="tom", J=J, method=method, epochs=epochs, n=data.n_data,
           held_out=int(held.sum()), inducing=int(sum(root.n_ip)), curve=[])
train_seconds, train_peak, record = 0.0, 0.0, None
saved = "local_data/models/tom_%d_%s" % (J, method)
predict_only = os.path.exists(saved)
if predict_only:
    # the trained model kept from an earlier run: prediction only
    m = geoml.persistence.load_model(saved)
    out["curve"].append(dict(epoch=epochs, seconds=float("nan"),
                             **scores()))
for chunk in range(0 if predict_only else epochs // every):
    common.gpu_peak_reset()
    start = time.perf_counter()
    if method == "svi":
        m.train_svi(every)
    else:
        # J * visits batches of N / (J * visits) rows: an epoch sees the
        # data once, as train_svi's does in batches of 400
        record = m.train_by_expert(
            every, batch_size=data.n_data // (J * visits),
            global_update=update, visits=visits)
    train_seconds += time.perf_counter() - start
    train_peak = max(train_peak, common.gpu_peak_mb())
    out["curve"].append(dict(epoch=(chunk + 1) * every,
                             seconds=train_seconds, **scores()))
    print(out["curve"][-1], flush=True)
if not predict_only:
    os.makedirs("local_data/models", exist_ok=True)
    m.save(saved)
out.update(train_seconds=train_seconds, train_peak_mb=train_peak,
           predict_only=predict_only)
if record is not None:
    out.update(mean_subset=float(np.mean([len(s) for s in
                                          record["subsets"]])),
               distinct_subsets=len(set(record["subsets"])),
               dropped_mean=float(np.mean(record["dropped"])),
               dropped_max=float(np.max(record["dropped"])))

table = m.expert_weights()
ordered = -np.sort(-table, axis=1)
for level in (0.9, 0.99, 0.999):
    k = (np.cumsum(ordered, axis=1) < level).sum(axis=1) + 1
    out["experts_for_%g" % level] = [float(k.mean()), int(k.max())]
print("concentration", {k: v for k, v in out.items()
                        if k.startswith("experts_for")}, flush=True)

lo, hi = coords.min(axis=0), coords.max(axis=0)
n = np.ceil((hi - lo) / 50.0).astype(int)


def blocks():
    return geoml.data.Blocks3D(start=lo + 25.0, n=n, step=[50.0] * 3,
                               discretization=[2, 2, 2])


full, part = blocks(), blocks()
common.gpu_peak_reset()
start = time.perf_counter()
m.predict(full, n_sim=20)
t_full, p_full = time.perf_counter() - start, common.gpu_peak_mb()
common.gpu_peak_reset()
start = time.perf_counter()
info = m.predict_by_expert(part, n_sim=20, coverage=0.99)
t_part, p_part = time.perf_counter() - start, common.gpu_peak_mb()
a = np.asarray(full.variables["Rock"].components["Ore"].probability.values)
b = np.asarray(part.variables["Rock"].components["Ore"].probability.values)
out["blocks"] = dict(locations=full.n_data, full_seconds=t_full,
                     full_peak_mb=p_full, part_seconds=t_part,
                     part_peak_mb=p_part, groups=len(info["subsets"]),
                     mean_set=float(np.average(
                         [len(s) for s in info["subsets"]],
                         weights=info["sizes"])),
                     left_out_max=float(np.max(info["left_out"])),
                     diff_max=float(np.max(np.abs(a - b))),
                     diff_mean=float(np.mean(np.abs(a - b))),
                     calls_changed=float(np.mean((a > 0.5) != (b > 0.5))))
print("blocks", out["blocks"], flush=True)
os.makedirs("experiments/batched_experts/results", exist_ok=True)
name = "tom_prediction.jsonl" if predict_only else "tom.jsonl"
with open("experiments/batched_experts/results/" + name, "a") as f:
    f.write(json.dumps(out) + "\n")
print("done", json.dumps({k: v for k, v in out.items() if k != "curve"}),
      flush=True)
