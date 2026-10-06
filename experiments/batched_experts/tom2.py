"""Tom's rock type again, for the open items: the round update in slots,
why training by expert called less ore, prediction on blocks in slots, and
replication.

Usage: python tom2.py <n_experts> <method> <epochs> <every> [seed]

As tom.py: a fifth of the holes held out (fixed by common.SEED), one
BasicGP of two outputs, 150 inducing points per expert,
`CategoricalGaussianIndicator(2)`. `method` is `svi` or `<update><visits>`
(slots; `-sets` for a trace per set); a batch is N / (J * visits) rows so
that an epoch reads the data once. `seed` seeds the model. Scored on the
held-out composites, with what tells calibration from ranking apart: the
mean probability of ore against its share, the area under the ROC curve,
and balanced accuracy at 0.5 and at the training share of ore. Then blocks
of 50 m with 2x2x2 sub-blocks, every expert against by expert. One JSON
line to results/tom2.jsonl.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402
import geoml.latent as latent  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.transform as tr  # noqa: E402

J, method, epochs, every = (int(sys.argv[1]), sys.argv[2], int(sys.argv[3]),
                            int(sys.argv[4]))
seed = int(sys.argv[5]) if len(sys.argv) > 5 else common.SEED
slots = not method.endswith("-sets")
core = method[:-len("-sets")] if not slots else method
update = core.rstrip("0123456789")
visits = int(core[len(update):] or 1)
decay = os.environ.get("BE_DECAY") or "steps"

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
share = float(np.mean(rock[~held] == "Ore"))

geoml.set_seed(seed)
ip = geoml.data.inducing.from_kmeans(data, 150 * J, seed=0)
experts = geoml.data.inducing.experts(ip, J, overlap=0.1, seed=0)
root = latent.BasicInput(experts, transform=tr.Isotropic(50.0))
leaf = latent.BasicGP(root, size=2, kernel=geoml.kernels.Matern32())
m = geoml.models.VGPNetwork(
    data, "Rock", lk.CategoricalGaussianIndicator(2), leaf,
    options=geoml.models.GPOptions(verbose=False, training_batch_size=400,
                                   prediction_batch_size=5000))


def diagnose(p):
    def balanced(cut):
        call = p > cut
        return float(np.mean([np.mean(call[truth]), np.mean(~call[~truth])]))

    return dict(balanced=balanced(0.5), balanced_at_share=balanced(share),
                brier=float(np.mean((p - truth) ** 2)),
                auc=float(roc_auc_score(truth, p)),
                mean_p=float(np.mean(p)), share_held=float(np.mean(truth)),
                ore_calls=float(np.mean(p > 0.5)),
                sharpness=float(np.mean(np.abs(p - 0.5))))


def scores(by_expert=False):
    t = geoml.data.PointData.from_array(coords[held], ["X", "Y", "Z"])
    if by_expert:
        m.predict_by_expert(t, n_sim=20)
    else:
        m.predict(t, n_sim=20)
    return diagnose(np.asarray(
        t.variables["Rock"].components["Ore"].probability.values))


out = dict(case="tom", J=J, method=method, epochs=epochs, seed=seed, decay=decay,
           clock="shared",
           n=data.n_data, held_out=int(held.sum()), share_train=share,
           curve=[])
train_seconds, train_peak, traces, record = 0.0, 0.0, 0, None
for chunk in range(epochs // every):
    common.gpu_peak_reset()
    start = time.perf_counter()
    if method == "svi":
        m.train_svi(every)
    else:
        record = m.train_by_expert(
            every, batch_size=data.n_data // (J * visits),
            global_update=update, visits=visits, slots=slots, decay=decay)
        traces = max(traces, record["traces"])
    train_seconds += time.perf_counter() - start
    train_peak = max(train_peak, common.gpu_peak_mb())
    out["curve"].append(dict(epoch=(chunk + 1) * every,
                             seconds=train_seconds, **scores()))
    print("epoch", out["curve"][-1], flush=True)
out.update(train_seconds=train_seconds, train_peak_mb=train_peak,
           traces=traces, final_by_expert=scores(by_expert=True))
print("diagnosis by expert", out["final_by_expert"], flush=True)
if record is not None:
    out.update(mean_subset=float(np.mean([len(s) for s in
                                          record["subsets"]])),
               max_subset=int(max(len(s) for s in record["subsets"])))
out["ranges"] = np.ravel(leaf.parameters["ranges"].get_value()).tolist()
out["transform_range"] = [float(np.ravel(p.get_value())[0])
                          for p in root.transform.all_parameters]

if os.environ.get("BE_BLOCKS", "1") == "1":
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
                         slots=info["slots"],
                         left_out_max=float(np.nanmax(info["left_out"])),
                         diff_max=float(np.max(np.abs(a - b))),
                         calls_changed=float(np.mean((a > 0.5) != (b > 0.5))))
    print("blocks", out["blocks"], flush=True)
os.makedirs("experiments/batched_experts/results", exist_ok=True)
with open("experiments/batched_experts/results/tom2.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("done", json.dumps({k: v for k, v in out.items() if k != "curve"}),
      flush=True)
