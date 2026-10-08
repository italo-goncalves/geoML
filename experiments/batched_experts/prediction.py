"""Step 6: prediction with every expert against prediction by expert.

Usage: python prediction.py <n_experts> [case]

The model is the step-1 baseline (train_svi, the same epochs as
scaling.py), trained once and saved under local_data/models/, reloaded
afterwards. Targets: a grid of points every 2.5 units and blocks of 10
units with 3x3 sub-blocks over the whole ground. Groupings: "home" always,
"exact" up to 16 experts. One JSON line to results/prediction.jsonl.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

J = int(sys.argv[1])
case = sys.argv[2] if len(sys.argv) > 2 else "shallow"
EPOCHS = {4: 200, 16: 60, 64: 20}
path = "local_data/models/%s_%d_svi" % (case, J)
data, test, yt, truth = common.dataset(J)
if os.path.exists(path):
    m = geoml.persistence.load_model(path)
else:
    m = common.model(data, J, deep=case == "deep",
                     propagation="independent" if case == "deep"
                     else "consensus")
    m.train_svi(EPOCHS[J])
    os.makedirs("local_data/models", exist_ok=True)
    m.save(path)

L = common.side(J)
targets = dict(
    points=lambda: geoml.data.Grid2D(start=[1.25, 1.25],
                                     n=[int(L / 2.5)] * 2, step=[2.5, 2.5]),
    blocks=lambda: geoml.data.Blocks2D(start=[5.0, 5.0], n=[int(L / 10)] * 2,
                                       step=[10.0, 10.0],
                                       discretization=[3, 3]))
out = dict(case=case, J=J)
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
        print(name, grouping, out[name][grouping], flush=True)
    print(name, "full %.1f s, %.0f MB" % (t_full, p_full), flush=True)
os.makedirs("experiments/batched_experts/results", exist_ok=True)
with open("experiments/batched_experts/results/prediction.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("done", flush=True)
