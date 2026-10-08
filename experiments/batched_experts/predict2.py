"""Prediction in slots on the saved every-expert models (prediction.py
trained them): a grid of points and blocks, every expert against by expert
in slots, packed and not.

Usage: python predict2.py <n_experts>
One JSON line to results/prediction2.jsonl.
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
m = geoml.persistence.load_model("local_data/models/shallow_%d_svi" % J)
L = common.side(J)
targets = dict(
    points=lambda: geoml.data.Grid2D(start=[1.25, 1.25],
                                     n=[int(L / 2.5)] * 2, step=[2.5, 2.5]),
    blocks=lambda: geoml.data.Blocks2D(start=[5.0, 5.0], n=[int(L / 10)] * 2,
                                       step=[10.0, 10.0],
                                       discretization=[3, 3]))
out = dict(case="shallow", J=J)
for name, build in targets.items():
    full = build()
    common.gpu_peak_reset()
    start = time.perf_counter()
    m.predict(full, n_sim=20)
    t_full, p_full = time.perf_counter() - start, common.gpu_peak_mb()
    a = np.asarray(full.values("v/prediction"))
    out[name] = dict(locations=full.n_data, full_seconds=t_full,
                     full_peak_mb=p_full)
    for label, slots, pack in (("packed", True, True),
                               ("unpacked", True, False),
                               ("ten_slots", 10, True)):
        part = build()
        common.gpu_peak_reset()
        start = time.perf_counter()
        info = m.predict_by_expert(part, n_sim=20, coverage=0.99,
                                   slots=slots, pack=pack)
        t_part, p_part = time.perf_counter() - start, common.gpu_peak_mb()
        b = np.asarray(part.values("v/prediction"))
        out[name][label] = dict(
            part_seconds=t_part, part_peak_mb=p_part,
            groups=len(info["subsets"]), slots=info["slots"],
            left_out_max=float(np.nanmax(info["left_out"])),
            over_coverage=float(np.mean(info["left_out"] > 0.0101)),
            diff_max=float(np.max(np.abs(a - b))),
            diff_mean=float(np.mean(np.abs(a - b))))
    print(name, out[name], flush=True)
os.makedirs("experiments/batched_experts/results", exist_ok=True)
with open("experiments/batched_experts/results/prediction2.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("done", json.dumps(out), flush=True)
