"""Consensus propagation under a subset, on the tree that concatenates the
coordinates: the second GP's inducing inputs are the first GP's outputs
at the inducing points, blended over every expert's opinion -- by expert,
over the active experts' only. How far does that move them, and the
prediction?

Usage: python probe_consensus.py <n_experts> <epochs>
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402
from geoml.latent import network  # noqa: E402

J, epochs = int(sys.argv[1]), int(sys.argv[2])
data, test, yt, truth = common.dataset(J)
m = common.model(data, J, deep="concat", propagation="consensus")
start = time.perf_counter()
m.train_svi(epochs)
seconds = time.perf_counter() - start
first = [n for n in m._nodes() if isinstance(n, network.BasicGP)
         and n.parent is n.root][0]
J1 = first.root.n_experts

with geoml.latent.propagation_rule("consensus"):
    m._refresh(m.options.jitter)
    full = [np.asarray(p) for p in first.inducing_points]
    spread = float(np.std(np.concatenate(full)))
    table = m.expert_weights()
    measured = np.any(m.has_value > 0, axis=1)
    w = table * measured[:, None]
    sets, _ = m._expert_subsets(w.T @ w, 0.99)
    worst, mean = [], []
    for subset in sets:
        size = len(subset)
        slots = network._Slots(size, tf.constant(list(subset), tf.int32),
                               tf.ones(size, tf.float64))
        with network.expert_slots(slots):
            m._refresh(m.options.jitter)
            points = np.asarray(first.slots_points)
            pmask = np.asarray(first.slots_point_mask)
        for p, k in enumerate(subset):
            n = int(pmask[p].sum())
            d = np.abs(points[p, :n] - full[k])
            worst.append(float(d.max()))
            mean.append(float(d.mean()))

t_full = geoml.data.PointData.from_array(np.asarray(test.coordinates),
                                         ["X", "Y"])
m.predict(t_full, n_sim=20)
t_part = geoml.data.PointData.from_array(np.asarray(test.coordinates),
                                         ["X", "Y"])
m.predict_by_expert(t_part, n_sim=20)
a = np.asarray(t_full.values("v/prediction"))
b = np.asarray(t_part.values("v/prediction"))
out = dict(J=J, epochs=epochs, train_seconds=seconds,
           first_layer_sd=spread,
           inducing_shift_max=float(np.max(worst)),
           inducing_shift_mean=float(np.mean(mean)),
           prediction_diff_max=float(np.max(np.abs(a - b))),
           prediction_diff_mean=float(np.mean(np.abs(a - b))),
           full=common.scores(t_full, yt, truth),
           by_expert=common.scores(t_part, yt, truth),
           mean_set=float(np.mean([len(s) for s in sets])))
print("done", json.dumps(out), flush=True)
os.makedirs("experiments/batched_experts/results", exist_ok=True)
with open("experiments/batched_experts/results/consensus.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
