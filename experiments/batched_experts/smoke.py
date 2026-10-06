"""Do the two methods run end to end, and does the by-expert prediction
with a coverage of one reproduce `predict`?"""
import sys

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402

J = 4
data, test, yt, truth = common.dataset(J)
m = common.model(data, J)
record = m.train_by_expert(epochs=2, batch_size=200)
print("bound", record["bound"], "seconds", np.round(record["seconds"], 1))
print("subsets", record["subsets"], "dropped", np.round(record["dropped"], 4))

full = test.copy() if hasattr(test, "copy") else None
import geoml  # noqa: E402
a = geoml.data.PointData.from_array(np.asarray(test.coordinates), ["X", "Y"])
b = geoml.data.PointData.from_array(np.asarray(test.coordinates), ["X", "Y"])
m.predict(a, n_sim=10)
info = m.predict_by_expert(b, n_sim=10, coverage=1.0)
print("groups", len(info["subsets"]), info["subsets"], info["sizes"])
pa = np.asarray(a.values("v/prediction"))
pb = np.asarray(b.values("v/prediction"))
print("coverage 1: max |diff| prediction %.3g, sims %.3g" % (
    np.max(np.abs(pa - pb)),
    np.max(np.abs(a.variables["v"].get_simulations()
                  - b.variables["v"].get_simulations()))))
c = geoml.data.PointData.from_array(np.asarray(test.coordinates), ["X", "Y"])
info = m.predict_by_expert(c, n_sim=10, coverage=0.99)
pc = np.asarray(c.values("v/prediction"))
print("coverage 0.99: groups %d, mean set %.2f, max |diff| %.3g, "
      "left out max %.3g" % (len(info["subsets"]),
                             np.mean([len(s) for s in info["subsets"]]),
                             np.max(np.abs(pa - pc)),
                             np.max(info["left_out"])))
