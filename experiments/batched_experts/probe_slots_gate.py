"""Where do the slot and per-set paths part? The parameters after one
step: which entries differ, and by how much."""
import sys

import numpy as np

sys.path.insert(0, "geoml/test")
import geoml  # noqa: E402
import test_batched_experts as t  # noqa: E402


class Stop(Exception):
    pass


def stop(event):
    if event.task == "train" and event.done == 1:
        raise Stop


out = {}
for slots in (False, True):
    m = t._model()
    before = t._local_values(m)
    try:
        with geoml.progress(stop):
            m.train_by_expert(1, batch_size=50, slots=slots)
    except Stop:
        pass
    out[slots] = (before, t._local_values(m))
a0, a1 = out[False]
b0, b1 = out[True]
print("same start:", np.array_equal(a0, b0))
da, db = a1 - a0, b1 - b0
diff = np.abs(da - db)
print("moved entries", int(np.sum(da != 0)), int(np.sum(db != 0)))
order = np.argsort(-diff)[:8]
for i in order:
    print(i, "step sets %.10e slots %.10e diff %.3e" % (da[i], db[i], diff[i]))
