"""Do the parameters move on a second call of train_by_expert?"""
import sys

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402

data, test, yt, truth = common.dataset(4)
m = common.model(data, 4)
leaf = m.leaves[0]


def snap():
    return (float(np.asarray(leaf.parameters["alpha_white_0"].get_value())
                  .ravel()[0]),
            float(np.asarray(leaf.parameters["ranges"].get_value()).ravel()[0]))


print("start", snap())
r = m.train_by_expert(1, batch_size=200)
print("after 1", snap(), r["bound"])
r = m.train_by_expert(1, batch_size=200)
print("after 2", snap(), r["bound"])
r = m.train_by_expert(2, batch_size=200)
print("after 2 more", snap(), r["bound"])
