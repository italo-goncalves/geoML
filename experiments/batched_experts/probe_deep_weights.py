"""In a deep tree, are the second GP's expert weights as local as the
first's? Experts needed for 99% of a point's weight, at each layer."""
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

J = 16
data, test, yt, truth = common.dataset(J)
m = common.model(data, J, deep=True, propagation="independent")
m.train_svi(10)
child = m.leaves[0]
parent = child.parent
root = parent.parent
x = tf.constant(np.asarray(data.coordinates)[:3000], tf.float64)
with geoml.latent.propagation_rule("independent"):
    m._refresh(m.options.jitter)
    xr, vr = root.propagate(x)
    _, _, w_parent, mu_p, var_p, _ = parent._moments(xr, vr)
    xp = tf.transpose(mu_p[:, :, 0])
    vp = tf.transpose(var_p)
    _, _, w_child, _, _, _ = child._moments(xp, vp)


def needed(w):
    w = np.asarray(w).mean(axis=1).T          # [n, J]
    w = -np.sort(-w, axis=1)
    k = (np.cumsum(w, axis=1) < 0.99).sum(axis=1) + 1
    return k.mean(), k.max()


print("first GP: experts for 99%% mean %.2f max %d" % needed(w_parent))
print("second GP: experts for 99%% mean %.2f max %d" % needed(w_child))
