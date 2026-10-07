"""Where do the gradients of a by-expert batch go?"""
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

data, test, yt, truth = common.dataset(4)
m = common.model(data, 4)
leaf = m.leaves[0]
idx = np.arange(200)
x = tf.constant(m.data.coordinates[idx], tf.float64)
y = tf.constant(m.y[idx], tf.float64)
h = tf.constant(m.has_value[idx], tf.float64)
v = [leaf.parameters["alpha_white_0"].variable,
     leaf.parameters["ranges"].variable]
for subset in (None, (0, 1, 2)):
    with geoml.latent.expert_subset(subset):
        with tf.GradientTape() as tape:
            m._refresh(m.options.jitter)
            data_term = m._data_log_lik(x, y, h, [{}], samples=10, seed=0)
        g = tape.gradient(data_term, v)
    print(subset, "data", float(data_term),
          ["None" if gi is None else float(tf.reduce_sum(tf.abs(gi)))
           for gi in g])


@tf.function
def traced(x, y, h):
    with tf.GradientTape() as tape:
        m._refresh(m.options.jitter)
        data_term = m._data_log_lik(x, y, h, [{}], samples=10, seed=0)
    return data_term, tape.gradient(data_term, v)


for subset in (None, (0, 1, 2)):
    with geoml.latent.expert_subset(subset):
        d, g = traced(x, y, h) if subset is None else \
            tf.function(traced.python_function)(x, y, h)
    print("traced", subset, float(d),
          ["None" if gi is None else float(tf.reduce_sum(tf.abs(gi)))
           for gi in g])
