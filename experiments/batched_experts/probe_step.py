"""Call one cached by-expert step directly and look at what it returns."""
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

data, test, yt, truth = common.dataset(4)
m = common.model(data, 4)
m.train_by_expert(1, batch_size=200)
st = m._by_expert
(key, step), = list(st["steps"].items())[:1]
subset = key[0]
print("subset", subset, "shared", [v.name for v in st["shared"]])
idx = np.arange(200)
leaf = m.leaves[0]
before = np.asarray(leaf.parameters["alpha_white_%d" % subset[0]]
                    .variable).ravel()[:2]
with geoml.latent.expert_subset(subset):
    bound, grads = step(
        tf.constant(m.data.coordinates[idx], tf.float64),
        tf.constant(m.y[idx], tf.float64),
        tf.constant(m.has_value[idx], tf.float64),
        tf.constant(m.data.get_batched_variance(idx)[0], tf.float64),
        tf.constant(1.0, tf.float64),
        tf.constant(np.full(len(subset), 1.0 / len(subset)), tf.float64))
after = np.asarray(leaf.parameters["alpha_white_%d" % subset[0]]
                   .variable).ravel()[:2]
print("bound", float(bound))
print("shared grads", [float(tf.reduce_sum(tf.abs(g))) for g in grads])
print("alpha before", before, "after", after)
print("concrete functions", len(step.experimental_get_tracing_count.__self__
                                 ._list_all_concrete_functions())
      if hasattr(step, "experimental_get_tracing_count") else "?")
print("trainable", leaf.parameters["ranges"].variable.trainable,
      leaf.parameters["alpha_white_0"].variable.trainable)
w = tf.Variable([1.0], dtype=tf.float64)
print("fresh trainable", w.trainable)
o = tf.keras.optimizers.Adam(1e-2)
o.build([w])
print("after build", w.trainable)
