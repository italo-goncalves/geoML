"""Do the per-expert optimizers step inside the traced step?"""
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402

data, test, yt, truth = common.dataset(4)
m = common.model(data, 4)
m.train_by_expert(1, batch_size=200)
st = m._by_expert
print("iterations", [int(o.iterations.numpy()) for o in st["optimizers"]],
      int(st["shared_optimizer"].iterations.numpy()))
print("local[0] ids", [id(v) for v in st["local"][0]][:3])
leaf = m.leaves[0]
print("param var id", id(leaf.parameters["alpha_white_0"].variable))
print("opt vars", [len(o.variables) for o in st["optimizers"]])

# the minimal case: a Keras Adam built outside, stepped inside a function
w = tf.Variable([1.0, 2.0], dtype=tf.float64)
opt = tf.keras.optimizers.Adam(1e-1)
opt.build([w])


@tf.function
def f():
    with tf.GradientTape() as t:
        loss = tf.reduce_sum(w ** 2)
    opt.apply_gradients(zip(t.gradient(loss, [w]), [w]))


f()
print("minimal", w.numpy(), int(opt.iterations.numpy()))
