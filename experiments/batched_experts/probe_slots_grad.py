"""The gradient of one batch's bound with respect to the experts' own
parameters, through the per-set path and through the slots (stacked from
the parameters themselves), term by term."""
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "geoml/test")
import geoml  # noqa: E402
import geoml.latent as latent  # noqa: E402
import test_batched_experts as t  # noqa: E402

m = t._model()
leaf = m.leaves[0]
subset = (0, 1, 2)
J = 4
idx = np.arange(50)
x = tf.constant(m.data.coordinates[idx], tf.float64)
y = tf.constant(m.y[idx], tf.float64)
hv = tf.constant(m.has_value[idx], tf.float64)
xv = tf.constant(m.data.get_batched_variance(idx)[0], tf.float64)
targets = {k: [leaf.parameters["%s_%d" % (n, k)].variable
               for n in ("alpha_white", "delta", "bias")] for k in subset}
flat = [v for k in subset for v in targets[k]]


def terms():
    m._refresh(m.options.jitter)
    data = m._data_log_lik(x, y, hv, [{}], x_var=xv,
                           samples=m.options.training_samples,
                           seed=m.options.seed)
    kl = leaf.expert_kl_terms()
    if isinstance(kl, list):
        kl = tf.stack(kl)
    return data, kl[:3]


def grads(context):
    out = {}
    with context:
        for name in ("data", "kl0"):
            with tf.GradientTape() as tape:
                data, kl = terms()
                f = data if name == "data" else kl[0]
            out[name] = (float(f), [np.asarray(tf.convert_to_tensor(g))
                                    if g is not None else None
                                    for g in tape.gradient(f, flat)])
    return out


a = grads(latent.expert_subset(subset))
slots = latent.network._Slots(4, tf.constant([0, 1, 2, J], tf.int32),
                              tf.constant([1., 1., 1., 0.], tf.float64))
b = grads(latent.network.expert_slots(slots))
names = ["%s_%d" % (n, k) for k in subset
         for n in ("alpha_white", "delta", "bias")]
for term in ("data", "kl0"):
    print(term, "value", a[term][0], b[term][0])
    for n, ga, gb in zip(names, a[term][1], b[term][1]):
        if ga is None or gb is None:
            print("  ", n, "None" if ga is None else "set", "None" if gb is None else "slot")
            continue
        print("   %-14s max|g| %.3e  max diff %.3e" % (
            n, np.max(np.abs(ga)), np.max(np.abs(ga - gb))))
