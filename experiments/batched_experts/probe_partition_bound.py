"""Why an epoch of a partition does not add up to the bound: the pieces."""
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "geoml/test")
import geoml  # noqa: E402
import test_batched_experts as t  # noqa: E402

m = t._model()
m.train_full(5)
for parameter in m._all_parameters:
    parameter.fix()
rows = np.arange(m.data.n_data)
x = tf.constant(m.data.coordinates, tf.float64)
y = tf.constant(m.y, tf.float64)
hv = tf.constant(m.has_value, tf.float64)
xv = tf.constant(m.data.get_batched_variance(rows)[0], tf.float64)
m._refresh(m.options.jitter)
data = float(m._data_log_lik(x, y, hv, [{}], x_var=xv,
                             samples=m.options.training_samples,
                             seed=m.options.seed))
kl = float(tf.add_n([n.kl_divergence() for n in m._nodes()]))
prior = float(m.log_prior())
print("full: data %.6f kl %.6f prior %.6f bound %.6f"
      % (data, kl, prior, data - kl + prior))
seen = []
with geoml.progress(seen.append):
    record = m.train_by_expert(1, coverage=1.0, sampling="partition")
print("epoch total %.6f" % record["bound"][0])
print("partition", record["partition"][0])
# the batches' data terms, recomputed eagerly under every expert
table = m.expert_weights()
blocks = m._expert_blocks()
batches = geoml.models.VGPNetwork._partition(
    table, rows, np.arange(4), np.random.default_rng(0), 1.0, blocks)
total = 0.0
for b in batches:
    i = b["rows"]
    total += float(m._data_log_lik(
        tf.constant(m.data.coordinates[i], tf.float64),
        tf.constant(m.y[i], tf.float64), tf.constant(m.has_value[i],
                                                      tf.float64), [{}],
        x_var=tf.constant(m.data.get_batched_variance(i)[0], tf.float64),
        samples=m.options.training_samples, seed=m.options.seed))
    print("batch", b["expert"], len(i), b["sets"])
print("data summed over a partition, every expert: %.6f" % total)
full_elbo = float(m._training_elbo(x, y, hv, [{}], x_var=xv,
                                   samples=m.options.training_samples,
                                   seed=m.options.seed,
                                   jitter=m.options.jitter))
loglik = float(m._log_lik(x, y, hv, [{}], x_var=xv,
                          samples=m.options.training_samples,
                          seed=m.options.seed))
print("_training_elbo %.6f  _log_lik %.6f  total_data %.1f  has_value %.1f"
      % (full_elbo, loglik, float(m.total_data), float(np.sum(m.has_value))))
print("kl after _training_elbo %.6f" % float(tf.add_n(
    [n.kl_divergence() for n in m._nodes()])))
