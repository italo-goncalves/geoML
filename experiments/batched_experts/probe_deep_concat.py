"""Does concatenating the coordinates into the second GP's input keep its
experts local? Experts needed for 99% of a point's weight, at each layer,
for a GP reading a GP and for a GP reading (coordinates, GP)."""
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402
from geoml import latent  # noqa: E402

J = 16
data, test, yt, truth = common.dataset(J)


def build(concat):
    geoml.set_seed(common.SEED)
    ip = geoml.data.inducing.from_kmeans(
        data, common.INDUCING_PER_EXPERT * J, seed=0)
    experts = geoml.data.inducing.experts(ip, J, overlap=0.1, seed=0)
    root = latent.BasicInput(experts, transform=geoml.transform.Isotropic(15.0))
    first = latent.BasicGP(root, size=1, kernel=geoml.kernels.Matern32())
    parent = latent.Concatenate(root, first) if concat else first
    leaf = latent.BasicGP(parent, size=1)
    options = geoml.models.GPOptions(
        verbose=False, training_batch_size=400, prediction_batch_size=5000,
        expert_propagation="independent")
    m = geoml.models.VGPNetwork(data, "v", geoml.likelihood.Gaussian(
        geoml.warping.ZScore(1)), leaf, options=options)
    return m, root, first, leaf


def needed(w):
    w = np.asarray(w).mean(axis=1).T          # [n, J]
    w = -np.sort(-w, axis=1)
    k = (np.cumsum(w, axis=1) < 0.99).sum(axis=1) + 1
    return k.mean(), k.max()


for concat in (False, True):
    m, root, first, leaf = build(concat)
    m.train_svi(10)
    x = tf.constant(np.asarray(data.coordinates)[:3000], tf.float64)
    with geoml.latent.propagation_rule("independent"):
        m._refresh(m.options.jitter)
        xr, vr = root.propagate(x)
        _, _, w1, mu1, var1, _ = first._moments(xr, vr)
        xp = tf.transpose(mu1[:, :, 0])
        vp = tf.transpose(var1)
        if concat:
            xp = tf.concat([xr, xp], axis=1)
            vp = tf.concat([vr, vp], axis=1)
        _, _, w2, _, _, _ = leaf._moments(xp, vp)
    m.predict(test, n_sim=20)
    s = common.scores(test, yt, truth)
    rmse, crps = s["rmse"], s["crps"]
    rng = {k: np.round(np.asarray(p.get_value()), 3).tolist()
           for k, p in leaf.parameters.items() if "range" in k}
    print("concat=%s  first GP %.2f / %d  second GP %.2f / %d  "
          "rmse %.3f crps %.3f  second range %s"
          % ((concat,) + needed(w1) + needed(w2) + (rmse, crps, rng)))
