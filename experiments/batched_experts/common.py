"""Shared pieces of the batched-experts experiments.

A synthetic field whose domain grows with the number of experts, so that
each expert covers ground of the same size whatever their number -- the
regime where training every expert at every step makes memory grow with
J, and an expert at a time should hold it flat.
"""
import numpy as np

import geoml
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.transform as tr
import geoml.warping as wp

SEED = 20261006
TILE = 50.0          # ground per expert, a square of this side
DATA_PER_EXPERT = 400
TEST_PER_EXPERT = 100
INDUCING_PER_EXPERT = 150
NOISE = 0.2


def side(n_experts):
    return TILE * np.sqrt(n_experts)


def field(x, rng_seed=SEED):
    """Two scales of random Fourier features plus a gentle trend."""
    rng = np.random.default_rng(rng_seed)
    out = 0.002 * x[:, 0]
    for scale, amp, m in ((40.0, 1.0, 60), (10.0, 0.5, 120)):
        w = rng.normal(size=(m, 2)) / scale
        phase = rng.uniform(0, 2 * np.pi, m)
        out = out + amp * np.sqrt(2.0 / m) * np.cos(x @ w.T + phase).sum(1)
    return out


def dataset(n_experts, seed=SEED):
    """Training and test points over a square of `n_experts` tiles."""
    rng = np.random.default_rng(seed + n_experts)
    L = side(n_experts)
    x = rng.uniform(0, L, (DATA_PER_EXPERT * n_experts, 2))
    xt = rng.uniform(0, L, (TEST_PER_EXPERT * n_experts, 2))
    y = field(x) + NOISE * rng.normal(size=len(x))
    yt = field(xt) + NOISE * rng.normal(size=len(xt))
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", y)
    test = geoml.data.PointData.from_array(xt, ["X", "Y"])
    return data, test, yt, field(xt)


def model(data, n_experts, batch=400, propagation="consensus", deep=False,
          seed=SEED):
    """`deep=True` puts a GP on the first GP alone; `deep="concat"` on the
    coordinates and the first GP together."""
    geoml.set_seed(seed)
    ip = geoml.data.inducing.from_kmeans(
        data, INDUCING_PER_EXPERT * n_experts, seed=0)
    experts = geoml.data.inducing.experts(ip, n_experts, overlap=0.1, seed=0)
    root = latent.BasicInput(experts, transform=tr.Isotropic(15.0))
    leaf = latent.BasicGP(root, size=1, kernel=geoml.kernels.Matern32())
    if deep == "concat":
        leaf = latent.BasicGP(latent.Concatenate(root, leaf), size=1)
    elif deep:
        leaf = latent.BasicGP(leaf, size=1)
    options = geoml.models.GPOptions(
        verbose=False, training_batch_size=batch,
        prediction_batch_size=5000, expert_propagation=propagation)
    return geoml.models.VGPNetwork(data, "v", lk.Gaussian(wp.ZScore(1)),
                                   leaf, options=options)


def scores(test, yt, truth):
    """Held-out rmse against the noiseless field and CRPS against the
    measurements, from the test container's prediction."""
    pred = np.asarray(test.values("v/prediction"), dtype=float)
    sims = np.asarray(test.variables["v"].get_simulations())
    return dict(rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
                crps=float(geoml.metrics.crps(yt, sims)))


def gpu_peak_reset():
    import tensorflow as tf
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def gpu_peak_mb():
    import tensorflow as tf
    try:
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
    except Exception:
        return float("nan")
