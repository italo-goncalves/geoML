"""`PowerSimplex` at the true power against `CenteredLogRatio` and
`ScaledSimplex`, on compositions drawn from the family.

Four parts with mean composition (0.4, 0.3, 0.2, 0.1), one smooth field per
part (Gaussian covariance, range 25 on a 100 x 100 square), noise of 0.3 on
the latent, and the way back at a true power of 0.25, 0.5 or 1. A fifth of
the rows is held out at random. The models train on the other rows that
hold no zero -- the usual likelihood takes no zero below a power of one, and
the centred log-ratio none at all -- and are scored on every held-out row,
zeros included: CRPS of a measurement per part, the coverage of its 90%
interval, and the share of measurement samples with a part at zero against
the share of held-out rows that have one. Each arm is chained with a
`ZScore` and trained full-batch for 1000 iterations; three seeds.

Usage:  python docs/benchmarks/power_simplex.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tensorflow as tf  # noqa: E402

import geoml  # noqa: E402
from geoml import warping as wp  # noqa: E402

D = 4
SCALE = np.array([[0.4, 0.3, 0.2, 0.1]])
LABELS = ["a", "b", "c", "d"]
SEEDS = (0, 1, 2)
ITERATIONS = 1000


def simulate(alpha, seed, n=600):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, (n, 2))
    d2 = ((coords[:, None] - coords[None]) ** 2).sum(-1)
    field = np.linalg.cholesky(np.exp(-d2 / (2 * 25.0 ** 2))
                               + 1e-8 * np.eye(n)) @ rng.normal(size=(n, D))
    truth = wp.PowerSimplex(D, alpha=alpha)
    truth.scale = tf.constant(SCALE, tf.float64)
    x = np.asarray(truth.backward(tf.constant(
        field + 0.3 * rng.normal(size=(n, D)))))
    test = rng.uniform(size=n) < 0.2
    return coords, x, test


def arms(alpha):
    return {
        "clr": lambda: wp.ChainedWarping(wp.CenteredLogRatio(D), wp.ZScore(D)),
        "scaled": lambda: wp.ChainedWarping(wp.ScaledSimplex(D),
                                            wp.ZScore(D)),
        "power": lambda: wp.ChainedWarping(wp.PowerSimplex(D, alpha=alpha),
                                           wp.ZScore(D)),
    }


def score(make, coords, x, test, seed):
    train = ~test & np.all(x > 0.0, axis=1)
    geoml.set_seed(100 + seed)
    point = geoml.data.PointData.from_array(coords[train], ["x", "y"])
    point.add_vector_variable("c", LABELS, x[train])
    gp = geoml.latent.BasicGP(geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(point, 60, seed=0),
        transform=geoml.transform.Isotropic(30.0)), size=D)
    model = geoml.models.VGPNetwork(
        point, "c", geoml.likelihood.Gaussian(make()), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=ITERATIONS)

    held = geoml.data.PointData.from_array(coords[test], ["x", "y"])
    samples = model.predict_measurements(held, n_sim=20, n_nodes=16)["c"]
    truth = x[test]
    crps = np.mean([geoml.metrics.crps(truth[:, k], samples[:, k, :])
                    for k in range(D)])
    low, high = np.quantile(samples, [0.05, 0.95], axis=2)
    cover = np.mean((truth >= low) & (truth <= high))
    zeros = np.mean(np.any(samples <= 0.0, axis=1))
    return crps, cover, zeros, np.mean(np.any(truth == 0.0, axis=1))


def main():
    for alpha in (0.25, 0.5, 1.0):
        rows = {}
        for seed in SEEDS:
            coords, x, test = simulate(alpha, seed)
            for name, make in arms(alpha).items():
                rows.setdefault(name, []).append(
                    score(make, coords, x, test, seed))
        print(f"true power {alpha}")
        for name, values in rows.items():
            crps, cover, zeros, observed = np.mean(values, axis=0)
            print(f"  {name:7s} crps {crps:.4f}  90% coverage {cover:.3f}  "
                  f"rows with a zero {zeros:.3f} (held out {observed:.3f})",
                  flush=True)


if __name__ == "__main__":
    main()
