"""`CensoredSimplex` against the usual likelihood, and its power trained.

The data are `power_simplex.py`'s: four parts with mean composition (0.4,
0.3, 0.2, 0.1), one smooth field per part (Gaussian covariance, range 25 on
a 100 x 100 square), noise of 0.3 on the latent, and the way back at a
true power of 0.25, 0.5 or 1. A fifth of the rows is held out; of the
others, a quarter have one or two parts hidden, as a campaign that did not
assay every element leaves them.

Arms, each a `PowerSimplex` chained with a `ZScore`:

* ``usual`` -- `Gaussian` at the true power, on the training rows that are
  complete and hold no zero, the only ones it can take;
* ``complete`` -- `CensoredSimplex` at the true power, on every complete
  training row, zeros included;
* ``censored`` -- the same on every training row, the partial ones too;
* ``trained`` -- as ``censored`` with the power trained from 1 (gate 2).

Scored on every held-out row, zeros included: the CRPS of a measurement per
part, the coverage of its 90% interval, and the share of measurement
samples with a part at zero against the share of held-out rows that have
one. The bound is never compared: a censored row's likelihood is a density
on the face its zeros lie on, a different measure for each pattern.

Usage:  python docs/benchmarks/censored_simplex.py [iterations] [seeds]
        [power ...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tensorflow as tf  # noqa: E402

import geoml  # noqa: E402
from geoml import likelihood as lk  # noqa: E402
from geoml import warping as wp  # noqa: E402

D = 4
SCALE = np.array([[0.4, 0.3, 0.2, 0.1]])
LABELS = ["a", "b", "c", "d"]


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
    x = np.where(x < 1e-12, 0.0, x)
    test = rng.uniform(size=n) < 0.2
    hidden = np.zeros((n, D), bool)
    partial = (~test) & (rng.uniform(size=n) < 0.25)
    for i in np.where(partial)[0]:
        hidden[i, rng.choice(D, rng.integers(1, 3), replace=False)] = True
    return coords, x, test, hidden


def arms(alpha):
    def chain(power, fixed=True):
        return wp.ChainedWarping(wp.PowerSimplex(D, alpha=power,
                                                 fixed=fixed), wp.ZScore(D))
    return {
        "usual": ("usual", lambda: lk.Gaussian(chain(alpha))),
        "complete": ("complete", lambda: lk.CensoredSimplex(chain(alpha))),
        "censored": ("all", lambda: lk.CensoredSimplex(chain(alpha))),
        "trained": ("all", lambda: lk.CensoredSimplex(chain(1.0, False))),
    }


def score(rows, make, coords, x, test, hidden, seed, iterations):
    complete = ~hidden.any(axis=1)
    if rows == "usual":
        train = ~test & complete & np.all(x > 0.0, axis=1)
    elif rows == "complete":
        train = ~test & complete
    else:
        train = ~test
    values = np.where(hidden, np.nan, x)[train]
    geoml.set_seed(100 + seed)
    point = geoml.data.PointData.from_array(coords[train], ["x", "y"])
    if rows == "usual":
        point.add_vector_variable("c", LABELS, values)
    else:
        point.add_mass_fraction_variable("c", LABELS, values)
    gp = geoml.latent.BasicGP(geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(point, 60, seed=0),
        transform=geoml.transform.Isotropic(30.0)), size=D)
    likelihood = make()
    model = geoml.models.VGPNetwork(
        point, "c", likelihood, gp,
        options=geoml.models.GPOptions(verbose=False))
    started = time.perf_counter()
    model.train_full(max_iter=iterations)
    spent = time.perf_counter() - started

    held = geoml.data.PointData.from_array(coords[test], ["x", "y"])
    samples = model.predict_measurements(held, n_sim=20, n_nodes=16)["c"]
    truth = x[test]
    crps = np.mean([geoml.metrics.crps(truth[:, k], samples[:, k, :])
                    for k in range(D)])
    low, high = np.quantile(samples, [0.05, 0.95], axis=2)
    cover = np.mean((truth >= low) & (truth <= high))
    zeros = np.mean(np.any(samples <= 0.0, axis=1))
    power = float(likelihood.warping.warpings[0].parameters["alpha"]
                  .get_value())
    return (crps, cover, zeros, np.mean(np.any(truth == 0.0, axis=1)),
            power, spent, int(train.sum()))


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    powers = [float(a) for a in sys.argv[3:]] or [0.25, 0.5, 1.0]
    for alpha in powers:
        results = {}
        for seed in seeds:
            coords, x, test, hidden = simulate(alpha, seed)
            for name, (rows, make) in arms(alpha).items():
                results.setdefault(name, []).append(
                    score(rows, make, coords, x, test, hidden, seed,
                          iterations))
        print(f"true power {alpha}")
        for name, values in results.items():
            crps, cover, zeros, observed, power, spent, n = \
                np.mean(values, axis=0)
            print(f"  {name:9s} crps {crps:.4f}  90% coverage {cover:.3f}  "
                  f"rows with a zero {zeros:.3f} (held out {observed:.3f})  "
                  f"power {power:.3f}  rows {n:.0f}  {spent:.0f} s",
                  flush=True)


if __name__ == "__main__":
    main()
