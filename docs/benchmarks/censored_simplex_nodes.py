"""The node counts of the censored simplex likelihood, measured.

Rows are drawn from the model itself -- three or four parts, a latent per
row near a fixed one, Gaussian noise per part, the way back of a
`PowerSimplex` at 0.5 or 1 -- and some of their parts hidden: none, one,
two or three. Each row is scored at the latent it was drawn from and at one
moved by a standard deviation per part (a misfit, so the improbable rows a
model meets while it trains are in the sample), with affine links and with
a `SinhArcsinh` before the `ZScore`. The reference is the test file's: a
dense grid in the shift, and the complete rows integrated over the hidden
parts' faces by Gauss-Legendre under the cosine substitution. Each count is
varied with the others at their defaults; the target is 1e-3 nats a row.

Usage:  python docs/benchmarks/censored_simplex_nodes.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "geoml", "test"))

import tensorflow as tf  # noqa: E402

import geoml.stats.censored as cs  # noqa: E402
import test_censored_simplex as ref  # noqa: E402

CASES = {3: (np.array([0.5, 0.3, 0.2]), np.array([0.35, 0.5, 0.6]),
             np.array([0.2, -0.3, 0.4])),
         4: (ref.SCALE4, ref.SD4, ref.LATENT4),
         5: (ref.SCALE5, ref.SD5, ref.LATENT5)}


def rows(size, alpha, kind, n_hidden, n, seed):
    scale, sd, latent = CASES[size]
    rng = np.random.default_rng(seed)
    f = latent[None, :] + 0.5 * rng.standard_normal((n, size))
    z = f + sd * rng.standard_normal((n, size))
    links = ref._links(kind, size)
    w = z if links is None else links.backward(tf.constant(z, tf.float64))
    x = ref._simplex(alpha, size, scale).backward(
        tf.constant(w, tf.float64)).numpy()
    x[x < 1e-12] = 0.0
    hidden = np.zeros((n, size), bool)
    for i in range(n):
        hidden[i, rng.choice(size, n_hidden, replace=False)] = True
    moved = f + sd * rng.choice([-1.0, 1.0], (n, size))
    return x, hidden, np.stack([f, moved], axis=2)


def reference(x, hidden, f, alpha, kind, size):
    scale, sd, _ = CASES[size]
    out = np.zeros((len(x), f.shape[2]))
    for i in range(len(x)):
        for s in range(f.shape[2]):
            row = np.where(hidden[i], np.nan, x[i])
            if hidden[i].any():
                out[i, s] = ref._reference_hidden(row, hidden[i], f[i, :, s],
                                                  alpha, kind, sd, scale)
            else:
                out[i, s] = ref._reference_complete(x[i], f[i, :, s], alpha,
                                                    kind, sd, scale)
    return out


def ours(x, hidden, f, alpha, kind, size, rule):
    scale, sd, _ = CASES[size]
    return ref._ours(np.where(hidden, 1.0, x), ~hidden, f, alpha, kind,
                     rule, sd, scale)


def main():
    sets = [(3, 0, 12), (3, 1, 12), (3, 2, 12), (4, 3, 4), (5, 4, 2)]
    rules = {"default": {}}
    for n_tau in (16, 20, 32):
        rules["n_tau=%d" % n_tau] = {"n_tau": n_tau}
    for n_legendre in (8, 12, 24):
        rules["n_legendre=%d" % n_legendre] = {"n_legendre": n_legendre}
    for n_sobol in (256, 4096):
        rules["n_sobol=%d" % n_sobol] = {"n_sobol": n_sobol}

    worst = {name: {} for name in rules}
    for alpha in (0.5, 1.0):
        for kind in ("affine", "sinh"):
            for size, n_hidden, n in sets:
                if size == 5 and kind == "sinh":
                    continue
                x, hidden, f = rows(size, alpha, kind, n_hidden, n,
                                    seed=n_hidden + 10 * size)
                started = time.time()
                truth = reference(x, hidden, f, alpha, kind, size)
                spent = time.time() - started
                for name, kwargs in rules.items():
                    # the Sobol rule serves four or more positive hidden
                    # parts only, which only the five-part rows have
                    if name != "default" \
                            and ("n_sobol" in name) != (size == 5):
                        continue
                    got = ours(x, hidden, f, alpha, kind, size,
                               cs.Rule(**kwargs))
                    error = float(np.max(np.abs(got - truth)))
                    key = (n_hidden, size)
                    worst[name][key] = max(worst[name].get(key, 0.0), error)
                print("alpha %.1f %-6s %d hidden of %d: reference %.0f s, "
                      "default worst %.2e, lowest log-lik %.1f"
                      % (alpha, kind, n_hidden, size, spent,
                         worst["default"][(n_hidden, size)], truth.min()),
                      flush=True)

    print("\nworst error per row, nats (over both powers and both links)")
    keys = sorted({k for errors in worst.values() for k in errors})
    print("%-16s" % "rule" + "".join("%14s" % ("%d hid/%d" % k)
                                      for k in keys))
    for name, errors in worst.items():
        print("%-16s" % name + "".join(
            "%14s" % ("%.2e" % errors[k] if k in errors else "-")
            for k in keys))


if __name__ == "__main__":
    main()
