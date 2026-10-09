# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A GP at uncertain inputs under the expected kernel's second moment.

Usage: python docs/benchmarks/second_moment.py walker
       python docs/benchmarks/second_moment.py train
       python docs/benchmarks/second_moment.py cost M [KERNEL [first]]

train -- the `GaussianInput` gate's two cases (`gaussian_input.py`):
eight inputs with 30% of their entries missing, given their conditional
moments, uncertain in training and test rows alike; and Walker Lake with
its reported locations jittered (sd 5 and 15), the test locations exact.
Held-out rmse, coverage of the central 90% of a measurement, CRPS and the
time a model takes, three seeds, for a model told nothing (or imputed),
the expected kernel's first moment alone, with the second moment, and
`UncertainInputGP`.

cost -- seconds a training iteration and the process's peak memory on
1000 uncertain locations with M inducing points (`first`: without the
second moment), one setting a process.

walker -- the moments, against the exact mixture. On Walker Lake, a
`GaussianInput` root and one `BasicGP` (100 inducing points, trained 150
iterations on the known locations), at 200 random locations whose input is
Gaussian with variance `level` times the squared range: the node's mean and
variance with the second moment (what a model does under the expected
kernel since step 2), with the expected kernel's first moment alone (step
1), under the marginal rule (Paciorek's inflated covariance, the moments
taken at one point) and by a 32-node quadrature over the input (what
`UncertainInputGP` does), each against the mixture by Monte Carlo, 3000
draws a location. Errors are the mean absolute difference over the
locations relative to the Monte Carlo value's mean. The gate: the second
moment within 2% in mean and variance at every level, for all four kernels.
"""

import sys

import numpy as np
import tensorflow as tf
from scipy.stats import norm, qmc

import geoml
from geoml.latent import network as _net

LEVELS = (0.01, 0.1, 0.3, 1.0, 3.0)      # input variance / squared range
KERNELS = ("Gaussian", "Exponential", "Matern32", "Matern52")
N_QUERY, N_MC, SEED = 200, 3000, 7


def walker_model(kernel_name, iterations=150):
    geoml.set_seed(1234)
    point, _ = geoml.datasets.walker()
    train = geoml.data.PointData.from_array(
        np.asarray(point.coordinates, float), ["X", "Y"])
    train.add_continuous_variable(
        "v", np.asarray(point.values("V/measurements"), float).ravel())
    inducing = geoml.data.inducing.from_kmeans(train, 100, seed=0)
    root = geoml.latent.GaussianInput(inducing,
                                      transform=geoml.transform.Isotropic(40.0))
    gp = geoml.latent.BasicGP(root, size=1,
                              kernel=getattr(geoml.kernels, kernel_name)())
    model = geoml.models.VGPNetwork(
        train, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(iterations)
    return model, root, gp


def at_points(gp, x, chunk=20000):
    """The posterior at known points, in chunks: means and variances."""
    mus, vs = [], []
    for start in range(0, x.shape[0], chunk):
        mu, v = gp.interpolate(tf.constant(x[start:start + chunk]), None)
        mus.append(mu.numpy()[0, :, 0])
        vs.append(v.numpy()[0])
    return np.concatenate(mus), np.concatenate(vs)


def monte_carlo(gp, u, var, rng):
    n, d = u.shape
    draws = u[:, None, :] + np.sqrt(var)[:, None, :] \
        * rng.normal(size=(n, N_MC, d))
    m, v = at_points(gp, draws.reshape(-1, d))
    m, v = m.reshape(n, N_MC), v.reshape(n, N_MC)
    return m.mean(1), v.mean(1) + m.var(1)


def quadrature(gp, u, var, q=32):
    n, d = u.shape
    nodes = norm.ppf(qmc.Sobol(d, scramble=True, seed=0).random(q))
    draws = u[:, None, :] + np.sqrt(var)[:, None, :] * nodes[None, :, :]
    m, v = at_points(gp, draws.reshape(-1, d))
    m, v = m.reshape(n, q), v.reshape(n, q)
    return m.mean(1), v.mean(1) + m.var(1)


def expected_kernel(gp, u, var):
    """The node's own moments at the uncertain inputs, with the second
    moment, and the variance of the first moment alone."""
    chain = (_net._Joint(tf.constant(u), tf.constant(var), None),)
    cov_cross, mu, second, _ = gp._expert_moments(None, None, chain)
    l = cov_cross[0].numpy()
    inv = gp.cov_smooth_inv[0].numpy()[0]
    first = 1.0 - np.einsum("nm,ml,nl->n", l, inv, l)
    return mu[0].numpy()[0, :, 0], second[0].numpy()[0], first


def walker():
    print("The moments at uncertain inputs on Walker Lake: error against "
          "the exact mixture (Monte Carlo, %d draws a location), relative "
          "to its mean" % N_MC)
    print("%-12s %6s | %8s %8s | %9s %9s %9s %9s | %9s %9s"
          % ("kernel", "var/r2", "MC mean", "MC var", "2nd var", "1st var",
             "marg var", "quad var", "2nd mean", "marg mean"))
    worst = {}
    for name in KERNELS:
        model, root, gp = walker_model(name)
        rng = np.random.default_rng(SEED)
        with model._propagation():
            model._refresh(model.options.jitter)
            u_raw = rng.uniform([0, 0], [260, 300], size=(N_QUERY, 2))
            u = root.propagate(tf.constant(u_raw), None)[0].numpy()
            w = np.asarray(gp.parameters["ranges"].get_value()).ravel()
            for level in LEVELS:
                var = np.broadcast_to(level * w ** 2, u.shape).copy()
                mc_m, mc_v = monte_carlo(gp, u, var, rng)
                ek_m, ek_v, first_v = expected_kernel(gp, u, var)
                mg_m, mg_v = gp.interpolate(tf.constant(u), tf.constant(var))
                mg_m, mg_v = mg_m.numpy()[0, :, 0], mg_v.numpy()[0]
                _, qu_v = quadrature(gp, u, var)

                def rel(a, b):
                    return np.mean(np.abs(a - b)) / np.mean(np.abs(b))

                errors = (rel(ek_v, mc_v), rel(first_v, mc_v),
                          rel(mg_v, mc_v), rel(qu_v, mc_v), rel(ek_m, mc_m),
                          rel(mg_m, mc_m))
                worst[name] = max(worst.get(name, 0.0), errors[0],
                                  errors[4])
                print("%-12s %6.2f | %8.4f %8.4f | %9.3f %9.3f %9.3f %9.3f "
                      "| %9.3f %9.3f" % ((name, level, np.mean(mc_m),
                                          np.mean(mc_v)) + errors))
    print("largest error of the second moment, mean or variance: "
          + ", ".join("%s %.3f" % kv for kv in worst.items()))


# --------------------------------------------------------------------------- #
# training on uncertain inputs
# --------------------------------------------------------------------------- #
class first_moment_only:
    """The expected kernel without the second moment, as step 1 left it."""

    def __enter__(self):
        self.kept = _net._second_moment_supported
        _net._second_moment_supported = lambda kernel: False

    def __exit__(self, *args):
        _net._second_moment_supported = self.kept


def arms_on(train, test, root, seed, fit_and_score):
    """The arms for a model told its inputs' variance."""
    out = {}
    with first_moment_only():
        out["first moment"] = fit_and_score(train, test, root(), seed)
    out["second moment"] = fit_and_score(train, test, root(), seed)
    out["UIGP"] = fit_and_score(train, test, root(), seed,
                                node=geoml.latent.UncertainInputGP)
    return out


def train_gate():
    """The `GaussianInput` gate's two cases (`gaussian_input.py`), arms
    rearranged: told nothing (or imputed), the expected kernel's first
    moment alone, with the second moment, and `UncertainInputGP`."""
    import os
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gaussian_input as gi

    def timed(fit):
        def run(*args, **kwargs):
            start = time.perf_counter()
            result = fit(*args, **kwargs)
            return result + (time.perf_counter() - start,)
        return run

    fit = timed(gi.fit_and_score)

    def report(title, rows):
        print("\n" + title)
        print("%-16s %8s %8s %8s %8s" % ("arm", "rmse", "cover90", "crps",
                                         "time s"))
        for arm in rows[0]:
            values = np.array([r[arm] for r in rows])
            print("%-16s %8.3f %8.3f %8.3f %8.0f"
                  % ((arm,) + tuple(values.mean(axis=0))))
        print("  (means of %d seeds)" % len(rows))

    # A. eight inputs, 30% of the entries missing, test rows uncertain too
    rows = []
    for seed in gi.SEEDS:
        rng = np.random.default_rng(seed)
        x_train = gi.correlated_inputs(rng, gi.N_TRAIN)
        x_test = gi.correlated_inputs(rng, gi.N_TEST)
        y_train = gi.target(x_train) + 0.1 * rng.normal(size=gi.N_TRAIN)
        y_test = gi.target(x_test) + 0.1 * rng.normal(size=gi.N_TEST)
        m_train = rng.uniform(size=x_train.shape) < gi.MISSING
        m_test = rng.uniform(size=x_test.shape) < gi.MISSING
        mean = np.nanmean(np.where(m_train, np.nan, x_train), axis=0)
        complete = ~m_train.any(axis=1)
        cov = np.cov(x_train[complete].T)
        labels = ["x%d" % i for i in range(gi.N_DIM)]
        inducing = geoml.data.inducing.from_kmeans(
            geoml.data.PointData.from_array(
                np.where(m_train, mean, x_train), labels), 100, seed=seed)

        def gaussian(x, v, y):
            g = geoml.data.GaussianData.from_array(x, v, labels)
            g.add_continuous_variable("v", y)
            return g

        def point(x, y):
            p = geoml.data.PointData.from_array(x, labels)
            p.add_continuous_variable("v", y)
            return p

        f_train, v_train = gi.conditional_moments(x_train, m_train, mean, cov)
        f_test, v_test = gi.conditional_moments(x_test, m_test, mean, cov)
        out = {"impute": fit(
            point(np.where(m_train, mean, x_train), y_train),
            point(np.where(m_test, mean, x_test), y_test),
            geoml.latent.BasicInput(
                inducing, transform=geoml.transform.AnisotropyARD(gi.N_DIM)),
            seed)}
        out.update(arms_on(
            gaussian(f_train, v_train, y_train),
            gaussian(f_test, v_test, y_test),
            lambda: geoml.latent.GaussianInput(
                inducing, transform=geoml.transform.AnisotropyARD(gi.N_DIM)),
            seed, fit))
        rows.append(out)
    report("A. 8-D inputs, 30% of entries missing, conditional moments; "
           "100 inducing points", rows)

    # B. jittered locations on Walker Lake
    for sd in (5.0, 15.0):
        rows = []
        for seed in gi.SEEDS:
            rng = np.random.default_rng(seed)
            point, grid = geoml.datasets.walker()
            truth = np.asarray(grid.values("V/measurements"),
                               dtype=float).ravel()
            coords = np.asarray(grid.coordinates, dtype=float)
            keep = np.isfinite(truth)
            coords, truth = coords[keep], truth[keep]
            idx = rng.choice(coords.shape[0], 2470, replace=False)
            reported = coords[idx[:470]] + sd * rng.normal(size=(470, 2))
            inducing = geoml.data.Grid2D(start=[0, 0], end=[260, 300],
                                         n=[15, 17])
            train_point = geoml.data.PointData.from_array(reported, ["X", "Y"])
            train_point.add_continuous_variable("v", truth[idx[:470]])
            train_gauss = geoml.data.GaussianData.from_array(
                reported, np.full_like(reported, sd ** 2), ["X", "Y"])
            train_gauss.add_continuous_variable("v", truth[idx[:470]])
            test = geoml.data.PointData.from_array(coords[idx[470:]],
                                                   ["X", "Y"])
            test.add_continuous_variable("v", truth[idx[470:]])
            out = {"ignored": fit(
                train_point, test,
                geoml.latent.BasicInput(
                    inducing, transform=geoml.transform.Isotropic(40.0)),
                seed)}
            out.update(arms_on(
                train_gauss, test,
                lambda: geoml.latent.GaussianInput(
                    inducing, transform=geoml.transform.Isotropic(40.0)),
                seed, fit))
            rows.append(out)
        report("B. Walker Lake, location error sd = %.0f; 255 inducing "
               "points" % sd, rows)


def cost(m, kernel_name="Matern32", n=1000, iterations=20):
    """Seconds an iteration and the process's peak memory, training on `n`
    uncertain locations with `m` inducing points; run in a fresh process
    per setting, the peak being the process's."""
    import resource
    import time
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (n, 2))
    data = geoml.data.GaussianData.from_array(
        x, np.full_like(x, 25.0), ["X", "Y"])
    data.add_continuous_variable("v", np.sin(x[:, 0] / 15)
                                 + np.cos(x[:, 1] / 20))
    geoml.set_seed(0)
    root = geoml.latent.GaussianInput(
        geoml.data.inducing.from_kmeans(data, m, seed=0),
        transform=geoml.transform.Isotropic(20.0))
    gp = geoml.latent.BasicGP(root, size=1,
                              kernel=getattr(geoml.kernels, kernel_name)())
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(2)                       # the trace
    start = time.perf_counter()
    model.train_full(iterations)
    seconds = (time.perf_counter() - start) / iterations
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print("m %d, %s, n %d: %.3f s an iteration, peak %.2f GB"
          % (m, kernel_name, n, seconds, peak))


def main(argv):
    command = argv[0] if argv else "walker"
    if command == "walker":
        walker()
    elif command == "train":
        train_gate()
    elif command == "cost":
        if len(argv) > 3 and argv[3] == "first":
            with first_moment_only():
                cost(int(argv[1]), argv[2])
        else:
            cost(int(argv[1]), argv[2] if len(argv) > 2 else "Matern32")
    else:
        raise SystemExit("unknown command %r" % command)


if __name__ == "__main__":
    main(sys.argv[1:])
