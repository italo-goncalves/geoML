"""Predictive moments of a `BasicGP` at Gaussian-uncertain inputs.

The measurement behind `latent.UncertainInputGP` (2026-09-03): `BasicGP`'s
path (Paciorek's inflated covariance, the moments taken at one point under
it) against the exact mixture moments over the input's Gaussian (Monte
Carlo), Girard's closed form (Gaussian kernel), the inflated-variance
substitution of that form for other kernels, a Sobol quadrature over the
input, and the first-order slope term. Verdict recorded in the 0.6.10
changelog: the current path understates the variance four- to tenfold and
misplaces the mean from a tenth of the squared range up; the closed form is
exact for the Gaussian kernel and wrong for every other; the quadrature is
within 4% for every kernel at 32 nodes -- and is what the node does.

Usage: python docs/benchmarks/uncertain_input_moments.py [walker|highdim|both]
"""
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy.stats import norm, qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import geoml  # noqa: E402

LEVELS = (0.01, 0.1, 0.3, 1.0, 3.0)      # input variance / squared range
N_QUERY, N_MC, SEED = 200, 3000, 7


def fit(train, inducing, kernel, transform, iterations=150):
    geoml.set_seed(1234)
    root = geoml.latent.GaussianInput(inducing, transform=transform)
    gp = geoml.latent.BasicGP(root, size=1, kernel=kernel)
    model = geoml.models.VGPNetwork(
        train, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(iterations)
    model.latent_network.refresh()          # eager state to read from
    return model, root, gp


class Internals:
    """The one expert's posterior, as numpy, in the transformed space."""
    def __init__(self, gp):
        self.z = gp.parent.inducing_points[0].numpy()            # [m, d]
        self.alpha = gp.alpha[0].numpy()[0, :, 0]                # [m]
        self.A = gp.cov_smooth_inv[0].numpy()[0]                 # [m, m]
        self.bias = float(np.asarray(gp.parameters["bias_0"].get_value()).ravel()[0])
        w = np.asarray(gp.parameters["ranges"].get_value()).ravel()
        self.w = np.broadcast_to(w, (self.z.shape[1],)).astype(float)
        self.kernelize = lambda r: gp.kernel.kernelize(tf.constant(r, tf.float64)).numpy()
        self.gp = gp


def current(it, u, var):
    mu, v = it.gp.interpolate(tf.constant(u), tf.constant(var))
    return mu.numpy()[0, :, 0], v.numpy()[0]


def exact_at_points(it, x, chunk=20000):
    """The posterior at *known* points, in chunks."""
    mus, vs = [], []
    for start in range(0, x.shape[0], chunk):
        mu, v = it.gp.interpolate(tf.constant(x[start:start + chunk]), None)
        mus.append(mu.numpy()[0, :, 0]); vs.append(v.numpy()[0])
    return np.concatenate(mus), np.concatenate(vs)


def monte_carlo(it, u, var, rng):
    n, d = u.shape
    draws = u[:, None, :] + np.sqrt(var)[:, None, :] * rng.normal(size=(n, N_MC, d))
    m, v = exact_at_points(it, draws.reshape(-1, d))
    m, v = m.reshape(n, N_MC), v.reshape(n, N_MC)
    return m.mean(1), v.mean(1) + m.var(1), v.mean(1), m.var(1)


def quadrature(it, u, var, q, seed=0):
    n, d = u.shape
    nodes = norm.ppf(qmc.Sobol(d, scramble=True, seed=seed).random(q))   # [q, d]
    draws = u[:, None, :] + np.sqrt(var)[:, None, :] * nodes[None, :, :]
    m, v = exact_at_points(it, draws.reshape(-1, d))
    m, v = m.reshape(n, q), v.reshape(n, q)
    return m.mean(1), v.mean(1) + m.var(1)


def slope(it, u, var, step=1e-4):
    """The current variance plus the first-order slope term."""
    mu, v = current(it, u, var)
    grad2 = np.zeros_like(u)
    for dd in range(u.shape[1]):
        e = np.zeros_like(u); e[:, dd] = step
        mp, _ = exact_at_points(it, u + e); mm, _ = exact_at_points(it, u - e)
        grad2[:, dd] = ((mp - mm) / (2 * step)) ** 2
    return mu, v + np.sum(var * grad2, axis=1)


def girard(it, u, var, substitute=False):
    """Girard's moments. geoML's Gaussian kernel is exp(-3 r^2 / w^2), i.e.
    a Gaussian with squared width s^2 = w^2 / 6 per dimension. With
    `substitute`, the kernel's own `kernelize` replaces the exponentials at
    the inflated distances (the Paciorek-style transfer to other kernels)."""
    z, alpha, A, w = it.z, it.alpha, it.A, it.w
    s2 = w ** 2 / 6.0                                              # [d]
    n, d = u.shape
    m = z.shape[0]

    # first moment: E[k(x, z_i)]
    c1 = np.prod((1 + var / s2) ** -0.5, axis=1)                   # [n]
    q1 = ((u[:, None, :] - z[None, :, :]) ** 2 / (2 * (s2 + var)[:, None, :])).sum(-1)
    if substitute:
        ell = c1[:, None] * it.kernelize(np.sqrt(q1 / 3.0))
    else:
        ell = c1[:, None] * np.exp(-q1)                            # [n, m]
    mean = ell @ alpha + it.bias

    # second moment: E[k(x, z_i) k(x, z_j)]
    c2 = np.prod((1 + 2 * var / s2) ** -0.5, axis=1)               # [n]
    zbar = 0.5 * (z[:, None, :] + z[None, :, :])                   # [m, m, d]
    pair = ((z[:, None, :] - z[None, :, :]) ** 2 / (4 * s2)).sum(-1)   # [m, m]
    if substitute:
        pair_factor = it.kernelize(np.sqrt(pair / 3.0))
    else:
        pair_factor = np.exp(-pair)
    var_e = np.empty(n); mean_var = np.empty(n); exp_var = np.empty(n)
    for i in range(n):
        qi = ((u[i, None, None, :] - zbar) ** 2 / (s2 + 2 * var[i])[None, None, :]).sum(-1)
        mid = it.kernelize(np.sqrt(qi / 3.0)) if substitute else np.exp(-qi)
        L = c2[i] * pair_factor * mid                              # [m, m]
        exp_var[i] = 1.0 - np.sum(A * L)
        mean_var[i] = alpha @ L @ alpha - (ell[i] @ alpha) ** 2
        var_e[i] = exp_var[i] + mean_var[i]
    return mean, var_e, exp_var, mean_var


def report(name, it, u, w, rng):
    print("\n%s  (m=%d, d=%d, ranges %s)" % (name, it.z.shape[0], it.z.shape[1],
                                            np.round(w, 3)))
    print("%-6s | %-9s %-9s | %-9s %-9s %-9s %-9s %-9s | %-9s %-9s"
          % ("var/w2", "MC mean", "MC var", "cur mean", "cur var", "girard v",
             "quad32 v", "slope v", "MC E[v]", "MC Var[m]"))
    print("        errors are mean |method - MC| over %d points, relative to the MC value's mean" % N_QUERY)
    for level in LEVELS:
        var = np.broadcast_to(level * w ** 2, u.shape).copy()
        mc_m, mc_v, mc_ev, mc_vm = monte_carlo(it, u, var, rng)
        cu_m, cu_v = current(it, u, var)
        gi_m, gi_v, _, _ = girard(it, u, var, substitute=not isinstance(
            it.gp.kernel, geoml.kernels.Gaussian))
        qu_m, qu_v = quadrature(it, u, var, 32)
        sl_m, sl_v = slope(it, u, var)
        rel = lambda a, b: np.mean(np.abs(a - b)) / np.mean(np.abs(b))
        print("%-6.2f | %-9.4f %-9.4f | %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f | %-9.4f %-9.4f"
              % (level, np.mean(mc_m), np.mean(mc_v),
                 rel(cu_m, mc_m), rel(cu_v, mc_v), rel(gi_v, mc_v),
                 rel(qu_v, mc_v), rel(sl_v, mc_v), np.mean(mc_ev), np.mean(mc_vm)))
        print("       girard mean err %.3f   quad mean err %.3f   cur var/MC var ratio %.3f"
              % (rel(gi_m, mc_m), rel(qu_m, mc_m), np.mean(cu_v) / np.mean(mc_v)))


def walker(kernel_name):
    rng = np.random.default_rng(SEED)
    point, grid = geoml.datasets.walker()
    train = geoml.data.PointData.from_array(np.asarray(point.coordinates, float), ["X", "Y"])
    train.add_continuous_variable("v", np.asarray(point.values("V/measurements"), float).ravel())
    inducing = geoml.data.inducing.from_kmeans(train, 100, seed=0)
    kernel = getattr(geoml.kernels, kernel_name)()
    model, root, gp = fit(train, inducing, kernel, geoml.transform.Isotropic(40.0))
    it = Internals(gp)
    u_raw = rng.uniform([0, 0], [260, 300], size=(N_QUERY, 2))
    u, _ = root.propagate(tf.constant(u_raw), None)
    report("Walker Lake, %s kernel" % kernel_name, it, u.numpy(), it.w, rng)


def highdim():
    rng = np.random.default_rng(SEED)
    n_dim, n = 8, 400
    x = rng.normal(size=(n, n_dim)) @ rng.normal(size=(n_dim, n_dim)) * 0.5
    y = np.sin(x[:, 0]) + 0.5 * np.cos(1.3 * x[:, 1]) + 0.3 * x[:, 2] * x[:, 3]
    labels = ["x%d" % i for i in range(n_dim)]
    train = geoml.data.PointData.from_array(x, labels)
    train.add_continuous_variable("v", y)
    inducing = geoml.data.inducing.from_kmeans(train, 100, seed=0)
    model, root, gp = fit(train, inducing, geoml.kernels.Gaussian(),
                          geoml.transform.AnisotropyARD(n_dim))
    it = Internals(gp)
    u, _ = root.propagate(tf.constant(x[:N_QUERY]), None)
    report("8-D synthetic, Gaussian kernel, ARD", it, u.numpy(), it.w, rng)
    # the quadrature's error against its budget, at one level
    var = np.broadcast_to(1.0 * it.w ** 2, u.shape).copy()
    mc_m, mc_v, _, _ = monte_carlo(it, u.numpy(), var, rng)
    for q in (16, 32, 64, 128):
        started = time.time()
        qm, qv = quadrature(it, u.numpy(), var, q)
        print("  quadrature Q=%3d: var err %.3f, mean err %.3f  (%.2f s)"
              % (q, np.mean(np.abs(qv - mc_v)) / np.mean(mc_v),
                 np.mean(np.abs(qm - mc_m)) / np.mean(np.abs(mc_m)), time.time() - started))
    started = time.time(); girard(it, u.numpy(), var); print("  girard closed form: %.2f s" % (time.time() - started))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("walker", "both"):
        for k in ("Gaussian", "Matern32", "Cubic", "Spherical"):
            walker(k)
    if which in ("highdim", "both"):
        highdim()
    print("done", flush=True)
