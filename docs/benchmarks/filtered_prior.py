"""The filtered prior written into the GP's own methods, Walker V, 2026-09-10.

Usage: python docs/benchmarks/filtered_prior.py <single|experts> [k=5]

Two rewrites of `BasicGP.refresh` for a terminal node, each with a filtering
diagonal Delta on the prior of the fold's inducing points (the
nearest-location rule: an inducing point belongs to the fold of its nearest
datum; every copy of a location is filtered, in every expert):

everywhere  the author's: the three steps -- the inducing means m = L a from
            the trained whitened values under the trained prior, the
            filtered prior K' = K + Delta, new whitened values a' =
            chol(K')^-1 m -- and then BasicGP.refresh with K' in place of K
            in every formula: its Cholesky factor and inverse, K' + D and its
            inverse (the variance), the simulations' root from chol(K') and
            delta, the dual weights K'^-1 L' a'. Nothing else is adjusted.
            As Delta grows the prediction tends to the mean
            k_k K_kk^-1 m_k + b and the variance 1 - k_k (K_kk + D_kk)^-1 k_k.
kept        the trained posterior q(u) = N(m, S) kept whole, S = (K^-1 +
            D^-1)^-1 under the trained prior, and the filtered prior in every
            quantity the prediction reads: K'^-1 m, K'^-1 - K'^-1 S K'^-1
            (written K'^-1 Delta K'^-1 + G (K + D)^-1 G^T, G = I - K'^-1 Delta,
            so no large matrices are subtracted), and the root G L^-T chol(W).
            As Delta grows: the same mean, and the variance
            1 - k_k K_kk^-1 k_k + k_k K_kk^-1 S_kk K_kk^-1 k_k.

Both are checked against BasicGP with no filter and against their
closed-form limits at a filter of 1e6. The cross-covariance k(x, z) and the
prior variance k(x, x) are not part of K and do not change. The refit, the
row solve and the earlier wrapper-based arms on this model and these folds
are in neutralized_sites_<layout>[_k<k>].txt.
"""
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import geoml  # noqa: E402
from geoml.latent.network import BasicGP  # noqa: E402

LAYOUT = sys.argv[1]
K_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
STEP, RANGE, ITER, JITTER = 26.0, 50.0, 500, 1e-6
FILTERS = (1e2, 1e6)
OUT = os.path.join(root, "docs", "benchmarks", "figures",
                   "filtered_prior_%s_k%d.txt" % (LAYOUT, K_FOLDS))


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


open(OUT, "w").close()

# the same model, seed and folds as neutralized_sites.py
geoml.set_seed(1234)
point, grid = geoml.datasets.walker()
if LAYOUT == "single":
    sets = [geoml.data.inducing.from_grid(point, STEP)]
else:
    sets = geoml.data.inducing.grid_experts(point, STEP, block=4)
base = geoml.latent.BasicInput(sets if len(sets) > 1 else sets[0],
                               transform=geoml.transform.Isotropic(RANGE))
gp = geoml.latent.BasicGP(base, size=1, kernel=geoml.kernels.Gaussian())
lik = geoml.likelihood.Gaussian(geoml.warping.ChainedWarping(
    geoml.warping.ZScore(1), geoml.warping.Spline(1)))
model = geoml.models.VGPNetwork(point, "V", lik, gp,
                                options=geoml.models.GPOptions(verbose=False))
t0 = time.time()
model.train_full(max_iter=ITER)
E = len(sets)
say("%s, k = %d: %d expert(s), %s inducing points; trained in %.0f s"
    % (LAYOUT, K_FOLDS, E, [s.n_data for s in sets], time.time() - t0))
assert not gp.children, "the rewrites cover a terminal node"

y = point.variables["V"].measurements.values.to_numpy().astype(float)
coords = np.asarray(point.coordinates, dtype=float)
point.spatial_k_fold(grid, K_FOLDS, seed=0)
fold = np.asarray(point.get_metadata("fold")).ravel()
folds = np.unique(fold)
tree = cKDTree(coords)
filtered = []                    # per expert: each inducing point's fold
for s in sets:
    _, nearest = tree.query(np.asarray(s.coordinates, dtype=float))
    filtered.append(fold[nearest])


def predict():
    model.predict(point, n_sim=50)
    v = point.variables["V"]
    return (v.prediction.values.to_numpy().astype(float).copy(),
            v.latent_mean.values.to_numpy().astype(float).copy(),
            v.latent_variance.values.to_numpy().astype(float).copy())


def rmse(pred, rows=slice(None)):
    return float(np.sqrt(np.mean((y[rows] - pred[rows]) ** 2)))


# the model as trained, through BasicGP's own refresh
reference = predict()

# ---- the rewrites -----------------------------------------------------------
FILTER = [tf.Variable(np.zeros(s.n_data), dtype=tf.float64, trainable=False)
          for s in sets]


def _refresh(mode, jitter):
    self = gp
    self.parent.refresh(jitter)
    out = dict(cov=[], cov_chol=[], cov_inv=[], cov_smooth=[],
               cov_smooth_chol=[], cov_smooth_inv=[], chol_r=[], alpha=[])
    for i, (ip, ip_var) in enumerate(zip(
            self.parent.inducing_points,
            self.parent.inducing_points_variance)):
        n = self.root.n_ip[i]
        eye = tf.eye(n, dtype=tf.float64)
        eye_s = tf.tile(eye[None, :, :], [self.size, 1, 1])
        a = self.parameters["alpha_white_%d" % i].get_value()
        d = self.parameters["delta_%d" % i].get_value()

        # the trained prior, and the inducing means under it (step 1)
        K = self.covariance_matrix(ip, ip, ip_var, ip_var) + eye * jitter
        L = tf.linalg.cholesky(K)
        m = tf.einsum("ab,sbc->sac", L, a)

        # the filtered prior (step 2)
        f = FILTER[i]
        Kf = K + tf.linalg.diag(f)
        Lf = tf.linalg.cholesky(Kf)
        Kf_inv = tf.linalg.cholesky_solve(Lf, eye)

        if mode == "everywhere":
            # new whitened values from the new Cholesky (step 3), then
            # BasicGP.refresh with Kf in place of K in every formula
            a_f = tf.linalg.triangular_solve(
                tf.tile(Lf[None, :, :], [self.size, 1, 1]), m, lower=True)
            smooth = Kf[None, :, :] + tf.linalg.diag(d)
            smooth_chol = tf.linalg.cholesky(smooth + eye_s * jitter)
            smooth_inv = tf.linalg.cholesky_solve(smooth_chol, eye_s)
            chol_r = BasicGP._whitened_root(Lf, d, eye_s)
            means = tf.einsum("ab,sbc->sac", Lf, a_f)
            alpha = tf.einsum("ab,sbc->sac", Kf_inv, means)
        else:
            # the trained posterior kept whole
            smooth = K[None, :, :] + tf.linalg.diag(d)
            smooth_chol = tf.linalg.cholesky(smooth + eye_s * jitter)
            trained_inv = tf.linalg.cholesky_solve(smooth_chol, eye_s)
            G = eye - Kf_inv * f[None, :]
            smooth_inv = tf.matmul(Kf_inv * f[None, :], Kf_inv)[None, :, :] \
                + tf.einsum("ab,sbc,dc->sad", G, trained_inv, G)
            chol_r = tf.einsum("ab,sbc->sac", G,
                               BasicGP._whitened_root(L, d, eye_s))
            alpha = tf.einsum("ab,sbc->sac", Kf_inv, m)

        out["cov"].append(Kf)
        out["cov_chol"].append(Lf)
        out["cov_inv"].append(Kf_inv)
        out["cov_smooth"].append(smooth)
        out["cov_smooth_chol"].append(smooth_chol)
        out["cov_smooth_inv"].append(smooth_inv)
        out["chol_r"].append(chol_r)
        out["alpha"].append(alpha)
    for name, values in out.items():
        setattr(self, name, tuple(values))


def use(mode):
    def refresh(jitter=1e-6):
        with tf.name_scope("filtered_refresh_" + mode):
            _refresh(mode, jitter)
    gp.refresh = refresh
    model._refresh_graph = None      # the traced refresh is rebuilt around it
    return refresh


def set_filter(f, size):
    for e in range(E):
        FILTER[e].assign(np.where(filtered[e] == f, size, 0.0)
                         if f is not None else np.zeros(sets[e].n_data))


def limit_check(mode, refresh):
    """The node's latent moments at a filter of 1e6 against the closed form
    of the mode's limit, per expert, on the first fold's rows."""
    f0 = folds[0]
    set_filter(f0, 1e6)
    refresh(JITTER)
    held = tf.constant(coords[fold == f0], tf.float64)
    xp, xv = gp.parent.propagate(held)
    cov_cross, mu, _, _, _, _ = gp._moments(xp, xv)
    worst_mean = worst_var = 0.0
    for e in range(E):
        ip = gp.parent.inducing_points[e]
        ipv = gp.parent.inducing_points_variance[e]
        n = sets[e].n_data
        K = np.asarray(gp.covariance_matrix(ip, ip, ipv, ipv)) \
            + np.eye(n) * JITTER
        L = np.linalg.cholesky(K)
        a = np.asarray(gp.parameters["alpha_white_%d" % e].get_value())[0, :, 0]
        d = np.asarray(gp.parameters["delta_%d" % e].get_value())[0]
        b = float(np.asarray(gp.parameters["bias_%d" % e].get_value()))
        m = L @ a
        keep = filtered[e] != f0
        k_all = np.asarray(cov_cross[e])
        k = k_all[:, keep]
        Kkk = K[np.ix_(keep, keep)]
        proj = np.linalg.solve(Kkk, k.T).T
        mean_ref = proj @ m[keep] + b
        if mode == "everywhere":
            smooth = Kkk + np.diag(d[keep]) + np.eye(keep.sum()) * JITTER
            var_ref = 1.0 - np.sum(np.linalg.solve(smooth, k.T).T * k, axis=1)
        else:
            W = np.linalg.inv(np.eye(n) + L.T @ np.diag(1.0 / d) @ L)
            S = L @ W @ L.T
            var_ref = 1.0 - np.sum(proj * k, axis=1) \
                + np.sum((proj @ S[np.ix_(keep, keep)]) * proj, axis=1)
        M = np.asarray(gp.cov_smooth_inv[e])[0]
        var_node = 1.0 - np.sum((k_all @ M) * k_all, axis=1)
        worst_mean = max(worst_mean, np.max(np.abs(
            np.asarray(mu[e])[0, :, 0] - mean_ref)))
        worst_var = max(worst_var, np.max(np.abs(var_node - var_ref)))
    set_filter(None, 0.0)
    return worst_mean, worst_var


results = {}
for mode in ("everywhere", "kept"):
    refresh = use(mode)
    set_filter(None, 0.0)
    identity = predict()
    say("%s -- no filter against BasicGP: max |diff| prediction %.1e, latent "
        "mean %.1e, latent variance %.1e" % ((mode,) + tuple(
            np.max(np.abs(a - b)) for a, b in zip(identity, reference))))
    say("%s -- filter 1e6 against its closed-form limit: max |diff| latent "
        "mean %.1e, variance %.1e" % ((mode,) + limit_check(mode, refresh)))
    for size in FILTERS:
        pred = np.full(len(y), np.nan)
        lvar = np.full(len(y), np.nan)
        for f in folds:
            rows = fold == f
            set_filter(f, size)
            p, _, v = predict()
            pred[rows], lvar[rows] = p[rows], v[rows]
        set_filter(None, 0.0)
        results[(mode, size)] = (pred, lvar)

say("")
say("%-28s %8s %22s" % ("arm", "rmse", "latent var, held rows"))
say("%-28s %8.1f %22.3f" % ("in-sample", rmse(reference[0]),
                            np.mean(reference[2])))
for (mode, size), (pred, lvar) in results.items():
    say("%-28s %8.1f %22.3f" % ("%s, %g" % (mode, size), rmse(pred),
                                np.mean(lvar)))
say("done")
