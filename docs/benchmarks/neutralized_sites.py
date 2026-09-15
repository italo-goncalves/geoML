"""Neutralizing a fold without refitting, Walker V, 2026-09-10.

Usage: python docs/benchmarks/neutralized_sites.py <single|experts> [k=5] [refits=200,1000]

The trained posterior of each expert is q(u) = N(m, S), m = L a (whitened
mean a), S = (K^-1 + D^-1)^-1, D = diag(delta). The mean at x is
k_x K^-1 m + b, which never reads delta. Arms, per spatial fold, no refit:

  delta-only   the proposal as stated: delta -> its upper bound (1e2) at
               the fold's inducing points, alpha untouched
  sites        the same points removed as pseudo-observations: ytil =
               (K + D) K^-1 m reproduces m, and m' = K (K + D')^-1 ytil'
               with the fold's ytil zeroed and its D at the bound
  krige        the fold's inducing values dropped and kriged from the kept
               posterior means (the sparse GP on the kept points); variance
               as in "sites"
  +ring        also removing inducing points within one step of a held-out
               datum
  rows         the held-out ROWS at infinite noise instead: with the
               hyperparameters, warping and noise frozen, the bound of a
               Gaussian leaf is quadratic in (alpha_white, bias), so the
               fold optimum is one ridge solve over the training rows;
               delta as trained
  rows+delta   the same with the fold's inducing points' delta at the bound
  prior        the filtering diagonal on the PRIOR instead (network.py, the
               `cov = covariance_matrix(ip, ip) + jitter` line): K' = K +
               1e2 at the fold's points, alpha, delta and bias as trained --
               the trained whitened mean is then read through chol(K')
  prior-reversed, prior-fold-last
               the same trained posterior whitened in another order of the
               inducing points before the prior is changed: the answer
               should not depend on an arbitrary order, and does
  mean-kept    the author's three steps: m = L a from the stored whitened
               mean, the diagonal on the prior, a' = chol(K')^-1 m, so the
               inducing means are kept and the order no longer matters; the
               mean at x becomes k_x K'^-1 m, which tends to kriging from the
               kept points' posterior means ("krige")
  mean-kept+ring  the same, with the one-step ring
  mean-zeroed  the three steps with the fold's inducing means set to zero
               (the prior mean) before the new whitened values are derived
  zeroed-no-prior  the fold's inducing means set to zero under the stored
               prior, their delta at the bound: the fold's inducing values
               pinned to the prior mean

Inducing points belong to the fold of their nearest datum; in the expert
layout every copy of a location goes. References: in-sample, the exact
optimum on all rows (how far Adam's state is from it), the honest refit on
the same folds (cross_validate, fresh state, 200 and 1000 iterations) and
the true error on Walker's exhaustive grid. rmse of the ground prediction
against the samples.
"""
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np
import scipy.linalg as sla
from scipy.spatial import cKDTree
import geoml
import tensorflow as tf

LAYOUT = sys.argv[1]
K_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
REFIT_ITERS = (tuple(int(v) for v in sys.argv[3].split(","))
               if len(sys.argv) > 3 else (200, 1000))
STEP, RANGE, ITER, BOUND = 26.0, 50.0, 500, 1e2
OUT = os.path.join(root, "docs", "benchmarks", "figures",
                   "neutralized_sites_%s.txt" % LAYOUT if K_FOLDS == 5 else
                   "neutralized_sites_%s_k%d.txt" % (LAYOUT, K_FOLDS))


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


open(OUT, "w").close()
geoml.set_seed(1234)
point, grid = geoml.datasets.walker()
if LAYOUT == "single":
    sets = [geoml.data.inducing.from_grid(point, STEP)]
else:
    sets = geoml.data.inducing.grid_experts(point, STEP, block=4)
root = geoml.latent.BasicInput(sets if len(sets) > 1 else sets[0],
                               transform=geoml.transform.Isotropic(RANGE))
gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
lik = geoml.likelihood.Gaussian(geoml.warping.ChainedWarping(
    geoml.warping.ZScore(1), geoml.warping.Spline(1)))
model = geoml.models.VGPNetwork(point, "V", lik, gp,
                                options=geoml.models.GPOptions(verbose=False))
t0 = time.time()
model.train_full(max_iter=ITER)
E = len(sets)
say("%s, k = %d: %d expert(s), %s inducing points; trained in %.0f s"
    % (LAYOUT, K_FOLDS, E, [s.n_data for s in sets], time.time() - t0))

y = point.variables["V"].measurements.values.to_numpy().astype(float)
coords = np.asarray(point.coordinates, dtype=float)
point.spatial_k_fold(grid, K_FOLDS, seed=0)
fold = np.asarray(point.get_metadata("fold")).ravel()
folds = np.unique(fold)


# the filtering diagonal on the prior, one per expert, read inside the
# traced graph: zero is bit-identical to the model as trained
BOOST = [tf.Variable(np.zeros(s.n_data), dtype=tf.float64, trainable=False)
         for s in sets]
_covariance = gp.covariance_matrix


def _boosted(x, y, x_var=None, y_var=None):
    out = _covariance(x, y, x_var, y_var)
    for i, ip in enumerate(gp.parent.inducing_points):
        if x is ip and y is ip:
            out = out + tf.linalg.diag(BOOST[i])
    return out


gp.covariance_matrix = _boosted


def predict():
    model.predict(point, n_sim=50)
    v = point.variables["V"]
    return (v.prediction.values.to_numpy().astype(float).copy(),
            v.latent_variance.values.to_numpy().astype(float).copy())


def rmse(pred, rows=slice(None)):
    return float(np.sqrt(np.mean((y[rows] - pred[rows]) ** 2)))


# the trained state, and each expert's pseudo-observations
model._refresh(1e-6)
state = []
for e in range(E):
    K = np.asarray(gp.cov[e])
    L = np.asarray(gp.cov_chol[e])
    a = np.asarray(gp.parameters["alpha_white_%d" % e].get_value())[0, :, 0]
    D = np.asarray(gp.parameters["delta_%d" % e].get_value())[0]
    m = L @ a
    ytil = m + D * sla.cho_solve((L, True), m)
    ip = np.asarray(sets[e].coordinates, dtype=float)
    b = float(np.asarray(gp.parameters["bias_%d" % e].get_value()))
    state.append(dict(K=K, L=L, a=a, D=D, ytil=ytil, ip=ip, b=b))
say("delta: share at the upper bound %.2f, median %.3g"
    % (np.mean(np.concatenate([s["D"] for s in state]) >= BOUND * 0.999),
       np.median(np.concatenate([s["D"] for s in state]))))

tree = cKDTree(coords)


def members(ip, f, ring):
    _, nearest = tree.query(ip)
    out = fold[nearest] == f
    if ring:
        held = cKDTree(coords[fold == f])
        dist, _ = held.query(ip)
        out |= dist <= STEP
    return out


def set_state(e, a, D):
    gp.parameters["alpha_white_%d" % e].set_value(a[None, :, None])
    gp.parameters["delta_%d" % e].set_value(D[None, :])


def restore():
    for e, s in enumerate(state):
        set_state(e, s["a"], s["D"])
        gp.parameters["bias_%d" % e].set_value(np.float64(s["b"]))
        BOOST[e].assign(np.zeros(len(s["a"])))


insample, v_in = predict()

# sanity: removing nothing reproduces the trained state
for e, s in enumerate(state):
    m2 = s["K"] @ np.linalg.solve(s["K"] + np.diag(s["D"]), s["ytil"])
    set_state(e, sla.solve_triangular(s["L"], m2, lower=True), s["D"])
check, _ = predict()
restore()
say("identity check, max |change| in prediction: %.2e" % np.max(np.abs(check - insample)))

arms = {k: np.full(len(y), np.nan) for k in
        ("delta-only", "sites", "sites+ring", "krige", "krige+ring",
         "prior", "prior-reversed", "prior-fold-last",
         "mean-kept", "mean-kept+ring", "mean-zeroed", "zeroed-no-prior")}
var_at = {k: np.full(len(y), np.nan) for k in arms}
clipped, removed_share = 0, {k: [] for k in arms}
t0 = time.time()
for f in folds:
    rows = fold == f
    for arm in arms:
        for e, s in enumerate(state):
            out = members(s["ip"], f, ring=arm.endswith("+ring"))
            removed_share[arm].append(out.mean())
            D2 = s["D"].copy()
            D2[out] = BOUND
            if arm == "delta-only":
                set_state(e, s["a"], D2)
                continue
            if arm == "mean-zeroed":
                boost = np.where(out, BOUND, 0.0)
                BOOST[e].assign(boost)
                m = s["L"] @ s["a"]
                m[out] = 0.0
                Lb = np.linalg.cholesky(s["K"] + np.diag(boost))
                a2 = sla.solve_triangular(Lb, m, lower=True)
                clipped += int(np.sum(np.abs(a2) > 10))
                set_state(e, a2, s["D"])
                continue
            if arm == "zeroed-no-prior":
                m = s["L"] @ s["a"]
                m[out] = 0.0
                a2 = sla.solve_triangular(s["L"], m, lower=True)
                clipped += int(np.sum(np.abs(a2) > 10))
                set_state(e, a2, D2)
                continue
            if arm.startswith("mean-kept"):
                boost = np.where(out, BOUND, 0.0)
                BOOST[e].assign(boost)
                m = s["L"] @ s["a"]
                Lb = np.linalg.cholesky(s["K"] + np.diag(boost))
                a2 = sla.solve_triangular(Lb, m, lower=True)
                clipped += int(np.sum(np.abs(a2) > 10))
                set_state(e, a2, s["D"])
                continue
            if arm.startswith("prior"):
                boost = np.where(out, BOUND, 0.0)
                BOOST[e].assign(boost)
                n = len(s["a"])
                if arm == "prior":
                    a2 = s["a"]
                else:
                    # the same posterior mean, whitened in another order, then
                    # read through that order's boosted Cholesky factor; the
                    # coefficient on the inducing values is beta = L'^-T a,
                    # carried back to the stored order's boosted factor
                    perm = (np.arange(n)[::-1] if arm == "prior-reversed"
                            else np.concatenate([np.where(~out)[0],
                                                 np.where(out)[0]]))
                    m = s["L"] @ s["a"]
                    KP = s["K"][np.ix_(perm, perm)]
                    aP = sla.solve_triangular(np.linalg.cholesky(KP), m[perm],
                                              lower=True)
                    LbP = np.linalg.cholesky(KP + np.diag(boost[perm]))
                    beta = np.empty(n)
                    beta[perm] = sla.solve_triangular(LbP.T, aP, lower=False)
                    Lb = np.linalg.cholesky(s["K"] + np.diag(boost))
                    a2 = Lb.T @ beta
                clipped += int(np.sum(np.abs(a2) > 10))
                set_state(e, a2, s["D"])
                continue
            if arm.startswith("krige"):
                # drop the fold's inducing values and krige them from the
                # kept posterior means: the prediction becomes the sparse GP
                # on the kept points alone; the variance as in "sites"
                m = s["L"] @ s["a"]
                keep = ~out
                m2 = m.copy()
                if keep.any() and out.any():
                    m2[out] = s["K"][np.ix_(out, keep)] @ np.linalg.solve(
                        s["K"][np.ix_(keep, keep)], m[keep])
                elif not keep.any():
                    m2[:] = 0.0
            else:
                y2 = s["ytil"].copy()
                y2[out] = 0.0
                m2 = s["K"] @ np.linalg.solve(s["K"] + np.diag(D2), y2)
            a2 = sla.solve_triangular(s["L"], m2, lower=True)
            clipped += int(np.sum(np.abs(a2) > 10))
            set_state(e, a2, D2)
        pred, var = predict()
        arms[arm][rows] = pred[rows]
        var_at[arm][rows] = var[rows]
        restore()
t_arms = time.time() - t0

# ---- the data's noise diagonal: held-out rows at infinite noise ----------
# With the hyperparameters, warping and noise frozen, the bound of a
# Gaussian leaf is quadratic in (alpha_white, bias): the fold optimum is one
# ridge solve over the training rows. Features phi = w_e(x) k_x L_e^-T for
# each expert's whitened mean, w_e(x) for its bias; KL prices alpha only.
x_all = tf.constant(coords, tf.float64)
y_w = np.asarray(lik.warping.forward(tf.constant(y[:, None]))[0])[:, 0]
sigma2 = float(np.asarray(lik.parameters["noise"].get_value()).ravel()[0])
sharp = float(lik.sharpness)


def features(deltas):
    for e, s in enumerate(state):
        set_state(e, s["a"], deltas[e])
    model._refresh(1e-6)
    xp, xv = gp.parent.propagate(x_all)
    cov_cross, _, w, _, _, _ = gp._moments(xp, xv)
    w = np.asarray(w)[:, 0, :]
    cols, pen = [], []
    for e, s in enumerate(state):
        kx = np.asarray(cov_cross[e])
        cols.append(w[e][:, None] * sla.solve_triangular(
            s["L"], kx.T, lower=True).T)
        cols.append(w[e][:, None])
        pen += [1.0] * kx.shape[1] + [0.0]
    return np.hstack(cols), np.array(pen)


def solve_and_set(train, deltas):
    phi, pen = features(deltas)
    A = sharp * phi[train].T @ phi[train] / sigma2 + np.diag(pen)
    theta = np.linalg.solve(A, sharp * phi[train].T @ y_w[train] / sigma2)
    k = 0
    for e, s in enumerate(state):
        n = len(s["a"])
        set_state(e, theta[k:k + n], deltas[e])
        gp.parameters["bias_%d" % e].set_value(np.float64(theta[k + n]))
        k += n + 1


# convergence check: every row kept, trained delta -- how far is Adam's
# state from the exact optimum it was heading for?
solve_and_set(np.ones(len(y), bool), [s["D"] for s in state])
exact_full, _ = predict()
restore()

rows_arms = {k: np.full(len(y), np.nan) for k in ("rows", "rows+delta")}
rows_var = {k: np.full(len(y), np.nan) for k in rows_arms}
t0 = time.time()
for f in folds:
    held = fold == f
    trained_D = [s["D"] for s in state]
    voronoi_D = []
    for e, s in enumerate(state):
        D2 = s["D"].copy()
        D2[members(s["ip"], f, ring=False)] = BOUND
        voronoi_D.append(D2)
    for arm, deltas in (("rows", trained_D), ("rows+delta", voronoi_D)):
        solve_and_set(~held, deltas)
        pred, var = predict()
        rows_arms[arm][held] = pred[held]
        rows_var[arm][held] = var[held]
        restore()
t_rows = time.time() - t0

refits = {}
for iterations in REFIT_ITERS:
    geoml.set_seed(99)
    t0 = time.time()
    oof, _ = geoml.models.cross_validate(model, folds="fold",
                                         iterations=iterations)
    v = oof.variables["V"]
    refits[iterations] = (
        v.prediction.values.to_numpy().astype(float),
        v.latent_variance.values.to_numpy().astype(float),
        time.time() - t0)

model.predict(grid, n_sim=20)
tp = grid.variables["V"].prediction.values.to_numpy().astype(float)
ty = grid.variables["V"].measurements.values.to_numpy().astype(float)
ok = np.isfinite(tp) & np.isfinite(ty)
truth = float(np.sqrt(np.mean((ty[ok] - tp[ok]) ** 2)))

say("alpha entries clipped at +-10 by the inducing-point arms: %d" % clipped)
say("")
say("%-12s %8s %22s %18s" % ("arm", "rmse", "latent var, held rows", "inducing removed"))
say("%-12s %8.1f %22.3f %18s" % ("in-sample", rmse(insample), np.mean(v_in), ""))
for arm in arms:
    say("%-12s %8.1f %22.3f %18.2f" % (arm, rmse(arms[arm]), np.mean(var_at[arm]),
                                       np.mean(removed_share[arm])))
say("%-12s %8.1f %22s %18s" % ("exact, all", rmse(exact_full), "", ""))
for arm in rows_arms:
    say("%-12s %8.1f %22.3f %18s" % (arm, rmse(rows_arms[arm]),
                                     np.mean(rows_var[arm]), ""))
for iterations, (pred, var, _) in refits.items():
    say("%-12s %8.1f %22.3f %18s" % ("refit-%d" % iterations, rmse(pred),
                                     np.nanmean(var), ""))
say("%-12s %8.1f" % ("truth", truth))
say("time: inducing-point arms %.1f s, row arms %.1f s; refits %s s"
    % (t_arms, t_rows, ", ".join("%.0f" % r[2] for r in refits.values())))
say("done")
