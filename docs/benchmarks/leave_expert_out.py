"""First look at leave-expert-out cross-validation on Walker V, 2026-09-10.

Usage: python docs/benchmarks/leave_expert_out.py <block>
  block 6 -> 4 experts, 4 -> 9 experts, 3 -> 16 experts (step 26, range 50)

Arms, all scored as rmse of the ground prediction against the samples:
  in-sample        the trained model on its own data
  leo-mix          sum_i w_i(x) * pred^{-i}(x)   (the proposal's weighting)
  leo-sq           sqrt(mean_x sum_i w_i(x) (y - pred^{-i}(x))^2)
  leo-home         pred^{-h(x)}(x), h = the expert with the largest weight
  refit-home       cross_validate with folds = h(x): same geometry, honest refit
  knndm            cross_validate with spatial_k_fold against the grid (today)
  truth            the trained model against Walker's exhaustive grid
Removal is exact: a mask multiplies the expert's weight before normalizing.
The literal reset (alpha 0, delta 1 or 1e2, bias 0) is measured beside it.
"""
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np
import tensorflow as tf
import geoml
from geoml.latent import network

BLOCK = int(sys.argv[1])
STEP, RANGE, ITER = 26.0, 50.0, 500
OUT = os.path.join(root, "docs", "benchmarks", "figures",
                   "leave_expert_out_block%d.txt" % BLOCK)


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


open(OUT, "w").close()
geoml.set_seed(1234)
point, grid = geoml.datasets.walker()
experts = geoml.data.inducing.grid_experts(point, STEP, block=BLOCK)
E = len(experts)
root = geoml.latent.BasicInput(experts, transform=geoml.transform.Isotropic(RANGE))
gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
lik = geoml.likelihood.Gaussian(geoml.warping.ChainedWarping(
    geoml.warping.ZScore(1), geoml.warping.Spline(1)))
model = geoml.models.VGPNetwork(point, "V", lik, gp,
                                options=geoml.models.GPOptions(verbose=False))
t0 = time.time()
model.train_full(max_iter=ITER)
say("block %d: %d experts of %d points; trained %d it in %.0f s"
    % (BLOCK, E, experts[0].n_data, ITER, time.time() - t0))

# the mask: exact removal, read inside the traced graph, so no retrace
MASK = tf.Variable(np.ones(E), dtype=tf.float64, trainable=False)


def masked(variances):
    explained = 1 - variances
    weights = explained / (variances + 1e-6) + 1e-6
    m = tf.reshape(MASK, [-1] + [1] * (len(variances.shape) - 1))
    weights = weights * m
    return weights / tf.reduce_sum(weights, axis=0, keepdims=True)


network._GPNode.get_expert_weights = staticmethod(masked)

y = point.variables["V"].measurements.values.to_numpy().astype(float)


def rmse(pred, rows=slice(None)):
    return float(np.sqrt(np.mean((y[rows] - pred[rows]) ** 2)))


def weights_at_data():
    model._refresh(1e-6)
    x = tf.constant(np.asarray(point.coordinates), tf.float64)
    xp, xv = gp.parent.propagate(x)
    _, _, w, _, _, _ = gp._moments(xp, xv)
    return np.asarray(w)[:, 0, :].T          # N x E


def predict_rows():
    model.predict(point, n_sim=50)
    return point.variables["V"].prediction.values.to_numpy().astype(float).copy()


W = weights_at_data()
home = W.argmax(axis=1)
top = W.max(axis=1)
shared = 1 - (W ** 2).sum(axis=1)
say("weights: max-weight quantiles 10/50/90%%: %s; share of points with max > 0.9: %.2f; "
    "mean 1-sum(w^2): %.3f; experts reaching a point (w > 0.01), mean: %.2f"
    % (np.round(np.quantile(top, [0.1, 0.5, 0.9]), 3), np.mean(top > 0.9),
       shared.mean(), np.mean((W > 0.01).sum(axis=1))))

insample = predict_rows()
removed = np.zeros((len(y), E))
t0 = time.time()
for i in range(E):
    m = np.ones(E); m[i] = 0.0
    MASK.assign(m)
    removed[:, i] = predict_rows()
MASK.assign(np.ones(E))
t_leo = time.time() - t0

leo_mix = (W * removed).sum(axis=1)
leo_sq = float(np.sqrt(np.mean((W * (y[:, None] - removed) ** 2).sum(axis=1))))
leo_home = removed[np.arange(len(y)), home]

# the literal reset of the expert with the most home points
c = int(np.bincount(home, minlength=E).argmax())
rows_c = home == c
names = ["alpha_white_%d" % c, "delta_%d" % c, "bias_%d" % c]
saved = {n: np.asarray(gp.parameters[n].get_value()).copy() for n in names}
for delta in (1.0, 1e2):
    gp.parameters[names[0]].set_value(np.zeros_like(saved[names[0]]))
    gp.parameters[names[1]].set_value(np.full_like(saved[names[1]], delta))
    gp.parameters[names[2]].set_value(np.zeros_like(saved[names[2]]))
    Wr = weights_at_data()
    pr = predict_rows()
    say("literal reset of expert %d with delta=%g: its weight on its home points "
        "mean %.3f (trained %.3f); rmse there %.3f vs masked removal %.3f"
        % (c, delta, Wr[rows_c, c].mean(), W[rows_c, c].mean(),
           rmse(pr, rows_c), rmse(removed[:, c], rows_c)))
for n in names:
    gp.parameters[n].set_value(saved[n])

# honest reference on the same geometry: folds = home expert
point.add_metadata("home", home)
geoml.set_seed(99)
t0 = time.time()
oof_home, _ = geoml.models.cross_validate(model, folds="home")
t_home = time.time() - t0
refit_home = oof_home.variables["V"].prediction.values.to_numpy().astype(float)

# today's route: kNNDM folds against the exhaustive grid
point.spatial_k_fold(grid, 5, seed=0)
geoml.set_seed(99)
t0 = time.time()
oof_knn, _ = geoml.models.cross_validate(model, folds="fold")
t_knn = time.time() - t0
knndm = oof_knn.variables["V"].prediction.values.to_numpy().astype(float)

# truth: the trained model on the exhaustive grid
model.predict(grid, n_sim=20)
truth_pred = grid.variables["V"].prediction.values.to_numpy().astype(float)
truth_y = grid.variables["V"].measurements.values.to_numpy().astype(float)
ok = np.isfinite(truth_y) & np.isfinite(truth_pred)
truth = float(np.sqrt(np.mean((truth_y[ok] - truth_pred[ok]) ** 2)))

say("")
say("%-12s %9s %8s" % ("arm", "rmse", "time s"))
for name, value, t in [("in-sample", rmse(insample), None),
                       ("leo-mix", rmse(leo_mix), t_leo),
                       ("leo-sq", leo_sq, None),
                       ("leo-home", rmse(leo_home), None),
                       ("refit-home", rmse(refit_home), t_home),
                       ("knndm", rmse(knndm), t_knn),
                       ("truth", truth, None)]:
    say("%-12s %9.2f %8s" % (name, value, "" if t is None else "%.0f" % t))

# where the leak lives: leo-home against refit-home, binned by the top weight
say("")
say("by top weight: bin, n, rmse leo-home, rmse refit-home, rmse in-sample")
for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]:
    rows = (top >= lo) & (top < hi)
    if rows.sum() == 0:
        continue
    say("[%.2f, %.2f) %4d %9.2f %9.2f %9.2f"
        % (lo, hi, rows.sum(), rmse(leo_home, rows), rmse(refit_home, rows),
           rmse(insample, rows)))
say("done")
