"""The gate for `latent.GaussianInput`: is declaring an input variance worth it?

Two cases, three seeds each, held-out scores of a *measurement* (coverage of
the central 90% band from `predict_measurements`) beside rmse and crps.

A. High-dimensional inputs with missing entries -- the case the node is
   built for. Eight correlated inputs, a smooth target, 30% of the entries
   missing at random in training and test rows alike. Arms:
     drop      BasicInput, training rows with any missing entry dropped,
               test rows imputed by the column mean (there is no other way
               to predict them through a deterministic root);
     impute    BasicInput, column-mean imputation everywhere;
     marginal  GaussianInput, a missing entry given its column's mean and
               variance;
     conditional  GaussianInput, a missing entry given its conditional mean
               and variance under a Gaussian fitted to the complete rows --
               the correlations between inputs put to work.
B. Uncertain locations -- the rare case. Walker Lake's exhaustive truth
   sampled at 470 true locations, the *reported* locations jittered by a
   Gaussian error of known standard deviation. Arms: BasicInput told nothing,
   GaussianInput told the variance.

The `+UIGP` arms (added 2026-09-03) swap `BasicGP` for `UncertainInputGP`,
which integrates over the declared variance instead of reading it through
the inflated kernel.

Usage:  python docs/benchmarks/gaussian_input.py [A|B|both]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import geoml  # noqa: E402

SEEDS = (11, 22, 33)
ITERATIONS = 250


def scores(y, pred, samples):
    keep = np.isfinite(y) & np.isfinite(pred)
    y, pred, samples = y[keep], pred[keep], samples[keep]
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    low, high = np.quantile(samples, [0.05, 0.95], axis=1)
    coverage = float(np.mean((y >= low) & (y <= high)))
    crps = float(geoml.metrics.crps(y, samples))
    return rmse, coverage, crps


def fit_and_score(train, test, root, seed, node=geoml.latent.BasicGP):
    geoml.set_seed(seed)
    gp = node(root, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        train, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(ITERATIONS)
    model.predict(test, n_sim=20)
    samples = model.predict_measurements(test, n_sim=20, n_nodes=8)["v"][:, 0, :]
    y = np.asarray(test.values("v/measurements"), dtype=float).ravel()
    pred = np.asarray(test.values("v/prediction"), dtype=float).ravel()
    return scores(y, pred, samples)


# --------------------------------------------------------------------------- #
# A. missing entries in a high-dimensional input
# --------------------------------------------------------------------------- #
N_DIM, N_TRAIN, N_TEST, MISSING = 8, 400, 300, 0.3


def correlated_inputs(rng, n):
    # eight inputs as a linear mix of four latent factors plus noise, so the
    # conditional arm has correlations to use
    factors = rng.normal(size=[n, 4])
    mix = rng.normal(size=[4, N_DIM])
    x = factors @ mix + 0.5 * rng.normal(size=[n, N_DIM])
    return x


def target(x):
    return (np.sin(x[:, 0]) + 0.5 * np.cos(1.3 * x[:, 1]) + 0.3 * x[:, 2] * x[:, 3]
            + 0.2 * np.sin(x[:, 4] + x[:, 5]))


def conditional_moments(x_obs, mask, mean, cov):
    """Conditional mean/variance of the missing entries of each row under a
    Gaussian, given the observed ones; observed entries keep zero variance."""
    filled = np.where(mask, mean, x_obs)
    var = np.zeros_like(filled)
    for r in range(filled.shape[0]):
        m = mask[r]
        if not m.any():
            continue
        o = ~m
        if not o.any():
            filled[r, m] = mean[m]
            var[r, m] = np.diag(cov)[m]
            continue
        c_oo = cov[np.ix_(o, o)] + 1e-9 * np.eye(o.sum())
        c_mo = cov[np.ix_(m, o)]
        gain = np.linalg.solve(c_oo, c_mo.T).T
        filled[r, m] = mean[m] + gain @ (x_obs[r, o] - mean[o])
        var[r, m] = np.diag(cov[np.ix_(m, m)] - gain @ c_mo.T)
    return filled, np.maximum(var, 1e-9)


def case_a(seed):
    rng = np.random.default_rng(seed)
    x_train, x_test = correlated_inputs(rng, N_TRAIN), correlated_inputs(rng, N_TEST)
    y_train = target(x_train) + 0.1 * rng.normal(size=N_TRAIN)
    y_test = target(x_test) + 0.1 * rng.normal(size=N_TEST)
    m_train = rng.uniform(size=x_train.shape) < MISSING
    m_test = rng.uniform(size=x_test.shape) < MISSING

    complete = ~m_train.any(axis=1)
    mean = np.nanmean(np.where(m_train, np.nan, x_train), axis=0)
    var = np.nanvar(np.where(m_train, np.nan, x_train), axis=0)
    cov = np.cov(x_train[complete].T)

    labels = ["x%d" % i for i in range(N_DIM)]
    inducing_pool = np.where(m_train, mean, x_train)
    inducing = geoml.data.inducing.from_kmeans(
        geoml.data.PointData.from_array(inducing_pool, labels), 100, seed=seed)

    def point(x, y):
        p = geoml.data.PointData.from_array(x, labels)
        p.add_continuous_variable("v", y)
        return p

    def gaussian(x, v, y):
        g = geoml.data.GaussianData.from_array(x, v, labels)
        g.add_continuous_variable("v", y)
        return g

    def basic():
        return geoml.latent.BasicInput(inducing, transform=geoml.transform.AnisotropyARD(N_DIM))

    def gauss():
        return geoml.latent.GaussianInput(inducing, transform=geoml.transform.AnisotropyARD(N_DIM))

    out = {}
    test_imputed = point(np.where(m_test, mean, x_test), y_test)
    out["drop"] = fit_and_score(
        point(x_train[complete], y_train[complete]), test_imputed, basic(), seed)
    out["impute"] = fit_and_score(
        point(np.where(m_train, mean, x_train), y_train), test_imputed, basic(), seed)
    out["marginal"] = fit_and_score(
        gaussian(np.where(m_train, mean, x_train), np.where(m_train, var, 0.0), y_train),
        gaussian(np.where(m_test, mean, x_test), np.where(m_test, var, 0.0), y_test),
        gauss(), seed)
    out["marginal+UIGP"] = fit_and_score(
        gaussian(np.where(m_train, mean, x_train), np.where(m_train, var, 0.0), y_train),
        gaussian(np.where(m_test, mean, x_test), np.where(m_test, var, 0.0), y_test),
        gauss(), seed, node=geoml.latent.UncertainInputGP)
    f_train, v_train = conditional_moments(x_train, m_train, mean, cov)
    f_test, v_test = conditional_moments(x_test, m_test, mean, cov)
    out["conditional"] = fit_and_score(
        gaussian(f_train, v_train, y_train), gaussian(f_test, v_test, y_test),
        gauss(), seed)
    out["conditional+UIGP"] = fit_and_score(
        gaussian(f_train, v_train, y_train), gaussian(f_test, v_test, y_test),
        gauss(), seed, node=geoml.latent.UncertainInputGP)
    return out


# --------------------------------------------------------------------------- #
# B. jittered locations on Walker Lake
# --------------------------------------------------------------------------- #
def case_b(seed, sd):
    rng = np.random.default_rng(seed)
    point, grid = geoml.datasets.walker()
    truth = np.asarray(grid.values("V/measurements"), dtype=float).ravel()
    coords = np.asarray(grid.coordinates, dtype=float)
    keep = np.isfinite(truth)
    coords, truth = coords[keep], truth[keep]

    # 470 samples at true locations, reported with error; 2000 test points
    # at exact locations
    idx = rng.choice(coords.shape[0], 2470, replace=False)
    train_true, test_idx = coords[idx[:470]], idx[470:]
    reported = train_true + sd * rng.normal(size=train_true.shape)
    variance = np.full_like(reported, sd ** 2)

    inducing = geoml.data.Grid2D(start=[0, 0], end=[260, 300], n=[15, 17])

    train_point = geoml.data.PointData.from_array(reported, ["X", "Y"])
    train_point.add_continuous_variable("v", truth[idx[:470]])
    train_gauss = geoml.data.GaussianData.from_array(reported, variance, ["X", "Y"])
    train_gauss.add_continuous_variable("v", truth[idx[:470]])
    test = geoml.data.PointData.from_array(coords[test_idx], ["X", "Y"])
    test.add_continuous_variable("v", truth[test_idx])

    out = {}
    out["ignored"] = fit_and_score(
        train_point, test,
        geoml.latent.BasicInput(inducing, transform=geoml.transform.Isotropic(40.0)),
        seed)
    out["declared"] = fit_and_score(
        train_gauss, test,
        geoml.latent.GaussianInput(inducing, transform=geoml.transform.Isotropic(40.0)),
        seed)
    out["declared+UIGP"] = fit_and_score(
        train_gauss, test,
        geoml.latent.GaussianInput(inducing, transform=geoml.transform.Isotropic(40.0)),
        seed, node=geoml.latent.UncertainInputGP)
    return out


def report(title, rows):
    print("\n" + title)
    print("%-14s %8s %8s %8s" % ("arm", "rmse", "cover90", "crps"))
    arms = list(rows[0].keys())
    for arm in arms:
        values = np.array([r[arm] for r in rows])
        print("%-14s %8.3f %8.3f %8.3f" % (arm, *values.mean(axis=0)))
    print("  (means of %d seeds; per seed:)" % len(rows))
    for seed, r in zip(SEEDS, rows):
        print("  seed %d: " % seed + "  ".join(
            "%s %.3f/%.2f/%.3f" % (arm, *r[arm]) for arm in arms))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("A", "both"):
        report("A. 8-D inputs, 30% of entries missing", [case_a(s) for s in SEEDS])
    if which in ("B", "both"):
        for sd in (5.0, 15.0):
            report("B. Walker Lake, location error sd = %.0f" % sd,
                   [case_b(s, sd) for s in SEEDS])
    print("done", flush=True)
