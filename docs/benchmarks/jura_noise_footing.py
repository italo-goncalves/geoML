"""Warping chains on Jura's metals, judged on the noise footing.

Under the epsilon-insensitive likelihood with a `ZScore -> Spline` warping
the variogram figure put copper and lead's fans at three times the data
curve and chromium's at one and a half (2026-09-09): a Laplace-like noise
in warped space pushed back through the spline's steep upper arm makes an
enormous variance in ppm squared for a right-skewed metal. The guide's
recommendation for new models is the parametric links, whose tails are
bounded. This measures it: the same inducing points, seed and spatial
folds for every arm, judged on (a) the lifted fan against the data curve,
in-sample and out-of-fold, per metal, and (b) the out-of-fold scores.

Usage: python docs/benchmarks/jura_noise_footing.py <chain> <likelihood> [seed] [iterations] [training_samples]
  chain: spline | boxcox | yeojohnson | boxcox-pca-sinharcsinh | spline-robust | boxcox-robust
  likelihood: epsilon | gaussian | mixture
The review of the first pass (2026-09-10) found 300 iterations far from
converged for the Gaussian arms and the mixture's eight-draw Monte Carlo
bound optimistic, hence the last two arguments.
"""
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["MPLBACKEND"] = "Agg"
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import geoml  # noqa: E402
import geoml.latent as gl  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.models as gm  # noqa: E402
import geoml.transform as tr  # noqa: E402
import geoml.warping as wp  # noqa: E402
from geoml.plots import prepare  # noqa: E402

CHAIN = sys.argv[1] if len(sys.argv) > 1 else "spline"
LIKELIHOOD = sys.argv[2] if len(sys.argv) > 2 else "epsilon"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
SAMPLES = int(sys.argv[5]) if len(sys.argv) > 5 else 8
# the first pass ran at 300 iterations and 8 samples and named its files
# without a budget; anything else says its budget in the name
TAG = "" if (ITERATIONS, SAMPLES) == (300, 8) else "_it%d_s%d" % (ITERATIONS, SAMPLES)
N_SIM, N_NODES, K = 64, 8, 5

CHAINS = {
    "spline": lambda k: wp.ChainedWarping(wp.ZScore(k), wp.Spline(k)),
    "boxcox": lambda k: wp.ChainedWarping(wp.BoxCox(k), wp.ZScore(k)),
    "yeojohnson": lambda k: wp.ChainedWarping(wp.YeoJohnson(k), wp.ZScore(k)),
    "boxcox-pca-sinharcsinh": lambda k: wp.ChainedWarping(
        wp.BoxCox(k), wp.RobustPCA(k, k), wp.ZScore(k), wp.SinhArcsinh(k), wp.ZScore(k)),
    # the robust ZScore leads the chain under a Mixture, as the guide says
    "spline-robust": lambda k: wp.ChainedWarping(wp.ZScore(k, robust=True), wp.Spline(k)),
    "boxcox-robust": lambda k: wp.ChainedWarping(wp.BoxCox(k), wp.ZScore(k, robust=True)),
}
LIKELIHOODS = {
    "epsilon": lambda k, w: lk.EpsilonInsensitive(warping=w),
    "gaussian": lambda k, w: lk.MultivariateGaussian(k, warping=w),
    # robustness with Gaussian tails: two scales of one Gaussian family
    "mixture": lambda k, w: lk.Mixture(w, n_components=2),
}


def bands(ratio):
    thirds = np.array_split(np.arange(len(ratio)), 3)
    return [float(np.nanmean(ratio[t])) for t in thirds]


def footing(container, labels):
    """Per metal: the lifted fan's mean over the data curve by lag band,
    the lift and the ground's long-lag variance as shares of the sill."""
    out = {}
    for p in prepare.variogram(container, "Elements", n_lags=15):
        data = np.asarray(p["data"]); fan = np.asarray(p["realizations"]).mean(axis=0)
        lift = np.zeros_like(data) if p["noise"] is None else np.asarray(p["noise"])
        raw = fan - lift
        short, mid, long = bands(fan / data)
        out[p["label"]] = dict(
            fan_over_data_short=short, fan_over_data_mid=mid, fan_over_data_long=long,
            lift_over_sill=float(np.nanmean(lift) / p["sill"]),
            ground_long_over_sill=float(np.nanmean(raw[-5:]) / p["sill"]))
    return out


def main():
    geoml.set_seed(SEED)
    train, held = geoml.datasets.jura()
    elements = list(train.get("Elements").labels)
    n_el = len(elements)
    y, has = train.variables["Elements"].get_measurements()
    y, has = np.asarray(y, float), np.asarray(has)
    sd = {c: float(np.std(y[has[:, j] == 1, j])) for j, c in enumerate(elements)}

    inducing = geoml.data.inducing.from_kmeans(train, 60, seed=0)
    leaf = gl.BasicGP(gl.BasicInput(inducing, transform=tr.Isotropic(1.0)), size=n_el)
    likelihood = LIKELIHOODS[LIKELIHOOD](n_el, CHAINS[CHAIN](n_el))
    model = gm.VGPNetwork(train, "Elements", likelihood, leaf,
                          options=gm.GPOptions(verbose=False, training_samples=SAMPLES))
    t0 = time.perf_counter()
    model.train_full(ITERATIONS)
    trained = time.perf_counter() - t0
    bound = float(model.training_log[-1])
    print("%s / %s / seed %d / %d iterations / %d samples: trained in %.0f s, bound %.1f"
          % (CHAIN, LIKELIHOOD, SEED, ITERATIONS, SAMPLES, trained, bound), flush=True)

    model.predict(train, n_sim=N_SIM, include_noise=True)
    in_sample = footing(train, elements)
    fig = geoml.plots.Explorer(train, continuous="Elements", model=model).variogram(n_lags=15)
    fig.suptitle("Jura in-sample, %s, %s" % (CHAIN, LIKELIHOOD), y=1.02)
    fig.savefig(os.path.join(root, "docs", "benchmarks", "figures",
                             "jura_footing_%s_%s%s_insample.png" % (CHAIN, LIKELIHOOD, TAG)), dpi=100, bbox_inches="tight")

    train.spatial_k_fold(held, k=K, seed=0)
    t0 = time.perf_counter()
    oof, scores = gm.cross_validate(model, iterations=200, n_sim=N_SIM, n_nodes=N_NODES)
    cv_time = time.perf_counter() - t0
    out_of_fold = footing(oof, elements)
    fig = geoml.plots.Explorer(oof, continuous="Elements", model=model).variogram(n_lags=15)
    fig.suptitle("Jura out-of-fold, %s, %s" % (CHAIN, LIKELIHOOD), y=1.02)
    fig.savefig(os.path.join(root, "docs", "benchmarks", "figures",
                             "jura_footing_%s_%s%s_oof.png" % (CHAIN, LIKELIHOOD, TAG)), dpi=100, bbox_inches="tight")
    pooled = scores[scores["fold"] == "all"].set_index("component")

    rows = []
    for c in elements:
        rows.append(dict(
            chain=CHAIN, likelihood=LIKELIHOOD, seed=SEED, iterations=ITERATIONS,
            training_samples=SAMPLES, metal=c, bound=bound,
            rmse_over_sd=float(pooled.loc[c, "rmse"] / sd[c]),
            crps_over_sd=float(pooled.loc[c, "crps"] / sd[c]),
            goodness=float(pooled.loc[c, "goodness"]),
            **{"in_" + k: v for k, v in in_sample[c].items()},
            **{"oof_" + k: v for k, v in out_of_fold[c].items()},
            train_seconds=round(trained, 1), cv_seconds=round(cv_time, 1)))
    table = pd.DataFrame(rows)
    print(table[["metal", "rmse_over_sd", "crps_over_sd", "goodness",
                 "in_fan_over_data_mid", "oof_fan_over_data_mid", "oof_lift_over_sill"]].to_string(index=False), flush=True)
    print("mean over metals: rmse/sd %.3f, crps/sd %.3f, goodness %.3f, |log oof fan/data| %.3f"
          % (table["rmse_over_sd"].mean(), table["crps_over_sd"].mean(), table["goodness"].mean(),
             np.abs(np.log(table[["oof_fan_over_data_short", "oof_fan_over_data_mid", "oof_fan_over_data_long"]].values)).mean()),
          flush=True)
    out = os.path.join(root, "docs", "benchmarks", "figures",
                       "jura_footing_%s_%s_seed%d%s.csv" % (CHAIN, LIKELIHOOD, SEED, TAG))
    table.to_csv(out, index=False)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
