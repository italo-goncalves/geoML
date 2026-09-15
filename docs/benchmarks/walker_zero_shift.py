"""Why chapter 13's Walker Lake fan sat above the data, 2026-09-15.

The manual's validation chapter drew the out-of-fold variogram fan over the
data and read "the model passes", while the data's curve sat under the
fan's lowest realization at 10 of 12 lags. The fan is exact (the noise lift
was checked against Monte Carlo on 2026-09-09), so the gap is the model's
own variance. This measures where it comes from, arm by arm: the model as
the chapter built it, and the same model changed in one place at a time.

Each arm reports the trained noise, exponent and range; the out-of-fold
scores of `cross_validate` (five spatial folds, 200 refit iterations); the
out-of-fold fan against the declustered data, lag by lag, with the noise
lift; the spread check by predicted grade; the declustered noise and
ground variance as shares of the data's; and the same fan predicted over
the exhaustive grid against the true field's own variogram, where there is
no clustering to argue about, with the lowest values the map would show.

The finding: Box-Cox's default shift, a millionth, puts Walker's 22 zero
samples so far down the logarithm that they pull the exponent's start from
0.60 (the zeros left out) to 0.375, and training keeps it near there
(0.42). The steeper inverse inflates the noise where the grade is high,
which is where the shortest, clustered pairs sit. A shift of one, the order
of the smallest positive value (2.1), trains exactly as Yeo-Johnson does
(the two maps agree on non-negative data) and puts the fan on the true
variogram.

Usage:  python docs/benchmarks/walker_zero_shift.py starts
        python docs/benchmarks/walker_zero_shift.py ARM
with ARM one of the keys of `ARMS`.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import geoml  # noqa: E402
from geoml.plots import prepare  # noqa: E402

ARMS = {
    "chapter": {},
    "shift1": {"shift": 1.0},
    "yeojohnson": {"link": "YeoJohnson"},
    "dense": {"step": 5.0, "block": 16},
    "long": {"iters": 2000},
    "shift1_long": {"shift": 1.0, "iters": 2000},
    "exponential": {"kernel": "Exponential"},
}


def starts():
    """Where Box-Cox's exponent starts, with and without the zeros."""
    walker, _ = geoml.datasets.walker()
    v = np.asarray(walker.values("V/measurements"), dtype=float)
    print("values: %d, zeros %d, smallest positive %g, largest %g"
          % (len(v), np.sum(v == 0), v[v > 0].min(), v.max()))
    for label, data, shift in (("shift 1e-6", v, 1e-6), ("shift 1", v, 1.0),
                               ("shift 1e-6, no zeros", v[v > 0], 1e-6)):
        warping = geoml.warping.BoxCox(1, shift=shift)
        warping.initialize(data[:, None])
        lam = float(warping.parameters["exponent"].get_value().numpy()[0])
        t = ((data + shift) ** lam - 1) / lam
        z = (t - t.mean()) / t.std()
        zeros = z[data == 0].min() if np.any(data == 0) else np.nan
        print("%-22s starts at %.3f; the zeros sit %.2f sd from the mean, "
              "the next value %.2f" % (label, lam, zeros,
                                       np.sort(z[data > 0])[0]))


def fan_table(container, label, **kwargs):
    panel = prepare.variogram(container, "V", n_lags=12, **kwargs)[0]
    fan = panel["realizations"]
    lift = panel["noise"]
    print("-- %s variogram, sill %.0f, VS %.1f"
          % (label, panel["sill"], panel["score"]))
    print("   lag   pairs     data    lift  fan-lo  fan-med  fan-hi  data/med")
    for k in range(len(panel["lag"])):
        print("   %5.0f %6d %8.0f %7.0f %7.0f %8.0f %7.0f   %5.2f" % (
            panel["lag"][k], panel["count"][k], panel["data"][k], lift[k],
            np.nanmin(fan[:, k]), np.nanmedian(fan[:, k]),
            np.nanmax(fan[:, k]),
            panel["data"][k] / np.nanmedian(fan[:, k])))


def arm(name):
    v = ARMS[name]
    geoml.set_seed(1234)
    walker, grid = geoml.datasets.walker()
    experts = geoml.data.inducing.grid_experts(grid, v.get("step", 10.0),
                                               block=v.get("block", 8))
    root = geoml.latent.BasicInput(experts,
                                   transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=getattr(
        geoml.kernels, v.get("kernel", "Spherical"))())
    link = getattr(geoml.warping, v.get("link", "BoxCox"))(
        1, **({"shift": v["shift"]} if "shift" in v else {}))
    likelihood = geoml.likelihood.Gaussian(
        geoml.warping.ChainedWarping(link, geoml.warping.ZScore(1)))
    model = geoml.models.VGPNetwork(
        walker, "V", likelihood, gp,
        options=geoml.models.GPOptions(verbose=False))
    start = time.time()
    model.train_full(max_iter=v.get("iters", 500))
    print("== %s: trained in %.0f s; noise %.4f, exponent %.4f, range %.2f"
          % (name, time.time() - start,
             float(np.ravel(likelihood.parameters["noise"].get_value())[0]),
             float(np.ravel(link.parameters["exponent"].get_value())[0]),
             float(np.ravel(root.transform.parameters["range"]
                            .get_value())[0])))

    walker.spatial_k_fold(grid, k=5, seed=0)
    oof, scores = geoml.models.cross_validate(model, iterations=200,
                                              n_sim=20)
    print(scores[scores["fold"] == "all"][
        ["rmse", "mae", "bias", "crps", "goodness"]].round(3).to_string())
    fan_table(oof, "out-of-fold")

    spread = prepare.spread_check(oof, "V", bins=6)[0]
    print("-- spread by predicted grade: centre, count, observed rms, "
          "noise sd, total sd")
    for k in range(len(spread["centre"])):
        print("   %6.0f %4d %7.0f %7.0f %7.0f" % (
            spread["centre"][k], spread["count"][k], spread["observed"][k],
            spread["noise"][k], spread["total"][k]))

    weights = np.asarray(geoml.math.geometry.declustering_weights(
        np.asarray(oof.coordinates, dtype=float),
        oof.values("V/measurements"))[0], dtype=float)
    weights = weights / weights.sum()
    measured = np.asarray(oof.values("V/measurements"), dtype=float)
    noise = float(np.sum(weights * np.asarray(
        oof.values("V/noise_variance"), dtype=float)))
    sims = np.asarray(oof.get("V").simulations[:, :], dtype=float)
    data_var = float(np.sum(weights * (measured
                                       - np.sum(weights * measured)) ** 2))
    sim_mean = float(np.sum(weights[:, None] * sims) / sims.shape[1])
    ground = float(np.sum(weights[:, None] * (sims - sim_mean) ** 2)
                   / sims.shape[1])
    print("-- declustered: data variance %.0f, noise %.0f (%.2f), ground "
          "realizations %.0f (%.2f), together %.2f of the data"
          % (data_var, noise, noise / data_var, ground, ground / data_var,
             (noise + ground) / data_var))

    model.predict(grid, n_sim=20)
    truth = np.asarray(grid.values("V/measurements"), dtype=float)
    prediction = np.asarray(grid.values("V/prediction"), dtype=float)
    print("-- truth: rmse %.1f, bias %.1f, true variance %.0f, mean noise "
          "variance %.0f" % (
              float(np.sqrt(np.mean((prediction - truth) ** 2))),
              float(np.mean(prediction - truth)), float(np.var(truth)),
              float(np.mean(np.asarray(grid.values("V/noise_variance"),
                                       dtype=float)))))
    fan_table(grid, "exhaustive grid", max_lag=190.0, decluster=False)
    grid.get("V").reset_quantiles([0.05, 0.5, 0.95])
    low = np.asarray(grid.values("V/quantiles/0.05"), dtype=float)
    sims = np.asarray(grid.get("V").simulations[:, :], dtype=float)
    print("-- lowest 0.05 quantile %.3f, lowest prediction %.3f, negative "
          "realization values %.4f%%" % (low.min(), prediction.min(),
                                          100 * np.mean(sims < 0)))


if __name__ == "__main__":
    if sys.argv[1] == "starts":
        starts()
    else:
        arm(sys.argv[1])
