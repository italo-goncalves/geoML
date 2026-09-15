"""The parametric links against `Spline`, on Walker Lake and Jura.

`BoxCox`, `YeoJohnson`, `Arcsinh` and `SinhArcsinh` (0.6.10) are
closed-form, trainable, elementwise marginal transforms. The question is
whether one of them can stand where the manual's chains put a `Spline`,
alone or in front of a shallower one: held-out rmse/sd, crps/sd, 90%
coverage and predicted-sd/data-sd of a *measurement*, the ELBO, training
time, the chain's back-transform round trip in data units, and the tail
check the spline lesson asks for -- 20 000 standard-normal latent draws
pushed through `backward`, how many come back non-finite and how far the
largest lands past the data's own maximum.

Walker Lake is chapter 15's model (a spherical kernel on a 10-unit lattice
of experts, 500 iterations) scored on 2000 points of the exhaustive truth;
Jura is the flow gate's model (150 k-means inducing points, phased
schedule) scored on the held-out 100 sites, the seven metals averaged.

Usage:  python docs/benchmarks/parametric_warpings.py [walker|jura|both] [arm,arm,...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tensorflow as tf  # noqa: E402

import geoml  # noqa: E402
from geoml import warping  # noqa: E402

N_SIM = 20
SEEDS = {"walker": (1234, 2345, 3456), "jura": (1234, 2345)}


def walker_arms():
    S = warping.Spline
    return {
        "baseline (spline 4)": lambda k: warping.ChainedWarping(
            warping.Scale(k), warping.Softplus(k), warping.ZScore(k), S(k, knots_per_arm=4)),
        "no spline": lambda k: warping.ChainedWarping(
            warping.Scale(k), warping.Softplus(k), warping.ZScore(k)),
        "boxcox": lambda k: warping.ChainedWarping(warping.BoxCox(k), warping.ZScore(k)),
        "yeojohnson": lambda k: warping.ChainedWarping(warping.YeoJohnson(k), warping.ZScore(k)),
        "arcsinh": lambda k: warping.ChainedWarping(warping.Arcsinh(k), warping.ZScore(k)),
        "sinharcsinh": lambda k: warping.ChainedWarping(
            warping.ZScore(k), warping.SinhArcsinh(k), warping.ZScore(k)),
        "boxcox + spline 2": lambda k: warping.ChainedWarping(
            warping.BoxCox(k), warping.ZScore(k), S(k, knots_per_arm=2)),
        "sinharcsinh + spline 2": lambda k: warping.ChainedWarping(
            warping.ZScore(k), warping.SinhArcsinh(k), warping.ZScore(k), S(k, knots_per_arm=2)),
    }


def jura_arms():
    S = warping.Spline
    return {
        "baseline (spline 5)": lambda k: warping.ChainedWarping(
            warping.Log(k), warping.RobustPCA(k, k), S(k, knots_per_arm=5), warping.ZScore(k)),
        "no spline": lambda k: warping.ChainedWarping(
            warping.Log(k), warping.RobustPCA(k, k), warping.ZScore(k)),
        "boxcox + pca": lambda k: warping.ChainedWarping(
            warping.BoxCox(k), warping.RobustPCA(k, k), warping.ZScore(k)),
        "pca + yeojohnson": lambda k: warping.ChainedWarping(
            warping.Log(k), warping.RobustPCA(k, k), warping.YeoJohnson(k), warping.ZScore(k)),
        "pca + sinharcsinh": lambda k: warping.ChainedWarping(
            warping.Log(k), warping.RobustPCA(k, k), warping.ZScore(k),
            warping.SinhArcsinh(k), warping.ZScore(k)),
        "boxcox + pca + sinharcsinh": lambda k: warping.ChainedWarping(
            warping.BoxCox(k), warping.RobustPCA(k, k), warping.ZScore(k),
            warping.SinhArcsinh(k), warping.ZScore(k)),
        "pca + sinharcsinh + spline 2": lambda k: warping.ChainedWarping(
            warping.Log(k), warping.RobustPCA(k, k), warping.ZScore(k),
            warping.SinhArcsinh(k), warping.ZScore(k), S(k, knots_per_arm=2)),
    }


# --------------------------------------------------------------------------- #
def walker_case():
    point, grid = geoml.datasets.walker()
    truth = np.asarray(grid.values("V/measurements"), dtype=float).ravel()
    coords = np.asarray(grid.coordinates, dtype=float)
    keep = np.isfinite(truth)
    idx = np.random.default_rng(7).choice(np.flatnonzero(keep), 2000, replace=False)
    validation = geoml.data.PointData.from_array(coords[idx], ["X", "Y"])
    validation.add_continuous_variable("V", truth[idx])

    def build(chain):
        root = geoml.latent.BasicInput(
            geoml.data.inducing.grid_experts(grid, 10.0, block=8),
            transform=geoml.transform.Isotropic(50))
        gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Spherical())
        return geoml.models.VGPNetwork(
            point, "V", geoml.likelihood.Gaussian(chain), gp,
            options=geoml.models.GPOptions(verbose=False))

    def train(model):
        model.train_full(max_iter=500)

    return point, validation, "V", [None], build, train


def jura_case():
    train_data, validation = geoml.datasets.jura()
    labels = list(train_data.get("Elements").labels)
    k = len(labels)

    def build(chain):
        root = geoml.latent.BasicInput(
            geoml.data.inducing.from_kmeans(train_data, 150, seed=0),
            transform=geoml.transform.Isotropic(1.0))
        gp = geoml.latent.BasicGP(root, size=k)
        return geoml.models.VGPNetwork(
            train_data, "Elements", geoml.likelihood.MultivariateGaussian(k, chain),
            gp, options=geoml.models.GPOptions(verbose=False))

    def train(model):
        for rate, iterations in ((2e-2, 300), (5e-3, 300)):
            model.set_learning_rate(rate)
            model.train_full(max_iter=iterations)

    return train_data, validation, "Elements", labels, build, train


def _path(name, label, leaf):
    return "%s/%s" % (name, leaf) if label is None else "%s/%s/%s" % (name, label, leaf)


def scores(validation, name, labels, samples):
    pooled = {"rmse_sd": [], "crps_sd": [], "cov90": [], "psd_sd": []}
    for i, label in enumerate(labels):
        y = np.asarray(validation.values(_path(name, label, "measurements")), float).ravel()
        pred = np.asarray(validation.values(_path(name, label, "prediction")), float).ravel()
        finite = np.isfinite(y) & np.isfinite(pred)
        y, pred = y[finite], pred[finite]
        draws = samples[finite, i, :]
        sd = float(y.std())
        q05 = np.quantile(draws, 0.05, axis=-1)
        q95 = np.quantile(draws, 0.95, axis=-1)
        pooled["rmse_sd"].append(np.sqrt(np.mean((pred - y) ** 2)) / sd)
        pooled["crps_sd"].append(geoml.metrics.crps(y, draws) / sd)
        pooled["cov90"].append(np.mean((y >= q05) & (y <= q95)))
        pooled["psd_sd"].append(draws.std(axis=-1).mean() / sd)
    return {key: float(np.mean(vals)) for key, vals in pooled.items()}


def matrix(container, name, labels):
    columns = [np.asarray(container.values(_path(name, label, "measurements")),
                          float).ravel() for label in labels]
    values = np.stack(columns, axis=-1)
    return values[np.all(np.isfinite(values), axis=1)]


def round_trip(chain, data):
    chain.refresh()
    warped, _ = chain.forward(tf.constant(data))
    return float(np.max(np.abs(np.asarray(chain.backward(warped)) - data)))


def tails(chain, data, seed=0):
    """20 000 standard-normal latent draws through the back-transform:
    how many are not finite, and the largest finite value relative to the
    data's own maximum (per column, the worst column)."""
    chain.refresh()
    draws = np.random.default_rng(seed).normal(size=[20000, data.shape[1]])
    back = np.asarray(chain.backward(tf.constant(draws)))
    bad = int(np.sum(~np.isfinite(back).all(axis=1)))
    finite = np.where(np.isfinite(back), np.abs(back), 0.0)
    ratio = float(np.max(finite.max(axis=0) / np.abs(data).max(axis=0)))
    return bad, ratio


def fitted_links(chain):
    """The parametric links' trained values, one line."""
    parts = []
    for link in getattr(chain, "warpings", [chain]):
        for key in ("exponent", "scale", "skewness", "tailweight"):
            if key in link.parameters:
                value = np.asarray(link.parameters[key].get_value()).ravel()
                parts.append("%s.%s=%s" % (type(link).__name__, key,
                                           np.round(value, 2).tolist()
                                           if value.size <= 3 else
                                           "%.2f..%.2f" % (value.min(), value.max())))
    return "  ".join(parts)


def run(dataset, only=None):
    case = walker_case if dataset == "walker" else jura_case
    arms = walker_arms() if dataset == "walker" else jura_arms()
    if only:
        arms = {k: v for k, v in arms.items() if k in only}
    train_data, validation, name, labels, build, train = case()
    k = len(labels)
    data = matrix(train_data, name, labels)
    print("\n=== %s: %d arms x %d seeds ===" % (dataset, len(arms), len(SEEDS[dataset])), flush=True)
    header = "%-30s %8s %8s %8s %8s %9s %7s %10s %6s %8s" % (
        "arm", "rmse/sd", "crps/sd", "cov90", "psd/sd", "elbo", "train", "roundtrip", "nonfin", "tail/max")
    print(header, flush=True)
    for arm, factory in arms.items():
        rows = []
        for seed in SEEDS[dataset]:
            geoml.set_seed(seed)
            chain = factory(k)
            model = build(chain)
            started = time.monotonic()
            try:
                train(model)
            except Exception as error:  # a broken arm is a result, not a crash
                print("%-30s seed %d FAILED: %r" % (arm, seed, error), flush=True)
                continue
            trained = time.monotonic() - started
            model.predict(validation, n_sim=N_SIM)
            samples = model.predict_measurements(validation, n_sim=N_SIM, n_nodes=8)[name]
            result = scores(validation, name, labels, samples)
            result["elbo"] = float(model.training_log[-1])
            result["train"] = trained
            result["roundtrip"] = round_trip(chain, data)
            result["nonfinite"], result["tail"] = tails(chain, data)
            rows.append(result)
            print("%-30s %8.3f %8.3f %8.3f %8.3f %9.0f %7.0f %10.2e %6d %8.1f   [seed %d] %s" % (
                arm, result["rmse_sd"], result["crps_sd"], result["cov90"], result["psd_sd"],
                result["elbo"], result["train"], result["roundtrip"], result["nonfinite"],
                result["tail"], seed, fitted_links(chain)), flush=True)
        if len(rows) > 1:
            mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            print("%-30s %8.3f %8.3f %8.3f %8.3f %9.0f %7.0f %10.2e %6.0f %8.1f   [mean of %d]" % (
                arm.upper(), mean["rmse_sd"], mean["crps_sd"], mean["cov90"], mean["psd_sd"],
                mean["elbo"], mean["train"], mean["roundtrip"], mean["nonfinite"], mean["tail"],
                len(rows)), flush=True)


if __name__ == "__main__":
    # the dataset, then an optional comma-separated subset of arm names
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for dataset in (["walker", "jura"] if which == "both" else [which]):
        run(dataset, only)
    print("done", flush=True)
