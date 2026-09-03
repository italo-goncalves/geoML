"""The benchmark gate for flow warpings, built before any flow.

Whichever flow goes on trial -- the discrete coupling flow, the CNF after
its overhaul -- faces this fixed bar: held-out scores on Jura and on the
Macpass assays against the standing recommendation (one initialized
marginal transform), a PCA-mixed route, and the CNF as it exists today.
A new candidate is one more entry in `ARMS`: a factory taking the number
of components and returning a warping.

Scores are of measurements on data the model never saw: rmse/sd, crps/sd
(the proper score standing in for held-out log-likelihood -- sample-based,
assumption-free), 90% coverage, predicted-sd/data-sd, plus training and
prediction wall time and the warping's own back-transform round trip
(``backward(forward(y))`` against ``y``), which is where the current CNF
is known to leak.

Usage, from anywhere:

    python docs/benchmarks/flow_warpings.py jura
    python docs/benchmarks/flow_warpings.py macpass /path/to/Macpass
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402

import geoml  # noqa: E402
from geoml import warping  # noqa: E402

SEED = 1234
N_SIM = 20
SCHEDULE = ((2e-2, 300), (5e-3, 300))  # the phased schedule that avoids the
# collapse regime the trained-rotation experiment hit at a flat 300


def marginal(k):
    return warping.ChainedWarping(
        warping.Scale(k), warping.Softplus(k), warping.ZScore(k),
        warping.Spline(k, knots_per_arm=10), warping.ZScore(k))


def pca_route(k):
    return warping.ChainedWarping(
        warping.Scale(k), warping.Softplus(k), warping.ZScore(k),
        warping.RobustPCA(k, k),
        warping.Spline(k, knots_per_arm=10), warping.ZScore(k))


def cnf_route(k):
    return warping.ChainedWarping(
        warping.Scale(k), warping.Softplus(k), warping.ZScore(k),
        warping.Spline(k, knots_per_arm=10), warping.ZScore(k),
        warping.ContinuousNormalizingFlow(k))


def cp_route(k):
    return warping.ChainedWarping(
        warping.Scale(k), warping.Softplus(k), warping.ZScore(k),
        warping.Spline(k, knots_per_arm=10), warping.ZScore(k),
        warping.TensorProductFlow(k, grid=9, rank=5))


ARMS = {"marginal": marginal, "pca": pca_route, "cnf": cnf_route,
        "cp": cp_route}


def jura_case():
    train, validation = geoml.datasets.jura()
    labels = list(train.get("Elements").labels)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(train, 150, seed=0),
        transform=geoml.transform.Isotropic(1.0))
    return train, validation, "Elements", labels, root


def macpass_case(path):
    holes = geoml.datasets.macpass(path)
    samples = holes.composite(5.0).as_point_data()

    # the biggest of the three widely separated deposits, as in
    # docs/variable-block-models.md section 11
    xy = np.asarray(samples.coordinates)[:, :2]
    group = hierarchy.fcluster(hierarchy.linkage(xy, method="single"),
                               900.0, criterion="distance")
    biggest = np.argmax(np.bincount(group))
    lo = xy[group == biggest].min(axis=0)
    hi = xy[group == biggest].max(axis=0)
    z = np.asarray(samples.coordinates)[:, 2]
    deposit = samples.subset_region(
        [lo[0] - 100, lo[1] - 100, z.min() - 10],
        [hi[0] + 100, hi[1] + 100, z.max() + 10])

    # complete cases of the three grades -- a mixing warping accepts
    # nothing less, so every arm is scored on the same rows
    labels = ["Ag_ppm", "Pb_pct", "Zn_pct"]
    values = np.stack(
        [np.asarray(deposit.values(c + "/measurements"), dtype=float).ravel()
         for c in labels], axis=-1)
    keep = np.all(np.isfinite(values), axis=1)
    coords = np.asarray(deposit.coordinates)[keep]
    values = values[keep]
    hole = np.asarray(deposit.values("_metadata/HOLEID"))[keep]

    # a fifth of the holes held out whole -- rows of one hole never split
    rng = np.random.default_rng(7)
    unique = np.unique(hole)
    held = rng.choice(unique, size=max(1, len(unique) // 5), replace=False)
    is_val = np.isin(hole, held)

    def build(mask):
        frame = pd.DataFrame(coords[mask], columns=["X", "Y", "Z"])
        data = geoml.data.PointData(frame, ["X", "Y", "Z"])
        data.add_vector_variable("assay", labels, values[mask])
        return data

    train, validation = build(~is_val), build(is_val)
    print("macpass: %d train rows, %d validation rows (%d of %d holes)"
          % (train.n_data, validation.n_data, len(held), len(unique)),
          flush=True)

    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(train, 400, seed=0),
        transform=geoml.transform.Anisotropy3D(
            maxrange=120.0, midrange_fct=0.6, minrange_fct=0.25,
            azimuth=45.0))
    return train, validation, "assay", labels, root


def scores(validation, name, labels, samples):
    pooled = {"rmse_sd": [], "crps_sd": [], "cov90": [], "psd_sd": []}
    for i, label in enumerate(labels):
        y = np.asarray(validation.values("%s/%s/measurements" % (name, label)),
                       dtype=float).ravel()
        pred = np.asarray(validation.values("%s/%s/prediction" % (name, label)),
                          dtype=float).ravel()
        finite = np.isfinite(y)
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


def round_trip(model, validation, name, labels):
    """Three numbers: the chain's max-abs and mean-abs round trip in the
    data's units, and -- when the chain ends in a flow -- the flow's own
    round trip in latent units, which separates a stiff field from a tail
    point the marginal links' inverses blow up."""
    matrix = np.stack(
        [np.asarray(validation.values("%s/%s/measurements" % (name, label)),
                    dtype=float).ravel() for label in labels], axis=-1)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    chain = model.likelihoods[0].warping
    chain.refresh()
    y = tf.constant(matrix, tf.float64)
    warped, _ = chain.forward(y)
    gap = np.abs(np.asarray(chain.backward(warped)) - matrix)

    latent = float("nan")
    links = getattr(chain, "warpings", [chain])
    if isinstance(links[-1], warping._ContinuousFlow):
        before = y
        for link in links[:-1]:
            before, _ = link.forward(before)
        through, _ = links[-1].forward(before)
        latent = float(np.max(np.abs(
            np.asarray(links[-1].backward(through)) - np.asarray(before))))
    return float(gap.max()), float(gap.mean()), latent


def run_arm(arm, factory, case):
    train, validation, name, labels, root = case()
    k = len(labels)

    gp = geoml.latent.BasicGP(root, size=k)
    model = geoml.models.VGPNetwork(
        train, name, geoml.likelihood.MultivariateGaussian(k, factory(k)),
        gp, options=geoml.models.GPOptions(verbose=False))

    started = time.monotonic()
    for rate, iterations in SCHEDULE:
        model.set_learning_rate(rate)
        model.train_full(max_iter=iterations)
    trained = time.monotonic() - started

    started = time.monotonic()
    model.predict(validation, n_sim=N_SIM)
    samples = model.predict_measurements(validation, n_sim=N_SIM,
                                         n_nodes=32)[name]
    predicted = time.monotonic() - started

    result = scores(validation, name, labels, samples)
    result["train_s"] = trained
    result["predict_s"] = predicted
    worst, typical, latent = round_trip(model, validation, name, labels)
    result["round_trip"] = worst
    result["round_trip_mean"] = typical
    result["latent_round_trip"] = latent
    result["elbo"] = float(model.training_log[-1])
    return result


def run(dataset, path=None, arms=None):
    case = jura_case if dataset == "jura" else (lambda: macpass_case(path))
    print("dataset: %s" % dataset, flush=True)
    for arm, factory in ARMS.items():
        if arms is not None and arm not in arms:
            continue
        geoml.set_seed(SEED)
        try:
            result = run_arm(arm, factory, case)
        except Exception as error:  # a broken arm is a result, not a crash
            print("%-9s FAILED: %r" % (arm, error), flush=True)
            continue
        print("%-9s rmse/sd %.3f  crps/sd %.3f  cov90 %.3f  psd/sd %.3f  "
              "elbo %.0f  train %.0f s  predict %.0f s  roundtrip %.2e "
              "(mean %.2e, latent %.2e)"
              % (arm, result["rmse_sd"], result["crps_sd"], result["cov90"],
                 result["psd_sd"], result["elbo"], result["train_s"],
                 result["predict_s"], result["round_trip"],
                 result["round_trip_mean"], result["latent_round_trip"]),
              flush=True)


if __name__ == "__main__":
    # dataset, then the Macpass directory (or "-" for Jura), then an
    # optional comma-separated subset of arms
    which = sys.argv[1] if len(sys.argv) > 1 else "jura"
    macpass_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" \
        else None
    chosen = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    run(which, macpass_path, chosen)
