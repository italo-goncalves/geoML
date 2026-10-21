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

"""Step 3 of the deep second moment: training, on Walker Lake.

The folded section's two networks on Walker's 470 samples: a hidden GP of
two outputs on the coordinates, and a leaf GP reading it alone (`dgp`) or
beside the coordinates (`concat`); 120 inducing points by k-means, one
expert; Gaussian kernels; chapter 15's warping (`BoxCox(shift=1)` then
`ZScore`) under a Gaussian likelihood. Arms, as on the folded section:

  moment   the 0.9.3 moment bound (the conditioned second moment off)
  second   the bound with the conditioned second moment, from scratch
  warmup   200 steps of the 64-draw doubly stochastic bound, then `second`

Scored against the exhaustive field at every sixth row and column (2 150
points): by the leaf's moments with the noise, in warped units (log score
a point, the share inside the central 90%, CRPS); and in V's units by 64
realizations of the ground and by samples of a measurement against the
truth (CRPS, coverage, spread). The exhaustive values carry the nugget the
model calls noise, so the measurement samples are the fair check; the
ground's realizations under-cover by however much the model put there.
Appends JSON lines to `phase3_walker.jsonl`.

Usage: python phase3_walker.py ITERATIONS NETWORK ARM SEED
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "doubly_stochastic"))
import gate_moments as gm  # noqa: E402
import ds_prototype as ds  # noqa: E402
import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402

N_IND, STEP, N_SIM = 120, 6, 64


def build(walker, network, seed):
    geoml.set_seed(seed)
    ip = geoml.data.inducing.from_kmeans(walker, N_IND, seed=seed)
    root = geoml.latent.BasicInput(ip, transform=geoml.transform.Isotropic(50))
    hidden = geoml.latent.BasicGP(root, size=2)
    below = hidden if network == "dgp" \
        else geoml.latent.Concatenate(hidden, root)
    leaf = geoml.latent.BasicGP(below, size=1)
    warping = geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1, shift=1.0), geoml.warping.ZScore(1))
    return geoml.models.VGPNetwork(
        walker, "V", geoml.likelihood.Gaussian(warping), leaf,
        options=geoml.models.GPOptions(verbose=False, propagation="joint",
                                       expert_propagation="independent"))


def targets(grid):
    rows = np.arange(grid.n_data).reshape(300, 260)[::STEP, ::STEP].ravel()
    x = np.asarray(grid.coordinates, dtype=float)[rows]
    truth = np.asarray(grid.values("V/measurements"), dtype=float)[rows]
    return x, truth


def moment_scores(model, x, truth):
    """The leaf's moments with the noise, in warped units."""
    likelihood = model.likelihoods[0]
    with model._propagation():
        model._refresh(model.options.jitter)
        mu, var = model.leaves[0].propagate(tf.constant(x))
    mu, var = np.asarray(mu)[:, 0], np.asarray(var)[:, 0]
    likelihood.warping.refresh()
    y, log_d = likelihood.warping.forward(tf.constant(truth[:, None]))
    y, log_d = np.asarray(y)[:, 0], np.asarray(log_d).ravel()
    noise = float(np.asarray(likelihood.parameters["noise"].get_value())
                  .ravel()[0])
    s = np.sqrt(var + noise)
    z = (y - mu) / s
    pit = stats.norm.cdf(z)
    crps = s * (z * (2 * pit - 1) + 2 * stats.norm.pdf(z)
                - 1 / np.sqrt(np.pi))
    return dict(log_score=float(np.mean(stats.norm.logpdf(z) - np.log(s)
                                        + log_d)),
                cover90=float(np.mean((pit > 0.05) & (pit < 0.95))),
                crps=float(np.mean(crps)))


def sample_scores(samples, truth):
    """CRPS, 90% coverage and spread of `samples` `[S, n]` against the
    truth, the CRPS's second term off the sorted sample."""
    s = np.sort(samples, axis=0)
    k = s.shape[0]
    pairs = 2.0 * np.sum((2 * np.arange(1, k + 1) - k - 1)[:, None] * s,
                         0) / k ** 2
    crps = np.mean(np.abs(samples - truth[None]), 0) - 0.5 * pairs
    lo, hi = np.quantile(samples, [0.05, 0.95], axis=0)
    return dict(crps=float(np.mean(crps)),
                cover90=float(np.mean((truth >= lo) & (truth <= hi))),
                spread=float(np.mean(np.std(samples, 0))))


def field_scores(model, x, truth):
    """In V's units: 64 realizations of the ground, and samples of a
    measurement -- the ground with the noise drawn, which is what the
    exhaustive field records (its values carry the nugget the model calls
    noise), 64 realizations of 8 noise values each -- against the truth."""
    target = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(target, n_sim=N_SIM, include_noise=True)
    sims = np.asarray(target.get("V").get_simulations(), dtype=float).T
    prediction = np.asarray(target.values("V/prediction"), dtype=float)
    measured = np.asarray(model.predict_measurements(
        target, n_sim=N_SIM, n_nodes=8)["V"], dtype=float)
    measured = measured.reshape(measured.shape[0], -1).T
    return dict(ground=sample_scores(sims, truth),
                measurement=sample_scores(measured, truth),
                rmse=float(np.sqrt(np.mean((prediction - truth) ** 2))))


def save_maps(model, grid, network, arm, seed):
    """Every other row and column of the field: the truth, the prediction,
    three realizations of the ground and one draw of a measurement, to
    `maps/walker_<network>_<arm>_<seed>.npz`."""
    rows = np.arange(grid.n_data).reshape(300, 260)[::2, ::2]
    x = np.asarray(grid.coordinates, dtype=float)[rows.ravel()]
    target = geoml.data.PointData.from_array(x, ["X", "Y"])
    model.predict(target, n_sim=3, include_noise=True)
    measured = np.asarray(model.predict_measurements(
        target, n_sim=1, n_nodes=1)["V"], dtype=float).reshape(len(x), -1)
    folder = os.path.join(gm.FIGURES, "maps")
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(
        os.path.join(folder, "walker_%s_%s_%d.npz" % (network, arm, seed)),
        shape=np.array(rows.shape),
        truth=np.asarray(grid.values("V/measurements"),
                         dtype=float)[rows.ravel()],
        prediction=np.asarray(target.values("V/prediction"), dtype=float),
        realizations=np.asarray(target.get("V").get_simulations(),
                                dtype=float),
        measurement=measured[:, 0])


def main(iterations, network, arm, seed):
    walker, grid = geoml.datasets.walker()
    model = build(walker, network, seed)
    start = time.perf_counter()
    if arm == "moment":
        net._UNCERTAIN_POINTS_SECOND_MOMENT = False
        model.train_full(iterations)
        net._UNCERTAIN_POINTS_SECOND_MOMENT = True
    elif arm == "second":
        model.train_full(iterations)
    elif arm == "warmup":
        ds.train_ds(model, 200, 64)
        model.train_full(iterations - 200)
    seconds = time.perf_counter() - start
    x, truth = targets(grid)
    row = dict(network=network, arm=arm, seed=seed, iterations=iterations,
               seconds=round(seconds, 1),
               bound=float(model.training_log[-1]),
               moments=moment_scores(model, x, truth),
               field=field_scores(model, x, truth))
    print(json.dumps(row), flush=True)
    if not os.environ.get("MAPS"):
        # a rerun for the maps has its row in the results already
        with open(os.path.join(gm.FIGURES, "phase3_walker.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    save_maps(model, grid, network, arm, seed)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
