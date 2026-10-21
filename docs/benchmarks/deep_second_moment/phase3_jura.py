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

"""Step 3 of the deep second moment: training, on chapter 16's network.

Chapter 16's Jura model as the chapter builds it -- a GP of two outputs
walked by a `GPWalk` and read by an isotropic Matern 3/2 rock GP, a
geological trend added to the metals' own GP, four experts, 250
iterations -- under a seed of the run's. The rock GP reads an uncertain
input with uncertain inducing points, so the conditioned second moment
is what the arms change:

  moment   the 0.9.3 moment bound (the conditioned second moment off)
  second   the bound with the conditioned second moment, from scratch

(The doubly stochastic warm-up of the folded section needs a leaf reading
a GP or a `Concatenate` with one expert, which this network is not.)
Scored on the 100 held-out sites as the chapter scores them, the model
predicting with the second moment either way: the metals' RMSE and CRPS
over each element's standard deviation, averaged, and goodness; the
rock's accuracy, and its Brier and log scores averaged over the
categories. Appends JSON lines to `phase3_jura.jsonl`.

Usage: python phase3_jura.py ITERATIONS ARM SEED
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def build(seed):
    geoml.set_seed(seed)
    train, validation = geoml.datasets.jura()
    elements = list(train.get("Elements").labels)
    rocks = list(train.get("Rock").labels)
    backbone = geoml.data.inducing.combine(
        train, geoml.data.Grid2D(start=[0, 0], end=[6, 6], n=[21, 21]))
    inducing = geoml.data.inducing.experts(backbone, 4, seed=1234)
    root = geoml.latent.BasicInput(
        inducing_points=inducing,
        transform=geoml.transform.Isotropic(0.05))
    displacement = geoml.latent.BasicGP(
        root, size=2, kernel=geoml.kernels.Gaussian())
    walked = geoml.latent.GPWalk(displacement, n_steps=5)
    rock_gp = geoml.latent.BasicGP(
        walked, size=len(rocks), kernel=geoml.kernels.Matern32(),
        isotropic=True)
    trend = geoml.latent.Linear(rock_gp, size=len(elements), unit_norm=False)
    metal_gp = geoml.latent.BasicGP(
        root, size=len(elements), kernel=geoml.kernels.Spherical())
    leaves = [rock_gp, geoml.latent.LinearCombination(trend, metal_gp)]
    warping = geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(len(elements)),
        geoml.warping.RobustPCA(len(elements), len(elements)),
        geoml.warping.ZScore(len(elements)),
        geoml.warping.SinhArcsinh(len(elements)),
        geoml.warping.ZScore(len(elements)))
    model = geoml.models.VGPNetwork(
        data=train,
        variables={
            "Rock": geoml.likelihood.CategoricalGaussianIndicator(len(rocks)),
            "Elements": geoml.likelihood.Laplace(warping=warping)},
        latent_network=leaves,
        options=geoml.models.GPOptions(prediction_batch_size=1000,
                                       jitter=1e-6, verbose=False))
    return model, validation


def scores(model, validation):
    model.predict(validation, n_sim=30, include_noise=True)
    metals = validation.get("Elements").compute_metrics()
    sd = np.nanstd(np.asarray(validation.get("Elements").get_measurements()[0],
                              dtype=float), axis=0)
    rmse = np.asarray(metals.loc["Root Mean Square Error (prediction)"],
                      dtype=float)
    crps = np.asarray(metals.loc["CRPS (simulations)"], dtype=float)
    goodness = np.asarray(metals.loc["Goodness (simulations)"], dtype=float)
    rock = validation.get("Rock").compute_metrics()
    measured = np.asarray(validation.get("Rock").get_measurements()[0])
    predicted = np.asarray(validation.values("Rock/predicted"))
    labels = list(validation.get("Rock").labels)
    if measured.ndim > 1:          # one indicator column per category
        measured = np.asarray(labels)[np.argmax(measured, axis=1)]
    return dict(metals_rmse_sd=float(np.mean(rmse / sd)),
                metals_crps_sd=float(np.mean(crps / sd)),
                metals_goodness=float(np.mean(goodness)),
                rock_accuracy=float(np.mean(predicted.astype(str)
                                            == measured.astype(str))),
                rock_brier=float(np.mean(np.asarray(
                    rock.loc["Brier score"], dtype=float))),
                rock_log=float(np.mean(np.asarray(
                    rock.loc["Log score"], dtype=float))))


def save_maps(model, arm, seed):
    """The chapter's map grid at half its resolution (101 x 101): the
    rock's predicted category and probabilities, three realizations of it
    (each the largest of its categories' latent draws, the indicator
    likelihood's rule), and every metal's prediction and three
    realizations, to `maps/jura_<arm>_<seed>.npz`."""
    grid = geoml.data.Grid2D(start=[0, 0], end=[6, 6], n=[101, 101])
    model.predict(grid, n_sim=3, include_noise=True)
    rock = grid.get("Rock")
    labels = list(rock.labels)
    latent = np.stack([np.asarray(grid.values("Rock/%s/simulations" % lab),
                                  dtype=float)
                       for lab in labels])                 # [K, n, 3]
    elements = list(grid.get("Elements").labels)
    folder = os.path.join(gm.FIGURES, "maps")
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(
        os.path.join(folder, "jura_%s_%d.npz" % (arm, seed)),
        shape=np.array([101, 101]),
        rocks=np.array(labels), elements=np.array(elements),
        rock_predicted=np.asarray(
            [labels.index(str(v)) for v in grid.values("Rock/predicted")]),
        rock_realizations=np.argmax(latent, axis=0),
        rock_entropy=np.asarray(grid.values("Rock/entropy"), dtype=float),
        metal_prediction=np.stack([
            np.asarray(grid.values("Elements/%s/prediction" % el),
                       dtype=float) for el in elements], -1),
        metal_realizations=np.stack([
            np.asarray(grid.get("Elements/%s" % el).get_simulations(),
                       dtype=float) for el in elements], -1))


def main(iterations, arm, seed):
    model, validation = build(seed)
    start = time.perf_counter()
    if arm == "moment":
        net._UNCERTAIN_POINTS_SECOND_MOMENT = False
        model.train_full(iterations)
        net._UNCERTAIN_POINTS_SECOND_MOMENT = True
    elif arm == "second":
        model.train_full(iterations)
    seconds = time.perf_counter() - start
    row = dict(arm=arm, seed=seed, iterations=iterations,
               seconds=round(seconds, 1),
               bound=float(model.training_log[-1]),
               **scores(model, validation))
    print(json.dumps(row), flush=True)
    if os.environ.get("MAPS"):
        # a rerun for the maps: its row is already in the results
        save_maps(model, arm, seed)
        return
    with open(os.path.join(gm.FIGURES, "phase3_jura.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
