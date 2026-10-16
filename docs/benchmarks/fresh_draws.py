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

"""Fresh training draws against a fixed set (`GPOptions(fresh_training_draws=)`).

Usage: python docs/benchmarks/fresh_draws.py CASE DRAWS FRESH INIT DRAW_SEED
  CASE       mixture | jura | deep16 | deep64
  DRAWS      training_samples (ignored by the deep cases, which name theirs)
  FRESH      0 or 1
  INIT       the seed given to `geoml.set_seed` before construction
  DRAW_SEED  `options.seed` while training: only the training draws see it
One JSON line to docs/benchmarks/figures/fresh_draws/results.jsonl.

Three places a training draw can act:

- `mixture`: gate 1 of `latent.GaussianMixture` (`gaussian_mixture_gates.py`,
  the `smooth` arm), a leaf that is not Gaussian and trains on its
  realizations; scored against the noiseless truth on 400 points.
- `jura`: Jura's seven metals under a warping that mixes them (`BoxCox ->
  RobustPCA -> ZScore -> SinhArcsinh -> ZScore`, a multivariate Gaussian),
  whose bound is a Monte Carlo average over the latent realizations;
  scored on Jura's held-out set.
- `deep16`, `deep64`: the folded section's GP beside the coordinates,
  trained on the doubly stochastic bound of `doubly_stochastic/` with 16 or
  64 draws through the hidden layer -- the basin question.

Measured for each: the objective twice -- the training log's own (the mean
of its last 50 steps) and an honest one, the bound re-evaluated at the
final parameters over 20 independent sets of 200 draws -- whose gap is how
far the parameters fitted their own draws; the accuracy; the predictions,
kept so that runs differing only in `DRAW_SEED` can be compared (the
prediction itself always drawn under one common seed); and the time a
step.
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "fresh_draws")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "doubly_stochastic"))

import geoml  # noqa: E402
import geoml.latent as latent  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.metrics as metrics  # noqa: E402
import geoml.transform as tr  # noqa: E402

PREDICTION_SEED = 999


def honest_bound(model, samples=200, sets=20):
    """The training bound at the final parameters, averaged over `sets`
    independent draw sets of `samples`: its mean and standard error."""
    inputs = [model.data.variables[v].training_input()
              for v in model.variables]
    x = tf.constant(model.data.coordinates, tf.float64)
    y = tf.constant(model.y, tf.float64)
    hv = tf.constant(model.has_value, tf.float64)
    x_var = tf.constant(model.data.get_batched_variance()[0], tf.float64)
    values = []
    with model._propagation():
        for k in range(sets):
            values.append(float(model._training_elbo(
                x, y, hv, inputs, x_var=x_var, samples=samples,
                seed=tf.constant(10 ** 6 + k, tf.int64),
                jitter=model.options.jitter)))
    return float(np.mean(values)), float(np.std(values) / np.sqrt(sets))


def timed_training(model, iterations):
    model.train_full(10)
    start = time.perf_counter()
    model.train_full(iterations - 10)
    return (time.perf_counter() - start) / (iterations - 10)


# --------------------------------------------------------------------------- #
# the GaussianMixture gate
# --------------------------------------------------------------------------- #
def mixture_case(draws, fresh, init, draw_seed):
    import gaussian_mixture_gates as gm
    data = gm.data("split")
    geoml.set_seed(init)
    weights = latent.BasicGP(gm.root(6.0), size=2)
    components = gm.root(4.0)
    leaf = latent.GaussianMixture(
        weights, [latent.BasicGP(components, size=1) for _ in range(2)])
    model = geoml.models.VGPNetwork(
        data, "v", lk.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False, training_samples=draws,
                                       fresh_training_draws=bool(fresh)))
    model.options.seed = draw_seed
    per = timed_training(model, gm.ITERATIONS)
    honest = honest_bound(model)

    x = np.linspace(0.0, 10.0, 400)
    a, b = gm.curves(x)
    truth = np.where(x >= gm.BOUNDARY, b, a)
    target = geoml.data.PointData.from_array(x[:, None], ["X"])
    model.options.seed = PREDICTION_SEED
    model.predict(target, n_sim=200)
    prediction = np.asarray(target.values("v/prediction"), dtype=float)
    sims = np.asarray(target.variables["v"].get_simulations())
    return model, per, honest, dict(
        rmse=float(np.sqrt(np.mean((prediction - truth) ** 2))),
        crps=float(metrics.crps(truth, sims))), prediction


# --------------------------------------------------------------------------- #
# Jura's metals under a mixing warping
# --------------------------------------------------------------------------- #
def jura_case(draws, fresh, init, draw_seed, iterations=600):
    train, held = geoml.datasets.jura()
    labels = list(train.get("Elements").labels)
    k = len(labels)
    geoml.set_seed(init)
    root = latent.BasicInput(
        geoml.data.inducing.from_kmeans(train, 120, seed=init),
        transform=tr.Isotropic(0.5))
    leaf = latent.BasicGP(root, size=k, kernel=geoml.kernels.Matern32())
    warping = geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(k), geoml.warping.RobustPCA(k, k),
        geoml.warping.ZScore(k), geoml.warping.SinhArcsinh(k),
        geoml.warping.ZScore(k))
    model = geoml.models.VGPNetwork(
        train, {"Elements": lk.MultivariateGaussian(k, warping=warping)}, None,
        leaf, options=geoml.models.GPOptions(
            verbose=False, jitter=1e-6, training_samples=draws,
            fresh_training_draws=bool(fresh)))
    model.options.seed = draw_seed
    per = timed_training(model, iterations)
    honest = honest_bound(model)

    model.options.seed = PREDICTION_SEED
    model.predict(held, n_sim=50)
    truth = np.column_stack([held.values("Elements/%s/measurements" % e)
                             for e in labels]).astype(float)
    pred = np.column_stack([held.values("Elements/%s/prediction" % e)
                            for e in labels]).astype(float)
    sd = np.nanstd(truth, axis=0)
    sims = [np.asarray(held.get("Elements/%s" % e).get_simulations())
            for e in labels]
    keep = np.all(np.isfinite(truth), axis=1)
    rmse = np.sqrt(np.nanmean((pred - truth) ** 2, axis=0)) / sd
    crps = np.array([metrics.crps(truth[keep, j], sims[j][keep])
                     for j in range(k)]) / sd
    return model, per, honest, dict(
        rmse=float(np.mean(rmse)), crps=float(np.mean(crps))), pred.ravel()


# --------------------------------------------------------------------------- #
# the folded section, draws through the hidden layer
# --------------------------------------------------------------------------- #
def deep_case(draws, fresh, init, draw_seed):
    import expected_kernel as ek
    import ds_prototype as ds
    import run_folded as rf
    data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
    new, holes = ek.drillholes(ek.N_NEW, ek.NEW_SEED)
    model = rf.build(data, "dgpc", seed=init)
    _, per = ds.train_ds(model, 2000, draws, ds.COUPLED, fresh=bool(fresh))
    inputs = [model.data.variables[v].training_input()
              for v in model.variables]
    x = tf.constant(model.data.coordinates, tf.float64)
    y = tf.constant(model.y, tf.float64)
    hv = tf.constant(model.has_value, tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        kl = float(tf.add_n([n.kl_divergence() for n in model._nodes()]))
        prior = float(model.log_prior())
        values = [float(ds.ds_data_term(model, x, y, hv, inputs, 200,
                                        10 ** 6 + k, ds.COUPLED))
                  for k in range(20)]
    honest = (float(np.mean(values) - kl + prior),
              float(np.std(values) / np.sqrt(20)))
    s = rf.summary(*rf.mixture_scores(model, new, ds.COUPLED), holes)
    return model, per, honest, dict(score_per_hole=s["score_per_hole"],
                                    crps=s["crps"], cover90=s["cover90"]), \
        None


def main(argv):
    case, draws, fresh, init, draw_seed = \
        argv[0], int(argv[1]), int(argv[2]), int(argv[3]), int(argv[4])
    if case == "mixture":
        model, per, honest, scores, pred = mixture_case(draws, fresh, init,
                                                        draw_seed)
    elif case == "jura":
        model, per, honest, scores, pred = jura_case(draws, fresh, init,
                                                     draw_seed)
    else:
        draws = int(case[4:])
        model, per, honest, scores, pred = deep_case(draws, fresh, init,
                                                     draw_seed)
    log = np.asarray(model.training_log)
    row = dict(case=case, draws=draws, fresh=bool(fresh), init=init,
               draw_seed=draw_seed, ms_per_iter=1000 * per,
               own_bound=float(log[-50:].mean()),
               honest_bound=honest[0], honest_se=honest[1],
               optimism=float(log[-50:].mean() - honest[0]),
               step_noise=float(np.std(np.diff(log[-200:]))), **scores,
               prediction=None if pred is None
               else [round(float(v), 5) for v in pred])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "prediction"}))


if __name__ == "__main__":
    main(sys.argv[1:])
