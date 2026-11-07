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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Experts weighted by their variances against fixed weights, tempered or
not, blended by their moments or balanced (`GPOptions(expert_weights=,
expert_temperature=, expert_blend=)`), on four models:

- `walker`: Walker Lake's V from its 470 samples, a Spherical GP on
  `grid_experts` (step 10, blocks of 8: 16 experts), scored against the
  exhaustive grid;
- `jura`: Jura's Co from its 259 training samples, a Matern32 GP on four
  experts of 120 k-means points, scored on the 100 validation samples;
- `tom`: Tom East's rock, ore against waste, on 20 experts of 500 k-means
  points, a fifth of the holes held out (`expert_overlap.py`'s model);
- `fold`: the folded vein's walk model (`walk_cost.py`'s, three experts of
  its 17 x 21 inducing points), scored against the true vein on the
  161 x 201 grid. The balanced blend is refused there.

Variants: `V1` today's rule, `V2` it tempered at 2, `C1` and `C2` fixed
weights, `P1` and `P2` fixed and balanced. Each run is one process, so
that a model's graphs are not kept by the next (CLAUDE.md, building models
in a loop).

Scores against the truth -- rmse, CRPS over the realizations, the 90%
interval's coverage for a grade; AUC, Brier and the log score for a rock --
overall and in the experts' seams (the locations where today's rule gives
no expert 0.9 of the weight), and the realizations' roughness there: the
sd of differences between neighbouring cells of the grid, in the seams
over away from them, against the same ratio of the truth (walker and
fold). Then the time and peak memory of predicting the grid (or, for tom,
a 20 m grid over the data's box) twice, the second timed.

Usage: python fixed_expert_weights.py CASE VARIANT [seed=1] [iters=]
                                     [ref=] [floor=]

`ref` sets the reference kernel's length in inducing spacings (4 by
default; the first runs were made at 1), `floor` the share of a
location's largest raw weight under which a weight is zero. Prints one
JSON line.

Measured 2026-11-06, RTX 4060 Ti / Ryzen 9 7950X (WSL), means of two
seeds; the full tables, the sweeps of the length and the floor and the
GeoScape network's time and memory are in `docs/expert-weights.md`:

| | Walker rmse | Jura rmse | Tom East AUC / log | vein Brier |
|---|---|---|---|---|
| V1 | 161 | 2.58 | 0.798 / 0.304 | 0.064 |
| V2 | 159 | 2.67 | 0.819 / 0.350 | 0.067 |
| C1 (ref 4, floor 1e-2) | 173 | 2.69 | 0.849 / 0.286 | 0.049 |
| P1 (ref 4, floor 1e-3) | 167 | 2.65 | -- | refused |
"""
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr

VARIANTS = {
    "V1": dict(),
    "V2": dict(expert_temperature=2.0),
    "C1": dict(expert_weights="coverage"),
    "C2": dict(expert_weights="coverage", expert_temperature=2.0),
    "P1": dict(expert_weights="coverage", expert_blend="balanced"),
    "P2": dict(expert_weights="coverage", expert_blend="balanced",
               expert_temperature=2.0),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def crps(truth, sims):
    """The sample CRPS, row by row: `E|X - y| - E|X - X'| / 2`."""
    first = np.mean(np.abs(sims - truth[:, None]), axis=1)
    s = np.sort(sims, axis=1)
    n = s.shape[1]
    k = np.arange(1, n + 1)
    spread = np.sum((2 * k - n - 1) * s, axis=1) / n ** 2
    return first - spread


def seams(model, container):
    """Where today's rule gives no expert 0.9 of the weight."""
    options = model.options.__dict__
    saved = {k: options.get(k) for k in VARIANTS["P2"]}
    for key, value in dict(expert_weights="variance", expert_temperature=1.0,
                           expert_blend="moments").items():
        options[key] = value
    try:
        table = model.expert_weights(container)
    finally:
        for key, value in saved.items():
            if value is None:
                options.pop(key, None)
            else:
                options[key] = value
    return np.max(table, axis=1) < 0.9


def roughness(field, shape, mask):
    """The sd of differences between neighbouring cells where `mask` holds,
    over where it does not, along both grid axes."""
    f = field.reshape(shape + (-1,))
    m = mask.reshape(shape)
    inside, outside = [], []
    for axis in (0, 1):
        d = np.diff(f, axis=axis)
        both = np.minimum(np.take(m, range(m.shape[axis] - 1), axis),
                          np.take(m, range(1, m.shape[axis]), axis))
        inside.append(d[both.astype(bool)])
        outside.append(d[~both.astype(bool)])
    inside, outside = np.concatenate(inside), np.concatenate(outside)
    return float(np.std(inside) / np.std(outside))


def timed_prediction(model, make, n_sim):
    import tensorflow as tf
    target = make()
    model.predict(target, n_sim=n_sim)
    gpu = bool(tf.config.list_physical_devices("GPU"))
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    target = make()
    start = time.perf_counter()
    model.predict(target, n_sim=n_sim)
    row = dict(predict_s=round(time.perf_counter() - start, 2),
               rss_gb=round(resource.getrusage(
                   resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
    if gpu:
        row["gpu_peak_gb"] = round(tf.config.experimental.get_memory_info(
            "GPU:0")["peak"] / 1e9, 3)
    return target, row


def options(variant, **extra):
    return geoml.models.GPOptions(verbose=False, **dict(VARIANTS[variant],
                                                         **extra))


def walker(variant, seed, iters):
    geoml.set_seed(seed)
    data, grid = geoml.datasets.walker()
    experts = geoml.data.inducing.grid_experts(grid, 10.0, block=8)
    root = gl.BasicInput(experts, transform=tr.Isotropic(50))
    gp = gl.BasicGP(root, size=1, kernel=kr.Spherical())
    likelihood = lk.Gaussian(geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1, shift=None), geoml.warping.ZScore(1)))
    model = geoml.models.VGPNetwork(data, "V", likelihood, gp,
                                    options=options(variant))
    start = time.perf_counter()
    model.train_full(iters or 500)
    row = dict(train_s=round(time.perf_counter() - start, 1),
               bound=float(model.training_log[-1]),
               experts=len(experts))

    def make():
        return geoml.data.Grid2D(start=[1, 1], n=[260, 300], step=[1, 1])
    target, timing = timed_prediction(model, make, 20)
    row.update(timing)
    truth = np.asarray(grid.values("V/measurements"), dtype=float)
    mean = np.asarray(target.values("V/prediction"), dtype=float)
    sims = np.asarray(target.get("V").simulations[:, :], dtype=float)
    seam = seams(model, target)
    low, high = np.quantile(sims, [0.05, 0.95], axis=1)
    inside = (truth >= low) & (truth <= high)
    score = crps(truth, sims)
    row.update(
        rmse=float(np.sqrt(np.mean((mean - truth) ** 2))),
        crps=float(np.mean(score)), coverage90=float(np.mean(inside)),
        seam_share=float(np.mean(seam)),
        seam_rmse=float(np.sqrt(np.mean((mean - truth)[seam] ** 2))),
        seam_crps=float(np.mean(score[seam])),
        seam_coverage90=float(np.mean(inside[seam])),
        roughness=roughness(sims, (300, 260), seam)
        / roughness(truth[:, None], (300, 260), seam))
    return row


def jura(variant, seed, iters):
    geoml.set_seed(seed)
    folder = os.path.join(os.path.dirname(geoml.__file__), "sample_data")
    frames = [pd.read_csv(os.path.join(folder, name))
              for name in ("jura_train.csv", "jura_val.csv")]
    train, test = (geoml.data.PointData(f, ["Xloc", "Yloc"]) for f in frames)
    for container, frame in zip((train, test), frames):
        container.add_continuous_variable(
            "Co", np.asarray(frame["Co"], dtype=float))
    ip = geoml.data.inducing.from_kmeans(train, 120, seed=0)
    experts = geoml.data.inducing.experts(ip, 4, seed=0)
    root = gl.BasicInput(experts, transform=tr.Isotropic(0.5))
    gp = gl.BasicGP(root, size=1, kernel=kr.Matern32())
    likelihood = lk.Gaussian(geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1), geoml.warping.ZScore(1)))
    model = geoml.models.VGPNetwork(train, "Co", likelihood, gp,
                                    options=options(variant))
    start = time.perf_counter()
    model.train_full(iters or 500)
    row = dict(train_s=round(time.perf_counter() - start, 1),
               bound=float(model.training_log[-1]))
    truth = np.asarray(test.values("Co/measurements"), dtype=float)
    model.predict(test, n_sim=200)
    mean = np.asarray(test.values("Co/prediction"), dtype=float)
    samples = model.predict_measurements(test, n_sim=50, n_nodes=4)["Co"]
    samples = np.asarray(samples, dtype=float).reshape(len(truth), -1)
    low, high = np.quantile(samples, [0.05, 0.95], axis=1)
    seam = seams(model, test)
    row.update(rmse=float(np.sqrt(np.mean((mean - truth) ** 2))),
               crps=float(np.mean(crps(truth, samples))),
               coverage90=float(np.mean((truth >= low) & (truth <= high))),
               seam_share=float(np.mean(seam)))
    return row


def tom(variant, seed, iters):
    from sklearn.metrics import roc_auc_score
    path = os.path.join(HERE, "..", "..", "local_data", "tom_east.csv")
    frame = pd.read_csv(path)
    rng = np.random.default_rng(0)
    holes = frame["HOLEID"].unique()
    held = frame["HOLEID"].isin(rng.choice(holes, len(holes) // 5,
                                           replace=False)).values
    train = frame[~held].reset_index(drop=True)
    data = geoml.data.PointData(train, ["X", "Y", "Z"])
    data.add_categorical_variable("rock", labels=["Waste", "Tom East"],
                                  measurements=train["Code_Simple"].values)
    test = frame[held].reset_index(drop=True)
    points = geoml.data.inducing.from_kmeans(data, 500, seed=0)
    experts = geoml.data.inducing.experts(points, 20, seed=0)
    geoml.set_seed(seed)
    root = gl.BasicInput(
        experts, transform=tr.Anisotropy3D(100, 0.75, 0.5, 345, 15, 70))
    leaf = gl.BasicGP(root, size=2, kernel=kr.Matern32())
    model = geoml.models.VGPNetwork(
        data, "rock", lk.CategoricalGaussianIndicator(2), leaf,
        options=options(variant))
    start = time.perf_counter()
    model.train_full(iters or 300)
    row = dict(train_s=round(time.perf_counter() - start, 1),
               bound=float(model.training_log[-1]))
    target = geoml.data.PointData(test, ["X", "Y", "Z"])
    model.predict(target, n_sim=10)
    p = np.clip(np.asarray(target.variables["rock"].components["Tom East"]
                           .probability.values, dtype=float), 1e-6, 1 - 1e-6)
    truth = (test["Code_Simple"] == "Tom East").values
    seam = seams(model, target)
    row.update(auc=float(roc_auc_score(truth, p)),
               brier=float(np.mean((p - truth) ** 2)),
               log_score=float(-np.mean(np.where(truth, np.log(p),
                                                 np.log(1 - p)))),
               seam_share=float(np.mean(seam)),
               seam_brier=float(np.mean(((p - truth) ** 2)[seam])))

    def make():
        return geoml.data.Grid3D.from_data(data, step=[20, 20, 20])
    _, timing = timed_prediction(model, make, 10)
    row.update(timing)
    return row


def fold(variant, seed, iters):
    truth = np.load(os.path.join(HERE, "fold_regression", "figures",
                                 "vein_truth.npz"))["ore"]
    holes = [(-4.5, 20), (-1.5, -20), (1.0, 0), (3.0, -40), (7.5, -20)]

    def cell(x, y):
        return (np.clip(np.round((y + 20) / 0.1).astype(int), 0, 200),
                np.clip(np.round((x + 8) / 0.1).astype(int), 0, 160))
    coords, rock = [], []
    for x0, angle in holes:
        t = np.arange(0, 40, 0.2)
        x = x0 + t * np.sin(np.radians(angle))
        y = -t * np.cos(np.radians(angle))
        keep = (x >= -8) & (x <= 8) & (y >= -20)
        x, y = x[keep], y[keep]
        coords.append(np.stack([x, y], axis=1))
        rock.append(np.where(truth[cell(x, y)], "ore", "waste"))
    coords, rock = np.concatenate(coords), np.concatenate(rock)
    geoml.set_seed(seed)
    data = geoml.data.PointData.from_array(coords)
    data.add_categorical_variable("Rock", ["waste", "ore"], rock)
    ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])
    experts = geoml.data.inducing.experts(ip, 3, seed=0)
    root = gl.BasicInput(experts, tr.Concatenate(tr.Isotropic(20),
                                                 tr.Isotropic(2)),
                         center=True)
    base_gp = gl.BasicGP(gl.SelectInput(root, [0, 1]), 1, fix_range=True)
    field = gl.BasicGP(gl.SelectInput(root, [2, 3]), size=2, fix_range=True)
    walk = gl.GPWalk(field, step=0.01, n_steps=10)
    reader = gl.BasicGP(walk, kernel=kr.Matern32(), isotropic=True)
    model = geoml.models.VGPNetwork(
        data=data, variables="Rock",
        likelihoods=lk.HierarchicalGaussianIndicator(2),
        latent_network=gl.Concatenate(base_gp, reader),
        options=options(variant, jitter=1e-6))
    start = time.perf_counter()
    model.train_full(iters or 500)
    row = dict(train_s=round(time.perf_counter() - start, 1),
               bound=float(model.training_log[-1]))

    def make():
        return geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
    target, timing = timed_prediction(model, make, 20)
    row.update(timing)
    p = np.clip(np.asarray(target.values("Rock/ore/probability"),
                           dtype=float), 1e-6, 1 - 1e-6)
    ore = truth.ravel().astype(bool)
    seam = seams(model, target)
    row.update(brier=float(np.mean((p - ore) ** 2)),
               log_score=float(-np.mean(np.where(ore, np.log(p),
                                                 np.log(1 - p)))),
               seam_share=float(np.mean(seam)),
               seam_brier=float(np.mean(((p - ore) ** 2)[seam])),
               roughness=roughness(p[:, None], (201, 161), seam)
               / roughness(ore[:, None].astype(float), (201, 161), seam))
    return row


if __name__ == "__main__":
    case, variant = sys.argv[1], sys.argv[2]
    opt = dict(seed="1", iters="", ref="", floor="")
    opt.update(dict(a.split("=", 1) for a in sys.argv[3:]))
    import geoml.latent.network as _net
    if opt["ref"]:
        # the reference kernel's length, in inducing spacings
        _net._REFERENCE_LENGTH = float(opt["ref"])
    if opt["floor"]:
        _net._WEIGHT_FLOOR = float(opt["floor"])
    run = dict(walker=walker, jura=jura, tom=tom, fold=fold)[case]
    row = dict(case=case, variant=variant, seed=int(opt["seed"]),
               geoml=geoml.__version__, ref=opt["ref"] or "4",
               floor=opt["floor"] or "1e-3")
    row.update(run(variant, int(opt["seed"]),
                   int(opt["iters"]) if opt["iters"] else None))
    print("RESULT " + json.dumps(row), flush=True)
