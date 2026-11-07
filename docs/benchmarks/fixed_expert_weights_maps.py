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

"""Maps of Walker Lake's V and Jura's Co under the variance rule and under
fixed weights, the models of `fixed_expert_weights.py` (seed 1): the
prediction under each rule, their difference, each rule's largest weight at
a location -- where it is under one, experts are blended -- and, for
Walker, the exhaustive truth and the two rules' errors.

Usage: python fixed_expert_weights_maps.py [walker|jura|both]
Writes figures/fixed_weights_walker.png and figures/fixed_weights_jura.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent as gl  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.transform as tr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = {"variance rule": {}, "fixed weights": dict(expert_weights="coverage")}


def walker_model(options):
    geoml.set_seed(1)
    data, grid = geoml.datasets.walker()
    experts = geoml.data.inducing.grid_experts(grid, 10.0, block=8)
    root = gl.BasicInput(experts, transform=tr.Isotropic(50))
    gp = gl.BasicGP(root, size=1, kernel=kr.Spherical())
    likelihood = lk.Gaussian(geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1, shift=None), geoml.warping.ZScore(1)))
    model = geoml.models.VGPNetwork(
        data, "V", likelihood, gp,
        options=geoml.models.GPOptions(verbose=False, **options))
    model.train_full(500)
    return model, data, grid


def jura_model(options):
    geoml.set_seed(1)
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
    model = geoml.models.VGPNetwork(
        train, "Co", likelihood, gp,
        options=geoml.models.GPOptions(verbose=False, **options))
    model.train_full(500)
    return model, train, test


def largest_weight(model, grid):
    return np.max(model.expert_weights(grid), axis=1)


def walker():
    out = {}
    for rule, options in RULES.items():
        model, data, truth_grid = walker_model(options)
        grid = geoml.data.Grid2D(start=[1, 1], n=[260, 300], step=[1, 1])
        model.predict(grid, n_sim=20)
        out[rule] = dict(mean=np.asarray(grid.values("V/prediction"), float),
                         top=largest_weight(model, grid))
    truth = np.asarray(truth_grid.values("V/measurements"), float)
    xy = np.asarray(data.coordinates)
    shape, extent = (300, 260), [0.5, 260.5, 0.5, 300.5]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
    vmax = np.quantile(truth, 0.98)

    def show(ax, field, title, cmap="viridis", vmin=0, top=None, points=False):
        im = ax.imshow(field.reshape(shape), origin="lower", extent=extent,
                       cmap=cmap, vmin=vmin, vmax=top)
        if points:
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c="w", lw=0)
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    v, f = out["variance rule"], out["fixed weights"]
    show(axes[0, 0], truth, "exhaustive V (truth), samples in white",
         top=vmax, points=True)
    show(axes[0, 1], v["mean"], "variance rule: prediction", top=vmax)
    show(axes[0, 2], f["mean"], "fixed weights: prediction", top=vmax)
    d = f["mean"] - v["mean"]
    lim = np.quantile(np.abs(d), 0.99)
    show(axes[0, 3], d, "fixed minus variance", cmap="RdBu_r", vmin=-lim,
         top=lim)
    err = [np.abs(r["mean"] - truth) for r in (v, f)]
    lim = np.quantile(np.concatenate(err), 0.98)
    rmse = [np.sqrt(np.mean((r["mean"] - truth) ** 2)) for r in (v, f)]
    show(axes[1, 0], err[0], "variance rule: |error| (rmse %.0f)" % rmse[0],
         cmap="magma", top=lim)
    show(axes[1, 1], err[1], "fixed weights: |error| (rmse %.0f)" % rmse[1],
         cmap="magma", top=lim)
    show(axes[1, 2], v["top"], "variance rule: largest expert weight",
         cmap="cividis", top=1)
    show(axes[1, 3], f["top"], "fixed weights: largest expert weight",
         cmap="cividis", top=1)
    fig.suptitle("Walker Lake V, 16 experts (grid_experts, step 10, blocks "
                 "of 8), spherical range 50", fontsize=14)
    path = os.path.join(HERE, "figures", "fixed_weights_walker.png")
    fig.savefig(path, dpi=110)
    print("wrote", path, "rmse", rmse, flush=True)


def jura():
    out = {}
    for rule, options in RULES.items():
        model, train, test = jura_model(options)
        grid = geoml.data.Grid2D(start=[0.25, 0.45], end=[5.25, 5.85],
                                 n=[201, 217])
        model.predict(grid, n_sim=20)
        model.predict(test, n_sim=20)
        truth = np.asarray(test.values("Co/measurements"), float)
        pred = np.asarray(test.values("Co/prediction"), float)
        out[rule] = dict(mean=np.asarray(grid.values("Co/prediction"), float),
                         top=largest_weight(model, grid),
                         rmse=float(np.sqrt(np.mean((pred - truth) ** 2))))
    xy = np.asarray(train.coordinates)
    co = np.asarray(train.values("Co/measurements"), float)
    vxy = np.asarray(test.coordinates)
    shape, extent = (217, 201), [0.25, 5.25, 0.45, 5.85]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    vmin, vmax = np.quantile(co, [0.02, 0.98])

    def show(ax, field, title, cmap="viridis", lo=None, hi=None,
             points=True):
        im = ax.imshow(field.reshape(shape), origin="lower", extent=extent,
                       cmap=cmap, vmin=lo, vmax=hi)
        if points:
            ax.scatter(xy[:, 0], xy[:, 1], s=4, c="w", lw=0)
            ax.scatter(vxy[:, 0], vxy[:, 1], s=6, c="r", marker="x", lw=0.8)
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    v, f = out["variance rule"], out["fixed weights"]
    sc = axes[0, 0].scatter(xy[:, 0], xy[:, 1], c=co, s=18, vmin=vmin,
                            vmax=vmax, cmap="viridis")
    axes[0, 0].scatter(vxy[:, 0], vxy[:, 1], s=10, c="r", marker="x")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlim(extent[:2]), axes[0, 0].set_ylim(extent[2:])
    axes[0, 0].set_title("Co samples (validation in red)")
    axes[0, 0].set_xticks([]), axes[0, 0].set_yticks([])
    fig.colorbar(sc, ax=axes[0, 0], shrink=0.8)
    show(axes[0, 1], v["mean"], "variance rule: prediction (validation "
         "rmse %.2f)" % v["rmse"], lo=vmin, hi=vmax)
    show(axes[0, 2], f["mean"], "fixed weights: prediction (validation "
         "rmse %.2f)" % f["rmse"], lo=vmin, hi=vmax)
    d = f["mean"] - v["mean"]
    lim = np.quantile(np.abs(d), 0.99)
    show(axes[1, 0], d, "fixed minus variance", cmap="RdBu_r", lo=-lim,
         hi=lim)
    show(axes[1, 1], v["top"], "variance rule: largest expert weight",
         cmap="cividis", lo=0, hi=1, points=False)
    show(axes[1, 2], f["top"], "fixed weights: largest expert weight",
         cmap="cividis", lo=0, hi=1, points=False)
    fig.suptitle("Jura Co, 4 experts of 120 k-means points, Matern32",
                 fontsize=14)
    path = os.path.join(HERE, "figures", "fixed_weights_jura.png")
    fig.savefig(path, dpi=110)
    print("wrote", path, "rmse", v["rmse"], f["rmse"], flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("walker", "both"):
        walker()
    if which in ("jura", "both"):
        jura()
