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

"""The folded vein drilled: a stationary model against the deep one.

The truth is a folded ore layer, `figures/vein_truth.npz` -- the ore region
(P > 0.5) of the fold ablation's `m32_joint_dall001` map, on the 161 x 201
grid over [-8, 8] x [-20, 0]. Five drillholes are collared at the top of
the section, four of them inclined, and sampled every 0.2 m along the hole
from the truth's nearest cell; every hole cuts the vein at least twice.
Two models, the same but for the walk: a base GP at range 20 beside a
Matern32 GP at range 2 that reads the coordinates (stationary, its ranges
per axis) or a walk over them (deep: a field of two outputs walked ten
steps of 0.01, read by an isotropic GP, as `GPWalk`'s docstring advises),
under the hierarchical indicator. Each trains `iterations` steps with the
model's own optimizer, predicts the grid and is scored against the truth:
cell accuracy, the ore's intersection over union, the Brier and log
scores of P(ore), and the ore bodies (P > 0.5, 8-connected, at least 20
cells; the truth is one). Prints one JSON line; saves the map, the training
log and the trained model.

Usage: python fold_drillholes.py NAME SEED stationary|deep [iterations=500]
                                 [holes=X/ANGLE_X/ANGLE_...]
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy import ndimage

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr

name, seed, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
opt = dict(iterations="500")
opt.update(dict(a.split("=", 1) for a in sys.argv[4:]))
out = os.environ.get("FOLD_OUT", ".")
here = os.path.dirname(os.path.abspath(__file__))
# collar east and the hole's angle from vertical, in degrees, + to the east;
# `holes=-4.5/5_-0.5/0_5.5/-8` drills another design
HOLES = [(-4.5, 20), (-1.5, -20), (1.0, 0), (3.0, -40), (7.5, -20)]
if "holes" in opt:
    HOLES = [tuple(float(v) for v in hole.split("/"))
             for hole in opt["holes"].split("_")]

truth = np.load(os.path.join(here, "figures", "vein_truth.npz"))["ore"]


def cell(x, y):
    return (np.clip(np.round((y + 20) / 0.1).astype(int), 0, 200),
            np.clip(np.round((x + 8) / 0.1).astype(int), 0, 160))


coords, rock = [], []
for x0, angle in HOLES:
    t = np.arange(0, 40, 0.2)
    x = x0 + t * np.sin(np.radians(angle))
    y = -t * np.cos(np.radians(angle))
    inside = (x >= -8) & (x <= 8) & (y >= -20)
    x, y = x[inside], y[inside]
    coords.append(np.stack([x, y], axis=1))
    rock.append(np.where(truth[cell(x, y)], "ore", "waste"))
coords, rock = np.concatenate(coords), np.concatenate(rock)

geoml.set_seed(seed)
data = geoml.data.PointData.from_array(coords)
data.add_categorical_variable("Rock", ["waste", "ore"], rock)
ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])
grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])

root = gl.BasicInput(ip, tr.Concatenate(tr.Isotropic(20), tr.Isotropic(2)),
                     center=True)
base_gp = gl.BasicGP(gl.SelectInput(root, [0, 1]), 1, fix_range=True)
if kind == "deep":
    field = gl.BasicGP(gl.SelectInput(root, [2, 3]), size=2, fix_range=True)
    walk = gl.GPWalk(field, step=0.01, n_steps=10)
    reader = gl.BasicGP(walk, kernel=kr.Matern32(), isotropic=True)
else:
    walk = None
    reader = gl.BasicGP(gl.SelectInput(root, [2, 3]), kernel=kr.Matern32())
model = geoml.models.VGPNetwork(
    data=data, variables="Rock",
    likelihoods=lk.HierarchicalGaussianIndicator(2),
    latent_network=gl.Concatenate(base_gp, reader),
    options=geoml.models.GPOptions(jitter=1e-6, verbose=False))

os.makedirs(out, exist_ok=True)
stem = os.path.join(out, "%s_s%d" % (name, seed))
start = time.perf_counter()
try:
    # one optimizer, as the model builds it, saved every 100 steps so a
    # run killed late keeps what it had
    left = int(opt["iterations"])
    while left > 0:
        model.train_full(min(left, 100))
        left -= 100
        model.save(stem + "_model")
        print("checkpoint: step %d, bound %.2f, %.0f s"
              % (len(model.training_log), model.training_log[-1],
                 time.perf_counter() - start), flush=True)
    seconds = time.perf_counter() - start
    model.predict(grid)
except Exception:
    traceback.print_exc()
    raise

p = np.asarray(grid.values("Rock/ore/probability"),
               dtype=float).reshape(201, 161)
labels, count = ndimage.label(p > 0.5, structure=np.ones((3, 3)))
sizes = ndimage.sum(np.ones_like(p), labels, range(1, count + 1))
q = np.clip(np.where(truth, p, 1 - p), 1e-6, 1)
fitted = np.asarray(model.training_log, dtype=float)
row = dict(name=name, kind=kind, seed=seed, settings=opt,
           version=geoml.__version__, samples=len(rock),
           accuracy=round(float(np.mean((p > 0.5) == truth)), 4),
           iou=round(float(np.sum((p > 0.5) & truth)
                           / np.sum((p > 0.5) | truth)), 4),
           brier=round(float(np.mean((p - truth) ** 2)), 4),
           log_score=round(float(np.mean(-np.log(q))), 4),
           bodies=int(np.sum(sizes >= 20)),
           data_accuracy=round(float(np.mean(
               (p[cell(coords[:, 0], coords[:, 1])] > 0.5) == (rock == "ore"))), 4),
           bound=round(float(fitted[-1]), 2), seconds=round(seconds, 1))
if walk is not None:
    row["amp"] = round(float(np.asarray(walk.parameters["amp"].get_value())), 3)
print(json.dumps(row), flush=True)
np.savez_compressed(stem + ".npz", p=p, log=fitted, coords=coords,
                    ore=rock == "ore")
