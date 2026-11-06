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

"""What a `GPWalk` costs to train and to predict.

The model is `fold_regression/fold_drillholes.py`'s, the folded vein
drilled by five holes (497 samples), a field of two outputs on 17 x 21
inducing points walked `n_steps` steps and read by an isotropic Matern32
GP, beside a base GP; `stationary` reads the coordinates instead of the
walk. A training step is timed as the difference between two calls of
different lengths, which cancels what a call costs once (its trace, when
it has one), reported apart as `call_overhead_s`. Then the 161 x 201 grid
is predicted twice, the second timed.

Usage: python walk_cost.py stationary|deep [n_steps=10] [steps=30]
                           [thin=1] [rep=1] [ipn=17,21] [n_sim=20]
                           [predict=1]

`thin=k` keeps every k-th sample, `rep=k` repeats the samples k times with
a small jitter, `ipn` sets the inducing grid. Prints one JSON line.

Measured 2026-11-05, 0.10.2, RTX 4060 Ti / Ryzen 9 7950X (WSL):

| | GPU | CPU |
|---|---|---|
| step, stationary / deep | 63 / 555 ms | 35 / 808 ms |
| a `train_full` call's retrace (deep) | 10.3 s | 9 s |
| grid prediction, stationary / deep | 0.7 / 12.8 s | 1.3 / 23 s |

On the GPU the deep step costs about 47 ms a walk step and grows linearly
in the samples (0.5 ms each) and as the cube of the inducing points (264,
565 and 2752 ms at 99, 357 and 775), the inducing points walking
themselves at every refresh. Of the deep prediction, the 20 realizations
are 57% (the reader's pathwise draws 5.3 s, the walk's 2.1 s).

After 0.10.3 (the boundary column gone, `k K^-1 k` shared, the broadcast
updates and `h R^T` on the GPU, the steps kept where they fit):

| | GPU | CPU |
|---|---|---|
| step, deep | 349 ms | 609 ms |
| a `train_full` call's retrace (deep) | 0 | 0 |
| grid prediction, deep | 12.2 s | -- |
| step at 775 inducing points | 1965 ms | -- |
"""
import json
import os
import sys
import time

import numpy as np

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr

kind = sys.argv[1]
opt = dict(n_steps="10", steps="30", thin="1", rep="1", ipn="17,21",
           n_sim="20", predict="1")
opt.update(dict(a.split("=", 1) for a in sys.argv[2:]))
here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "fold_regression")
HOLES = [(-4.5, 20), (-1.5, -20), (1.0, 0), (3.0, -40), (7.5, -20)]
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
coords, rock = coords[::int(opt["thin"])], rock[::int(opt["thin"])]
if int(opt["rep"]) > 1:
    jitter = np.random.default_rng(0)
    coords = np.concatenate(
        [coords + (i > 0) * jitter.normal(0, 0.05, coords.shape)
         for i in range(int(opt["rep"]))])
    rock = np.concatenate([rock] * int(opt["rep"]))

geoml.set_seed(1)
data = geoml.data.PointData.from_array(coords)
data.add_categorical_variable("Rock", ["waste", "ore"], rock)
ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0],
                       n=[int(v) for v in opt["ipn"].split(",")])
grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
root = gl.BasicInput(ip, tr.Concatenate(tr.Isotropic(20), tr.Isotropic(2)),
                     center=True)
base_gp = gl.BasicGP(gl.SelectInput(root, [0, 1]), 1, fix_range=True)
if kind == "deep":
    field = gl.BasicGP(gl.SelectInput(root, [2, 3]), size=2, fix_range=True)
    walk = gl.GPWalk(field, step=0.01, n_steps=int(opt["n_steps"]))
    reader = gl.BasicGP(walk, kernel=kr.Matern32(), isotropic=True)
else:
    reader = gl.BasicGP(gl.SelectInput(root, [2, 3]), kernel=kr.Matern32())
model = geoml.models.VGPNetwork(
    data=data, variables="Rock",
    likelihoods=lk.HierarchicalGaussianIndicator(2),
    latent_network=gl.Concatenate(base_gp, reader),
    options=geoml.models.GPOptions(jitter=1e-6, verbose=False))

import tensorflow as tf  # noqa: E402

row = dict(geoml=geoml.__version__, kind=kind, samples=len(rock),
           device="gpu" if tf.config.list_physical_devices("GPU") else "cpu",
           **opt)
start = time.perf_counter()
model.train_full(3)
row["first_call_s"] = round(time.perf_counter() - start, 2)
steps = int(opt["steps"])
start = time.perf_counter()
model.train_full(2)
short = time.perf_counter() - start
start = time.perf_counter()
model.train_full(2 + steps)
step = (time.perf_counter() - start - short) / steps
row["step_ms"] = round(step * 1e3, 1)
row["call_overhead_s"] = round(short - 2 * step, 2)
row["bound"] = float(model.training_log[-1])
if opt["predict"] == "1":
    model.predict(grid, n_sim=int(opt["n_sim"]))
    start = time.perf_counter()
    model.predict(grid, n_sim=int(opt["n_sim"]))
    row["predict_s"] = round(time.perf_counter() - start, 2)
print(json.dumps(row), flush=True)
