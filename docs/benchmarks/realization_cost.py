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

"""What the realizations of a grid prediction cost, on the folded vein's
walk model (`walk_cost.py`'s): the whole prediction at one realization and
at `n_sim`, then one batch of the grid through the first leaf, traced, with
the reader's pathwise draw and the walk's realizations knocked out in
turn. Switching off the reader's draw also prunes the walk's realizations,
which only the reader reads.

Usage: python realization_cost.py [n_sim=20] [train=30] [prec=float32|float64]

Prints one JSON line. Measured 2026-11-06, RTX 4060 Ti / Ryzen 9 7950X
(WSL), 20 realizations on the 161 x 201 grid:

| | 0.10.3 | 0.10.4 |
|---|---|---|
| GPU, prediction | 12.3 s | 5.9 s |
| GPU, one realization | 4.8 s | 4.6 s |
| CPU float64, prediction | 22.6 s | 21.7 s |
| CPU float32, prediction | -- | 17.8 s |

At 0.10.3 a GPU batch of 2158 rows cost 0.28 s plus 26 ms a realization:
the reader's draw 0.39 s of 20 realizations, the walk's 0.14 s. Of the
draw, timed apart on its own tensors (four chunks of five): the conditioned
kernel 45%, the inducing draw `k R G^-1 Phi(Z)` 20%, the residual (a
triangular solve of every row's `k`, then the prior) 27%, the features 3%,
the blend 5%. The batch's size does not matter (12.3 to 12.7 s from 1000 to
8000 rows). 0.10.4 solves the triangle once per call and joins the two
products, takes the features' products in single precision where the
quadrature is single (the default on a GPU), and compiles the chunks with
XLA on a GPU where no tape records them: a GPU batch then costs 0.29 s
plus 4.4 ms a realization, what is left being the moments; a CPU batch in
float64, 1.16 s plus 72 ms a realization, the reader's draw 1.0 s of 20 and
the walk's 0.45 s.
"""
import json
import os
import sys
import time

import numpy as np

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.latent.network as net
import geoml.likelihood as lk
import geoml.transform as tr

opt = dict(n_sim="20", train="30", prec="")
opt.update(dict(a.split("=", 1) for a in sys.argv[1:]))
here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "fold_regression")
HOLES = [(-4.5, 20), (-1.5, -20), (1.0, 0), (3.0, -40), (7.5, -20)]
truth = np.load(os.path.join(here, "figures", "vein_truth.npz"))["ore"]

coords, rock = [], []
for x0, angle in HOLES:
    t = np.arange(0, 40, 0.2)
    x = x0 + t * np.sin(np.radians(angle))
    y = -t * np.cos(np.radians(angle))
    inside = (x >= -8) & (x <= 8) & (y >= -20)
    x, y = x[inside], y[inside]
    coords.append(np.stack([x, y], axis=1))
    rows = np.clip(np.round((y + 20) / 0.1).astype(int), 0, 200)
    cols = np.clip(np.round((x + 8) / 0.1).astype(int), 0, 160)
    rock.append(np.where(truth[rows, cols], "ore", "waste"))
coords, rock = np.concatenate(coords), np.concatenate(rock)

geoml.set_seed(1)
data = geoml.data.PointData.from_array(coords)
data.add_categorical_variable("Rock", ["waste", "ore"], rock)
ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])
grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
root = gl.BasicInput(ip, tr.Concatenate(tr.Isotropic(20), tr.Isotropic(2)),
                     center=True)
base_gp = gl.BasicGP(gl.SelectInput(root, [0, 1]), 1, fix_range=True)
field = gl.BasicGP(gl.SelectInput(root, [2, 3]), size=2, fix_range=True)
walk = gl.GPWalk(field, step=0.01, n_steps=10)
reader = gl.BasicGP(walk, kernel=kr.Matern32(), isotropic=True)
options = {"quadrature_precision": opt["prec"]} if opt["prec"] else {}
model = geoml.models.VGPNetwork(
    data=data, variables="Rock",
    likelihoods=lk.HierarchicalGaussianIndicator(2),
    latent_network=gl.Concatenate(base_gp, reader),
    options=geoml.models.GPOptions(jitter=1e-6, verbose=False, **options))
model.train_full(int(opt["train"]))

import tensorflow as tf  # noqa: E402

n_sim = int(opt["n_sim"])
row = dict(geoml=geoml.__version__, rows=int(model._prediction_rows()),
           device="gpu" if tf.config.list_physical_devices("GPU") else "cpu",
           **opt)


def sync(x):
    tf.nest.map_structure(lambda t: t.numpy(), x)


def timed(fn, reps=3):
    sync(fn())
    start = time.perf_counter()
    for _ in range(reps):
        sync(fn())
    return round((time.perf_counter() - start) / reps, 3)


for s in (1, n_sim):
    model.predict(grid, n_sim=s)
    start = time.perf_counter()
    model.predict(grid, n_sim=s)
    row["predict_%d_s" % s] = round(time.perf_counter() - start, 2)

rows = row["rows"]
x = tf.constant(grid.get_batched_coordinates(
    np.arange(min(rows, grid.n_data)))[0], tf.float64)
leaf = model.leaves[0]
with model._propagation(rows=rows):
    gl.refresh_cached(model.leaves, model.options.jitter, owner=model)


def batch(s):
    @tf.function
    def f(x):
        with net.once_per_pass():
            return leaf.predict(x, n_sim=s, seed=[3, 0])[:3]
    with model._propagation(rows=rows):
        return timed(lambda: f(x))


row["batch_1_s"] = batch(1)
row["batch_s"] = batch(n_sim)
walked = net.GPWalk._walk_realizations
drawn = net.BasicGP._pathwise_simulate


@tf.autograph.experimental.do_not_convert
def still(self, n, starts, *args):
    return starts


@tf.autograph.experimental.do_not_convert
def no_draw(self, n, seed):
    if self is not reader:
        return drawn(self, n, seed)
    return tf.zeros([self.size, tf.shape(self._path_state[5])[0], n],
                    tf.float64)


net.GPWalk._walk_realizations = still
row["batch_no_walk_draws_s"] = batch(n_sim)
net.GPWalk._walk_realizations = walked
net.BasicGP._pathwise_simulate = no_draw
row["batch_no_reader_draws_s"] = batch(n_sim)
net.BasicGP._pathwise_simulate = drawn
print(json.dumps(row), flush=True)
