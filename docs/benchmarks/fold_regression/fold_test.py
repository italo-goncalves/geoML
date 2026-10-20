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

"""Does the folded ore fold? The deep model of non_stationary_test.py
(2022 VGP article), on whatever geoML is first on the path.

Builds the script's deep model -- a walked field over the second half of a
two-scale input, a Cubic GP reading the walk, beside a base GP --, trains
it on the script's schedule (100 steps at 5e-2, 100 at 1e-2), predicts the
ore probability on a 161 x 201 grid, and measures the fold: the ore region
(P > 0.5) split into 8-connected bodies, and whether one body joins all
five ore intervals' centres. Prints a JSON line and saves the map.

Usage: python fold_test.py NAME SEED RANGE [OUT_DIR] [PROPAGATION]
"""
import inspect
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy import ndimage

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr

name, seed, deep_range = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else "."
propagation = sys.argv[5] if len(sys.argv) > 5 else "marginal"
CENTRES = [(-5, -13), (5, -13), (0, -4), (0, -10), (0, -16)]

if hasattr(geoml, "set_seed"):
    geoml.set_seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

z1 = np.linspace(0, -20, 101)
rock_1 = np.where((z1 < -11.5) & (z1 > -14.5), "ore", "waste")
rock_2 = np.where((z1 < -2.5) & (z1 > -5.5), "ore", "waste")
rock_2 = np.where((z1 < -8.5) & (z1 > -11.5), "ore", rock_2)
rock_2 = np.where((z1 < -14.5) & (z1 > -17.5), "ore", rock_2)
coords = np.stack([
    np.concatenate([-5 * np.ones_like(z1), np.zeros_like(z1),
                    5 * np.ones_like(z1)]),
    np.concatenate([z1, z1, z1])], axis=1)
rock = np.concatenate([rock_1, rock_2, rock_1[::-1]])
point_data = geoml.data.PointData.from_array(coords)
point_data.add_categorical_variable("Rock_deep", ["waste", "ore"], rock)
ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])
grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])

deep_in = gl.BasicInput(ip, tr.Concatenate(tr.Isotropic(20),
                                           tr.Isotropic(deep_range)),
                        center=True)
base_gp = gl.BasicGP(gl.SelectInput(deep_in, [0, 1]), 1, fix_range=True)
deep_gp_1 = gl.BasicGP(gl.SelectInput(deep_in, [2, 3]), size=2,
                       fix_range=True)
walk = gl.GPWalk(deep_gp_1, step=0.01, n_steps=10)
deep_gp_2 = gl.BasicGP(walk, kernel=kr.Cubic())
deep_out = gl.Concatenate(base_gp, deep_gp_2)

options = dict(jitter=1e-6)
accepted = inspect.signature(geoml.models.GPOptions).parameters
if "propagation" in accepted:
    options["propagation"] = propagation
if "verbose" in accepted:
    options["verbose"] = False
model = geoml.models.VGPNetwork(
    data=point_data, variables="Rock_deep",
    likelihoods=lk.HierarchicalGaussianIndicator(2),
    latent_network=deep_out, options=geoml.models.GPOptions(**options))
zero_bias = bool(os.environ.get("ZERO_BIAS"))
if zero_bias:
    # every GP node's prior mean back at zero, as before v0.3.5
    for node in [base_gp, deep_gp_1, deep_gp_2]:
        for key, par in node.parameters.items():
            if key.startswith("bias"):
                par.set_value(0.0)
                par.fix()
start = time.perf_counter()
model.set_learning_rate(5e-2)
model.train_full(100)
model.set_learning_rate(1e-2)
model.train_full(100)
seconds = time.perf_counter() - start
model.predict(grid)

try:
    p = np.asarray(grid.values("Rock_deep/ore/probability"), dtype=float)
except Exception:
    p = np.asarray(grid.variables["Rock_deep"].components["ore"]
                   .probability.values, dtype=float)
p = p.reshape(201, 161)
labels, count = ndimage.label(p > 0.5, structure=np.ones((3, 3)))
sizes = ndimage.sum(np.ones_like(p), labels, range(1, count + 1))
bodies = int(np.sum(sizes >= 20))
cells = [int(labels[int(round((y + 20) / 0.1)), int(round((x + 8) / 0.1))])
         for x, y in CENTRES]
joined = len(set(cells)) == 1 and cells[0] != 0
# accuracy at the data, by the nearest grid cell
rows = np.round((coords[:, 1] + 20) / 0.1).astype(int)
cols = np.round((coords[:, 0] + 8) / 0.1).astype(int)
accuracy = float(np.mean((p[rows, cols] > 0.5) == (rock == "ore")))
log = list(getattr(model, "training_log", []))
row = dict(name=name + ("_zerobias" if zero_bias else ""), seed=seed, range=deep_range, propagation=propagation,
           version=getattr(geoml, "__version__", "?"),
           folded=bool(joined), bodies=bodies, centres_in=cells,
           ore_share=float(np.mean(p > 0.5)), data_accuracy=accuracy,
           bound=float(log[-1]) if log else None, seconds=round(seconds, 1))
print(json.dumps(row), flush=True)
os.makedirs(out, exist_ok=True)
np.savez_compressed(os.path.join(out, "%s_r%g_s%d.npz" % (row["name"], deep_range,
                                                         seed)), p=p)
