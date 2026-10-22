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

"""Why the folded ore stops folding under the 0.9.4 defaults.

`fold_test.py`'s model -- a walked field over the second half of a
two-scale input, a GP reading the walk, beside a base GP, the hierarchical
indicator -- on the current tree, one factor changed at a time: the
reader's kernel, the propagation, the range prior, `full_variance`, the
walk's starting amplitude. Trains on the script's schedule (100 steps at
5e-2, 100 at 1e-2), predicts the 161 x 201 grid, and prints one JSON line:
the ore bodies (P > 0.5, 8-connected, at least 20 cells), P on the fold's
two limbs between the holes, the uncertain share (0.25 < P < 0.75), the
bound, and what training did to the parameters that decide the fold -- the
walk's amplitude and reach, the reader's ranges, the field's ranges, and
under the joint rule the variance of the reader's inputs at its inducing
points. Saves the map.

Usage: python fold_ablation.py NAME SEED [key=value ...]
  reader=Cubic|Matern32|Gaussian|Exponential  propagation=marginal|joint
  range_prior=2.0|none  full=1|0  amp=1.0  iterations=100+100  isotropic=0|1
  delta=1.0  delta_on=field|all  tolerance=none|0.01  field_range=1
Saves the training log beside the map, and the trained model.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr

name, seed = sys.argv[1], int(sys.argv[2])
opt = dict(reader="Cubic", propagation="marginal", range_prior="2.0",
           full="1", amp="1.0", iterations="100+100", isotropic="0",
           delta="1.0", delta_on="field", tolerance="none", field_range="1")
opt.update(dict(a.split("=", 1) for a in sys.argv[3:]))
out = os.environ.get("FOLD_OUT", ".")
LIMBS = [(-2.5, -11.2), (2.5, -8.5)]

geoml.set_seed(seed)
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
data = geoml.data.PointData.from_array(coords)
data.add_categorical_variable("Rock_deep", ["waste", "ore"], rock)
ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])
grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])

prior = None if opt["range_prior"] == "none" else float(opt["range_prior"])
deep_in = gl.BasicInput(ip, tr.Concatenate(tr.Isotropic(20), tr.Isotropic(2)),
                        center=True)
base_gp = gl.BasicGP(gl.SelectInput(deep_in, [0, 1]), 1, fix_range=True)
field = gl.BasicGP(gl.SelectInput(deep_in, [2, 3]), size=2, fix_range=True)
# the walk's field at a longer range, still fixed: a smoother displacement
ranges = field.parameters["ranges"]
ranges.set_value(np.full(np.shape(ranges.get_value()), float(opt["field_range"])))
walk = gl.GPWalk(field, step=0.01, n_steps=10)
walk.parameters["amp"].set_value(float(opt["amp"]))
reader = gl.BasicGP(walk, kernel=getattr(kr, opt["reader"])(),
                    range_prior=prior, isotropic=opt["isotropic"] == "1")
# the variational variances' start: `delta` sets S = K - K (K + D)^-1 K,
# D = diag(delta), so a small one starts the inducing values confident
nodes = {"field": [field], "all": [base_gp, field, reader]}[opt["delta_on"]]
for node in nodes:
    for key, par in node.parameters.items():
        if key.startswith("delta"):
            par.set_value(np.full(np.shape(par.get_value()), float(opt["delta"])))
deep_out = gl.Concatenate(base_gp, reader)

options = dict(jitter=1e-6, verbose=False, propagation=opt["propagation"],
               full_variance=opt["full"] == "1")
if opt["tolerance"] != "none":
    options["training_tolerance"] = float(opt["tolerance"])
model = geoml.models.VGPNetwork(
    data=data, variables="Rock_deep",
    likelihoods=lk.HierarchicalGaussianIndicator(2), latent_network=deep_out,
    options=geoml.models.GPOptions(**options))

start = time.perf_counter()
os.makedirs(out, exist_ok=True)
# with `checkpoint=N`, each phase trains N steps at a time and the model and
# the log so far are saved after each: a phase split into chunks replays one
# call to the bit, and a run killed late keeps what it had
every = int(opt.get("checkpoint", "0"))
for rate, steps in zip((5e-2, 1e-2), opt["iterations"].split("+")):
    model.set_learning_rate(rate)
    left = int(steps)
    while left > 0:
        now = min(left, every) if every else left
        model.train_full(now)
        left -= now
        if every:
            model.save(os.path.join(out, "%s_s%d_model" % (name, seed)))
            np.save(os.path.join(out, "%s_s%d_log.npy" % (name, seed)),
                    np.asarray(model.training_log, dtype=float))
            print("checkpoint: step %d, bound %.2f, %.0f s"
                  % (len(model.training_log), model.training_log[-1],
                     time.perf_counter() - start), flush=True)
seconds = time.perf_counter() - start
log = np.asarray(model.training_log, dtype=float)
model.save(os.path.join(out, "%s_s%d_model" % (name, seed)))
model.predict(grid)

p = np.asarray(grid.values("Rock_deep/ore/probability"),
               dtype=float).reshape(201, 161)
labels, count = ndimage.label(p > 0.5, structure=np.ones((3, 3)))
sizes = ndimage.sum(np.ones_like(p), labels, range(1, count + 1))


def at(x, y):
    return float(p[int(round((y + 20) / 0.1)), int(round((x + 8) / 0.1))])


amp = float(np.asarray(walk.parameters["amp"].get_value()))
row = dict(name=name, seed=seed, settings=opt, version=geoml.__version__,
           bodies=int(np.sum(sizes >= 20)),
           limbs=[round(at(x, y), 3) for x, y in LIMBS],
           uncertain=round(float(np.mean((p > 0.25) & (p < 0.75))), 3),
           ore_share=round(float(np.mean(p > 0.5)), 3),
           bound=round(float(model.training_log[-1]), 2),
           amp=round(amp, 3), reach=round(0.01 * 10 * amp, 3),
           reader_ranges=np.round(np.asarray(
               reader.parameters["ranges"].get_value()).ravel(), 3).tolist(),
           delta_after={label: round(float(np.median(np.asarray(
               node.parameters["delta_0"].get_value()))), 4)
               for label, node in (("base", base_gp), ("field", field),
                                   ("reader", reader))},
           seconds=round(seconds, 1), steps=len(log),
           gain_last_100=round(float(log[-1] - log[max(0, len(log) - 101)]), 2),
           field_range=float(np.asarray(ranges.get_value()).ravel()[0]))
# the variance of the reader's inputs at its own inducing points: how
# uncertain the walked coordinates are where the reader is anchored
model._refresh(model.options.jitter)
variance = getattr(walk, "inducing_points_variance", None)
if variance is not None:
    v = np.asarray(variance[0], dtype=float)
    row["walked_ip_sd"] = [round(float(q), 4) for q in
                           np.sqrt(np.percentile(v, [50, 90], axis=None))]
print(json.dumps(row), flush=True)
os.makedirs(out, exist_ok=True)
np.savez_compressed(os.path.join(out, "%s_s%d.npz" % (name, seed)), p=p,
                    log=log)
