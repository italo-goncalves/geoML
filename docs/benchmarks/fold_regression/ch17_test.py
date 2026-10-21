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

"""Chapter 17's deep model as the 0.8.0 chapter wrote it, on whatever geoML
is first on the path: the vein database, the merged domains, the dense and
sparse conversions, a GPWalk over an Anisotropy3D input read by a one-field
implicit model, 120 full steps at 2e-2. Scores the vein surface (the
contour of the Vein indicator at zero on a 61 x 71 x 61 grid): its number
of separate pieces and its area; and the in-sample agreement off the
contacts. The stationary model alongside, the same way. Prints JSON lines
and saves each model's indicator section through the middle of the grid.

Usage: python ch17_test.py NAME SEED DATA_DIR OUT_DIR
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import geoml

name, seed, data_dir, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], \
    sys.argv[4]
geoml.set_seed(seed)
# `PROPAGATION=marginal` reads a deep input by its marginals, as before 0.9.0
PROPAGATION = {"propagation": os.environ["PROPAGATION"]}     if os.environ.get("PROPAGATION") else {}
if os.environ.get("PROPAGATION"):
    name = name + "_" + os.environ["PROPAGATION"]

collar = pd.read_csv(os.path.join(data_dir, "vein-collar.csv"))
lito = pd.read_csv(os.path.join(data_dir, "vein-lito.csv"),
                   usecols=["HOLE", "FROM", "TO", "SIMPLE LITO"])
holes = geoml.data.DrillholeData(collar, hole="HOLE", x="EAST", y="NORTH",
                                 z="RL", length="TDEPTH")
holes.add_intervals("lito", lito, hole="HOLE", fr="FROM", to="TO",
                    categorical=["SIMPLE LITO"])
merged = holes.merge_domains(("lito", "SIMPLE LITO"))
holes.add_intervals("lito", merged)
dense = holes.as_classification_input(("lito", "SIMPLE LITO"), length=2.0)
sparse = holes.as_classification_input(("lito", "SIMPLE LITO"), length=10.0)


def implicit_model(root_node, data, iterations=120):
    field = geoml.latent.BasicGP(root_node, size=1)
    indicators = geoml.latent.Linear(field, size=2)
    model = geoml.models.VGPNetwork(
        data, "SIMPLE LITO",
        geoml.likelihood.CategoricalGaussianIndicator(2), indicators,
        options=geoml.models.GPOptions(jitter=1e-6, verbose=False,
                                       **PROPAGATION))
    model.set_learning_rate(2e-2)
    model.train_full(max_iter=iterations)
    return model


def anisotropy():
    return geoml.transform.Anisotropy3D(100, 1.0, 0.5, azimuth=270, dip=45)


grid = geoml.data.Grid3D(start=[24850, 15700, 1300],
                         end=[25150, 16050, 1600], n=[61, 71, 61])
interior = ~dense.values("SIMPLE LITO/boundary").astype(bool)
truth = dense.values("SIMPLE LITO/measurements_a")
os.makedirs(out, exist_ok=True)
for kind in ("stationary", "deep"):
    start = time.perf_counter()
    root = geoml.latent.BasicInput(inducing_points=sparse,
                                   transform=anisotropy(), center=True)
    if kind == "deep":
        root = geoml.latent.GPWalk(geoml.latent.BasicGP(root, size=3),
                                   n_steps=5)
    model = implicit_model(root, dense)
    seconds = time.perf_counter() - start
    model.predict(grid)
    indicator = np.asarray(grid.values(
        "SIMPLE LITO/Vein/indicator_predicted"), dtype=float)
    surface = grid.get("SIMPLE LITO/Vein/indicator_predicted") \
        .get_contour(0.0)
    try:
        pieces = len(surface.split())
    except Exception:
        pieces = -1
    model.predict(dense)
    predicted = dense.values("SIMPLE LITO/predicted")
    row = dict(name=name, seed=seed, model=kind,
               version=getattr(geoml, "__version__", "?"),
               pieces=pieces, area=float(surface.area),
               triangles=int(len(surface.triangles)),
               agreement=float(np.mean(predicted[interior]
                                       == truth[interior])),
               vein_share=float(np.mean(indicator > 0)),
               bound=float(model.training_log[-1]),
               seconds=round(seconds, 1))
    print(json.dumps(row), flush=True)
    cube = indicator.reshape(61, 71, 61)        # z, y, x
    np.savez_compressed(os.path.join(out, "ch17_%s_%s_s%d.npz"
                                     % (name, kind, seed)),
                        section=cube[:, 35, :], plan=cube[30, :, :],
                        vertices=np.asarray(surface.coordinates),
                        triangles=np.asarray(surface.triangles))
