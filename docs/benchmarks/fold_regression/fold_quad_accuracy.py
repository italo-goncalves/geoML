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

"""The conditioned quadrature's rule, judged on one trained model.

`predict RULE MODEL_DIR OUT.npz` loads a model `fold_ablation.py` saved and
predicts the 161 x 201 grid under one rule of the conditioned second
moment's quadrature -- `sobol1024` (the reference), `hermite7` (the
default), `hermite5`, `hermite7xla`, `hermite5xla` -- saving the ore's
probability and its indicator's latent mean and variance, and the time the
prediction took (after one traced). One rule a process: the Hermite count
is read at trace time and is not in the trace's key.

`compare REF.npz RUN.npz...` prints, against the reference, each run's
largest and median difference in probability and latent mean, and its
latent variance's largest and median relative difference.

Usage: python fold_quad_accuracy.py predict RULE MODEL_DIR OUT.npz
       python fold_quad_accuracy.py compare REF.npz RUN.npz...
"""
import os
import sys
import time

import numpy as np

COLUMNS = {"p": "Rock_deep/ore/probability",
           "mean": "Rock_deep/ore/indicator_mean",
           "var": "Rock_deep/ore/indicator_variance"}


def predict(rule, model_dir, out):
    import geoml
    import geoml.latent.network as net
    if rule == "sobol1024":
        net._QUADRATURE_RULE, net._QUADRATURE_NODES = "sobol", 1024
    else:
        net._HERMITE_POINTS = int(rule[len("hermite"):][0])
        net._QUADRATURE_XLA = rule.endswith("xla")
    model = geoml.persistence.load_model(model_dir)
    grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
    model.predict(grid)
    start = time.perf_counter()
    model.predict(grid)
    seconds = time.perf_counter() - start
    np.savez_compressed(out, seconds=seconds, **{
        key: np.asarray(grid.values(path), dtype=float).ravel()
        for key, path in COLUMNS.items()})
    print("%s: %.1f s" % (rule, seconds), flush=True)


def compare(reference, runs):
    ref = np.load(reference)
    for run in runs:
        got = np.load(run)
        dp = np.abs(got["p"] - ref["p"])
        dm = np.abs(got["mean"] - ref["mean"])
        rv = np.abs(got["var"] / ref["var"] - 1.0)
        print("%-24s %5.1f s  P: max %.4f median %.5f | mean: max %.4f "
              "median %.5f | variance: max %.3f median %.4f"
              % (os.path.basename(run), float(got["seconds"]), dp.max(),
                 np.median(dp), dm.max(), np.median(dm), rv.max(),
                 np.median(rv)))


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(*sys.argv[2:5])
    else:
        compare(sys.argv[2], sys.argv[3:])
