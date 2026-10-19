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

"""Step 3 of the deep second moment: training, on the folded section.

Arms, each on the GP on a GP and beside the coordinates, three seeds,
the same number of steps:
  moment   the 0.9.3 moment bound (the conditioned second moment off)
  second   the bound with the conditioned second moment, from scratch
  warmup   200 steps of the 64-draw doubly stochastic bound
           (`doubly_stochastic/ds_prototype.train_ds`), then `second`
Each model is scored on the 200 new holes by its moments -- with the
second moment, as the package now reports them -- (log score per hole,
90% coverage, CRPS; `run_folded.summary`), and its 64 realizations
against the true field (`run_folded.ensemble_scores`). Appends JSON lines
to `phase3_folded.jsonl`.

Usage: python phase3_folded.py ITERATIONS NETWORK ARM SEED
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import gate_fields as gf  # noqa: E402
import ds_prototype as ds  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def main(iterations, network, arm, seed):
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    new, holes = gm.ek.drillholes(gm.ek.N_NEW, gm.ek.NEW_SEED)
    keep, gm.ek.SEED = gm.ek.SEED, seed
    try:
        model = gm.build(data, network, kr.Gaussian)
    finally:
        gm.ek.SEED = keep
    start = time.perf_counter()
    if arm == "moment":
        net._UNCERTAIN_POINTS_SECOND_MOMENT = False
        model.train_full(iterations)
        net._UNCERTAIN_POINTS_SECOND_MOMENT = True
    elif arm == "second":
        model.train_full(iterations)
    elif arm == "warmup":
        ds.train_ds(model, 200, 64)
        model.train_full(iterations - 200)
    seconds = time.perf_counter() - start
    points, truth = gm.rf.grid()
    row = dict(network=network, arm=arm, seed=seed, iterations=iterations,
               seconds=round(seconds, 1),
               bound=float(model.training_log[-1]),
               moments=gm.rf.summary(*gm.rf.moment_scores(model, new), holes),
               field=gm.rf.ensemble_scores(
                   gf.realizations(model, points, True), truth))
    print(json.dumps(row), flush=True)
    with open(os.path.join(gm.FIGURES, "phase3_folded.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
