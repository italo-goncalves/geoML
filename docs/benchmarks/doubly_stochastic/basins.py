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

"""Bias or basin? dgpc: (1) the accurate 15x15 grid from scratch; (2) the
5x5 grid's model continued 2000 steps with 64 draws; (3) the 64-draw
model continued 2000 steps on the 5x5 grid. Each scored on the new holes
by the mixture."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_kernel as ek  # noqa: E402
import ds_prototype as ds  # noqa: E402
import run_folded as rf  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
new, holes = ek.drillholes(ek.N_NEW, ek.NEW_SEED)


def score(model, label):
    s = rf.summary(*rf.mixture_scores(model, new, ds.COUPLED), holes)
    print("  %-44s new hole %.2f  crps %.3f  final bound %.1f"
          % (label, s["score_per_hole"], s["crps"], model.training_log[-1]),
          flush=True)


print("dgpc seed %d" % seed)
model = rf.build(data, "dgpc", seed=seed)
_, per = ds.train_ds(model, 2000, 16, hermite=15)
score(model, "15x15 grid, 2000 steps (%.0f ms/step)" % (1000 * per))

model = rf.build(data, "dgpc", seed=seed)
ds.train_ds(model, 2000, 16, hermite=5)
score(model, "5x5 grid, 2000 steps")
ds.train_ds(model, 2000, 64, ds.COUPLED)
score(model, "  ... then 64 draws, 2000 more")

model = rf.build(data, "dgpc", seed=seed)
ds.train_ds(model, 2000, 64, ds.COUPLED)
score(model, "64 draws, 2000 steps")
ds.train_ds(model, 2000, 16, hermite=5)
score(model, "  ... then the 5x5 grid, 2000 more")
