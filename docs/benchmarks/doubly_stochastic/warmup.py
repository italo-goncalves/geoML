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

"""A sampled warm-up, then the deterministic second-moment bound (5x5
grid), 2000 steps in all. Usage: python warmup.py NET SEED WARM_STEPS
DRAWS. One JSON line to warmup.jsonl."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_kernel as ek, ds_prototype as ds, run_folded as rf
net, seed, warm, draws = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
new, holes = ek.drillholes(ek.N_NEW, ek.NEW_SEED)
model = rf.build(data, net, seed=seed)
start = time.perf_counter()
ds.train_ds(model, warm, draws, ds.COUPLED)
ds.train_ds(model, 2000 - warm, 16, hermite=5)
seconds = time.perf_counter() - start
s = rf.summary(*rf.mixture_scores(model, new, ds.COUPLED), holes)
row = dict(network=net, seed=seed, warm=warm, draws=draws, seconds=seconds, **s)
open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "doubly_stochastic"), "warmup.jsonl"), "a").write(json.dumps(row) + "\n")
print(json.dumps(row))
