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

"""Cost and memory of a training step and a prediction, moment against
doubly stochastic, on the folded section's GP-on-a-GP at larger sizes.

Usage: python scale.py N_HOLES N_IND ARM S
  ARM: moment | ds (coupled)
One process per configuration, so the peak resident memory is its own.
Prints one JSON line.
"""
import json
import os
import resource
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_kernel as ek  # noqa: E402
import ds_prototype as ds  # noqa: E402
import run_folded as rf  # noqa: E402


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv):
    n_holes, n_ind, arm, s = int(argv[0]), int(argv[1]), argv[2], int(argv[3])
    ek.N_IND = n_ind
    data, _ = ek.drillholes(n_holes, ek.SEED)
    model = rf.build(data, "dgp")
    base = rss_mb()
    steps = 10
    if arm == "moment":
        model.train_full(3)
        start = time.perf_counter()
        model.train_full(steps)
        per = (time.perf_counter() - start) / steps
    else:
        _, per = ds.train_ds(model, 10 + steps, s, ds.COUPLED)
    train_peak = rss_mb()

    # prediction at 10 000 locations: the moments, or the mixture over S draws
    x = tf.constant(np.column_stack([
        np.random.default_rng(0).uniform(0.2, 4.8, 10000),
        np.random.default_rng(1).uniform(0.0, 3.0, 10000)]), tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        if arm == "moment":
            fn = tf.function(lambda: model.leaves[0].predict(x, n_sim=1)[:2])
        else:
            fn = tf.function(lambda: ds.leaf_moments(model, x, s, 1,
                                                     ds.COUPLED))
        fn()
        start = time.perf_counter()
        for _ in range(3):
            fn()
        pred = (time.perf_counter() - start) / 3
    print(json.dumps(dict(n=n_holes * ek.N_PER_HOLE, m=n_ind, arm=arm, s=s,
                          ms_per_iter=1000 * per, predict_ms=1000 * pred,
                          rss_before_mb=base, train_peak_mb=train_peak,
                          total_peak_mb=rss_mb())))


if __name__ == "__main__":
    main(sys.argv[1:])
