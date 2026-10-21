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

"""What a reader's pathwise realizations cost in memory and time, by chunk.

Walker Lake's GP on a GP with the second moment (`phase3_walker.build`),
trained 200 steps, predicting N_SIM realizations at N_POINTS points of the
exhaustive field: the realizations' chunk forced to each size
(`_REALIZATIONS_PER_CHUNK`; "budget" for the default budget's choice), the
prediction traced once, then timed with the device's peak memory reset.
Prints a JSON line per arm, with the bytes per realization and per element
of its `[n, m, d]` and `[n, size (J + 1)]` arrays.

Usage: python chunk_predict.py N_POINTS N_SIM ARM...
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase3_walker as pw  # noqa: E402
import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def main(n_points, n_sim, arms):
    walker, grid = geoml.datasets.walker()
    model = pw.build(walker, "dgp", 11)
    model.train_full(200)
    rows = np.random.default_rng(0).choice(grid.n_data, n_points,
                                           replace=False)
    x = np.asarray(grid.coordinates, dtype=float)[rows]
    model.options.prediction_batch_size = n_points
    gpu = bool(tf.config.list_physical_devices("GPU"))
    leaf = model.leaves[0]
    for arm in arms:
        net.BasicGP._REALIZATIONS_PER_CHUNK = None if arm == "budget" \
            else int(arm)
        target = geoml.data.PointData.from_array(x, ["X", "Y"])
        model.predict(target, n_sim=n_sim)
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
            before = tf.config.experimental.get_memory_info("GPU:0")[
                "current"]
        start = time.perf_counter()
        model.predict(target, n_sim=n_sim)
        seconds = time.perf_counter() - start
        peak = (tf.config.experimental.get_memory_info("GPU:0")["peak"]
                - before) if gpu else float("nan")
        chunk = n_sim if arm == "budget" else int(arm)
        n_features = max(1024, 2 * max(int(m) for m in leaf.root.n_ip)) + 1
        elements = n_points * (pw.N_IND * 2 + leaf.size * n_features)
        print(json.dumps(dict(
            arm=arm, seconds=round(seconds, 2), peak_mb=round(peak / 1e6, 1),
            bytes_per_draw_element=round(peak / (chunk * elements), 1),
            first=float(np.asarray(target.get("V").get_simulations())[0, 0]))),
            flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
