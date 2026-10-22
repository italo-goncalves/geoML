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

"""Where a grid prediction of the folded-ore model goes, by self time.

Loads a model `fold_ablation.py` saved, predicts the 161 x 201 grid once
to trace, times a second prediction with the pathwise draws on and with
them off (`_PATHWISE_REALIZATIONS`, a separate trace), then traces one
prediction with them on through the TF profiler and sums its op self time
by name scope and op type (`deep_second_moment/profile_steps.summarize`).

Usage: python fold_predict_profile.py MODEL_DIR LOGDIR
"""
import os
import sys
import time

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "deep_second_moment"))
from profile_steps import summarize  # noqa: E402
import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def timed(model, grid):
    model.predict(grid)
    start = time.perf_counter()
    model.predict(grid)
    return time.perf_counter() - start


def main(model_dir, logdir):
    model = geoml.persistence.load_model(model_dir)
    grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
    on = timed(model, grid)
    net._PATHWISE_REALIZATIONS = False
    off = timed(model, grid)
    net._PATHWISE_REALIZATIONS = True
    print("prediction: %.1f s with the pathwise draws, %.1f s without"
          % (on, off), flush=True)
    options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=3, device_tracer_level=1)
    tf.profiler.experimental.start(logdir, options=options)
    model.predict(grid)
    tf.profiler.experimental.stop()
    summarize(logdir, "one grid prediction, self time")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
