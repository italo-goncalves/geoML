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

"""Chapter 17's failing refine, replayed from the saved walked model.

Loads the model the chapter trained (saved by a runner that checkpoints
it), runs the chapter's `refine`, and logs every prediction batch: its
blocks, its rows, and the split the block set hands the model. ARM `auto`
keeps the model's own batch size, `fixed` the options' 20000 rows. Prints
the whole traceback on failure.

Usage: python ch17_replay.py MODEL_DIR auto|fixed
"""
import sys
import time
import traceback

import numpy as np

import geoml


def main(path, arm):
    model = geoml.persistence.load_model(path)
    if arm == "fixed":
        model._prediction_rows = lambda: int(
            model.options.prediction_batch_size)
    print(arm, "rows a batch:", model._prediction_rows(), flush=True)
    blocks = geoml.data.BlockSet3D(
        start=[24860, 15710, 1310], n=[15, 18, 15], step=[20.0, 20.0, 20.0],
        discretization=(2, 2, 2), max_levels=3)
    original = type(blocks).get_batched_coordinates

    def logged(self, batch):
        coords, splits = original(self, batch)
        rows = np.shape(coords)[0]
        print("  batch: %d blocks, %d rows" % (len(batch), rows),
              flush=True)
        if splits is not None and rows % int(splits):
            print("  batch of %d blocks: %d rows, splits %s -- NOT A "
                  "MULTIPLE" % (len(batch), rows, splits), flush=True)
        return coords, splits

    type(blocks).get_batched_coordinates = logged

    # every block-mean in the prediction checks its rows against the block
    # count, naming the line that called it, where a reshape would only
    # report the sizes
    import tensorflow as tf
    import geoml.likelihood as lk
    aggregate = lk._aggregate

    def checked(x, n_splits=None, fun=tf.reduce_mean):
        if n_splits is not None:
            caller = traceback.extract_stack()[-2]
            tf.debugging.Assert(
                tf.equal(tf.math.floormod(tf.shape(x)[0], n_splits), 0),
                ["_aggregate from %s:%d" % (caller.filename, caller.lineno),
                 tf.shape(x), n_splits], summarize=10)
        return aggregate(x, n_splits, fun)

    lk._aggregate = checked
    start = time.perf_counter()
    try:
        refined = geoml.models.refine(model, blocks)
        print(arm, "refined to %d blocks in %.0f s"
              % (refined.n_data, time.perf_counter() - start), flush=True)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
