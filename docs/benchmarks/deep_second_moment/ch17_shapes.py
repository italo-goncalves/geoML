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

"""The refine failure in a few calls: one-batch predictions of the walked
model on chapter 17's blocks, of the block counts the failing pass met.

Each call predicts the first COUNT blocks of the chapter's 20 m set (a
`where` mask, one batch), in the order given; the first failure is retried
eagerly (`tf.config.run_functions_eagerly`) for the Python line.

Usage: python ch17_shapes.py MODEL_DIR COUNT...
"""
import sys
import traceback

import numpy as np
import tensorflow as tf

import geoml


def main(path, counts):
    # every block mean prints, from inside the graph, the shape it is given
    # and the block count it divides by, and the line that called it
    import geoml.likelihood as lk
    aggregate = lk._aggregate

    def printed(x, n_splits=None, fun=tf.reduce_mean):
        if n_splits is not None:
            line = traceback.extract_stack()[-2].lineno
            tf.print("  _aggregate at line", line, "shape", tf.shape(x),
                     "n_splits", n_splits, output_stream=sys.stdout)
        return aggregate(x, n_splits, fun)

    lk._aggregate = printed
    model = geoml.persistence.load_model(path)
    blocks = geoml.data.BlockSet3D(
        start=[24860, 15710, 1310], n=[15, 18, 15], step=[20.0, 20.0, 20.0],
        discretization=(2, 2, 2), max_levels=3)
    for count in counts:
        where = np.zeros(blocks.n_data, bool)
        where[:count] = True
        try:
            model.predict(blocks, n_sim=20, where=where)
            print("%4d blocks: ok" % count, flush=True)
        except Exception as error:
            print("%4d blocks: FAILED %s" % (count, str(error)[:300]),
                  flush=True)
            tf.config.run_functions_eagerly(True)
            try:
                model.predict(blocks, n_sim=20, where=where)
                print("%4d blocks: ok eagerly" % count, flush=True)
            except Exception:
                traceback.print_exc()
            return


if __name__ == "__main__":
    main(sys.argv[1], [int(c) for c in sys.argv[2:]])
