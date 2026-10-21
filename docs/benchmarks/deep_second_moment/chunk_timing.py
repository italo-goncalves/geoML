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

"""How the conditioned quadrature's chunking costs a training step.

Walker Lake's GP on a GP with the second moment (`phase3_walker.build`),
a fresh model per arm: the wall time of a training step and the bound
after 25 steps (which every arm must share), with the quadrature's 64
nodes taken 4, 16 or 64 at a time through `_chunked_sums` (a loop,
recomputed under a local tape in the backward pass), or all 64 by plain
autodiff (no loop, no recompute; the intermediates kept). Prints a JSON
line per arm.

Usage: python chunk_timing.py [ARM...]   ARM like loop4, loop64, plain
"""
import json
import os
import sys
import time

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase3_walker as pw  # noqa: E402
import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402

LOOPED = net._chunked_sums


def plain_sums(fn, inputs, n_chunks):
    """Every chunk unrolled and differentiated by TensorFlow itself."""
    total = None
    for c in range(n_chunks):
        out = fn(tf.constant(c), *inputs)
        total = out if total is None else tuple(a + b
                                                for a, b in zip(total, out))
    return total


def run(arm):
    walker, _ = geoml.datasets.walker()
    if arm == "plain":
        net._chunked_sums = plain_sums
        net.BasicGP._NODES_PER_CHUNK = 64
    else:
        net._chunked_sums = LOOPED
        net.BasicGP._NODES_PER_CHUNK = int(arm[len("loop"):])
    model = pw.build(walker, "dgp", 11)
    model.train_full(5)
    start = time.perf_counter()
    model.train_full(20)
    step = (time.perf_counter() - start) / 20
    print(json.dumps(dict(arm=arm, ms_a_step=round(step * 1e3, 1),
                          bound=float(model.training_log[-1]))), flush=True)


if __name__ == "__main__":
    for arm in sys.argv[1:] or ["loop4", "loop16", "loop64", "plain"]:
        run(arm)
