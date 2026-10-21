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

"""What the conditioned quadrature costs in memory and time, by chunk.

Walker Lake's GP on a GP with the second moment (`phase3_walker.build`), on
the GPU, a fresh model per arm: the quadrature's chunk forced to each size
(`_NODES_PER_CHUNK`, `QUADRATURE_PLAIN=1` for one chunk by plain
autodiff), five steps traced, then ten timed with the device's peak memory
reset before them. Prints a JSON line per arm, with the bytes per node and
per element of an `[n, m, d]` array the slope implies.

Usage: python chunk_memory.py ARM...   ARM like 4, 16, 64, plain, xla16
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


def run(arm):
    walker, _ = geoml.datasets.walker()
    xla = arm.startswith("xla")
    name = arm[3:] if xla else arm
    net._QUADRATURE_XLA = xla
    net._QUADRATURE_RULE = "hermite" if name == "hermite" else "sobol"
    # a number forces the chunk; "plain", "hermite": every node at once;
    # "budget": the default budget's choice
    forced = name not in ("plain", "hermite", "budget")
    net.BasicGP._NODES_PER_CHUNK = int(name) if forced else None
    model = pw.build(walker, "dgp", 11)
    if name in ("plain", "hermite"):
        model.options.quadrature_memory = 1e15
    q = 25 if name == "hermite" else 64
    chunk = min(int(name), q) if forced else q
    model.train_full(5)
    gpu = bool(tf.config.list_physical_devices("GPU"))
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
        before = tf.config.experimental.get_memory_info("GPU:0")["current"]
    start = time.perf_counter()
    model.train_full(10)
    step = (time.perf_counter() - start) / 10
    peak = (tf.config.experimental.get_memory_info("GPU:0")["peak"]
            - before) if gpu else float("nan")
    n, m, d = walker.n_data, pw.N_IND, 2
    print(json.dumps(dict(
        arm=arm, chunk=chunk, ms_a_step=round(step * 1e3, 1),
        peak_mb=round(peak / 1e6, 1),
        bytes_per_node_element=round(peak / (chunk * n * m * d), 1),
        bound=float(model.training_log[-1]))), flush=True)


if __name__ == "__main__":
    for arm in sys.argv[1:]:
        run(arm)
