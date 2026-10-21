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

"""Where the time of a deep model goes, by the TF profiler.

Walker Lake's GP on a GP (`phase3_walker.build`): the wall time of a
training step under the moment bound and under the second moment, then a
traced run of each phase -- STEPS training steps with the second moment,
and a prediction of the pathwise realizations at N_POINTS points -- whose
op events are summed by name scope (to DEPTH levels) and by op type. Ops
run on several threads at once, so the sums exceed the wall time; the
shares are what to read.

Usage: python profile_steps.py LOGDIR [STEPS N_POINTS N_SIM]
"""
import collections
import glob
import os
import re
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase3_walker as pw  # noqa: E402
import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402

DEPTH = int(os.environ.get("DEPTH", "4"))
TOP = int(os.environ.get("TOP", "30"))


def summarize(logdir, title):
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    path = sorted(glob.glob(logdir + "/**/*.xplane.pb", recursive=True))[-1]
    space = xplane_pb2.XSpace()
    space.ParseFromString(open(path, "rb").read())
    by_scope = collections.Counter()
    by_type = collections.Counter()
    total = 0.0
    for plane in space.planes:
        names = {k: v.name for k, v in plane.event_metadata.items()}
        stat_names = {k: v.name for k, v in plane.stat_metadata.items()}
        for line in plane.lines:
            for ev in line.events:
                op = None
                for st in ev.stats:
                    if stat_names.get(st.metadata_id) == "tf_op":
                        which = st.WhichOneof("value")
                        op = getattr(st, which)
                        if which == "ref_value":
                            op = plane.stat_metadata[op].name
                if not op:
                    # at tracer level 3 an op's event is named `scope/op:Type`
                    event = names.get(ev.metadata_id, "")
                    if ":" not in event or "::" in event:
                        continue
                    op = event
                ms = ev.duration_ps / 1e9
                name, _, kind = op.partition(":")
                kind = kind or names.get(ev.metadata_id, "?")
                scope = re.sub(r"_\d+(?=/|$)", "", name)
                scope = "/".join(scope.split("/")[:DEPTH])
                by_scope[scope] += ms
                by_type[kind] += ms
                total += ms
    print("=== %s: op time %.0f ms in all" % (title, total))
    print("--- by scope")
    if not total:
        print("no op events in the trace")
        return
    for scope, ms in by_scope.most_common(TOP):
        print("%9.1f ms %5.1f%%  %s" % (ms, 100 * ms / total, scope))
    print("--- by op type")
    for kind, ms in by_type.most_common(20):
        print("%9.1f ms %5.1f%%  %s" % (ms, 100 * ms / total, kind))


def main(logdir, steps, n_points, n_sim):
    walker, grid = geoml.datasets.walker()
    # wall time a step, after the trace, each arm on a model of its own (a
    # training step's trace is kept per model)
    for conditioned in (False, True):
        net._UNCERTAIN_POINTS_SECOND_MOMENT = conditioned
        model = pw.build(walker, "dgp", 11)
        model.train_full(5)
        start = time.perf_counter()
        model.train_full(20)
        print("%s: %.1f ms a training step" % (
            "second moment" if conditioned else "moment bound",
            (time.perf_counter() - start) / 20 * 1e3), flush=True)
    net._UNCERTAIN_POINTS_SECOND_MOMENT = True
    options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=3, device_tracer_level=1)
    tf.profiler.experimental.start(os.path.join(logdir, "train"),
                                   options=options)
    model.train_full(steps)
    tf.profiler.experimental.stop()
    summarize(os.path.join(logdir, "train"), "training, %d steps" % steps)

    rows = np.random.default_rng(0).choice(grid.n_data, n_points,
                                           replace=False)
    target = geoml.data.PointData.from_array(
        np.asarray(grid.coordinates, dtype=float)[rows], ["X", "Y"])
    model.predict(target, n_sim=n_sim)            # traced once
    start = time.perf_counter()
    model.predict(target, n_sim=n_sim)
    print("prediction: %.1f s for %d points and %d realizations"
          % (time.perf_counter() - start, n_points, n_sim), flush=True)
    tf.profiler.experimental.start(os.path.join(logdir, "predict"),
                                   options=options)
    model.predict(target, n_sim=n_sim)
    tf.profiler.experimental.stop()
    summarize(os.path.join(logdir, "predict"), "prediction")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], int(args[1]) if len(args) > 1 else 20,
         int(args[2]) if len(args) > 2 else 5000,
         int(args[3]) if len(args) > 3 else 16)
