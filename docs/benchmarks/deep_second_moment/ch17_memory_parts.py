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

"""Where chapter 17's prediction memory goes, part by part.

The chapter's walked model (`ch17_memory.build`), trained one epoch, then
one prediction batch of random points in the chapter's block box under each
setting: the defaults; the pathwise realizations off; the conditioned
second moment off; both off; 2 realizations instead of 20; 5000 rows
instead of 20000. The process's peak resident memory is reset before each
(`/proc/self/clear_refs`, Linux), so each line is that prediction's own
peak above what the process already held. One setting a process.

Usage: python ch17_memory_parts.py SETTING
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch17_memory as cm  # noqa: E402
import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def _status(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    return 0


def _reset_peak():
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def measure(model, label, rows=20000, n_sim=20):
    x = np.random.default_rng(0).uniform(
        [24850, 15700, 1300], [25150, 16060, 1600], [rows, 3])
    target = geoml.data.PointData.from_array(x, ["X", "Y", "Z"])
    model.options.prediction_batch_size = rows
    base = _status("VmRSS")
    _reset_peak()
    start = time.perf_counter()
    model.predict(target, n_sim=n_sim)
    seconds = time.perf_counter() - start
    peak = _status("VmHWM")
    print("%-28s rows %5d sims %2d  %6.1f s  peak above base %6.2f GB"
          % (label, rows, n_sim, seconds, (peak - base) / 1e9), flush=True)


SETTINGS = {
    "defaults": dict(),
    "pathwise_off": dict(pathwise=False),
    "conditioned_off": dict(conditioned=False),
    "both_off": dict(pathwise=False, conditioned=False),
    "sims2": dict(n_sim=2),
    "rows5000": dict(rows=5000),
    "rows10000": dict(rows=10000),
}


def main(name):
    """One setting, in a process of its own: memory a prediction frees
    stays in the allocator, so a second prediction in one process reuses it
    and its peak says nothing."""
    setting = dict(SETTINGS[name])
    net._PATHWISE_REALIZATIONS = setting.pop("pathwise", True)
    net._UNCERTAIN_POINTS_SECOND_MOMENT = setting.pop("conditioned", True)
    model = cm.build("on")
    model.train_svi(1)
    measure(model, name, **setting)


if __name__ == "__main__":
    main(sys.argv[1])
