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

"""Every expected-kernel call one joint-rule training step makes.

`fold_profile.build("joint")`, with `_expected_kernel` wrapped to record,
at trace time, the static shape of each call's `[..., n, m, d]` difference
and the chain of geoML functions that made it; one training step traces
them all. Prints the calls grouped by caller chain and shape, with how many
elements a component touches, largest first.

Usage: python fold_kernel_calls.py
"""
import collections
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fold_profile as fp  # noqa: E402
import geoml.latent.network as net  # noqa: E402

CALLS = collections.Counter()
original = net._expected_kernel


def recorded(kernel, ranges, mean_x, var_x, mean_y, var_y, cov=None,
             gradient=False):
    shape = (tuple(mean_x.shape[:-1]) + (mean_y.shape[-2],)
             + (mean_x.shape[-1],))
    chain = " < ".join(
        f.name for f in reversed(traceback.extract_stack()[:-1])
        if "geoml" in f.filename and f.name not in ("recorded",))[:160]
    CALLS[(chain, str(shape), cov is not None, bool(gradient))] += 1
    return original(kernel, ranges, mean_x, var_x, mean_y, var_y, cov,
                    gradient)


net._expected_kernel = recorded
model = fp.build("joint")
model.train_full(1)
rows = []
for (chain, shape, has_cov, gradient), count in CALLS.items():
    dims = [int(s) if s not in ("None",) else 0
            for s in shape.strip("()").replace(" ", "").split(",") if s]
    rows.append((int(np.prod(dims)) * count, count, shape, has_cov,
                 gradient, chain))
for elements, count, shape, has_cov, gradient, chain in sorted(rows,
                                                               reverse=True):
    print("%10d elements  x%-3d %-22s cov=%-5s grad=%-5s %s"
          % (elements, count, shape, has_cov, gradient, chain))
