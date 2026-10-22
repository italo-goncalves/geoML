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

"""What the folded-ore model costs, and where, under each propagation.

`fold_ablation.py`'s model with the Matern32 reader. For the marginal rule,
the joint rule, and the joint rule with `full_variance` off: the wall time
of a training step (after five traced ones) and of a prediction of the
161 x 201 grid (after one traced one). Then a TF profiler trace of STEPS
joint-rule training steps, its op time summed by name scope and op type
(`deep_second_moment/profile_steps.summarize`). Ops run on several threads
at once, so the sums exceed the wall time; read the shares.

Usage: python fold_profile.py LOGDIR [STEPS]
"""
import os
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "deep_second_moment"))
from profile_steps import summarize  # noqa: E402
import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent as gl  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.transform as tr  # noqa: E402


def build(propagation, full=True):
    geoml.set_seed(1)
    z1 = np.linspace(0, -20, 101)
    rock_1 = np.where((z1 < -11.5) & (z1 > -14.5), "ore", "waste")
    rock_2 = np.where((z1 < -2.5) & (z1 > -5.5), "ore", "waste")
    rock_2 = np.where((z1 < -8.5) & (z1 > -11.5), "ore", rock_2)
    rock_2 = np.where((z1 < -14.5) & (z1 > -17.5), "ore", rock_2)
    coords = np.stack([
        np.concatenate([-5 * np.ones_like(z1), np.zeros_like(z1),
                        5 * np.ones_like(z1)]),
        np.concatenate([z1, z1, z1])], axis=1)
    data = geoml.data.PointData.from_array(coords)
    data.add_categorical_variable(
        "Rock_deep", ["waste", "ore"],
        np.concatenate([rock_1, rock_2, rock_1[::-1]]))
    ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])
    deep_in = gl.BasicInput(
        ip, tr.Concatenate(tr.Isotropic(20), tr.Isotropic(2)), center=True)
    base_gp = gl.BasicGP(gl.SelectInput(deep_in, [0, 1]), 1, fix_range=True)
    field = gl.BasicGP(gl.SelectInput(deep_in, [2, 3]), size=2,
                       fix_range=True)
    walk = gl.GPWalk(field, step=0.01, n_steps=10)
    reader = gl.BasicGP(walk, kernel=kr.Matern32())
    return geoml.models.VGPNetwork(
        data=data, variables="Rock_deep",
        likelihoods=lk.HierarchicalGaussianIndicator(2),
        latent_network=gl.Concatenate(base_gp, reader),
        options=geoml.models.GPOptions(jitter=1e-6, verbose=False,
                                       propagation=propagation,
                                       full_variance=full))


def main(logdir, steps):
    grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
    for label, propagation, full in (("marginal", "marginal", True),
                                     ("joint", "joint", True),
                                     ("joint, full_variance off", "joint",
                                      False)):
        model = build(propagation, full)
        model.train_full(5)
        start = time.perf_counter()
        model.train_full(20)
        step = (time.perf_counter() - start) / 20
        model.predict(grid)
        start = time.perf_counter()
        model.predict(grid)
        print("%-26s %7.1f ms a training step, %6.1f s a grid prediction"
              % (label, step * 1e3, time.perf_counter() - start), flush=True)
        if label == "joint":
            options = tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=3, device_tracer_level=1)
            tf.profiler.experimental.start(os.path.join(logdir, "train"),
                                           options=options)
            model.train_full(steps)
            tf.profiler.experimental.stop()
            summarize(os.path.join(logdir, "train"),
                      "joint rule, %d training steps" % steps)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
