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

"""What each piece of the joint rule costs on the folded-ore model.

`fold_profile.build`'s model, one arm a process (a traced step stays
resident, so arms do not share one): the wall time of a training step
(20 after 5 traced) and of the 161 x 201 grid's prediction (one after one
traced), with pieces switched off --

  marginal          the marginal rule
  joint_plain       the joint rule, full_variance off
  joint_nocond      full variance, the conditioned second moment off
                    (`_UNCERTAIN_POINTS_SECOND_MOMENT`)
  joint_nopath      full variance, the pathwise draws off
                    (`_PATHWISE_REALIZATIONS`)
  joint             the default
  joint_norecomp    the default, the walk's steps kept rather than
                    recomputed in the backward pass (`tf.recompute_grad`
                    replaced by the identity, as the suite's test does)
  plain_norecomp    joint_plain likewise
  joint_h5          the default, 5 x 5 Gauss-Hermite nodes for the
                    conditioned quadrature instead of 7 x 7
  joint_xla         the default, XLA on the quadrature's and the pathwise
                    draws' chunk bodies (`_QUADRATURE_XLA`)
  joint_h5_xla      both

Prints a JSON line.

Usage: python fold_cost.py ARM
"""
import json
import os
import sys
import time

import geoml  # noqa: F401
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fold_profile as fp  # noqa: E402
import geoml.latent.network as net  # noqa: E402

ARMS = {
    "marginal": dict(propagation="marginal"),
    "joint_plain": dict(full=False),
    "joint_nocond": dict(conditioned=False),
    "joint_nopath": dict(pathwise=False),
    "joint": dict(),
    "joint_norecomp": dict(recompute=False),
    "plain_norecomp": dict(full=False, recompute=False),
    "joint_h5": dict(hermite=5),
    "joint_xla": dict(xla=True),
    "joint_h5_xla": dict(hermite=5, xla=True),
}


def main(arm):
    setting = dict(ARMS[arm])
    net._UNCERTAIN_POINTS_SECOND_MOMENT = setting.get("conditioned", True)
    net._PATHWISE_REALIZATIONS = setting.get("pathwise", True)
    # the quadrature's rule (Gauss-Hermite points a dimension) and XLA on a
    # quadrature or pathwise chunk's body; set before the first trace, the
    # Hermite count not being in the trace's key
    net._HERMITE_POINTS = setting.get("hermite", 7)
    net._QUADRATURE_XLA = setting.get("xla", False)
    if not setting.get("recompute", True):
        tf.recompute_grad = lambda f: f
    model = fp.build(setting.get("propagation", "joint"),
                     setting.get("full", True))
    grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[161, 201])
    model.train_full(5)
    start = time.perf_counter()
    model.train_full(20)
    step = (time.perf_counter() - start) / 20
    model.predict(grid)
    start = time.perf_counter()
    model.predict(grid)
    predict = time.perf_counter() - start
    print(json.dumps(dict(arm=arm, step_ms=round(step * 1e3, 1),
                          predict_s=round(predict, 2),
                          bound=round(float(model.training_log[-1]), 4))),
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
