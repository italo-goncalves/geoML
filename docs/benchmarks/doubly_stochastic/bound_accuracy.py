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

"""Is the Hermite grid an accurate integral of the data term? For a model
trained by the 5x5 grid and one trained by 64 draws a step (dgpc, seed
11), the data term at the end of training by grids of 3..21 nodes a
dimension and by 4000 draws."""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_kernel as ek  # noqa: E402
import ds_prototype as ds  # noqa: E402
import run_folded as rf  # noqa: E402

net, seed = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 \
    else ("dgpc", 11)
data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
for arm in ("quad5", "ds64"):
    model = rf.build(data, net, seed=seed)
    if arm == "quad5":
        ds.train_ds(model, 2000, 16, hermite=5)
    else:
        ds.train_ds(model, 2000, 64, ds.COUPLED)
    inputs = [model.data.variables[v].training_input()
              for v in model.variables]
    x = tf.constant(model.data.coordinates, tf.float64)
    y = tf.constant(model.y, tf.float64)
    hv = tf.constant(model.has_value, tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        terms = {k: float(ds.quadrature_data_term(model, x, y, hv, inputs, k))
                 for k in (3, 5, 9, 15, 21)}
        draws = [float(ds.ds_data_term(model, x, y, hv, inputs, 500, c,
                                       ds.COUPLED)) for c in range(8)]
        mean, var, _ = ds._parent_moments(model.leaves[0], x)
        leaf_r = model.leaves[0].parameters["ranges"].get_value().numpy()
    print("%s %s-trained: data term by grid %s; by 4000 draws %.2f +- %.2f"
          % (net, arm, {k: round(v, 2) for k, v in terms.items()},
             np.mean(draws), np.std(draws) / np.sqrt(len(draws))))
    print("    hidden sd at the data: median %.3f, max %.3f; leaf ranges %s"
          % (np.median(np.sqrt(var.numpy()[:, :2])),
             np.sqrt(var.numpy()[:, :2]).max(), np.round(leaf_r.ravel(), 3)))
