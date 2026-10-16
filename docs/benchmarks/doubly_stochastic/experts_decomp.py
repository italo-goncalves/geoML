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

import os, sys
import numpy as np, tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "moment", sys.argv[1], sys.argv[2]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experts.py")).read().split("model, sizes = build()")[0])
import ds_experts as dx
model, sizes = build()
model.train_full(2000)
x = tf.constant(np.asarray(new.coordinates), tf.float64)
with model._propagation():
    model._refresh(model.options.jitter)
    mu_m, v_m = model.leaves[0].propagate(x)
    mask = np.array(ds.uncertain_dims(model.leaves[0].parent), float)
    s1 = s2 = sv = 0.0; n = 0
    for c in range(40):
        nodes = tf.random.stateless_normal([50, x.shape[0], len(mask)], [3, c], dtype=tf.float64) * mask
        mu, var = dx.expert_moments(model, x, nodes)
        mu, var = mu.numpy()[:, :, 0], var.numpy()[:, :, 0]
        s1, s2, sv, n = s1 + mu.sum(0), s2 + (mu**2).sum(0), sv + var.sum(0), n + 50
    mean = s1 / n; exact = sv / n + s2 / n - mean**2
    hv = ds._parent_moments.__wrapped__ if False else None
    m = model.leaves[0].parent.propagate(x)
v_m = v_m.numpy()[:, 0]
print("%s expert(s), seed %s: moment var %.3f, exact %.3f (E[var] %.3f + Var[mu] %.3f), median share captured %.2f; hidden sd at new holes median %.3f"
      % (sys.argv[3], sys.argv[2], v_m.mean(), exact.mean(), (sv/n).mean(), (s2/n-mean**2).mean(), np.median(v_m/exact), np.median(np.sqrt(m[1].numpy()[:, :2]))))
