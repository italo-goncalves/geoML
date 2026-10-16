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

"""Gates for the prototype: averaged over draws, the coupled cross-
covariance is the joint expected kernel and the independent one the
expected kernel without the covariance; the mixture's mean is the moment
model's mean. On the folded section's two deep networks, briefly trained."""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_kernel as ek  # noqa: E402
import ds_prototype as ds  # noqa: E402
from geoml.latent import network as net  # noqa: E402

data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
new, _ = ek.drillholes(20, ek.NEW_SEED)
x = tf.constant(np.asarray(new.coordinates), tf.float64)
S = 20000

for network in ("dgp", "dgpc"):
    model = ek.folded_model(data, network, "joint", iterations=200)
    leaf = model.leaves[0]
    with model._propagation():
        model._refresh(model.options.jitter)
        mean, var, cov = ds._parent_moments(leaf, x)
        ips, ipvs = leaf._parent_points(net._active_experts(leaf.root))
        joint = leaf.expected_covariance(mean, var, ips[0], ipvs[0], cov)
        alone = leaf.expected_covariance(mean, var, ips[0], ipvs[0], None)
        sums = {k: 0.0 for k in ("c", "c2", "i", "i2", "m", "m2", "sd")}
        mu_m, _ = leaf.propagate(x)
        for chunk in range(S // 1000):
            eps = tf.random.stateless_normal([1000] + list(mean.shape),
                                             [3, chunk], dtype=tf.float64)
            h = mean[None] + tf.sqrt(var)[None] * eps
            kc = ds.conditional_kernel(leaf, h, mean, var, cov, ds.COUPLED)
            ki = ds.conditional_kernel(leaf, h, mean, var, cov,
                                       ds.INDEPENDENT)
            mu_s, _ = ds.leaf_conditionals(leaf, kc)
            for k, t in (("c", kc), ("i", ki), ("m", mu_s)):
                t = t.numpy()
                sums[k] = sums[k] + t.sum(0)
                sums[k + "2"] = sums[k + "2"] + (t ** 2).sum(0)
        mean_of = {k: sums[k] / S for k in ("c", "i", "m")}
        se = {k: np.sqrt(np.maximum(sums[k + "2"] / S - mean_of[k] ** 2, 0)
                         / S) for k in ("c", "i", "m")}
        err_c = np.abs(mean_of["c"] - joint.numpy()); se_c = se["c"]
        err_i = np.abs(mean_of["i"] - alone.numpy()); se_i = se["i"]
        err_mu = np.abs(mean_of["m"] - mu_m.numpy()); se_mu = se["m"]
        gap = np.abs(joint.numpy() - alone.numpy())
        wrong = np.abs(mean_of["c"] - alone.numpy())
        mix_sd = np.sqrt(np.maximum(sums["m2"] / S - mean_of["m"] ** 2, 0))
    print("%-5s gate A coupled: max err %.2e (%.1f s.e.)   gate B independent:"
          " max err %.2e (%.1f s.e.)   gate C mean: max err %.2e (%.1f s.e.)"
          % (network, err_c.max(), (err_c / np.maximum(se_c, 1e-12)).max(),
             err_i.max(), (err_i / np.maximum(se_i, 1e-12)).max(),
             err_mu.max(), (err_mu / np.maximum(se_mu, 1e-12)).max()))
    print("      joint vs uncoupled kernel differ by up to %.3f; the coupled"
          " average sits %.2e from the joint and %.3f from the uncoupled"
          % (gap.max(), err_c.max(), wrong.max()))
    print("      parent sd at the points: median %.3f, max %.3f; sd of the"
          " leaf mean over draws: median %.3f"
          % (np.median(np.sqrt(var.numpy())), np.sqrt(var.numpy()).max(),
             np.median(mix_sd)))
