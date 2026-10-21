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

"""Where the pathwise realizations' short-range roughness comes from, and
how the inducing points' number bears on it.

On the folded section's networks trained with the conditioned second
moment (seed 11), the leaf's realizations on the true field's grid, with
the location-keyed noise switched off, and then one more part of the draw
at a time: the fit of the inducing values' draw (`_fitted_draw`), the
leaf's inducing values' draw (its normals zeroed), the hidden layer's draw
(its normals zeroed: the leaf reads the hidden layer's mean). For each,
the semivariance at the grid's shortest lag against the variance at a
point. With `N_IND` set, the same with that many inducing points, and the
new holes' log score. Appends to `roughness.jsonl`.

Usage: python roughness.py ITERATIONS [N_IND]
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def leaf_draws(model, x, leaf, hidden):
    with model._propagation():
        model._refresh(model.options.jitter)
        with net.once_per_pass():
            _, var = leaf.propagate(x)
            sims = np.asarray(leaf.simulate(400, [3, 1]))[0]
    f = sims.reshape(31, 47, -1)
    return dict(shortest_lag=float(0.5 * np.mean((f[:, 1:] - f[:, :-1]) ** 2)),
                sill=float(np.mean(sims.var(1))),
                var_ratio=float(np.median(sims.var(1)
                                          / np.asarray(var)[:, 0])))


def main(iterations, n_ind):
    gm.ek.N_IND = n_ind
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    new, holes = gm.ek.drillholes(gm.ek.N_NEW, gm.ek.NEW_SEED)
    points, _ = gm.rf.grid()
    x = tf.constant(points, tf.float64)
    real_loc, real_sim, real_fit = (net._location_normals,
                                    net._simulation_normals, net._fitted_draw)
    for network in ("dgp", "concat"):
        keep, gm.ek.SEED = gm.ek.SEED, 11
        try:
            model = gm.build(data, network, kr.Gaussian)
        finally:
            gm.ek.SEED = keep
        model.train_full(iterations)
        leaf = model.leaves[0]
        hidden = leaf.parent if network == "dgp" else leaf.parent.parents[0]
        row = dict(network=network, n_inducing=n_ind, iterations=iterations,
                   moments=gm.rf.summary(*gm.rf.moment_scores(model, new),
                                         holes))
        row["as_built"] = leaf_draws(model, x, leaf, hidden)
        net._location_normals = lambda c, n, s, k: real_loc(c, n, s, k) * 0.0
        try:
            row["no_noise"] = leaf_draws(model, x, leaf, hidden)
            net._fitted_draw = lambda ksk, kr_: (
                tf.ones_like(kr_), real_fit(ksk, kr_)[1])
            row["no_noise_no_fit"] = leaf_draws(model, x, leaf, hidden)
            net._fitted_draw = real_fit
            for name, target in (("no_leaf_values", leaf.name),
                                 ("hidden_at_mean", hidden.name)):
                net._simulation_normals = (
                    lambda shape, seed, key=None, t=target:
                    real_sim(shape, seed, key) * (0.0 if key == t else 1.0))
                row["no_noise_" + name] = leaf_draws(model, x, leaf, hidden)
                net._simulation_normals = real_sim
        finally:
            net._location_normals, net._simulation_normals = real_loc, real_sim
            net._fitted_draw = real_fit
        print(json.dumps(row), flush=True)
        with open(os.path.join(gm.FIGURES, "roughness.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30)
