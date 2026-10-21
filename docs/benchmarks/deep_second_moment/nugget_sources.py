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

"""Where the pathwise realizations' nugget comes from, on the folded
section: the variance added as independent noise per location -- the
hidden layer's realization gap and the leaf's own unexplained variance --
against what each layer reports, and the inducing points' spacing against
the ranges each layer learned.

Usage: python nugget_sources.py [ITERATIONS]
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def main(iterations):
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    points, _ = gm.rf.grid()
    x = tf.constant(points, tf.float64)
    for network in ("dgp", "concat"):
        keep, gm.ek.SEED = gm.ek.SEED, 11
        try:
            model = gm.build(data, network, kr.Gaussian)
        finally:
            gm.ek.SEED = keep
        model.train_full(iterations)
        leaf = model.leaves[0]
        hidden = leaf.parent if network == "dgp" else leaf.parent.parents[0]
        root = leaf.root
        with model._propagation():
            model._refresh(model.options.jitter)
            with net.once_per_pass():
                _, leaf_var = leaf.propagate(x)
                coords, chain, weights, uncertain = leaf._path_state
                _, hidden_var = hidden.propagate(x)
                gap = np.asarray(hidden._realization_gap).T     # [n, 2]
                # the leaf's own unexplained variance at its input's mean,
                # with no conditioning: 1 - k K^-1 kᵀ
                k = np.asarray(leaf.expected_covariance(
                    chain[0].mean, None, *leaf._parent_points([0]))[0]
                    if False else leaf.expected_covariance(
                        chain[0].mean, tf.zeros_like(chain[0].mean),
                        leaf._parent_points([0])[0][0],
                        leaf._parent_points([0])[1][0]))
                inv_k = np.asarray(leaf.cov_inv[0])
                own = np.maximum(1.0 - np.einsum("nm,mk,nk->n", k, inv_k, k),
                                 0.0)
        z = np.asarray(root.inducing_points[0])
        spacing = np.median(cKDTree(z).query(z, k=2)[0][:, 1])
        x_tr = np.asarray(root.propagate(x)[0])
        row = dict(
            network=network, n_inducing=int(len(z)),
            n_data=int(data.n_data),
            inducing_spacing_input_units=float(spacing),
            hidden_ranges=np.asarray(
                hidden.parameters["ranges"].get_value()).ravel().tolist(),
            leaf_ranges=np.asarray(
                leaf.parameters["ranges"].get_value()).ravel().tolist(),
            hidden_gap_share=float(np.median(gap / np.asarray(hidden_var))),
            leaf_own_share=float(np.median(own / np.asarray(leaf_var)[:, 0])),
            grid_to_inducing=float(np.median(cKDTree(z).query(x_tr)[0])))
        print(json.dumps(row), flush=True)


if __name__ == "__main__" and not os.environ.get("DECOMPOSE"):
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)


def decomposition(iterations):
    """The realizations' variance with every noise, without the leaf's own
    and without the input gap, and the semivariance at the grid's
    shortest lag against the field's sill, on the leaf."""
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    points, _ = gm.rf.grid()
    x = tf.constant(points, tf.float64)
    real = net._location_normals
    for network in ("dgp", "concat"):
        keep, gm.ek.SEED = gm.ek.SEED, 11
        try:
            model = gm.build(data, network, kr.Gaussian)
        finally:
            gm.ek.SEED = keep
        model.train_full(iterations)
        leaf = model.leaves[0]
        out = {}
        for drop in (None, "/own", "/input", "both"):
            def keyed(coords, count, seed, key, drop=drop):
                z = real(coords, count, seed, key)
                gone = drop == "both" or (drop and key.endswith(drop))
                return z * 0.0 if gone else z
            net._location_normals = keyed
            try:
                with model._propagation():
                    model._refresh(model.options.jitter)
                    with net.once_per_pass():
                        _, var = leaf.propagate(x)
                        sims = np.asarray(leaf.simulate(400, [3, 1]))[0]
            finally:
                net._location_normals = real
            f = sims.reshape(31, 47, -1)
            nugget = 0.5 * np.mean((f[:, 1:] - f[:, :-1]) ** 2)
            out[drop or "all"] = dict(
                var_ratio=float(np.median(sims.var(1) / np.asarray(var)[:, 0])),
                shortest_lag_semivariance=float(nugget),
                sill=float(np.mean(sims.var(1))))
        print(json.dumps(dict(network=network, **out)), flush=True)


if __name__ == "__main__" and os.environ.get("DECOMPOSE"):
    decomposition(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
