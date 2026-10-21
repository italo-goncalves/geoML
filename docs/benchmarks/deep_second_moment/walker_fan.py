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

"""Why chapter 13's simulation fan sits above the data under full variance.

Chapter 13's Walker model (saved by the release run's checkpointing
runner), predicted at the 470 samples with N_SIM realizations; the
semivariogram, by distance class, of the data, of the full realizations
plus the pair-averaged `noise_variance` (as the `variogram` figure draws
it), and of the same realizations without each GP node's own residual (the
smooth draws, `smooth_draws`) plus the noise. Declustered as the figure
declusters it. Prints one line per lag.

Usage: python walker_fan.py MODEL_DIR [N_SIM]
"""
import sys

import numpy as np
import tensorflow as tf

import geoml
import geoml.latent.network as net


def semivariogram(values, xy, weights, edges):
    """Weighted semivariance of `values` (`[n]` or `[n, S]`, averaged over
    the realizations) by distance class."""
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    w = weights[:, None] * weights[None, :]
    upper = np.triu(np.ones_like(d, bool), 1)
    v = values if values.ndim == 2 else values[:, None]
    gamma = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pairs = upper & (d >= lo) & (d < hi)
        i, j = np.nonzero(pairs)
        sq = 0.5 * ((v[i] - v[j]) ** 2).mean(axis=1)
        gamma.append(np.sum(w[i, j] * sq) / np.sum(w[i, j]))
    return np.array(gamma)


def main(path, n_sim=50):
    model = geoml.persistence.load_model(path)
    data = model.data
    xy = np.asarray(data.coordinates, dtype=float)
    y = np.asarray(data.values("V/measurements"), dtype=float)
    try:
        weights = np.asarray(data.get_metadata("declustering"), dtype=float)
    except (KeyError, ValueError):
        weights = np.ones(len(y))
    target = geoml.data.PointData.from_array(xy, ["X", "Y"])
    model.predict(target, n_sim=n_sim)
    full = np.asarray(target.variables["V"].get_simulations(), dtype=float)
    noise = np.asarray(target.values("V/noise_variance"), dtype=float)

    # the smooth draws: the same leaf's realizations without the residual,
    # through the likelihood's back-transform as the prediction takes them
    leaf, lik = model.leaves[0], model.likelihoods[0]
    model._refresh(model.options.jitter)
    with model._propagation(), net.once_per_pass():
        x = tf.constant(xy, tf.float64)
        leaf.predict(x, n_sim=n_sim, seed=[model.options.seed, 0])
        with net.smooth_draws():
            smooth = leaf.simulate(n_sim, [model.options.seed, 0])
    smooth = tf.transpose(smooth, [1, 0, 2])
    values, _ = lik._values_and_noise(smooth, True, None)
    smooth = np.asarray(values, dtype=float)[:, 0, :]

    edges = np.linspace(0, 190, 13)
    data_g = semivariogram(y, xy, weights, edges)
    noise_pairs = 0.5 * (noise[:, None] + noise[None, :])
    w = weights[:, None] * weights[None, :]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    upper = np.triu(np.ones_like(d, bool), 1)
    lift = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pairs = upper & (d >= lo) & (d < hi)
        lift.append(np.sum(w[pairs] * noise_pairs[pairs]) / np.sum(w[pairs]))
    lift = np.array(lift)
    full_g = semivariogram(full, xy, weights, edges) + lift
    smooth_g = semivariogram(smooth, xy, weights, edges) + lift
    print("lag   data    full+noise  smooth+noise   noise")
    for k in range(len(data_g)):
        print("%5.0f %8.0f %10.0f %12.0f %9.0f" % (
            0.5 * (edges[k] + edges[k + 1]), data_g[k], full_g[k],
            smooth_g[k], lift[k]))
    print("mean ratio to the data: full %.3f, smooth %.3f" % (
        np.mean(full_g / data_g), np.mean(smooth_g / data_g)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
