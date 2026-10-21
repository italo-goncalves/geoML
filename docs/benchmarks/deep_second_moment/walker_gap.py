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

"""How much of chapter 13's reported variance its inducing points leave out.

Chapter 13's Walker model -- a GP on a lattice of inducing points cut into
experts, the chapter's warping, 500 iterations -- for a kernel and a
lattice spacing. At the 470 samples it reports, as shares of the reported
latent variance (medians and quartiles): the explained part `|k R|²` the
realizations carry with `full_variance` off, and the gap `1 - k K^-1 kᵀ`
they add when it is on; then the realizations with the full variance on:
their variance against the reported one, and how much of it is one offset
shared by every location (the spread across realizations of each one's mean
over the samples, against what the reported covariance implies were the
locations independent).

Usage: python walker_gap.py KERNEL SPACING [grid|lattice|data]
`grid` is the chapter's `grid_experts`; `lattice` the same lattice cut into
16 experts by `experts`; `data` that lattice with the samples added before
the cut.
"""
import sys
import time

import numpy as np
import tensorflow as tf

import geoml
import geoml.latent.network as net


def build(kernel, spacing, mode="grid"):
    geoml.set_seed(1234)
    walker, grid = geoml.datasets.walker()
    inducing = geoml.data.inducing
    if mode == "grid":
        experts = inducing.grid_experts(grid, float(spacing), block=8)
    elif mode == "lattice":
        experts = inducing.experts(
            inducing.from_grid(grid, float(spacing)), 16, seed=0)
    elif mode == "data":
        experts = inducing.experts(inducing.combine(
            inducing.from_grid(grid, float(spacing)), walker), 16, seed=0)
    else:
        raise ValueError(mode)
    root = geoml.latent.BasicInput(experts,
                                   transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1,
                              kernel=getattr(geoml.kernels, kernel)())
    warping = geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1, shift=1.0), geoml.warping.ZScore(1))
    model = geoml.models.VGPNetwork(
        walker, "V", geoml.likelihood.Gaussian(warping), gp,
        options=geoml.models.GPOptions(verbose=False))
    return model, gp, walker, root


def quartiles(v):
    return "%.3f [%.3f, %.3f]" % tuple(np.percentile(v, [50, 25, 75]))


def main(kernel, spacing, mode="grid"):
    model, gp, walker, root = build(kernel, spacing, mode)
    start = time.perf_counter()
    model.train_full(500)
    print("%s, lattice %s (%s): %d inducing points in %d experts, trained "
          "in %.0f s" % (kernel, spacing, mode, sum(int(n) for n in root.n_ip),
                      len(root.n_ip), time.perf_counter() - start))
    print("  range %.1f, bound %.1f" % (float(np.asarray(
        root.transform.parameters["range"].get_value()).ravel()[0]),
        model.training_log[-1]))

    x = tf.constant(np.asarray(walker.coordinates, dtype=float), tf.float64)
    model._refresh(model.options.jitter)
    with model._propagation(), net.once_per_pass():
        mu, var = gp.propagate(x)
        cov_cross, _, weights = gp._swept()
        seen = tf.stack([tf.reduce_sum(tf.einsum("nm,smj->snj", k, r) ** 2,
                                       -1)
                         for k, r in zip(cov_cross, gp.chol_r)], 0)
        explained = tf.reduce_sum(seen * weights, 0)
        sims = gp.simulate(400, [model.options.seed, 0])
    var = np.asarray(var, dtype=float)[:, 0]
    explained = np.asarray(explained, dtype=float)[0]
    sims = np.asarray(sims, dtype=float).reshape(len(var), -1)  # [n, S]
    print("  explained / reported: %s" % quartiles(explained / var))
    print("  gap / reported:       %s" % quartiles(1.0 - explained / var))
    ratio = np.var(sims, axis=1) / var
    print("  realizations (full variance on) / reported: %s"
          % quartiles(ratio))
    # each realization's mean over the samples, less the mean of the means:
    # a shared offset shows as a spread far above the independent case
    offsets = sims.mean(axis=0) - sims.mean()
    independent = np.sqrt(var.sum()) / len(var)
    print("  spread of a realization's mean over the samples: %.3f, "
          "%.1f times the independent case (%.3f)"
          % (offsets.std(), offsets.std() / independent, independent))


if __name__ == "__main__":
    main(*sys.argv[1:4])
