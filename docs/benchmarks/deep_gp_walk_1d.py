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

"""A deep GP by a walk along a vector field, in one dimension: the 2022
VGP paper's demonstration of non-stationarity, run on the current package.

Usage: python docs/benchmarks/deep_gp_walk_1d.py [CSV]

CSV is the digitized curve of Hedge et al. (two columns, `;`-separated,
decimal comma), by default where the paper's figures were made; it is not
in the repository. The curve is flat on one side and steep on the other,
which no stationary covariance fits. A GP (the displacement field) reads
the coordinate, `GPWalk` moves the coordinate along it in ten steps, and a
second GP reads the moved coordinate: where the walk spreads points apart
the reader sees the curve change slowly, where it packs them together,
quickly.

The figure (`figures/deep_gp_walk_1d.png`), top to bottom: the prediction
in the input space with its 95% interval; the inducing points' trajectories
along the walk (mean and two standard deviations, every fifth point) and
where they start and end; the prediction in the warped space; the
displacement field with two standard deviations; and the covariance of the
field and of the output -- the reader's expected kernel over the walked
coordinates, taken as independent -- over the input.

Under the expected kernel (`GPOptions(propagation="joint")`, the default
since 0.9.0) the walk is one random field and `GPWalk.compute_path`, which
steps by the marginal rule of the versions before, is not the walk the model
trains; the trajectories are read from the walk itself, stopped after each
step.
"""
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from cmcrameri import cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import geoml
import geoml.latent as gl
import geoml.latent.network as net
import geoml.likelihood as lk
import geoml.transform as tr
import geoml.warping as wp

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "/mnt/c/Users/Ítalo/OneDrive/Python/Lab/Modelagem geológica",
    "2022 Artigo VGP", "Hedge et al.csv")

df = pd.read_csv(CSV, header=None, sep=";", decimal=",")
x = df.values[:, 0] - 2.5
y = df.values[:, 1] / 3

points = geoml.data.PointData(pd.DataFrame({"x": x, "y": y}), coordinates="x")
points.add_continuous_variable("y", y)

geoml.set_seed(2022)
ip = geoml.data.Grid1D(start=-10, end=10, n=101)
tree_input = gl.BasicInput(ip, tr.Isotropic(2), fix_transform=True)
layer_1 = gl.BasicGP(tree_input, name="field")
layer_2 = gl.GPWalk(layer_1, step=0.01, n_steps=10, name="walk")
layer_3 = gl.BasicGP(layer_2, name="reader")

gp = geoml.models.VGPNetwork(
    data=points,
    variables="y",
    likelihoods=lk.Gaussian(wp.ZScore(1)),
    latent_network=layer_3,
    options=geoml.models.GPOptions(jitter=1e-6, training_samples=50,
                                   verbose=False))
start = time.perf_counter()
gp.set_learning_rate(2e-2)
gp.train_full(1000)
print("bound after 1000 iterations: %.2f, trained in %.0f s"
      % (gp.training_log[-1], time.perf_counter() - start))

grid = geoml.data.Grid1D(start=-10, end=10, n=5001)
gp.predict(grid, n_sim=100)
grid.variables["y"].reset_quantiles([0.025, 0.5, 0.975])

rng_1 = float(np.squeeze(tree_input.transform.parameters["range"]
                         .get_value()))
amp = float(np.squeeze(layer_2.parameters["amp"].get_value()))
print("walk: amp %.3f, reach %.3f ranges of the input; field range %.3f, "
      "reader range %.3f" % (
          amp, layer_2.step * layer_2.n_steps * amp,
          float(np.squeeze(layer_1.parameters["ranges"].get_value())),
          float(np.squeeze(layer_3.parameters["ranges"].get_value()))))


def walked(z, n_steps=None):
    """The walk's mean and variance at `z`, stopped after `n_steps`."""
    full = layer_2.n_steps
    if n_steps is not None:
        layer_2.n_steps = n_steps
    try:
        mean, var = layer_2.propagate(tf.constant(z, tf.float64))
    finally:
        layer_2.n_steps = full
    return np.asarray(mean), np.asarray(var)


coarse = np.linspace(-10, 10, 1001)[:, None]
with gp._propagation():
    gp._refresh(gp.options.jitter)
    # the field's covariance over the input, and the reader's over the
    # walked coordinate
    start_c = tree_input.propagate(tf.constant(coarse))[0]
    cov_1 = np.asarray(layer_1.covariance_matrix(start_c, start_c))
    m_c, v_c = walked(coarse)
    cov_2 = np.asarray(net._expected_kernel(
        layer_3.kernel, layer_3.parameters["ranges"].get_value(),
        tf.constant(m_c), tf.constant(v_c), tf.constant(m_c),
        tf.constant(v_c)))

    # the inducing points' trajectories, one step at a time
    start_ip = np.asarray(tree_input.propagate(tf.constant(ip.coordinates))[0])
    path_mu, path_var = [start_ip], [np.zeros_like(start_ip)]
    for k in range(1, layer_2.n_steps + 1):
        m, v = walked(ip.coordinates, k)
        path_mu.append(m)
        path_var.append(v)
    path_mu = np.stack(path_mu)[..., 0] * rng_1        # [steps + 1, points]
    path_std = np.sqrt(np.stack(path_var)[..., 0]) * rng_1

    # the displacement field, and the prediction in the warped space
    field_mu, field_var = layer_1.propagate(tf.constant(grid.coordinates))
    field_mu = np.asarray(field_mu)[:, 0]
    field_std = np.sqrt(np.asarray(field_var)[:, 0])
    x_2 = walked(grid.coordinates)[0][:, 0] * rng_1
    mu_f, var_f = layer_3.propagate(tf.constant(grid.coordinates))
    x_w = walked(points.coordinates)[0][:, 0] * rng_1

warping = gp.likelihoods[0].warping
warp_mean = float(np.squeeze(warping.parameters["mean"].get_value()))
warp_std = float(np.squeeze(warping.parameters["std"].get_value()))
mu_f = np.asarray(mu_f)[:, 0] * warp_std + warp_mean
std_f = np.sqrt(np.asarray(var_f)[:, 0]) * warp_std

matplotlib.rc("font", size=8)
fig = plt.figure(constrained_layout=True, figsize=(6, 7))
gs = GridSpec(3, 1, figure=fig, height_ratios=(0.45, 0.15, 0.4))
gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], hspace=0.0)
ax0 = fig.add_subplot(gs1[0])
ax1 = fig.add_subplot(gs1[1])
ax2 = fig.add_subplot(gs1[2])

steps = np.linspace(1, 0, path_mu.shape[0])
for i in range(0, path_mu.shape[1], 5):
    ax1.fill_betweenx(steps, path_mu[:, i] - 2 * path_std[:, i],
                      path_mu[:, i] + 2 * path_std[:, i],
                      color="royalblue", alpha=0.1)
    ax1.plot(path_mu[:, i], steps, "-k", alpha=0.5)
    ax0.vlines(path_mu[0, i], -2, 2, colors="k", alpha=0.5)
    ax2.vlines(path_mu[-1, i], -2, 2, colors="k", alpha=0.5)
ax1.set_xlim(-10, 10)
ax1.set_ylim(0, 1)
ax1.yaxis.set_ticks([])
ax1.xaxis.set_visible(False)
ax1.set_ylabel("Warping\ntrajectories")

smoothing = 5
quantiles = grid.variables["y"].quantiles
ax0.scatter(x, y, facecolor="w", edgecolor="k", zorder=3)
ax0.fill_between(grid.coordinates[:, 0],
                 quantiles[0.025].as_series(sigma=smoothing),
                 quantiles[0.975].as_series(sigma=smoothing),
                 color="royalblue", alpha=0.5)
ax0.plot(grid.coordinates[:, 0], quantiles[0.5].as_series(sigma=smoothing),
         "-", color="royalblue")
ax0.set_xlim(-10, 10)
ax0.set_ylim(-2, 2)
ax0.set_ylabel("Input\nspace")
ax0.xaxis.set_visible(False)
ax0.yaxis.set_ticks([])

ax2.scatter(x_w, y, facecolor="w", edgecolor="k", zorder=3)
ax2.fill_between(x_2, mu_f - 2 * std_f, mu_f + 2 * std_f,
                 color="royalblue", alpha=0.5)
ax2.plot(x_2, mu_f, "-", color="royalblue")
ax2.set_xlim(-10, 10)
ax2.set_ylim(-2, 2)
ax2.set_ylabel("Warped\nspace")
ax2.yaxis.set_ticks([])
ax2.set_xlabel("Input variable")

ax3 = fig.add_subplot(gs[1])
ax3.hlines(0, -10, 10, linestyle="dashed", color="k", alpha=0.5)
ax3.fill_between(grid.coordinates[:, 0], field_mu - 2 * field_std,
                 field_mu + 2 * field_std, color="royalblue", alpha=0.5)
ax3.plot(grid.coordinates[:, 0], field_mu, "-", color="royalblue")
ax3.set_xlim(-10, 10)
ax3.set_ylim(-3, 3)
ax3.yaxis.set_ticks([])
ax3.set_ylabel("Displacement\nfield")
ax3.set_xlabel("Input variable")

gs3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2])
for column, (cov, title) in enumerate((
        (cov_1, "Displacement field covariance"),
        (cov_2, "Output variable covariance"))):
    ax = fig.add_subplot(gs3[column])
    ax.imshow(cov, extent=[-10, 10, 10, -10], cmap=cm.nuuk)
    ax.set_title(title)

out = os.path.join(HERE, "figures", "deep_gp_walk_1d.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
