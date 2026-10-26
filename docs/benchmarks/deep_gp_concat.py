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

"""A deep GP by input concatenation, in one dimension: the 2021 VGP paper's
demonstration, run on the current package.

Usage: python docs/benchmarks/deep_gp_concat.py

The data change character at zero -- `cos(x)` on the left, `cos(x²)` on the
right, its frequency growing with `x` -- which no stationary covariance
fits. GP A reads `x`; GP B reads `(x, GP A)`, the input concatenated with
the hidden layer, so GP A can stretch the space where the data oscillate
faster while `x` keeps the field local. Both GPs have unit ranges, the
input's transform fixed, so every bit of non-stationarity has to come from
GP A.

The figure (`figures/deep_gp_concat.png`): the prediction with its 95%
interval; the latent space, GP B's mean over a grid of `(x, GP A)` with
GP A's path through it and its spread at the inducing points; and both
GPs' covariance matrices among the inducing points -- GP B's is the
expected kernel under GP A's uncertainty, so it is no longer a function of
`x - x'` alone.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from cmcrameri import cm
from matplotlib.gridspec import GridSpec

import geoml
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr

HERE = os.path.dirname(os.path.abspath(__file__))

geoml.set_seed(2021)
x = np.linspace(-5, 5, 51)
y = np.where(x < 0, np.cos(x), np.cos(x ** 2))

points = geoml.data.PointData(pd.DataFrame({"x": x, "y": y}), coordinates="x")
points.add_continuous_variable("y", y)

tree_input = gl.BasicInput(
    geoml.data.Grid1D(start=-10, end=10, n=101),
    tr.Isotropic(6.5), fix_transform=True)
layer_1 = gl.BasicGP(tree_input, fix_range=True, name="GP_A")
concat_1 = gl.Concatenate(tree_input, layer_1)
layer_2 = gl.BasicGP(concat_1, fix_range=True, name="GP_B")

gp = geoml.models.VGPNetwork(
    data=points,
    variables="y",
    likelihoods=lk.Gaussian(),
    latent_network=layer_2,
    options=geoml.models.GPOptions(jitter=1e-6, training_samples=50,
                                   verbose=False))
gp.set_learning_rate(5e-2)
gp.train_full(500)
gp.set_learning_rate(1e-2)
gp.train_full(1000)
print("bound after 1500 iterations: %.2f" % gp.training_log[-1])

grid = geoml.data.Grid1D(start=-10, end=10, n=2001)
gp.predict(grid, n_sim=100)
grid.variables["y"].reset_quantiles([0.025, 0.5, 0.975])

# GP A along the prediction grid and at its own inducing points
hidden = gp.predict_node(layer_1, geoml.data.Grid1D(start=-10, end=10,
                                                    n=2001))
path = np.asarray(hidden.components[hidden.labels[0]].latent_mean.values)
at_points = gp.predict_node(layer_1, geoml.data.Grid1D(start=-10, end=10,
                                                       n=101))
part = at_points.components[at_points.labels[0]]
ip_mean = np.asarray(part.latent_mean.values)
ip_sd = np.sqrt(np.asarray(part.latent_variance.values))

# GP B's mean over the latent space: certain inputs (x through the input's
# fixed transform, GP A), read by GP B under the rule it was trained under
a_lim = (-3.5, 4.5)
grid_exp = geoml.data.Grid2D(start=[-10, a_lim[0]], end=[10, a_lim[1]],
                             n=[501, 101])
z = np.asarray(grid_exp.coordinates).copy()
z[:, :1] = np.asarray(tree_input.transform(tf.constant(z[:, :1])))
with gp._propagation():
    gp._refresh(gp.options.jitter)
    mean_exp, _ = layer_2.interpolate(tf.constant(z, tf.float64))
    cov_a = np.asarray(layer_1.cov[0])
    cov_b = np.asarray(layer_2.cov[0])
mean_exp = np.squeeze(np.asarray(mean_exp))
image = mean_exp.reshape([101, 501])
lat_max = np.max(np.abs(mean_exp))

matplotlib.rc("font", size=8)
fig = plt.figure(constrained_layout=True, figsize=(6, 7))
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.5, 1, 0.1, 1.5])

ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(x, y, facecolor="w", edgecolor="k", zorder=3)
ax1.fill_between(grid.coordinates[:, 0],
                 grid.variables["y"].quantiles[0.025].values,
                 grid.variables["y"].quantiles[0.975].values,
                 color="royalblue", alpha=0.5)
ax1.plot(grid.coordinates[:, 0], grid.variables["y"].quantiles[0.5].values,
         "-", color="royalblue")
ax1.plot(np.linspace(-10, 10, 101), np.zeros([101]) - 2, "|k")
ax1.set_title("Prediction")
ax1.set_xlim(-10, 10)
ax1.set_ylim(-2, 2)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"Output variable ($z$)")

ax2 = fig.add_subplot(gs[1, :])
im = ax2.imshow(image, origin="lower", extent=[-10, 10, *a_lim],
                cmap=cm.cork, vmin=-lat_max, vmax=lat_max)
ax2.plot(grid.coordinates[:, 0], path, "-k")
ax2.errorbar(np.linspace(-10, 10, 101), ip_mean, ip_sd, ecolor="k",
             fmt="none", alpha=0.25)
ax2.set_xlim(-10, 10)
ax2.set_ylim(*a_lim)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel("GP A")
ax2.set_title("Latent space")
ax2.set_aspect("auto")

ax3 = fig.add_subplot(gs[2, :])
plt.colorbar(im, cax=ax3, orientation="horizontal")
ax3.set_xlabel("GP B")

for column, (cov, title) in enumerate(((cov_a, "GP A"), (cov_b, "GP B"))):
    ax = fig.add_subplot(gs[3, column])
    ax.imshow(cov, extent=[-10, 10, 10, -10], cmap=cm.nuuk)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$x'$")
    ax.set_title("Covariance matrix - %s" % title)

out = os.path.join(HERE, "figures", "deep_gp_concat.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
