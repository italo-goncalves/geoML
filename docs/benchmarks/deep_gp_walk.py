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

"""A deep GP by a walk along a vector field, in two dimensions: the 2022
VGP paper's demonstration, run on the current package.

Usage: python docs/benchmarks/deep_gp_walk.py

Three vertical holes, 20 m deep, 5 m apart: the outer two cut one ore
layer at 11.5-14.5 m, the middle one three layers. A stationary model can
only connect the layers as flat bands; a model whose coordinates walk
along a learned vector field (`GPWalk`) can bend the space so that the
middle hole's layers fold onto the outer ones.

Both models share a coarse base GP (range 20) for the waste-ore balance
and give the rock type a second latent variable, read from coordinates at
range 10 (stationary, a `Cubic` kernel) or range 2 walked along a GP's
field (deep). Under the expected kernel (`GPOptions(propagation="joint")`,
the default since 0.9.0) the GP reading the walk takes a scale mixture of
Gaussians -- the `Cubic` of the 2022 script is refused there -- so it reads
with the `Gaussian` kernel, isotropic as `GPWalk` asks.

The figure (`figures/deep_gp_walk.png`): the probability of ore under each
model, the learned vector field (colour its norm, width its standard
deviation) and the grid's coordinates after the walk, rescaled to the
section, coloured by the deep model's probability.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

import geoml
import geoml.kernels as kr
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.transform as tr

HERE = os.path.dirname(os.path.abspath(__file__))

z1 = np.linspace(0, -20, 101)
rock_1 = np.where((z1 < -11.5) & (z1 > -14.5), "ore", "waste")
rock_2 = np.where((z1 < -2.5) & (z1 > -5.5), "ore", "waste")
rock_2 = np.where((z1 < -8.5) & (z1 > -11.5), "ore", rock_2)
rock_2 = np.where((z1 < -14.5) & (z1 > -17.5), "ore", rock_2)
rock = np.concatenate([rock_1, rock_2, rock_1[::-1]])

point_data = geoml.data.PointData.from_array(np.stack([
    np.concatenate([-5 * np.ones_like(z1), np.zeros_like(z1),
                    5 * np.ones_like(z1)]),
    np.concatenate([z1, z1, z1])], axis=1))
point_data.add_categorical_variable("Rock_stat", ["waste", "ore"], rock)
point_data.add_categorical_variable("Rock_deep", ["waste", "ore"], rock)

ip = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[17, 21])
grid = geoml.data.Grid2D(start=[-8, -20], end=[8, 0], n=[321, 201])

# stationary model
geoml.set_seed(2022)
stat_in = gl.BasicInput(ip, tr.Concatenate(tr.Isotropic(20),
                                           tr.Isotropic(10)))
base_gp = gl.BasicGP(gl.SelectInput(stat_in, [0, 1]), 1, fix_range=True)
stat_gp = gl.BasicGP(gl.SelectInput(stat_in, [2, 3]), kernel=kr.Cubic(),
                     fix_range=True)
stat_out = gl.Concatenate(base_gp, stat_gp)

stat_model = geoml.models.VGPNetwork(
    data=point_data,
    variables="Rock_stat",
    likelihoods=lk.HierarchicalGaussianIndicator(2),
    latent_network=stat_out,
    options=geoml.models.GPOptions(verbose=False))
start = time.perf_counter()
stat_model.set_learning_rate(5e-2)
stat_model.train_full(200)
stat_model.set_learning_rate(1e-2)
stat_model.train_full(300)
print("stationary: bound %.2f, trained in %.0f s"
      % (stat_model.training_log[-1], time.perf_counter() - start))
stat_model.predict(grid)

# deep model
geoml.set_seed(2022)
deep_in = gl.BasicInput(ip, tr.Concatenate(tr.Isotropic(20),
                                           tr.Isotropic(2)), center=True)
base_gp = gl.BasicGP(gl.SelectInput(deep_in, [0, 1]), 1, fix_range=True)
field = gl.BasicGP(gl.SelectInput(deep_in, [2, 3]), size=2, fix_range=True,
                   name="field")
walk = gl.GPWalk(field, step=0.01, n_steps=10, name="walk")
reader = gl.BasicGP(walk, kernel=kr.Gaussian(), isotropic=True)
deep_out = gl.Concatenate(base_gp, reader)

deep_model = geoml.models.VGPNetwork(
    data=point_data,
    variables="Rock_deep",
    likelihoods=lk.HierarchicalGaussianIndicator(2),
    latent_network=deep_out,
    options=geoml.models.GPOptions(jitter=1e-6, verbose=False))
start = time.perf_counter()
deep_model.set_learning_rate(5e-2)
deep_model.train_full(100)
deep_model.set_learning_rate(1e-2)
deep_model.train_full(100)
print("deep: bound %.2f, trained in %.0f s"
      % (deep_model.training_log[-1], time.perf_counter() - start))
start = time.perf_counter()
deep_model.predict(grid)
print("deep: predicted %d locations in %.0f s"
      % (grid.n_data, time.perf_counter() - start))


def node_moments(model, node, coordinates):
    """A node's mean and variance at `coordinates`, one column a part."""
    target = geoml.data.PointData.from_array(np.asarray(coordinates))
    latent = model.predict_node(node, target, n_sim=1)
    parts = [latent.components[label] for label in latent.labels]
    return (np.stack([np.asarray(p.latent_mean.values) for p in parts], 1),
            np.stack([np.asarray(p.latent_variance.values) for p in parts],
                     1))


shape = grid.grid_size[::-1]
im_stat = grid.variables["Rock_stat"].components["ore"].probability \
    .as_image()
im_deep = grid.variables["Rock_deep"].components["ore"].probability \
    .as_image()

mu, var = node_moments(deep_model, field, grid.coordinates)
x = np.reshape(grid.coordinates[:, 0], shape)
y = np.reshape(grid.coordinates[:, 1], shape)
mu_x, mu_y = np.reshape(mu[:, 0], shape), np.reshape(mu[:, 1], shape)
std = np.sqrt(np.reshape(var[:, 0] + var[:, 1], shape))
speed = np.sqrt(mu_x ** 2 + mu_y ** 2)

walked_grid, _ = node_moments(deep_model, walk, grid.coordinates)
walked_data, _ = node_moments(deep_model, walk, point_data.coordinates)
low = np.min(np.concatenate([walked_grid, walked_data]), axis=0)
high = np.max(np.concatenate([walked_grid, walked_data]), axis=0)


def to_section(points):
    """Walked coordinates rescaled onto the section's box."""
    out = (points - low) / (high - low)
    return np.stack([out[:, 0] * 16 - 8, out[:, 1] * 20 - 20], axis=1)


walked_grid, walked_data = to_section(walked_grid), to_section(walked_data)
is_ore = (rock == "ore").astype(float)

extent = grid.bounding_box.as_array().T.ravel()
fig, ax = plt.subplots(2, 3, figsize=[10, 11], gridspec_kw=dict(
    width_ratios=[1, 1, 0.05], wspace=0.05, hspace=0.1))
for axis, image, title in ((ax[0, 0], im_stat, "Shallow GP (stationary)"),
                           (ax[0, 1], im_deep, "Deep GP (GPWalk)")):
    im = axis.imshow(image, origin="lower", extent=extent, vmin=0, vmax=1,
                     cmap=cm.berlin)
    axis.scatter(point_data.coordinates[:, 0], point_data.coordinates[:, 1],
                 c=is_ore, cmap=cm.buda)
    axis.set_title(title)
    axis.set_xlim(-8, 8)
    axis.set_ylim(-20, 0)

st = ax[1, 0].streamplot(x, y, mu_x, mu_y, color=speed,
                         linewidth=2 * std / std.max(), density=2,
                         cmap=cm.lajolla_r)
ax[1, 0].set_aspect("equal")
ax[1, 0].set_title("Learned vector field")
ax[1, 0].set_xlim(-8, 8)
ax[1, 0].set_ylim(-20, 0)

ax[1, 1].scatter(walked_grid[:, 0], walked_grid[:, 1], marker="o", s=0.05,
                 cmap=cm.berlin, vmin=0, vmax=1,
                 c=grid.variables["Rock_deep"].components["ore"]
                 .probability.values)
ax[1, 1].scatter(walked_data[:, 0], walked_data[:, 1], c=is_ore,
                 cmap=cm.buda)
ax[1, 1].set_aspect("equal")
ax[1, 1].set_title("Displaced coordinates")
ax[1, 1].set_xlim(-8, 8)
ax[1, 1].set_ylim(-20, 0)

ax[0, 1].tick_params(labelbottom=False, labelleft=False)
ax[0, 0].tick_params(labelbottom=False)
ax[1, 1].tick_params(labelleft=False)
plt.colorbar(st.lines, cax=ax[1, 2], label="Vector field norm")
plt.colorbar(im, cax=ax[0, 2], label="Probability of ore")

out = os.path.join(HERE, "figures", "deep_gp_walk.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
