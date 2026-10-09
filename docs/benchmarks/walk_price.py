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

"""What keeps a `GPWalk` network honest under the expected kernel.

Usage: python docs/benchmarks/walk_price.py SEED [ARM ...]

The folded section of `expected_kernel.py`: training holes drawn with SEED,
the model seeded with SEED, 2000 iterations, scored on the same 200 new
holes. Each arm prints its score a new hole, its calibration (the mean
squared standardized residual: one is right, above one the intervals are
too narrow), the trained deformation's spread and displacement at the data
in the leaf's ranges, the walk's `amp` and the leaf's range. Measured
2026-10-09 on seeds 2026, 1 and 2 (`docs/expected-kernel.md`, "The walk").

The arms, by what prices the deformation:

- `default`: the library -- the field's KL, and the leaf's ranges held
  where they start (`VGPNetwork._hold_walk_ranges`); `held 0.5`, `held 2`
  hold them at another value;
- `displacement`: the term 0.9.0's development carried, `1/2 sum (moved -
  start)² / spread²` over the walked inducing points, the leaf free;
  `displacement, held` and `displacement, prior 50` beside a held or
  priced leaf;
- `none`: the field's KL alone, the leaf free;
- `amp 0.3`, `amp 1`, `amp 3`: a proper prior, no displacement -- `amp`
  scales the field, so fixing it fixes the deformation's prior scale;
- `prior 10`, `prior 50`: the leaf's range prior made stronger;
- `uncertain amp, rate 1` and `rate 3`: `log amp ~ N(log amp, s²)` against
  an exponential prior on `amp`, linearized -- one deviation shared by
  every point, along its displacement.
"""

import sys

import numpy as np
import tensorflow as tf

import geoml
import geoml.parameter as gpr

sys.path.insert(0, "docs/benchmarks")
import expected_kernel as ek  # noqa: E402

GPWalk = geoml.latent.GPWalk
HOLD = geoml.models.VGPNetwork._hold_walk_ranges
WALK = GPWalk._joint_walk
KL = GPWalk.kl_divergence
TERMS = GPWalk.expert_kl_terms


def displacement(self):
    """The development's displacement term, one expert."""
    start = self.walker.inducing_points[0]
    moved = self.inducing_points[0]
    spread = self.inducing_points_variance[0]
    return [0.5 * tf.reduce_sum((moved - start) ** 2 / spread)]


def scaled_walk(self, mean, var0, cov0, field):
    end, a, h, r, start = WALK(self, mean, var0, cov0, field)
    if "amp_sd" not in self.parameters:
        return end, a, h, r, start
    sd = self.parameters["amp_sd"].get_value()
    eye = tf.eye(self.size, dtype=tf.float64)
    extra = sd * (end - mean)[..., :, :, None] * eye
    return end, a, tf.concat([h, extra[..., None]], axis=-1), r, start


def scale_kl(self):
    total = KL(self)
    if "amp_sd" not in self.parameters:
        return total
    mu = tf.math.log(self.parameters["amp"].get_value())
    s = self.parameters["amp_sd"].get_value()
    rate = self._amp_rate
    # KL(N(mu, s²) on log amp || exponential(rate) on amp, in log amp)
    return total - tf.math.log(s) - 0.5 * np.log(2 * np.pi * np.e) \
        - np.log(rate) + rate * tf.exp(mu + 0.5 * s ** 2) - mu


GPWalk._joint_walk = scaled_walk
GPWalk.kl_divergence = scale_kl

# name: (displacement term, leaf held at, leaf range prior, amp fixed at,
#        uncertain amp's prior rate); a leaf held at None trains its ranges
ARMS = {
    "default": (False, 1.0, 2.0, None, None),
    "held 0.5": (False, 0.5, 2.0, None, None),
    "held 2": (False, 2.0, 2.0, None, None),
    "displacement": (True, None, 2.0, None, None),
    "displacement, held": (True, 1.0, 2.0, None, None),
    "displacement, prior 50": (True, None, 50.0, None, None),
    "none": (False, None, 2.0, None, None),
    "amp 0.3": (False, None, 2.0, 0.3, None),
    "amp 1": (False, None, 2.0, 1.0, None),
    "amp 3": (False, None, 2.0, 3.0, None),
    "prior 10": (False, None, 10.0, None, None),
    "prior 50": (False, None, 50.0, None, None),
    "uncertain amp, rate 1": (False, None, 2.0, None, 1.0),
    "uncertain amp, rate 3": (False, None, 2.0, None, 3.0),
}


def run(seed, name, data, new, holes):
    priced, held, leaf_prior, amp, rate = ARMS[name]
    GPWalk.expert_kl_terms = displacement if priced else TERMS
    geoml.models.VGPNetwork._hold_walk_ranges = HOLD if held is not None \
        else (lambda self: None)
    geoml.set_seed(seed)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, ek.N_IND, seed=seed))
    walk = geoml.latent.GPWalk(geoml.latent.BasicGP(root, size=2))
    if rate is not None:
        walk._add_parameter("amp_sd", gpr.PositiveParameter(0.3, 1e-3, 3.0))
        walk._amp_rate = rate
    leaf = geoml.latent.BasicGP(walk, size=1, range_prior=leaf_prior)
    if held is not None:
        leaf.parameters["ranges"].set_value(np.full([1, 1, 2], held))
    model = geoml.models.VGPNetwork(
        data, "V", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False))
    for p in model.likelihoods[0].warping.all_parameters:
        p.fix()
    if amp is not None:
        walk.parameters["amp"].set_value(amp)
        walk.parameters["amp"].fix()
    model.train_full(2000)
    lpd, z, _ = ek.log_scores(model, new)
    per_hole = np.array([lpd[holes == h].sum() for h in np.unique(holes)])
    x = tf.constant(np.asarray(data.coordinates), tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        begin = walk.walker.propagate(x)[0].numpy()
        moved, var = (t.numpy() for t in walk.propagate(x))
    leaf_r = np.asarray(leaf.parameters["ranges"].get_value()).ravel()
    sd = np.sqrt(np.mean(var / leaf_r ** 2))
    shift = np.mean(np.sqrt(np.sum((moved - begin) ** 2 / leaf_r ** 2, 1)))
    print("%-5d %-24s %8.2f %6.2f %6.3f %6.3f %6.2f %6.2f"
          % (seed, name, per_hole.mean(), np.mean(z ** 2), sd, shift,
             float(walk.parameters["amp"].get_value()), float(np.mean(leaf_r))),
          flush=True)


def main(argv):
    seed = int(argv[0])
    data, _ = ek.drillholes(ek.N_HOLES, seed)
    new, holes = ek.drillholes(ek.N_NEW, ek.NEW_SEED)
    print("%-5s %-24s %8s %6s %6s %6s %6s %6s"
          % ("seed", "arm", "score", "calib", "sd/r", "move/r", "amp",
             "leaf r"))
    for name in argv[1:] or ARMS:
        run(seed, name, data, new, holes)


if __name__ == "__main__":
    main(sys.argv[1:])
