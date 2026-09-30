"""Refining on the mean against refining on every realization, 2026-09-29.

Usage: python docs/benchmarks/refine_on_the_mean.py

The case of `test_blockset.py`'s `test_ground_without_data_is_not_refined`:
the data fill the west half of a 160 x 80 x 40 m box, a pod whose shell
crosses the cut-off among them; the inducing points cover both halves, so
the realizations east of the data are as rough as the prior. The same model
(same seed) refines the same coarse block set twice, once with the criterion
as it stood before 0.8.5 -- the share of realizations whose sub-blocks
straddle the cut-off, over `tolerance=0.05` -- and once with the prediction's
sub-blocks. The reference is the same model predicted on a grid of points at
the finest block size, 5 m. Reported per arm: the blocks made, in each half,
and the volume on the wrong side of the cut-off -- the 5 m cells whose block
in the refined set says above where the grid says below, or the reverse.
(The pod meets the top and bottom of the box, so its contour is an open
sheet with no volume to compare.)
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))

import numpy as np                                       # noqa: E402
import tensorflow as tf                                  # noqa: E402

import geoml                                             # noqa: E402
import geoml.likelihood as lk                            # noqa: E402

CUTOFF = 1.0
CELL = 5.0


def old_divided(x, cutoffs, n_splits=None):
    """`likelihood._divided` as it stood in 0.8.4 -- every realization
    judged on its own, then the share of them that found the block divided --
    with 0.8.4's default `tolerance=0.05` applied here, `needs_splitting`
    reading a flag since."""
    cuts = lk._cutoff_matrix(cutoffs, x.dtype)
    if n_splits is None:
        return tf.zeros(
            tf.concat([tf.shape(x)[:2], [tf.shape(cuts)[-1]]], axis=0),
            dtype=x.dtype)
    n = tf.cast(tf.shape(x)[0] / n_splits, dtype=tf.int32)
    grouped = tf.reshape(
        x, tf.concat([[n_splits, n], tf.shape(x)[1:]], axis=0))
    below = tf.cast(grouped[..., None] <= cuts[:, None, :], x.dtype)
    share = tf.reduce_mean(below, axis=1)
    straddles = tf.cast((share > 0.0) & (share < 1.0), x.dtype)
    return tf.cast(tf.reduce_mean(straddles, axis=2) > 0.05, x.dtype)


def model():
    geoml.set_seed(1234)
    tf.random.set_seed(1234)
    rng = np.random.default_rng(1234)
    xyz = rng.uniform([0, 0, 0], [80, 80, 40], size=[300, 3])
    radius = np.linalg.norm(xyz - np.array([40.0, 40.0, 20.0]), axis=1)
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable(
        "au", 4.0 * np.exp(-(radius / 20.0) ** 2) + 0.02)
    point.variables["au"].set_cutoffs([CUTOFF])
    east = np.stack(np.meshgrid(np.arange(90, 160, 20.0),
                                np.arange(10, 80, 20.0), [10.0, 30.0],
                                indexing="ij"), axis=-1).reshape(-1, 3)
    ip = geoml.data.inducing.combine(
        geoml.data.inducing.from_kmeans(point, 60, seed=0), east)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(15.0))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    m = geoml.models.VGPNetwork(
        point, "au", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=10))
    m.train_full(max_iter=100)
    return m


def blocks():
    return geoml.data.BlockSet3D([10, 10, 10], [8, 4, 2], [20.0] * 3,
                                 discretization=(2, 2, 2), max_levels=2)


def main():
    grid = geoml.data.Grid3D(start=[CELL / 2] * 3, n=[32, 16, 8],
                             step=[CELL] * 3)
    model().predict(grid, n_sim=20)
    truth = grid.values("au/prediction") > CUTOFF

    rows = []
    for arm, divided in (("every realization", old_divided),
                         ("the mean", lk._divided)):
        current = lk._divided
        lk._divided = divided
        try:
            refined = geoml.models.refine(model(), blocks(), n_sim=20)
        finally:
            lk._divided = current
        east = np.asarray(refined.coordinates)[:, 0] > 80
        block = refined.index_data(grid)
        called = refined.values("au/prediction")[block] > CUTOFF
        wrong = int(np.count_nonzero(called != truth)) * CELL ** 3
        rows.append((arm, refined.n_data, int((~east).sum()),
                     int(east.sum()), wrong))

    print("above the cut-off on the 5 m grid: %.0f m3"
          % (truth.sum() * CELL ** 3))
    print("%-18s %7s %6s %6s %12s" % ("criterion", "blocks", "west", "east",
                                      "wrong side"))
    for arm, total, west, east, wrong in rows:
        print("%-18s %7d %6d %6d %9.0f m3" % (arm, total, west, east, wrong))


if __name__ == "__main__":
    main()
