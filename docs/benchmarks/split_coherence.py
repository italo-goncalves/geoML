"""What tells a block worth cutting from one only noise straddles, 2026-10-29.

Usage: python docs/benchmarks/split_coherence.py [RUN12.npz]

The rule from 0.8.5 to 0.9.11 cut a block wherever the prediction at its
sub-blocks straddled a cut-off. On a GeoScape model of the Tom East deposit
(Run 12) it cut 93% of the blocks, most of them far from any sample: the
background sat near the cut-offs, and the mean of 25 rough realizations
wobbles across a cut-off near it by sampling noise alone (with 200 the far
marks fell from 81% to 36% at one cut-off, from 51% to 3% at another).

Here the candidates are scored against blocks whose worth is known. Four
known fields -- the pod of `refine_on_the_mean.py`, the ore body and the
rock plane of `test_blockset.py` (the plane trained to 300 iterations),
and a rough random field sampled densely in one half and sparsely in the
other -- each at 8 and 20 realizations, on the coarse block set and on
blocks half its size. A block's worth is what cutting it into its
sub-blocks buys: the share of its volume on the wrong side of the known
field when it is called as a whole, less the share when each sub-block is
called on its own, read on a 3 x 3 x 3 lattice inside each sub-block. The
negatives are Run 12's straddling blocks more than 200 m from a sample,
whose sub-block realizations (400 near, 400 far, 25 and 200 realizations,
predicted as points from the saved network) are not in the repository;
pass the file to include them.

Measured (2026-10-29): coherence -- the share of the variation inside a
block its realizations hold in common, each realization's shift of the
whole block set aside (`likelihood._coherence`) -- ran 0.55 to 0.99 on the
312 straddling blocks worth cutting and 0.24 to 0.68 on the soft rock
contact (30 iterations, probabilities moving at most 0.26 across a block),
against at most 0.127 on Run 12's far blocks at 25 realizations and 0.020
at 200, the Beta null exactly. At max(0.2, the noise floor) it kept every
block today's rule marked on the known fields, and their whole benefit,
and none of Run 12's far blocks. The contrast of the sub-blocks'
probabilities lost real benefit at a half and passed noise at 0.3; the
F-test alone marked far blocks at 200 realizations, a tiny real gradient
being detectable with enough of them; the within-block share of the
variance told nothing apart.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..", "geoml",
                                                "test")))
sys.path.insert(0, here)

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402
import tensorflow as tf                                  # noqa: E402
from scipy import spatial, stats                         # noqa: E402

import geoml                                             # noqa: E402

THRESHOLDS = {"coherence": [0.1, 0.2, 0.3, 0.5],
              "contrast": [0.3, 0.5],
              "log10p": [3, 5]}


def grf_field(xyz, seed=7, n=300, length=25.0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 3)) / length
    b = rng.uniform(0, 2 * np.pi, n)
    a = rng.normal(size=n) * np.sqrt(2.0 / n)
    return np.cos(np.asarray(xyz) @ w.T + b) @ a


def case(name):
    """A trained model, its coarse block set, the cut-off, the known field
    as a test of location and the variable's path."""
    if name == "pod":
        import refine_on_the_mean as bench
        centre, width = np.array([40.0, 40.0, 20.0]), 20.0
        return (bench.model(), bench.blocks(), 1.0,
                lambda p: 4.0 * np.exp(-(np.linalg.norm(p - centre, axis=1)
                                         / width) ** 2) + 0.02 > 1.0, "au")
    if name == "ore":
        import test_blockset as t
        centre, width = np.array([80.0, 80.0, 80.0]), 28.0
        return (t._ore_model(), t._ore_blocks(), 1.0,
                lambda p: 4.0 * np.exp(-(np.linalg.norm(p - centre, axis=1)
                                         / width) ** 2) + 0.02 > 1.0, "au")
    if name == "rock":
        import test_blockset as t
        model = t._rock_model()
        model.train_full(max_iter=300)
        return (model, t._ore_blocks(), 0.0,
                lambda p: p[:, 0] + 0.4 * p[:, 1] < 110, "rock")
    geoml.set_seed(1234)
    rng = np.random.default_rng(1234)
    xyz = np.concatenate([rng.uniform([0, 0, 0], [80, 80, 40], (500, 3)),
                          rng.uniform([80, 0, 0], [160, 80, 40], (25, 3))])
    point = geoml.data.PointData.from_array(xyz)
    point.add_continuous_variable("g", grf_field(xyz))
    point.variables["g"].set_cutoffs([0.3])
    east = np.stack(np.meshgrid(np.arange(90, 160, 20.0),
                                np.arange(10, 80, 20.0), [10.0, 30.0],
                                indexing="ij"), axis=-1).reshape(-1, 3)
    ip = geoml.data.inducing.combine(
        geoml.data.inducing.from_kmeans(point, 150, seed=0), east)
    root = geoml.latent.BasicInput(
        [ip], transform=geoml.transform.Isotropic(20.0))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Matern52())
    model = geoml.models.VGPNetwork(
        point, "g", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False, training_samples=10))
    model.train_full(max_iter=300)
    blocks = geoml.data.BlockSet3D([10, 10, 10], [8, 4, 2], [20.0] * 3,
                                   discretization=(2, 2, 2), max_levels=2)
    return model, blocks, 0.3, lambda p: grf_field(p) > 0.3, "g"


def halved(blocks):
    """The same box in blocks half the size: a later pass's decisions."""
    step = np.asarray(blocks.block_size[0]) / 2
    lo = np.asarray(blocks.bounding_box.min).ravel()
    shape = np.round((np.asarray(blocks.bounding_box.max).ravel() - lo)
                     / step).astype(int)
    return geoml.data.BlockSet3D(lo + step / 2, shape, step,
                                 discretization=(2, 2, 2), max_levels=2)


def labelled(model, blocks, cut, truth, path, n_sim):
    """The sub-block realizations of every block and what cutting it buys,
    as a share of its volume."""
    n, k = blocks.n_data, blocks.rows_per_location
    coords, _ = blocks.get_batched_coordinates(np.arange(n))
    labels = list(blocks.coordinate_labels)
    points = geoml.data.PointData(pd.DataFrame(coords, columns=labels),
                                  labels)
    model.predict(points, n_sim=n_sim)
    if path == "rock":
        variable = points.variables["rock"]
        granite = variable.components["granite"]
        draws = np.stack([np.asarray(c.simulations)
                          for c in variable.components.values()], axis=1)
        v = model.likelihoods[0]._category_draws(
            tf.constant(draws)).numpy()[:, 0]
        prob = granite.probability.values.to_numpy().reshape(n, k)
        sub, whole = prob > 0.5, prob.mean(axis=1) > 0.5
        straddle = (prob.min(1) <= 0.5) & (prob.max(1) > 0.5)
    else:
        v = np.asarray(points.values(path + "/simulations"), float)
        pred = points.values(path + "/prediction").reshape(n, k)
        sub, whole = pred > cut, pred.mean(axis=1) > cut
        straddle = (pred.min(1) <= cut) & (pred.max(1) > cut)
    v = v.reshape(n, k, n_sim)
    size = blocks.block_size / np.asarray(blocks.discretization)
    lattice = np.stack(np.meshgrid(*[np.array([-1, 0, 1]) / 3.0] * 3,
                                   indexing="ij"), axis=-1).reshape(-1, 3)
    fine = coords.reshape(n, k, 1, 3) + lattice[None, None] \
        * np.repeat(size, k, axis=0).reshape(n, k, 1, 3)
    inside = truth(fine.reshape(-1, 3)).reshape(n, k, -1)
    benefit = (whole[:, None, None] != inside).mean(axis=(1, 2)) \
        - (sub[:, :, None] != inside).mean(axis=(1, 2))
    return v, cut, straddle, benefit * blocks.block_volume \
        / blocks.block_volume.max()


def measures(v, cut):
    n, k, S = v.shape
    coherence = geoml.likelihood._coherence(
        tf.constant(v[:, :, None, :])).numpy()[:, 0]
    p = (v > cut).mean(axis=2)
    # the coherence's own F statistic, as a significance
    f = coherence / (1 - coherence + 1e-300) * (S - 1)
    log10p = -stats.f.logsf(f, k - 1, (k - 1) * (S - 1)) / np.log(10)
    return {"coherence": coherence, "contrast": p.max(1) - p.min(1),
            "log10p": log10p}


def main():
    rows = []
    for name in ("pod", "ore", "rock", "grf"):
        model, coarse, cut, truth, path = case(name)
        for scale, blocks in (("coarse", coarse), ("half", halved(coarse))):
            for n_sim in (8, 20):
                v, cut, straddle, benefit = labelled(model, blocks, cut,
                                                     truth, path, n_sim)
                rows.append(("%s %s %d" % (name, scale, n_sim),
                             measures(v, cut), straddle, benefit))
                print("built", rows[-1][0], flush=True)

    far = {}
    if len(sys.argv) > 1:
        d = np.load(sys.argv[1])
        half = len(d["near"])
        for S in (25, 200):
            parts = []
            for metal, cuts in {"Zn": [5, 10, 15], "Pb": [5, 10, 15],
                                "Ag": [100, 200, 300]}.items():
                v = d["sims_%s_%d" % (metal, S)].astype(float)[half:]
                m = v.mean(axis=2)
                for c in cuts:
                    straddle = (m.min(1) <= c) & (m.max(1) > c)
                    parts.append({key: val[straddle] for key, val
                                  in measures(v, c).items()})
            far[S] = {key: np.concatenate([p[key] for p in parts])
                      for key in parts[0]}

    good = np.concatenate([m["coherence"][s & (b > 0)]
                           for _, m, s, b in rows])
    print("\ncoherence of the straddling blocks worth cutting: %d, "
          "%.3f to %.3f" % (len(good), good.min(), good.max()))
    for S, values in far.items():
        print("coherence of Run 12's far straddling blocks, %d "
              "realizations: %d, at most %.3f"
              % (S, len(values["coherence"]), values["coherence"].max()))
    for S in (8, 20, 25, 200):
        print("noise floor at %3d realizations, 8 sub-blocks: %.3f"
              % (S, stats.beta.ppf(1 - 1e-3, 3.5, 3.5 * (S - 1))))

    print("\nkept benefit / marked blocks, as shares of the straddle's, per "
          "case; Run 12 far marked at 25 / 200")
    for metric, thresholds in THRESHOLDS.items():
        for t in thresholds:
            cells = []
            for _, m, s, b in rows:
                mark = s & (m[metric] >= t)
                cells.append("%4.2f/%4.2f" % (
                    b[mark].sum() / b[s].sum() if b[s].sum() > 0 else 1.0,
                    mark.sum() / max(s.sum(), 1)))
            fp = " / ".join("%5.1f%%" % (100 * np.mean(far[S][metric] >= t))
                            for S in far)
            print("%-9s >= %-4g %s   %s" % (metric, t, " ".join(cells), fp))
    print("columns:", ", ".join(name for name, _, _, _ in rows))


if __name__ == "__main__":
    main()
