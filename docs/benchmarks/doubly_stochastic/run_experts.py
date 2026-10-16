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

"""The folded section's GP beside the coordinates with several experts.
Usage: python run_experts.py ARM SEED [N_EXPERTS]
  ARM: gate | moment | draws64 | warm (200 steps of 64 draws, then the 5x5
  grid). One JSON line to experts.jsonl."""
import json, os, resource, sys, time
import numpy as np, tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_kernel as ek, ds_prototype as ds, ds_experts as dx, run_folded as rf
import geoml
from scipy import stats
from geoml.latent import network as net

arm, seed = sys.argv[1], int(sys.argv[2])
n_experts = int(sys.argv[3]) if len(sys.argv) > 3 else 3
data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
new, holes = ek.drillholes(ek.N_NEW, ek.NEW_SEED)


def build():
    geoml.set_seed(seed)
    points = geoml.data.inducing.from_kmeans(data, 60, seed=seed)
    sets = geoml.data.inducing.experts(points, n_experts, overlap=0.25, seed=seed)
    root = geoml.latent.BasicInput(sets)
    hidden = geoml.latent.BasicGP(root, size=2)
    leaf = geoml.latent.BasicGP(geoml.latent.Concatenate(hidden, root), size=1)
    model = geoml.models.VGPNetwork(
        data, "V", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False, propagation="joint",
                                       expert_propagation="independent"))
    for p in model.likelihoods[0].warping.all_parameters:
        p.fix()
    return model, [s.n_data for s in sets]


def mixture(model, target, draws=256):
    x = tf.constant(np.asarray(target.coordinates), tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        mask = np.array(ds.uncertain_dims(model.leaves[0].parent), float)
        nodes = tf.random.stateless_normal([draws, x.shape[0], len(mask)], [7, 1],
                                           dtype=tf.float64) * mask
        mu, var = dx.expert_moments(model, x, nodes)
    mu, var = mu.numpy()[:, :, 0], var.numpy()[:, :, 0]
    y, log_d, noise = rf.warped(model, target)
    sd = np.sqrt(var + noise)
    z = (y[None] - mu) / sd
    lpd = np.logaddexp.reduce(stats.norm.logpdf(z) - np.log(sd), 0) - np.log(draws) + log_d
    per_hole = np.array([lpd[holes == h].sum() for h in np.unique(holes)])
    return float(per_hole.mean())


model, sizes = build()
if arm == "gate":
    # per expert: the coupled kernel averaged over 20000 draws against the
    # joint expected kernel of that expert
    x = tf.constant(np.asarray(new.coordinates)[:120], tf.float64)
    leaf = model.leaves[0]
    model.train_full(200)
    with model._propagation():
        model._refresh(model.options.jitter)
        chain = net._chain_of(leaf.parent.propagate(x))
        ids = net._active_experts(leaf.root)
        ips, ipvs = leaf._parent_points(ids)
        for p in range(len(ids)):
            e = chain[p]
            joint = leaf.expected_covariance(e.mean, e.variance, ips[p], ipvs[p],
                                             e.covariance).numpy()
            acc = 0.0
            for c in range(20):
                eps = tf.random.stateless_normal([1000] + list(e.mean.shape), [9, c],
                                                 dtype=tf.float64)
                h = e.mean[None] + tf.sqrt(e.variance)[None] * eps
                acc = acc + dx._kernel(leaf, h, e, ips[p], ipvs[p]).numpy().sum(0)
            err = np.abs(acc / 20000 - joint)
            print("expert %d (%d points): coupled average vs joint expected kernel, "
                  "max err %.2e (largest entry %.3f)" % (p, sizes[p], err.max(), joint.max()))
    sys.exit()

start = time.perf_counter()
if arm == "moment":
    model.train_full(10)
    t = time.perf_counter(); model.train_full(1990); per = (time.perf_counter() - t) / 1990
elif arm == "draws64":
    per = dx.train(model, 2000, draws=64)
else:
    dx.train(model, 200, draws=64)
    per = dx.train(model, 1800, hermite=5)
seconds = time.perf_counter() - start
moment = rf.summary(*rf.moment_scores(model, new), holes)
row = dict(arm=arm, seed=seed, n_experts=n_experts, expert_sizes=sizes,
           seconds=seconds, ms_per_iter=1000 * per,
           moment_score=moment["score_per_hole"], moment_crps=moment["crps"],
           mixture_score=mixture(model, new),
           peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "doubly_stochastic"), "experts.jsonl"), "a").write(json.dumps(row) + "\n")
print(json.dumps(row))
