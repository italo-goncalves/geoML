"""Where the pathwise realizations leave the moments: the realized input
against the chain, and the conditional moments over it against the
quadrature's."""
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402

data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
new, _ = gm.ek.drillholes(gm.ek.N_NEW, gm.ek.NEW_SEED)
pick = np.random.default_rng(3).choice(new.n_data, gm.N_POINTS, replace=False)
x = np.asarray(new.coordinates, dtype=float)[pick]
model = gm.build(data, "dgp", kr.Gaussian)
model.train_full(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
leaf = model.leaves[0]
S = 4000
with model._propagation():
    model._refresh(model.options.jitter)
    with net.once_per_pass():
        moments = leaf.propagate(tf.constant(x))
        chain = leaf._path_state[1][0]
        hid = leaf.parent
        h = np.asarray(hid.simulate(S, [3, 1]))           # [d, n, S]
        gap = np.asarray(hid._realization_gap)            # [d, n]
print("chain mean vs realized mean, in chain sds (max):",
      np.max(np.abs(h.mean(-1).T - np.asarray(chain.mean))
             / np.sqrt(np.asarray(chain.variance))))
real_var = h.var(-1).T
print("realized var / chain var (median):",
      np.median(real_var / np.asarray(chain.variance)))
print("(realized var + gap) / chain var (median):",
      np.median((real_var + gap.T) / np.asarray(chain.variance)))
print("gap / chain var (median):", np.median(gap.T / np.asarray(chain.variance)))

# the conditioned kernel over the realized inputs against the quadrature
with model._propagation():
    model._refresh(model.options.jitter)
    with net.once_per_pass():
        moments = leaf.propagate(tf.constant(x))
        coords, chains, weights = leaf._path_state
        noise = np.random.default_rng(0).normal(size=h.shape)
        hs = h + np.sqrt(np.maximum(gap, 0))[:, :, None] * noise   # [d, n, S]
        hs = tf.constant(np.transpose(hs, [1, 2, 0]))               # [n, S, d]
        ips, ipvs = leaf._parent_points([0])
        k = np.asarray(leaf._realized_covariance(hs, chains[0], ips[0],
                                                 ipvs[0]))          # [n, S, m]
        l = np.asarray(leaf.expected_covariance(
            chains[0].mean, chains[0].variance, ips[0], ipvs[0],
            chains[0].covariance))
        sums, q = leaf._conditioned_sums(
            chains[0].mean, chains[0].variance, chains[0].covariance,
            ips[0], ipvs[0], leaf.cov_smooth_inv[0], leaf.alpha[0])
inv = np.asarray(leaf.cov_smooth_inv[0])[0]
alpha = np.asarray(leaf.alpha[0])[0, :, 0]
kbar_s = k.mean(1)
kbar_q = np.asarray(sums[0])[:, :] / q
print("max |mean_s k - l|:", np.abs(kbar_s - l).max(),
      " max |mean_q k - l|:", np.abs(kbar_q - l).max(), " max l:", l.max())
ksk_s = np.einsum("nsi,ij,nsj->ns", k, inv, k).mean(1)
ksk_q = np.asarray(sums[1])[0] / q
f = k @ alpha
print("E[kSk]: draws median %.4f, quadrature median %.4f" % (
    np.median(ksk_s), np.median(ksk_q)))
var_f_q = np.asarray(sums[3])[0] / q - (np.asarray(sums[2])[0] / q) ** 2
print("Var(k alpha): draws median %.4f, quadrature median %.4f" % (
    np.median(f.var(1)), np.median(var_f_q)))

root_r = np.asarray(leaf.chol_r[0])[0]
kr2 = (np.einsum("nsm,mj->nsj", k, root_r) ** 2).sum(-1)
left = np.maximum(1.0 - np.einsum("nsi,ij,nsj->ns", k, inv, k) - kr2, 0.0)
print("share of draws whose left variance is clamped at zero: %.3f" % np.mean(
    1.0 - np.einsum("nsi,ij,nsj->ns", k, inv, k) - kr2 < 0))
manual = f.var(1) + kr2.mean(1) + left.mean(1)
with model._propagation():
    model._refresh(model.options.jitter)
    with net.once_per_pass():
        m, v = leaf.propagate(tf.constant(x))
        sims = np.asarray(leaf.simulate(S, [3, 1]))[0]
print("simulate's variance / analytic (median): %.3f" % np.median(
    sims.var(1) / np.asarray(v)[:, 0]))
print("manual terms / analytic (median): %.3f" % np.median(
    manual / np.asarray(v)[:, 0]))
print("simulate's mean - analytic mean, in sds (median |.|): %.3f" % np.median(
    np.abs(sims.mean(1) - np.asarray(m)[:, 0]) / np.sqrt(np.asarray(v)[:, 0])))

with model._propagation():
    model._refresh(model.options.jitter)
    parts = []
    for c in range(10):
        with net.once_per_pass():
            m, v = leaf.propagate(tf.constant(x))
            parts.append(np.asarray(leaf.simulate(400, [11, c]))[0])
pooled = np.concatenate(parts, 1)
print("ten calls pooled: variance / analytic (median) %.3f" % np.median(
    pooled.var(1) / np.asarray(v)[:, 0]))
print("within one call of 400 (median) %.3f" % np.median(
    parts[0].var(1) / np.asarray(v)[:, 0]))
means = np.stack([p.mean(1) for p in parts])
print("spread of the calls' means / expected (median) %.3f" % np.median(
    means.var(0) / (np.asarray(v)[:, 0] / 400)))
