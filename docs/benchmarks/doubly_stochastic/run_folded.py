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

"""The folded section under three ways of training and scoring a deep GP.

Usage: python run_folded.py ITERATIONS NETWORK ARM [S_TRAIN]

ARM is one of
  moment       trained on the moment bound (today's), 0.9.x default
  ds-coupled   trained on the doubly stochastic bound, inducing inputs
               conditioned on each draw (the same model as `moment`)
  ds-indep     trained on the doubly stochastic bound, inducing inputs
               independent of the draw (Salimbeni's bound as written)
  vgp          the flat GP, for reference (NETWORK ignored)

Every trained model is scored both ways it can be: by its moments
(Gaussian predictive) and by the mixture over S_SCORE draws of the hidden
layer, on the benchmark's 200 new drillholes; and its realizations against
the true field on a grid. Writes a JSON line to results.jsonl.
"""
import json
import os
import resource
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_kernel as ek  # noqa: E402
import ds_prototype as ds  # noqa: E402
import geoml  # noqa: E402
import geoml.likelihood as lk  # noqa: E402

S_SCORE, N_FIELDS = 256, 64
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "doubly_stochastic")
os.makedirs(HERE, exist_ok=True)


def build(data, network, rule="joint", seed=None):
    """`expected_kernel.folded_model` without the training; `seed` moves
    the initialization and the k-means start, never the data."""
    seed = ek.SEED if seed is None else seed
    geoml.set_seed(seed)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, ek.N_IND, seed=seed))
    if network == "vgp":
        leaf = geoml.latent.BasicGP(root, size=1)
    else:
        hidden = geoml.latent.BasicGP(root, size=2)
        below = hidden if network == "dgp" \
            else geoml.latent.Concatenate(hidden, root)
        leaf = geoml.latent.BasicGP(below, size=1)
    options = geoml.models.GPOptions(verbose=False, propagation=rule,
                                     expert_propagation="independent")
    model = geoml.models.VGPNetwork(data, "V", geoml.likelihood.Gaussian(),
                                    leaf, options=options)
    for p in model.likelihoods[0].warping.all_parameters:
        p.fix()
    return model


def warped(model, data):
    likelihood = model.likelihoods[0]
    likelihood.warping.refresh()
    y, _ = data.variables["V"].get_measurements()
    y_w, log_d = likelihood.warping.forward(tf.constant(y, tf.float64))
    noise = float(np.reshape(np.asarray(
        likelihood.parameters["noise"].get_value()), -1)[0])
    return (np.reshape(np.asarray(y_w), -1), np.reshape(np.asarray(log_d), -1),
            noise)


def mixture_scores(model, data, coupling, n_draws=S_SCORE, seed=5):
    """Log predictive density, PIT and CRPS per sample from the mixture of
    the leaf's conditionals over draws of its parent, in warped units (the
    log-density takes the warping's log-derivative)."""
    x = tf.constant(np.asarray(data.coordinates), tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        if isinstance(model.leaves[0].parent, geoml.latent.BasicInput):
            mu, var = model.leaves[0].propagate(x)
            mu, var = mu.numpy()[None, :, 0], var.numpy()[None, :, 0]
        else:
            mu, var = ds.leaf_moments(model, x, n_draws, seed, coupling)
            mu, var = mu.numpy()[:, :, 0], var.numpy()[:, :, 0]
    y, log_d, noise = warped(model, data)
    sd = np.sqrt(var + noise)
    from scipy import stats
    z = (y[None] - mu) / sd
    lpd = np.logaddexp.reduce(stats.norm.logpdf(z) - np.log(sd), axis=0) \
        - np.log(mu.shape[0]) + log_d
    pit = stats.norm.cdf(z).mean(0)
    # CRPS of the mixture by sampling it
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, mu.shape[0], [400, mu.shape[1]])
    draws = mu[pick, np.arange(mu.shape[1])] \
        + sd[pick, np.arange(mu.shape[1])] * rng.standard_normal(pick.shape)
    crps = np.mean(np.abs(draws - y[None]), 0) \
        - 0.5 * np.mean(np.abs(draws[::2] - draws[1::2]), 0)
    return lpd, pit, crps


def moment_scores(model, data):
    lpd, z, _ = ek.log_scores(model, data)
    from scipy import stats
    pit = stats.norm.cdf(z)
    y, _, noise = warped(model, data)
    # Gaussian CRPS in closed form, warped units
    x = tf.constant(np.asarray(data.coordinates), tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        mu, var, _, _ = model.leaves[0].predict(x, n_sim=1)
    mu, var = np.asarray(mu)[0, :, 0], np.asarray(var)[0]
    s = np.sqrt(var + noise)
    w = (y - mu) / s
    crps = s * (w * (2 * stats.norm.cdf(w) - 1) + 2 * stats.norm.pdf(w)
                - 1 / np.sqrt(np.pi))
    return lpd, pit, crps


def summary(lpd, pit, crps, holes):
    per_hole = np.array([lpd[holes == h].sum() for h in np.unique(holes)])
    return dict(score_per_hole=float(per_hole.mean()),
                median_per_hole=float(np.median(per_hole)),
                cover90=float(np.mean((pit > 0.05) & (pit < 0.95))),
                crps=float(np.mean(crps)))


def grid():
    gx, gd = np.meshgrid(np.linspace(0.2, 4.8, 47), np.linspace(0.0, 3.0, 31))
    return np.column_stack([gx.ravel(), gd.ravel()]), ek.folded(gx, gd).ravel()


def ensemble_scores(sims, truth):
    """CRPS and 90% coverage of the truth by an ensemble `[S, n]`."""
    s = np.sort(sims, axis=0)
    lo, hi = np.quantile(sims, [0.05, 0.95], axis=0)
    crps = np.mean(np.abs(sims - truth[None]), 0) \
        - 0.5 * np.mean(np.abs(sims[:, None] - sims[None]), (0, 1))
    return dict(crps=float(np.mean(crps)),
                cover90=float(np.mean((truth >= lo) & (truth <= hi))),
                spread=float(np.mean(np.std(sims, 0))))


def field_scores(model):
    points, truth = grid()
    out = {}
    target = geoml.data.PointData.from_array(points, ["X", "Y"])
    model.predict(target, n_sim=N_FIELDS)
    today = np.asarray(target.variables["V"].get_simulations()).T  # [S, n]
    out["today"] = ensemble_scores(today, truth)
    if isinstance(model.leaves[0].parent, geoml.latent.BasicInput):
        return out
    x = tf.constant(points, tf.float64)
    backward = model.likelihoods[0].warping.backward
    for kind in ("explained", "full"):
        with model._propagation():
            model._refresh(model.options.jitter)
            f = ds.leaf_fields(model, x, N_FIELDS, 9, kind, ds.COUPLED)
        sims = np.asarray(backward(tf.reshape(f, [-1, 1]))).reshape(
            N_FIELDS, -1)
        out["ds_" + kind] = ensemble_scores(sims, truth)
    return out


def timed_predictions(model, data, coupling):
    """Seconds for one moment prediction and one 256-draw mixture at the
    200 new holes, after a warm-up of each."""
    x = tf.constant(np.asarray(data.coordinates), tf.float64)
    out = {}
    with model._propagation():
        model._refresh(model.options.jitter)
        for name, fn in (
                ("moment", lambda: model.leaves[0].predict(x, n_sim=1)),
                ("mixture", lambda: ds.leaf_moments(model, x, S_SCORE, 1,
                                                    coupling))):
            fn()
            start = time.perf_counter()
            for _ in range(3):
                fn()
            out[name] = (time.perf_counter() - start) / 3
    return out


def exact_moments(model, x, n_draws=4000, chunk=None, seed=21):
    """The leaf's exact predictive mean and variance through the hidden
    layer, by coupled draws: the mean of the conditional means, and the
    variance split as E[var_s] + Var[mu_s]; each `[n]`."""
    if chunk is None:
        # about 2e7 numbers in a [chunk, n, m, d] tensor, a few such alive
        leaf = model.leaves[0]
        m = int(leaf.root.n_ip[0])
        chunk = max(1, int(2e7 // (int(x.shape[0]) * m * leaf.parent.size)))
        chunk = min(chunk, n_draws)
    n_draws = (n_draws // chunk) * chunk
    s1 = s2 = sv = 0.0
    with model._propagation():
        model._refresh(model.options.jitter)
        for c in range(n_draws // chunk):
            mu, var = ds.leaf_moments(model, x, chunk, seed + c, ds.COUPLED)
            mu, var = mu.numpy()[:, :, 0], var.numpy()[:, :, 0]
            s1, s2, sv = s1 + mu.sum(0), s2 + (mu ** 2).sum(0), sv + var.sum(0)
    mean = s1 / n_draws
    return mean, sv / n_draws, s2 / n_draws - mean ** 2


def decomposition(model, data, holes):
    """At the new holes: the moment path's variance against the exact one,
    E[var_s] (the moment variance less tr(M Cov k)) plus Var[mu_s]; and the
    log score of the moment Gaussian, of a Gaussian with the exact two
    moments, and of the mixture (`mixture` in the row)."""
    from scipy import stats
    x = tf.constant(np.asarray(data.coordinates), tf.float64)
    with model._propagation():
        model._refresh(model.options.jitter)
        mu_m, v_m = model.leaves[0].propagate(x)
    mu_m, v_m = mu_m.numpy()[:, 0], v_m.numpy()[:, 0]
    mean, e_var, var_mu = exact_moments(model, x)
    y, log_d, noise = warped(model, data)
    exact = e_var + var_mu
    lpd = stats.norm.logpdf(y, mean, np.sqrt(exact + noise)) + log_d
    per_hole = np.array([lpd[holes == h].sum() for h in np.unique(holes)])
    return dict(moment_var=float(v_m.mean()), e_var=float(e_var.mean()),
                var_mu=float(var_mu.mean()), exact_var=float(exact.mean()),
                share_captured=float(np.median(v_m / exact)),
                mean_gap=float(np.max(np.abs(mean - mu_m))),
                exact_gauss_per_hole=float(per_hole.mean()), noise=noise)


def map_grid():
    gx, gd = np.meshgrid(np.linspace(0, 5, 101), np.linspace(-0.2, 3.2, 69))
    return gx, gd, np.column_stack([gx.ravel(), gd.ravel()])


def save_maps(model, row, path):
    """Everything the figures need, on the figure grid, in data units: the
    moment and exact predictive mean and sd, the hidden layer's mean and sd,
    three of today's realizations and three doubly stochastic ones, and the
    training log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gx, gd, points = map_grid()
    x = tf.constant(points, tf.float64)
    w = model.likelihoods[0].warping
    zero, one = (float(np.asarray(w.backward(tf.constant([[v]], tf.float64)))
                 [0, 0]) for v in (0.0, 1.0))
    scale = one - zero
    out = dict(shape=np.array(gx.shape), log=np.array(model.training_log))
    with model._propagation():
        model._refresh(model.options.jitter)
        mu, var = model.leaves[0].propagate(x)
        out["moment_mean"] = zero + scale * mu.numpy()[:, 0]
        out["moment_sd"] = scale * np.sqrt(var.numpy()[:, 0])
        parent = model.leaves[0].parent
        if not isinstance(parent, geoml.latent.BasicInput):
            hm, hv, _ = ds._parent_moments(model.leaves[0], x)
            out["hidden_mean"], out["hidden_sd"] = hm.numpy(), np.sqrt(hv.numpy())
            f = ds.leaf_fields(model, x, 3, 31, "full", ds.COUPLED)
            out["ds_fields"] = zero + scale * f.numpy()[:, :, 0]
    if not isinstance(model.leaves[0].parent, geoml.latent.BasicInput):
        mean, e_var, var_mu = exact_moments(model, x, n_draws=1000)
        out["exact_mean"] = zero + scale * mean
        out["exact_sd"] = scale * np.sqrt(e_var + var_mu)
        out["e_var_sd"] = scale * np.sqrt(e_var)
    target = geoml.data.PointData.from_array(points, ["X", "Y"])
    model.predict(target, n_sim=3)
    out["today_fields"] = np.asarray(target.variables["V"].get_simulations()).T
    np.savez_compressed(path, **out)


def main(argv):
    iterations, network, arm = int(argv[0]), argv[1], argv[2]
    s_train = int(argv[3]) if len(argv) > 3 else 16
    seed = int(argv[4]) if len(argv) > 4 else ek.SEED
    if arm == "vgp":
        network = "vgp"
    data, _ = ek.drillholes(ek.N_HOLES, ek.SEED)
    new, holes = ek.drillholes(ek.N_NEW, ek.NEW_SEED)
    in_holes = np.repeat(np.arange(ek.N_HOLES), ek.N_PER_HOLE)
    model = build(data, network, seed=seed)
    start = time.perf_counter()
    if arm in ("moment", "vgp"):
        model.train_full(10)
        began = time.perf_counter()
        model.train_full(iterations - 10)
        per_iter = (time.perf_counter() - began) / (iterations - 10)
    elif arm.startswith("quad"):
        _, per_iter = ds.train_ds(model, iterations, s_train,
                                  hermite=int(arm[4:]))
    else:
        coupling = ds.COUPLED if arm == "ds-coupled" else ds.INDEPENDENT
        _, per_iter = ds.train_ds(model, iterations, s_train, coupling)
    seconds = time.perf_counter() - start
    row = dict(network=network, arm=arm, iterations=iterations, seed=seed,
               s_train=s_train if arm.startswith("ds") else None,
               hermite=int(arm[4:]) if arm.startswith("quad") else None,
               train_seconds=seconds, ms_per_iter=1000 * per_iter, final_bound=float(model.training_log[-1]),
               noise=float(np.reshape(np.asarray(model.likelihoods[0]
                     .parameters["noise"].get_value()), -1)[0]))
    row["moment"] = summary(*moment_scores(model, new), holes)
    row["moment_in"] = summary(*moment_scores(model, data), in_holes)
    if network != "vgp":
        coupling = ds.INDEPENDENT if arm == "ds-indep" else ds.COUPLED
        row["mixture"] = summary(*mixture_scores(model, new, coupling), holes)
        row["mixture_in"] = summary(*mixture_scores(model, data, coupling),
                                    in_holes)
        row["predict_seconds"] = timed_predictions(model, new, coupling)
    if network != "vgp":
        row["decomposition"] = decomposition(model, new, holes)
    row["fields"] = field_scores(model)
    save_maps(model, row, os.path.join(
        HERE, "maps", "%s_%s_S%s_seed%d_it%d.npz"
        % (network, arm, s_train, seed, iterations)))
    row["peak_rss_mb"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
