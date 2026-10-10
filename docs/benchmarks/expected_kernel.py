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

"""The gates of the expected kernel (0.9.0).

Usage: python docs/benchmarks/expected_kernel.py kernel|folded [iterations]

`kernel` (gate 1, the function alone): E[k(h(x), h(y))] against the
average of k over 4e5 draws of a jointly Gaussian input at twelve
locations in two dimensions, for every kernel the expected kernel takes;
the error with the covariance between locations dropped, and the marginal
rule of the versions before; and a figure of one pair of locations as the
input grows uncertain. The same check through a network is
`test_expected_kernel.py`.

`folded` (gate 2): the research project's folded section, a VGP and two
deep networks (a GP on a GP, and on the GP beside the coordinates) under
both rules, scored by the log predictive density of 200 new drillholes,
with the calibration of their standardized residuals (one is calibrated).
"""

import os
import sys

import numpy as np
import tensorflow as tf

import geoml
from geoml.latent import network as net

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

KERNELS = {
    "Gaussian": geoml.kernels.Gaussian,
    "Exponential": geoml.kernels.Exponential,
    "Matern32": geoml.kernels.Matern32,
    "Matern52": geoml.kernels.Matern52,
    "RationalQuadratic": lambda: geoml.kernels.RationalQuadratic(0.7),
}


def joint_inputs(n=12, dim=2, seed=3):
    """A random Gaussian input at `n` locations, one field per dimension,
    correlated between locations: means `[n, dim]`, covariances
    `[n, n, dim]`."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(-0.8, 0.8, [n, dim])
    covs = []
    for _ in range(dim):
        features = rng.normal(size=[n, 3]) * 0.3
        sites = rng.uniform(0, 1, [n, 1])
        covs.append(features @ features.T
                    + 0.15 * np.exp(-(sites - sites.T) ** 2 / 0.1)
                    + 1e-9 * np.eye(n))
    return mean, np.stack(covs, axis=-1)


def kernel_values(kernel, d):
    return np.asarray(kernel.kernelize(tf.constant(d, tf.float64)))


def monte_carlo(kernel, ranges, mean, cov, draws=400_000, seed=11):
    rng = np.random.default_rng(seed)
    n, dim = mean.shape
    roots = [np.linalg.cholesky(cov[:, :, s]) for s in range(dim)]
    total, square, done = np.zeros([n, n]), np.zeros([n, n]), 0
    while done < draws:
        size = min(20_000, draws - done)
        h = np.stack([mean[:, s][None, :]
                      + rng.normal(size=[size, n]) @ roots[s].T
                      for s in range(dim)], axis=-1)
        d = np.sqrt(np.sum(((h[:, :, None] - h[:, None]) / ranges) ** 2, -1))
        k = kernel_values(kernel, d)
        total += k.sum(0)
        square += (k ** 2).sum(0)
        done += size
    avg = total / draws
    return avg, np.sqrt(np.maximum(square / draws - avg ** 2, 0) / draws)


def expected(kernel, ranges, mean, var, cov):
    r = tf.constant(np.asarray(ranges, float).reshape([1, 1, -1]), tf.float64)
    m = tf.constant(mean, tf.float64)
    v = None if var is None else tf.constant(var, tf.float64)
    c = None if cov is None else tf.constant(cov, tf.float64)
    return np.asarray(net._expected_kernel(kernel, r, m, v, m, v, c))


def marginal(kernel, ranges, mean, var):
    """The rule of the versions before 0.9.0 (`BasicGP.covariance_matrix`):
    the variances added to the squared range with weight one half, the
    locations independent, Paciorek's normalization."""
    r2 = np.asarray(ranges, float) ** 2
    vx, vy = var[:, None, :], var[None, :, :]
    total = r2 + (vx + vy) / 2
    d = np.sqrt(np.sum((mean[:, None] - mean[None]) ** 2 / total, -1))
    norm = np.prod(vx + r2, -1) ** 0.25 * np.prod(vy + r2, -1) ** 0.25 \
        / np.sqrt(np.prod(total, -1))
    return kernel_values(kernel, d) * norm


def kernel_gate():
    mean, cov = joint_inputs()
    var = np.stack([np.diag(cov[:, :, s]) for s in range(cov.shape[2])], -1)
    ranges = np.array([0.7, 1.1])
    off = ~np.eye(len(mean), dtype=bool)
    print("Gate 1, the function: 12 locations, 2 dimensions, 4e5 draws")
    print("%-18s %10s %8s %14s %14s" % ("kernel", "max error", "in s.e.",
                                       "no covariance", "marginal rule"))
    for name, make in KERNELS.items():
        kernel = make()
        avg, se = monte_carlo(kernel, ranges, mean, cov)
        joint = np.abs(expected(kernel, ranges, mean, var, cov) - avg)[off]
        alone = np.abs(expected(kernel, ranges, mean, var, None) - avg)[off]
        old = np.abs(marginal(kernel, ranges, mean, var) - avg)[off]
        print("%-18s %10.4f %8.1f %14.4f %14.4f"
              % (name, joint.max(), (joint / se[off]).max(), alone.max(),
                 old.max()))
    pair_figure()


def pair_figure():
    """Two locations half a range apart, their inputs equally uncertain:
    the expected kernel, the draws' average and the marginal rule as the
    uncertainty grows, with the inputs independent and correlated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sds = np.geomspace(0.02, 2.0, 25)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True,
                             sharey=True)
    for row, name in enumerate(("Gaussian", "Matern32")):
        kernel = KERNELS[name]()
        for col, rho in enumerate((0.0, 0.9)):
            ax = axes[row, col]
            joint, independent, old, mc = [], [], [], []
            for sd in sds:
                mean = np.array([[0.0], [0.5]])
                c = sd ** 2 * np.array([[1.0, rho], [rho, 1.0]])[:, :, None]
                var = np.full([2, 1], sd ** 2)
                joint.append(expected(kernel, [1.0], mean, var, c)[0, 1])
                independent.append(
                    expected(kernel, [1.0], mean, var, None)[0, 1])
                old.append(marginal(kernel, [1.0], mean, var)[0, 1])
                mc.append(monte_carlo(kernel, np.array([1.0]), mean,
                                      c + 1e-12 * np.eye(2)[:, :, None],
                                      draws=40_000)[0][0, 1])
            ax.plot(sds, joint, color="C0", linewidth=2,
                    label="expected kernel")
            if rho > 0:
                ax.plot(sds, independent, color="C0", linestyle=":",
                        label="expected kernel, correlation dropped")
            ax.plot(sds, old, color="C3", linestyle="--",
                    label="marginal rule (before 0.9.0)")
            ax.plot(sds, mc, "o", color="black", markersize=3,
                    label="average over draws")
            ax.set_xscale("log")
            ax.set_title("%s kernel, inputs correlated at %.1f"
                         % (name, rho))
            if row == 1:
                ax.set_xlabel("input standard deviation (ranges)")
            if col == 0:
                ax.set_ylabel("covariance of the two locations")
            ax.grid(alpha=0.3)
    axes[0, 1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Two locations half a range apart, their input uncertain")
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "expected_kernel_pair.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("figure:", path)


# --------------------------------------------------------------------------- #
# gate 2: the folded section
# --------------------------------------------------------------------------- #
# The research project's synthetic case (Spatial cross-validation,
# experiments/bncv_prototype.py): a 2-D section, 16 vertical drillholes of 12
# samples, thin layers along a folded stratigraphic coordinate, Gaussian
# noise of 0.15; 30 k-means inducing points; scored on 200 new drillholes
# drawn with another seed.
N_HOLES, N_PER_HOLE, N_IND, SEED, NEW_SEED, N_NEW = 16, 12, 30, 2026, 777, 200
FOLD_AMP, FOLD_WAVE, LAYER_PERIOD = 0.8, 3.0, 1.5


def folded(x, depth):
    s = depth - FOLD_AMP * np.sin(2 * np.pi * x / FOLD_WAVE)
    return np.sin(2 * np.pi * s / LAYER_PERIOD) + 0.3 * np.sin(1.1 * x)


def drillholes(n_holes, seed):
    import pandas as pd
    rng = np.random.default_rng(seed)
    xh = np.sort(rng.uniform(0.2, 4.8, n_holes))
    rows = []
    for h, x0 in enumerate(xh):
        depth = np.sort(rng.uniform(0.0, 3.0, N_PER_HOLE))
        x = x0 + 0.05 * depth
        v = folded(x, depth) + 0.15 * rng.standard_normal(N_PER_HOLE)
        rows += [(a, b, c, h) for a, b, c in zip(x, depth, v)]
    frame = pd.DataFrame(rows, columns=["X", "Y", "V", "hole"])
    data = geoml.data.PointData(frame, ["X", "Y"])
    data.add_continuous_variable("V", frame["V"].values)
    return data, frame["hole"].values


def folded_model(data, network, rule, iterations=2000):
    geoml.set_seed(SEED)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, N_IND, seed=SEED))
    if network == "vgp":
        leaf = geoml.latent.BasicGP(root, size=1)
    elif network == "walk":
        leaf = geoml.latent.BasicGP(geoml.latent.GPWalk(
            geoml.latent.BasicGP(root, size=2)), size=1)
    else:
        hidden = geoml.latent.BasicGP(root, size=2)
        below = hidden if network == "dgp" \
            else geoml.latent.Concatenate(hidden, root)
        leaf = geoml.latent.BasicGP(below, size=1)
    options = geoml.models.GPOptions(
        verbose=False, propagation=rule,
        expert_propagation="independent" if rule == "joint" else "consensus")
    model = geoml.models.VGPNetwork(data, "V", geoml.likelihood.Gaussian(),
                                    leaf, options=options)
    # the warping's centre duplicates the GP's bias: kept where the data
    # put it, as the research did
    for p in model.likelihoods[0].warping.all_parameters:
        p.fix()
    model.train_full(iterations)
    return model


def log_scores(model, data):
    """Each sample's log predictive density, and its standardized residual,
    from the leaf's moments through the likelihood's noise and warping."""
    import geoml.likelihood as lk
    with model._propagation():
        model._refresh(model.options.jitter)
        x = tf.constant(np.asarray(data.coordinates), tf.float64)
        mu, var, _, _ = model.leaves[0].predict(x, n_sim=1)
    mu, var = np.asarray(mu)[0, :, 0], np.asarray(var)[0]
    likelihood = model.likelihoods[0]
    likelihood.warping.refresh()
    y, _ = data.variables["V"].get_measurements()
    y_w, log_d = likelihood.warping.forward(tf.constant(y, tf.float64))
    nodes = np.sqrt(2 * var)[:, None] * lk._ROOTS_64.numpy()[None, :] \
        + mu[:, None]
    logp = likelihood._make_distribution(
        tf.constant(nodes[:, None, :])).log_prob(
        y_w[:, :, None]).numpy()[:, 0, :]
    lpd = np.logaddexp.reduce(logp + np.log(lk._WEIGHTS_64.numpy())[None],
                              axis=1) + np.reshape(np.asarray(log_d), -1)
    noise = float(np.reshape(np.asarray(
        likelihood.parameters["noise"].get_value()), -1)[0])
    z = (np.reshape(np.asarray(y_w), -1) - mu) / np.sqrt(var + noise)
    return lpd, z, var


def folded_gate(iterations=2000, networks=("vgp", "dgp", "dgpc", "walk")):
    import time
    data, _ = drillholes(N_HOLES, SEED)
    new, holes = drillholes(N_NEW, NEW_SEED)
    print("Gate 2, the folded section: 16 holes, %d Adam iterations, "
          "scored on 200 new holes" % iterations)
    print("%-6s %-9s %12s %14s %12s %12s %8s"
          % ("model", "rule", "score/hole", "median/hole", "calibration",
             "in-sample", "time s"))
    rows = {}
    for network in networks:
        for rule in (("joint",) if network == "vgp"
                     else ("marginal", "joint")):
            start = time.perf_counter()
            model = folded_model(data, network, rule, iterations)
            seconds = time.perf_counter() - start
            lpd, z, _ = log_scores(model, new)
            per_hole = np.array([lpd[holes == h].sum()
                                 for h in np.unique(holes)])
            lpd_in, _, _ = log_scores(model, data)
            rows[(network, rule)] = (per_hole.mean(), np.mean(z ** 2), model)
            reach = ""
            for node in model._nodes():
                if isinstance(node, geoml.latent.GPWalk):
                    reach = "  reach %.2f" % (
                        node.step * node.n_steps
                        * float(node.parameters["amp"].get_value()))
            print("%-6s %-9s %12.2f %14.2f %12.2f %12.1f %8.0f%s"
                  % (network, rule, per_hole.mean(), np.median(per_hole),
                     np.mean(z ** 2), lpd_in.sum() / N_HOLES, seconds,
                     reach))
    folded_figure(rows, data)
    return rows


def folded_figure(rows, data):
    """The fitted field and its standard deviation for each model, with the
    truth, along the section."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gx, gd = np.meshgrid(np.linspace(0, 5, 101), np.linspace(-0.2, 3.2, 69))
    grid = np.column_stack([gx.ravel(), gd.ravel()])
    keys = list(rows)
    fig, axes = plt.subplots(2, len(keys) + 1,
                             figsize=(3.2 * (len(keys) + 1), 6.4),
                             sharex=True, sharey=True, layout="constrained")
    holes = np.asarray(data.coordinates)
    truth = folded(gx, gd)
    axes[0, 0].pcolormesh(gx, gd, truth, cmap="RdBu_r", vmin=-1.4, vmax=1.4)
    axes[0, 0].set_title("truth")
    axes[1, 0].axis("off")
    for c, key in enumerate(keys, start=1):
        model = rows[key][2]
        target = geoml.data.PointData.from_array(grid, ["X", "Y"])
        model.predict(target, n_sim=1)
        variable = target.variables["V"]
        mean = np.asarray(variable.prediction.values).reshape(gx.shape)
        sd = np.sqrt(np.asarray(variable.latent_variance.values)
                     ).reshape(gx.shape)
        axes[0, c].pcolormesh(gx, gd, mean, cmap="RdBu_r", vmin=-1.4,
                              vmax=1.4)
        axes[0, c].set_title("%s, %s\n%.1f a new hole"
                             % (key[0], key[1], rows[key][0]))
        image = axes[1, c].pcolormesh(gx, gd, sd, cmap="viridis", vmin=0,
                                      vmax=1)
        axes[1, c].set_title("latent sd")
    for ax in axes.ravel()[:len(keys) + 1].tolist() + axes[1, 1:].tolist():
        ax.plot(holes[:, 0], holes[:, 1], ".", color="black", markersize=2)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel("depth")
    axes[1, 1].set_ylabel("depth")
    fig.colorbar(image, ax=axes[1, -1], label="latent sd")
    fig.suptitle("The folded section: fitted fields under both rules")
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "expected_kernel_folded.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("figure:", path)


# --------------------------------------------------------------------------- #
# gate 3: chapter 5's deep model
# --------------------------------------------------------------------------- #
def chapter5_model(rule, deep=True, iterations=100):
    """Chapter 5's two-layer model on Walker Lake -- the outer kernel a
    Matern32, which the expected kernel takes where the chapter had a
    spherical -- or a flat one on the same input."""
    walker, walker_grid = geoml.datasets.walker()
    geoml.set_seed(1234)
    experts = geoml.data.inducing.grid_experts(walker_grid, 10.0, block=8)
    root = geoml.latent.BasicInput(
        experts, transform=geoml.transform.Isotropic(50))
    below = root
    if deep:
        inner = geoml.latent.BasicGP(root, size=2,
                                     kernel=geoml.kernels.Gaussian())
        below = geoml.latent.Concatenate(root, inner)
    outer = geoml.latent.BasicGP(below, size=1,
                                 kernel=geoml.kernels.Matern32())
    warping = geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1, shift=1.0), geoml.warping.ZScore(1))
    model = geoml.models.VGPNetwork(
        walker, "V", geoml.likelihood.Gaussian(warping), outer,
        options=geoml.models.GPOptions(
            verbose=False, propagation=rule,
            expert_propagation="independent" if rule == "joint"
            else "consensus"))
    model.train_full(max_iter=iterations)
    return model, walker_grid


def chapter5_gate(iterations=100):
    import time
    from geoml import metrics
    print("Gate 3, chapter 5's deep model on Walker Lake, %d iterations; "
          "scored against the exhaustive grid at every 20th node"
          % iterations)
    print("%-6s %-9s %8s %8s %10s %10s %8s"
          % ("model", "rule", "rmse", "crps", "cover 90%", "final ELBO",
             "time s"))
    for deep, rules in ((False, ("joint",)), (True, ("marginal", "joint"))):
        for rule in rules:
            start = time.perf_counter()
            model, grid = chapter5_model(rule, deep, iterations)
            seconds = time.perf_counter() - start
            rows = np.arange(grid.n_data)[::20]
            coords = np.asarray(grid.coordinates)[rows]
            truth = np.asarray(grid.variables["V"].get_measurements()[0]
                               )[rows, 0]
            target = geoml.data.PointData.from_array(coords, ["X", "Y"])
            model.predict(target, n_sim=50)
            variable = target.variables["V"]
            prediction = np.asarray(variable.prediction.values)
            sims = np.asarray(variable.get_simulations())
            _, covered = metrics.coverage(truth, sims, [0.9])
            print("%-6s %-9s %8.1f %8.1f %10.2f %10.1f %8.0f"
                  % ("deep" if deep else "flat", rule,
                     metrics.rmse(truth, prediction),
                     metrics.crps(truth, sims), covered[0],
                     model.training_log[-1], seconds))


def main(argv):
    command = argv[0] if argv else "kernel"
    if command == "chapter5":
        chapter5_gate(int(argv[1]) if len(argv) > 1 else 100)
        return
    if command == "kernel":
        kernel_gate()
    elif command == "folded":
        folded_gate(int(argv[1]) if len(argv) > 1 else 2000)
    elif command == "walk":
        folded_gate(int(argv[1]) if len(argv) > 1 else 2000,
                    networks=("vgp", "walk"))
    else:
        raise SystemExit("unknown command %r" % command)


if __name__ == "__main__":
    main(sys.argv[1:])
