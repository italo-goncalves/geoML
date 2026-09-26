"""The two gates of `latent.GaussianMixture`, 2026-09-25.

Roadmap item L. One-dimensional synthetic data, 120 noisy samples of
`sin(2 pi x / 4)` on [0, 10] at noise 0.1, scored against the noiseless
truth on 400 points of the line.

- **split**: the sinusoid below x = 5 and the same in opposite phase above
  it, a jump of two at the boundary. The mixture must beat one GP on RMSE
  and CRPS and put its switch at the boundary.
- **cross**: every sample drawn from one of the two at random, so the two
  curves cross throughout. The prediction should be bimodal away from the
  crossings: at least a fifth of the realizations near each curve.

Each arm names what differs: `gp` one GP on a root of range 2; `shared` the
mixture with its weights and components on that root; `smooth` components
on a root of range 4 and weights of range 6; `samples50` that at 50
training draws instead of 20. The node's own `amplitude` sharpens the
softmax in every mixture arm.

Findings (split): the shared root collapses onto one component (share 1.00
everywhere, the GP's scores); smooth components switch at 4.79, RMSE 0.161
and CRPS 0.043 against the GP's 0.192 and 0.078, the amplitude at its
ceiling of 100. A component as flexible as the one GP can fit the jump on
its own, so nothing asks the weights to switch; one smoother than the jump
cannot. Drawn (`figure`), one component follows the data left of the
boundary and again past 7.5, and the other only bridges [5, 7].

`deep` and `deep_spherical` are the construction in use (manual chapter
5): the input and an inner two-column GP concatenated under an outer GP,
the outer kernel the default or spherical. On the split, 0.151 / 0.049 and
0.162 / 0.052: as close to the jump as the mixture, overshooting it inside
too narrow a band, so the mixture keeps the better CRPS. Not bimodal on
the crossing curves either. The cross gate fails: the mixture blends
the two curves into their mean, no realization near either (the bound
trains on `E_q[log p(y | f)]`, which rewards agreeing with each sample, and
the weights never learn to switch sample by sample).

Usage:  python docs/benchmarks/gaussian_mixture_gates.py
        python docs/benchmarks/gaussian_mixture_gates.py figure
"""
import time

import numpy as np
import tensorflow as tf

import geoml
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.metrics as metrics
import geoml.transform as tr

PERIOD, BOUNDARY, NOISE, N, ITERATIONS = 4.0, 5.0, 0.1, 120, 1000


def curves(x):
    a = np.sin(2 * np.pi * x / PERIOD)
    return a, -a


def samples(case, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 10.0, N))
    a, b = curves(x)
    which = x >= BOUNDARY if case == "split" else rng.uniform(size=N) < 0.5
    return x, np.where(which, b, a) + rng.normal(0.0, NOISE, N)


def data(case):
    x, y = samples(case)
    container = geoml.data.PointData.from_array(x[:, None], ["X"])
    container.add_continuous_variable("v", y)
    return container


def root(range_):
    grid = geoml.data.Grid1D(start=-0.5, n=45, step=0.25)
    return latent.BasicInput(grid, transform=tr.Isotropic(range_))


def leaf(arm, case):
    geoml.set_seed(1234)
    if arm == "gp":
        return latent.BasicGP(root(2.0), size=1)
    if arm in ("deep", "deep_spherical"):
        # the construction in use (manual chapter 5): an inner GP warping
        # the input, read by an outer GP beside the real coordinates
        base = root(2.0)
        inner = latent.BasicGP(base, size=2, kernel=geoml.kernels.Gaussian())
        outer_kernel = geoml.kernels.Spherical() \
            if arm == "deep_spherical" else None
        return latent.BasicGP(latent.Concatenate(base, inner), size=1,
                              kernel=outer_kernel)
    if arm == "shared":
        base = root(2.0)
        weights, components = latent.BasicGP(base, size=2), base
    else:
        weights_range = 0.2 if case == "cross" else 6.0
        weights = latent.BasicGP(root(weights_range), size=2)
        components = root(4.0)
    return latent.GaussianMixture(
        weights, [latent.BasicGP(components, size=1) for _ in range(2)])


def trained(arm, case):
    draws = 50 if arm == "samples50" else 20
    model = geoml.models.VGPNetwork(
        data(case), "v", lk.Gaussian(), leaf(arm, case),
        options=geoml.models.GPOptions(verbose=False,
                                       training_samples=draws))
    model.train_full(max_iter=ITERATIONS)
    return model


def run(arm, case):
    start = time.time()
    model = trained(arm, case)
    seconds = time.time() - start

    x = np.linspace(0.0, 10.0, 400)
    a, b = curves(x)
    target = geoml.data.PointData.from_array(x[:, None], ["X"])
    model.predict(target, n_sim=200)
    prediction = np.asarray(target.values("v/prediction"), dtype=float)
    sims = np.asarray(target.variables["v"].get_simulations())
    log = np.asarray(model.training_log)
    row = {"arm": arm, "case": case, "seconds": seconds,
           "bound": log[-1],
           # the bound's own noise: the spread of its steps once settled
           "bound_noise": np.std(np.diff(log[-200:]))}

    if case == "split":
        truth = np.where(x >= BOUNDARY, b, a)
        row["rmse"] = np.sqrt(np.mean((prediction - truth) ** 2))
        row["crps"] = metrics.crps(truth, sims)
        if isinstance(model.leaves[0], latent.GaussianMixture):
            node = model.leaves[0]
            model._refresh(1e-6)
            mu, _ = node.weights.propagate(tf.constant(x[:, None]))
            first = np.asarray(tf.nn.softmax(mu, axis=1))[:, 0]
            changes = x[1:][np.diff(first > 0.5) != 0]
            row["switch"] = changes[np.argmin(np.abs(changes - BOUNDARY))] \
                if len(changes) else np.nan
    else:
        apart = np.abs(a - b) > 0.75
        near_a = (np.abs(sims - a[:, None]) < 0.25).mean(axis=1)[apart]
        near_b = (np.abs(sims - b[:, None]) < 0.25).mean(axis=1)[apart]
        row["crps_a"] = metrics.crps(a, sims)
        row["bimodal"] = np.mean((near_a >= 0.2) & (near_b >= 0.2))
        row["near_a"], row["near_b"] = near_a.mean(), near_b.mean()
    return row


def figure(path="docs/benchmarks/figures/gaussian_mixture_gate1.png"):
    """The split gate drawn: one GP, the mixture, and the mixture's share
    of its first component along the line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geoml.plots.style as style

    xd, yd = samples("split")
    x = np.linspace(0.0, 10.0, 400)
    a, b = curves(x)
    truth = np.where(x >= BOUNDARY, b, a)
    fits = {}
    for arm in ("gp", "deep", "smooth"):
        model = trained(arm, "split")
        target = geoml.data.PointData.from_array(x[:, None], ["X"])
        model.predict(target, n_sim=200)
        fits[arm] = (model,
                     np.asarray(target.values("v/prediction"), dtype=float),
                     np.asarray(target.variables["v"].get_simulations()))

    model, _, _ = fits["smooth"]
    node = model.leaves[0]
    model._refresh(1e-6)
    grid = tf.constant(x[:, None], tf.float64)
    _, _, weights, _ = node.weights.predict(grid, n_sim=200, seed=[1, 0])
    amplitude = node.parameters["amplitude"].get_value()
    shares = np.asarray(tf.nn.softmax(weights * tf.sqrt(amplitude),
                                      axis=0))[0]
    # each component's mean in the data's units: the latent field back
    # through the likelihood's warping
    warping = model.likelihoods[0].warping
    components = [np.asarray(warping.backward(c.propagate(grid)[0]))[:, 0]
                  for c in node.components]

    with style.context():
        fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 3, 3, 2]})
        titles = {"gp": "One GP", "deep": "Deep GP (input and inner GP)",
                  "smooth": "Gaussian mixture"}
        for ax, arm in zip(axes, ("gp", "deep", "smooth")):
            _, prediction, sims = fits[arm]
            low, high = np.percentile(sims, [5, 95], axis=1)
            rmse = np.sqrt(np.mean((prediction - truth) ** 2))
            ax.fill_between(x, low, high, color=style.color(0), alpha=0.25,
                            lw=0, label="90% of the realizations")
            for s in range(3):
                ax.plot(x, sims[:, s], color=style.color(0), lw=0.6,
                        alpha=0.6, label="realizations" if s == 0 else None)
            ax.plot(x, prediction, color=style.color(0), lw=1.8,
                    label="prediction")
            ax.plot(x, truth, color="black", lw=1.0, ls="--", label="truth")
            if arm == "smooth":
                for k, c in enumerate(components):
                    ax.plot(x, c, color=style.color(k + 2), lw=1.0, ls=":",
                            label="component %d" % (k + 1))
            ax.scatter(xd, yd, s=9, color="0.35", zorder=3, label="data")
            ax.axvline(BOUNDARY, color="0.5", lw=0.8)
            ax.set_ylim(-2.2, 2.2)
            ax.set_ylabel("v")
            ax.set_title("%s: RMSE %.3f, CRPS %.3f against the truth"
                         % (titles[arm], rmse, metrics.crps(truth, sims)))
            ax.legend(loc="lower left", fontsize=7, ncol=3)
        ax = axes[3]
        low, high = np.percentile(shares, [5, 95], axis=1)
        ax.fill_between(x, low, high, color=style.color(2), alpha=0.25, lw=0,
                        label="90% of the realizations")
        ax.plot(x, shares.mean(axis=1), color=style.color(2), lw=1.8,
                label="mean share")
        ax.axvline(BOUNDARY, color="0.5", lw=0.8, label="boundary")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("share of component 1")
        ax.set_xlabel("x")
        ax.set_title("The weights' softmax, amplitude %.2f"
                     % float(np.ravel(amplitude)[0]))
        ax.legend(loc="center left", fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["figure"]:
        figure()
        sys.exit()
    for case, arms in (("split", ("gp", "deep", "deep_spherical", "shared",
                                  "smooth", "samples50")),
                       ("cross", ("gp", "deep", "deep_spherical",
                                  "smooth"))):
        for arm in arms:
            row = run(arm, case)
            print("  ".join("%s=%s" % (k, ("%.3f" % v) if isinstance(
                v, float) else v) for k, v in row.items()), flush=True)
