"""Figures of the `LikelihoodMixture` gates, 2026-10-01.

Usage: python docs/benchmarks/likelihood_mixture_figures.py [Tom East.csv]
       python docs/benchmarks/likelihood_mixture_figures.py latent [csv]

Writes `docs/benchmarks/figures/likelihood_mixture_gates.png` -- the
crossing, split and skew gates -- and, given the Tom East file, a figure of
its own into `local_data/`, which the repository never holds, since the
data are not ours to publish.

With `latent`, the phase 2 figures instead: the split gate under shares
that change from place to place (`likelihood_mixture_latent.png`), and,
given the file, maps of Tom East -- a long section and a plan through the
deposit -- and the out-of-fold scores, both into `local_data/`.
"""
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

import gaussian_mixture_gates as gates                  # noqa: E402
import geoml                                            # noqa: E402


def gate_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 10.0, 400)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # crossing: the measurements by true curve, realizations by population
    model, container, which = gates.mixture_model("cross")
    target = geoml.data.PointData.from_array(x[:, None], ["X"])
    model.predict(target, n_sim=20)
    sims = target.variables["v"].get_simulations()
    labels = model.likelihoods[0].labels(20)
    xd = np.asarray(container.coordinates)[:, 0]
    yd = container.values("v/measurements")
    ax = axes[0, 0]
    for s in range(20):
        ax.plot(x, sims[:, s], lw=0.6, alpha=0.6,
                color="tab:orange" if labels[s] else "tab:blue")
    ax.scatter(xd, yd, c=np.where(which, "k", "grey"), s=10, zorder=3)
    ax.set_title("Crossing curves: 20 realizations, coloured by population;\n"
                 "measurements black / grey by the curve they came from")

    # split: the mixture against one GP
    ax = axes[0, 1]
    a, b = gates.curves(x)
    truth = np.where(x >= gates.BOUNDARY, b, a)
    for arm, colour in (("gp", "tab:green"), ("mixture", "tab:red")):
        if arm == "gp":
            m = gates.trained("gp", "split")
        else:
            m, _, _ = gates.mixture_model("split")
        t = geoml.data.PointData.from_array(x[:, None], ["X"])
        m.predict(t, n_sim=100)
        s = t.variables["v"].get_simulations()
        ax.fill_between(x, *np.percentile(s, [5, 95], axis=1),
                        color=colour, alpha=0.2)
        ax.plot(x, t.values("v/prediction"), color=colour,
                label="one GP" if arm == "gp" else "mixture, fixed shares")
    ax.plot(x, truth, "k--", lw=1, label="ground")
    ax.legend(fontsize=8)
    ax.set_title("Split regimes: prediction and 90% band")

    # skew: separate warpings against one shared warping
    for col, arm in enumerate(("separate", "shared")):
        m, c, w = gates.mixture_model("skew", arm)
        first = m.responsibilities(c, store=False)["v"][:, 0] > 0.5
        hit = first == w
        if hit.mean() < 0.5:
            hit = ~hit
        ax = axes[1, col]
        xs = np.asarray(c.coordinates)[:, 0]
        ys = c.values("v/measurements")
        ax.scatter(xs[hit], ys[hit], s=12, c="tab:blue",
                   label="assigned to its own population")
        ax.scatter(xs[~hit], ys[~hit], s=24, c="tab:red", marker="x",
                   label="assigned to the other")
        ax.set_title("Two populations, %s warping: %.0f%% assigned right"
                     % ("a separate" if arm == "separate" else "one shared",
                        100 * hit.mean()))
        ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("wrote", path)


def tom_east_figure(csv, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tom_east_mixture as tom

    data, _ = tom.load(csv)
    m = tom.model("mixture", data)
    shares = m.responsibilities(data, store=False)["metals"]
    ore = np.asarray(data.get_metadata("ore"), dtype=bool)
    first = shares[:, 0] > 0.5
    if np.mean(first == ore) < 0.5:
        shares = shares[:, ::-1]
    values = data.values("metals/Ag/measurements"), \
        data.values("metals/Zn/measurements")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, colour, title in (
            (axes[0], np.where(ore, "tab:red", "tab:blue"),
             "Logged rock type: red Tom East, blue Waste"),
            (axes[1], shares[:, 1],
             "The mixture's share of the second population")):
        drawn = ax.scatter(values[0], values[1], c=colour, s=8,
                           cmap="coolwarm", vmin=0, vmax=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Ag, g/t")
        ax.set_ylabel("Zn, %")
        ax.set_title(title)
    fig.colorbar(drawn, ax=axes[1])

    # out-of-fold CRPS, as measured by tom_east_mixture.py on 2026-10-01
    crps = {"recommended chain": (68.3, 3.96, 4.17),
            "plain BoxCox, ZScore": (52.4, 4.20, 3.90),
            "mixture": (49.4, 3.67, 3.10)}
    ax = axes[2]
    base = np.asarray(crps["recommended chain"])
    for i, (arm, row) in enumerate(crps.items()):
        ax.bar(np.arange(3) + 0.27 * (i - 1), np.asarray(row) / base, 0.27,
               label=arm)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Ag", "Pb", "Zn"])
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_ylabel("out-of-fold CRPS, against the recommended chain")
    ax.set_title("Folds by hole: lower is better")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("wrote", path)


def latent_gate_figure(path):
    """The split gate under latent shares: the prediction against one GP,
    each realization coloured by the population it takes at each place,
    and the expected share beside the fraction of realizations in the
    second population."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 10.0, 400)
    a, b = gates.curves(x)
    truth = np.where(x >= gates.BOUNDARY, b, a)
    n_sim = 100
    model, container, _ = gates.mixture_model("split", "latent")
    amplitude = float(model.likelihoods[0].parameters["amplitude"]
                      .get_value())
    target = geoml.data.PointData.from_array(x[:, None], ["X"])
    model.predict(target, n_sim=n_sim)
    sims = target.variables["v"].get_simulations()
    labels = np.asarray(target.variables["v"].population)
    share = target.variables["v"].responsibilities[1].values.to_numpy()
    xd = np.asarray(container.coordinates)[:, 0]
    yd = container.values("v/measurements")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    gp = gates.trained("gp", "split")
    t = geoml.data.PointData.from_array(x[:, None], ["X"])
    gp.predict(t, n_sim=n_sim)
    for values, prediction, colour, name in (
            (t.variables["v"].get_simulations(), t.values("v/prediction"),
             "tab:green", "one GP"),
            (sims, target.values("v/prediction"), "tab:red",
             "mixture, latent shares")):
        ax.fill_between(x, *np.percentile(values, [5, 95], axis=1),
                        color=colour, alpha=0.2)
        ax.plot(x, prediction, color=colour, label=name)
    ax.plot(x, truth, "k--", lw=1, label="ground")
    ax.scatter(xd, yd, s=6, c="grey", zorder=3)
    ax.legend(fontsize=8)
    ax.set_title("Prediction and 90% band")

    ax = axes[1]
    for s in range(20):
        ax.scatter(x, sims[:, s], s=1.5, c=np.where(
            labels[:, s] == 1, "tab:orange", "tab:blue"))
    ax.axvline(gates.BOUNDARY, color="k", lw=0.6, ls=":")
    ax.set_title("20 realizations, each point coloured by the population\n"
                 "it takes there: blue the first, orange the second")

    ax = axes[2]
    ax.plot(x, share, color="tab:red", label="expected share")
    ax.plot(x, labels.mean(axis=1), color="k", lw=0.8, ls="--",
            label="fraction of %d realizations" % n_sim)
    # which population takes the right-hand regime is the model's choice
    right = (x >= gates.BOUNDARY).astype(float)
    if share[x >= gates.BOUNDARY].mean() < 0.5:
        right = 1 - right
    ax.step(x, right, color="grey", lw=0.8, where="post", label="ground")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("The second population's share; amplitude %.1f"
                 % amplitude)
    for ax in axes:
        ax.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("wrote", path)


def _section_points(data, fixed, axis, step, slab):
    """A regular sheet through the assayed intervals, `axis` held at
    `fixed`; with the intervals within `slab` of it."""
    xyz = np.asarray(data.coordinates)
    free = [i for i in range(3) if i != axis]
    lo = np.floor(xyz[:, free].min(axis=0) / step) * step - 2 * step
    hi = np.ceil(xyz[:, free].max(axis=0) / step) * step + 2 * step
    u = np.arange(lo[0], hi[0] + step / 2, step)
    v = np.arange(lo[1], hi[1] + step / 2, step)
    uu, vv = np.meshgrid(u, v)
    points = np.zeros([uu.size, 3])
    points[:, free[0]], points[:, free[1]] = uu.ravel(), vv.ravel()
    points[:, axis] = fixed
    near = np.abs(xyz[:, axis] - fixed) <= slab
    return points, uu.shape, (u, v), free, near


def tom_east_maps(csv, path, arm="latent_rock"):
    """A long section and a plan through Tom East: the rock type's
    probability (where the arm models it), the expected share of the ore
    population, the Zn prediction, and one realization's populations;
    faded more than 40 m from any assayed interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    import tom_east_mixture as tom

    data, _ = tom.load(csv)
    if arm == "latent_rock":
        data.spatial_k_fold(data, k=5, groups="HOLEID")
        m = tom.model(arm, tom.load_logged(csv, data), assayed=data)
    else:
        m = tom.model(arm, data)
    with_rock = arm == "latent_rock"
    n_sim = 20
    # the realization whose point sits mid-way: one at either end takes the
    # same population almost everywhere
    shown = int(np.argmin(np.abs(
        geoml.likelihood._interleaved_points(n_sim) - 0.5)))
    xyz = np.asarray(data.coordinates)
    ore = np.asarray(data.get_metadata("ore"), dtype=bool)
    zn = data.values("metals/Zn/measurements")
    tree = cKDTree(xyz)
    centre = xyz.mean(axis=0)
    views = [("Long section, X = %.0f" % centre[0], 0, centre[0]),
             ("Plan, Z = %.0f" % centre[2], 2, centre[2])]
    names = "XYZ"

    columns = 4 if with_rock else 3
    fig, axes = plt.subplots(2, columns, figsize=(5 * columns, 10.5))
    for row, (title, axis, fixed) in enumerate(views):
        points, shape, (u, v), free, near = _section_points(
            data, fixed, axis, 4.0, 15.0)
        target = geoml.data.PointData.from_array(points, ["X", "Y", "Z"])
        m.predict(target, n_sim=n_sim)
        metals = target.variables["metals"]
        share = metals.responsibilities[1].values.to_numpy()
        prediction = target.values("metals/Zn/prediction")
        population = np.asarray(metals.population)[:, shown].astype(float)
        far = tree.query(points)[0] > 40.0
        extent = (u[0], u[-1], v[0], v[-1])
        layers = [
            (share, "Expected share of the ore population", "coolwarm",
             dict(vmin=0, vmax=1)),
            (prediction, "Zn prediction, %", "viridis",
             dict(norm=colors.LogNorm(0.05, 20))),
            (population, "Realization %d: its population" % shown,
             "coolwarm",
             dict(vmin=0, vmax=1))]
        if with_rock:
            rock = target.variables["rock"].components["Tom East"]\
                .probability.values.to_numpy()
            layers.insert(0, (rock, "Rock type: probability of Tom East",
                              "coolwarm", dict(vmin=0, vmax=1)))
        dots = xyz[near][:, free]
        for col, (values, name, cmap, scale) in enumerate(layers):
            ax = axes[row, col]
            grid = np.ma.masked_invalid(values.reshape(shape))
            drawn = ax.imshow(grid, origin="lower", extent=extent,
                              cmap=cmap, aspect="equal", **scale)
            ax.imshow(np.where(far.reshape(shape), 1.0, np.nan),
                      origin="lower", extent=extent, cmap="Greys",
                      vmin=0, vmax=1.4, alpha=0.75, aspect="equal")
            if name.startswith("Zn"):
                ax.scatter(*dots.T, c=zn[near], s=10, cmap=cmap,
                           edgecolors="k", linewidths=0.3,
                           norm=colors.LogNorm(0.05, 20))
            else:
                ax.scatter(*dots.T, c=np.where(ore[near], "tab:red",
                                               "tab:blue"),
                           s=10, edgecolors="k", linewidths=0.3)
            ax.set_xlabel(names[free[0]])
            ax.set_ylabel(names[free[1]])
            ax.set_title("%s\n%s" % (title, name), fontsize=10)
            fig.colorbar(drawn, ax=ax, shrink=0.75)
    fig.suptitle("Tom East, a mixture of two populations with shares %s. "
                 "Dots: assayed intervals within 15 m, red Tom East, blue "
                 "Waste (Zn on the Zn maps); faded beyond 40 m of any assay"
                 % ("tied to the logged rock type" if with_rock
                    else "from the metals alone"), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print("wrote", path)


def tom_east_scores(path):
    """The out-of-fold scores of 2026-10-01, folds by hole built against
    the assayed intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crps = {"one Gaussian, recommended chain": (84.6, 4.25, 3.48),
            "one Gaussian, BoxCox, ZScore": (46.0, 3.77, 3.45),
            "mixture, fixed shares": (46.7, 3.32, 2.89),
            "mixture, latent shares": (45.6, 3.28, 2.71),
            "mixture, shares tied to rock": (45.8, 3.30, 2.83)}
    # the rock-tied arm, its rock type trained on every logged interval:
    # balanced accuracy out of fold, the mean of the two categories' recall
    balanced = {"assayed (798)": (0.568, 0.640),
                "all logged (4585)": (0.598, 0.734)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    base = np.asarray(crps["one Gaussian, recommended chain"])
    width = 0.16
    for i, (arm, row) in enumerate(crps.items()):
        ax.bar(np.arange(3) + width * (i - 2), np.asarray(row) / base,
               width, label=arm)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Ag", "Pb", "Zn"])
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_ylabel("out-of-fold CRPS, against the recommended chain")
    ax.set_title("Lower is better")
    ax.set_ylim(0, 1.6)
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax = axes[1]
    rows = np.arange(len(balanced))
    for i, (name, colour) in enumerate((("rock type's own call", "tab:red"),
                                        ("expected shares", "tab:blue"))):
        ax.barh(rows + 0.38 * (i - 0.5),
                [pair[i] for pair in balanced.values()], 0.38,
                color=colour, label=name)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--", label="chance")
    ax.set_yticks(rows)
    ax.set_yticklabels(list(balanced))
    ax.set_xlim(0.4, 0.8)
    ax.set_xlabel("out of fold, balanced accuracy against Code_Simple")
    ax.set_title("Shares tied to a rock type trained on every logged "
                 "interval")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Tom East, five folds by hole (160 / 160 / 160 / 159 / "
                 "159 intervals)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    local = os.path.join(here, "..", "..", "local_data")
    if sys.argv[1:2] == ["latent"]:
        latent_gate_figure(os.path.join(
            here, "figures", "likelihood_mixture_latent.png"))
        if len(sys.argv) > 2:
            tom_east_scores(os.path.join(local, "tom_east_scores.png"))
            tom_east_maps(sys.argv[2],
                          os.path.join(local, "tom_east_maps.png"))
            tom_east_maps(sys.argv[2],
                          os.path.join(local, "tom_east_maps_metals.png"),
                          arm="latent")
        sys.exit()
    gate_figure(os.path.join(here, "figures", "likelihood_mixture_gates.png"))
    if len(sys.argv) > 1:
        tom_east_figure(sys.argv[1], os.path.join(
            here, "..", "..", "local_data", "tom_east_mixture.png"))
