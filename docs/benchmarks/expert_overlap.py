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

"""How `inducing.experts` divides points, against three compact
alternatives.

Usage: python docs/benchmarks/expert_overlap.py [Tom East.csv [--train]]

The 0.8.7 algorithm (`current`, kept here as it was) caps k-means
clusters at n/k points, placing points greedily in order of preference, and
lends each expert the points nearest its centre in its own Mahalanobis
metric (roadmap, section 2). Measured against it, each with a size band of
+-10% and the overlap borrowed by Euclidean distance to the nearest core
member -- `transport` being what `inducing.experts` does since:

- `trading`: the capped clusters, then points moved or swapped between
  clusters wherever that brings them nearer their centre, sizes kept in
  the band;
- `transport`: k-means whose assignment step is a transport problem with
  every cluster's size bounded by the band (Bradley, Bennett & Demiriz
  2000);
- `bisection`: the points split at a quantile of their principal axis,
  recursively, the quantile the share of experts on each side.

Geometry, per expert and reported as the worst over them: the core's
farthest member over its median radius, its most isolated member's
distance to the nearest fellow member over the core's median spacing, and
a borrowed point's distance to the nearest core member in the same
spacings; the expert sizes. With `--train`, a rock model per algorithm on
Tom East -- ore against waste, a fifth of the holes held out -- scored by
held-out AUC and Brier over three seeds.
"""

import sys
import time

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.spatial as spatial
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import geoml
from geoml.data import inducing


# --------------------------------------------------------------------- #
# the algorithms: points -> core labels
# --------------------------------------------------------------------- #
def current(points, k, band, seed):
    """0.8.7's assignment: k-means clusters capped at n/k points, each
    point offered its preferred cluster in order of how much the choice
    costs it."""
    n = len(points)
    centres = KMeans(k, n_init=10, random_state=seed).fit(
        points).cluster_centers_
    distance = _squared(points, centres)
    capacity = int(np.ceil(n / k))
    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(k, dtype=int)
    for i in np.argsort(distance.min(axis=1) - distance.max(axis=1)):
        for j in np.argsort(distance[i]):
            if counts[j] < capacity:
                labels[i], counts[j] = j, counts[j] + 1
                break
    return labels


def _covariance(members, floor=1e-3):
    """0.8.7's cluster covariance, its small eigenvalues floored."""
    n_dim = members.shape[1]
    if members.shape[0] > n_dim:
        covariance = np.atleast_2d(np.cov(members, rowvar=False))
    else:
        covariance = np.eye(n_dim) * max(float(members.var()), 1.0)
    values, vectors = np.linalg.eigh(covariance)
    if values.max() <= 0:
        return np.eye(n_dim)
    values = np.maximum(values, floor * values.max())
    return (vectors * values) @ vectors.T


def _bounds(n, k, band):
    return int(np.floor(n / k * (1 - band))), int(np.ceil(n / k * (1 + band)))


def _squared(points, centres):
    return ((points[:, None, :] - centres[None]) ** 2).sum(-1)


def trading(points, k, band, seed, sweeps=50):
    n = len(points)
    lo, hi = _bounds(n, k, band)
    labels = current(points, k, band, seed)
    for _ in range(sweeps):
        changed = 0
        centres = np.array([points[labels == j].mean(axis=0)
                            for j in range(k)])
        cost = _squared(points, centres)
        sizes = np.bincount(labels, minlength=k)
        # moves, the largest gain first, while the band allows them
        gain = cost[np.arange(n), labels] - cost.min(axis=1)
        for i in np.argsort(-gain):
            if gain[i] <= 0:
                break
            j, c = int(np.argmin(cost[i])), labels[i]
            if sizes[c] > lo and sizes[j] < hi:
                labels[i] = j
                sizes[c] -= 1
                sizes[j] += 1
                changed += 1
        # swaps between two clusters, each point preferring the other's
        for a in range(k):
            for b in range(a + 1, k):
                pa = np.flatnonzero(labels == a)
                pb = np.flatnonzero(labels == b)
                ga = cost[pa, a] - cost[pa, b]
                gb = cost[pb, b] - cost[pb, a]
                pa, ga = pa[np.argsort(-ga)], np.sort(ga)[::-1]
                pb, gb = pb[np.argsort(-gb)], np.sort(gb)[::-1]
                for m in range(min(len(pa), len(pb))):
                    if ga[m] + gb[m] <= 0:
                        break
                    labels[pa[m]], labels[pb[m]] = b, a
                    changed += 1
        if not changed:
            break
    return labels


def _bounded_assignment(cost, lo, hi):
    """The assignment of every point to one cluster minimizing `cost`,
    every cluster holding between `lo` and `hi` points: a transport
    problem, whose vertices are integral."""
    n, k = cost.shape
    rows = np.repeat(np.arange(n), k)
    cols = np.arange(n * k)
    each = sparse.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n * k))
    clusters = sparse.csr_matrix(
        (np.ones(n * k), (np.tile(np.arange(k), n), cols)), shape=(k, n * k))
    result = optimize.linprog(
        cost.ravel(), A_ub=sparse.vstack([clusters, -clusters]),
        b_ub=np.concatenate([np.full(k, hi), np.full(k, -lo)]),
        A_eq=each, b_eq=np.ones(n), bounds=(0, 1), method="highs")
    return np.argmax(result.x.reshape(n, k), axis=1)


def transport(points, k, band, seed, rounds=30):
    n = len(points)
    lo, hi = _bounds(n, k, band)
    centres = KMeans(k, n_init=10, random_state=seed).fit(
        points).cluster_centers_
    labels = None
    for _ in range(rounds):
        new = _bounded_assignment(_squared(points, centres), lo, hi)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        centres = np.array([points[labels == j].mean(axis=0)
                            for j in range(k)])
    return labels


def bisection(points, k, band, seed):
    labels = np.zeros(len(points), dtype=int)

    def split(index, k, first):
        if k == 1:
            labels[index] = first
            return
        centred = points[index] - points[index].mean(axis=0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        order = index[np.argsort(centred @ axis)]
        left = k // 2
        n_left = int(round(len(index) * left / k))
        split(order[:n_left], left, first)
        split(order[n_left:], k - left, first + left)

    split(np.arange(len(points)), k, 0)
    return labels


ALGORITHMS = dict(current=current, trading=trading, transport=transport,
                  bisection=bisection)


# --------------------------------------------------------------------- #
# the overlap and the measures
# --------------------------------------------------------------------- #
def borrowed(points, labels, k, overlap, mahalanobis):
    """Each expert's borrowed indices: by the core's Mahalanobis distance
    from its centre, as 0.8.7 does, or by Euclidean distance to the nearest
    core member."""
    out = []
    for j in range(k):
        core = labels == j
        outside = np.flatnonzero(~core)
        count = min(int(round(overlap * core.sum())), outside.size)
        members = points[core]
        if mahalanobis:
            chol = np.linalg.cholesky(_covariance(members))
            solved = np.linalg.solve(chol, (points - members.mean(axis=0)).T)
            distance = np.sqrt((solved ** 2).sum(axis=0))[outside]
        else:
            distance = spatial.cKDTree(members).query(points[outside])[0]
        out.append(outside[np.argsort(distance)[:count]])
    return out


def measure(points, labels, lent, k):
    rows = []
    for j in range(k):
        members = points[labels == j]
        radius = np.linalg.norm(members - members.mean(axis=0), axis=1)
        tree = spatial.cKDTree(members)
        nearest = tree.query(members, k=2)[0][:, 1]
        spacing = np.median(nearest)
        gaps = tree.query(points[lent[j]])[0] / spacing
        rows.append(dict(size=len(members),
                         farthest=radius.max() / np.median(radius),
                         isolated=nearest.max() / spacing,
                         borrowed=gaps.max() if gaps.size else 0.0))
    table = pd.DataFrame(rows)
    return dict(sizes="%d-%d" % (table["size"].min(), table["size"].max()),
                farthest=table["farthest"].max(),
                isolated=table["isolated"].max(),
                borrowed=table["borrowed"].max(),
                borrowed_median=table["borrowed"].median())


def geometry(name, points, ks=(5, 20), band=0.1, overlap=0.1, seed=0):
    rows = []
    for k in ks:
        for algorithm, divide in ALGORITHMS.items():
            start = time.perf_counter()
            labels = divide(points, k, band, seed)
            seconds = time.perf_counter() - start
            lent = borrowed(points, labels, k, overlap,
                            mahalanobis=algorithm == "current")
            rows.append(dict(k=k, algorithm=algorithm, seconds=seconds,
                             **measure(points, labels, lent, k)))
    table = pd.DataFrame(rows).set_index(["k", "algorithm"])
    print("== %s: %d points" % (name, len(points)))
    with pd.option_context("display.width", 250, "display.precision", 2,
                           "display.max_columns", None):
        print(table)


def expert_sets(points, labels, k, overlap, mahalanobis):
    lent = borrowed(points, labels, k, overlap, mahalanobis)
    return [geoml.data.PointData.from_array(
        points[np.union1d(np.flatnonzero(labels == j), lent[j])],
        ["X", "Y", "Z"]) for j in range(k)]


# --------------------------------------------------------------------- #
# the cases
# --------------------------------------------------------------------- #
def synthetic(seed=0):
    """Drillhole-like lines: a dense block of holes 10 m apart beside a
    sparse spread 60 m apart, samples every 2 m down holes dipping 60
    degrees, reduced to 500 k-means points."""
    rng = np.random.default_rng(seed)
    collars = np.vstack([
        np.column_stack([g.ravel() for g in np.meshgrid(
            np.arange(0, 100, 10.0), np.arange(0, 100, 10.0))]),
        np.column_stack([g.ravel() for g in np.meshgrid(
            np.arange(150, 600, 60.0), np.arange(0, 600, 60.0))])])
    collars = collars + rng.normal(0, 2, collars.shape)
    down = np.arange(0, 200, 2.0)
    direction = np.array([0.0, np.cos(np.radians(60)),
                          -np.sin(np.radians(60))])
    samples = np.vstack([np.column_stack([c[0] + 0 * down, c[1] + 0 * down,
                                          0 * down]) + down[:, None]
                         * direction for c in collars])
    data = geoml.data.PointData.from_array(samples, ["X", "Y", "Z"])
    return np.asarray(inducing.from_kmeans(data, 500, seed=0).coordinates)


def tom_east(path):
    frame = pd.read_csv(path)
    rng = np.random.default_rng(0)
    holes = frame["HOLEID"].unique()
    held = frame["HOLEID"].isin(rng.choice(holes, len(holes) // 5,
                                           replace=False)).values
    train = frame[~held].reset_index(drop=True)
    data = geoml.data.PointData(train, ["X", "Y", "Z"])
    data.add_categorical_variable("rock", labels=["Waste", "Tom East"],
                                  measurements=train["Code_Simple"].values)
    test = frame[held].reset_index(drop=True)
    return data, test


def train(data, test, sets, seed):
    import geoml.latent as latent
    import geoml.likelihood as lk
    import geoml.transform as tr
    geoml.set_seed(seed)
    root = latent.BasicInput(
        sets, transform=tr.Anisotropy3D(100, 0.75, 0.5, 345, 15, 70))
    leaf = latent.BasicGP(root, size=2, kernel=geoml.kernels.Matern32())
    model = geoml.models.VGPNetwork(
        data, "rock", lk.CategoricalGaussianIndicator(2), leaf,
        options=geoml.models.GPOptions(verbose=False))
    start = time.perf_counter()
    model.train_full(300)
    seconds = time.perf_counter() - start
    target = geoml.data.PointData(test, ["X", "Y", "Z"])
    model.predict(target, n_sim=10)
    p = np.asarray(target.variables["rock"].components["Tom East"]
                   .probability.values)
    truth = (test["Code_Simple"] == "Tom East").values
    return dict(auc=roc_auc_score(truth, p),
                brier=float(np.mean((p - truth) ** 2)), seconds=seconds)


def scores(path, ks=(5, 20), band=0.1, overlap=0.1, seeds=(1, 2, 3)):
    data, test = tom_east(path)
    points = np.asarray(inducing.from_kmeans(data, 500, seed=0).coordinates)
    rows = []
    for k in ks:
        for algorithm, divide in ALGORITHMS.items():
            labels = divide(points, k, band, 0)
            sets = expert_sets(points, labels, k, overlap,
                               mahalanobis=algorithm == "current")
            for seed in seeds:
                rows.append(dict(k=k, algorithm=algorithm, seed=seed,
                                 **train(data, test, sets, seed)))
                print(rows[-1], flush=True)
    table = pd.DataFrame(rows).groupby(["k", "algorithm"])[
        ["auc", "brier", "seconds"]].agg(["mean", "std"])
    with pd.option_context("display.width", 250, "display.precision", 3,
                           "display.max_columns", None):
        print(table)


def figure(name, points, k, path, band=0.1, overlap=0.1, seed=0):
    """One column per algorithm: the cores in plan and in a vertical
    section along the survey's long horizontal axis, each core's convex
    hull outlined, and a line from every borrowed point to the nearest
    member of the core that borrowed it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    centred = points - points.mean(axis=0)
    axis = np.linalg.svd(centred[:, :2], full_matrices=False)[2][0]
    along = centred[:, :2] @ axis
    views = (("plan", centred[:, 0], centred[:, 1], "east (m)", "north (m)"),
             ("section", along, centred[:, 2], "along the long axis (m)",
              "elevation (m)"))
    colours = plt.get_cmap("tab20" if k > 10 else "tab10")
    fig, axes = plt.subplots(2, len(ALGORITHMS), figsize=(5 * len(ALGORITHMS),
                                                          10), squeeze=False)
    for c, (algorithm, divide) in enumerate(ALGORITHMS.items()):
        labels = divide(points, k, band, seed)
        lent = borrowed(points, labels, k, overlap,
                        mahalanobis=algorithm == "current")
        for r, (view, u, v, xlabel, ylabel) in enumerate(views):
            ax = axes[r, c]
            uv = np.column_stack([u, v])
            for j in range(k):
                core = labels == j
                colour = colours(j % colours.N)
                ax.scatter(u[core], v[core], s=6, color=colour, linewidths=0)
                if core.sum() >= 3:
                    hull = ConvexHull(uv[core])
                    ring = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(uv[core][ring, 0], uv[core][ring, 1], color=colour,
                            linewidth=1)
                members = np.flatnonzero(core)
                tree = spatial.cKDTree(points[members])
                nearest = members[tree.query(points[lent[j]])[1]]
                for a, b in zip(lent[j], nearest):
                    ax.plot([u[a], u[b]], [v[a], v[b]], color="black",
                            linewidth=0.6)
            ax.set_aspect("equal")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title("%s, %s" % (algorithm, view))
    fig.suptitle("%s: %d experts; outlines are cores, black lines join a "
                 "borrowed point to the nearest member of its borrower"
                 % (name, k))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv):
    geometry("synthetic, uneven density and lines", synthetic())
    if argv:
        data, _ = tom_east(argv[0])
        geometry("Tom East, 500 k-means points", np.asarray(
            inducing.from_kmeans(data, 500, seed=0).coordinates))
        if "--train" in argv:
            scores(argv[0])
        if "--figures" in argv:
            points = np.asarray(
                inducing.from_kmeans(data, 500, seed=0).coordinates)
            for k in (5, 20):
                figure("Tom East", points, k,
                       "local_data/experts_tom_east_%d.png" % k)
                figure("Synthetic", synthetic(), k,
                       "local_data/experts_synthetic_%d.png" % k)


if __name__ == "__main__":
    main(sys.argv[1:])
