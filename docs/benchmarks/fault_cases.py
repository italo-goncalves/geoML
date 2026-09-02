"""Synthetic cases for the fault transforms, with figures.

Run from the repository root; figures land beside this file.

A: the notebook's case -- a horizontal contact offset by 3 across an
   inclined fault, rock types; base model, BellFault2D, ImplicitFault and
   FaultDisplacement side by side.
B: a grade that is a vertical gradient, offset along a curved fault whose
   normals are derived from its trace; plain, repulsion and displacement.
C: two faults, the younger cutting the older; plain, two repulsions, and
   a FaultNetwork that restores youngest first.
D: two curvilinear faults ending on an older curvilinear fault; plain,
   three repulsion coordinates behind a Linear head, the network with the
   abutting declared, and the same network without it as the control.
"""
import os
import sys
import time

import numpy as np

# the repository root, so the working tree's geoml is what runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent as gl  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.transform as tr  # noqa: E402
from geoml.math.geometry import point_normals  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
N_GRID = 121


def image(values, n=N_GRID):
    """A flat grid field as an image: a Grid2D orders x fastest, so the
    flat field reshapes to rows of y, as `as_image` does."""
    return np.asarray(values, dtype=float).reshape(n, n)


def grid(n=N_GRID):
    return geoml.data.Grid2D(start=[-5, -5], end=[5, 5], n=[n, n])


def train(model, phased=False, iterations=1000):
    started = time.monotonic()
    if phased:
        model.set_learning_rate(0.1)
        model.train_full(iterations // 2)
        model.set_learning_rate(0.01)
        model.train_full(iterations // 2)
    else:
        model.train_full(iterations)
    return time.monotonic() - started


def throw_of(transform):
    for t in getattr(transform, "transforms", [transform]):
        if "throw" in t.parameters:
            return float(t.parameters["throw"].get_value())
    return float("nan")


def draw_search(ax, throws, scores, best, true, label):
    ax.plot(throws, scores, "o-", color="k", ms=4)
    ax.axvline(true, color="tab:green", lw=1, label="true %.2f" % true)
    ax.axvline(best, color="tab:red", lw=1, ls="--", label="chosen %.2f" % best)
    ax.set_xlabel("candidate throw along the fault"); ax.set_ylabel("bound after a 40-iteration burst")
    ax.set_title(label, fontsize=9); ax.legend(fontsize=8)


# ------------------------------------------------------------------ case A
def case_a():
    print("== case A: the notebook's contact ==", flush=True)
    fault_end = np.array([2.0, -5.0]) * 3          # 5x + 2y = 0
    fault_start = -fault_end
    reject = 3.0
    coords = geoml.data.Grid2D(start=[-5, -5], end=[5, 5], n=[6, 6]).coordinates
    coords = np.asarray(coords, dtype=float)
    n_data = coords.shape[0]
    side = (-fault_end[1] * coords[:, 0] + fault_end[0] * coords[:, 1]) < 0
    label_0 = np.where(coords[:, 1] > 0, "top", "bottom")
    label_1 = np.where(coords[:, 1] > reject, "top", "bottom")
    final = np.where(side, label_0, label_1)
    c0 = np.stack([np.linspace(-5, 5, 11), np.zeros(11)], axis=1)
    c0 = c0[(-fault_end[1] * c0[:, 0] + fault_end[0] * c0[:, 1]) < 0]
    c1 = np.stack([np.linspace(-5, 5, 11), np.zeros(11) + reject], axis=1)
    c1 = c1[(-fault_end[1] * c1[:, 0] + fault_end[0] * c1[:, 1]) > 0]
    contacts = np.concatenate([c0, c1])
    all_coords = np.concatenate([coords, c0, c1])
    top = np.concatenate([final, ["top"] * len(contacts)])
    bottom = np.concatenate([final, ["bottom"] * len(contacts)])

    # the fault as observations: points along the line, normals toward
    # the side where the contact sits higher (5x + 2y > 0)
    t = np.linspace(-1.0, 1.0, 25)[:, None]
    fault_points = t * fault_end[None, :]
    normal = np.array([5.0, 2.0]) / np.sqrt(29.0)
    fault_normals = np.tile(normal, [25, 1])

    def data(name):
        pts = geoml.data.PointData.from_array(all_coords, ["X", "Y"])
        pts.add_rock_type_variable(name, labels=["top", "bottom"],
                                   measurements_a=top, measurements_b=bottom)
        return pts

    def categorical(transform, deep, phased=False, search=None):
        # as in the notebook: a Linear head between a concatenated transform
        # and the GP lets the model learn anisotropy in the expanded space
        root = gl.BasicInput(grid(11), transform)
        if deep:
            gp = gl.BasicGP(gl.Linear(root, size=3), kernel=kr.Cubic())
        else:
            gp = gl.BasicGP(root, kernel=kr.Cubic(), fix_range=True)
        net = gl.Linear(gp, size=2)
        pts = data("rock")
        model = geoml.models.VGPNetwork(
            pts, "rock", lk.CategoricalGaussianIndicator(2), net,
            options=geoml.models.GPOptions(verbose=False))
        searches = []
        for fault, candidates in (search or []):
            best, scores = geoml.models.search_throw(model, fault, candidates, iterations=40)
            searches.append((candidates, scores, best))
        seconds = train(model, phased=phased)
        g = grid()
        model.predict(g)
        prob = g.variables["rock"].components["top"].probability.as_image()
        return prob, root, seconds, searches

    def aniso():
        a = tr.Anisotropy2D(3, 0.5, 90)
        a.parameters["azimuth"].fix()
        return a

    geoml.set_seed(1)
    cases = []
    prob, root, s, _ = categorical(aniso(), deep=False)
    cases.append(("Base model (%.0f s)" % s, prob, root))
    prob, root, s, _ = categorical(tr.Concatenate(aniso(), tr.BellFault2D(fault_start, fault_end)), deep=True)
    cases.append(("BellFault2D, the old segment (%.0f s)" % s, prob, root))
    implicit = tr.ImplicitFault(fault_points, fault_normals, reach=8.0, mode="step")
    prob, root, s, _ = categorical(tr.Concatenate(aniso(), implicit), deep=True)
    cases.append(("ImplicitFault, repulsion (%.0f s)" % s, prob, root))
    # from zero the categorical likelihood did not move the throw; the
    # search on the bound chooses it among candidates, training refines it
    true_throw = reject / (5.0 / np.sqrt(29.0))     # 3 of vertical offset along the fault
    displacement = tr.FaultDisplacement(fault_points, fault_normals, reach=30.0, width=0.1)
    chained = tr.ChainedTransform(displacement, aniso())
    prob, root, s, searches = categorical(chained, deep=False, phased=True,
                                          search=[(displacement, np.linspace(-8.0, 8.0, 17))])
    cases.append(("FaultDisplacement, throw searched then trained\nthrow %.2f, exact %.2f along the fault (%.0f s)" % (throw_of(chained), true_throw, s), prob, root))
    print("   trained; throw %.2f (searched %.2f, exact %.2f)" % (throw_of(chained), searches[0][2], true_throw), flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (title, prob, root) in zip(axes[0], cases):
        ax.imshow(prob, origin="lower", vmin=0, vmax=1, extent=(-5, 5, -5, 5), cmap="RdBu")
        ax.contour(np.linspace(-5, 5, N_GRID), np.linspace(-5, 5, N_GRID), prob, levels=[0.5], colors="k")
        ax.scatter(coords[final == "top", 0], coords[final == "top", 1], c="b", s=25, edgecolor="w")
        ax.scatter(coords[final == "bottom", 0], coords[final == "bottom", 1], c="r", s=25, edgecolor="w")
        ax.scatter(contacts[:, 0], contacts[:, 1], c="k", s=15)
        ax.plot([fault_start[0], fault_end[0]], [fault_start[1], fault_end[1]], "--", color="0.3")
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_title(title, fontsize=10)
    # under the hood: what the kernel sees
    g = grid(61)
    gc = np.asarray(g.coordinates, dtype=float)
    labels = ["the throw search", "extra coordinate (BellFault2D)",
              "extra coordinate (ImplicitFault)", "restored coordinates (FaultDisplacement)"]
    for ax, (title, prob, root), label in zip(axes[1], cases, labels):
        if label == "the throw search":
            candidates, scores, best = searches[0]
            draw_search(ax, candidates, scores, best, true_throw, "bound per candidate throw, model restored each time")
            continue
        x_tr = np.asarray(root.transform(tf.constant(gc)))
        if x_tr.shape[1] == 3:
            ax.imshow(image(x_tr[:, 2], 61), origin="lower", extent=(-5, 5, -5, 5), cmap="PuOr")
            ax.plot([fault_start[0], fault_end[0]], [fault_start[1], fault_end[1]], "--", color="0.3")
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        else:
            # the restoration alone, in world units, coloured by the true
            # side: the contact should become one straight line at y = 0
            first = root.transform.transforms[0] if hasattr(root.transform, "transforms") else root.transform
            restored = np.asarray(first(tf.constant(gc)))
            side_grid = (-fault_end[1] * gc[:, 0] + fault_end[0] * gc[:, 1]) < 0
            truth_top = np.where(side_grid, gc[:, 1] > 0, gc[:, 1] > reject)
            ax.scatter(restored[:, 0], restored[:, 1], c=np.where(truth_top, "b", "r"), s=4)
            ax.axhline(0.0, color="k", lw=0.8)
            ax.plot([fault_start[0], fault_end[0]], [fault_start[1], fault_end[1]], "--", color="0.3")
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
            ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
    fig.suptitle("Case A: the notebook's contact, offset by 3 across an inclined fault -- top probability (row 1) and what the kernel sees (row 2)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case_a.png"), dpi=100)
    plt.close(fig)
    print("   saved case_a.png", flush=True)


# ------------------------------------------------------------------ case B
def case_b():
    print("== case B: a grade offset along a curved fault ==", flush=True)
    geoml.set_seed(2)
    y = np.linspace(-6.0, 6.0, 41)
    # bends wider than the throw (radius of curvature 3.2 against 2.5):
    # beyond a bend's centre of curvature the normal lines cross and a
    # hanging wall sliding along the surface must fold, whatever the frame
    # -- at amplitude 1.5 (radius 1.7) 1.9% of it did
    trace = np.stack([0.8 * np.sin(np.pi * y / 5.0), y], axis=1)
    derived = point_normals(trace, k=6)                    # concavity rule
    rightward = point_normals(trace, k=6, orient=[1.0, 0.0])
    truth = tr.FaultDisplacement(trace, rightward, throw=2.5, reach=30.0, width=0.05)

    rng = np.random.default_rng(3)
    samples = rng.uniform(-5.0, 5.0, size=[180, 2])
    restored = np.asarray(truth(tf.constant(samples)))
    values = restored[:, 1] + rng.normal(scale=0.1, size=len(samples))
    data = geoml.data.PointData.from_array(samples, ["X", "Y"])
    data.add_continuous_variable("v", values)
    g = grid()
    gc = np.asarray(g.coordinates, dtype=float)
    truth_field = np.asarray(truth(tf.constant(gc)))[:, 1]

    searches = []

    def continuous(transform, phased=False, search=None):
        root = gl.BasicInput(grid(11), transform)
        gp = gl.BasicGP(root, size=1)
        model = geoml.models.VGPNetwork(data, "v", lk.Gaussian(), gp,
                                        options=geoml.models.GPOptions(verbose=False))
        for fault, candidates, label in (search or []):
            best, scores = geoml.models.search_throw(model, fault, candidates, iterations=40)
            searches.append((candidates, scores, best, label))
        seconds = train(model, phased=phased, iterations=800)
        gg = grid()
        model.predict(gg)
        pred = np.asarray(gg.variables["v"].prediction.values.to_numpy(), dtype=float).ravel()
        rmse = float(np.sqrt(np.mean((pred - truth_field) ** 2)))
        return pred, root, seconds, rmse

    panels = [("Truth: v = restored y", truth_field, None, "")]
    pred, root, s, rmse = continuous(tr.Isotropic(3.0))
    panels.append(("Plain Isotropic (rmse %.2f, %.0f s)" % (rmse, s), pred, root, ""))
    pred, root, s, rmse = continuous(tr.Concatenate(
        tr.Isotropic(3.0), tr.ImplicitFault(trace, rightward, reach=8.0)))
    panels.append(("ImplicitFault repulsion (rmse %.2f, %.0f s)" % (rmse, s), pred, root, ""))
    # from zero the throw trains at the phased rate when it moves before the
    # kernel range shrinks to explain the jump; from the initialization this
    # run draws it did not, so the search chooses it first here too
    fault = tr.FaultDisplacement(trace, rightward, reach=30.0, width=0.1)
    chained = tr.ChainedTransform(fault, tr.Isotropic(3.0))
    pred, root, s, rmse = continuous(chained, phased=True,
                                     search=[(fault, np.linspace(-4.0, 4.0, 9), "throw")])
    panels.append(("FaultDisplacement, throw searched then trained (rmse %.2f, %.0f s)\nthrow %.2f, true 2.50 along the fault" % (rmse, s, throw_of(chained)), pred, root, ""))
    print("   trained; throw %.2f (searched %.2f)" % (throw_of(chained), searches[0][2]), flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    vmin, vmax = np.percentile(truth_field, [1, 99])
    for ax, (title, field, root, _) in zip(axes[0], panels):
        ax.imshow(image(field), origin="lower", extent=(-5, 5, -5, 5), vmin=vmin, vmax=vmax, cmap="viridis")
        ax.contour(np.linspace(-5, 5, N_GRID), np.linspace(-5, 5, N_GRID), image(field), levels=np.arange(-8, 9, 1.0), colors="w", linewidths=0.5)
        ax.plot(trace[:, 0], trace[:, 1], "--", color="k", lw=1)
        ax.scatter(samples[:, 0], samples[:, 1], c=values, s=12, vmin=vmin, vmax=vmax, edgecolor="k", linewidths=0.3)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_title(title, fontsize=10)
    # row 2: the fitted surface with its derived normals, and what the kernel sees
    ax = axes[1, 0]
    surface = fault.surface
    field = np.asarray(surface(gc))
    ax.imshow(image(field), origin="lower", extent=(-5, 5, -5, 5), cmap="PuOr")
    ax.contour(np.linspace(-5, 5, N_GRID), np.linspace(-5, 5, N_GRID), image(field), levels=[0.0], colors="k")
    ax.quiver(trace[::2, 0], trace[::2, 1], derived[::2, 0], derived[::2, 1], color="k", scale=25, width=0.004)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_title("HermiteRBF field of the trace, zero set,\nnormals derived toward the concavity", fontsize=10)
    candidates, scores, best, _ = searches[0]
    draw_search(axes[1, 1], candidates, scores, best, 2.5, "bound per candidate throw, model restored each time")
    for ax, (title, field, root, _) in zip(axes[1, 2:], panels[2:]):
        x_tr = np.asarray(root.transform(tf.constant(gc)))
        if x_tr.shape[1] == 3:
            ax.imshow(image(x_tr[:, 2]), origin="lower", extent=(-5, 5, -5, 5), cmap="PuOr")
            ax.plot(trace[:, 0], trace[:, 1], "--", color="k", lw=1)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_title("extra coordinate the repulsion adds", fontsize=10)
        else:
            gc61 = np.asarray(grid(61).coordinates, dtype=float)
            x61 = np.asarray(root.transform(tf.constant(gc61)))
            t61 = np.asarray(truth(tf.constant(gc61)))[:, 1]
            ax.scatter(x61[:, 0], x61[:, 1], c=t61, s=5, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_title("restored coordinates, coloured by the truth:\ncontinuous again if the slip is right", fontsize=10)
    fig.suptitle("Case B: a vertical grade gradient offset by 2.5 along a curved fault fitted from its trace")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case_b.png"), dpi=100)
    plt.close(fig)
    print("   saved case_b.png", flush=True)


# ------------------------------------------------------------------ case C
def case_c():
    print("== case C: two faults, the younger cutting the older ==", flush=True)
    geoml.set_seed(5)
    y = np.linspace(-7.0, 7.0, 36)
    # the two traces cross at (0.75, 1.875): F1's upper part lies on F2's
    # positive side and is displaced by it, so F1 is observed in two pieces
    f1_original = np.stack([0.4 * y, y], axis=1)                 # older
    f2_trace = np.stack([1.5 - 0.4 * y, y], axis=1)              # younger
    n1 = point_normals(f1_original, k=6, orient=[1.0, 0.0])
    n2 = point_normals(f2_trace, k=6, orient=[1.0, 0.0])
    d1 = np.array([0.4, 1.0]); d1 /= np.linalg.norm(d1)
    d2 = np.array([-0.4, 1.0]); d2 /= np.linalg.norm(d2)
    # the 2-D tangent is the rightward normal turned a quarter turn, which
    # is d1 and d2: throws along them with vertical components 2 and -1.5
    throw1 = 2.0 / d1[1]
    throw2 = -1.5 / d2[1]
    # F1 as observed: its trace displaced forward by F2 on F2's positive side
    forward_f2 = tr.FaultDisplacement(f2_trace, n2, throw=-throw2, reach=40.0, width=0.05)
    f1_observed = np.asarray(forward_f2(tf.constant(f1_original)))
    n1_observed = point_normals(f1_observed, k=6, orient=[1.0, 0.0])

    truth_net = tr.FaultNetwork([
        tr.FaultDisplacement(f2_trace, n2, throw=throw2, reach=40.0, width=0.05),
        tr.FaultDisplacement(f1_observed, n1_observed, throw=throw1, reach=40.0, width=0.05)])
    rng = np.random.default_rng(6)
    samples = rng.uniform(-5.0, 5.0, size=[220, 2])
    values = np.asarray(truth_net(tf.constant(samples)))[:, 1] + rng.normal(scale=0.1, size=len(samples))
    data = geoml.data.PointData.from_array(samples, ["X", "Y"])
    data.add_continuous_variable("v", values)
    g = grid()
    gc = np.asarray(g.coordinates, dtype=float)
    truth_field = np.asarray(truth_net(tf.constant(gc)))[:, 1]

    searches = []

    def continuous(transform, phased=False, search=None):
        root = gl.BasicInput(grid(11), transform)
        gp = gl.BasicGP(root, size=1)
        model = geoml.models.VGPNetwork(data, "v", lk.Gaussian(), gp,
                                        options=geoml.models.GPOptions(verbose=False))
        for fault, candidates, label in (search or []):
            best, scores = geoml.models.search_throw(model, fault, candidates, iterations=40)
            searches.append((candidates, scores, best, label))
        seconds = train(model, phased=phased, iterations=800)
        gg = grid()
        x_tr = np.asarray(root.transform(tf.constant(gc)))
        bad = ~np.all(np.isfinite(x_tr), axis=1)
        print("   transform non-finite at %d of %d grid points" % (bad.sum(), len(gc)), gc[bad][:5].tolist(), flush=True)
        model.predict(gg)
        pred = np.asarray(gg.variables["v"].prediction.values.to_numpy(), dtype=float).ravel()
        print("   prediction non-finite at %d grid points; training log finite: %s" % ((~np.isfinite(pred)).sum(), bool(np.all(np.isfinite(model.training_log)))), flush=True)
        rmse = float(np.sqrt(np.nanmean((pred - truth_field) ** 2)))
        return pred, root, seconds, rmse

    panels = [("Truth: two faults, F2 cuts F1", truth_field, None)]
    pred, root, s, rmse = continuous(tr.Isotropic(3.0))
    panels.append(("Plain Isotropic (rmse %.2f, %.0f s)" % (rmse, s), pred, root))
    pred, root, s, rmse = continuous(tr.Concatenate(
        tr.Isotropic(3.0), tr.ImplicitFault(f2_trace, n2, reach=8.0),
        tr.ImplicitFault(f1_observed, n1_observed, reach=8.0)))
    panels.append(("Two ImplicitFaults, no topology (rmse %.2f, %.0f s)" % (rmse, s), pred, root))
    # from zero both throws stayed near zero in a thousand iterations; the
    # search on the bound chooses each -- the younger first, then the older
    # with the younger set -- and joint training refines them
    younger = tr.FaultDisplacement(f2_trace, n2, reach=40.0, width=0.1)
    older = tr.FaultDisplacement(f1_observed, n1_observed, reach=40.0, width=0.1)
    network = tr.FaultNetwork([younger, older])
    chained = tr.ChainedTransform(network, tr.Isotropic(3.0))
    candidates = np.linspace(-4.0, 4.0, 9)
    pred, root, s, rmse = continuous(chained, phased=True,
                                     search=[(younger, candidates, "F2 (younger), F1 at zero"),
                                             (older, candidates, "F1 (older), F2 as chosen")])
    t1 = float(older.parameters["throw"].get_value()); t2 = float(younger.parameters["throw"].get_value())
    panels.append(("FaultNetwork, throws searched then trained (rmse %.2f, %.0f s)\nF2 throw %.2f true %.2f; F1 throw %.2f true %.2f"
                   % (rmse, s, t2, throw2, t1, throw1), pred, root))
    print("   trained; F2 %.2f true %.2f; F1 %.2f true %.2f" % (t2, throw2, t1, throw1), flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    vmin, vmax = np.percentile(truth_field, [1, 99])
    for ax, (title, field, root) in zip(axes[0], panels):
        ax.imshow(image(field), origin="lower", extent=(-5, 5, -5, 5), vmin=vmin, vmax=vmax, cmap="viridis")
        ax.contour(np.linspace(-5, 5, N_GRID), np.linspace(-5, 5, N_GRID), image(field), levels=np.arange(-8, 9, 1.0), colors="w", linewidths=0.5)
        ax.plot(f1_observed[:, 0], f1_observed[:, 1], "--", color="k", lw=1)
        ax.plot(f2_trace[:, 0], f2_trace[:, 1], "-", color="k", lw=1)
        ax.scatter(samples[:, 0], samples[:, 1], c=values, s=10, vmin=vmin, vmax=vmax, edgecolor="k", linewidths=0.3)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_title(title, fontsize=9)
    gc61 = np.asarray(grid(61).coordinates, dtype=float)
    t61 = np.asarray(truth_net(tf.constant(gc61)))[:, 1]
    ax = axes[1, 0]
    restored_f1 = np.asarray(network._restored_points(1))
    ax.scatter(restored_f1[:, 0], restored_f1[:, 1], c="tab:red", s=30, label="F1 after undoing the trained F2")
    ax.scatter(f1_observed[:, 0], f1_observed[:, 1], c="k", s=8, label="F1 as observed (two pieces)")
    ax.plot(f1_original[:, 0], f1_original[:, 1], ":", color="0.5", lw=1, label="F1 before F2 (truth)")
    ax.plot(f2_trace[:, 0], f2_trace[:, 1], "-", color="k", lw=1, label="F2")
    ax.legend(fontsize=8); ax.set_xlim(-6, 6); ax.set_ylim(-7, 7); ax.set_aspect("equal")
    ax.set_title("the older fault's pieces, before and after undoing the younger", fontsize=9)
    ax = axes[1, 1]
    for (candidates, scores, best, label), colour, true in zip(searches, ("tab:blue", "tab:orange"), (throw2, throw1)):
        ax.plot(candidates, scores, "o-", color=colour, ms=4, label=label)
        ax.axvline(true, color=colour, lw=1, ls=":")
        ax.axvline(best, color=colour, lw=1, ls="--")
    ax.set_xlabel("candidate throw along the fault (dotted true, dashed chosen)"); ax.set_ylabel("bound after a 40-iteration burst")
    ax.set_title("the two throw searches, younger first", fontsize=9); ax.legend(fontsize=8)
    ax = axes[1, 2]
    x_tr = np.asarray(panels[2][2].transform(tf.constant(gc)))
    ax.imshow(image(x_tr[:, 2]) + image(x_tr[:, 3]), origin="lower", extent=(-5, 5, -5, 5), cmap="PuOr")
    ax.set_title("the two repulsion coordinates, summed", fontsize=9)
    ax = axes[1, 3]
    x61 = np.asarray(panels[3][2].transform(tf.constant(gc61)))
    ax.scatter(x61[:, 0], x61[:, 1], c=t61, s=5, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title("restored coordinates coloured by the truth:\ncontinuous if both slips are right", fontsize=9)
    fig.suptitle("Case C: two faults -- the younger (F2) cuts the older (F1); the network refits F1 on the coordinates F2 restores")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case_c.png"), dpi=100)
    plt.close(fig)
    print("   saved case_c.png", flush=True)


# ------------------------------------------------------------------ case D
def case_d():
    print("== case D: two curvilinear faults ending on a third ==", flush=True)
    geoml.set_seed(8)

    def f0_y(x):                       # the older fault: inclined, gently curved
        return -1.0 + 0.35 * x + 0.6 * np.sin(np.pi * x / 6.0)

    xs = np.linspace(-7.0, 7.0, 36)
    f0_trace = np.stack([xs, f0_y(xs)], axis=1)
    n0 = point_normals(f0_trace, k=6, orient=[0.0, 1.0])       # hanging wall above
    ys = np.linspace(-7.0, 7.0, 71)
    younger_traces = []
    for x_of_y in (lambda y: -2.0 + 0.5 * np.sin(np.pi * y / 5.0),
                   lambda y: 2.5 + 0.4 * np.sin(np.pi * (y + 1.0) / 5.0)):
        x = x_of_y(ys)
        above = ys > f0_y(x) + 0.15                              # they end on F0
        younger_traces.append(np.stack([x[above], ys[above]], axis=1))
    f1_trace, f2_trace = younger_traces
    n1 = point_normals(f1_trace, k=6, orient=[1.0, 0.0])
    n2 = point_normals(f2_trace, k=6, orient=[1.0, 0.0])
    throw0, throw1, throw2 = 1.5, 1.0, -0.8
    abutting = [(0, 2, 1), (1, 2, 1)]         # F1, F2 stop on F0's upper side

    def faults(t0, t1, t2, width):
        return [tr.FaultDisplacement(f1_trace, n1, throw=t1, reach=40.0, width=width),
                tr.FaultDisplacement(f2_trace, n2, throw=t2, reach=40.0, width=width),
                tr.FaultDisplacement(f0_trace, n0, throw=t0, reach=40.0, width=width)]

    truth_net = tr.FaultNetwork(faults(throw0, throw1, throw2, 0.05), abutting=abutting)
    rng = np.random.default_rng(9)
    samples = rng.uniform(-5.0, 5.0, size=[260, 2])
    values = np.asarray(truth_net(tf.constant(samples)))[:, 1] + rng.normal(scale=0.1, size=len(samples))
    data = geoml.data.PointData.from_array(samples, ["X", "Y"])
    data.add_continuous_variable("v", values)
    g = grid()
    gc = np.asarray(g.coordinates, dtype=float)
    truth_field = np.asarray(truth_net(tf.constant(gc)))[:, 1]

    searches = {}

    def continuous(label, transform, head=None, phased=False, search=None):
        root = gl.BasicInput(grid(11), transform)
        node = gl.Linear(root, size=head) if head else root
        gp = gl.BasicGP(node, size=1)
        model = geoml.models.VGPNetwork(data, "v", lk.Gaussian(), gp,
                                        options=geoml.models.GPOptions(verbose=False))
        for fault, candidates, name in (search or []):
            best, scores = geoml.models.search_throw(model, fault, candidates, iterations=40)
            searches.setdefault(label, []).append((candidates, scores, best, name))
        seconds = train(model, phased=phased, iterations=800)
        gg = grid()
        model.predict(gg)
        pred = np.asarray(gg.variables["v"].prediction.values.to_numpy(), dtype=float).ravel()
        print("   %s: prediction non-finite at %d grid points" % (label, (~np.isfinite(pred)).sum()), flush=True)
        rmse = float(np.sqrt(np.nanmean((pred - truth_field) ** 2)))
        return pred, root, seconds, rmse

    panels = [("Truth: F1 and F2 (dashed) end on the older F0", truth_field, None)]
    pred, root, s, rmse = continuous("plain", tr.Isotropic(3.0))
    panels.append(("Plain Isotropic (rmse %.2f, %.0f s)" % (rmse, s), pred, root))
    repulsion = tr.Concatenate(
        tr.Isotropic(3.0),
        tr.ImplicitFault(f0_trace, n0, reach=40.0),
        tr.ImplicitFault(f1_trace, n1, reach=1.0),
        tr.ImplicitFault(f2_trace, n2, reach=1.0))
    pred, root, s, rmse = continuous("repulsion", repulsion, head=5)
    panels.append(("Three ImplicitFaults + Linear head, no topology (rmse %.2f, %.0f s)" % (rmse, s), pred, root))
    # the same three repulsions with F1 and F2 declared to stop on F0: the
    # trimmed sign, zero beyond F0 by construction
    blocks = tr.Concatenate(
        tr.Isotropic(3.0),
        tr.ImplicitFaultBlocks(
            [tr.ImplicitFault(f0_trace, n0, reach=40.0),
             tr.ImplicitFault(f1_trace, n1, reach=40.0),
             tr.ImplicitFault(f2_trace, n2, reach=40.0)],
            abutting=[(1, 0, 1), (2, 0, 1)]))
    pred, root, s, rmse = continuous("blocks", blocks, head=5)
    panels.append(("ImplicitFaultBlocks + Linear head, F1 and F2 stop on F0 (rmse %.2f, %.0f s)" % (rmse, s), pred, root))
    candidates = np.linspace(-4.0, 4.0, 9)

    def network_model(label, with_abutting):
        f1, f2, f0 = faults(0.0, 0.0, 0.0, 0.1)
        net = tr.FaultNetwork([f1, f2, f0], abutting=abutting if with_abutting else ())
        chained = tr.ChainedTransform(net, tr.Isotropic(3.0))
        pred, root, s, rmse = continuous(label, chained, phased=True,
                                         search=[(f0, candidates, "F0 (oldest), others at zero"),
                                                 (f1, candidates, "F1, F0 as chosen"),
                                                 (f2, candidates, "F2, F0 and F1 as chosen")])
        t = [float(f.parameters["throw"].get_value()) for f in (f0, f1, f2)]
        print("   %s: throws F0 %.2f (true %.2f), F1 %.2f (true %.2f), F2 %.2f (true %.2f)"
              % (label, t[0], throw0, t[1], throw1, t[2], throw2), flush=True)
        return pred, root, s, rmse, t, net

    pred, root, s, rmse, t, net = network_model("network with abutting", True)
    network_panel = ("FaultNetwork, F1 and F2 declared to stop on F0 (rmse %.2f, %.0f s)\nthrows F0 %.2f F1 %.2f F2 %.2f, true %.1f %.1f %.1f"
                     % (rmse, s, t[0], t[1], t[2], throw0, throw1, throw2), pred, root)
    pred_c, root_c, s_c, rmse_c, t_c, _ = network_model("network without abutting", False)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    vmin, vmax = np.percentile(truth_field, [1, 99])

    def draw_field(ax, field, title):
        ax.imshow(image(field), origin="lower", extent=(-5, 5, -5, 5), vmin=vmin, vmax=vmax, cmap="viridis")
        ax.contour(np.linspace(-5, 5, N_GRID), np.linspace(-5, 5, N_GRID), image(field), levels=np.arange(-8, 9, 1.0), colors="w", linewidths=0.5)
        ax.plot(f0_trace[:, 0], f0_trace[:, 1], "-", color="k", lw=1.2)
        for trace in (f1_trace, f2_trace):
            ax.plot(trace[:, 0], trace[:, 1], "--", color="k", lw=1)
        ax.scatter(samples[:, 0], samples[:, 1], c=values, s=8, vmin=vmin, vmax=vmax, edgecolor="k", linewidths=0.3)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_title(title, fontsize=9)

    # row 1: truth, the untrimmed repulsion, the trimmed one, the network;
    # the plain model's number goes to the title
    plain_rmse = float(panels[1][0].split("rmse ")[1].split(",")[0])
    for ax, (title, field, root) in zip(axes[0], [panels[0], panels[2], panels[3], network_panel]):
        draw_field(ax, field, title)

    def draw_sum(ax, root, title):
        x_tr = np.asarray(root.transform(tf.constant(gc)))
        ax.imshow(image(x_tr[:, 2] + x_tr[:, 3] + x_tr[:, 4]), origin="lower", extent=(-5, 5, -5, 5), cmap="PuOr")
        ax.plot(f0_trace[:, 0], f0_trace[:, 1], "-", color="k", lw=1.2)
        for trace in (f1_trace, f2_trace):
            ax.plot(trace[:, 0], trace[:, 1], "--", color="k", lw=1)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_title(title, fontsize=9)

    draw_sum(axes[1, 0], panels[2][2], "the three repulsion coordinates summed, no topology:\nF1 and F2 continue past F0 until their reach fades them")
    draw_sum(axes[1, 1], panels[3][2], "the same with ImplicitFaultBlocks:\nF1 and F2 are zero beyond F0 by construction")
    draw_field(axes[1, 3], pred_c, "control: the network with no abutting declared\n(rmse %.2f) throws F0 %.2f F1 %.2f F2 %.2f" % (rmse_c, t_c[0], t_c[1], t_c[2]))
    ax = axes[1, 2]
    for (candidates_, scores, best, name), colour, true in zip(searches["network with abutting"], ("tab:green", "tab:blue", "tab:orange"), (throw0, throw1, throw2)):
        ax.plot(candidates_, scores, "o-", color=colour, ms=4, label=name)
        ax.axvline(true, color=colour, lw=1, ls=":")
        ax.axvline(best, color=colour, lw=1, ls="--")
    ax.set_xlabel("candidate throw (dotted true, dashed chosen)"); ax.set_ylabel("bound after a 40-iteration burst")
    ax.set_title("the three throw searches of the network, oldest first", fontsize=9); ax.legend(fontsize=8)
    fig.suptitle("Case D: two curvilinear faults (F1, F2) ending on an older curvilinear fault (F0) -- plain isotropic rmse %.2f; repulsion without and with topology, and the network" % plain_rmse)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case_d.png"), dpi=100)
    plt.close(fig)
    print("   saved case_d.png", flush=True)


# ------------------------------------------------------------------ case E
def case_e():
    """Case D's geometry with the improved displacement: profiled throws on
    the younger faults in the truth, and the network with fault-parallel
    flow, bell profiles, trainable drag width, the throw of F0 read from
    markers, a refined search for F1 and F2 at a wide step and the step
    annealed, against the network as it was."""
    print("== case E: case D revisited with the improved displacement ==", flush=True)
    geoml.set_seed(11)

    def f0_y(x):
        return -1.0 + 0.35 * x + 0.6 * np.sin(np.pi * x / 6.0)

    xs = np.linspace(-7.0, 7.0, 36)
    f0_trace = np.stack([xs, f0_y(xs)], axis=1)
    n0 = point_normals(f0_trace, k=6, orient=[0.0, 1.0])
    ys = np.linspace(-7.0, 7.0, 71)
    traces = []
    for x_of_y in (lambda y: -2.0 + 0.5 * np.sin(np.pi * y / 5.0),
                   lambda y: 2.5 + 0.4 * np.sin(np.pi * (y + 1.0) / 5.0)):
        x = x_of_y(ys)
        above = ys > f0_y(x) + 0.15
        traces.append(np.stack([x[above], ys[above]], axis=1))
    f1_trace, f2_trace = traces
    n1 = point_normals(f1_trace, k=6, orient=[1.0, 0.0])
    n2 = point_normals(f2_trace, k=6, orient=[1.0, 0.0])
    throw0, throw1, throw2 = 1.5, 1.4, -1.2          # F1, F2: the profiles' peaks
    abutting = [(0, 2, 1), (1, 2, 1)]

    def faults(t0, t1, t2, width, improved):
        if improved:
            return [tr.FaultDisplacement(f1_trace, n1, throw=t1, reach=40.0, width=width, profile="bell", drag=True),
                    tr.FaultDisplacement(f2_trace, n2, throw=t2, reach=40.0, width=width, profile="bell", drag=True),
                    tr.FaultDisplacement(f0_trace, n0, throw=t0, reach=40.0, width=width, drag=True)]
        return [tr.FaultDisplacement(f1_trace, n1, throw=t1, reach=40.0, width=width, flow_steps=0),
                tr.FaultDisplacement(f2_trace, n2, throw=t2, reach=40.0, width=width, flow_steps=0),
                tr.FaultDisplacement(f0_trace, n0, throw=t0, reach=40.0, width=width, flow_steps=0)]

    truth_faults = faults(throw0, throw1, throw2, 0.05, improved=True)
    truth_net = tr.FaultNetwork(truth_faults, abutting=abutting)
    rng = np.random.default_rng(12)
    samples = rng.uniform(-5.0, 5.0, size=[260, 2])
    values = np.asarray(truth_net(tf.constant(samples)))[:, 1] + rng.normal(scale=0.1, size=len(samples))
    data = geoml.data.PointData.from_array(samples, ["X", "Y"])
    data.add_continuous_variable("v", values)
    g = grid()
    gc = np.asarray(g.coordinates, dtype=float)
    truth_field = np.asarray(truth_net(tf.constant(gc)))[:, 1]

    # one horizon, restored y = 0, seen on both walls of F0 left of F1: the
    # footwall keeps it at y = 0, the hanging wall's is found by bisection
    foot_x = np.linspace(2.6, 4.8, 12)
    footwall = np.stack([foot_x, np.zeros(12)], axis=1)
    hang_x = np.linspace(-4.8, -3.0, 12)
    lo, hi = f0_y(hang_x) + 0.3, np.full(12, 5.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        restored_y = np.asarray(truth_net(tf.constant(np.stack([hang_x, mid], axis=1))))[:, 1]
        lo = np.where(restored_y < 0, mid, lo)
        hi = np.where(restored_y < 0, hi, mid)
    hanging = np.stack([hang_x, 0.5 * (lo + hi)], axis=1)

    def build(transform, head=None):
        root = gl.BasicInput(grid(11), transform)
        node = gl.Linear(root, size=head) if head else root
        model = geoml.models.VGPNetwork(data, "v", lk.Gaussian(), gl.BasicGP(node, size=1),
                                        options=geoml.models.GPOptions(verbose=False))
        return model, root

    def score(model, label):
        gg = grid()
        model.predict(gg)
        pred = np.asarray(gg.variables["v"].prediction.values.to_numpy(), dtype=float).ravel()
        rmse = float(np.sqrt(np.nanmean((pred - truth_field) ** 2)))
        print("   %s: rmse %.3f, non-finite %d" % (label, rmse, (~np.isfinite(pred)).sum()), flush=True)
        return pred, rmse

    started = time.monotonic()
    model, _ = build(tr.Isotropic(3.0))
    train(model, iterations=800)
    plain, plain_rmse = score(model, "plain")
    blocks = tr.Concatenate(tr.Isotropic(3.0), tr.ImplicitFaultBlocks(
        [tr.ImplicitFault(f0_trace, n0, reach=40.0), tr.ImplicitFault(f1_trace, n1, reach=40.0),
         tr.ImplicitFault(f2_trace, n2, reach=40.0)], abutting=[(1, 0, 1), (2, 0, 1)]))
    model, _ = build(blocks, head=5)
    train(model, iterations=800)
    blocks_pred, blocks_rmse = score(model, "blocks")
    candidates = np.linspace(-4.0, 4.0, 9)

    # the network as it was: straight step, one throw per fault, fixed
    # width, coarse search, phased training
    started = time.monotonic()
    f1, f2, f0 = faults(0.0, 0.0, 0.0, 0.1, improved=False)
    old_net = tr.FaultNetwork([f1, f2, f0], abutting=abutting)
    model, _ = build(tr.ChainedTransform(old_net, tr.Isotropic(3.0)))
    for fault in (f0, f1, f2):
        geoml.models.search_throw(model, fault, candidates, iterations=40)
    train(model, phased=True, iterations=800)
    old_pred, old_rmse = score(model, "network as it was")
    old_throws = [float(f.parameters["throw"].get_value()) for f in (f0, f1, f2)]
    old_seconds = time.monotonic() - started

    # the improved network: flow, profiles, drag, F0 from markers, refined
    # search at a wide step, the step annealed while the widths train
    started = time.monotonic()
    f1, f2, f0 = faults(0.0, 0.0, 0.0, 1.0, improved=True)
    new_net = tr.FaultNetwork([f1, f2, f0], abutting=abutting)
    model, new_root = build(tr.ChainedTransform(new_net, tr.Isotropic(3.0)))
    marker_throw = f0.throw_from_markers(hanging, footwall)
    print("   F0's throw from the markers: %.3f (true %.2f)" % (marker_throw, throw0), flush=True)
    passes = {}
    for fault, name in ((f1, "F1"), (f2, "F2")):
        _, passes[name] = geoml.models.search_throw(model, fault, candidates, iterations=40, refine=1)
    for width, rate, iterations in ((1.0, 0.1, 300), (0.4, 0.1, 300), (0.15, 0.01, 300)):
        for fault in (f0, f1, f2):
            fault.set_width(width)
        model.set_learning_rate(rate)
        model.train_full(iterations)
    new_pred, new_rmse = score(model, "network improved")
    new_throws = [float(f.parameters["throw"].get_value()) for f in (f0, f1, f2)]
    new_widths = [f.width for f in (f0, f1, f2)]
    new_seconds = time.monotonic() - started
    print("   throws as it was %s, improved %s, true %s; widths trained to %s"
          % (np.round(old_throws, 2), np.round(new_throws, 2), (throw0, throw1, throw2), np.round(new_widths, 3)), flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    vmin, vmax = np.percentile(truth_field, [1, 99])

    def draw_field(ax, field, title):
        ax.imshow(image(field), origin="lower", extent=(-5, 5, -5, 5), vmin=vmin, vmax=vmax, cmap="viridis")
        ax.contour(np.linspace(-5, 5, N_GRID), np.linspace(-5, 5, N_GRID), image(field), levels=np.arange(-8, 9, 1.0), colors="w", linewidths=0.5)
        ax.plot(f0_trace[:, 0], f0_trace[:, 1], "-", color="k", lw=1.2)
        for trace in (f1_trace, f2_trace):
            ax.plot(trace[:, 0], trace[:, 1], "--", color="k", lw=1)
        ax.scatter(samples[:, 0], samples[:, 1], c=values, s=8, vmin=vmin, vmax=vmax, edgecolor="k", linewidths=0.3)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_title(title, fontsize=9)

    draw_field(axes[0, 0], truth_field, "Truth: F1 and F2 with bell profiles (peaks %.1f, %.1f) end on F0 (throw %.1f)" % (throw1, throw2, throw0))
    draw_field(axes[0, 1], blocks_pred, "ImplicitFaultBlocks + Linear head (rmse %.2f); plain %.2f" % (blocks_rmse, plain_rmse))
    draw_field(axes[0, 2], old_pred, "FaultNetwork as it was (rmse %.2f, %.0f s)\nthrows F0 %.2f F1 %.2f F2 %.2f" % (old_rmse, old_seconds, *old_throws))
    draw_field(axes[0, 3], new_pred, "FaultNetwork improved: flow, profiles, drag, markers, refined search, annealing\n(rmse %.2f, %.0f s) throws F0 %.2f F1 %.2f F2 %.2f" % (new_rmse, new_seconds, *new_throws))
    ax = axes[1, 0]
    draw_field(ax, truth_field, "the marker horizon on both walls of F0:\nthrow read from them %.2f, true %.2f" % (marker_throw, throw0))
    ax.scatter(hanging[:, 0], hanging[:, 1], c="w", s=30, edgecolor="k", zorder=5, label="hanging-wall markers")
    ax.scatter(footwall[:, 0], footwall[:, 1], c="k", s=30, edgecolor="w", zorder=5, label="footwall markers")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1, 1]
    for k, (trace, truth_fault, fault, peak, colour, name) in enumerate((
            (f1_trace, truth_faults[0], f1, throw1, "tab:blue", "F1"),
            (f2_trace, truth_faults[1], f2, throw2, "tab:orange", "F2"))):
        t = tf.constant(trace)
        ax.plot(trace[:, 1], peak * np.asarray(truth_fault._profile(t)), "-", color=colour, label="%s true" % name)
        ax.plot(trace[:, 1], float(fault.parameters["throw"].get_value()) * np.asarray(fault._profile(t)), "--", color=colour, label="%s learned" % name)
    ax.set_xlabel("y along the trace"); ax.set_ylabel("throw"); ax.legend(fontsize=8)
    ax.set_title("throw profiles along F1 and F2, true and learned", fontsize=9)
    ax = axes[1, 2]
    for name, colour, true in (("F1", "tab:blue", throw1), ("F2", "tab:orange", throw2)):
        for level, (candidates_, scores) in enumerate(passes[name]):
            ax.plot(candidates_, scores, "o-" if level == 0 else "s--", color=colour, ms=4, label="%s pass %d" % (name, level + 1))
        ax.axvline(true, color=colour, lw=1, ls=":")
    ax.set_xlabel("candidate throw (dotted true)"); ax.set_ylabel("bound after a 40-iteration burst")
    ax.set_title("coarse then fine searches at a wide step", fontsize=9); ax.legend(fontsize=8)
    ax = axes[1, 3]
    gc61 = np.asarray(grid(61).coordinates, dtype=float)
    t61 = np.asarray(truth_net(tf.constant(gc61)))[:, 1]
    x61 = np.asarray(new_net(tf.constant(gc61)))
    ax.scatter(x61[:, 0], x61[:, 1], c=t61, s=5, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title("restored coordinates of the improved network,\ncoloured by the truth", fontsize=9)
    fig.suptitle("Case E: case D's geometry with the improved displacement -- fault-parallel flow, bell profiles, trainable drag width, markers, refined search, annealed step")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case_e.png"), dpi=100)
    plt.close(fig)
    print("   saved case_e.png", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    for case in which:
        {"a": case_a, "b": case_b, "c": case_c, "d": case_d, "e": case_e}[case]()
    print("done", flush=True)
