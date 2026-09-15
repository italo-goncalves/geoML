"""The leaf-only refit against E1's protocol, on a model with an interior.

Chapter 16's Jura tree: a two-column displacement GP walked (`GPWalk`), the
rock GP on the walked coordinates, the metals as their own GP plus a
`Linear` trend read from the rock GP. The interior is the displacement
field -- the only GP no likelihood touches -- and `refit="leaves"` keeps it
as all the data taught it while `refit="variational"` forgets it too.

Two modes, per seed, on the same five spatial folds and inducing set:

`reuse` -- the driver's arms from one model trained on all the rows:
fresh-all and fresh-leaves at 50 and 200 iterations, warm-200
(`refit="all"`, the leak reference E1 measured 3-8% past the gold),
scratch-400 (a fresh model per fold, the gold) and the base model's
in-sample score, which is what an out-of-fold score approaches when the
model remembers.

`honest` -- the same leaf-only refit from an interior that never saw the
held-out rows: a fresh model per fold trained 300 iterations (scratch-300),
continued to 600 (scratch-600, is the gold converged?), then its terminal
GP nodes re-initialized and refit 200 iterations at everything else fixed
(scratch-leaves). If scratch-leaves scores like fresh-leaves the refit
protocol is better; if it scores like scratch the interior was remembering.

Usage: python docs/benchmarks/leaf_refit.py <seed> [reuse|honest]
"""
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import geoml  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent as gl  # noqa: E402
import geoml.likelihood as lk  # noqa: E402
import geoml.models as gm  # noqa: E402
import geoml.transform as tr  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODE = sys.argv[2] if len(sys.argv) > 2 else "reuse"
K, N_SIM, N_NODES = 5, 20, 16
VARIABLES = ["Rock", "Elements"]


def build(data, inducing, n_rock, n_el, seed):
    geoml.set_seed(seed)
    root_node = gl.BasicInput(inducing, transform=tr.Isotropic(1.0))
    displacement = gl.BasicGP(root_node, size=2, kernel=kr.Gaussian())
    walked = gl.GPWalk(displacement, n_steps=5)
    rock_gp = gl.BasicGP(walked, size=n_rock, kernel=kr.Matern32())
    trend = gl.Linear(rock_gp, size=n_el, unit_norm=False)
    metal_gp = gl.BasicGP(root_node, size=n_el, kernel=kr.Spherical())
    leaves = [rock_gp, gl.LinearCombination(trend, metal_gp)]
    return gm.VGPNetwork(
        data, VARIABLES,
        [lk.CategoricalGaussianIndicator(n_rock), lk.MultivariateGaussian(n_el)],
        leaves, options=gm.GPOptions(verbose=False, training_samples=8))


def score_fold(model, oof, held, acc):
    """The driver's scoring, verbatim, for the hand-built arms."""
    held_points = oof[held]
    truths = {}
    for v, _ in model._measured_variables():
        y_true, has_value = held_points.variables[v].get_measurements()
        y_true = np.asarray(y_true, dtype=float)
        has_value = np.asarray(has_value)
        if y_true.ndim == 1:
            y_true = y_true[:, None]
        if has_value.ndim == 1:
            has_value = has_value[:, None]
        parts = getattr(held_points.variables[v], "labels", None)
        truths[v] = (y_true, has_value, [v] if parts is None else list(parts))
    for batch, samples in model.measurement_batches(
            held_points, n_sim=N_SIM, n_nodes=N_NODES):
        for v, sample in samples.items():
            y_true, has_value, components = truths[v]
            for c, component in enumerate(components):
                column = has_value[batch, min(c, has_value.shape[1] - 1)]
                measured = column == 1
                if not measured.any():
                    continue
                truth = y_true[batch, c][measured]
                draw = sample[measured, c, :]
                gm._accumulate(acc.setdefault((v, component), gm._fresh_scores()),
                               truth, draw)


def new_oof(train):
    oof = train[np.ones(train.n_data, dtype=bool)]
    for v in VARIABLES:
        oof.variables[v].allocate_simulations(N_SIM)
    return oof


def table_of(acc):
    return pd.DataFrame([dict(gm._scores_from(a), variable=v, component=c, fold="all")
                         for (v, c), a in acc.items()])


def summarize(arm, iterations, oof, scores, sd, seconds):
    pooled = scores[scores["fold"] == "all"].set_index("component")
    rock = oof.variables["Rock"].compute_metrics()
    return dict(
        arm=arm, iterations=iterations,
        rmse_over_sd=float(np.mean([pooled.loc[c, "rmse"] / sd[c] for c in sd])),
        crps_over_sd=float(np.mean([pooled.loc[c, "crps"] / sd[c] for c in sd])),
        goodness=float(pooled["goodness"].mean()),
        rock_balanced_accuracy=float(rock.loc["Balanced accuracy"].iloc[0]),
        seconds=round(seconds, 1))


def reuse_mode(train, inducing, n_rock, n_el, labels, folds, sd):
    model = build(train, inducing, n_rock, n_el, SEED)
    t0 = time.perf_counter()
    model.train_full(300)
    print("seed %d: base model trained in %.0f s, bound %.1f"
          % (SEED, time.perf_counter() - t0, model.training_log[-1]), flush=True)

    rows = []
    # in-sample: the base model on its own rows
    t0 = time.perf_counter()
    within = new_oof(train)
    model.predict(within, n_sim=N_SIM, include_noise=True)
    acc = {}
    score_fold(model, within, np.ones(train.n_data, dtype=bool), acc)
    rows.append(summarize("in-sample", 300, within, table_of(acc), sd,
                          time.perf_counter() - t0))
    print(rows[-1], flush=True)

    for arm, refit, iterations in [("fresh-all", "variational", 50),
                                   ("fresh-leaves", "leaves", 50),
                                   ("fresh-all", "variational", 200),
                                   ("fresh-leaves", "leaves", 200),
                                   ("warm", "all", 200)]:
        geoml.set_seed(SEED)
        t0 = time.perf_counter()
        oof, scores = gm.cross_validate(
            model, refit=refit, iterations=iterations, n_sim=N_SIM, n_nodes=N_NODES)
        rows.append(summarize(arm, iterations, oof, scores, sd, time.perf_counter() - t0))
        print(rows[-1], flush=True)

    t0 = time.perf_counter()
    oof = new_oof(train)
    acc = {}
    for i, fold in enumerate(folds):
        held = labels == fold
        fold_model = build(train[~held], inducing, n_rock, n_el, SEED + 1000 * (i + 1))
        fold_model.train_full(400)
        fold_model.predict(oof, n_sim=N_SIM, include_noise=True, where=held)
        score_fold(fold_model, oof, held, acc)
    rows.append(summarize("scratch", 400, oof, table_of(acc), sd, time.perf_counter() - t0))
    print(rows[-1], flush=True)
    return rows


def honest_mode(train, inducing, n_rock, n_el, labels, folds, sd):
    """A fresh model per fold: scored at 300, at 600, and after its terminal
    GP nodes are re-initialized and refit 200 iterations from the 600 state
    with everything else fixed -- the leaf-only refit from an interior that
    never saw the held-out rows."""
    oofs = {arm: new_oof(train) for arm in ("scratch-300", "scratch-600", "scratch-leaves")}
    accs = {arm: {} for arm in oofs}
    seconds = {arm: 0.0 for arm in oofs}
    bounds = []
    for i, fold in enumerate(folds):
        held = labels == fold
        fold_model = build(train[~held], inducing, n_rock, n_el, SEED + 1000 * (i + 1))
        t0 = time.perf_counter()
        fold_model.train_full(300)
        fold_model.predict(oofs["scratch-300"], n_sim=N_SIM, include_noise=True, where=held)
        score_fold(fold_model, oofs["scratch-300"], held, accs["scratch-300"])
        seconds["scratch-300"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        fold_model.train_full(300)
        bounds.append((fold_model.training_log[299], fold_model.training_log[-1]))
        fold_model.predict(oofs["scratch-600"], n_sim=N_SIM, include_noise=True, where=held)
        score_fold(fold_model, oofs["scratch-600"], held, accs["scratch-600"])
        seconds["scratch-600"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        gm._fresh_variational_state(fold_model, nodes=gm._terminal_gp_nodes(fold_model))
        fold_model._reset_optimizer()
        fold_model.train_full(200)
        fold_model.predict(oofs["scratch-leaves"], n_sim=N_SIM, include_noise=True, where=held)
        score_fold(fold_model, oofs["scratch-leaves"], held, accs["scratch-leaves"])
        seconds["scratch-leaves"] += time.perf_counter() - t0
        print("seed %d fold %d: bound at 300 %.1f, at 600 %.1f"
              % (SEED, i + 1, bounds[-1][0], bounds[-1][1]), flush=True)
    rows = []
    for arm, iterations in (("scratch-300", 300), ("scratch-600", 600), ("scratch-leaves", 200)):
        rows.append(summarize(arm, iterations, oofs[arm], table_of(accs[arm]), sd, seconds[arm]))
        print(rows[-1], flush=True)
    return rows


def main():
    train, held_out = geoml.datasets.jura()
    n_rock = len(train.get("Rock").labels)
    elements = list(train.get("Elements").labels)
    n_el = len(elements)
    y, has = train.variables["Elements"].get_measurements()
    y, has = np.asarray(y, float), np.asarray(has)
    sd = {c: float(np.std(y[has[:, j] == 1, j])) for j, c in enumerate(elements)}

    inducing = geoml.data.inducing.from_kmeans(train, 60, seed=0)
    train.spatial_k_fold(held_out, k=K, seed=SEED)
    labels = np.asarray(train.get_metadata("fold"))
    folds = np.unique(labels)

    rows = (reuse_mode if MODE == "reuse" else honest_mode)(
        train, inducing, n_rock, n_el, labels, folds, sd)
    table = pd.DataFrame(rows)
    table.insert(0, "seed", SEED)
    print(table.to_string(index=False))
    out = os.path.join(root, "docs", "benchmarks", "figures",
                       "leaf_refit_%s_seed%d.csv" % (MODE, SEED))
    table.to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    main()
