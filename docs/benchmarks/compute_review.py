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

"""Where training and prediction spend their time: roadmap item L, phase 0.

Usage: python docs/benchmarks/compute_review.py CASE OUT [options]
  CASE    walker | experts | gradient | saved
  OUT     a directory: `results.json` and `reference.npz` land there
  --model PATH     the saved model (`saved` only): chapter 17's walked
                   model, or the GeoScape walk network (`A_model`)
  --rows N         predict at the first N data rows only (default all)
  --n-sim N        realizations a prediction draws (default 10)
  --steps N        training iterations timed (default 20; by expert, epochs)
  --parts LIST     comma-separated, of step,refresh,plain,predict,roots,cv
                   (default all but cv for `saved`)
  --no-xla         `network._QUADRATURE_XLA = False` for the whole run

The cases (the roadmap's A, B, E; C and D are `saved`):

- `walker`  Walker Lake, one expert of 300 k-means points, a Gaussian kernel
            at certain inputs -- the shared prior draw's realizations.
- `experts` chapter 15's model (16 experts of `grid_experts`, an
            exponential kernel) trained by expert.
- `gradient` a `GradientConstrainedInput` on `example_fold`, as `test_vgp`
            builds it.

What each part measures, every time with `time.perf_counter` and the
resident memory sampled every 50 ms in a thread (its peak over the part):

- `step`     the seconds of a training iteration (median past the first
             two), read off `geoml.progress`, and the training log.
- `refresh`  one traced refresh of every node plus the KL and their
             gradient: the work item 4 (one factorization) can cut.
- `plain`    the training step again with the first layer's
             `covariance_matrix` replaced by `_plain_covariance`: what item
             6 can cut (the difference).
- `predict`  `predict` into a container of coordinates alone, then into a
             copy of the data rows (the measurement pass behind `pit_*`),
             each twice and the second call timed (the first traces):
             per-batch seconds, the peak, and how many times
             `_chunked_sums`/`_chunked_map` ran *eagerly* (each one a
             `tf.function` built and, on the CPU, compiled with XLA: item
             1a), how often `_shared_prior_simulate` and the likelihoods'
             `_values_and_noise` were entered (item 2: per trace in a
             graph, per call eagerly).
- `roots`    the plain prediction again with the conditioned quadrature's
             `root_sums` loop answered by zeros: what item 3 can cut, at
             most (only where a node takes the quadrature).
- `cv`       `cross_validate` on three spatial folds, a short refit.

`reference.npz` keeps the predictions, realizations, `pit_*` columns and
the training log for the later gates; for a bit-for-bit gate run both
sides one-threaded (`TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
OMP_NUM_THREADS=1`).
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import tensorflow as tf  # noqa: E402

import geoml  # noqa: E402
import geoml.latent.network as net  # noqa: E402
import geoml.likelihood as lik  # noqa: E402


# --------------------------------------------------------------------------- #
# measuring
# --------------------------------------------------------------------------- #
def _rss():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


class Meter:
    """Seconds and the resident memory's peak and start over a block."""

    def __enter__(self):
        self.start_bytes = _rss()
        self.peak = self.start_bytes
        self._done = False

        def sample():
            while not self._done:
                self.peak = max(self.peak, _rss())
                time.sleep(0.05)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.started
        self._done = True
        self._thread.join()
        self.peak = max(self.peak, _rss())
        return False

    def report(self):
        return {"seconds": self.seconds,
                "rss_start_gb": self.start_bytes / 1e9,
                "rss_peak_gb": self.peak / 1e9,
                "rss_growth_gb": (self.peak - self.start_bytes) / 1e9}


class Counts:
    """Counts entries into the functions the items act on."""

    def __init__(self):
        self.counts = {}
        self._patched = []

    def wrap(self, owner, name, label=None):
        original = getattr(owner, name)
        label = label or name

        def counted(*args, **kwargs):
            key = label + (" eager" if tf.executing_eagerly() else " traced")
            self.counts[key] = self.counts.get(key, 0) + 1
            return original(*args, **kwargs)

        counted.__name__ = getattr(original, "__name__", name)
        setattr(owner, name, counted)
        self._patched.append((owner, name, original))

    def reset(self):
        self.counts = {}
        return self

    def restore(self):
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)


def _batch_times(events):
    """The seconds between consecutive events (time, done)."""
    return [t1 - t0 for (t0, _), (t1, _) in zip(events[:-1], events[1:])]


# --------------------------------------------------------------------------- #
# the cases
# --------------------------------------------------------------------------- #
def walker_model():
    geoml.set_seed(1234)
    walker, _ = geoml.datasets.walker()
    points = geoml.data.inducing.from_kmeans(walker, 300, seed=0)
    root = geoml.latent.BasicInput(points,
                                   transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    warping = geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1, shift=1.0), geoml.warping.ZScore(1))
    model = geoml.models.VGPNetwork(
        walker, "V", geoml.likelihood.Gaussian(warping), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=100)
    return model


def experts_model():
    geoml.set_seed(1234)
    walker, walker_grid = geoml.datasets.walker()
    experts = geoml.data.inducing.grid_experts(walker_grid, 10.0, block=8)
    root = geoml.latent.BasicInput(experts,
                                   transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1,
                              kernel=geoml.kernels.Exponential())
    warping = geoml.warping.ChainedWarping(
        geoml.warping.BoxCox(1, shift=1.0), geoml.warping.ZScore(1))
    model = geoml.models.VGPNetwork(
        walker, "V", geoml.likelihood.Gaussian(warping), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_by_expert(epochs=5)
    return model


def gradient_model():
    geoml.set_seed(1234)
    tf.random.set_seed(1234)
    point, dirs, _ = geoml.datasets.example_fold()
    ip = geoml.data.Grid2D(start=[10, 10], n=[8, 8], step=[11, 11])
    covariance = geoml.kernels.Covariance(
        geoml.kernels.Gaussian(), geoml.transform.Isotropic(30))
    root = geoml.latent.GradientConstrainedInput(ip, dirs, covariance,
                                                 size=1)
    model = geoml.models.VGPNetwork(
        point, "rock_num", geoml.likelihood.Gaussian(), root,
        options=geoml.models.GPOptions(verbose=False, training_samples=8))
    model.train_full(max_iter=50)
    return model


def _by_expert(model):
    return len(model.leaves[0].root.n_ip) > 1


# --------------------------------------------------------------------------- #
# the parts
# --------------------------------------------------------------------------- #
def part_step(model, steps):
    events = []

    def callback(p):
        if p.task == "train" and not p.within:
            events.append((time.perf_counter(), p.done))

    with geoml.progress(callback), Meter() as meter:
        if _by_expert(model):
            times = model.train_by_expert(epochs=steps)["seconds"]
        else:
            model.train_full(max_iter=steps)
            times = _batch_times(events)
    out = meter.report()
    out["unit"] = "epoch" if _by_expert(model) else "iteration"
    out["median"] = statistics.median(times[1:]) if len(times) > 1 \
        else out["seconds"]
    out["count"] = len(events)
    return out


def part_refresh(model, repeats=10):
    variables = model.get_unfixed_variables()

    @tf.function
    def refresh():
        with tf.GradientTape() as tape:
            model._refresh(model.options.jitter)
            kl = tf.add_n([node.kl_divergence() for node in model._nodes()])
        return kl, tape.gradient(
            kl, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    with model._propagation(rows=model.data.n_data):
        refresh()                                       # traced
        times = []
        with Meter() as meter:
            for _ in range(repeats):
                start = time.perf_counter()
                kl, _ = refresh()
                float(kl)
                times.append(time.perf_counter() - start)
    out = meter.report()
    out["median"] = statistics.median(times)
    return out


def _first_layer(model):
    return [n for n in model._nodes() if isinstance(n, net.BasicGP)
            and n.parent is n.root
            and type(n.root) is net.BasicInput]


def part_plain(model, steps):
    nodes = _first_layer(model)
    if not nodes:
        return {"skipped": "no GP node reads a certain input directly"}
    originals = {id(n): n.covariance_matrix for n in nodes}
    for n in nodes:
        n.covariance_matrix = (lambda node: lambda x, y, var_x=None,
                               var_y=None: node._plain_covariance(x, y))(n)
    model._step = None                                  # retrace the step
    try:
        out = part_step(model, steps)
    finally:
        for n in nodes:
            del n.covariance_matrix
        model._step = None
    out["nodes"] = [n.name for n in nodes]
    del originals
    return out


def _targets(model, rows):
    data = model.data
    keep = np.arange(data.n_data if rows is None else min(rows, data.n_data))
    measured = data[keep]
    plain = geoml.data.PointData.from_array(
        np.asarray(data.coordinates)[keep])
    return plain, measured


def _predict_once(model, target, n_sim, counts):
    events = []

    def callback(p):
        if p.task == "predict" and not p.within:
            events.append((time.perf_counter(), p.done))

    counts.reset()
    rows = model._prediction_rows()
    start = time.perf_counter()
    with geoml.progress(callback), Meter() as meter:
        model.predict(target, n_sim=n_sim)
    out = meter.report()
    # the event for batch i comes once batch i - 1 has landed
    batches = _batch_times(events)
    out["batches"] = batches
    out["batch_count"] = len(events)
    out["rows_per_batch_estimate"] = rows
    out["batch_median"] = statistics.median(batches) if batches else None
    out["prediction_seconds"] = (events[-1][0] - start) if events else None
    out["after_batches_seconds"] = meter.seconds - out["prediction_seconds"] \
        if events else None
    out["counts"] = dict(counts.counts)
    return out


def _arrays(target, label):
    out = {}
    paths = [path for pattern in ("**/latent_mean", "**/latent_variance",
                                  "**/prediction", "**/simulations")
             for path in target.select(pattern)]
    for path in paths:
        values = target.values(str(path))
        if values is not None:
            out["%s|%s" % (label, path)] = np.asarray(values, float)
    for name in list(target.metadata):
        if name.startswith(("pit_", "warped_")):
            out["%s|_metadata/%s" % (label, name)] = np.asarray(
                target.get_metadata(name), float)
    return out


def part_predict(model, rows, n_sim, counts, arrays):
    plain, measured = _targets(model, rows)
    out = {"rows": plain.n_data}
    # each target twice: the first call traces (its time kept as
    # `first`), the second is the one timed
    for label, target in (("plain", plain), ("measured", measured)):
        first = _predict_once(model, target, n_sim, counts)
        out[label] = _predict_once(model, target, n_sim, counts)
        out[label]["first"] = {k: first[k] for k in (
            "seconds", "rss_growth_gb", "counts")}
        arrays.update(_arrays(target, label))
    out["measurement_pass_seconds"] = \
        out["measured"]["seconds"] - out["plain"]["seconds"]
    return out


def _zero_root_sums(original):
    def patched(fn, inputs, n_chunks):
        if getattr(fn, "__name__", "") != "root_sums":
            return original(fn, inputs, n_chunks)
        mean, root_r = inputs[0], inputs[9]
        lead = tf.shape(mean)[:-2]
        n = tf.shape(mean)[-2:-1]
        s, j = tf.shape(root_r)[-3:-2], tf.shape(root_r)[-1:]
        return (tf.zeros(tf.concat([lead, s, n, j], 0), tf.float64),
                tf.zeros(tf.concat([lead, s, n], 0), tf.float64),
                tf.zeros(tf.concat([lead, s, n], 0), tf.float64))
    return patched


def part_roots(model, rows, n_sim, counts):
    if not any(getattr(n, "_takes_quadrature", lambda: False)()
               for n in model._nodes()):
        return {"skipped": "no node takes the conditioned quadrature"}
    plain, _ = _targets(model, rows)
    original = net._chunked_sums
    net._chunked_sums = _zero_root_sums(original)
    model._compiled = {}                                # retrace
    try:
        first = _predict_once(model, plain, n_sim, counts)   # traces
        out = _predict_once(model, plain, n_sim, counts)
        out["first"] = {k: first[k] for k in (
            "seconds", "rss_growth_gb", "counts")}
    finally:
        net._chunked_sums = original
        model._compiled = {}
    return out


def part_cv(model, arrays):
    walker_grid = None
    if model.data.n_dim == 2:
        lower = np.min(model.data.coordinates, 0)
        upper = np.max(model.data.coordinates, 0)
        n = np.maximum(((upper - lower) / 5).astype(int), 2)
        walker_grid = geoml.data.Grid2D(start=lower, n=n, step=5)
    geoml.set_seed(0)
    model.data.spatial_k_fold(walker_grid if walker_grid is not None
                              else model.data, k=3, seed=0)
    geoml.set_seed(0)
    with Meter() as meter:
        oof, scores = geoml.models.cross_validate(model, iterations=30,
                                                  n_sim=10)
    out = meter.report()
    out["scores"] = json.loads(scores.to_json(orient="records"))
    arrays.update(_arrays(oof, "cv"))
    return out


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["walker", "experts", "gradient",
                                         "saved"])
    parser.add_argument("out")
    parser.add_argument("--model")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--n-sim", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--parts")
    parser.add_argument("--no-xla", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.no_xla:
        net._QUADRATURE_XLA = False

    default = "step,refresh,plain,predict,roots" if args.case == "saved" \
        else "step,refresh,plain,predict,roots,cv"
    parts = (args.parts or default).split(",")

    counts = Counts()
    counts.wrap(net, "_chunked_sums")
    counts.wrap(net, "_chunked_map")
    counts.wrap(net.BasicGP, "_shared_prior_simulate")
    counts.wrap(lik._ContinuousLikelihood, "_values_and_noise")

    results = {"case": args.case, "model": args.model, "rows": args.rows,
               "n_sim": args.n_sim, "steps": args.steps,
               "quadrature_xla": net._quadrature_xla(),
               "gpu": net._gpu_visible(),
               "threads": os.environ.get("TF_NUM_INTRAOP_THREADS"),
               "version": geoml.__version__}
    arrays = {}

    def save():
        with open(os.path.join(args.out, "results.json"), "w") as file:
            json.dump(results, file, indent=1, default=str)
        np.savez(os.path.join(args.out, "reference.npz"), **arrays)

    with Meter() as meter:
        if args.case == "saved":
            model = geoml.persistence.load_model(args.model)
        else:
            model = {"walker": walker_model, "experts": experts_model,
                     "gradient": gradient_model}[args.case]()
    results["build"] = meter.report()
    results["nodes"] = [type(n).__name__ for n in model._nodes()]
    print("built in %.1f s" % meter.seconds, flush=True)

    for part in parts:
        print("part %s" % part, flush=True)
        try:
            if part == "step":
                results[part] = part_step(model, args.steps)
                arrays["training_log"] = np.asarray(model.training_log)
            elif part == "refresh":
                results[part] = part_refresh(model)
            elif part == "plain":
                results[part] = part_plain(model, args.steps)
            elif part == "predict":
                results[part] = part_predict(model, args.rows, args.n_sim,
                                             counts, arrays)
            elif part == "roots":
                results[part] = part_roots(model, args.rows, args.n_sim,
                                           counts)
            elif part == "cv":
                results[part] = part_cv(model, arrays)
        except Exception:
            import traceback
            results[part] = {"error": traceback.format_exc()}
            traceback.print_exc()
        save()
        print(json.dumps({part: {k: v for k, v in results[part].items()
                                 if k not in ("batches", "scores")}
                                if isinstance(results[part], dict) else
                                results[part]},
                         default=str)[:2000], flush=True)
    counts.restore()


if __name__ == "__main__":
    main()
