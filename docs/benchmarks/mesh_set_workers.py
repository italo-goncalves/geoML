"""How a mesh set's realizations scale with workers, and where the time
goes, 2026-09-14.

Usage: python docs/benchmarks/mesh_set_workers.py <folder holding
       BM_sub_blocked> <label> [workers, comma-separated] [realizations]

The Assen FeO_total set at four cut-offs, its first `realizations` (8 by
default) contoured with each number of workers in turn (1, 2, 4 and 8 by
default). Timed apart: reading the realizations' columns, the pool as a
whole, each realization's task inside its worker -- the wall clock, the CPU
the worker spent, the threads it ran, and its stages (the cut, the paint,
the cut to the box, the classification, the comparison with the
prediction's shell, the measures) -- the parent writing every mesh to the
store, and the bytes each task sent back. `WORKER_BLAS_THREADS=n` in the
environment holds OpenBLAS to n threads in each task, the experiment that
found its spinning; the workers hold it to one themselves since. On
2026-09-11 the set of 25 realizations took 27 s a realization on eight
workers against 59 s in one process, 2.2 times as fast. The results go to
`docs/benchmarks/figures/mesh_set_workers_<label>.txt`: `baseline` as the
workers then were, `blas1` and `blas1_wide` with one BLAS thread, the
second on 24 realizations at 8, 12 and 24 workers.
"""
import os
import sys
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
root = os.environ.get("GEOML_ROOT") or os.path.abspath(
    os.path.join(here, "..", ".."))
sys.path.insert(0, root)
import numpy as np  # noqa: E402

import geoml  # noqa: E402
import geoml.data.meshsets as msm  # noqa: E402

FOLDER = sys.argv[1]
LABEL = sys.argv[2]
WORKERS = [int(w) for w in (sys.argv[3] if len(sys.argv) > 3
                            else "1,2,4,8").split(",")]
COUNT = int(sys.argv[4]) if len(sys.argv) > 4 else 8
CUTOFFS = [0.6, 0.7, 0.8, 0.85]
OUT = os.path.join(here, "figures", "mesh_set_workers_%s.txt" % LABEL)


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def threads():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return None


# the task, timed where it runs: in a forked worker, whose copy of this
# module the patches are part of -- the task whole, and its stages
_task = msm._realization_task
STAGES = {}


def _staged(name, function):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            STAGES[name] = STAGES.get(name, 0.0) + time.perf_counter() - start
    return wrapped


import geoml.data.blocks as blk  # noqa: E402
blk.BlockSet3D._cut_to_contour = _staged("cut",
                                         blk.BlockSet3D._cut_to_contour)
blk._painted_contour = _staged("paint", blk._painted_contour)
blk._clip_to_box = _staged("clip", blk._clip_to_box)
blk.mesh3d = _staged("classify", blk.mesh3d)
msm._against = _staged("against", msm._against)
msm._measure = _staged("measure", msm._measure)


def _timed_task(position):
    if os.environ.get("WORKER_BLAS_THREADS"):
        # an experiment: OpenBLAS keeps a thread per CPU in every process
        import threadpoolctl
        threadpoolctl.threadpool_limits(
            int(os.environ["WORKER_BLAS_THREADS"]), user_api="blas")
    STAGES.clear()
    wall, cpu = time.perf_counter(), time.process_time()
    number, arrays, measures, taken, failed = _task(position)
    sent = sum(np.asarray(a).nbytes for mesh in arrays.values()
               for a in (mesh if isinstance(mesh, (tuple, list))
                         else [mesh]) if a is not None
               and hasattr(a, "nbytes"))
    measures[next(iter(measures))]["_timing"] = (
        time.perf_counter() - wall, time.process_time() - cpu, threads(),
        sent, dict(STAGES))
    return number, arrays, measures, taken, failed


msm._realization_task = _timed_task
spent = {"read": 0.0, "write": 0.0}
_read = msm._read_realizations
_write = msm._write_arrays


def _timed_read(*args, **kwargs):
    start = time.perf_counter()
    out = _read(*args, **kwargs)
    spent["read"] += time.perf_counter() - start
    return out


def _timed_write(*args, **kwargs):
    start = time.perf_counter()
    out = _write(*args, **kwargs)
    spent["write"] += time.perf_counter() - start
    return out


msm._read_realizations = _timed_read
msm._write_arrays = _timed_write
timings = []
_each = msm._each_realization


def _collected(count, workers, first=0):
    start = time.perf_counter()
    for number, arrays, measures, taken, failed in _each(count, workers,
                                                         first=first):
        timing = measures[next(iter(measures))].pop("_timing")
        timings.append(timing)
        yield number, arrays, measures, taken, failed
    spent["pool"] = spent.get("pool", 0.0) + time.perf_counter() - start


msm._each_realization = _collected


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").close()
    say("label %s, geoML from %s, BLAS threads in the workers: %s"
        % (LABEL, root, os.environ.get("WORKER_BLAS_THREADS", "as they come")))
    blocks = geoml.data.BlockSet3D.open(os.path.join(FOLDER, "BM_sub_blocked"))
    say("%d blocks, %d CPUs, FeO_total at %s, realizations 0-%d"
        % (blocks.n_data, os.cpu_count(), CUTOFFS, COUNT - 1))
    start = time.perf_counter()
    msm.MeshSet(blocks, "Comp/FeO_total", cutoffs=CUTOFFS, simulations=False)
    alone = time.perf_counter() - start
    say("the prediction's shells alone: %.0f s" % alone)
    scratch = tempfile.mkdtemp(prefix="mesh_set_workers_")
    rows = []
    for workers in WORKERS:
        spent.update(read=0.0, write=0.0, pool=0.0)
        timings.clear()
        start = time.perf_counter()
        msm.MeshSet(blocks, "Comp/FeO_total", cutoffs=CUTOFFS,
                    simulations=list(range(COUNT)), workers=workers,
                    store=os.path.join(scratch, "w%d.zarr" % workers))
        total = time.perf_counter() - start - alone
        wall = np.array([t[0] for t in timings])
        cpu = np.array([t[1] for t in timings])
        rows.append((workers, total))
        say("\n== %d worker%s: %.0f s past the prediction, %.1f s a "
            "realization" % (workers, "" if workers == 1 else "s", total,
                             total / COUNT))
        say("  reading the realizations %.1f s; the pool %.0f s, of it the "
            "parent writing %.1f s" % (spent["read"], spent["pool"],
                                       spent["write"]))
        say("  a task in its worker: %.1f s (%.1f-%.1f), %.1f s of CPU, "
            "%.2f cores, up to %d threads, %.0f MB sent back"
            % (wall.mean(), wall.min(), wall.max(), cpu.mean(),
               cpu.sum() / wall.sum(), max(t[2] for t in timings),
               np.mean([t[3] for t in timings]) / 1e6))
        names = sorted({n for t in timings for n in t[4]})
        say("  its stages, mean s: " + ", ".join(
            "%s %.1f" % (n, np.mean([t[4].get(n, 0.0) for t in timings]))
            for n in names))
    one = dict(rows).get(1)
    if one:
        say("\nspeed-up: %s" % ", ".join("%d: %.1fx" % (w, one / t)
                                         for w, t in rows))
    say("done")
