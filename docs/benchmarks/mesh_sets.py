"""Mesh sets on the Assen block model, 2026-09-10.

Usage: python docs/benchmarks/mesh_sets.py <folder holding BM_sub_blocked>
       [workers]

The gate the design set before the defaults were fixed: on a real model
(908 237 blocks, 25 realizations per variable), how much consecutive shells
cross each other -- as contoured, and after simplifying -- and what a
realization costs. Also the categorical set of the six rocks: how much the
bodies overlap, the gap they leave, and how their volumes spread across the
realizations.

FeO_total is stored as a fraction and declares no cut-offs, so the shells
are taken at 0.6, 0.7, 0.8 and 0.85 -- about its quartiles and its ninth
decile. The rock model was trained with `CategoricalGaussianIndicator`,
whose rule is the set's default, `"largest"`.
"""
import os
import resource
import sys
import tempfile
import time

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import geoml  # noqa: E402

FOLDER = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CUTOFFS = [0.6, 0.7, 0.8, 0.85]
OUT = os.path.join(root, "docs", "benchmarks", "figures", "mesh_sets.txt")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def peak():
    """Peak resident memory, this process and its workers, in GB."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    workers = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return own / 1e6, workers / 1e6


def timed(label, function):
    t0 = time.perf_counter()
    value = function()
    say("%-58s %7.1f s" % (label, time.perf_counter() - t0))
    return value


open(OUT, "w").close()
scratch = tempfile.mkdtemp(prefix="mesh_sets_bench_")
blocks = geoml.data.BlockSet3D.open(os.path.join(FOLDER, "BM_sub_blocked"))
say("%d blocks, %d CPUs, %d workers" % (blocks.n_data, os.cpu_count(),
                                         WORKERS))

say("")
say("== FeO_total at %s" % CUTOFFS)
timed("the prediction's shells alone", lambda: geoml.data.MeshSet(
    blocks, "Comp/FeO_total", cutoffs=CUTOFFS, simulations=False))
timed("one realization besides, in this process", lambda: geoml.data.MeshSet(
    blocks, "Comp/FeO_total", cutoffs=CUTOFFS, simulations=[0], workers=1,
    store=os.path.join(scratch, "one.zarr")))
fe = timed("every realization, %d workers" % WORKERS,
           lambda: geoml.data.MeshSet(
               blocks, "Comp/FeO_total", cutoffs=CUTOFFS, workers=WORKERS,
               store=os.path.join(scratch, "fe.zarr")))
say("peak memory: %.1f GB here, %.1f GB the largest worker" % peak())
say("failures:", fe.failures)
say(fe.table())
say(fe.check())
say(fe.volume_dispersion())
say(fe.connectivity())
worst = [fe.simulations[i].check()["volume"].max()
         for i in range(len(fe.simulations))]
say("largest crossing in any realization's set: %.4g m3" % max(worst))
simple = timed("simplified to 1 m, nested again", lambda: fe.simplify(1.0))
say("what nesting the simplified shells took back:")
say(simple.repairs)
say("triangles as contoured and simplified:",
    [len(fe[c].triangles) for c in CUTOFFS],
    [len(simple[c].triangles) for c in CUTOFFS])

say("")
say("== Simple Rock, one body per rock")
rock = timed("the prediction's bodies and every realization's",
             lambda: geoml.data.MeshSet(
                 blocks, "Simple Rock", workers=WORKERS,
                 store=os.path.join(scratch, "rock.zarr")))
say("peak memory: %.1f GB here, %.1f GB the largest worker" % peak())
say("failures:", rock.failures)
say(rock.table())
checked = rock.check()
say(checked)
say(rock.volume_dispersion())
gaps = [rock.simulations[i].check().set_index("kind").loc["gap", "share"]
        for i in (0, 12, 24)]
overlaps = [rock.simulations[i].check().query("kind == 'overlap'")[
    "volume"].sum() for i in (0, 12, 24)]
say("realizations 0, 12, 24: overlap %s m3, gap share %s"
    % (np.round(overlaps, 4).tolist(), np.round(gaps, 5).tolist()))
say("done")
