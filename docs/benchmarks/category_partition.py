"""A categorical realization's bodies: contoured one category at a time, or
all of them from one contour of the draws, 2026-09-14.

Usage: python docs/benchmarks/category_partition.py <folder holding
       BM_sub_blocked> <label>

The Assen rock model's realizations 0, 12 and 24, the six rocks of each made
into a mesh set both ways, in one process: each rock contoured on its own
field -- its draw against the best of the others', taken block by block,
which is how every realization was made until this day -- and every rock
from one cut and one paint of the draws, each field read off the draws'
corner means (`BlockSet3D._contour_fields`). For each, how much of the
model the realization's bodies claim twice and how much none of them
claims, as `MeshSet.check()` measures it, and the time a realization took.
The results go to `docs/benchmarks/figures/category_partition_<label>.txt`.
"""
import os
import sys
import time
import warnings

here = os.path.dirname(os.path.abspath(__file__))
root = os.environ.get("GEOML_ROOT") or os.path.abspath(
    os.path.join(here, "..", ".."))
sys.path.insert(0, root)
import numpy as np  # noqa: E402

import geoml  # noqa: E402
import geoml.data.blocks as blk  # noqa: E402
import geoml.data.meshsets as msm  # noqa: E402

FOLDER = sys.argv[1]
LABEL = sys.argv[2]
REALIZATIONS = [0, 12, 24]
OUT = os.path.join(here, "figures", "category_partition_%s.txt" % LABEL)


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def one_at_a_time(*args, **kwargs):
    # what a realization falls back to when the partition in one contour
    # fails, and so what every realization did before it existed
    raise RuntimeError("each category on its own")


def shares(checked, box):
    """The volume claimed twice and the volume claimed by none, as shares of
    the model: `check()` gives each overlap as a share of the smaller body."""
    return (100 * checked.query("kind == 'overlap'")["volume"].sum() / box,
            100 * checked.set_index("kind").loc["gap", "volume"] / box)


def measured(made, box):
    return [shares(made.simulations[i].check(), box)
            for i in range(len(made.simulations))]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").close()
    say("label %s, geoML from %s" % (LABEL, root))
    blocks = geoml.data.BlockSet3D.open(os.path.join(FOLDER, "BM_sub_blocked"))
    say("%d blocks, realizations %s, one process" % (blocks.n_data,
                                                    REALIZATIONS))
    together = blk.BlockSet3D._contour_fields
    box = float(np.prod(np.asarray(blocks.lattice_shape)
                        * np.asarray(blocks.base_step, dtype=float)))
    start = time.perf_counter()
    prediction = msm.MeshSet(blocks, "Simple Rock", simulations=False)
    alone = time.perf_counter() - start
    say("the prediction's bodies: %.0f s; overlap %.4f%% of the model, gap "
        "%.4f%%" % ((alone,) + shares(prediction.check(), box)))
    for route in ("one at a time", "all at once"):
        blk.BlockSet3D._contour_fields = one_at_a_time \
            if route == "one at a time" else together
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            made = msm.MeshSet(blocks, "Simple Rock",
                               simulations=REALIZATIONS, workers=1)
            took = time.perf_counter() - start - alone
        say("\n== %s: %.0f s a realization, %d failures" % (
            route, took / len(REALIZATIONS), len(made.failures)))
        for number, (overlap, gap) in zip(REALIZATIONS, measured(made, box)):
            say("  realization %2d: overlap %.4f%%, gap %.4f%%" % (
                number, overlap, gap))
    blk.BlockSet3D._contour_fields = together
    say("done")
