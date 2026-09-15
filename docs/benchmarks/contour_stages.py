"""What one contour of a block model costs, stage by stage, and what it gets
wrong, 2026-09-13.

Usage: python docs/benchmarks/contour_stages.py <folder holding
       BM_sub_blocked> <label> [stages,census,slabs] [census workers]

The baseline the plan for the roadmap's section-4 mesh items is measured
against, run again after each step under a new label; the results go to
`docs/benchmarks/figures/contour_stages_<label>.txt`. Set `GEOML_ROOT` to
import geoML from another checkout -- the copy of the code as it stood
before a change, say -- rather than from this one.

stages
    The Assen prediction of FeO_total closed above at 0.6, 0.7, 0.8 and
    0.85, contoured the way a mesh set contours it
    (`_contour_values(..., fallback=False)`), at supersample 0 and at 1 -- the
    second cuts a lattice about the size a Tom v6 contour reaches. Each stage
    is timed and its peak memory read as growth over what the process held
    when it began (Linux `clear_refs`): the cut to the finest size, the
    paint, and the classification. Also the lattice at each level, the
    triangles, and the edges more than two triangles share, which no winding
    repair settles. Two realizations the 2026-09-11 benchmark retried or lost
    are added: FeO_total's 19 at 0.8, and Limestone's 17 of the rocks.
census
    Every realization's FeO_total shells and rock bodies contoured at their
    own level only (`_NUDGES = (0.0,)` while the realizations are made), so
    a shell the set would have retried is recorded as a failure, with why.
slabs
    Three categories in slabs filling an 80 m box of 10 m blocks: the gap
    `check()` reads, most of it where the caps round the box's edges.
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
import pandas as pd  # noqa: E402

import geoml  # noqa: E402
import geoml.data.blocks as blk  # noqa: E402
import geoml.data.meshsets as msm  # noqa: E402
import geoml.math.geometry as gmt  # noqa: E402

FOLDER = sys.argv[1]
LABEL = sys.argv[2]
PARTS = (sys.argv[3] if len(sys.argv) > 3 else "stages,census,slabs") \
    .split(",")
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
CUTOFFS = [0.6, 0.7, 0.8, 0.85]
GB = 1024 ** 3
OUT = os.path.join(here, "figures", "contour_stages_%s.txt" % LABEL)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# the stages, measured by wrapping the functions the contour calls
# --------------------------------------------------------------------------- #
stats = {}


def _reset():
    stats.clear()
    stats.update(cut_s=0.0, cut_gb=0.0, paint_s=0.0, paint_gb=0.0,
                 classify_s=0.0, levels=[], cells=None, kind=None,
                 triangles=0, shared=0)


# the wrappers also run inside a mesh set's contours, whose rows nobody reads
_reset()


def _timed_stage(name, function):
    def wrapped(*args, **kwargs):
        meter = msm._PeakMeter()
        start = time.perf_counter()
        out = function(*args, **kwargs)
        stats[name + "_s"] += time.perf_counter() - start
        growth = meter.growth()
        if growth is not None:
            stats[name + "_gb"] = max(stats[name + "_gb"], growth / GB)
        return out
    return wrapped


_cut = blk.BlockSet3D._cut_to_contour
blk.BlockSet3D._cut_to_contour = _timed_stage("cut", _cut)
_paint = blk._painted_contour


def _painted(origin, size, *args, **kwargs):
    stats["cells"] = len(origin)
    return _timed_stage("paint", _paint)(origin, size, *args, **kwargs)


blk._painted_contour = _painted
if hasattr(blk.BlockSet3D, "_shared_corners"):
    _shared = blk.BlockSet3D._shared_corners

    def _counted(origin, *args, **kwargs):
        stats["levels"].append(len(origin))
        return _shared(origin, *args, **kwargs)

    blk.BlockSet3D._shared_corners = staticmethod(_counted)
_mesh3d = blk.mesh3d


def _classified(points, triangles, *args, **kwargs):
    start = time.perf_counter()
    mesh = _mesh3d(points, triangles, *args, **kwargs)
    stats["classify_s"] += time.perf_counter() - start
    stats["kind"] = type(mesh).__name__
    stats["triangles"] = len(triangles)
    stats["shared"] = int(np.sum(gmt._edge_counts(
        points, triangles, 6, False) > 2))
    return mesh


blk.mesh3d = _classified


def contour_row(name, blocks, values, level, supersample):
    _reset()
    start = time.perf_counter()
    mesh = blocks._contour_values(values, level, name,
                                  supersample=supersample, close="above",
                                  fallback=False)
    total = time.perf_counter() - start
    healed = None
    if mesh is not None and type(mesh).__name__ == "Mesh3D":
        healed = type(mesh.heal()).__name__
    return {"contour": name, "level": level, "supersample": supersample,
            "total_s": round(total, 1), "cut_s": round(stats["cut_s"], 1),
            "paint_s": round(stats["paint_s"], 1),
            "classify_s": round(stats["classify_s"], 1),
            "cut_gb": round(stats["cut_gb"], 2),
            "paint_gb": round(stats["paint_gb"], 2),
            "levels": "/".join("%.2fM" % (n / 1e6) for n in stats["levels"]),
            "painted_M": None if stats["cells"] is None
            else round(stats["cells"] / 1e6, 2),
            "triangles": stats["triangles"], "shared_edges": stats["shared"],
            "kind": stats["kind"], "healed": healed}


def stages(blocks):
    say("\n== stages of one contour, in this process")
    _, column = blk._contour_column(blocks, "Comp/FeO_total")
    prediction = np.asarray(column.values, dtype=float).ravel()
    rows = []
    for supersample in (0, 1):
        for level in CUTOFFS:
            rows.append(contour_row("FeO_total", blocks, prediction, level,
                                    supersample))
            say(rows[-1])
    realization = np.asarray(
        blocks.values("Comp/FeO_total/simulations/19"), dtype=float).ravel()
    rows.append(contour_row("FeO_total r19", blocks, realization, 0.8, 0))
    say(rows[-1])
    rock = blocks.variables["Simple Rock"]
    labels = list(rock.labels)
    draws = np.stack([np.asarray(blocks.values(
        "Simple Rock/%s/simulations/17" % label), dtype=float).ravel()
        for label in labels], axis=1)
    fields = msm._category_fields(draws, "largest")
    j = labels.index("Limestone")
    rows.append(contour_row("Limestone r17", blocks, fields[:, j], 0.0, 0))
    say(rows[-1])
    say(pd.DataFrame(rows).set_index(["contour", "level", "supersample"]))


# --------------------------------------------------------------------------- #
# the census: every realization at its own level only
# --------------------------------------------------------------------------- #
def census(blocks, scratch):
    say("\n== census: realizations contoured at their own level only, %d "
        "workers" % WORKERS)
    realizations = msm.MeshSet._contour_realizations

    def own_level_only(self, *args, **kwargs):
        saved = msm._NUDGES
        msm._NUDGES = (0.0,)
        try:
            return realizations(self, *args, **kwargs)
        finally:
            msm._NUDGES = saved

    msm.MeshSet._contour_realizations = own_level_only
    try:
        for name, path, cutoffs in (("FeO_total", "Comp/FeO_total", CUTOFFS),
                                    ("rocks", "Simple Rock", None)):
            start = time.perf_counter()
            made = msm.MeshSet(blocks, path, cutoffs=cutoffs,
                               workers=WORKERS,
                               store=os.path.join(scratch, name + ".zarr"))
            took = time.perf_counter() - start
            moved = {key: value for key, value in made._nudge.items()
                     if value != 0.0}
            say("%s: %.0f s, %d realizations x %d meshes; the prediction's "
                "meshes moved off their level: %s"
                % (name, took, len(made.simulations), len(made), moved))
            say("  not closed at their own level: %d" % len(made.failures))
            for failure in made.failures:
                say("   realization %s, %s: %s" % (
                    failure["realization"], failure["key"],
                    failure["error"][:90]))
    finally:
        msm.MeshSet._contour_realizations = realizations


# --------------------------------------------------------------------------- #
# the slabs
# --------------------------------------------------------------------------- #
def slabs():
    say("\n== three slabs filling an 80 m box of 10 m blocks")
    blocks = geoml.data.BlockSet3D([0, 0, 0], [8, 8, 8], [10.0, 10.0, 10.0],
                                   discretization=(2, 2, 2), max_levels=1)
    x = np.asarray(blocks.coordinates)[:, 0]
    blocks.add_rock_type_variable("Rock", labels=["A", "B", "C"])
    rock = blocks.variables["Rock"]
    draws = np.stack([15.0 - np.abs(x - middle)
                      for middle in (5.0, 35.0, 65.0)], axis=1)
    skew = msm._category_fields(draws, "largest")
    for j, name in enumerate("ABC"):
        rock.components[name].indicator_predicted.values[:] = skew[:, j]
    rock.predicted.values[:] = np.argmax(draws, axis=1)
    made = msm.MeshSet(blocks, "Rock", simulations=False)
    say(made.check())


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").close()
    say("label %s, geoML from %s" % (LABEL, root))
    model = None
    if "stages" in PARTS or "census" in PARTS:
        model = geoml.data.BlockSet3D.open(os.path.join(FOLDER,
                                                        "BM_sub_blocked"))
        say("%d blocks, %d CPUs" % (model.n_data, os.cpu_count()))
    if "stages" in PARTS:
        stages(model)
    if "census" in PARTS:
        census(model, tempfile.mkdtemp(prefix="contour_census_"))
    if "slabs" in PARTS:
        slabs()
    say("done")
