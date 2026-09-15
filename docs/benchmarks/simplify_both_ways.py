"""`Mesh3D.simplify` measured both ways on real shells, 2026-09-14.

Usage: python docs/benchmarks/simplify_both_ways.py <folder of the Assen
       shells> <label>

The Assen BIF and Hematite shells simplified at 0.5, 1 and 2 m: the
triangles kept, the time, the kind, and how far the answer strays from the
original either way -- the simplified faces (centroids and edge midpoints)
from the original surface, and the original's vertices from the simplified
one. `GEOML_ROOT` picks the checkout. Also, where the checkout's `simplify`
takes a quadric pre-pass, whether that pre-pass is still a body. The
results go to `docs/benchmarks/figures/simplify_both_ways_<label>.txt`.
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
import geoml.data.meshes as msh  # noqa: E402

FOLDER = sys.argv[1]
LABEL = sys.argv[2]
OUT = os.path.join(here, "figures", "simplify_both_ways_%s.txt" % LABEL)


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def deviations(original, simplified):
    """How far the simplified surface strays from the original, both ways."""
    points = np.asarray(simplified.coordinates, dtype=float)
    triangles = np.asarray(simplified.triangles)
    probes = np.concatenate([points[triangles].mean(axis=1),
                             (points[triangles[:, 0]]
                              + points[triangles[:, 1]]) / 2,
                             (points[triangles[:, 1]]
                              + points[triangles[:, 2]]) / 2,
                             (points[triangles[:, 2]]
                              + points[triangles[:, 0]]) / 2])
    shift = np.round(np.ravel(original.bounding_box.min))
    with msh._DistanceQueries(
            original._polydata().translate(-shift)) as measure:
        out = float(np.abs(measure.query(probes - shift)).max())
    # the original's vertices a triangle uses: the BIF shell carries 35 that
    # none does, up to 0.72 m off its own surface
    used = np.unique(np.asarray(original.triangles))
    with msh._DistanceQueries(
            simplified._polydata().translate(-shift)) as measure:
        back = float(np.abs(measure.query(
            np.asarray(original.coordinates, dtype=float)[used]
            - shift)).max())
    return out, back


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").close()
    say("label %s, geoML from %s" % (LABEL, root))
    for name in ("BIF", "Hematite"):
        shell = geoml.data.Solid3D.open(os.path.join(FOLDER, name + ".zarr"))
        polydata = shell._polydata()
        n = polydata.n_cells
        rough = polydata.decimate(1.0 - 50_000.0 / n,
                                  volume_preservation=True)
        rough = rough.clean().triangulate()
        try:
            kind = type(msh._rebuilt_as(geoml.data.Solid3D, rough)).__name__
        except Exception as error:
            kind = type(error).__name__
        say("\n%s: %d triangles, volume %.6g; its quadric pre-pass to %d "
            "triangles comes back %s" % (name, n, shell.volume,
                                         rough.n_cells, kind))
        say("  %-7s %9s %8s %-8s %11s %7s %7s %s" % (
            "budget", "triangles", "time", "kind", "volume", "out", "back",
            "whole?"))
        for budget in (0.5, 1.0, 2.0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                slim = shell.simplify(budget)
                took = time.perf_counter() - start
            out, back = deviations(shell, slim)
            whole = any("as it came" in str(w.message) for w in caught)
            say("  %-7g %9d %7.1fs %-8s %11.6g %7.3f %7.3f %s" % (
                budget, len(slim.triangles), took, type(slim).__name__,
                slim.volume, out, back, "returned whole" if whole else ""))
    say("done")
