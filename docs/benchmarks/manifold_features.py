"""Manifold's own mesh operations against geoML's, 2026-09-10.

Usage: python docs/benchmarks/manifold_features.py <folder of the Assen shells>

What Manifold offers that geoML already does itself, measured on the same
bodies:

simplify    `Mesh3D.simplify(max_error)` -- decimation to a geometric error
            budget, enforced by measuring -- against `Manifold.simplify(tol)`
            at the same number. Both answers are measured the same way: the
            deviation of the simplified faces from the original (centroids
            and edge midpoints) and of the original vertices from the
            simplified surface, the triangles kept, the time, the volume, and
            whether the answer is still a consistent body.
split       `Mesh3D.split()` against `Manifold.decompose()`, on a body in many
            pieces: the count of pieces, their volumes, the time.
level_set   `Grid3D` contouring (VTK flying edges on sampled values) against
            `Manifold.level_set` on the same field, handed over as a function.

The Assen shells sit at mine coordinates, so Manifold gets them in the local
frame and in double precision, as the booleans do.
"""
import os
import sys
import time

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import manifold3d  # noqa: E402
import numpy as np  # noqa: E402

import geoml  # noqa: E402
import geoml.math.geometry as gmt  # noqa: E402
from geoml.data.meshes import (  # noqa: E402
    _DistanceQueries, _from_manifold, _to_manifold, mesh3d)

FOLDER = sys.argv[1]
OUT = os.path.join(root, "docs", "benchmarks", "figures",
                   "manifold_features.txt")


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


open(OUT, "w").close()


def deviations(original, simplified):
    """How far the simplified surface strays from the original, both ways."""
    points = np.asarray(simplified.coordinates, dtype=float)
    triangles = np.asarray(simplified.triangles)
    probes = np.concatenate([points[triangles].mean(axis=1),
                             (points[triangles[:, 0]]
                              + points[triangles[:, 1]]) / 2])
    shift = np.round(np.ravel(original.bounding_box.min))
    with _DistanceQueries(original._polydata().translate(-shift)) as measure:
        out = float(np.abs(measure.query(probes - shift)).max())
    with _DistanceQueries(
            simplified._polydata().translate(-shift)) as measure:
        back = float(np.abs(measure.query(
            np.asarray(original.coordinates, dtype=float) - shift)).max())
    return out, back


def manifold_simplify(solid, tolerance):
    shift = np.round(np.ravel(solid.bounding_box.min))
    body = _to_manifold(solid, shift)
    t0 = time.perf_counter()
    slim = body.simplify(tolerance)
    slim.num_tri()
    worked = time.perf_counter() - t0
    return _from_manifold(slim, shift), worked


def report_simplify(label, solid, budgets):
    say("")
    say("simplify, %s: %d triangles, volume %.6g" % (
        label, len(np.asarray(solid.triangles)), solid.volume))
    say("  %-9s %-7s %9s %8s %-8s %11s %9s %9s" % (
        "engine", "budget", "triangles", "time", "kind", "volume",
        "out", "back"))
    for budget in budgets:
        t0 = time.perf_counter()
        ours = solid.simplify(budget)
        ours_time = time.perf_counter() - t0
        theirs, theirs_time = manifold_simplify(solid, budget)
        for engine, mesh, worked in (("geoML", ours, ours_time),
                                     ("Manifold", theirs, theirs_time)):
            out, back = deviations(solid, mesh)
            volume = mesh.volume if isinstance(mesh, geoml.data.Solid3D) \
                else gmt.signed_volume(np.asarray(mesh.coordinates),
                                       np.asarray(mesh.triangles))
            say("  %-9s %-7g %9d %7.1fs %-8s %11.6g %9.3g %9.3g" % (
                engine, budget, len(np.asarray(mesh.triangles)), worked,
                type(mesh).__name__, volume, out, back))


# ---- simplify ---------------------------------------------------------------
# the test suite's contoured ball, and two real shells
grid = geoml.data.Grid3D(start=[0, 0, 0], n=[40, 40, 40], step=[0.5] * 3)
distance = np.linalg.norm(np.asarray(grid.coordinates) - 9.75, axis=1)
grid.add_continuous_variable("v", distance)
ball = grid.variables["v"].measurements.get_contour(6.0)
report_simplify("contoured ball", ball, [0.05, 0.2, 0.5])

for name in ("BIF", "Hematite"):
    shell = geoml.data.Solid3D.open(os.path.join(FOLDER, name + ".zarr"))
    report_simplify("Assen %s" % name, shell, [0.5, 2.0])

# ---- split ------------------------------------------------------------------
shell = geoml.data.Solid3D.open(os.path.join(FOLDER, "Diabase.zarr"))
say("")
t0 = time.perf_counter()
pieces = shell.split()
ours_time = time.perf_counter() - t0
shift = np.round(np.ravel(shell.bounding_box.min))
body = _to_manifold(shell, shift)
t0 = time.perf_counter()
parts = body.decompose()
theirs_time = time.perf_counter() - t0
ours_volumes = sorted((p.volume for p in pieces
                       if isinstance(p, geoml.data.Solid3D)), reverse=True)
theirs_volumes = sorted((p.volume() for p in parts), reverse=True)
say("split, Assen Diabase: geoML %d pieces in %.1f s, Manifold %d in %.1f s; "
    "largest volumes %s against %s" % (
        len(pieces), ours_time, len(parts), theirs_time,
        np.round(ours_volumes[:3], 1), np.round(theirs_volumes[:3], 1)))
t0 = time.perf_counter()
back = [_from_manifold(p, shift) for p in parts]
say("  and handing Manifold's pieces back as meshes: %.1f s more"
    % (time.perf_counter() - t0))

# ---- level_set --------------------------------------------------------------
say("")
for n in (40, 80):
    step = 20.0 / n
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[n, n, n], step=[step] * 3)
    values = 6.0 - np.linalg.norm(np.asarray(grid.coordinates) - 9.75,
                                  axis=1)
    grid.add_continuous_variable("v", values)
    t0 = time.perf_counter()
    ours = grid.variables["v"].measurements.get_contour(0.0)
    ours_time = time.perf_counter() - t0

    def field(x, y, z):
        # positive inside, as Manifold reads a level set
        return 6.0 - ((x - 9.75) ** 2 + (y - 9.75) ** 2
                      + (z - 9.75) ** 2) ** 0.5

    t0 = time.perf_counter()
    theirs = manifold3d.Manifold.level_set(
        field, [0.0, 0.0, 0.0, 20.0, 20.0, 20.0], step, 0.0)
    theirs.num_tri()
    theirs_time = time.perf_counter() - t0
    exact = 4.0 / 3.0 * np.pi * 6.0 ** 3
    say("level_set, a ball on %d^3: geoML %s %d tris in %.2f s, volume %.2f%% "
        "off; Manifold %d tris in %.2f s, volume %.2f%% off" % (
            n, type(ours).__name__, len(np.asarray(ours.triangles)),
            ours_time, 100 * (ours.volume - exact) / exact,
            theirs.num_tri(), theirs_time,
            100 * (theirs.volume() - exact) / exact))
say("done")
