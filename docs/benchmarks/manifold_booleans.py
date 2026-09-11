"""Solid booleans through Manifold against the implicit engine, 2026-09-10.

Usage: python docs/benchmarks/manifold_booleans.py <folder of the Assen shells>

Every pair of the six Assen rock shells (325k-681k triangles each, at mine
coordinates) -- the workload behind the implicit engine's 95 s -- intersected
three ways, and a few pairs also joined and subtracted:

implicit    `Solid3D.intersection` as it stands: `_resolved`, then the
            signed-distance grid, exact to the step its warning names.
manifold64  manifold3d directly: the vertices welded, moved to the pair's
            rounded lower corner and handed over in float64 (`Mesh64`), the
            answer read back with `to_mesh64` and classified by `mesh3d`.
accessor32  pyvista-manifold's `mesh.manifold.<operation>`, which casts every
            vertex to float32: in the same local frame, and at the mine
            coordinates as given.

Every Manifold call runs in a forked child, so a crash -- VTK's exact filter
segfaulted on contour-derived shells like these -- is a row of the table
rather than the end of the run.
"""
import itertools
import multiprocessing as mp
import os
import re
import sys
import time
import warnings

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# GEOML_ROOT points the `implicit` arm at a checkout that still has the
# signed-distance grid, which the working tree no longer does since the swap
sys.path.insert(0, os.environ.get("GEOML_ROOT", root))
import manifold3d  # noqa: E402
import numpy as np  # noqa: E402
import pyvista_manifold  # noqa: E402,F401 -- registers the accessor

import geoml  # noqa: E402
import geoml.math.geometry as gmt  # noqa: E402
from geoml.data.meshes import mesh3d  # noqa: E402

FOLDER = sys.argv[1]
ROCKS = ["BIF", "Calcitic Hematite", "Diabase", "Hematite", "Limestone",
         "Shale"]
TIMEOUT = 900
OUT = os.path.join(root, "docs", "benchmarks", "figures",
                   "manifold_booleans.txt")


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


open(OUT, "w").close()
solids = {name: geoml.data.Solid3D.open(os.path.join(FOLDER, name + ".zarr"))
          for name in ROCKS}
for name, solid in solids.items():
    say("%-18s %7d triangles, volume %.4g" % (
        name, len(np.asarray(solid.triangles)), solid.volume))


def corner(a, b):
    """The pair's rounded lower corner, as `_local_frame` takes it."""
    return np.round(np.minimum(np.ravel(a.bounding_box.min),
                               np.ravel(b.bounding_box.min)))


def summary(points, triangles):
    """An answer as geoML would hold it: its class, volume and size."""
    if len(triangles) == 0:
        return "empty", 0.0, 0
    mesh = mesh3d(points, triangles, gmt.vertex_normals(points, triangles))
    # about the vertices' own centre, whichever package version is loaded:
    # about the origin, tetrahedra at mine coordinates cancel down to a
    # small body's volume within rounding
    corners = (points - points.mean(axis=0))[triangles]
    volume = np.sum(np.einsum("ij,ij->i", corners[:, 0],
                              np.cross(corners[:, 1], corners[:, 2]))) / 6
    return type(mesh).__name__, float(volume), len(triangles)


def manifold64(a, b, operation):
    shift = corner(a, b)

    def convert(solid):
        points, triangles = gmt.weld(
            np.asarray(solid.coordinates, dtype=float) - shift,
            solid.triangles)
        return manifold3d.Manifold(manifold3d.Mesh64(
            vert_properties=np.ascontiguousarray(points),
            tri_verts=np.ascontiguousarray(triangles, dtype=np.uint64)))

    t0 = time.perf_counter()
    first, second = convert(a), convert(b)
    converted = time.perf_counter() - t0
    t0 = time.perf_counter()
    answer = {"intersection": lambda: first ^ second,
              "union": lambda: first + second,
              "difference": lambda: first - second}[operation]()
    answer.num_tri()                 # the tree is evaluated on demand
    worked = time.perf_counter() - t0
    mesh = answer.to_mesh64()
    points = np.asarray(mesh.vert_properties)[:, :3] + shift
    triangles = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    kind, volume, count = summary(points, triangles)
    return {"status": "%s/%s" % (first.status().name,
                                 second.status().name),
            "convert": converted, "time": worked, "kind": kind,
            "volume": volume, "tris": count}


def accessor32(a, b, operation, local):
    shift = corner(a, b) if local else np.zeros(3)
    first = a._polydata().translate(-shift)
    second = b._polydata().translate(-shift)
    valid = "%s/%s" % (first.manifold.is_valid, second.manifold.is_valid)
    t0 = time.perf_counter()
    answer = getattr(first.manifold, operation)(second)
    worked = time.perf_counter() - t0
    if answer.n_points == 0:
        return {"valid": valid, "time": worked, "kind": "empty",
                "volume": 0.0, "tris": 0}
    points = np.asarray(answer.points, dtype=float) + shift
    kind, volume, count = summary(points, np.asarray(answer.regular_faces))
    return {"valid": valid, "time": worked, "kind": kind, "volume": volume,
            "tris": count}


def isolated(function, *args):
    """`function(*args)` in a forked child; a crash is reported, not fatal."""
    context = mp.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def run():
        try:
            send.send(("ok", function(*args)))
        except Exception as error:        # a refusal is a result too
            send.send(("error", repr(error)))

    process = context.Process(target=run)
    process.start()
    send.close()
    result = None
    try:
        if receive.poll(TIMEOUT):
            result = receive.recv()
        else:
            process.kill()
            result = ("timeout", None)
    except EOFError:
        pass
    process.join()
    if result is None:
        result = ("crashed, exit code %s" % process.exitcode, None)
    return result


def implicit(a, b, operation):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        answer = getattr(a, operation)(b)
        worked = time.perf_counter() - t0
    step = None
    for warning in caught:
        found = re.search(r"step of ([0-9.e+-]+)", str(warning.message))
        if found:
            step = float(found.group(1))
    if answer.n_data == 0:
        return {"time": worked, "step": step, "kind": "empty",
                "volume": 0.0, "tris": 0}
    kind, volume, count = summary(np.asarray(answer.coordinates),
                                  np.asarray(answer.triangles))
    return {"time": worked, "step": step, "kind": kind, "volume": volume,
            "tris": count}


def row(label, result):
    state, values = result
    if state != "ok":
        return "%-12s %s" % (label, state if values is None
                             else "%s: %s" % (state, values))
    extra = ""
    if "status" in values:
        extra = " status %s, converted in %.1f s" % (values["status"],
                                                     values["convert"])
    if "valid" in values:
        extra = " valid %s" % values["valid"]
    if "step" in values:
        extra = " step %s" % values["step"]
    return "%-12s %7.1f s  %-8s volume %.6g  %8d tris%s" % (
        label, values["time"], values["kind"], values["volume"],
        values["tris"], extra)


totals = {"implicit": 0.0, "manifold64": 0.0}
volumes = []
jobs = [(pair, "intersection") for pair in itertools.combinations(ROCKS, 2)]
jobs += [(("Hematite", "Calcitic Hematite"), "union"),
         (("Hematite", "Calcitic Hematite"), "difference"),
         (("BIF", "Shale"), "difference")]
for (first, second), operation in jobs:
    a, b = solids[first], solids[second]
    say("")
    say("%s %s %s" % (first, operation, second))
    reference = ("ok", implicit(a, b, operation))
    say(row("implicit", reference))
    exact = isolated(manifold64, a, b, operation)
    say(row("manifold64", exact))
    say(row("accessor32", isolated(accessor32, a, b, operation, True)))
    say(row("  at mine", isolated(accessor32, a, b, operation, False)))
    if exact[0] == "ok":
        totals["implicit"] += reference[1]["time"]
        totals["manifold64"] += exact[1]["time"] + exact[1]["convert"]
        volumes.append((first, second, operation, reference[1]["volume"],
                        exact[1]["volume"], reference[1]["step"]))

say("")
say("times, every job both engines finished: implicit %.1f s, manifold64 "
    "%.1f s (conversion included)" % (totals["implicit"],
                                      totals["manifold64"]))
say("%-34s %14s %14s %9s %7s" % ("job", "implicit", "manifold64", "diff",
                                 "step"))
for first, second, operation, v_implicit, v_exact, step in volumes:
    say("%-34s %14.6g %14.6g %8.2f%% %7s" % (
        "%s %s %s" % (first[:8], operation[:5], second[:8]), v_implicit,
        v_exact, 100 * (v_implicit - v_exact) / v_exact if v_exact else 0.0,
        step))
say("done")
