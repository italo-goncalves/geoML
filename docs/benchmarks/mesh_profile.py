"""Where the time of the mesh operations goes, function by function,
2026-11-11.

Usage: python docs/benchmarks/mesh_profile.py <folder holding
       BM_sub_blocked and Surfaces> <label> [parts, comma-separated]

The profile the roadmap's "Faster mesh operations" starts from, on the
Assen model (908 237 blocks, 25 realizations) and its shells (325k-2.5M
triangles, mine coordinates). Each operation runs once under cProfile, in
this process, and is reported as its wall time, the geoML functions with
the most cumulative time, and the functions of any library with the most
time of their own -- which is where VTK, Manifold and numpy show. The
results go to `docs/benchmarks/figures/mesh_profile_<label>.txt`. Set
`GEOML_ROOT` to import geoML from another checkout.

contour     FeO_total's prediction closed above at 0.7, at supersample 0
            and 1.
set         A mesh set of FeO_total at 0.6, 0.7, 0.8 and 0.85 and of the
            rocks, the prediction and two realizations each, in one
            process, into a store.
booleans    Hematite and Uncertainty (2.5M and 650k triangles):
            intersection, difference and union.
simplify    Uncertainty at 1 m and at 0.25 m.
split       Hematite and Uncertainty separated into their pieces.
"""
import cProfile
import hashlib
import os
import pstats
import sys
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
root = os.environ.get("GEOML_ROOT") or os.path.abspath(
    os.path.join(here, "..", ".."))
sys.path.insert(0, root)
import numpy as np  # noqa: E402

import geoml  # noqa: E402

FOLDER = sys.argv[1]
LABEL = sys.argv[2]
PARTS = (sys.argv[3] if len(sys.argv) > 3
         else "contour,set,booleans,simplify,split").split(",")
OUT = os.path.join(here, "figures", "mesh_profile_%s.txt" % LABEL)
SHELLS = os.path.join(FOLDER, "Surfaces")


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def _name(key):
    path, line, function = key
    if path == "~":
        return function
    short = path.split("site-packages/")[-1]
    if root in path:
        short = os.path.relpath(path, root)
    return "%s:%d %s" % (short, line, function)


def profiled(label, function, rows=14):
    """Runs `function` once under cProfile and reports where it went."""
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    try:
        out = function()
    finally:
        profile.disable()
    wall = time.perf_counter() - start
    stats = pstats.Stats(profile).stats
    say("\n== %s: %.1f s" % (label, wall))
    ours = sorted(((v[3], v[0], k) for k, v in stats.items()
                   if os.path.join(root, "geoml") in k[0]), reverse=True)
    say("   cumulative, geoML's functions")
    for cumulative, calls, key in ours[:rows]:
        say("   %7.2f s %5.1f%% %7d  %s"
            % (cumulative, 100 * cumulative / wall, calls, _name(key)))
    own = sorted(((v[2], v[0], k) for k, v in stats.items()), reverse=True)
    say("   own time, any function")
    for total, calls, key in own[:rows]:
        say("   %7.2f s %5.1f%% %7d  %s"
            % (total, 100 * total / wall, calls, _name(key)))
    return out


def describe(mesh):
    # the digest of the arrays, to compare a mesh between two labels
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(
        np.asarray(mesh.coordinates, dtype=float) + 0.0).tobytes())
    digest.update(np.ascontiguousarray(
        np.asarray(mesh.triangles, dtype=np.int64)).tobytes())
    return "%s, %d triangles, digest %s" % (
        type(mesh).__name__, len(mesh.triangles), digest.hexdigest()[:12])


open(OUT, "w").close()
say("geoML", geoml.__version__, "from", root)

if "contour" in PARTS or "set" in PARTS:
    start = time.perf_counter()
    blocks = geoml.data.BlockSet3D.open(os.path.join(FOLDER, "BM_sub_blocked"))
    say("opened %d blocks in %.1f s" % (blocks.n_data,
                                        time.perf_counter() - start))

if "contour" in PARTS:
    for supersample in (0, 1):
        shell = profiled(
            "contour FeO_total at 0.7, supersample %d" % supersample,
            lambda: blocks.get_contour("Comp/FeO_total", 0.7, close="above",
                                       supersample=supersample))
        say("  ", describe(shell))

if "set" in PARTS:
    with tempfile.TemporaryDirectory() as scratch:
        grade = profiled(
            "mesh set of FeO_total, 4 cut-offs, prediction and 2 "
            "realizations", lambda: geoml.data.MeshSet(
                blocks, "Comp/FeO_total", cutoffs=[0.6, 0.7, 0.8, 0.85],
                simulations=[0, 1], workers=1,
                store=os.path.join(scratch, "grade.zarr")))
        rock = profiled(
            "mesh set of the rocks, prediction and 2 realizations",
            lambda: geoml.data.MeshSet(
                blocks, "Simple Rock", simulations=[0, 1], workers=1,
                store=os.path.join(scratch, "rock.zarr")))
        for name, meshes in (("FeO_total", grade), ("the rocks", rock)):
            for world, shells in [("prediction", meshes)] + [
                    ("realization %d" % i, meshes.simulations[i])
                    for i in range(len(meshes.simulations))]:
                for key in shells:
                    say("   %s, %s, %s: %s" % (name, world, key,
                                               describe(shells[key])))

if "booleans" in PARTS or "simplify" in PARTS or "split" in PARTS:
    start = time.perf_counter()
    hematite = geoml.data.Solid3D.open(os.path.join(SHELLS, "Hematite.zarr"))
    uncertainty = geoml.data.Solid3D.open(
        os.path.join(SHELLS, "Uncertainty.zarr"))
    say("opened Hematite (%d triangles) and Uncertainty (%d) in %.1f s"
        % (len(hematite.triangles), len(uncertainty.triangles),
           time.perf_counter() - start))

if "booleans" in PARTS:
    for operation in ("intersection", "difference", "union"):
        answer = profiled(
            "Uncertainty %s Hematite" % operation,
            lambda: getattr(uncertainty, operation)(hematite))
        say("  ", describe(answer), "volume %.1f" % answer.volume)

if "simplify" in PARTS:
    for budget in (1.0, 0.25):
        answer = profiled("simplify Uncertainty to %.2f m" % budget,
                          lambda: uncertainty.simplify(budget))
        say("  ", describe(answer))

if "split" in PARTS:
    for name, mesh in (("Uncertainty", uncertainty), ("Hematite", hematite)):
        pieces = profiled("split %s" % name, mesh.split)
        say("   %d pieces" % len(pieces))
        for piece in pieces:
            say("  ", describe(piece))
