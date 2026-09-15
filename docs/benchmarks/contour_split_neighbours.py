"""The shells the edge split closed on Assen, 2026-09-13: each against the
bodies a ten-thousandth of the span either side (this checkout), or against
what the old retry returned (`GEOML_ROOT` pointing at the code as it stood
before the split).

Usage: python docs/benchmarks/contour_split_neighbours.py > out.txt

The eight shells are the ones `contour_stages.py`'s census found closing at
their own level only once edges four triangles share were split: the
FeO_total prediction at 0.7, five of its realizations, and two rock bodies.
Results: `figures/contour_split_neighbours_{new,old}.txt`.
"""
import glob, os, sys
root = os.environ.get("GEOML_ROOT") or glob.glob("/mnt/c/Users/*/OneDrive/Python/Pacotes/geoML-claude")[0]
sys.path.insert(0, root)
import numpy as np
import geoml, geoml.data.blocks as blk, geoml.data.meshsets as msm
assen = glob.glob("/mnt/c/Users/*/OneDrive/Python/Projetos/Modelagem geol*gica/2025 Assen Fe deposit/Assen geoml v6")[0]
blocks = geoml.data.BlockSet3D.open(os.path.join(assen, "BM_sub_blocked"))
labels = list(blocks.variables["Simple Rock"].labels)

def field(case):
    kind, number, key = case
    if kind == "FeO":
        if number is None:
            return np.asarray(blk._contour_column(blocks, "Comp/FeO_total")[1].values, dtype=float).ravel()
        return np.asarray(blocks.values("Comp/FeO_total/simulations/%d" % number), dtype=float).ravel()
    draws = np.stack([np.asarray(blocks.values("Simple Rock/%s/simulations/%d" % (l, number)), dtype=float).ravel()
                      for l in labels], axis=1)
    return msm._category_fields(draws, "largest")[:, labels.index(key)]

cases = [("FeO", None, 0.7), ("FeO", 2, 0.8), ("FeO", 8, 0.7), ("FeO", 14, 0.6), ("FeO", 19, 0.8),
         ("FeO", 19, 0.85), ("rock", 0, "Calcitic Hematite"), ("rock", 3, "Limestone")]
old = "geoml_baseline" in root
for case in cases:
    values = field(case)
    level = case[2] if case[0] == "FeO" else 0.0
    finite = values[np.isfinite(values)]
    span = float(finite.max() - finite.min())
    if old:
        mesh, nudge = msm._shell(blocks, values, level, "above", 0, "g")
        print("%-28s old retry: volume %.1f at %+.0e of the span" % (case, mesh.volume, nudge / span), flush=True)
    else:
        row = []
        for shift in (-1e-4, 0.0, 1e-4):
            mesh = blocks._contour_values(values, level + shift * span, "g", close="above", fallback=False)
            row.append("%s %.1f" % (type(mesh).__name__, getattr(mesh, "volume", float("nan"))))
        print("%-28s -1e-4: %s | level: %s | +1e-4: %s" % ((case,) + tuple(row)), flush=True)
