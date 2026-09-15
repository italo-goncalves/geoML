"""How close a block model's closed contours come to the truth, and how often
they close, 2026-09-14.

Usage: python docs/benchmarks/closed_contours.py <label>
       python docs/benchmarks/closed_contours.py compare <label> <label>

The first form measures the checkout it imports -- `GEOML_ROOT` picks
another, the code as it stood before a change, say -- and writes every case
to `docs/benchmarks/figures/closed_contours_<label>.json`; the second puts
two such runs side by side in `closed_contours_<label>_<label>.txt`.

balls
    A ball of radius 45 m in a 160 m box of 10 m blocks, meeting the box at
    a face, an edge, a corner, and centred on a face, closed above and below
    at supersample 0 and 1: the volume against the ball cut to the box,
    by Monte Carlo on two million points.
layers
    1 000 small random models, a field rising to the top face with noise
    on it, contoured near the top and closed on the side a random draw
    picks: what each came back as, and its volume against the volume of
    the blocks on the kept side.
"""
import json
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(here, "figures")


def measure(label):
    root = os.environ.get("GEOML_ROOT") or os.path.abspath(
        os.path.join(here, "..", ".."))
    sys.path.insert(0, root)
    import numpy as np
    import geoml

    record = {"root": root, "balls": {}, "layers": {}}
    blocks = geoml.data.BlockSet3D([0, 0, 0], [16, 16, 16], [10.0] * 3,
                                   discretization=(2, 2, 2), max_levels=1)
    low = np.ravel(blocks.bounding_box.min)
    high = np.ravel(blocks.bounding_box.max)
    box = float(np.prod(high - low))
    sample = np.random.default_rng(0).uniform(low, high, size=(2000000, 3))
    for name, centre in (("face", [30.0, 75.0, 75.0]),
                         ("edge", [30.0, 30.0, 75.0]),
                         ("corner", [30.0, 30.0, 30.0]),
                         ("half outside", [-5.0, 75.0, 75.0])):
        values = 45.0 - np.linalg.norm(np.asarray(blocks.coordinates)
                                       - np.asarray([centre]), axis=1)
        exact = (np.linalg.norm(sample - np.asarray(centre), axis=1)
                 < 45.0).mean() * box
        for supersample in (0, 1):
            for side, truth in (("above", exact), ("below", box - exact)):
                mesh = blocks._contour_values(
                    values, 0.0, "g", supersample=supersample, close=side,
                    fallback=False)
                kind = type(mesh).__name__
                error = 100 * (mesh.volume / truth - 1) \
                    if kind == "Solid3D" else None
                record["balls"]["%s %s %d" % (name, side, supersample)] = \
                    [kind, error]

    start = time.perf_counter()
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        n = [int(rng.integers(3, 6)), int(rng.integers(3, 6)),
             int(rng.integers(2, 4))]
        model = geoml.data.BlockSet3D([0, 0, 0], n, [10.0, 10.0, 10.0],
                                      discretization=(2, 2, 2), max_levels=1)
        if rng.random() < 0.7:
            model = model.split(
                np.flatnonzero(rng.random(model.n_data) < 0.5))
        z = np.asarray(model.coordinates)[:, 2]
        values = z / (10.0 * n[2]) + 0.15 * rng.standard_normal(model.n_data)
        side = "above" if rng.random() < 0.5 else "below"
        if side == "below":
            values = -values
        level = float(np.quantile(values, rng.uniform(0.7, 0.97)))
        mesh = model._contour_values(values, level, "g", close=side,
                                     fallback=False)
        kind = type(mesh).__name__
        kept = values > level if side == "above" else values < level
        volume = np.prod(np.asarray(model.block_size, dtype=float), axis=1)
        record["layers"][str(seed)] = [
            kind, float(mesh.volume) if kind == "Solid3D" else None,
            float(volume[kept].sum())]
    record["layers_s"] = time.perf_counter() - start
    with open(os.path.join(FIGURES, "closed_contours_%s.json" % label),
              "w") as f:
        json.dump(record, f)
    print("%s: geoML from %s, layers in %.0f s" % (label, root,
                                                   record["layers_s"]))


def compare(first, second):
    import numpy as np

    runs = []
    for label in (first, second):
        with open(os.path.join(FIGURES,
                               "closed_contours_%s.json" % label)) as f:
            runs.append(json.load(f))
    out = os.path.join(FIGURES, "closed_contours_%s_%s.txt" % (first, second))
    lines = ["%s: geoML from %s" % (label, run["root"])
             for label, run in zip((first, second), runs)]
    lines += ["", "== balls, the volume against Monte Carlo (%)",
              "%-22s %12s %12s" % ("case", first, second)]
    for case in runs[0]["balls"]:
        cells = [("%+.2f" % run["balls"][case][1])
                 if run["balls"][case][0] == "Solid3D"
                 else run["balls"][case][0] for run in runs]
        lines.append("%-22s %12s %12s" % (case, *cells))
    for label, run in zip((first, second), runs):
        errors = [abs(e) for kind, e in run["balls"].values()
                  if kind == "Solid3D"]
        lines.append("%s: worst %.2f%%, mean %.2f%%"
                     % (label, max(errors), np.mean(errors)))
    lines += ["", "== 1 000 random thin layers"]
    for label, run in zip((first, second), runs):
        kinds = {}
        for kind, _, _ in run["layers"].values():
            kinds[kind] = kinds.get(kind, 0) + 1
        ratio = np.array([v / k for kind, v, k in run["layers"].values()
                          if kind == "Solid3D" and k > 0])
        lines.append("%s: %s; volume against the kept blocks' quartiles "
                     "%.3f %.3f %.3f" % (label, kinds,
                                         *np.quantile(ratio, [0.25, 0.5,
                                                              0.75])))
    lost = [seed for seed in runs[0]["layers"]
            if runs[0]["layers"][seed][0] == "Solid3D"
            and runs[1]["layers"][seed][0] == "NoneType"]
    gained = [seed for seed in runs[0]["layers"]
              if runs[0]["layers"][seed][0] == "NoneType"
              and runs[1]["layers"][seed][0] == "Solid3D"]
    lines.append("a body in %s and none in %s: %d; the reverse: %d"
                 % (first, second, len(lost), len(gained)))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        measure(sys.argv[1])
