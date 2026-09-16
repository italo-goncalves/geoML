"""What splitting the realization axis costs and buys, 2026-09-15.

Usage: python docs/benchmarks/realization_chunks.py [rows] [n_sim]

`ArrayStore` chunked the location axis alone, so a chunk held every
realization of a band of rows. That suits the row-wise reductions -- a band
is complete, and a quantile over a row needs all of it -- and it makes
reading ONE realization read them all: `simulation(i)` on a (5 000 000, 100)
store walks 4 GB to hand back 40 MB, which is GeoScape's item 3.

The question is how many realizations a chunk should hold. Narrower chunks
make a realization's read visit a smaller share of the store; they also make
the reductions rechunk, since a block of a wide-chunked dask array no longer
holds whole rows. This measures both directions at three widths -- every
realization in one chunk (the old policy), a quarter of them, and ten --
against the three things that read a simulations store:

one realization
    `variable.simulation(i)`, the primitive behind every per-realization
    pass. Averaged over five, spread across the column groups.
quantiles
    `row_quantiles([0.1, 0.5, 0.9])` computed whole: one pass over the store
    that must see each row complete.
bands
    `row_bands()` with a mean per band, what a reduction written by hand
    does. Reads the store once whatever the chunking.

Run it as root to have the page cache dropped before each measurement,
without which a store that fits in RAM is read from memory and every width
looks alike; the report says which way it ran.

Results go to `docs/benchmarks/figures/realization_chunks.txt`.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.environ.get("GEOML_ROOT", os.getcwd()))

import geoml.storage as storage  # noqa: E402

WIDTHS = ("all", "quarter", "ten")


def _chunks(rows, n_sim, width, itemsize=8):
    cols = {"all": n_sim, "quarter": max(1, n_sim // 4), "ten": 10}[width]
    cols = min(cols, n_sim)
    per_row = max(cols * itemsize, 1)
    return (max(1, min(rows, storage._TARGET_CHUNK_BYTES // per_row)), cols)


def _drop_caches():
    """Empty the page cache, so a read is measured against the disk.

    Root only; returns whether it worked, which the report records -- a
    measurement taken out of RAM is a different measurement and must not be
    read as this one.
    """
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as handle:
            handle.write("3")
        return True
    except (OSError, PermissionError):
        return False


def _timed(call, cold=False):
    if cold:
        _drop_caches()
    start = time.perf_counter()
    result = call()
    return time.perf_counter() - start, result


def run(rows, n_sim):
    values = np.random.default_rng(0).normal(size=(rows, n_sim))
    cold = _drop_caches()
    lines = ["(%d, %d) float64, %.2f GB, %s"
             % (rows, n_sim, values.nbytes / 1024 ** 3,
                "page cache dropped before each read" if cold
                else "WARM: run as root to measure against the disk"), ""]
    lines.append("%-9s %-16s %8s %8s %8s %7s"
                 % ("width", "chunks", "one", "quant", "bands", "files"))

    for width in WIDTHS:
        chunks = _chunks(rows, n_sim, width)
        directory = tempfile.mkdtemp(prefix="geoml_chunks_")
        path = os.path.join(directory, "s.zarr")
        try:
            store = storage.ArrayStore.allocate(
                values.shape, dtype=values.dtype, fill_value=0.0,
                chunks=chunks, backend="zarr", store=path)
            store[:] = values

            # one realization, five of them across the column groups
            picks = np.linspace(0, n_sim - 1, 5).astype(int)
            one = np.mean([_timed(lambda i=i: np.asarray(store[:, i]),
                                  cold)[0] for i in picks])

            quant = _timed(
                lambda: store.row_quantiles([0.1, 0.5, 0.9]).compute(),
                cold)[0]

            def bands():
                return [np.asarray(store[band]).mean()
                        for band in store.row_bands()]
            band = _timed(bands, cold)[0]

            files = sum(len(names) for _, _, names in os.walk(path))
            lines.append("%-9s %-16s %8.2f %8.2f %8.2f %7d"
                         % (width, "%d x %d" % chunks, one, quant, band,
                            files))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    report = run(n_rows, sims)
    print(report)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures", "realization_chunks.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(report)
