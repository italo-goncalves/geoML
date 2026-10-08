# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""What a byte shuffle and float32 realizations save on disk.

Usage: python realization_codecs.py STORE [PATH ...]

Reads each `simulations` array of a container store (every one under STORE
by default, or the variable paths given, `Comp/FeO_total` say) and writes it
again three ways, in the store's own chunks: as 0.8.6 wrote it (float64,
`bytes` then `zstd`), float64 with a byte shuffle before the `zstd`, and
float32 with the shuffle -- what 0.8.7 writes. Prints the bytes on disk,
the write and read seconds and the largest difference float32 makes, in
units of each array's own spread between realizations.
"""

import os
import sys
import tempfile
import time

import numpy as np
import zarr
from zarr.codecs import Shuffle, ZstdCodec


def size(path):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, files in os.walk(path) for f in files)


def arrays_in(group):
    for name, array in group.arrays():
        if name == "simulations":
            yield "", array
    for name, sub in group.groups():
        for n, a in arrays_in(sub):
            yield (name + "/" + n).rstrip("/"), a


def main():
    store, paths = sys.argv[1], sys.argv[2:]
    group = zarr.open_group(store, mode="r")
    found = ([(p, group[p + "/simulations"]) for p in paths] if paths
             else list(arrays_in(group)))
    ways = (("0.8.6, float64", "f8", "auto"),
            ("float64, shuffled", "f8", [Shuffle(elementsize=8), ZstdCodec()]),
            ("float32, shuffled", "f4", [Shuffle(elementsize=4), ZstdCodec()]))
    total = {w: 0 for w, _, _ in ways}
    with tempfile.TemporaryDirectory() as work:
        for name, source in found:
            values = np.asarray(source[...], dtype=np.float64)
            spread = np.nanmedian(np.nanstd(values, axis=1))
            print("%s: %s, %.0f MB raw" % (name or "/", values.shape,
                                           values.nbytes / 1024 ** 2))
            for k, (way, dtype, codecs) in enumerate(ways):
                path = os.path.join(work, "%s %d" % (name.replace("/", "_"),
                                                      k))
                start = time.perf_counter()
                target = zarr.create_array(
                    store=path, shape=values.shape, chunks=source.chunks,
                    dtype=dtype, compressors=codecs, fill_value=np.nan)
                target[...] = values
                written = time.perf_counter() - start
                start = time.perf_counter()
                back = target[...]
                read = time.perf_counter() - start
                on_disk = size(path)
                total[way] += on_disk
                line = "  %-18s %7.1f MB  write %5.2f s  read %5.2f s" % (
                    way, on_disk / 1024 ** 2, written, read)
                if dtype == "f4":
                    error = np.nanmax(np.abs(back.astype(np.float64) - values))
                    line += "  largest change %.1e of the spread" % (
                        error / spread)
                print(line)
    base = total[ways[0][0]]
    for way, _, _ in ways:
        print("%-18s %8.1f MB, %.2f of 0.8.6's" % (
            way, total[way] / 1024 ** 2, total[way] / base))


if __name__ == "__main__":
    main()
