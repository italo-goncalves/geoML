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

"""Chapter 17's surface block on its walked model, with the traceback.

The chapter's runner reports an error by its type alone; this runs block 5
of the chapter -- refine, contour, predict onto the surface -- on the
walked model (`ch17_memory.build`, trained EPOCHS epochs) and prints the
whole error, timing each stage.

Usage: python ch17_surface.py [EPOCHS]
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch17_memory as cm  # noqa: E402
import geoml  # noqa: E402


def main(epochs=2):
    model = cm.build("on")
    model.train_svi(epochs)
    blocks = geoml.data.BlockSet3D(
        start=[24860, 15710, 1310], n=[15, 18, 15], step=[20.0, 20.0, 20.0],
        discretization=(2, 2, 2), max_levels=3)
    try:
        start = time.perf_counter()
        blocks = geoml.models.refine(model, blocks)
        print("refined to %d blocks in %.0f s, peak %.2f GB"
              % (blocks.n_data, time.perf_counter() - start, cm.peak_gb()),
              flush=True)
        surface = blocks.get_contour("SIMPLE LITO/Vein/indicator_predicted",
                                     0.0)
        start = time.perf_counter()
        model.predict(surface)
        print("surface of %d triangles predicted in %.0f s, peak %.2f GB"
              % (len(surface.triangles), time.perf_counter() - start,
                 cm.peak_gb()), flush=True)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
