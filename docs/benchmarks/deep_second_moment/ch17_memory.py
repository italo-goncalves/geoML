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

"""What chapter 17's walked model costs in memory to predict a block set.

The chapter's deep model (`GPWalk` under a one-field implicit model, 5606
training points, 617 inducing points), trained two epochs -- memory does
not depend on the trained values -- then the first pass of the chapter's
`refine`: the 4050 blocks of its 20 m set, eight sub-blocks each, at the
default prediction batch. `full_variance` on or off. Prints the process's
peak resident memory after training and after the prediction.

Usage: python ch17_memory.py on|off [BATCH]
"""
import os
import resource
import sys
import time

import pandas as pd

import geoml

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "manual", "data")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def build(arm):
    """The chapter's walked model, `full_variance` on or off."""
    geoml.set_seed(1234)
    collar = pd.read_csv(os.path.join(DATA, "vein-collar.csv"))
    lito = pd.read_csv(os.path.join(DATA, "vein-lito.csv"),
                       usecols=["HOLE", "FROM", "TO", "SIMPLE LITO"])
    holes = geoml.data.DrillholeData(
        collar, hole="HOLE", x="EAST", y="NORTH", z="RL", length="TDEPTH")
    holes.add_intervals("lito", lito, hole="HOLE", fr="FROM", to="TO",
                        categorical=["SIMPLE LITO"])
    holes.add_intervals("lito", holes.merge_domains(("lito", "SIMPLE LITO")))
    dense = holes.as_classification_input(("lito", "SIMPLE LITO"), length=1.0)
    sparse = holes.as_classification_input(("lito", "SIMPLE LITO"),
                                           length=10.0)
    deep_input = geoml.latent.BasicInput(
        inducing_points=sparse,
        transform=geoml.transform.Anisotropy3D(100, 1.0, 0.5, azimuth=270,
                                               dip=45),
        center=True)
    walked = geoml.latent.GPWalk(
        geoml.latent.BasicGP(deep_input, size=3), n_steps=5)
    field = geoml.latent.BasicGP(walked, size=1, isotropic=True)
    return geoml.models.VGPNetwork(
        dense, "SIMPLE LITO", geoml.likelihood.CategoricalGaussianIndicator(2),
        geoml.latent.Linear(field, size=2),
        options=geoml.models.GPOptions(
            jitter=1e-6, training_batch_size=500, verbose=False,
            full_variance=arm == "on"))


def main(arm, batch=None):
    model = build(arm)
    if batch is not None:
        model.options.prediction_batch_size = int(batch)
    start = time.perf_counter()
    model.train_svi(2)
    print(arm, "trained %.0f s, peak %.2f GB"
          % (time.perf_counter() - start, peak_gb()), flush=True)
    blocks = geoml.data.BlockSet3D(
        start=[24860, 15710, 1310], n=[15, 18, 15], step=[20.0, 20.0, 20.0],
        discretization=(2, 2, 2), max_levels=3)
    start = time.perf_counter()
    model.predict(blocks)
    print(arm, "predicted %d blocks %.0f s, peak %.2f GB"
          % (blocks.n_data, time.perf_counter() - start, peak_gb()),
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
