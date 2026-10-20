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

"""Gate 1a's residual on the GP on a GP: is it the quadrature?

The conditioned second moment at 64, 256 and 1024 symmetric Sobol nodes,
and on a Gauss-Hermite grid of 3, 4, 5 and 7 points a dimension (the
hidden layer has two outputs, so 9 to 49 nodes), against 16000 coupled
draws, on the GP on a GP with a Gaussian and a Matern 5/2 leaf. Writes
`gate_nodes.json`.

Usage: python gate_nodes.py [ITERATIONS]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_moments as gm  # noqa: E402
import geoml.kernels as kr  # noqa: E402
import geoml.latent.network as net  # noqa: E402


def main(iterations):
    data, _ = gm.ek.drillholes(gm.ek.N_HOLES, gm.ek.SEED)
    new, _ = gm.ek.drillholes(gm.ek.N_NEW, gm.ek.NEW_SEED)
    pick = np.random.default_rng(3).choice(new.n_data, gm.N_POINTS,
                                           replace=False)
    x = np.asarray(new.coordinates, dtype=float)[pick]
    rules = [("sobol", n) for n in (64, 256, 1024)]         + [("hermite", p) for p in (3, 4, 5, 7)]
    rows = []
    for name, kernel in (("Gaussian", kr.Gaussian),
                         ("Matern52", kr.Matern52)):
        model = gm.build(data, "dgp", kernel)
        net._UNCERTAIN_POINTS_SECOND_MOMENT = False
        model.train_full(iterations)
        net._UNCERTAIN_POINTS_SECOND_MOMENT = True
        ex_mean, ex_evar, ex_vmu = gm.rf.exact_moments(model, x, 16000)
        ex_var = ex_evar + ex_vmu
        for rule, count in rules:
            net._QUADRATURE_RULE = rule
            net._HERMITE_POINTS = count if rule == "hermite" else 5
            net._QUADRATURE_NODES = count if rule == "sobol" else 64
            net._QUADRATURE.clear()
            mean, var = gm.analytic(model, x, True)
            ratio = var / ex_var
            row = dict(kernel=name, rule=rule,
                       nodes=count if rule == "sobol" else count ** 2,
                       var_ratio_median=float(np.median(ratio)),
                       var_ratio_mean=float(np.mean(ratio)),
                       var_error_median=float(np.median(np.abs(ratio - 1))),
                       var_ratio_5=float(np.percentile(ratio, 5)),
                       var_ratio_95=float(np.percentile(ratio, 95)),
                       mean_error_sd=float(np.max(np.abs(mean - ex_mean)
                                                  / np.sqrt(ex_var))))
            print(json.dumps(row), flush=True)
            rows.append(row)
    net._QUADRATURE_RULE = "sobol"
    net._QUADRATURE_NODES = 64
    net._QUADRATURE.clear()
    with open(os.path.join(gm.FIGURES, "gate_nodes.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
