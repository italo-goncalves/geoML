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

"""The training curves `fold_ablation.py` saved beside its maps.

Two panels on one step axis: the whole curve, and the bound from -150 up,
where the runs part; a dotted line where the learning rate drops from 5e-2
to 1e-2 (step 100). Each curve's label gives where it stopped.

Usage: python fold_curves.py MAPS_DIR OUT.png NAME...
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "long_default_r1": "default delta, field range 1",
    "long_default_r3": "default delta, field range 3",
    "long_conf_r1": "delta 0.01 on every GP, field range 1",
    "long_conf_r3": "delta 0.01 on every GP, field range 3",
    "conf_r1_3000": "delta 0.01 on every GP, 3000 steps, no tolerance",
}
STYLES = {"long_default_r1": ("#2a6fdb", "-"), "long_default_r3": ("#2a6fdb", "--"),
          "conf_r1_3000": ("#8b1a1a", "-"),
          "long_conf_r1": ("#d2691e", "-"), "long_conf_r3": ("#d2691e", "--")}


def main(maps, out, names):
    figure, (whole, near) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name in names:
        log = np.load(os.path.join(maps, "%s_s1.npz" % name))["log"]
        colour, style = STYLES.get(name, (None, "-"))
        label = "%s: step %d, %.1f" % (LABELS.get(name, name), len(log),
                                        log[-1])
        for ax in (whole, near):
            ax.plot(np.arange(1, len(log) + 1), log, color=colour,
                    linestyle=style, lw=1.4, label=label)
    for ax in (whole, near):
        ax.axvline(100, color="0.5", lw=0.8, linestyle=":")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    whole.set_ylabel("ELBO")
    whole.set_title("whole run")
    near.set_ylim(-150, -40)
    near.set_title("from -150 up")
    near.legend(fontsize=8.5, loc="lower right")
    figure.suptitle(os.environ.get(
        "FOLD_TITLE", "Folded ore, joint rule: training stopped by "
        "training_tolerance=0.01 (the rate drops at step 100)"))
    figure.tight_layout()
    figure.savefig(out, dpi=120)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
