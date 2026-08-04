# geoML - machine learning models for geospatial data
# Copyright (C) 2021  Ítalo Gomes Gonçalves
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
"""
One place defining what a geoML figure looks like.

Every function in this sub-package draws inside `context()`, a
`matplotlib.rc_context`, so the settings below apply to the figure being drawn
and to nothing else: importing geoML does not change how anyone else's plots
look. Call `use()` to take them globally, if that is what you want.

`PALETTE` is what tells one category from another, and it is the same list
wherever a category is drawn, so a rock type keeps its colour from a histogram
to a map to the dashboard. Change it here and every figure follows.
"""
import matplotlib as _mpl


PALETTE = [
    "#1f6f8b",   # teal blue
    "#e0761f",   # orange
    "#1e7b45",   # green
    "#9b2226",   # dark red
    "#5f5aa2",   # violet
    "#b58b00",   # ochre
    "#00a5a8",   # cyan
    "#6c757d",   # slate
]

# For a continuous value, where the order of the colours has to mean something.
# `cividis` reads the same to colour-vision-deficient eyes, and both its ends
# stay visible against a white background.
SEQUENTIAL = "cividis"

# what a measured but unassigned location looks like
MISSING_COLOR = "#c9c9c9"

# behind a label that has to be read over the data it sits on, such as the
# loading arrows in a biplot: enough to lift the text off the points without
# hiding them
LABEL_BOX = {
    "boxstyle": "round,pad=0.25",
    "facecolor": "white",
    "edgecolor": "black",
    "linewidth": 0.6,
    "alpha": 0.5,
}

RC = {
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",

    "font.family": "sans-serif",
    "font.size": 9,

    "axes.prop_cycle": _mpl.cycler(color=PALETTE),
    "axes.facecolor": "white",
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.labelcolor": "#2b2b2b",
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.grid": True,
    "axes.axisbelow": True,          # data over the grid, never under it

    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,

    "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",

    "legend.frameon": False,
    "legend.fontsize": 8,

    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",

    "image.cmap": SEQUENTIAL,
}


def context():
    """The geoML settings, for the figure being drawn and nothing else."""
    return _mpl.rc_context(RC)


def use():
    """Takes the geoML settings globally, for every figure from here on."""
    _mpl.rcParams.update(RC)


def color(index):
    """The colour of the `index`-th category, cycling when there are many."""
    return PALETTE[index % len(PALETTE)]
