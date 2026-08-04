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
Figures: exploring a data set, and reporting on a model.

`Explorer` is the way in. It holds a container and a choice of variables, and
answers with figures:

    eda = geoml.plots.Explorer(point, continuous="Elements",
                               categorical="Rock")
    eda.histogram()
    eda.pairs()
    eda.pca(explained=0.9)
    eda.scene()

Give it a model and it will report on one as well -- how training went, what
the data looks like after the warping the model actually trained with, and
whether the predictions and their spread hold up:

    eda = geoml.plots.Explorer(point, continuous="Elements", model=model)
    eda.training_curve()
    eda.transformed_pairs()
    eda.prediction_scatter(component="Zn")
    eda.simulation_pairs()
    eda.accuracy()

On a block model carrying simulations, `grade_tonnage()` draws how much
material clears each cut-off and how good it is, in volume or -- given a
density, which may itself be simulated -- in mass.

The colours are the Explorer's to keep, so a category looks the same in every
figure it draws. Pass your own when the defaults are not what the map calls
for -- a list, or a dict naming the categories:

    geoml.plots.Explorer(point, continuous="Zn", categorical="rock",
                         palette={"granite": "#d99", "basalt": "#48a"},
                         cmap="viridis")

Every figure is drawn under the settings in `style`, applied to that figure
alone, so importing geoML never changes how anyone else's plots look. The
arithmetic behind the figures lives in `prepare` and draws nothing, so the
numbers can be had on their own -- and so a second way of showing them reads
the same functions rather than working them out again.
"""
from geoml.plots.explorer import Explorer
from geoml.plots.style import PALETTE, SEQUENTIAL, context, use
from geoml.plots import prepare, style

__all__ = ["Explorer", "PALETTE", "SEQUENTIAL", "context", "use",
           "prepare", "style"]
