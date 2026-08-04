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

`Interactive` is the same set of figures again, under the same names and the
same arguments, in plotly rather than matplotlib. `Explorer` is the one to save
and print; `Interactive` is the one to look at, and it does what a printed
figure cannot: zoom one panel of a scatter matrix and its whole row and column
follow, so a cluster can be traced across every pair at once; click a category
in the legend and it leaves every panel; turn a three-dimensional scene around;
and hand `scene()` a list of variables rather than one, for a menu to switch
between them over the same ground.

`Dashboard` puts several of them on one page and links their selections. A
figure answers one question, and most questions worth asking of spatial data
are about two things at once -- those samples out on the tail of the histogram,
where are they on the map? Drag a box over any panel and the same locations
light up in all the others:

    eda = geoml.plots.Interactive(point, continuous="Elements",
                                  categorical="Rock", model=model)
    board = eda.dashboard()
    board.write_html("jura.html")     # opens anywhere, with no network

`board` renders in a notebook as well, and `Dashboard([...])` takes figures
rather than names when the page is to be composed by hand.

The colours belong to the object, so a category looks the same in every figure
it draws. Pass your own when the defaults are not what the map calls for -- a
list, or a dict naming the categories:

    geoml.plots.Explorer(point, continuous="Zn", categorical="rock",
                         palette={"granite": "#d99", "basalt": "#48a"},
                         cmap="viridis")

Every matplotlib figure is drawn under the settings in `style`, applied to that
figure alone, so importing geoML never changes how anyone else's plots look;
`style.TEMPLATE` is the same look for plotly. The arithmetic behind the figures
lives in `prepare` and draws nothing, so the numbers can be had on their own --
and so that both ways of showing them read the same functions rather than
working them out twice.
"""
from geoml.plots.dashboard import Dashboard
from geoml.plots.explorer import Explorer
from geoml.plots.interactive import Interactive
from geoml.plots.style import PALETTE, SEQUENTIAL, TEMPLATE, context, use
from geoml.plots import prepare, style

__all__ = ["Explorer", "Interactive", "Dashboard", "PALETTE", "SEQUENTIAL",
           "TEMPLATE", "context", "use", "prepare", "style"]
