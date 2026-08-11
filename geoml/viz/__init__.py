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
Visualization backends the containers and models export through: `plotly`
(figure dicts), `pyvista` (VTK objects) and `graphviz` (DOT text -- written,
never rendered, so Graphviz itself is not a dependency). The figures
themselves live in `geoml.plots`.
"""
from geoml.viz import graphviz, plotly, pyvista
