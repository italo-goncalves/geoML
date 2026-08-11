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
Spatial data: the containers and their variables (`core`), drillholes and
their conversion to point data (`drillhole`), and the helpers that build
inducing-point sets (`inducing`).

This facade re-exports everything `geoml.data` held when it was one module,
so `geoml.data.PointData` keeps resolving -- in user code and in every
saved store.
"""
from geoml.data.core import *
# The private bases other modules and user code hold instances of.
from geoml.data.core import _Attribute, _SpatialData, _Variable
from geoml.data import core, drillhole, inducing
