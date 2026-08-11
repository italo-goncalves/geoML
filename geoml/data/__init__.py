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
Spatial data, one module per concern: `base` the container tree (errors,
paths, traversal, the attribute), `variables` the variable family,
`containers` the point-based containers, `grids` the regular grids,
`meshes` the triangulated surfaces and solids, `blocks` the block models,
`io` the Zarr persistence, `drillhole` the drilling databases and
`inducing` the inducing-point helpers.

This facade re-exports everything `geoml.data` held when it was one
module, so `geoml.data.PointData` keeps resolving -- in user code and in
every saved store.
"""
from geoml.data.base import *
from geoml.data.variables import *
from geoml.data.containers import *
from geoml.data.grids import *
from geoml.data.meshes import *
from geoml.data.blocks import *
from geoml.data.io import *
# The private bases other modules and user code hold instances of.
from geoml.data.base import _Attribute
from geoml.data.variables import _Variable
from geoml.data.containers import _SpatialData
from geoml.data import (base, variables, containers, grids, meshes, blocks,
                        io, drillhole, inducing)
from geoml.data.drillhole import DrillholeData, IntervalTable
