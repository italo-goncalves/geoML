# geoML - machine learning models for geospatial data
# Copyright (C) 2024  Ítalo Gomes Gonçalves
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

import numpy as _np
import pandas as _pd
import pyvista as _pv


def structure_discs(coordinates, dip, azimuth, size=1, **kwargs):
    # conversions
    dip = -dip * _np.pi / 180
    azimuth = (90 - azimuth) * _np.pi / 180
    strike = azimuth - _np.pi / 2

    # dip and strike vectors
    dipvec = _np.concatenate([
        _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
        _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
        _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1)
    strvec = _np.concatenate([
        _np.array(_np.cos(strike), ndmin=2).transpose(),
        _np.array(_np.sin(strike), ndmin=2).transpose(),
        _np.zeros([dipvec.shape[0], 1])], axis=1)

    normals = dipvec[:, [1, 2, 0]] * strvec[:, [2, 0, 1]] \
              - dipvec[:, [2, 0, 1]] * strvec[:, [1, 2, 0]]

    # surfaces
    discs = _pv.MultiBlock()
    strike_tubes = _pv.MultiBlock()
    dip_tubes = _pv.MultiBlock()

    for i, point in enumerate(coordinates):
        discs.append(
            _pv.Cylinder(point, normals[i], radius=size / 2, height=size / 10)
        )
        strike_tubes.append(
            _pv.Cylinder(point, strvec[i], size / 9, size)
        )
        dip_tubes.append(
            _pv.Cylinder(point + dipvec[i] * size / 4,
                         dipvec[i], size / 9, size / 2)
        )

    return discs, strike_tubes, dip_tubes
