# geoML - machine learning models for geospatial data
# Copyright (C) 2025  Ítalo Gomes Gonçalves
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
from sklearn.decomposition import PCA as _PCA


def rotation_matrix(azimuth=0.0, dip=0.0, rake=0.0):
    # conversion to radians
    azimuth = azimuth * (_np.pi / 180)
    dip = dip * (_np.pi / 180)
    rake = rake * (_np.pi / 180)

    # conversion to mathematical coordinates
    dip = - dip

    # rotation matrix
    # x and y axes are switched
    # rotation over z is with sign reversed
    rx = _np.stack([_np.cos(rake), 0, _np.sin(rake),
                    0, 1, 0,
                    -_np.sin(rake), 0, _np.cos(rake)], -1)
    rx = _np.reshape(rx, [3, 3])
    ry = _np.stack([1, 0, 0,
                    0, _np.cos(dip), -_np.sin(dip),
                    0, _np.sin(dip), _np.cos(dip)], -1)
    ry = _np.reshape(ry, [3, 3])
    rz = _np.stack([_np.cos(azimuth), _np.sin(azimuth), 0,
                    -_np.sin(azimuth), _np.cos(azimuth), 0,
                    0, 0, 1], -1)
    rz = _np.reshape(rz, [3, 3])

    rot = _np.matmul(_np.matmul(rz, ry), rx)
    return rot.T


def rotation_matrix_from_points(points):
    pca = _PCA()
    pca.fit(points)
    rotmat = pca.components_[[1, 0, 2]]
    normal = vector_product(rotmat[0], rotmat[1])
    prod = _np.sum(normal * rotmat[2])
    if prod < 0:
        rotmat[1] *= -1
    # elif rotmat[0, 0] > 0:
    #     rotmat[0] *= -1
    #     rotmat[1] *= -1
    return rotmat


def azimuth_from_xy(x, y):
    ang = _np.degrees(_np.atan2(y, x))
    ang = 90 - ang
    if ang < 0:
        ang += 360
    return ang


def dip_from_vec(vec):
    if vec.shape != (3,):
        raise ValueError('Vector must be 3D')
    x, y, z = vec
    proj = _np.sqrt(x**2 + y**2)
    dip = - _np.degrees(_np.atan2(z, proj))
    return dip


def angles_from_rotation_matrix(rotmat):
    rotmat = rotmat.T

    if rotmat.shape == (2, 2):
        return azimuth_from_xy(rotmat[0, 1], rotmat[1, 1])
    elif rotmat.shape != (3, 3):
        raise ValueError('Rotation matrix must be 2D or 3D')

    az = azimuth_from_xy(rotmat[0, 1], rotmat[1, 1])
    dip = dip_from_vec(rotmat[:, 1])

    rotmat_2 = rotation_matrix(az, dip, 0)
    rotmat_3 = _np.matmul(rotmat_2, rotmat).T
    rake = - dip_from_vec(rotmat_3[:, 0])

    if dip < 0:
        dip = - dip
        rake = - rake
        if az < 180:
            az += 180
        else:
            az -= 180

    return az, dip, rake


def vector_product(vec1, vec2):
    vec1 = _np.asarray(vec1)
    vec2 = _np.asarray(vec2)

    normalvec = vec1[[1, 2, 0]] * vec2[[2, 0, 1]] \
                - vec1[[2, 0, 1]] * vec2[[1, 2, 0]]
    return normalvec
