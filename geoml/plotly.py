# geoML - machine learning models for geospatial data
# Copyright (C) 2020  Ítalo Gomes Gonçalves
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
import collections as _collections


def _deep_update(d, u):
    """
    https://stackoverflow.com/questions/3232943/
    update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, _collections.Mapping):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def aspect_ratio_2d(vertical_exaggeration=1):
    return {"xaxis": {"constrain": "domain",
                      "showexponent": "none",
                      "exponentformat": "none"},
            "yaxis": {"scaleanchor": "x",
                      "showexponent": "none",
                      "exponentformat": "none",
                      "scaleratio": vertical_exaggeration}}


def aspect_ratio_3d(bounding_box, vertical_exaggeration=1):
    # aspect ratio
    aspect = bounding_box.max[0] - bounding_box.min[0]
    aspect = aspect / aspect[0]
    aspect[-1] = aspect[-1] * vertical_exaggeration

    # expanded bounding box
    exp_box = _np.stack([bounding_box.min[0] - 0.1 * aspect,
                         bounding_box.max[0] + 0.1 * aspect],
                        axis=0)

    return {"scene": {"aspectratio": {"x": aspect[0],
                                      "y": aspect[1],
                                      "z": aspect[2]},
                      "xaxis": {"showexponent": "none",
                                "exponentformat": "none",
                                "range": exp_box[:, 0]},
                      "yaxis": {"showexponent": "none",
                                "exponentformat": "none",
                                "range": exp_box[:, 1]},
                      "zaxis": {"showexponent": "none",
                                "exponentformat": "none",
                                "range": exp_box[:, 2]}}}


def bounding_box_3d(bounding_box, **kwargs):
    idx = _np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    idy = _np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1])
    idz = _np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
    obj = {"type": "scatter3d",
           "mode": "lines",
           "x": bounding_box[idx, 0],
           "y": bounding_box[idy, 1],
           "z": bounding_box[idz, 2],
           "hoverinfo": "x+y+z",
           "name": "bounding box"}
    obj = _deep_update(obj, kwargs)
    return [obj]


def planes_3d(coordinates, dip, azimuth, size=1, **kwargs):
    # conversions
    dip = -dip * _np.pi / 180
    azimuth = (90 - azimuth) * _np.pi / 180
    strike = azimuth - _np.pi / 2

    # dip and strike vectors
    dipvec = _np.concatenate([
        _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
        _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
        _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1)*size
    strvec = _np.concatenate([
        _np.array(_np.cos(strike), ndmin=2).transpose(),
        _np.array(_np.sin(strike), ndmin=2).transpose(),
        _np.zeros([dipvec.shape[0], 1])], axis=1)*size

    # surfaces
    surf = []
    for i, point in enumerate(coordinates):
        surf_points = _np.stack([
            point + dipvec[i],
            point - 0.5*strvec[i] - 0.5*dipvec[i],
            point + 0.5*strvec[i] - 0.5*dipvec[i]
        ], axis=0)
        obj = {"type": "mesh3d",
               "x": surf_points[:, 0],
               "y": surf_points[:, 1],
               "z": surf_points[:, 2],
               "hoverinfo": "x+y+z+text"}
        obj = _deep_update(obj, kwargs)
        surf.append(obj)
    return surf


def arrows_3d(coordinates, directions, size=1, **kwargs):
    head = coordinates + size / 2 * directions
    tail = coordinates - size / 2 * directions

    x = []
    y = []
    z = []
    for i in range(head.shape[0]):
        x.extend([tail[i, 0], head[i, 0], None])
        y.extend([tail[i, 1], head[i, 1], None])
        z.extend([tail[i, 2], head[i, 2], None])

    obj = {"type": "scatter3d",
           "mode": "lines",
           "x": x,
           "y": y,
           "z": z,
           "hoverinfo": "none"}

    obj = _deep_update(obj, kwargs)
    return [obj]


def numeric_points_3d(coordinates, values, **kwargs):
    # x, y, z, and text as lists
    text = [str(num) for num in values]
    x = coordinates[:, 0].tolist()
    y = coordinates[:, 1].tolist()
    z = coordinates[:, 2].tolist()

    obj = {"mode": "markers",
           "text": text,
           "marker": {"color": values}}

    obj.update({"type": "scatter3d",
                "x": x,
                "y": y,
                "z": z,
                "hoverinfo": "x+y+z+text"})
    obj = _deep_update(obj, kwargs)
    return [obj]


def categorical_points_3d(coordinates, values, colors, **kwargs):
    """
    Visualization of categorical data.

    Parameters
    ----------
    coordinates : ndarray
        The 3D coordinates.
    values : ndarray, Series
        The categories to plot.
    colors : dict
        Dictionary mapping the data labels to plotly color
        specifications.
    kwargs
        Additional arguments passed on to plotly's scatter3d
        function.

    Returns
    -------
    out : list
        A singleton list containing the data to plot in plotly's format.
    """
    # x, y, z, and colors as lists
    color_val = _np.arange(len(colors))
    colors_dict = dict(zip(colors.keys(), color_val))
    colors2 = _pd.Series(values).map(colors_dict).tolist()
    colors3 = [colors.get(item) for item in colors]
    cmax = len(color_val) - 1

    text = values.tolist()
    x = coordinates[:, 0].tolist()
    y = coordinates[:, 1].tolist()
    z = coordinates[:, 2].tolist()

    obj = {"mode": "markers",
           "text": text,
           "marker": {"color": colors2,
                      "colorscale": [[i / cmax, colors3[i]]
                                     for i in color_val]}}

    obj.update({"type": "scatter3d",
                "x": x,
                "y": y,
                "z": z,
                "hoverinfo": "x+y+z+text"})
    obj = _deep_update(obj, kwargs)
    return [obj]


def isosurface(verts, faces, values=None, **kwargs):
    obj = {"type": "mesh3d",
           "x": verts[:, 0],
           "y": verts[:, 1],
           "z": verts[:, 2],
           "i": faces[:, 0],
           "j": faces[:, 1],
           "k": faces[:, 2]}
    if values is not None:
        obj["intensity"] = values
    obj = _deep_update(obj, kwargs)
    return [obj]


def segments_3d(coordinates, labels, colors, **kwargs):
    # x, y, z, and colors as lists
    labels = _pd.Series(labels)
    color_val = _np.arange(len(colors))
    colors_dict = dict(zip(colors.keys(), color_val))
    colors2 = labels.map(colors_dict).tolist()
    colors3 = [colors.get(item) for item in colors]
    cmax = len(color_val) - 1
    n_data = coordinates.shape[0]
    text = labels.tolist()
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    x = x.tolist()
    y = y.tolist()
    z = z.tolist()

    # inserting None elements to break undesired lines
    # points object will always have an even number of rows
    # None elements must be inserted every 3rd position
    indexes = _np.flip(_np.arange(start=0, stop=n_data, step=2))
    for idx in indexes:
        colors2.insert(idx, "rgb(0,0,0)")  # dummy color
        x.insert(idx, None)
        y.insert(idx, None)
        z.insert(idx, None)
        text.insert(idx, None)

    obj = {"type": "scatter3d",
           "x": x,
           "y": y,
           "z": z,
           "hoverinfo": "x+y+z+text",
           "mode": "lines",
           "text": text,
           "line": {"color": colors2,
                    "colorscale": [[i / cmax, colors3[i]]
                                   for i in color_val]}}

    obj = _deep_update(obj, kwargs)
    return [obj]


def numeric_section_3d(gridded_x, gridded_y, gridded_z, values, **kwargs):
    obj = {"type": "surface",
           "x": gridded_x,
           "y": gridded_y,
           "z": gridded_z,
           "surfacecolor": values}
    obj = _deep_update(obj, kwargs)
    return [obj]


def mpl_to_plotly(cmap, level):
    rgba = _np.array(cmap(level))
    rgb = _np.round(rgba[:-1] * 255)
    rgb = tuple(int(num) for num in rgb)
    color_str = "rgb(%d,%d,%d)" % rgb
    return color_str
