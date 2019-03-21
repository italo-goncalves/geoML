# geoML - machine learning models for geospatial data
# Copyright (C) 2019  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["Points1D", "Points2D", "Points3D",
           "Grid1D", "Grid2D", "Grid3D",
           "Directions2D", "Directions3D",
           "DrillholeData",
           "merge"]

import pandas as _pd
import numpy as _np
# import plotly.graph_objs as go
import collections as _collections
import os as _os

from skimage import measure as _measure


def _update(d, u):
    """
    https://stackoverflow.com/questions/3232943/
    update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, _collections.Mapping):
            d[k] = _update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _bounding_box(coords):
    """
    Computes a point set's bounding box and its diagonal
    """
    bbox = _np.array([[_np.min(coords[:, i]) for i in range(coords.shape[1])],
                      [_np.max(coords[:, i]) for i in range(coords.shape[1])]])
    d = _np.sqrt(sum([_np.diff(bbox[:, i]) ** 2 for i in range(bbox.shape[1])]))
    return bbox, d


class _SpatialData(object):
    """Abstract class for spatial data in general"""

    def __init__(self):
        self._ndim = None
        self.bounding_box = None

    def __repr__(self):
        return self.__str__()

    @property
    def ndim(self):
        return self._ndim

    def aspect_ratio(self, vex=1):
        """
        Returns a list with plotly layout data.
        """
        if self._ndim == 1:
            raise ValueError("aspect not available for one-dimensional " +
                             "objects")
        elif self._ndim == 2:
            return {"xaxis": {"constrain": "domain",
                              "showexponent": "none",
                              "exponentformat": "none"},
                    "yaxis": {"scaleanchor": "x",
                              "showexponent": "none",
                              "exponentformat": "none",
                              "scaleratio": vex}}
        elif self._ndim == 3:
            # aspect ratio
            aspect = _np.apply_along_axis(lambda x: x[1] - x[0],
                                          0,
                                          self.bounding_box)
            aspect = aspect / aspect[0]
            aspect[-1] = aspect[-1] * vex

            # expanded bounding box
            def expand_10_perc(x):
                dif = x[1] - x[0]
                return [x[0] - 0.1 * dif, x[1] + 0.1 * dif]

            bbox = _np.apply_along_axis(lambda x: expand_10_perc(x),
                                        0,
                                        self.bounding_box)
            return {"scene": {"aspectratio": {"x": aspect[0],
                                              "y": aspect[1],
                                              "z": aspect[2]},
                              "xaxis": {"showexponent": "none",
                                        "exponentformat": "none",
                                        "range": bbox[:, 0]},
                              "yaxis": {"showexponent": "none",
                                        "exponentformat": "none",
                                        "range": bbox[:, 1]},
                              "zaxis": {"showexponent": "none",
                                        "exponentformat": "none",
                                        "range": bbox[:, 2]}}}


class _PointData(_SpatialData):
    """
        Abstract class for data represented as points in arbitrary locations.
    """
    def __init__(self, coords, data=None):
        super().__init__()
        self.coords = None
        self.coords_label = None
        if data is not None:
            if len(coords) != data.shape[0]:
                raise ValueError("number of rows of data and length of "
                                 "coords do not match")
        else:
            data = _pd.DataFrame()
        self.data = _pd.DataFrame(data)

    def as_data_frame(self):
        """
        Conversion of a spatial object to a data frame.
        """
        df = _pd.DataFrame(self.coords,
                           columns=self.coords_label)
        df = _pd.concat([df, self.data], axis=1)
        return df


class Points1D(_PointData):
    """
    Point data in 1D

    Attributes
    ----------
    coords : array
        A matrix with the spatial coordinates.
    data : data frame
        The coordinates' attributes.
    coords_label : str
        Label for spatial coordinates.
    bounding_box : array
        Object's spatial limits.
    diagonal : double
        The extent of the data's range, the diagonal of the bounding box.
    """
    def __init__(self, coords, data=None, coords_label=None):
        """
        Initializer for Points1D

        Parameters
        ----------
        coords : array, pandas series or data frame containing one column
            A vector with the spatial coordinates.
        data : data frame
            The coordinates' attributes (optional).
        coords_label : str
            Optional string to label the coordinates.
            Extracted from coords object if available.
        """
        super().__init__(coords, data)
        if isinstance(coords, _pd.core.frame.DataFrame):
            if coords_label is None:
                coords_label = coords.columns
            coords = coords.values
            if coords.shape[1] != 1:
                raise ValueError("coords data frame must have only one column")
        if isinstance(coords, _pd.core.series.Series):
            if coords_label is None:
                coords_label = coords.name
            coords = coords.values
        if coords_label is None:
            coords_label = "X"

        self._ndim = 1
        self.coords = _np.array(coords, ndmin=2).transpose()
        self.bounding_box = _np.array([[coords.min()], [coords.max()]])
        self.diagonal = coords.max() - coords.min()
        self.coords_label = coords_label

    def __str__(self):
        s = "Object of class " + self.__class__.__name__ + " with " \
            + str(len(self.coords)) + " data locations\n\n"

        if self.data is not None:
            s += "Data preview:\n\n"
            s += str(self.data.head())
        return s

    def draw_numeric(self, column, **kwargs):
        raise NotImplementedError()

    def draw_categorical(self, column, colors, **kwargs):
        raise NotImplementedError()


class Points2D(_PointData):
    """
    Point data in 2D.

    Attributes
    ----------
    coords : array
        A matrix with the spatial coordinates.
    data : data frame
        The coordinates' attributes.
    coords_label : str
        Label for spatial coordinates.
    bounding_box : array
        Object's spatial limits.
    diagonal : double
        The extent of the data's range, the diagonal of the bounding box.
    """
    def __init__(self, coords, data=None, coords_label=None):
        """
        Initializer for Points2D

        Parameters
        ----------
        coords : array or data frame
            The spatial coordinates. Must contain two columns.
        data : data frame
            The coordinates' attributes (optional)
        coords_label : str
            Optional string to label the coordinates.
            Extracted from coords object if available.
        """
        super().__init__(coords, data)
        if isinstance(coords, _pd.core.frame.DataFrame):
            if coords_label is None:
                coords_label = coords.columns
            coords = coords.values
        if coords_label is None:
            coords_label = ["X", "Y"]

        self.coords = coords
        self.bounding_box, self.diagonal = _bounding_box(coords)
        self.coords_label = coords_label
        self._ndim = 2

    def __str__(self):
        s = "Object of class " + self.__class__.__name__ + " with " \
            + str(self.coords.shape[0]) + " data locations\n\n"

        if self.data is not None:
            s += "Data preview:\n\n"
            s += str(self.data.head())
        return s

    def draw_numeric(self, column, **kwargs):
        raise NotImplementedError()

    def draw_categorical(self, column, colors, **kwargs):
        raise NotImplementedError()


class Points3D(_PointData):
    """
    Point data in 3D.

    Attributes
    ----------
    coords : array
        A matrix with the spatial coordinates.
    data : data frame
        The coordinates' attributes.
    coords_label : str
        Label for spatial coordinates.
    bounding_box : array
        Object's spatial limits.
    diagonal : double
        The extent of the data's range, the diagonal of the bounding box.
    """

    def __init__(self, coords, data=None, coords_label=None):
        """
        Initializer for Points3D.

        Parameters
        ----------
        coords : array or data frame
            The spatial coordinates. Must contain two columns.
        data : data frame
            The coordinates' attributes (optional)
        coords_label : str
            Optional string to label the coordinates.
            Extracted from coords object if available.
        """
        super().__init__(coords, data)
        if isinstance(coords, _pd.core.frame.DataFrame):
            if coords_label is None:
                coords_label = coords.columns
            coords = coords.values
        if coords_label is None:
            coords_label = ["X", "Y", "Z"]

        self.coords = coords
        self.bounding_box, self.diagonal = _bounding_box(coords)
        self.coords_label = coords_label
        self._ndim = 3

    def __str__(self):
        s = "Object of class " + self.__class__.__name__ \
            + " with " + str(self.coords.shape[0]) + " data locations\n\n"

        if self.data is not None:
            s += "Data preview:\n\n"
            s += str(self.data.head())
        return s

    def draw_numeric(self, column, **kwargs):
        raise NotImplementedError()

    def draw_categorical(self, column, colors, **kwargs):
        raise NotImplementedError()


class Grid1D(Points1D):
    """
    Equally spaced points in 1D.

    Attributes
    ----------
    step_size : list
        Distance between grid nodes.
    grid : list
        The grid coordinates.
    grid_size : list
        The number of points in grid.
    """
    def __init__(self, start, n, step=None, end=None):
        """
        Initializer for Grid1D.

        Parameters
        ----------
        start :
            Starting point for grid.
        n : int
            Number of grid nodes.
        step :
            Spacing between grid nodes.
        end :
            Last grid point.


        Either step or end must be given. If both are given, end is ignored.
        """
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        if step is not None:
            end = start + (n - 1) * step
        grid = _np.linspace(start, end, n)

        super().__init__(grid)
        self.step_size = [step]
        self.grid = [grid]
        self.grid_size = [n]


class Grid2D(Points2D):
    """
    Equally spaced points in 2D.

    Attributes
    ----------
    step_size : list
        Distance between grid nodes.
    grid : list
        The grid coordinates.
    grid_size : list
        The number of points in grid.
    """

    def __init__(self, start, n, step=None, end=None):
        """
        Initializer for Grid2D.

        Parameters
        ----------
        start : length 2 array, list, or tuple
            Starting point for grid.
        n : length 2 array, list, or tuple of ints
            Number of grid nodes.
        step : length 2 array, list, or tuple
            Spacing between grid nodes.
        end : length 2 array, list, or tuple
            Last grid point.


        Either step or end must be given. If both are given, end is ignored.
        """
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        start = _np.array(start)
        n = _np.array(n)
        if step is not None:
            step = _np.array(step)
            end = start + (n - 1) * step
        else:
            end = _np.array(end)
        grid_x = _np.linspace(start[0], end[0], n[0])
        grid_y = _np.linspace(start[1], end[1], n[1])
        coords = _np.array([(x, y) for y in grid_y for x in grid_x])

        super().__init__(coords)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y]
        self.grid_size = n.tolist()

    def as_image(self, column):
        """
        Reshapes the data in the form of a matrix for plotting.

        The output can be used in plotting functions such as
        matplotlib's imshow(). If you use it, do not forget to set
        origin="lower".

        Parameters
        ----------
        column : str
            The name of the column to use.

        Returns
        -------
        image : array
            A 2-dimensional array.
        """
        image = _np.reshape(self.data[column].values,
                            newshape=self.grid_size,
                            order="F")
        image = image.transpose()
        #image = _np.flip(image, 0)
        return image


class Grid3D(Points3D):
    """
    Equally spaced points in 3D.

    Attributes
    ----------
    step_size : list
        Distance between grid nodes.
    grid : list
        The grid coordinates.
    grid_size : list
        The number of points in grid.
    """

    def __init__(self, start, n, step=None, end=None):
        """
        Initializer for Grid3D.

        Parameters
        ----------
        start : length 2 array, list, or tuple
            Starting point for grid.
        n : length 2 array, list, or tuple of ints
            Number of grid nodes.
        step : length 2 array, list, or tuple
            Spacing between grid nodes.
        end : length 2 array, list, or tuple
            Last grid point.


        Either step or end must be given. If both are given, end is ignored.
        """
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        start = _np.array(start)
        n = _np.array(n)
        if step is not None:
            step = _np.array(step)
            end = start + (n - 1) * step
        else:
            end = _np.array(end)

        grid_x = _np.linspace(start[0], end[0], n[0])
        grid_y = _np.linspace(start[1], end[1], n[1])
        grid_z = _np.linspace(start[2], end[2], n[2])
        coords = _np.array([(x, y, z) for z in grid_z for y in grid_y
                            for x in grid_x])

        super().__init__(coords)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y, grid_z]
        self.grid_size = n.tolist()

    def get_contour(self, column, value):
        """
        Calls the marching cubes algorithm to extract a isosurface.

        Parameters
        ----------
        column : str
            The column with the variable to contour.
        value : double
            The value on which to calculate the isosurface.

        Returns
        -------
        verts : array
            Mesh vertices.
        faces : array
            Mesh faces.
        normals : array
            Mesh normals.
        values : array
            Value that can be used for color coding.


        This method calls the skimage.measure.marching_cubes_lewiner().
        See the original documentation for details.
        """
        cube = _np.reshape(self.data[column].values,
                           self.grid_size, order="F")
        verts, faces, normals, values = _measure.marching_cubes_lewiner(
            cube, value, gradient_direction="ascent",
            allow_degenerate=False, spacing=self.step_size)
        return verts, faces, normals, values

    def draw_contour(self, column, value, **kwargs):
        """Creates plotly object with the contour at the specified value."""
        obj = {}

        verts, faces, normals, values = self.get_contour(column, value)
        for i in range(3):
            verts[:, i] += self.grid[i][0]
        obj.update({"type": "mesh3d",
                    "x": verts[:, 0],
                    "y": verts[:, 1],
                    "z": verts[:, 2],
                    "i": faces[:, 0],
                    "j": faces[:, 1],
                    "k": faces[:, 2]})
        obj = _update(obj, kwargs)
        return [obj]

    def draw_section_numeric(self, column, axis=0, **kwargs):
        """
        Returns a list with all possible sections along the specified axis,
        in plotly format.
        """
        if (axis > 3) | (axis < 0):
            raise ValueError("axis must be 0, 1, or 2")

        sections = []
        minval = self.data[column].values.min()
        maxval = self.data[column].values.max()

        # sections in X
        if axis == 0:
            for i in range(self.grid_size[0]):
                value = self.grid[0][i]
                idx = self.coords[:, 0] == value
                y, z = _np.meshgrid(self.grid[1], self.grid[2], indexing="ij")
                col = self.data.loc[idx, column].values
                col = _np.reshape(col,
                                  [self.grid_size[1], self.grid_size[2]],
                                  order="F")
                sections.append({
                    "type": "surface",
                    "x": _np.ones_like(y) * value,
                    "y": y,
                    "z": z,
                    "surfacecolor": col,
                    "cmin": minval,
                    "cmax": maxval})
                sections[i] = _update(sections[i], kwargs)

        if axis == 1:
            for i in range(self.grid_size[1]):
                value = self.grid[1][i]
                idx = self.coords[:, 1] == value
                x, z = _np.meshgrid(self.grid[0], self.grid[2], indexing="ij")
                col = self.data.loc[idx, column].values
                col = _np.reshape(col,
                                  [self.grid_size[0], self.grid_size[2]],
                                  order="F")
                sections.append({
                    "type": "surface",
                    "x": x,
                    "y": _np.ones_like(x) * value,
                    "z": z,
                    "surfacecolor": col,
                    "cmin": minval,
                    "cmax": maxval})
                sections[i] = _update(sections[i], kwargs)

        if axis == 2:
            for i in range(self.grid_size[2]):
                value = self.grid[2][i]
                idx = self.coords[:, 2] == value
                x, y = _np.meshgrid(self.grid[0], self.grid[1], indexing="ij")
                col = self.data.loc[idx, column].values
                col = _np.reshape(col,
                                  [self.grid_size[0], self.grid_size[1]],
                                  order="F")
                sections.append({
                    "type": "surface",
                    "x": x,
                    "y": y,
                    "z": _np.ones_like(x) * value,
                    "surfacecolor": col,
                    "cmin": minval,
                    "cmax": maxval})
                sections[i] = _update(sections[i], kwargs)

        return sections

    def draw_section_categorical(self, column, colors, axis=0, **kwargs):
        """
        Returns a list with all possible sections along the specified axis,
        in plotly format.
        """
        if (axis > 3) | (axis < 0):
            raise ValueError("axis must be 0, 1, or 2")

        color_val = _np.arange(len(colors))
        colors_dict = dict(zip(colors.keys(), color_val))
        colors2 = self.data[column].map(colors_dict).values
        colors3 = [colors.get(item) for item in colors]

        # fooling the continuous colorbar
        minval = 0
        maxval = len(color_val)
        colorscale = [[(i + 0.5) / maxval, colors3[i]] for i in color_val]
        for i in _np.flip(color_val):
            colorscale.insert(i + 1, [(i + 1) / maxval, colors3[i]])
            colorscale.insert(i, [(i + 0) / maxval, colors3[i]])

        sections = []
        # sections in X
        if axis == 0:
            for i in range(self.grid_size[0]):
                value = self.grid[0][i]
                idx = self.coords[:, 0] == value
                y, z = _np.meshgrid(self.grid[1], self.grid[2], indexing="ij")
                col = _np.reshape(colors2[idx],
                                  [self.grid_size[1], self.grid_size[2]],
                                  order="F")
                sections.append({
                    "type": "surface",
                    "x": _np.ones_like(y) * value,
                    "y": y,
                    "z": z,
                    "surfacecolor": col + 0.5,
                    "colorscale": colorscale,
                    "cmin": minval,
                    "cmax": maxval,
                    "colorbar": {"tickmode": "array",
                                 "ticktext": [item for item in colors],
                                 "tickvals": color_val + 0.5}})
                sections[i] = _update(sections[i], kwargs)

        if axis == 1:
            for i in range(self.grid_size[1]):
                value = self.grid[1][i]
                idx = self.coords[:, 1] == value
                x, z = _np.meshgrid(self.grid[0], self.grid[2], indexing="ij")
                col = _np.reshape(colors2[idx],
                                  [self.grid_size[0], self.grid_size[2]],
                                  order="F")
                sections.append({
                    "type": "surface",
                    "x": x,
                    "y": _np.ones_like(x) * value,
                    "z": z,
                    "surfacecolor": col + 0.5,
                    "colorscale": colorscale,
                    "cmin": minval,
                    "cmax": maxval,
                    "colorbar": {"tickmode": "array",
                                 "ticktext": [item for item in colors],
                                 "tickvals": color_val + 0.5}})
                sections[i] = _update(sections[i], kwargs)

        if axis == 2:
            for i in range(self.grid_size[2]):
                value = self.grid[2][i]
                idx = self.coords[:, 2] == value
                x, y = _np.meshgrid(self.grid[0], self.grid[1], indexing="ij")
                col = _np.reshape(colors2[idx],
                                  [self.grid_size[0], self.grid_size[1]],
                                  order="F")
                sections.append({
                    "type": "surface",
                    "x": x,
                    "y": y,
                    "z": _np.ones_like(x) * value,
                    "surfacecolor": col + 0.5,
                    "colorscale": colorscale,
                    "cmin": minval,
                    "cmax": maxval,
                    "colorbar": {"tickmode": "array",
                                 "ticktext": [item for item in colors],
                                 "tickvals": color_val + 0.5}})
                sections[i] = _update(sections[i], kwargs)

        return sections


class _DirectionalData(_PointData):
    """
    Abstract class for directional data
    """

    def as_data_frame(self):
        """
        Conversion of a spatial object to a data frame.
        """
        df_coords = _pd.DataFrame(self.coords,
                                  columns=self.coords_label)
        df_directions = _pd.DataFrame(self.directions,
                                      columns=self.directions_label)
        df = _pd.concat([df_coords, df_directions, self.data], axis=1)
        return df


class Directions2D(_DirectionalData, Points2D):
    """
    Spatial directions in 2D.
    """

    def __init__(self, coords, directions, data=None, coords_label=None,
                 directions_label=None):
        """
        Initializer for Directions2D.

        Parameters
        ----------
        coords : a 2-dimensional array or data frame
            Array-like object with the spatial coordinates.
        directions : a 2-dimensional array or data frame
            It is expected that the rows have unit norm.
        data : pandas DataFrame (optional)
            A data frame with additional attributes.
        coords_label : str
            Optional string to label the coordinates.
            Extracted from coords object if available.
        directions_label : str
            Optional string to label the directions.
            Extracted from directions object if available.
        """
        if coords.shape != directions.shape:
            raise ValueError("shape of coords and directions do not match")
        super().__init__(coords, data, coords_label)
        if isinstance(directions, _pd.core.frame.DataFrame):
            if directions_label is None:
                directions_label = directions.columns
            directions = directions.values
        if directions_label is None:
            directions_label = ["d" + s for s in self.coords_label]

        self.directions = directions
        self.directions_label = directions_label

    @classmethod
    def from_azimuth(cls, coords, azimuth, data=None):
        """
        Creates a Directions2D object from azimuth data.

        Parameters
        ----------
        coords : array or data frame
            The coordinates containing the azimuth measurements.
        azimuth : array
            Azimuth, between 0 and 360 degrees.
        data : data frame
            Additional data.

        Returns
        -------
        A Directions2D object.
        """
        directions = _pd.DataFrame({"dX": _np.sin(azimuth / 180 * _np.pi),
                                    "dY": _np.cos(azimuth / 180 * _np.pi)})
        return Directions2D(coords, directions, data)

    def draw_arrows(self, size=1):
        raise NotImplementedError()


class Directions3D(_DirectionalData, Points3D):
    """
    Spatial directions in 3D.
    """

    def __init__(self, coords, directions, data=None, coords_label=None,
                 directions_label=None):
        """
        Initializer for Directions3D

        Parameters
        ----------
        coords : a 2-dimensional array or data frame
            Array-like object with the spatial coordinates.
        directions : a 2-dimensional array or data frame
            It is expected that the rows have unit norm.
        data : pandas DataFrame (optional)
            A data frame with additional attributes.
        coords_label : str
            Optional string to label the coordinates.
            Extracted from coords object if available.
        directions_label : str
            Optional string to label the directions.
            Extracted from directions object if available.
        """
        if coords.shape != directions.shape:
            raise ValueError("shape of coords and directions do not match")
        super().__init__(coords, data, coords_label)
        if isinstance(directions, _pd.core.frame.DataFrame):
            if directions_label is None:
                directions_label = directions.columns
            directions = directions.values
        if directions_label is None:
            directions_label = ["d" + s for s in self.coords_label]

        self.directions = directions
        self.directions_label = directions_label

    def draw_arrows(self, size=1):
        raise NotImplementedError()

    @classmethod
    def from_planes(cls, coords, azimuth, dip, data=None):
        """
        Creates a Directions3D object from planar information,
        represented by azimuth direction and dip.
        
        Parameters
        ----------
        coords : array-like or data frame with 3 columns
            The data points' spatial coordinates.
        azimuth : array
            Azimuth direction, between 0 and 360 degrees.
        dip : array
            Dip, between 0 and 90 degrees.
        data : pandas DataFrame
            A data frame with additional data.
        
        Returns
        -------
        A Directions3D object.
        """
        # conversions
        dip = -dip * _np.pi / 180
        azimuth = (90 - azimuth) * _np.pi / 180
        strike = azimuth - _np.pi / 2
        if isinstance(coords, _pd.core.frame.DataFrame):
            coords = coords.values

        # dip and strike vectors
        dipvec = _np.concatenate([
            _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
            _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
            _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1)
        strvec = _np.concatenate([
            _np.array(_np.cos(strike), ndmin=2).transpose(),
            _np.array(_np.sin(strike), ndmin=2).transpose(),
            _np.zeros([dipvec.shape[0], 1])], axis=1)

        # result
        vecs = _np.concatenate([dipvec, strvec], axis=0)
        coords = _np.concatenate([coords, coords], axis=0)
        if data is not None:
            data = _pd.concat([data, data], axis=0)
        return cls(coords, vecs, data)

    @classmethod
    def from_lines(cls, coords, azimuth, dip, data=None):
        """
        Creates a Directions3D object from linear information.
        
        Parameters
        ----------
        coords : array-like or data frame with 3 columns.
            The data coordinates.
        azimuth : array
            Azimuth direction, between 0 and 360 degrees.
        dip : array
            Dip, between 0 and 90 degrees.
        data : data frame
            A data frame with additional data.
        
        Returns
        -------
        A Directions3D object.
        """
        # conversions
        dip = -dip * _np.pi / 180
        azimuth = (90 - azimuth) * _np.pi / 180
        if isinstance(coords, _pd.core.frame.DataFrame):
            coords = coords.values

        # dip and strike vectors
        dipvec = _np.concatenate([
            _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
            _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
            _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1)

        # result
        return cls(coords, dipvec, data)

    @classmethod
    def plane_normal(cls, coords, azimuth, dip, data=None):
        """
        Creates a Directions3D object representing the normal
        vectors of the planes parameterized by azimuth and dip.
        
        Parameters
        ----------
        coords : array-like or data frame with 3 columns.
            The data coordinates.
        azimuth : array
            Azimuth direction, between 0 and 360 degrees.
        dip : array
            Dip, between 0 and 90 degrees.
        data : data frame
            A data frame with additional data.
        
        Returns
        -------
        A Directions3D object.
        """
        n_data = coords.shape[0]
        plane_dirs = cls.from_planes(coords, azimuth, dip)
        vec1 = plane_dirs.directions[0:n_data, :]
        vec2 = plane_dirs.directions[n_data:(2 * n_data), :]

        normalvec = vec1[:, [1, 2, 0]] * vec2[:, [2, 0, 1]] \
                    - vec1[:, [2, 0, 1]] * vec2[:, [1, 2, 0]]
        normalvec = _np.apply_along_axis(lambda x: x/_np.sqrt(_np.sum(x ** 2)),
                                         axis=0,
                                         arr=normalvec)

        # result
        return cls(coords, normalvec, data)


class DrillholeData(_SpatialData):
    """
    Drillhole data
    
    Attributes
    ----------
    coords_from, coords_to :
        Coordinates for the start and end of each segment.
    data :
        The drillhole segments' attributes. The column HOLEID is reserved.
    lengths :
        The length of each segment
    """

    def __init__(self, collar=None, assay=None, survey=None,
                 holeid="HOLEID", x="X", y="Y", z="Z", fr="FROM", to="TO",
                 **kwargs):
        """
        Initializer for DrillholeData.

        Parameters
        ----------
        collar : DataFrame
            The coordinates of the drillhole collars.
        assay : DataFrame
            The interval data.
        survey : DataFrame
            The hole survey data.
        holeid : str
            Column with the hole index.
        x,y,z : str
            Columns in collar with the coordinates.
        fr,to : str
            Columns in assay with the interval measurements.


        The column HOLEID in the output is reserved.
        """

        if "coords_from" in kwargs:
            # internal instantiation
            self.coords_from = kwargs["coords_from"]
            self.coords_to = kwargs["coords_to"]
            self.data = kwargs["data"]
        else:
            # instantiation from input
            if survey is not None:
                raise NotImplementedError("survey data is not yet supported")

            # column names
            collar_names = collar.columns
            collar_names = collar_names[[a not in [x, y, z]
                                         for a in collar_names.tolist()]]
            assay_names = assay.columns
            assay_names = assay_names[[a not in [holeid, fr, to]
                                       for a in assay_names.tolist()]]
            if (holeid not in collar_names) & (holeid not in assay_names):
                raise ValueError("holeid column must be present in collar "
                                 "and assay data frames")

            # processing data
            df = _pd.merge(collar,
                           assay,
                           on=holeid,
                           suffixes=("_collar", "_assay"))
            df_from = _pd.DataFrame({"X": 0, "Y": 0, "Z": df[fr]})
            df_to = _pd.DataFrame({"X": 0, "Y": 0, "Z": df[to]})
            coords_from = df.loc[:, [x, y, z]].values - df_from.values
            coords_to = df.loc[:, [x, y, z]].values - df_to.values
            df = df.drop([x, y, z], axis=1)
            df = df.rename(columns={holeid: "HOLEID"})

            # output
            self.coords_from = coords_from
            self.coords_to = coords_to
            self.data = df

        self._ndim = self.coords_from.shape[1]
        self.lengths = _np.sqrt(_np.sum((self.coords_from
                                         - self.coords_to) ** 2,
                                axis=1))

        coords2 = _np.concatenate([self.coords_from, self.coords_to], axis=0)
        self.bounding_box, self.diagonal = _bounding_box(coords2)

    def __str__(self):
        s = "Object of class " + self.__class__.__name__ + "\n\n"
        s += str(self.coords_from.shape[0]) + " drillhole segments in "
        s += str(self._ndim) + " dimension"
        if self._ndim > 1:
            s += "s"
        s += "\n\n"
        if self.data is not None:
            s += "Data preview:\n\n"
            s += str(self.data.head())
        return s

    def as_points(self, locations=_np.array([0.05, 0.5, 0.95])):
        if locations.__class__ is not _np.ndarray:
            locations = _np.array(locations)
        if (locations.min() < 0) | (locations.max() > 1):
            raise ValueError("locations must contain values between " +
                             "0 and 1, inclusive")

        dif = self.coords_to - self.coords_from
        points = [self.coords_from + dif * x for x in locations]
        points = _np.concatenate(points, axis=0)
        df = _pd.concat([self.data] * len(locations), ignore_index=True)

        return Points3D(points, df)

    def merge_segments(self, by):
        """
        Merges redundant segments according to the by column in order to
        form longer segments.
        
        Parameters
        ----------
        by : name of the column with categorical data
        """
        coords_dif = self.coords_from - self.coords_to
        directions = _np.apply_along_axis(lambda x: x / self.lengths,
                                          axis=0, arr=coords_dif)

        # finding mergeable segments
        # condition 1 - segments sharing a point
        # (coords_to[i,:] == coords_from[i-1,:])
        start_end = _np.apply_along_axis(
            lambda x: all(x < 1e-6), axis=1,
            arr=self.coords_to[0:(self.coords_to.shape[0] - 1), :]
                - self.coords_from[1:self.coords_from.shape[0], :])
        # condition 2 - parallelism
        dir_from = directions[0:(directions.shape[0] - 1), :]
        dir_to = directions[1:directions.shape[0], :]
        parallel = _np.apply_along_axis(
            lambda x: all(x < 1e-6), axis=1,
            arr=dir_from - dir_to)
        # condition 3 - same value in "by" column
        val_from = self.data.loc[0:(self.coords_from.shape[0] - 1), by].values
        val_to = self.data.loc[1:self.coords_from.shape[0], by].values
        same_value = [val_from[i] == val_to[i] for i in range(len(val_to))]
        # final vector
        mergeable = start_end & parallel & same_value

        # merged object
        # find contiguous mergeable segments
        merge_ids = _np.split(
            _np.arange(self.coords_from.shape[0]),
            _np.where(_np.concatenate([[False], ~mergeable]))[0])
        # merge_ids may contain empty elements
        merge_ids = [x for x in merge_ids if x.size > 0]
        # coordinates
        new_coords_from = _np.zeros([len(merge_ids), self._ndim])
        new_coords_to = _np.zeros([len(merge_ids), self._ndim])
        for i in range(len(merge_ids)):
            new_coords_from[i, :] = self.coords_from[merge_ids[i][0], :]
            new_coords_to[i, :] = self.coords_to[merge_ids[i][-1], :]
        # data
        cols = by if by == "HOLEID" else ["HOLEID", by]
        new_df = self.data.loc[[x[0] for x in merge_ids], cols]
        new_df = new_df.reset_index(drop=True)  # very important
        # initialization
        return DrillholeData(coords_from=new_coords_from,
                             coords_to=new_coords_to,
                             data=new_df)

    def get_contacts(self, by):
        """
        Returns a Points3D with the coordinates of the contacts between
        two different categories.
        
        Parameters
        ----------
        by : name of the column with the category
        """
        merged = self.merge_segments(by)
        points = merged.as_points(_np.array([0, 1]))

        # finding duplicates
        dup_label = _pd.DataFrame(points.coords).duplicated(keep="last")
        dup1 = _np.where(dup_label.values)[0]
        dup2 = dup1 - 1

        # new object
        new_points = points.coords[dup1, :]
        new_data = _pd.DataFrame(
            {"HOLEID": points.data.loc[dup2, "HOLEID"].values,
             by + "_up": points.data.loc[dup2, by].values,
             by + "_down": points.data.loc[dup1, by].values})
        new_data = new_data.reset_index(drop=True)
        return Points3D(new_points, new_data)

    def as_classification_input(self, by,
                                locations=_np.array([0.25, 0.5, 0.75])):
        """
        Returns a Points3D object in a format suitable for use as input to
        a classification model.
        """
        merged = self.merge_segments(by)
        point = merged.as_points(locations)
        point.data[by + "_up"] = point.data[by]
        point.data[by + "_down"] = point.data[by]
        contacts = merged.get_contacts(by)
        out = merge(point, contacts)
        out.data = out.data.drop(by, axis=1)
        return out

    def draw_categorical(self, column, colors, **kwargs):
        # converting to points and reordering
        merged = self.merge_segments(column)
        points = Points3D(
            _np.concatenate([merged.coords_from, merged.coords_to], axis=0),
            _pd.concat([merged.data] * 2, axis=0).reset_index(drop=True))
        seq = _np.arange(merged.coords_from.shape[0])
        sort_idx = _np.argsort(_np.concatenate([seq, seq + 0.1]))
        points.coords = points.coords[sort_idx, :]
        points.data = points.data.loc[sort_idx, :].reset_index(drop=True)

        # x, y, z, and colors as lists
        color_val = _np.arange(len(colors))
        colors_dict = dict(zip(colors.keys(), color_val))
        colors2 = points.data[column].map(colors_dict).tolist()
        colors3 = [colors.get(item) for item in colors]
        cmax = len(color_val) - 1
        n_data = points.coords.shape[0]
        text = points.data[column].tolist()
        x = points.coords[:, 0].tolist()
        y = points.coords[:, 1].tolist() if self._ndim > 1 else [None] * n_data
        z = points.coords[:, 2].tolist() if self._ndim > 2 else [None] * n_data

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

        obj = {"mode": "lines",
               "name": column,
               "text": text,
               "line": {"color": colors2,
                        "colorscale": [[i / cmax, colors3[i]]
                                       for i in color_val]}}
        if self._ndim == 1:
            obj.update({"type": "scatter",
                        "x": x,
                        "y": _np.repeat(0, points.coords.shape[0]),
                        "hoveron": "fills",
                        "hoverinfo": "x+text"})
        elif self._ndim == 2:
            obj.update({"type": "scatter",
                        "x": x,
                        "y": y,
                        "hoveron": "fills",
                        "hoverinfo": "x+y+text"})
        elif self._ndim == 3:
            obj.update({"type": "scatter3d",
                        "x": x,
                        "y": y,
                        "z": z,
                        "hoverinfo": "x+y+z+text"})
        obj = _update(obj, kwargs)
        return [obj]

    def draw_numeric(self, column, **kwargs):
        raise NotImplementedError()


def merge(x, y):
    """
    Merges point objects
    """
    if x.ndim != y.ndim:
        raise ValueError("dimensions of x and y do not match")
    if isinstance(x, _PointData) & isinstance(y, _PointData):
        # coordinates
        new_coords = _np.concatenate([x.coords, y.coords], axis=0)

        # data
        all_cols = _pd.unique(x.data.columns.values.tolist() +
                              y.data.columns.values.tolist())
        df_x = _pd.DataFrame(columns=all_cols)
        df_y = _pd.DataFrame(columns=all_cols)
        for col in x.data.columns:
            df_x[col] = x.data[col]
        for col in y.data.columns:
            df_y[col] = y.data[col]
        new_data = _pd.concat([df_x, df_y]).reset_index(drop=True)

        return x.__class__(new_coords, new_data)
    else:
        raise ValueError("x and y are not compatible for merging")


class Examples(object):
    """Example data."""
    def __new__(cls, *args, **kwargs):
        return None

    @staticmethod
    def walker():
        """
        Walker lake dataset.

        Returns
        -------
        walker_point : geoml.data.Points2D
            Dataset with 470 samples.
        walker_grid : geoml.data.Grid2D
            Full data.
        """
        path = _os.path.dirname(__file__)
        path_walker = _os.path.join(path, "sample_data\\walker.dat")
        path_walker_ex = _os.path.join(path, "sample_data\\walker_ex.dat")

        walker = _pd.read_table(path_walker)
        walker_ex = _pd.read_table(path_walker_ex, sep=",")

        walker_point = Points2D(walker[["X", "Y"]],
                                walker.drop(["X", "Y"], axis=1))
        walker_grid = Grid2D(start=[1, 1], n=[260, 300], step=[1, 1])
        walker_grid.data = walker_ex.drop(["X", "Y"], axis=1)

        return walker_point, walker_grid

    @staticmethod
    def ararangua():
        """
        Drillhole data from Araranguá town, Brazil.

        Returns
        -------
        ara_dh : geoml.data.DrillholeData
            A dataset with 13 drillholes.
        """
        path = _os.path.dirname(__file__)
        file = _os.path.join(path, "sample_data\\Araranguá.xlsx")

        ara_lito = _pd.read_excel(file, sheet_name="Lito")
        ara_collar = _pd.read_excel(file, sheet_name="Collar")

        ara_dh = DrillholeData(ara_collar,
                               ara_lito,
                               holeid="Hole ID",
                               fr="From",
                               to="To")
        return ara_dh

    @staticmethod
    def example_fold():
        """
        Example directional data.

        Returns
        -------
        point : geoml.data.Points2D
            Some coordinates with two rock labels.
        dirs : geoml.data.Directions2D
            Structural measurements representing a fold.
        """
        ex_point = _pd.DataFrame(
            {"X": _np.array([25, 40, 60, 85,
                             5, 10, 45, 50, 55, 75, 90,
                             15, 20, 30, 50, 65, 75, 90, 25, 50, 65, 75]),
             "Y": _np.array([25, 60, 50, 15,
                             50, 80, 10, 30, 10, 75, 90,
                             15, 35, 65, 85, 65, 50, 20, 10, 50, 20, 10]),
             "rock": _np.concatenate([_np.repeat("rock_a", 4),
                                      _np.repeat("rock_b", 7),
                                      _np.repeat("boundary", 11)])})
        ex_point["label_1"] = _pd.Categorical(_np.concatenate(
            [_np.repeat("rock_a", 4),
             _np.repeat("rock_b", 7),
             _np.repeat("rock_a", 11)]))
        ex_point["label_2"] = _pd.Categorical(_np.concatenate(
            [_np.repeat("rock_a", 4),
             _np.repeat("rock_b", 7),
             _np.repeat("rock_b", 11)]))

        ex_dir = _pd.DataFrame(
            {"X": _np.array([40, 50, 70, 90, 30, 20, 10]),
             "Y": _np.array([40, 85, 70, 30, 50, 60, 10]),
             "strike": _np.array([30, 90, 325, 325, 30, 30, 30])})

        point = Points2D(ex_point[["X", "Y"]],
                         ex_point.drop(["X", "Y"], axis=1))
        dirs = Directions2D.from_azimuth(ex_dir[["X", "Y"]], ex_dir["strike"])

        return point, dirs

    @staticmethod
    def sunspot_number():
        """
        Sunspot number data.

        This data is downloaded from the Royal Observatory of Belgium
        SILSO website (http://sidc.be/silso/home), and is distributed under
        the CC BY-NC4.0 license (https://goo.gl/PXrLYd).

        Returns
        -------
        yearly - Point1D
            Yearly averages since 1700.
        monthly - Point1D
            Monthly averages since 1700.
        daily - Point1D
            Daily total sunspot number since 1818.
        """
        yearly_df = _pd.read_table(
            "http://sidc.be/silso/INFO/snytotcsv.php",
            sep=";", header=None)
        yearly_df.set_axis(["year", "sn", "sn_std",
                             "n_obs", "definitive"],
                            axis="columns", inplace=True)
        yearly = Points1D(_np.arange(1700, yearly_df.shape[0] + 1700),
                          yearly_df)

        monthly_df = _pd.read_table(
            "http://sidc.oma.be/silso/INFO/snmtotcsv.php",
            sep=";", header=None)
        monthly_df.set_axis(["year", "month", "year_frac", "sn", "sn_std",
                             "n_obs", "definitive"],
                            axis="columns", inplace=True)
        monthly = Points1D(_np.arange(1, monthly_df.shape[0] + 1), monthly_df)

        daily_df = _pd.read_table("http://sidc.oma.be/silso/INFO/sndtotcsv.php",
                                  sep=";", header=None)
        daily_df.set_axis(["year", "month", "day", "year_frac", "sn", "sn_std",
                           "n_obs", "definitive"],
                          axis="columns", inplace=True)
        daily = Points1D(_np.arange(1, daily_df.shape[0] + 1), daily_df)

        return yearly, monthly, daily
