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
# MERCHANTABILITY or FITNESS FOR matrix PARTICULAR PURPOSE.  See the
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
import collections as _collections
import copy as _copy

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
    d = _np.squeeze(d)
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

    def draw_bounding_box(self, **kwargs):
        idx = _np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        idy = _np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1])
        idz = _np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
        obj = {"type": "scatter3d",
               "mode": "lines",
               "x": self.bounding_box[idx, 0],
               "y": self.bounding_box[idy, 1],
               "z": self.bounding_box[idz, 2],
               "hoverinfo": "x+y+z",
               "name": "bounding box"}
        obj = _update(obj, kwargs)
        return [obj]


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

    def aggregate_categorical(self, column, grid):
        """
        Aggregation of categorical data in a regular grid.

        Parameters
        ----------
        column : str
            The name of the column with the data to work on.
        grid :
            matrix Grid*D object.

        Returns
        -------
        out :
            matrix point object with the aggregated data.

        This method is used to downsample large point clouds. Given a grid
        with regularly spaced coordinates, the label assigned to each
        grid coordinate is the dominant label among all the data points in
        its vicinity, considering that each coordinate is the center of a
        rectangular cell. Cells without any data are dropped.
        """
        if self.ndim != grid.ndim:
            raise ValueError("grid dimension different from object's")

        # writing grid id
        if grid.ndim == 1:
            grid_id = _np.array(range(int(grid.grid_size[0])))
            cols = ["xid"]
        if grid.ndim == 2:
            grid_id = _np.array(
                [(x, y) for y in range(int(grid.grid_size[1]))
                        for x in range(int(grid.grid_size[0]))])
            cols = ["xid", "yid"]
        if grid.ndim == 3:
            grid_id = _np.array([(x, y, z)
                                for z in range(int(grid.grid_size[2]))
                                for y in range(int(grid.grid_size[1]))
                                for x in range(int(grid.grid_size[0]))])
            cols = ["xid", "yid", "zid"]
        grid_id = _pd.DataFrame(grid_id, columns=cols)

        # resetting grid
        grid = _copy.deepcopy(grid)
        grid.data = grid_id

        # identifying cell id
        # raw_data = self.as_data_frame()
        raw_data = _pd.DataFrame(self.data)
        raw_data["xid"] = _np.round(
            (self.coords[:, 0] - grid.grid[0][0]
             - grid.step_size[0] / 2) / grid.step_size[0])
        if self.ndim >= 2:
            raw_data["yid"] = _np.round(
                (self.coords[:, 1] - grid.grid[1][0]
                 - grid.step_size[1] / 2) / grid.step_size[1])
        if self.ndim == 3:
            raw_data["zid"] = _np.round(
                (self.coords[:, 2] - grid.grid[2][0]
                 - grid.step_size[2] / 2) / grid.step_size[2])

        # counting values inside cells
        grid_full = grid.as_data_frame()
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + [column]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = data_3.set_index(cols) \
            .join(grid_full.set_index(cols)) \
            .reset_index(drop=True)
        out = self.__class__(data_4.loc[:, self.coords_label],
                             data_4[column])
        return out

    def aggregate_numeric(self, column, grid):
        """
        Aggregation of numeric data in a regular grid.

        Parameters
        ----------
        column : str
            The name of the column with the data to work on.
        grid :
            matrix Grid*D object.

        Returns
        -------
        out :
            matrix point object with the aggregated data.

        This method is used to downsample large point clouds. Aggregation is
        done by taking the mean of the data values in each grid cell.
        """
        if self.ndim != grid.ndim:
            raise ValueError("grid dimension different from object's")

        # writing grid id
        if grid.ndim == 1:
            grid_id = _np.array(range(int(grid.grid_size[0])))
            cols = ["xid"]
        if grid.ndim == 2:
            grid_id = _np.array(
                [(x, y) for y in range(int(grid.grid_size[1]))
                        for x in range(int(grid.grid_size[0]))])
            cols = ["xid", "yid"]
        if grid.ndim == 3:
            grid_id = _np.array([(x, y, z)
                                for z in range(int(grid.grid_size[2]))
                                for y in range(int(grid.grid_size[1]))
                                for x in range(int(grid.grid_size[0]))])
            cols = ["xid", "yid", "zid"]
        grid_id = _pd.DataFrame(grid_id, columns=cols)

        # resetting grid
        grid = _copy.deepcopy(grid)
        grid.data = grid_id

        # identifying cell id
        # raw_data = self.as_data_frame()
        raw_data = _pd.DataFrame(self.data.loc[:, column])
        raw_data["xid"] = _np.round(
            (self.coords[:, 0] - grid.grid[0][0]
             - grid.step_size[0] / 2) / grid.step_size[0])
        if self.ndim >= 2:
            raw_data["yid"] = _np.round(
                (self.coords[:, 1] - grid.grid[1][0]
                 - grid.step_size[1] / 2) / grid.step_size[1])
        if self.ndim == 3:
            raw_data["zid"] = _np.round(
                (self.coords[:, 2] - grid.grid[2][0]
                 - grid.step_size[2] / 2) / grid.step_size[2])

        # mean of values inside cells
        grid_full = grid.as_data_frame()
        # raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols).mean()
        # data_2.reset_index(level=data_2.index.names, inplace=True)

        # output
        data_3 = data_2.join(grid_full.set_index(cols)) \
                       .reset_index(drop=True)
        out = self.__class__(data_3.loc[:, self.coords_label],
                             data_3[column])
        return out


class Points1D(_PointData):
    """
    Point data in 1D

    Attributes
    ----------
    coords : array
        matrix matrix with the spatial coordinates.
    data : data frame
        The coordinates' attributes.
    coords_label : list
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
            matrix vector with the spatial coordinates.
        data : data frame
            The coordinates' attributes (optional).
        coords_label : str
            Optional string to label the coordinates.
            Extracted from coords object if available.
        """
        super().__init__(coords, data)
        if isinstance(coords_label, str):
            coords_label = [coords_label]
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
            coords_label = ["X"]

        self._ndim = 1
        self.coords = _np.array(coords, ndmin=2, dtype=_np.float).transpose()
        if coords.shape[0] > 0:
            self.bounding_box = _np.array([[_np.min(coords)],
                                           [_np.max(coords)]])
            self.diagonal = _np.max(coords) - _np.min(coords)
        else:
            self.bounding_box = _np.zeros([2, 1])
            self.diagonal = 0
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

    def subset_region(self, xmin, xmax, include_min=True, include_max=False):
        df = self.as_data_frame()
        keep = (df[self.coords_label[0]] >= xmin) \
               & (df[self.coords_label[0]] <= xmax)
        if not include_min:
            keep = keep & (df[self.coords_label[0]] > xmin)
        if not include_max:
            keep = keep & (df[self.coords_label[0]] < xmax)
        df = df.loc[keep, :]
        df.reset_index(drop=True, inplace=True)
        coords = df[self.coords_label]
        data = df.drop(self.coords_label, axis=1)
        return Points1D(coords, data)

    def divide_by_region(self, n=1):
        break_points = _np.linspace(self.bounding_box[0],
                                    self.bounding_box[1],
                                    n + 1)
        divided = [self.subset_region(break_points[i], break_points[i+1],
                                      include_max=(i == n-1))
                   for i in range(n)]
        return divided

    def assign_region(self, n=1, prefix="region"):
        break_points = _np.linspace(self.bounding_box[0],
                                    self.bounding_box[1],
                                    n + 1)
        region = _np.zeros(self.coords.shape[0], dtype=_np.int32)

        for i in range(n):
            keep = (self.coords[:, 0] >= break_points[i]) \
                   & (self.coords[:, 0] <= break_points[i+1])
            if i < n-1:
                keep = keep & (self.coords[:, 0] < break_points[i+1])
            region[keep] = i

        if self.data is None:
            self.data = _pd.DataFrame(
                region, columns=[prefix + "_" + self.coords_label[0]])
        else:
            self.data[prefix + "_" + self.coords_label[0]] = region


class Points2D(_PointData):
    """
    Point data in 2D.

    Attributes
    ----------
    coords : array
        matrix matrix with the spatial coordinates.
    data : data frame
        The coordinates' attributes.
    coords_label : list
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
        coords_label : list
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

        self.coords = _np.array(coords, dtype=_np.float)
        if coords.shape[0] > 0:
            self.bounding_box, self.diagonal = _bounding_box(coords)
        else:
            self.bounding_box = _np.zeros([2, 2])
            self.diagonal = 0
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

    def subset_region(self, xmin, xmax, ymin, ymax,
                      include_min_x=True, include_max_x=False,
                      include_min_y=True, include_max_y=False):
        df = self.as_data_frame()
        keep = (df[self.coords_label[0]] >= xmin) \
               & (df[self.coords_label[0]] <= xmax) \
               & (df[self.coords_label[1]] >= ymin) \
               & (df[self.coords_label[1]] <= ymax)
        if not include_min_x:
            keep = keep & (df[self.coords_label[0]] > xmin)
        if not include_min_y:
            keep = keep & (df[self.coords_label[1]] > ymin)
        if not include_max_x:
            keep = keep & (df[self.coords_label[0]] < xmax)
        if not include_max_y:
            keep = keep & (df[self.coords_label[1]] < ymax)
        df = df.loc[keep, :]
        df.reset_index(drop=True, inplace=True)
        coords = df[self.coords_label]
        data = df.drop(self.coords_label, axis=1)
        return Points2D(coords, data)

    def divide_by_region(self, nx=1, ny=1):
        break_points_x = _np.linspace(self.bounding_box[0, 0],
                                      self.bounding_box[1, 0],
                                      nx + 1)
        break_points_y = _np.linspace(self.bounding_box[0, 1],
                                      self.bounding_box[1, 1],
                                      ny + 1)

        divided = [[
            self.subset_region(
                break_points_x[i], break_points_x[i + 1],
                break_points_y[j], break_points_y[j + 1],
                include_max_x=(i == nx - 1),
                include_max_y=(j == ny - 1)
            )
            for j in range(ny)]
            for i in range(nx)]
        return divided

    def assign_region(self, nx=1, ny=1, prefix="region"):
        break_points_x = _np.linspace(self.bounding_box[0, 0],
                                      self.bounding_box[1, 0],
                                      nx + 1)
        break_points_y = _np.linspace(self.bounding_box[0, 1],
                                      self.bounding_box[1, 1],
                                      ny + 1)

        region = _np.zeros([self.coords.shape[0], 2], dtype=_np.int32)

        for i in range(nx):
            keep = (self.coords[:, 0] >= break_points_x[i]) \
                   & (self.coords[:, 0] <= break_points_x[i + 1])
            if i < nx - 1:
                keep = keep & (self.coords[:, 0] < break_points_x[i + 1])
            region[keep, 0] = i
        for j in range(ny):
            keep = (self.coords[:, 1] >= break_points_y[j]) \
                   & (self.coords[:, 1] <= break_points_y[j + 1])
            if j < ny - 1:
                keep = keep & (self.coords[:, 1] < break_points_y[j + 1])
            region[keep, 1] = j

        if self.data is None:
            self.data = _pd.DataFrame(
                region,
                columns=[prefix + "_" + lab for lab in self.coords_label])
        else:
            for d in range(self.ndim):
                self.data[prefix + "_" + self.coords_label[d]] = region[:, d]


class Points3D(_PointData):
    """
    Point data in 3D.

    Attributes
    ----------
    coords : array
        matrix matrix with the spatial coordinates.
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

        self.coords = _np.array(coords, dtype=_np.float)
        if coords.shape[0] > 0:
            self.bounding_box, self.diagonal = _bounding_box(coords)
        else:
            self.bounding_box = _np.zeros([2, 3])
            self.diagonal = 0
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
        # x, y, z, and text as lists
        values = self.data[column].tolist()
        text = [str(num) for num in values]
        x = self.coords[:, 0].tolist()
        y = self.coords[:, 1].tolist()
        z = self.coords[:, 2].tolist()

        obj = {"mode": "markers",
               "name": column,
               "text": text,
               "marker": {"color": values}}

        obj.update({"type": "scatter3d",
                    "x": x,
                    "y": y,
                    "z": z,
                    "hoverinfo": "x+y+z+text"})
        obj = _update(obj, kwargs)
        return [obj]

    def draw_categorical(self, column, colors, **kwargs):
        """
        Visualization of categorical data.

        Parameters
        ----------
        column : str
            Name of the categorical variable to plot.
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
        colors2 = self.data[column].map(colors_dict).tolist()
        colors3 = [colors.get(item) for item in colors]
        cmax = len(color_val) - 1

        text = self.data[column].tolist()
        x = self.coords[:, 0].tolist()
        y = self.coords[:, 1].tolist()
        z = self.coords[:, 2].tolist()

        obj = {"mode": "markers",
               "name": column,
               "text": text,
               "marker": {"color": colors2,
                          "colorscale": [[i / cmax, colors3[i]]
                                         for i in color_val]}}

        obj.update({"type": "scatter3d",
                    "x": x,
                    "y": y,
                    "z": z,
                    "hoverinfo": "x+y+z+text"})
        obj = _update(obj, kwargs)
        return [obj]

    def subset_region(self, xmin, xmax, ymin, ymax, zmin, zmax,
                      include_min_x=True, include_max_x=False,
                      include_min_y=True, include_max_y=False,
                      include_min_z=True, include_max_z=False
                      ):
        df = self.as_data_frame()
        keep = (df[self.coords_label[0]] >= xmin) \
               & (df[self.coords_label[0]] <= xmax) \
               & (df[self.coords_label[1]] >= ymin) \
               & (df[self.coords_label[1]] <= ymax) \
               & (df[self.coords_label[2]] >= zmin) \
               & (df[self.coords_label[2]] <= zmax)
        if not include_min_x:
            keep = keep & (df[self.coords_label[0]] > xmin)
        if not include_min_y:
            keep = keep & (df[self.coords_label[1]] > ymin)
        if not include_min_z:
            keep = keep & (df[self.coords_label[2]] > zmin)
        if not include_max_x:
            keep = keep & (df[self.coords_label[0]] < xmax)
        if not include_max_y:
            keep = keep & (df[self.coords_label[1]] < ymax)
        if not include_max_z:
            keep = keep & (df[self.coords_label[2]] < zmax)
        df = df.loc[keep, :]
        df.reset_index(drop=True, inplace=True)
        coords = df[self.coords_label]
        data = df.drop(self.coords_label, axis=1)
        return Points3D(coords, data)

    def divide_by_region(self, nx=1, ny=1, nz=1):
        break_points_x = _np.linspace(self.bounding_box[0, 0],
                                      self.bounding_box[1, 0],
                                      nx + 1)
        break_points_y = _np.linspace(self.bounding_box[0, 1],
                                      self.bounding_box[1, 1],
                                      ny + 1)
        break_points_z = _np.linspace(self.bounding_box[0, 2],
                                      self.bounding_box[1, 2],
                                      nz + 1)

        divided = [[[
            self.subset_region(
                break_points_x[i], break_points_x[i + 1],
                break_points_y[j], break_points_y[j + 1],
                break_points_z[k], break_points_z[k + 1],
                include_max_x=(i == nx - 1),
                include_max_y=(j == ny - 1),
                include_max_z=(k == nz - 1)
            )
            for k in range(nz)]
            for j in range(ny)]
            for i in range(nx)]
        return divided

    def assign_region(self, nx=1, ny=1, nz=1, prefix="region"):
        break_points_x = _np.linspace(self.bounding_box[0, 0],
                                      self.bounding_box[1, 0],
                                      nx + 1)
        break_points_y = _np.linspace(self.bounding_box[0, 1],
                                      self.bounding_box[1, 1],
                                      ny + 1)
        break_points_z = _np.linspace(self.bounding_box[0, 2],
                                      self.bounding_box[1, 2],
                                      nz + 1)

        region = _np.zeros([self.coords.shape[0], 3], dtype=_np.int32)

        for i in range(nx):
            keep = (self.coords[:, 0] >= break_points_x[i]) \
                   & (self.coords[:, 0] <= break_points_x[i + 1])
            if i < nx - 1:
                keep = keep & (self.coords[:, 0] < break_points_x[i + 1])
            region[keep, 0] = i
        for j in range(ny):
            keep = (self.coords[:, 1] >= break_points_y[j]) \
                   & (self.coords[:, 1] <= break_points_y[j + 1])
            if j < ny - 1:
                keep = keep & (self.coords[:, 1] < break_points_y[j + 1])
            region[keep, 1] = j
        for k in range(nz):
            keep = (self.coords[:, 2] >= break_points_z[k]) \
                   & (self.coords[:, 2] <= break_points_z[k + 1])
            if k < nz - 1:
                keep = keep & (self.coords[:, 2] < break_points_z[k + 1])
            region[keep, 2] = k

        if self.data is None:
            self.data = _pd.DataFrame(
                region,
                columns=[prefix + "_" + lab for lab in self.coords_label])
        else:
            for d in range(self.ndim):
                self.data[prefix + "_" + self.coords_label[d]] = region[:, d]

    def draw_planes(self, dip, azimuth, size=1, **kwargs):
        dip = self.data[dip]
        azimuth = self.data[azimuth]

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
        for i, point in enumerate(self.coords):
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
            obj = _update(obj, kwargs)
            surf.append(obj)
        return surf

    def export_planes(self, dip, azimuth, filename, size=1):
        dip = self.data[dip]
        azimuth = self.data[azimuth]

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

        points = _np.stack([
            self.coords + dipvec,
            self.coords - 0.5*strvec - 0.5*dipvec,
            self.coords + 0.5*strvec - 0.5*dipvec
        ], axis=0)
        points = _np.reshape(points, [3*points.shape[1], points.shape[2]],
                             order="F")
        idx = _np.reshape(_np.arange(points.shape[0]), self.coords.shape)

        # export
        with open(filename, 'w') as out_file:
            out_file.write(
                str(points.shape[0]) + " " + str(idx.shape[0]) + "\n")
            for line in points:
                out_file.write(" ".join(str(elem) for elem in line) + "\n")
            for line in idx:
                out_file.write(" ".join(str(elem) for elem in line) + "\n")


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
        else:
            step = (end - start) / (n - 1)
        grid = _np.linspace(start, end, n, dtype=_np.float)

        super().__init__(grid)
        self.step_size = [step]
        self.grid = [grid]
        self.grid_size = [int(n)]


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
            step = _np.array([(end[0]-start[0])/(n[0]-1),
                              (end[1]-start[1])/(n[1]-1)])
        grid_x = _np.linspace(start[0], end[0], n[0])
        grid_y = _np.linspace(start[1], end[1], n[1])
        coords = _np.array([(x, y) for y in grid_y for x in grid_x],
                           dtype=_np.float)

        super().__init__(coords)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y]
        # self.grid_size = n.tolist()
        self.grid_size = [int(num) for num in n]

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
            matrix 2-dimensional array.
        """
        image = _np.reshape(self.data[column].values,
                            newshape=self.grid_size,
                            order="F")
        image = image.transpose()
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
            step = _np.array([(end[0] - start[0]) / (n[0] - 1),
                              (end[1] - start[1]) / (n[1] - 1),
                              (end[2] - start[2]) / (n[2] - 1)])

        grid_x = _np.linspace(start[0], end[0], n[0])
        grid_y = _np.linspace(start[1], end[1], n[1])
        grid_z = _np.linspace(start[2], end[2], n[2])
        coords = _np.array([(x, y, z) for z in grid_z for y in grid_y
                            for x in grid_x], dtype=_np.float)

        super().__init__(coords)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y, grid_z]
        # self.grid_size = n.tolist()
        self.grid_size = [int(num) for num in n]

    def as_cube(self, column):
        """
        Returns a rank-3 array filled with the specified variable.

        Parameters
        ----------
        column : str
            The column with the variable to use.

        Returns
        -------
        cube : array
            matrix cubic array.
        """
        cube = _np.reshape(self.data[column].values,
                           self.grid_size, order="F")
        return cube

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


        This method calls skimage.measure.marching_cubes_lewiner().
        See the original documentation for details.
        """
        cube = self.as_cube(column)
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
                    "k": faces[:, 2],
                    "text": column})
        obj = _update(obj, kwargs)
        return [obj]

    def export_contour(self, column, value, filename, triangles=True):
        verts, faces, normals, values = self.get_contour(column, value)
        for i in range(3):
            verts[:, i] += self.grid[i][0]
        with open(filename, 'w') as out_file:
            if triangles:
                out_file.write(
                    str(verts.shape[0]) + " " + str(faces.shape[0]) + "\n")
            for line in verts:
                out_file.write(" ".join(str(elem) for elem in line) + "\n")
            if triangles:
                for line in faces:
                    out_file.write(" ".join(str(elem) for elem in line) + "\n")

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


class Directions2D(Points2D):
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
            matrix data frame with additional attributes.
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
        matrix Directions2D object.
        """
        directions = _pd.DataFrame({"dX": _np.sin(azimuth / 180 * _np.pi),
                                    "dY": _np.cos(azimuth / 180 * _np.pi)})
        return Directions2D(coords, directions, data)

    def draw_arrows(self, size=1):
        raise NotImplementedError()

    def subset_region(self, xmin, xmax, ymin, ymax,
                      include_min_x=True, include_max_x=False,
                      include_min_y=True, include_max_y=False):
        df = self.as_data_frame()
        keep = (df[self.coords_label[0]] >= xmin) \
               & (df[self.coords_label[0]] <= xmax) \
               & (df[self.coords_label[1]] >= ymin) \
               & (df[self.coords_label[1]] <= ymax)
        if not include_min_x:
            keep = keep & (df[self.coords_label[0]] > xmin)
        if not include_min_y:
            keep = keep & (df[self.coords_label[1]] > ymin)
        if not include_max_x:
            keep = keep & (df[self.coords_label[0]] < xmax)
        if not include_max_y:
            keep = keep & (df[self.coords_label[1]] < ymax)
        df = df.loc[keep, :]
        df.reset_index(drop=True, inplace=True)
        coords = df[self.coords_label]
        directions = df[self.directions_label]
        data = df.drop(list(self.coords_label) + list(self.directions_label),
                       axis=1)
        return Directions2D(coords, directions, data)

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


class Directions3D(Points3D):
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
            matrix data frame with additional attributes.
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

    def draw_arrows(self, size=1, **kwargs):
        head = self.coords + size / 2 * self.directions
        tail = self.coords - size / 2 * self.directions

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

        obj = _update(obj, kwargs)
        return [obj]

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
            matrix data frame with additional data.
        
        Returns
        -------
        matrix Directions3D object.
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
            matrix data frame with additional data.
        
        Returns
        -------
        matrix Directions3D object.
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
            matrix data frame with additional data.
        
        Returns
        -------
        matrix Directions3D object.
        """
        n_data = coords.shape[0]
        plane_dirs = cls.from_planes(coords, azimuth, dip)
        vec1 = plane_dirs.directions[0:n_data, :]
        vec2 = plane_dirs.directions[n_data:(2 * n_data), :]

        normalvec = vec1[:, [1, 2, 0]] * vec2[:, [2, 0, 1]] \
                    - vec1[:, [2, 0, 1]] * vec2[:, [1, 2, 0]]
        normalvec = _np.apply_along_axis(lambda x: x/_np.sqrt(_np.sum(x ** 2)),
                                         axis=1,
                                         arr=normalvec)

        # result
        return cls(coords, normalvec, data)

    def subset_region(self, xmin, xmax, ymin, ymax, zmin, zmax,
                      include_min_x=True, include_max_x=False,
                      include_min_y=True, include_max_y=False,
                      include_min_z=True, include_max_z=False
                      ):
        df = self.as_data_frame()
        keep = (df[self.coords_label[0]] >= xmin) \
               & (df[self.coords_label[0]] <= xmax) \
               & (df[self.coords_label[1]] >= ymin) \
               & (df[self.coords_label[1]] <= ymax) \
               & (df[self.coords_label[2]] >= zmin) \
               & (df[self.coords_label[2]] <= zmax)
        if not include_min_x:
            keep = keep & (df[self.coords_label[0]] > xmin)
        if not include_min_y:
            keep = keep & (df[self.coords_label[1]] > ymin)
        if not include_min_z:
            keep = keep & (df[self.coords_label[2]] > zmin)
        if not include_max_x:
            keep = keep & (df[self.coords_label[0]] < xmax)
        if not include_max_y:
            keep = keep & (df[self.coords_label[1]] < ymax)
        if not include_max_z:
            keep = keep & (df[self.coords_label[2]] < zmax)
        df = df.loc[keep, :]
        df.reset_index(drop=True, inplace=True)
        coords = df[self.coords_label]
        directions = df[self.directions_label]
        data = df.drop(list(self.coords_label) + list(self.directions_label),
                       axis=1)
        return Directions3D(coords, directions, data)

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

    def __init__(self, collar=None, assay=None, survey=None, holeid="HOLEID",
                 x="X", y="Y", z="Z", fr="FROM", to="TO", **kwargs):
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
        super().__init__()
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
        if (_np.min(locations) < 0) | (_np.max(locations) > 1):
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
        # condition 4 - same hole
        hole_from = self.data.loc[0:(self.coords_from.shape[0] - 1), "HOLEID"]\
                        .values
        hole_to = self.data.loc[1:self.coords_from.shape[0], "HOLEID"].values
        same_hole = [hole_from[i] == hole_to[i] for i in range(len(hole_to))]
        # final vector
        mergeable = start_end & parallel & same_value & same_hole

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
        all_cols = []
        if x.data is not None:
            all_cols += x.data.columns.values.tolist()
        if y.data is not None:
            all_cols += y.data.columns.values.tolist()
        all_cols = _pd.unique(all_cols)

        df_x = _pd.DataFrame(columns=all_cols)
        df_y = _pd.DataFrame(columns=all_cols)
        if x.data is not None:
            for col in x.data.columns:
                df_x[col] = x.data[col]
        if y.data is not None:
            for col in y.data.columns:
                df_y[col] = y.data[col]
        new_data = _pd.concat([df_x, df_y]).reset_index(drop=True)

        if x.ndim == 1:
            return Points1D(new_coords, new_data)
        if x.ndim == 2:
            return Points2D(new_coords, new_data)
        if x.ndim == 3:
            return Points3D(new_coords, new_data)
    else:
        raise ValueError("x and y are not compatible for merging")


class PointsND(object):
    """General, high-dimensional data"""

    def __init__(self, data, columns_x, columns_y=None):
        self._ndim = len(columns_x)
        self.coords_label = columns_x
        self.coords = data.loc[:, columns_x].values
        self.bounding_box, self.diagonal = _bounding_box(self.coords)
        if columns_y is not None:
            self.data = _pd.DataFrame(data.loc[:, columns_y])
        else:
            self.data = _pd.DataFrame()

    def __repr__(self):
        return self.__str__()

    @property
    def ndim(self):
        return self._ndim

    def as_data_frame(self):
        """
        Conversion to a data frame.
        """
        df = _pd.DataFrame(self.coords,
                           columns=self.coords_label)
        df = _pd.concat([df, self.data], axis=1)
        return df

    def __str__(self):
        s = "Object of class " + self.__class__.__name__ + " with " \
            + str(len(self.coords)) + " data points\n\n"

        if self.data is not None:
            s += "Data preview:\n\n"
            s += str(self.data.head())
        return s


class CirculantGrid1D(Grid1D):
    """
    Grid with circulant structure.
    """
    def __init__(self, grid, expand=0.2):
        n_points = float(grid.grid_size[0]) * (1 + expand)
        n_points = int(_np.ceil(n_points))
        if n_points % 2 == 0:
            n_points += 1
        start = grid.grid[0][0]
        step = grid.step_size[0]
        super().__init__(start, n_points, step)
        self.point_zero = _np.array([[self.grid[0][n_points // 2]]])
