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
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["PointData",
           "Grid1D", "Grid2D", "Grid3D",
           "DirectionalData",
           "DrillholeData",
           "batch_index",
           "export_planes"]

import pandas as pd
import tensorflow as _tf
import pandas as _pd
import numpy as _np
import copy as _copy
import collections as _col
import pyvista as _pv
import itertools as _iter

from skimage import measure as _measure

import geoml.interpolation as _gint
import geoml.plotly as _py


def bounding_box(points):
    """
    Computes a point set's bounding box and its diagonal.

    Parameters
    ----------
    points : array
        A set of coordinates.

    Returns
    -------
    bbox : array-like
        Array with the box's minimum and maximum values in each direction.
    d : float
        The box's diagonal length.
    """
    if len(points.shape) < 2:
        points = _np.expand_dims(points, axis=0)
    bbox = _np.array([[_np.min(points[:, i]) for i in range(points.shape[1])],
                      [_np.max(points[:, i]) for i in range(points.shape[1])]])
    d = _np.sqrt(sum([
        _np.diff(bbox[:, i]) ** 2 for i in range(bbox.shape[1])]))
    d = _np.squeeze(d)
    return bbox, d


class BoundingBox(object):
    """
    An n-dimensional box.
    """
    def __init__(self, min_values, max_values):
        """
        An n-dimensional box.

        Parameters
        ----------
        min_values : array
            The box's minimum values in each direction.
        max_values : array
            The box's maximum values in each direction.
        """
        min_values = _np.array(min_values, ndmin=2)
        max_values = _np.array(max_values, ndmin=2)

        if min_values.shape != max_values.shape:
            raise ValueError("min_values and max_values must have the"
                             "same size")

        self._n_dim = min_values.shape[1]
        self._min = min_values
        self._max = max_values
        self._diagonal = _np.sqrt(_np.sum((max_values - min_values)**2))

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def diagonal(self):
        return self._diagonal

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def as_array(self):
        return _np.concatenate([self.min, self.max], axis=0)

    def __repr__(self):
        return self.as_array().__repr__()

    def overlaps_with(self, other):
        """
        Checks if box overlaps with another box.

        Parameters
        ----------
        other : BoundingBox
            The other box.

        Returns
        -------
        check : bool
            The checking result.
        """
        if other.n_dim != self.n_dim:
            raise ValueError("box dimensions mismatch")

        checks = []
        for min_A, max_A, min_B, max_B in zip(
                self.min[0], self.max[0], other.min[0], other.max[0]):
            checks.append(min_A > max_B)
            checks.append(min_B > max_A)
        return not any(checks)

    @classmethod
    def from_array(cls, array):
        """
        Builds bounding box from an array with minimum and maximum coordinates.

        Parameters
        ----------
        array : array
            Array with the minimum and maximum coordinates.

        Returns
        -------
        box : BoundingBox
            A box object.
        """
        if len(array.shape) < 2:
            array = _np.expand_dims(array, axis=0)
        min_values = _np.min(array, axis=0, keepdims=True)
        max_values = _np.max(array, axis=0, keepdims=True)
        return BoundingBox(min_values, max_values)

    def contains_points(self, array):
        """
        Checks if points are contained within the box.

        Parameters
        ----------
        array : array
            A set of coordinates.

        Returns
        -------
        check : bool
            True if all points are contained within the box.
        """
        if array.shape[1] != self.n_dim:
            raise ValueError("box and points dimensions mismatch")

        check_1 = array >= self.min
        check_2 = array <= self.max

        contains = _np.all(_np.concatenate([check_1, check_2], axis=1), axis=1)
        return contains


class _Variable(object):
    """Representation of a dependent random variable."""

    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        self._length = 1

    @property
    def length(self):
        return self._length

    @classmethod
    def from_variable(cls, coordinates, variable):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def as_data_frame(self, **kwargs):
        raise NotImplementedError

    def get_measurements(self):
        raise NotImplementedError

    def prediction_input(self):
        return {}

    def training_input(self, idx=None):
        return {}

    def copy_to(self, coordinates):
        coordinates.variables[self.name] = self.__class__.from_variable(
            coordinates, self
        )

    def update(self, idx, **kwargs):
        raise NotImplementedError

    def allocate_simulations(self, n_sim):
        raise NotImplementedError

    def set_coordinates(self, coordinates):
        raise NotImplementedError

    def fill_pyvista_cube(self, cube, prefix=None):
        raise NotImplementedError

    def fill_pyvista_points(self, points, prefix=None):
        raise NotImplementedError

    class _Attribute(object):
        """A specific sequence of variable values, tied to data locations."""

        def __init__(self, coordinates, values=None, dtype=None):
            self.coordinates = coordinates

            if dtype is None:
                dtype = float

            if values is None:
                values = _np.array([_np.nan] * coordinates.n_data,
                                   dtype=dtype)
            values = _np.array(values, ndmin=1, dtype=dtype)
            if len(values.shape) > 1:
                values = _np.squeeze(values)

            if len(values.shape) != 1:
                raise ValueError("Values must be 1-dimensional")

            if len(values) != coordinates.n_data:
                raise ValueError("Values and coordinates size mismatch")

            self.values = values

        def __str__(self):
            return self.values.__str__()

        def __repr__(self):
            return self.values.__repr__()

        def __getitem__(self, item):
            new_obj = _copy.deepcopy(self)
            new_obj.values = _np.array(self.values[item], ndmin=1)
            return new_obj

        def as_image(self):
            """
            Reshapes the data in the form of a matrix for plotting.

            The output can be used in plotting functions such as
            matplotlib's `imshow()`. If you use it, do not forget to set
            `origin="lower"`.

            Returns
            -------
            image : _np.array
                A 2-dimensional array.
            """
            if not isinstance(self.coordinates, Grid2D):
                raise ValueError("method only available for Grid2D data"
                                 "objects")

            image = _np.reshape(self.values, #.astype(float),
                                newshape=self.coordinates.grid_size,
                                order="F")
            image = image.transpose()
            return image

        def as_cube(self):
            """
            Returns a rank-3 array filled with the specified variable.

            Returns
            -------
            cube : _np.array
                A rank-3 array.
            """
            if not isinstance(self.coordinates, Grid3D):
                raise ValueError("method only available for Grid3D data"
                                 "objects")

            cube = _np.reshape(self.values,  # .astype(float),
                               self.coordinates.grid_size, order="F")
            return cube

        def get_contour(self, value):
            """
            Isosurface extraction.

            This method calls `skimage.measure.marching_cubes()`.
            See the original documentation for details.

            Parameters
            ----------
            value : double
                The value on which to calculate the isosurface.

            Returns
            -------
            surf : Surface3D
                A `Surface3D` object.
            """
            cube = self.as_cube()
            verts, faces, normals, values = _measure.marching_cubes(
                cube, value, gradient_direction="ascent",
                allow_degenerate=False, spacing=self.coordinates.step_size)
            for i in range(3):
                verts[:, i] += self.coordinates.grid[i][0]
            # return verts, faces, normals, values
            return Surface3D(verts, faces, normals)

        def export_contour(self, value, filename, triangles=True,
                           offset=None):
            verts, faces, normals, values = self.get_contour(value)

            if offset is None:
                offset = _np.zeros([1, 3])
            else:
                offset = _np.array(_np.squeeze(offset))[None, :]
            verts = verts + offset
            
            with open(filename, 'w') as out_file:
                if triangles:
                    out_file.write(
                        str(verts.shape[0]) + " " + str(faces.shape[0]) + "\n")
                for line in verts:
                    out_file.write(" ".join(str(elem) for elem in line) + "\n")
                if triangles:
                    for line in faces:
                        out_file.write(
                            " ".join(str(elem) for elem in line) + "\n")

        def fill_pyvista_cube(self, cube, label):
            if self.values.dtype == object:
                if not all(self.values == ""):
                    cube.point_arrays[label] = self.as_cube() \
                        .transpose([2, 0, 1]).ravel()
            elif not all(_np.isnan(self.values)):
                cube.point_arrays[label] = self.as_cube() \
                    .transpose([2, 0, 1]).ravel()

        def fill_pyvista_points(self, points, label):
            if self.values.dtype == object:
                if not all(self.values == ""):
                    points.point_arrays[label] = self.values
            elif not all(_np.isnan(self.values)):
                points.point_arrays[label] = self.values

        def draw_contour(self, value, **kwargs):
            """Creates plotly object with the contour at the specified value."""
            surf_obj = self.get_contour(value)
            return _py.isosurface(
                surf_obj.coordinates, surf_obj.triangles, **kwargs)

        def draw_numeric(self, **kwargs):
            if self.coordinates.n_dim != 3:
                raise NotImplemented("method currently available only for"
                                     "3D coordinates")

            if isinstance(self.coordinates, Section3D):
                values = _np.reshape(self.values, self.coordinates.grid_shape,
                                     order="F")
                gridded_x = _np.reshape(self.coordinates.coordinates[:, 0],
                                        self.coordinates.grid_shape,
                                        order="F")
                gridded_y = _np.reshape(self.coordinates.coordinates[:, 1],
                                        self.coordinates.grid_shape,
                                        order="F")
                gridded_z = _np.reshape(self.coordinates.coordinates[:, 2],
                                        self.coordinates.grid_shape,
                                        order="F")

                return _py.numeric_section_3d(gridded_x, gridded_y, gridded_z,
                                              values, **kwargs)

            if isinstance(self.coordinates, Surface3D):
                return _py.isosurface(self.coordinates.coordinates,
                                      self.coordinates.triangles,
                                      values=self.values)

            return _py.numeric_points_3d(
                self.coordinates.coordinates,
                self.values,
                **kwargs)

        def draw_categorical(self, colors, **kwargs):
            if self.coordinates.n_dim != 3:
                raise NotImplemented("method currently available only for"
                                     "3D coordinates")

            return _py.categorical_points_3d(
                self.coordinates.coordinates,
                self.values,
                colors,
                **kwargs)


class ContinuousVariable(_Variable):
    """
    Representation of a continuous random variable.

    Attributes
    ----------
    measurements : _Attribute
        The raw measurements.
    latent_mean : _Attribute
        The mean of the latent Gaussian representation.
    latent_variance : _Attribute
        The variance of the latent Gaussian representation.
    simulations : list
        Draws from the variable's posterior distribution.
    quantiles : dict
        The variables quantiles, indexed by the corresponding percentile.
    probabilities : dict
        Cumulative distribution probabilities, indexed by the corresponding
        quantile.
    """
    def __init__(self, name, coordinates, measurements=None,
                 quantiles=None, probabilities=(0.025, 0.5, 0.975)):
        super().__init__(name, coordinates)

        if measurements is None:
            self.measurements = self._Attribute(coordinates)
        else:
            self.measurements = self._Attribute(coordinates, measurements)

        self.latent_mean = self._Attribute(coordinates)
        self.latent_variance = self._Attribute(coordinates)

        self.simulations = []

        self.quantiles = _col.OrderedDict()
        self.reset_quantiles(probabilities)

        self.probabilities = _col.OrderedDict()
        self.reset_probabilities(quantiles)

    def get_measurements(self):
        values = self.measurements.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    def reset_quantiles(self, probabilities=None):
        """
        Resets the variable's quantiles.

        Parameters
        ----------
        probabilities : array
            An array of probabilities, ordered values from 0 to 1 (exclusive),
            on which the models will compute the corresponding quantiles.
        """
        self.quantiles = _col.OrderedDict()
        if probabilities is not None:
            for p in probabilities:
                self.quantiles[p] = self._Attribute(self.coordinates)

    def reset_probabilities(self, quantiles=None):
        """
        Resets the variable's probabilities.

        Parameters
        ----------
        quantiles : array
            An array of quantiles, ordered values on which the models will
            compute the corresponding cumulative probabilities.
        """
        self.probabilities = _col.OrderedDict()
        if quantiles is not None:
            for q in quantiles:
                self.probabilities[q] = self._Attribute(self.coordinates)

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = ContinuousVariable(variable.name, coordinates,
                                     quantiles=None, probabilities=None)
        if len(variable.quantiles) > 0:
            new_var.reset_quantiles(variable.quantiles.keys())
        if len(variable.probabilities) > 0:
            new_var.reset_probabilities(variable.probabilities.keys())

        return new_var

    def __getitem__(self, item):
        # new_obj = _copy.deepcopy(self)
        self.measurements = self.measurements[item]
        self.latent_mean = self.latent_mean[item]
        self.latent_variance = self.latent_variance[item]

        for i, sim in enumerate(self.simulations):
            self.simulations[i] = sim[item]

        if len(self.quantiles) > 0:
            for key, val in self.quantiles.items():
                self.quantiles[key] = val[item]

        if len(self.probabilities) > 0:
            for key, val in self.probabilities.items():
                self.probabilities[key] = val[item]

        return self

    def as_data_frame(self, measurements=True, latent=True,
                      simulations=True, quantiles=True,
                      probabilities=True, **kwargs):
        """
        Converts the object to a DataFrame.

        Parameters
        ----------
        measurements : bool
            Whether to include the measurements.
        latent : bool
            Whether to include the latent Gaussian variable.
        simulations : bool
            Whether to include the latent Gaussian variable.
        quantiles : bool
            Whether to include the quantiles.
        probabilities : bool
            Whether to include the probabilities.
        kwargs : dict
            Ignored.

        Returns
        -------
        df : pd.DataFrame
            The converted object.
        """
        df = _pd.DataFrame({})

        if measurements:
            df[self.name] = self.measurements.values

        if latent:
            df[self.name + "_latent_mean"] = self.latent_mean.values
            df[self.name + "_latent_variance"] = self.latent_variance.values

        if simulations:
            for i, sim in enumerate(self.simulations):
                df[self.name + "_sim_" + str(i)] = sim.values

        if quantiles:
            if len(self.quantiles) > 0:
                for key, val in self.quantiles.items():
                    df[self.name + "_q" + str(key)] = val.values

        if probabilities:
            if len(self.probabilities) > 0:
                for key, val in self.probabilities.items():
                    df[self.name + "_p" + str(key)] = val.values

        return df

    def prediction_input(self):
        d = {"quantiles": None, "probabilities": None}
        if len(self.quantiles) > 0:
            d["probabilities"] = _tf.constant(
                [k for k in self.quantiles.keys()], _tf.float64)
        if len(self.probabilities) > 0:
            d["quantiles"] = _tf.constant(
                [k for k in self.probabilities.keys()], _tf.float64)

        return d

    def update(self, idx, **kwargs):
        self.latent_mean.values[idx] = kwargs["mean"].numpy()
        self.latent_variance.values[idx] = kwargs["variance"].numpy()

        if "simulations" in kwargs.keys():
            sims = kwargs["simulations"].numpy()
            for s in range(sims.shape[1]):
                self.simulations[s].values[idx] = sims[:, s]

        if "quantiles" in kwargs.keys():
            quant = kwargs["quantiles"].numpy()
            prob_keys = self.quantiles.keys()
            for i, p in zip(range(quant.shape[1]), prob_keys):
                self.quantiles[p].values[idx] = quant[:, i]

        if "probabilities" in kwargs.keys():
            prob = kwargs["probabilities"].numpy()
            quant_keys = self.probabilities.keys()
            for i, q in zip(range(prob.shape[1]), quant_keys):
                self.probabilities[q].values[idx] = prob[:, i]

    def allocate_simulations(self, n_sim):
        self.simulations = [self._Attribute(self.coordinates)
                            for _ in range(n_sim)]

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.measurements.coordinates = coordinates
        self.latent_mean.coordinates = coordinates
        self.latent_variance.coordinates = coordinates

        for sim in self.simulations:
            sim.coordinates = coordinates

        if len(self.quantiles) > 0:
            for p in self.quantiles.values():
                p.coordinates = coordinates

        if len(self.probabilities) > 0:
            for q in self.probabilities.values():
                q.coordinates = coordinates

    def fill_pyvista_cube(self, cube, prefix=None):
        self.measurements.fill_pyvista_cube(cube, self.name)
        self.latent_mean.fill_pyvista_cube(
            cube, self.name + " - latent mean")
        self.latent_variance.fill_pyvista_cube(
            cube, self.name + " - latent variance")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_cube(
                cube, self.name + " - simulation %d" % i)

        for p in self.quantiles.keys():
            self.quantiles[p].fill_pyvista_cube(
                cube, self.name + " - quantile %f" % p)

        for q in self.probabilities.keys():
            self.probabilities[q].fill_pyvista_cube(
                cube, self.name + " - probability %f" % q)

    def fill_pyvista_points(self, points, prefix=None):
        self.measurements.fill_pyvista_points(points, self.name)
        self.latent_mean.fill_pyvista_points(
            points, self.name + " - latent mean")
        self.latent_variance.fill_pyvista_points(
            points, self.name + " - latent variance")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_points(
                points, self.name + " - simulation %d" % i)

        for p in self.quantiles.keys():
            self.quantiles[p].fill_pyvista_points(
                points, self.name + " - quantile %f" % p)

        for q in self.probabilities.keys():
            self.probabilities[q].fill_pyvista_points(
                points, self.name + " - probability %f" % q)


class _Component(_Variable):
    def __init__(self, name, coordinates, measurements=None):
        super().__init__(name, coordinates)
        n_data = coordinates.n_data

        if measurements is None:
            self.measurements = self._Attribute(coordinates)
        else:
            self.measurements = self._Attribute(coordinates, measurements)

        self.probability = self._Attribute(coordinates, _np.zeros(n_data))
        self.indicator_mean = self._Attribute(coordinates)
        self.indicator_variance = self._Attribute(coordinates)
        self.indicator_predicted = self._Attribute(coordinates)
        self.simulations = []

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        new_obj.probability = self.probability[item]
        new_obj.measurements = self.measurements[item]
        new_obj.indicator_mean = self.indicator_mean[item]
        new_obj.indicator_variance = self.indicator_variance[item]
        new_obj.indicator_predicted = self.indicator_predicted[item]

        for i, sim in enumerate(self.simulations):
            new_obj.simulations[i] = sim[item]

        return new_obj

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.probability.coordinates = coordinates
        self.indicator_mean.coordinates = coordinates
        self.indicator_variance.coordinates = coordinates
        self.indicator_predicted.coordinates = coordinates

        for sim in self.simulations:
            sim.coordinates = coordinates

    def as_data_frame(self, probability=True, predictions=True,
                      simulations=False):
        df = _pd.DataFrame({})

        if probability:
            df[self.name + "_probability"] = self.probability.values
            df[self.name + "_indicator"] = self.measurements.values

        if predictions:
            df[self.name + "_indicator_mean"] = self.indicator_mean.values
            df[self.name + "_indicator_variance"] = \
                self.indicator_variance.values
            df[self.name + "_indicator_predicted"] = \
                self.indicator_predicted.values

        if simulations:
            for i, sim in enumerate(self.simulations):
                df[self.name + "_sim_" + str(i)] = sim.values

        return df

    def update(self, idx, **kwargs):
        self.indicator_mean.values[idx] = kwargs["mean"].numpy()
        self.indicator_variance.values[idx] = kwargs["variance"].numpy()
        self.probability.values[idx] = kwargs["probability"].numpy()

        if "indicator" in kwargs:
            self.indicator_predicted.values[idx] = kwargs["indicator"].numpy()

        sims = kwargs["simulations"].numpy()
        for s in range(sims.shape[1]):
            self.simulations[s].values[idx] = sims[:, s]

    def allocate_simulations(self, n_sim):
        self.simulations = [self._Attribute(self.coordinates)
                            for _ in range(n_sim)]

    def fill_pyvista_cube(self, cube, prefix=None):
        label = prefix + " - " + self.name

        self.measurements.fill_pyvista_cube(cube, label)
        self.indicator_mean.fill_pyvista_cube(
            cube, label + " - indicator mean")
        self.indicator_variance.fill_pyvista_cube(
            cube, label + " - indicator variance")
        self.indicator_predicted.fill_pyvista_cube(
            cube, label + " - indicator predicted")
        self.probability.fill_pyvista_cube(
            cube, label + " - probability")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_cube(
                cube, label + " - simulation %d" % i)

    def fill_pyvista_points(self, points, prefix=None):
        label = prefix + " - " + self.name

        self.measurements.fill_pyvista_points(points, label)
        self.indicator_mean.fill_pyvista_points(
            points, label + " - indicator mean")
        self.indicator_variance.fill_pyvista_points(
            points, label + " - indicator variance")
        self.indicator_predicted.fill_pyvista_points(
            points, label + " - indicator predicted")
        self.probability.fill_pyvista_points(
            points, label + " - probability")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_points(
                points, label + " - simulation %d" % i)


class CompositionalVariable(_Variable):
    def __init__(self, name, coordinates, labels, measurements=None):
        super().__init__(name, coordinates)

        self.labels = labels
        self._length = len(labels)

        self.components = {}
        for i, label in enumerate(labels):
            self.components[label] = _Component(
                label,
                coordinates,
                measurements[:, i] if measurements is not None else None)

    def get_measurements(self):
        out = [self.components[label].measurements.values
               for label in self.labels]
        out = _np.stack(out, axis=1)
        total = _np.sum(out, axis=1, keepdims=True)
        total = _np.where(_np.abs(total - 1) < 1e-10, 1.0, _np.nan)
        out = out * total

        has_value = _np.all(~ _np.isnan(out), axis=1, keepdims=True) * 1.0
        has_value = _np.tile(has_value, [1, self.length])
        out[_np.isnan(out)] = 0
        return out, has_value

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        for comp in self.components.values():
            comp.set_coordinates(coordinates)

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = CompositionalVariable(variable.name, coordinates,
                                        variable.labels)
        return new_var

    @classmethod
    def from_data_frame(cls, name, coordinates, df, columns=None,
                        *args, **kwargs):
        new_var = CompositionalVariable(
            name,
            coordinates,
            labels=columns,
            measurements=df.loc[:, columns].values)
        return new_var

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)

        for name, comp in self.components.items():
            new_obj.components[name] = comp[item]

        return new_obj

    def as_data_frame(self, probability=True, predictions=True, **kwargs):
        all_dfs = []
        for key, val in self.components.items():
            cat_df = val.as_data_frame(
                probability=probability,
                predictions=predictions,
                **kwargs)
            cat_df.columns = [self.name + "_" + col for col in cat_df.columns]
            all_dfs.append(cat_df)
        all_dfs = _pd.concat(all_dfs, axis=1)

        return all_dfs

    def update(self, idx, **kwargs):
        mean = _tf.unstack(kwargs["mean"], axis=1)
        variance = _tf.unstack(kwargs["variance"], axis=1)
        probability = _tf.unstack(kwargs["probability"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)

        for lb, m, v, p, s in zip(
                self.labels, mean, variance,
                probability, simulations):
            self.components[lb].update(idx, **{
                "mean": m,
                "variance": v,
                "probability": p,
                "simulations": s
            })

    def allocate_simulations(self, n_sim):
        for comp in self.labels:
            self.components[comp].allocate_simulations(n_sim)

    def fill_pyvista_cube(self, cube, prefix=None):
        for comp in self.labels:
            self.components[comp].fill_pyvista_cube(cube, self.name)

    def fill_pyvista_points(self, points, prefix=None):
        for comp in self.labels:
            self.components[comp].fill_pyvista_points(points, self.name)


class _Category(_Variable):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)
        n_data = coordinates.n_data

        self.probability = self._Attribute(coordinates, _np.zeros(n_data))
        self.indicator = self._Attribute(coordinates, - _np.ones(n_data))
        self.indicator_mean = self._Attribute(coordinates)
        self.indicator_variance = self._Attribute(coordinates)
        self.indicator_predicted = self._Attribute(coordinates)
        self.simulations = []

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        new_obj.probability = self.probability[item]
        new_obj.indicator = self.indicator[item]
        new_obj.indicator_mean = self.indicator_mean[item]
        new_obj.indicator_variance = self.indicator_variance[item]
        new_obj.indicator_predicted = self.indicator_predicted[item]

        for i, sim in enumerate(self.simulations):
            new_obj.simulations[i].values = sim.values[item]

        return new_obj

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.probability.coordinates = coordinates
        self.indicator_mean.coordinates = coordinates
        self.indicator_variance.coordinates = coordinates
        self.indicator_predicted.coordinates = coordinates

        for sim in self.simulations:
            sim.coordinates = coordinates

    def as_data_frame(self, probability=True, predictions=True,
                      simulations=False):
        df = _pd.DataFrame({})

        if probability:
            df[self.name + "_probability"] = self.probability.values
            df[self.name + "_indicator"] = self.indicator.values

        if predictions:
            df[self.name + "_indicator_mean"] = self.indicator_mean.values
            df[self.name + "_indicator_variance"] = \
                self.indicator_variance.values
            df[self.name + "_indicator_predicted"] = \
                self.indicator_predicted.values

        if simulations:
            for i, sim in enumerate(self.simulations):
                df[self.name + "_sim_" + str(i)] = sim.values

        return df

    def update(self, idx, **kwargs):
        self.indicator_predicted.values[idx] = kwargs["indicator"].numpy()
        self.indicator_mean.values[idx] = kwargs["mean"].numpy()
        self.indicator_variance.values[idx] = kwargs["variance"].numpy()
        self.probability.values[idx] = kwargs["probability"].numpy()

        sims = kwargs["simulations"].numpy()
        for s in range(sims.shape[1]):
            self.simulations[s].values[idx] = sims[:, s]

    def allocate_simulations(self, n_sim):
        self.simulations = [self._Attribute(self.coordinates)
                            for _ in range(n_sim)]

    def fill_pyvista_cube(self, cube, prefix=None):
        label = prefix + " - " + self.name

        self.indicator.fill_pyvista_cube(
            cube, label + " - indicator")
        self.indicator_mean.fill_pyvista_cube(
            cube, label + " - indicator mean")
        self.indicator_variance.fill_pyvista_cube(
            cube, label + " - indicator variance")
        self.indicator_predicted.fill_pyvista_cube(
            cube, label + " - indicator predicted")
        self.probability.fill_pyvista_cube(
            cube, label + " - probability")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_cube(
                cube, label + " - simulation %d" % i)

    def fill_pyvista_points(self, points, prefix=None):
        label = prefix + " - " + self.name

        self.indicator.fill_pyvista_points(
            points, label + " - indicator")
        self.indicator_mean.fill_pyvista_points(
            points, label + " - indicator mean")
        self.indicator_variance.fill_pyvista_points(
            points, label + " - indicator variance")
        self.indicator_predicted.fill_pyvista_points(
            points, label + " - indicator predicted")
        self.probability.fill_pyvista_points(
            points, label + " - probability")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_points(
                points, label + " - simulation %d" % i)


class RockTypeVariable(CompositionalVariable):
    def __init__(self, name, coordinates, labels=None, measurements_a=None,
                 measurements_b=None):
        if measurements_b is None:
            measurements_b = measurements_a

        if labels is None:
            if measurements_a is None:
                raise Exception("either the labels or measurements"
                                "must be provided")
            cat_a = _pd.Categorical(measurements_a)
            cat_b = _pd.Categorical(measurements_b)
            labels = _pd.api.types.union_categoricals([cat_a, cat_b])
            labels = labels.categories.values

        n_cat = len(labels)
        n_data = coordinates.n_data

        avg_vals = None
        if measurements_a is not None:
            vals_a = _np.zeros([n_data, n_cat])
            vals_b = vals_a.copy()
            for i, label in enumerate(labels):
                vals_a[measurements_a == label, i] = 1
                vals_b[measurements_b == label, i] = 1
            avg_vals = 0.5 * (vals_a + vals_b)

        super().__init__(name, coordinates,
                         labels=labels, measurements=avg_vals)
        # self._length *= 2

        self.predicted = self._Attribute(
            coordinates, _np.array([""]*n_data), dtype=object)
        self.entropy = self._Attribute(coordinates)
        self.uncertainty = self._Attribute(coordinates)

        if measurements_a is None:
            self.measurements_a = self._Attribute(
                coordinates, _np.array([""] * n_data), dtype=object)
            self.measurements_b = self._Attribute(
                coordinates, _np.array([""] * n_data), dtype=object)
            self.boundary = self._Attribute(
                coordinates, [False]*n_data, dtype=bool)
        else:
            self.measurements_a = self._Attribute(
                coordinates, measurements_a, dtype=object)
            self.measurements_b = self._Attribute(
                coordinates, measurements_b, dtype=object)
            self.boundary = self._Attribute(
                coordinates, measurements_a != measurements_b, dtype=bool)

    # def get_measurements(self):
    #     out = [self.components[label].measurements.values
    #            for label in self.labels]
    #     out = _np.stack(out, axis=1)
    #     return _np.concatenate([out, out], axis=1)

    def set_coordinates(self, coordinates):
        super().set_coordinates(coordinates)
        self.predicted.coordinates = coordinates
        self.entropy.coordinates = coordinates
        self.uncertainty.coordinates = coordinates
        self.boundary.coordinates = coordinates

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = RockTypeVariable(variable.name, coordinates,
                                   variable.labels)
        return new_var

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col_a=None, col_b=None,
                        *args, **kwargs):
        labels = _pd.api.types.union_categoricals(
            [_pd.Categorical(df[col_a]),
             _pd.Categorical(df[col_b])])
        labels = labels.categories.values

        new_var = RockTypeVariable(name, coordinates, labels,
                                   measurements_a=df[col_a].values,
                                   measurements_b=df[col_b].values)
        return new_var

    def __getitem__(self, item):
        new_obj = super().__getitem__(item)

        new_obj.predicted = self.predicted[item]
        new_obj.entropy = self.entropy[item]
        new_obj.uncertainty = self.uncertainty[item]
        new_obj.boundary = self.boundary[item]
        new_obj.measurements_a = self.measurements_a[item]
        new_obj.measurements_b = self.measurements_b[item]

        return new_obj

    def as_data_frame(self, probability=True, predictions=True, **kwargs):
        df = super().as_data_frame(probability, predictions, **kwargs)
        df[self.name + "_a"] = self.measurements_a.values
        df[self.name + "_b"] = self.measurements_b.values

        if predictions:
            df[self.name + "_predicted"] = self.predicted.values
            df[self.name + "_entropy"] = self.entropy.values
            df[self.name + "_uncertainty"] = self.uncertainty.values

        return df

    def update(self, idx, **kwargs):
        self.entropy.values[idx] = kwargs["entropy"].numpy()
        self.uncertainty.values[idx] = kwargs["uncertainty"].numpy()

        max_prob = _np.argmax(kwargs["probability"].numpy(), axis=1)
        self.predicted.values[idx] = _np.array(self.labels)[max_prob]
        # self.predicted.values[idx] = self.labels[max_prob]

        mean = _tf.unstack(kwargs["mean"], axis=1)
        variance = _tf.unstack(kwargs["variance"], axis=1)
        indicators = _tf.unstack(kwargs["indicators"], axis=1)
        probability = _tf.unstack(kwargs["probability"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)

        for lb, m, v, i, p, s in zip(
                self.labels, mean, variance, indicators,
                probability, simulations):
            self.components[lb].update(idx, **{
                "mean": m,
                "variance": v,
                "indicator": i,
                "probability": p,
                "simulations": s
            })

    def training_input(self, idx=None):
        if idx is None:
            idx = _np.arange(self.coordinates.n_data)
        return {"is_boundary": _tf.constant(self.boundary.values[idx, None],
                                            _tf.bool)}

    def fill_pyvista_cube(self, cube, prefix=None):
        self.measurements_a.fill_pyvista_cube(
            cube, self.name + " - measurements_a")
        self.measurements_b.fill_pyvista_cube(
            cube, self.name + " - measurements_b")
        self.predicted.fill_pyvista_cube(
            cube, self.name + " - predicted")
        self.entropy.fill_pyvista_cube(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_cube(
            cube, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_cube(cube, self.name)

    def fill_pyvista_points(self, points, prefix=None):
        self.measurements_a.fill_pyvista_points(
            points, self.name + " - measurements_a")
        self.measurements_b.fill_pyvista_points(
            points, self.name + " - measurements_b")
        self.predicted.fill_pyvista_points(
            points, self.name + " - predicted")
        self.entropy.fill_pyvista_points(
            points, self.name + " - entropy")
        self.uncertainty.fill_pyvista_points(
            points, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_points(points, self.name)


class CategoricalVariable(RockTypeVariable):
    def __init__(self, name, coordinates, labels, measurements=None):
        super().__init__(name, coordinates, labels, measurements_a=measurements)

    def fill_pyvista_cube(self, cube, prefix=None):
        self.measurements_a.fill_pyvista_cube(
            cube, self.name + " - measurements")
        self.predicted.fill_pyvista_cube(
            cube, self.name + " - predicted")
        self.entropy.fill_pyvista_cube(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_cube(
            cube, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_cube(cube, self.name)

    def fill_pyvista_points(self, points, prefix=None):
        self.measurements_a.fill_pyvista_points(
            points, self.name + " - measurements")
        self.predicted.fill_pyvista_points(
            points, self.name + " - predicted")
        self.entropy.fill_pyvista_points(
            points, self.name + " - entropy")
        self.uncertainty.fill_pyvista_points(
            points, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_points(points, self.name)


class OrderedRockType(RockTypeVariable):
    def __init__(self, name, coordinates, labels=None, measurements_a=None,
                 measurements_b=None):
        super().__init__(name, coordinates, labels, measurements_a,
                         measurements_b)
        self._length = 1

        if measurements_b is None:
            measurements_b = measurements_a

        implicit_values = -0.5 * _np.ones_like(measurements_a)
        for i in range(len(labels[:-1])):
            implicit_values = _np.where(
                (measurements_a == labels[i])
                & (measurements_b == labels[i + 1]),
                i,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i + 1])
                & (measurements_b == labels[i]),
                i,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i])
                & (measurements_b == labels[i]),
                i - 0.5,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i + 1])
                & (measurements_b == labels[i + 1]),
                i + 0.5,
                implicit_values
            )
            # implicit_values[(measurements_a == labels[i])
            #                 & (measurements_b == labels[i + 1])] = i
            # implicit_values[(measurements_a == labels[i + 1])
            #                 & (measurements_b == labels[i])] = i
            # implicit_values[(measurements_a == labels[i])
            #                 & (measurements_b == labels[i])] = i - 0.5
            # implicit_values[(measurements_a == labels[i + 1])
            #                 & (measurements_b == labels[i + 1])] = i + 0.5

        self.implicit_values = self._Attribute(coordinates, implicit_values)

    def get_measurements(self):
        values = self.implicit_values.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    def __getitem__(self, item):
        new_obj = super().__getitem__(item)
        new_obj.implicit_values = self.implicit_values[item]
        return new_obj


class BinaryVariable(_Variable):
    def __init__(self, name, coordinates, labels, measurements=None):
        super().__init__(name, coordinates)
        n_data = coordinates.n_data

        self.labels = labels
        self._length = 1
        if len(labels) != 2:
            raise ValueError("there must be exactly 2 labels - found %d"
                             % len(labels))

        self.indicator = self._Attribute(
            coordinates, _np.array([_np.nan]*n_data))
        if measurements is None:
            self.measurements = self._Attribute(
                coordinates, [""]*n_data, dtype=object)
            self.weights = self._Attribute(coordinates)
        else:
            self.measurements = self._Attribute(
                coordinates, measurements, dtype=object)
            self.weights = self._Attribute(coordinates, _np.ones(n_data))
            self.indicator.values[measurements == labels[0]] = 1
            self.indicator.values[measurements == labels[1]] = 0

        self.predicted = self._Attribute(
            coordinates, [""]*n_data, dtype=object)
        self.probability = self._Attribute(coordinates, _np.zeros(n_data))
        self.entropy = self._Attribute(coordinates)
        self.uncertainty = self._Attribute(coordinates)

        self.latent_mean = self._Attribute(coordinates)
        self.latent_variance = self._Attribute(coordinates)

        self.simulations = []

        if measurements is not None:
            for label in labels:
                idx = measurements == label
                n_in_label = _np.sum(idx)
                if n_in_label > 0:
                    self.weights.values[idx] = \
                        n_data / (n_in_label * len(labels))

    def get_measurements(self):
        values = self.indicator.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = BinaryVariable(variable.name, coordinates, variable.labels)
        return new_var

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        new_obj.probability = self.probability[item]
        new_obj.indicator = self.indicator[item]
        new_obj.latent_mean = self.latent_mean[item]
        new_obj.latent_variance = self.latent_variance[item]
        new_obj.predicted = self.predicted[item]
        new_obj.entropy = self.entropy[item]
        new_obj.uncertainty = self.uncertainty[item]
        new_obj.measurements = self.measurements[item]
        new_obj.weights = self.weights[item]

        for i, sim in enumerate(self.simulations):
            new_obj.simulations[i].values = sim.values[item]

        return new_obj

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.probability.coordinates = coordinates
        self.indicator.coordinates = coordinates
        self.latent_mean.coordinates = coordinates
        self.latent_variance.coordinates = coordinates
        self.predicted.coordinates = coordinates
        self.entropy.coordinates = coordinates
        self.uncertainty.coordinates = coordinates
        self.measurements.coordinates = coordinates
        self.weights.coordinates = coordinates

        for sim in self.simulations:
            sim.coordinates = coordinates

    def as_data_frame(self, measurements=True, latent=True,
                      predictions=True, simulations=False):
        df = _pd.DataFrame({})

        if measurements:
            df[self.name + "_measurements"] = self.measurements.values
            df[self.name + "_weights"] = self.weights.values

        if predictions:
            df[self.name + "_predicted"] = self.predicted.values
            df[self.name + "_probability"] = self.probability.values
            df[self.name + "_entropy"] = self.entropy.values
            df[self.name + "_uncertainty"] = self.uncertainty.values

        if latent:
            df[self.name + "_latent_mean"] = self.latent_mean.values
            df[self.name + "_latent_variance"] = self.latent_variance.values

        if simulations:
            for i, sim in enumerate(self.simulations):
                df[self.name + "_sim_" + str(i)] = sim.values

        return df

    def update(self, idx, **kwargs):
        prob = kwargs["probability"].numpy()
        mean = kwargs["mean"].numpy()
        var = kwargs["variance"].numpy()
        entropy = kwargs["entropy"].numpy()
        uncertainty = kwargs["uncertainty"].numpy()
        sims = kwargs["simulations"].numpy()

        if len(prob.shape) > 1:
            prob = prob[:, 0]
            mean = mean[:, 0]
            var = var[:, 0]
            # entropy = entropy[:, 0]
            # uncertainty = uncertainty[:, 0]
            sims = sims[:, 0, :]

        label_idx = _np.zeros(prob.shape, dtype=_np.int64)  # positive class
        label_idx[prob < 0.5] = 1  # negative class

        self.predicted.values[idx] = _np.array(self.labels)[label_idx]
        self.latent_mean.values[idx] = mean
        self.latent_variance.values[idx] = var
        self.entropy.values[idx] = entropy
        self.uncertainty.values[idx] = uncertainty
        self.probability.values[idx] = prob

        for s in range(sims.shape[1]):
            self.simulations[s].values[idx] = sims[:, s]

    def allocate_simulations(self, n_sim):
        self.simulations = [self._Attribute(self.coordinates)
                            for _ in range(n_sim)]

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col, positive_class):
        labels = _pd.Categorical(df[col])
        labels = labels.categories.values.tolist()

        pos = None
        for i, label in enumerate(labels):
            if label == positive_class:
                pos = i
        labels.pop(pos)
        labels.append(positive_class)
        labels = labels[::-1]

        new_var = BinaryVariable(name, coordinates, labels,
                                 measurements=df[col].values)
        return new_var

    def fill_pyvista_cube(self, cube, prefix=None):
        self.indicator.fill_pyvista_cube(
            cube, self.name + " - indicator")
        self.latent_mean.fill_pyvista_cube(
            cube, self.name + " - latent mean")
        self.latent_variance.fill_pyvista_cube(
            cube, self.name + " - latent variance")
        self.predicted.fill_pyvista_cube(
            cube, self.name + " - predicted")
        self.probability.fill_pyvista_cube(
            cube, self.name + " - probability")
        self.entropy.fill_pyvista_cube(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_cube(
            cube, self.name + " - uncertainty")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_cube(
                cube, self.name + " - simulation %d" % i)

    def fill_pyvista_points(self, points, prefix=None):
        self.indicator.fill_pyvista_points(
            points, self.name + " - indicator")
        self.latent_mean.fill_pyvista_points(
            points, self.name + " - latent mean")
        self.latent_variance.fill_pyvista_points(
            points, self.name + " - latent variance")
        self.predicted.fill_pyvista_points(
            points, self.name + " - predicted")
        self.probability.fill_pyvista_points(
            points, self.name + " - probability")
        self.entropy.fill_pyvista_points(
            points, self.name + " - entropy")
        self.uncertainty.fill_pyvista_points(
            points, self.name + " - uncertainty")

        for i, sim in enumerate(self.simulations):
            sim.fill_pyvista_points(
                points, self.name + " - simulation %d" % i)


class AnomalyVariable(BinaryVariable):
    def __init__(self, name, coordinates, label, measurements=None):
        labels = [label, "_dummy"]
        super().__init__(name, coordinates, labels, measurements)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col, positive_class):
        new_var = AnomalyVariable(name, coordinates, positive_class,
                                  measurements=df[col].values)
        return new_var


class _SpatialData(object):
    """Abstract class for spatial data in general"""

    def __init__(self):
        self._n_dim = None
        self._bounding_box = None
        self._n_data = None
        self._diagonal = None
        self.variables = {}

    def __repr__(self):
        return self.__str__()

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def n_data(self):
        return self._n_data

    @property
    def diagonal(self):
        return self._bounding_box.diagonal

    def aspect_ratio(self, vertical_exaggeration=1):
        """
        Returns a list with plotly layout data.
        """
        if self._n_dim == 2:
            return _py.aspect_ratio_2d(vertical_exaggeration)
        elif self._n_dim == 3:
            return _py.aspect_ratio_3d(self.bounding_box, vertical_exaggeration)
        else:
            raise ValueError("aspect ratio only available for 2- and "
                             "3-dimensional data objects")

    def draw_bounding_box(self, **kwargs):
        if self._n_dim == 2:
            raise NotImplementedError
        elif self._n_dim == 3:
            return _py.bounding_box_3d(self.bounding_box, **kwargs)
        else:
            raise ValueError("bounding_box only available for 2- and "
                             "3-dimensional data objects")


class _PointBased(_SpatialData):
    """Abstract class for data objects based on points"""
    def __init__(self):
        super().__init__()
        self.coordinates = None

    def __str__(self):
        s = "Object of class %s with %s data locations\n\n" \
            % (self.__class__.__name__, str(self.n_data))

        if len(self.variables) > 0:
            s += "Variables:\n"
            for name, var in self.variables.items():
                s += "    %s: %s\n" % (name, var.__class__.__name__)
        return s

    def add_continuous_variable(self, name, measurements=None,
                                quantiles=None,
                                probabilities=(0.025, 0.5, 0.975)):
        self.variables[name] = ContinuousVariable(
            name, self, measurements, quantiles=quantiles,
            probabilities=probabilities)

    def add_categorical_variable(self, name, labels, measurements=None):
        self.variables[name] = CategoricalVariable(
            name, self, labels, measurements)

    def add_rock_type_variable(self, name, labels=None, measurements_a=None,
                               measurements_b=None, ordered=False):
        if ordered:
            self.variables[name] = OrderedRockType(
                name, self, labels, measurements_a, measurements_b)
        else:
            self.variables[name] = RockTypeVariable(
                name, self, labels, measurements_a, measurements_b)

    def add_binary_variable(self, name, labels, measurements=None):
        self.variables[name] = BinaryVariable(name, self, labels, measurements)

    def add_anomaly_variable(self, name, label, measurements=None):
        self.variables[name] = AnomalyVariable(name, self, label, measurements)

    def add_compositional_variable(self, name, labels, measurements=None):
        self.variables[name] = CompositionalVariable(
            name, self, labels, measurements)

    def get_data_variance(self):
        return _np.zeros_like(self.coordinates)


class PointData(_PointBased):
    """
        Data represented as points in arbitrary locations.
    """

    def __init__(self, data, coordinates):
        """

        Parameters
        ----------
        data : _pd.DataFrame
        coordinates : str or list
        """
        super().__init__()

        if isinstance(coordinates, str):
            coordinates = [coordinates]

        self.coordinate_labels = coordinates
        self.coordinates = _np.array(data.loc[:, coordinates], ndmin=2)

        self._n_dim = self.coordinates.shape[1]
        self._n_data = self.coordinates.shape[0]
        if self._n_data > 0:
            self._bounding_box = BoundingBox.from_array(self.coordinates)
        else:
            self._bounding_box = BoundingBox.from_array(
                _np.zeros([2, self.n_dim]))

    def as_data_frame(self, **kwargs):
        """
        Conversion of a spatial object to a data frame.
        """
        df = [_pd.DataFrame(self.coordinates,
                            columns=self.coordinate_labels)]
        for variable in self.variables.values():
            df.append(variable.as_data_frame(**kwargs))
        df = _pd.concat(df, axis=1)
        return df

    @classmethod
    def from_array(cls, coordinates, coordinate_labels=None):
        df = _pd.DataFrame(coordinates)
        if coordinate_labels is not None:
            df.columns = coordinate_labels
        else:
            n_dim = coordinates.shape[1]
            if n_dim <= 3:
                df.columns = ["X", "Y", "Z"][0:n_dim]
            else:
                df.columns = ["V" + str(i) for i in range(n_dim)]
        return PointData(df, df.columns)

    def __getitem__(self, item):
        self_copy = _copy.deepcopy(self)
        new_obj = PointData.from_array(self_copy.coordinates[item])
        new_obj.coordinate_labels = self_copy.coordinate_labels
        for name, var in self_copy.variables.items():
            new_obj.variables[name] = var[item]
            new_obj.variables[name].set_coordinates(new_obj)
        return new_obj

    def subset_region(self, min_val, max_val,
                      include_min=None, include_max=None):
        if not (isinstance(min_val, list)
                or isinstance(min_val, tuple)
                or isinstance(min_val, _np.ndarray)):
            min_val = [min_val]
        if not (isinstance(max_val, list)
                or isinstance(max_val, tuple)
                or isinstance(max_val, _np.ndarray)):
            max_val = [max_val]
        if include_min is None:
            include_min = [True] * self.n_dim
        if include_max is None:
            include_max = [False] * self.n_dim
        if not (isinstance(include_min, list)
                or isinstance(include_min, tuple)):
            include_min = [include_min]
        if not (isinstance(include_max, list)
                or isinstance(include_max, tuple)):
            include_max = [include_max]

        checks = (len(min_val) == self.n_dim,
                  len(max_val) == self.n_dim,
                  len(include_min) == self.n_dim,
                  len(include_max) == self.n_dim)
        if not all(checks):
            raise ValueError("all arguments must match the data dimension")

        keep = _np.array([True] * self.n_data)
        for i in range(self.n_dim):
            keep = keep & (self.coordinates[:, i] >= min_val[i])
            if not include_min[i]:
                keep = keep & (self.coordinates[:, i] > min_val[i])
            keep = keep & (self.coordinates[:, i] <= max_val[i])
            if not include_max[i]:
                keep = keep & (self.coordinates[:, i] < max_val[i])

        return self[keep] if sum(keep) > 0 else None

    def as_pyvista(self):
        if not self.n_dim == 3:
            raise ValueError("as_pyvista method is only supported "
                             "for 3-dimensional data")

        pv_points = _pv.PolyData(self.coordinates)

        for var in self.variables.keys():
            self.variables[var].fill_pyvista_points(pv_points)

        return pv_points


class GaussianData(PointData):
    def __init__(self, data, coordinates_mean, coordinates_variance):
        super().__init__(data, coordinates_mean)
        self.variance = _np.array(data.loc[:, coordinates_variance], ndmin=2)

    @classmethod
    def from_array(cls, coordinates, coordinates_variance,
                   coordinate_labels=None):
        df = _pd.DataFrame(coordinates)
        if coordinate_labels is None:
            n_dim = coordinates.shape[1]
            if n_dim <= 3:
                coordinate_labels = ["X", "Y", "Z"][0:n_dim]
            else:
                coordinate_labels = ["V" + str(i) for i in range(n_dim)]

        df.columns = coordinate_labels
        var_labels = [s + "_var" for s in coordinate_labels]
        df_var = _pd.DataFrame(coordinates_variance, columns=var_labels)
        df = _pd.concat([df, df_var], axis=1)
        return GaussianData(df, coordinate_labels, var_labels)

    def get_data_variance(self):
        return self.variance


class _GriddedData(PointData):
    def __init__(self, data, coordinates):
        super().__init__(data, coordinates)
        self.grid = None
        self.grid_size = None
        self.step_site = None

    def index_data(self, data):
        if data.n_dim != self.n_dim:
            raise ValueError("Data dimension mismatch. Expected dimension %d"
                             " and found %d." % (self.n_dim, data.n_dim))

        cell_id = [
            _np.ceil((data.coordinates[:, i] - self.grid[i][0]
                       - self.step_size[i]/2) / self.step_size[i]).astype(int)
            for i in range(self.n_dim)
        ]

        return _np.stack(cell_id, axis=1)


class Grid1D(_GriddedData):
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

    def __init__(self, start, n, step=None, end=None, label=None):
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
        label : str
            The label for the coordinate.


        Either step or end must be given. If both are given, end is ignored.
        """
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        if step is not None:
            end = start + (n - 1) * step
        else:
            step = (end - start) / (n - 1)
        grid = _np.linspace(start, end, n, dtype=float)

        if label is None:
            label = "X"
        grid_df = _pd.DataFrame({label: grid})

        super().__init__(grid_df, label)
        self.step_size = [step]
        self.grid = [grid]
        self.grid_size = [int(n)]

    def aggregate_categorical(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([x for x in range(int(self.grid_size[0]))])
        cols = ["xid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements_a.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data_2 = raw_data.copy()
        raw_data_2["value"] = data.variables[variable].measurements_b.values
        raw_data = _pd.concat([raw_data, raw_data_2],
                              axis=0).reset_index(drop=True)

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = CategoricalVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_binary(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([x for x in range(int(self.grid_size[0]))])
        cols = ["xid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = BinaryVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_numeric(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.arange(int(self.grid_size[0]))[:, None]
        cols = ["xid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])

        # aggregating
        data_2 = raw_data.groupby(cols).mean()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # output
        data_3 = grid_full.join(data_2.set_index(cols), on=cols) \
            .reset_index(drop=False)
        self.variables[variable] = ContinuousVariable(
            variable, self, data_3["value"].values
        )


class Grid2D(_GriddedData):
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

    def __init__(self, start, n, step=None, end=None, labels=None):
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
        labels : list
            The labels for the coordinates.


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
                              (end[1] - start[1]) / (n[1] - 1)])
        grid_x = _np.linspace(start[0], end[0], n[0])
        grid_y = _np.linspace(start[1], end[1], n[1])
        coords = _np.array(list(_iter.product(grid_y, grid_x)),
                           dtype=float)[:, ::-1]

        if labels is None:
            labels = ["X", "Y"]
        grid = _pd.DataFrame(coords, columns=labels)

        super().__init__(grid, labels)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y]
        self.grid_size = [int(num) for num in n]

    def aggregate_categorical(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y)
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements_a.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data_2 = raw_data.copy()
        raw_data_2["value"] = data.variables[variable].measurements_b.values
        raw_data = _pd.concat([raw_data, raw_data_2],
                              axis=0).reset_index(drop=True)

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = CategoricalVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_binary(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y)
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = BinaryVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_numeric(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y)
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])

        # aggregating
        data_2 = raw_data.groupby(cols).mean()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # output
        data_3 = grid_full.join(data_2.set_index(cols), on=cols)\
            .reset_index(drop=False)
        self.variables[variable] = ContinuousVariable(
            variable, self, data_3["value"].values
        )


class Grid3D(_GriddedData):
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

    def __init__(self, start, n, step=None, end=None, labels=None):
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
        labels : list
            The labels for the coordinates.


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
        coords = _np.array(list(_iter.product(grid_z, grid_y, grid_x)),
                           dtype=float)[:, ::-1]

        if labels is None:
            labels = ["X", "Y", "Z"]
        grid = _pd.DataFrame(coords, columns=labels)

        super().__init__(grid, labels)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y, grid_z]
        self.grid_size = [int(num) for num in n]

    def aggregate_categorical(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y, z)
                             for z in range(int(self.grid_size[2]))
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid", "zid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements_a.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data["zid"] = _np.round(
            (data.coordinates[:, 2] - self.grid[2][0]
             - self.step_size[2] / 2) / self.step_size[2])
        raw_data_2 = raw_data.copy()
        raw_data_2["value"] = data.variables[variable].measurements_b.values
        raw_data = _pd.concat([raw_data, raw_data_2],
                              axis=0).reset_index(drop=True)

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = CategoricalVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_binary(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y, z)
                             for z in range(int(self.grid_size[2]))
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid", "zid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data["zid"] = _np.round(
            (data.coordinates[:, 2] - self.grid[2][0]
             - self.step_size[2] / 2) / self.step_size[2])

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = BinaryVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_numeric(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y, z)
                             for z in range(int(self.grid_size[2]))
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid", "zid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values,
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data["zid"] = _np.round(
            (data.coordinates[:, 2] - self.grid[2][0]
             - self.step_size[2] / 2) / self.step_size[2])

        # aggregating
        data_2 = raw_data.groupby(cols).mean()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # output
        data_3 = grid_full.join(data_2.set_index(cols), on=cols) \
            .reset_index(drop=False)
        self.variables[variable] = ContinuousVariable(
            variable, self, data_3["value"].values
        )

    def make_interpolator(self, coordinates):
        return _gint.cubic_conv_3d(coordinates,
                                   self.grid[0], self.grid[1], self.grid[2])

    def as_pyvista(self):
        pv_grid = _pv.StructuredGrid(*_np.meshgrid(*self.grid))

        for var in self.variables.keys():
            self.variables[var].fill_pyvista_cube(pv_grid)

        return pv_grid


class GridND(_GriddedData):
    """
    Implicit grid in N dimensions.
    """
    def __init__(self, start, n, step=None, end=None, labels=None):
        super().__init__()
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        start = _np.array(start)
        n = _np.array(n)
        if step is not None:
            step = _np.array(step)
            end = start + (n - 1) * step
        else:
            end = _np.array(end)
            step = _np.array([(e - st) / (n_ - 1)
                              for st, e, n_ in zip(start, end, n)])

        self.grid = []
        for st, e, n_ in zip(start, end, n):
            self.grid.append(_np.linspace(st, e, n_))

        self.step_size = step.tolist()
        self.grid_size = [int(num) for num in n]
        self.labels = labels

        self._n_dim = len(self.grid)
        self._n_data = _np.prod(self.grid_size)
        self._bounding_box = BoundingBox.from_array(
            _np.stack([start, end], axis=0))

    def __str__(self):
        s = "Object of class %s with %s data locations\n" \
            % (self.__class__.__name__, str(self.n_data))

        return s


class DirectionalData(PointData):
    def __init__(self, data, coordinates, directions):
        """

        Parameters
        ----------
        data : _pd.DataFrame
        coordinates : str or list
        """
        super().__init__(data, coordinates)

        if isinstance(directions, str):
            directions = [directions]
        if len(directions) != self.n_dim:
            raise ValueError("arguments coordinates and directions must"
                             "have the same length")

        self.direction_labels = directions
        self.directions = _np.array(data.loc[:, directions], ndmin=2)

        all_coords = _np.concatenate([self._bounding_box.as_array(),
                                      self.directions],
                                     axis=0)
        self._bounding_box, self._diagonal = bounding_box(all_coords)

    def as_data_frame(self, full=False):
        """
        Conversion of a spatial object to a data frame.
        """
        df = [_pd.DataFrame(self.coordinates, columns=self.coordinate_labels),
              _pd.DataFrame(self.directions, columns=self.direction_labels)]
        for variable in self.variables.values():
            df.append(variable.as_data_frame(full))
        df = _pd.concat(df, axis=1)
        return df

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        new_obj.coordinates = self.coordinates[item]
        if len(new_obj.coordinates.shape) < 2:
            new_obj.coordinates = _np.expand_dims(new_obj.coordinates, axis=0)
        new_obj.directions = self.directions[item]
        if len(new_obj.directions.shape) < 2:
            new_obj.directions = _np.expand_dims(new_obj.directions, axis=0)
        new_obj._n_data = new_obj.coordinates.shape[0]

        all_coords = _np.concatenate([new_obj.coordinates, new_obj.directions],
                                     axis=0)
        new_obj._bounding_box, new_obj._diagonal = bounding_box(all_coords)

        for name, var in new_obj.variables.items():
            new_obj.variables[name] = var[item]
            new_obj.variables[name].set_coordinates(new_obj)
        return new_obj

    @classmethod
    def from_azimuth(cls, data, coordinates, azimuth):
        azimuth = data[azimuth]
        data = data.copy()
        data["dX"] = _np.sin(azimuth / 180 * _np.pi)
        data["dY"] = _np.cos(azimuth / 180 * _np.pi)
        return DirectionalData(data, coordinates, ["dX", "dY"])

    @classmethod
    def from_planes(cls, data, coordinates, azimuth, dip):
        # conversions
        dip = -data[dip].values * _np.pi / 180
        azimuth = (90 - data[azimuth].values) * _np.pi / 180
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

        # result
        vecs = _np.concatenate([dipvec, strvec], axis=0)
        vecs = _pd.DataFrame(vecs, columns=["dX", "dY", "dZ"])
        data = _pd.concat([data, data], axis=0).reset_index(drop=True)
        data = _pd.concat([data, vecs], axis=1)
        return DirectionalData(data, coordinates, ["dX", "dY", "dZ"])

    @classmethod
    def from_azimuth_and_dip(cls, data, coordinates, azimuth, dip):
        dip = -data[dip] * _np.pi / 180
        azimuth = (90 - data[azimuth]) * _np.pi / 180

        dipvec = _np.concatenate([
            _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
            _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
            _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1)

        # result
        vecs = _pd.DataFrame(dipvec, columns=["dX", "dY", "dZ"])
        data = _pd.concat([data, vecs], axis=1)
        return DirectionalData(data, coordinates, ["dX", "dY", "dZ"])

    @classmethod
    def from_normals(cls, data, coordinates, azimuth, dip):
        n_data = data.shape[0]
        plane_dirs = cls.from_planes(data, coordinates, azimuth, dip)
        vec1 = plane_dirs.directions[0:n_data, :]
        vec2 = plane_dirs.directions[n_data:(2 * n_data), :]

        normalvec = vec1[:, [1, 2, 0]] * vec2[:, [2, 0, 1]] \
                    - vec1[:, [2, 0, 1]] * vec2[:, [1, 2, 0]]
        normalvec = _np.apply_along_axis(
            lambda x: x / _np.sqrt(_np.sum(x ** 2)),
            axis=1,
            arr=normalvec)

        # result
        vecs = _pd.DataFrame(- normalvec, columns=["dX", "dY", "dZ"])
        data = _pd.concat([data, vecs], axis=1)
        return DirectionalData(data, coordinates, ["dX", "dY", "dZ"])


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
                 x="X", y="Y", z="Z", fr="FROM", to="TO", az=None,
                 dip=None):
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
        collar = collar.copy()

        if survey is not None:
            raise NotImplementedError("survey data is not yet supported")

        if az is None:
            az = "AZIMUTH"
            collar[az] = 0
        if dip is None:
            dip = "DIP"
            collar[dip] = 90

        assay = assay.drop([x, y, z], axis=1, errors="ignore")

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

        directions = DirectionalData.from_azimuth_and_dip(
            df, [x, y, z], az, dip
        ).directions

        # df_from = _pd.DataFrame({"X": 0, "Y": 0, "Z": df[fr]})
        # df_to = _pd.DataFrame({"X": 0, "Y": 0, "Z": df[to]})
        # coords_from = df.loc[:, [x, y, z]].values - df_from.values
        # coords_to = df.loc[:, [x, y, z]].values - df_to.values

        coords_from = df.loc[:, [x, y, z]].values \
                      + directions * df[fr].values[:, None]
        coords_to = df.loc[:, [x, y, z]].values \
                      + directions * df[to].values[:, None]

        df = df.drop([x, y, z], axis=1)
        df = df.rename(columns={holeid: "HOLEID"})

        # output
        self.coords_from = coords_from
        self.coords_to = coords_to
        self.data = df
        self.coordinate_labels = [x, y, z]

        self._n_dim = self.coords_from.shape[1]
        self._n_data = self.coords_from.shape[0]
        self.lengths = _np.sqrt(_np.sum((self.coords_from
                                         - self.coords_to) ** 2,
                                        axis=1))

        coords2 = _np.concatenate([self.coords_from, self.coords_to], axis=0)
        # self._bounding_box, self._diagonal = bounding_box(coords2)
        self._bounding_box = BoundingBox.from_array(coords2)

    def __str__(self):
        s = "Object of class " + self.__class__.__name__ + "\n\n"
        s += str(self.coords_from.shape[0]) + " drillhole segments in "
        s += str(self._n_dim) + " dimensions\n\n"
        if self.data is not None:
            s += "Data preview:\n\n"
            s += str(self.data.head())
        return s

    def as_data_frame(self):
        df = _pd.concat([
            _pd.DataFrame(
                self.coords_from,
                columns=[s + "_from" for s in self.coordinate_labels]),
            _pd.DataFrame(
                self.coords_to,
                columns=[s + "_to" for s in self.coordinate_labels]),
            self.data
        ], axis=1)
        return df

    def segment_fixed(self, interval=5):
        dif = self.coords_to - self.coords_from
        length = _np.sqrt(_np.sum(dif ** 2, axis=1))
        points = []
        newdata = []
        for i, ln in enumerate(length):
            n = int(_np.ceil(ln / interval))
            position = _np.cumsum(_np.ones(n) / (n + 1))
            points.extend([self.coords_from[i, :] + dif[i, :] * x
                           for x in position])
            newdata.append(self.data.iloc[[i] * n, :])
        points = _np.stack(points, axis=0)
        points = _pd.DataFrame(points, columns=self.coordinate_labels)
        df = _pd.concat(newdata, ignore_index=True)
        df = _pd.concat([points, df], axis=1)

        return PointData(df, self.coordinate_labels), df

    def segment_relative(self, locations=(0.05, 0.5, 0.95)):
        if locations.__class__ is not _np.ndarray:
            locations = _np.array(locations)
        if (_np.min(locations) < 0) | (_np.max(locations) > 1):
            raise ValueError("locations must contain values between " +
                             "0 and 1, inclusive")

        dif = self.coords_to - self.coords_from
        points = [self.coords_from + dif * x for x in locations]
        points = _np.concatenate(points, axis=0)
        points = _pd.DataFrame(points, columns=self.coordinate_labels)
        df = _pd.concat([self.data] * len(locations), ignore_index=True)
        df = _pd.concat([points, df], axis=1)

        return PointData(df, self.coordinate_labels), df

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
        hole_from = self.data.loc[0:(self.coords_from.shape[0] - 1), "HOLEID"] \
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
        new_coords_from = _np.zeros([len(merge_ids), self._n_dim])
        new_coords_to = _np.zeros([len(merge_ids), self._n_dim])
        for i in range(len(merge_ids)):
            new_coords_from[i, :] = self.coords_from[merge_ids[i][0], :]
            new_coords_to[i, :] = self.coords_to[merge_ids[i][-1], :]
        # data
        cols = by if by == "HOLEID" else ["HOLEID", by]
        new_df = self.data.loc[[x[0] for x in merge_ids], cols]
        new_df = new_df.reset_index(drop=True)  # very important

        # initialization
        new_obj = _copy.deepcopy(self)
        new_obj.coords_from = new_coords_from
        new_obj.coords_to = new_coords_to
        new_obj.data = new_df

        new_obj._n_data = new_obj.coords_from.shape[0]
        new_obj.lengths = _np.sqrt(_np.sum((new_obj.coords_from
                                            - new_obj.coords_to) ** 2,
                                           axis=1))

        coords2 = _np.concatenate(
            [new_obj.coords_from, new_obj.coords_to], axis=0)
        new_obj._bounding_box, new_obj._diagonal = bounding_box(coords2)

        return new_obj

    def get_contacts(self, by):
        """
        Returns a PointData with the coordinates of the contacts between
        two different categories.
        
        Parameters
        ----------
        by : name of the column with the category
        """
        merged = self.merge_segments(by)
        points, df = merged.segment_relative(_np.array([0, 1]))
        points.add_categorical_variable(
            by, _pd.unique(df[by]), measurements=df[by].values)
        points.add_categorical_variable(
            "HOLEID", _pd.unique(df["HOLEID"]),
            measurements=df["HOLEID"].values)

        # finding duplicates
        dup_label = _pd.DataFrame(points.coordinates).duplicated(keep="last")
        dup1 = _np.where(dup_label.values)[0]
        dup2 = dup1 - 1

        # new object
        new_points = points.coordinates[dup1, :]
        new_points = _pd.DataFrame(new_points, columns=self.coordinate_labels)
        new_data = _pd.DataFrame(
            {"HOLEID": points.variables["HOLEID"].measurements_a.values[dup2],
             by + "_a": points.variables[by].measurements_a.values[dup2],
             by + "_b": points.variables[by].measurements_b.values[dup1]})
        new_data = new_data.reset_index(drop=True)
        new_data = _pd.concat([new_points, new_data], axis=1)

        new_obj = PointData(new_data, self.coordinate_labels)
        new_obj.add_rock_type_variable(
            by, points.variables[by].labels,
            measurements_a=new_data[by + "_a"].values,
            measurements_b=new_data[by + "_b"].values)
        return new_obj

    def as_classification_input(self, by, interval=5):
        """
        Returns a PointData object in a format suitable for use as input to
        a classification model.
        """
        merged = self.merge_segments(by)
        points, df = merged.segment_fixed(interval)
        points.add_categorical_variable(
            by, _pd.unique(df[by]), measurements=df[by].values)

        df = points.as_data_frame()
        contacts = merged.get_contacts(by)
        contacts = contacts.as_data_frame()
        new_data = _pd.concat([df, contacts], axis=0)

        new_obj = PointData(new_data, self.coordinate_labels)
        new_obj.add_rock_type_variable(
            by, points.variables[by].labels,
            measurements_a=new_data[by + "_a"].values,
            measurements_b=new_data[by + "_b"].values)
        return new_obj

    def as_pyvista(self):
        # empty object
        drill_coords = []
        cell_links = []
        for i in range(self.n_data):
            drill_coords.append(self.coords_from[i])
            drill_coords.append(self.coords_to[i])
            cell_links.append([2, 2 * i, 2 * i + 1])
        drill_coords = _np.stack(drill_coords, axis=0)
        cell_links = _np.stack(cell_links, axis=0)

        pv_dh = _pv.PolyData(drill_coords, lines=cell_links,
                             n_lines=self.n_data)

        # scalars
        df = self.data.dropna(axis=1, how="all")
        for col in df.columns:
            # fixing special characters
            if df[col].dtype == 'object':
                df[col] = df[col].str.normalize('NFKD')\
                    .str.encode('ascii', errors='ignore').str.decode('utf-8')
            pv_dh.cell_arrays[col] = df[col].values

        return pv_dh

    def draw_categorical(self, column, colors, **kwargs):
        # converting to points and reordering
        merged = self.merge_segments(column)
        points = _np.concatenate([merged.coords_from, merged.coords_to], axis=0)
        df = _pd.concat([
            _pd.DataFrame(points, columns=self.coordinate_labels),
            _pd.concat([merged.data] * 2, axis=0).reset_index(drop=True)
        ], axis=1)
        seq = _np.arange(merged.n_data)
        sort_idx = _np.argsort(_np.concatenate([seq, seq + 0.1]))
        df = df.iloc[sort_idx, :].reset_index(drop=True)
        # return df
        return _py.segments_3d(df.loc[:, self.coordinate_labels].values,
                               df[column].values, colors, **kwargs)

    def draw_numeric(self, column, **kwargs):
        raise NotImplementedError()


def batch_index(n_data, batch_size):
    n_batches = int(_np.ceil(n_data / batch_size))
    idx = [_np.arange(i * batch_size,
                      _np.minimum((i + 1) * batch_size,
                                  n_data))
           for i in range(n_batches)]
    return idx


def export_planes(coordinates, dip, azimuth, filename, size=1):
    # conversions
    dip = -dip * _np.pi / 180
    azimuth = (90 - azimuth) * _np.pi / 180
    strike = azimuth - _np.pi / 2

    # dip and strike vectors
    dipvec = _np.concatenate([
        _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
        _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
        _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1) * size
    strvec = _np.concatenate([
        _np.array(_np.cos(strike), ndmin=2).transpose(),
        _np.array(_np.sin(strike), ndmin=2).transpose(),
        _np.zeros([dipvec.shape[0], 1])], axis=1) * size

    points = _np.stack([
        coordinates + dipvec,
        coordinates - 0.5 * strvec - 0.5 * dipvec,
        coordinates + 0.5 * strvec - 0.5 * dipvec
    ], axis=0)
    points = _np.reshape(points, [3 * points.shape[1], points.shape[2]],
                         order="F")
    idx = _np.reshape(_np.arange(points.shape[0]), coordinates.shape)

    # export
    with open(filename, 'w') as out_file:
        out_file.write(
            str(points.shape[0]) + " " + str(idx.shape[0]) + "\n")
        for line in points:
            out_file.write(" ".join(str(elem) for elem in line) + "\n")
        for line in idx:
            out_file.write(" ".join(str(elem) for elem in line) + "\n")


class Section3D(PointData):
    def __init__(self, center, azimuth, dip, width, height, n_x, n_y,
                 coordinate_labels=("X", "Y", "Z")):
        grid = Grid2D(start=[- width/2, - height/2],
                      end=[width/2, height/2],
                      n=[n_x, n_y])
        base_coords = _np.concatenate(
            [grid.coordinates, _np.zeros([grid.n_data, 1])], axis=1)

        azimuth = azimuth * _np.pi / 180
        dip = - dip * _np.pi / 180
        ry = _np.reshape(_np.array(
            [1, 0, 0,
             0, _np.cos(dip), -_np.sin(dip),
             0, _np.sin(dip), _np.cos(dip)]
        ), [3, 3])
        rz = _np.reshape(_np.array(
            [_np.cos(azimuth), _np.sin(azimuth), 0,
             -_np.sin(azimuth), _np.cos(azimuth), 0,
             0, 0, 1]
        ), [3, 3])
        rotation_matrix = _np.matmul(rz, ry).T

        center = _np.array(center, ndmin=2)
        rotated_coords = _np.matmul(base_coords, rotation_matrix) + center

        df = _pd.DataFrame(rotated_coords, columns=coordinate_labels)
        super().__init__(df, coordinate_labels)
        self.grid_shape = [n_x, n_y]


class Surface3D(_PointBased):
    def __init__(self, points, triangles, normals):
        super().__init__()

        if points.shape[1] != 3:
            raise ValueError("points must be an array with 3 columns")
        if triangles.shape[1] != 3:
            raise ValueError("triangles must be an array with 3 columns")
        if normals.shape[1] != 3:
            raise ValueError("normals must be an array with 3 columns")

        if triangles.shape[1] != normals.shape[1]:
            raise ValueError("triangles and normals must have the same"
                             "number of lines")

        self.coordinates = points
        self.triangles = triangles
        self.normals = normals

        self._n_dim = 3
        self._n_data = self.coordinates.shape[0]
        if self._n_data > 0:
            self._bounding_box = BoundingBox.from_array(self.coordinates)
        else:
            self._bounding_box = BoundingBox.from_array(
                _np.zeros([2, self.n_dim]))

    def export_micromine(self, points_filename="points",
                         triangles_filename="triangles",
                         offset=[0, 0, 0], **kwargs):
        points_df = [
            _pd.DataFrame({"id": _np.arange(self.n_data)}),
            _pd.DataFrame(self.coordinates, columns=["EAST", "NORTH", "RL"])
        ]
        for variable in self.variables.values():
            points_df.append(variable.as_data_frame(**kwargs))
        points_df = _pd.concat(points_df, axis=1)
        points_df["EAST"] += offset[0]
        points_df["NORTH"] += offset[1]
        points_df["RL"] += offset[2]
        points_df.to_csv(points_filename + ".csv", index=False)

        triangles_df = pd.DataFrame(
            self.triangles, columns=["PointId1", "PointId2", "PointId3"])
        triangles_df.to_csv(triangles_filename + ".csv", index=False)


def _blockdata(cls):
    # Decorator to extend functionality of some classes
    # Used to avoid multiple inheritance

    old_init = cls.__init__

    def new_init(self, start, n, step, discretization=None):
        old_init(self, start, n, step)
        if discretization is None:
            discretization = [1] * self.n_dim
        self.discretization = discretization

        self._bounding_box = BoundingBox(
            _np.min(self.coordinates, axis=0) - _np.array(self.step_size) / 2,
            _np.max(self.coordinates, axis=0) + _np.array(self.step_size) / 2,
        )

        sub_grid = _np.array(
            list(_iter.product(
                *[_np.arange(d) for d in self.discretization[::-1]]
            )),
            dtype=float)[:, ::-1]
        sub_grid -= (_np.array(self.discretization)[None, :] - 1) / 2
        sub_grid *= _np.array(self.step_size)[None, :]
        sub_grid /= (_np.array(self.discretization)[None, :] + 1)
        self.sub_grid = sub_grid

    def discretized_coordinates(self, index):
        center = _np.array([g[i] for g, i in zip(self.grid, index)])[None, :]
        return self.sub_grid + center

    def inducing_grid(self, index):
        center = _np.array([g[i] for g, i in zip(self.grid, index)])[None, :]
        grid = self.sub_grid
        discr = _np.array(self.discretization)[None, :]
        grid = grid * (discr + 1) / (discr - 1) + center
        return PointData.from_array(grid)

    cls.__init__ = new_init
    cls.discretized_coordinates = discretized_coordinates
    cls.inducing_grid = inducing_grid

    return cls


@_blockdata
class Blocks1D(Grid1D):
    pass


@_blockdata
class Blocks2D(Grid2D):
    pass


@_blockdata
class Blocks3D(Grid3D):
    pass
