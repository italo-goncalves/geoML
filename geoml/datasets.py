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
# MERCHANTABILITY or FITNESS FOR matrix PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os as _os
import pandas as _pd
import numpy as _np

import geoml.data as _data


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
    path_walker = _os.path.join(path, "sample_data/walker.dat")
    path_walker_ex = _os.path.join(path, "sample_data/walker_ex.dat")

    walker_sample = _pd.read_table(path_walker)
    walker_ex = _pd.read_table(path_walker_ex, sep=",")

    walker_point = _data.Points2D(walker_sample[["X", "Y"]],
                                  walker_sample.drop(["X", "Y"], axis=1))
    walker_grid = _data.Grid2D(start=[1, 1], n=[260, 300], step=[1, 1])
    walker_grid.data = walker_ex.drop(["X", "Y"], axis=1)

    return walker_point, walker_grid


def ararangua():
    """
    Drillhole data from Araranguá town, Brazil.

    Returns
    -------
    ara_dh : geoml.data.DrillholeData
        A dataset with 13 drillholes.
    """
    path = _os.path.dirname(__file__)
    file = _os.path.join(path, "sample_data/Araranguá.xlsx")

    ara_lito = _pd.read_excel(file, sheet_name="Lito")
    ara_collar = _pd.read_excel(file, sheet_name="Collar")

    ara_dh = _data.DrillholeData(ara_collar,
                                 ara_lito,
                                 holeid="Hole ID",
                                 fr="From",
                                 to="To")
    return ara_dh


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

    point = _data.Points2D(ex_point[["X", "Y"]],
                           ex_point.drop(["X", "Y"], axis=1))
    dirs = _data.Directions2D.from_azimuth(ex_dir[["X", "Y"]], ex_dir["strike"])

    return point, dirs


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
    yearly = _data.Points1D(_np.arange(1700, yearly_df.shape[0] + 1700,
                                       dtype=_np.float),
                            yearly_df)

    monthly_df = _pd.read_table(
        "http://sidc.oma.be/silso/INFO/snmtotcsv.php",
        sep=";", header=None)
    monthly_df.set_axis(["year", "month", "year_frac", "sn", "sn_std",
                         "n_obs", "definitive"],
                        axis="columns", inplace=True)
    monthly = _data.Points1D(_np.arange(1, monthly_df.shape[0] + 1,
                                        dtype=_np.float), monthly_df)

    daily_df = _pd.read_table("http://sidc.oma.be/silso/INFO/sndtotcsv.php",
                              sep=";", header=None)
    daily_df.set_axis(["year", "month", "day", "year_frac", "sn", "sn_std",
                       "n_obs", "definitive"],
                      axis="columns", inplace=True)
    daily = _data.Points1D(_np.arange(1, daily_df.shape[0] + 1,
                                      dtype=_np.float), daily_df)

    return yearly, monthly, daily


def andrade():
    """
    Structural measurements in Cerro do Andrade, Caçapava do Sul, Brazil.

    Returns
    -------
    points : geoml.data.Points3D
        Base point data format.
    normals : geoml.data.Directions3D
        Normals to the foliation planes.
    """
    path = _os.path.dirname(__file__)
    file = _os.path.join(path, "sample_data/andrade.txt")

    raw_data = _pd.read_table(file)

    points = _data.Points3D(
        coords=raw_data[["X", "Y", "Z"]],
        data=raw_data.drop(["X", "Y", "Z"], axis=1))

    normals = _data.Directions3D.plane_normal(
        coords=raw_data[["X", "Y", "Z"]],
        azimuth=raw_data["dip_dir"],
        dip=raw_data["dip"])

    return points, normals
