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

import os as _os
import pandas as _pd
import numpy as _np

import geoml.data as _data


def walker():
    """
    Walker lake dataset.

    Returns
    -------
    walker_point : geoml.data.PointData
        Dataset with 470 samples.
    walker_grid : geoml.data.Grid2D
        Full data.
    """
    path = _os.path.dirname(__file__)
    path_walker = _os.path.join(path, "sample_data/walker.dat")
    path_walker_ex = _os.path.join(path, "sample_data/walker_ex.dat")

    walker_sample = _pd.read_table(path_walker, na_values=-999) * 1.0
    walker_ex = _pd.read_table(path_walker_ex, sep=",")

    walker_point = _data.PointData(walker_sample, ["X", "Y"])
    walker_point.add_continuous_variable("V", walker_sample["V"].values)
    walker_point.add_continuous_variable("U", walker_sample["U"].values)

    walker_grid = _data.Grid2D(start=[1, 1], n=[260, 300], step=[1, 1])
    walker_grid.add_continuous_variable("V", walker_ex["V"].values)
    walker_grid.add_continuous_variable("U", walker_ex["U"].values)

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
    point : geoml.data.PointData
        Some coordinates with two rock labels.
    dirs : geoml.data.DirectionalData
        Structural measurements_a representing a fold.
    """
    ex_point = _pd.DataFrame(
        {"X": _np.array([
            # rock a
            25, 40, 60, 85, 89, 76, 66, 74, 64, 50, 50, 31, 21, 25, 16, 30,
            # rock b
            5, 10, 45, 50, 55, 75, 90, 91, 74, 64,
            76, 66, 50, 50, 29, 19, 14, 25,
            # boundary
            15, 20, 30, 50, 65, 75, 90, 25, 50, 65, 75,
        ]),
         "Y": _np.array([
             # rock a
             25, 60, 50, 15, 19, 11, 21, 49, 64, 51, 84, 64, 34, 11, 16, 45,
             # rock b
             50, 80, 10, 30, 10, 75, 90, 21, 9, 19,
             51, 66, 49, 86, 66, 36, 14, 9,
             # boundary
             15, 35, 65, 85, 65, 50, 20, 10, 50, 20, 10,
         ]),
         "rock": _np.concatenate([_np.repeat("a", 16),
                                  _np.repeat("b", 18),
                                  _np.repeat("boundary", 11)])})
    ex_point["label_1"] = _np.concatenate(
        [_np.repeat("a", 16),
         _np.repeat("b", 18),
         _np.repeat("a", 11)])
    ex_point["label_2"] = _np.concatenate(
        [_np.repeat("a", 16),
         _np.repeat("b", 18),
         _np.repeat("b", 11)])

    ex_dir = _pd.DataFrame(
        {"X": _np.array([40, 50, 70, 90, 30, 20, 20]),
         "Y": _np.array([40, 85, 70, 30, 50, 60, 10]),
         "strike": _np.array([30, 90, 145, 145, 30, 30, 30])})
    ex_dir["azimuth"] = ex_dir["strike"] - 90

    point = _data.PointData(ex_point, ["X", "Y"])
    point.add_rock_type_variable("rock", labels=["a", "b"],
                                 measurements_a=ex_point["label_1"].values,
                                 measurements_b=ex_point["label_2"].values)

    vals = _np.ones(ex_point.shape[0])
    vals[ex_point["label_1"] == "b"] = -1
    vals[ex_point["label_1"] != ex_point["label_2"]] = 0
    point.add_continuous_variable("rock_num", vals)

    dirs = _data.DirectionalData.from_azimuth(
        ex_dir, ["X", "Y"], "strike"
    )

    normals = _data.DirectionalData.from_azimuth(
        ex_dir, ["X", "Y"], "azimuth"
    )

    return point, dirs, normals


def sunspot_number():
    """
    Sunspot number data.

    This data is downloaded from the Royal Observatory of Belgium
    SILSO website (http://sidc.be/silso/home), and is distributed under
    the CC BY-NC4.0 license (https://goo.gl/PXrLYd).

    Returns
    -------
    out - dict
        Dict containing the processed and original data.
    """
    yearly_df = _pd.read_table(
        "http://sidc.be/silso/INFO/snytotcsv.php",
        sep=";", header=None)
    yearly_df.set_axis(["year", "sn", "sn_std",
                        "n_obs", "definitive"],
                       axis="columns", inplace=True)
    # yearly = _data.Points1D(_np.arange(1700, yearly_df.shape[0] + 1700,
    #                                    dtype=float),
    #                         yearly_df)
    yearly = _data.PointData(yearly_df, "year")
    yearly.add_continuous_variable("sn", yearly_df["sn"].values)

    monthly_df = _pd.read_table(
        "http://sidc.oma.be/silso/INFO/snmtotcsv.php",
        sep=";", header=None)
    monthly_df.set_axis(["year", "month", "year_frac", "sn", "sn_std",
                         "n_obs", "definitive"],
                        axis="columns", inplace=True)
    monthly_df["idx"] = _np.arange(1, monthly_df.shape[0] + 1, dtype=float)
    # monthly = _data.Points1D(_np.arange(1, monthly_df.shape[0] + 1,
    #                                     dtype=float), monthly_df)
    monthly = _data.PointData(monthly_df, "idx")
    monthly.add_continuous_variable("sn", monthly_df["sn"].values)

    daily_df = _pd.read_table("http://sidc.oma.be/silso/INFO/sndtotcsv.php",
                              sep=";", header=None)
    daily_df.set_axis(["year", "month", "day", "year_frac", "sn", "sn_std",
                       "n_obs", "definitive"],
                      axis="columns", inplace=True)
    daily_df["idx"] = _np.arange(1, daily_df.shape[0] + 1, dtype=float)
    daily = _data.PointData(daily_df, "idx")
    daily.add_continuous_variable("sn", daily_df["sn"].values)

    out = {"points": {"yearly": yearly,
                      "monthly": monthly,
                      "daily": daily},
           "data_frames": {"yearly": yearly_df,
                           "monthly": monthly_df,
                           "daily": daily_df}}

    return out


def andrade():
    """
    Structural measurements in Cerro do Andrade, Caçapava do Sul, Brazil.

    Returns
    -------
    planes : geoml.data.DirectionalData
        Directions parallel to the foliation planes.
    normals : geoml.data.DirectionalData
        Normals to the foliation planes.
    """
    path = _os.path.dirname(__file__)
    file = _os.path.join(path, "sample_data/andrade.txt")

    raw_data = _pd.read_table(file, sep=",")

    planes = _data.DirectionalData.from_planes(
        raw_data, ["X", "Y", "Z"], "azimuth", "dip"
    )
    normals = _data.DirectionalData.from_normals(
        raw_data, ["X", "Y", "Z"], "azimuth", "dip"
    )

    return planes, normals, raw_data


def jura():
    """
    Jura mountains dataset (Goovaerts, 1997).

    Returns
    -------
    jura_train : geoml.data.PointData
        Training data (2 categorical variables and 7 continuous).
    jura_val : geoml.data.PointData
        Validation data (2 categorical variables and 7 continuous).
    """
    elements = ["Cd", "Co", "Cr", "Cu", "Ni", "Pb", "Zn"]

    path = _os.path.dirname(__file__)
    # file = _os.path.join(path, "sample_data/jura_sample.dat")
    #
    # raw_data = _pd.read_table(file, sep=" ", header=None,
    #                           skiprows=13, engine="python",
    #                           skipinitialspace=True)
    # raw_data.columns = ["X", "Y", "Landuse", "Rock"] + elements
    # raw_data["Landuse"] = raw_data["Landuse"].astype("str")
    # raw_data["Rock"] = raw_data["Rock"].astype("str")
    #
    # landuse_labels = _np.unique(raw_data["Landuse"])
    # rock_labels = _np.unique(raw_data["Rock"])
    #
    # jura_train = _data.PointData(raw_data, coordinates=["X", "Y"])
    # jura_train.add_categorical_variable("Landuse", landuse_labels,
    #                                     raw_data["Landuse"])
    # jura_train.add_categorical_variable("Rock", rock_labels, raw_data["Rock"])
    # for el in elements:
    #     jura_train.add_continuous_variable(el, raw_data[el])

    file_a = _os.path.join(path, "sample_data/jura_train.csv")

    train_df = _pd.read_csv(file_a)

    landuse_labels = _np.unique(train_df["Landuse"])
    rock_labels = _np.unique(train_df["Rock"])

    jura_train = _data.PointData(train_df, coordinates=["Xloc", "Yloc"])
    jura_train.add_categorical_variable("Landuse", landuse_labels,
                                        train_df["Landuse"])
    jura_train.add_categorical_variable("Rock", rock_labels, train_df["Rock"])
    for el in elements:
        jura_train.add_continuous_variable(el, train_df[el])

    file_b = _os.path.join(path, "sample_data/jura_val.csv")

    val_df = _pd.read_csv(file_b)

    jura_val = _data.PointData(val_df, coordinates=["Xloc", "Yloc"])
    jura_val.add_categorical_variable("Landuse", landuse_labels,
                                      val_df["Landuse"])
    jura_val.add_categorical_variable("Rock", rock_labels, val_df["Rock"])
    for el in elements:
        jura_val.add_continuous_variable(el, val_df[el])

    return jura_train, jura_val


def arctic_lake():
    """
    A compositional dataset.

    Returns
    -------
    arctic_lake_data : geoml.data.PointData

    References
    ----------
    Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R. (2015).
    Modeling and Analysis of Compositional Data. John Wiley & Sons.

    """
    path = _os.path.dirname(__file__)
    file = _os.path.join(path, "sample_data/Arctic_lake.csv")

    raw_data = _pd.read_csv(file)

    arctic_lake_data = _data.PointData(raw_data, ["Depth (m)"])
    arctic_lake_data.add_compositional_variable(
        "comp", labels=['Sand', 'Silt', 'Clay'],
        measurements=raw_data.values[:, :3] / 100)

    return arctic_lake_data
