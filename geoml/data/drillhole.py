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

__all__ = ["DrillholeData", "IntervalTable"]

from collections.abc import Sequence
from typing import cast as _cast

import numpy as _np
import pandas as _pd
import copy as _copy
import warnings as _warnings
import pyvista as _pv

import geoml._types as _types
import geoml.data.containers as _data
import geoml.viz.plotly as _py

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoml.data.geoh5 import Workspace as _GeoH5Workspace


# canonical names for the three columns every interval table must have
HOLE = "HOLEID"
FROM = "FROM"
TO = "TO"

# what a sample's own length is called once it is a point, and where along
# the hole it sits -- what a contact profile measures from
LENGTH = "LENGTH"
DEPTH = "DEPTH"

ROLES = ("grade", "categorical", "density", "recovery", "flag", "ignore")

# what to divide a column by to turn it into a fraction of the whole
UNITS = {"fraction": 1.0, "ratio": 1.0, "1": 1.0,
         "%": 100.0, "pct": 100.0, "percent": 100.0, "wt%": 100.0,
         "ppm": 1e6, "g/t": 1e6, "mg/kg": 1e6,
         "ppb": 1e9, "ug/kg": 1e9, "mg/t": 1e9}

_TOL = 1e-6


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str) or not hasattr(x, "__iter__"):
        return [x]
    return list(x)


# --------------------------------------------------------------------------- #
# desurveying
# --------------------------------------------------------------------------- #
def _inclination(dip, positive_down=True):
    """
    Converts a dip angle in degrees to an inclination in radians, measured
    from the downward vertical (0 = straight down, pi/2 = horizontal).

    geoML's convention is that a positive dip points downwards, matching
    `DirectionalData.from_azimuth_and_dip()`. Many drillhole databases use the
    opposite sign, hence `positive_down`.
    """
    dip = _np.asarray(dip, dtype=float)
    if not positive_down:
        dip = -dip
    return _np.pi / 2 - dip * _np.pi / 180


def _min_curvature(md_0, md_1, inc_0, inc_1, azm_0, azm_1):
    """
    The minimum curvature step between two survey stations.

    The hole is assumed to follow a circular arc between the stations, which is
    the industry standard. Returns the easting, northing and vertical (positive
    downwards) displacements.
    """
    dogleg = _np.cos(inc_1 - inc_0) \
             - _np.sin(inc_0) * _np.sin(inc_1) * (1 - _np.cos(azm_1 - azm_0))
    dogleg = _np.arccos(_np.clip(dogleg, -1.0, 1.0))

    # the ratio factor tends to 1 as the arc straightens; guard the division
    straight = dogleg < 1e-9
    safe = _np.where(straight, 1.0, dogleg)
    ratio = _np.where(straight, 1.0, 2.0 / safe * _np.tan(safe / 2.0))

    half = 0.5 * (md_1 - md_0) * ratio
    east = half * (_np.sin(inc_0) * _np.sin(azm_0)
                   + _np.sin(inc_1) * _np.sin(azm_1))
    north = half * (_np.sin(inc_0) * _np.cos(azm_0)
                    + _np.sin(inc_1) * _np.cos(azm_1))
    vertical = half * (_np.cos(inc_0) + _np.cos(inc_1))
    return east, north, vertical


class _HoleTrace(object):
    """
    The path of a single hole, as a sequence of survey stations.

    Coordinates are never stored for the interval data; they are produced on
    demand by `coordinates_at()`, so compositing and re-sampling stay exact.
    """

    def __init__(self, collar, depth, inclination, azimuth):
        self.collar = _np.asarray(collar, dtype=float)
        self.depth = _np.asarray(depth, dtype=float)
        self.inclination = _np.asarray(inclination, dtype=float)
        self.azimuth = _np.asarray(azimuth, dtype=float)
        self.station_coordinates = self._integrate()

    def _integrate(self):
        n = len(self.depth)
        coordinates = _np.tile(self.collar, [n, 1])
        if n > 1:
            east, north, vertical = _min_curvature(
                self.depth[:-1], self.depth[1:],
                self.inclination[:-1], self.inclination[1:],
                self.azimuth[:-1], self.azimuth[1:])
            coordinates[1:, 0] = self.collar[0] + _np.cumsum(east)
            coordinates[1:, 1] = self.collar[1] + _np.cumsum(north)
            coordinates[1:, 2] = self.collar[2] - _np.cumsum(vertical)
        return coordinates

    def _attitude_at(self, depth, station):
        """Attitude interpolated between the bracketing stations."""
        following = _np.minimum(station + 1, len(self.depth) - 1)
        span = self.depth[following] - self.depth[station]
        position = _np.where(span > 0,
                             (depth - self.depth[station])
                             / _np.where(span > 0, span, 1.0),
                             0.0)
        position = _np.clip(position, 0.0, 1.0)

        inclination = self.inclination[station] + position * (
                self.inclination[following] - self.inclination[station])
        # the azimuth is taken through the shorter arc, so a hole crossing
        # north does not swing all the way around
        turn = (self.azimuth[following] - self.azimuth[station]
                + _np.pi) % (2 * _np.pi) - _np.pi
        azimuth = self.azimuth[station] + position * turn
        return inclination, azimuth

    def coordinates_at(self, depth):
        """
        The coordinates at the given depths along the hole.

        Depths beyond the last survey station continue in a straight line, in
        the direction of that station.
        """
        depth = _np.atleast_1d(_np.asarray(depth, dtype=float))
        station = _np.searchsorted(self.depth, depth, side="right") - 1
        station = _np.clip(station, 0, len(self.depth) - 1)

        inclination, azimuth = self._attitude_at(depth, station)
        east, north, vertical = _min_curvature(
            self.depth[station], depth,
            self.inclination[station], inclination,
            self.azimuth[station], azimuth)

        base = self.station_coordinates[station]
        return _np.stack([base[:, 0] + east,
                          base[:, 1] + north,
                          base[:, 2] - vertical], axis=1)


# --------------------------------------------------------------------------- #
# interval aggregation helpers
# --------------------------------------------------------------------------- #
def _overlap_pairs(target_from, target_to, source_from, source_to):
    """
    Matches target intervals to the source intervals they overlap.

    Both sets must be sorted by depth, and the source intervals must not
    overlap each other (which `IntervalTable.validate()` checks). Returns the
    matching target indices, source indices, and the length of each overlap.
    """
    first = _np.searchsorted(source_to, target_from, side="right")
    last = _np.searchsorted(source_from, target_to, side="left")
    counts = _np.maximum(last - first, 0)

    total = int(counts.sum())
    if total == 0:
        empty = _np.zeros(0, dtype=int)
        return empty, empty, _np.zeros(0)

    target_index = _np.repeat(_np.arange(len(target_from)), counts)
    offset = _np.arange(total) - _np.repeat(_np.cumsum(counts) - counts, counts)
    source_index = offset + _np.repeat(first, counts)

    overlap = _np.minimum(target_to[target_index], source_to[source_index]) \
              - _np.maximum(target_from[target_index], source_from[source_index])

    keep = overlap > 0
    return target_index[keep], source_index[keep], overlap[keep]


def _weighted_mean(values, target_index, source_index, weight, n_target):
    """Length-weighted mean, skipping missing values."""
    value = _np.asarray(values, dtype=float)[source_index]
    weight = _np.where(_np.isnan(value), 0.0, weight)

    total = _np.bincount(target_index, weights=weight * _np.nan_to_num(value),
                         minlength=n_target)
    norm = _np.bincount(target_index, weights=weight, minlength=n_target)

    with _np.errstate(invalid="ignore", divide="ignore"):
        out = total / norm
    out[norm <= 0] = _np.nan
    return out


def _weighted_majority(values, target_index, source_index, weight, n_target):
    """The category accounting for the greatest length within each target."""
    codes, labels = _pd.factorize(_pd.Series(values))
    n_labels = len(labels)
    out = _np.full(n_target, None, dtype=object)
    if n_labels == 0:
        return out

    code = codes[source_index]
    known = code >= 0
    if not known.any():
        return out

    flat = target_index[known] * n_labels + code[known]
    totals = _np.bincount(flat, weights=weight[known],
                          minlength=n_target * n_labels)
    totals = totals.reshape(n_target, n_labels)

    found = totals.sum(axis=1) > 0
    out[found] = _np.asarray(labels, dtype=object)[totals.argmax(axis=1)[found]]
    return out


def _longest_contributor(values, target_index, source_index, weight, n_target):
    """The value of the single longest interval contributing to each target."""
    out = _np.full(n_target, None, dtype=object)
    if len(target_index) == 0:
        return out

    # sorting by weight within each target makes the last write the largest
    order = _np.lexsort((weight, target_index))
    pick = _np.full(n_target, -1, dtype=int)
    pick[target_index[order]] = source_index[order]

    found = pick >= 0
    out[found] = _np.asarray(values, dtype=object)[pick[found]]
    return out


def _fixed_runs(zones, length):
    """
    Splits each zone into runs of the given length.

    A zone shorter than `length` yields a single run, and the residual of a
    longer zone is merged into its last run, so no stub intervals are created.
    """
    # a zone with a missing or inverted depth cannot be split; validate()
    # reports these, but only raises when asked to
    usable = _np.isfinite(zones[FROM].values) \
             & _np.isfinite(zones[TO].values) \
             & (zones[TO].values > zones[FROM].values)
    zones = zones.loc[usable]

    start = zones[FROM].values
    end = zones[TO].values
    counts = _np.maximum(
        1, _np.floor((end - start) / length + _TOL)).astype(int)

    zone_index = _np.repeat(_np.arange(len(start)), counts)
    step = _np.arange(int(counts.sum())) \
           - _np.repeat(_np.cumsum(counts) - counts, counts)

    run_from = start[zone_index] + step * length
    run_to = _np.where(step == counts[zone_index] - 1,
                       end[zone_index], run_from + length)

    # the originating zone is kept so callers can carry its attributes over
    out = _pd.DataFrame({HOLE: zones[HOLE].values[zone_index],
                         FROM: run_from, TO: run_to, "_zone": zone_index})
    return out


def _merge_runs(data, column):
    """Merges touching intervals of the same hole that share a category."""
    hole = data[HOLE].values
    start = data[FROM].values
    end = data[TO].values
    value = data[column].values

    opens = _np.ones(len(data), dtype=bool)
    if len(data) > 1:
        same = (value[1:] == value[:-1]) \
               | (_pd.isna(value[1:]) & _pd.isna(value[:-1]))
        opens[1:] = ~((hole[1:] == hole[:-1])
                      & (_np.abs(start[1:] - end[:-1]) <= _TOL)
                      & same)

    group = _np.cumsum(opens) - 1
    merged = _pd.DataFrame({HOLE: hole, FROM: start, TO: end,
                            column: value, "_run": group})
    merged = merged.groupby("_run", sort=True).agg(
        {HOLE: "first", FROM: "min", TO: "max", column: "first"})
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# interval table
# --------------------------------------------------------------------------- #
class IntervalTable(object):
    """
    One interval file, such as an assay or a lithology table.

    The table holds depths only -- coordinates come from the hole traces in the
    `DrillholeData` object that owns it, so re-compositing never accumulates
    positional error.

    Each value column is given a role, which decides how it is composited:

    ``grade``
        Numeric. Length-weighted mean, weighted also by the density column if
        one is declared. Missing values are skipped rather than propagated.
    ``categorical``
        Rock type, alteration, and the like. The category holding the greatest
        length within a composite wins.
    ``density``
        Numeric, and also the weight applied to the grades. Composited as a
        length-weighted mean of its own.
    ``recovery``
        Numeric: the share of each interval actually recovered. Composited as
        a length-weighted mean, like the density -- a fraction of a length is
        exactly what length-weighting averages -- and never applied as a
        weight to anything: how much a poorly recovered assay should count is
        a modelling decision, not a compositing one. Conversion carries it as
        *metadata* rather than as a variable, beside `HOLEID` and `LENGTH`:
        it describes the sample, and the models must not see it.
    ``flag``
        A categorical marker (drilling method, say) treated like a category.
    ``ignore``
        Carried through untouched: a composite takes the value of its single
        longest contributing interval. Free-text columns land here.

    Columns that are not declared explicitly are given the ``grade`` role if
    they are numeric and ``ignore`` otherwise; `set_role()` corrects them
    afterwards. `__str__` prints the roles in force.

    Attributes
    ----------
    data : DataFrame
        The intervals, sorted by hole and depth, with the hole, from and to
        columns renamed to HOLEID, FROM and TO.
    roles : dict
        Column name to role.
    name : str
        A label for the table, used in messages.
    """

    def __init__(self, data, hole="HOLEID", fr="FROM", to="TO",
                 grades=None, categorical=None, density=None, recovery=None,
                 flags=None, ignore=None, name=None):
        """
        Initializer for IntervalTable.

        Parameters
        ----------
        data : DataFrame
            The interval file.
        hole : str
            Column with the hole name.
        fr, to : str
            Columns with the start and end depths of each interval.
        grades, categorical, density, recovery, flags, ignore : str or list
            Columns to assign to each role.
        name : str
            A label for the table.
        """
        for column, label in ((hole, "hole"), (fr, "fr"), (to, "to")):
            if column not in data.columns:
                raise ValueError(
                    f"column {column} (given as {label}) is not in the data "
                    f"frame; found {list(data.columns)}")

        data = data.rename(columns={hole: HOLE, fr: FROM, to: TO}).copy()
        data[HOLE] = data[HOLE].astype(str).str.strip()
        data[FROM] = _pd.to_numeric(data[FROM], errors="coerce")
        data[TO] = _pd.to_numeric(data[TO], errors="coerce")

        self.data = data.sort_values([HOLE, FROM]).reset_index(drop=True)
        self.name = name
        self.roles = {}

        for column in self.value_columns:
            self.roles[column] = \
                "grade" if _pd.api.types.is_numeric_dtype(self.data[column]) \
                else "ignore"
        for role, columns in (("grade", grades), ("categorical", categorical),
                              ("density", density), ("recovery", recovery),
                              ("flag", flags), ("ignore", ignore)):
            for column in _as_list(columns):
                self.set_role(column, role)

    @classmethod
    def _from_canonical(cls, data, roles, name=None):
        """Builds a table from data that is already in canonical form."""
        new_table = cls.__new__(cls)
        new_table.data = data
        new_table.roles = dict(roles)
        new_table.name = name
        return new_table

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        s = "Object of class " + self.__class__.__name__
        if self.name is not None:
            s += f" named '{self.name}'"
        s += "\n\n"
        s += f"{len(self)} intervals in {len(self.holes)} holes, "
        s += f"totalling {self.length.sum():.1f} m\n\n"
        s += "Column roles:\n"
        for column in self.value_columns:
            s += f"  {column}: {self.roles[column]}\n"
        return s

    def __repr__(self):
        return self.__str__()

    @property
    def value_columns(self):
        return [c for c in self.data.columns if c not in (HOLE, FROM, TO)]

    @property
    def holes(self):
        return _pd.unique(self.data[HOLE])

    @property
    def length(self):
        return self.data[TO].values - self.data[FROM].values

    @property
    def density_column(self):
        columns = self.columns_with_role("density")
        if len(columns) > 1:
            raise ValueError(
                f"table {self.name} declares more than one density column: "
                f"{columns}")
        return columns[0] if len(columns) == 1 else None

    def columns_with_role(self, role: str) -> list[str]:
        if role not in ROLES:
            raise ValueError(f"unknown role {role}; expected one of {ROLES}")
        return [c for c in self.value_columns if self.roles.get(c) == role]

    def set_role(self, column: str, role: str) -> "IntervalTable":
        """Declares (or redeclares) how a column is to be composited."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role}; expected one of {ROLES}")
        if column not in self.data.columns:
            raise ValueError(
                f"column {column} is not in table {self.name}; found "
                f"{self.value_columns}")
        if column in (HOLE, FROM, TO):
            raise ValueError(f"column {column} is reserved")
        self.roles[column] = role
        return self

    def rename(self, columns: "dict[str, str]") -> "IntervalTable":
        """
        Renames value columns, carrying their roles with them.

        The roles are held beside the data, keyed by column name, so renaming
        the data frame's columns directly leaves them behind: the column
        quietly stops being a grade, and printing or compositing the table
        raises. Rename here instead — or rename the data frame before it is
        added, which is the other place where the two cannot drift apart.

        Column names become variable names in `as_point_data()`, so this is the
        last chance to tidy them before they reach a model.

        Parameters
        ----------
        columns : dict
            Maps each current column name to its new one.

        Returns
        -------
        self, so that calls can be chained, as with `set_role`.
        """
        columns = dict(columns)

        unknown = [c for c in columns if c not in self.data.columns]
        if len(unknown) > 0:
            raise ValueError(
                f"column(s) {sorted(map(str, unknown))} are not in table "
                f"{self.name}; found {self.value_columns}")

        touched = [c for c in list(columns) + list(columns.values())
                   if c in (HOLE, FROM, TO)]
        if len(touched) > 0:
            raise ValueError(
                f"column(s) {sorted(set(touched))} are reserved; the hole and "
                f"depth columns are named when the table is built")

        new_names = list(columns.values())
        kept = [c for c in self.data.columns if c not in columns]
        taken = sorted(set(new_names) & set(kept))
        repeated = sorted({c for c in new_names if new_names.count(c) > 1})
        if len(taken) > 0 or len(repeated) > 0:
            raise ValueError(
                f"the new names would collide: {taken + repeated} would name "
                f"more than one column")

        self.data = self.data.rename(columns=columns)
        self.roles = {columns.get(str(name), name): role
                      for name, role in self.roles.items()}
        return self

    def validate(self, on_error="warn", collar=None):
        """
        Checks the intervals for the problems that break compositing.

        Overlapping intervals, non-positive lengths, missing depths and holes
        with no collar are errors: the aggregation assumes intervals within a
        hole are sorted and disjoint. Gaps and intervals running past the end
        of the hole are only reported, since assay tables are legitimately
        incomplete.

        Parameters
        ----------
        on_error : str
            One of "warn", "raise" or "ignore".
        collar : DataFrame
            Collar table indexed by hole, used to check hole names and lengths.

        Returns
        -------
        report : DataFrame
            One row per problem found, with its hole, depths and severity.
        """
        if on_error not in ("warn", "raise", "ignore"):
            raise ValueError("on_error must be 'warn', 'raise' or 'ignore'")

        data = self.data
        hole = data[HOLE].values
        start = data[FROM].values
        end = data[TO].values
        found = []

        def report(mask, issue, severity):
            index = _np.where(mask)[0]
            if len(index) > 0:
                found.append(_pd.DataFrame(
                    {HOLE: hole[index], FROM: start[index], TO: end[index],
                     "issue": issue, "severity": severity}))

        report(_np.isnan(start) | _np.isnan(end), "missing depth", "error")

        with _np.errstate(invalid="ignore"):
            report(end <= start, "non-positive length", "error")

        if len(data) > 1:
            same_hole = hole[1:] == hole[:-1]
            with _np.errstate(invalid="ignore"):
                overlaps = same_hole & (start[1:] < end[:-1] - _TOL)
                gaps = same_hole & (start[1:] > end[:-1] + _TOL)
            report(_np.concatenate([[False], overlaps]),
                   "overlaps the previous interval", "error")
            report(_np.concatenate([[False], gaps]),
                   "gap after the previous interval", "warning")

        if collar is not None:
            report(~_pd.Index(hole).isin(collar.index),
                   "hole is not in the collar table", "error")
            if "LENGTH" in collar.columns:
                total = collar["LENGTH"].reindex(hole).values
                with _np.errstate(invalid="ignore"):
                    report(end > total + _TOL,
                           "interval runs past the end of the hole", "warning")

        if len(found) == 0:
            return _pd.DataFrame(
                columns=[HOLE, FROM, TO, "issue", "severity"])

        found = _pd.concat(found, ignore_index=True)
        if on_error != "ignore":
            self._announce(found, on_error)
        return found

    def _announce(self, found, on_error):
        summary = found.groupby(["severity", "issue"]).size()
        lines = [f"  {count} interval(s): {issue} ({severity})"
                 for (severity, issue), count in summary.items()]
        message = f"problems in interval table {self.name}:\n" \
                  + "\n".join(lines)

        if on_error == "raise" and (found["severity"] == "error").any():
            raise ValueError(message)
        _warnings.warn(message, stacklevel=3)

    def subset_holes(self, holes: "Sequence[str]") -> "IntervalTable":
        """A copy of this table containing the given holes only."""
        keep = self.data[HOLE].isin(list(holes))
        return self._from_canonical(
            self.data.loc[keep].reset_index(drop=True), self.roles, self.name)

    def category_legend(self, column: str) -> _pd.DataFrame:
        """
        The distinct values of a column, and how much of the hole each holds.

        This is the starting point for grouping categories without a GUI: write
        it out with `to_csv()`, edit the `group` column in a spreadsheet -- it
        starts as a copy of the label -- and read it back into
        `group_categories()`. The rows are sorted by length, so the codes that
        carry the deposit come first and a long legend can be worked down from
        the top.

        Parameters
        ----------
        column : str
            The column to describe.

        Intervals carrying no value get a row of their own, with a missing
        label, so the legend accounts for every interval in the table and the
        unlogged ground is visible while the grouping is being decided. It says
        nothing about the depths no interval covers at all --
        `DrillholeData.fill_unlogged()` turns those into intervals, and
        `validate()` reports them.

        Returns
        -------
        legend : DataFrame
            One row per value, with the number of intervals it appears in, the
            total length it accounts for, and the `group` column to edit.
        """
        if column not in self.data.columns:
            raise ValueError(
                f"column {column} is not in table {self.name}; found "
                f"{self.value_columns}")

        frame = _pd.DataFrame({"label": self.data[column].values,
                               "length": self.length})
        legend = frame.groupby("label", dropna=False).agg(
            n_intervals=("length", "size"), length=("length", "sum"))
        legend = legend.sort_values("length", ascending=False).reset_index()
        legend["group"] = legend["label"]
        return legend

    def group_categories(self, column, groups, new_column=None, other=None):
        """
        Lumps the values of a column into fewer categories.

        Logging codes are usually far too many to model directly, and which of
        them behave alike is a judgement only the geologist can make.

        Parameters
        ----------
        column : str
            The column to group.
        groups : dict or DataFrame
            Either a mapping from each new category to the values it takes in,
            as ``{"ore": ["MSST", "SLTST"], "waste": "SHALE"}``, or a legend
            from `category_legend()` with its `group` column edited.
        new_column : str
            Where to put the result. The column is replaced by default; naming
            a new one keeps the original codes alongside the groups.
        other : str
            What to call the values that no group claims. They keep their own
            label by default, with a warning naming them. Missing values stay
            missing either way.

        Returns
        -------
        table : IntervalTable
            A new table; this one is left alone.
        """
        if column not in self.data.columns:
            raise ValueError(
                f"column {column} is not in table {self.name}; found "
                f"{self.value_columns}")

        if isinstance(groups, _pd.DataFrame):
            if not {"label", "group"}.issubset(groups.columns):
                raise ValueError(
                    "a legend must carry the 'label' and 'group' columns that "
                    "category_legend() produces")
            groups = {name: list(rows["label"])
                      for name, rows in groups.groupby("group")}

        mapping = {}
        for name, members in groups.items():
            for label in _as_list(members):
                if mapping.get(label, name) != name:
                    raise ValueError(
                        f"value {label!r} was placed in both "
                        f"{mapping[label]!r} and {name!r}")
                mapping[label] = name

        value = _pd.Series(self.data[column].values)
        unknown = set(mapping) - set(value.dropna().unique())
        if len(unknown) > 0:
            _warnings.warn(
                f"{len(unknown)} value(s) named in the groups do not appear in "
                f"{column}: {sorted(map(str, unknown))[:5]}")

        grouped = value.map(mapping)
        loose = grouped.isna() & value.notna()
        if loose.any():
            left_out = sorted(map(str, set(value[loose])))
            if other is None:
                _warnings.warn(
                    f"{len(left_out)} value(s) of {column} were left out of "
                    f"the groups and kept their own label: {left_out[:5]}. "
                    f"Name them, or pass other= to lump them together")
                grouped = grouped.where(~loose, value)
            else:
                grouped = grouped.where(~loose, other)

        target = column if new_column is None else new_column
        data = self.data.copy()
        data[target] = grouped.values

        roles = dict(self.roles)
        roles[target] = "categorical"
        return self._from_canonical(data, roles, self.name)

    def aggregate_onto(self, targets):
        """
        Composites this table onto the given target intervals.

        Parameters
        ----------
        targets : DataFrame
            Intervals with the HOLEID, FROM and TO columns, sorted by hole and
            depth.

        Returns
        -------
        table : IntervalTable
            A new table on the target support, with the same column roles.
        """
        source = self.data
        source_rows = source.groupby(HOLE, sort=False).indices
        n_target = targets.shape[0]

        target_index, source_index, overlap = [], [], []
        for hole, rows in targets.groupby(HOLE, sort=False).indices.items():
            matching = source_rows.get(hole)
            if matching is None:
                continue
            local_target, local_source, local_overlap = _overlap_pairs(
                targets[FROM].values[rows], targets[TO].values[rows],
                source[FROM].values[matching], source[TO].values[matching])
            target_index.append(rows[local_target])
            source_index.append(matching[local_source])
            overlap.append(local_overlap)

        if len(target_index) > 0:
            target_index = _np.concatenate(target_index)
            source_index = _np.concatenate(source_index)
            overlap = _np.concatenate(overlap)
        else:
            target_index = source_index = _np.zeros(0, dtype=int)
            overlap = _np.zeros(0)

        out = _pd.DataFrame({HOLE: targets[HOLE].values,
                             FROM: targets[FROM].values,
                             TO: targets[TO].values})

        # grades are weighted by mass where a density is available, by length
        # otherwise; the density itself is only ever weighted by length
        density = self.density_column
        grade_weight = overlap
        if density is not None:
            value = source[density].values.astype(float)[source_index]
            grade_weight = overlap * _np.where(_np.isnan(value), 1.0, value)

        for column in self.columns_with_role("grade"):
            out[column] = _weighted_mean(
                source[column].values, target_index, source_index,
                grade_weight, n_target)
        # the density and the recovery are only ever weighted by length: the
        # first is what the mass weighting is built from, and the second is a
        # fraction of a length, which is exactly what length-weighting
        # averages
        for role in ("density", "recovery"):
            for column in self.columns_with_role(role):
                out[column] = _weighted_mean(
                    source[column].values, target_index, source_index,
                    overlap, n_target)
        for role in ("categorical", "flag"):
            for column in self.columns_with_role(role):
                out[column] = _weighted_majority(
                    source[column].values, target_index, source_index,
                    overlap, n_target)
        for column in self.columns_with_role("ignore"):
            out[column] = _longest_contributor(
                source[column].values, target_index, source_index,
                overlap, n_target)

        return self._from_canonical(out[self.data.columns], self.roles,
                                    self.name)


# --------------------------------------------------------------------------- #
# drillhole data
# --------------------------------------------------------------------------- #
class DrillholeData(_data._SpatialData):
    """
    Drillhole data: a set of collars, their traces, and any number of interval
    tables.

    This object is not fed to a model. It is a staging area in which the raw
    files are checked, desurveyed and composited, and from which point data is
    produced with `as_point_data()` or `as_classification_input()`.

    The hole path is computed by minimum curvature from the survey table. Holes
    with no survey are taken as straight, following the collar dip and azimuth,
    or vertical if those are absent.

    Attributes
    ----------
    collar : DataFrame
        Indexed by hole, with the X, Y and Z coordinates, the hole LENGTH, and
        the fallback DIP and AZIMUTH.
    intervals : dict
        Name to `IntervalTable`.
    """

    def __init__(self, collar, survey=None, hole="HOLEID", x="X", y="Y",
                 z="Z", length=None, dip=None, azimuth=None, depth="DEPTH",
                 dip_positive_down=True):
        """
        Initializer for DrillholeData.

        Parameters
        ----------
        collar : DataFrame
            One row per hole, with its coordinates.
        survey : DataFrame
            Downhole survey stations. Optional.
        hole : str
            Column with the hole name, in every table.
        x, y, z : str
            Columns in collar with the coordinates.
        length : str
            Column in collar with the total length of the hole. Optional, but
            needed to check intervals running past the end of a hole.
        dip, azimuth : str
            Columns with the attitude, in collar (as a fallback for holes with
            no survey) and in survey.
        depth : str
            Column in survey with the depth of each station.
        dip_positive_down : bool
            Whether a positive dip points downwards, as in the rest of geoML.
            Databases recording downward holes as negative need False here.

        Interval tables are added afterwards with `add_intervals()`.
        """
        super().__init__()
        self._n_dim = 3
        self.intervals = {}
        self.dip_positive_down = dip_positive_down

        for column in (hole, x, y, z):
            if column not in collar.columns:
                raise ValueError(
                    f"column {column} is not in the collar table; found "
                    f"{list(collar.columns)}")

        holes = collar[hole].astype(str).str.strip()
        if holes.duplicated().any():
            repeated = holes[holes.duplicated()].unique()
            raise ValueError(f"repeated holes in the collar table: "
                             f"{list(repeated)}")

        self.collar = _pd.DataFrame(
            {"X": collar[x].values.astype(float),
             "Y": collar[y].values.astype(float),
             "Z": collar[z].values.astype(float),
             "LENGTH": collar[length].values.astype(float)
             if length is not None else _np.nan,
             "DIP": collar[dip].values.astype(float)
             if dip is not None and dip in collar.columns else _np.nan,
             "AZIMUTH": collar[azimuth].values.astype(float)
             if azimuth is not None and azimuth in collar.columns else _np.nan},
            index=_pd.Index(holes.values, name=HOLE))

        self._traces = self._build_traces(survey, hole, depth, dip, azimuth)
        self._n_data = self.collar.shape[0]
        self._update_bounding_box()

    def _build_traces(self, survey, hole, depth, dip, azimuth):
        stations = {}
        if survey is not None:
            for column in (hole, depth):
                if column not in survey.columns:
                    raise ValueError(
                        f"column {column} is not in the survey table; found "
                        f"{list(survey.columns)}")
            if dip is None or azimuth is None:
                raise ValueError(
                    "dip and azimuth columns are needed to use a survey table")

            survey = _pd.DataFrame(
                {HOLE: survey[hole].astype(str).str.strip().values,
                 "DEPTH": _pd.to_numeric(survey[depth],
                                         errors="coerce").values,
                 "DIP": _pd.to_numeric(survey[dip], errors="coerce").values,
                 "AZIMUTH": _pd.to_numeric(survey[azimuth],
                                           errors="coerce").values})
            survey = survey.dropna().sort_values([HOLE, "DEPTH"])

            unknown = set(survey[HOLE]) - set(self.collar.index)
            if len(unknown) > 0:
                _warnings.warn(
                    f"{len(unknown)} hole(s) in the survey table have no "
                    f"collar and were dropped: {sorted(unknown)[:5]}")
                survey = survey[survey[HOLE].isin(self.collar.index)]
            stations = dict(tuple(survey.groupby(HOLE, sort=False)))

        traces = {}
        vertical = 0
        for name, row in self.collar.iterrows():
            table = stations.get(name)
            if table is not None and table.shape[0] > 0:
                station_depth = table["DEPTH"].values
                station_dip = table["DIP"].values
                station_azimuth = table["AZIMUTH"].values
            else:
                # a straight hole along the collar attitude; vertical when the
                # collar does not record one
                station_depth = _np.zeros(1)
                if _np.isnan(row["DIP"]):
                    vertical += 1
                    station_dip = _np.array(
                        [90.0 if self.dip_positive_down else -90.0])
                else:
                    station_dip = _np.array([row["DIP"]])
                station_azimuth = _np.array(
                    [0.0 if _np.isnan(row["AZIMUTH"]) else row["AZIMUTH"]])

            if station_depth[0] > _TOL:
                # the trace must start at the collar
                station_depth = _np.concatenate([[0.0], station_depth])
                station_dip = _np.concatenate([station_dip[:1], station_dip])
                station_azimuth = _np.concatenate(
                    [station_azimuth[:1], station_azimuth])

            traces[name] = _HoleTrace(
                row[["X", "Y", "Z"]].values.astype(float),
                station_depth,
                _inclination(station_dip, self.dip_positive_down),
                station_azimuth * _np.pi / 180)

        if vertical > 0:
            _warnings.warn(
                f"{vertical} hole(s) have neither survey nor collar dip, and "
                f"were taken as vertical")
        return traces

    def _desurveyed_points(self):
        """The holes as a point cloud: collars, stations, toes, interval ends.

        What the bounding box is measured from, and what a grid fitting a
        rotation to the drilling (`RotatedGrid3D.from_data`) reads -- the
        drillholes never expose `coordinates` of their own, being interval
        data rather than points.
        """
        points = [self.collar[["X", "Y", "Z"]].values]
        for name, trace in self._traces.items():
            points.append(trace.station_coordinates)

        toe = self.collar["LENGTH"]
        known = toe.notna()
        if known.any():
            points.append(self.coordinates_at(
                toe.index[known].values, toe[known].values))

        for table in self.intervals.values():
            points.append(self.coordinates_at(table.data[HOLE].values,
                                              table.data[TO].values))

        points = _np.concatenate(points, axis=0)
        return points[_np.all(_np.isfinite(points), axis=1)]

    def _update_bounding_box(self):
        self._bounding_box = _data.BoundingBox.from_array(
            self._desurveyed_points())

    def __str__(self):
        s = "Object of class " + self.__class__.__name__ + "\n\n"
        s += f"{self.n_holes} drillholes, "
        s += f"{self._survey_count} of them surveyed\n"
        total = self.collar["LENGTH"].sum()
        if _np.isfinite(total) and total > 0:
            s += f"{total:.1f} m drilled\n"
        s += "\n"
        if len(self.intervals) == 0:
            s += "No interval tables"
        else:
            s += "Interval tables:\n"
            for name, table in self.intervals.items():
                s += f"  {name}: {len(table)} intervals, columns " \
                     f"{table.value_columns}\n"
        return s

    def __repr__(self):
        # a container shows its summary when named on its own, as the ones in
        # `data` do
        return self.__str__()

    @property
    def n_holes(self):
        return self.collar.shape[0]

    @property
    def _survey_count(self):
        return sum(1 for t in self._traces.values() if len(t.depth) > 1)

    @classmethod
    def from_geoh5(cls, workspace: "_types.PathLike | _GeoH5Workspace",
                   name: "str | None" = None) -> "DrillholeData":
        """
        Reads drillholes from a geoh5 workspace.

        Every hole becomes a collar row named as the file names it, its
        survey stations come across with **geoh5's own dip convention
        converted** — there a vertical downward hole reads -90, so the
        object is built with `dip_positive_down=False` — and each named
        *property group* holding FROM/TO columns becomes one interval
        table of that name, gathered across the holes. Referenced
        (coded) columns take the categorical role; depth-associated
        readings, which cover no interval, are left out. `name` narrows
        the read to one drillhole group, or to one hole.

        Needs the `geoh5py` package: `pip install geoml[geoh5]`.

        Parameters
        ----------
        workspace
            Path of the workspace to read, or an open
            `geoml.data.geoh5.Workspace`.
        name
            A drillhole group, or a single hole, when the file holds
            more than one.

        Returns
        -------
        data : DrillholeData
            Ready for validation, compositing and conversion, like any
            other drillhole database.

        Raises
        ------
        ValueError
            If the workspace holds no drillholes, or none of the given
            name.
        """
        import geoml.data.geoh5 as _geoh5io
        collar, survey, tables = _geoh5io.read_drillholes(workspace, name)
        data = cls(collar, survey if len(survey) else None,
                   hole="HOLEID", x="X", y="Y", z="Z",
                   dip="DIP", azimuth="AZIMUTH", depth="DEPTH",
                   dip_positive_down=False)
        for label, frame in tables.items():
            text = [column for column in frame.columns
                    if column not in ("HOLEID", "FROM", "TO")
                    and not _pd.api.types.is_numeric_dtype(frame[column])]
            data.add_intervals(label, frame,
                               categorical=text if text else None)
        return data

    def add_intervals(self, name: str, data: _pd.DataFrame,
                      on_error: str = "warn",
                      **kwargs) -> "DrillholeData":
        """
        Adds an interval table.

        Parameters
        ----------
        name : str
            A label for the table, used to refer to it later.
        data : DataFrame or IntervalTable
            The interval file. A data frame is passed to `IntervalTable` along
            with the remaining arguments.
        on_error : str
            Passed to `IntervalTable.validate()`: "warn", "raise" or "ignore".

        Returns
        -------
        self, so that calls can be chained.
        """
        table = data if isinstance(data, IntervalTable) \
            else IntervalTable(data, name=name, **kwargs)
        table.name = name

        unknown = set(table.holes) - set(self.collar.index)
        if len(unknown) > 0:
            _warnings.warn(
                f"{len(unknown)} hole(s) in table {name} have no collar and "
                f"were dropped: {sorted(unknown)[:5]}")
            table = table.subset_holes(
                sorted(set(table.holes) - unknown))

        table.validate(on_error=on_error, collar=self.collar)
        self.intervals[name] = table
        self._update_bounding_box()
        return self

    def rename(self, table, columns):
        """
        Renames columns of one of the interval tables.

        The roles travel with the columns; see `IntervalTable.rename()`. This
        is the way to tidy the names of a database that was built for you — by
        `datasets.macpass()`, say — where the data frames were never in reach.

        Parameters
        ----------
        table : str
            Name of the interval table.
        columns : dict
            Maps each current column name to its new one.

        Returns
        -------
        self, so that calls can be chained.
        """
        if table not in self.intervals:
            raise ValueError(
                f"there is no table named {table}; found "
                f"{list(self.intervals.keys())}")
        self.intervals[table].rename(columns)
        return self

    def rename_table(self, name: str, new_name: str) -> "DrillholeData":
        """
        Renames an interval table, keeping its position.

        Parameters
        ----------
        name : str
            The table's current name.
        new_name : str
            What to call it instead.

        Returns
        -------
        self, so that calls can be chained.
        """
        if name not in self.intervals:
            raise ValueError(
                f"there is no table named {name}; found "
                f"{list(self.intervals.keys())}")
        if new_name != name and new_name in self.intervals:
            raise ValueError(
                f"there is already a table named {new_name}; drop it first "
                f"if it is to be replaced")
        # rebuilt rather than popped back in, so the table keeps its place --
        # the order is what `as_point_data` merges by default
        self.intervals = {new_name if key == name else key: table
                          for key, table in self.intervals.items()}
        self.intervals[new_name].name = new_name
        return self

    def drop_table(self, name: str) -> "DrillholeData":
        """
        Removes an interval table.

        Parameters
        ----------
        name : str
            The table to remove.

        Returns
        -------
        self, so that calls can be chained.
        """
        if name not in self.intervals:
            raise ValueError(
                f"there is no table named {name}; found "
                f"{list(self.intervals.keys())}")
        del self.intervals[name]
        self._update_bounding_box()
        return self

    def validate(self, on_error="warn"):
        """Validates every interval table. Returns a combined report."""
        found = [table.validate(on_error=on_error, collar=self.collar)
                 .assign(table=name)
                 for name, table in self.intervals.items()]
        # a clean table reports an empty frame, which carries no dtypes for the
        # concatenation to go by
        found = [report for report in found if len(report) > 0]
        if len(found) == 0:
            return _pd.DataFrame(
                columns=[HOLE, FROM, TO, "issue", "severity", "table"])
        return _pd.concat(found, ignore_index=True)

    def coordinates_at(self, holes: _types.ArrayLike,
                       depths: _types.ArrayLike) -> _np.ndarray:
        """
        The coordinates of the given depths in the given holes.

        Parameters
        ----------
        holes : array
            Hole name of each point.
        depths : array
            Depth of each point along its hole.

        Returns
        -------
        coordinates : array
            An (n_points, 3) array. Holes with no collar give NaN.
        """
        holes = _np.asarray(holes, dtype=object)
        depths = _np.asarray(depths, dtype=float)
        out = _np.full([len(depths), 3], _np.nan, dtype=float)

        for name in _pd.unique(holes):
            trace = self._traces.get(name)
            if trace is None:
                continue
            rows = _np.where(holes == name)[0]
            out[rows] = trace.coordinates_at(depths[rows])
        return out

    # ----------------------------------------------------------------------- #
    # compositing
    # ----------------------------------------------------------------------- #
    def _resolve_domain(self, domain):
        """Finds the table and column a domain refers to."""
        if isinstance(domain, (tuple, list)):
            name, column = domain
            return self.intervals[name], column

        if domain in self.intervals:
            table = self.intervals[domain]
            columns = table.columns_with_role("categorical")
            if len(columns) != 1:
                raise ValueError(
                    f"table {domain} has {len(columns)} categorical columns "
                    f"{columns}; name one explicitly, as ('{domain}', column)")
            return table, columns[0]

        for table in self.intervals.values():
            if domain in table.columns_with_role("categorical"):
                return table, domain

        raise ValueError(f"no categorical column or table named {domain}")

    def _hole_extent(self):
        """The depth range covered by the interval tables, per hole."""
        if len(self.intervals) == 0:
            raise _data.NoDataError("no interval tables have been added")

        covered = _pd.concat(
            [t.data[[HOLE, FROM, TO]] for t in self.intervals.values()],
            ignore_index=True)
        extent = covered.groupby(HOLE, sort=True).agg({FROM: "min", TO: "max"})
        return extent.reset_index()

    def composite(self, length: float = 1.0,
                  domain: str | None = None) -> "DrillholeData":
        """
        Composites every interval table onto a common support.

        Runs of the given length are laid out along each hole and every table
        is aggregated onto them, so the tables come out sharing one support and
        can be merged row for row by `as_point_data()`.

        Numeric columns are averaged weighted by length, or by length times
        density where the table declares a density column, skipping missing
        values. Categories are decided by the greatest length within the run.
        A run shorter than `length` is kept whole and the residual of a longer
        one is merged into its last run, so no stubs are produced.

        Parameters
        ----------
        length : float
            The length of the composites.
        domain : str or tuple
            A categorical column whose boundaries the composites must honour,
            given as a table name, a column name, or a (table, column) pair.
            Without it the runs ignore geology, which smears contacts.

        Returns
        -------
        drillholes : DrillholeData
            A new object with the same collars and traces.
        """
        if domain is None:
            zones = self._hole_extent()
        else:
            table, column = self._resolve_domain(domain)
            zones = _merge_runs(table.data, column)

        return self._composite_onto(_fixed_runs(zones, length))

    def composite_fixed(self, length: float = 1.0) -> "DrillholeData":
        """
        Composites onto runs of fixed length, ignoring any geology.

        The same as `composite()` with no domain; kept as a separate name
        because ignoring domain boundaries is a deliberate choice.
        """
        return self.composite(length=length, domain=None)

    def composite_to(self, table: str) -> "DrillholeData":
        """
        Composites every table onto the intervals of one of them.

        This is the way to bring assays onto the geological intervals that were
        logged, rather than onto an arbitrary run length.

        Parameters
        ----------
        table : str or IntervalTable
            The table whose intervals become the support.
        """
        source = self.intervals[table] if isinstance(table, str) else table
        targets = source.data[[HOLE, FROM, TO]].copy()
        return self._composite_onto(targets)

    def _composite_onto(self, targets):
        targets = targets.sort_values([HOLE, FROM]).reset_index(drop=True)

        new_object = _copy.copy(self)
        new_object.intervals = {
            name: table.aggregate_onto(targets)
            for name, table in self.intervals.items()}
        new_object._update_bounding_box()
        return new_object

    def merge_domains(self, domain):
        """
        Merges touching intervals sharing a category into single runs.

        Parameters
        ----------
        domain : str or tuple
            The categorical column, as in `composite()`.

        Returns
        -------
        table : IntervalTable
            A table with one interval per run.
        """
        table, column = self._resolve_domain(domain)
        merged = _merge_runs(table.data, column)
        return IntervalTable._from_canonical(
            merged, {column: "categorical"}, name=f"{table.name}_merged")

    # ----------------------------------------------------------------------- #
    # subsetting and grouping
    # ----------------------------------------------------------------------- #
    def subset_holes(self, holes):
        """
        The given holes only, with everything logged in them.

        Parameters
        ----------
        holes : str or list
            The names of the holes to keep.

        Returns
        -------
        drillholes : DrillholeData
            A new object with the selected collars, their traces, and every
            interval table cut down to them.
        """
        wanted = set(str(hole).strip() for hole in _as_list(holes))
        keep = [hole for hole in self.collar.index if hole in wanted]
        if len(keep) == 0:
            raise _data.NoDataError(
                "none of the given holes are in the collar table")

        unknown = wanted - set(keep)
        if len(unknown) > 0:
            _warnings.warn(
                f"{len(unknown)} of the given holes are not in the collar "
                f"table and were skipped: {sorted(unknown)[:5]}")

        new_object = _copy.copy(self)
        new_object.collar = self.collar.loc[keep]
        new_object._traces = {hole: self._traces[hole] for hole in keep}
        new_object.intervals = {name: table.subset_holes(keep)
                                for name, table in self.intervals.items()}
        new_object._n_data = len(keep)
        new_object._update_bounding_box()
        return new_object

    def subset_region(self, min_val, max_val):
        """
        The holes whose collar falls within a bounding box.

        The box is applied to the collars, not to the individual intervals, so
        a hole is kept or dropped whole: a deviated hole that wanders out of
        the box still comes through entire. Both bounds are inclusive.

        Parameters
        ----------
        min_val, max_val : array
            Opposite corners of the box, as [x, y] or [x, y, z]. Two values
            select in plan view, which is the usual way to carve out an area.
            The `min` and `max` of a `BoundingBox` can be passed directly.

        Returns
        -------
        drillholes : DrillholeData
        """
        min_val = _np.asarray(min_val, dtype=float).ravel()
        max_val = _np.asarray(max_val, dtype=float).ravel()
        if len(min_val) != len(max_val) or len(min_val) not in (2, 3):
            raise ValueError(
                "min_val and max_val must be the same length, either 2 (a box "
                "in plan view) or 3")

        position = self.collar[["X", "Y", "Z"]].values[:, :len(min_val)]
        inside = _np.all((position >= min_val) & (position <= max_val), axis=1)
        if not inside.any():
            raise _data.NoDataError("no collar falls within the given box")
        return self.subset_holes(self.collar.index[inside])

    def filter_intervals(self, table, column, values, whole_holes=False):
        """
        The intervals whose column holds one of the given values.

        Parameters
        ----------
        table : str
            The interval table to filter.
        column : str
            The column to look at.
        values : object or list
            The values to keep.
        whole_holes : bool
            Whether to keep every hole containing a matching interval, with all
            of its data, rather than the matching intervals alone. Cutting the
            intervals themselves leaves the tables on different supports, so
            the result has to be composited again before `as_point_data()`;
            with `whole_holes` nothing is cut but the holes.

        Returns
        -------
        drillholes : DrillholeData
        """
        source = self.intervals[table]
        if column not in source.data.columns:
            raise ValueError(
                f"column {column} is not in table {table}; found "
                f"{source.value_columns}")

        values = _as_list(values)
        match = source.data[column].isin(values).values
        if not match.any():
            raise _data.NoDataError(
                f"no interval of {table} has {column} in {values}")

        if whole_holes:
            return self.subset_holes(
                _pd.unique(source.data[HOLE].values[match]))

        new_object = _copy.copy(self)
        new_object.intervals = dict(self.intervals)
        new_object.intervals[table] = IntervalTable._from_canonical(
            source.data.loc[match].reset_index(drop=True), source.roles,
            source.name)
        new_object._update_bounding_box()
        return new_object

    def category_legend(self, domain):
        """
        The distinct values of a categorical column, and how much each holds.

        See `IntervalTable.category_legend()`. The column is named as in
        `composite()`.
        """
        table, column = self._resolve_domain(domain)
        return table.category_legend(column)

    def group_categories(self, domain, groups, new_column=None, other=None):
        """
        Lumps the values of a categorical column into fewer categories.

        See `IntervalTable.group_categories()`. The column is named as in
        `composite()`.

        Returns
        -------
        drillholes : DrillholeData
            A new object, with the table holding the column replaced.
        """
        table, column = self._resolve_domain(domain)
        name = next(key for key, value in self.intervals.items()
                    if value is table)

        new_object = _copy.copy(self)
        new_object.intervals = dict(self.intervals)
        new_object.intervals[name] = table.group_categories(
            column, groups, new_column=new_column, other=other)
        return new_object

    def fill_unlogged(self, domain, label, ends=True):
        """
        Gives the ground a log does not account for a category of its own.

        Where only the intervals of interest are logged -- the mineralised
        ones, say -- everything else is left implicit, and a model has no way
        of knowing it was called anything. This makes the statement explicit,
        so the unlogged ground can be modelled as the waste it is, in the two
        ways it can be missing:

        1. Intervals that exist but carry no value, which take `label`.
        2. Depths no interval covers, which become new intervals carrying
           `label` and nothing else. With `ends` this reaches from the collar
           to the bottom of the hole, so a hole with no intervals at all comes
           out logged from end to end.

        The result covers each hole continuously, so `validate()` finds no
        gaps in it. The table changes support, though, so the object has to be
        composited again before `as_point_data()`.

        Parameters
        ----------
        domain : str or tuple
            The categorical column, as in `composite()`.
        label : str
            The category to give the unlogged ground. Required: it asserts
            something about the rock that nobody wrote down.
        ends : bool
            Whether to fill from the collar down to the first interval and
            from the last one to the bottom of the hole. The bottom is the
            LENGTH in the collar table, so holes with no recorded length keep
            whatever runs past their last interval.

        Returns
        -------
        drillholes : DrillholeData
            A new object, with the table holding the column replaced.
        """
        table, column = self._resolve_domain(domain)
        name = next(key for key, value in self.intervals.items()
                    if value is table)

        data = table.data
        filled = data.copy()
        filled.loc[_pd.isna(filled[column]), column] = label

        hole = data[HOLE].values
        start = data[FROM].values
        end = data[TO].values
        added = []

        if len(data) > 1:
            with _np.errstate(invalid="ignore"):
                gap = (hole[1:] == hole[:-1]) & (start[1:] > end[:-1] + _TOL)
            above = _np.where(gap)[0]
            added.append(_pd.DataFrame({HOLE: hole[above], FROM: end[above],
                                        TO: start[above + 1]}))

        if ends:
            covered = data.groupby(HOLE).agg(top=(FROM, "min"),
                                             bottom=(TO, "max"))
            extent = self.collar[["LENGTH"]].join(covered)
            # a hole with nothing logged in it is covered from 0 to 0, and so
            # is filled in one piece by the trailing interval below
            top = extent["top"].fillna(0.0).values
            bottom = extent["bottom"].fillna(0.0).values
            toe = extent["LENGTH"].values
            names = extent.index.values

            with _np.errstate(invalid="ignore"):
                leading = top > _TOL
                trailing = _np.isfinite(toe) & (toe > bottom + _TOL)
            added.append(_pd.DataFrame({HOLE: names[leading], FROM: 0.0,
                                        TO: top[leading]}))
            added.append(_pd.DataFrame({HOLE: names[trailing],
                                        FROM: bottom[trailing],
                                        TO: toe[trailing]}))

            empty = int(covered.reindex(self.collar.index)["top"].isna().sum())
            if empty > 0:
                _warnings.warn(
                    f"{empty} hole(s) have no interval in table {name} at all, "
                    f"and were logged as {label!r} from end to end")
            unknown = int(_np.sum(~_np.isfinite(toe)))
            if unknown > 0:
                _warnings.warn(
                    f"{unknown} hole(s) have no recorded length, so nothing "
                    f"was added past their last interval")

        added = [frame for frame in added if len(frame) > 0]
        if len(added) > 0:
            added = _pd.concat(added, ignore_index=True)
            added[column] = label
            filled = _pd.concat([filled, added], ignore_index=True)
        filled = filled.sort_values([HOLE, FROM]).reset_index(drop=True)

        new_object = _copy.copy(self)
        new_object.intervals = dict(self.intervals)
        new_object.intervals[name] = IntervalTable._from_canonical(
            filled, table.roles, table.name)
        new_object._update_bounding_box()
        return new_object

    # ----------------------------------------------------------------------- #
    # conversion
    # ----------------------------------------------------------------------- #
    def _common_support(self, tables):
        """Checks the tables share one support and returns it."""
        reference = tables[0]
        for table in tables[1:]:
            same = len(table) == len(reference)
            if same:
                same = _np.array_equal(table.data[HOLE].values,
                                       reference.data[HOLE].values) \
                       and _np.allclose(table.data[FROM].values,
                                        reference.data[FROM].values) \
                       and _np.allclose(table.data[TO].values,
                                        reference.data[TO].values)
            if not same:
                raise ValueError(
                    f"tables {reference.name} and {table.name} are on "
                    f"different supports; composite the object first, with "
                    f"composite() or composite_to()")
        return reference.data[[HOLE, FROM, TO]]

    def as_point_data(self, tables=None, position=0.5, compositional=None,
                      vector=None, drop_missing=True):
        """
        Converts the interval data to points at the centre of each interval.

        Every requested table must be on the same support, which compositing
        produces; the tables are then merged row for row. Numeric columns
        become continuous variables and categorical ones become categorical
        variables, unless they are claimed by a compositional or vector group.

        The hole each point came from, its depth down that hole and the
        length it stands for are carried as the metadata columns `HOLEID`,
        `DEPTH` and `LENGTH`, which the models never see: they are what a
        leave-one-hole-out split, a contact profile and a support-weighted
        statistic read. Columns with the ``recovery`` role ride beside them,
        for the same reason -- they describe the sample, not the ground.

        Parameters
        ----------
        tables : str or list
            Which interval tables to convert. All of them by default.
        position : float
            Where in each interval the point sits, from 0 (top) to 1 (bottom).
            The default puts it at the centre.
        compositional : dict
            Groups of columns to convert to compositional variables, as
            ``{name: {"columns": {"Pb_pct": "%", "Ag_ppm": "ppm"}, "rest":
            True}}``. Each column is named with its unit, which is needed to
            add the parts up; see `UNITS`, or give a number to divide by. A
            row missing any part is marked missing entirely; non-positive
            parts are replaced by half the smallest positive value of their
            column, since a log-ratio transform cannot take a zero; and with
            "rest" a further part is added holding whatever is left of the
            whole, so that the composition sums to one. Without "rest" the
            parts are closed instead.
        vector : dict
            Groups of columns to convert to vector variables, as
            ``{name: [columns]}``.
        drop_missing : bool
            Whether to drop points whose values are all missing, which happens
            where a composite falls in a gap.

        Returns
        -------
        point : geoml.data.PointData
        """
        tables = self._select_tables(tables)
        support = self._common_support(tables)

        depth = support[FROM].values \
                + position * (support[TO].values - support[FROM].values)
        coordinates = self.coordinates_at(support[HOLE].values, depth)

        merged = _pd.concat(
            [support.reset_index(drop=True)]
            + [t.data[t.value_columns].reset_index(drop=True) for t in tables],
            axis=1)

        claimed = set()
        for group in (compositional or {}).values():
            claimed.update(group["columns"] if isinstance(group, dict)
                           else group)
        for columns in (vector or {}).values():
            claimed.update(columns)

        numeric, categorical = [], []
        for table in tables:
            numeric.extend(c for c in table.columns_with_role("grade")
                           + table.columns_with_role("density")
                           if c not in claimed)
            categorical.extend(c for c in table.columns_with_role("categorical")
                               if c not in claimed)

        keep = _np.all(_np.isfinite(coordinates), axis=1)
        if drop_missing and len(numeric) > 0:
            values = merged[numeric].values.astype(float)
            keep = keep & ~_np.all(_np.isnan(values), axis=1)
        coordinates, merged = coordinates[keep], merged.loc[keep] \
            .reset_index(drop=True)

        frame = _pd.concat(
            [_pd.DataFrame(coordinates, columns=["X", "Y", "Z"]), merged],
            axis=1)
        point = _data.PointData(frame, ["X", "Y", "Z"])

        # where the sample came from, which the models never see: the hole is
        # what a leave-one-hole-out split needs, and the length is the support
        # the value stands for. The recovery is of the same kind -- it
        # describes the sample rather than the ground, so it rides beside
        # them instead of becoming a variable
        point.add_metadata(HOLE, merged[HOLE].values)
        point.add_metadata(DEPTH, merged[FROM].values + position
                           * (merged[TO].values - merged[FROM].values))
        point.add_metadata(LENGTH, merged[TO].values - merged[FROM].values)
        for table in tables:
            for column in table.columns_with_role("recovery"):
                point.add_metadata(column,
                                   frame[column].values.astype(float))

        for column in numeric:
            point.add_continuous_variable(
                column, frame[column].values.astype(float))
        for column in categorical:
            labels = _pd.unique(frame[column].dropna())
            point.add_categorical_variable(
                column, labels, measurements=frame[column].values)
        for name, columns in (vector or {}).items():
            point.add_vector_variable(
                name, list(columns),
                frame[list(columns)].values.astype(float))
        for name, group in (compositional or {}).items():
            columns, parts = _prepare_composition(frame, group)
            point.add_compositional_variable(name, columns, parts)

        return point

    def _select_tables(self, tables):
        if tables is None:
            tables = list(self.intervals.keys())
        if isinstance(tables, str):
            tables = [tables]
        if len(tables) == 0:
            raise _data.NoDataError("no interval tables to convert")
        return [self.intervals[t] if isinstance(t, str) else t
                for t in tables]

    def get_contacts(self, domain: "str | tuple[str, str]"
                     ) -> "_data.PointData":
        """
        The points where the category changes, as a rock type variable.

        Each contact carries the category above it and the category below it,
        which is what an implicit model needs to place a boundary. The hole it
        came from and its depth down it are carried as the metadata columns
        `HOLEID` and `DEPTH`; a contact has no length, so none is recorded.

        Parameters
        ----------
        domain : str or tuple
            The categorical column, as in `composite()`.

        Returns
        -------
        point : geoml.data.PointData
        """
        table, column = self._resolve_domain(domain)
        runs = _merge_runs(table.data, column)

        hole = runs[HOLE].values
        value = runs[column].values
        touching = (hole[1:] == hole[:-1]) \
                   & (_np.abs(runs[FROM].values[1:]
                              - runs[TO].values[:-1]) <= _TOL) \
                   & (value[1:] != value[:-1])
        above = _np.where(touching)[0]
        if len(above) == 0:
            raise _data.NoDataError(f"no contacts found in {column}")
        below = above + 1

        coordinates = self.coordinates_at(hole[above], runs[TO].values[above])
        frame = _pd.DataFrame(coordinates, columns=["X", "Y", "Z"])
        frame[HOLE] = hole[above]

        keep = _np.all(_np.isfinite(coordinates), axis=1)
        frame = frame.loc[keep].reset_index(drop=True)

        point = _data.PointData(frame, ["X", "Y", "Z"])
        point.add_metadata(HOLE, frame[HOLE].values)
        point.add_metadata(DEPTH, runs[TO].values[above][keep])
        point.add_rock_type_variable(
            column, labels=_pd.unique(value[~_pd.isna(value)]),
            measurements_a=value[above][keep],
            measurements_b=value[below][keep])
        return point

    def as_classification_input(self, domain: "str | tuple[str, str]",
                                length: float = 5.0,
                                label_order: "Sequence[str] | None" = None
                                ) -> "_data.PointData":
        """
        Converts a categorical column to point data for a classification model.

        Unlike the compositing done for grades, this treats the category as
        known at a point rather than over an interval: points are dropped along
        each run of constant category, and the contacts between runs are added
        as points carrying both neighbouring categories. A boundary therefore
        enters the model with zero effective support, which is what lets an
        implicit model honour it exactly.

        The hole and the length are carried as the metadata columns `HOLEID`
        and `LENGTH`, the latter being zero at the contacts.

        Parameters
        ----------
        domain : str or tuple
            The categorical column, as in `composite()`.
        length : float
            Spacing of the points dropped along each run.
        label_order : list
            The order of the categories, for an ordered rock type. The order
            found in the data by default.

        Returns
        -------
        point : geoml.data.PointData
        """
        table, column = self._resolve_domain(domain)
        runs = _merge_runs(table.data, column)

        interior = _fixed_runs(runs[[HOLE, FROM, TO]], length)
        depth = 0.5 * (interior[FROM].values + interior[TO].values)

        coordinates = self.coordinates_at(interior[HOLE].values, depth)
        value = runs[column].values[interior["_zone"].values]

        keep = _np.all(_np.isfinite(coordinates), axis=1) & ~_pd.isna(value)
        frame = _pd.DataFrame(coordinates[keep], columns=["X", "Y", "Z"])
        frame[column + "_a"] = value[keep]
        frame[column + "_b"] = value[keep]
        frame[HOLE] = interior[HOLE].values[keep]
        frame[LENGTH] = interior[TO].values[keep] - interior[FROM].values[keep]

        contacts = self.get_contacts(domain)
        contact_frame = _pd.DataFrame(_np.asarray(contacts.coordinates),
                                      columns=["X", "Y", "Z"])
        # `get_contacts` builds a rock-type variable, which is the one
        # kind holding the pair of classes a contact lies between
        contact_variable = _cast("_data.RockTypeVariable",
                                 contacts.variables[column])
        contact_frame[column + "_a"] = \
            contact_variable.measurements_a.to_numpy()
        contact_frame[column + "_b"] = \
            contact_variable.measurements_b.to_numpy()
        contact_frame[HOLE] = contacts.get_metadata(HOLE)
        # a contact is a point, not an interval — the zero support is the
        # whole reason it is added
        contact_frame[LENGTH] = 0.0

        frame = _pd.concat([frame, contact_frame], ignore_index=True)

        labels = label_order if label_order is not None \
            else _pd.unique(runs[column].dropna())
        point = _data.PointData(frame, ["X", "Y", "Z"])
        point.add_metadata(HOLE, frame[HOLE].values)
        point.add_metadata(LENGTH, frame[LENGTH].values)
        point.add_rock_type_variable(
            column, labels=labels,
            measurements_a=frame[column + "_a"].values,
            measurements_b=frame[column + "_b"].values,
            ordered=label_order is not None)
        return point

    # ----------------------------------------------------------------------- #
    # visualization
    # ----------------------------------------------------------------------- #
    def as_pyvista(self, table=None):
        """
        The interval data as a pyvista object, one line per interval.

        Parameters
        ----------
        table : str
            Which interval table to export. The first one by default.
        """
        table = self._select_tables(table)[0]
        data = table.data

        start = self.coordinates_at(data[HOLE].values, data[FROM].values)
        end = self.coordinates_at(data[HOLE].values, data[TO].values)

        keep = _np.all(_np.isfinite(start), axis=1) \
               & _np.all(_np.isfinite(end), axis=1)
        start, end, data = start[keep], end[keep], data.loc[keep]

        n = start.shape[0]
        points = _np.empty([2 * n, 3])
        points[0::2] = start
        points[1::2] = end
        lines = _np.column_stack([_np.full(n, 2), _np.arange(0, 2 * n, 2),
                                  _np.arange(1, 2 * n, 2)])

        drillholes = _pv.PolyData(points, lines=lines)
        drillholes.cell_data[HOLE] = data[HOLE].values.astype(str)
        for column in table.value_columns:
            values = data[column]
            if values.isna().all():
                continue
            if values.dtype == "object":
                values = values.astype(str).str.normalize("NFKD") \
                    .str.encode("ascii", errors="ignore").str.decode("utf-8")
            drillholes.cell_data[column] = values.values
        return drillholes

    def draw_categorical(self, domain, colors, **kwargs):
        """Draws the merged runs of a categorical column with plotly."""
        table, column = self._resolve_domain(domain)
        runs = _merge_runs(table.data, column)

        start = self.coordinates_at(runs[HOLE].values, runs[FROM].values)
        end = self.coordinates_at(runs[HOLE].values, runs[TO].values)

        n = runs.shape[0]
        points = _np.empty([2 * n, 3])
        points[0::2] = start
        points[1::2] = end
        values = _np.repeat(runs[column].values, 2)

        return _py.segments_3d(points, values, colors, **kwargs)


def _divisor(unit):
    """What to divide a column by to turn its unit into a fraction."""
    if isinstance(unit, (int, float)) and not isinstance(unit, bool):
        if unit <= 0:
            raise ValueError(f"a unit divisor must be positive, got {unit}")
        return float(unit)

    key = str(unit).strip().lower()
    if key not in UNITS:
        raise ValueError(
            f"unknown unit {unit!r}; expected one of {sorted(UNITS)}, or a "
            f"number to divide the column by")
    return UNITS[key]


def _prepare_composition(frame, group):
    """
    Builds the parts of a compositional variable.

    Handles the processing a raw assay table needs before it can be modelled,
    in the order the steps have to happen:

    1. Every column is divided by its unit, so all the parts become fractions
       of the same whole and can be added up. This is why the unit of each
       column has to be declared.
    2. A row missing any one of its parts is marked missing entirely. The
       parts of a composition only carry information relative to each other,
       so a row that is short of one of them cannot be used.
    3. Non-positive parts are replaced by half the smallest positive value of
       their own column, the usual substitution for values below detection.
       A log-ratio transform cannot take a zero.
    4. The rest is added, holding whatever is left of the whole: one minus the
       sum of the other parts. Where the parts leave no room -- they already
       account for everything, or for more than everything -- the rest is held
       at ``min_rest``, the smallest positive part found anywhere, and those
       samples are scaled down to fit. Every part is then strictly positive
       and the composition sums to one.
    """
    if not isinstance(group, dict) or not isinstance(group.get("columns"),
                                                     dict):
        raise ValueError(
            "the columns of a composition must be given as a mapping from "
            "column name to unit, as {'Pb_pct': '%', 'Ag_ppm': 'ppm'}; the "
            f"units are needed to add the parts up. Expected one of "
            f"{sorted(UNITS)}, or a number to divide the column by")

    columns = list(group["columns"].keys())
    rest = group.get("rest", False)

    parts = frame[columns].values.astype(float)
    for i, column in enumerate(columns):
        parts[:, i] = parts[:, i] / _divisor(group["columns"][column])

    missing = _np.any(_np.isnan(parts), axis=1)
    if missing.any():
        partly = missing.sum() - int(_np.all(_np.isnan(parts), axis=1).sum())
        if partly > 0:
            _warnings.warn(
                f"{partly} sample(s) are missing some but not all parts of "
                f"the composition; they were marked missing entirely")
        parts[missing] = _np.nan

    for i, column in enumerate(columns):
        with _np.errstate(invalid="ignore"):
            replace = parts[:, i] <= 0
        if not replace.any():
            continue
        positive = parts[parts[:, i] > 0, i]
        if positive.size == 0:
            _warnings.warn(
                f"every value of {column} is non-positive, so there is no "
                f"scale to replace them with; they were left alone")
            continue
        parts[replace, i] = 0.5 * positive.min()

    if not rest:
        # a missing row sums to NaN, and so stays missing
        with _np.errstate(invalid="ignore", divide="ignore"):
            return columns, parts / parts.sum(axis=1, keepdims=True)

    with _np.errstate(invalid="ignore"):
        positive = parts[parts > 0]
    if positive.size == 0:
        raise ValueError(
            "the composition has no positive value anywhere, so there is no "
            "room to place a rest in")
    minimum = positive.min()

    total = parts.sum(axis=1)
    residual = 1.0 - total
    with _np.errstate(invalid="ignore"):
        crowded = residual < minimum
    if crowded.any():
        _warnings.warn(
            f"{crowded.sum()} sample(s) leave no room for a rest, their parts "
            f"already accounting for the whole; they were scaled down so that "
            f"the rest could take the smallest part found, {minimum:.3g}. "
            f"Check the units if this affects many samples")
        parts[crowded] *= ((1.0 - minimum) / total[crowded])[:, None]
        residual = _np.where(crowded, minimum, residual)

    parts = _np.concatenate([parts, residual[:, None]], axis=1)
    return columns + ["rest"], parts
