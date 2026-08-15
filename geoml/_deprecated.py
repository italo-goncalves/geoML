# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
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

"""One warning, said the same way by every 0.6.0 shim.

The shims kept the pre-0.6.0 module paths importable when the package was
split into subpackages. They said "for one release" in a docstring and
nowhere else, so nobody importing one has ever been told at runtime; this
is that notice, ahead of removing them in 0.7.0.
"""

import warnings as _warnings

# Old path -> where it went. The one source of truth for both the warning
# text and the package's lazy `__getattr__`, so a shim cannot be renamed in
# one place and missed in the other.
MOVED = {
    "drillhole": "geoml.data.drillhole",
    "geometry": "geoml.math.geometry",
    "graphviz": "geoml.viz.graphviz",
    "inducing": "geoml.data.inducing",
    "interpolation": "geoml.math.interpolate",
    "plotly": "geoml.viz.plotly",
    "probability": "geoml.stats.probability",
    "pyvista": "geoml.viz.pyvista",
    "random": "geoml.stats.random",
    "tftools": "geoml.math.tf and geoml.math.linalg",
}


def warn_moved(old, stacklevel=3):
    """Announces that `geoml.<old>` is a 0.6.0 alias due for removal.

    Parameters
    ----------
    old : str
        The bare module name, as a key of `MOVED`.
    stacklevel : int
        How far up to attribute the warning. The default points past this
        function and its caller, at the code that reached for the old path
        -- which is what decides whether Python's default filter shows it
        at all, since `DeprecationWarning` is displayed when it is
        attributed to `__main__`.
    """
    _warnings.warn(
        "geoml.%s is a deprecated alias kept from the 0.6.0 reorganization "
        "and will be removed in 0.7.0; import %s instead."
        % (old, MOVED[old]),
        DeprecationWarning, stacklevel=stacklevel)
