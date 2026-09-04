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
"""Type aliases shared by the annotated public interface.

The names here spell out the argument shapes the package accepts in more
than one form, so that a signature says what the docstring would otherwise
have to explain. They are aliases, not classes: nothing imports this module
at runtime except for the annotations themselves.

Only plain types live here. Anything whose type is a geoML class is
annotated with that class directly, in the module that already imports it,
which keeps this module free of the circular imports a central registry of
container and likelihood types would need.
"""
import os as _os
from collections.abc import Mapping, Sequence
from typing import Union

import numpy as _np
import numpy.typing as _npt

__all__ = ["ArrayLike", "FloatArray", "IndexArray", "PathLike", "Where",
           "Cutoffs", "Bins", "Labels", "Unit", "Units"]

#: Anything NumPy can turn into an array: a list, a tuple, a Series, an array.
ArrayLike = _npt.ArrayLike

#: An array of doubles, which is what every computation here produces.
FloatArray = _npt.NDArray[_np.float64]

#: Whole numbers that point at something -- which locations were kept, where
#: each value belongs -- rather than measuring it.
IndexArray = _npt.NDArray[_np.integer]

#: A filesystem location, as a string or as anything `os.fspath` accepts.
PathLike = Union[str, _os.PathLike]

#: Which locations an operation visits: one boolean per location, the
#: indices themselves, the name of a boolean metadata column, or `None` for
#: all of them.
Where = Union[_np.ndarray, Sequence[int], str, None]

#: The values a variable is judged against: one number or several.
Cutoffs = Union[float, Sequence[float], None]

#: How a range is divided: a count, which gives equal-count bins, or the
#: edges themselves.
Bins = Union[int, Sequence[float]]

#: The names of a variable's components or categories, in order.
Labels = Sequence[str]

# What a column is measured in: a name from `data.variables.UNITS`, or the
# number to divide by.
Unit = Union[str, float]

# Units for a variable's several components: one per label, or a mapping
# from label to unit. `None` leaves them undeclared.
Units = Union[Mapping[str, Unit], Sequence[Unit], None]
