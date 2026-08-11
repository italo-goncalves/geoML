# geoML - machine learning models for geospatial data
# Copyright (C) 2021  Ítalo Gomes Gonçalves
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
"""
Latent-variable networks: the modelling paradigms a `VGPNetwork` composes.

`network` is the inducing-point paradigm and the one this namespace
re-exports -- `geoml.latent.BasicGP` and its siblings resolve here, which is
also what lets a saved model rebuild. `fourier` (variational Fourier
features) is an orthogonal paradigm kept alongside it; import it explicitly
as `geoml.latent.fourier` -- it is not re-exported.
"""
from geoml.latent.network import *
from geoml.latent import network
