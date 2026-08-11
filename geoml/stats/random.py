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

"""
The generator that every random initialization draws from.

Parameters are initialized at random when an object is built — a latent node,
a kernel's transform, an orthonormal matrix — which happens before any model
exists to carry `options.seed`. So the draws come from one generator held
here, seeded with `set_seed` before the objects are built:

    import geoml
    geoml.set_seed(1234)

    latent = geoml.latent.BasicGP(data, kernel=kernel)
    model = geoml.models.VGPNetwork(data, variables, likelihoods, latent)

Building the same objects again after the same call gives the same starting
parameters. Left alone, the generator is seeded from the operating system, so
each run differs — as it did before.

Training and prediction do not draw from here: their Monte Carlo sampling is
already reproducible from `options.seed` through stateless TensorFlow
sampling.
"""

import numpy as _np

_rng = _np.random.default_rng()


def set_seed(seed):
    """
    Seeds the generator used to initialize parameters.

    Parameters
    ----------
    seed : int
        The seed. `None` draws a fresh one from the operating system, which is
        how the generator starts out.

    Notes
    -----
    This must be called before the objects are built, since that is when the
    initial values are drawn. It replaces the generator, so anything already
    built keeps the values it was given.
    """
    global _rng
    _rng = _np.random.default_rng(seed)


def rng():
    """
    The generator every random initialization draws from.

    Returns
    -------
    numpy.random.Generator
    """
    return _rng
