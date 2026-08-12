__version__ = '0.6.1'
__author__ = 'Ítalo Gomes Gonçalves'

# TensorFlow's C++ INFO wall -- device initialization, XLA compilation,
# ptxas register spills -- is noise to a modelling session. Filtered here
# because it only works before TensorFlow is first imported, which the
# imports below do; `setdefault` leaves a user's own setting alone, and an
# explicit TF_CPP_MIN_LOG_LEVEL=0 brings everything back.
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# The public surface: the modules a user reaches for. Internal machinery
# (parameter, persistence, storage) and the one-release deprecation shims
# stay importable but unadvertised.
__all__ = [
    'data',
    'datasets',
    'kernels',
    'likelihood',
    'latent',
    'math',
    'metrics',
    'models',
    'plots',
    'stats',
    'transform',
    'viz',
    'warping',
]

from . import *
from .stats.random import set_seed

# Deprecated shims (moved in 0.6.0), imported so the old attribute paths
# keep working for one release: geoml.drillhole -> geoml.data.drillhole,
# geoml.inducing -> geoml.data.inducing, geoml.geometry / interpolation /
# tftools -> geoml.math.*, geoml.probability / random -> geoml.stats.*,
# geoml.plotly / pyvista / graphviz -> geoml.viz.*.
from geoml import (drillhole, geometry, graphviz, inducing, interpolation,
                   plotly, probability, pyvista, random, tftools)
# Internal, unadvertised, but kept reachable as attributes: parameter and
# storage arrive through the import graph regardless; persistence does not.
from geoml import persistence

# The everyday names, one import up: containers and models. Kernels,
# likelihoods and warpings stay module-qualified -- `Gaussian` alone names
# three different things.
from geoml.data import (PointData, Grid1D, Grid2D, Grid3D, BlockSet3D,
                        DrillholeData)
from geoml.models import VGPNetwork
