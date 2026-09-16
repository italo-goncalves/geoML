__version__ = '0.6.13'
__author__ = 'Ítalo Gomes Gonçalves'

# TensorFlow's C++ INFO wall -- device initialization, XLA compilation,
# ptxas register spills -- is noise to a modelling session. Filtered here
# because it only works before TensorFlow is first imported, which the
# imports below do; `setdefault` leaves a user's own setting alone, and an
# explicit TF_CPP_MIN_LOG_LEVEL=0 brings everything back.
import os as _os

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# The public surface: the modules a user reaches for. Internal machinery
# (parameter, persistence, storage) stays importable but unadvertised.
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

# Internal, unadvertised, but kept reachable as attributes: parameter and
# storage arrive through the import graph regardless; persistence does not.
from geoml import persistence

# The everyday names, one import up: containers and models. Kernels,
# likelihoods and warpings stay module-qualified -- `Gaussian` alone names
# three different things.
from geoml.data import (PointData, Grid1D, Grid2D, Grid3D, BlockSet3D,
                        DrillholeData)
from geoml.models import VGPNetwork

# What a long call is doing, and how to stop it. The module is private so
# that `geoml.progress` is the context manager rather than a module holding
# one of that name.
from geoml._progress import Cancelled, Progress, progress

# TensorFlow's retracing notice, for this package's own graphs only, is
# noise: the retraces it reports are deliberate, and the message carries the
# repr of the function it names, which for a bound method is the entire
# model. `geoml.math.tf.silence_retracing_notices(False)` puts them back.
from geoml.math.tf import silence_retracing_notices as _silence_retracing
_silence_retracing()
