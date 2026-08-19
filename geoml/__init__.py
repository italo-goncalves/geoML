__version__ = '0.6.7'
__author__ = 'Ítalo Gomes Gonçalves'

# TensorFlow's C++ INFO wall -- device initialization, XLA compilation,
# ptxas register spills -- is noise to a modelling session. Filtered here
# because it only works before TensorFlow is first imported, which the
# imports below do; `setdefault` leaves a user's own setting alone, and an
# explicit TF_CPP_MIN_LOG_LEVEL=0 brings everything back.
import importlib as _importlib
import os as _os
import warnings as _warnings

from geoml import _deprecated

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

# The 0.6.0 shims are resolved on demand rather than imported here. They
# used to be imported eagerly, so that `geoml.geometry` kept working as an
# attribute after the module moved -- but now that each one announces its
# own deprecation, importing all ten up front would fire ten warnings at
# every `import geoml`, including for the great majority of users who never
# touch an old path. A module-level `__getattr__` (PEP 562) keeps the
# attribute working and moves the warning to the moment someone actually
# reaches for it. `import geoml.tftools` still warns from the shim itself.
def __getattr__(name):
    if name in _deprecated.MOVED:
        # The shim warns from its own module body, which is right for
        # `import geoml.tftools` and wrong here: a warning attributed to
        # geoml/tftools.py is one Python's default filter hides, where the
        # same warning attributed to the caller is shown. So the import is
        # silenced and the warning re-issued from here, pointing at whoever
        # asked -- and the user gets one warning rather than two.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", DeprecationWarning)
            module = _importlib.import_module("geoml." + name)
        _deprecated.warn_moved(name)
        return module
    raise AttributeError("module 'geoml' has no attribute %r" % name)


def __dir__():
    return sorted(list(globals()) + list(_deprecated.MOVED))
# Internal, unadvertised, but kept reachable as attributes: parameter and
# storage arrive through the import graph regardless; persistence does not.
from geoml import persistence

# The everyday names, one import up: containers and models. Kernels,
# likelihoods and warpings stay module-qualified -- `Gaussian` alone names
# three different things.
from geoml.data import (PointData, Grid1D, Grid2D, Grid3D, BlockSet3D,
                        DrillholeData)
from geoml.models import VGPNetwork

# TensorFlow's retracing notice, for this package's own graphs only, is
# noise: the retraces it reports are deliberate, and the message carries the
# repr of the function it names, which for a bound method is the entire
# model. `geoml.math.tf.silence_retracing_notices(False)` puts them back.
from geoml.math.tf import silence_retracing_notices as _silence_retracing
_silence_retracing()
