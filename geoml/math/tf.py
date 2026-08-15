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
"""TensorFlow helpers in everyday use. The larger numerical machinery
(solvers, Lanczos, Kronecker products) is `geoml.math.linalg`."""

import logging as _logging

import tensorflow as _tf


# The traced functions whose retracing notices this package answers for.
# `_predict_raw` retraces on a new batch shape, a new `n_sim` and either
# value of `include_noise`, all of which are baked into the graph on
# purpose; `refresh_cached`'s inner function takes no arguments at all and
# is traced once per network, so a session holding many models (a
# cross-validation, a manual) trips TensorFlow's counter without anything
# being retraced twice.
_RETRACING = "triggered tf.function retracing"
_OURS = ("_predict_raw", "refresh_cached")


class _RetracingFilter(_logging.Filter):
    """Drops TensorFlow's retracing notice for this package's own graphs.

    The notice is worth having in general and is noise here, for two
    reasons. It fires on retraces that are deliberate, since the quantities
    that trigger them are the ones a prediction is allowed to vary; and it
    interpolates the *repr of the function it names*, which for a bound
    method is the whole model -- every node, every parameter -- so a single
    notice can run to hundreds of lines and bury the output it appears in.

    Only messages naming this package's functions are dropped. Anything
    TensorFlow says about a user's own `tf.function` still comes through.
    """

    def filter(self, record):
        message = record.getMessage()
        if _RETRACING not in message:
            return True
        return not any(name in message for name in _OURS)


_installed = _RetracingFilter()


def silence_retracing_notices(silence: bool = True) -> None:
    """
    Whether to drop TensorFlow's retracing notice for geoML's own graphs.

    On by default, and installed when `geoml` is imported. Call it with
    `False` to hear them, which is worth doing if a prediction seems to be
    spending its time compiling rather than computing.

    Parameters
    ----------
    silence
        `True` to drop the notices, `False` to let them through.

    See Also
    --------
    geoml.models.GPOptions : `jit_predict`, the other tracing-related knob.
    """
    logger = _tf.get_logger()
    logger.removeFilter(_installed)
    if silence:
        logger.addFilter(_installed)


@_tf.function(
    input_signature=[_tf.TensorSpec(shape=[None, None], dtype=_tf.float64),
                     _tf.TensorSpec(shape=[None, None], dtype=_tf.float64)])
def pairwise_dist(mat_a, mat_b):
    """
    Computes pairwise distances between each elements of matrix and
    each elements of mat_b.

    Args:
    mat_a,    [m,d] matrix
    mat_b,    [n,d] matrix

    Returns:
    dist,    [m,n] matrix of pairwise distances

    code from
    https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
    """
    with _tf.name_scope('pairwise_dist'):
        # squared norms of each row in matrix and mat_b
        na = _tf.reduce_sum(_tf.square(mat_a), 1)
        nb = _tf.reduce_sum(_tf.square(mat_b), 1)

        # na as a row and nb as a column vectors
        na = _tf.reshape(na, [-1, 1])
        nb = _tf.reshape(nb, [1, -1])

        # return pairwise euclidean difference matrix
        dist = na - 2 * _tf.matmul(mat_a, mat_b, False, True) + nb
        dist = _tf.sqrt(_tf.maximum(dist, 0.0))
    return dist


def training_step(optimizer, loss, variables):
    # For some reason optimizer.minimize() is not present in TensorFlow v2.11 onwards
    with _tf.GradientTape() as tape:
        result = loss()

    grads = tape.gradient(result, variables)
    optimizer.apply_gradients(zip(grads, variables))


def ensure_rank_2(x):
    x = _tf.cond(_tf.equal(_tf.rank(x), 1),
                 lambda: x[:, None],
                 lambda: x)
    return x


def batched_dataset(y_data, batch_size, shuffle=True):
    n_data = y_data.shape[0]
    # 1. Create a dataset from the large y_data array
    ds_data = _tf.data.Dataset.from_tensor_slices(y_data)

    # 2. Create a dataset of the indices (0, 1, 2, ... n_data-1)
    ds_indices = _tf.data.Dataset.range(n_data)

    # 3. Zip them together
    ds = _tf.data.Dataset.zip((ds_data, ds_indices))

    # 4. Shuffle (buffer_size=n_data) and batch
    if shuffle:
        ds = ds.shuffle(n_data)
    ds = ds.batch(batch_size)

    # 5. Prefetch for performance
    ds = ds.prefetch(_tf.data.AUTOTUNE)

    return ds
