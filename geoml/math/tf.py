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

import tensorflow as _tf


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
