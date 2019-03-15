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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as _tf
import numpy as _np


def pairwise_dist(A, B):
    """
    Computes pairwise distances between each elements of A and
    each elements of B.

    Args:
    A,    [m,d] matrix
    B,    [n,d] matrix

    Returns:
    D,    [m,n] matrix of pairwise distances

    code from
    https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
    """
    with _tf.variable_scope('pairwise_dist'):
        # squared norms of each row in A and B
        na = _tf.reduce_sum(_tf.square(A), 1)
        nb = _tf.reduce_sum(_tf.square(B), 1)

        # na as a row and nb as a column vectors
        na = _tf.reshape(na, [-1, 1])
        nb = _tf.reshape(nb, [1, -1])

        # return pairwise euclidean difference matrix
        D = na - 2 * _tf.matmul(A, B, False, True) + nb
        D = _tf.sqrt(_tf.maximum(D, 0.0))
    return D


def safe_chol(A):
    """
    Conditioning of a matrix for Cholesky decomposition
    """
    with _tf.variable_scope("safe_chol"):
        A = 0.5 * (A + _tf.transpose(A))
        e, v = _tf.self_adjoint_eig(A)
#        e = tf.where(e > 1e-14, e, 1e-14*tf.ones_like(e))
#        A = tf.matmul(tf.matmul(v,tf.matrix_diag(e),transpose_a=True),v)
        jitter = _tf.where(e > 1e-6, _tf.zeros_like(e),
                           1e-6 * _tf.ones_like(e))
        A = A + _tf.matrix_diag(jitter)
        L = _tf.linalg.cholesky(A)
    return L


def softmax(A):
    """
    Softmax by rows of a matrix
    """
    with _tf.variable_scope("softmax"):
        s = _tf.shape(A)
        E = _tf.exp(A)
        E_total = _tf.reduce_sum(E, axis=1, keepdims=True)
        E_total = _tf.tile(E_total, [1, s[1]]) + _np.exp(-10.0)
        out = E / E_total
    return out


def prod_n(inputs, name=None):
    """
    Multiplies all input tensors element-wise.

    Parameters
    ----------
    inputs : list
        A list of Tensor objects, all with the same shape and type.
    name : str
        A name for the operation (optional).

    Returns
    -------
    A Tensor of same shape and type as the elements of inputs.
    """
    if name is None:
        name = "prod_n"
    with _tf.name_scope(name):
        s = _tf.shape(inputs[0])
        out = _tf.ones_like(s, dtype=inputs[0].dtype)
        for tensor in inputs:
            out = out * tensor
        return out
