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
        E = _tf.exp(A - _tf.reduce_max(A))
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
        # s = _tf.shape(inputs[0])
        # out = _tf.ones(s, dtype=inputs[0].dtype)
        # for tensor in inputs:
        #     out = out * tensor
        out = _tf.stack(inputs, axis=0)
        out = _tf.reduce_prod(out, axis=0)
        return out


def safe_logdet(mat):
    with _tf.name_scope("safe_det"):
        eig = _tf.linalg.eigvalsh(mat)
        min_eig = _tf.exp(_tf.constant(-10.0, dtype=_tf.float64))
        out = _tf.where(eig > min_eig,
                        eig,
                        _tf.ones_like(eig) * min_eig)
        out = _tf.reduce_sum(_tf.log(out))
        return out


def efficient_matrix_product(mat, nugget, b):
    """
    Computes efficient matrix products using the compressed feature
    matrix structure.

    Parameters
    ----------
    mat : Tensor
        A matrix.
    nugget : Tensor
        A vector, the diagonal of a nugget matrix.
    b : Tensor
        A generic matrix with matching shape.

    Returns
    -------
    out : Tensor
        The computed matrix product.

    This method computes tf.matmul(tf.matmul(mat, mat, False, True)
    + tf.diag(nugget), b) in an efficient manner, especially if mat has more
    rows than columns.
    """
    with _tf.name_scope("efficient_matrix_product"):
        # Efficient multiplication of nugget and b
        sb = _tf.shape(b)
        nug = _tf.reshape(nugget, [sb[0], 1])
        nug = _tf.tile(nug, [1, sb[1]])
        out_0 = nug * b

        # matrix multiplication
        out_1 = _tf.matmul(mat, b, True, False)
        out_1 = _tf.matmul(mat, out_1)
    return out_0 + out_1


def conjugate_gradient_block(mat, nugget, b, tol=1e-3, jitter=1e-9):
    """
    Conjugate gradient solver using the compressed feature matrix
    structure.

    Parameters
    ----------
    mat : Tensor
        A matrix.
    nugget : Tensor
        A vector, the diagonal of a nugget matrix.
    b : Tensor
        A generic matrix with matching shape.
    tol : double
        The tolerance for ending the loop. It is tested against the
        Frobenius norm of the residual matrix.
    jitter : double
        A small number to improve numerical instability.

    Returns
    -------
    x : Tensor
        The solution of the system Ax=b, where A is decomposed in
        feature matrix form (i.e. A=tf.matmul(mat, mat, False, True)).
    """
    with _tf.name_scope("conjugate_gradient"):
        # initialization
        r = -b
        p = -r
        x = _tf.zeros_like(b)
        rtr = _tf.matmul(r, r, True, False)
        i = _tf.constant(0)
        tol = _tf.constant(tol, _tf.float64)
        sx = _tf.shape(x)
        reg = _tf.eye(sx[1], dtype=_tf.float64) * jitter

        # TensorFlow loop
        def cond(i_, r_, p_, x_, rtr_):

            cond_0 = _tf.less(i_, sx[0])

            val = _tf.sqrt(_tf.reduce_mean(r_ * r_))
            cond_1 = _tf.greater(val, tol)

            return _tf.reduce_all(_tf.stack([cond_0, cond_1]))

        def body(i_, r_, p_, x_, rtr_):
            ap = efficient_matrix_product(mat, nugget, p_)
            alpha = _tf.linalg.solve(
                _tf.matmul(p_, ap, True, False) + reg,
                rtr_
            )
            x_ = x_ + _tf.matmul(p_, alpha)
            r_ = r_ + _tf.matmul(ap, alpha)
            rtr_new = _tf.matmul(r_, r_, True, False)
            p_ = -r_ + _tf.matmul(p_, _tf.linalg.solve(rtr_ + reg, rtr_new))
            return i_ + 1, r_, p_, x_, rtr_new

        out = _tf.while_loop(cond, body, (i, r, p, x, rtr))
    return out[3]


def efficient_logdet(mat, nugget, use_sylvester=True):
    """
    Efficient log-determinant computation.

    This method uses the special structure of the compressed feature
    matrices and Sylvester's determinant theorem to compute the
    Gaussian Process determinant in an efficient manner, allowing the
    use of a larger number of training points.

    Parameters
    ----------
    mat : Tensor
        A matrix.
    nugget : Tensor
        A vector, the diagonal of a nugget matrix.
    use_sylvester : bool
        Whether to use Sylvester's determinant theorem (faster for
        a large number of training data).

    Returns
    -------
    det : scalar tensor
        The computed log determinant.
    """
    if use_sylvester:
        with _tf.name_scope("determinant_sylvester"):
            # shape of feature tensor
            s = _tf.shape(mat)  # rows x columns
            eye = _tf.eye(s[1], s[1], dtype=_tf.float64)

            # nugget term
            det = _tf.reduce_sum(_tf.log(nugget), name="nugget_term")

            # features term
            nug = _tf.reshape(nugget, [s[0], 1])
            nug = _tf.tile(nug, [1, s[1]])
            prod = _tf.matmul(mat, mat / nug, True, False) + eye
            chol = _tf.linalg.cholesky(prod)
            det = det + 2.0*_tf.reduce_sum(_tf.log(_tf.matrix_diag_part(chol)))
        return det
    else:
        with _tf.name_scope("determinant_standard"):
            prod = _tf.matmul(mat, mat, False, True) + _tf.diag(nugget)
            chol = _tf.linalg.cholesky(prod)
            det = 2.0 * _tf.reduce_sum(_tf.log(_tf.diag_part(chol)))
        return det


def extract_features(mat, min_var):
    with _tf.name_scope("extract_features"):
        # eigen decomposition
        values, vectors = _tf.self_adjoint_eig(mat)
        values = _tf.reverse(values, axis=[0])
        vectors = _tf.reverse(vectors, axis=[1])

        # number of principal components needed to reach min_var
        total_var = _tf.cumsum(values)
        keep = _tf.where(_tf.greater(
            total_var, min_var * _tf.reduce_sum(values)))[0] + 1
        keep = _tf.squeeze(keep)

        # compressed matrix
        delta = _tf.diag(_tf.sqrt(values[0:keep]))
        q = _tf.matmul(vectors[:, 0:keep], delta)
    return q
