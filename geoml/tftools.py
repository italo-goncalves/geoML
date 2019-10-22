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
    Softmax by rows of a matrix.
    """
    with _tf.variable_scope("softmax"):
        s = _tf.shape(A)
        E = _tf.exp(A - _tf.reduce_max(A))
        E_total = _tf.reduce_sum(E, axis=1, keepdims=True)
        E_total = _tf.tile(E_total, [1, s[1]]) + _np.exp(-10.0)
        out = E / E_total
    return out


def composition_close(A):
    """
    Divides each row of a matrix by its sum.
    """
    with _tf.variable_scope("composition_close"):
        s = _tf.shape(A)
        total = _tf.reduce_sum(A, axis=1, keepdims=True)
        total = _tf.tile(total, [1, s[1]]) + _np.exp(-10.0)
        out = A / total
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


def conjugate_gradient(matmul_fun, b, x_0=None, tol=1e-3, jitter=1e-9,
                       max_iter=100):
    """
    Conjugate gradient solver.

    Parameters
    ----------
    matmul_fun : function
        Function to compute A times b. Must accept and return a single Tensor
        with the same shape as b.
    b : Tensor
        A vector with matching shape.
    x_0 : Tensor
        An initial guess at the solution. Defaults to zero.
    tol : double
        The tolerance for ending the loop. It is tested against the
        Frobenius norm of the residual matrix.
    jitter : double
        A small number to improve numerical stability.

    Returns
    -------
    x : Tensor
        The solution of the system Ax=b, where A is implicitly provided in
        matmul_fun.
    """
    with _tf.name_scope("conjugate_gradient"):
        # initialization
        # r = -b
        if x_0 is None:
            x_0 = _tf.zeros_like(b)
        r = matmul_fun(x_0) - b
        p = -r
        # x = _tf.zeros_like(b)
        x = x_0
        rtr = _tf.reduce_sum(r * r)
        i = _tf.constant(0)
        tol = _tf.constant(tol, b.dtype)
        sx = _tf.shape(x)

        # TensorFlow loop
        def cond(i_, r_, p_, x_, rtr_):

            cond_0 = _tf.less(i_, sx[0])

            val = _tf.sqrt(_tf.reduce_mean(r_ * r_))
            cond_1 = _tf.greater(val, tol)

            cond_2 = _tf.less(i_, max_iter)

            return _tf.reduce_all(_tf.stack([cond_0, cond_1, cond_2]))

        def body(i_, r_, p_, x_, rtr_):
            ap = matmul_fun(p_)
            alpha = rtr_ / (_tf.reduce_sum(p_ * ap)) # + jitter)
            x_ = x_ + alpha * p_
            r_ = r_ + alpha * ap
            rtr_new = _tf.reduce_sum(r_ * r_)
            p_ = -r_ + p_ * rtr_new / (rtr_) # + jitter)
            return i_ + 1, r_, p_, x_, rtr_new

        out = _tf.while_loop(cond, body, (i, r, p, x, rtr))
    return out[3]


# def conjugate_gradient_pre_cond(matmul_fun, pre_comp_matmul_fun,
#                                 b, x_0=None, tol=1e-3, jitter=1e-9):
#     """
#     Conjugate gradient solver with preconditioning.
#
#     Parameters
#     ----------
#     matmul_fun : function
#         Function to compute A times b. Must accept and return a single Tensor
#         with the same shape as b.
#     pre_comp_matmul_fun : function
#         Function to compute the product of the preconditioning matrix with a
#         vector.
#     b : Tensor
#         A vector with matching shape.
#     x_0 : Tensor
#         An initial guess at the solution. Defaults to zero.
#     tol : double
#         The tolerance for ending the loop. It is tested against the
#         Frobenius norm of the residual matrix.
#     jitter : double
#         A small number to improve numerical stability.
#
#     Returns
#     -------
#     x : Tensor
#         The solution of the system Ax=b, where A is implicitly provided in
#         matmul_fun.
#     """
#     with _tf.name_scope("conjugate_gradient"):
#         # initialization
#         if x_0 is None:
#             x_0 = _tf.zeros_like(b)
#         r = matmul_fun(x_0) - b
#         p = -r
#         x = x_0
#         z = pre_comp_matmul_fun(r)
#         rtr = _tf.reduce_sum(z * r)
#         i = _tf.constant(0)
#         tol = _tf.constant(tol, b.dtype)
#         sx = _tf.shape(x)
#
#         # TensorFlow loop
#         def cond(i_, r_, p_, x_, rtr_):
#
#             cond_0 = _tf.less(i_, sx[0])
#
#             val = _tf.sqrt(_tf.reduce_mean(r_ * r_))
#             cond_1 = _tf.greater(val, tol)
#
#             return _tf.reduce_all(_tf.stack([cond_0, cond_1]))
#
#         def body(i_, r_, p_, x_, rtr_):
#             ap = matmul_fun(p_)
#             alpha = rtr_ / (_tf.reduce_sum(p_ * ap) + jitter)
#             x_ = x_ + alpha * p_
#             r_ = r_ + alpha * ap
#             z_ = pre_comp_matmul_fun(r_)
#             rtr_new = _tf.reduce_sum(z_ * r_)
#             p_ = -z_ + p_ * rtr_new / (rtr_ + jitter)
#             return i_ + 1, r_, p_, x_, rtr_new
#
#         out = _tf.while_loop(cond, body, (i, r, p, x, rtr))
#     return out[3]


def conjugate_gradient_block(matmul_fun, b, tol=1e-3, jitter=1e-9,
                             x_0=None, max_iter=100):
    """
    Conjugate gradient solver using the compressed feature matrix
    structure.

    Parameters
    ----------
    matmul_fun : function
        Function to compute A times b. Must accept and return a single Tensor
        with the same shape as b.
    b : Tensor
        A generic matrix with matching shape.
    tol : double
        The tolerance for ending the loop. It is tested against the
        Frobenius norm of the residual matrix.
    jitter : double
        A small number to improve numerical stability.

    Returns
    -------
    x : Tensor
        The solution of the system Ax=b, where A is implicitly provided in
        matmul_fun.
    """
    with _tf.name_scope("conjugate_gradient_block"):
        # initialization
        if x_0 is None:
            x_0 = _tf.zeros_like(b)
        # x = b
        r = matmul_fun(x_0) - b
        # r = _tf.zeros_like(b)
        # qr_q, qr_r = _tf.linalg.qr(r)
        # r = qr_q

        p = -r
        # x = _tf.zeros_like(b)
        x = x_0
        i = _tf.constant(0)
        tol = _tf.constant(tol, b.dtype)
        sx = _tf.shape(r)

        keep = _tf.fill([1, sx[1]], True)

        def update_cols(tensor, indices, updates):
            return _tf.transpose(_tf.tensor_scatter_nd_update(
                _tf.transpose(tensor),
                indices,
                _tf.transpose(updates)))

        def add_to_cols(tensor, indices, updates):
            return _tf.transpose(_tf.tensor_scatter_nd_add(
                _tf.transpose(tensor),
                indices,
                _tf.transpose(updates)))

        # TensorFlow loop
        def cond(i_, r_, p_, x_, keep_):
            # cond_0 = _tf.less(i_, sx[0])
            cond_0 = _tf.less(i_, max_iter)
            cond_1 = _tf.reduce_any(keep_)
            idx = _tf.squeeze(_tf.where(_tf.squeeze(keep_)))
            cond_2 = _tf.greater(_tf.rank(idx), 0)
            return _tf.reduce_all(_tf.stack([cond_0, cond_1, cond_2]))

        def body(i_, r_, p_, x_, keep_):
            idx = _tf.squeeze(_tf.where(_tf.squeeze(keep_)))
            r_sub = _tf.gather(r_, idx, axis=1)
            p_sub = _tf.gather(p_, idx, axis=1)
            rtr_ = _tf.matmul(r_sub, r_sub, True, False)

            ap = matmul_fun(p_sub)
            mat = _tf.matmul(p_sub, ap, True, False)
            mat = mat + _tf.eye(_tf.shape(mat)[0], dtype=mat.dtype) * jitter
            alpha = _tf.linalg.solve(mat, rtr_)
            r_sub = r_sub + _tf.matmul(ap, alpha)

            idx = _tf.expand_dims(idx, axis=1)
            x_ = add_to_cols(x_, idx, _tf.matmul(p_sub, alpha))
            r_ = update_cols(r_, idx, r_sub)

            # unequal convergence
            res = _tf.math.reduce_euclidean_norm(r_, axis=0, keepdims=True)
            keep_new = _tf.less(res, tol)

            rtr_new = _tf.matmul(r_sub, r_sub, True, False)
            rtr_ = rtr_ + _tf.eye(_tf.shape(rtr_)[0], dtype=rtr_.dtype) * jitter
            p_sub = -r_sub + _tf.matmul(p_sub, _tf.linalg.solve(rtr_, rtr_new))
            p_ = update_cols(p_, idx, p_sub)
            return i_ + 1, r_, p_, x_, keep_new

        out = _tf.while_loop(cond, body, (i, r, p, x, keep),
                             swap_memory=False,
                             parallel_iterations=1)
    # return _tf.matmul(out[3], qr_r)
    return out[3]


def extract_features(mat, min_var, pseudo_inverse=False):
    with _tf.name_scope("extract_features"):
        # eigen decomposition
        # values, vectors = _tf.self_adjoint_eig(mat)
        # values = _tf.reverse(values, axis=[0])
        # vectors = _tf.reverse(vectors, axis=[1])
        values, _, vectors = _tf.linalg.svd(mat)

        # number of principal components needed to reach min_var
        total_var = _tf.cumsum(values**2)
        keep = _tf.where(_tf.greater(
            total_var, min_var * _tf.reduce_sum(values**2)))[0] + 1
        keep = _tf.squeeze(keep)

        # compressed matrix
        # delta = _tf.sqrt(values[0:keep])
        delta = values[0:keep]
        if pseudo_inverse:
            delta = 1.0 / delta
        q = _tf.matmul(vectors[:, 0:keep], _tf.diag(delta))
    return q


def determinant_taylor(matmul_fn, size, n=10, m=30, seed=1234):
    with _tf.name_scope("determinant_approximation"):
        size_fl = _tf.cast(size, _tf.float64)

        def det_matmul(rhs):
            return rhs - matmul_fn(rhs) / size_fl

        rnd = _tf.random.stateless_uniform([size, n], [seed, 0],
                                           dtype=_tf.float64,
                                           minval=0, maxval=1)
        rnd = _tf.round(rnd) * 2 - 1
        mod = _tf.math.reduce_euclidean_norm(rnd, axis=0, keepdims=True)
        rnd = rnd / mod
        mod = _tf.squeeze(mod)

        trace = []
        prod = rnd
        for i in range(m):
            prod = det_matmul(prod)
            tr = _tf.reduce_mean(
                _tf.reduce_sum(rnd * prod, axis=0) * mod)
            trace.append(tr / (i + 1))
        trace = _tf.stack(trace) * size

        # with _tf.control_dependencies([_tf.print(trace)]):

        logdet = - _tf.reduce_sum(trace)
    return logdet + size_fl * _tf.math.log(size_fl)


def lanczos(matmul_fn, q_0, m=30):
    with _tf.name_scope("lanczos"):
        q_0 = q_0 / _tf.math.reduce_euclidean_norm(q_0)
        r = matmul_fn(q_0)
        alpha = _tf.reduce_sum(r * q_0)
        r = r - alpha * q_0
        beta = _tf.math.reduce_euclidean_norm(r)

        alpha = _tf.expand_dims(alpha, axis=0)
        beta = _tf.expand_dims(beta, axis=0)
        q_mat = q_0
        i = _tf.constant(1)

        def loop_fn(i_, q_mat_, beta_, r_, alpha_):
            v = _tf.expand_dims(q_mat_[:, - 1], axis=1)
            q_i = r_ / beta_[- 1]
            q_mat_ = _tf.concat([q_mat_, q_i], axis=1)
            beta_i = beta[-1]
            r_ = matmul_fn(q_i) - beta_i * v
            alpha_i = _tf.reduce_sum(q_i * r_)
            alpha_i = _tf.expand_dims(alpha_i, axis=0)
            alpha_ = _tf.concat([alpha_, alpha_i], axis=0)
            r_ = r_ - alpha_i * q_i

            # reorthogonalization
            r_ = r_ - _tf.matmul(q_mat_, _tf.matmul(q_mat_, r_, True, False))
            r_ = r_ - _tf.matmul(q_mat_, _tf.matmul(q_mat_, r_, True, False))
            r_ = r_ - _tf.matmul(q_mat_, _tf.matmul(q_mat_, r_, True, False))

            beta_i = _tf.math.reduce_euclidean_norm(r_)
            beta_i = _tf.expand_dims(beta_i, axis=0)
            beta_ = _tf.concat([beta_, beta_i], axis=0)
            return i_ + 1, q_mat_, beta_, r_, alpha_

        def cond_fn(i_, q_mat_, beta_, r_, alpha_):
            cond_0 = _tf.less(i_, m)
            # cond_1 = _tf.greater(beta_[-2], 0.0)
            return _tf.reduce_all(_tf.stack([cond_0]))

        i, q_mat, beta, r, alpha = _tf.while_loop(
            cond_fn, loop_fn, (i, q_mat, beta, r, alpha),
            shape_invariants=(_tf.TensorShape([]),
                              _tf.TensorShape([None, None]),
                              _tf.TensorShape([None]),
                              _tf.TensorShape([None, None]),
                              _tf.TensorShape([None])),
            swap_memory=False,
            parallel_iterations=1)

        keep = _tf.squeeze(_tf.where(_tf.greater(beta, 1e-12)))
        keep = _tf.cond(_tf.greater(_tf.rank(keep), 0),
                        lambda: keep,
                        lambda: _tf.constant([0], keep.dtype))
        beta = _tf.gather(beta, keep)
        beta = _tf.cond(_tf.greater(_tf.rank(beta), 0),
                        lambda: beta,
                        lambda: _tf.expand_dims(beta, axis=0))
        pos = _tf.shape(beta)[0]
        alpha = alpha[0:pos]
        q_mat = q_mat[:, 0:pos]

        alpha_min = _tf.reduce_min(alpha)
        alpha = _tf.cond(_tf.greater(alpha_min, 0.0),
                         lambda: alpha,
                         lambda: alpha - 1.1 * alpha_min)

        alpha = _tf.linalg.tensor_diag(alpha)
        beta = _tf.linalg.tensor_diag(
            _tf.concat([beta[0:(pos-1)], [0.0]], axis=0))
        mat_t = alpha + _tf.roll(beta, 1, axis=1) + _tf.roll(beta, 1, axis=0)

    return mat_t, q_mat


def determinant_lanczos(matmul_fn, size, n=10, m=100, seed=1234):
    with _tf.name_scope("determinant_approximation_lanczos"):

        # Rademacher random vectors
        rnd = _tf.random.stateless_uniform([size, n], [seed, 0],
                                           dtype=_tf.float64,
                                           minval=0, maxval=1)
        rnd = _tf.round(rnd) * 2 - 1

        def loop_fn(vec):
            vec = _tf.expand_dims(vec, axis=1)
            mat_t, _ = lanczos(matmul_fn, vec, m + 1)
            eigvals, eigvecs = _tf.linalg.eigh(mat_t)
            eigvals = _tf.reverse(eigvals, axis=[0])

            eig_min = _tf.reduce_min(eigvals)
            eigvals = _tf.cond(_tf.greater(eig_min, 0.0),
                               lambda: eigvals,
                               lambda: eigvals - 1.01 * eig_min)

            eigvecs = _tf.reverse(eigvecs, axis=[1])
            tau = eigvecs[0, :]
            return _tf.reduce_sum(
                tau**2 * _tf.math.log(eigvals)
            )

        det = _tf.map_fn(loop_fn, _tf.transpose(rnd),
                         swap_memory=False,
                         parallel_iterations=1)
        logdet = size / n * _tf.reduce_sum(det)
    return logdet
