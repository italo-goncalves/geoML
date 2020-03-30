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
# MERCHANTABILITY or FITNESS FOR matrix PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as _tf
import numpy as _np


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


@_tf.function
def composition_close(matrix):
    """
    Divides each row of a matrix by its sum.
    """
    with _tf.name_scope("composition_close"):
        s = _tf.shape(matrix)
        total = _tf.reduce_sum(matrix, axis=1, keepdims=True)
        total = _tf.tile(total, [1, s[1]]) + _np.exp(-10.0)
        out = matrix / total
    return out


def prod_n(inputs, name=None):
    """
    Multiplies all input tensors element-wise.

    Parameters
    ----------
    inputs : list
        matrix list of Tensor objects, all with the same shape and type.
    name : str
        matrix name for the _operation (optional).

    Returns
    -------
    matrix Tensor of same shape and type as the elements of inputs.
    """
    if name is None:
        name = "prod_n"
    with _tf.name_scope(name):
        out = _tf.stack(inputs, axis=0)
        out = _tf.reduce_prod(out, axis=0)
        return out


def conjugate_gradient(matmul_fun, b, x_0=None, tol=1e-3, jitter=1e-9,
                       max_iter=100):
    """
    Conjugate gradient solver.

    Parameters
    ----------
    matmul_fun : function
        Function to compute matrix times b. Must accept and return a single Tensor
        with the same shape as b.
    b : Tensor
        matrix vector with matching shape.
    x_0 : Tensor
        An initial guess at the solution. Defaults to zero.
    tol : double
        The tolerance for ending the loop. It is tested against the
        Frobenius norm of the residual matrix.
    jitter : double
        matrix small number to improve numerical stability.

    Returns
    -------
    x : Tensor
        The solution of the system Ax=b, where matrix is implicitly provided in
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


def conjugate_gradient_block(matmul_fun, b, tol=1e-3, jitter=1e-9,
                             x_0=None, max_iter=100):
    """
    Conjugate gradient solver using the compressed feature matrix
    structure.

    Parameters
    ----------
    matmul_fun : function
        Function to compute matrix times b. Must accept and return a single Tensor
        with the same shape as b.
    b : Tensor
        matrix generic matrix with matching shape.
    tol : double
        The tolerance for ending the loop. It is tested against the
        Frobenius norm of the residual matrix.
    jitter : double
        matrix small number to improve numerical stability.

    Returns
    -------
    x : Tensor
        The solution of the system Ax=b, where matrix is implicitly provided in
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


def lanczos(matmul_fn, q_0, m=30):
    with _tf.name_scope("lanczos"):
        q_0 = q_0 / _tf.math.reduce_euclidean_norm(q_0)
        r = matmul_fn(q_0)
        alpha = _tf.reduce_sum(r * q_0)
        r = r - alpha * q_0
        beta = _tf.math.reduce_euclidean_norm(r)

        alpha = _tf.expand_dims(alpha, axis=0)
        beta = _tf.expand_dims(beta, axis=0)
        mat_q = q_0
        i = _tf.constant(1)

        def loop_fn(i_, mat_q_, beta_, r_, alpha_):
            v = _tf.expand_dims(mat_q_[:, - 1], axis=1)
            beta_i = beta_[-1]
            q_i = r_ / beta_i
            mat_q_ = _tf.concat([mat_q_, q_i], axis=1)
            r_ = matmul_fn(q_i) - beta_i * v
            alpha_i = _tf.reduce_sum(q_i * r_)
            alpha_i = _tf.expand_dims(alpha_i, axis=0)
            alpha_ = _tf.concat([alpha_, alpha_i], axis=0)
            r_ = r_ - alpha_i * q_i

            # reorthogonalization
            r_ = r_ - _tf.matmul(mat_q_, _tf.matmul(mat_q_, r_, True, False))
            r_ = r_ - _tf.matmul(mat_q_, _tf.matmul(mat_q_, r_, True, False))
            r_ = r_ - _tf.matmul(mat_q_, _tf.matmul(mat_q_, r_, True, False))

            beta_i = _tf.math.reduce_euclidean_norm(r_)
            beta_i = _tf.expand_dims(beta_i, axis=0)
            beta_ = _tf.concat([beta_, beta_i], axis=0)
            return i_ + 1, mat_q_, beta_, r_, alpha_

        def cond_fn(i_, mat_q_, beta_, r_, alpha_):
            cond_0 = _tf.less(i_, m)
            return _tf.reduce_all(_tf.stack([cond_0]))

        i, mat_q, beta, r, alpha = _tf.while_loop(
            cond_fn, loop_fn, (i, mat_q, beta, r, alpha),
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
        mat_q = mat_q[:, 0:pos]

        alpha_min = _tf.reduce_min(alpha)
        alpha = _tf.cond(_tf.greater(alpha_min, 0.0),
                         lambda: alpha,
                         lambda: alpha - 1.1 * alpha_min)

        alpha = _tf.linalg.tensor_diag(alpha)
        beta = _tf.linalg.tensor_diag(
            _tf.concat([beta[0:(pos-1)], [0.0]], axis=0))
        mat_t = alpha + _tf.roll(beta, 1, axis=1) + _tf.roll(beta, 1, axis=0)

    return mat_t, mat_q


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


def repeat_vector(vec, n, repeat_each=_tf.constant(True)):
    with _tf.name_scope("repeat_vector"):
        vec = _tf.expand_dims(vec, axis=-1)
        vec = _tf.tile(vec, [1, n])
        vec = _tf.cond(repeat_each,
                       lambda: _tf.reshape(vec, [-1]),
                       lambda: _tf.reshape(_tf.transpose(vec), [-1]))
    return vec


@_tf.function
def sparse_kronecker_by_rows(mat_x, mat_y):
    with _tf.name_scope("sparse_kronecker_by_rows"):
        sh = _tf.cast(_tf.shape(mat_x), _tf.int64)
        n_rows = sh[0]
        nx = sh[1]
        ny = _tf.cast(_tf.shape(mat_y)[1], _tf.int64)

        # pre-allocation
        rows = _tf.ones([0], _tf.int64)
        cols = _tf.ones([0], _tf.int64)
        vals = _tf.zeros([0], _tf.float64)
        i = _tf.constant(0, _tf.int64)

        def loop_fn(i_, rows_, cols_, vals_):
            slice_x = _tf.sparse.slice(mat_x, [i_, 0], [1, nx])
            slice_y = _tf.sparse.slice(mat_y, [i_, 0], [1, ny])

            nvals_x = _tf.shape(slice_x.values)[0]
            nvals_y = _tf.shape(slice_y.values)[0]

            vals_x = repeat_vector(slice_x.values, nvals_y,
                                   repeat_each=_tf.constant(False))
            vals_y = repeat_vector(slice_y.values, nvals_x)
            vals_xy = vals_x * vals_y

            cols_x = repeat_vector(slice_x.indices[:, 1], nvals_y,
                                   repeat_each=_tf.constant(False))
            cols_y = repeat_vector(slice_y.indices[:, 1], nvals_x)
            cols_xy = cols_x + cols_y*nx

            rows_xy = _tf.ones_like(cols_xy) * i_

            # update
            rows_ = _tf.concat([rows_, rows_xy], axis=0)
            cols_ = _tf.concat([cols_, cols_xy], axis=0)
            vals_ = _tf.concat([vals_, vals_xy], axis=0)

            return i_ + 1, rows_, cols_, vals_

        def cond_fn(i_, rows_, cols_, vals_):
            return _tf.less(i_, n_rows)

        i, rows, cols, vals = _tf.while_loop(
            cond_fn, loop_fn, loop_vars=[i, rows, cols, vals],
            parallel_iterations=100,
            shape_invariants=[_tf.TensorShape([]), _tf.TensorShape([None]),
                              _tf.TensorShape([None]), _tf.TensorShape([None])]
        )

        # output
        indices = _tf.stack([rows, cols], axis=1)
        mat_new = _tf.sparse.SparseTensor(indices, vals,
                                          dense_shape=[n_rows, nx*ny])
    return mat_new


# def lanczos_conjgate_gradient(matmul_fn, size, n=100, m=100,
#                               seed=1234, tol=1e-6):
#     """Matrix determinant based on conjugate gradients.
#
#     Based on Gardner et al. (2018)
#     """
#     with _tf.name_scope("lanczos_conjgate_gradient"):
#
#         # Rademacher random vectors
#         rnd = _tf.random.stateless_uniform([size, n], [seed, 0],
#                                            dtype=_tf.float64,
#                                            minval=0, maxval=1)
#         rnd = _tf.round(rnd) * 2 - 1
#         rnd = rnd / _tf.math.reduce_euclidean_norm(rnd, axis=0, keepdims=True)
#
#         # initialization
#         mat_u = _tf.zeros_like(rnd)
#         mat_r = matmul_fn(mat_u) - rnd
#         mat_z = mat_r
#         mat_d = mat_z
#
#         alpha = _tf.zeros([0, n], _tf.float64)
#         beta = _tf.zeros([0, n], _tf.float64)
#         i = _tf.constant(0)
#
#         # conjugate gradient
#         def loop_fn(i_, alpha_, beta_, u_, r_, z_, d_):
#             mat_v = matmul_fn(d_)
#             alpha_i = _tf.reduce_sum(r_ * z_, axis=0, keepdims=True) \
#                 / _tf.reduce_sum(d_ * mat_v, axis=0, keepdims=True)
#             u_ = u_ + alpha_i * d_
#             r_ = r_ - alpha_i * mat_v
#
#             z_new = r_
#             beta_i = _tf.reduce_sum(z_new**2, axis=0, keepdims=True) \
#                 / _tf.reduce_sum(z_**2, axis=0, keepdims=True)
#             d_ = z_new - beta_i * d_
#
#             alpha_ = _tf.concat([alpha_, alpha_i], axis=0)
#             beta_ = _tf.concat([beta_, beta_i], axis=0)
#
#             return i_ + 1, alpha_, beta_, u_, r_, z_, d_
#
#         def cond_fn(i_, alpha_, beta_, u_, r_, z_, d_):
#             cond_0 = _tf.less(i_, m)
#             # cond_1 = _tf.less(_tf.sqrt(_tf.reduce_mean(r_*r_)),
#             #                   _tf.constant(tol, _tf.float64))
#             return _tf.reduce_all(_tf.stack([cond_0]))
#
#         i, alpha, beta, mat_u, mat_r, mat_z, mat_d = _tf.while_loop(
#             cond_fn, loop_fn,
#             (i, alpha, beta, mat_u, mat_r, mat_z, mat_d),
#             shape_invariants=(_tf.TensorShape([]),
#                               _tf.TensorShape([None, n]),
#                               _tf.TensorShape([None, n]),
#                               _tf.TensorShape([size, n]),
#                               _tf.TensorShape([size, n]),
#                               _tf.TensorShape([size, n]),
#                               _tf.TensorShape([size, n])),
#             swap_memory=False,
#             parallel_iterations=1)
#
#         # fixing numerical issues
#         alpha_min = _tf.reduce_min(alpha, axis=0, keepdims=True)
#         alpha = _tf.where(_tf.greater(alpha, 0.0),
#                           alpha,
#                           alpha - 1.01 * alpha_min)
#
#         # lanczos and determinant
#         final_m = _tf.shape(alpha)[0]
#         idx = _tf.range(final_m)
#
#         lanczos_beta = _tf.sqrt(beta) / alpha
#         lanczos_beta = _tf.gather(lanczos_beta, _tf.expand_dims(idx, axis=1))
#
#         beta_alpha = beta / alpha
#         beta_alpha = _tf.roll(beta_alpha, shift=1, axis=0)
#         beta_alpha = _tf.tensor_scatter_nd_update(
#             beta_alpha, [[0]], _tf.zeros([1, n], _tf.float64))
#         lanczos_alpha = 1/alpha + beta_alpha
#
#         lanczos_alpha = _tf.transpose(lanczos_alpha)
#         lanczos_beta = _tf.transpose(_tf.squeeze(lanczos_beta, axis=1))
#         batch_t = _tf.zeros([n, final_m, final_m], _tf.float64)
#         batch_t = _tf.linalg.set_diag(batch_t, lanczos_alpha) \
#                   + _tf.roll(_tf.linalg.set_diag(batch_t, lanczos_beta), 1, 2) \
#                   + _tf.roll(_tf.linalg.set_diag(batch_t, lanczos_beta), 1, 1)
#
#         eigvals, eigvecs = _tf.linalg.eigh(batch_t)
#         eigvals = _tf.maximum(eigvals, 1e-9)
#         first_row = eigvecs[:, 0, :]
#
#         logdet = _tf.cast(size, _tf.float64) * _tf.reduce_mean(
#             _tf.reduce_sum(first_row**2 * _tf.math.log(eigvals), axis=1)
#         )
#
#     return logdet


def lanczos_gp_solve(matmul_fn, y, n=20, m=30, seed=1234):
    """Lanczos decomposition and determinant estimation"""
    with _tf.name_scope("lanczos_gp_solve"):
        n_data = _tf.shape(y)[0]

        # Rademacher random vectors
        rnd = _tf.random.stateless_uniform([n_data, n], [seed, 0],
                                           dtype=_tf.float64,
                                           minval=0, maxval=1)
        rnd = _tf.round(rnd) * 2 - 1
        rnd = rnd / _tf.math.reduce_euclidean_norm(rnd, axis=0, keepdims=True)

        # initialization
        q_0 = _tf.concat([y, rnd], axis=1)
        q_0 = q_0 / _tf.math.reduce_euclidean_norm(q_0, axis=0, keepdims=True)
        r = matmul_fn(q_0)
        alpha = _tf.reduce_sum(r * q_0, axis=0, keepdims=True)  # [it, n+1]
        r = r - alpha * q_0  # [n_data, n+1]
        beta = _tf.math.reduce_euclidean_norm(r, axis=0, keepdims=True)

        mat_q = _tf.expand_dims(q_0, axis=0)  # [it, n_data, n+1]
        i = _tf.constant(1)

        # batch lanczos
        def loop_fn(i_, mat_q_, beta_, r_, alpha_):
            v = mat_q_[-1, :, :]  # [n_data, n+1]
            beta_i = _tf.expand_dims(beta_[-1, :], axis=0)  # [1, n+1]
            q_i = r_ / beta_i
            mat_q_ = _tf.concat([mat_q_, _tf.expand_dims(q_i, 0)], axis=0)
            r_ = matmul_fn(q_i) - beta_i * v
            alpha_i = _tf.reduce_sum(q_i * r_, axis=0, keepdims=True)
            alpha_ = _tf.concat([alpha_, alpha_i], axis=0)
            r_ = r_ - alpha_i * q_i

            # reorthogonalization
            mat_q_ = _tf.transpose(mat_q_, perm=[2, 1, 0])  # [n+1, n_data, it]
            r_ = _tf.expand_dims(_tf.transpose(r_), axis=2)  # [n+1, n_data, 1]
            r_ = r_ - _tf.matmul(mat_q_, _tf.matmul(mat_q_, r_, True, False))
            r_ = r_ - _tf.matmul(mat_q_, _tf.matmul(mat_q_, r_, True, False))
            r_ = r_ - _tf.matmul(mat_q_, _tf.matmul(mat_q_, r_, True, False))
            r_ = _tf.transpose(_tf.squeeze(r_, axis=2))
            mat_q_ = _tf.transpose(mat_q_, perm=[2, 1, 0])

            beta_i = _tf.math.reduce_euclidean_norm(r_, axis=0, keepdims=True)
            beta_ = _tf.concat([beta_, beta_i], axis=0)
            return i_ + 1, mat_q_, beta_, r_, alpha_

        def cond_fn(i_, mat_q_, beta_, r_, alpha_):
            return _tf.less(i_, m)

        i, mat_q, beta, r, alpha = _tf.while_loop(
            cond_fn, loop_fn, (i, mat_q, beta, r, alpha),
            shape_invariants=(_tf.TensorShape([]),
                              _tf.TensorShape([None, None, None]),
                              _tf.TensorShape([None, None]),
                              _tf.TensorShape([None, None]),
                              _tf.TensorShape([None, None])),
            swap_memory=False,
            parallel_iterations=1)

        # post-processing
        alpha_min = _tf.reduce_min(alpha)
        alpha = _tf.where(_tf.greater(alpha, 0.0),
                          alpha,
                          alpha - 1.01 * alpha_min)
        beta = _tf.where(_tf.greater(beta, 1e-12),
                         beta,
                         _tf.ones_like(beta)*1e-12)
        beta = beta[0:(m-1), :]
        beta = _tf.concat([beta, _tf.zeros([1, n+1], _tf.float64)], axis=0)

        # determinant
        batch_t = _tf.zeros([n+1, m, m], _tf.float64)
        batch_t = _tf.linalg.set_diag(batch_t, _tf.transpose(alpha)) \
                  + _tf.roll(_tf.linalg.set_diag(batch_t, _tf.transpose(beta)),
                                                 1, 2) \
                  + _tf.roll(_tf.linalg.set_diag(batch_t, _tf.transpose(beta)),
                                                 1, 1)

        eigvals, eigvecs = _tf.linalg.eigh(batch_t)
        eigvals = _tf.maximum(eigvals, 1e-9)
        first_row = eigvecs[1::, 0, :]

        logdet = _tf.cast(n_data, _tf.float64) * _tf.reduce_mean(
            _tf.reduce_sum(first_row ** 2 * _tf.math.log(eigvals[1::, :]), axis=1)
        )

        # decomposition for solving the system
        eigvals_system = _tf.maximum(eigvals[0, :], 1e-9)
        phi = _tf.linalg.diag(_tf.sqrt(1 / eigvals_system))
        phi = _tf.matmul(eigvecs[0, :, :], phi)
        phi = _tf.matmul(mat_q[:, :, 0], phi, True, False)

        solved = _tf.matmul(phi, y, True, False)
        solved = _tf.matmul(phi, solved)

    return phi, solved, logdet


@_tf.function
def highest_value_probability(mu, var, seed, n_samples=10000):
    sh = _tf.shape(mu)
    n_data, n_values = sh[0], sh[1]

    rnd = _tf.random.stateless_normal([1, n_values, n_samples - n_values],
                                      seed=[seed, 0], dtype=_tf.float32)
    samples = _tf.expand_dims(_tf.sqrt(var), 2)*rnd + _tf.expand_dims(mu, 2)
    max_ind = _tf.math.argmax(samples, axis=1)

    def count_fn(i):
        which = _tf.cast(_tf.equal(max_ind, i), _tf.float32)
        return _tf.reduce_sum(which, axis=1) + 1.0

    counts = _tf.map_fn(count_fn, _tf.range(n_values, dtype=_tf.int64),
                        dtype=_tf.float32)
    counts = _tf.transpose(counts)

    prob = counts/_tf.reduce_sum(counts, axis=1, keepdims=True)
    return prob
