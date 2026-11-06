# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
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

"""What each operation of a `GPWalk` step costs, forward and with its
gradient, on random tensors of a walk's shapes: `n` walkers, `m` inducing
points, a field of two outputs. Each case is its own traced function, run
30 times after three warm-ups.

TensorFlow's profiler cannot say this: under WSL it records no GPU kernel,
and on the CPU the op times it sums are some 25 times the wall time, the
ops of one step running at once on oversubscribed threads.

Usage: python walk_ops.py [n=497] [m=357]

Measured 2026-11-05 at n=497, m=357, milliseconds forward / with the
gradient, RTX 4060 Ti and Ryzen 9 7950X (WSL):

| | GPU | CPU |
|---|---|---|
| expected kernel and slope | 1.3 / 2.1 | 4.3 / 9.2 |
| the same under XLA | 0.3 / 0.9 | 1.0 / 6.7 |
| Stein term, three operands | 2.1 / 4.5 | 3.1 / 5.5 |
| Stein term through `q`, with `cov P` | 1.1 / 2.3 | 1.6 / 3.8 |
| `cov (K + D)^-1 cov`, an output each | 1.0 / 2.1 | 1.9 / 3.0 |
| `cov K^-1 cov`, shared | 0.5 / 1.1 | 1.0 / 1.3 |
| `h` update, einsum | 0.3 / 4.7 | 2.1 / 3.9 |
| `h` update, broadcast sum | 0.3 / 0.9 | 2.5 / 7.5 |
| walked covariance `h h` | 1.4 / 3.8 | 3.0 / 5.2 |

The batched 2 x 2 products of the einsum form are slow in float64 on the
GPU, their gradient above all, and the broadcast form builds a temporary
twice the size of `h`, which the CPU pays for.
"""
import sys
import time

import numpy as np
import tensorflow as tf

import geoml.kernels as kr
import geoml.latent.network as net

n = int(sys.argv[1]) if len(sys.argv) > 1 else 497
m = int(sys.argv[2]) if len(sys.argv) > 2 else 357
d = 2
rng = np.random.default_rng(0)


def tensor(*shape, scale=0.1):
    return tf.constant(rng.normal(size=shape) * scale, tf.float64)


mean, u = tensor(n, d, scale=3.0), tensor(m, d, scale=3.0)
v, u_var = tf.abs(tensor(n, d)), tf.abs(tensor(m, d))
c = tensor(n, m, d, scale=0.001)
cov, grad = tensor(n, m), tensor(n, m, d)
root_r, smooth_inv, P = tensor(d, m, m), tensor(d, m, m), tensor(d, m, m)
k_inv = tensor(m, m)
h, q, jac, g = tensor(n, d, d, m), tensor(n, d, d, m), tensor(n, d, d), \
    tensor(d, n, m)
eye = tf.eye(d, dtype=tf.float64)
ranges = tf.constant([1.0, 1.0], tf.float64)


def expected(mean, v, c):
    return net._expected_kernel(kr.Gaussian(), ranges, mean, v, u, u_var, c,
                                True)


def h_einsum(jac, h, g):
    return h + 0.01 * (tf.einsum("...ij,...jsq->...isq", jac, h)
                       + tf.einsum("...snq,ds->...ndsq", g, eye))


def h_broadcast(jac, h, g):
    mixed = tf.reduce_sum(jac[:, :, :, None, None] * h[:, None], axis=2)
    return h + 0.01 * (mixed + eye[:, :, None]
                       * tf.transpose(g, [1, 0, 2])[:, None, :, :])


CASES = {
    "expected kernel and slope": (expected, [mean, v, c]),
    "the same under XLA": (tf.function(expected, jit_compile=True),
                           [mean, v, c]),
    "Stein term, three operands": (
        lambda grad, h: tf.einsum("...njd,...sjl,...ndsl->...ns",
                                  grad, root_r, h), [grad, h]),
    "Stein term through q, with cov P": (
        lambda grad, q, cov: (tf.einsum("...njd,...ndsj->...ns", grad, q),
                              tf.einsum("...nm,...smj->...snj", cov, P)),
        [grad, q, cov]),
    "cov (K + D)^-1 cov, an output each": (
        lambda cov: tf.reduce_sum(tf.einsum("...nm,...sml->...snl", cov,
                                            smooth_inv) * cov[None], -1),
        [cov]),
    "cov K^-1 cov, shared": (
        lambda cov: tf.reduce_sum(tf.matmul(cov, k_inv) * cov, -1), [cov]),
    "h update, einsum": (h_einsum, [jac, h, g]),
    "h update, broadcast sum": (h_broadcast, [jac, h, g]),
    "walked covariance h h": (
        lambda h: tf.einsum("...ndsq,...jdsq->...njd", h, h[:200]), [h]),
}


def timed(fn, reps=30):
    for _ in range(3):
        fn()
    start = time.perf_counter()
    for _ in range(reps):
        out = fn()
    tf.nest.map_structure(lambda x: x.numpy(), out)
    return (time.perf_counter() - start) / reps * 1e3


print("n=%d m=%d %s" % (n, m, "gpu" if tf.config.list_physical_devices("GPU")
                        else "cpu"))
for name, (fn, args) in CASES.items():
    held = [tf.Variable(x) for x in args]

    @tf.function
    def forward():
        return fn(*held)

    @tf.function
    def with_gradient():
        with tf.GradientTape() as tape:
            out = tf.nest.flatten(fn(*held))
            loss = tf.add_n([tf.reduce_sum(o) for o in out])
        return loss, tape.gradient(loss, held)

    print("%7.2f / %7.2f ms  %s" % (timed(forward), timed(with_gradient),
                                     name), flush=True)
