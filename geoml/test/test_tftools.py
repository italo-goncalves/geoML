import numpy as np
import tensorflow as tf

import geoml

# conjugate gradient block
a = tf.random.stateless_normal([1000, 1000], [1234, 0])
a = tf.matmul(a, a, True, False)
a = a + tf.eye(1000) * 1e-3
b = tf.random.stateless_normal([1000, 1000], [4321, 0])
b = tf.matmul(a, b, True, False)
a = tf.cast(a, tf.float64)
b = tf.cast(b, tf.float64)
solved = geoml.tftools.conjugate_gradient_block(
        lambda x: tf.matmul(a, x), b)
    
assert not np.any(np.isnan(solved))

# conjugate gradient normal
b = tf.expand_dims(b[:, 0], 1)
solved = geoml.tftools.conjugate_gradient(lambda x: tf.matmul(a, x), b)

# lanczos
a = tf.linalg.diag(tf.constant([0, 1, 2, 3, 4, 100000], tf.float64))
b = tf.ones([6, 1], tf.float64)
mat_t, q = geoml.tftools.lanczos(lambda x: tf.matmul(a, x), b, 6)
t=mat_t.numpy()

# lanczos determinant
chol = tf.linalg.cholesky(a)
true_det = 2*tf.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(chol)))

det_0 = geoml.tftools.determinant_lanczos(
        lambda x: tf.matmul(a, x), 1000, n=20, m=100)

phi, solved_1, det_1 = geoml.tftools.lanczos_gp_solve(
        lambda x: tf.matmul(a, x), y=b, n=20, m=100, seed=1234)

assert tf.sqrt(tf.reduce_mean((solved - solved_1)**2)) < 1e-6

#det_2 = geoml.tftools.lanczos_conjgate_gradient(
#        lambda x: tf.matmul(a, x), 1000, n=20, m=100)