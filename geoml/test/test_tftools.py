import numpy as np
import tensorflow as tf

import geoml

# conjugate gradient
g = tf.Graph()
with g.as_default():
    a = tf.random.stateless_normal([1000, 1000], [1234, 0])
    a = tf.matmul(a, a, True, False)
    a = a + tf.eye(1000) * 1e-6
    b = tf.random.stateless_normal([1000, 1000], [4321, 0])
    b = tf.matmul(a, b, True, False)
    solved = geoml.tftools.conjugate_gradient_block(
            lambda x: tf.matmul(a, x), b)
with tf.Session(graph=g) as sess:
    out = solved.eval()
    
assert not np.any(np.isnan(out))

# lanczos
g = tf.Graph()
with g.as_default():
    a = tf.diag(tf.constant([0, 1, 2, 3, 4, 100000], tf.float32))
    b = tf.ones([6, 1], tf.float32)
    mat_t, q = geoml.tftools.lanczos(
            lambda x: tf.matmul(a, x), b, 6)
with tf.Session(graph=g) as sess:
    mat_t = mat_t.eval()
    q = q.eval()
    