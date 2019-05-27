import numpy as np
import tensorflow as tf

import geoml

# cubic_conv_1D
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 100)

W = geoml.interpolation.cubic_conv_1d(x, xnew)

g = tf.Graph()
with g.as_default():
    W_tf = geoml.interpolation.cubic_conv_1d_tf(x, xnew)
    # W_tf = tf.sparse.to_dense(W_tf)
    init = tf.global_variables_initializer()
with tf.Session(graph=g) as sess:
    sess.run(init)
    W2 = W_tf.eval(session=sess)

assert (np.abs(W - W2) < 1e-9).all()

# MonotonicSpline
x = np.array([0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15])
y = np.array([10, 10, 10, 10, 10, 10, 10.5, 15, 50, 60, 85])
x2 = np.linspace(-2, 17, 1000)
sp = geoml.interpolation.MonotonicSpline(x, y)
y2 = sp(x2)