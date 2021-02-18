import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import geoml
import timeit

# cubic_conv_1d
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 100)

x = tf.constant(x)
xnew = tf.constant(xnew)
mat = geoml.interpolation.cubic_conv_1d(x, xnew)
w = mat.matmul(tf.eye(11, dtype=tf.float64))

# speed
setup = """
import tensorflow as tf
import numpy as np
import geoml

x = np.linspace(0, 10, 1001)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 10000)

x = tf.constant(x)
xnew = tf.constant(xnew)
# warming up
geoml.interpolation.cubic_conv_1d(x, xnew)
"""
timeit.timeit("geoml.interpolation.cubic_conv_1d(x, xnew)", 
              setup=setup, number=100) # ~0.6s


# cubic_conv_2d
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 100)
ynew = np.random.uniform(0, 10, 100)

x = tf.constant(x)
xnew = tf.constant(xnew)
mat = geoml.interpolation.cubic_conv_2d(x, xnew, x, ynew)
w = mat.matmul(tf.eye(11*11, dtype=tf.float64))

setup = """
import numpy as np
import tensorflow as tf
import geoml

x = np.linspace(0, 10, 1001)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 10000)
ynew = np.random.uniform(0, 10, 10000)

x = tf.constant(x)
xnew = tf.constant(xnew)
geoml.interpolation.cubic_conv_2d(x, xnew, x, ynew)
"""
timeit.timeit("geoml.interpolation.cubic_conv_2d(x, xnew, x, ynew)", 
              setup=setup, number=100) # ~2.2s



# cubic_conv_3d
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 100)
ynew = np.random.uniform(0, 10, 100)
znew = np.random.uniform(0, 10, 100)

x = tf.constant(x)
xnew = tf.constant(xnew)
mat = geoml.interpolation.cubic_conv_3d(x, xnew, x, ynew, x, znew)
w = mat.matmul(tf.eye(11*11*11, dtype=tf.float64))

setup = """
import numpy as np
import tensorflow as tf
import geoml

x = np.linspace(0, 10, 1001)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 10000)
ynew = np.random.uniform(0, 10, 10000)
znew = np.random.uniform(0, 10, 10000)

x = tf.constant(x)
xnew = tf.constant(xnew)
geoml.interpolation.cubic_conv_3d(x, xnew, x, ynew, x, znew)
"""
timeit.timeit("geoml.interpolation.cubic_conv_3d(x, xnew, x, ynew, x, znew)", 
              setup=setup, number=100) # 3.5s


# interpolating spline
n_funs = 10
x = tf.sort(tf.random.stateless_uniform([10, n_funs], maxval=10,
                                        dtype=tf.float64, 
                                        seed=[4321, 0]),
            axis=0)
y = tf.math.sin(x*3)
xnew = tf.cast(tf.linspace(-1.0, 11.0, 1001), tf.float64)
xnew = tf.tile(xnew[:, None], [1, n_funs])

spline = geoml.interpolation.CubicSpline()
ynew = spline.interpolate(x, y, xnew)
ynew_d1 = spline.interpolate_d1(x, y, xnew)

plt.plot(xnew.numpy()[:, 0], ynew.numpy()[:, 0], "-r")
plt.plot(x.numpy()[:, 0], y.numpy()[:, 0], "ok")
plt.plot(xnew.numpy()[:, 0], ynew_d1.numpy()[:, 0], "-g")

y_mono = tf.cumsum(tf.math.abs(y), axis=0)
spline_mono = geoml.interpolation.MonotonicCubicSpline()

ynew_mono = spline_mono.interpolate(x, y_mono, xnew)
ynew_mono_d1 = spline_mono.interpolate_d1(x, y_mono, xnew)

plt.plot(xnew.numpy()[:, 0], ynew_mono.numpy()[:, 0], "-r")
plt.plot(x.numpy()[:, 0], y_mono.numpy()[:, 0], "ok")
plt.plot(xnew.numpy()[:, 0], ynew_mono_d1.numpy()[:, 0], "-g")
plt.hlines(0, -1, 11, linestyles="dashed")


setup = """
import numpy as np
import tensorflow as tf
import geoml

n_funs = 100

x = tf.sort(tf.random.stateless_uniform([10, n_funs], maxval=10, dtype=tf.float64, 
                                        seed=[4321, 0]))
y = tf.math.sin(x*3)
y_mono = tf.cumsum(tf.math.abs(y), axis=0)
spline_mono = geoml.interpolation.MonotonicCubicSpline()

xnew = tf.cast(tf.linspace(-1.0, 11.0, 10001), tf.float64)
xnew = tf.tile(xnew[:, None], [1, n_funs])
ynew_mono = spline_mono.interpolate(x, y_mono, xnew)
"""
timeit.timeit("ynew_mono = spline_mono.interpolate(x, y_mono, xnew)", 
              setup=setup, number=10) # 1.66s

setup = """
import numpy as np
import tensorflow as tf
import geoml

n_funs = 10

x = tf.sort(tf.random.stateless_uniform([100, n_funs], maxval=10, dtype=tf.float64, 
                                        seed=[4321, 0]))
y = tf.math.sin(x*3)
y_mono = tf.cumsum(tf.math.abs(y), axis=0)
spline_mono = geoml.interpolation.MonotonicCubicSpline()

xnew = tf.cast(tf.linspace(-1.0, 11.0, 10001), tf.float64)
xnew = tf.tile(xnew[:, None], [1, n_funs])
ynew_mono = spline_mono.interpolate(x, y_mono, xnew)
"""
timeit.timeit("ynew_mono = spline_mono.interpolate(x, y_mono, xnew)", 
              setup=setup, number=10) # 1.63s