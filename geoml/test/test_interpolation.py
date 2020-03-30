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
w = geoml.interpolation.cubic_conv_1d(x, xnew)
w2 = tf.sparse.to_dense(w)


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
              setup=setup, number=100) # ~1.6s


# cubic_conv_2d
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, [100, 2])

x = tf.constant(x)
xnew = tf.constant(xnew)
w = geoml.interpolation.cubic_conv_2d(x, x, xnew)
w2 = tf.sparse.to_dense(w)

setup = """
import numpy as np
import tensorflow as tf
import geoml

x = np.linspace(0, 10, 1001)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, [10000, 2])

x = tf.constant(x)
xnew = tf.constant(xnew)
geoml.interpolation.cubic_conv_2d(x, x, xnew)
"""
timeit.timeit("geoml.interpolation.cubic_conv_2d(x, x, xnew)", 
              setup=setup, number=10) # ~125s



# cubic_conv_3d
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, [100, 3])

x = tf.constant(x)
xnew = tf.constant(xnew)
w = geoml.interpolation.cubic_conv_3d(x, x, x, xnew)
w2 = tf.sparse.to_dense(w)

setup = """
import numpy as np
import tensorflow as tf
import geoml

x = np.linspace(0, 10, 1001)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, [10000, 3])

x = tf.constant(x)
xnew = tf.constant(xnew)
geoml.interpolation.cubic_conv_3d(x, x, x, xnew)
"""
timeit.timeit("geoml.interpolation.cubic_conv_3d(x, x, x, xnew)", 
              setup=setup, number=10) # ??s


# interpolating spline
x = tf.sort(tf.random.stateless_uniform([10], maxval=10, dtype=tf.float64, 
                                        seed=[4321, 0]))
y = tf.math.sin(x*3)
xnew = tf.cast(tf.linspace(-1.0, 11.0, 1001), tf.float64)

spline = geoml.interpolation.CubicSpline(x, y)
ynew = spline.interpolate(xnew)
ynew_d1 = spline.interpolate_d1(xnew)

plt.plot(xnew.numpy(), ynew.numpy(), "-r")
plt.plot(x.numpy(), y.numpy(), "ok")
plt.plot(xnew.numpy(), ynew_d1.numpy(), "-g")
plt.plot(x.numpy(), spline.d.numpy(), "og")

y_mono = tf.cumsum(tf.math.abs(y))
spline_mono = geoml.interpolation.MonotonicCubicSpline(x, y_mono)

ynew_mono = spline_mono.interpolate(xnew)
ynew_mono_d1 = spline_mono.interpolate_d1(xnew)

plt.plot(xnew.numpy(), ynew_mono.numpy(), "-r")
plt.plot(x.numpy(), y_mono.numpy(), "ok")
plt.plot(xnew.numpy(), ynew_mono_d1.numpy(), "-g")
plt.plot(x.numpy(), spline_mono.d.numpy(), "og")
plt.hlines(0, -1, 11, linestyles="dashed")


setup = """
import numpy as np
import tensorflow as tf
import geoml

x = tf.sort(tf.random.stateless_uniform([10], maxval=10, dtype=tf.float64, 
                                        seed=[4321, 0]))
y = tf.math.sin(x*3)
y_mono = tf.cumsum(tf.math.abs(y))
spline_mono = geoml.interpolation.MonotonicCubicSpline(x, y_mono)

xnew = tf.cast(tf.linspace(-1.0, 11.0, 10001), tf.float64)
ynew_mono = spline_mono.interpolate(xnew)
"""
timeit.timeit("ynew_mono = spline_mono.interpolate(xnew)", 
              setup=setup, number=10) # 0.013s

setup = """
import numpy as np
import tensorflow as tf
import geoml

x = tf.sort(tf.random.stateless_uniform([10000], maxval=10, dtype=tf.float64, 
                                        seed=[4321, 0]))
y = tf.math.sin(x*3)
y_mono = tf.cumsum(tf.math.abs(y))
spline_mono = geoml.interpolation.MonotonicCubicSpline(x, y_mono)
"""
timeit.timeit("spline_mono = geoml.interpolation.MonotonicCubicSpline(x, y_mono)", 
              setup=setup, number=10) # 0.034s