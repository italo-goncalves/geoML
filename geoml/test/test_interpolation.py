import numpy as np
# import tensorflow as tf

import geoml
import timeit

# cubic_conv_1D
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 100)

w_np = geoml.interpolation.cubic_conv_1d(x, xnew)
w_sp = geoml.interpolation.cubic_conv_1d_sparse(x, xnew)
w_sp = w_sp.todense()
w_par = geoml.interpolation.cubic_conv_1d_parallel(x, xnew)
w_par = w_par.todense()

assert (np.abs(w_np - w_sp) < 1e-9).all()
assert (np.abs(w_np - w_par) < 1e-9).all()

# speed
setup = """
import numpy as np
import geoml

x = np.linspace(0, 10, 1001)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 10000)
"""
timeit.timeit("geoml.interpolation.cubic_conv_1d(x, xnew)", 
              setup=setup, number=10)
timeit.timeit("geoml.interpolation.cubic_conv_1d_sparse(x, xnew)", 
              setup=setup, number=10)
timeit.timeit("geoml.interpolation.cubic_conv_1d_parallel(x, xnew)", 
              setup=setup, number=10)

setup = """
import numpy as np
import geoml

x = np.linspace(0, 10, 1001)
np.random.seed(1234)
xnew = np.random.uniform(0, 1, [10000, 2])
"""
timeit.timeit("geoml.interpolation.cubic_conv_2d_sparse(x, x, xnew)", 
              setup=setup, number=10)
timeit.timeit("geoml.interpolation.cubic_conv_2d_parallel(x, x, xnew)", 
              setup=setup, number=10)


# gaussian
#x = np.linspace(0, 10, 11)
#np.random.seed(1234)
#xnew = np.random.uniform(0, 10, 100)
#xnew = np.linspace(0, 10, 1001)
#
#w_inv = geoml.interpolation.inverse_distance_1d_sparse(
#        x, xnew, radius=5, power=2, epsilon=0.01)
#w_inv = w_inv.todense()
#
#y = np.sin(x) + 2*np.cos(x+1) + 2
#ynew = np.matmul(w_inv, np.expand_dims(y, 1))
#plt.plot(x, y, "ok", xnew, ynew, "-r")
