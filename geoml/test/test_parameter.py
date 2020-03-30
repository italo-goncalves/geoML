from geoml.parameter import *
import numpy as np

p_real = RealParameter(1, -10, 10)
p_real.set_value(12)
assert p_real.value.numpy() == 10.0
assert p_real.value_transformed.numpy() == 10.0

p_real2 = RealParameter(np.arange(5), -10*np.ones([5]), 10*np.ones([5]))

p_positive = PositiveParameter(np.exp(np.arange(5)), 
                               0.01*np.ones([5]), 
                               100*np.ones([5]))

p_compositional = CompositionalParameter(np.exp(np.arange(5)))
