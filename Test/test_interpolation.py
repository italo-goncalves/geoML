import numpy as np

import geoml

#%% cubic_conv_1D
x = np.linspace(0, 10, 11)
np.random.seed(1234)
xnew = np.random.uniform(0, 10, 100)

W = geoml.interpolation.cubic_conv_1D(x, xnew)