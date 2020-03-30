import numpy as np
import matplotlib.pyplot as plt

import geoml

y = np.random.uniform(0.1, 1, 10).cumsum()
warp = geoml.warping.ZScore()
warp.refresh(y)
plt.plot(y, warp.forward(y), "ok")
plt.plot(y, warp.derivative(y), "ok")

x = np.random.uniform(0.1, 10, 10).cumsum()
warp = geoml.warping.Spline(5)
warp.parameters["warped_partition"].set_value(
        np.array([0.15, 0.05, 0.2, 0.4, 0.1, 0.1]))
warp.refresh(x)
x2 = np.linspace(-20, 120, 101)
plt.figure()
plt.hlines([-5, 0, 5], -20, 120, linestyles="dashed")
plt.plot(x2, warp.forward(x2), "r-")
plt.plot(warp.base_spline.x, warp.base_spline.y, "or")
plt.plot(warp.backward(np.linspace(-5, 5, 101)), np.linspace(-5, 5, 101), "g-")
plt.plot(warp.reverse_spline.y, warp.reverse_spline.x, "|g")
plt.plot(x2, warp.derivative(x2), "b-")
