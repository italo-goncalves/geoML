import numpy as np
import matplotlib.pyplot as plt

import geoml

y = np.random.uniform(0.1, 1, 10).cumsum()
warp = geoml.warping.ZScore()
warp.refresh(y)
plt.plot(y, warp.forward(y), "ok")
plt.plot(y, warp.derivative(y), "ok")

y = np.random.uniform(0.1, 1, 10).cumsum()
x = np.linspace(0, 10, 10)
warp = geoml.warping.Spline(10)
warp.params["warp"].set_value(np.concatenate([[y[0]], np.diff(y)]))
#warp.params["original"].set_value(x)
warp.refresh(x)
x2 = np.linspace(-2, 12, 101)
plt.figure()
plt.plot(x2, warp.forward(x2), "r-", x, y, "or")
plt.plot(x2, warp.derivative(x2), "b-")

plt.figure()
plt.plot(x2, warp.backward(x2), "r-", y, x, "or")

plt.figure()
plt.plot(warp.forward(x), y, "ob")
plt.plot(x, warp.backward(y), "or")