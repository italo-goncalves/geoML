import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.tools as ptls
# from tensorboard import main as tb

import geoml

walker, walker_ex = geoml.data.Examples.walker()

kernels=[geoml.kernels.SphericalKernel(geoml.transform.Identity()),
         geoml.kernels.SphericalKernel(geoml.transform.Isotropic(50))]
warping=[geoml.warping.Softplus(), geoml.warping.Spline(5)]
gp = geoml.models.InterpolatingBands(
    walker, "V", walker_ex,
    kernels=kernels, warping=warping,
    base_transform=geoml.transform.Anisotropy2D(360),
    n_directions=20, quick=False, n_points=400)
#gp.cov_model.variance.set_value(np.array([0.96, 0.04]))
#gp.cov_model.set_warping_parameter(
#    1, "warp", np.array([-0.5765, 0.635, 0.4184, 0.6075, 0.2089]))
gp.train(seed=1234, max_iter=2000)

gp.predict(name="Vpred", batch_size=20000)

logdir = "C:/Dropbox/Python/Tensorboard"
writer = tf.summary.FileWriter(logdir, gp.graph)

plt.figure(figsize=(12,12))
plt.subplot(2, 2, 1)
plt.imshow(walker_ex.as_image("Vpred_p0.025"), vmin=0, vmax=1700, origin="lower")
plt.colorbar()
plt.title("2.5% quantile")
plt.subplot(2, 2, 2)
plt.imshow(walker_ex.as_image("Vpred_p0.5"), vmin=0, vmax=1700, origin="lower")
plt.colorbar()
plt.title("Median")
plt.subplot(2, 2, 3)
plt.imshow(walker_ex.as_image("Vpred_p0.975"), vmin=0, vmax=1700, origin="lower")
plt.colorbar()
plt.title("97.5% quantile")
plt.subplot(2, 2, 4)
plt.imshow(walker_ex.as_image("V"), vmin=0, vmax=1700, origin="lower")
plt.colorbar()
plt.title("True values")
plt.show()

gp.simulate(n_sim=100, batch_size=20000, add_noise=True)

plt.figure(figsize=(12, 12))
for i in np.arange(1, 5):
    plt.subplot(2, 2, i)
    sim = "sim_" + "{:0>3d}".format(i)
    plt.imshow(walker_ex.as_image(sim), vmin=0, vmax=1700, origin="lower")
    plt.colorbar()
plt.show()


# variogram
def gamma(val, axis=0, lag=1):
    sh = val.shape
    if lag >= sh[axis]:
        raise ValueError("lag too high")

    def dif_sq(x, lag):
        x1 = x[0:(len(x)-lag)]
        x2 = x[lag:len(x)]
        return np.mean((x1-x2)**2)

    dif = np.apply_along_axis(lambda x: dif_sq(x, lag), axis, val)
    return 0.5*np.mean(dif)


def vg(val, axis, lags):
    return np.array([gamma(val, axis=axis, lag=lag) for lag in lags])


lags = np.arange(1, 201)
vg_true = [vg(walker_ex.as_image("V"), axis=0, lags=lags),
           vg(walker_ex.as_image("V"), axis=1, lags=lags)]
vg_sim = [[vg(walker_ex.as_image("sim_" + "{:0>3d}".format(i)),
              axis=0, lags=lags),
           vg(walker_ex.as_image("sim_" + "{:0>3d}".format(i)),
              axis=1, lags=lags)] for i in range(100)]
vg_sim = np.array(vg_sim)  # n_sim x axis x lag

vg_data_ns_lag = np.array([
    8.615893, 15.368443, 24.101943, 34.438619, 44.000312, 53.982558,
    63.679293, 73.910500, 83.698979, 94.118934, 103.866408, 114.348585,
    121.950557,
])
vg_data_ns_gamma = np.array([
    33785.13, 55055.79, 63448.89, 78094.37, 83008.55, 90716.19,
    88650.36, 100486.62, 89501.55, 103380.57, 101038.36, 98677.53, 98355.88,
])

vg_data_ew_lag = np.array([
    6.532345, 14.811752, 24.644694, 34.811028, 44.207668, 54.854384,
    64.435300, 74.860619, 84.470791, 95.027901, 104.510051, 115.044951,
    122.089810,
])
vg_data_ew_gamma = np.array([
    47575.80, 74973.09, 92091.18, 97078.22, 98291.57, 105718.70,
    79138.05, 94530.47, 82491.43, 93578.60, 83801.35, 90487.75,
    74419.25,
])

plt.subplot(1, 2, 1)
for i in range(100):
    plt.plot(lags, vg_sim[i, 0, :], "b-", alpha=0.1)
plt.plot(lags, vg_true[0], "k-")
# plt.plot(vg_data_ns_lag, vg_data_ns_gamma, "ko")
# plt.show()

plt.subplot(1, 2, 2)
for i in range(100):
    plt.plot(lags, vg_sim[i, 1, :], "b-", alpha=0.1)
plt.plot(lags, vg_true[1], "k-")
# plt.plot(vg_data_ew_lag, vg_data_ew_gamma, "ko")
plt.show()

feed = gp.cov_model.feed_dict()
feed.update(
    {gp.tf_handles["y_tf"]: np.resize(gp.y, (len(gp.y), 1))})

with tf.Session(graph=gp.graph) as session:
    session.run(gp.tf_handles["init"], feed_dict=feed)
    # tf_handles = [gp.tf_handles["Q"],
    #            gp.tf_handles["features_x"]]
    features_x = session.run(gp.tf_handles["features_x"], feed_dict=feed)