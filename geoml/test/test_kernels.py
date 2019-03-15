import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import geoml

angle = np.array(np.linspace(0, 2*np.pi, 100), ndmin = 2).transpose()
coords = np.append(np.cos(angle), np.sin(angle), axis = 1)
dirs = np.random.normal(size=[100, 2])
for i in range(dirs.shape[0]): dirs[i,:] = dirs[i,:]/np.sqrt(np.sum(dirs[i,:]**2))
dirs = np.ones_like(coords)

tr = geoml.transform.Isotropic(5)
anis_2D = geoml.transform.Anisotropy2D(200, 0.1, 60)
ker_gauss = geoml.kernels.GaussianKernel(tr)
ker_sph = geoml.kernels.SphericalKernel(anis_2D)
ker_cos = geoml.kernels.CosineKernel(tr)
cov = geoml.kernels.CovarianceModelRegression(
        [ker_cos],
        [geoml.warping.Spline(5)])
cov.variance.set_value(np.array([0.99, 0.01]))

x_ = np.reshape(np.linspace(0, 9, 10), [10,1])
x_d = np.reshape(np.linspace(0, 9, 10), [10,1]) + 0.5
# covariance model
g = tf.Graph()
with g.as_default():
    cov.init_tf_placeholder()
    x = tf.constant(x_, dtype=tf.float64)
    xd = tf.constant(x_d, dtype=tf.float64)
    d = tf.constant(np.ones([10,1]), dtype=tf.float64)
    nugget = cov.variance.tf_val[-1]
    K = tf.add(cov.covariance_matrix(x, x),
               cov.nugget.nugget_matrix(x) * nugget)
    K1 = cov.covariance_matrix_d1(x, xd, d)
    K2 = tf.add(cov.covariance_matrix_d2(xd, xd, d, d),
                cov.nugget.nugget_matrix(xd) * 0)
    Kfull = tf.concat([tf.concat([K, K1], axis=1),
                       tf.concat([tf.transpose(K1), K2], axis=1)],
                axis = 0)
    L = tf.linalg.cholesky(Kfull)
#    x2 = cov.kernels[1].transform.backward(x)
    init = tf.global_variables_initializer()


#print(cov)

feed = cov.feed_dict()
with tf.Session(graph = g) as session:
    session.run(init, feed_dict = feed)
#    covmat_full = covmat_full.eval(session = session, feed_dict = feed)
#    L = L.eval(session = session, feed_dict = feed)
#    anis = cov.kernels[1].transform._anis.eval(session = session, feed_dict = feed)
#    anis_inv = cov.kernels[1].transform._anis_inv.eval(session = session, feed_dict = feed)
    tmp = K.eval(session = session, feed_dict = feed)

plt.imshow(tmp)
e = np.linalg.eig(tmp)
tmp = tmp + np.diag(np.repeat(1e-6, tmp.shape[0]))
L = np.linalg.cholesky(tmp)
plt.imshow(L)

#%% classification
cov = geoml.kernels.CovarianceModelClassif(
        [[geoml.kernels.CubicKernel(geoml.transform.Isotropic(10)), 
          geoml.kernels.CubicKernel(geoml.transform.Isotropic(5))], 
        [geoml.kernels.GaussianKernel(geoml.transform.Isotropic(10))]])

x_ = np.reshape(np.linspace(0, 9, 10), [10,1])
x_d = np.reshape(np.linspace(0, 9, 10), [10,1]) + 0.5

g = tf.Graph()
with g.as_default():
    cov.init_tf_placeholder()
    x = tf.constant(x_, dtype=tf.float64)
#    xd = tf.constant(x_d, dtype=tf.float64)
#    d = tf.constant(np.ones([10,1]), dtype=tf.float64)
#    nugget = cov.variance.tf_val[-1]
    K = cov.covariance_matrix(x, x)
    L = tf.linalg.cholesky(K)
    init = tf.global_variables_initializer()

feed = cov.feed_dict()
with tf.Session(graph = g) as session:
    session.run(init, feed_dict = feed)
#    covmat_full = covmat_full.eval(session = session, feed_dict = feed)
#    L = L.eval(session = session, feed_dict = feed)
#    anis = cov.kernels[1].transform._anis.eval(session = session, feed_dict = feed)
#    anis_inv = cov.kernels[1].transform._anis_inv.eval(session = session, feed_dict = feed)
    tmp = L.eval(session = session, feed_dict = feed)

plt.imshow(tmp[0,:,:])
