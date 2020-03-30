import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import geoml

# 1D
x = np.reshape(np.linspace(0, 9, 1000), [1000,1])
tr = geoml.transform.Isotropic(5)
ker_gauss = geoml.kernels.Gaussian(tr)
ker_sph = geoml.kernels.Spherical(tr)
ker_cos = geoml.kernels.Cosine(tr)

cov_gauss = ker_gauss.covariance_matrix(x, x)
cov_sph = ker_sph.covariance_matrix(x, x)
cov_cos = ker_cos.covariance_matrix(x, x)

ker_comp_1 = geoml.kernels.Sum(ker_gauss, ker_sph)
cov_comp_1 = ker_comp_1.covariance_matrix(x, x)

ker_comp_2 = geoml.kernels.Product(ker_comp_1, ker_cos)
cov_comp_2 = ker_comp_2.covariance_matrix(x, x)

ker_comp_2.components[0].components[1].transform.parameters["range"].fix()
print(ker_comp_2)

plt.plot(x, cov_comp_2.numpy()[:, 0])
