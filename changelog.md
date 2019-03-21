## version 0.1.1
* Added support for products of kernels.
* Fixed bug in `geoml.tftools.prod_n()` function.
* `geoml.kernels.GaussianKernel.covariance_matrix(x, y)` now
adds some jitter to the covariance matrix when x == y, to
avoid Cholesky decomposition problems.
* `geoml.warping.Softplus` now corrects non-positive values
in its input.
* Corrected a bug where the wrong nugget would be added
while making a prediction.
* Added sunspot number data and example notebook.