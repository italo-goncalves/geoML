## version 0.5.0
* Memory-efficient data storage: variable arrays are now backed by NumPy in RAM
or chunked Zarr on disk, chosen automatically by size, so large projects no
longer need to hold every array in memory
* Simulations are stored as a single `(n_data, n_sim)` array (a dedicated
dimension) rather than a list of separate arrays
* Grid and block coordinates are now generated lazily: a regular grid no longer
holds its full `(n_data, n_dim)` coordinate array in memory, but produces the
requested rows on demand, so very large grids can be built and predicted into
without the coordinates dominating RAM
* Containers can be persisted to and reloaded from a single Zarr store with
`to_zarr()` / `open()`, covering all point-based containers and variable types
* Quantiles and probabilities are computed lazily in a single pass over the
simulations, without materializing them
* `reset_probabilities()` is now the inverse of `reset_quantiles()`: it takes
cutoff values and returns cumulative probabilities in (0, 1)
* Fixed construction of `OrderedRockType` and rock-type/binary variables from
string measurements
* Fixed a bug where a parameter shared between components (e.g. a transform
reused by several kernels) was counted more than once by `get_parameter_values`
and `save_state`
* New dependencies: `zarr`, `xarray`, `dask`

## version 0.4.1
* Support for predictions over blocks with discretization
* Delta method for change of support

## version 0.4.0
* Implemented multivariate data and likelihoods
* Warpings are now multivariate
* New warpings: `PCA`, `RobustPCA`, `Rotation`, and `ContinuousNormalizingFlow`
* Streamlined code to deal with compositional data
* Streamlined quantiles and percentiles computation, now directly from simulations
* Added option to smooth data when converting it to data to image or cube format
* Added convenience methods to compute model metrics


## version 0.3.5
* Support for rotated 3D grids
* Simulations now have the option to include the likelihood variance
* Improvements and bug fixes
* Product of Experts is now the default for GP latent variables

## version 0.3.4
* Improved pyvista integration
* Multiscale modeling did not work
* Compatibility with TensorFlow v2.11

## version 0.3.3
* Improvements and optimizations
* Heteroscedastic likelihoods
* A likelihood for conformable layers
* A likelihood for limited time-aware implicit modeling
* Ability to use a random field to move data points
* Ellipsoidal trends
* Limited support for faults
* (Experimental) multiscale modeling (refinement with increasingly denser inducing point sets)

## version 0.3.2
* Tensorized latent variable removed for now (back to the old chalkboard).
* Full deep GP implementation.
* Added Jura dataset.
* Added likelihoods for compositional variables.
* `kernels` module now has a specific object for covariance functions.
* Implemented the ability to convert data objects to `pyvista`.
* Implemented `Surface3D` data object.

## version 0.3.1
* New autoregressive latent variable.
* New tensorized latent variable.
* New Product of Experts latent variable.
* Spline warping is now centered at zero.
* Ensembles now handled in a specific model.

## version 0.3.0
* Introduced variational models.
* All training is now gradient-based.
* Changed the data manipulation system, defining variable types (continuous,
categorical, etc.) and Attributes (mean, variance, entropy, etc.) that can be
accessed and plotted directly.
* Plotly interface is now in a specific module.
* New `Section3D` data object.

## version 0.2.0
* Code updated to TensorFlow 2.0.
* Cubic convolution implemented in TensorFlow.
* Simplified kernel API.
* New auxiliary functions in `tftools` module.
* Cubic splines implemented in TensorFlow.
* Added Cerro do Andrade dataset.
* Notebooks moved to Google Colaboratory.
* Training is now based on Particle Swarm Optimization.
* Included a `GPOptions` class to control verbosity, batch size and
other model options.
* The covariance matrix for directional data is now computed with a 
finite difference method.
* Drillholes can now be segmented in roughly fixed intervals.

## version 0.1.3
* Introduced parallel sparse cubic convolution interpolation.
* Introduced the sparse GP model.
* Introduced the SPICE model for scalability.
* New auxiliary functions in `tftools` module, such as 
conjugate gradient solver and Lanczos decomposition.
* Limited support for modeling non-spatial data.

## version 0.1.2
* Introduced the `jitter` parameter in the models, for improved 
numerical stability.
* Corrected a bug where setting the value of a parameter beyond its limits
would not extend the limits.
* Implemented the `prod_n()` and `safe_logdet()` functions in the 
`tftools` module.
* Implemented the `aggregate_categorical()` method for point data methods.
* Optimized the batch prediction process.
* Implemented saving and loading for model parameters.
* Implemented the neural network transform.
* Now multiple transforms can be chained and/or shared between multiple
kernels.
* The classification algorithms now output an uncertainty metric, weighting
the entropy and predictive variance.

## version 0.1.1
* Added support for products of kernels.
* Fixed bug in `geoml.tftools.prod_n()` function.
* `geoml.kernels.Gaussian.covariance_matrix(x, y)` now
adds some jitter to the covariance matrix when x == y, to
avoid Cholesky decomposition problems.
* `geoml.warping.Softplus` now corrects non-positive values
in its input.
* Corrected a bug where the wrong nugget would be added
while making a prediction.
* Added sunspot number data and example notebook.
* Implemented cubic convolution fully in TensorFlow.
* Set the default `n_knots=5` in `geoml.warping.CubicSpline.__init__()`.