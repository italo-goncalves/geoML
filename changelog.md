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
* Implemented `Surface3D` da[setup.py](setup.py)ta object.

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