## version 0.5.0
* `DrillholeData` was rewritten from scratch, in a new `drillhole` module.
It is now built from a collar and an optional survey table, and interval files
(assay, lithology, and any others) are added to it separately, each as an
`IntervalTable`. Hole paths are desurveyed by minimum curvature, so deviated
holes are positioned properly; holes with no survey are taken as straight,
along the collar attitude. Databases that record a downward hole with a
negative dip are supported through `dip_positive_down=False`
* Interval tables declare the role of each column — grade, categorical,
density, flag or ignore — which decides how it is composited. Roles can be
redeclared afterwards
* Interval data is checked on entry: overlapping intervals, non-positive
lengths, missing depths and holes with no collar are reported as errors, and
gaps and intervals running past the end of the hole as warnings, with the
policy chosen by `on_error`
* New compositing methods, all of which return a new object with every table on
one shared support: `composite()` (fixed length, honouring the boundaries of a
named categorical domain), `composite_fixed()` and `composite_to()` (onto the
intervals of another table). Numeric columns are averaged weighted by length,
or by length times density where a density column is declared, skipping
missing values; categories are decided by the greatest length in the composite
* `as_point_data()` converts the interval data to points, merging the tables
row for row. Compositional and vector variables are assembled here, with the
processing raw assays need, in order: each column is divided by its unit
("%", "ppm", "g/t", …, or a number), which has to be declared and is what
lets parts measured in different units be added together; a sample missing any
one part is marked missing entirely, since the parts only carry information
relative to each other; non-positive parts are replaced by half the smallest
positive value of their own column, the usual substitution for values below
detection, which a log-ratio transform cannot otherwise take; and a rest part
is added, holding whatever is left of the whole, so a barren sample is nearly
all rest and a rich one much less so. Where the parts leave no room, the rest
is held at the smallest part found and the sample is scaled down to fit, with
a warning — which usually means a unit was declared wrongly
* The database can be cut down before it is used: `subset_region()` keeps the
holes whose collar falls within a bounding box, `subset_holes()` those named
explicitly, and `filter_intervals()` the intervals holding a given value — or,
with `whole_holes`, every hole in which that value was logged
* Logging codes can be lumped into modelling domains with
`group_categories()`, which takes a mapping from each new category to the codes
it takes in, and reports the codes left out. Since there is no GUI to click
them together, `category_legend()` lists the distinct codes with the number of
intervals and the metreage each accounts for, sorted by length and with a
`group` column ready to be edited: write it to a spreadsheet, fill the column
in, and read it straight back into `group_categories()`. Intervals carrying no
value get a row of their own, so the legend accounts for the whole table
* `fill_unlogged()` gives a category to the ground a log does not account for,
which is what a database holds when only the intervals of interest were
logged. Intervals that exist but carry no value take the new label, and the
depths no interval covers become intervals of their own, reaching from the
collar to the bottom of the hole — so a hole nobody logged comes out labelled
end to end, and the log covers every hole continuously afterwards
* `as_classification_input()` still produces points of zero effective support,
including the contacts between rock types carrying both neighbouring classes
* New `datasets.macpass()` loader for the Macmillan Pass drillhole database,
published by Fireweed Metals Corp. The data is not distributed with geoML; see
the docstring for where to get it and the terms that apply
* Memory-efficient data storage: variable arrays are now backed by NumPy in RAM
or chunked Zarr on disk, chosen automatically by size, so large projects no
longer need to hold every array in memory
* Simulations are stored as a single `(n_data, n_sim)` array (a dedicated
dimension) rather than a list of separate arrays
* New `variable.simulation(i)` and `variable.n_sim`: a single simulation can
again be handled as an attribute, with the usual `as_image()`, `as_cube()`,
`smooth()` and contouring helpers
* Grid and block coordinates are now generated lazily: a regular grid no longer
holds its full `(n_data, n_dim)` coordinate array in memory, but produces the
requested rows on demand, so very large grids can be built and predicted into
without the coordinates dominating RAM
* Point coordinates and `GaussianData`'s input variance are stored the same way
as variable arrays (NumPy in RAM or Zarr on disk, by size), so a large point
cloud no longer has to stay in memory
* Fixed a bug where the input variance was rebuilt for the whole object on every
prediction batch, making prediction quadratic in the number of points; it is now
generated for the requested batch only
* Fixed `GaussianData`: predicting into one no longer fails, subsetting keeps the
object's class and variance instead of degrading to `PointData`, and the variance
is included in `as_data_frame()`
* Containers can be persisted to and reloaded from a single Zarr store with
`to_zarr()` / `open()`, covering all point-based containers and variable types
* Trained models can be saved and loaded whole with `model.save()` and
`Model.open()`, in a new `persistence` module: structure, trained parameters
and training data are all kept, so a reloaded model predicts exactly as it did
and can be trained further. It remembers its variables and their types, and so
creates them on any new data object it predicts on. `Model.open(path, data=...)`
rebuilds the same model around a different data set
* Quantiles and probabilities are computed lazily in a single pass over the
simulations, without materializing them
* `reset_probabilities()` is now the inverse of `reset_quantiles()`: it takes
cutoff values and returns cumulative probabilities in (0, 1)
* Fixed prediction of categorical variables, which failed with a `TypeError`
after the prediction posterior started being cached between batches
* Fixed variables being degraded to a more basic type when built from an
existing one, either by predicting on a new data object or through
`from_data_frame()`: a categorical variable became a rock type variable, an
ordered rock type became an unordered one, and an anomaly became a plain binary
variable. The same applied to data objects, where `Blocks2D.from_bounding_box()`
and its relatives returned a plain grid
* `CategoricalVariable` can now be built from a data frame with a single
measurements column, instead of the two contact columns a rock type needs
* Fixed the `n_sim` argument being ignored when predicting with the standard
`GP` model, which failed for any value other than 50
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