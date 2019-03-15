.. geoml documentation master file, created by
   sphinx-quickstart on Mon Mar 11 15:07:40 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

geoml - Machine Learning for geospatial data
============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This package aims to provide a number of machine learning models
especially adapted to work with geologic and spatial data in
general. Spatial data is different from the typical machine
learning dataset in many aspects:

* The data is by definition low-dimensional, so a deep learning model
  may struggle to perform adequately;
* There may be information about the gradient of the variable being
  modeled (wind speed, geological foliations, etc.) that one wishes to
  include in the modeling process;
* In problems such as geological modeling, the boundary between class
  labels is of special importance, and in fact one may have data points
  that lie exactly in such boundaries.

So far all implemented models are based on the Gaussian Process (GP)
model (also known as kriging) which is already widely used for modeling
spatial data. Parameters such as range, sill, anisotropy, etc.
need not be explored manually, as they are optimized automatically by
means of a genetic algorithm accelerated with TensorFlow.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
