# geoml.kernels

Covariance functions. A kernel is handed to a GP node together with a
{doc}`transform <transform>`, which is what carries the ranges and the
anisotropy: the kernel says how correlation falls with distance, and the
transform says what distance means.

```{eval-rst}
.. automodule:: geoml.kernels
   :members: Gaussian, Spherical, Exponential, Cubic, Matern32, Matern52,
             Constant, Linear, Cosine, Sum, Product, Scale
   :show-inheritance:
```
