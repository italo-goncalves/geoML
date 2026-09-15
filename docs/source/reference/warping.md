# geoml.warping

How a variable's own units are turned into the scale the latent Gaussian
lives on, and back. A warping is given to a likelihood, and chains compose
left to right — `Log` then `ZScore` then `Spline` is the usual shape for a
positive, skewed grade.

A warping's second return value is the **log** of its Jacobian
determinant, which is what makes the objective a density in the data's own
units rather than the latent one; chains add theirs. Each also declares
whether one component of its input can reach another of its output, which
decides how the likelihood integrates the noise.

```{eval-rst}
.. automodule:: geoml.warping
   :members: Identity, ZScore, Center, Scale, Log, Softplus, Sigmoid,
             BoxCox, YeoJohnson, Arcsinh, SinhArcsinh,
             Spline, ChainedWarping, Rotation, PCA, RobustPCA,
             CenteredLogRatio, ScaledSimplex, ContinuousNormalizingFlow,
             TensorProductFlow
   :show-inheritance:
```
