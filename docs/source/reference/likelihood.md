# geoml.likelihood

`CensoredSimplex` is the likelihood for a `MassFractionVariable`: a zero
part scores the probability of its latent value lying below the point
where the part vanishes, and a part not assayed is integrated over every
value it could take. Its bound is a density on a different measure for
each pattern of zeros, so compare it with another likelihood by held-out
scores, never by the bound.

```{eval-rst}
.. automodule:: geoml.likelihood
   :members: Gaussian, CensoredSimplex, Laplace, Gamma, StudentT,
             EpsilonInsensitive, Huber, Mixture, LikelihoodMixture, MultivariateGaussian, MultivariateLaplace,
             MultivariateEpsilonInsensitive, MultivariateHuber, Bernoulli,
             CategoricalGaussianIndicator, GradientIndicator
   :show-inheritance:
```
