# geoml.stats

## The package generator

One seed governs everything: call {func}`~geoml.stats.random.set_seed`
**before** the objects are built, and parameter initialization, the
training draws and the simulation stream all follow from it. A model's
options draw their own seed from this generator at construction, and a
saved model keeps the number it drew, so its simulations replay on reload.

```{eval-rst}
.. automodule:: geoml.stats.random
   :members: set_seed, rng, sobol_engine
```

## Distributions

Custom TensorFlow-Probability distributions, used by the likelihoods and
the spline warpings.

```{eval-rst}
.. automodule:: geoml.stats.probability
   :members: EpsilonInsensitive, Huber, SplineBased, SmoothEmpirical,
             BinnedEmpirical, EmpiricalGaussianMixture,
             hazen_plotting_positions, hazen_binned_plotting_points
   :exclude-members: kl_divergence, prob, cdf, log_prob, log_cdf,
                     survival_function, log_survival_function, quantile,
                     mean, stddev, variance, mode, entropy, sample,
                     cross_entropy, covariance
   :show-inheritance:
```

```{note}
The distribution methods every TensorFlow-Probability distribution has --
`sample`, `log_prob`, `quantile` and the rest -- are left out above: their
docstrings come from TFP's own base class, which copies them into each
subclass, and they document TFP rather than geoML. Read them in
[the TFP reference](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution).
```
