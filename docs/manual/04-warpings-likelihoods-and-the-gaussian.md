# 4. Warpings, likelihoods, and where the Gaussian sits

"Grades are not Gaussian" is true and beside the point. In this package
the Gaussian assumption lives in exactly one place, the **latent field**
$f(\mathbf{x})$, and the data reaches it through two adapters that are
part of the model and trained with it. A **warping** reshapes the
variable's distribution, and a **likelihood** describes how a measurement
scatters around the ground. This chapter is about both, and about the
doctrine that falls out of keeping them separate: a prediction and an
assay are different kinds of number.

## 4.1 Warpings: normal scores, made trainable

The normal score transform is standard practice. Rank the data, map it to
Gaussian quantiles, model, back-transform. A warping is the same idea as a
*parametric, trainable, invertible* function $g$, with the model working
on $g^{-1}(z)$ and every prediction returned through $g$. Because it is
part of the model, its parameters are fitted by the same ELBO, and because
it is a function rather than a rank table, it extrapolates beyond the
data's range in a controlled way. That is where the rank version is
weakest.

Warpings chain, and the chain is read data-side first:

- `Log`, `Softplus` for positivity: grades, thicknesses.
- `Scale` divides by the data's range, so what follows starts on
  well-behaved numbers instead of on assay units.
- `ZScore` centres and scales. `ZScore(size, robust=True)` initializes on
  winsorized data, so that one gross outlier cannot set the scale.
- `Spline` is a monotone spline for asymmetry the fixed links do not
  catch, and it is the trainable heart of most chains.
- `PCA`, `RobustPCA` and `Rotation` are the multivariate links, which
  decorrelate the columns so the latent fields can be modelled
  independently.
- `CenteredLogRatio` handles compositions, off the simplex and back.

A chain that returns only positive values is the everyday case, and it is
worth spelling out once:

```
data  ->  Scale  ->  Softplus  ->  ZScore  ->  Spline  ->  latent field
```

Read backwards, the field passes through the spline, is un-standardized,
goes through a softplus (which cannot be negative), and is scaled back to
assay units by a positive factor. No realization can come back negative,
which is exactly the defect chapter 2 left open. Chapters 11 and 15 use
this chain as their default.

One distinction the package tracks carefully: a warping is
**elementwise** if each output component depends on its own input alone
(`Softplus`, `Spline`), and *mixing* if not (`PCA`, `Rotation`,
`CenteredLogRatio`). The flag decides how the noise integrals below are
computed, a cheap per-component quadrature for elementwise chains and a
genuinely multivariate rule for mixing ones. Every warping's declaration
is checked against a numerical Jacobian in the test suite, because a false
claim would make the model integrate the wrong thing.

## 4.2 Likelihoods: what the nugget really is

The likelihood is the measurement model, and the familiar choices line up
with familiar estimators:

- `Gaussian` gives squared error, the classical choice, and the one that
  recovers kriging exactly.
- `Laplace` and `Huber` are the absolute-error flavours, where heavy tails
  survive without dragging the fit.
- `EpsilonInsensitive` is the GP analogue of support-vector regression:
  errors smaller than $\epsilon$ cost nothing, which is a natural reading
  of assay precision.
- `Mixture` models the noise as several scales of one family sharing the
  same latent location, classically a narrow component for genuine
  readings and a wide one for bad ones. A heavy tail merely *survives* an
  outlier, while a mixture **names it**: `responsibilities` reports, per
  sample, how likely each component is to have produced that reading. Its
  docstring carries the options, and `ZScore(robust=True)` is the warping
  to lead it with.

The nugget of chapter 2 is now visible for what it is: the `noise`
parameter of the likelihood, measurement error plus sub-resolution
variability, sitting on the latent scale and *outside* the ground.

## 4.3 The doctrine: noise is integrated out, never drawn

What should a prediction report? This package's answer, everywhere and
without exception, is the **ground**: the value left once the measurement
scatter is averaged over,

$$
h(\mathbf{x}) = \mathbb{E}_\epsilon\left[\,g(f(\mathbf{x}) +
\epsilon)\,\right],
$$

computed by quadrature rather than by simulating noise. Three consequences
follow, each easy to trip over coming from classical simulation practice.

1. **The correction is deterministic and one-directional.** On a skewed
   variable $g$ is convex, so by Jensen's inequality leaving the noise out
   biases every value low. Turning `include_noise` off is not a neutral
   simplification, it is a different and usually wrong answer.
2. **Simulated realizations are of the ground too.** They carry the
   model's uncertainty, not the assay scatter, which is why comparing them
   directly to held-out assays is a category error.
3. **When an assay is the question, ask for a measurement.**
   `predict_measurements` returns the predictive distribution of a
   *sample*, the same computation stopped one step earlier, and it is what
   every honest comparison against data uses. Chapter 13 is built on it.

Three variances, three questions, one table worth memorizing:

| column | question it answers |
|---|---|
| `latent_variance` | how sure is the model about the ground here? |
| `dispersion` | how much does the ground vary *inside* this block? (chapter 8) |
| `noise_variance` | how far would an assay of this value scatter? |

## 4.4 Seeing the doctrine on Walker Lake

The model is chapter 3's, with a positivity chain in place of the plain
`ZScore`. The inducing points are the same grid of experts, so the only
thing that changed between the two chapters is the warping.

```python
import geoml
import numpy as np

geoml.set_seed(1234)
walker, walker_grid = geoml.datasets.walker()

# positivity first, then centring, then a trainable spline for the skew
warping = geoml.warping.ChainedWarping(
    geoml.warping.Softplus(1),
    geoml.warping.ZScore(1),
    geoml.warping.Spline(1, knots_per_arm=4))

experts = geoml.data.inducing.grid_experts(walker_grid, 10.0, block=8)

root = geoml.latent.BasicInput(
    experts,
    transform=geoml.transform.Isotropic(50))

gp = geoml.latent.BasicGP(
    root,
    size=1,
    kernel=geoml.kernels.Spherical())

model = geoml.models.VGPNetwork(
    walker, "V",
    geoml.likelihood.Gaussian(warping),
    gp,
    options=geoml.models.GPOptions(verbose=False))

model.train_full(max_iter=250)
```

First consequence, the Jensen shift. The same locations predicted with
and without the noise integral:

```python
subset = geoml.data.PointData.from_array(
    np.asarray(walker_grid.coordinates)[::50])

model.predict(subset, n_sim=20, include_noise=True)
with_noise = subset.values("V/prediction").copy()

model.predict(subset, n_sim=20, include_noise=False)
without = subset.values("V/prediction")

print("mean shift:", round(float(np.mean(with_noise - without)), 2))
```

The shift is positive. The integrated prediction sits above the naive
back-transform, exactly as Jensen says it must on a convex $g$, and it is
not small on a variable this skewed.

Second consequence, ground versus measurement. The stored simulations
against the measurement samples, at the training locations:

```python
model.predict(walker, n_sim=20, include_noise=True)
samples = model.predict_measurements(walker, n_sim=20)["V"]

# 470 locations by 20 realizations reads whole without trouble; on a block
# model it would be read a band at a time instead (chapter 10)
ground = np.asarray(walker.get("V/simulations"))

print("ground spread:     ",
      round(float(np.mean(np.std(ground, axis=1))), 1))
print("measurement spread:",
      round(float(np.mean(np.std(samples[:, 0, :], axis=1))), 1))
```

The measurement distribution is the wider one, because it carries the
assay scatter the ground has had averaged out, and the gap between the two
numbers is the noise the model fitted. Which of them an interval should be
cut from depends only on what the interval is *for*: ground for resources,
measurements for checking against assays.

> **In the code.** Warpings live in `geoml.warping`, each declaring
> `elementwise` truthfully, and likelihoods in `geoml.likelihood`. The
> noise integral is `integrated_backward`, which also returns
> `noise_variance` for free, and `models.predict_measurements` is the
> measurement-side door. The node counts behind the integrals (eight
> Gauss–Hermite for elementwise chains, 64 Sobol for mixing ones) were set
> by measurement on real assays, and `test_noise_integration.py` owns the
> machinery.

## Further reading

The 2022 regression paper for warped variational GPs and the robust
likelihoods on real data; the 2020 sunspot paper for warping a bounded,
asymmetric series; Rasmussen & Williams (2006, §9) for warped GPs in the
classical setting.

## References

Gonçalves, Í. G. *et al.* (2020). Sunspot cycle prediction using warped
Gaussian process regression. *Advances in Space Research*.
<https://www.sciencedirect.com/science/article/pii/S0273117719308026>

Gonçalves, Í. G. *et al.* (2022). Learning spatial patterns with
variational Gaussian processes: regression. *Computers & Geosciences*.
<https://doi.org/10.1016/j.cageo.2022.105056>

Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
Machine Learning*. MIT Press.
