# 7. Simulation

The map of means answers one question per location, and most questions a
mine asks are not of that kind. What tonnage clears the cut-off? What
revenue, at prices nobody knows yet? How likely is this stope to pay?
These are functions of the *whole field at once*, and a nonlinear function
of an average is not the average of the function. That is the oldest
reason geostatistics simulates. This chapter covers how simulation works
here, and how derived quantities are computed from it without leaving the
data structures.

## 7.1 What a realization is

A realization is one coherent draw of the latent field at every requested
location, carrying the spatial correlation, warped out to the variable's
units. `predict(..., n_sim=...)` stores them beside the prediction, and
three properties are guaranteed rather than hoped for.

- **The noise is not in them.** Realizations are of the *ground*, which is
  chapter 4's doctrine. The assay scatter is integrated into the reported
  value and never drawn into the ensemble. Classical simulation practice
  conflates the two, and this package deliberately does not.
- **Batch invariance.** A location's simulated value does not depend on
  which batch it was computed in, so predicting twice, or in pieces, gives
  the same ensemble.
- **Reproducibility.** One call to `geoml.set_seed(...)` *before the model
  is built* fixes everything: initialization, training draws and the
  simulation stream. A saved model keeps the seed it drew, so its
  simulations replay on reload (chapter 11).

For tight simulation budgets there is one option worth knowing.
`GPOptions(qmc_simulations=True)` swaps the random draws for quasi-random
Sobol ones. Measured on Walker Lake, the ensemble mean lands 7–37× closer
to the exact posterior and the tail quantiles about 25% closer at the same
`n_sim`, and the ensemble's own spread pays nothing for it. It is off by
default only because saved models must replay the streams they recorded.

Storage respects scale. Realizations live in a chunked store that is read
in bands of rows, and `variable.simulation(i)` reads a single realization
as a column. A block model's ensemble can run to hundreds of gigabytes,
and nothing in the package, or in this manual, materializes one whole.

## 7.2 Derived variables: arithmetic on the ensemble

A net smelter return, a metal content, a revenue: these are functions of
the simulated variables, applied **realization by realization**, because
the answer's uncertainty comes from the inputs' and the function bends.
The container does this itself.

The model is chapter 4's, unchanged.

```python
import geoml
import numpy as np

geoml.set_seed(1234)
walker, walker_grid = geoml.datasets.walker()

warping = geoml.warping.ChainedWarping(
    geoml.warping.BoxCox(1),
    geoml.warping.ZScore(1))

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

subset = geoml.data.PointData.from_array(
    np.asarray(walker_grid.coordinates)[::20])
model.predict(subset, n_sim=50, include_noise=True)
```

`derive` walks the parents' realizations and applies the function once per
realization. The example is a payable value: only the part of the grade
above a cut-off earns anything, at a price nobody has set yet. The
optional `simulation=` keyword receives the realization's index, which is
how that *external* uncertainty rides along, one draw per realization.

```python
price = np.random.default_rng(7).normal(1.0, 0.2, size=50)

payable = subset.derive(
    "payable",
    lambda v, simulation: np.maximum(v - 300.0, 0.0) * price[simulation],
    ["V"])

print("mean of derived sims:  ",
      round(float(np.mean(subset.values("payable/prediction"))), 1))
print("function of the means: ",
      round(float(np.mean(
          np.maximum(subset.values("V/prediction") - 300.0, 0.0)
          * price.mean())), 1))
```

The two numbers differ, and the first is the right one. Wherever the grade
straddles the cut-off, some realizations pay and some do not, and applying
the hinge to the mean throws that half-paying ground away. The prediction
of a derived variable is the mean of its derived realizations, never the
function of the parents' predictions. The nonlinear case is the reason the
class exists.

What comes back is a `DerivedVariable`, and it is worth knowing how it
differs from the `ContinuousVariable` it is built on. Everything a
continuous variable can do, it can do: quantiles, cut-offs, contours,
grade–tonnage curves and Zarr persistence all work unchanged. What it
cannot do is go back into a model. Every model refuses it as an input,
because its uncertainty is inherited from its parents rather than
modelled, and the package will not let that distinction blur. Two smaller
differences follow from the same idea. Metadata may join the arguments as
per-location constants (`["V", "_metadata/density"]`), and unsimulated
variables are refused outright, since there is no ensemble to walk. The
*function* is not persisted either. A reloaded derived variable is data,
and rerunning the deriving script is how it is refreshed.

> **In the code.** `container.derive(names, function, arguments)` in
> `data/containers.py`, and `DerivedVariable` in `data/variables.py`. The
> banded walk and the column reads follow the same storage discipline as
> everything else in the package (chapter 10).

## Further reading

Chilès & Delfiner (2012, ch. 7) for why geostatistics simulates; the 0.6.2
changelog for the quasi-Monte-Carlo measurements; chapter 8 for what
realizations mean on blocks, where the same ensemble also decides where a
model should refine itself.

## References

Chilès, J.-P., & Delfiner, P. (2012). *Geostatistics: Modeling Spatial
Uncertainty* (2nd ed.). Wiley.
