# geoML

Machine learning models for spatial and geoscientific data, built on
**variational Gaussian processes**. Kriging's assumptions, without kriging's
limits: variables of any kind in one model, non-Gaussian data through trained
warpings, block support without a change-of-support formula, and simulation
throughout.

```{code-block} python
import geoml

geoml.set_seed(1234)
walker, grid = geoml.datasets.walker()

inducing = geoml.data.inducing.from_kmeans(walker, 100, seed=0)
gp = geoml.latent.BasicGP(
    geoml.latent.BasicInput(inducing,
                            transform=geoml.transform.Isotropic(50)),
    size=1, kernel=geoml.kernels.Spherical())
warping = geoml.warping.ChainedWarping(
    geoml.warping.Softplus(1), geoml.warping.ZScore(1))

model = geoml.models.VGPNetwork(
    walker, "V", geoml.likelihood.Gaussian(warping), gp)
model.train_full(max_iter=300)
model.predict(grid, n_sim=50)

grid.variables["V"].reset_quantiles([0.05, 0.5, 0.95])
```

## Installing

```{code-block} bash
pip install -e .
```

Python 3.10 or newer. The dependencies are declared in `pyproject.toml`;
`pip install -e .[dev]` adds what the documentation and the type check need.

## Where to go

- **[The API reference](reference/index)** documents every public class and
  function, module by module.
- **[Internals](internals/index)** collects the design records: what was
  built, what was measured, and why the package settled where it did.

```{toctree}
:maxdepth: 2
:hidden:

reference/index
internals/index
```

## Citing

geoML implements the methods of a series of papers on Gaussian processes for
geological modelling; see the repository's README for the current list.

## Licence

GPL-3, dual-licensed. See the repository for details.
