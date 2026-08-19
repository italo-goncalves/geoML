# API reference

The public modules, grouped by what they are for. Everything listed here is
importable from `geoml`; the modules not listed are internal and may change
without notice.

## Modelling

```{toctree}
:maxdepth: 1

models
latent
kernels
transform
warping
likelihood
```

## Data

```{toctree}
:maxdepth: 1

datasets
data
inducing
drillhole
geoh5
```

## Figures and export

```{toctree}
:maxdepth: 1

plots
viz
```

## Support

```{toctree}
:maxdepth: 1

metrics
storage
math
stats
```

```{note}
`parameter`, `persistence` and `storage`'s internals are not listed: they
are the machinery underneath, and they change without notice. The tensor
code inside the latent nodes and the likelihoods carries no type
annotations on purpose — see the typing note in `CLAUDE.md`.
```
