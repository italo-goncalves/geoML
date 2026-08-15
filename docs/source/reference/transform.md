# geoml.transform

What distance means before a {doc}`kernel <kernels>` sees it: the ranges,
the anisotropy ellipsoid, and the projections. Chained transforms compose,
so a periodic direction and an anisotropic one can sit in the same model.

## Isotropic and anisotropic

```{eval-rst}
.. automodule:: geoml.transform
   :members: Identity, Isotropic, Anisotropy2D, Anisotropy3D, AnisotropyARD
   :show-inheritance:
```

## Angles given rather than fitted

The `Math` and `Dynamic` variants take the ellipsoid's orientation from
somewhere other than training — a structural measurement, or another node.

```{eval-rst}
.. currentmodule:: geoml.transform
.. autoclass:: Anisotropy2DMath
   :show-inheritance:
.. autoclass:: Anisotropy2DDynamic
   :show-inheritance:
.. autoclass:: Anisotropy3DMath
   :show-inheritance:
.. autoclass:: Anisotropy3DDynamic
   :show-inheritance:
```

## Composing and reshaping

```{eval-rst}
.. currentmodule:: geoml.transform
.. autoclass:: ChainedTransform
   :show-inheritance:
.. autoclass:: Concatenate
   :show-inheritance:
.. autoclass:: SelectVariables
   :show-inheritance:
.. autoclass:: ProjectionTo1D
   :show-inheritance:
.. autoclass:: RandomProjections
   :show-inheritance:
.. autoclass:: NormalizeWithBoundingBox
   :show-inheritance:
.. autoclass:: Periodic
   :show-inheritance:
.. autoclass:: BellFault2D
   :show-inheritance:
```
