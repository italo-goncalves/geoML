# geoml.latent

The nodes a model's latent network is composed from. A network is built
bottom-up — an input node, one or more GP nodes above it, operations that
combine them — and handed to
{class}`~geoml.models.VGPNetwork` as its `latent_network`.

The signatures here are unannotated on purpose: everything below the
constructors is tensor code, where a `tf.Tensor` in and a `tf.Tensor` out
says nothing about the rank, dtype or axis order that actually goes wrong.
The constructors' own arguments are documented in their docstrings.

## Inputs

```{eval-rst}
.. automodule:: geoml.latent.network
   :members: BasicInput, GradientConstrainedInput
   :show-inheritance:
```

## Gaussian process nodes

```{eval-rst}
.. currentmodule:: geoml.latent.network
.. autoclass:: BasicGP
   :members:
   :show-inheritance:
.. autoclass:: AdditiveGP
   :show-inheritance:
.. autoclass:: MultiStructureGP
   :show-inheritance:
```

## Combining and reshaping

```{eval-rst}
.. currentmodule:: geoml.latent.network
.. autoclass:: Add
   :show-inheritance:
.. autoclass:: Multiply
   :show-inheritance:
.. autoclass:: LinearCombination
   :show-inheritance:
.. autoclass:: Linear
   :show-inheritance:
.. autoclass:: ProductOfExperts
   :show-inheritance:
.. autoclass:: Stack
   :show-inheritance:
.. autoclass:: Concatenate
   :show-inheritance:
.. autoclass:: SelectInput
   :show-inheritance:
```

## Shaping a field

```{eval-rst}
.. currentmodule:: geoml.latent.network
.. autoclass:: Bias
   :show-inheritance:
.. autoclass:: Scale
   :show-inheritance:
.. autoclass:: Exponentiation
   :show-inheritance:
.. autoclass:: RadialTrend
   :show-inheritance:
.. autoclass:: GPWalk
   :show-inheritance:
```

## When nodes do not fit together

```{eval-rst}
.. currentmodule:: geoml.latent.network
.. autoexception:: NodeIncompatibilityError
.. autoexception:: BrokenPropagationError
.. autoexception:: SizeIncompatibilityError
```
