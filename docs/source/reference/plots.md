# geoml.plots

The figures, in two backends that draw the same set under the same names:
`Explorer` in matplotlib, to print, and `Interactive` in plotly, to look at.
Both are built from a container plus a choice of variable, and several
plotly figures can be linked on one page with `Dashboard`.

Three of them are only honest on data the model has not seen — `accuracy`,
`spread_check` and `variogram` — since at a training location a model
interpolates its own measurement.

## Matplotlib

```{eval-rst}
.. automodule:: geoml.plots.explorer
   :members: Explorer
   :show-inheritance:
```

## Plotly

```{eval-rst}
.. automodule:: geoml.plots.interactive
   :members: Interactive
   :show-inheritance:
.. automodule:: geoml.plots.dashboard
   :members: Dashboard
   :show-inheritance:
```

## The arithmetic behind them

`prepare` holds what both backends read and imports no plotting library, so
the numbers behind a figure can be had without drawing it — which is also
how they are tested.

```{eval-rst}
.. automodule:: geoml.plots.prepare
   :members: variable, variable_or_component, continuous_parts,
             component_names, prediction_values, spread_check, variogram,
             swath, categorical_swath, proportions, contact,
             grade_tonnage, training_curve, moving_average, realizations,
             realization_store
```

## Style

```{eval-rst}
.. automodule:: geoml.plots.style
   :members: context, use, PALETTE, SEQUENTIAL, TEMPLATE
```
