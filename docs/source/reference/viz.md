# geoml.viz

Export rather than drawing: each of these writes something another tool
renders. The containers' own `as_pyvista` methods are documented with
{doc}`the containers <data>`; what lives here is what those exports need
around them.

## Diagrams

`to_dot` writes a model or a latent network as a Graphviz DOT diagram —
nodes for inputs, latent variables and variables, edges for the wiring, and
a shared node drawn once, which the printed tree cannot show. The module
imports nothing: Graphviz is needed to render the text, never to write it.

```{eval-rst}
.. automodule:: geoml.viz.graphviz
   :members: to_dot
```

## PyVista

```{eval-rst}
.. automodule:: geoml.viz.pyvista
   :members: structure_discs, camera_orbit
```

## Plotly

```{eval-rst}
.. automodule:: geoml.viz.plotly
   :members: aspect_ratio_2d
```
