# geoml.catalogue

Every class and function a model or a script may use, described as data,
for programs that build geoML models without reading its code. Write it
with

```bash
python -m geoml.catalogue catalogue.json
```

or read it in Python with `geoml.catalogue.build()`. The same geoML version
writes the same bytes. Each entry is keyed by the dotted path a saved model
records, so a catalogue entry and a saved spec name a class the same way.

```{eval-rst}
.. automodule:: geoml.catalogue
   :members: build, dumps
```

What introspection cannot read off a class, the class declares in a
`_catalogue` attribute, assigned where its module ends: its category, its
parents, how its size follows from its arguments, whether inducing points
pass through it, and which variables a likelihood accepts. The package's
tests build every class against its declaration.

Since geoML 0.8.0 the catalogue is format 2. Every entry carries a
`description`, the docstring's account of when to use it and how it works;
every bound a constructor clips a value to is a constraint on its argument;
a size rule may name a `fallback` for a parameter left null; a categorical
likelihood names its `contour_rule`; `containers` lists what a script
builds or predicts into, with the methods it calls on each; and each
variable type lists the columns it writes, with their role and scale. The design record is
*The catalogue*, under internals.
