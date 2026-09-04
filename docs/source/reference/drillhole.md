# geoml.data.drillhole

An interval table gives each of its columns a **role**, which decides how a
composite is formed, and may give it a **unit** — `units={"Pb": "%", "Ag":
"ppm"}`, or `set_unit` afterwards. The unit travels with the column through
renaming and compositing and lands on the variable the conversion builds,
where it names an axis and, for a part of a composition, converts the part
to a fraction of the whole at the model's door.

```{eval-rst}
.. automodule:: geoml.data.drillhole
   :members: DrillholeData, IntervalTable
   :show-inheritance:
```
