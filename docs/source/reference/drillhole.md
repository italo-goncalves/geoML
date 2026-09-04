# geoml.data.drillhole

An interval table gives each of its columns a **role**, which decides how a
composite is formed, and may give it a **unit** — `units={"Pb": "%", "Ag":
"ppm"}` at construction, or afterwards through
`IntervalTable.set_unit(column, unit)` for one column and
`DrillholeData.set_unit(table, {column: unit})` for a table at a time. The
unit travels with the column through renaming and compositing and lands on
the variable the conversion builds, where it names an axis and, for a part
of a composition, converts the part to a fraction of the whole at the
model's door.

```{eval-rst}
.. automodule:: geoml.data.drillhole
   :members: DrillholeData, IntervalTable
   :show-inheritance:
```
