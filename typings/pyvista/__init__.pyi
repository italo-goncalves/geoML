# Untyped on purpose, as `typings/tensorflow` is: PyVista ships partial
# annotations, and the gaps (`pyvista.PolyData` among them) would report as
# errors on code that works. Declaring the module untyped keeps the check on
# the layer it can verify.
from typing import Any


def __getattr__(name: str) -> Any: ...
