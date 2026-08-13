# A deliberate blank: TensorFlow's own stubs are partial, and where they are
# wrong they bury the annotations this package does declare under errors
# nobody can act on (`tf.matmul` overloads, `Tensor | None` from every
# `tf.Variable`). Declaring the module untyped makes every tensor `Any`, so
# the type check reports on the numpy/pandas/container layer -- the half a
# checker can verify -- and stays quiet about the half it cannot.
from typing import Any


def __getattr__(name: str) -> Any: ...
