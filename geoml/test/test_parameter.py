"""Tests for the Parametric base class, focused on shared-parameter dedup.

A Parametric shared between two parents (e.g. one transform reused by several
kernels) must contribute its parameters exactly once to the top-level flat list,
to get_parameter_values, and to the saved state.
"""
import pickle

import numpy as np

import geoml.parameter as gpr


class _Leaf(gpr.Parametric):
    def __init__(self, value=1.0):
        super().__init__()
        self._add_parameter("p", gpr.RealParameter(value, 0.0, 10.0))

    def pretty_print(self, depth=0):
        return "Leaf"


class _Node(gpr.Parametric):
    """Registers whatever children it is given."""

    def __init__(self, *children):
        super().__init__()
        self.children = [self._register(c) for c in children]

    def pretty_print(self, depth=0):
        return "Node"


def test_distinct_parameters_all_counted():
    top = _Node(_Leaf(), _Leaf())
    assert len(top.all_parameters) == 2
    value, shape, position, mn, mx = top.get_parameter_values(complete=True)
    assert len(value) == 2


def test_shared_parameter_counted_once():
    leaf = _Leaf()
    a = _Node(leaf)
    b = _Node(leaf)               # same leaf shared by two parents
    top = _Node(a, b)

    assert len(a.all_parameters) == 1
    assert len(top.all_parameters) == 1     # not 2
    value, shape, position, mn, mx = top.get_parameter_values(complete=True)
    assert len(value) == 1
    assert len(top.get_unfixed_variables()) == 1


def test_add_parameter_is_idempotent_by_identity():
    node = _Node()
    p = gpr.RealParameter(2.0, 0.0, 10.0)
    node._add_parameter("x", p)
    node._add_parameter("x_again", p)       # same object, another name
    assert len(node.all_parameters) == 1
    assert len(node.parameters) == 2


def test_shared_parameter_save_load_roundtrip(tmp_path):
    leaf = _Leaf(value=3.0)
    top = _Node(_Node(leaf), _Node(leaf))

    path = str(tmp_path / "state.pkl")
    top.save_state(path)

    with open(path, "rb") as f:
        value, shape, position, mn, mx = pickle.load(f)
    assert len(value) == 1                   # saved once, not twice

    leaf.parameters["p"].set_value(7.0)
    assert np.isclose(float(leaf.parameters["p"].get_value()), 7.0)

    top.load_state(path)
    assert np.isclose(float(leaf.parameters["p"].get_value()), 3.0)
