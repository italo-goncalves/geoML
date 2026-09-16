"""The catalogue (`geoml.catalogue`), and every claim in it checked.

A program building models without reading geoML's code offers what the
catalogue lists and trusts what it declares: how many parents a node takes,
how its size follows from its arguments, whether inducing points pass
through it, which variables a likelihood may be bound to. A public class
without a declaration fails here, and so does a declaration that says
something its class does not do.
"""
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import zarr

import geoml
import geoml.catalogue as catalogue
import geoml.kernels as kr
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.persistence as persistence
import geoml.transform as tr
import geoml.warping as wp

CATEGORIES = {"input", "latent", "function", "operation", "transform",
              "kernel", "covariance", "warping", "likelihood"}
REFERENCES = {"ref:kernel", "ref:covariance", "ref:transform", "ref:warping",
              "ref:likelihood", "data:PointData", "data:DirectionalData",
              "data:BlockSet3D"}
PLAIN = {"int", "float", "bool", "str", "enum", "node", "json"}
NODES = [c for c in catalogue.classes()
         if c.__module__ == "geoml.latent.network"]


def _known(kind):
    base = kind[:-2] if kind.endswith("[]") else kind
    return base in PLAIN | REFERENCES


@pytest.fixture(scope="module")
def built():
    return catalogue.build()


# --------------------------------------------------------------------------- #
# what is declared
# --------------------------------------------------------------------------- #
def test_every_public_class_declares_itself():
    """Each on its own class: a subclass would otherwise inherit its
    parent's declaration and say the wrong thing without a word."""
    missing = [catalogue.path(c) for c in catalogue.classes()
               if "_catalogue" not in c.__dict__]
    assert missing == []


def test_every_entry_speaks_the_vocabulary(built):
    for name, entry in built["classes"].items():
        assert entry["category"] in CATEGORIES, name
        assert entry["stability"] in ("public", "experimental", "internal")
        for p in entry["params"]:
            assert _known(p["type"]), (name, p["name"], p["type"])
            if p["type"] == "enum":
                assert p["constraints"]["choices"], (name, p["name"])


def test_every_declared_argument_is_one_the_constructor_takes():
    for cls in catalogue.classes():
        declared = set(cls.__dict__["_catalogue"].get("params", {}))
        taken = set(inspect.signature(cls).parameters)
        assert declared <= taken, (catalogue.path(cls), declared - taken)


def test_nothing_offered_is_json_by_accident(built):
    """`json` means there is no better word; a class that is offered says
    so on purpose, in its declaration, or names a real type."""
    for cls in catalogue.classes():
        entry = built["classes"][catalogue.path(cls)]
        if entry["stability"] == "internal":
            continue
        declared = cls.__dict__["_catalogue"].get("params", {})
        for p in entry["params"]:
            if p["type"] == "json":
                assert declared.get(p["name"], {}).get("type") == "json", \
                    (catalogue.path(cls), p["name"])


def test_every_node_takes_a_name(built):
    for name, entry in built["classes"].items():
        if entry["category"] in ("input", "latent", "function",
                                 "operation") \
                and entry["stability"] != "internal":
            assert "name" in [p["name"] for p in entry["params"]], name


def test_a_likelihood_accepts_the_variable_types_there_are(built):
    for name, entry in built["classes"].items():
        if entry["category"] == "likelihood":
            assert set(entry["accepts"]) <= set(built["variable_types"])


def test_the_workflow_names_what_the_catalogue_describes(built):
    named = []
    for step in built["workflow"].values():
        for value in step.values():
            named += value if isinstance(value, list) else [value]
    assert [n for n in named if n.startswith("geoml.")
            and n not in built["functions"]] == []
    assert built["workflow"]["fit"]["load"] == "geoml.models.VGPNetwork.open"
    for name in built["functions"]:
        catalogue.resolve(name)


def test_the_catalogue_says_which_versions_it_describes(built):
    assert built["catalogue_format"] == catalogue.CATALOGUE_FORMAT
    assert built["geoml_version"] == geoml.__version__
    assert built["persistence_format"] == persistence._GEOML_MODEL_FORMAT


def test_the_same_geoml_writes_the_same_bytes(tmp_path):
    """The reading side pins a catalogue by its hash, so two builds may not
    differ in a byte -- which a set iterated into the output would make
    them do, differently under each hash seed."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(geoml.__file__)))
    written = []
    for seed in ("0", "4242"):
        target = tmp_path / ("catalogue_%s.json" % seed)
        env = dict(os.environ, PYTHONHASHSEED=seed, TF_CPP_MIN_LOG_LEVEL="3",
                   PYTHONPATH=os.pathsep.join(
                       [root] + [p for p in [os.environ.get("PYTHONPATH")]
                                 if p]))
        subprocess.run([sys.executable, "-m", "geoml.catalogue",
                        str(target)], env=env, check=True)
        written.append(target.read_bytes())
    assert written[0] == written[1]
    assert written[0].decode("utf-8") == catalogue.dumps()


# --------------------------------------------------------------------------- #
# nodes: built against their declarations
# --------------------------------------------------------------------------- #
def _points(n=12, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return geoml.data.PointData.from_array(rng.uniform(0.0, 100.0, (n, dim)))


def _root(transform=None):
    geoml.set_seed(1234)
    return latent.BasicInput(
        _points(), transform=transform or tr.Isotropic(30.0))


def _gp(parent, size):
    return latent.BasicGP(parent, size=size)


def _directions():
    frame = pd.DataFrame({"c0": [20.0, 50.0, 80.0], "c1": [30.0, 60.0, 40.0],
                          "dx": [1.0, 0.0, 1.0], "dy": [0.0, 1.0, 0.0]})
    return geoml.data.DirectionalData(frame, ["c0", "c1"], ["dx", "dy"])


def _calls(cls):
    """`(args, kwargs)` pairs building `cls` on stand-in parents, at sizes
    that differ where a size can."""
    root = _root()
    if cls in (latent.BasicInput, latent.GaussianInput):
        return [((_points(),), {"transform": tr.Isotropic(30.0)}),
                ((_points(),), {"transform": tr.ProjectionTo1D(2)})]
    if cls is latent.GradientConstrainedInput:
        return [((_points(), _directions(),
                  kr.Covariance(kr.Gaussian(), tr.Isotropic(30.0))),
                 {"size": s}) for s in (1, 2)]
    if cls in (latent.BasicGP, latent.AdditiveGP, latent.UncertainInputGP,
               latent.MultiStructureGP, latent.RadialTrend):
        return [((root,), {"size": s}) for s in (1, 3)]
    if cls is latent.Linear:
        return [((_gp(root, 3),), {"size": s}) for s in (1, 2)]
    if cls is latent.SelectInput:
        return [((_gp(root, 3), [0, 2]), {}), ((_gp(root, 3), [1]), {})]
    if cls in (latent.Bias, latent.Scale, latent.Exponentiation):
        return [((_gp(root, s),), {}) for s in (1, 3)]
    if cls is latent.GPWalk:
        # the field must be the walker's size, and the root's is 2
        return [((_gp(root, 2),), {})]
    if cls in (latent.Stack, latent.Concatenate):
        return [((_gp(root, 1), _gp(root, 2)), {}), ((_gp(root, 3),), {})]
    if cls in (latent.LinearCombination, latent.Add,
               latent.ProductOfExperts, latent.Multiply):
        return [((_gp(root, 2), _gp(root, 2)), {}),
                ((_gp(root, 3), _gp(root, 3), _gp(root, 3)), {})]
    raise AssertionError("no stand-in parents for %s: add them to `_calls`"
                         % cls.__name__)


def _bound(cls, args, kwargs):
    arguments = inspect.signature(cls).bind(*args, **kwargs)
    arguments.apply_defaults()
    return arguments.arguments


def _parents(cls, arguments):
    param = cls._catalogue["parents"]["param"]
    if param is None:
        return []
    given = arguments[param]
    return list(given) if isinstance(given, tuple) else [given]


def _expected(rule, arguments, parents):
    kind = rule["rule"]
    if kind == "const":
        return rule["value"]
    if kind == "param":
        return arguments[rule["param"]]
    if kind == "len":
        return len(arguments[rule["param"]])
    if kind in ("same_as_parent", "common"):
        return parents[0].size
    if kind == "sum":
        return sum(p.size for p in parents)
    if kind == "input_dimension":
        points = arguments["inducing_points"]
        transform = arguments["transform"] or tr.Identity()
        return int(np.asarray(transform(np.ones([1, points.n_dim]))).shape[1])
    raise AssertionError("no reading for a %r size rule" % kind)


def _blocked(cls):
    """`cls`'s first call with a parent that passes no inducing points."""
    args, kwargs = _calls(cls)[0]
    first = args[0]
    product = latent.Multiply(_gp(first.root, first.size),
                              _gp(first.root, first.size))
    return (product,) + tuple(args[1:]), kwargs


def test_every_node_has_stand_in_parents():
    for cls in NODES:
        assert _calls(cls)


@pytest.mark.parametrize("cls", NODES, ids=lambda c: c.__name__)
def test_a_node_is_the_size_it_declares(cls):
    for args, kwargs in _calls(cls):
        node = cls(*args, **kwargs)
        arguments = _bound(cls, args, kwargs)
        assert node.size == _expected(cls._catalogue["size"], arguments,
                                      _parents(cls, arguments))


@pytest.mark.parametrize(
    "cls", [c for c in NODES if c._catalogue["size"]["rule"] == "common"],
    ids=lambda c: c.__name__)
def test_a_node_of_common_size_refuses_two_sizes(cls):
    root = _root()
    with pytest.raises(latent.SizeIncompatibilityError):
        cls(_gp(root, 1), _gp(root, 2))


@pytest.mark.parametrize("cls", NODES, ids=lambda c: c.__name__)
def test_inducing_points_pass_through_a_node_as_it_declares(cls):
    claim = cls._catalogue["propagates_inducing"]
    args, kwargs = _calls(cls)[0]
    node = cls(*args, **kwargs)
    if claim is False:
        with pytest.raises(latent.BrokenPropagationError):
            _gp(node, 1)
        return
    _gp(node, 1)
    parents = cls._catalogue["parents"]
    if claim == "parents" and parents["max"] != 0 \
            and not cls._catalogue["requires_propagation"] \
            and "category" not in parents:
        # and nothing passes through where a parent passes nothing on
        args, kwargs = _blocked(cls)
        with pytest.raises(latent.BrokenPropagationError):
            _gp(cls(*args, **kwargs), 1)


@pytest.mark.parametrize(
    "cls", [c for c in NODES if c._catalogue["parents"]["max"] != 0
            and "category" not in c._catalogue["parents"]],
    ids=lambda c: c.__name__)
def test_a_node_needs_inducing_points_as_it_declares(cls):
    args, kwargs = _blocked(cls)
    if cls._catalogue["requires_propagation"]:
        with pytest.raises(latent.BrokenPropagationError):
            cls(*args, **kwargs)
    else:
        cls(*args, **kwargs)


# --------------------------------------------------------------------------- #
# transforms and warpings: the widths they declare
# --------------------------------------------------------------------------- #
def _line(y=50.0):
    t = np.linspace(20.0, 80.0, 13)
    return np.stack([t, np.full_like(t, y)], axis=1), \
        np.tile([0.0, -1.0], [13, 1])


def _plane():
    y, z = np.meshgrid(np.linspace(0.0, 100.0, 5), np.linspace(0.0, 100.0, 5))
    points = np.stack([np.full(y.size, 50.0), y.ravel(), z.ravel()], axis=1)
    return points, np.tile([1.0, 0.0, 0.0], [len(points), 1])


def _transform_calls(cls):
    """`(args, kwargs, width in)` building `cls`."""
    line, down = _line()
    plane, across = _plane()
    column = np.stack([np.full(13, 50.0), np.linspace(20.0, 80.0, 13)],
                      axis=1)
    return {
        tr.Identity: ((), {}, 3), tr.Isotropic: ((2.0,), {}, 3),
        tr.Anisotropy2D: ((), {}, 2), tr.Anisotropy2DMath: ((), {}, 2),
        tr.Anisotropy2DDynamic: ((), {}, 2), tr.Anisotropy3D: ((), {}, 3),
        tr.Anisotropy3DMath: ((), {}, 3), tr.Anisotropy3DDynamic: ((), {}, 3),
        tr.ProjectionTo1D: ((3,), {}, 3), tr.AnisotropyARD: ((3,), {}, 3),
        tr.ChainedTransform: ((tr.Isotropic(2.0), tr.ProjectionTo1D(3)), {},
                              3),
        tr.SelectVariables: (([0, 2],), {}, 3),
        tr.NormalizeWithBoundingBox: (
            (geoml.data.BoundingBox.from_array(np.array([[0.0, 0.0, 0.0],
                                                        [1.0, 2.0, 3.0]])),),
            {}, 3),
        tr.Periodic: ((), {}, 3),
        tr.Concatenate: ((tr.Identity(), tr.ProjectionTo1D(3)), {}, 3),
        tr.RandomProjections: ((3, 5), {}, 3),
        tr.BellFault2D: ((np.array([20.0, 50.0]), np.array([80.0, 50.0])),
                         {}, 2),
        tr.ImplicitFault: ((line, down), {"reach": 30.0}, 2),
        tr.FaultDisplacement: ((plane, across), {"throw": 5.0,
                                                 "reach": 1000.0}, 3),
        tr.FaultNetwork: (([tr.FaultDisplacement(plane, across, throw=5.0,
                                                 reach=1000.0)],), {}, 3),
        tr.ImplicitFaultBlocks: (
            ([tr.ImplicitFault(line, down, reach=30.0),
              tr.ImplicitFault(column, np.tile([1.0, 0.0], [13, 1]),
                               reach=30.0)],), {}, 2),
    }[cls]


def _width(rule, arguments, given):
    if rule is None:
        return given
    kind = rule["rule"]
    if kind == "same_as_parent":
        return given
    if kind == "sum":
        return sum(int(np.asarray(t(np.ones([1, given]))).shape[1])
                   for t in arguments["transforms"])
    return _expected(rule, arguments, [])


TRANSFORMS = [c for c in catalogue.classes()
              if c.__module__ == "geoml.transform"]
WARPINGS = [c for c in catalogue.classes() if c.__module__ == "geoml.warping"]


@pytest.mark.parametrize("cls", TRANSFORMS, ids=lambda c: c.__name__)
def test_a_transform_takes_and_gives_the_widths_it_declares(cls):
    args, kwargs, width = _transform_calls(cls)
    transform = cls(*args, **kwargs)
    arguments = _bound(cls, args, kwargs)
    size = cls._catalogue["size"]
    if size["in"] is not None:
        assert width == _width(size["in"], arguments, width)
    x = np.random.default_rng(0).uniform(0.0, 100.0, (7, width))
    out = np.asarray(transform(tf.constant(x))).shape[1]
    if size["out"]["rule"] != "custom":
        assert out == _width(size["out"], arguments, width)


def _warping_calls(cls):
    return {wp.Identity: (3,), wp.Spline: (3,), wp.ZScore: (3,),
            wp.Center: (3,), wp.Softplus: (3,), wp.Log: (3,), wp.Scale: (3,),
            wp.BoxCox: (3,), wp.YeoJohnson: (3,), wp.Arcsinh: (3,),
            wp.SinhArcsinh: (3,), wp.Sigmoid: (3,),
            wp.ContinuousNormalizingFlow: (2,), wp.TensorProductFlow: (2,),
            wp.PCA: (4, 2), wp.RobustPCA: (4, 2), wp.CenteredLogRatio: (3,),
            wp.Rotation: (3,), wp.ScaledSimplex: (3,),
            wp.ChainedWarping: (wp.ZScore(3), wp.PCA(3, 2))}[cls]


@pytest.mark.parametrize("cls", WARPINGS, ids=lambda c: c.__name__)
def test_a_warping_takes_and_gives_the_widths_it_declares(cls):
    args = _warping_calls(cls)
    warping = cls(*args)
    arguments = _bound(cls, args, {})
    size = cls._catalogue["size"]
    if size["in"]["rule"] != "custom":
        assert warping.size_in == _expected(size["in"], arguments, [])
    if size["out"]["rule"] == "same_as_parent":
        assert warping.size_out == warping.size_in
    elif size["out"]["rule"] != "custom":
        assert warping.size_out == _expected(size["out"], arguments, [])


# --------------------------------------------------------------------------- #
# likelihoods: every pairing they declare, built and trained a step
# --------------------------------------------------------------------------- #
_VARIABLE_NAMES = {"continuous": "continuous", "vector": "vector",
                   "compositional": "composition", "rock_type": "rock_type",
                   "categorical": "categorical",
                   "ordered_rock_type": "ordered", "binary": "binary",
                   "anomaly": "anomaly"}


@pytest.fixture(scope="module")
def every_variable():
    n = 24
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 100.0, (n, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=["X", "Y"]),
                                 ["X", "Y"])
    rock = np.where(coords[:, 0] > 50.0, "granite", "basalt")
    point.add_continuous_variable("continuous", rng.uniform(1.0, 2.0, n))
    point.add_vector_variable("vector", ["a", "b"],
                              rng.uniform(1.0, 2.0, (n, 2)))
    point.add_compositional_variable(
        "composition", ["s", "t"], rng.dirichlet([3.0, 4.0], n))
    point.add_rock_type_variable("rock_type", labels=("granite", "basalt"),
                                 measurements_a=rock)
    point.add_categorical_variable("categorical", ("granite", "basalt"), rock)
    point.add_rock_type_variable("ordered", labels=("granite", "basalt"),
                                 measurements_a=rock, ordered=True)
    point.add_binary_variable("binary", labels=("basalt", "granite"),
                              measurements=rock)
    point.add_anomaly_variable("anomaly", "granite", measurements=rock)
    return point


def _likelihood(cls, length):
    """`cls` sized for a variable of `length` columns; Gamma warped by the
    identity, its data being positive and its density zero below zero."""
    if cls in (lk.MultivariateGaussian, lk.MultivariateLaplace,
               lk.MultivariateEpsilonInsensitive, lk.MultivariateHuber):
        return cls(length)
    if cls is lk.Gamma:
        return cls(wp.Identity(length))
    if cls is lk.Mixture:
        return cls(wp.ZScore(length))
    if cls in (lk.CategoricalGaussianIndicator,
               lk.HierarchicalGaussianIndicator):
        return cls(2)
    if cls is lk.OrderedGaussianIndicator:
        return cls(1)
    if cls in (lk.Bernoulli, lk.BernoulliMaximumMargin):
        return cls()
    return cls(wp.ZScore(length))


PAIRINGS = [(cls, kind) for cls in catalogue.classes()
            if cls.__module__ == "geoml.likelihood"
            for kind in cls._catalogue["accepts"]]


@pytest.mark.parametrize("cls,kind", PAIRINGS,
                         ids=lambda v: getattr(v, "__name__", v))
def test_a_likelihood_trains_on_every_variable_type_it_accepts(
        cls, kind, every_variable):
    name = _VARIABLE_NAMES[kind]
    variable = every_variable.variables[name]
    likelihood = _likelihood(cls, variable.length)
    geoml.set_seed(1234)
    root = latent.BasicInput(_points(), transform=tr.Isotropic(30.0))
    model = geoml.models.VGPNetwork(
        every_variable, name, likelihood,
        latent.BasicGP(root, size=likelihood.size),
        options=geoml.models.GPOptions(verbose=False, training_samples=4))
    model.train_full(max_iter=1)
    assert np.isfinite(model.training_log[-1])


# --------------------------------------------------------------------------- #
# persistence: every class offered survives a save
# --------------------------------------------------------------------------- #
def _instances(cls):
    if cls in NODES:
        args, kwargs = _calls(cls)[0]
        return cls(*args, **kwargs)
    if cls in TRANSFORMS:
        args, kwargs, _ = _transform_calls(cls)
        return cls(*args, **kwargs)
    if cls in WARPINGS:
        return cls(*_warping_calls(cls))
    if cls.__module__ == "geoml.likelihood":
        return _likelihood(cls, 1)
    covariance = kr.Covariance(kr.Gaussian(), tr.Isotropic(2.0))
    return {kr.Covariance: lambda: covariance, kr.Linear: kr.Linear,
            kr.Sum: lambda: kr.Sum(covariance,
                                   kr.Covariance(kr.Cubic())),
            kr.Product: lambda: kr.Product(covariance,
                                           kr.Covariance(kr.Cubic())),
            kr.Scale: lambda: kr.Scale(covariance)}.get(cls, cls)()


OFFERED = [c for c in catalogue.classes()
           if c._catalogue.get("stability", "public") != "internal"]


@pytest.mark.parametrize("cls", OFFERED, ids=lambda c: c.__name__)
def test_every_class_offered_survives_a_save(cls, tmp_path):
    """A model is saved by replaying its constructors, so a class that
    records no arguments, or takes one no store can hold, would build in
    the editor and fail at the save."""
    obj = _instances(cls)
    store = str(tmp_path / "object.zarr")
    group = zarr.open_group(store, mode="w")
    node = persistence._encode(obj, persistence._Writer(group, store))
    back = persistence._decode(json.loads(json.dumps(node)),
                               persistence._Reader(group, store))
    assert type(back) is cls
