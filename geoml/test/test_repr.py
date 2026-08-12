"""What objects look like when printed.

Two contracts. ``repr`` identifies an object on one line, as the call that would
build it, and must never raise — every kernel and every likelihood used to raise
``NotImplementedError``, which took a debugger or a failing assertion down with
it. ``str`` lays out the state: parameter values, and everything registered
inside the object.
"""
import inspect

import numpy as np
import pytest

import geoml
import geoml.parameter as gpr


def _buildable(module, base):
    """Every class in ``module`` that can be built with no arguments."""
    found = []
    for name, cls in vars(module).items():
        if not isinstance(cls, type) or name.startswith("_"):
            continue
        if not issubclass(cls, base) or cls is base:
            continue
        required = [p for p in inspect.signature(cls).parameters.values()
                    if p.default is p.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if len(required) > 0:
            continue
        try:
            found.append(pytest.param(cls(), id=name))
        except Exception:      # needs more than its signature says
            continue
    return found


KERNELS = _buildable(geoml.kernels, gpr.Parametric)
LIKELIHOODS = _buildable(geoml.likelihood, gpr.Parametric)
TRANSFORMS = _buildable(geoml.transform, gpr.Parametric)


def _network():
    """A two-branch network sharing one input node."""
    geoml.set_seed(1234)
    ip = geoml.data.Grid2D(start=[1, 1], n=[6, 6], step=[44, 50])
    root = geoml.latent.BasicInput(
        ip, transform=geoml.transform.Isotropic(50))
    return geoml.latent.Add(
        geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian()),
        geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Cubic()))


def _model():
    point, _ = geoml.datasets.walker()
    return geoml.models.VGPNetwork(
        point, "V", geoml.likelihood.Gaussian(), _network(),
        options=geoml.models.GPOptions(verbose=False))


@pytest.mark.parametrize("obj", KERNELS + LIKELIHOODS + TRANSFORMS)
def test_every_object_reprs_on_one_line(obj):
    text = repr(obj)
    assert text.startswith(obj.__class__.__name__)
    assert "\n" not in text
    assert "object at 0x" not in text


@pytest.mark.parametrize("obj", KERNELS + LIKELIHOODS + TRANSFORMS)
def test_every_object_prints_its_state(obj):
    text = str(obj)
    assert obj.__class__.__name__ in text


def test_the_kernels_and_likelihoods_are_covered():
    """The sweep above is only a regression net if it caught these."""
    assert len(KERNELS) >= 5 and len(LIKELIHOODS) >= 3


def test_repr_carries_the_current_value_not_the_one_built_with():
    """Ranges move during training; a repr showing the initial one would lie."""
    transform = geoml.transform.Isotropic(50)
    assert "50" in repr(transform)
    transform.parameters["range"].set_value(123.0)
    assert "123" in repr(transform)
    assert "50" not in repr(transform)


def test_large_parameters_are_summarized_not_dumped():
    """A GP node holds an inducing value per point; printing them all is noise."""
    node = _network().parents[0]
    assert repr(node) == "BasicGP(size=1, kernel=Gaussian())"
    assert "shape (1, 36, 1)" in str(node)


def test_str_lays_out_the_whole_composition():
    text = str(_network())
    for name in ("Add", "BasicGP", "BasicInput", "Isotropic", "Cubic"):
        assert name in text
    assert "range: " in text          # the parameters come with it


def test_a_node_reports_its_size():
    network = _network()
    assert "size 1" in str(network)
    assert "size=1" in repr(network)


def test_the_parents_stay_out_of_a_node_repr():
    """They are the composition, which str lays out; nesting them repeats it."""
    node = _network().parents[0]
    assert "BasicInput" not in repr(node)
    assert "BasicInput" in str(node)


def test_a_model_keeps_its_summary():
    model = _model()
    assert repr(model).startswith("Variational Gaussian process model")
    assert str(model) == repr(model)
    assert "V (Gaussian)" in repr(model)


def test_a_vector_variable_names_its_components():
    """`repr` says which variable this is; `str` says what is on it, so the
    columns need not be looked up in the source."""
    train, _ = geoml.datasets.jura()
    text = str(train.variables["Elements"])

    assert text.startswith("VectorVariable('Elements', n_data=259)")
    assert "components: Cd, Co, Cr, Cu, Ni, Pb, Zn" in text
    assert "columns: uncertainty" in text
    # the one-line identity is unchanged by any of this
    assert repr(train.variables["Elements"]) == \
        "VectorVariable('Elements', n_data=259)"


def test_a_categorical_variable_names_its_categories():
    train, _ = geoml.datasets.jura()
    text = str(train.variables["Rock"])

    assert "categories: Argovian, Kimmeridgian" in text
    assert "measurements_a" in text and "predicted" in text


def test_a_variable_lists_what_can_be_read_off_it():
    point, _ = geoml.datasets.walker()
    variable = point.variables["V"]

    assert "columns: measurements, latent_mean, latent_variance, prediction" \
        in str(variable)

    variable.allocate_simulations(8)
    variable.reset_quantiles([0.1, 0.9])
    text = str(variable)
    assert "simulations: 8" in text
    assert "quantiles: 0.1, 0.9" in text


def test_a_container_still_lists_its_variables_in_one_line_each():
    """Containers name the class rather than printing each variable, so a
    variable growing a longer `str` does not grow theirs."""
    point, _ = geoml.datasets.walker()
    assert "    V: ContinuousVariable\n" in str(point)


def test_parameters_and_variables_identify_themselves():
    point, _ = geoml.datasets.walker()
    assert repr(point.variables["V"]) == "ContinuousVariable('V', n_data=470)"

    parameter = gpr.PositiveParameter(2.5, 0.1, 10)
    assert repr(parameter) == "PositiveParameter(2.5)"
    parameter.fix()
    assert "fixed=True" in repr(parameter)


def test_options_report_their_settings():
    text = repr(geoml.models.GPOptions(jitter=1e-6))
    assert text.startswith("GPOptions(")
    assert "seed=" in text and "jitter=1e-06" in text


def test_printing_does_not_read_the_stored_arrays(monkeypatch):
    """A container's repr must stay cheap on a variable backed by disk."""
    import geoml.storage as storage
    point, _ = geoml.datasets.walker()

    def explode(self, *args, **kwargs):
        raise AssertionError("the backing array was read")

    monkeypatch.setattr(storage.ArrayStore, "__array__", explode)
    assert "PointData" in repr(point)
    assert "ContinuousVariable" in repr(point.variables["V"])
