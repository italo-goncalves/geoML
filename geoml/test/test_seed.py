"""The seed that makes a build reproducible.

Parameters are drawn when an object is constructed, from the package generator
that ``geoml.set_seed`` seeds — and a model's options draw their own ``seed``
from the same generator when built, so the one call governs the initial
parameters, the training draws and the simulation stream alike. These tests pin
that contract: the same seed gives the same starting parameters and the same
training seed, and building a model never disturbs the generator the caller is
using for their own work.
"""
import numpy as np
import pytest

import geoml
import geoml.parameter as gpr


def _build():
    """A small network on Walker Lake. Not trained: only the initial draws."""
    point, _ = geoml.datasets.walker()
    inducing = geoml.data.Grid2D(start=[1, 1], n=[6, 6], step=[44, 50])
    root = geoml.latent.BasicInput(
        inducing, transform=geoml.transform.Isotropic(50))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    return geoml.models.VGPNetwork(
        point, "V", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False))


def _values(model):
    return model.get_parameter_values(complete=True)[0]


def test_the_same_seed_gives_the_same_starting_parameters():
    geoml.set_seed(1234)
    first = _values(_build())
    geoml.set_seed(1234)
    second = _values(_build())
    assert np.array_equal(first, second)


def test_a_different_seed_starts_somewhere_else():
    geoml.set_seed(1234)
    first = _values(_build())
    geoml.set_seed(4321)
    other = _values(_build())
    assert not np.array_equal(first, other)


def test_the_seed_covers_only_what_is_built_after_it():
    """One call does not make every build alike — the generator moves on."""
    geoml.set_seed(1234)
    first = _values(_build())
    second = _values(_build())
    assert not np.array_equal(first, second)


def test_the_orthonormal_matrices_follow_the_package_seed():
    """These drew from TensorFlow's global generator, which no seed reached."""
    geoml.set_seed(7)
    first = gpr.OrthonormalMatrix(4, 2).get_value().numpy()
    geoml.set_seed(7)
    second = gpr.OrthonormalMatrix(4, 2).get_value().numpy()
    assert np.allclose(first, second)


def test_random_projections_keeps_its_own_seed():
    first = geoml.transform.RandomProjections(3, 5, seed=99)
    second = geoml.transform.RandomProjections(3, 5, seed=99)
    other = geoml.transform.RandomProjections(3, 5, seed=100)
    assert np.array_equal(first.projections.numpy(),
                          second.projections.numpy())
    assert not np.array_equal(first.projections.numpy(),
                              other.projections.numpy())


def test_building_leaves_the_callers_generator_alone():
    """Initialization drew from NumPy's global generator, and
    ``RandomProjections`` reset it outright, so building changed whatever the
    caller drew next."""
    np.random.seed(0)
    expected = np.random.normal(size=5)

    np.random.seed(0)
    geoml.set_seed(1234)
    _build()
    geoml.transform.RandomProjections(3, 5, seed=99)

    assert np.array_equal(np.random.normal(size=5), expected)


def test_the_training_seed_is_drawn_from_the_package_generator():
    """``options.seed`` is no longer an argument: it is drawn when the options
    are built, so ``set_seed`` is the one knob and there is no second one to
    forget. A saved model keeps the number it drew — persistence restores the
    options ``vars`` wholesale, never calling the constructor."""
    with pytest.raises(TypeError):
        geoml.models.GPOptions(seed=1)

    geoml.set_seed(101)
    first = geoml.models.GPOptions(verbose=False).seed
    geoml.set_seed(101)
    second = geoml.models.GPOptions(verbose=False).seed
    geoml.set_seed(202)
    other = geoml.models.GPOptions(verbose=False).seed

    assert first == second
    assert first != other


def test_training_leaves_the_callers_generator_alone():
    """Training seeded the global generator to shuffle its batches."""
    geoml.set_seed(1234)
    model = _build()

    np.random.seed(0)
    expected = np.random.normal(size=5)

    np.random.seed(0)
    model.train_full(max_iter=3)

    assert np.array_equal(np.random.normal(size=5), expected)
