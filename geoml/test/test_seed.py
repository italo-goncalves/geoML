"""The seed that makes a build reproducible.

Parameters are drawn when an object is constructed — before any model exists to
carry ``options.seed`` — so the draws come from the package generator that
``geoml.set_seed`` seeds. These tests pin both halves of that contract: the same
seed gives the same starting parameters, and building a model never disturbs the
generator the caller is using for their own work.
"""
import numpy as np

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


def test_training_leaves_the_callers_generator_alone():
    """Training seeded the global generator to shuffle its batches."""
    geoml.set_seed(1234)
    model = _build()

    np.random.seed(0)
    expected = np.random.normal(size=5)

    np.random.seed(0)
    model.train_full(max_iter=3)

    assert np.array_equal(np.random.normal(size=5), expected)
