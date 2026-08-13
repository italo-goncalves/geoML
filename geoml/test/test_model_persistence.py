"""Saving and loading trained models.

A saved model must come back complete: same structure, same trained parameters,
same variables and variable types, so that it predicts exactly as it did before
and can be trained further. The structure is rebuilt by replaying the
constructor calls recorded by ``Parametric``, so the tests below check the
things that replay can get wrong -- shared objects, post-construction state
(fixed flags, limits) and the training data itself.
"""
import os

import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.persistence as persistence


COLS = ["X", "Y"]


def _points(n=40, seed=1234):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    point.add_continuous_variable("v", np.sin(coords[:, 0] / 25.0))
    return point, coords


def _inducing(n=6, seed=99):
    rng = np.random.default_rng(seed)
    return geoml.data.PointData(
        pd.DataFrame(rng.uniform(0, 100, (n, 2)), columns=COLS), COLS)


def _model(point=None, max_iter=5, seed=1234):
    """A small trained continuous model."""
    import tensorflow as tf
    geoml.set_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if point is None:
        point, _ = _points()
    root = geoml.latent.BasicInput(
        _inducing(), transform=geoml.transform.Isotropic(40))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(
        verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), gp, options=options)
    model.train_full(max_iter=max_iter)
    return model


def _grid():
    return geoml.data.Grid2D(start=[0, 0], n=[8, 8], step=[10, 10])


def _predict(model, n_sim=3):
    grid = _grid()
    model.predict(grid, n_sim=n_sim)
    return np.asarray(grid.variables["v"].latent_mean.values), grid


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #
def test_loaded_model_predicts_identically(tmp_path):
    model = _model()
    before, _ = _predict(model)

    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    assert type(loaded) is geoml.models.VGPNetwork
    after, _ = _predict(loaded)
    assert np.allclose(before, after)


def test_parameters_and_structure_survive(tmp_path):
    model = _model()
    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    assert len(loaded.all_parameters) == len(model.all_parameters)
    for saved, restored in zip(model.all_parameters, loaded.all_parameters):
        assert np.allclose(saved.variable.numpy(), restored.variable.numpy())
        assert np.allclose(saved.min_transformed.numpy(),
                           restored.min_transformed.numpy())
        assert np.allclose(saved.max_transformed.numpy(),
                           restored.max_transformed.numpy())

    assert loaded.training_log == model.training_log
    assert loaded.options.seed == model.options.seed
    assert loaded.options.training_samples == model.options.training_samples


def test_training_data_comes_back(tmp_path):
    model = _model()
    coords = np.asarray(model.data.coordinates)
    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    assert loaded.data.n_data == model.data.n_data
    assert np.allclose(np.asarray(loaded.data.coordinates), coords)
    assert np.allclose(loaded.y, model.y)


# --------------------------------------------------------------------------- #
# variable types
# --------------------------------------------------------------------------- #
def test_loaded_model_remembers_variable_types(tmp_path):
    """A reloaded model must create the right variable on a new container."""
    import tensorflow as tf
    geoml.set_seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 100, (40, 2))
    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    point.add_binary_variable(
        "b", labels=("waste", "ore"),
        measurements=np.where(coords[:, 0] > 50, "ore", "waste"))

    root = geoml.latent.BasicInput(
        _inducing(), transform=geoml.transform.Isotropic(40))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(
        verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "b", geoml.likelihood.Bernoulli(), gp, options=options)
    model.train_full(max_iter=3)

    grid = _grid()
    model.predict(grid, n_sim=2)
    before = np.asarray(grid.variables["b"].probability.values).copy()

    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    assert loaded.variables == ["b"]
    assert type(loaded.likelihoods[0]) is geoml.likelihood.Bernoulli
    assert type(loaded.data.variables["b"]) is geoml.data.BinaryVariable
    assert list(loaded.data.variables["b"].labels) == ["waste", "ore"]

    # it builds that variable on a container that has never seen it
    new_grid = _grid()
    loaded.predict(new_grid, n_sim=2)
    created = new_grid.variables["b"]
    assert type(created) is geoml.data.BinaryVariable
    assert list(created.labels) == ["waste", "ore"]
    assert np.allclose(before, np.asarray(created.probability.values))


def test_categorical_variable_definition_survives(tmp_path):
    """Labels and components of a rock type must come back with the model."""
    import tensorflow as tf
    geoml.set_seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 100, (40, 2))
    labels = ("granite", "basalt", "shale")
    rock = np.array([labels[i % 3] for i in range(40)])

    point = geoml.data.PointData(pd.DataFrame(coords, columns=COLS), COLS)
    point.add_rock_type_variable("rt", labels=labels, measurements_a=rock)

    root = geoml.latent.BasicInput(
        _inducing(), transform=geoml.transform.Isotropic(40))
    gp = geoml.latent.BasicGP(root, size=len(labels),
                              kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(
        verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "rt",
        geoml.likelihood.CategoricalGaussianIndicator(len(labels)),
        gp, options=options)
    model.train_full(max_iter=3)

    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    assert type(loaded.likelihoods[0]) is \
        geoml.likelihood.CategoricalGaussianIndicator
    restored = loaded.data.variables["rt"]
    assert type(restored) is geoml.data.RockTypeVariable
    assert list(restored.labels) == list(labels)
    assert list(restored.components.keys()) == list(labels)

    grid = _grid()
    loaded.predict(grid, n_sim=2)
    created = grid.variables["rt"]
    assert type(created) is geoml.data.RockTypeVariable
    assert list(created.labels) == list(labels)
    probabilities = np.stack(
        [np.asarray(created.components[lb].probability.values)
         for lb in labels], axis=1)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# things replaying constructors can get wrong
# --------------------------------------------------------------------------- #
def test_shared_nodes_are_not_duplicated(tmp_path):
    """A node feeding two children must stay one object after loading.

    Parameters are matched by position in a flat list, so duplicating a shared
    object would shift every position after it.
    """
    import tensorflow as tf
    geoml.set_seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    point, _ = _points()
    transform = geoml.transform.Isotropic(40)
    root = geoml.latent.BasicInput(_inducing(), transform=transform)
    gp_a = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    gp_b = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Cubic())
    network = geoml.latent.Add(gp_a, gp_b)
    options = geoml.models.GPOptions(
        verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), network, options=options)
    model.train_full(max_iter=3)

    before, _ = _predict(model)
    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    left, right = loaded.latent_network.parents
    assert left.parent is right.parent          # the shared root
    assert len(loaded.all_parameters) == len(model.all_parameters)
    assert np.allclose(before, _predict(loaded)[0])


def test_fixed_flags_and_limits_survive(tmp_path):
    """Both can be changed after construction, so rebuilding cannot infer them."""
    model = _model()
    parameter = model.latent_network.parameters["ranges"]
    parameter.set_limits(min_val=np.full(parameter.shape, 0.5),
                         max_val=np.full(parameter.shape, 123.0))
    parameter.set_value(np.full(parameter.shape, 50.0))
    parameter.fix()

    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    restored = loaded.latent_network.parameters["ranges"]
    assert restored.fixed
    assert np.allclose(restored.get_value().numpy(),
                       parameter.get_value().numpy())
    assert np.allclose(restored.max_transformed.numpy(),
                       parameter.max_transformed.numpy())
    assert len(loaded.get_unfixed_variables()) == \
        len(model.get_unfixed_variables())


def test_fix_transform_flag_survives(tmp_path):
    """`BasicInput(fix_transform=True)` fixes parameters inside the transform."""
    import tensorflow as tf
    geoml.set_seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    point, _ = _points()
    root = geoml.latent.BasicInput(
        _inducing(), transform=geoml.transform.Isotropic(40),
        fix_transform=True)
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(verbose=False, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), gp, options=options)

    path = str(tmp_path / "model.zarr")
    model.save(path)
    loaded = geoml.models.VGPNetwork.open(path)

    assert all(p.fixed for p in loaded.latent_network.parent.transform
               .all_parameters)
    assert len(loaded.get_unfixed_variables()) == \
        len(model.get_unfixed_variables())


# --------------------------------------------------------------------------- #
# using the loaded model
# --------------------------------------------------------------------------- #
def test_loaded_model_can_be_trained_further(tmp_path):
    model = _model(max_iter=5)
    path = str(tmp_path / "model.zarr")
    model.save(path)

    loaded = geoml.models.VGPNetwork.open(path)
    before = loaded.latent_network.parameters["ranges"].get_value().numpy()
    loaded.train_full(max_iter=5)

    # training continues the restored log instead of starting a new one
    assert len(loaded.training_log) == 10
    assert loaded.training_log[:5] == model.training_log
    assert np.all(np.isfinite(loaded.training_log))
    after = loaded.latent_network.parameters["ranges"].get_value().numpy()
    assert not np.allclose(before, after)      # training actually moved it


def test_load_around_new_data(tmp_path):
    model = _model()
    path = str(tmp_path / "model.zarr")
    model.save(path)

    rng = np.random.default_rng(77)
    coords = rng.uniform(0, 100, (25, 2))
    new_point = geoml.data.PointData(
        pd.DataFrame(coords, columns=COLS), COLS)
    new_point.add_continuous_variable("v", np.cos(coords[:, 1] / 20.0))

    loaded = geoml.models.VGPNetwork.open(path, data=new_point)
    assert loaded.data is new_point
    assert loaded.data.n_data == 25

    # the trained parameters are the starting point, and training works
    saved_range = model.latent_network.parameters["ranges"].get_value()
    loaded_range = loaded.latent_network.parameters["ranges"].get_value()
    assert np.allclose(saved_range.numpy(), loaded_range.numpy())

    loaded.train_full(max_iter=3)
    assert np.all(np.isfinite(loaded.training_log))


def test_saving_twice_overwrites(tmp_path):
    model = _model()
    path = str(tmp_path / "model.zarr")
    model.save(path)
    model.save(path)

    loaded = geoml.models.VGPNetwork.open(path)
    assert len(loaded.all_parameters) == len(model.all_parameters)


# --------------------------------------------------------------------------- #
# failure modes
# --------------------------------------------------------------------------- #
def test_only_geoml_classes_are_rebuilt():
    with pytest.raises(persistence.ModelFormatError):
        persistence._resolve("os.system")


def test_opening_something_else(tmp_path):
    point, _ = _points()
    path = str(tmp_path / "container.zarr")
    point.to_zarr(path)

    with pytest.raises(persistence.ModelFormatError):
        geoml.models.VGPNetwork.open(path)


def test_new_data_must_have_the_variables(tmp_path):
    model = _model()
    path = str(tmp_path / "model.zarr")
    model.save(path)

    rng = np.random.default_rng(3)
    empty = geoml.data.PointData(
        pd.DataFrame(rng.uniform(0, 100, (10, 2)), columns=COLS), COLS)

    with pytest.raises(KeyError):
        geoml.models.VGPNetwork.open(path, data=empty)
