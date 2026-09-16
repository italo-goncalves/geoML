"""The seed that makes a build reproducible, and what else it takes.

Parameters are drawn when an object is constructed, from the package generator
that ``geoml.set_seed`` seeds — and a model's options draw their own ``seed``
from the same generator when built, so the one call governs the initial
parameters, the training draws and the simulation stream alike. These tests pin
that contract: the same seed gives the same starting parameters and the same
training seed, and building a model never disturbs the generator the caller is
using for their own work.
"""
import os
import subprocess
import sys
import textwrap

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


# --------------------------------------------------------------------------- #
# nothing rides on what a process happens to do
# --------------------------------------------------------------------------- #
def test_the_robust_pca_start_follows_the_package_seed():
    """FastMCD drew its starting subsets from NumPy's global generator, which
    no seed reaches, so chapter 16's model began from one of two robust
    covariances depending on the process, and everything below moved with
    it."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(60, 3)),
                        rng.normal(loc=8.0, size=(8, 3))])

    def started():
        geoml.set_seed(11)
        warping = geoml.warping.RobustPCA(3, 3)
        warping.initialize(x.copy())
        return (np.asarray(warping.eigvals), np.asarray(warping.eigvecs))

    first, second = started(), started()
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_a_tree_is_walked_in_the_order_it_was_built():
    """`VGPNetwork._nodes` reads this order and sums the KL in it. A set of
    nodes iterates by memory address: a different order in every process."""
    geoml.set_seed(5)
    root = geoml.latent.BasicInput(
        geoml.data.Grid2D(start=[1, 1], n=[4, 4], step=[66, 75]),
        transform=geoml.transform.Isotropic(50))
    gps = [geoml.latent.BasicGP(root, size=1) for _ in range(3)]

    assert [node.name for node in geoml.latent.Add(*gps).get_unique_parents()] \
        == [gp.name for gp in gps] + [root.name]


def test_the_trainable_variables_keep_their_registration_order():
    geoml.set_seed(5)
    model = _build()
    assert [id(v) for v in model.get_unfixed_variables()] \
        == [id(p.variable) for p in model.all_parameters if not p.fixed]


_ANOTHER_PROCESS = '''
import json
import numpy as np
import geoml

geoml.set_seed(1234)
rng = np.random.default_rng(0)
coords = rng.uniform(0.0, 100.0, (60, 2))
point = geoml.data.PointData.from_array(coords)
point.add_vector_variable("m", ["a", "b", "c"], np.stack([
    np.sin(coords[:, 0] / 20.0), np.cos(coords[:, 1] / 25.0),
    coords[:, 0] / 100.0], axis=1) + rng.normal(0.0, 0.1, (60, 3)))
root = geoml.latent.BasicInput(
    geoml.data.Grid2D(start=[0, 0], n=[5, 5], step=[25, 25]),
    transform=geoml.transform.Isotropic(40))
network = geoml.latent.Add(*[geoml.latent.BasicGP(root, size=3)
                             for _ in range(2)])
warping = geoml.warping.ChainedWarping(geoml.warping.RobustPCA(3, 3),
                                       geoml.warping.ZScore(3))
model = geoml.models.VGPNetwork(
    point, "m", geoml.likelihood.MultivariateGaussian(3, warping), network,
    options=geoml.models.GPOptions(verbose=False, training_samples=4))
model.train_full(max_iter=5)
grid = geoml.data.Grid2D(start=[10, 10], n=[4, 4], step=[20, 20])
model.predict(grid, n_sim=3)
print(json.dumps({
    "log": [float(v).hex() for v in model.training_log],
    "prediction": [float(v).hex() for v in
                   np.asarray(grid.get("m/a/prediction").values)[:4]],
    "simulations": [float(v).hex() for v in
                    np.asarray(grid.get("m/a").simulations)[0]]}))
'''


def test_a_model_comes_out_the_same_in_another_process(tmp_path):
    """The contract, as `docs/source/reference/reproducibility.md` states
    it: the same code, versions and device, `set_seed` before anything is
    built, and every number comes back the same in any process. Two orders
    that changed with the process, and a robust fit reading NumPy's global
    generator, used to break it."""
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(_ANOTHER_PROCESS), encoding="utf-8")
    root = os.path.dirname(os.path.dirname(os.path.abspath(geoml.__file__)))

    answers = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, CUDA_VISIBLE_DEVICES="",
                   TF_CPP_MIN_LOG_LEVEL="3",
                   PYTHONPATH=os.pathsep.join(
                       [root] + [p for p in [os.environ.get("PYTHONPATH")]
                                 if p]))
        answers.append(subprocess.run(
            [sys.executable, str(script)], env=env, check=True,
            capture_output=True, text=True).stdout.strip().splitlines()[-1])

    assert answers[0] == answers[1]
