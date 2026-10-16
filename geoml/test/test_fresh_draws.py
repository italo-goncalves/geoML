"""`GPOptions(fresh_training_draws=True)`: a Monte Carlo bound averaged over
new realizations at every step instead of one fixed set.

Pinned here: the first step draws exactly what the fixed set does and the
later ones do not; training split into chunks replays one call bit for bit;
a model that trains by quadrature reads no draws and trains to the same
bits either way; and the option survives a save, a model saved before it
existed opening with it off.
"""
import numpy as np

import geoml
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.persistence as persistence
import geoml.transform as tr


def _mixture_model(fresh, seed=1234):
    """A `GaussianMixture` leaf, which trains on its realizations."""
    geoml.set_seed(seed)
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 10.0, 60))
    y = np.sin(2 * np.pi * x / 4) * np.where(x >= 5, -1, 1) \
        + rng.normal(0.0, 0.1, 60)
    data = geoml.data.PointData.from_array(x[:, None], ["X"])
    data.add_continuous_variable("v", y)

    def root(r):
        return latent.BasicInput(geoml.data.Grid1D(start=-0.5, n=25, step=0.5),
                                 transform=tr.Isotropic(r))

    leaf = latent.GaussianMixture(
        latent.BasicGP(root(6.0), size=2),
        [latent.BasicGP(root(4.0), size=1) for _ in range(2)])
    return geoml.models.VGPNetwork(
        data, "v", lk.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False, training_samples=8,
                                       fresh_training_draws=fresh))


def test_the_first_step_draws_the_fixed_set_and_the_later_ones_do_not():
    fixed, fresh = _mixture_model(False), _mixture_model(True)
    fixed.train_full(5)
    fresh.train_full(5)
    assert fixed.training_log[0] == fresh.training_log[0]
    assert not np.allclose(fixed.training_log[1:], fresh.training_log[1:])


def test_training_in_chunks_replays_one_call():
    whole = _mixture_model(True)
    whole.train_full(30)
    chunked = _mixture_model(True)
    for _ in range(3):
        chunked.train_full(10)
    np.testing.assert_array_equal(chunked.training_log, whole.training_log)


def test_a_new_optimizer_starts_the_draws_again():
    """`set_learning_rate` starts a phase, and the phase its own count: the
    first step after it draws what the first step of training did."""
    model = _mixture_model(True)
    model.train_full(4)
    assert int(model._draw_count.numpy()) == 4
    model.set_learning_rate(1e-2)
    model.train_full(1)
    assert int(model._draw_count.numpy()) == 1


def test_a_model_trained_by_quadrature_is_untouched():
    """A Gaussian leaf under a Gaussian likelihood integrates by quadrature
    and reads no realization: the option changes no bit of its training."""
    def walker(fresh):
        geoml.set_seed(1234)
        walker_point, _ = geoml.datasets.walker()
        root = latent.BasicInput(
            geoml.data.Grid2D(start=[1, 1], n=[6, 6], step=[45, 50]),
            transform=tr.Isotropic(50))
        return geoml.models.VGPNetwork(
            walker_point, "V", lk.Gaussian(), latent.BasicGP(root, size=1),
            options=geoml.models.GPOptions(verbose=False, training_samples=8,
                                           fresh_training_draws=fresh))
    off, on = walker(False), walker(True)
    off.train_full(10)
    on.train_full(10)
    np.testing.assert_array_equal(off.training_log, on.training_log)


def test_the_option_is_saved_and_an_older_save_opens_with_it_off(tmp_path):
    model = _mixture_model(True)
    model.train_full(2)
    path = str(tmp_path / "fresh")
    model.save(path)
    assert persistence.load_model(path).options.fresh_training_draws

    # an options object as one saved before the option existed rebuilds it:
    # `persistence` updates `vars()` on a bare instance, so the class
    # default answers
    old = geoml.models.GPOptions.__new__(geoml.models.GPOptions)
    vars(old).update({k: v for k, v in vars(model.options).items()
                      if k != "fresh_training_draws"})
    assert old.fresh_training_draws is False
