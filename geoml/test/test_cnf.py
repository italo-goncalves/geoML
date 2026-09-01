"""The continuous flows after the 0.6.9 overhaul.

The field is a fixed function settled once by ``refresh`` -- Sobol anchors
in the box crossed with time knots for `ContinuousNormalizingFlow`, a
CP-structured grid for `TensorProductFlow` -- and both directions integrate
it with the same adaptive solver, so the guarantees here are the ones the
old mismatched midpoint schemes could not give: a round trip at solver
tolerance, a log-determinant that matches the numerical Jacobian, an
identity far from the box, gradients through the solve to every parameter,
and a model that saves and reloads with nothing data-dependent left behind.
Every test runs over both representations.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import geoml
import geoml.warping as wp

KINDS = ["dense", "cp"]


def _build(kind, size, rtol=1e-6):
    if kind == "dense":
        return wp.ContinuousNormalizingFlow(size, inducing_points=12,
                                            n_steps=6, rtol=rtol)
    return wp.TensorProductFlow(size, grid=7, rank=3, n_steps=6, rtol=rtol)


def _flow(kind, size=3, rtol=1e-6, scale=0.5, seed=101):
    """A flow with a field worth testing: the fresh init is deliberately
    near identity, so the amplitude is set to something sizable."""
    geoml.set_seed(seed)
    flow = _build(kind, size, rtol)
    rng = np.random.default_rng(seed)
    if kind == "dense":
        flow.parameters["alpha_white"].set_value(
            rng.normal(scale=scale, size=[12, size, 6]))
    else:
        flow.parameters["weights"].set_value(
            rng.normal(scale=scale, size=[3]))
    flow.refresh()
    return flow


@pytest.mark.parametrize("kind", KINDS)
def test_round_trip_is_solver_tolerance(kind):
    flow = _flow(kind)
    x = np.random.default_rng(0).normal(size=[50, 3])

    warped, _ = flow.forward(tf.constant(x))
    back = flow.backward(warped)

    assert float(np.max(np.abs(np.asarray(back) - x))) < 1e-4


@pytest.mark.parametrize("kind", KINDS)
def test_log_determinant_matches_the_jacobian(kind):
    flow = _flow(kind, rtol=1e-10)
    rows = np.random.default_rng(1).normal(size=[6, 3])
    reported = np.asarray(flow.forward(tf.constant(rows))[1])

    step = 1e-5
    numeric = []
    for row in rows:
        jacobian = np.zeros([3, 3])
        for column in range(3):
            bump = np.zeros([1, 3])
            bump[0, column] = step
            plus = np.asarray(flow.forward(tf.constant(row[None] + bump))[0])
            minus = np.asarray(flow.forward(tf.constant(row[None] - bump))[0])
            jacobian[:, column] = (plus - minus)[0] / (2 * step)
        numeric.append(np.log(abs(np.linalg.det(jacobian))))

    assert np.allclose(reported, numeric, atol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_the_map_is_identity_far_from_the_box(kind):
    flow = _flow(kind)
    far = np.full([4, 3], 20.0)

    warped, log_det = flow.forward(tf.constant(far))

    assert float(np.max(np.abs(np.asarray(warped) - far))) < 1e-8
    assert float(np.max(np.abs(np.asarray(log_det)))) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_forward_before_refresh_raises(kind):
    geoml.set_seed(7)
    flow = _build(kind, 3)
    with pytest.raises(RuntimeError, match="refresh"):
        flow.forward(np.zeros([2, 3]))


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_reach_every_parameter_through_the_solve(kind):
    """Every parameter, not just one: the solver's adjoint computes
    gradients for the state, for Variables read inside the field and for
    its `constants` -- a tensor merely captured by the closure is cut,
    which is how `alpha_white` and `amp` were silently detached from
    training in the first draft of the overhaul."""
    flow = _flow(kind)
    x = tf.constant(np.random.default_rng(2).normal(size=[20, 3]))

    with tf.GradientTape() as tape:
        flow.refresh()
        warped, log_det = flow.forward(x)
        loss = tf.reduce_sum(warped ** 2) + tf.reduce_sum(log_det)
    gradients = tape.gradient(loss, flow.get_unfixed_variables())

    assert all(g is not None and float(tf.reduce_max(tf.abs(g))) > 0
               for g in gradients)


@pytest.mark.parametrize("kind", KINDS)
def test_the_geometry_is_construction_not_data(kind):
    geoml.set_seed(1)
    first = _build(kind, 3)
    geoml.set_seed(2)
    second = _build(kind, 3)

    # different package seeds draw different weights but identical
    # geometry: the anchors or grid are constants, so nothing
    # data-dependent needs persisting
    geometry = "anchors" if kind == "dense" else "nodes"
    weights = "alpha_white" if kind == "dense" else "factors_space"
    assert np.array_equal(np.asarray(getattr(first, geometry)),
                          np.asarray(getattr(second, geometry)))
    assert not np.array_equal(
        np.asarray(first.parameters[weights].get_value()),
        np.asarray(second.parameters[weights].get_value()))


@pytest.mark.parametrize("kind", KINDS)
def test_a_model_with_a_flow_saves_and_reloads(kind, tmp_path):
    geoml.set_seed(42)
    rng = np.random.default_rng(42)
    coordinates = rng.uniform(0, 100, size=[30, 2])
    values = rng.normal(size=[30, 2])

    data = geoml.data.PointData(
        pd.DataFrame(coordinates, columns=["X", "Y"]), ["X", "Y"])
    data.add_vector_variable("v", ["a", "b"], values)

    if kind == "dense":
        flow = wp.ContinuousNormalizingFlow(2, inducing_points=8, n_steps=4)
    else:
        flow = wp.TensorProductFlow(2, grid=5, rank=2, n_steps=4)
    chain = wp.ChainedWarping(wp.ZScore(2), flow)
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 8, seed=0),
        transform=geoml.transform.Isotropic(30.0))
    gp = geoml.latent.BasicGP(root, size=2)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.MultivariateGaussian(2, chain), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(max_iter=3)

    target = geoml.data.PointData(
        pd.DataFrame(rng.uniform(0, 100, size=[10, 2]), columns=["X", "Y"]),
        ["X", "Y"])
    model.predict(target, n_sim=5)
    before = np.asarray(target.values("v/a/prediction"))

    path = str(tmp_path / "flow_model")
    geoml.persistence.save_model(model, path)
    restored = geoml.persistence.load_model(path)

    fresh = geoml.data.PointData(
        pd.DataFrame(np.asarray(target.coordinates), columns=["X", "Y"]),
        ["X", "Y"])
    restored.predict(fresh, n_sim=5)
    after = np.asarray(fresh.values("v/a/prediction"))

    assert np.allclose(before, after)
