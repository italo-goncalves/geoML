"""Faults as transforms: a surface fitted from its observations that
repels points across it or restores its hanging wall, and several of them
in age order.

Planes have exact answers -- the field is the signed distance through the
drift alone -- so a restoration is checked to the metre and the repulsion
against `BellFault2D` to solver precision. The one trained case is the
decision the plan recorded: the displacement variant ships if a model with
it beats one without on a displaced layer.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import geoml
from geoml.transform import (ImplicitFault, FaultDisplacement, FaultNetwork,
                             BellFault2D, Isotropic, ChainedTransform)


def _plane(axis, position, n=400, seed=0):
    """Points on the plane `x[axis] = position` inside [0, 100]^3, with
    their normals along +axis."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 100.0, size=[n, 3])
    points[:, axis] = position
    normals = np.zeros([n, 3])
    normals[:, axis] = 1.0
    return points, normals


def _away(points, axis, position, margin=3.0):
    return np.abs(points[:, axis] - position) > margin


def _point_data(coordinates, values):
    frame = pd.DataFrame(coordinates, columns=["X", "Y", "Z"])
    data = geoml.data.PointData(frame, ["X", "Y", "Z"])
    data.add_continuous_variable("v", values)
    return data


def test_the_repulsion_matches_bell_fault_across_the_midpoint():
    t = np.linspace(20.0, 80.0, 31)
    points = np.stack([t, np.full_like(t, 50.0)], axis=1)
    # BellFault2D is positive below its segment; so are normals pointing down
    normals = np.tile([0.0, -1.0], [31, 1])
    fault = ImplicitFault(points, normals, mode="decay", reach=30.0)
    bell = BellFault2D(np.array([20.0, 50.0]), np.array([80.0, 50.0]))
    bell.parameters["range"].set_value(10.0)
    amp = float(bell.parameters["amp"].get_value())
    # BellFault2D's range acts on the across distance over the half-length
    rng = float(bell.parameters["range"].get_value()) * 30.0
    fault.parameters["amp"].set_value(amp)
    fault.parameters["range"].set_value(rng)
    assert float(fault.parameters["range"].get_value()) == pytest.approx(rng)

    across = np.linspace(-40.0, 40.0, 17)
    query = np.stack([np.full_like(across, 50.0), 50.0 + across], axis=1)
    mine = np.asarray(fault(tf.constant(query)))[:, 0]
    theirs = np.asarray(bell(tf.constant(query)))[:, 0]
    assert np.allclose(mine, theirs, atol=1e-6)
    # and the feature dies out past the observations
    far = np.array([[200.0, 55.0]])
    assert abs(float(np.asarray(fault(tf.constant(far)))[0, 0])) < 1e-9
    step = np.asarray(ImplicitFault(points, normals, mode="step",
                                    reach=30.0)(tf.constant(query)))[:, 0]
    # the normals point down, so the field is positive below the segment
    assert step[0] > 0 > step[-1]


def test_a_displaced_layer_is_restored_with_the_true_slip():
    points, normals = _plane(axis=0, position=50.0)
    fault = FaultDisplacement(points, normals, throw=20.0, reach=1000.0,
                              width=0.5)
    rng = np.random.default_rng(1)
    query = rng.uniform(0.0, 100.0, size=[300, 3])
    query = query[_away(query, 0, 50.0)]
    restored = np.asarray(fault(tf.constant(query)))
    expected = query[:, 2] - 20.0 * (query[:, 0] > 50.0)
    assert np.allclose(restored[:, 2], expected, atol=0.1)
    assert np.allclose(restored[:, :2], query[:, :2], atol=1e-9)


def test_training_finds_the_slip_and_beats_the_plain_transform():
    """The gate for the displacement variant: on a layer offset by 20 m
    across a fault, the model that can restore it must beat the one that
    cannot on held-out data, and read the slip off the data."""
    geoml.set_seed(4)
    rng = np.random.default_rng(2)
    samples = rng.uniform(0.0, 100.0, size=[700, 3])
    samples = samples[_away(samples, 0, 50.0, margin=2.0)]
    value = samples[:, 2] - 20.0 * (samples[:, 0] > 50.0) \
        + rng.normal(scale=0.5, size=len(samples))
    rows = np.arange(len(samples))
    train = _point_data(samples[rows < 400], value[rows < 400])
    test = _point_data(samples[rows >= 400], value[rows >= 400])
    axes = np.linspace(5.0, 95.0, 6)
    lattice = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"),
                       axis=-1).reshape(-1, 3)
    inducing = geoml.data.PointData.from_array(lattice)
    points, normals = _plane(axis=0, position=50.0, n=200)

    def _score(transform):
        root = geoml.latent.BasicInput(inducing, transform=transform)
        gp = geoml.latent.BasicGP(root, size=1)
        model = geoml.models.VGPNetwork(
            train, "v", geoml.likelihood.Gaussian(), gp,
            options=geoml.models.GPOptions(verbose=False))
        # the slip moves slowly at the default rate (measured: a tenth of
        # the way in 300 iterations); the phased pattern reads it off
        model.set_learning_rate(0.1)
        model.train_full(max_iter=500)
        model.predict(test, n_sim=1)
        predicted = np.asarray(test.values("v/prediction")).ravel()
        return float(np.sqrt(np.mean((predicted - value[rows >= 400]) ** 2)))

    fault = FaultDisplacement(points, normals, reach=1000.0, width=1.0)
    with_fault = _score(ChainedTransform(fault, Isotropic(30.0)))
    plain = _score(Isotropic(30.0))
    throw = float(fault.parameters["throw"].get_value())
    strike = float(fault.parameters["strike_slip"].get_value())
    print("rmse with fault %.3f, plain %.3f, throw %.2f strike %.2f"
          % (with_fault, plain, throw, strike))
    assert with_fault < 0.4 * plain
    assert abs(throw - 20.0) < 3.0


def test_points_on_one_normal_line_slide_together_on_a_curved_fault():
    """The slip direction is read at the foot on the surface: two hanging-
    wall points on the same normal line of a curved fault are moved by
    the same vector, though the field's gradient differs between them,
    and the restoration does not fold. The bends are wider than the
    throw (radius of curvature 5.1 against 2.5): beyond a bend's centre
    of curvature the normal lines of any curve cross, and a hanging wall
    sliding along the surface must fold there, whatever the frame."""
    from geoml.math.geometry import point_normals
    y = np.linspace(-6.0, 6.0, 41)
    amplitude = 0.5
    trace = np.stack([amplitude * np.sin(np.pi * y / 5.0), y], axis=1)
    normals = point_normals(trace, k=6, orient=[1.0, 0.0])
    # the straight step reads the frame at the foot: two points on one
    # normal line move by the same vector
    straight = FaultDisplacement(trace, normals, throw=2.5, reach=30.0,
                                 width=0.05, flow_steps=0)
    # the curve's own normal at that point, not the PCA estimate, so the
    # two points genuinely share a foot
    foot = trace[25]
    slope = amplitude * (np.pi / 5.0) * np.cos(np.pi * y[25] / 5.0)
    normal = np.array([1.0, -slope]) / np.hypot(1.0, slope)
    near, far = foot + 0.3 * normal, foot + 1.2 * normal
    pair = tf.constant(np.stack([near, far]))
    moved = np.asarray(straight(pair)) - np.asarray(pair)
    assert np.linalg.norm(moved[0]) == pytest.approx(2.5, abs=0.05)
    assert np.allclose(moved[0], moved[1], atol=0.03)
    # and the field's own gradient does differ between the two points
    grads = np.asarray(straight.surface.gradient(pair))
    grads = grads / np.linalg.norm(grads, axis=1, keepdims=True)
    assert np.linalg.norm(grads[0] - grads[1]) > 0.005

    # the flow follows each point's own level set, so the two chords
    # differ a little, and the restoration's Jacobian keeps its sign on
    # the hanging wall: no fold
    fault = FaultDisplacement(trace, normals, throw=2.5, reach=30.0,
                              width=0.05)
    flowed = np.asarray(fault(pair)) - np.asarray(pair)
    # an arc of length 2.5 on a curve of radius about five: the chord is
    # nearly the throw long and turned by a few degrees from the tangent
    assert np.linalg.norm(flowed[0]) == pytest.approx(2.5, abs=0.15)
    cosine = flowed[0] @ moved[0] / np.linalg.norm(flowed[0]) \
        / np.linalg.norm(moved[0])
    assert cosine > 0.95
    h = 0.05
    axis = np.linspace(-4.5, 4.5, 41)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    base = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    at = np.asarray(fault(tf.constant(base)))
    dx = (np.asarray(fault(tf.constant(base + [h, 0.0]))) - at) / h
    dy = (np.asarray(fault(tf.constant(base + [0.0, h]))) - at) / h
    det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    hanging = np.asarray(fault.surface(tf.constant(base))) > 0.3
    assert np.all(det[hanging] > 0.5)


def test_implicit_fault_blocks_trim_the_fault_that_ends_on_another():
    """F1 (vertical, x = 0) ends on F0 (horizontal, y = 0) from above: its
    coordinate is +-amp above F0 and exactly zero below, where the fault
    does not exist, while an untrimmed ImplicitFault keeps repelling there."""
    from geoml.transform import ImplicitFaultBlocks
    f0_points = np.stack([np.linspace(-5.0, 5.0, 21), np.zeros(21)], axis=1)
    f0 = ImplicitFault(f0_points, np.tile([0.0, 1.0], [21, 1]), reach=30.0)
    f1_points = np.stack([np.zeros(20), np.linspace(0.25, 5.0, 20)], axis=1)
    f1 = ImplicitFault(f1_points, np.tile([1.0, 0.0], [20, 1]), reach=30.0)
    blocks = ImplicitFaultBlocks([f0, f1], abutting=[(1, 0, 1)])
    query = tf.constant(np.array([[-1.0, 2.0], [1.0, 2.0],
                                  [-1.0, -2.0], [1.0, -2.0]]))
    out = np.asarray(blocks(query))
    a0 = float(f0.parameters["amp"].get_value())
    a1 = float(f1.parameters["amp"].get_value())
    assert out.shape == (4, 2)
    assert np.allclose(out[:, 0], [a0, a0, -a0, -a0])
    assert np.allclose(out[:, 1], [-a1, a1, 0.0, 0.0])
    # the same fault untrimmed still jumps below F0, within its reach
    alone = np.asarray(f1(query))[:, 0]
    assert alone[2] < 0 < alone[3]
    # in decay mode the trimmed distance fades the coordinate past F0
    f1_decay = ImplicitFault(f1_points, np.tile([1.0, 0.0], [20, 1]),
                             reach=30.0, mode="decay")
    f1_decay.parameters["range"].set_value(2.0)
    decayed = np.asarray(ImplicitFaultBlocks([f0, f1_decay],
                                             abutting=[(1, 0, 1)])(query))
    assert abs(decayed[3, 1]) < 0.4 * abs(decayed[1, 1])
    with pytest.raises(ValueError, match="different"):
        ImplicitFaultBlocks([f0, f1], abutting=[(1, 1, 1)])
    with pytest.raises(ValueError, match="ImplicitFault"):
        ImplicitFaultBlocks([f0, FaultDisplacement(f1_points,
                                                    np.tile([1.0, 0.0],
                                                            [20, 1]))])


def test_the_throw_search_picks_the_true_one():
    """Where the throw does not train from zero, the search on the bound
    finds it among candidates; here the true 20 against 0, 10 and 30."""
    geoml.set_seed(9)
    rng = np.random.default_rng(3)
    samples = rng.uniform(0.0, 100.0, size=[300, 3])
    samples = samples[_away(samples, 0, 50.0, margin=2.0)]
    value = samples[:, 2] - 20.0 * (samples[:, 0] > 50.0) \
        + rng.normal(scale=0.5, size=len(samples))
    train = _point_data(samples, value)
    axes = np.linspace(5.0, 95.0, 5)
    lattice = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"),
                       axis=-1).reshape(-1, 3)
    points, normals = _plane(axis=0, position=50.0, n=150)
    fault = FaultDisplacement(points, normals, reach=1000.0, width=1.0)
    root = geoml.latent.BasicInput(geoml.data.PointData.from_array(lattice),
                                   transform=ChainedTransform(fault,
                                                              Isotropic(30.0)))
    model = geoml.models.VGPNetwork(
        train, "v", geoml.likelihood.Gaussian(),
        geoml.latent.BasicGP(root, size=1),
        options=geoml.models.GPOptions(verbose=False))
    best, passes = geoml.models.search_throw(
        model, fault, [0.0, 10.0, 20.0, 30.0], iterations=40, refine=1)
    assert len(passes) == 2
    coarse, scores = passes[0]
    assert np.argmax(scores) == 2
    fine, _ = passes[1]
    assert np.allclose(fine, [10.0, 15.0, 20.0, 25.0, 30.0])
    assert abs(best - 20.0) <= 5.0
    assert float(fault.parameters["throw"].get_value()) == best
    assert model.training_log == []   # the bursts are not left in the log


def test_two_faults_restore_the_older_surface_and_the_layer():
    """Older fault F1 (y = 50, slip 10 up) cut by younger F2 (x = 50,
    slip 15 along y): F1 is observed in two pieces, y = 50 and y = 65."""
    f1_points, f1_normals = _plane(axis=1, position=50.0, n=300, seed=5)
    f1_points[f1_points[:, 0] > 50.0, 1] += 15.0   # displaced by F2
    f2_points, f2_normals = _plane(axis=0, position=50.0, n=300, seed=6)
    # F1 (normals +y): the up-dip direction is +z, so throw 10 is 10 up;
    # F2 (normals +x): its strike is +y, so strike_slip 15 is 15 along y
    older = FaultDisplacement(f1_points, f1_normals, throw=10.0,
                              reach=1000.0, width=0.5)
    younger = FaultDisplacement(f2_points, f2_normals, strike_slip=15.0,
                                reach=1000.0, width=0.5)
    network = FaultNetwork([younger, older])

    # undoing F2 puts F1's observations back on one plane, with a hard
    # step, so even the ones inside the smooth step's band come back whole
    restored = np.asarray(network._restored_points(1))
    assert np.allclose(restored[:, 1], 50.0, atol=1e-6)

    rng = np.random.default_rng(7)
    query = rng.uniform(0.0, 100.0, size=[400, 3])
    pre_f2 = query.copy()
    pre_f2[:, 1] -= 15.0 * (query[:, 0] > 50.0)
    keep = _away(query, 0, 50.0) & _away(pre_f2, 1, 50.0)
    query, pre_f2 = query[keep], pre_f2[keep]
    expected_z = query[:, 2] - 10.0 * (pre_f2[:, 1] > 50.0)
    out = np.asarray(network(tf.constant(query)))
    assert np.allclose(out[:, 1], pre_f2[:, 1], atol=0.1)
    assert np.allclose(out[:, 2], expected_z, atol=0.1)

    # a fault that stops against an older one does not displace it, so the
    # older surface is observed whole; declared abutting on F1's negative
    # side, F2 acts nowhere on the positive one and fully on the negative
    whole_points, whole_normals = _plane(axis=1, position=50.0, n=300,
                                         seed=5)
    whole = FaultDisplacement(whole_points, whole_normals, throw=10.0,
                              reach=1000.0, width=0.5)
    stopped = FaultNetwork([younger, whole], abutting=[(0, 1, -1)])
    rng = np.random.default_rng(8)
    probe = rng.uniform(0.0, 100.0, size=[300, 3])
    probe = probe[_away(probe, 0, 50.0) & _away(probe, 1, 50.0)]
    out = np.asarray(stopped(tf.constant(probe)))
    positive = probe[:, 1] > 50.0
    assert np.allclose(out[positive, 1], probe[positive, 1], atol=0.1)
    expected = probe[:, 1] - 15.0 * (probe[:, 0] > 50.0)
    assert np.allclose(out[~positive, 1], expected[~positive], atol=0.1)
    # and F1's own observations were not moved by F2
    assert np.allclose(np.asarray(stopped._restored_points(1)),
                       whole_points)
    with pytest.raises(ValueError, match="younger"):
        FaultNetwork([younger, older], abutting=[(1, 0, 1)])


def test_the_flow_is_exact_on_a_plane_and_follows_a_curved_level_set():
    """Fault-parallel flow: on a plane it is the straight step to the
    last digit; on a curved fault the moved points stay on the level set
    of the field they started on, where the straight step drifts off it."""
    from geoml.math.geometry import point_normals
    points, normals = _plane(axis=0, position=50.0)
    straight = FaultDisplacement(points, normals, throw=20.0, reach=1000.0,
                                 width=0.5, flow_steps=0)
    flow = FaultDisplacement(points, normals, throw=20.0, reach=1000.0,
                             width=0.5)
    query = tf.constant(np.random.default_rng(1).uniform(0, 100, [50, 3]))
    assert np.allclose(np.asarray(straight(query)), np.asarray(flow(query)),
                       atol=1e-9)

    y = np.linspace(-6.0, 6.0, 41)
    trace = np.stack([0.8 * np.sin(np.pi * y / 5.0), y], axis=1)
    normals = point_normals(trace, k=6, orient=[1.0, 0.0])
    kinds = [FaultDisplacement(trace, normals, throw=2.5, reach=30.0,
                               width=0.05, flow_steps=steps)
             for steps in (0, 4)]
    hanging = np.random.default_rng(2).uniform(-4.0, 4.0, size=[300, 2])
    hanging = hanging[np.asarray(kinds[0].surface(hanging)) > 0.5]
    drift = [np.abs(np.asarray(kind.surface(kind(tf.constant(hanging))))
                    - np.asarray(kind.surface(hanging))).max()
             for kind in kinds]
    assert drift[1] < 0.05 * 2.5
    assert drift[1] < 0.3 * drift[0]


def test_the_profile_peaks_at_the_centre_and_dies_at_the_tips():
    trace = np.stack([np.zeros(41), np.linspace(-5.0, 5.0, 41)], axis=1)
    normals = np.tile([1.0, 0.0], [41, 1])
    fault = FaultDisplacement(trace, normals, throw=2.0, reach=100.0,
                              width=0.05, profile="bell")
    # the extent starts a fifth past the observations: six here
    assert float(fault.parameters["extent"].get_value()) == pytest.approx(6.0)
    query = tf.constant(np.array([[1.0, 0.0], [1.0, 5.0], [1.0, 7.0]]))
    moved = np.linalg.norm(np.asarray(fault(query)) - np.asarray(query),
                           axis=1)
    assert moved[0] == pytest.approx(2.0, abs=0.05)
    assert 0.0 < moved[1] < 0.3
    assert moved[2] == pytest.approx(0.0, abs=1e-9)


def test_the_throw_is_read_from_markers_and_the_width_anneals():
    points, normals = _plane(axis=0, position=50.0)
    fault = FaultDisplacement(points, normals, reach=1000.0, width=0.5)
    # one horizon, restored z = 40: at z = 60 on the hanging wall of a
    # 20 m throw, z = 40 on the footwall
    rng = np.random.default_rng(3)
    hanging = np.stack([rng.uniform(55, 95, 30), rng.uniform(0, 100, 30),
                        np.full(30, 60.0)], axis=1)
    footwall = np.stack([rng.uniform(5, 45, 30), rng.uniform(0, 100, 30),
                         np.full(30, 40.0)], axis=1)
    throw = fault.throw_from_markers(hanging, footwall)
    assert throw == pytest.approx(20.0, abs=0.5)
    assert float(fault.parameters["throw"].get_value()) == throw

    assert fault.width == pytest.approx(0.5)
    assert fault.parameters["width"].fixed
    fault.set_width(2.0)
    assert fault.width == pytest.approx(2.0)
    drag = FaultDisplacement(points, normals, reach=1000.0, width=0.5,
                             drag=True)
    assert not drag.parameters["width"].fixed


def test_the_fault_transforms_save_and_load(tmp_path):
    points, normals = _plane(axis=0, position=50.0, n=100)
    fault = FaultDisplacement(points, normals, throw=20.0, strike_slip=3.0,
                              reach=1000.0, width=0.5)
    rng = np.random.default_rng(0)
    lattice = geoml.data.PointData.from_array(
        rng.uniform(0.0, 100.0, size=[40, 3]))
    root = geoml.latent.BasicInput(
        lattice, transform=ChainedTransform(fault, Isotropic(30.0)))
    gp = geoml.latent.BasicGP(root, size=1)
    coordinates = rng.uniform(0.0, 100.0, size=[30, 3])
    data = _point_data(coordinates, coordinates[:, 2])
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(), gp,
        options=geoml.models.GPOptions(verbose=False))
    geoml.persistence.save_model(model, str(tmp_path / "fault"))
    loaded = geoml.persistence.load_model(str(tmp_path / "fault"))
    query = tf.constant(np.array([[70.0, 30.0, 40.0], [20.0, 30.0, 40.0]]))
    before = np.asarray(root.transform(query))
    after = np.asarray(loaded.latent_network.root.transform(query))
    assert np.allclose(before, after)
