"""The parameter-free implicit surface: a polyharmonic Hermite RBF through
points with their normals, and the normals derived when absent.

Every case has an analytic answer -- a plane is the drift, a sphere has a
known inside, a squashed sphere a known normal -- so the interpolant, the
sign conventions, the derived normals and the greedy centre selection are
each checked against a number rather than a picture.
"""
import numpy as np
import pytest

import geoml
from geoml.math.geometry import point_normals
from geoml.math.rbf import HermiteRBF


def _sphere(n=400, radius=50.0, centre=(10.0, -20.0, 5.0)):
    """Points spread evenly over a sphere, with their outward normals."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    ring = np.sqrt(1.0 - z ** 2)
    angle = np.pi * (1.0 + np.sqrt(5.0)) * i
    unit = np.stack([ring * np.cos(angle), ring * np.sin(angle), z], axis=1)
    return np.asarray(centre) + radius * unit, unit


def _plane(n=300, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 100.0, size=[n, 2])
    return np.concatenate([xy, np.zeros([n, 1])], axis=1)


def test_a_plane_is_reproduced_by_the_drift():
    points = _plane()
    field = HermiteRBF(points, normals=np.tile([0.0, 0.0, 1.0], [300, 1]))
    rng = np.random.default_rng(1)
    query = rng.uniform(-50.0, 150.0, size=[200, 3])
    assert np.allclose(np.asarray(field(query)), query[:, 2], atol=1e-6)
    assert np.allclose(np.asarray(field.gradient(query)), [0.0, 0.0, 1.0],
                       atol=1e-6)


def test_a_sphere_has_its_inside_behind_the_normals():
    points, normals = _sphere()
    field = HermiteRBF(points, normals=normals)
    assert field.n_centres == 400
    assert field.max_residual < 1e-6
    assert np.max(np.abs(np.asarray(field.gradient(points)) - normals)) < 0.05
    centre = np.array([[10.0, -20.0, 5.0]])
    assert np.asarray(field(centre))[0] < 0.0
    inner, _ = _sphere(50, radius=35.0)
    outer, _ = _sphere(50, radius=70.0)
    assert np.all(np.asarray(field(inner)) < 0.0)
    assert np.all(np.asarray(field(outer)) > 0.0)


def test_derived_normals_point_to_the_concavity():
    points, outward = _sphere()
    normals = point_normals(points, k=10)
    # a sphere curves inward everywhere: every derived normal points in
    assert np.all(np.einsum("ni,ni->n", normals, outward) < -0.9)
    field = HermiteRBF(points)  # derives them itself
    assert np.asarray(field(np.array([[10.0, -20.0, 5.0]])))[0] > 0.0


def test_a_flat_patch_is_consistent_and_says_the_sign_is_arbitrary():
    points = _plane()
    with pytest.warns(UserWarning, match="flat"):
        normals = point_normals(points, k=10)
    assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1e-9)
    assert len(set(np.sign(normals[:, 2]))) == 1
    upward = point_normals(points, k=10, orient=[0.0, 0.0, 1.0])
    assert np.all(upward[:, 2] > 0.999)


def test_a_single_trace_cannot_fix_a_normal():
    t = np.linspace(0.0, 100.0, 60)
    line = np.stack([t, 0.3 * t, -0.5 * t], axis=1)
    with pytest.raises(ValueError, match="line"):
        point_normals(line, k=8)


def test_a_few_normals_or_one_are_enough():
    """Nothing is asked per point: normals on a subset, aligned with NaN
    rows elsewhere, at their own locations, or a single one."""
    points = _plane()
    some = np.full([300, 3], np.nan)
    some[::60] = [0.0, 0.0, 1.0]
    field = HermiteRBF(points, normals=some)
    query = np.random.default_rng(2).uniform(-50.0, 150.0, size=[100, 3])
    assert np.allclose(np.asarray(field(query)), query[:, 2], atol=1e-6)

    sphere, outward = _sphere()
    one = HermiteRBF(sphere, normals=(sphere[:1], outward[:1]))
    assert one.max_residual < 1e-6
    assert len(one.gradient_points) == 1
    centre = np.array([[10.0, -20.0, 5.0]])
    assert np.asarray(one(centre))[0] < 0.0
    far, _ = _sphere(30, radius=90.0)
    assert np.all(np.asarray(one(far)) > 0.0)
    # a normal measured away from the points still pins the field
    elsewhere = HermiteRBF(sphere, normals=(np.array([[10.0, -20.0, 60.0]]),
                                            np.array([[0.0, 0.0, 1.0]])))
    assert np.asarray(elsewhere(centre))[0] < 0.0


def test_the_off_surface_form_in_two_dimensions():
    angle = np.linspace(0.0, 2 * np.pi, 120, endpoint=False)
    unit = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    circle = 30.0 * unit
    for basis in ("thin_plate", "linear"):
        field = HermiteRBF(circle, normals=unit, basis=basis)
        assert field.max_residual < 1e-6
        assert np.asarray(field(np.zeros([1, 2])))[0] < 0.0
        assert np.all(np.asarray(field(45.0 * unit[::7])) > 0.0)
        assert np.all(np.asarray(field(15.0 * unit[::7])) < 0.0)


def test_greedy_centres_meet_the_budget_with_fewer_points():
    geoml.set_seed(3)
    points, normals = _sphere(n=1500)
    field = HermiteRBF(points, normals=normals, max_error=0.05)
    assert field.n_centres < 1500
    assert field.max_residual <= 0.05
    residual = np.abs(np.asarray(field(points)))
    assert residual.max() <= 0.05


def test_a_fixed_transform_keeps_the_given_normals():
    points, normals = _sphere()
    field = HermiteRBF(points, normals=normals,
                       transform=geoml.transform.Isotropic(10.0))
    assert field.max_residual < 1e-6
    # the gradient comes back in the original coordinates: the normals
    assert np.max(np.abs(np.asarray(field.gradient(points)) - normals)) < 0.05


def test_the_contour_closes_around_the_sphere():
    points, normals = _sphere()
    field = HermiteRBF(points, normals=normals)
    grid = geoml.data.Grid3D(start=[-70.0, -100.0, -75.0], n=[41, 41, 41],
                             step=[4.0, 4.0, 4.0])
    mesh = field.contour(grid)
    assert mesh.closed
    assert mesh.volume == pytest.approx(4.0 / 3.0 * np.pi * 50.0 ** 3,
                                        rel=0.05)
