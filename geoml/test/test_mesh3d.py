"""The mesh hierarchy: Mesh3D, and the promises Surface3D and Solid3D make.

A `Mesh3D` measures itself as it is built — area, closed, consistent — and the
two subclasses turn those measurements into invariants, so an assignment need
only be handed the right type. `mesh3d` picks the class the geometry calls
for; `heal` repairs; `split` separates the pieces; and the booleans work
around VTK, which answers with nothing whenever two bodies do not cross.
"""
import numpy as np
import pytest
import pyvista as pv

import geoml
from geoml.data import Mesh3D, Surface3D, Solid3D, mesh3d


def _arrays(mesh):
    mesh = mesh.triangulate()
    points = np.asarray(mesh.points, dtype=float)
    return points, mesh.faces.reshape(-1, 4)[:, 1:]


def _build(mesh, cls=None):
    points, triangles = _arrays(mesh)
    normals = geoml.math.geometry.vertex_normals(points, triangles)
    if cls is None:
        return mesh3d(points, triangles, normals)
    return cls(points, triangles, normals)


def _sheet():
    points = np.array([[0.0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    return Surface3D(points, triangles,
                     geoml.math.geometry.vertex_normals(points, triangles))


# -- what a mesh knows about itself --------------------------------------- #

def test_a_sheet_measures_its_area():
    assert np.isclose(_sheet().area, 4.0 * 3.0)


def test_a_body_measures_its_area_and_volume():
    box = _build(pv.Box(bounds=(0, 2, 0, 3, 0, 4)), Solid3D)

    assert np.isclose(box.volume, 2.0 * 3.0 * 4.0)
    assert np.isclose(box.area, 2 * (2 * 3 + 2 * 4 + 3 * 4))


def test_the_measurements_are_taken_at_construction():
    sheet = _sheet()

    assert sheet.closed is False
    assert sheet.consistent is True
    assert not hasattr(sheet, "volume")     # a sheet encloses nothing


def test_a_sphere_measures_up():
    sphere = _build(pv.Sphere(radius=2.0, theta_resolution=60,
                              phi_resolution=60), Solid3D)

    # a triangulation falls just inside the sphere it approximates
    assert 0.98 < sphere.volume / (4 / 3 * np.pi * 8) < 1.0
    assert 0.98 < sphere.area / (4 * np.pi * 4) < 1.0


# -- the promises the subclasses make ------------------------------------- #

def test_a_closed_mesh_cannot_be_a_surface():
    points, triangles = _arrays(pv.Box())
    with pytest.raises(ValueError, match="this mesh closes"):
        Surface3D(points, triangles,
                  geoml.math.geometry.vertex_normals(points, triangles))


def test_an_open_mesh_cannot_be_a_solid():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    with pytest.raises(ValueError, match="does not close"):
        Solid3D(points, triangles,
                geoml.math.geometry.vertex_normals(points, triangles))


def test_a_body_whose_triangles_disagree_cannot_be_a_solid():
    points, triangles = _arrays(pv.Box())
    triangles = triangles.copy()
    triangles[0] = triangles[0][::-1]
    with pytest.raises(ValueError, match="disagree about which way is out"):
        Solid3D(points, triangles,
                geoml.math.geometry.vertex_normals(points, triangles))


def test_the_factory_picks_the_class_the_geometry_calls_for():
    assert isinstance(_build(pv.Box()), Solid3D)
    assert isinstance(_build(pv.Plane()), Surface3D)

    # closed, but its triangles disagree: neither a sheet nor a body
    points, triangles = _arrays(pv.Box())
    triangles = triangles.copy()
    triangles[0] = triangles[0][::-1]
    broken = mesh3d(points, triangles,
                    geoml.math.geometry.vertex_normals(points, triangles))
    assert type(broken) is Mesh3D


def test_faces_that_bound_nothing_are_dropped():
    """Neither a zero-area triangle nor a twinned face bounds anything, and
    both are read as defects: a sliver carries a self-edge belonging to one
    triangle, which counts as a boundary, and a twin makes its edges run
    twice the same way round, which counts as a winding failure. A contour
    emits both where the level set pinches at a cell corner, so they go
    before the geometry is asked what class it calls for.

    How many copies of a twin to drop is decided by its edges rather than
    assumed, and the two cases below are why. A flap standing clear of the
    surface owns its edges outright, so both copies go and nothing is left
    hanging; a wall written twice shares its edges with the body, so
    dropping both would leave each of them bounding one face -- an open edge
    where there is no hole -- and one copy stays.
    """
    points, triangles = _arrays(pv.Box())
    sliver = np.array([[triangles[0, 0], triangles[0, 1], triangles[0, 1]]])

    # a wall recorded twice: one copy survives, and the box stays a box
    twinned = np.concatenate([triangles, sliver, triangles[1:2]])
    assert not isinstance(
        mesh3d(points, twinned,
               geoml.math.geometry.vertex_normals(points, twinned)), Solid3D)
    kept, clean = geoml.math.geometry.drop_degenerate_faces(points, twinned)
    assert len(clean) == len(triangles)
    assert len(kept) == len(points)
    assert isinstance(
        mesh3d(kept, clean,
               geoml.math.geometry.vertex_normals(kept, clean)), Solid3D)

    # a flap standing clear of it: both copies go, and so does the flap
    flap = np.array([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0], [0.0, 1.0, 9.0]])
    offset = len(points)
    face = [[offset, offset + 1, offset + 2]]
    loose = np.concatenate([points, flap])
    kept, clean = geoml.math.geometry.drop_degenerate_faces(
        loose, np.concatenate([triangles, face, face]))
    assert len(clean) == len(triangles)
    assert geoml.math.geometry.reversed_edges(kept, clean) == 0


def test_a_doubled_patch_is_resurrected_from_its_rim_inward():
    """The twin groups couple, which is why the rule is worked out on the
    surface rather than per face: in a doubled patch -- what decimation
    makes of a collapsed thin feature -- the middle face sees every
    neighbour as a twin, so any rule scoring its edges in isolation drops
    it while the rim stays, and the patch tears down the middle. Reported
    from a real model as `NotClosedError: 32 of its edges belong to a
    single triangle` out of `simplify`. Here an octahedron has a face and
    all three of its neighbours written twice; the rim's copies come back
    on the first pass and the middle's on the second, once the rim's
    return leaves its edges half-supported.
    """
    v = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    doubled = np.concatenate([f, f[[0, 1, 3, 4]]])

    kept, clean = geoml.math.geometry.drop_degenerate_faces(v, doubled)

    assert len(clean) == len(f)
    assert geoml.math.geometry.open_edges(kept, clean) == 0
    body = mesh3d(kept, clean,
                  geoml.math.geometry.vertex_normals(kept, clean))
    assert isinstance(body, Solid3D)


def test_a_contour_that_closes_comes_back_as_a_solid():
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[24, 24, 24], step=[1, 1, 1])
    radius = np.linalg.norm(
        np.asarray(grid.coordinates) - np.array([[11.5, 11.5, 11.5]]), axis=1)
    grid.add_continuous_variable("v", radius)

    shell = grid.variables["v"].measurements.get_contour(8.0)

    assert isinstance(shell, Solid3D)
    assert 0.95 < shell.volume / (4 / 3 * np.pi * 8 ** 3) < 1.05


# -- healing --------------------------------------------------------------- #

def test_healing_closes_a_tube():
    tube = _build(pv.Cylinder(radius=1.0, height=2.0, capping=False))
    assert isinstance(tube, Surface3D)

    healed = tube.heal(hole_size=3.0)

    assert isinstance(healed, Solid3D)
    assert np.isclose(healed.volume, np.pi * 2, rtol=0.01)


def test_healing_settles_triangles_that_disagree():
    points, triangles = _arrays(pv.Box(bounds=(0, 1, 0, 1, 0, 1)))
    triangles = triangles.copy()
    triangles[0] = triangles[0][::-1]
    broken = mesh3d(points, triangles,
                    geoml.math.geometry.vertex_normals(points, triangles))
    assert not broken.consistent

    healed = broken.heal()

    assert isinstance(healed, Solid3D)
    assert np.isclose(healed.volume, 1.0)


def test_healing_settles_a_disagreement_reorienting_cannot():
    """The winding failure that is not a winding failure. A zero-thickness
    flap makes each of its edges run twice the same way round, and it is
    the one such disagreement VTK cannot walk across: measured on this very
    mesh, `clean`, `compute_normals(consistent_normals=True,
    auto_orient_normals=True)` and `triangulate` left all three reversed
    edges exactly as they were. Since every error message in the module
    sends the user to `heal`, it drops what bounds nothing first.
    """
    points, triangles = _arrays(pv.Box(bounds=(0, 1, 0, 1, 0, 1)))
    flap = np.array([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0], [0.0, 1.0, 9.0]])
    offset = len(points)
    twin = [[offset, offset + 1, offset + 2]]
    dirty_points = np.concatenate([points, flap])
    dirty = np.concatenate([triangles, twin, twin])

    broken = mesh3d(dirty_points, dirty,
                    geoml.math.geometry.vertex_normals(dirty_points, dirty))
    assert type(broken) is Mesh3D
    assert not broken.consistent

    healed = broken.heal()

    assert isinstance(healed, Solid3D)
    assert np.isclose(healed.volume, 1.0)


def test_healing_leaves_a_hole_too_large_to_cover():
    tube = _build(pv.Cylinder(radius=1.0, height=2.0, capping=False))

    healed = tube.heal(hole_size=0.1)

    assert isinstance(healed, Surface3D)     # still open, and says so


# -- splitting ------------------------------------------------------------- #

def test_a_mesh_in_one_piece_splits_into_itself():
    box = _build(pv.Box(), Solid3D)
    assert box.split() == [box]


def test_two_bodies_in_one_mesh_come_apart():
    here = _build(pv.Sphere(radius=1.0, center=(0, 0, 0)), Solid3D)
    away = _build(pv.Sphere(radius=1.0, center=(10, 0, 0)), Solid3D)

    both = here.union(away)
    pieces = both.split()

    assert len(pieces) == 2
    assert all(isinstance(piece, Solid3D) for piece in pieces)
    assert np.isclose(sum(piece.volume for piece in pieces), both.volume)


# -- booleans -------------------------------------------------------------- #

def _spheres(gap):
    return (_build(pv.Sphere(radius=1.0, center=(0, 0, 0)), Solid3D),
            _build(pv.Sphere(radius=1.0, center=(gap, 0, 0)), Solid3D))


def test_bodies_that_cross():
    here, there = _spheres(1.2)

    union = here.union(there)
    common = here.intersection(there)
    rest = here.difference(there)

    assert union.volume > here.volume
    assert 0 < common.volume < here.volume
    assert np.isclose(union.volume, here.volume + there.volume - common.volume,
                      rtol=0.02)
    assert np.isclose(rest.volume, here.volume - common.volume, rtol=0.02)


def test_bodies_that_stand_apart():
    here, there = _spheres(10.0)

    # VTK answers with nothing for all three; these are worked out instead
    assert np.isclose(here.union(there).volume, here.volume + there.volume)
    assert here.intersection(there).volume == 0.0
    assert np.isclose(here.difference(there).volume, here.volume)


def test_one_body_inside_another():
    small = _build(pv.Sphere(radius=1.0), Solid3D)
    big = _build(pv.Sphere(radius=3.0), Solid3D)

    assert np.isclose(small.union(big).volume, big.volume)
    assert np.isclose(small.intersection(big).volume, small.volume)
    assert small.difference(big).volume == 0.0


def test_a_body_with_a_cavity():
    small = _build(pv.Sphere(radius=1.0), Solid3D)
    big = _build(pv.Sphere(radius=3.0), Solid3D)

    shell = big.difference(small)

    assert isinstance(shell, Solid3D)
    assert np.isclose(shell.volume, big.volume - small.volume)
    assert len(shell.split()) == 2       # the outer wall and the cavity's

    # the hollow reads as outside, which is the point of the cavity
    data = geoml.data.PointData(
        __import__("pandas").DataFrame(
            [[0.0, 0, 0], [2.0, 0, 0], [5.0, 0, 0]],
            columns=["X", "Y", "Z"]), ["X", "Y", "Z"])
    data.assign_from_solid(shell, "rock")
    assert list(data.get_metadata("rock")) == ["outside", "inside", "outside"]


def test_an_empty_result_is_still_a_body():
    here, there = _spheres(10.0)
    empty = here.intersection(there)

    assert isinstance(empty, Solid3D)
    assert empty.n_data == 0
    assert empty.volume == 0.0
    assert empty.area == 0.0


def test_a_body_cannot_be_unioned_with_a_sheet():
    box = _build(pv.Box(), Solid3D)
    with pytest.raises(geoml.data.MeshTypeError, match="no volume"):
        box.union(_sheet())


def test_a_body_can_only_be_combined_with_a_mesh():
    box = _build(pv.Box(), Solid3D)
    with pytest.raises(geoml.data.MeshTypeError, match="or cut by"):
        box.union("not a mesh at all")


# -- persistence ----------------------------------------------------------- #

@pytest.mark.parametrize("builder", [
    lambda: _sheet(),
    lambda: _build(pv.Box(bounds=(0, 2, 0, 3, 0, 4)), Solid3D),
])
def test_a_mesh_keeps_its_class_through_zarr(tmp_path, builder):
    mesh = builder()
    path = str(tmp_path / "mesh.zarr")
    mesh.to_zarr(path)

    reloaded = geoml.data.PointData.open(path)

    assert type(reloaded) is type(mesh)
    assert np.allclose(np.asarray(reloaded.coordinates),
                       np.asarray(mesh.coordinates))
    assert np.array_equal(np.asarray(reloaded.triangles),
                          np.asarray(mesh.triangles))
    assert np.isclose(reloaded.area, mesh.area)
