"""Closing a contour, terrains, and cutting one kind of mesh with another.

`get_contour(close=...)` pads the cube so a surface running out of the grid
closes inside it. `DTM3D` is a sheet that promises never to fold over, which
is what lets a body be divided into what lies under it and what lies over it.
Cutting keeps the caller's kind: a sheet cut by a body is a sheet, a body cut
by a sheet is a body.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import pyvista as pv

import geoml
from geoml.data import (Mesh3D, Surface3D, Solid3D, DTM3D, mesh3d,
                        MeshTypeError, NotSingleValuedError, NotClosedError,
                        InconsistentMeshError)


def _build(mesh, cls=None):
    mesh = mesh.triangulate()
    points = np.asarray(mesh.points, dtype=float)
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    normals = geoml.math.geometry.vertex_normals(points, triangles)
    return mesh3d(points, triangles, normals) if cls is None \
        else cls(points, triangles, normals)


def _terrain(height=5.0, slope=0.0, extent=20.0, cls=DTM3D, n=9):
    """A flat or tilted sheet spanning [-extent, extent] in x and y."""
    axis = np.linspace(-extent, extent, n)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    gz = height + slope * gx
    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    index = np.arange(points.shape[0]).reshape(gx.shape)
    triangles = []
    for i in range(index.shape[0] - 1):
        for j in range(index.shape[1] - 1):
            a, b = index[i, j], index[i, j + 1]
            c, d = index[i + 1, j + 1], index[i + 1, j]
            triangles += [[a, b, c], [a, c, d]]
    triangles = np.array(triangles)
    return cls(points, triangles,
               geoml.math.geometry.vertex_normals(points, triangles))


def _grid_with_a_ball(centre, radius_value=6.0):
    """A field whose contour runs out of the grid on one side."""
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[20, 20, 20], step=[1, 1, 1])
    radius = np.linalg.norm(
        np.asarray(grid.coordinates) - np.array([centre]), axis=1)
    grid.add_continuous_variable("v", radius)
    return grid


# -- closing a contour at the grid's edge ---------------------------------- #

def test_a_contour_running_out_of_the_grid_is_open():
    grid = _grid_with_a_ball([3.0, 9.5, 9.5])
    shell = grid.variables["v"].measurements.get_contour(6.0)

    assert isinstance(shell, Surface3D)
    assert not shell.closed


def test_closing_shuts_it_against_the_boundary():
    grid = _grid_with_a_ball([3.0, 9.5, 9.5])
    shell = grid.variables["v"].measurements.get_contour(6.0, close="below")

    assert isinstance(shell, Solid3D)
    assert shell.volume > 0


def test_closing_leaves_a_contour_that_already_closed_alone():
    grid = _grid_with_a_ball([9.5, 9.5, 9.5])
    open_ended = grid.variables["v"].measurements.get_contour(6.0)
    closed = grid.variables["v"].measurements.get_contour(6.0, close="below")

    assert isinstance(open_ended, Solid3D)
    assert np.isclose(open_ended.volume, closed.volume, rtol=0.01)


def test_the_two_sides_close_different_regions():
    grid = _grid_with_a_ball([3.0, 9.5, 9.5])
    below = grid.variables["v"].measurements.get_contour(6.0, close="below")
    above = grid.variables["v"].measurements.get_contour(6.0, close="above")

    # "below" is the ball, "above" everything else in the grid
    assert below.volume < above.volume


def test_an_unknown_side_is_refused():
    grid = _grid_with_a_ball([9.5, 9.5, 9.5])
    with pytest.raises(ValueError, match="close takes"):
        grid.variables["v"].measurements.get_contour(6.0, close="outside")


# -- terrains -------------------------------------------------------------- #

def test_a_terrain_is_a_surface():
    terrain = _terrain()

    assert isinstance(terrain, Surface3D)
    assert not terrain.closed
    assert np.isclose(terrain.area, 40.0 * 40.0)


def _folded_sheet():
    """Two triangles over the same footprint, facing opposite ways."""
    points = np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0],
                       [0.0, 0, 1], [0, 2, 1], [2, 0, 1]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    return points, triangles, geoml.math.geometry.vertex_normals(points, triangles)


def test_a_sheet_that_folds_over_is_not_a_terrain():
    points, triangles, normals = _folded_sheet()

    Surface3D(points, triangles, normals)          # allowed without the promise
    with pytest.raises(NotSingleValuedError, match="folds over"):
        DTM3D(points, triangles, normals)


def test_a_closed_body_is_not_a_terrain_either():
    points = np.asarray(pv.Box().triangulate().points, dtype=float)
    triangles = pv.Box().triangulate().faces.reshape(-1, 4)[:, 1:]
    with pytest.raises(MeshTypeError):
        DTM3D(points, triangles,
              geoml.math.geometry.vertex_normals(points, triangles))


def test_a_terrain_survives_zarr(tmp_path):
    terrain = _terrain()
    path = str(tmp_path / "dtm.zarr")
    terrain.to_zarr(path)

    reloaded = geoml.data.PointData.open(path)

    assert type(reloaded) is DTM3D
    assert np.isclose(reloaded.area, terrain.area)


# -- a sheet cut by a body ------------------------------------------------- #

def test_a_sheet_cut_by_a_body_stays_a_sheet():
    # the cut follows the sheet's own triangles, so a coarse sheet gives a
    # coarse circle however fine the ball is
    sheet = _terrain(height=0.0, extent=10.0, cls=Surface3D, n=61)
    ball = _build(pv.Sphere(radius=4.0, center=(0, 0, 0),
                            theta_resolution=90, phi_resolution=90), Solid3D)

    inside = sheet.intersection(ball)
    outside = sheet.difference(ball)

    assert isinstance(inside, Surface3D)
    assert isinstance(outside, Surface3D)
    # the sheet crosses the ball's equator, a disc of radius 4
    assert np.isclose(inside.area, np.pi * 16, rtol=0.02)
    assert np.isclose(inside.area + outside.area, sheet.area, rtol=1e-6)


def test_a_sheet_clear_of_the_body_comes_back_whole_or_empty():
    sheet = _terrain(height=50.0, extent=10.0, cls=Surface3D)
    ball = _build(pv.Sphere(radius=4.0, center=(0, 0, 0)), Solid3D)

    assert sheet.intersection(ball).n_data == 0
    assert np.isclose(sheet.difference(ball).area, sheet.area)


def test_a_sheet_is_cut_by_a_terrain_through_its_underneath():
    """A sheet has no inside, but a single-valued one has an underneath, so
    a flat sheet cut below a tilted terrain keeps the half lying under it."""
    sheet = _terrain(height=0.0, extent=10.0, cls=Surface3D, n=41)
    topo = _terrain(height=0.0, slope=0.5, cls=DTM3D)

    below = sheet.intersection(topo)
    above = sheet.difference(topo)

    assert isinstance(below, Surface3D)
    assert isinstance(above, Surface3D)
    # the terrain crosses the sheet along x=0: half below, half above
    assert np.isclose(below.area, sheet.area / 2, rtol=0.02)
    assert np.isclose(below.area + above.area, sheet.area, rtol=1e-6)


def test_a_sheet_cannot_be_cut_by_what_has_no_underneath():
    sheet = _terrain(cls=Surface3D)
    with pytest.raises(NotSingleValuedError, match="folds over"):
        sheet.intersection(Surface3D(*_folded_sheet()))


# -- a body cut by a sheet ------------------------------------------------- #

def test_a_body_cut_by_a_terrain_stays_a_body():
    ball = _build(pv.Sphere(radius=4.0, center=(0, 0, 0)), Solid3D)
    ground = _terrain(height=0.0)

    below = ball.intersection(ground)
    above = ball.difference(ground)

    assert isinstance(below, Solid3D)
    assert isinstance(above, Solid3D)
    # the terrain cuts the ball through its centre
    assert np.isclose(below.volume, ball.volume / 2, rtol=0.02)
    assert np.isclose(below.volume + above.volume, ball.volume, rtol=0.02)


def test_a_tilted_terrain_still_halves_it():
    ball = _build(pv.Sphere(radius=4.0, center=(0, 0, 0)), Solid3D)

    below = ball.intersection(_terrain(height=0.0, slope=0.5))

    assert np.isclose(below.volume, ball.volume / 2, rtol=0.05)


def test_a_body_wholly_under_the_terrain_is_kept_whole():
    ball = _build(pv.Sphere(radius=1.0, center=(0, 0, -20)), Solid3D)
    ground = _terrain(height=0.0)

    assert np.isclose(ball.intersection(ground).volume, ball.volume, rtol=0.02)
    assert ball.difference(ground).volume == 0.0


def test_a_sheet_that_folds_cannot_divide_a_body():
    ball = _build(pv.Sphere(radius=1.0, center=(1, 1, 0)), Solid3D)
    folded = Surface3D(*_folded_sheet())

    with pytest.raises(NotSingleValuedError, match="folds over"):
        ball.intersection(folded)


def test_a_sheet_too_small_to_cross_the_body_is_refused():
    ball = _build(pv.Sphere(radius=4.0), Solid3D)
    short = _terrain(height=0.0, extent=1.0)

    with pytest.raises(MeshTypeError, match="does not reach across"):
        ball.intersection(short)


def test_a_body_and_a_sheet_have_no_union():
    ball = _build(pv.Sphere(radius=4.0), Solid3D)
    with pytest.raises(MeshTypeError, match="no volume"):
        ball.union(_terrain(height=0.0))


def test_cutting_a_block_model_down_to_the_ground():
    """The reason for all of this: a body trimmed to the topography."""
    pit = _build(pv.Box(bounds=(-5, 5, -5, 5, -10, 10)), Solid3D)
    ground = _terrain(height=0.0, slope=0.2)

    rock = pit.intersection(ground)

    assert isinstance(rock, Solid3D)
    assert np.isclose(rock.volume, pit.volume / 2, rtol=0.02)

    data = geoml.data.PointData(
        pd.DataFrame([[0.0, 0, -5], [0.0, 0, 5]], columns=["X", "Y", "Z"]),
        ["X", "Y", "Z"])
    data.assign_from_solid(rock, "rock")
    assert list(data.get_metadata("rock")) == ["inside", "outside"]


# -- the numerics at mine-grid coordinates ---------------------------------- #

def _mine_site(offset):
    """An irregular TIN terrain and a box through it, `offset` away from the
    origin -- the geometry that broke VTK's boolean at UTM coordinates."""
    from scipy.spatial import Delaunay

    # built at the origin and translated afterwards, so the two sites hold
    # exactly the same geometry and their cuts can be compared to the metre
    rng = np.random.default_rng(42)
    n = 40
    axis = np.linspace(0, 4000, n)
    gx, gy = np.meshgrid(axis, axis)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    inner = ((xy[:, 0] > axis[1]) & (xy[:, 0] < axis[-2])
             & (xy[:, 1] > axis[1]) & (xy[:, 1] < axis[-2]))
    xy[inner] += rng.uniform(-35, 35, [inner.sum(), 2])
    z = (300 + 40 * np.sin(xy[:, 0] / 700)
         + 30 * np.cos(xy[:, 1] / 900) + rng.normal(0, 4, len(xy)))
    tri = Delaunay(xy).simplices
    points = np.column_stack([xy, z]) + offset
    dtm = Surface3D(points, tri, geoml.math.geometry.vertex_normals(points, tri))

    box = _build(pv.Box(bounds=(offset[0] + 1500, offset[0] + 2500,
                                offset[1] + 1500, offset[1] + 2500,
                                offset[2] + 150, offset[2] + 450)), Solid3D)
    return dtm, box


def test_the_booleans_hold_at_mine_grid_coordinates():
    """VTK's intersection filter works to absolute tolerances, so this same
    geometry used to come back empty or unclosed at UTM-scale coordinates
    while succeeding at the origin; the local frame is what holds it."""
    near_dtm, near_box = _mine_site(np.zeros(3))
    far_dtm, far_box = _mine_site(np.array([500000.0, 7000000.0, 0.0]))

    reference = near_box.intersection(near_dtm)
    far = far_box.intersection(far_dtm)

    assert isinstance(far, Solid3D)
    assert far.volume > 0
    # to VTK's own cut-line noise: measured ~1.6e-6 apart on this geometry
    assert np.isclose(far.volume, reference.volume, rtol=1e-5)


def test_clipping_a_sheet_holds_at_mine_grid_coordinates():
    """The sheet-cut-by-body path shares the local frame: the piece of the
    terrain inside the box must not depend on where the mine sits."""
    near_dtm, near_box = _mine_site(np.zeros(3))
    far_dtm, far_box = _mine_site(np.array([500000.0, 7000000.0, 0.0]))

    reference = near_dtm.intersection(near_box)
    far = far_dtm.intersection(far_box)

    assert isinstance(far, Surface3D)
    assert far.area > 0
    assert np.isclose(far.area, reference.area, rtol=1e-6)


def test_the_booleans_are_exact_on_the_triangles():
    """Two boxes overlapping in a box: every answer is exact, not a grid's
    approximation of it."""
    one = _build(pv.Box(bounds=(0, 2, 0, 2, 0, 2)), Solid3D)
    two = _build(pv.Box(bounds=(1, 3, 0, 2, 0, 2)), Solid3D)

    assert np.isclose(one.intersection(two).volume, 4.0, rtol=1e-12)
    assert np.isclose(one.union(two).volume, 12.0, rtol=1e-12)
    assert np.isclose(one.difference(two).volume, 4.0, rtol=1e-12)


def _box(low, high, offset):
    """A box in double precision. pyvista's own sources hold float32
    points, which cannot place a millimetre at mine coordinates: a box
    edge at 9.999 lands on 10.0 once moved 500 km."""
    unit = pv.Box(bounds=(0, 1, 0, 1, 0, 1)).triangulate()
    points = np.asarray(unit.points, dtype=float) \
        * (np.asarray(high) - np.asarray(low)) + low + offset
    triangles = unit.faces.reshape(-1, 4)[:, 1:]
    return mesh3d(points, triangles,
                  geoml.math.geometry.vertex_normals(points, triangles))


@pytest.mark.parametrize("offset", [np.zeros(3),
                                    np.array([500000.0, 7000000.0, 300.0])])
def test_an_answer_touching_itself_comes_back_a_body(offset):
    """Two quadrants of a slab left meeting along one edge: Manifold keeps
    the edge twice, and welded by position it would be an edge four
    triangles share -- a Mesh3D with no volume."""
    slab = _box([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], offset)
    first = _box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], offset)
    second = _box([1.0, 1.0, 0.0], [2.0, 2.0, 1.0], offset)
    left = slab.difference(first).difference(second)
    assert isinstance(left, Solid3D)
    assert left.volume == pytest.approx(2.0, abs=1e-4)
    assert len(left.split()) == 2
    # and it goes back into Manifold as a body
    assert left.union(first).volume == pytest.approx(3.0, abs=1e-4)


@pytest.mark.parametrize("offset", [np.zeros(3),
                                    np.array([500000.0, 7000000.0, 300.0])])
def test_a_film_thinner_than_any_grid_is_measured_exactly(offset):
    """Adjacent rock domains meet along films far thinner than any grid
    step the booleans could afford, and a grid inflated them to its own
    thickness: the Assen intersections read up to 5000 times the exact
    volume. A millimetre film, at the origin and at mine coordinates."""
    one = _box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], offset)
    two = _box([9.999, 0.0, 0.0], [20.0, 10.0, 10.0], offset)

    film = one.intersection(two)
    assert isinstance(film, Solid3D)
    assert np.isclose(film.volume, 0.001 * 10 * 10, rtol=1e-6)


def test_a_body_manifold_refuses_is_named_not_answered_with_nothing(
        monkeypatch):
    """An engine that fails by returning nothing reads exactly like two
    bodies that never meet -- the trap VTK's filter set -- so a body
    Manifold will not take is refused out loud."""
    import geoml.data.meshes as meshes

    one = _build(pv.Box(bounds=(0, 2, 0, 2, 0, 2)), Solid3D)
    two = _build(pv.Box(bounds=(1, 3, 0, 2, 0, 2)), Solid3D)
    weld = meshes._gmt.weld

    def spoiled(points, triangles, precision=6):
        points, triangles = weld(points, triangles, precision)
        return np.full_like(points, np.nan), triangles

    monkeypatch.setattr(meshes._gmt, "weld", spoiled)
    with pytest.raises(InconsistentMeshError, match="Manifold cannot take"):
        one.intersection(two)


def test_the_contoured_shell_cuts_after_all():
    """The real failing case: VTK's exact filter dropped whole patches when a
    contoured shell met a box. The volume must be right whichever engine
    answers."""
    shell = _contoured_shell()
    box = _build(pv.Box(bounds=(5.0, 15.0, 5.0, 15.0, 5.0, 15.0)), Solid3D)

    cut = shell.intersection(box)

    rng = np.random.default_rng(0)
    pts = rng.uniform([5, 5, 5], [15, 15, 15], [200000, 3])
    reference = (np.linalg.norm(pts - [9.5, 9.5, 9.5], axis=1)
                 < 6.0).mean() * 1000.0

    assert isinstance(cut, Solid3D)
    assert np.isclose(cut.volume, reference, rtol=0.03)


# -- fewer triangles, the same shape ---------------------------------------- #

def _contoured_shell():
    grid = _grid_with_a_ball([9.5, 9.5, 9.5])
    return grid.variables["v"].measurements.get_contour(6.0)


def test_simplify_honours_its_error_budget():
    """The argument is geometric -- how far the surface may move, in its own
    units -- and it is enforced by measurement, since the decimator's own
    error metric runs loose at tight budgets (measured 7x over)."""
    shell = _contoured_shell()
    budget = 0.5
    slim = shell.simplify(budget)

    assert isinstance(slim, Solid3D)
    assert len(slim.triangles) < 0.5 * len(shell.triangles)
    assert np.isclose(slim.volume, shell.volume, rtol=0.05)

    # an independent probe of the promise: points on the simplified faces
    # must sit within the budget of the original surface
    pts = np.asarray(slim.coordinates)
    tri = np.asarray(slim.triangles)
    probes = np.concatenate([pts[tri].mean(axis=1),
                             (pts[tri[:, 0]] + pts[tri[:, 1]]) / 2])
    distance = pv.PolyData(np.ascontiguousarray(probes)) \
        .compute_implicit_distance(shell._polydata())["implicit_distance"]
    assert np.abs(np.asarray(distance)).max() <= budget * 1.001


def _block_shell(centre, radius=45.0, n=8):
    """A closed body contoured out of a block model, which is the shape the
    package actually produces and the one VTK's boolean could not survive."""
    step = 160.0 / n
    blocks = geoml.data.BlockSet3D([0, 0, 0], [n, n, n], [step] * 3,
                                   discretization=(2, 2, 2), max_levels=1)
    values = radius - np.linalg.norm(
        np.asarray(blocks.coordinates) - np.asarray([centre]), axis=1)
    blocks.add_continuous_variable("g")
    blocks.variables["g"].prediction.values[:] = values
    blocks.variables["g"].allocate_simulations(1)
    blocks.variables["g"].simulations[:, :] = values[:, None]
    return blocks.get_contour("g", 0.0, close="above")


@pytest.mark.parametrize("operation", ["intersection", "union", "difference"])
def test_two_block_model_shells_combine(operation):
    """The reported crash. `vtkIntersectionPolyDataFilter` **segfaults** on
    contour-derived bodies -- measured from 836 triangles upwards, at the
    origin and at mine-grid coordinates, and on the same shells cleaned,
    de-slivered and decimated -- so VTK's filter is not consulted at all, and
    Manifold, which takes these shells, answers in-process.

    A regression here does not fail this test: it takes the whole test
    runner down with it, which is the loudest signal available and the
    reason VTK's engine was dropped rather than guarded.
    """
    one, two = _block_shell([60.0, 80.0, 80.0]), _block_shell([100.0, 80.0, 80.0])
    assert isinstance(one, Solid3D) and isinstance(two, Solid3D)

    combined = getattr(one, operation)(two)

    assert isinstance(combined, Solid3D)
    assert combined.volume > 0
    if operation == "union":
        assert combined.volume > max(one.volume, two.volume)
    if operation == "intersection":
        assert combined.volume < min(one.volume, two.volume)


def test_a_flap_left_by_decimation_does_not_break_the_rebuild():
    """Collapsing an edge through a thin feature folds it into a
    zero-thickness flap -- the same triangle twice, bounding nothing. Every
    edge of it then runs twice the same way round, so the constructor reads
    the body as inconsistently wound and refuses.

    It cannot be repaired by reorienting, which is what `_rebuilt_as` used
    to try alone: a flap is non-manifold, VTK's orientation pass cannot walk
    across it, and it returns having spread the disagreement rather than
    settled it. Measured on a decimated shell, 2 reversed edges went in and
    3 came out. So the flap is dropped first, and only genuine winding is
    reoriented.
    """
    from geoml.data.meshes import _rebuilt_as

    box = _build(pv.Box())
    points = np.asarray(box.coordinates)
    triangles = np.asarray(box.triangles)

    # a flap standing clear of the box: two coincident triangles, sharing
    # every edge with each other and nothing else
    flap = np.array([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0], [0.0, 1.0, 9.0]])
    offset = len(points)
    dirty_points = np.concatenate([points, flap])
    dirty = np.concatenate([triangles,
                            [[offset, offset + 1, offset + 2]],
                            [[offset, offset + 1, offset + 2]]])
    assert geoml.math.geometry.reversed_edges(dirty_points, dirty) > 0

    faces = np.concatenate(
        [np.full([len(dirty), 1], 3, int), dirty], axis=1).ravel()
    rebuilt = _rebuilt_as(Solid3D, pv.PolyData(dirty_points, faces))

    assert isinstance(rebuilt, Solid3D)
    assert np.isclose(rebuilt.volume, box.volume, rtol=1e-9)


def test_simplify_never_raises_out_of_a_workflow():
    """Decimation can break a body however clean its input -- collapse a
    thin feature into a membrane -- and `simplify` holds a remedy no other
    rebuild has: cutting less, whose error only shrinks, down to not
    cutting at all. So it retries gently and, as the last resort, returns
    the mesh as it came with a warning, rather than raising out of
    `get_contour(..., simplify=)` -- which it was reported doing, twice,
    from the same real model.
    """
    shell = _contoured_shell()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for budget in (0.3, 0.5, 1.0, 3.0):
            slim = shell.simplify(budget)
            assert isinstance(slim, Solid3D)
            assert len(slim.triangles) <= len(shell.triangles)


def test_simplify_keeps_a_terrain_a_terrain():
    ground = _terrain(height=5.0, slope=0.3, n=17)
    slim = ground.simplify(0.5)

    assert isinstance(slim, DTM3D)
    assert len(slim.triangles) < 0.5 * len(ground.triangles)


def test_simplify_wants_a_positive_error():
    shell = _contoured_shell()
    with pytest.raises(ValueError, match="must be positive"):
        shell.simplify(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        shell.simplify(-1.0)


def test_smooth_holds_the_volume():
    """Taubin's filter does not shrink what it smooths -- measured 0.04%
    volume drift on a contoured shell -- and the kind is kept. Accuracy
    against the true level set is another matter, and the docstring says
    so: contour with `supersample` for that."""
    shell = _contoured_shell()
    smoothed = shell.smooth()

    assert isinstance(smoothed, Solid3D)
    assert np.isclose(smoothed.volume, shell.volume, rtol=0.01)


def test_get_contour_simplifies_on_the_way_out():
    grid = _grid_with_a_ball([9.5, 9.5, 9.5])
    full = grid.variables["v"].measurements.get_contour(6.0)
    slim = grid.variables["v"].measurements.get_contour(6.0, simplify=0.5)

    assert isinstance(slim, Solid3D)
    assert len(slim.triangles) < 0.5 * len(full.triangles)
    assert np.isclose(slim.volume, full.volume, rtol=0.05)


def test_a_flipped_triangle_is_repaired_not_refused():
    """Decimation can leave a triangle wound against its neighbours; that is
    bookkeeping, not shape, so the rebuild repairs the winding and carries
    on -- while a genuinely open mesh still refuses, since closing a hole
    would be inventing geometry."""
    from geoml.data.meshes import _rebuilt_as

    ball = pv.Sphere(radius=4.0, theta_resolution=30,
                     phi_resolution=30).triangulate()
    faces = ball.faces.reshape(-1, 4).copy()
    faces[10, 1:] = faces[10, 3:0:-1]
    flipped = pv.PolyData(ball.points, faces.ravel())

    body = _rebuilt_as(Solid3D, flipped)
    assert isinstance(body, Solid3D)
    assert body.consistent and body.closed

    holed = pv.PolyData(ball.points, np.delete(faces, 10, axis=0).ravel())
    with pytest.raises(NotClosedError):
        _rebuilt_as(Solid3D, holed)


def test_clip_meshes_cuts_everything_below_in_one_pass():
    """The batch form: one extruded ground serves every cut, each mesh
    coming back its own kind and matching the one-by-one answer."""
    topo = _terrain(height=0.0, slope=0.5, cls=DTM3D)
    ball = _build(pv.Sphere(radius=4.0, center=(0, 0, 0)), Solid3D)
    sheet = _terrain(height=0.0, extent=10.0, cls=Surface3D, n=41)

    below = topo.clip_meshes([ball, sheet])

    assert isinstance(below[0], Solid3D)
    assert below[0].closed
    assert isinstance(below[1], Surface3D)
    assert np.isclose(below[0].volume, ball.intersection(topo).volume,
                      rtol=0.01)
    assert np.isclose(below[1].area, sheet.intersection(topo).area,
                      rtol=0.01)


def test_clip_meshes_refuses_what_it_cannot_cut():
    topo = _terrain(height=0.0, cls=DTM3D)
    shapeless = Mesh3D(np.zeros([0, 3]), np.zeros([0, 3], dtype=int),
                       np.zeros([0, 3]))
    with pytest.raises(MeshTypeError, match="bodies and sheets"):
        topo.clip_meshes([shapeless])
