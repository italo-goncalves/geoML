"""Closing a contour, terrains, and cutting one kind of mesh with another.

`get_contour(close=...)` pads the cube so a surface running out of the grid
closes inside it. `DTM3D` is a sheet that promises never to fold over, which
is what lets a body be divided into what lies under it and what lies over it.
Cutting keeps the caller's kind: a sheet cut by a body is a sheet, a body cut
by a sheet is a body.
"""
import numpy as np
import pandas as pd
import pytest
import pyvista as pv

import geoml
from geoml.data import (Mesh3D, Surface3D, Solid3D, DTM3D, mesh3d,
                        MeshTypeError, NotSingleValuedError, NotClosedError)


def _build(mesh, cls=None):
    mesh = mesh.triangulate()
    points = np.asarray(mesh.points, dtype=float)
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    normals = geoml.geometry.vertex_normals(points, triangles)
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
               geoml.geometry.vertex_normals(points, triangles))


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
    return points, triangles, geoml.geometry.vertex_normals(points, triangles)


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
              geoml.geometry.vertex_normals(points, triangles))


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


def test_a_sheet_can_only_be_cut_by_a_body():
    sheet = _terrain(cls=Surface3D)
    with pytest.raises(MeshTypeError, match="only be cut by a Solid3D"):
        sheet.intersection(_terrain(cls=Surface3D))


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
