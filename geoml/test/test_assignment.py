"""Assigning a surface to a container's locations.

`assign_from_surface` compares each location with the sheet's elevation over
it; `assign_from_solid` asks whether it falls inside a closed body. Both write
a metadata column. A grid answers the first once per column of cells rather
than once per cell — the shortcut is checked here against the general path —
and a block model can measure the blocks the surface cuts through.
"""
import numpy as np
import pandas as pd
import pytest
import pyvista as pv

import geoml


def _sheet(z0=5.0, slope_x=0.0, slope_y=0.0, extent=10.0):
    """An open sheet over [0, extent]^2, at z = z0 + slope . (x, y)."""
    xy = np.array([[0.0, 0.0], [extent, 0.0], [extent, extent], [0.0, extent]])
    height = z0 + slope_x * xy[:, 0] + slope_y * xy[:, 1]
    points = np.column_stack([xy, height])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    return geoml.data.Surface3D(
        points, triangles, geoml.math.geometry.vertex_normals(points, triangles))


def _sheet_from(xy, triangles, height=5.0):
    """A flat sheet at `height`, over whatever shape the triangles make."""
    points = np.column_stack([np.asarray(xy, dtype=float),
                              np.full(len(xy), float(height))])
    triangles = np.asarray(triangles)
    return geoml.data.Surface3D(
        points, triangles, geoml.math.geometry.vertex_normals(points, triangles))


def _box(bounds=(2.0, 8.0, 2.0, 8.0, 2.0, 8.0)):
    """A watertight body: every edge belongs to two triangles."""
    mesh = pv.Box(bounds=bounds).triangulate()
    points = np.asarray(mesh.points, dtype=float)
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    return geoml.data.Solid3D(
        points, triangles, geoml.math.geometry.vertex_normals(points, triangles))


def _points(coordinates):
    labels = ["X", "Y", "Z"]
    return geoml.data.PointData(
        pd.DataFrame(np.asarray(coordinates, dtype=float), columns=labels),
        labels)


# -- the sheet ------------------------------------------------------------ #

def test_locations_are_split_by_the_sheet():
    data = _points([[5, 5, 1], [5, 5, 9], [5, 5, 5.5]])
    data.assign_from_surface(_sheet(z0=5.0), "ground")

    assert list(data.get_metadata("ground")) == ["below", "above", "above"]


def test_a_location_beyond_the_sheet_is_left_empty():
    data = _points([[5, 5, 1], [50, 50, 1]])
    data.assign_from_surface(_sheet(z0=5.0), "ground")

    assert list(data.get_metadata("ground")) == ["below", ""]


def test_the_footprint_follows_the_triangles_not_their_hull():
    # an L, missing the [5, 10] x [5, 10] quadrant that its hull still covers
    ell = _sheet_from([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)],
                      [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)])
    data = _points([[7.5, 2.5, 1], [7.5, 7.5, 1], [2.5, 7.5, 1]])

    data.assign_from_surface(ell, "ground")

    assert list(data.get_metadata("ground")) == ["below", "", "below"]


def test_a_hole_in_the_sheet_is_a_hole_in_the_answer():
    ring = _sheet_from(
        [(0, 0), (10, 0), (10, 10), (0, 10),
         (4, 4), (6, 4), (6, 6), (4, 6)],
        [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    data = _points([[1, 1, 1], [5, 5, 1], [9, 9, 1]])

    data.assign_from_surface(ring, "ground")

    assert list(data.get_metadata("ground")) == ["below", "", "below"]


def test_a_tilted_sheet_is_followed():
    # z = x, so the answer differs at the two ends of the same height band
    data = _points([[2, 5, 1], [2, 5, 3], [8, 5, 7], [8, 5, 9]])
    data.assign_from_surface(_sheet(z0=0.0, slope_x=1.0), "ground")

    assert list(data.get_metadata("ground")) == \
        ["below", "above", "below", "above"]


def test_the_sides_can_be_renamed():
    data = _points([[5, 5, 1], [5, 5, 9]])
    data.assign_from_surface(_sheet(z0=5.0), "ground", labels=("air", "rock"))

    assert list(data.get_metadata("ground")) == ["rock", "air"]


def test_a_closed_body_is_refused_as_a_sheet():
    data = _points([[5, 5, 5]])
    with pytest.raises(ValueError, match="closed"):
        data.assign_from_surface(_box(), "ground")


def test_a_two_dimensional_container_is_refused():
    grid = geoml.data.Grid2D(start=[0, 0], n=[3, 3], step=[1, 1])
    with pytest.raises(geoml.data.DimensionMismatchError):
        grid.assign_from_surface(_sheet(), "ground")


# -- the solid ------------------------------------------------------------ #

def test_locations_inside_a_body():
    data = _points([[5, 5, 5], [0, 0, 0], [2.5, 5, 5], [5, 5, 9]])
    data.assign_from_solid(_box(), "ore")

    assert list(data.get_metadata("ore")) == \
        ["inside", "outside", "inside", "outside"]


def test_a_sheet_is_refused_as_a_solid():
    data = _points([[5, 5, 5]])
    with pytest.raises(ValueError, match="not closed"):
        data.assign_from_solid(_sheet(), "ore")


def _from_pyvista(mesh):
    mesh = mesh.triangulate()
    points = np.asarray(mesh.points, dtype=float)
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    return geoml.data.mesh3d(
        points, triangles, geoml.math.geometry.vertex_normals(points, triangles))


def test_a_body_closed_in_space_counts_as_closed():
    # pyvista's own Cylinder indexes the caps' corners apart from the tube's,
    # so every seam looks like a boundary until the vertices are welded
    cylinder = _from_pyvista(pv.Cylinder(radius=1.0, height=2.0, capping=True))
    assert cylinder.n_data > np.unique(
        np.asarray(cylinder.coordinates), axis=0).shape[0]

    data = _points([[0, 0, 0], [5, 0, 0]])
    data.assign_from_solid(cylinder, "ore")

    assert list(data.get_metadata("ore")) == ["inside", "outside"]


def test_a_cylinder_with_one_end_open_is_refused():
    tube = pv.Cylinder(radius=1.0, height=2.0, capping=False).triangulate()
    disc = pv.Disc(center=(-1.0, 0, 0), inner=0.0, outer=1.0,
                   normal=(1, 0, 0), r_res=1, c_res=tube.n_points // 2)
    one_open = _from_pyvista(tube.merge(disc))

    data = _points([[0, 0, 0]])
    with pytest.raises(ValueError, match="not closed"):
        data.assign_from_solid(one_open, "ore")


def test_triangles_disagreeing_which_way_is_out_are_refused():
    box = _box()
    triangles = np.asarray(box.triangles).copy()
    triangles[0] = triangles[0][::-1]
    points = np.asarray(box.coordinates)
    broken = geoml.data.Mesh3D(
        points, triangles, geoml.math.geometry.vertex_normals(points, triangles))
    assert broken.closed and not broken.consistent

    data = _points([[5, 5, 5]])
    with pytest.raises(ValueError, match="disagree about which way is out"):
        data.assign_from_solid(broken, "ore")


def test_a_body_wound_inwards_is_turned_round():
    box = _box()
    points = np.asarray(box.coordinates)
    inside_out = np.asarray(box.triangles)[:, ::-1]
    # closed and consistent, only facing the wrong way
    assert geoml.math.geometry.signed_volume(points, inside_out) < 0

    turned = geoml.data.Solid3D(
        points, inside_out, geoml.math.geometry.vertex_normals(points, inside_out))
    assert turned.volume > 0

    data = _points([[5, 5, 5], [0, 0, 0]])
    data.assign_from_solid(turned, "ore")

    assert list(data.get_metadata("ore")) == ["inside", "outside"]


# -- the grid shortcut ---------------------------------------------------- #

def test_the_grid_shortcut_matches_the_general_path():
    surface = _sheet(z0=5.0, slope_x=0.3, slope_y=-0.2)
    # reaches past the sheet's footprint, so empty answers are compared too
    grid = geoml.data.Grid3D(start=[0.5, 0.5, 0.5], n=[9, 7, 4],
                             step=[1.4, 2.1, 2.7])

    grid.assign_from_surface(surface, "fast")
    geoml.data._SpatialData.assign_from_surface(grid, surface, "slow")

    fast = grid.get_metadata("fast")
    assert np.array_equal(fast, grid.get_metadata("slow"))
    assert set(fast) == {"above", "below", ""}


def test_a_rotated_grid_takes_the_general_path():
    surface = _sheet(z0=5.0, extent=100.0)
    # rotated well inside the sheet's footprint, so nothing falls off its edge
    grid = geoml.data.RotatedGrid3D(start=[20, 20, 1], n=[4, 4, 4],
                                    step=[2, 2, 2], azimuth=30.0, dip=20.0)
    assert grid._transform is not None

    grid.assign_from_surface(surface, "ground")

    height = np.asarray(grid.coordinates)[:, 2]
    expected = np.where(height < 5.0, "below", "above")
    assert np.array_equal(grid.get_metadata("ground"), expected)
    assert set(grid.get_metadata("ground")) == {"above", "below"}


# -- the blocks ----------------------------------------------------------- #

def _column_of_blocks(discretization):
    """One column of three blocks, centred at z = 2, 6 and 10."""
    return geoml.data.Blocks3D(start=[5, 5, 2], n=[1, 1, 3], step=[1, 1, 4],
                               discretization=discretization)


def test_the_block_flag_follows_the_centre():
    blocks = _column_of_blocks([1, 1, 4])
    # sub-blocks sit at the centre +/- 1.5 and +/- 0.5
    blocks.assign_from_surface(_sheet(z0=6.2), "ground", fraction="share")

    assert list(blocks.get_metadata("ground")) == ["below", "below", "above"]
    assert np.allclose(blocks.get_metadata("share"), [1.0, 0.5, 0.0])


def _blocks_past_the_sheet():
    """Centres at x = 2, 8 and 14; the sheet only spans x in [0, 10]."""
    return geoml.data.Blocks3D(start=[2, 2, 2], n=[3, 1, 1], step=[6, 1, 4],
                               discretization=[3, 1, 4])


def test_a_block_the_sheet_never_reaches_is_not_measured():
    blocks = _blocks_past_the_sheet()
    blocks.assign_from_surface(_sheet(z0=6.2), "ground", fraction="share")

    assert list(blocks.get_metadata("ground")) == ["below", "below", ""]
    assert np.allclose(blocks.get_metadata("share"), [1.0, 1.0, np.nan],
                       equal_nan=True)


def test_an_uncovered_block_can_be_counted_as_nothing():
    blocks = _blocks_past_the_sheet()
    blocks.assign_from_surface(_sheet(z0=6.2), "ground", fraction="share",
                               uncovered=0.0)

    assert list(blocks.get_metadata("ground")) == ["below", "below", ""]
    assert np.allclose(blocks.get_metadata("share"), [1.0, 1.0, 0.0])


def test_a_surface_can_be_required_to_cover_everything():
    blocks = _blocks_past_the_sheet()
    with pytest.raises(ValueError, match="does not reach 1 of the 3"):
        blocks.assign_from_surface(_sheet(z0=6.2), "ground", uncovered="raise")

    assert "ground" not in blocks.metadata


def test_covering_everything_passes_the_requirement():
    data = _points([[5, 5, 1], [4, 4, 9]])
    data.assign_from_surface(_sheet(z0=5.0), "ground", uncovered="raise")

    assert list(data.get_metadata("ground")) == ["below", "above"]


def test_the_requirement_reaches_the_grid_shortcut():
    grid = geoml.data.Grid3D(start=[5, 5, 1], n=[2, 2, 2], step=[20, 1, 1])
    assert grid._transform is None

    with pytest.raises(ValueError, match="does not reach"):
        grid.assign_from_surface(_sheet(z0=5.0), "ground", uncovered="raise")


def test_an_unknown_uncovered_rule_is_refused():
    data = _points([[5, 5, 1]])
    with pytest.raises(ValueError, match="uncovered takes"):
        data.assign_from_surface(_sheet(), "ground", uncovered="skip")


def test_the_fraction_is_optional():
    blocks = _column_of_blocks([1, 1, 4])
    blocks.assign_from_surface(_sheet(z0=6.2), "ground")

    assert "ground" in blocks.metadata
    assert "share" not in blocks.metadata


def test_the_solid_fraction_measures_the_blocks_it_cuts():
    blocks = _column_of_blocks([1, 1, 4])
    # the body reaches up to z = 8, cutting the block centred at 10 in part
    blocks.assign_from_solid(_box(bounds=(2, 8, 2, 8, 2, 8)), "ore",
                             fraction="share")

    assert list(blocks.get_metadata("ore")) == ["inside", "inside", "outside"]
    # centre 2: sub-blocks at 0.5, 1.5, 2.5, 3.5 -> two inside
    # centre 6: 4.5, 5.5, 6.5, 7.5 -> all four inside
    # centre 10: 8.5 ... 11.5 -> none
    assert np.allclose(blocks.get_metadata("share"), [0.5, 1.0, 0.0])


def test_the_fraction_survives_chunking():
    # 1000 sub-blocks a block puts the chunk at 1000 blocks, so 1331 of them
    # is more than one pass; the loop must not scramble them
    blocks = geoml.data.Blocks3D(start=[1, 1, 1], n=[11, 11, 11],
                                 step=[1, 1, 1], discretization=[10, 10, 10])
    surface = _sheet(z0=6.4, extent=30.0)

    blocks.assign_from_surface(surface, "ground", fraction="share")

    coordinates, _ = blocks.get_batched_coordinates(
        np.arange(blocks.n_data))
    below = coordinates[:, 2] < 6.4
    expected = below.reshape([blocks.n_data, 1000]).mean(axis=1)
    assert np.allclose(blocks.get_metadata("share"), expected)


# -- what the column is for ----------------------------------------------- #

def test_a_point_set_can_be_cut_down_by_the_column():
    data = _points([[5, 5, 1], [5, 5, 9], [5, 5, 2]])
    data.add_continuous_variable("v", [10.0, 20.0, 30.0])
    data.assign_from_surface(_sheet(z0=5.0), "ground")

    kept = data[data.get_metadata("ground") == "below"]

    assert kept.n_data == 2
    assert np.allclose(
        np.asarray(kept.variables["v"].measurements.values).ravel(),
        [10.0, 30.0])
    assert list(kept.get_metadata("ground")) == ["below", "below"]


def test_the_column_reaches_the_data_frame():
    data = _points([[5, 5, 1], [5, 5, 9]])
    data.assign_from_surface(_sheet(z0=5.0), "ground")

    assert list(data.as_data_frame()["ground"]) == ["below", "above"]
