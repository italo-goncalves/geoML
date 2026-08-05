"""Surface I/O: the DXF round trip, and the contour export.

`Surface3D.export_dxf` writes one MESH entity, whose vertex list survives
`from_dxf` untouched. The 3DFACE and POLYFACE entities other software writes
repeat the coordinates of every shared corner instead, and are welded back on
the way in; normals are computed there too, since a DXF file carries none.
"""
import numpy as np
import pytest
import ezdxf

import geoml


def _pyramid():
    """Four triangles around a shared apex, meeting pairwise along an edge."""
    points = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    return geoml.data.Surface3D(
        points, triangles, geoml.geometry.vertex_normals(points, triangles))


def _square():
    """Two triangles in the z = 0 plane, wound counter-clockwise."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    return points, triangles


def _sphere_grid():
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[12, 12, 12], step=[1, 1, 1])
    radius = np.linalg.norm(
        np.asarray(grid.coordinates) - np.array([[5.5, 5.5, 5.5]]), axis=1)
    grid.add_continuous_variable("v", radius)
    return grid


def test_mesh_roundtrip_is_exact(tmp_path):
    surface = _pyramid()
    path = str(tmp_path / "surface.dxf")
    surface.export_dxf(path)

    back = geoml.data.Surface3D.from_dxf(path)

    assert np.array_equal(np.asarray(back.coordinates),
                          np.asarray(surface.coordinates))
    assert np.array_equal(back.triangles, surface.triangles)


def test_export_writes_a_single_mesh_entity(tmp_path):
    path = str(tmp_path / "surface.dxf")
    _pyramid().export_dxf(path)

    model = ezdxf.readfile(path).modelspace()
    assert len(model.query("MESH")) == 1
    assert len(model.query("3DFACE")) == 0
    assert len(model.query("POLYLINE")) == 0


def test_offset_shifts_the_written_coordinates(tmp_path):
    surface = _pyramid()
    path = str(tmp_path / "surface.dxf")
    offset = np.array([[1000.0, 2000.0, 30.0]])
    surface.export_dxf(path, offset=offset[0])

    back = geoml.data.Surface3D.from_dxf(path)

    assert np.allclose(np.asarray(back.coordinates) - offset,
                       np.asarray(surface.coordinates))


def test_normals_are_computed_on_the_way_in(tmp_path):
    points, triangles = _square()
    surface = geoml.data.Surface3D(points, triangles, np.zeros_like(points))
    path = str(tmp_path / "flat.dxf")
    surface.export_dxf(path)

    back = geoml.data.Surface3D.from_dxf(path)

    # counter-clockwise in the z = 0 plane, so every vertex normal is +z --
    # whatever the surface was built with, the file never held it
    assert back.normals.shape == points.shape
    assert np.allclose(back.normals, np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(np.linalg.norm(back.normals, axis=1), 1.0)


def test_3dface_entities_are_welded(tmp_path):
    points, triangles = _square()
    path = str(tmp_path / "faces.dxf")
    document = ezdxf.new()
    model = document.modelspace()
    for triangle in triangles:
        corners = points[triangle]
        model.add_3dface([corners[0], corners[1], corners[2], corners[2]])
    document.saveas(path)

    back = geoml.data.Surface3D.from_dxf(path)

    # six corners were written; the two the triangles share are one vertex
    assert back.n_data == 4
    assert back.triangles.shape == (2, 3)
    assert np.allclose(np.sort(np.asarray(back.coordinates), axis=0),
                       np.sort(points, axis=0))


def test_polyface_mesh_is_welded(tmp_path):
    points, triangles = _square()
    path = str(tmp_path / "polyface.dxf")
    document = ezdxf.new()
    polyface = document.modelspace().add_polyface()
    for triangle in triangles:
        polyface.append_face(points[triangle].tolist())
    document.saveas(path)

    back = geoml.data.Surface3D.from_dxf(path)

    assert back.n_data == 4
    assert back.triangles.shape == (2, 3)


def test_a_quad_face_is_triangulated(tmp_path):
    points, _ = _square()
    path = str(tmp_path / "quad.dxf")
    document = ezdxf.new()
    mesh = document.modelspace().add_mesh()
    with mesh.edit_data() as mesh_data:
        mesh_data.vertices = points.tolist()
        mesh_data.faces = [[0, 1, 2, 3]]
    document.saveas(path)

    back = geoml.data.Surface3D.from_dxf(path)

    assert back.n_data == 4
    assert np.array_equal(back.triangles, np.array([[0, 1, 2], [0, 2, 3]]))


def test_several_meshes_are_concatenated(tmp_path):
    points, triangles = _square()
    path = str(tmp_path / "two_bodies.dxf")
    document = ezdxf.new()
    model = document.modelspace()
    for shift in (0.0, 10.0):
        mesh = model.add_mesh()
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = (points + np.array([[shift, 0, 0]])).tolist()
            mesh_data.faces = triangles.tolist()
    document.saveas(path)

    back = geoml.data.Surface3D.from_dxf(path)

    assert back.n_data == 8
    assert back.triangles.shape == (4, 3)
    # the second body indexes its own vertices, not the first body's
    assert back.triangles[2:].min() == 4
    assert back.triangles.max() == 7


def test_a_file_without_a_surface_raises(tmp_path):
    path = str(tmp_path / "line_only.dxf")
    document = ezdxf.new()
    document.modelspace().add_line((0, 0, 0), (1, 1, 1))
    document.saveas(path)

    with pytest.raises(ValueError, match="no triangulated surface"):
        geoml.data.Surface3D.from_dxf(path)


def test_a_contour_travels_through_dxf(tmp_path):
    grid = _sphere_grid()
    surface = grid.variables["v"].measurements.get_contour(3.0)
    assert surface.n_data > 0

    path = str(tmp_path / "contour.dxf")
    surface.export_dxf(path)
    back = geoml.data.Surface3D.from_dxf(path)

    assert np.array_equal(np.asarray(back.coordinates),
                          np.asarray(surface.coordinates))
    assert np.array_equal(back.triangles, surface.triangles)


def test_export_contour_writes_the_triangulation(tmp_path):
    """It used to unpack the Surface3D get_contour returns, and so raised."""
    grid = _sphere_grid()
    surface = grid.variables["v"].measurements.get_contour(3.0)
    n_points, n_triangles = surface.n_data, surface.triangles.shape[0]

    path = tmp_path / "contour.txt"
    grid.variables["v"].measurements.export_contour(3.0, str(path))

    lines = path.read_text().splitlines()
    assert lines[0].split() == [str(n_points), str(n_triangles)]
    assert len(lines) == 1 + n_points + n_triangles
