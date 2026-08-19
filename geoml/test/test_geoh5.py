"""The geoh5 interchange: surfaces and points, both ways.

Skipped whole when `geoh5py` is not installed — the dependency is an
optional extra (`geoml[geoh5]`), and the package must work without it.
Every test writes into its own temporary workspace; `geoh5py` can create
files from scratch, so nothing binary is bundled.
"""
import numpy as np
import pandas as pd
import pytest

geoh5py = pytest.importorskip("geoh5py")

import geoml
import geoml.data.geoh5 as geoh5io

COLS = ["X", "Y", "Z"]


def _tetrahedron():
    """The smallest closed body, wound outward."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    return geoml.data.Solid3D(
        points, triangles,
        geoml.math.geometry.vertex_normals(points, triangles))


def _sheet():
    """Two triangles that close nothing."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.5], [1.0, 1.0, 0.5]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    return geoml.data.Surface3D(
        points, triangles,
        geoml.math.geometry.vertex_normals(points, triangles))


def _point_data(n=12, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.uniform(0, 100, (n, 3)), columns=COLS)
    return geoml.data.PointData(frame, COLS), rng


# --------------------------------------------------------------------------- #
# surfaces
# --------------------------------------------------------------------------- #
def test_a_body_comes_back_a_body(tmp_path):
    body = _tetrahedron()
    file = str(tmp_path / "model.geoh5")
    body.to_geoh5(file, name="tetra")

    back = geoml.data.Mesh3D.from_geoh5(file)

    assert type(back).__name__ == "Solid3D"
    assert np.allclose(back.coordinates, body.coordinates)
    assert np.array_equal(back.triangles, body.triangles)
    assert np.isclose(back.volume, body.volume)


def test_a_sheet_comes_back_a_sheet(tmp_path):
    sheet = _sheet()
    file = str(tmp_path / "model.geoh5")
    sheet.to_geoh5(file, name="sheet")

    back = geoml.data.Mesh3D.from_geoh5(file)
    assert type(back).__name__ == "Surface3D"
    assert np.allclose(back.coordinates, sheet.coordinates)


def test_a_workspace_accumulates_and_names_decide(tmp_path):
    file = str(tmp_path / "model.geoh5")
    _tetrahedron().to_geoh5(file, name="ore")
    _sheet().to_geoh5(file, name="topo")

    listing = geoh5io.contents(file)
    assert listing == {"ore": "Surface", "topo": "Surface"}

    ore = geoml.data.Mesh3D.from_geoh5(file, name="ore")
    assert type(ore).__name__ == "Solid3D"

    # two surfaces and no name is a question, not a guess
    with pytest.raises(ValueError, match="pass `name=`"):
        geoml.data.Mesh3D.from_geoh5(file)


def test_a_missing_name_lists_what_the_file_holds(tmp_path):
    file = str(tmp_path / "model.geoh5")
    _tetrahedron().to_geoh5(file, name="ore")

    with pytest.raises(ValueError, match="'ore'"):
        geoml.data.Mesh3D.from_geoh5(file, name="waste")


def test_a_typo_in_the_path_is_not_an_empty_workspace(tmp_path):
    with pytest.raises(FileNotFoundError):
        geoml.data.Mesh3D.from_geoh5(str(tmp_path / "nowhere.geoh5"))


# --------------------------------------------------------------------------- #
# points
# --------------------------------------------------------------------------- #
def test_points_carry_their_columns_both_ways(tmp_path):
    point, rng = _point_data()
    grade = rng.uniform(0, 5, 12)
    grade[3] = np.nan
    point.add_continuous_variable("grade", grade)
    rocks = np.array(["granite", "basalt"] * 6, dtype=object)
    rocks[5] = None                       # not measured here
    point.add_categorical_variable("rock", labels=["basalt", "granite"],
                                   measurements=rocks)
    point.add_metadata("weight", rng.uniform(0, 1, 12))

    file = str(tmp_path / "model.geoh5")
    point.to_geoh5(file, name="samples")
    back = geoml.data.PointData.from_geoh5(file)

    assert np.allclose(back.coordinates, point.coordinates)
    measured = back.variables["grade - measurements"] \
        .measurements.values.to_numpy()
    assert np.allclose(measured, grade, equal_nan=True)

    rock = back.variables["rock - measurements_a"]
    assert list(rock.labels) == ["basalt", "granite"]
    values = rock.measurements_a.to_numpy()
    assert values[0] == "granite" and values[1] == "basalt"
    assert values[5] == ""                # missing stayed missing

    # metadata came back as a column too -- a foreign reader cannot know
    # it was metadata, so it reads as a variable named what the file says
    weight = back.variables["weight"].measurements.values.to_numpy()
    assert np.allclose(weight, point.get_metadata("weight"))


def test_the_column_names_map_back_to_paths(tmp_path):
    point, rng = _point_data()
    point.add_continuous_variable("grade", rng.uniform(0, 5, 12))

    file = str(tmp_path / "model.geoh5")
    point.to_geoh5(file, name="samples")

    import json
    workspace = geoh5py.workspace.Workspace(file)
    try:
        samples = workspace.get_entity("samples")[0]
        table = json.loads(samples.metadata["geoml_paths"])
    finally:
        workspace.close()
    assert table["grade - measurements"] == "grade/measurements"


def test_a_flat_container_says_it_cannot_go(tmp_path):
    frame = pd.DataFrame(np.zeros((5, 2)), columns=["X", "Y"])
    flat = geoml.data.PointData(frame, ["X", "Y"])
    with pytest.raises(ValueError, match="3-dimensional"):
        flat.to_geoh5(str(tmp_path / "model.geoh5"))


# --------------------------------------------------------------------------- #
# block models
# --------------------------------------------------------------------------- #
def _refined_blocks(seed=3):
    """A mixed-level model carrying a grade and a rock type."""
    rng = np.random.default_rng(seed)
    blocks = geoml.data.BlockSet3D(start=[10.0, 20.0, 5.0], n=[4, 3, 2],
                                   step=[8.0, 8.0, 4.0], max_levels=2)
    blocks = blocks.split([0, 5])
    blocks = blocks.split(np.arange(blocks.n_data) >= blocks.n_data - 4)
    grade = rng.uniform(0, 5, blocks.n_data)
    grade[2] = np.nan
    blocks.add_continuous_variable("grade", grade)
    blocks.add_categorical_variable(
        "rock", labels=["basalt", "granite"],
        measurements=np.asarray(["granite", "basalt"]
                                * (blocks.n_data // 2 + 1),
                                dtype=object)[:blocks.n_data])
    return blocks, grade


def test_a_refined_model_round_trips_whole(tmp_path):
    blocks, grade = _refined_blocks()
    file = str(tmp_path / "model.geoh5")
    blocks.to_geoh5(file, name="bm")

    back = geoml.data.BlockSet3D.from_geoh5(file)

    assert type(back) is geoml.data.BlockSet3D
    assert back.n_data == blocks.n_data
    assert back.max_levels == blocks.max_levels
    assert np.allclose(back.coordinates, blocks.coordinates)
    assert np.allclose(back.block_size, blocks.block_size)
    assert np.array_equal(back.level, blocks.level)
    values = back.variables["grade - measurements"] \
        .measurements.values.to_numpy()
    assert np.allclose(values, grade, equal_nan=True)
    rock = back.variables["rock - measurements_a"].measurements_a.to_numpy()
    assert rock[0] == "granite" and rock[1] == "basalt"
    # the box did not grow: the padding is in the counts, never in cells
    assert np.allclose(np.ravel(back.bounding_box.min),
                       np.ravel(blocks.bounding_box.min))
    assert np.allclose(np.ravel(back.bounding_box.max),
                       np.ravel(blocks.bounding_box.max))
    assert "imported" not in back.metadata


def test_a_rotated_model_keeps_its_place_in_the_world(tmp_path):
    blocks = geoml.data.RotatedBlockSet3D(
        start=[500.0, 800.0, 50.0], n=[3, 2, 2], step=[10.0, 10.0, 5.0],
        azimuth=35.0, max_levels=1)
    blocks = blocks.split([1, 4])
    file = str(tmp_path / "model.geoh5")
    blocks.to_geoh5(file, name="bm")

    back = geoml.data.BlockSet3D.from_geoh5(file)

    assert type(back) is geoml.data.RotatedBlockSet3D
    assert np.isclose(back.azimuth, 35.0)
    assert np.allclose(back.coordinates, blocks.coordinates)
    assert np.allclose(back.block_size, blocks.block_size)


def test_a_dipping_model_says_geoh5_cannot_hold_it(tmp_path):
    blocks = geoml.data.RotatedBlockSet3D(
        start=[0.0, 0.0, 0.0], n=[2, 2, 2], step=[1.0, 1.0, 1.0],
        azimuth=10.0, dip=20.0, max_levels=1)
    with pytest.raises(ValueError, match="vertical axis"):
        blocks.to_geoh5(str(tmp_path / "model.geoh5"))


def test_an_uneven_discretization_is_refused_not_resampled(tmp_path):
    blocks = geoml.data.BlockSet3D(start=[0.0, 0.0, 0.0], n=[2, 2, 2],
                                   step=[2.0, 2.0, 1.0],
                                   discretization=(2, 2, 1), max_levels=1)
    with pytest.raises(ValueError, match="2x2x2"):
        blocks.to_geoh5(str(tmp_path / "model.geoh5"))


def _foreign_octree(tmp_path, cells, u_cell_size=5.0, w_cell_size=2.0,
                    rotation=0.0, counts=(8, 8, 8)):
    file = str(tmp_path / "foreign.geoh5")
    workspace = geoh5py.workspace.Workspace.create(file)
    octree_cells = np.array(cells, dtype=[("I", "<i4"), ("J", "<i4"),
                                          ("K", "<i4"), ("NCells", "<i4")])
    tree = geoh5py.objects.Octree.create(
        workspace, origin=[100.0, 200.0, 50.0],
        u_count=counts[0], v_count=counts[1], w_count=counts[2],
        u_cell_size=u_cell_size, v_cell_size=4.0, w_cell_size=w_cell_size,
        rotation=rotation, octree_cells=octree_cells, name="foreign")
    tree.add_data({"grade": {
        "values": np.arange(len(cells), dtype=float),
        "association": "CELL"}})
    workspace.close()
    return file


def test_a_partial_foreign_octree_is_filled_and_marked(tmp_path):
    # a 2-cube and three 1-cubes: the rest of their 4x4x4 corner and
    # nothing beyond it should be filled in, unvalued
    cells = [(0, 0, 0, 2), (2, 0, 0, 1), (2, 1, 0, 1), (3, 3, 3, 1)]
    file = _foreign_octree(tmp_path, cells)

    back = geoml.data.BlockSet3D.from_geoh5(file)

    covered = np.prod(back.block_size, axis=1).sum()
    assert np.isclose(covered, np.prod(4 * np.array([5.0, 4.0, 2.0])))
    imported = np.asarray(back.metadata["imported"].values)
    assert imported.sum() == 4 and len(imported) == back.n_data
    grade = back.variables["grade"].measurements.values.to_numpy()
    # file cells keep their values in file order; the fill holds nothing
    assert np.allclose(grade[:4], np.arange(4.0))
    assert np.all(np.isnan(grade[4:]))


def test_an_upside_down_octree_lands_the_right_way_up(tmp_path):
    file = _foreign_octree(tmp_path, [(0, 0, 0, 8)], w_cell_size=-2.0)

    back = geoml.data.BlockSet3D.from_geoh5(file)

    assert back.n_data == 1
    # origin was the TOP at z=50, eight cells of 2 running down
    assert np.allclose(back.coordinates[0],
                       [100 + 20.0, 200 + 16.0, 50 - 8.0])
    assert np.allclose(back.block_size[0], [40.0, 32.0, 16.0])


def test_an_overlapping_foreign_octree_is_refused(tmp_path):
    file = _foreign_octree(tmp_path, [(0, 0, 0, 2), (1, 1, 1, 1)])
    with pytest.raises(ValueError, match="inside larger cells"):
        geoml.data.BlockSet3D.from_geoh5(file)


def test_a_w_flip_under_rotation_is_fine(tmp_path):
    # z is the rotation's own axis, so a downward w composes no reflection
    file = _foreign_octree(tmp_path, [(0, 0, 0, 8)], rotation=15.0,
                           w_cell_size=-2.0)
    assert geoml.data.BlockSet3D.from_geoh5(file).n_data == 1


def test_a_reflected_rotation_is_refused(tmp_path):
    file = _foreign_octree(tmp_path, [(0, 0, 0, 8)], u_cell_size=-5.0,
                           rotation=15.0)
    with pytest.raises(ValueError, match="reflection"):
        geoml.data.BlockSet3D.from_geoh5(file)


def test_surfaces_and_points_share_one_workspace(tmp_path):
    file = str(tmp_path / "model.geoh5")
    _tetrahedron().to_geoh5(file, name="ore")
    point, rng = _point_data(n=6)
    point.add_continuous_variable("grade", rng.uniform(0, 5, 6))
    point.to_geoh5(file, name="samples")

    listing = geoh5io.contents(file)
    assert listing == {"ore": "Surface", "samples": "Points"}
    assert type(geoml.data.Mesh3D.from_geoh5(file)).__name__ == "Solid3D"
    assert geoml.data.PointData.from_geoh5(file).n_data == 6


# --------------------------------------------------------------------------- #
# regular block models
# --------------------------------------------------------------------------- #
def test_a_uniform_model_round_trips_as_a_blockmodel(tmp_path):
    blocks = geoml.data.Blocks3D(start=[10.0, 20.0, 5.0], n=[3, 4, 2],
                                 step=[2.0, 3.0, 4.0])
    rng = np.random.default_rng(5)
    grade = rng.uniform(0, 5, blocks.n_data)
    blocks.add_continuous_variable("grade", grade)
    file = str(tmp_path / "model.geoh5")
    blocks.to_geoh5(file, name="bm")

    assert geoh5io.contents(file) == {"bm": "BlockModel"}
    back = geoml.data.Blocks3D.from_geoh5(file)

    assert type(back) is geoml.data.Blocks3D
    assert np.allclose(back.coordinates, blocks.coordinates)
    values = back.variables["grade - measurements"] \
        .measurements.values.to_numpy()
    # the ordering survived the u-fastest storage and came back geoML's
    assert np.allclose(values, grade)


def test_a_rotated_blockmodel_lands_where_geoh5py_says(tmp_path):
    file = str(tmp_path / "foreign.geoh5")
    workspace = geoh5py.workspace.Workspace.create(file)
    model = geoh5py.objects.BlockModel.create(
        workspace, origin=[100.0, 200.0, 50.0],
        u_cell_delimiters=np.array([0.0, 2.0, 4.0, 6.0]),
        v_cell_delimiters=np.array([0.0, 3.0, 6.0]),
        z_cell_delimiters=np.array([0.0, 4.0]),
        rotation=30.0, name="bm")
    model.add_data({"row": {"values": np.arange(6.0),
                            "association": "CELL"}})
    truth = np.asarray(model.centroids, dtype=float)
    workspace.close()

    back = geoml.data.Blocks3D.from_geoh5(file)

    assert type(back) is geoml.data.RotatedBlockSet3D
    assert back.max_levels == 0
    # same cells, in whatever order: match each centroid to its row
    ours = np.asarray(back.coordinates)
    values = back.variables["row"].measurements.values.to_numpy()
    for world, value in zip(truth, np.arange(6.0)):
        row = np.argmin(np.linalg.norm(ours - world, axis=1))
        assert np.allclose(ours[row], world)
        assert np.isclose(values[row], value)


def test_a_tartan_model_is_refused_with_its_axis_named(tmp_path):
    file = str(tmp_path / "foreign.geoh5")
    workspace = geoh5py.workspace.Workspace.create(file)
    geoh5py.objects.BlockModel.create(
        workspace, origin=[0.0, 0.0, 0.0],
        u_cell_delimiters=np.array([0.0, 1.0, 3.0]),   # 1 then 2: tartan
        v_cell_delimiters=np.array([0.0, 1.0]),
        z_cell_delimiters=np.array([0.0, 1.0]),
        rotation=0.0, name="bm")
    workspace.close()
    with pytest.raises(ValueError, match="tartan"):
        geoml.data.Blocks3D.from_geoh5(file)


# --------------------------------------------------------------------------- #
# drillholes
# --------------------------------------------------------------------------- #
def _foreign_drillholes(tmp_path):
    file = str(tmp_path / "holes.geoh5")
    workspace = geoh5py.workspace.Workspace.create(file)
    group = geoh5py.groups.DrillholeGroup.create(workspace, name="campaign")
    first = geoh5py.objects.Drillhole.create(
        workspace, collar=[100.0, 200.0, 50.0],
        surveys=np.array([[0.0, 0.0, -90.0], [10.0, 5.0, -85.0]]),
        parent=group, name="DH001")
    first.add_data({
        "Au": {"values": np.array([1.0, 2.0, 3.0]),
               "from-to": np.array([[0.0, 2.0], [2.0, 5.0], [5.0, 9.0]])},
        "rock": {"values": np.array([1, 2, 1], dtype=np.int32),
                 "from-to": np.array([[0.0, 2.0], [2.0, 5.0], [5.0, 9.0]]),
                 "value_map": {1: "granite", 2: "basalt"},
                 "type": "referenced"},
    }, property_group="assay")
    second = geoh5py.objects.Drillhole.create(
        workspace, collar=[110.0, 210.0, 51.0],
        surveys=np.array([[0.0, 90.0, -60.0]]),
        parent=group, name="DH002")
    second.add_data({
        "Au": {"values": np.array([4.0]),
               "from-to": np.array([[1.0, 6.0]])},
    }, property_group="assay")
    workspace.close()
    return file


def test_drillholes_come_back_as_a_database(tmp_path):
    file = _foreign_drillholes(tmp_path)

    holes = geoml.data.DrillholeData.from_geoh5(file, name="campaign")

    assert sorted(holes.collar.index) == ["DH001", "DH002"]
    assert np.allclose(holes.collar.loc["DH001", ["X", "Y", "Z"]],
                       [100.0, 200.0, 50.0])
    table = holes.intervals["assay"]
    assert len(table) == 4
    assert table.roles["Au"] == "grade"
    assert table.roles["rock"] == "categorical"
    # geoh5's -90 dip is straight down: the desurveyed trace must descend
    points = holes.as_point_data("assay")
    coordinates = np.asarray(points.coordinates)
    assert coordinates[:, 2].max() <= 50.0


def test_one_hole_can_be_read_alone(tmp_path):
    file = _foreign_drillholes(tmp_path)
    one = geoml.data.DrillholeData.from_geoh5(file, name="DH002")
    assert list(one.collar.index) == ["DH002"]


# --------------------------------------------------------------------------- #
# the workspace object
# --------------------------------------------------------------------------- #
def test_the_workspace_reads_like_a_dict_of_geoml_objects(tmp_path):
    file = str(tmp_path / "project.geoh5")
    blocks, _ = _refined_blocks()
    uniform = geoml.data.Blocks3D(start=[0.0, 0.0, 0.0], n=[2, 2, 2],
                                  step=[1.0, 1.0, 1.0])
    point, _ = _point_data(n=5)
    with geoh5io.Workspace(file) as project:
        _tetrahedron().to_geoh5(project, name="ore")
        point.to_geoh5(project, name="samples")
        blocks.to_geoh5(project, name="octree")
        uniform.to_geoh5(project, name="regular")

        assert type(project["ore"]).__name__ == "Solid3D"
        assert type(project["samples"]).__name__ == "PointData"
        assert type(project["octree"]).__name__ == "BlockSet3D"
        assert type(project["regular"]).__name__ == "Blocks3D"
        assert "ore" in project and "nothing" not in project
        assert len(project) == 4
        assert sorted(project) == ["octree", "ore", "regular", "samples"]

        printed = repr(project)
        assert "ore: Surface" in printed
        assert "octree: Octree" in printed

        with pytest.raises(KeyError, match="nothing named"):
            project["missing"]


def test_folders_shape_the_project_tree(tmp_path):
    file = str(tmp_path / "project.geoh5")
    with geoh5io.Workspace(file) as project:
        _tetrahedron().to_geoh5(project, name="hematite",
                                folder="Surfaces/Ore")
        _sheet().to_geoh5(project, name="hematite",
                          folder="Surfaces/Waste")
        _tetrahedron().to_geoh5(project, name="loose")

        assert project.contents() == {
            "Surfaces/Ore/hematite": "Surface",
            "Surfaces/Waste/hematite": "Surface",
            "loose": "Surface"}
        # a bare name shared by two folders is a question, answered with
        # the qualified names
        with pytest.raises(ValueError, match="Surfaces/Ore/hematite"):
            project["hematite"]
        assert type(project["Surfaces/Ore/hematite"]).__name__ == "Solid3D"
        assert type(project["Surfaces/Waste/hematite"]).__name__ \
            == "Surface3D"
        assert "hematite" in project and "loose" in project

    # the groups nest, and repeated exports reuse them rather than
    # planting a second Surfaces folder
    workspace = geoh5py.workspace.Workspace(file)
    tree = [(str(group.name), str(getattr(group.parent, "name", "")))
            for group in workspace.groups]
    workspace.close()
    assert tree.count(("Surfaces", "Workspace")) == 1
    assert ("Ore", "Surfaces") in tree and ("Waste", "Surfaces") in tree


def test_replace_is_scoped_to_the_folder(tmp_path):
    file = str(tmp_path / "project.geoh5")
    with geoh5io.Workspace(file) as project:
        _tetrahedron().to_geoh5(project, name="body", folder="A")
        _tetrahedron().to_geoh5(project, name="body", folder="B")
        # rewriting A's must leave B's alone
        _sheet().to_geoh5(project, name="body", folder="A")

        assert project.contents() == {"A/body": "Surface",
                                      "B/body": "Surface"}
        assert type(project["A/body"]).__name__ == "Surface3D"
        assert type(project["B/body"]).__name__ == "Solid3D"


def test_a_foldered_name_reaches_the_classmethods_too(tmp_path):
    file = str(tmp_path / "project.geoh5")
    _tetrahedron().to_geoh5(file, name="body", folder="Surfaces/Ore")
    back = geoml.data.Mesh3D.from_geoh5(file, name="Surfaces/Ore/body")
    assert type(back).__name__ == "Solid3D"
    # and the bare name still works while it is unique
    assert type(geoml.data.Mesh3D.from_geoh5(file, name="body")).__name__ \
        == "Solid3D"


def test_a_drillhole_group_reads_by_its_name_too(tmp_path):
    file = _foreign_drillholes(tmp_path)
    with geoh5io.Workspace(file) as project:
        holes = project["campaign"]
        assert type(holes).__name__ == "DrillholeData"
        assert len(holes.collar) == 2
        assert "campaign" in project
        assert "campaign" in repr(project)
def test_a_workspace_object_keeps_a_model_together(tmp_path):
    file = str(tmp_path / "project.geoh5")
    blocks, _ = _refined_blocks()
    with geoh5io.Workspace(file) as project:
        _tetrahedron().to_geoh5(project, name="ore")
        _sheet().to_geoh5(project, name="topo")
        blocks.to_geoh5(project, name="blocks")
        # and readable back through the same open handle
        ore = geoml.data.Mesh3D.from_geoh5(project, name="ore")
        assert type(ore).__name__ == "Solid3D"
        assert set(project.contents()) == {"ore", "topo", "blocks"}
    assert geoh5io.contents(file) == {"ore": "Surface", "topo": "Surface",
                                      "blocks": "Octree"}


def test_rewriting_a_name_replaces_the_object(tmp_path):
    file = str(tmp_path / "project.geoh5")
    point, _ = _point_data(n=6)
    point.add_continuous_variable("grade", np.zeros(6))
    point.to_geoh5(file, name="samples")
    point.variables["grade"].measurements.values[:] = np.arange(6.0)
    point.to_geoh5(file, name="samples")

    assert geoh5io.contents(file) == {"samples": "Points"}
    back = geoml.data.PointData.from_geoh5(file)
    values = back.variables["grade - measurements"] \
        .measurements.values.to_numpy()
    assert np.allclose(values, np.arange(6.0))


def test_replace_false_keeps_both_and_says_so(tmp_path):
    file = str(tmp_path / "project.geoh5")
    mesh = _tetrahedron()
    mesh.to_geoh5(file, name="ore")
    mesh.to_geoh5(file, name="ore", replace=False)
    with pytest.raises(ValueError, match="pass `name=`"):
        geoml.data.Mesh3D.from_geoh5(file, name="ore")


def test_a_replacement_in_an_open_workspace_leaves_no_ghost(tmp_path):
    """geoh5py defers removals to close, so the live listings would show
    the replaced object until then; the wrapper remembers the ghosts."""
    file = str(tmp_path / "project.geoh5")
    with geoh5io.Workspace(file) as project:
        _tetrahedron().to_geoh5(project, name="ore")
        _sheet().to_geoh5(project, name="ore")
        assert project.contents() == {"ore": "Surface"}
        back = geoml.data.Mesh3D.from_geoh5(project, name="ore")
        assert type(back).__name__ == "Surface3D"
    assert geoh5io.contents(file) == {"ore": "Surface"}
