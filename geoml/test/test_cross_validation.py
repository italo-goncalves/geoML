"""Cross-validation: the folds, the driver, and the calibration.

A random fold on spatial data is answered by its neighbours, so the folds
here are built to mimic the prediction task instead: held-out points should
sit as far from the training data as the prediction locations do
(`spatial_k_fold`), a whole drill hole stands or falls together, and the
scores are read on data the model never saw.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import spatial, stats

import geoml


def _clustered_points(seed=0, n_holes=12, per_hole=30, spread=2.0):
    """Tightly clustered samples inside a much wider box -- the drillhole
    geometry that makes random folds lie."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(10, 90, size=[n_holes, 2])
    xy = (centers[:, None, :]
          + rng.normal(scale=spread, size=[n_holes, per_hole, 2]))
    df = pd.DataFrame(xy.reshape(-1, 2), columns=["X", "Y"])
    points = geoml.data.PointData(df, ["X", "Y"])
    points.add_metadata("hole", np.repeat(np.arange(n_holes), per_hole))
    return points


def _grid():
    return geoml.data.Grid2D(start=[0, 0], n=[50, 50], step=[2, 2])


def _w_of(points, fold, target):
    """The criterion, recomputed independently of the method under test."""
    coords = np.asarray(points.coordinates)
    pooled = np.concatenate([
        spatial.cKDTree(coords[fold != f]).query(
            coords[fold == f], k=1)[0]
        for f in np.unique(fold)])
    return stats.wasserstein_distance(pooled, target)


# --------------------------------------------------------------------------- #
# the folds
# --------------------------------------------------------------------------- #
def test_the_chosen_folds_mimic_the_task_better_than_random_ones():
    points = _clustered_points()
    w, target, pooled = points.spatial_k_fold(_grid(), k=4, seed=0)

    rng = np.random.default_rng(1)
    random_w = [
        _w_of(points, rng.integers(0, 4, points.n_data), target)
        for _ in range(5)]

    assert w < min(random_w)
    assert np.isclose(w, _w_of(points, points.get_metadata("fold"), target))


def test_a_group_stands_or_falls_together():
    points = _clustered_points()
    points.spatial_k_fold(_grid(), k=4, groups="hole")

    fold = points.get_metadata("fold")
    hole = points.get_metadata("hole")
    for h in np.unique(hole):
        assert np.unique(fold[hole == h]).size == 1


def test_the_fold_column_is_written_with_every_fold_present():
    points = _clustered_points()
    points.spatial_k_fold(_grid(), k=4, groups="hole")

    fold = points.get_metadata("fold")
    assert set(np.unique(fold)) == {0, 1, 2, 3}


def test_grouped_folds_are_deterministic():
    a = _clustered_points()
    b = _clustered_points()
    a.spatial_k_fold(_grid(), k=4, groups="hole")
    b.spatial_k_fold(_grid(), k=4, groups="hole")

    assert np.array_equal(a.get_metadata("fold"), b.get_metadata("fold"))


def test_too_few_groups_refuse():
    points = _clustered_points(n_holes=3)
    with pytest.raises(ValueError, match="3 groups"):
        points.spatial_k_fold(_grid(), k=5, groups="hole")


def test_the_match_is_reported_honestly():
    """The returned distances are what the criterion was computed on: the
    target from the prediction grid, the pooled ones from the folds."""
    points = _clustered_points()
    w, target, pooled = points.spatial_k_fold(_grid(), k=4, seed=0)

    assert target.size == _grid().n_data
    assert pooled.size == points.n_data
    assert w >= 0
