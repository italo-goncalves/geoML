# geoML - machine learning models for geospatial data
# Copyright (C) 2021  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Block models: the `_blockdata` fan-out shared by `Blocks1D/2D/3D`, the
variable-size `BlockSet3D` on its integer lattice, `RotatedBlockSet3D`,
and the sub-block geometry behind the mesh assignments and `crossed_by`.
"""
import copy as _copy
import itertools as _iter
import json as _json

import numpy as _np
import pandas as _pd
import pyvista as _pv

import geoml.math.geometry as _gmt

from geoml.data.base import *
from geoml.data.base import _Attribute, _closing_value
from geoml.data.variables import *
from geoml.data.variables import _Variable
from geoml.data.containers import *
from geoml.data.containers import _PointBased, _SpatialData
from geoml.data.grids import *
from geoml.data.grids import (
    _TOL_COVER, _aggregate_onto, _cover_box, _fitted_rotation)
from geoml.data.meshes import *
from geoml.data.meshes import (
    _below_sheet, _mesh_test, _uncovered_rule, _within_body)

def _sub_block_shares(blocks, test, rows=None):
    """The share of each block's sub-blocks that `test` accepts.

    Walked in chunks: a 5 x 5 x 5 discretization is 125 sub-blocks for
    every location, and materializing all of them at once is what the
    batching in `get_batched_coordinates` exists to avoid. `rows` narrows it
    to the blocks worth asking about; the rest come back `nan`.
    """
    per_block = blocks.rows_per_location
    chunk = max(1, 1000000 // per_block)
    if rows is None:
        rows = _np.arange(blocks._n_data)

    shares = _np.full(blocks._n_data, _np.nan)
    for start in range(0, len(rows), chunk):
        index = rows[start:start + chunk]
        coordinates, _ = blocks.get_batched_coordinates(index)
        # block-major, so every block's sub-blocks are one row
        accepted = test(coordinates).reshape([len(index), per_block])
        shares[index] = accepted.mean(axis=1)
    return shares


def _blocks_from_surface(blocks, base, surface, name, labels, fraction,
                         uncovered):
    """`assign_from_surface` for anything made of blocks."""
    base(blocks, surface, name, labels, uncovered=uncovered)

    if fraction is not None:
        shares = _sub_block_shares(blocks, _below_sheet(surface))
        # a sheet that misses the centre describes the block no better
        # than it describes a location, and left the flag empty for the
        # same reason -- so the fraction says so rather than reading 0
        missed = _np.asarray(blocks.metadata[name].values).ravel() < 0
        shares[missed] = _uncovered_rule(uncovered)[1]
        blocks.add_metadata(fraction, shares)


def _blocks_from_solid(blocks, base, solid, name, labels, fraction):
    """`assign_from_solid` for anything made of blocks."""
    base(blocks, solid, name, labels)

    if fraction is not None:
        blocks.add_metadata(
            fraction, _sub_block_shares(blocks, _within_body(solid)))


def _blockdata(cls):
    # Decorator to extend functionality of some classes
    # Used to avoid multiple inheritance

    old_init = cls.__init__

    def new_init(self, start, n, step=None, end=None, labels=None, discretization=None):
        old_init(self, start=start, n=n, step=step, end=end, labels=labels)
        if discretization is None:
            discretization = [1] * self.n_dim
        self.discretization = discretization

        lo = _np.array([axis.min() for axis in self.grid])
        hi = _np.array([axis.max() for axis in self.grid])
        self._bounding_box = BoundingBox(
            lo - _np.array(self.step_size) / 2,
            hi + _np.array(self.step_size) / 2,
        )

        sub_grid = _np.array(
            list(_iter.product(
                *[_np.arange(d) for d in self.discretization[::-1]]
            )),
            dtype=float)[:, ::-1]
        sub_grid -= (_np.array(self.discretization)[None, :] - 1) / 2
        sub_grid *= _np.array(self.step_size)[None, :]
        sub_grid /= _np.array(self.discretization)[None, :]
        self.sub_grid = sub_grid

    def discretized_coordinates(self, index):
        center = _np.array([g[i] for g, i in zip(self.grid, index)])[None, :]
        return self.sub_grid + center

    def inducing_grid(self, index):
        center = _np.array([g[i] for g, i in zip(self.grid, index)])[None, :]
        grid = self.sub_grid
        discr = _np.array(self.discretization)[None, :]
        # grid = grid * (discr + 1) / (discr - 1) + center
        grid = grid * (discr + 2) / (discr + 1) + center
        return PointData.from_array(grid)

    def get_batched_coordinates(self, index):
        if index is None:
            index = _np.arange(self._n_data)

        centers = _np.asarray(self.coordinates[index])
        # block-major, one temporary instead of one per block: block b owns
        # rows b*k to (b+1)*k, which is what `_aggregate` averages back
        coords = (centers[:, None, :] + self.sub_grid[None, :, :])
        coords = coords.reshape(-1, centers.shape[1])

        splits = None if _np.prod(self.discretization) == 1 else len(index)
        return coords, splits

    @property
    def rows_per_location(self):
        return int(_np.prod(self.discretization))

    def _batch_rows(self, index=None):
        # one row per discretization point of every block in the batch
        n, _ = _PointBased._batch_rows(self, index)
        n_sub = int(_np.prod(self.discretization))
        return n * n_sub, (None if n_sub == 1 else n)

    # captured before being replaced, as `old_init` is: the block versions
    # only add the partial blocks on top of what the grid already answers
    base_assign_from_surface = cls.assign_from_surface
    base_assign_from_solid = cls.assign_from_solid

    def _sub_block_fraction(self, test):
        """The share of each block's sub-blocks that `test` accepts."""
        return _sub_block_shares(self, test)

    def assign_from_surface(self, surface, name, labels=("above", "below"),
                            fraction=None, uncovered=_np.nan):
        """
        As `Grid3D.assign_from_surface`, measuring the partial blocks on
        request.

        The flag in `name` follows the block centre, as a whole-block code
        does everywhere else. Name a `fraction` column as well and the share
        of each block lying below the sheet is measured over the sub-blocks
        `discretization` already defines — what a tonnage near surface needs,
        where counting a half-buried block whole is the error.

        Where the sheet reaches part of a block but not all of it, the
        sub-blocks past its edge count as not below. Where it does not reach
        the block at all — the centre included, which is what leaves the flag
        empty — the fraction is `uncovered` instead of a measurement.

        Parameters
        ----------
        surface : Surface3D
            The sheet to compare against.
        name : str
            Name of the metadata column holding the whole-block flag.
        labels : tuple
            What to call the two sides, in the order (above, below).
        fraction : str, optional
            Name of a second metadata column, to hold the share of each block
            below the sheet. Costs `prod(discretization)` queries per block.
        uncovered : float or "raise"
            What the `fraction` column records for a block the sheet does not
            reach: `numpy.nan` by default, so it cannot pass for a block
            genuinely above ground, or `0.0` to count it as nothing. Pass
            `"raise"` to refuse a surface that does not cover every block.
        """
        _blocks_from_surface(self, base_assign_from_surface, surface, name,
                             labels, fraction, uncovered)

    def assign_from_solid(self, solid, name, labels=("outside", "inside"),
                          fraction=None):
        """
        As `_SpatialData.assign_from_solid`, measuring the partial blocks on
        request.

        `fraction` behaves as it does in `assign_from_surface`: the flag in
        `name` follows the block centre, while the column named here holds the
        share of each block's sub-blocks falling inside the body — the share
        of its volume, for the regular sub-blocks a discretization defines.

        Parameters
        ----------
        solid : Surface3D
            The closed body to test against.
        name : str
            Name of the metadata column holding the whole-block flag.
        labels : tuple
            What to call the two sides, in the order (outside, inside).
        fraction : str, optional
            Name of a second metadata column, to hold the share of each block
            inside the body. Costs `prod(discretization)` queries per block.
        """
        _blocks_from_solid(self, base_assign_from_solid, solid, name, labels,
                           fraction)

    cls.__init__ = new_init
    cls.discretized_coordinates = discretized_coordinates
    cls.inducing_grid = inducing_grid
    cls.get_batched_coordinates = get_batched_coordinates
    cls._batch_rows = _batch_rows
    cls.rows_per_location = rows_per_location
    cls._sub_block_fraction = _sub_block_fraction
    cls.assign_from_surface = assign_from_surface
    cls.assign_from_solid = assign_from_solid

    return cls


@_blockdata
class Blocks1D(Grid1D):
    pass


@_blockdata
class Blocks2D(Grid2D):
    pass


@_blockdata
class Blocks3D(Grid3D):
    def as_pyvista(self, simulations=False, include="**"):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        pv_blocks = _pv.ImageData(
            dimensions=_np.array(self.grid_size) + 1,
            spacing=self.step_size,
            origin=_np.array([ax[0] for ax in self.grid]) - _np.array(self.step_size) / 2
        )

        return self._finish_pyvista(pv_blocks, "blocks", simulations,
                                    include)


def _ghost_shell(origin, size, shape):
    """Mirror images of the boundary cells, one per box face they touch.

    A ghost is its partner's reflection across the shared face, so their
    corners coincide exactly and the corner averaging cannot tear along the
    box surface -- the mismatch between neighbouring ghosts of different
    sizes does not matter, the fill value being one constant no surface
    runs through. Edge and corner diagonals get no ghost: no crossing can
    happen there either.
    """
    ghost_origin, ghost_size = [], []
    for axis in range(3):
        at_low = origin[:, axis] == 0
        if at_low.any():
            mirrored = origin[at_low].copy()
            mirrored[:, axis] -= size[at_low][:, axis]
            ghost_origin.append(mirrored)
            ghost_size.append(size[at_low])
        at_high = origin[:, axis] + size[:, axis] == shape[axis]
        if at_high.any():
            mirrored = origin[at_high].copy()
            mirrored[:, axis] += size[at_high][:, axis]
            ghost_origin.append(mirrored)
            ghost_size.append(size[at_high])
    if not ghost_origin:
        return (_np.zeros([0, 3], dtype=origin.dtype),
                _np.zeros([0, 3], dtype=size.dtype))
    return _np.concatenate(ghost_origin), _np.concatenate(ghost_size)


def _contour_column(blocks, path):
    """The column a contour path names, and that path in full.

    A path naming a column (`"metals/zn/prediction"`) is taken as it stands;
    one naming a variable or component defaults to its `prediction`. A bare
    segment that resolves to nothing on its own is searched for anywhere in
    the tree, which is what lets a component be named without spelling its
    owner -- refused when the name belongs to more than one place.
    """
    path = VariablePath(str(path))
    try:
        found = blocks.get(path)
    except KeyError as err:
        if len(path.parts) != 1:
            raise ValueError(str(err))
        matches = {p: thing
                   for p, thing in blocks.select("**/" + path.parts[0]).items()
                   if not isinstance(thing, _Attribute)}
        if len(matches) > 1:
            raise ValueError(
                "%r sits in more than one place: %s; give the full path"
                % (str(path), ", ".join(str(p) for p in matches)))
        if len(matches) == 0:
            # raises with the canonical listing of what there is to name
            blocks._variable_or_component(path.parts[0])
            raise ValueError(
                "no variable or component named %r" % str(path))
        (path, found), = matches.items()

    if isinstance(found, _Attribute):
        return path, found

    try:
        column = blocks.get(path / "prediction")
    except KeyError:
        parts = getattr(found, "components", None) or {}
        if parts:
            raise ValueError(
                "%r is made of components and holds no prediction of its "
                "own; name one of %s"
                % (str(path), ", ".join(str(name) for name in parts)))
        raise ValueError("%r holds no prediction to contour" % str(path))
    return path / "prediction", column


class BlockSet3D(PointData):
    """
    Blocks of several sizes, on one integer lattice.

    A block model where the interesting ground can be carried finely and the
    rest coarsely. On a real deposit the ground worth resolving is a small
    part of the volume, and a uniform model at the resolution that part needs
    spends almost all of its cells saying nothing: refining 5 m only where it
    is wanted takes a 29-million-cell model to under 700 000.

    Every block's position and size are whole numbers of a **base cell**, the
    finest the model may go, which is `step / discretization ** max_levels`.
    Working in those integers rather than in metres is what makes it exact:
    blocks meet without a tolerance, a block is a whole number of its own
    children, and regrouping conserves mass to the last digit. It also means
    the model can say which of its answers rest on coarse blocks, which is the
    one thing a mixed-support model has to be able to prove -- see
    `docs/variable-block-models.md`.

    It is built **full**: the blocks tile their box exactly, and every
    operation keeps them that way. Ground to leave out is filtered, not
    removed, so that grouping is always safe -- a half-populated group would
    average over blocks that are not there and quietly weigh the answer wrong.

    `discretization` does two jobs, and they are the same job. It is how
    finely a block is sampled to average it, and it is how a block splits:
    each sub-block becomes a child. So the refinement ratio is the
    discretization, per axis and not necessarily two -- `[2, 2, 1]` refines in
    plan and leaves the bench height alone. Being the same at every level is
    what lets a block of any size fan out into the same number of rows, so the
    model sees one shape whatever it is looking at and nothing downstream has
    to know that levels exist. It costs a coarse block some accuracy in its
    own average, always by overstating how variable it is, which errs towards
    splitting it -- and splitting is what removes the error.

    Parameters
    ----------
    start : array-like
        Centre of the first (coarsest) block.
    n : array-like
        Number of blocks along each axis, at the coarsest level.
    step : array-like
        Size of a coarsest-level block.
    discretization : array-like
        Sub-blocks per axis, used at every level, and so also the ratio a
        block splits by. An axis given 1 is never refined.
    max_levels : int
        How many times a block may be split. Fixes the base cell, and so the
        lattice everything else is counted in.
    labels : list
        Coordinate names.

    Attributes
    ----------
    block_size : array
        `(n_data, 3)`, each block's size in the coordinates' own units. Not
        called `step_size`: that name means one size for the whole object, and
        anything reading it would take the product of this array for a volume.
    block_volume : array
        `(n_data,)`, what each block is worth in a tonnage.
    level : array
        `(n_data,)`, 0 for a coarsest block up to `max_levels` for a base one.
    """

    def __init__(self, start, n, step, discretization=(2, 2, 2), max_levels=3,
                 labels=("X", "Y", "Z")):
        if int(max_levels) < 0:
            raise ValueError("max_levels cannot be negative; got %r"
                             % max_levels)

        self.max_levels = int(max_levels)
        self.discretization = [int(d) for d in discretization]
        if len(self.discretization) != 3 or any(
                d < 1 for d in self.discretization):
            raise ValueError(
                "discretization needs three positive numbers; got %r"
                % (discretization,))
        if self.max_levels > 0 and _np.prod(self.discretization) == 1:
            raise ValueError(
                "a discretization of one sub-block cannot refine anything, so "
                "max_levels=%d would have nothing to do; give a coarser "
                "discretization or max_levels=0" % self.max_levels)

        n = _np.asarray(n, dtype=_np.int64)
        step = _np.asarray(step, dtype=float)
        start = _np.asarray(start, dtype=float)
        if not (n.shape == step.shape == start.shape == (3,)):
            raise ValueError(
                "start, n and step are three numbers each; got %s, %s and %s"
                % (start.shape, n.shape, step.shape))

        # Splitting a block hands each of its sub-blocks a block of its own, so
        # the discretization is also the refinement ratio -- per axis, and not
        # necessarily two. That is what keeps sub-block `j` and child `j` the
        # same corner of the parent, which is what lets the criterion read off
        # a coarse prediction mean anything about the children it would make.
        # It also lets an axis be left alone: [2, 2, 1] refines in plan and
        # keeps the bench height.
        self.base_step = step / self._coarse_size
        # the box corner, not the first centre: a block's origin is its lower
        # corner, which is what makes the lattice arithmetic come out integer
        self.box_corner = start - step / 2
        self.lattice_shape = n * self._coarse_size

        cells = _np.stack(_np.meshgrid(
            *[_np.arange(k) for k in n], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self._origin = cells * self._coarse_size
        self._level = _np.zeros(len(cells), dtype=_np.int64)
        # nothing has been predicted yet, so every block is new
        self._fresh = _np.ones(len(cells), dtype=bool)

        self._sub_grid = _gmt.unit_sub_grid(self.discretization)

        super().__init__(
            _pd.DataFrame(self._centres(), columns=list(labels)), list(labels))
        self._bounding_box = self._lattice_bounding_box()

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def _centres(self):
        return self.box_corner + (self._origin + self._size / 2) \
            * self.base_step

    @property
    def _coarse_size(self):
        """A coarsest block's extent, in base cells, along each axis.

        Not the number of sub-blocks, which is `prod(discretization)` and is
        the same at every level -- this is how far a level-0 block reaches
        across the lattice, and so how many base cells it would come to if it
        were split all the way down.
        """
        return _np.array(self.discretization,
                         dtype=_np.int64) ** self.max_levels

    @property
    def _size(self):
        """Each block's extent in base cells, from its level.

        Held as the level rather than the size: a level is one small number
        against three, and the two cannot then disagree.
        """
        ratio = _np.array(self.discretization, dtype=_np.int64)
        return self._coarse_size[None, :] \
            // ratio[None, :] ** self._level[:, None]

    @property
    def block_size(self):
        return self._size * self.base_step

    @property
    def block_volume(self):
        return _np.prod(self.block_size, axis=1)

    @property
    def level(self):
        return self._level

    @property
    def rows_per_location(self):
        return int(_np.prod(self.discretization))

    def is_full(self):
        """Whether the blocks tile their box exactly -- no gap, no overlap.

        Volume alone cannot answer: a gap and an overlap of the same size
        cancel. So the base cells are counted, in an array the shape of the
        lattice, which costs a byte per base cell (29 MB for a 30-million-cell
        model). Construction is full and both `split` and `group` preserve it,
        so this is for checking something that came from elsewhere rather than
        for routine use.
        """
        covered = int(_np.prod(self._size, axis=1).sum())
        if covered != int(_np.prod(self.lattice_shape)):
            return False

        seen = _np.zeros(tuple(self.lattice_shape), dtype=_np.uint8)
        for origin, size in zip(self._origin, self._size):
            seen[origin[0]:origin[0] + size[0],
                 origin[1]:origin[1] + size[1],
                 origin[2]:origin[2] + size[2]] += 1
        return bool(seen.min() == 1 and seen.max() == 1)

    _NO_SUBSET = (
        "a %s cannot be subsetted: it is structurally complete by design, and "
        "removing blocks would leave a group averaging over children that are "
        "no longer there -- the mass-conservation error the lattice exists to "
        "prevent. Exclude ground by value instead, with a metadata column "
        "(`add_metadata`); `predict(..., where=...)` visits part of a model "
        "without making a smaller one. To hand the blocks to something else, "
        "`as_data_frame` carries a size per row, `as_pyvista` writes them as "
        "hexahedra, and `to_zarr` round-trips the lattice whole.")

    def __getitem__(self, item):
        # PointData's would hand back a plain PointData, silently: no origin,
        # no level, no size, so the size columns vanish, `as_pyvista` draws
        # points rather than blocks and `grade_tonnage` refuses. Better to say
        # so here than to return something that looks usable.
        raise TypeError(self._NO_SUBSET % type(self).__name__)

    def subset_region(self, min_val, max_val,
                      include_min=None, include_max=None):
        raise TypeError(self._NO_SUBSET % type(self).__name__)

    # ------------------------------------------------------------------ #
    # refinement
    # ------------------------------------------------------------------ #
    def split(self, mask, carry=True, labels=("X", "Y", "Z")):
        """A new block set with each marked block cut into its own sub-blocks.

        A block becomes `prod(discretization)` children, one per sub-block and
        in the same order, so the values a coarse prediction already holds for
        those sub-blocks describe the blocks this makes.

        A block that was **not** split keeps what was predicted for it: it is
        the same block on the same support, so its value is still the right
        answer and arriving at it again would be work for nothing. The
        children start missing, and `unpredicted()` says which they are, so
        `predict(..., where=...)` visits only them. A parent's value is never
        handed down -- that would manufacture children agreeing exactly, which
        is the one thing refining is meant to find out rather than assume.

        Parameters
        ----------
        mask : array-like
            One boolean per block, or the indices of the blocks to split.
        carry : bool
            Whether to bring the variables and metadata across. `False` gives
            bare geometry, for building a mesh to predict onto from scratch.
        """
        mask = _np.asarray(mask)
        if mask.dtype != bool:
            index = _np.zeros(self.n_data, dtype=bool)
            index[mask] = True
            mask = index
        if mask.shape != (self.n_data,):
            raise ValueError(
                "the mask needs one value per block: got %d for %d"
                % (mask.size, self.n_data))

        finest = self._level >= self.max_levels
        if _np.any(mask & finest):
            raise ValueError(
                "%d block(s) are already at the finest level the lattice "
                "allows (max_levels=%d); build the set with more levels to "
                "go further"
                % (int(_np.count_nonzero(mask & finest)), self.max_levels))

        # one child per sub-block, in the sub-blocks' own order
        corner = _gmt.sub_block_index(self.discretization)
        child_size = self._size[mask] // _np.array(self.discretization)
        children = (self._origin[mask][:, None, :]
                    + corner[None, :, :] * child_size[:, None, :]
                    ).reshape(-1, 3)
        child_level = _np.repeat(self._level[mask] + 1, len(corner))

        new = _copy.copy(self)
        new._origin = _np.concatenate([self._origin[~mask], children])
        new._level = _np.concatenate([self._level[~mask], child_level])
        # the blocks this made, so that predicting can be told to visit only
        # them without first being told what was predicted before
        new._fresh = _np.concatenate([
            _np.zeros(int(_np.count_nonzero(~mask)), dtype=bool),
            _np.ones(len(children), dtype=bool)])
        new.variables = {}
        new.metadata = {}
        new._init_coordinates(new._centres(), list(labels))
        # `_init_coordinates` has just measured the box off the coordinates,
        # which are block *centres*: that box is half a block short on every
        # side, and short by a different amount after every split, since
        # refining an edge block moves its centre nearer the boundary. Two
        # sets over the same ground would then disagree about their extent.
        # The real box is the lattice, which no refinement changes.
        new._bounding_box = new._lattice_bounding_box()

        if carry:
            # A block that was not split is the same block on the same
            # support, so what was predicted for it still stands; only the new
            # children have nothing yet. Metadata goes to the children as
            # well, and by inheritance rather than as missing: it describes the
            # ground, and a child occupies its parent's ground.
            parent = _np.concatenate(
                [_np.flatnonzero(~mask),
                 _np.repeat(_np.flatnonzero(mask), len(corner))])
            for name, variable in self.variables.items():
                new.variables[name] = variable.carry_to(
                    new, ~mask, len(children))
            for name, column in self.metadata.items():
                values = _np.asarray(column.values)[parent]
                fresh = _Attribute(new, values, dtype=values.dtype)
                fresh.labels = column.labels
                new.metadata[name] = fresh
        return new

    def group(self, mask, carry=True, labels=("X", "Y", "Z")):
        """The inverse of `split`: whole families of children, back into the
        parent they came from.

        A block is grouped with its siblings, so the mask must name **every**
        child of a parent or none of them. A partial family would average over
        children that are not there and mis-weight the parent -- the
        mass-conservation error the lattice exists to prevent -- so it is
        refused rather than approximated. That check is what makes conversion
        between supports two-directional: `group` undoes `split` exactly, and
        the blocks tile their box afterwards as they did before.

        A block that was **not** grouped keeps what it holds, exactly as in
        `split` and for the same reason: it is the same block on the same
        support, so its value is still the right answer for it. The parents
        are the ones on new ground, and they come back missing --
        `unpredicted()` names them and `predict(..., where=...)` fills them.

        A parent's value is never averaged from its children. Coarsening is a
        change of support and almost nothing survives it: a parent's spread is
        not its children's, its within-block dispersion is larger by exactly
        what the grouping absorbed, and a category has no mean. The one thing
        that would come across exactly is a realization, and a variable is
        more than its realizations. Metadata does come across, from the first
        child -- it describes the ground rather than the model, and where the
        children disagree about it there is no right answer to be had.

        Parameters
        ----------
        mask : array-like
            One boolean per block, or the indices of the blocks to group.
        carry : bool
            Whether to bring across what the blocks that were not grouped
            hold. `False` gives bare geometry.
        labels : list
            Coordinate names.
        """
        mask = _np.asarray(mask)
        if mask.dtype != bool:
            index = _np.zeros(self.n_data, dtype=bool)
            index[mask] = True
            mask = index
        if mask.shape != (self.n_data,):
            raise ValueError(
                "the mask needs one value per block: got %d for %d"
                % (mask.size, self.n_data))

        coarsest = self._level < 1
        if _np.any(mask & coarsest):
            raise ValueError(
                "%d block(s) are already at the coarsest level the lattice "
                "has and belong to no parent"
                % int(_np.count_nonzero(mask & coarsest)))

        ratio = _np.array(self.discretization, dtype=_np.int64)
        family = self._size[mask] * ratio[None, :]
        parent = (self._origin[mask] // family) * family
        level = self._level[mask] - 1

        # one key a parent, level included: two families of different sizes can
        # share a lower corner, and they are not the same parent
        shape = self.lattice_shape
        key = (((parent[:, 0] * shape[1] + parent[:, 1]) * shape[2]
                + parent[:, 2]) * (self.max_levels + 1) + level)
        _, first, counts = _np.unique(key, return_index=True,
                                      return_counts=True)

        whole = int(_np.prod(self.discretization))
        if _np.any(counts != whole):
            short = _np.flatnonzero(counts != whole)
            raise ValueError(
                "grouping takes whole families: %d parent(s) were named by "
                "only some of their %d children, the first by %d. A partial "
                "family averages over blocks that are not there, which is the "
                "one thing the lattice exists to prevent"
                % (len(short), whole, int(counts[short[0]])))

        new = _copy.copy(self)
        new._origin = _np.concatenate([self._origin[~mask], parent[first]])
        new._level = _np.concatenate([self._level[~mask], level[first]])
        # the blocks this made, on a support nothing has been predicted on
        new._fresh = _np.concatenate([
            _np.zeros(int(_np.count_nonzero(~mask)), dtype=bool),
            _np.ones(len(first), dtype=bool)])
        new.variables = {}
        new.metadata = {}
        new._init_coordinates(new._centres(), list(labels))
        # the lattice is the box, not the spread of the centres -- see `split`
        new._bounding_box = new._lattice_bounding_box()

        if carry:
            source = _np.concatenate(
                [_np.flatnonzero(~mask), _np.flatnonzero(mask)[first]])
            for name, variable in self.variables.items():
                new.variables[name] = variable.carry_to(
                    new, ~mask, len(first))
            for name, column in self.metadata.items():
                values = _np.asarray(column.values)[source]
                fresh = _Attribute(new, values, dtype=values.dtype)
                fresh.labels = column.labels
                new.metadata[name] = fresh
        return new

    def block_shares(self, split_on=None):
        """How often each decision cuts a block in two.

        A continuous variable contributes one column per cut-off it declares,
        a categorical one per category, and both mean the same thing: the
        share of realizations in which this block's sub-blocks fall on both
        sides of something that matters. A grade is judged against the grades
        someone declared; an indicator against zero, that being where one
        category stops winning and its rival starts.

        Returns a dict of name -> `(n_data,)` array, empty where nothing
        declared a decision to make.
        """
        chosen = list(self.variables) if split_on is None else \
            [str(name) for name in _np.atleast_1d(split_on)]
        shares = {}
        for name in chosen:
            if name not in self.variables:
                raise ValueError(
                    "no variable named %r to split on; found %s"
                    % (name, ", ".join(sorted(self.variables)) or "none"))
            # each variable reports its own -- see `_Variable.split_shares`
            for key, values in self.variables[name].split_shares().items():
                shares[("%s %s" % (name, key)).strip()] = values
        return shares

    def needs_splitting(self, split_on=None, tolerance=0.05):
        """Which blocks hold more than one answer, and so are worth cutting.

        A block whose sub-blocks agree, realization by realization, holds one
        answer however finely it is cut. One whose sub-blocks disagree holds
        two, and cutting is what separates them.

        Note what this does *not* mark: a block the model is merely unsure
        about. Realizations either side of a cut-off are the model not
        knowing, and no amount of cutting will settle that -- the answer to it
        is another drillhole. Only disagreement *within* a realization counts.

        The test is over every decision at once and any one is enough, which
        is the cautious way round on purpose: a block worth splitting for one
        variable is worth splitting whatever the others say. Name `split_on`
        to narrow it -- letting every element of a polymetallic deposit vote
        marks most of the model and gives back the saving.

        Blocks already at the finest level are never marked, there being
        nothing to cut them into.

        Parameters
        ----------
        split_on : str or list, optional
            Which variables get a say. All of them by default.
        tolerance : float
            The share of realizations that must find the block divided. Small
            but not zero, so that one realization in twenty does not carry it.
        """
        shares = self.block_shares(split_on)
        mask = _np.zeros(self.n_data, dtype=bool)
        for values in shares.values():
            mask |= _np.asarray(values, dtype=float) > tolerance
        return mask & (self._level < self.max_levels)

    def _by_level(self):
        """Where the blocks of each level sit, as a sorted table of origins.

        Sorted rather than rasterized on purpose: naming the block covering a
        point is then a `searchsorted`, where painting the base lattice would
        cost a cell for every one the model exists to avoid carrying.
        """
        shape = self.lattice_shape
        tables = []
        for k in range(self.max_levels + 1):
            rows = _np.flatnonzero(self._level == k)
            here = self._origin[rows]
            key = ((here[:, 0] * shape[1] + here[:, 1]) * shape[2]
                   + here[:, 2])
            order = _np.argsort(key, kind="stable")
            tables.append((key[order], rows[order]))
        return tables

    def unbalanced(self, gap=1):
        """Blocks with a neighbour more than `gap` levels finer than they are.

        A block whose own sub-blocks agree is never marked by
        `needs_splitting`, and rightly so -- cutting it would not change the
        answer it gives. But the field can still turn sharply inside it, and
        nothing in the block itself says so. That a *neighbour* was cut twice
        while this block was not cut at all is the evidence, and it lives
        outside the block.

        It matters for what is drawn rather than for what is decided. A
        contour reads a block through its eight corners, so a coarse block
        beside much finer ones is a crude straight guess across a long span
        laid right where the surface runs. Levelling the jump measured three
        times closer to the true surface for 35% more blocks, where refining a
        whole level deeper without it bought almost nothing for 2.6 times as
        many -- deeper refinement widens the jumps as fast as it narrows the
        blocks. `models.refine` therefore cuts these as it goes.

        Blocks already at the finest level are never marked, there being
        nothing to cut them into.

        Parameters
        ----------
        gap : int
            How many levels of difference to tolerate. One is the usual 2:1
            balance: a block may meet blocks one level finer, not two.
        """
        ratio = _np.array(self.discretization, dtype=_np.int64)
        shape = self.lattice_shape
        tables = self._by_level()
        size = self._size
        marked = _np.zeros(self.n_data, dtype=bool)

        # asked from the fine side: a fine block steps one cell past each of
        # its faces and names the coarse block covering that point, which is
        # exact where asking the coarse block what lies beyond its face is not
        # -- a cell of its own size holds blocks that do not touch it
        for fine in range(int(gap) + 1, self.max_levels + 1):
            rows = _np.flatnonzero(self._level == fine)
            if len(rows) == 0:
                continue
            origin, extent = self._origin[rows], size[rows]

            for coarse in range(fine - int(gap)):
                key_of, owner = tables[coarse]
                if len(owner) == 0:
                    continue
                cell = self._coarse_size // ratio ** coarse

                for axis in range(3):
                    for side in (-1, 1):
                        beyond = origin.copy()
                        beyond[:, axis] += (extent[:, axis] if side > 0
                                            else -1)
                        inside = ((beyond[:, axis] >= 0)
                                  & (beyond[:, axis] < shape[axis]))
                        owning = (beyond // cell) * cell
                        key = ((owning[:, 0] * shape[1] + owning[:, 1])
                               * shape[2] + owning[:, 2])
                        at = _np.minimum(_np.searchsorted(key_of, key),
                                         len(key_of) - 1)
                        hit = inside & (key_of[at] == key)
                        marked[owner[at[hit]]] = True

        return marked & (self._level < self.max_levels)

    def _lattice_bounding_box(self):
        """The lattice is the box, not the spread of the centres -- see
        `split`. A rotated subclass answers with the rotated corners."""
        return BoundingBox(
            self.box_corner,
            self.box_corner + self.lattice_shape * self.base_step)

    @classmethod
    def from_data(cls, data, step, margin=0.1, decimals=0,
                  discretization=(2, 2, 2), max_levels=3):
        """
        A block model covering another object's bounding box.

        As `Grid3D.from_data`, counting blocks rather than nodes: the
        margined box's lower *corner* is floored to `decimals`, and enough
        blocks follow to cover the far side, so the corner is round and the
        margin never shrinks.

        Parameters
        ----------
        data
            Any spatial object, drillholes included.
        step
            The coarse block size, one number or one per direction.
        margin : float or array
            A fraction of the data's extent; see `Grid3D.from_data`.
        decimals : int
            Decimals to floor the box corner to.
        discretization, max_levels
            As the constructor takes them.
        """
        corner, n, step, labels = _cover_box(
            data, step, margin, decimals, n_dim=3, cells=True)
        return cls(start=corner + step / 2, n=n, step=step,
                   discretization=discretization, max_levels=max_levels,
                   labels=labels if labels else ("X", "Y", "Z"))

    # ------------------------------------------------------------------ #
    # sample data
    # ------------------------------------------------------------------ #
    def index_data(self, data):
        """Which block each of `data`'s locations falls in.

        One row index per location, `-1` for anything outside the box. Note
        this is **not** what a grid's `index_data` returns -- a cell index per
        axis -- because blocks of several sizes have no per-axis index to
        return. Which block *is* the answer here.

        The lattice makes it cheap: a location's base cell is arithmetic, and
        the block covering that cell is the one whose origin is the cell's
        ancestor at that block's own level, so one `searchsorted` per level
        finds it and every location is settled within `max_levels + 1` of them.
        """
        if data.n_dim != self.n_dim:
            raise DimensionMismatchError(
                "Data dimension mismatch. Expected dimension %d and found %d."
                % (self.n_dim, data.n_dim))

        coordinates = _np.asarray(data.coordinates, dtype=float)
        cell = _np.floor(
            (coordinates - self.box_corner) / self.base_step).astype(_np.int64)
        shape = self.lattice_shape

        found = _np.full(len(coordinates), -1, dtype=_np.int64)
        left = _np.flatnonzero(
            _np.all((cell >= 0) & (cell < shape), axis=1))
        ratio = _np.array(self.discretization, dtype=_np.int64)
        tables = self._by_level()

        for k in range(self.max_levels + 1):
            if len(left) == 0:
                break
            key_of, owner = tables[k]
            if len(owner) == 0:
                continue
            size = self._coarse_size // ratio ** k
            ancestor = (cell[left] // size) * size
            key = ((ancestor[:, 0] * shape[1] + ancestor[:, 1]) * shape[2]
                   + ancestor[:, 2])
            at = _np.minimum(_np.searchsorted(key_of, key), len(key_of) - 1)
            hit = key_of[at] == key
            found[left[hit]] = owner[at[hit]]
            left = left[~hit]

        return found

    def _cell_of(self, data):
        # which block, already this object's own row index
        return self.index_data(data)

    def aggregate(self, data, variables=None, metadata=True):
        """Carries another object's measurements onto the blocks holding them.

        As `Grid3D.aggregate` -- one method, the operation following each
        variable's kind -- over blocks of several sizes.
        """
        _aggregate_onto(self, data, variables, metadata)
        return self

    def assign_from_surface(self, surface, name, labels=("above", "below"),
                            fraction=None, uncovered=_np.nan):
        """
        As `Blocks3D.assign_from_surface`, over blocks of several sizes.

        The flag in `name` follows the block centre; naming a `fraction`
        column measures the share of each block below the sheet over the
        sub-blocks `discretization` defines, scaled to each block's own size.
        `crossed_by` asks the same question and answers with the blocks worth
        cutting.
        """
        _blocks_from_surface(self, PointData.assign_from_surface, surface,
                             name, labels, fraction, uncovered)

    def assign_from_solid(self, solid, name, labels=("outside", "inside"),
                          fraction=None):
        """
        As `Blocks3D.assign_from_solid`, over blocks of several sizes.

        `fraction` holds the share of each block's volume inside the body, and
        `crossed_by` turns the same test into the blocks worth cutting.
        """
        _blocks_from_solid(self, PointData.assign_from_solid, solid, name,
                           labels, fraction)

    def crossed_by(self, mesh):
        """Which blocks a mesh passes through, and so which are worth cutting.

        A block is crossed when its sub-blocks fall on **both** sides of the
        mesh -- the question `needs_splitting` asks of a cut-off, asked of
        geometry instead. A topography, a vein wall, a lease boundary: a block
        the surface runs through holds two answers whatever is predicted into
        it, and no amount of prediction will separate them.

        One entirely above or entirely below is left alone however close it
        lies, which is what keeps this from refining a whole domain. Blocks
        already at the finest level are never marked, as elsewhere.

        Hand it to `split`, or give the mesh to `models.refine`, which unions
        this with the other two criteria and repeats until nothing is left to
        cut::

            blocks = blocks.split(blocks.crossed_by(topography))

        A sheet that covers only part of the model counts a sub-block past its
        edge as not below, the way `fraction` does, so a block straddling the
        sheet's own boundary reads as crossed. That is usually wanted -- the
        edge is a real feature of the ground being described -- but it is why
        a sheet should reach across the model when it is not.

        Parameters
        ----------
        mesh : Surface3D or Solid3D
            The sheet or closed body to test against. A `Mesh3D` that is
            neither has no sides and is refused.

        Returns
        -------
        array
            One boolean per block.
        """
        splittable = _np.flatnonzero(self._level < self.max_levels)
        if len(splittable) == 0:
            return _np.zeros(self.n_data, dtype=bool)

        shares = _sub_block_shares(self, _mesh_test(mesh), rows=splittable)
        # nan everywhere else, and a comparison against it is False, so the
        # blocks that were never asked about answer no on their own
        with _np.errstate(invalid="ignore"):
            return (shares > 0) & (shares < 1)

    def unpredicted(self, variable=None):
        """Which blocks have nothing in them yet.

        What `split` leaves behind: hand it to `predict(..., where=...)` and
        only the blocks the refinement created are visited. Without a variable
        it is the blocks the last split made, which is what a container knows
        without having to be told what was predicted into it; naming one reads
        its missing values instead, which stays true however the object was
        arrived at.
        """
        if variable is None:
            return _np.array(self._fresh, dtype=bool)
        return _np.isnan(_np.asarray(
            self.variables[variable].prediction.values))

    # ------------------------------------------------------------------ #
    # what the model asks for
    # ------------------------------------------------------------------ #
    def get_batched_coordinates(self, index=None):
        if index is None:
            index = _np.arange(self._n_data)

        centres = _np.asarray(self.coordinates[index])
        size = self.block_size[index]
        # block-major, one temporary rather than one per block: block b owns
        # rows b*k to (b+1)*k, which is what `_aggregate` averages back. The
        # only difference from a fixed-size block model is that the sub-grid
        # is scaled by each block's own size.
        coords = centres[:, None, :] + self._sub_grid[None, :, :] \
            * size[:, None, :]
        coords = coords.reshape(-1, centres.shape[1])

        splits = None if self.rows_per_location == 1 else len(centres)
        return coords, splits

    def _batch_rows(self, index=None):
        n, _ = _PointBased._batch_rows(self, index)
        k = self.rows_per_location
        return n * k, (None if k == 1 else n)

    def as_data_frame(self, metadata=True, **kwargs):
        df = super().as_data_frame(metadata=metadata, **kwargs)
        for i, label in enumerate(self.coordinate_labels):
            df["_" + label] = self.block_size[:, i]
        return df

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    def as_pyvista(self, simulations=False, include="**"):
        """
        Converts this object to a pyvista one, carrying its variables.

        One hexahedron per block, written out corner by corner, rather than
        the `ImageData` a regular block model exports: implicit geometry can
        only say one spacing, and the point here is that there is more than
        one. The cells are welded, so blocks that touch share the corners they
        meet at -- which is what lets anything be contoured across them.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        mesh = self._hex_mesh(self._origin, self._size, self.base_step)
        return self._finish_pyvista(mesh, "cells", simulations, include)

    def _hex_mesh(self, origin, size, step):
        """One welded hexahedron per block, on any lattice of blocks -- the
        object's own, or the finer one a contour is drawn on. `origin` and
        `size` count that lattice's own cells, `step` says how big one is."""
        low = self.box_corner + origin * step
        size = size * step
        points = (low[:, None, :]
                  + _gmt.HEX_CORNERS[None, :, :] * size[:, None, :]).reshape(-1, 3)

        connectivity = _np.arange(len(points)).reshape(-1, 8)
        cells = _np.hstack(
            [_np.full((len(connectivity), 1), 8), connectivity]).ravel()
        kinds = _np.full(len(connectivity), _pv.CellType.HEXAHEDRON,
                         dtype=_np.uint8)
        # Welded, and it matters: written corner by corner every cell owns its
        # own eight, so neighbours share nothing, and anything reading values
        # at a corner sees one block there instead of the several that meet.
        # A contour over an unwelded mesh comes back empty for that reason.
        return _pv.UnstructuredGrid(cells, kinds, points).clean(
            tolerance=1e-9, produce_merge_map=False)

    @staticmethod
    def _shared_corners(origin, size, shape):
        """Each block's eight corners, as indices into the distinct lattice
        points -- the welding of `_hex_mesh`, done in integers."""
        corners = (origin[:, None, :]
                   + _gmt.HEX_CORNERS[None, :, :] * size[:, None, :])
        span = shape + 1
        key = ((corners[..., 0] * span[1] + corners[..., 1]) * span[2]
               + corners[..., 2])
        return _np.unique(key.ravel(), return_inverse=True)[1].reshape(-1, 8)

    @staticmethod
    def _at_corners(corners, values):
        """The mean over the blocks meeting at each corner, read back per
        block -- what `cell_data_to_point_data` puts there, and so what decides
        where the surface runs. A block holding no value contributes nothing,
        rather than carrying its absence into every corner it touches.
        """
        total = _np.zeros(int(corners.max()) + 1)
        count = _np.zeros(len(total))
        known = _np.isfinite(values)
        _np.add.at(total, corners[known].ravel(), _np.repeat(values[known], 8))
        _np.add.at(count, corners[known].ravel(), 1.0)
        mean = _np.divide(total, count, out=_np.full(len(total), _np.nan),
                          where=count > 0)
        return mean[corners]

    def _cut_to_contour(self, values, value, margin=1, supersample=1,
                        cap=None):
        """The blocks a surface runs through, cut small -- in the mesh handed
        to VTK, not in the model.

        A hexahedron is contoured from its own eight corners, so where a
        coarse block meets finer ones it cannot see the corners they place in
        the middle of the shared face. The two sides then draw different
        curves there and the surface tears along every interface it crosses.
        Cutting the surface's neighbourhood to one size puts the interfaces
        out of its way, and one ring of margin covers the surface shifting as
        the values are read at the finer size.

        Cutting `supersample` levels past the model's own finest also smooths
        it. What VTK contours is a trilinear reading of the corner averages,
        continuous but with a crease at every face it crosses, and on a field
        with structure at a few blocks those creases are what looks blocky.
        Averaging onto corners again at each finer level composes into a
        rounder reconstruction, so the surface both turns less sharply and
        sits closer to the level set it is meant to be: on a rough test field,
        one level took the mean angle between neighbouring triangles from 9.1
        to 5.7 degrees and halved the distance from the true surface, matching
        a model of 3.4 times as many blocks that had to be predicted.

        Nothing is predicted, at any level. A child takes its parent's value
        plus what the corners say about the shape running across it, and that
        correction averages to zero over the children, so a block's estimate
        is exactly the mean of the children standing in for it.

        Returns
        -------
        origin, size, step, values
            The finer lattice: origins and sizes counted in its own cells,
            and how big one of those is. `(None, None, None, None)` if nothing
            was cut and the blocks as they stand will do.
        """
        ratio = _np.array(self.discretization, dtype=_np.int64)
        offset = _gmt.sub_block_index(self.discretization)
        weights = _gmt.trilinear_weights(self.discretization)

        # a lattice `supersample` levels finer than the model's own, so that
        # cutting below the base cell is still whole numbers of a cell
        unit = ratio ** int(supersample)
        origin = self._origin * unit
        size = self._size * unit
        step = self.base_step / unit
        shape = self.lattice_shape * unit

        values = _np.asarray(values, dtype=float)
        cut = False

        for _ in range(self.max_levels + int(supersample)):
            corners = self._shared_corners(origin, size, shape)
            at_corner = self._at_corners(corners, values)
            # a corner with no value must not decide anything either way
            low = _np.where(_np.isnan(at_corner), _np.inf, at_corner).min(1)
            high = _np.where(_np.isnan(at_corner), -_np.inf, at_corner).max(1)
            marked = (low < value) & (high > value)
            if cap is not None:
                # a closing cap runs through the boundary cells the kept
                # region touches, and it tears between mismatched sizes like
                # any other piece of the surface -- so those cells are cut
                # with the rest (the cap side: +1 keeps above, -1 below)
                at_face = _np.any(
                    (origin == 0) | (origin + size == shape), axis=1)
                kept = values > value if cap > 0 else values < value
                marked |= at_face & kept
            near = _gmt.grow(corners, marked, margin)
            near &= _np.all(size >= ratio, axis=1)
            if not _np.any(near):
                break

            cut = True
            child_size = size[near] // ratio
            children = (origin[near][:, None, :]
                        + offset[None, :, :] * child_size[:, None, :]
                        ).reshape(-1, 3)
            shaped = at_corner[near] @ weights.T
            shaped += (values[near] - at_corner[near].mean(axis=1))[:, None]
            # a block beside one that was never predicted has no shape to read
            # off its corners, so it hands its own value down unchanged
            shaped = _np.where(_np.isfinite(shaped), shaped,
                               values[near][:, None])

            origin = _np.concatenate([origin[~near], children])
            size = _np.concatenate(
                [size[~near], _np.repeat(child_size, len(offset), axis=0)])
            values = _np.concatenate([values[~near], shaped.ravel()])

        if not cut:
            return None, None, None, None
        return origin, size, step, values

    def get_contour(self, path, value, supersample=1, simplify=None,
                    close=False):
        """
        Isosurface through blocks of more than one size.

        `marching_cubes` wants a rectangular array and there is none to give
        it, so the cells are handed to VTK instead, which contours an
        unstructured grid directly. On a model of one block size the two agree
        exactly; here the answer is the one a regular grid could not have
        produced without carrying every block at the finest size.

        Values live on the cells and an isosurface needs them on the corners,
        so they are averaged onto the corners first -- the blocks meeting at a
        corner are what decide where the surface passes.

        The blocks the surface runs through are cut to the finest size the
        lattice allows before any of that, in the mesh handed to VTK and not
        in the model: a coarse block cannot see the corners its finer
        neighbours place in the middle of the face they share, so the two
        sides draw different curves there and the surface tears along every
        such interface it crosses. Cutting its neighbourhood to one size puts
        the interfaces out of the way. Nothing is predicted -- a child reads
        its parent's value and the shape its corners carry -- and the surface
        comes back the one a model carried at the finest size throughout would
        have given. See `_cut_to_contour`.

        Parameters
        ----------
        path : str
            The column to contour, named the way the tree names it:
            `"grade/prediction"` is that column, `"grade"` alone defaults to
            the variable's prediction, and `"Elements/Zn"` reaches a
            component (only the components of a composition hold a grade). A
            single bare segment that is no variable of its own is searched
            for anywhere in the tree, so `"Zn"` still finds `"Elements/Zn"`
            as long as only one variable holds a `Zn`.
        value : float
            The value to draw the surface at.
        supersample : int
            How many levels past the model's own finest block to cut the mesh
            to. Costs `prod(discretization)` times the cells per level, around
            the surface only, and buys a rounder and *closer* surface rather
            than merely a prettier one -- what VTK reads between block corners
            is trilinear, and creasing at every face is what looks blocky. One
            level is worth roughly predicting a model several times the size;
            past that it flattens off. Zero to leave the mesh at the model's
            own resolution.
        simplify : float, optional
            A geometric error budget, in coordinate units: the surface is
            simplified until pushing further would move it more than this
            (see `Mesh3D.simplify`). Pairs naturally with `supersample`,
            which buys accuracy in triangles this then spends back where
            the surface is flat. None returns the full triangulation.
        close : bool or str
            Whether to close the surface where it runs out of the model, so
            that what comes back is a body rather than a sheet with a hole
            in the side -- `"above"` (or `True`) keeps the region where the
            values exceed `value`, a grade shell, and `"below"` the region
            under it, as on a grid. Done with a shell of ghost cells
            mirroring the boundary blocks, each its partner's own size, so
            the closing cap cannot tear whatever the refinement did to the
            boundary.

        Returns
        -------
        surf : Solid3D, Surface3D or Mesh3D
            Whichever the geometry calls for, as `get_contour` on a grid.
        """
        path, _ = _contour_column(self, path)

        # the export records each column's label beside its path, so the
        # label is looked up rather than reconstructed -- rebuilding it by
        # hand is what used to break whenever the spelling changed
        mesh = self.as_pyvista()
        table = {}
        if "geoml_paths" in mesh.field_data:
            table = _json.loads(str(mesh.field_data["geoml_paths"][0]))
        label = {p: lb for lb, p in table.items()}.get(str(path))
        if label is None or label not in mesh.cell_data:
            raise ValueError(
                "nothing under %r to contour; the mesh carries %s"
                % (str(path),
                   ", ".join(sorted(mesh.cell_data.keys())) or "nothing"))

        value = float(value)
        values = _np.asarray(mesh.cell_data[label], dtype=float)
        span = (float(_np.nanmin(values)), float(_np.nanmax(values)))
        fill = cap = None
        if close:
            fill = _closing_value(close, values)
            cap = 1 if fill < value else -1

        origin, size, step, cell_values = self._cut_to_contour(
            values, value, supersample=supersample, cap=cap)
        if origin is None:
            origin, size = self._origin, self._size
            step, cell_values = self.base_step, values
        else:
            # let the coarse mesh go before the finer one is built: a block
            # model is large enough that holding both is worth avoiding
            mesh = None

        if close:
            # the box measured in whichever lattice the mesh is drawn on
            shape = _np.asarray(self.lattice_shape) * _np.round(
                self.base_step / step).astype(int)
            ghost_origin, ghost_size = _ghost_shell(origin, size, shape)
            origin = _np.concatenate([origin, ghost_origin])
            size = _np.concatenate([size, ghost_size])
            cell_values = _np.concatenate(
                [cell_values, _np.full(len(ghost_origin), fill)])

        if close or mesh is None:
            mesh = self._hex_mesh(origin, size, step)
            mesh.cell_data[label] = cell_values

        surface = mesh.cell_data_to_point_data().contour(
            [value], scalars=label)
        if surface.n_cells == 0:
            raise ValueError(
                "no surface at %g; %r runs from %g to %g"
                % (value, label, span[0], span[1]))

        surface = surface.triangulate()
        verts = _np.asarray(surface.points, dtype=float)
        faces = _np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
        surface = mesh3d(verts, faces, _gmt.vertex_normals(verts, faces))
        if simplify is not None:
            surface = surface.simplify(simplify)
        return surface


class RotatedBlockSet3D(BlockSet3D):
    """
    A variable-size block model rotated about its starting block.

    The lattice is `BlockSet3D`'s, untouched: splitting, grouping, the
    refinement criteria and the integer arithmetic all happen in the
    unrotated frame, which is what keeps them exact. The rotation is applied
    where coordinates *leave* -- the block centres, the sub-block fan-out a
    prediction reads, the exported hexahedra -- and removed where coordinates
    *come in* (`index_data`, and so `aggregate`). Every mesh test and
    assignment reads sub-block positions through `get_batched_coordinates`,
    so geometry against surfaces and solids works in world coordinates with
    nothing overridden.
    """

    def __init__(self, start, n, step, azimuth=0.0, dip=0.0, rake=0.0,
                 discretization=(2, 2, 2), max_levels=3,
                 labels=("X", "Y", "Z")):
        self.azimuth = float(azimuth)
        self.dip = float(dip)
        self.rake = float(rake)
        # the pivot: the first coarse block's centre, as `RotatedGrid3D`
        # turns about its own origin
        self._pivot = _np.asarray(start, dtype=float)
        super().__init__(start, n, step, discretization=discretization,
                         max_levels=max_levels, labels=labels)

    def rotation_matrix(self):
        return _gmt.rotation_matrix(self.azimuth, self.dip, self.rake)

    def _to_world(self, coordinates):
        return _np.matmul(coordinates - self._pivot,
                          self.rotation_matrix()) + self._pivot

    def _centres(self):
        return self._to_world(super()._centres())

    def _lattice_bounding_box(self):
        corner = self.box_corner
        far = corner + self.lattice_shape * self.base_step
        corners = _np.array(list(_iter.product(
            *zip(corner, far))), dtype=float)
        return BoundingBox.from_array(self._to_world(corners))

    def index_data(self, data):
        # into the lattice frame: the inverse of the map the centres left by
        return super().index_data(
            rotate(data, self._pivot, self.azimuth, self.dip, self.rake,
                   reverse=True))

    def get_batched_coordinates(self, index=None):
        if index is None:
            index = _np.arange(self._n_data)
        # fan out in the lattice frame, where a sub-block is an axis-aligned
        # offset, then rotate every row: the offsets turn with the blocks
        centres = BlockSet3D._centres(self)[index]
        size = self.block_size[index]
        coords = centres[:, None, :] + self._sub_grid[None, :, :] \
            * size[:, None, :]
        coords = self._to_world(coords.reshape(-1, 3))
        splits = None if self.rows_per_location == 1 else len(centres)
        return coords, splits

    def _hex_mesh(self, origin, size, step):
        # built on the lattice, turned as one piece: the welding is by shared
        # corner indices, which a rotation cannot tear
        mesh = super()._hex_mesh(origin, size, step)
        mesh.points = self._to_world(_np.asarray(mesh.points, dtype=float))
        return mesh

    @classmethod
    def from_data(cls, data, step, margin=0.1, decimals=0,
                  discretization=(2, 2, 2), max_levels=3):
        """
        A rotated block model fitted to another object's spread.

        As `RotatedGrid3D.from_data` -- the rotation fitted to the points and
        rounded to `decimals` (degrees) before the box is measured -- counting
        blocks rather than nodes.
        """
        points, azimuth, dip, rake = _fitted_rotation(data, decimals)
        mat = _gmt.rotation_matrix(azimuth, dip, rake)

        centre = _np.mean(points, axis=0, keepdims=True)
        unrotated = _np.matmul(points - centre, mat.T)

        margin = _np.asarray(margin, dtype=float)
        if margin.shape == ():
            margin = _np.full([2, 3], float(margin))
        elif margin.shape == (2,):
            margin = _np.stack([margin] * 3, axis=1)

        low = unrotated.min(axis=0)
        high = unrotated.max(axis=0)
        extent = high - low
        low = low - extent * margin[0]
        high = high + extent * margin[1]

        step = _np.broadcast_to(
            _np.asarray(step, dtype=float).ravel(), (3,))
        n = _np.maximum(
            _np.ceil((high - low) / step - _TOL_COVER).astype(int), 1)

        # `start` is the first block's centre, half a step past the corner
        start = _np.squeeze(
            _np.matmul((low + step / 2)[None, :], mat) + centre)
        start = _np.round(start, decimals)

        labels = getattr(data, "coordinate_labels", None)
        return cls(start=start, n=n, step=_np.array(step), azimuth=azimuth,
                   dip=dip, rake=rake, discretization=discretization,
                   max_levels=max_levels,
                   labels=labels if labels else ("X", "Y", "Z"))


