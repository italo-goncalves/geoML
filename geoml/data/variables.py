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
The variable family: `_Variable` and the concrete kinds a container holds
(continuous, vector, compositional, categorical, binary), each declaring
its own columns for the tree machinery in `base` to fold over. Constructed
by a container's `add_*_variable` methods, never directly.
"""
import collections as _col

import numpy as _np
import pandas as _pd
import tensorflow as _tf
import sklearn.metrics as _skmetrics

import geoml.metrics as _gmlmetrics
import geoml.storage as _storage

from geoml.data.base import *
from geoml.data.base import (
    _Attribute, _TreeNode, _carry_rows, _copy_for_subset, _encode,
    _path_key, _subset_simulations)

class _Variable(_TreeNode):
    """Representation of a dependent random variable."""

    # Subclasses that can be simulated replace this with an (n_data, n_sim)
    # store; declared here so the accessors below work on every variable.
    simulations = None

    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        self._length = 1
        self.metrics = None

    # What this variable's ``labels`` are called when it prints itself.
    _LABEL_KIND = "labels"

    def __repr__(self):
        s = "%s(%r, n_data=%s" % (
            self.__class__.__name__, self.name, self.coordinates.n_data)
        if self.n_sim > 0:
            s += ", n_sim=%d" % self.n_sim
        return s + ")"

    def __str__(self):
        """
        The variable, what it is made of, and what can be read off it.

        ``repr`` says which variable this is; this says what is on it, so that
        the columns do not have to be looked up in the source -- a variable
        carries different ones depending on what it models.

        Only the names are listed, not whether anything has been written into
        them: knowing that means reading each column, and a column on a large
        model lives on disk, which is too much to do for a print.
        """
        lines = [repr(self)]

        labels = getattr(self, "labels", None)
        if labels is not None:
            lines.append("    %s: %s" % (
                self._LABEL_KIND, ", ".join(str(label) for label in labels)))

        columns = [name for name, value in vars(self).items()
                   if isinstance(value, _Attribute)]
        if len(columns) > 0:
            lines.append("    columns: " + ", ".join(columns))

        for name in ("quantiles", "probabilities"):
            keys = getattr(self, name, None)
            if keys:
                lines.append("    %s: %s"
                             % (name, ", ".join(str(key) for key in keys)))

        if self.n_sim > 0:
            lines.append("    simulations: %d" % self.n_sim)

        return "\n".join(lines)

    @property
    def length(self):
        return self._length

    @classmethod
    def from_variable(cls, coordinates, variable):
        raise NotImplementedError

    def split_shares(self):
        """Which decisions this variable carries, and how often each cuts a
        block in two.

        `{name: (n_data,) array}`, the name saying which decision *within*
        this variable -- a cut-off for a grade, a boundary for a category --
        and empty where the variable declares none.

        Asked of the variable rather than worked out from outside, which used
        to be a necessity (`divided` was an `_Attribute` on a category and an
        `OrderedDict` on a grade) and is now only about the names: the
        storage is one shape, but a grade's decision renders `@ 1.5` because
        someone declared that number, while a category's stays bare -- its
        zero is an artefact of the log-odds, and `granite @ 0` would read as
        noise. `_share_label` is that one difference.
        """
        shares = {}
        for cutoff, attribute in (getattr(self, "divided", None) or {}).items():
            shares[self._share_label(cutoff)] = attribute.values.to_numpy()
        for label, component in (
                getattr(self, "components", None) or {}).items():
            for key, values in component.split_shares().items():
                shares[("%s %s" % (label, key)).strip()] = values
        return shares

    def _share_label(self, cutoff):
        return "@ %g" % cutoff

    def _fill_pyvista(self, write, simulations=False, prefix=None,
                      include="**"):
        """`write(label, attribute)` for every filled column, named by path.

        The same enumeration the data frame reads (`_export_leaves`), handed
        to a writer instead of a frame -- in place of the twenty-one
        per-class methods that each named their columns again, no two
        agreeing on the spelling. A label is `render(path, "pretty")`:
        `assay - Zn - noise_variance`, the same segments every other export
        uses.
        """
        root = None if prefix is None \
            else VariablePath(prefix) / self._node_name
        for path, attribute in self._export_leaves(include, simulations, root):
            write(render(path, "pretty"), attribute)

    def fill_pyvista_cube(self, cube, prefix=None, sigma=None,
                          simulations=False, include="**"):
        self._fill_pyvista(
            lambda label, attribute: attribute.fill_pyvista_cube(
                cube, label, sigma=sigma),
            simulations, prefix, include)

    def fill_pyvista_points(self, points, prefix=None, simulations=False,
                            include="**"):
        self._fill_pyvista(
            lambda label, attribute: attribute.fill_pyvista_points(
                points, label),
            simulations, prefix, include)

    def fill_pyvista_blocks(self, cube, prefix=None, sigma=None,
                            simulations=False, include="**"):
        self._fill_pyvista(
            lambda label, attribute: attribute.fill_pyvista_blocks(
                cube, label, sigma=sigma),
            simulations, prefix, include)

    def fill_pyvista_cells(self, mesh, prefix=None, simulations=False,
                           include="**"):
        self._fill_pyvista(
            lambda label, attribute: attribute.fill_pyvista_cells(mesh, label),
            simulations, prefix, include)

    def carry_to(self, coordinates, keep, n_new):
        """This variable on a longer set of locations, keeping what still fits.

        The locations of `coordinates` are the `keep` ones of this variable, in
        their old order, followed by `n_new` that did not exist before. The
        first take their values across; the rest come back missing, which is
        what marks them as still to be predicted.

        Used when a block set is refined: a block that was not split is the
        same block on the same support, and its value is still the right
        answer for it, so re-predicting it would spend the work to arrive at
        the number already there.
        """
        new = self.__class__.from_variable(coordinates, self)
        self._copy_attrs_into(new)
        self._carry_into(new, _np.asarray(keep, dtype=bool))
        return new

    def _carry_into(self, new, keep):
        """Fill an already-built variable with the `keep` rows of this one.

        Apart from `carry_to` so that a variable holding components fills the
        ones its own `from_variable` just created, rather than building a
        second set that would go nowhere.
        """
        n_kept = int(_np.count_nonzero(keep))

        for role in self._ZARR_ATTRS:
            old = getattr(self, role, None)
            fresh = getattr(new, role, None)
            if old is None or fresh is None or not old._has_content():
                continue
            fresh.labels = old.labels
            fresh.values[:n_kept] = _np.asarray(old.values)[keep]

        if self._ZARR_HAS_SIMS and self.simulations is not None:
            new.allocate_simulations(self.simulations.shape[1])
            _carry_rows(self.simulations, new.simulations, keep)

        for family in self._DICT_FAMILIES:
            target = getattr(new, family)
            for key, attr in (getattr(self, family, None) or {}).items():
                fresh = self._Attribute(new.coordinates)
                fresh.values[:n_kept] = _np.asarray(attr.values)[keep]
                target[key] = fresh

        for name, component in (getattr(self, "components", None) or {}).items():
            component._carry_into(new.components[name], keep)

    def _subset_into(self, new, item):
        """Fill a copy with the `item` rows of this node, column by column.

        Driven by the declarations rather than written out per class: a
        column named in `_ZARR_ATTRS` and forgotten in a hand-written subset
        survives at its old length, and only whatever reads it afterwards
        finds out. That is how a `BinaryVariable` shipped a subset whose
        `probability` was never cut -- the old override wrote it into a dead
        `average` attribute instead.
        """
        for role in self._ZARR_ATTRS:
            column = getattr(self, role, None)
            if column is not None:
                setattr(new, role, column[item])
        for family in self._DICT_FAMILIES:
            target = getattr(new, family)
            for key, attribute in (getattr(self, family, None) or {}).items():
                target[key] = attribute[item]
        if getattr(self, "simulations", None) is not None:
            new.simulations = _subset_simulations(self.simulations, item)
        theirs = new.child_nodes()
        for label, child in self.child_nodes().items():
            child._subset_into(theirs[label], item)

    def __getitem__(self, item):
        new_obj = _copy_for_subset(self)
        self._subset_into(new_obj, item)
        return new_obj

    def get_measurements(self):
        raise NotImplementedError

    def get_simulations(self):
        raise NotImplementedError

    @property
    def n_sim(self):
        """Number of simulations available, or 0 if none were drawn."""
        return 0 if self.simulations is None else self.simulations.shape[1]

    def simulation(self, index):
        """
        A single simulation, in the form of an `_Attribute`.

        Simulations are kept in one `(n_data, n_sim)` array, so a single one is
        not an attribute in its own right. This wraps the corresponding column,
        giving it the usual helpers (`as_image()`, `as_cube()`, `smooth()`,
        `get_contour()`, `draw_*()`, ...).

        Parameters
        ----------
        index : int
            Position of the simulation, from 0 to `n_sim - 1`.

        Returns
        -------
        attribute : _Attribute
            A copy of the simulation. Modifying it (with `smooth()`, for
            instance) does not affect the stored simulations; assign it back
            with `variable.simulations[:, index] = attribute.values` if that is
            the intention.
        """
        if self.simulations is None:
            raise NoDataError(
                f'No simulations available for variable {self.name}.')

        return self._Attribute(self.coordinates, self.simulations[:, index])

    def get_predictions(self):
        raise NotImplementedError

    def prediction_input(self):
        return {}

    def training_input(self, idx=None):
        return {}

    def copy_to(self, coordinates):
        new = self.__class__.from_variable(coordinates, self)
        self._copy_attrs_into(new)
        coordinates.variables[self.name] = new

    def update(self, idx, **kwargs):
        raise NotImplementedError

    def allocate_simulations(self, n_sim):
        raise NotImplementedError

    def compute_metrics(self, **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Zarr persistence (see _SpatialData.to_zarr / _SpatialData.open)
    # ------------------------------------------------------------------ #
    # Scalar ``_Attribute`` roles to persist; overridden per subclass.
    _ZARR_ATTRS = ()
    _ZARR_HAS_SIMS = False        # has a (n_data, n_sim) simulations store

    def _save_attr(self, group, prefix, role):
        """Write one ``_Attribute``'s store into ``group``; None-valued -> skip.

        String/object attributes are stored as fixed-length unicode; everything
        else is streamed as a numeric/bool Zarr array.
        """
        attr = getattr(self, role, None)
        if attr is None:
            return None
        key = prefix + "/" + role
        store = attr.values
        if _np.dtype(store.dtype) == object:
            unicode = _np.asarray(store).astype(str)
            if unicode.dtype.itemsize == 0:
                unicode = unicode.astype("<U1")
            target = group.create_array(
                name=key, shape=unicode.shape, chunks=unicode.shape,
                dtype=unicode.dtype)
            target[:] = unicode
            return {"key": key, "encoding": "str"}
        store.write_into(group, key)
        info = {"key": key, "encoding": "array"}
        if attr.labels is not None:
            # a coded attribute's categories are its own — a measurement
            # column's are not the variable's. Stringified, as the variable's
            # own labels already are: this goes into JSON, and categories can
            # come out of pandas as NumPy integers.
            info["labels"] = [str(label) for label in attr.labels]
        return info

    def _load_attr(self, group, info):
        z = group[info["key"]]
        if info["encoding"] == "str":
            return _storage.ArrayStore.from_numpy(
                _np.asarray(z[:]).astype(object))
        return _storage.ArrayStore.wrap_zarr(z)

    def _zarr_save(self, group, prefix):
        # str(name): a component is named after its category, which pandas can
        # hand over as a NumPy integer, and this is JSON
        meta = {"class": type(self).__name__, "name": str(self.name),
                "attrs": {}}
        if hasattr(self, "labels"):
            meta["labels"] = [str(x) for x in self.labels]
        for role in self._ZARR_ATTRS:
            info = self._save_attr(group, prefix, role)
            if info is not None:
                meta["attrs"][role] = info
        if self._ZARR_HAS_SIMS and self.simulations is not None:
            key = prefix + "/simulations"
            self.simulations.write_into(group, key)
            meta["simulations"] = key
        # the dict families and the node's own facts, off the declarations,
        # with the Zarr key spelling the same string `get` takes:
        # `assay/Zn/quantiles/0.5`
        for family in self._DICT_FAMILIES:
            entries = []
            for at, attr in (getattr(self, family, None) or {}).items():
                key = prefix + "/" + family + "/" + _path_key(at)
                attr.values.write_into(group, key)
                entries.append({"key": key, "at": float(at)})
            if entries:
                meta.setdefault("dicts", {})[family] = entries
        facts = {name: value for name, value in self.node_attrs().items()
                 if value is not None}
        if facts:
            meta["node_attrs"] = facts
        if getattr(self, "components", None):
            meta["components"] = {}
            for cname, comp in self.components.items():
                # str: this is a JSON key, and a category can be a NumPy
                # integer. The labels above are stringified for the same
                # reason, so the rebuilt variable's components match.
                meta["components"][str(cname)] = comp._zarr_save(
                    group, prefix + "/" + str(cname))
        return meta

    def _zarr_load(self, group, prefix, meta):
        for role, info in meta.get("attrs", {}).items():
            attribute = getattr(self, role)
            store = self._load_attr(group, info)
            if attribute.labels is not None and info["encoding"] == "str":
                # written before the categorical attributes held codes; a
                # value outside the variable's labels is lost here, there
                # being nothing else to read the categories from
                store = _storage.ArrayStore.from_numpy(
                    _encode(_np.asarray(store), attribute.labels))
            if "labels" in info:
                attribute.labels = list(info["labels"])
            attribute.values = store
        if meta.get("simulations") is not None:
            self.simulations = _storage.ArrayStore.wrap_zarr(
                group[meta["simulations"]])
        for family, entries in meta.get("dicts", {}).items():
            target = getattr(self, family)
            for info in entries:
                attr = self._Attribute(self.coordinates)
                attr.values = _storage.ArrayStore.wrap_zarr(group[info["key"]])
                target[info["at"]] = attr
        for name, value in meta.get("node_attrs", {}).items():
            setattr(self, name, value)
        for cname, cmeta in meta.get("components", {}).items():
            self.components[cname]._zarr_load(
                group, prefix + "/" + str(cname), cmeta)


# `_Attribute` used to be defined inside `_Variable`; the alias keeps its
# `self._Attribute(...)` call sites working.
_Variable._Attribute = _Attribute


class ContinuousVariable(_Variable):
    """
    Representation of a continuous random variable.

    Attributes
    ----------
    measurements : _Attribute
        The raw measurements.
    latent_mean : _Attribute
        The mean of the latent Gaussian representation.
    latent_variance : _Attribute
        The variance of the latent Gaussian representation.
    dispersion : _Attribute
        How much the locations inside each block differ among themselves --
        the variance over a block's sub-blocks, averaged over the
        realizations, in the variable's own units rather than the latent
        ones. A different question from `latent_variance`, which is how sure
        the model is *of* the block: a well-known block can still be
        heterogeneous, and that is what decides whether cutting it finer
        would tell anyone anything. Filled only where the container
        discretizes; elsewhere a location has no interior and this stays
        missing rather than zero.
    noise_variance : _Attribute
        How far a fresh *measurement* here would fall from the value above --
        the likelihood noise carried into the variable's own units, averaged
        over the realizations. The third of three variances and the third
        question: `latent_variance` is how sure the model is of the value,
        `dispersion` is how much the ground varies inside a block, and this is
        how much a sample of it would scatter. A prediction reports the ground,
        with the noise integrated out, so this is what has to be added back to
        compare against an assay. Missing where the prediction was made with
        `include_noise=False`, there being no integration to read it from.
    simulations : ArrayStore
        Draws from the variable's posterior distribution, in a single
        `(n_data, n_sim)` array. Use `simulation()` to get one of them as an
        `_Attribute`.
    quantiles : dict
        The variables quantiles, indexed by the corresponding percentile.
    probabilities : dict
        Cumulative distribution probabilities, indexed by the corresponding
        quantile.
    """
    _ZARR_ATTRS = ("measurements", "latent_mean", "latent_variance",
                   "prediction", "dispersion", "noise_variance")
    _ZARR_HAS_SIMS = True
    _DICT_FAMILIES = ("quantiles", "probabilities", "proportions", "divided")
    _NODE_ATTRS = ("cutoffs",)

    def __init__(self, name, coordinates, measurements=None):
        super().__init__(name, coordinates)

        if measurements is None:
            self.measurements = self._Attribute(coordinates)
        else:
            self.measurements = self._Attribute(coordinates, measurements)

        self.latent_mean = self._Attribute(coordinates)
        self.latent_variance = self._Attribute(coordinates)
        self.prediction = self._Attribute(coordinates)

        # How much the locations *inside* each block differ among themselves,
        # which is a different question from how sure the model is of the block
        # (`latent_variance`). Filled only where the container discretizes;
        # everywhere else a location has no interior and this stays zero.
        self.dispersion = self._Attribute(coordinates)

        # What a sample taken here would read, as against what the ground
        # holds: the likelihood noise in this variable's units. A prediction
        # integrates that noise out, so this is the piece to add back before
        # comparing with an assay.
        self.noise_variance = self._Attribute(coordinates)

        # A single (n_data, n_sim) store (NumPy or Zarr by size); None until
        # ``allocate_simulations`` is called.
        self.simulations = None

        self.quantiles = _col.OrderedDict()
        self.probabilities = _col.OrderedDict()

        # The grades a decision turns on -- a mining cut-off, a contaminant
        # limit. Declared on the data and carried to whatever is predicted
        # from it; `None` means this variable takes no part in any decision,
        # which is the answer for the rest component of a composition.
        self.cutoffs = None
        # For each of them, how much of each block sits at or below it, over
        # the sub-blocks and the realizations both -- the recoverable share,
        # and what a partial-block report wants.
        self.proportions = _col.OrderedDict()
        # And for each, how often the cut-off passes *through* the block:
        # the share of realizations whose sub-blocks fall on both sides. A
        # different question, and the one that says whether cutting the block
        # finer would settle anything. See `likelihood._divided`.
        self.divided = _col.OrderedDict()

    def set_cutoffs(self, cutoffs):
        """The grades this variable is judged against.

        They travel with the variable, so a model trained on data that
        declares them hands them to every block model predicted from it, and
        `refine` knows what the blocks have to be resolved against without
        being told a second time.
        """
        self.cutoffs = None if cutoffs is None else \
            [float(c) for c in _np.atleast_1d(cutoffs)]
        return self

    def prediction_input(self):
        return {} if self.cutoffs is None else {"cutoffs": self.cutoffs}

    def get_measurements(self):
        values = self.measurements.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    def get_simulations(self):
        return _np.asarray(self.simulations)

    def get_predictions(self):
        return self.prediction.values.to_numpy()

    def reset_quantiles(self, probabilities=None):
        """
        Resets the variable's quantiles.

        Parameters
        ----------
        probabilities : array
            An array of probabilities, ordered values from 0 to 1 (exclusive),
            on which to compute the corresponding quantiles.
        """
        if self.simulations is None:
            raise NoDataError(f'No simulations available for variable {self.name}.')

        self.quantiles = _col.OrderedDict()
        if probabilities is not None:
            # All quantiles are computed lazily in a single chunk-by-chunk
            # pass over the simulations; the (n_data, n_sim) array is never
            # fully materialized.
            columns = self.simulations.row_quantiles(probabilities)
            targets = []
            for p in probabilities:
                attr = self._Attribute(self.coordinates)
                self.quantiles[p] = attr
                targets.append(attr.values)
            _storage.store_columns(columns, targets)

    def reset_probabilities(self, quantiles=None):
        """
        Resets the variable's probabilities.

        Parameters
        ----------
        quantiles : array
            An array of quantiles (cutoff values in the variable's units),
            ordered, on which to compute the corresponding cumulative
            probabilities in the (0, 1) interval.
        """
        if self.simulations is None:
            raise NoDataError(f'No simulations available for variable {self.name}.')

        self.probabilities = _col.OrderedDict()
        if quantiles is not None:
            # Empirical CDF, the inverse of reset_quantiles: for each cutoff,
            # the fraction of simulations at or below it, in (0, 1). Computed
            # lazily in a single chunk-by-chunk pass. (The previous
            # implementation misused np.percentile, treating the cutoff as a
            # percent rank.)
            columns = self.simulations.row_cdf(quantiles)
            targets = []
            for q in quantiles:
                attr = self._Attribute(self.coordinates)
                self.probabilities[q] = attr
                targets.append(attr.values)
            _storage.store_columns(columns, targets)

    @classmethod
    def from_variable(cls, coordinates, variable):
        # the facts the variable carries -- its cut-offs -- follow separately,
        # by `_copy_attrs_into` in `copy_to`/`carry_to`, off the `_NODE_ATTRS`
        # declaration rather than named here again
        return cls(variable.name, coordinates)

    def update(self, idx, **kwargs):
        # The likelihood speaks in `(rows, components, ...)` whatever the
        # number of components; a scalar variable takes its single column.
        # Flat arrays still arrive from the legacy closed-form model and from
        # a vector variable distributing columns to its parts.
        def column(key):
            arr = kwargs[key].numpy()
            return arr[:, 0] if arr.ndim > 1 else arr

        self.prediction.values[idx] = column("average_sim")

        if "mean" in kwargs.keys():
            self.latent_mean.values[idx] = column("mean")
            self.latent_variance.values[idx] = column("variance")

        if "dispersion" in kwargs.keys():
            self.dispersion.values[idx] = column("dispersion")

        if "noise_variance" in kwargs.keys():
            self.noise_variance.values[idx] = column("noise_variance")

        for key, target in (("proportions", self.proportions),
                            ("divided", self.divided)):
            if key not in kwargs.keys():
                continue
            values = kwargs[key].numpy()
            if values.ndim == 3:
                values = values[:, 0, :]
            cutoffs = self.cutoffs or []
            if values.shape[1] == 0:
                # a component of a vector variable that declared none, whose
                # columns its parent has already trimmed away
                continue
            # the model asked the *training* variable what the cut-offs were,
            # and the answer is being filed against this one; they match
            # unless somebody has moved one of them since
            if len(cutoffs) != values.shape[1]:
                raise ValueError(
                    "%r was predicted against %d cut-off(s) but declares %s; "
                    "set them on the data the model was trained from, and let "
                    "`copy_to` carry them"
                    % (self.name, values.shape[1], self.cutoffs))
            for i, cutoff in enumerate(cutoffs):
                if cutoff not in target:
                    target[cutoff] = self._Attribute(self.coordinates)
                target[cutoff].values[idx] = values[:, i]

        if "simulations" in kwargs.keys():
            # Whole (batch, n_sim) block written as one region into the store.
            sims = kwargs["simulations"].numpy()
            if sims.ndim == 3:
                sims = sims[:, 0, :]
            self.simulations[idx, :] = sims

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

    def compute_metrics(self, alpha=0.05):
        y_true, has_value = self.get_measurements()
        if _np.sum(has_value) == 0:
            raise ValueError('No measurements available')

        y_pred = self.prediction.values.to_numpy()
        sims = self.get_simulations()

        has_value = has_value[:, 0]
        y_true = y_true[has_value == 1]
        y_pred = y_pred[has_value == 1]
        sims = sims[has_value == 1]

        metrics = {
            'Root Mean Square Error (prediction)': _skmetrics.root_mean_squared_error(y_true, y_pred),
            'Mean Absolute Error (prediction)': _skmetrics.mean_absolute_error(y_true, y_pred),
            'Median Absolute Error (prediction)': _skmetrics.median_absolute_error(y_true, y_pred),
            'Root Mean Square Error (simulations)': _skmetrics.root_mean_squared_error(
                _np.broadcast_to(y_true, sims.shape), sims),
            'Mean Absolute Error (simulations)': _skmetrics.mean_absolute_error(
                _np.broadcast_to(y_true, sims.shape), sims),
            'Median Absolute Error (simulations)': _skmetrics.median_absolute_error(
                _np.broadcast_to(y_true, sims.shape), sims),
        }

        bias_2, variance = _gmlmetrics.bias_variance_decomposition(y_true, sims)
        metrics['Bias squared (simulations)'] = bias_2
        metrics['Variance (simulations)'] = variance

        if not isinstance(alpha, (list, tuple)):
            alpha = [alpha]
        for a in alpha:
            metrics[f'Interval score ({a})'] = _gmlmetrics.interval_score(y_true, sims, a)

        self.metrics = _pd.Series(metrics, name=self.name)
        return self.metrics


class VectorVariable(_Variable):
    _ZARR_ATTRS = ("uncertainty",)
    _LABEL_KIND = "components"

    def __init__(self, name, coordinates, labels, measurements=None):
        super().__init__(name, coordinates)

        if isinstance(measurements, _pd.DataFrame):
            measurements = measurements.values

        self.labels = labels
        self._length = len(labels)

        self.components = {}
        for i, label in enumerate(labels):
            self.components[label] = ContinuousVariable(
                label,
                coordinates,
                measurements[:, i] if measurements is not None else None,
            )

        self.uncertainty = self._Attribute(coordinates)

    def get_measurements(self):
        # not allowing partial missing data
        out = [self.components[label].measurements.values.to_numpy()
               for label in self.labels]
        out = _np.stack(out, axis=1)

        has_value = _np.all(~ _np.isnan(out), axis=1, keepdims=True) * 1.0
        has_value = _np.tile(has_value, [1, self.length])
        out = _np.where(has_value == 0.0, 1.0, out)
        return out, has_value

    def prediction_input(self):
        """The components' cut-offs, as one row each.

        They are declared per component -- two grades are judged against two
        different numbers -- but the model sees the variable whole, so they
        travel as a matrix with a row per component. A component declaring
        fewer than the widest is padded with infinity, which nothing is ever
        above, so its spare columns come back empty and `update` drops them.
        """
        declared = [self.components[label].cutoffs or [] for label in
                    self.labels]
        widest = max(len(row) for row in declared) if declared else 0
        if widest == 0:
            return {}
        return {"cutoffs": [row + [_np.inf] * (widest - len(row))
                            for row in declared]}

    def get_simulations(self):
        sims = _np.stack([self.components[v].get_simulations() for v in self.labels], axis=2)
        return sims

    def get_predictions(self):
        pred = _np.stack([self.components[v].get_predictions() for v in self.labels], axis=1)
        return pred

    @classmethod
    def from_variable(cls, coordinates, variable):
        # the components are built fresh by `__init__`; what they *know* --
        # their cut-offs -- follows by `_copy_attrs_into`, which walks the
        # two trees in step so nothing is named here again
        return cls(variable.name, coordinates, variable.labels)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, columns=None,
                        *args, **kwargs):
        new_var = cls(
            name,
            coordinates,
            labels=columns,
            measurements=df.loc[:, columns].values,
        )
        return new_var

    def update(self, idx, **kwargs):
        prediction = _tf.unstack(kwargs["average_sim"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)
        # each component is dispersed inside a block, and measured, on its own
        # account
        dispersion = _tf.unstack(kwargs["dispersion"], axis=1)
        noise = _tf.unstack(kwargs["noise_variance"], axis=1)
        blank = [None] * len(self.labels)
        shares = {
            key: (_tf.unstack(kwargs[key], axis=1)
                  if key in kwargs.keys() else blank)
            for key in ("proportions", "divided")}

        for i, (lb, p, s, d, nv) in enumerate(zip(self.labels, prediction,
                                                  simulations, dispersion,
                                                  noise)):
            values = {"average_sim": p, "simulations": s, "dispersion": d,
                      "noise_variance": nv}
            for key, unstacked in shares.items():
                if unstacked[i] is not None:
                    # the matrix was padded out to the widest component, so
                    # this one takes only the cut-offs it declared
                    declared = len(self.components[lb].cutoffs or [])
                    values[key] = unstacked[i][:, :declared]
            self.components[lb].update(idx, **values)

        self.uncertainty.values[idx] = kwargs["uncertainty"].numpy()

    def allocate_simulations(self, n_sim):
        for comp in self.labels:
            self.components[comp].allocate_simulations(n_sim)

    def reset_quantiles(self, probabilities=None):
        for el in self.labels:
            self.components[el].reset_quantiles(probabilities)

    def reset_probabilities(self, quantiles=None):
        for el in self.labels:
            self.components[el].reset_probabilities(quantiles)

    def compute_metrics(self, alpha=0.05):
        metrics = [self.components[comp].compute_metrics(alpha) for comp in self.labels]
        metrics = _pd.concat(metrics, axis=1)
        metrics.columns = self.labels
        self.metrics = metrics
        return self.metrics


class _Component(ContinuousVariable):
    def __init__(self, name, coordinates, measurements=None):
        super().__init__(name, coordinates, measurements)
        self.latent_mean = None
        self.latent_variance = None

    def update(self, idx, **kwargs):
        self.prediction.values[idx] = kwargs["prediction"].numpy()
        self.simulations[idx, :] = kwargs["simulations"].numpy()

        if "dispersion" in kwargs.keys():
            self.dispersion.values[idx] = kwargs["dispersion"].numpy()

        if "noise_variance" in kwargs.keys():
            self.noise_variance.values[idx] = \
                kwargs["noise_variance"].numpy()

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

    def get_simulations(self):
        return _np.asarray(self.simulations)


class CompositionalVariable(VectorVariable):
    def __init__(self, name, coordinates, labels, measurements=None):
        super().__init__(name, coordinates, labels, measurements)
        for i, label in enumerate(labels):
            self.components[label] = _Component(
                label,
                coordinates,
                measurements[:, i] if measurements is not None else None)

    def get_measurements(self):
        # not allowing partial missing data
        out = [self.components[label].measurements.values.to_numpy()
               for label in self.labels]

        out = _np.stack(out, axis=1)
        total = _np.sum(out, axis=1, keepdims=True)
        total = _np.where(_np.abs(total - 1) < 1e-10, 1.0, _np.nan)
        out = out * total

        has_value = _np.all(~ _np.isnan(out), axis=1, keepdims=True) * 1.0
        has_value = _np.tile(has_value, [1, self.length]).astype(float)
        # out[_np.isnan(out)] = 1
        out = _np.where(has_value == 0.0, 1.0, out).astype(float)
        return out, has_value

        # allowing partial missing data
        # out = [self.components[label].measurements.values
        #        for label in self.labels]
        # out = _np.stack(out, axis=1)
        # has_value = ~ _np.isnan(out)
        # out[_np.isnan(out)] = 1
        # return out, has_value

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels)
        return new_var

    @classmethod
    def from_data_frame(cls, name, coordinates, df, columns=None,
                        *args, **kwargs):
        new_var = cls(
            name,
            coordinates,
            labels=columns,
            measurements=df.loc[:, columns].values)
        return new_var

    def update(self, idx, **kwargs):
        prediction = _tf.unstack(kwargs["average_sim"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)
        # a part varies inside a block, and is assayed, on its own account
        dispersion = _tf.unstack(kwargs["dispersion"], axis=1)
        noise = _tf.unstack(kwargs["noise_variance"], axis=1)

        for lb, p, s, d, nv in zip(
                self.labels,
                prediction, simulations, dispersion, noise):
            self.components[lb].update(idx, **{
                "prediction": p,
                "simulations": s,
                "dispersion": d,
                "noise_variance": nv
            })

        self.uncertainty.values[idx] = kwargs["uncertainty"].numpy()

    def compute_metrics(self, alpha=0.05):
        metrics = [self.components[comp].compute_metrics(alpha) for comp in self.labels]
        metrics = _pd.concat(metrics, axis=1)
        metrics.columns = self.labels

        comp_true, has_value = self.get_measurements()
        comp_pred = _np.stack([self.components[c].prediction.values.to_numpy()
                               for c in self.labels], axis=1)
        comp_true = comp_true[has_value[:, 0] == 1]
        comp_pred = comp_pred[has_value[:, 0] == 1]
        ad = _gmlmetrics.aitchison_distance(comp_true, comp_pred)
        metrics.loc['Aitchison distance', :] = ad

        self.metrics = metrics
        return self.metrics


class _Category(_Variable):
    _ZARR_ATTRS = ("probability", "indicator", "indicator_mean",
                   "indicator_variance", "indicator_predicted")
    _ZARR_HAS_SIMS = True
    _DICT_FAMILIES = ("proportions", "divided")

    def __init__(self, name, coordinates, indicator):
        super().__init__(name, coordinates)
        n_data = coordinates.n_data

        self.probability = self._Attribute(coordinates, _np.zeros(n_data))
        self.indicator = self._Attribute(coordinates, indicator)
        self.indicator_mean = self._Attribute(coordinates)
        self.indicator_variance = self._Attribute(coordinates)
        self.indicator_predicted = self._Attribute(coordinates)
        # How much of each block this category holds, and whether the block
        # is cut in two by this category's boundary -- see
        # `likelihood._divided`. The same dicts a grade keeps, keyed by the
        # one cut-off a category has: zero on `ind_skew`, its log-odds
        # against its best rival, which is not a number anyone declared but
        # the level set the contact *is*. One shape for both kinds, so
        # nothing downstream asks which it is holding. `proportions` is a
        # different thing from `probability`, which is how sure the model is
        # that the block as a whole belongs here.
        self.proportions = _col.OrderedDict()
        self.divided = _col.OrderedDict()
        self.simulations = None

    def _share_label(self, cutoff):
        # the zero is an artefact of the log-odds, not a number anyone
        # declared, so the share keeps a bare name where a grade's says
        # `@ 1.5`
        return ""

    def update(self, idx, **kwargs):
        self.indicator_predicted.values[idx] = kwargs["indicator"].numpy()
        self.indicator_mean.values[idx] = kwargs["mean"].numpy()
        self.indicator_variance.values[idx] = kwargs["variance"].numpy()
        self.probability.values[idx] = kwargs["probability"].numpy()

        for family in ("proportions", "divided"):
            if family in kwargs.keys():
                target = getattr(self, family)
                if 0.0 not in target:
                    target[0.0] = self._Attribute(self.coordinates)
                target[0.0].values[idx] = kwargs[family].numpy()

        self.simulations[idx, :] = kwargs["simulations"].numpy()

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

class RockTypeVariable(_Variable):
    _ZARR_ATTRS = ("predicted", "entropy", "uncertainty",
                   "measurements_a", "measurements_b", "boundary")
    _LABEL_KIND = "categories"

    def __init__(self, name, coordinates, labels=None, measurements_a=None,
                 measurements_b=None):
        # Coerce to numpy arrays: with plain Python lists the element-wise
        # comparisons below (indicator building, boundary detection) would
        # silently collapse to scalars.
        if measurements_a is not None:
            measurements_a = _np.asarray(measurements_a)
        if measurements_b is not None:
            measurements_b = _np.asarray(measurements_b)

        if measurements_b is None:
            measurements_b = measurements_a

        if labels is None:
            if measurements_a is None:
                raise Exception("either the labels or measurements"
                                "must be provided")
            cat_a = _pd.Categorical(measurements_a)
            cat_b = _pd.Categorical(measurements_b)
            labels = _pd.api.types.union_categoricals([cat_a, cat_b])
            labels = labels.categories.values

        n_cat = len(labels)
        n_data = coordinates.n_data

        avg_vals = None
        if measurements_a is not None:
            vals_a = _np.zeros([n_data, n_cat])
            vals_b = vals_a.copy()
            for i, label in enumerate(labels):
                vals_a[measurements_a == label, i] = 1
                vals_b[measurements_b == label, i] = 1
            avg_vals = 0.5 * (vals_a + vals_b)

        super().__init__(name, coordinates)
        self.labels = labels
        self._length = len(labels)

        self.components = {}
        for i, label in enumerate(labels):
            self.components[label] = _Category(
                label,
                coordinates,
                avg_vals[:, i] if avg_vals is not None else None,
            )

        # the categories are known here, so these three hold codes into
        # `labels` rather than one string object per data location
        self.predicted = self._Attribute.encoded(coordinates, labels=labels)
        self.entropy = self._Attribute(coordinates)
        self.uncertainty = self._Attribute(coordinates)

        if measurements_a is None:
            self.measurements_a = self._Attribute.encoded(
                coordinates, labels=labels)
            self.measurements_b = self._Attribute.encoded(
                coordinates, labels=labels)
            self.boundary = self._Attribute(
                coordinates, [False]*n_data, dtype=bool)
        else:
            # their own categories, not the variable's: a measurement outside
            # `labels` is still worth keeping as what it says
            self.measurements_a = self._Attribute.encoded(
                coordinates, measurements_a)
            self.measurements_b = self._Attribute.encoded(
                coordinates, measurements_b)
            self.boundary = self._Attribute(
                coordinates, measurements_a != measurements_b, dtype=bool)

    def get_measurements(self):
        # not allowing partial missing data
        out = [self.components[label].indicator.values.to_numpy()
               for label in self.labels]

        out = _np.stack(out, axis=1)
        total = _np.sum(out, axis=1, keepdims=True)
        total = _np.where(_np.abs(total - 1) < 1e-10, 1.0, _np.nan)
        out = out * total

        has_value = _np.all(~ _np.isnan(out), axis=1, keepdims=True) * 1.0
        has_value = _np.tile(has_value, [1, self.length]).astype(float)
        out = _np.where(has_value == 0.0, 1.0, out).astype(float)
        return out, has_value

    def allocate_simulations(self, n_sim):
        for comp in self.labels:
            self.components[comp].allocate_simulations(n_sim)

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels)
        return new_var

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col_a=None, col_b=None,
                        *args, **kwargs):
        labels = _pd.api.types.union_categoricals(
            [_pd.Categorical(df[col_a]),
             _pd.Categorical(df[col_b])])
        labels = labels.categories.values

        new_var = cls(name, coordinates, labels,
                      measurements_a=df[col_a].values,
                      measurements_b=df[col_b].values)
        return new_var

    def __getitem__(self, item):
        new_obj = _copy_for_subset(self)

        new_obj.entropy = self.entropy[item]
        new_obj.uncertainty = self.uncertainty[item]
        new_obj.boundary = self.boundary[item]
        new_obj.measurements_a = self.measurements_a[item]
        new_obj.measurements_b = self.measurements_b[item]

        # Only the categories still present: slicing is a pre-processing step
        # before training, and a category with no data left in it has nothing
        # to teach the model. The parent's order is kept, since it is the order
        # of the components and of the codes below.
        present = set(new_obj.measurements_a.to_numpy()) \
            | set(new_obj.measurements_b.to_numpy())
        labels = [label for label in self.labels if label in present]
        if len(labels) == 0:
            # nothing is measured here at all — a prediction target, say, whose
            # categories come from the model rather than from the data
            labels = list(self.labels)
        new_obj.labels = labels
        new_obj._length = len(labels)

        # the components and the prediction follow the labels: `update` writes
        # the winning label's position, which the dropped ones would shift
        new_obj.components = {label: self.components[label][item]
                              for label in labels}
        predicted = self.predicted[item]
        predicted.values = _encode(predicted.to_numpy(), labels)
        predicted.labels = list(labels)
        new_obj.predicted = predicted

        return new_obj

    def update(self, idx, **kwargs):
        self.entropy.values[idx] = kwargs["entropy"].numpy()
        self.uncertainty.values[idx] = kwargs["uncertainty"].numpy()

        # the winning category's position is the code
        self.predicted.values[idx] = _np.argmax(
            kwargs["probability"].numpy(), axis=1)

        mean = _tf.unstack(kwargs["mean"], axis=1)
        variance = _tf.unstack(kwargs["variance"], axis=1)
        indicators = _tf.unstack(kwargs["indicators"], axis=1)
        probability = _tf.unstack(kwargs["probability"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)
        # the share of each block this category holds, one column per label
        blank = [None] * len(self.labels)
        proportions = _tf.unstack(kwargs["proportions"], axis=1) \
            if "proportions" in kwargs.keys() else blank
        divided = _tf.unstack(kwargs["divided"], axis=1) \
            if "divided" in kwargs.keys() else blank

        for lb, m, v, i, p, s, share, cut in zip(
                self.labels, mean, variance, indicators,
                probability, simulations, proportions, divided):
            values = {"mean": m, "variance": v, "indicator": i,
                      "probability": p, "simulations": s}
            if share is not None:
                values["proportions"] = share
                values["divided"] = cut
            self.components[lb].update(idx, **values)

    def training_input(self, idx=None):
        if idx is None:
            idx = _np.arange(self.coordinates.n_data)
        return {"is_boundary": _tf.constant(
            self.boundary.values.to_numpy()[idx, None], _tf.bool)}

    def compute_metrics(self, **kwargs):
        y_pred = self.predicted.to_numpy()
        y_true_a = self.measurements_a.to_numpy()
        y_true_b = self.measurements_b.to_numpy()

        valid = y_true_a == y_true_b

        y_pred = y_pred[valid]
        y_true = y_true_a[valid]

        series = []
        for lab in self.labels:
            yp = _np.where(y_pred == lab, 1, 0)
            yt = _np.where(y_true == lab, 1, 0)
            d = {
                'Balanced accuracy': _skmetrics.balanced_accuracy_score(yt, yp),
                'Jaccard': _skmetrics.jaccard_score(yt, yp),
                'Matthews': _skmetrics.matthews_corrcoef(yt, yp),
            }
            series.append(_pd.Series(d, name=lab))

        self.metrics = _pd.concat(series, axis=1)
        return self.metrics


class CategoricalVariable(RockTypeVariable):
    def __init__(self, name, coordinates, labels=None, measurements=None):
        super().__init__(name, coordinates, labels=labels, measurements_a=measurements)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, measurements_col=None,
                        *args, **kwargs):
        labels = _pd.Categorical(df[measurements_col])
        labels = labels.categories.values

        new_var = cls(name, coordinates, labels,
                      measurements=df[measurements_col].values)
        return new_var

class OrderedRockType(RockTypeVariable):
    _ZARR_ATTRS = RockTypeVariable._ZARR_ATTRS + ("implicit_values",)

    def __init__(self, name, coordinates, labels=None, measurements_a=None,
                 measurements_b=None):
        super().__init__(name, coordinates, labels, measurements_a,
                         measurements_b)
        self._length = 1

        if measurements_a is None:
            # Labels-only construction (prediction targets, reloading from
            # disk): no measured contacts, all implicit values missing.
            self.implicit_values = self._Attribute(coordinates)
            return

        measurements_a = _np.asarray(measurements_a)
        if measurements_b is None:
            measurements_b = measurements_a
        else:
            measurements_b = _np.asarray(measurements_b)

        # Labels may have been derived from the measurements by the parent.
        self.implicit_values = self._Attribute(
            coordinates,
            self._implicit_values(measurements_a, measurements_b, self.labels))

    @staticmethod
    def _implicit_values(measurements_a, measurements_b, labels):
        """Where each pair of measurements sits in the label sequence.

        A function of the labels, so it has to be recomputed whenever they
        change — slicing away a category shifts every position after it.
        """
        # Built as a float array from the start; the previous
        # ``-0.5 * _np.ones_like(measurements_a)`` crashed on string inputs.
        implicit_values = _np.full(len(measurements_a), -0.5)
        for i in range(len(labels[:-1])):
            implicit_values = _np.where(
                (measurements_a == labels[i])
                & (measurements_b == labels[i + 1]),
                i,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i + 1])
                & (measurements_b == labels[i]),
                i,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i])
                & (measurements_b == labels[i]),
                i - 0.5,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i + 1])
                & (measurements_b == labels[i + 1]),
                i + 0.5,
                implicit_values
            )
            # implicit_values[(measurements_a == labels[i])
            #                 & (measurements_b == labels[i + 1])] = i
            # implicit_values[(measurements_a == labels[i + 1])
            #                 & (measurements_b == labels[i])] = i
            # implicit_values[(measurements_a == labels[i])
            #                 & (measurements_b == labels[i])] = i - 0.5
            # implicit_values[(measurements_a == labels[i + 1])
            #                 & (measurements_b == labels[i + 1])] = i + 0.5

        return implicit_values

    def get_measurements(self):
        values = self.implicit_values.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    def __getitem__(self, item):
        new_obj = super().__getitem__(item)
        implicit = self.implicit_values[item]
        if new_obj.measurements_a._has_content():
            # positions in the label sequence, and the slice may have dropped
            # labels, so these are recomputed rather than carried over
            implicit.values = self._implicit_values(
                new_obj.measurements_a.to_numpy(),
                new_obj.measurements_b.to_numpy(),
                new_obj.labels)
        new_obj.implicit_values = implicit
        return new_obj


class BinaryVariable(_Variable):
    _ZARR_ATTRS = ("indicator", "measurements", "weights", "predicted",
                   "probability", "entropy", "uncertainty",
                   "latent_mean", "latent_variance")
    _LABEL_KIND = "categories"
    _ZARR_HAS_SIMS = True

    def __init__(self, name, coordinates, labels=None, measurements=None):
        super().__init__(name, coordinates)
        n_data = coordinates.n_data

        # Coerce to a numpy array: with a plain Python list the element-wise
        # comparisons below (indicator and weights) silently collapse to
        # scalars and the indicators stay NaN.
        if measurements is not None:
            measurements = _np.asarray(measurements)

        if labels is None:
            if measurements is None:
                raise Exception("either the labels or measurements"
                                "must be provided")
            cat = _pd.Categorical(measurements)
            labels = cat.categories.values

        self.labels = labels
        self._length = 1
        if len(labels) != 2:
            raise ValueError(f"There must be exactly 2 labels - found {len(labels)}.")

        self.indicator = self._Attribute(
            coordinates, _np.array([_np.nan]*n_data))
        if measurements is None:
            self.measurements = self._Attribute.encoded(
                coordinates, labels=labels)
            self.weights = self._Attribute(coordinates)
        else:
            # their own categories: an `AnomalyVariable` labels everything that
            # is not the anomaly `_dummy`, but the measurements say what it was
            self.measurements = self._Attribute.encoded(
                coordinates, measurements)
            self.weights = self._Attribute(coordinates, _np.ones(n_data))
            self.indicator.values[measurements == labels[0]] = 1
            self.indicator.values[measurements == labels[1]] = 0

        self.predicted = self._Attribute.encoded(coordinates, labels=labels)
        self.probability = self._Attribute(coordinates, _np.zeros(n_data))
        self.entropy = self._Attribute(coordinates)
        self.uncertainty = self._Attribute(coordinates)

        self.latent_mean = self._Attribute(coordinates)
        self.latent_variance = self._Attribute(coordinates)

        self.simulations = None

        if measurements is not None:
            for label in labels:
                idx = measurements == label
                n_in_label = _np.sum(idx)
                if n_in_label > 0:
                    self.weights.values[idx] = \
                        n_data / (n_in_label * len(labels))

    def get_measurements(self):
        values = self.indicator.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels)
        return new_var

    def update(self, idx, **kwargs):
        prob = kwargs["probability"].numpy()
        mean = kwargs["mean"].numpy()
        var = kwargs["variance"].numpy()
        entropy = kwargs["entropy"].numpy()
        uncertainty = kwargs["uncertainty"].numpy()
        sims = kwargs["simulations"].numpy()

        if len(prob.shape) > 1:
            prob = prob[:, 0]
            mean = mean[:, 0]
            var = var[:, 0]
            # entropy = entropy[:, 0]
            # uncertainty = uncertainty[:, 0]
            sims = sims[:, 0, :]

        label_idx = _np.zeros(prob.shape, dtype=int)  # positive class
        label_idx[prob < 0.5] = 1  # negative class

        # the label's position is the code
        self.predicted.values[idx] = label_idx
        self.latent_mean.values[idx] = mean
        self.latent_variance.values[idx] = var
        self.entropy.values[idx] = entropy
        self.uncertainty.values[idx] = uncertainty
        self.probability.values[idx] = prob

        self.simulations[idx, :] = sims

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col, positive_class):
        labels = _pd.Categorical(df[col])
        labels = labels.categories.values.tolist()

        pos = None
        for i, label in enumerate(labels):
            if label == positive_class:
                pos = i
        labels.pop(pos)
        labels.append(positive_class)
        labels = labels[::-1]

        new_var = cls(name, coordinates, labels,
                      measurements=df[col].values)
        return new_var

class AnomalyVariable(BinaryVariable):
    def __init__(self, name, coordinates, label, measurements=None):
        labels = [label, "_dummy"]
        super().__init__(name, coordinates, labels, measurements)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col, positive_class):
        new_var = cls(name, coordinates, positive_class,
                      measurements=df[col].values)
        return new_var

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels[0])
        return new_var


