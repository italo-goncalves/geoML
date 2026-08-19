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
What a set of figures is about, before anything is drawn.

A container, a choice of variables, and the colours they are to be drawn in --
none of which depends on what does the drawing. `Explorer` renders it with
matplotlib and `Interactive` with plotly, and they are meant to answer the same
questions the same way, so what the question *is* lives here rather than twice
over.

Nothing in this module imports a plotting library.
"""
import numpy as _np

import geoml.data as _data
import geoml.plots.prepare as _prep
import geoml.plots.style as _style


class Selection(object):
    """
    One data set and a choice of variables, held on to.

    Parameters
    ----------
    data
        Any container from the `data` module.
    continuous : str
        The name of a continuous or vector variable. `pairs` and `pca` need a
        vector one, having nothing to pair otherwise.
    categorical : str
        The name of a categorical variable, used to split and colour the rest.
    model
        A trained model, for the figures that need one.
    palette : list or dict
        The colours for the categories. A list is taken in order, cycling if
        there are more categories than colours; a dict names them, which is
        what a mapping convention wants -- `{"granite": "#d99", "basalt":
        "#48a"}` -- and any category it leaves out falls back to the package
        palette. Defaults to `style.PALETTE`.
    cmap : str or Colormap
        The colours for a continuous value. Defaults to `style.SEQUENTIAL`.
    uncertainty : str or array
        Where to read how sure the model is at each location, for the figures
        that can leave out what it is unsure of. Which column that is depends
        on what was modelled -- `"latent_variance"` on a continuous variable,
        `"uncertainty"` on a vector or categorical one, or a metadata column
        written alongside -- so it is named rather than guessed at. A bare name
        is looked for on the variable being drawn and on the one containing it,
        then in the metadata; a path (`"Variable/column"`) says exactly
        where; and an
        array of one value per location is taken as it comes. See
        `prepare.uncertainty_values`.
    """

    def __init__(self, data, continuous=None, categorical=None, model=None,
                 palette=None, cmap=None, uncertainty=None):
        self.data = data
        self.model = model
        # measurement-level simulations, per variable, built on demand by
        # `_measurement_samples` and never stored on the container
        self._measurements = None
        self.palette = palette
        self.cmap = cmap if cmap is not None else _style.SEQUENTIAL
        self.uncertainty = uncertainty

        self.continuous = None
        if continuous is not None:
            self.continuous = _prep.variable(data, continuous)

        self.categorical = None
        if categorical is not None:
            self.categorical = _prep.variable(data, categorical)
            # Fails here rather than halfway through drawing -- and a variable
            # carrying no category at all is the case worth catching, since
            # what it produces is not an error but a figure with nothing on
            # it. Every series is drawn per category, so none of them means no
            # traces, silently.
            _, _, present = _prep.category_values(self.categorical)
            if len(present) == 0:
                raise ValueError(
                    "%r has no category measured anywhere, so there is "
                    "nothing to split or colour by. Its labels are %s -- if "
                    "those look like the values rather than the categories, "
                    "they went in as `labels` where they were meant to be "
                    "`measurements`"
                    % (categorical,
                       ", ".join(str(label) for label
                                 in getattr(self.categorical, "labels",
                                            [])[:5]) or "empty"))

    def __repr__(self):
        return "%s(%s, continuous=%r, categorical=%r)" % (
            type(self).__name__,
            type(self.data).__name__,
            getattr(self.continuous, "name", None),
            getattr(self.categorical, "name", None))

    # ------------------------------------------------------------------ #
    # what there is to draw
    # ------------------------------------------------------------------ #
    def _require_continuous(self, caller):
        if self.continuous is None:
            raise ValueError("%s needs a continuous variable; the %s was "
                             "built without one"
                             % (caller, type(self).__name__))
        return self.continuous

    @staticmethod
    def _check_measured(var, measured):
        """Nothing measured means nothing to explore, and a clearer error than
        the one an empty array eventually raises somewhere in NumPy."""
        if not _np.any(measured):
            raise ValueError(
                "%r has no measurements here; a grid predicted onto carries "
                "values everywhere and observations nowhere, and this figure "
                "is a comparison against what was observed" % var.name)

    def _require_categorical(self, caller):
        if self.categorical is None:
            raise ValueError("%s needs a categorical variable; the %s was "
                             "built without one"
                             % (caller, type(self).__name__))
        return self.categorical

    def _require_model(self, caller):
        if self.model is None:
            raise ValueError("%s needs a model; the %s was built without one"
                             % (caller, type(self).__name__))
        return self.model

    def _measurement_samples(self, var, caller, n_sim=20):
        """What a measurement at each location would read, from the model.

        A prediction reports the ground, with the likelihood noise integrated
        out, so the simulations stored on the container are intervals for a
        quantity no assay observes; scoring them against measured values asks
        the model a question it did not answer. This asks the right one.

        Kept for as long as this selection lives, since every component of a
        vector variable wants the same array, and a figure redrawn wants it
        again. Nothing reaches the container.
        """
        model = self._require_model(caller)
        if self._measurements is None:
            self._measurements = model.predict_measurements(
                self.data, n_sim=n_sim)
        if var.name not in self._measurements:
            raise ValueError(
                "%s has nothing to say about %r: its likelihood carries no "
                "warping, so there is no value for a measurement of it to "
                "scatter around" % (caller, var.name))
        return self._measurements[var.name]

    def _require_vector(self, caller):
        var = self._require_continuous(caller)
        if not isinstance(var, _data.VectorVariable):
            raise TypeError(
                "%s needs a vector variable to pair up; %r is a %s"
                % (caller, var.name, type(var).__name__))
        return var

    @staticmethod
    def _check_kind(kind):
        if kind not in ("scatter", "hist2d"):
            raise ValueError(
                "kind must be 'scatter' or 'hist2d'; got %r" % kind)

    @staticmethod
    def _is_categorical(var):
        return isinstance(var, (_data.RockTypeVariable, _data.BinaryVariable))

    # ------------------------------------------------------------------ #
    # colours
    # ------------------------------------------------------------------ #
    def _color(self, index, label):
        """The colour of one category: the user's, or the package's."""
        if isinstance(self.palette, dict):
            return self.palette.get(label, _style.color(index))
        if self.palette is not None:
            return self.palette[index % len(self.palette)]
        return _style.color(index)

    def _series(self) -> "list[tuple]":
        """What to draw as separate colours: the categories, or everything.

        One pair per series, `(label, mask)`. Without a categorical variable
        there is one nameless series over every location, and its mask is
        `None` rather than an array of ones -- the callers that index with it
        are the ones that checked `self.categorical` first.
        """
        if self.categorical is None:
            return [(None, None)]
        values, measured, labels = _prep.category_values(self.categorical)
        return _prep.groups(values, measured, labels)

    def _color_variable(self, color):
        """
        What to colour a scene by.

        A named `color` may be a variable or one component of a vector
        variable, so that `scene(color="Zn")` works without reaching into
        `Elements` by hand. Asked for nothing, it takes what is held -- but a
        vector variable is several numbers per location and not one colour
        scale, so it says so rather than quietly drawing the first component
        under the whole variable's name.
        """
        if isinstance(color, (list, tuple)):
            # `Interactive` overrides this by never getting here: a list means
            # a menu, which is a thing a drawn figure can do and a printed one
            # cannot
            raise TypeError(
                "%s draws one variable at a time; a list of them is a menu "
                "to choose between, which only `Interactive` can offer"
                % type(self).__name__)

        if color is not None:
            return _prep.variable_or_component(self.data, color)

        var = self.continuous if self.continuous is not None \
            else self.categorical
        if var is None:
            raise ValueError("scene needs a variable to colour by")

        if not self._is_categorical(var) and var.length > 1:
            if self.categorical is not None:
                return self.categorical
            raise ValueError(
                "%r has %d components, which is not one colour scale; pass "
                "color= with one of %s -- or, on an `Interactive`, a list of "
                "them, or [%r] for all of them, to get a menu to switch "
                "between"
                % (var.name, var.length,
                   _prep.component_names(var), var.name))
        return var
