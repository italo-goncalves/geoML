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
Looking at a data set before modelling it.

An `Explorer` is a choice of data and variables -- one continuous (or vector)
and one categorical -- held on to, so that the figures can be asked for one
after another without repeating it. The categorical variable is what splits and
colours every other figure, which is the question worth asking of most
geoscientific data: does this population behave as one, or as several?

    eda = geoml.plots.Explorer(point, continuous="Elements",
                               categorical="Rock")
    eda.histogram()
    eda.pairs()
    eda.pca(explained=0.9)
    eda.scene()
"""
import matplotlib.colors as _mcolors
import matplotlib.pyplot as _plt
import numpy as _np
import scipy.stats as _stats

import geoml.data as _data
import geoml.metrics as _gmet
import geoml.plots.prepare as _prep
import geoml.plots.style as _style


class Explorer(object):
    """
    Exploratory figures for one data set and a choice of variables.

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
        A trained model, for the figures that need one. Not used yet.
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
        then in the metadata; `"Variable.column"` says exactly where; and an
        array of one value per location is taken as it comes. See
        `prepare.uncertainty_values`.
    """

    def __init__(self, data, continuous=None, categorical=None, model=None,
                 palette=None, cmap=None, uncertainty=None):
        self.data = data
        self.model = model
        self.palette = palette
        self.cmap = cmap if cmap is not None else _style.SEQUENTIAL
        self.uncertainty = uncertainty

        self.continuous = None
        if continuous is not None:
            self.continuous = _prep.variable(data, continuous)

        self.categorical = None
        if categorical is not None:
            self.categorical = _prep.variable(data, categorical)
            # fails here rather than halfway through drawing
            _prep.category_values(self.categorical)

    def __repr__(self):
        return "Explorer(%s, continuous=%r, categorical=%r)" % (
            type(self.data).__name__,
            getattr(self.continuous, "name", None),
            getattr(self.categorical, "name", None))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _require_continuous(self, caller):
        if self.continuous is None:
            raise ValueError("%s needs a continuous variable; the Explorer was "
                             "built without one" % caller)
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

    def _require_model(self, caller):
        if self.model is None:
            raise ValueError("%s needs a model; the Explorer was built "
                             "without one" % caller)
        return self.model

    def _require_vector(self, caller):
        var = self._require_continuous(caller)
        if not isinstance(var, _data.VectorVariable):
            raise TypeError(
                "%s needs a vector variable to pair up; %r is a %s"
                % (caller, var.name, type(var).__name__))
        return var

    def _color(self, index, label):
        """The colour of one category: the user's, or the package's."""
        if isinstance(self.palette, dict):
            return self.palette.get(label, _style.color(index))
        if self.palette is not None:
            return self.palette[index % len(self.palette)]
        return _style.color(index)

    def _series(self):
        """What to draw as separate colours: the categories, or everything."""
        if self.categorical is None:
            return [(None, None)]
        values, measured, labels = _prep.category_values(self.categorical)
        return _prep.groups(values, measured, labels)

    @staticmethod
    def _legend_fraction(figure):
        """
        How much of the width to leave for a legend outside the axes.

        A fixed share is too little on a small figure and too much on a large
        one: a category's name is so many characters wide whatever the figure
        is, so the strip is measured in inches and turned into a fraction.
        """
        return 1.0 - min(0.34, 1.6 / figure.get_figwidth())

    def _legend(self, figure, axes, corner=False, anchor=0.87):
        """
        The categories, named once for the whole figure.

        A matrix leaves its upper triangle empty, so the legend goes in that
        corner. A grid of panels may be full, so there the legend goes outside
        and the layout is told to leave room for it.
        """
        if self.categorical is None:
            return
        handles, labels = axes.get_legend_handles_labels()
        if len(handles) == 0:
            # nothing was drawn per category -- counts pool them
            return
        # the corner sits under the title, which a two-component matrix is
        # small enough to collide with
        figure.legend(handles, labels, title=self.categorical.name,
                      loc="upper right" if corner else "center left",
                      bbox_to_anchor=(0.98, 0.94) if corner else (anchor, 0.5))

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    def histogram(self, bins=25, figsize=None):
        """
        The distribution of the continuous variable, one panel per component.

        Split by category when there is one: the populations are drawn over
        each other with a common set of bins, so their spreads can be compared
        rather than only their shapes.
        """
        var = self._require_continuous("histogram")
        values, measured, labels = _prep.numeric_values(var)
        self._check_measured(var, measured)
        series = self._series()

        rows, columns = _prep.grid_shape(len(labels))
        with _style.context():
            figure, axes = _plt.subplots(
                rows, columns, squeeze=False,
                figsize=figsize or (3.2 * columns, 2.6 * rows))
            flat = axes.ravel()

            for i, label in enumerate(labels):
                column = values[:, i]
                edges = _np.histogram_bin_edges(column[measured], bins=bins)
                for j, (name, mask) in enumerate(series):
                    keep = measured if mask is None else mask
                    flat[i].hist(column[keep], bins=edges,
                                 color=self._color(j, name), alpha=0.65,
                                 label=name)
                flat[i].set_title(label)
                flat[i].set_ylabel("count")

            for extra in flat[len(labels):]:
                extra.set_visible(False)

            fraction = self._legend_fraction(figure)
            self._legend(figure, flat[0], anchor=fraction + 0.02)
            figure.suptitle(var.name)
            figure.tight_layout(
                rect=(0, 0, fraction if self.categorical is not None else 1, 1))
        return figure

    def pairs(self, kind="scatter", alpha=0.7, bins=60, log_counts=False,
              log=False, principal_components=0, upper=None, figsize=None,
              size=6):
        """
        Every component against every other, with its distribution down the
        diagonal.

        Parameters
        ----------
        kind : str
            `"scatter"`, or `"hist2d"` to count the points into cells instead,
            for when there are too many to draw one by one. Counts make a
            single surface, so the categories are pooled.
        alpha : float
            How transparent each point is, for `kind="scatter"`.
        bins : int
            Cells along each axis, for `kind="hist2d"`.
        log_counts : bool
            Colour the cells by the logarithm of the count. Worth turning on
            whenever a few cells hold most of the data, which is most of the
            time with a skewed variable: on a linear scale they take the whole
            colour range and everything else reads as empty.
        log : bool
            Draw the data on a log scale: the centred log-ratio for a
            composition, whose parts carry a constant sum, and an ordinary
            logarithm otherwise. Geochemical data is usually closer to
            symmetric this way, and it is what makes the principal components
            of a composition drawable -- they are directions in the log-ratios,
            so this is the space they belong to.
        principal_components : int
            Draw this many principal components over the data, in the data's
            own axes -- the reverse of `pca`, which puts the data on the
            components' axes. Each is a line through the mean, one standard
            deviation of that component long either way, so its length says
            how much of the spread it accounts for and its slant says which
            measurements move together. The sign of a component means nothing,
            hence a line rather than an arrow.
        upper : str
            What to put in the upper triangle, which is otherwise left empty:
            `"hist2d"` or `"density"` for where the mass is with the
            categories pooled, or `"correlation"` for the coefficient alone.
            The lower triangle says who each point is; the upper says where
            the data as a whole sits.
        """
        self._check_kind(kind)
        var = self._require_vector("pairs")
        values, measured, labels = _prep.numeric_values(var)
        self._check_measured(var, measured)
        series = self._series()
        n = len(labels)

        compositional = isinstance(var, _data.CompositionalVariable)
        if log:
            # only the measured rows: the gaps are padded with 1.0 to keep the
            # array square, and that padding is not data to be checked or
            # transformed
            transformed = values.copy()
            transformed[measured] = _prep.logarithm(values[measured],
                                                    compositional)
            values = transformed
            labels = ["%s(%s)" % ("clr" if compositional else "log", label)
                      for label in labels]

        analysis = None
        if principal_components > 0:
            if compositional and not log:
                raise TypeError(
                    "the principal components of %r are those of its log-"
                    "ratios, which are not directions in the proportions "
                    "drawn here. Pass log=True to draw the log-ratios, which "
                    "is the space they belong to, or use `pca()` to put the "
                    "data on the components' own axes" % var.name)
            analysis = _prep.principal_components(values[measured],
                                                  explained=1.0)

        with _style.context():
            figure, axes = _plt.subplots(
                n, n, squeeze=False, figsize=figsize or (2.0 * n, 2.0 * n))
            self._draw_matrix(axes, values, labels, series, measured, size,
                              kind=kind, alpha=alpha, bins=bins,
                              log_counts=log_counts, upper=upper)

            if analysis is not None:
                for row in range(n):
                    for column in range(row):
                        self._draw_axes_of_variation(
                            axes[row][column], analysis, column, row,
                            principal_components)
            # the corner is only free while the upper triangle is empty
            outside = upper is not None and self.categorical is not None
            fraction = self._legend_fraction(figure)
            self._legend(figure, axes[0][0], corner=upper is None,
                         anchor=fraction + 0.02)
            figure.suptitle("%s by %s" % (var.name, self.categorical.name)
                            if self.categorical is not None else var.name)
            figure.tight_layout(
                rect=(0, 0, fraction if outside else 1, 0.97))
        return figure

    def pca(self, explained=0.9, log=False, kind="scatter", alpha=0.7,
            bins=60, log_counts=False, figsize=None, size=6):
        """
        The same pairs plot, on principal components instead of measurements.

        Only the components carrying `explained` of the variance are drawn.
        Each panel also holds the loadings: an arrow per original column,
        showing what it contributes to the two components on the axes. A
        composition is opened up with the centred log-ratio first -- see
        `prepare.centred_log_ratio` for why.

        Parameters
        ----------
        explained : float
            Share of the total variance to reach, between 0 and 1.
        log : bool
            Take the components of the logarithms rather than of the
            measurements. A composition is opened up either way -- there is no
            useful PCA of proportions, whose covariance the constant sum makes
            singular -- so this decides the matter only for everything else.
        kind, alpha, bins, log_counts
            As in `pairs`.
        """
        self._check_kind(kind)
        var = self._require_vector("pca")
        self._check_measured(var, _prep.numeric_values(var)[1])
        analysis = _prep.component_analysis(var, explained, log)

        scores = analysis["scores"]
        ratio = analysis["ratio"]
        n = analysis["n_components"]
        labels = ["PC%d (%.1f%%)" % (i + 1, 100 * ratio[i]) for i in range(n)]

        measured = analysis["measured"]
        series = [(name, mask[measured]) for name, mask in self._series()] \
            if self.categorical is not None else [(None, None)]

        with _style.context():
            figure, axes = _plt.subplots(
                n, n, squeeze=False, figsize=figsize or (2.2 * n, 2.2 * n))
            self._draw_matrix(axes, scores, labels, series,
                              _np.ones(len(scores), dtype=bool), size,
                              kind=kind, alpha=alpha, bins=bins,
                              log_counts=log_counts)

            for row in range(n):
                for column in range(row):
                    self._draw_loadings(
                        axes[row][column], analysis["loadings"],
                        analysis["labels"], column, row, scores)

            self._legend(figure, axes[0][0], corner=True)
            figure.suptitle("%s — %.0f%% of the variance in %d components"
                            % (var.name, 100 * _np.sum(ratio[:n]), n))
            figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figure

    def scene(self, color=None, figsize=None, size=14):
        """
        Where the data is, coloured by a variable.

        The coordinates decide the drawing: a value against position in 1D, a
        map in 2D, a scatter in 3D. For anything more than a look at a 3D data
        set, `as_pyvista()` on the container is the better road.

        Parameters
        ----------
        color : str
            The variable to colour by. Defaults to the continuous variable the
            Explorer holds, or the categorical one if that is all there is.
        """
        coordinates = _np.asarray(self.data.coordinates)
        n_dim = coordinates.shape[1]
        if n_dim > 3:
            raise ValueError(
                "a scene needs 1, 2 or 3 coordinates; this data has %d"
                % n_dim)

        var = self._color_variable(color)

        labels = getattr(self.data, "coordinate_labels", None) \
            or ["axis %d" % (i + 1) for i in range(n_dim)]

        with _style.context():
            figure = _plt.figure(figsize=figsize or (6.5, 5.5))
            if n_dim == 3:
                axes = figure.add_subplot(projection="3d")
            else:
                axes = figure.add_subplot()

            if self._is_categorical(var):
                self._scene_categories(axes, coordinates, var, n_dim, size)
                axes.legend(title=var.name, loc="best")
            else:
                self._scene_values(figure, axes, coordinates, var, n_dim, size)

            axes.set_xlabel(labels[0])
            if n_dim == 1:
                axes.set_ylabel(var.name)
            else:
                axes.set_ylabel(labels[1])
            if n_dim == 3:
                axes.set_zlabel(labels[2])
                axes.grid(False)
            elif n_dim == 2:
                # a map: the two axes are the same thing, so the same scale
                axes.set_aspect("equal", adjustable="datalim")
            axes.set_title(var.name)
            figure.tight_layout()
        return figure

    # ------------------------------------------------------------------ #
    # figures that need the model
    # ------------------------------------------------------------------ #
    def training_curve(self, window=None, figsize=None):
        """
        The ELBO against the iteration it was measured at.

        Every value is an estimate, from a sample of the latent variables and,
        under `train_svi`, a sample of the data too, so the curve is noisy
        whether or not training has settled. The running mean over it is the
        line to read: flat means finished, still climbing means it is not.

        Parameters
        ----------
        window : int
            Points to average over. Defaults to a fiftieth of the log.
        """
        self._require_model("training_curve")
        curve = _prep.training_curve(self.model, window)

        with _style.context():
            figure, axes = _plt.subplots(figsize=figsize or (6.5, 4.0))
            axes.plot(curve["iteration"], curve["value"], linewidth=0.8,
                      alpha=0.45, color=_style.color(0), label="ELBO")
            if len(curve["smooth"]) < len(curve["value"]):
                axes.plot(curve["smooth_iteration"], curve["smooth"],
                          color=_style.color(1),
                          label="mean of %d" % curve["window"])
            axes.set_xlabel("iteration")
            axes.set_ylabel("ELBO")
            axes.set_title("Training")
            axes.legend(loc="lower right")
            figure.tight_layout()
        return figure

    def transformed_pairs(self, kind="scatter", alpha=0.7, bins=60,
                          log_counts=False, upper=None, figsize=None, size=6):
        """
        The measurements as the model sees them, after its warping.

        Two things worth checking before trusting a fitted model, and both are
        easier to see than to test. Down the diagonal, whether the warping
        made each variable Gaussian: the standard normal it is aiming at is
        drawn over each histogram. Off the diagonal, whether what is left is
        independent: a round cloud with a correlation near zero is what the
        model assumes, and a tilted or curved one is structure it will not
        capture.

        The columns are numbered rather than named: a warping may rotate the
        data or bend it, so a column is generally a mixture of what was
        measured rather than any one of it.

        Parameters
        ----------
        kind, alpha, bins, log_counts
            As in `pairs`.
        upper : str
            What to put in the upper triangle, otherwise left empty:
            `"hist2d"`, `"density"` or `"correlation"`, as in `pairs`.
            `"density"` earns its place here: the contours of a pair the
            warping has done its work on are round and centred, and any lean
            or corner in them is the dependence the model is about to assume
            away.
        """
        self._check_kind(kind)
        self._require_model("transformed_pairs")
        var = self._require_continuous("transformed_pairs")
        self._check_measured(var, _prep.numeric_values(var)[1])
        values, _, labels = _prep.warped_values(self.model, var.name)
        n = len(labels)

        with _style.context():
            figure, axes = _plt.subplots(
                n, n, squeeze=False, figsize=figsize or (2.2 * n, 2.2 * n))
            self._draw_matrix(axes, values, labels, [(None, None)],
                              _np.ones(len(values), dtype=bool), size,
                              density=True, kind=kind, alpha=alpha,
                              bins=bins, log_counts=log_counts, upper=upper)

            for row in range(n):
                self._draw_normal(axes[row][row], values[:, row])
                for column in range(row):
                    self._annotate_correlation(
                        axes[row][column], values[:, column], values[:, row])

            figure.suptitle("%s, as the model sees it" % var.name)
            figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figure

    def simulation_pairs(self, kind="hist2d", bins=60, log_counts=False,
                         most=100000, margin=0.1, alpha=0.6, figsize=None,
                         size=6):
        """
        What was measured against what was simulated.

        The measurements fill the lower triangle and the simulations the
        upper, in the same form and between the same limits, so the two halves
        of the matrix can be read against each other: a simulation that
        reproduces the data has an upper half that mirrors the lower one. Down
        the diagonal the measured histogram carries the simulated density over
        it, which is the same comparison one variable at a time.

        Both halves are counted into cells by default. Simulations come in
        location-by-realization blocks that run to millions of values, and a
        scatter of that many points is a filled rectangle whatever its
        transparency.

        Parameters
        ----------
        kind : str
            `"hist2d"`, the default here, or `"scatter"`.
        bins, log_counts, alpha, size
            As in `pairs`.
        most : int
            About how many simulated values to draw per component. They are
            taken by striding through the block, so the sample spans locations
            and realizations alike; the same stride is used for every
            component, since it is the pairs that are being looked at.
        margin : float
            How far past the measured range to look, as a share of it.
            Simulated values outside that window are left out: a few
            realizations reaching far beyond anything measured would otherwise
            set the scale for every panel and squeeze the comparison into a
            corner of it. They are dropped rather than pinned to the edge,
            which would pile the whole tail into the last cell and read as a
            mode that is not there.
        """
        self._check_kind(kind)
        var = self._require_continuous("simulation_pairs")
        values, measured, labels = _prep.numeric_values(var)
        self._check_measured(var, measured)

        data = values[measured]
        sample = _prep.simulation_sample(var, most)
        n = len(labels)

        # The measured range with room around it, shared by a panel and its
        # mirror -- without which the halves cannot be compared by eye at all.
        # Simulated values outside it are set aside per panel rather than for
        # the whole matrix: a realization that runs away in one component
        # still has something to say about the others.
        limits = _prep.padded_range(data, margin)
        inside = _np.column_stack(
            [(sample[:, i] >= low) & (sample[:, i] <= high)
             for i, (low, high) in enumerate(limits)])

        with _style.context():
            figure, axes = _plt.subplots(
                n, n, squeeze=False, figsize=figsize or (2.1 * n, 2.1 * n))

            for row in range(n):
                for column in range(n):
                    panel = axes[row][column]
                    if row == column:
                        panel.hist(data[:, row], bins=25, density=True,
                                   range=limits[row], color=_style.color(0),
                                   alpha=0.65, label="measured")
                        self._draw_kde(panel, sample[inside[:, row], row],
                                       limits[row])
                    else:
                        measurements = column < row
                        source = data
                        if not measurements:
                            source = sample[inside[:, column]
                                            & inside[:, row]]
                        self._draw_points(
                            panel, source[:, column], source[:, row], size,
                            kind, alpha, self._cells(len(source), bins),
                            log_counts,
                            color=_style.color(0 if measurements else 1))
                        panel.set_ylim(limits[row])
                    panel.set_xlim(limits[column])

                    if row == n - 1:
                        panel.set_xlabel(labels[column])
                    else:
                        panel.set_xticklabels([])
                    if column == 0:
                        panel.set_ylabel(labels[row])
                    else:
                        panel.set_yticklabels([])

            # every panel is in use here, so the legend goes outside
            fraction = self._legend_fraction(figure)
            handles, names = axes[0][0].get_legend_handles_labels()
            figure.legend(handles, names, loc="center left",
                          bbox_to_anchor=(fraction + 0.02, 0.5))
            figure.suptitle(
                "%s: measured below the diagonal, simulated above" % var.name)
            figure.tight_layout(rect=(0, 0, fraction, 0.97))
        return figure

    @staticmethod
    def _cells(n_points, most):
        """
        How many cells to count `n_points` into, at most `most` of them.

        The two halves of a comparison rarely hold comparable numbers -- a few
        hundred measurements against a hundred thousand simulated values -- and
        binning both the same way leaves the sparse half as scattered single
        counts that read as noise. Roughly the square root of the count keeps
        several points in a typical cell either way.
        """
        return int(_np.clip(_np.sqrt(n_points), 10, most))

    def _draw_kde(self, panel, values, limits, most=5000):
        """The simulated density, as a line over the measured histogram."""
        if len(values) > most:
            values = values[::len(values) // most]
        if len(values) < 3 or _np.ptp(values) == 0:
            return

        grid = _np.linspace(limits[0], limits[1], 200)
        panel.plot(grid, _stats.gaussian_kde(values)(grid),
                   color=_style.color(1), linewidth=1.6, label="simulated")

    def prediction_scatter(self, component=None, kind="scatter", alpha=0.6,
                           bins=60, log_counts=False, figsize=None, size=10):
        """
        What was predicted against what was measured.

        A single variable is drawn with its two distributions along the sides,
        which is where a bias shows that the scatter alone hides: the same
        cloud can sit on the 1:1 line while the predicted values are packed
        into a narrower range than the real ones. Several components are drawn
        as a panel each; name one with `component` to get the margins for it.

        Parameters
        ----------
        component : str
            One component of a vector variable, drawn on its own.
        kind : str
            `"scatter"`, or `"hist2d"` for a block model, where the points are
            past counting and a scatter is a solid mass whatever its
            transparency.
        alpha : float
            How transparent each point is. Worth lowering as the points pile
            up, until there are enough to want `kind="hist2d"` instead.
        bins : int
            Bins along each axis, for `kind="hist2d"`.
        log_counts : bool
            Colour the cells by the logarithm of the count.
        """
        self._check_kind(kind)
        var = self._require_continuous("prediction_scatter")
        true, predicted, labels = _prep.prediction_values(self.data, var.name)

        if component is not None:
            if component not in labels:
                raise KeyError("no component %r in %r; found %s"
                               % (component, var.name, ", ".join(labels)))
            index = labels.index(component)
            true, predicted = true[:, [index]], predicted[:, [index]]
            labels = [component]

        if len(labels) == 1:
            return self._joint_scatter(true[:, 0], predicted[:, 0], labels[0],
                                       figsize, size, kind, alpha, bins,
                                       log_counts)

        rows, columns = _prep.grid_shape(len(labels))
        with _style.context():
            figure, axes = _plt.subplots(
                rows, columns, squeeze=False,
                figsize=figsize or (3.0 * columns, 2.8 * rows))
            flat = axes.ravel()
            for i, label in enumerate(labels):
                self._draw_agreement(flat[i], true[:, i], predicted[:, i],
                                     size, kind, alpha, bins, log_counts)
                flat[i].set_title(label)
                flat[i].set_xlabel("measured")
                flat[i].set_ylabel("predicted")
            for extra in flat[len(labels):]:
                extra.set_visible(False)
            figure.suptitle("%s: predicted against measured" % var.name)
            figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figure

    def accuracy(self, probabilities=None, figsize=None):
        """
        Whether the simulated spread is the spread the errors actually have.

        For each probability, the share of true values that fall inside the
        interval holding that share of the simulations. On the 1:1 line the
        model knows what it does not know; below it the intervals are too
        narrow for the errors they have to cover, and above it the model is
        hedging. Deutsch's goodness statistic sums that up in one number.
        """
        var = self._require_continuous("accuracy")
        components = getattr(var, "components", None)
        parts = [var] if components is None else \
            [components[label] for label in var.labels]

        with _style.context():
            figure, axes = _plt.subplots(figsize=figsize or (5.0, 5.0))
            axes.plot([0, 1], [0, 1], color="#4a4a4a", linewidth=1.0,
                      linestyle="--", label="perfect")

            for i, part in enumerate(parts):
                measured = part.measurements.values.to_numpy().astype(float)
                simulations = _np.asarray(part.get_simulations(), dtype=float)
                keep = ~_np.isnan(measured)
                # a grid predicted onto has simulations everywhere and
                # measurements nowhere, and there is nothing to check against
                self._check_measured(part, keep)
                nominal, observed = _gmet.coverage(
                    measured[keep], simulations[keep], probabilities)
                axes.plot(nominal, observed, marker="o", markersize=3,
                          color=self._color(i, str(part.name)),
                          label="%s (G = %.2f)"
                                % (part.name, _gmet.goodness(nominal,
                                                             observed)))

            axes.set_xlabel("probability of the interval")
            axes.set_ylabel("share of true values inside it")
            axes.set_title("Accuracy")
            axes.set_aspect("equal")
            axes.legend(loc="upper left")
            figure.tight_layout()
        return figure

    def grade_tonnage(self, component=None, density=None, cutoffs=30,
                      max_uncertainty=None, figsize=None):
        """
        How much material clears each cut-off, and how good it is.

        Tonnage falls and grade rises as the cut-off climbs, and where the two
        cross is the question the curve is drawn to answer. Simulations are
        carried through one by one and drawn as a family, with the median over
        them picked out: the spread between the thin lines is what the model
        does not know about the answer.

        Parameters
        ----------
        component : str
            Which grade, when the variable is a vector one. A cut-off applies
            to a single number, so there is nothing to guess here.
        density : float or str
            A number, a metadata column, or a `ContinuousVariable` -- and in
            that last case its simulations are matched with the grade's, one
            to one. Without a density the curve is in volume.
        cutoffs : int or array-like
            The grades to cut at, or how many of them to spread evenly across
            the range of the data.
        max_uncertainty : float
            Leave out the blocks the model doubts more than this, reading the
            column named when the Explorer was built. A block the model cannot
            speak for is not tonnage, and counting it flatters the answer at
            exactly the cut-offs where there is least data to go on.
        """
        var = self._require_continuous("grade_tonnage")
        name = var.name
        if component is not None:
            name = component
        elif var.length > 1:
            raise ValueError(
                "%r holds %d components and a cut-off applies to one grade; "
                "name one with component= (%s)"
                % (var.name, var.length,
                   ", ".join(str(label) for label in var.labels)))

        curves = _prep.grade_tonnage(
            self.data, name, density, cutoffs,
            uncertainty=self.uncertainty, max_uncertainty=max_uncertainty)

        cutoff = curves["cutoff"]
        many = curves["tonnage"].shape[1] > 1

        with _style.context():
            figure, axes = _plt.subplots(figsize=figsize or (6.5, 4.5))
            grade_axes = axes.twinx()

            # Two scales share one frame, and a single grid would belong to
            # the left one without saying so -- a reader following the grade
            # curve to a line reads a tonnage off it. Each scale gets its own
            # horizontal lines in the colour of its curve, faint enough to sit
            # under both; the cut-off axis is shared, so its lines stay grey.
            axes.grid(False)
            grade_axes.grid(False)
            axes.grid(True, axis="x", color="#d9d9d9", linewidth=0.6)
            for panel, index in ((axes, 0), (grade_axes, 1)):
                panel.grid(True, axis="y", color=_style.color(index),
                           alpha=0.25, linewidth=0.8)
                panel.set_axisbelow(True)
                panel.tick_params(axis="y", colors=_style.color(index))

            for panel, key, index in ((axes, "tonnage", 0),
                                      (grade_axes, "grade", 1)):
                if many:
                    panel.plot(cutoff, curves[key], color=_style.color(index),
                               linewidth=0.6, alpha=0.25)
                panel.plot(cutoff, _np.median(curves[key], axis=1),
                           color=_style.color(index), linewidth=2.0,
                           label=key)

            axes.set_xlabel("cut-off grade (%s)" % name)
            axes.set_ylabel(curves["unit"] + " above the cut-off",
                            color=_style.color(0))
            grade_axes.set_ylabel("mean grade above the cut-off",
                                  color=_style.color(1))
            title = "Grade and tonnage"
            if curves["kept"] < curves["total"]:
                # an uncertainty handed over as values has no name to give
                named = self.uncertainty \
                    if isinstance(self.uncertainty, str) else "uncertainty"
                title += " (%d of %d blocks, %s <= %g)" % (
                    curves["kept"], curves["total"], named, max_uncertainty)
            axes.set_title(title)

            handles = [axes.lines[-1], grade_axes.lines[-1]]
            axes.legend(handles, [line.get_label() for line in handles],
                        loc="center right")
            figure.tight_layout()
        return figure

    def _color_variable(self, color):
        """
        What to colour a scene by.

        A named `color` may be a variable or one component of a vector
        variable, so that `scene(color="Zn")` works without reaching into
        `Elements` by hand. Asked for nothing, it takes what the Explorer
        holds -- but a vector variable is several numbers per location and not
        one colour scale, so it says so rather than quietly drawing the first
        component under the whole variable's name.
        """
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
                "color= with one of %s"
                % (var.name, var.length,
                   ", ".join(str(label) for label in var.labels)))
        return var

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_categorical(var):
        return isinstance(var, (_data.RockTypeVariable, _data.BinaryVariable))

    def _draw_points(self, panel, x, y, size, kind, alpha, bins, log_counts,
                     color=None, label=None):
        """One panel of points, as a cloud or as counts in cells.

        Counted into cells, empty cells are left unpainted rather than drawn
        as the bottom of the colour scale, which would fill the panel with a
        background that looks like data.
        """
        if kind == "hist2d":
            return panel.hist2d(
                x, y, bins=bins, cmap=self.cmap, cmin=1,
                norm=_mcolors.LogNorm() if log_counts else None)
        return panel.scatter(x, y, s=size, alpha=alpha, linewidths=0,
                             color=color, label=label)

    @staticmethod
    def _check_kind(kind, categorical=None):
        if kind not in ("scatter", "hist2d"):
            raise ValueError(
                "kind must be 'scatter' or 'hist2d'; got %r" % kind)

    def _draw_matrix(self, axes, values, labels, series, measured, size,
                     density=False, kind="scatter", alpha=0.7, bins=60,
                     log_counts=False, upper=None):
        """A scatter matrix: pairs below the diagonal, distributions along it.

        The upper triangle is the lower one transposed, so it is left out --
        half the ink for all of the information, and it leaves a corner for the
        legend to sit in without covering any data.
        """
        if kind == "hist2d" and len(series) > 1:
            # counts make one surface, and several laid over each other read
            # as none of them, so the categories are pooled here
            series = [(None, None)]

        n = len(labels)
        for row in range(n):
            for column in range(n):
                panel = axes[row][column]
                if column > row:
                    if upper is None:
                        panel.set_visible(False)
                        continue
                    self._draw_upper(panel, values[measured, column],
                                     values[measured, row], upper, bins,
                                     log_counts)
                else:
                    for j, (name, mask) in enumerate(series):
                        keep = measured if mask is None else mask
                        if row == column:
                            panel.hist(values[keep, row], bins=20,
                                       color=self._color(j, name), alpha=0.65,
                                       density=density, label=name)
                        else:
                            self._draw_points(
                                panel, values[keep, column],
                                values[keep, row], size, kind, alpha, bins,
                                log_counts, color=self._color(j, name),
                                label=name)

                # An upper panel is never on the bottom row nor in the first
                # column, so it keeps no tick labels of its own: its scales are
                # the ones already written along the edges of the matrix.
                if row == n - 1:
                    panel.set_xlabel(labels[column])
                else:
                    panel.set_xticklabels([])
                if column == 0:
                    panel.set_ylabel(labels[row])
                else:
                    panel.set_yticklabels([])

    @staticmethod
    def _draw_axes_of_variation(panel, analysis, x_column, y_column, count):
        """
        The principal components, drawn back onto the data they came from.

        A component is a direction in the space of the measurements, so in a
        panel showing two of them it is the pair of entries belonging to those
        two -- no rescaling to the panel needed, unlike the loadings in `pca`.
        Each line runs one standard deviation of that component either side of
        the mean, which puts the components in the data's own units: the first
        is longest because it carries the most variance, and a line lying flat
        along an axis means that measurement moves on its own.
        """
        mean = analysis["mean"]
        loadings = analysis["loadings"]
        deviations = _np.sqrt(_np.maximum(analysis["eigenvalues"], 0.0))

        for k in range(min(count, loadings.shape[1])):
            step = loadings[:, k] * deviations[k]
            x = [mean[x_column] - step[x_column], mean[x_column] + step[x_column]]
            y = [mean[y_column] - step[y_column], mean[y_column] + step[y_column]]

            panel.annotate("", xy=(x[1], y[1]), xytext=(x[0], y[0]),
                           arrowprops={"arrowstyle": "<->", "color": "#2b2b2b",
                                       "linewidth": 1.1, "alpha": 0.85})

            # A component that barely involves either of these two measurements
            # is a short line, which is the honest answer -- but it is too
            # short to hang a name on, and every such panel would collect a
            # pile of labels over one spot.
            width = _np.ptp(panel.get_xlim()) or 1.0
            height = _np.ptp(panel.get_ylim()) or 1.0
            reach = _np.hypot((x[1] - x[0]) / width, (y[1] - y[0]) / height)
            if reach > 0.15:
                panel.text(x[1], y[1], "PC%d" % (k + 1), fontsize=7,
                           color="#2b2b2b", ha="left", va="bottom",
                           clip_on=True, bbox=_style.LABEL_BOX)

    def _draw_upper(self, panel, x, y, upper, bins, log_counts):
        """
        The upper triangle: the same pair, told the other way.

        The lower triangle answers "which category is this point"; up here the
        categories are pooled on purpose, so the question becomes "where does
        the data sit, all of it together" -- which a colour-split scatter is
        poor at, since whichever category is drawn last hides the rest.
        """
        if upper == "hist2d":
            panel.hist2d(x, y, bins=bins, cmap=self.cmap, cmin=1,
                         norm=_mcolors.LogNorm() if log_counts else None)
        elif upper == "density":
            self._draw_density(panel, x, y)
        elif upper == "correlation":
            correlation = _np.corrcoef(x, y)[0, 1]
            panel.text(0.5, 0.5, "%.2f" % correlation,
                       transform=panel.transAxes, ha="center", va="center",
                       # the stronger it is, the larger it reads
                       fontsize=9 + 14 * abs(correlation),
                       color=_style.color(0) if correlation >= 0
                       else _style.color(3))
            panel.set_xticks([])
            panel.set_yticks([])
            panel.grid(False)
        else:
            raise ValueError(
                "upper must be 'hist2d', 'density' or 'correlation'; got %r"
                % upper)

    def _draw_density(self, panel, x, y, grid=60, most=4000):
        """Smoothed contours of where the points are.

        The estimate costs one pass over the data for every grid cell, so a
        long column is thinned first: a density is a shape, and a few thousand
        points settle it as well as a few million do.
        """
        if len(x) > most:
            step = len(x) // most
            x, y = x[::step], y[::step]
        if len(x) < 3 or _np.ptp(x) == 0 or _np.ptp(y) == 0:
            return

        x_axis = _np.linspace(_np.min(x), _np.max(x), grid)
        y_axis = _np.linspace(_np.min(y), _np.max(y), grid)
        mesh_x, mesh_y = _np.meshgrid(x_axis, y_axis)

        kernel = _stats.gaussian_kde(_np.vstack([x, y]))
        density = kernel(_np.vstack([mesh_x.ravel(), mesh_y.ravel()]))

        # lines rather than filled bands: a filled contour paints its lowest
        # level over the whole panel, and half a matrix of dark squares next to
        # the light scatter half reads as two figures pasted together
        panel.contour(mesh_x, mesh_y, density.reshape(grid, grid),
                      levels=6, cmap=self.cmap, linewidths=0.9)

    @staticmethod
    def _draw_loadings(panel, loadings, labels, x_component, y_component,
                       scores):
        """An arrow per original column, over the cloud of scores.

        Each axis is stretched to its own component's spread. A single scale
        for the whole figure would suit the first component, which carries most
        of the variance, and send every arrow off the edge of the others.
        Loadings are unit length, so this keeps them inside the panel.
        """
        x_reach = _np.max(_np.abs(scores[:, x_component])) or 1.0
        y_reach = _np.max(_np.abs(scores[:, y_component])) or 1.0

        for i, label in enumerate(labels):
            x = loadings[i, x_component] * x_reach
            y = loadings[i, y_component] * y_reach
            panel.annotate("", xy=(x, y), xytext=(0, 0),
                           arrowprops={"arrowstyle": "->", "color": "#2b2b2b",
                                       "linewidth": 0.9, "alpha": 0.8})
            panel.text(x, y, label, fontsize=7, color="#2b2b2b",
                       ha="left", va="bottom", clip_on=True,
                       bbox=_style.LABEL_BOX)

    @staticmethod
    def _draw_normal(panel, values):
        """The normal of the same mean and spread, over the histogram.

        Fitted rather than standard, so that the curve asks about the shape
        alone. A warping's parameters are trained along with everything else
        and need not leave the data at unit variance -- the GP's amplitude
        absorbs a scale factor -- so a standard normal would call a perfectly
        symmetric result skewed. What is left over is the question the warping
        is answerable for: a distribution still leaning shows as a histogram
        sliding out from under the curve.
        """
        mean, deviation = _np.mean(values), _np.std(values)
        if deviation <= 0:
            return
        low, high = panel.get_xlim()
        x = _np.linspace(low, high, 200)
        density = _np.exp(-0.5 * ((x - mean) / deviation) ** 2) \
            / (deviation * _np.sqrt(2 * _np.pi))
        panel.plot(x, density, color="#2b2b2b", linewidth=1.0, alpha=0.8)

    @staticmethod
    def _annotate_correlation(panel, x, y):
        """What the eye is being asked about, as a number."""
        panel.text(0.05, 0.9, "r = %.2f" % _np.corrcoef(x, y)[0, 1],
                   transform=panel.transAxes, fontsize=7, color="#2b2b2b",
                   va="top", bbox=_style.LABEL_BOX)

    def _draw_agreement(self, panel, true, predicted, size, kind="scatter",
                        alpha=0.6, bins=60, log_counts=False):
        """Predicted against measured, with the line they would sit on."""
        self._draw_points(panel, true, predicted, size, kind, alpha, bins,
                          log_counts, color=_style.color(0))

        low = min(_np.min(true), _np.min(predicted))
        high = max(_np.max(true), _np.max(predicted))
        margin = 0.05 * (high - low) if high > low else 1.0
        panel.plot([low, high], [low, high], color="#4a4a4a", linewidth=1.0,
                   linestyle="--", zorder=0)
        # the same limits on both axes, so the line runs at 45 degrees and a
        # cloud leaning off it is leaning visibly
        panel.set_xlim(low - margin, high + margin)
        panel.set_ylim(low - margin, high + margin)

    def _joint_scatter(self, true, predicted, label, figsize, size,
                       kind="scatter", alpha=0.6, bins=60, log_counts=False):
        """One variable, with the two distributions along the sides."""
        with _style.context():
            figure = _plt.figure(figsize=figsize or (5.5, 5.5))
            grid = figure.add_gridspec(
                2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                wspace=0.06, hspace=0.06)

            main = figure.add_subplot(grid[1, 0])
            top = figure.add_subplot(grid[0, 0], sharex=main)
            right = figure.add_subplot(grid[1, 1], sharey=main)

            self._draw_agreement(main, true, predicted, size, kind, alpha,
                                 bins, log_counts)
            top.hist(true, bins=25, color=_style.color(0), alpha=0.65)
            right.hist(predicted, bins=25, orientation="horizontal",
                       color=_style.color(0), alpha=0.65)

            top.tick_params(labelbottom=False)
            right.tick_params(labelleft=False)
            top.set_ylabel("count")
            right.set_xlabel("count")
            main.set_xlabel("measured")
            main.set_ylabel("predicted")
            top.set_title(label)
        return figure

    def _scene_categories(self, axes, coordinates, var, n_dim, size):
        values, measured, labels = _prep.category_values(var)
        for j, (name, mask) in enumerate(_prep.groups(values, measured,
                                                      labels)):
            self._scatter(axes, coordinates[mask], n_dim, size,
                          color=self._color(j, name), label=name)

    def _scene_values(self, figure, axes, coordinates, var, n_dim, size):
        values, measured, _ = _prep.numeric_values(var)
        drawn = self._scatter(axes, coordinates[measured], n_dim, size,
                              values=values[measured, 0])
        if n_dim > 1:
            # in 1D the value is the vertical axis, and carries no colour
            figure.colorbar(drawn, ax=axes, label=var.name, shrink=0.8)

    def _scatter(self, axes, coordinates, n_dim, size, values=None,
                 color=None, label=None):
        """One scatter call, whatever the number of coordinates."""
        if n_dim == 1:
            # nothing to put on the other axis, so the value goes there --
            # unless the colour is the whole message, and then it is a strip
            height = values if values is not None \
                else _np.zeros(len(coordinates))
            position = [coordinates[:, 0], height]
        elif n_dim == 2:
            position = [coordinates[:, 0], coordinates[:, 1]]
        else:
            position = [coordinates[:, 0], coordinates[:, 1],
                        coordinates[:, 2]]

        options = {"s": size, "linewidths": 0, "label": label}
        if color is not None:
            options["color"] = color
        elif n_dim > 1:
            options["c"] = values
            options["cmap"] = self.cmap
        return axes.scatter(*position, **options)
