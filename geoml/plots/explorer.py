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

.. code-block:: python

    eda = geoml.plots.Explorer(point, continuous="Elements",
                               categorical="Rock")
    eda.histogram()
    eda.pairs()
    eda.pca(explained=0.9)
    eda.scene()

These are matplotlib figures, meant to be saved and printed. `Interactive` is
the same set of figures in plotly, for looking at on a screen.
"""
import matplotlib.colors as _mcolors
import matplotlib.pyplot as _plt
import numpy as _np

import geoml.data as _data
import geoml.metrics as _gmet
import geoml.plots.base as _base
import geoml.plots.prepare as _prep
import geoml.plots.style as _style


class Explorer(_base.Selection):
    """
    Exploratory figures for one data set and a choice of variables.

    Takes its arguments from `base.Selection`: the container, a continuous and
    a categorical variable, a model for the figures that need one, and the
    colours to draw them in.
    """

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
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
    def histogram(self, bins=25, figsize=None) -> "_plt.Figure":
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
              size=6) -> "_plt.Figure":
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
            bins=60, log_counts=False, figsize=None, size=6) -> "_plt.Figure":
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

    def scene(self, color=None, clip=None, figsize=None, size=14) -> "_plt.Figure":
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
        clip : pair of floats
            Where to end the colour scale, as quantiles: `[0, 0.99]` for a
            variable with a long right tail, which is most assays. Without it
            one value far from the rest takes the whole scale and leaves
            everything else in a single shade. Nothing is dropped -- the
            points beyond the ends take the end colour. Has no effect in 1D,
            where the value is an axis rather than a colour.
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
                self._scene_values(figure, axes, coordinates, var, n_dim,
                                   size, clip)

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
    def training_curve(self, window=None, figsize=None) -> "_plt.Figure":
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
                          log_counts=False, upper=None, figsize=None, size=6) -> "_plt.Figure":
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
                         size=6) -> "_plt.Figure":
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
                            kind, alpha, _prep.cells(len(source), bins),
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
    def _draw_kde(panel, values, limits):
        """The simulated density, as a line over the measured histogram."""
        curve = _prep.density_curve(values, limits)
        if curve is None:
            return
        panel.plot(curve[0], curve[1], color=_style.color(1), linewidth=1.6,
                   label="simulated")

    def prediction_scatter(self, component=None, kind="scatter", alpha=0.6,
                           bins=60, log_counts=False, figsize=None, size=10) -> "_plt.Figure":
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
        true, predicted, labels, _ = _prep.prediction_values(self.data,
                                                             var.name)

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

    def accuracy(self, probabilities=None, figsize=None) -> "_plt.Figure":
        """
        Whether the simulated spread is the spread the errors actually have.

        For each probability, the share of true values that fall inside the
        interval holding that share of the simulations. On the 1:1 line the
        model knows what it does not know; below it the intervals are too
        narrow for the errors they have to cover, and above it the model is
        hedging. Deutsch's goodness statistic sums that up in one number.

        The intervals come from the model rather than from the container: what
        is stored there is the ground, the likelihood noise having been
        integrated out, and an assay is a measurement of the ground rather
        than the ground itself. Scoring the stored simulations against
        measured values would ask the model a question it never answered, and
        it would fail -- so this figure needs the model the selection was
        built with.
        """
        var = self._require_continuous("accuracy")
        parts = _prep.continuous_parts(var)

        # a grid predicted onto has simulations everywhere and measurements
        # nowhere, and there is nothing to check against -- said before the
        # model is asked for anything, so it fails fast
        measured = [part.measurements.values.to_numpy().astype(float)
                    for part in parts]
        for part, values in zip(parts, measured):
            self._check_measured(part, ~_np.isnan(values))
        samples = self._measurement_samples(var, "accuracy")

        with _style.context():
            figure, axes = _plt.subplots(figsize=figsize or (5.0, 5.0))
            axes.plot([0, 1], [0, 1], color="#4a4a4a", linewidth=1.0,
                      linestyle="--", label="perfect")

            for i, part in enumerate(parts):
                simulations = _np.asarray(samples[:, i, :], dtype=float)
                keep = ~_np.isnan(measured[i])
                nominal, observed = _gmet.coverage(
                    measured[i][keep], simulations[keep], probabilities)
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

    def reliability(self, bins=10, figsize=None) -> "_plt.Figure":
        """
        Whether a claimed probability is the frequency it claims.

        One curve per category: the locations binned by the probability
        the model assigned to it, each bin's mean claim against the share
        of its locations actually measured as that category. On the
        diagonal a 70% claim is that category 70% of the time; below it
        the model is overconfident, above it hedging. The legend carries
        each curve's expected calibration error, the count-weighted mean
        distance from the diagonal.

        The same locations count as in `confusion_matrix`: a contact has
        two measurements and no one truth, and locations missing either
        the measurement or the prediction are left out.

        Only honest on data the model has not seen: at a training
        location the claim was fitted to its own outcome. The out-of-fold
        container `models.cross_validate` fills is the honest input.

        Parameters
        ----------
        bins : int or sequence
            How many bins, or where their edges are. A count gives
            **equal-count** bins over the claimed probabilities; pass
            explicit edges for equal width.
        """
        var = self._require_categorical("reliability")
        panels = _prep.reliability(self.data, var.name, bins=bins)

        with _style.context():
            figure, axes = _plt.subplots(figsize=figsize or (5.0, 5.0))
            axes.plot([0, 1], [0, 1], color="#4a4a4a", linewidth=1.0,
                      linestyle="--", label="perfect")
            for i, panel in enumerate(panels):
                axes.plot(panel["claimed"], panel["observed"], marker="o",
                          markersize=3,
                          color=self._color(i, panel["label"]),
                          label="%s (ECE = %.2f)"
                                % (panel["label"], panel["ece"]))
            axes.set_xlabel("claimed probability")
            axes.set_ylabel("share measured as the category")
            axes.set_title("Reliability")
            axes.set_aspect("equal")
            axes.legend(loc="upper left")
            figure.tight_layout()
        return figure

    def confusion_matrix(self, figsize=None) -> "_plt.Figure":
        """
        What was measured against what the model called there, counted.

        Rows are the measured categories, columns the predicted ones, so
        the diagonal is agreement and each row reads as one category's
        fate. The shading is each cell's share of its measured row --
        categories are as unbalanced as rock types usually are, and raw
        counts would light the dominant row and hide what happens to a
        rare one -- and the counts are written in the cells.

        Contacts do not count: a rock type variable carries two
        measurements there, and neither alone is the truth the prediction
        is measured against. Locations missing either the measurement or
        the prediction are left out likewise.

        Only honest on data the model has not seen: at a training location
        the prediction interpolates its own measurement, and the diagonal
        congratulates the model on remembering it. The out-of-fold
        container `models.cross_validate` fills is the honest input.
        """
        var = self._require_categorical("confusion_matrix")
        table = _prep.confusion_matrix(self.data, var.name)
        counts, labels = table["counts"], table["labels"]
        k = len(labels)

        with _style.context():
            side = max(3.4, 1.4 + 0.45 * k)
            figure, axes = _plt.subplots(figsize=figsize
                                         or (side + 1.1, side))
            image = axes.imshow(table["share"], vmin=0.0, vmax=1.0,
                                cmap=self.cmap)
            axes.grid(False)
            axes.set_xticks(range(k), labels, rotation=45, ha="right",
                            rotation_mode="anchor")
            axes.set_yticks(range(k), labels)
            axes.set_xlabel("predicted")
            axes.set_ylabel("measured")
            # a count is readable over any cell as long as its ink
            # disagrees with the shading behind it
            face = image.cmap(image.norm(table["share"]))
            dark = face[..., :3] @ (0.299, 0.587, 0.114) < 0.5
            for i in range(k):
                for j in range(k):
                    axes.text(j, i, str(int(counts[i, j])), ha="center",
                              va="center", fontsize=8,
                              color="white" if dark[i, j] else "#2b2b2b")
            figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04,
                            label="share of the measured category")
            axes.set_title("%s: %d locations, %.0f%% agreement"
                           % (var.name, int(counts.sum()),
                              100.0 * table["agreement"]))
            figure.tight_layout()
        return figure

    def swath(self, predicted, axis=0, bins=12, where=None, weights=None,
              quantiles=(0.05, 0.95), figsize=None) -> "_plt.Figure":
        """
        The data's mean against the model's, slab by slab along one axis.

        The check that localizes conditional bias instead of aggregating it
        away: a model unbiased overall can run high in one part of the
        deposit and low in another, and only a mean per slab shows where.
        Two corrections make the comparison fair. The data's means are
        declustered -- the stored `"declustering"` column, else weights
        computed here -- so a crowded patch of holes speaks once; and the
        model's means run only over the ground the data informs, which
        `where` names. Where the model carries simulations the band
        between two quantiles of the realizations' slab means is drawn,
        which a kriging swath cannot.

        Draws the continuous variable when one was given, else the
        categorical one as stacked shares: the data's declustered share of
        each category against the model's mean predicted probability.

        Parameters
        ----------
        predicted
            The grid or block model carrying the model's prediction.
        axis : int or str
            Which coordinate the slabs cut across, by index or by label.
        bins : int or sequence
            How many slabs, of **equal width**, or where their edges are.
        where
            Which locations of `predicted` take part: a boolean mask or
            the name of a boolean metadata column, as `assign_from_data`
            writes. Everything, by default.
        weights
            One declustering weight per sample, overriding the stored
            column.
        quantiles
            The two quantiles of the realizations' slab means drawn as a
            band.
        """
        with _style.context():
            if self.continuous is not None:
                return self._continuous_swath(predicted, axis, bins, where,
                                              weights, quantiles, figsize)
            var = self._require_categorical("swath")
            result = _prep.categorical_swath(self.data, predicted, var.name,
                                             axis, bins, weights, where)
            figure = _plt.figure(figsize=figsize or (7.0, 4.6))
            grid = figure.add_gridspec(2, 1, height_ratios=[3, 1])
            top = figure.add_subplot(grid[0])
            bottom = figure.add_subplot(grid[1], sharex=top)
            self._draw_category_swath(top, bottom, result)
            top.set_title("%s: %sshares along %s"
                          % (var.name,
                             "declustered " if result["declustered"] else "",
                             result["axis"]))
            bottom.set_xlabel(result["axis"])
            figure.tight_layout()
        return figure

    def _continuous_swath(self, predicted, axis, bins, where, weights,
                          quantiles, figsize):
        var = self.continuous
        panels = _prep.swath(self.data, predicted, var.name, axis=axis,
                             bins=bins, weights=weights, where=where,
                             quantiles=quantiles)
        rows, columns = _prep.grid_shape(len(panels))
        size = figsize or (4.6 * columns, 1.0 + 3.8 * rows)
        figure = _plt.figure(figsize=size)
        grid = figure.add_gridspec(2 * rows, columns,
                                   height_ratios=[3, 1] * rows)
        top = None
        for i, panel in enumerate(panels):
            row, column = divmod(i, columns)
            top = figure.add_subplot(grid[2 * row, column])
            bottom = figure.add_subplot(grid[2 * row + 1, column], sharex=top)
            self._draw_swath(top, bottom, panel, quantiles)
            top.set_title(panel["label"])
            bottom.set_xlabel(panel["axis"])
        handles, labels = top.get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", frameon=False,
                      fontsize="small", ncol=min(3, len(labels)),
                      bbox_to_anchor=(0.5, 0.995))
        figure.suptitle("%s: declustered data against the model along %s"
                        % (var.name, panels[0]["axis"]), y=1.03)
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        return figure

    def _draw_swath(self, top, bottom, panel, quantiles):
        """One component of `swath`: means and band above, support below."""
        x, model = _prep.step_path(panel["lo"], panel["hi"],
                                   panel["model_mean"])
        _, data = _prep.step_path(panel["lo"], panel["hi"],
                                  panel["data_mean"])
        model_color, data_color = self._color(0, "model"), \
            self._color(3, "data")
        if panel["band_lo"] is not None:
            _, low = _prep.step_path(panel["lo"], panel["hi"],
                                     panel["band_lo"])
            _, high = _prep.step_path(panel["lo"], panel["hi"],
                                      panel["band_hi"])
            top.fill_between(x, low, high, color=model_color, alpha=0.2,
                             linewidth=0,
                             label="model: %.0f-%.0f%% of realizations"
                             % (100 * quantiles[0], 100 * quantiles[1]))
        top.plot(x, model, color=model_color, linewidth=1.8,
                 label="model mean")
        top.plot(x, data, color=data_color, linewidth=1.2, label="data mean")
        top.plot(panel["centre"], panel["data_mean"], "o", markersize=4,
                 color=data_color)
        top.set_ylabel(panel["label"])

        width = 0.9 * (panel["hi"] - panel["lo"])
        bottom.bar(panel["centre"], panel["data_count"], width=width,
                   color=data_color, alpha=0.5, label="samples")
        bottom.set_ylabel("samples")
        cells = bottom.twinx()
        _, count = _prep.step_path(panel["lo"], panel["hi"],
                                   panel["model_count"])
        cells.plot(x, count, color=model_color, linewidth=1.0, alpha=0.7)
        cells.set_ylabel("model cells")
        cells.grid(False)

    def _draw_category_swath(self, top, bottom, result):
        """Stacked shares per slab, the data's beside the model's."""
        width = 0.42 * (result["hi"] - result["lo"])
        left = result["centre"] - 0.5 * width
        right = result["centre"] + 0.5 * width
        base_data = _np.zeros_like(result["centre"])
        base_model = _np.zeros_like(result["centre"])
        for k, label in enumerate(result["labels"]):
            color = self._color(k, label)
            data_share = _np.nan_to_num(result["data_share"][:, k])
            model_share = _np.nan_to_num(result["model_share"][:, k])
            top.bar(left, data_share, width=width, bottom=base_data,
                    color=color, label=label)
            top.bar(right, model_share, width=width, bottom=base_model,
                    color=color, hatch="//", edgecolor="white",
                    linewidth=0.5)
            base_data = base_data + data_share
            base_model = base_model + model_share
        top.set_ylim(0.0, 1.0)
        top.set_ylabel("share (data | model, hatched)")
        top.legend(frameon=False, fontsize="small")

        bottom.bar(result["centre"], result["data_count"],
                   width=0.9 * (result["hi"] - result["lo"]),
                   color=self._color(3, "data"), alpha=0.5)
        bottom.set_ylabel("samples")

    def spread_check(self, bins=8, figsize=None) -> "_plt.Figure":
        """
        Whether the noise the model fitted is the noise the data has.

        A residual holds two things at once -- how wrong the model was about
        the ground, and how far the assay fell from the ground -- so it is
        read against the two together. The band is the noise, the line the
        whole claim, the points what the errors actually did. On the line is
        calibrated, below it is hedging, above it is over-confident.

        The level axis is what says which term is at fault. A warping bends,
        so the noise grows with the value while the model's own uncertainty
        does not: a shortfall widening with the grade is the noise, a flat one
        is the posterior. Points inside the band alone are the plainest case
        -- the fitted noise over-explains the errors by itself.

        Only honest on data the model has not seen: at a training location it
        interpolates its own measurement, and the residual is not an error.

        Parameters
        ----------
        bins : int or sequence
            How many bins, or where their edges are. A count gives
            **equal-count** bins, since a predicted grade is skewed and equal
            width would leave the top bins with a sample each; pass
            `np.linspace(...)` to ask for equal width instead.
        """
        var = self._require_continuous("spread_check")
        panels = _prep.spread_check(self.data, var.name, bins=bins)

        rows, columns = _prep.grid_shape(len(panels))
        with _style.context():
            size = figsize or (4.2 * columns, 0.6 + 3.4 * rows)
            # three entries explaining a band, a line and a set of points do
            # not fit inside a panel without covering the very curve they
            # explain, so the key gets an axes of its own above them all --
            # side by side where the figure is wide enough for it
            side_by_side = size[0] >= 8.0
            figure = _plt.figure(figsize=size)
            grid = figure.add_gridspec(
                rows + 1, columns,
                height_ratios=[0.3 if side_by_side else 0.8] + [1.0] * rows)

            key = figure.add_subplot(grid[0, :])
            key.axis("off")

            drawn = []
            for i, panel in enumerate(panels):
                row, column = divmod(i, columns)
                axes = figure.add_subplot(grid[row + 1, column])
                self._draw_spread(axes, panel)
                axes.set_title(panel["label"])
                axes.set_xlabel("predicted value")
                axes.set_ylabel("spread")
                drawn.append(axes)

            handles, labels = drawn[0].get_legend_handles_labels()
            key.legend(handles, labels, loc="center", frameon=False,
                       ncol=len(labels) if side_by_side else 1,
                       fontsize="small")
            figure.suptitle("%s: claimed spread against observed" % var.name)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
        return figure

    def _draw_spread(self, axes, panel):
        """One component of `spread_check`."""
        x, noise = _prep.step_path(panel["lo"], panel["hi"], panel["noise"])
        _, total = _prep.step_path(panel["lo"], panel["hi"], panel["total"])

        axes.fill_between(x, 0.0, noise, color=self._color(0, "noise"),
                          alpha=0.25, label="claimed: measurement noise")
        axes.plot(x, total, color=self._color(0, "noise"), linewidth=1.6,
                  label="claimed: noise and uncertainty")
        # the points span their bin, so a wide one reads as a thin stretch of
        # data rather than as a wide interval
        axes.errorbar(panel["centre"], panel["observed"],
                      yerr=panel["observed_error"],
                      xerr=_np.stack([panel["centre"] - panel["lo"],
                                      panel["hi"] - panel["centre"]]),
                      fmt="o", markersize=4, capsize=0, linewidth=1.0,
                      color=self._color(3, "observed"),
                      label="observed: rms residual")
        axes.set_ylim(bottom=0.0)

    def variogram(self, n_lags=15, max_lag=None, direction=None,
                  tolerance=45.0, residuals=False, decluster=True,
                  figsize=None) -> "_plt.Figure":
        """
        The data's spatial structure, against the fan the simulations make.

        The experimental semivariogram of the measurements, with one thin
        curve per realization on the same pairs. A model that learned the
        spatial structure scatters its fan *around* the data's curve; a
        kernel too smooth sags below it at short lags, and a nugget fitted
        into the range lifts it there. Neither shows in `accuracy` or
        `spread_check`, which judge one location at a time.

        The measurements carry the likelihood noise and the realizations do
        not, so the fan is raised by what an independent error at each
        location adds to a semivariogram, taken from `noise_variance`.
        Without that the two curves are not the same quantity and every
        model looks over-smooth by a nugget.

        With `residuals=True` it is the variogram of `measured - predicted`
        and the fan is dropped: structure left in the residuals is structure
        the model missed -- honest on cross-validated predictions
        (`models.cross_validate`).

        Each panel's title carries `VS`, the variogram score of
        :func:`geoml.metrics.variogram_score` over the same locations and
        weights: the eye's verdict on the fan as one number, for comparing
        two models without squinting. Lower is better, but only against
        another model on the same data -- the score keeps a bias the curves
        are corrected for, and never reaches zero.

        Parameters
        ----------
        n_lags : int
            Number of equal-width lag bins.
        max_lag : float, optional
            Longest separation considered; half the bounding-box diagonal by
            default.
        direction : array-like, optional
            Direction vector for a directional variogram; omnidirectional
            when absent. The anisotropy ellipsoid's principal axes are the
            directions worth asking about.
        tolerance : float
            Angular tolerance around `direction`, in degrees.
        residuals : bool
            Variogram of the residuals instead, without the fan.
        decluster : bool or float
            Weight pairs by cell-declustering weights, so that the curve
            estimates the field's variogram rather than the sampling's.
            `True` chooses the cell size, a number fixes it, `False` leaves
            the pairs raw.
        """
        var = self._require_continuous("variogram")
        panels = _prep.variogram(
            self.data, var.name, n_lags=n_lags, max_lag=max_lag,
            direction=direction, tolerance=tolerance, residuals=residuals,
            decluster=decluster)

        rows, columns = _prep.grid_shape(len(panels))
        with _style.context():
            size = figsize or (4.2 * columns, 0.6 + 3.4 * rows)
            figure, axes_grid = _plt.subplots(
                rows, columns, figsize=size, squeeze=False)
            for i, panel in enumerate(panels):
                row, column = divmod(i, columns)
                axes = axes_grid[row][column]
                self._draw_variogram(axes, panel)
                axes.set_title(panel["label"] if panel["score"] is None
                               else "%s (VS = %.3g)"
                                    % (panel["label"], panel["score"]))
                axes.set_xlabel("lag distance")
                axes.set_ylabel("semivariance")
            for i in range(len(panels), rows * columns):
                row, column = divmod(i, columns)
                axes_grid[row][column].axis("off")
            axes_grid[0][0].legend(loc="lower right", fontsize="small",
                                   frameon=False)
            what = "residual variogram" if residuals else \
                "variogram and simulation fan"
            figure.suptitle("%s: %s" % (var.name, what))
            figure.tight_layout(rect=(0, 0, 1, 0.95))
        return figure

    def _draw_variogram(self, axes, panel):
        """One component of `variogram`."""
        fan = panel["realizations"]
        if fan is not None:
            for r in range(fan.shape[0]):
                axes.plot(panel["lag"], fan[r],
                          color=self._color(0, "realizations"),
                          alpha=0.2, linewidth=0.7,
                          label="realizations" if r == 0 else None)
        axes.axhline(panel["sill"], color=self._color(2, "sill"),
                     linestyle="--", linewidth=1.0, label="data variance")
        axes.plot(panel["lag"], panel["data"], "o-", markersize=4,
                  linewidth=1.4, color=self._color(3, "data"), label="data")
        axes.set_ylim(bottom=0.0)

    def grade_tonnage(self, component=None, density=None, cutoffs=30,
                      max_uncertainty=None, log_mass=False, figsize=None) -> "_plt.Figure":
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
        log_mass : bool
            Put the tonnage on a logarithmic scale. Most of a deposit clears
            the low cut-offs, so on a linear axis the high ones are a flat
            line along the bottom and the spread between the realizations
            there -- which is where the decision usually is -- cannot be seen
            at all. A cut-off that nothing clears has no logarithm and drops
            out of the curve rather than being drawn at the axis floor.
        """
        var = self._require_continuous("grade_tonnage")
        name = var.name
        if component is not None:
            name = component
        elif var.length > 1:
            raise ValueError(
                "%r holds %d components and a cut-off applies to one grade; "
                "name one with component= (%s)"
                % (var.name, var.length, _prep.component_names(var)))

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

            if log_mass:
                # only the tonnage: the grade axis spans one order of
                # magnitude at most and a log scale would say nothing
                axes.set_yscale("log")

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

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #
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

    def _draw_density(self, panel, x, y):
        """Smoothed contours of where the points are."""
        grid = _prep.density_grid(x, y)
        if grid is None:
            return
        x_axis, y_axis, density = grid

        # lines rather than filled bands: a filled contour paints its lowest
        # level over the whole panel, and half a matrix of dark squares next to
        # the light scatter half reads as two figures pasted together
        panel.contour(*_np.meshgrid(x_axis, y_axis), density,
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

        What is left over is the question the warping is answerable for: a
        distribution still leaning shows as a histogram sliding out from under
        the curve.
        """
        curve = _prep.normal_curve(values, *panel.get_xlim())
        if curve is None:
            return
        panel.plot(curve[0], curve[1], color="#2b2b2b", linewidth=1.0,
                   alpha=0.8)

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

    def _scene_values(self, figure, axes, coordinates, var, n_dim, size,
                      clip=None):
        values, measured, _ = _prep.numeric_values(var)
        drawn = self._scatter(axes, coordinates[measured], n_dim, size,
                              values=values[measured, 0],
                              limits=_prep.color_limits(values[measured, 0],
                                                        clip))
        if n_dim > 1:
            # in 1D the value is the vertical axis, and carries no colour
            figure.colorbar(drawn, ax=axes, label=var.name, shrink=0.8,
                            extend="both" if clip is not None else "neither")

    def _scatter(self, axes, coordinates, n_dim, size, values=None,
                 color=None, label=None, limits=None):
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
            if limits is not None:
                # the scale is bounded, not the data: a point past the end
                # keeps its own number and takes the end colour
                options["vmin"], options["vmax"] = limits
        return axes.scatter(*position, **options)
