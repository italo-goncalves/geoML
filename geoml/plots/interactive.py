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
The same figures, for looking at rather than for printing.

`Interactive` answers every question `Explorer` does, with the same method
names and the same arguments, and hands back a plotly figure instead of a
matplotlib one:

    eda = geoml.plots.Interactive(point, continuous="Elements",
                                  categorical="Rock")
    eda.pairs().show()
    eda.scene()

What that buys is the things a printed figure cannot do: zooming a panel of a
scatter matrix takes its whole row and column with it, so a cluster can be
followed across every pair at once; clicking a category in the legend takes it
out of every panel; and a three-dimensional scene can be turned around.

Every trace drawn from locations carries, in `customdata`, the row of the
container each of its points came from. That is what makes a `Dashboard` able
to link one figure's selection to another's -- see `dashboard`. Figures whose
points are not locations, such as the training curve, carry none, and a
simulated value is a realization rather than a place, so the simulated half of
`simulation_pairs` carries none either.

The numbers all come from `prepare`, the same functions `Explorer` reads, so
the two backends cannot drift into showing different things.
"""
import numpy as _np
import plotly.graph_objects as _go
from plotly.subplots import make_subplots as _subplots

import geoml.data as _data
import geoml.metrics as _gmet
import geoml.plots.base as _base
import geoml.plots.dashboard as _dash
import geoml.plots.prepare as _prep
import geoml.plots.style as _style


# What a point that fell outside the selection looks like. Plotly's own default
# is a light dimming, which on a crowded panel is not enough to tell the chosen
# points from the rest.
UNSELECTED = {"marker": {"opacity": 0.08}}

# the grey of the reference lines, matching the matplotlib figures
REFERENCE = "#4a4a4a"
INK = "#2b2b2b"


class Interactive(_base.Selection):
    """
    Exploratory figures for one data set, as plotly figures.

    Takes its arguments from `base.Selection`: the container, a continuous and
    a categorical variable, a model for the figures that need one, and the
    colours to draw them in. Every method mirrors the one of the same name on
    `Explorer`, down to the arguments, except that a figure is sized in pixels
    (`height`, `width`) rather than in inches.
    """

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _finish(figure, title=None, height=None, width=None, **layout):
        """The package's look, applied to a figure on its way out."""
        figure.update_layout(template=_style.TEMPLATE, height=height,
                             width=width, **layout)
        if title is not None:
            figure.update_layout(title_text=title)
        return figure

    @staticmethod
    def _axis(row, column, columns):
        """Plotly's suffix for the axes of panel `(row, column)`, 1-based."""
        index = (row - 1) * columns + column
        return "" if index == 1 else str(index)

    def _points(self, figure, x, y, rows, size, alpha, color, name,
                legend, row, column):
        """One trace of locations, each carrying the row it came from."""
        figure.add_trace(
            _go.Scatter(
                x=x, y=y, mode="markers",
                customdata=rows,
                marker={"size": size, "opacity": alpha, "color": color,
                        "line": {"width": 0}},
                unselected=UNSELECTED,
                name=name, legendgroup=name, showlegend=legend,
                hovertemplate="%{x:.4g}, %{y:.4g}"
                              "<br>row %{customdata}<extra></extra>"),
            row=row, col=column)

    def _cells(self, figure, x, y, bins, log_counts, row, column):
        """Points counted into cells, as a heatmap.

        Plotly's own `Histogram2d` bins for itself but paints every cell,
        including the empty ones, and has no way to spread the colours over the
        logarithm of a count. Counting first settles both: an empty cell is a
        gap in the array and is left unpainted, and the colour can be taken
        from whichever of the two the caller asked for.

        No colour bar: these go in the panels of a matrix, where a bar beside
        each would cost more room than the counts are worth. The hover says
        how many points a cell holds, which is the question one actually asks
        of a cell.
        """
        counted = _prep.counts_2d(x, y, bins, log_counts)
        figure.add_trace(
            _go.Heatmap(
                x=counted["x"], y=counted["y"], z=counted["z"],
                # the count goes in `text`, not in `customdata`: a dashboard
                # reads `customdata` as the row a point came from, and a cell
                # is a number of points rather than one of them
                text=counted["count"],
                colorscale=self.cmap, showscale=False, hoverongaps=False,
                hovertemplate="%{x:.4g}, %{y:.4g}"
                              "<br>%{text:d} points<extra></extra>"),
            row=row, col=column)

    def _density(self, figure, x, y, row, column):
        """Smoothed contours of where the points are."""
        grid = _prep.density_grid(x, y)
        if grid is None:
            return
        x_axis, y_axis, density = grid
        figure.add_trace(
            _go.Contour(
                x=x_axis, y=y_axis, z=density, showscale=False,
                colorscale=self.cmap, ncontours=7,
                contours={"coloring": "lines"}, line={"width": 1.2},
                hoverinfo="skip"),
            row=row, col=column)

    @staticmethod
    def _line(figure, x, y, color, row=None, column=None, width=1.6,
              name=None, legend=False, dash=None, hover=False):
        """A curve that is not made of locations: a fit, a reference, a mean."""
        figure.add_trace(
            _go.Scatter(x=x, y=y, mode="lines", name=name,
                        legendgroup=name, showlegend=legend,
                        line={"color": color, "width": width, "dash": dash},
                        hoverinfo=None if hover else "skip"),
            row=row, col=column)

    def _bars(self, figure, values, rows, edges, color, name, legend,
              row, column, density=False):
        """A histogram, over bins chosen outside so panels can share them."""
        figure.add_trace(
            _go.Histogram(
                x=values, customdata=rows,
                xbins={"start": edges[0], "end": edges[-1],
                       "size": edges[1] - edges[0]},
                autobinx=False,
                histnorm="probability density" if density else "",
                marker={"color": color}, opacity=0.65,
                unselected={"marker": {"opacity": 0.15}},
                name=name, legendgroup=name, showlegend=legend),
            row=row, col=column)

    # ------------------------------------------------------------------ #
    # several at once
    # ------------------------------------------------------------------ #
    def dashboard(self, figures=("scene", "histogram", "pairs"), **options):
        """
        The named figures on one page, sharing a selection.

        Each name is a method of this class, called with its own defaults.
        That is the short way; for figures drawn with arguments of your own,
        or captioned, or from more than one data set, build a `Dashboard`
        directly -- it takes figures rather than names:

            geoml.plots.Dashboard(
                [("Where the cadmium is", eda.scene(color="Cd")),
                 eda.pairs(log=True, upper="density")],
                title="Jura", columns=2)

        Parameters
        ----------
        figures : sequence of str
            Which figures, in the order they are to be laid out. Asking for
            one this Interactive cannot draw raises whatever that figure
            would have raised on its own -- there is no guessing here about
            what a data set can support.
        options
            Passed to `Dashboard`: `title`, `columns`, `plotlyjs`, `hint`.
        """
        options.setdefault(
            "title", "%s — %s" % (type(self.data).__name__,
                                  getattr(self.continuous, "name", None)
                                  or getattr(self.categorical, "name", "")))
        return _dash.Dashboard([getattr(self, name)() for name in figures],
                               **options)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    def histogram(self, bins=25, height=None, width=None) -> "_go.Figure":
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
        # a panel here carries a title of its own as well as the tick labels
        # of the one above, and both live in the gap between the two
        figure = _subplots(rows=rows, cols=columns, subplot_titles=labels,
                           vertical_spacing=min(0.4 / rows, 0.2))

        for i, label in enumerate(labels):
            row, column = divmod(i, columns)
            edges = _np.histogram_bin_edges(values[measured, i], bins=bins)
            for j, (name, mask) in enumerate(series):
                keep = measured if mask is None else mask
                self._bars(figure, values[keep, i], _np.where(keep)[0], edges,
                           self._color(j, name), name or var.name, i == 0,
                           row + 1, column + 1)

        figure.update_yaxes(title_text="count", col=1)
        return self._finish(
            figure, title=var.name,
            height=height or 100 + 250 * rows, width=width,
            barmode="overlay",
            legend={"title": {"text": getattr(self.categorical, "name", "")}})

    def pairs(self, kind="scatter", alpha=0.7, bins=60, log_counts=False,
              log=False, principal_components=0, upper=None, size=5,
              height=None, width=None) -> "_go.Figure":
        """
        Every component against every other, with its distribution down the
        diagonal.

        Parameters
        ----------
        kind : str
            `"scatter"`, or `"hist2d"` to count the points into cells instead,
            for when there are too many to draw one by one. Counts make a
            single surface, so the categories are pooled -- and a counted cell
            is not a location, so a matrix drawn this way takes no part in a
            dashboard's linked selection.
        alpha : float
            How opaque each point is, for `kind="scatter"`.
        bins : int
            Cells along each axis, for `kind="hist2d"`.
        log_counts : bool
            Colour the cells by the logarithm of the count. Worth turning on
            whenever a few cells hold most of the data, which is most of the
            time with a skewed variable.
        log : bool
            Draw the data on a log scale: the centred log-ratio for a
            composition, whose parts carry a constant sum, and an ordinary
            logarithm otherwise.
        principal_components : int
            Draw this many principal components over the data, in the data's
            own axes -- the reverse of `pca`. Each is a line through the mean,
            one standard deviation of that component long either way.
        upper : str
            What to put in the upper triangle, which is otherwise left empty:
            `"hist2d"` or `"density"` for where the mass is with the categories
            pooled, or `"correlation"` for the coefficient alone.
        """
        self._check_kind(kind)
        var = self._require_vector("pairs")
        values, measured, labels = _prep.numeric_values(var)
        self._check_measured(var, measured)
        series = self._series()

        compositional = isinstance(var, _data.CompositionalVariable)
        if log:
            # only the measured rows: the gaps are padded with 1.0 to keep the
            # array square, and that padding is not data to be transformed
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

        n = len(labels)
        figure = self._matrix(values, labels, series, measured,
                              _np.arange(len(values)), kind=kind, alpha=alpha,
                              bins=bins, log_counts=log_counts, upper=upper,
                              size=size)

        if analysis is not None:
            for row in range(n):
                for column in range(row):
                    self._axes_of_variation(figure, analysis, column, row,
                                            principal_components)

        title = "%s by %s" % (var.name, self.categorical.name) \
            if self.categorical is not None else var.name
        return self._finish(
            figure, title=title, height=height or 80 + 170 * n,
            width=width or 80 + 180 * n,
            legend={"title": {"text": getattr(self.categorical, "name", "")}})

    def pca(self, explained=0.9, log=False, kind="scatter", alpha=0.7,
            bins=60, log_counts=False, size=5, height=None, width=None) -> "_go.Figure":
        """
        The same pairs plot, on principal components instead of measurements.

        Only the components carrying `explained` of the variance are drawn.
        Each panel also holds the loadings: an arrow per original column,
        showing what it contributes to the two components on the axes. A
        composition is opened up with the centred log-ratio first.

        Parameters
        ----------
        explained : float
            Share of the total variance to reach, between 0 and 1.
        log : bool
            Take the components of the logarithms rather than of the
            measurements. A composition is opened up either way.
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

        figure = self._matrix(
            scores, labels, series, _np.ones(len(scores), dtype=bool),
            _np.where(measured)[0], kind=kind, alpha=alpha, bins=bins,
            log_counts=log_counts, upper=None, size=size)

        for row in range(n):
            for column in range(row):
                self._loadings(figure, analysis["loadings"],
                               analysis["labels"], column, row, scores, n)

        return self._finish(
            figure,
            title="%s — %.0f%% of the variance in %d components"
                  % (var.name, 100 * _np.sum(ratio[:n]), n),
            height=height or 80 + 190 * n, width=width or 80 + 200 * n,
            legend={"title": {"text": getattr(self.categorical, "name", "")}})

    def scene(self, color=None, clip=None, size=4, height=None, width=None) -> "_go.Figure":
        """
        Where the data is, coloured by a variable.

        The coordinates decide the drawing: a value against position in 1D, a
        map in 2D, a scatter that can be turned around in 3D -- which is the
        one plotly is worth reaching for on its own account.

        A three-dimensional scene takes no part in a dashboard's linked
        selection. Plotly has no box or lasso over a 3D scatter to select
        with, and no per-point selection state on one to show a selection
        made elsewhere. Several of them on one page do turn together, though.

        Parameters
        ----------
        color : str or list
            The variable to colour by. Defaults to the continuous variable
            held, or the categorical one if that is all there is.

            A **list** of them puts a menu on the figure instead, one entry
            per choice, and the ground is drawn once with the values swapped
            over it. That is the answer to several variables over the same
            body of rock: five grades become one scene with five entries on a
            menu rather than five scenes, which is lighter, and it holds the
            point cloud still while the variable changes -- the comparison one
            is actually making. A name in the list may be a variable, one
            component of a vector variable, or a vector variable standing for
            all of its components, so `color=["Elements"]` names every grade
            in it.

            Only the locations measured in all of them are drawn; see
            `prepare.color_choices`.
        clip : pair of floats
            Where to end the colour scale, as quantiles: `[0, 0.99]` for a
            variable with a long right tail, which is most assays, and
            `[0.01, 0.99]` to take both ends. Without it one value far from
            the rest takes the whole scale and leaves everything else in a
            single shade. Nothing is dropped -- the points beyond the ends
            take the end colour, and the hover still reports what was
            measured. A menu clips each of its choices by its own quantiles,
            since they are different variables and share nothing but the
            ground. Has no effect in 1D, where the value is an axis rather
            than a colour.
        """
        coordinates = _np.asarray(self.data.coordinates)
        n_dim = coordinates.shape[1]
        if n_dim > 3:
            raise ValueError(
                "a scene needs 1, 2 or 3 coordinates; this data has %d"
                % n_dim)

        labels = getattr(self.data, "coordinate_labels", None) \
            or ["axis %d" % (i + 1) for i in range(n_dim)]

        figure = _go.Figure()
        if isinstance(color, (list, tuple)):
            title, menu = self._scene_menu(figure, coordinates, n_dim, color,
                                           size, clip)
        else:
            title, menu = self._scene_one(figure, coordinates, n_dim,
                                          color, size, clip), None

        if n_dim == 3:
            layout = {"scene": {
                "xaxis": {"title": {"text": labels[0]}},
                "yaxis": {"title": {"text": labels[1]}},
                "zaxis": {"title": {"text": labels[2]}},
                "aspectmode": "data"}}
        else:
            layout = {"xaxis": {"title": {"text": labels[0]}},
                      "yaxis": {"title": {"text": title if n_dim == 1
                                          else labels[1]}}}
            if n_dim == 2:
                # a map: the two axes are the same thing, so the same scale
                layout["yaxis"]["scaleanchor"] = "x"

        if menu is not None:
            layout["updatemenus"] = [menu]
            # the menu sits over the figure, where the top margin is
            layout["margin"] = {"l": 60, "r": 30, "t": 90, "b": 50}

        return self._finish(figure, title=title,
                            height=height or 560, width=width,
                            legend={"title": {"text": title}}, **layout)

    def _scene_one(self, figure, coordinates, n_dim, color, size, clip=None):
        """One variable over the coordinates, and what to call the figure."""
        var = self._color_variable(color)

        if self._is_categorical(var):
            values, measured, names = _prep.category_values(var)
            for j, (name, mask) in enumerate(_prep.groups(values, measured,
                                                          names)):
                self._scene_trace(figure, coordinates[mask], n_dim, size,
                                  rows=_np.where(mask)[0], name=name,
                                  color=self._color(j, name))
        else:
            values, measured, _ = _prep.numeric_values(var)
            self._scene_trace(figure, coordinates[measured], n_dim, size,
                              rows=_np.where(measured)[0], name=str(var.name),
                              values=values[measured, 0],
                              limits=_prep.color_limits(values[measured, 0],
                                                        clip))
        return str(var.name)

    def _scene_menu(self, figure, coordinates, n_dim, names, size, clip=None):
        """
        One set of points, and a menu saying which values colour them.

        The ground is drawn once and only the values change, which is both
        lighter than a trace per choice -- a block model's coordinates are the
        bulk of it, and there is no sense in carrying five copies -- and truer
        to what is being asked: the cloud stays where it is while the variable
        over it changes.
        """
        values, rows, labels = _prep.color_choices(self.data, names)
        # each choice by its own quantiles: they are different variables and
        # share nothing but the ground they are drawn over
        limits = [_prep.color_limits(values[:, i], clip)
                  for i in range(len(labels))]

        self._scene_trace(figure, coordinates[rows], n_dim, size, rows=rows,
                          name=labels[0], values=values[:, 0],
                          limits=limits[0])

        buttons = []
        for i, label in enumerate(labels):
            if n_dim == 1:
                # in 1D the value is the vertical axis and carries no colour,
                # so switching means moving the points rather than repainting
                trace = {"y": [values[:, i]]}
                layout = {"title": {"text": label},
                          "yaxis": {"title": {"text": label}}}
            else:
                low, high = limits[i] or (None, None)
                trace = {"marker.color": [values[:, i]],
                         "marker.colorbar.title.text": label,
                         "marker.cauto": limits[i] is None,
                         "marker.cmin": low, "marker.cmax": high}
                layout = {"title": {"text": label}}
            buttons.append({"label": label, "method": "update",
                            "args": [trace, layout]})

        return labels[0], {"buttons": buttons, "direction": "down",
                           "showactive": True, "x": 1.0, "xanchor": "right",
                           "y": 1.12, "yanchor": "top",
                           "bgcolor": "white", "bordercolor": "#d9d9d9",
                           "font": {"size": 10}}

    def _scene_trace(self, figure, coordinates, n_dim, size, rows,
                     name=None, color=None, values=None, limits=None):
        """One scatter, whatever the number of coordinates."""
        marker = {"size": size, "line": {"width": 0}}
        if color is not None:
            marker["color"] = color
        elif n_dim > 1:
            marker["color"] = values
            marker["colorscale"] = self.cmap
            marker["colorbar"] = {"title": {"text": name}, "thickness": 14}
            if limits is not None:
                marker["cmin"], marker["cmax"] = limits

        if n_dim == 3:
            # no customdata: a 3D scene cannot be brushed and cannot show a
            # brush made elsewhere, so a row index on it would promise a link
            # that is not there
            figure.add_trace(_go.Scatter3d(
                x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
                mode="markers", marker=marker, name=name,
                showlegend=color is not None))
            return

        if n_dim == 1:
            # nothing to put on the other axis, so the value goes there --
            # unless the colour is the whole message, and then it is a strip
            position = {"x": coordinates[:, 0],
                        "y": values if values is not None
                        else _np.zeros(len(coordinates))}
        else:
            position = {"x": coordinates[:, 0], "y": coordinates[:, 1]}

        figure.add_trace(_go.Scatter(
            mode="markers", marker=marker, customdata=rows, name=name,
            legendgroup=name, showlegend=color is not None,
            unselected=UNSELECTED,
            hovertemplate="%{x:.4g}, %{y:.4g}"
                          "<br>row %{customdata}<extra></extra>",
            **position))

    # ------------------------------------------------------------------ #
    # figures that need the model
    # ------------------------------------------------------------------ #
    def training_curve(self, window=None, height=None, width=None) -> "_go.Figure":
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

        figure = _go.Figure()
        self._line(figure, curve["iteration"], curve["value"],
                   _style.color(0), width=1.0, name="ELBO", legend=True,
                   hover=True)
        if len(curve["smooth"]) < len(curve["value"]):
            self._line(figure, curve["smooth_iteration"], curve["smooth"],
                       _style.color(1), width=2.0, legend=True, hover=True,
                       name="mean of %d" % curve["window"])

        return self._finish(
            figure, title="Training", height=height or 420, width=width,
            xaxis={"title": {"text": "iteration"}},
            yaxis={"title": {"text": "ELBO"}},
            hovermode="x unified")

    def transformed_pairs(self, kind="scatter", alpha=0.7, bins=60,
                          log_counts=False, upper=None, size=5, height=None,
                          width=None) -> "_go.Figure":
        """
        The measurements as the model sees them, after its warping.

        Two things worth checking before trusting a fitted model, and both are
        easier to see than to test. Down the diagonal, whether the warping made
        each variable Gaussian: the normal it is aiming at is drawn over each
        histogram. Off the diagonal, whether what is left is independent: a
        round cloud with a correlation near zero is what the model assumes, and
        a tilted or curved one is structure it will not capture.

        The columns are numbered rather than named: a warping may rotate the
        data or bend it, so a column is generally a mixture of what was
        measured rather than any one of it.

        Parameters
        ----------
        kind, alpha, bins, log_counts, upper
            As in `pairs`.
        """
        self._check_kind(kind)
        self._require_model("transformed_pairs")
        var = self._require_continuous("transformed_pairs")
        self._check_measured(var, _prep.numeric_values(var)[1])
        values, measured, labels = _prep.warped_values(self.model, var.name)
        n = len(labels)

        figure = self._matrix(
            values, labels, [(None, None)],
            _np.ones(len(values), dtype=bool), _np.where(measured)[0],
            kind=kind, alpha=alpha, bins=bins, log_counts=log_counts,
            upper=upper, size=size, density=True)

        for row in range(n):
            self._normal(figure, values[:, row], row)
            for column in range(row):
                self._correlation(figure, values[:, column], values[:, row],
                                  column, row, n, corner=True)

        return self._finish(
            figure, title="%s, as the model sees it" % var.name,
            height=height or 80 + 190 * n, width=width or 80 + 200 * n,
            showlegend=False)

    def simulation_pairs(self, kind="hist2d", bins=60, log_counts=False,
                         most=100000, margin=0.1, alpha=0.6, size=5,
                         height=None, width=None) -> "_go.Figure":
        """
        What was measured against what was simulated.

        The measurements fill the lower triangle and the simulations the upper,
        in the same form and between the same limits, so the two halves of the
        matrix can be read against each other: a simulation that reproduces the
        data has an upper half that mirrors the lower one. Down the diagonal
        the measured histogram carries the simulated density over it, which is
        the same comparison one variable at a time.

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
            About how many simulated values to draw per component.
        margin : float
            How far past the measured range to look, as a share of it.
            Simulated values outside that window are left out.
        """
        self._check_kind(kind)
        var = self._require_continuous("simulation_pairs")
        values, measured, labels = _prep.numeric_values(var)
        self._check_measured(var, measured)

        data = values[measured]
        rows = _np.where(measured)[0]
        sample = _prep.simulation_sample(var, most)
        n = len(labels)

        # The measured range with room around it, shared by a panel and its
        # mirror -- without which the halves cannot be compared by eye at all.
        limits = _prep.padded_range(data, margin)
        inside = _np.column_stack(
            [(sample[:, i] >= low) & (sample[:, i] <= high)
             for i, (low, high) in enumerate(limits)])

        figure = _subplots(rows=n, cols=n, shared_xaxes=True,
                           horizontal_spacing=0.02, vertical_spacing=0.02)

        for row in range(n):
            for column in range(n):
                first = row == 0 and column == 0
                if row == column:
                    edges = _np.histogram_bin_edges(
                        data[:, row], bins=25, range=limits[row])
                    self._bars(figure, data[:, row], rows, edges,
                               _style.color(0), "measured", first,
                               row + 1, column + 1, density=True)
                    curve = _prep.density_curve(sample[inside[:, row], row],
                                                limits[row])
                    if curve is not None:
                        self._line(figure, curve[0], curve[1],
                                   _style.color(1), row + 1, column + 1,
                                   name="simulated", legend=first)
                else:
                    measurements = column < row
                    source, identity = data, rows
                    if not measurements:
                        keep = inside[:, column] & inside[:, row]
                        # a simulated value is a realization, not a place
                        source, identity = sample[keep], None
                    self._pair(
                        figure, source[:, column], source[:, row], identity,
                        kind, alpha, size,
                        _prep.cells(len(source), bins), log_counts,
                        _style.color(0 if measurements else 1),
                        "measured" if measurements else "simulated",
                        False, row + 1, column + 1)
                    figure.update_yaxes(range=list(limits[row]),
                                        row=row + 1, col=column + 1)
                figure.update_xaxes(range=list(limits[column]),
                                    row=row + 1, col=column + 1)

        self._label_edges(figure, labels, n)
        return self._finish(
            figure,
            title="%s: measured below the diagonal, simulated above"
                  % var.name,
            height=height or 80 + 180 * n, width=width or 80 + 190 * n,
            barmode="overlay")

    def prediction_scatter(self, component=None, kind="scatter", alpha=0.6,
                           bins=60, log_counts=False, size=6, height=None,
                           width=None) -> "_go.Figure":
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
            past counting.
        alpha : float
            How opaque each point is.
        bins : int
            Bins along each axis, for `kind="hist2d"`.
        log_counts : bool
            Colour the cells by the logarithm of the count.
        """
        self._check_kind(kind)
        var = self._require_continuous("prediction_scatter")
        true, predicted, labels, rows = _prep.prediction_values(self.data,
                                                                var.name)

        if component is not None:
            if component not in labels:
                raise KeyError("no component %r in %r; found %s"
                               % (component, var.name, ", ".join(labels)))
            index = labels.index(component)
            true, predicted = true[:, [index]], predicted[:, [index]]
            labels = [component]

        if len(labels) == 1:
            return self._joint(true[:, 0], predicted[:, 0], rows, labels[0],
                               kind, alpha, bins, log_counts, size, height,
                               width)

        panels, columns = _prep.grid_shape(len(labels))
        figure = _subplots(rows=panels, cols=columns, subplot_titles=labels,
                           horizontal_spacing=0.08)
        for i, label in enumerate(labels):
            row, column = divmod(i, columns)
            self._agreement(figure, true[:, i], predicted[:, i], rows, kind,
                            alpha, bins, log_counts, size, row + 1, column + 1,
                            legend=False)
        figure.update_xaxes(title_text="measured", row=panels)
        figure.update_yaxes(title_text="predicted", col=1)

        return self._finish(
            figure, title="%s: predicted against measured" % var.name,
            height=height or 100 + 300 * panels, width=width,
            showlegend=False)

    def accuracy(self, probabilities=None, height=None, width=None) -> "_go.Figure":
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
        than the ground itself. So this figure needs the model the selection
        was built with.
        """
        var = self._require_continuous("accuracy")
        components = getattr(var, "components", None)
        parts = [var] if components is None else \
            [components[label] for label in var.labels]

        # nothing measured means nothing to check against, said before the
        # model is asked for anything
        measured = [part.measurements.values.to_numpy().astype(float)
                    for part in parts]
        for part, values in zip(parts, measured):
            self._check_measured(part, ~_np.isnan(values))
        samples = self._measurement_samples(var, "accuracy")

        figure = _go.Figure()
        self._line(figure, [0, 1], [0, 1], REFERENCE, width=1.0,
                   name="perfect", legend=True, dash="dash")

        for i, part in enumerate(parts):
            simulations = _np.asarray(samples[:, i, :], dtype=float)
            keep = ~_np.isnan(measured[i])
            nominal, observed = _gmet.coverage(
                measured[i][keep], simulations[keep], probabilities)
            figure.add_trace(_go.Scatter(
                x=nominal, y=observed, mode="lines+markers",
                marker={"size": 6},
                line={"color": self._color(i, str(part.name)), "width": 1.8},
                name="%s (G = %.2f)"
                     % (part.name, _gmet.goodness(nominal, observed)),
                hovertemplate="nominal %{x:.2f}"
                              "<br>observed %{y:.2f}<extra></extra>"))

        return self._finish(
            figure, title="Accuracy", height=height or 520, width=width or 560,
            xaxis={"title": {"text": "probability of the interval"}},
            yaxis={"title": {"text": "share of true values inside it"},
                   "scaleanchor": "x"},
            legend={"x": 0.02, "y": 0.98, "xanchor": "left",
                    "yanchor": "top"})

    def spread_check(self, bins=8, height=None, width=None) -> "_go.Figure":
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
        is the posterior.

        Only honest on data the model has not seen.

        Parameters
        ----------
        bins : int or sequence
            How many bins, or where their edges are. A count gives
            **equal-count** bins; pass `np.linspace(...)` for equal width.
        """
        var = self._require_continuous("spread_check")
        panels = _prep.spread_check(self.data, var.name, bins=bins)
        labels = [panel["label"] for panel in panels]

        rows, columns = _prep.grid_shape(len(panels))
        figure = _subplots(rows=rows, cols=columns, subplot_titles=labels,
                           horizontal_spacing=0.1)
        for i, panel in enumerate(panels):
            row, column = divmod(i, columns)
            self._spread(figure, panel, row + 1, column + 1, legend=i == 0)
        figure.update_xaxes(title_text="predicted value", row=rows)
        figure.update_yaxes(title_text="spread", col=1, rangemode="tozero")

        return self._finish(
            figure, title="%s: claimed spread against observed" % var.name,
            height=height or 120 + 320 * rows, width=width,
            # above the panels rather than beside them, as in `Explorer`: the
            # entries are long and a column of them squeezes every panel
            legend={"orientation": "h", "x": 0.5, "xanchor": "center",
                    "y": 1.0, "yanchor": "bottom"})

    def _spread(self, figure, panel, row, column, legend):
        """One component of `spread_check`."""
        x, noise = _prep.step_path(panel["lo"], panel["hi"], panel["noise"])
        _, total = _prep.step_path(panel["lo"], panel["hi"], panel["total"])
        claimed = self._color(0, "noise")

        figure.add_trace(_go.Scatter(
            x=x, y=noise, mode="lines", fill="tozeroy",
            line={"color": claimed, "width": 0.8},
            fillcolor="rgba(31, 111, 139, 0.25)",
            name="claimed: measurement noise", showlegend=legend,
            hovertemplate="noise %{y:.4g}<extra></extra>"), row=row, col=column)
        figure.add_trace(_go.Scatter(
            x=x, y=total, mode="lines", line={"color": claimed, "width": 1.8},
            name="claimed: noise and uncertainty", showlegend=legend,
            hovertemplate="claimed %{y:.4g}<extra></extra>"),
            row=row, col=column)
        figure.add_trace(_go.Scatter(
            x=panel["centre"], y=panel["observed"], mode="markers",
            marker={"size": 7, "color": self._color(3, "observed")},
            error_y={"type": "data", "array": panel["observed_error"]},
            error_x={"type": "data",
                     "array": panel["hi"] - panel["centre"],
                     "arrayminus": panel["centre"] - panel["lo"]},
            customdata=_np.stack([panel["count"],
                                  panel["observed"] / panel["total"]], axis=1),
            name="observed: rms residual", showlegend=legend,
            hovertemplate="observed %{y:.4g}<br>%{customdata[0]} samples"
                          "<br>observed/claimed %{customdata[1]:.2f}"
                          "<extra></extra>"), row=row, col=column)

    def variogram(self, n_lags=15, max_lag=None, direction=None,
                  tolerance=45.0, residuals=False, height=None, width=None) -> "_go.Figure":
        """
        The data's spatial structure, against the fan the simulations make.

        The experimental semivariogram of the measurements, with one thin
        curve per realization on the same pairs. A model that learned the
        spatial structure scatters its fan *around* the data's curve; a
        kernel too smooth sags below it at short lags, and a nugget fitted
        into the range lifts it there. Neither shows in `accuracy` or
        `spread_check`, which judge one location at a time.

        With `residuals=True` it is the variogram of `measured - predicted`
        and the fan is dropped: structure left in the residuals is structure
        the model missed -- honest on cross-validated predictions
        (`models.cross_validate`).

        Parameters
        ----------
        n_lags : int
            Number of equal-width lag bins.
        max_lag : float, optional
            Longest separation considered; half the bounding-box diagonal by
            default.
        direction : array-like, optional
            Direction vector for a directional variogram; omnidirectional
            when absent.
        tolerance : float
            Angular tolerance around `direction`, in degrees.
        residuals : bool
            Variogram of the residuals instead, without the fan.
        """
        var = self._require_continuous("variogram")
        panels = _prep.variogram(
            self.data, var.name, n_lags=n_lags, max_lag=max_lag,
            direction=direction, tolerance=tolerance, residuals=residuals)
        labels = [panel["label"] for panel in panels]

        rows, columns = _prep.grid_shape(len(panels))
        figure = _subplots(rows=rows, cols=columns, subplot_titles=labels,
                           horizontal_spacing=0.1)
        for i, panel in enumerate(panels):
            row, column = divmod(i, columns)
            self._variogram_panel(figure, panel, row + 1, column + 1,
                                  legend=i == 0)
        figure.update_xaxes(title_text="lag distance", row=rows)
        figure.update_yaxes(title_text="semivariance", col=1,
                            rangemode="tozero")

        what = "residual variogram" if residuals else \
            "variogram and simulation fan"
        return self._finish(
            figure, title="%s: %s" % (var.name, what),
            height=height or 120 + 320 * rows, width=width,
            legend={"orientation": "h", "x": 0.5, "xanchor": "center",
                    "y": 1.0, "yanchor": "bottom"})

    def _variogram_panel(self, figure, panel, row, column, legend):
        """One component of `variogram`."""
        fan = panel["realizations"]
        if fan is not None:
            for r in range(fan.shape[0]):
                figure.add_trace(_go.Scatter(
                    x=panel["lag"], y=fan[r], mode="lines",
                    line={"color": self._color(0, "realizations"),
                          "width": 0.7},
                    opacity=0.25, name="realizations",
                    legendgroup="realizations",
                    showlegend=legend and r == 0,
                    hoverinfo="skip"), row=row, col=column)
        figure.add_trace(_go.Scatter(
            x=[panel["lag"][0], panel["lag"][-1]],
            y=[panel["sill"], panel["sill"]], mode="lines",
            line={"color": self._color(2, "sill"), "width": 1.0,
                  "dash": "dash"},
            name="data variance", showlegend=legend,
            hovertemplate="variance %{y:.4g}<extra></extra>"),
            row=row, col=column)
        figure.add_trace(_go.Scatter(
            x=panel["lag"], y=panel["data"], mode="lines+markers",
            marker={"size": 6, "color": self._color(3, "data")},
            line={"color": self._color(3, "data"), "width": 1.6},
            customdata=panel["count"],
            name="data", showlegend=legend,
            hovertemplate="lag %{x:.4g}<br>semivariance %{y:.4g}"
                          "<br>%{customdata} pairs<extra></extra>"),
            row=row, col=column)

    def grade_tonnage(self, component=None, density=None, cutoffs=30,
                      max_uncertainty=None, log_mass=False, height=None,
                      width=None) -> "_go.Figure":
        """
        How much material clears each cut-off, and how good it is.

        Tonnage falls and grade rises as the cut-off climbs, and where the two
        cross is the question the curve is drawn to answer. Simulations are
        carried through one by one and drawn as a family, with the median over
        them picked out: the spread between the thin lines is what the model
        does not know about the answer.

        Two scales share one frame, so each gets gridlines in the colour of its
        own curve -- a reader following the grade curve to a grey line would
        otherwise read a tonnage off it.

        Parameters
        ----------
        component : str
            Which grade, when the variable is a vector one.
        density : float or str
            A number, a metadata column, or a `ContinuousVariable` -- and in
            that last case its simulations are matched with the grade's, one to
            one. Without a density the curve is in volume.
        cutoffs : int or array-like
            The grades to cut at, or how many of them to spread evenly across
            the range of the data.
        max_uncertainty : float
            Leave out the blocks the model doubts more than this, reading the
            column named when the Interactive was built.
        log_mass : bool
            Put the tonnage on a logarithmic scale. Most of a deposit clears
            the low cut-offs, so on a linear axis the high ones are a flat
            line along the bottom and the spread between the realizations
            there -- which is where the decision usually is -- cannot be seen.
            A cut-off that nothing clears has no logarithm and drops out.
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

        figure = _go.Figure()
        for key, index, axis in (("tonnage", 0, "y"), ("grade", 1, "y2")):
            if many:
                # every realization in one trace, cut apart by the gaps a NaN
                # leaves: a hundred traces of a hundred points each is a
                # hundred legend entries and a far heavier page
                x, y = self._family(cutoff, curves[key])
                figure.add_trace(_go.Scatter(
                    x=x, y=y, mode="lines", yaxis=axis, showlegend=False,
                    line={"color": _style.color(index), "width": 0.6},
                    opacity=0.3, hoverinfo="skip"))
            figure.add_trace(_go.Scatter(
                x=cutoff, y=_np.median(curves[key], axis=1), mode="lines",
                yaxis=axis, name=key,
                line={"color": _style.color(index), "width": 2.5},
                hovertemplate="cut-off %{x:.4g}<br>" + key +
                              " %{y:.4g}<extra></extra>"))

        title = "Grade and tonnage"
        if curves["kept"] < curves["total"]:
            # an uncertainty handed over as values has no name to give
            named = self.uncertainty \
                if isinstance(self.uncertainty, str) else "uncertainty"
            title += " (%d of %d blocks, %s ≤ %g)" % (
                curves["kept"], curves["total"], named, max_uncertainty)

        return self._finish(
            figure, title=title, height=height or 470, width=width,
            xaxis={"title": {"text": "cut-off grade (%s)" % name}},
            # only the tonnage: the grade axis spans one order of magnitude at
            # most and a log scale would say nothing
            yaxis={"title": {"text": curves["unit"] + " above the cut-off",
                             "font": {"color": _style.color(0)}},
                   "tickfont": {"color": _style.color(0)},
                   "type": "log" if log_mass else "linear",
                   "gridcolor": self._faint(_style.color(0))},
            yaxis2={"title": {"text": "mean grade above the cut-off",
                              "font": {"color": _style.color(1)}},
                    "tickfont": {"color": _style.color(1)},
                    "gridcolor": self._faint(_style.color(1)),
                    "overlaying": "y", "side": "right", "showgrid": True},
            hovermode="x unified",
            legend={"x": 0.98, "y": 0.5, "xanchor": "right"})

    @staticmethod
    def _family(x, curves):
        """A bundle of curves as one trace, separated by gaps.

        A NaN breaks a plotly line, so every realization can be laid end to end
        in a single trace instead of costing one of its own.
        """
        n_cutoffs, n_curves = curves.shape
        gap = _np.full([1, n_curves], _np.nan)
        stacked = _np.vstack([curves, gap])
        repeated = _np.concatenate([_np.asarray(x, dtype=float), [_np.nan]])
        return (_np.tile(repeated, n_curves),
                stacked.T.reshape(-1))

    @staticmethod
    def _faint(color):
        """A hex colour, watered down to something a gridline can be."""
        red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        return "rgba(%d,%d,%d,0.25)" % (red, green, blue)

    # ------------------------------------------------------------------ #
    # the matrix
    # ------------------------------------------------------------------ #
    def _matrix(self, values, labels, series, measured, rows, kind="scatter",
                alpha=0.7, bins=60, log_counts=False, upper=None, size=5,
                density=False):
        """A scatter matrix: pairs below the diagonal, distributions along it.

        The upper triangle is the lower one transposed, so it is left out
        unless something else is asked for it -- half the ink for all of the
        information.
        """
        if kind == "hist2d" and len(series) > 1:
            # counts make one surface, and several laid over each other read
            # as none of them, so the categories are pooled here
            series = [(None, None)]

        n = len(labels)
        figure = _subplots(rows=n, cols=n, shared_xaxes=True,
                           horizontal_spacing=0.02, vertical_spacing=0.02)
        # a category belongs in the legend once, not once per panel; the rest
        # of its traces join it by `legendgroup`, so clicking the entry still
        # takes the category out of the whole matrix
        named = False

        for row in range(n):
            for column in range(n):
                if column > row:
                    if upper is None:
                        continue
                    self._upper(figure, values[measured, column],
                                values[measured, row], upper, bins,
                                log_counts, row, column, n)
                    continue

                for j, (name, mask) in enumerate(series):
                    keep = measured if mask is None else mask
                    color = self._color(j, name)
                    legend = name is not None and not named
                    if row == column:
                        edges = _np.histogram_bin_edges(values[keep, row],
                                                        bins=20)
                        self._bars(figure, values[keep, row], rows[keep],
                                   edges, color, name, legend,
                                   row + 1, column + 1, density=density)
                    else:
                        self._pair(figure, values[keep, column],
                                   values[keep, row], rows[keep], kind, alpha,
                                   size, bins, log_counts, color, name,
                                   legend, row + 1, column + 1)
                named = True

        self._label_edges(figure, labels, n)
        self._match_rows(figure, n, upper)
        return self._finish(figure, barmode="overlay")

    def _pair(self, figure, x, y, rows, kind, alpha, size, bins, log_counts,
              color, name, legend, row, column):
        """One off-diagonal panel: a cloud of points, or counts in cells."""
        if kind == "hist2d":
            self._cells(figure, x, y, bins, log_counts, row, column)
        else:
            self._points(figure, x, y, rows, size, alpha, color, name,
                         legend, row, column)

    def _upper(self, figure, x, y, upper, bins, log_counts, row, column, n):
        """
        The upper triangle: the same pair, told the other way.

        The lower triangle answers "which category is this point"; up here the
        categories are pooled on purpose, so the question becomes "where does
        the data sit, all of it together".
        """
        if upper == "hist2d":
            self._cells(figure, x, y, bins, log_counts, row + 1, column + 1)
        elif upper == "density":
            self._density(figure, x, y, row + 1, column + 1)
        elif upper == "correlation":
            # An axis carrying no trace is dropped on the way to the drawn
            # figure, and an annotation pinned to it goes with it -- so the
            # number needs a panel to sit in before it can be written there.
            figure.add_trace(
                _go.Scatter(x=[None], y=[None], mode="markers",
                            showlegend=False, hoverinfo="skip"),
                row=row + 1, col=column + 1)
            self._correlation(figure, x, y, column, row, n)
            figure.update_xaxes(showticklabels=False, showgrid=False,
                                row=row + 1, col=column + 1)
            figure.update_yaxes(showticklabels=False, showgrid=False,
                                row=row + 1, col=column + 1)
        else:
            raise ValueError(
                "upper must be 'hist2d', 'density' or 'correlation'; got %r"
                % upper)

    def _correlation(self, figure, x, y, column, row, n, corner=False):
        """What the eye is being asked about, as a number."""
        correlation = _np.corrcoef(x, y)[0, 1]
        suffix = self._axis(row + 1, column + 1, n)
        place = {"x": 0.05, "y": 0.95, "xanchor": "left", "yanchor": "top",
                 "text": "r = %.2f" % correlation, "font": {"size": 9}} \
            if corner else \
            {"x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle",
             "text": "%.2f" % correlation,
             # the stronger it is, the larger it reads
             "font": {"size": 10 + 14 * abs(correlation),
                      "color": _style.color(0) if correlation >= 0
                      else _style.color(3)}}

        figure.add_annotation(
            xref="x%s domain" % suffix, yref="y%s domain" % suffix,
            showarrow=False, bgcolor="rgba(255,255,255,0.5)",
            bordercolor=INK if corner else None,
            borderwidth=0.6 if corner else 0, borderpad=2, **place)

    def _normal(self, figure, values, row):
        """The normal of the same mean and spread, over the histogram."""
        curve = _prep.normal_curve(values, _np.min(values), _np.max(values))
        if curve is None:
            return
        self._line(figure, curve[0], curve[1], INK, row + 1, row + 1,
                   width=1.2)

    def _axes_of_variation(self, figure, analysis, x_column, y_column, count):
        """
        The principal components, drawn back onto the data they came from.

        A component is a direction in the space of the measurements, so in a
        panel showing two of them it is the pair of entries belonging to those
        two. Each line runs one standard deviation of that component either
        side of the mean, which puts the components in the data's own units:
        the first is longest because it carries the most variance, and a line
        lying flat along an axis means that measurement moves on its own. The
        sign of a component means nothing, hence a line rather than an arrow.
        """
        mean = analysis["mean"]
        loadings = analysis["loadings"]
        deviations = _np.sqrt(_np.maximum(analysis["eigenvalues"], 0.0))

        for k in range(min(count, loadings.shape[1])):
            step = loadings[:, k] * deviations[k]
            x = [mean[x_column] - step[x_column],
                 mean[x_column] + step[x_column]]
            y = [mean[y_column] - step[y_column],
                 mean[y_column] + step[y_column]]
            figure.add_trace(
                _go.Scatter(x=x, y=y, mode="lines+text",
                            text=["", "PC%d" % (k + 1)],
                            textposition="top right",
                            textfont={"size": 8, "color": INK},
                            line={"color": INK, "width": 1.4},
                            opacity=0.85, showlegend=False,
                            hovertemplate="PC%d<extra></extra>" % (k + 1)),
                row=y_column + 1, col=x_column + 1)

    def _loadings(self, figure, loadings, labels, x_component, y_component,
                  scores, n):
        """An arrow per original column, over the cloud of scores.

        Each axis is stretched to its own component's spread. A single scale
        for the whole figure would suit the first component, which carries most
        of the variance, and send every arrow off the edge of the others.

        Two annotations per column, not one. Plotly draws an annotation's text
        at the *tail* of its arrow and points the head at the position given,
        so a single annotation carrying both would pile every name on the
        origin the arrows all start from. The arrow is drawn with no text, and
        the name is placed at the head, which is where the eye goes looking
        for it.
        """
        x_reach = _np.max(_np.abs(scores[:, x_component])) or 1.0
        y_reach = _np.max(_np.abs(scores[:, y_component])) or 1.0
        suffix = self._axis(y_component + 1, x_component + 1, n)
        axes = {"xref": "x%s" % suffix, "yref": "y%s" % suffix}

        for i, label in enumerate(labels):
            x = loadings[i, x_component] * x_reach
            y = loadings[i, y_component] * y_reach

            figure.add_annotation(
                x=x, y=y, ax=0, ay=0, text="",
                axref="x%s" % suffix, ayref="y%s" % suffix,
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=0.9,
                arrowcolor=INK, opacity=0.9, **axes)
            figure.add_annotation(
                x=x, y=y, text=label, showarrow=False,
                font={"size": 8, "color": INK},
                bgcolor="rgba(255,255,255,0.5)", bordercolor=INK,
                borderwidth=0.6, borderpad=2,
                xanchor="left", yanchor="bottom", **axes)

    @staticmethod
    def _label_edges(figure, labels, n):
        """
        The names, written once along the bottom and once down the side.

        A panel away from the edges keeps no tick labels of its own: its scales
        are the ones already written along the edges of the matrix, and
        repeating them in every cell is what makes a scatter matrix unreadable.
        The x axes were shared down each column when the figure was made, which
        already clears theirs.
        """
        for i, label in enumerate(labels):
            figure.update_xaxes(title_text=label, row=n, col=i + 1)
            figure.update_yaxes(title_text=label, row=i + 1, col=1)
        for row in range(1, n + 1):
            for column in range(2, n + 1):
                figure.update_yaxes(showticklabels=False, row=row, col=column)

    def _match_rows(self, figure, n, upper):
        """
        Zooming one panel takes its whole row and column with it.

        `make_subplots` was asked to share the x axis down each column, which
        is what puts one variable on one scale wherever it is drawn
        horizontally. The same has to hold across a row -- except for the
        diagonal, whose vertical axis is a count and not the variable at all,
        and which would be flattened out of existence by being matched to it.
        """
        for row in range(1, n + 1):
            columns = [c for c in range(1, n + 1)
                       if c != row and (c < row or upper is not None)]
            if len(columns) < 2:
                continue
            reference = "y%s" % self._axis(row, columns[0], n)
            for column in columns[1:]:
                figure.update_yaxes(matches=reference, row=row, col=column)

    # ------------------------------------------------------------------ #
    # predicted against measured
    # ------------------------------------------------------------------ #
    def _agreement(self, figure, true, predicted, rows, kind, alpha, bins,
                   log_counts, size, row, column, legend=True):
        """Predicted against measured, with the line they would sit on."""
        self._pair(figure, true, predicted, rows, kind, alpha, size, bins,
                   log_counts, _style.color(0), "measured", legend, row,
                   column)

        low = min(_np.min(true), _np.min(predicted))
        high = max(_np.max(true), _np.max(predicted))
        margin = 0.05 * (high - low) if high > low else 1.0
        self._line(figure, [low, high], [low, high], REFERENCE, row, column,
                   width=1.0, dash="dash")

        # the same limits on both axes, so the line runs at 45 degrees and a
        # cloud leaning off it is leaning visibly
        window = [low - margin, high + margin]
        figure.update_xaxes(range=window, row=row, col=column)
        figure.update_yaxes(range=window, row=row, col=column)

    def _joint(self, true, predicted, rows, label, kind, alpha, bins,
               log_counts, size, height, width):
        """One variable, with the two distributions along the sides."""
        figure = _subplots(
            rows=2, cols=2, column_widths=[0.82, 0.18],
            row_heights=[0.18, 0.82], shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.02, vertical_spacing=0.02)

        self._agreement(figure, true, predicted, rows, kind, alpha, bins,
                        log_counts, size, 2, 1, legend=False)
        self._bars(figure, true, rows,
                   _np.histogram_bin_edges(true, bins=25),
                   _style.color(0), "measured", False, 1, 1)
        figure.add_trace(
            _go.Histogram(
                y=predicted, customdata=rows, nbinsy=25,
                marker={"color": _style.color(0)}, opacity=0.65,
                unselected={"marker": {"opacity": 0.15}},
                showlegend=False),
            row=2, col=2)

        figure.update_xaxes(title_text="measured", row=2, col=1)
        figure.update_yaxes(title_text="predicted", row=2, col=1)
        figure.update_yaxes(title_text="count", row=1, col=1)
        figure.update_xaxes(title_text="count", row=2, col=2)
        return self._finish(figure, title=label, height=height or 620,
                            width=width or 640, showlegend=False,
                            barmode="overlay")
