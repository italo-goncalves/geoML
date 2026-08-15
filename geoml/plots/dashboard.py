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
Several figures on one page, sharing a selection.

A figure answers one question. Most questions worth asking of spatial data are
about two things at once -- *those* samples, the ones out on the tail of the
histogram, where are they on the map? -- and no single figure answers that. A
dashboard does: drag a box over any panel and the same locations light up in
every other one.

.. code-block:: python

    eda = geoml.plots.Interactive(point, continuous="Elements",
                                  categorical="Rock")
    board = eda.dashboard()
    board.write_html("jura.html")

`board` renders itself in a notebook, and `write_html` writes a page that
carries plotly with it, so it can be sent to someone who has neither geoML nor
a network.

Take the long way for control over what goes on it, and over how it is laid
out -- a nested entry is a row of its own, so the rows need not be equal:

.. code-block:: python

    geoml.plots.Dashboard(
        [("Where the samples are", eda.scene(color="Cd")),
         [eda.histogram(), eda.training_curve()],
         eda.pairs(log=True, upper="density")],
        title="Jura, before modelling")

What links the panels is `customdata`: every trace `Interactive` draws from
locations carries the row of the container each of its points came from, and
the script this module writes matches selections by those rows. Anything
without them -- a counted cell, a simulated value, a training curve, a 3D
scene -- is simply left alone when a selection is made, which is the honest
answer for a panel whose points are not places.

3D scenes are linked the other way. There is no brushing them, but there is
turning them, and a page of several answers a question one of them cannot: the
same body of ground under two variables, seen from the same angle. Turn one and
the rest follow.
"""
import html as _html
import uuid as _uuid

import plotly.graph_objects as _go
import plotly.io as _pio
import plotly.offline as _poff


# Bound after every panel has been drawn. Selections are matched on the row
# index each point carries, not on position, so a point may be a bar in one
# panel and a dot in another and still be the same sample.
_SCRIPT = r"""
(function () {
  var ids = %(ids)s;
  var busy = false;
  var queued = null;
  var turning = false;
  var showing = {};
  var weight = {};
  var carries = {};

  function chosen(event) {
    if (!event || !event.points || !event.points.length) { return null; }
    var rows = {};
    var any = false;
    for (var i = 0; i < event.points.length; i++) {
      var row = event.points[i].customdata;
      if (row !== undefined && row !== null) { rows[row] = true; any = true; }
    }
    // A selection holding nothing counts as no selection, for two reasons.
    // Dragging a box over empty ground reads as a slip rather than as an
    // instruction to dim the world; and taking a box off a panel makes plotly
    // announce an empty selection, which would otherwise come back round as
    // an order to dim every other panel just after they were cleared.
    return any ? rows : null;
  }

  function apply(id, rows) {
    var gd = document.getElementById(id);
    if (!gd || !gd.data) { return; }
    survey(id);

    // A box dragged over a panel outlives the points it picked: plotly keeps
    // it in the layout and works the selection out from it again, so the box
    // has to go before the points will, or it simply puts them back.
    if (rows === null && gd.layout.selections
        && gd.layout.selections.length) {
      Plotly.relayout(gd, {selections: null});
    }

    var picked = [];
    var every = [];
    for (var t = 0; t < gd.data.length; t++) {
      var carried = gd.data[t].customdata;
      every.push(t);
      if (rows === null || !carried || !carried.length) {
        picked.push(null);
        continue;
      }
      var here = [];
      for (var i = 0; i < carried.length; i++) {
        if (rows[carried[i]]) { here.push(i); }
      }
      picked.push(here);
    }
    Plotly.restyle(gd, {selectedpoints: picked}, every);
    showing[id] = rows !== null;
  }

  // What a panel costs to redraw, and whether it can show a selection at all.
  // Plotly redraws every trace of a figure whatever changed, at a cost of
  // about a millisecond each, so both are worth knowing before asking it to.
  function survey(id) {
    if (weight[id] !== undefined) { return; }
    var gd = document.getElementById(id);
    weight[id] = gd && gd.data ? gd.data.length : 0;
    carries[id] = false;
    for (var t = 0; gd && gd.data && t < gd.data.length; t++) {
      var c = gd.data[t].customdata;
      if (c && c.length) { carries[id] = true; break; }
    }
  }

  function wanted(id, rows) {
    survey(id);
    // a panel drawn from no locations -- a training curve, a grade-tonnage --
    // can never show a selection, and redrawing it would only cost
    if (!carries[id]) { return false; }
    if (rows !== null) { return true; }

    // Clearing: worth doing where there is something to clear, which is
    // either a selection this script put on the panel or a box the user drew
    // on it themselves. The second is the case of dragging over one panel and
    // then double-clicking on another -- the box belongs to the first, which
    // has no `showing` of ours to go by.
    var gd = document.getElementById(id);
    return !!showing[id] || !!(gd && gd.layout && gd.layout.selections
                               && gd.layout.selections.length);
  }

  function share(from, rows) {
    // the newest request wins: a walk in progress finishes the panel it is on
    // and then picks this up, rather than the request being dropped
    queued = {from: from, rows: rows};
    if (!busy) { drain(); }
  }

  function drain() {
    if (!queued) { busy = false; return; }
    busy = true;
    var job = queued;
    queued = null;

    var todo = ids.filter(function (id) {
      // the panel the selection was made on is plotly's to draw: it has the
      // box on it, and restyling it from here would fight whoever is dragging
      return id !== job.from && wanted(id, job.rows);
    });
    // Lightest first. The work is the same either way, but a map that lights
    // up at once and a scatter matrix that follows reads as quick, where
    // everything arriving together after a frozen half-second does not.
    todo.sort(function (a, b) { return weight[a] - weight[b]; });

    (function step(i) {
      if (i >= todo.length) { drain(); return; }
      apply(todo[i], job.rows);
      // hand the browser back to whoever is using it: this is what lets each
      // panel appear as it is finished instead of all of them at the end
      window.setTimeout(function () { step(i + 1); }, 0);
    })(0);
  }

  // ---- the 3D scenes turn together ---------------------------------- //
  function scenesOf(gd) {
    var names = [];
    if (!gd || !gd.layout) { return names; }
    for (var key in gd.layout) {
      if (/^scene[0-9]*$/.test(key)) { names.push(key); }
    }
    return names;
  }

  function sameView(a, b) {
    return !!a && !!b && JSON.stringify(a) === JSON.stringify(b);
  }

  function turn(from, camera) {
    if (turning) { return; }
    turning = true;
    ids.forEach(function (id) {
      if (id === from) { return; }
      var gd = document.getElementById(id);
      var move = {};
      scenesOf(gd).forEach(function (name) {
        // Skipping a scene already at this view is what ends the exchange.
        // Plotly answers a relayout with a relayout event of its own, and it
        // arrives after this function has finished, so a flag alone would not
        // stop the panels handing the camera back and forth for ever.
        if (!sameView(gd.layout[name].camera, camera)) {
          move[name + '.camera'] = camera;
        }
      });
      if (Object.keys(move).length) { Plotly.relayout(gd, move); }
    });
    turning = false;
  }

  function turned(event) {
    if (!event) { return null; }
    for (var key in event) {
      if (/\.camera$/.test(key)) { return event[key]; }
    }
    return null;
  }

  function wire() {
    for (var i = 0; i < ids.length; i++) {
      var gd = document.getElementById(ids[i]);
      // the panels are drawn by scripts of their own, and a graph div only
      // takes listeners once plotly has been through it
      if (!gd || typeof gd.on !== 'function') {
        window.setTimeout(wire, 50);
        return;
      }
    }
    ids.forEach(function (id) {
      var gd = document.getElementById(id);
      gd.on('plotly_selected', function (event) {
        share(id, chosen(event));
      });
      gd.on('plotly_deselect', function () {
        share(null, null);
      });
      gd.on('plotly_doubleclick', function () {
        share(null, null);
      });
      gd.on('plotly_relayout', function (event) {
        var camera = turned(event);
        if (camera) { turn(id, camera); }
      });
    });
  }

  wire();
})();
"""

_STYLE = """
.geoml-board {
  font-family: sans-serif;
  color: #2b2b2b;
  background: white;
  padding: 8px 16px 24px 16px;
}
.geoml-board h1 {
  font-size: 20px;
  font-weight: 600;
  margin: 12px 0 4px 0;
}
.geoml-board p.geoml-hint {
  font-size: 12px;
  color: #6c757d;
  margin: 0 0 16px 0;
}
.geoml-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.geoml-row {
  display: grid;
  gap: 16px;
  align-items: start;
}
.geoml-card {
  border: 1px solid #e2e2e2;
  border-radius: 6px;
  padding: 4px;
  overflow: hidden;
  min-width: 0;
}
/* the wrapper plotly puts around its graph div, which has to carry the
   card's height down to it for a responsive plot to have one to fill */
.geoml-card > div { height: 100%; width: 100%; }
@media (max-width: 900px) {
  .geoml-row { grid-template-columns: minmax(0, 1fr) !important; }
}
"""

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>%(title)s</title>
<style>%(style)s</style>
%(library)s
</head>
<body>
%(body)s
</body>
</html>
"""

HINT = ("Drag a box or a lasso over any panel and the same locations light up "
        "in the others. Double-click to clear.")

# said only when there is more than one scene for it to be true of
HINT_3D = " Turning one 3D scene turns the rest with it."

# a trace that lives in a 3D scene, where there is no box to drag
_SPATIAL = ("scatter3d", "surface", "mesh3d", "cone", "volume", "isosurface",
            "streamtube")


class Dashboard(object):
    """
    A page of figures whose selections are shared.

    Parameters
    ----------
    figures : list
        The figures, in the order they are to be laid out. An entry may be:

        - a **figure**;
        - a **`(caption, figure)` pair** -- the caption replaces whatever title
          the figure came with, which is how the same figure serves in one
          place under its own name and in another under the question it is
          there to answer;
        - a **list or tuple of either**, which is a row of its own, however
          many panels are in it.

        Nest one entry and the whole list is read as rows, a bare figure
        counting as a row of one; nest none and the figures are dealt out
        `columns` to a row. Rows are what a page usually wants: a map beside a
        histogram, and the scatter matrix that goes with them across the full
        width underneath.

            Dashboard([eda.scene(),
                       [eda.histogram(), eda.training_curve()],
                       eda.pairs()])
    title : str
        The heading over the page.
    columns : int
        How many panels to a row, when the rows were not given outright. A
        narrow window falls back to one, whatever this says.
    plotlyjs : str
        `"embed"` writes plotly into the page, some four megabytes of it, and
        the result opens anywhere with no network at all -- which is what
        makes it a report rather than a link. `"cdn"` fetches the library
        instead, leaving a file small enough to keep in a repository or a
        notebook, and needing a connection to open.
    hint : str or bool
        The line under the heading saying what can be done with the page.
        False leaves it out.
    """

    def __init__(self, figures, title=None, columns=2, plotlyjs="embed",
                 hint=True):
        if plotlyjs not in ("embed", "cdn"):
            raise ValueError(
                "plotlyjs must be 'embed', which carries the library in the "
                "page, or 'cdn', which fetches it; got %r" % plotlyjs)
        if not figures:
            raise ValueError("a dashboard needs at least one figure")

        self.title = title
        self.columns = max(1, int(columns))
        self.plotlyjs = plotlyjs

        # `figures` is flat whatever the caller passed, since the ids and the
        # linking script want one list; `rows` says which of them share a row
        self.figures, self.shapes, self.rows = [], [], []
        for entry in self._as_rows(figures, self.columns):
            row = []
            for item in entry:
                caption, figure = item if self._is_captioned(item) \
                    else (None, item)
                figure, shape = self._prepare(figure, caption)
                row.append(len(self.figures))
                self.figures.append(figure)
                self.shapes.append(shape)
            self.rows.append(row)

        if hint is True:
            hint = HINT + (HINT_3D if self._scenes() > 1 else "")
        self.hint = hint

        self.id = _uuid.uuid4().hex[:10]

    def __repr__(self):
        return "Dashboard(%d figures in %d rows)" % (len(self.figures),
                                                     len(self.rows))

    @staticmethod
    def _is_captioned(entry):
        """
        A `(caption, figure)` pair, and not a row holding two figures.

        Both arrive as a tuple, so they are told apart by what is in front: a
        caption is a name, and nothing else that can be passed here is.
        """
        return (isinstance(entry, tuple) and len(entry) == 2
                and isinstance(entry[0], str))

    @classmethod
    def _as_rows(cls, figures, columns):
        """
        The entries grouped into the rows they are to be laid out in.

        Nesting one entry says the caller is arranging the page themselves, so
        every entry is then a row and a bare figure is a row of one. Nesting
        none leaves it to `columns`.
        """
        def is_row(entry):
            return isinstance(entry, (list, tuple)) \
                and not cls._is_captioned(entry)

        if any(is_row(entry) for entry in figures):
            return [list(entry) if is_row(entry) else [entry]
                    for entry in figures]
        return [figures[i:i + columns]
                for i in range(0, len(figures), columns)]

    def _scenes(self):
        """How many panels hold a 3D scene, which turn together."""
        return sum(any(trace.type in _SPATIAL for trace in figure.data)
                   for figure in self.figures)

    @staticmethod
    def _prepare(figure, caption):
        """
        A copy of the figure, freed to fill whatever column it lands in, and
        the proportions it was drawn at.

        The copy is the point: a dashboard takes the size and the drag mode
        into its own hands, and the figure it was handed belongs to whoever
        built it and may be on screen already.

        The proportions are kept because a figure is not a rectangle to be
        stretched. A scatter matrix is drawn about as tall as it is wide, and
        pouring one into half a page turns every panel into a slot. Handing
        the shape to the page as an aspect ratio makes the matrix smaller
        instead of narrower, which is a matrix one can still read.
        """
        figure = _go.Figure(figure)
        shape = ((figure.layout.width or 900),
                 (figure.layout.height or 450))
        figure.update_layout(width=None, height=None, autosize=True)
        if caption is not None:
            figure.update_layout(title_text=caption)

        # A box to drag is the whole point of the page, so it is the tool the
        # panel opens with -- except where there is no box to drag, and asking
        # a 3D scene to select would only take away its rotation.
        if not any(trace.type in _SPATIAL for trace in figure.data):
            figure.update_layout(dragmode="select")
        return figure, shape

    def height(self, page=1000):
        """
        How tall the page comes out, on one `page` pixels wide.

        Only an estimate, and only wanted for the height to give the frame a
        notebook shows this in: the panels are laid out by the browser, and
        what they end up as depends on the width they are given. A row is as
        tall as the tallest panel in it, and a panel is as tall as its own
        proportions make it once it has been given its share of the width.
        """
        total = 0.0
        for row in self.rows:
            column = (page - 32 - 16 * (len(row) - 1)) / len(row)
            total += max(column * self.shapes[i][1] / self.shapes[i][0]
                         for i in row)
        return int(total + 16 * len(self.rows) + 130)

    # ------------------------------------------------------------------ #
    # the page
    # ------------------------------------------------------------------ #
    def _divs(self):
        """The rows of cards, and the ids the panels were given."""
        pieces, ids = [], []
        for row in self.rows:
            pieces.append('<div class="geoml-row" style="grid-template-'
                          'columns: repeat(%d, minmax(0, 1fr))">' % len(row))
            for i in row:
                div_id = "geoml-%s-%d" % (self.id, i)
                width, height = self.shapes[i]
                ids.append(div_id)
                pieces.append(
                    '<div class="geoml-card" style="aspect-ratio: %g / %g">%s'
                    '</div>'
                    % (width, height,
                       _pio.to_html(self.figures[i], include_plotlyjs=False,
                                    full_html=False, div_id=div_id,
                                    config={"responsive": True,
                                            "displaylogo": False})))
            pieces.append("</div>")
        return pieces, ids

    def to_html(self, full=True):
        """
        The page, as text.

        Parameters
        ----------
        full : bool
            A whole document, head and all. False gives the piece that goes
            inside one, for pasting into a page of your own -- and then
            plotly has to be loaded by that page, since a fragment carries no
            head to put it in.
        """
        pieces, ids = self._divs()

        body = ['<div class="geoml-board">']
        if self.title:
            body.append("<h1>%s</h1>" % _html.escape(str(self.title)))
        if self.hint:
            body.append('<p class="geoml-hint">%s</p>'
                        % _html.escape(str(self.hint)))
        body.append('<div class="geoml-grid">')
        body.extend(pieces)
        body.append("</div>")
        body.append("<script>%s</script>" % (_SCRIPT % {"ids": repr(ids)}))
        body.append("</div>")
        body = "\n".join(body)

        if not full:
            return "<style>%s</style>\n%s" % (_STYLE, body)

        return _PAGE % {
            "title": _html.escape(str(self.title or "geoML")),
            "style": _STYLE,
            "library": self._library(),
            "body": body}

    def _library(self):
        """Plotly itself: written into the page, or fetched by it."""
        if self.plotlyjs == "embed":
            return "<script>%s</script>" % _poff.get_plotlyjs()
        return ('<script src="https://cdn.plot.ly/plotly-%s.min.js" '
                'charset="utf-8"></script>' % _poff.get_plotlyjs_version())

    def write_html(self, path):
        """Write the page to a file, and hand back the path."""
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.to_html())
        return path

    def _repr_html_(self):
        """
        What a notebook shows.

        The page goes in an iframe rather than straight into the notebook's
        own document. A dashboard is a page with a script, and a notebook is
        not obliged to run scripts it is handed -- JupyterLab and VS Code do
        not, which is why a bare `<script>` works in one notebook and silently
        does nothing in the next. An iframe is a document in its own right and
        runs its own, everywhere. It also keeps the styles and the element ids
        to itself, so several dashboards in one notebook cannot tread on each
        other.
        """
        return ('<iframe srcdoc="%s" width="100%%" height="%d" '
                'style="border:none" frameborder="0"></iframe>'
                % (_html.escape(self.to_html(), quote=True), self.height()))

    def show(self, path=None):
        """Write the page and open it in a browser."""
        import tempfile
        import webbrowser

        if path is None:
            handle = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8")
            with handle as file:
                file.write(self.to_html())
            path = handle.name
        else:
            self.write_html(path)

        webbrowser.open("file://%s" % path)
        return path
