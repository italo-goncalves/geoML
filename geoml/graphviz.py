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
Models and latent networks as Graphviz diagrams.

This module writes a DOT description and stops there, the way `plotly` builds a
figure and leaves the drawing to someone else: nothing is imported, so Graphviz
is needed to look at a diagram but never to produce one. Render it with

    dot -Tpng network.dot -o network.png

or paste it into any Graphviz viewer.

A diagram carries what the printed tree cannot: which node feeds which, once,
for a node shared by several branches. Boxes are coloured by the part they
play -- the coordinates that go in, the latent nodes, the warpings on the way
out, and the variables that come out -- and every edge is labelled with the
number of variables travelling along it.
"""
import geoml.latent as _latent
import geoml.warping as _warp


PALETTE = {
    "input": "#1f6f8b",
    "latent": "#3f3f3f",
    "warping": "#e0761f",
    "output": "#1e7b45",
}

_LEGEND = (("input", "Input variable"), ("latent", "Latent variable"),
           ("warping", "Transformation"), ("output", "Output variable"))


def _escape(text):
    """Quotes, backslashes and line breaks, as DOT wants them."""
    return (str(text).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n"))


def _parents(node):
    """The nodes feeding this one, whichever kind of node it is."""
    parents = getattr(node, "parents", None)
    if parents is not None:
        return list(parents)
    parent = getattr(node, "parent", None)
    return [] if parent is None else [parent]


def _label(node, coordinates=None):
    """The node's name, over what it is.

    The class is left off when the name already carries it, which is what an
    automatic name looks like: `BasicGP_2` over `Matern32`, not over
    `BasicGP (Matern32)`.
    """
    class_name = type(node).__name__
    name = getattr(node, "name", None) or class_name

    if coordinates is not None:
        return "%s\n%s" % (name, ", ".join(str(c) for c in coordinates))

    kernel = getattr(node, "kernel", None)
    detail = class_name if not name.startswith(class_name) else ""
    if kernel is not None:
        kernel = type(kernel).__name__
        detail = "%s (%s)" % (detail, kernel) if detail else kernel
    return "%s\n%s" % (name, detail) if detail else name


class _Diagram(object):
    """Collects boxes and arrows, then writes them out as DOT."""

    def __init__(self):
        self.statements = []
        self.ids = {}

    def box(self, key, label, role):
        """Declares a box once, and returns the identifier that refers to it.

        Boxes are identified by position, not by name: a name may repeat (two
        subnetworks built apart can carry the same automatic one) and may hold
        any character the user likes, neither of which an identifier tolerates.
        """
        if key in self.ids:
            return self.ids[key]

        name = "n%d" % len(self.ids)
        self.ids[key] = name
        if role == "concat":
            # the glyph goes inside the circle, so the node's name goes beside
            # it -- a concatenation is a node like any other and can be looked
            # up by name
            self.statements.append(
                '    %s [label="+", xlabel="%s", shape=circle, width=0.35, '
                'fillcolor="#ffffff", fontcolor="#000000", penwidth=1];'
                % (name, _escape(label)))
        else:
            self.statements.append(
                '    %s [label="%s", fillcolor="%s"];'
                % (name, _escape(label), PALETTE[role]))
        return name

    def arrow(self, tail, head, size, both=False):
        """Draws an arrow, headed at both ends when the step is invertible."""
        self.statements.append(
            '    %s -> %s [label=" %s"%s];'
            % (tail, head, size, ", dir=both" if both else ""))


def _add_network(diagram, top, coordinates=None):
    """Walks everything feeding `top`, and returns the box it ended on.

    Every node is boxed before any arrow is drawn: a node's role depends on
    what feeds it, so it cannot be settled while looking at it from below.
    """
    nodes, seen, stack = [], set(), [top]
    while len(stack) > 0:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        stack.extend(_parents(node))

    for node in nodes:
        if len(_parents(node)) == 0:
            role = "input"
        elif isinstance(node, _latent.Stack):
            role = "concat"
        else:
            role = "latent"
        diagram.box(
            id(node), _label(node, coordinates if role == "input" else None),
            role)

    for node in nodes:
        for parent in _parents(node):
            diagram.arrow(diagram.ids[id(parent)], diagram.ids[id(node)],
                          parent.size)

    return diagram.ids[id(top)]


def _add_head(diagram, model, source):
    """The likelihoods hanging off the network: warpings, then variables.

    The warpings are drawn backwards, the way the model generates a value
    rather than the way it reads one, so the arrows run with the rest of the
    diagram. `Identity` is left out, having nothing to show.
    """
    for name, likelihood, size in zip(model.variables, model.likelihoods,
                                      model.lik_sizes):
        warping = getattr(likelihood, "warping", None)
        chain = list(getattr(warping, "warpings", [warping]))
        chain = [w for w in chain
                 if w is not None and not isinstance(w, _warp.Identity)]

        # a warping is a two-way step -- forward to read a value into the
        # latent space, backward to generate one -- so every arrow from the
        # first warping onwards is headed at both ends. The one that leaves the
        # latent network is not: only that direction generates.
        tail, both = source, False
        for wrapping in reversed(chain):
            head = diagram.box(id(wrapping), type(wrapping).__name__, "warping")
            diagram.arrow(tail, head, size, both=both)
            tail, size, both = head, wrapping.size_in, True

        variable = model.data.variables.get(name)
        label = "%s\n%s" % (name, type(likelihood).__name__)
        head = diagram.box(("variable", name), label, "output")
        diagram.arrow(tail, head,
                      size if variable is None else variable.length, both=both)


def _legend():
    """The legend, as a cluster stacked along the flow.

    The entries are chained by invisible edges: nothing joins them otherwise,
    and Graphviz puts unconnected nodes on one rank, which spreads the legend
    into a row as wide as the diagram.
    """
    lines = ['    subgraph cluster_legend {',
             '        label="Legend";', '        color="#9a9a9a";',
             '        fontcolor="#000000";']
    for role, text in _LEGEND:
        lines.append('        legend_%s [label="%s", fillcolor="%s"];'
                     % (role, text, PALETTE[role]))
    lines.append('        legend_concat [label="+", shape=circle, width=0.35, '
                 'fillcolor="#ffffff", fontcolor="#000000", penwidth=1];')
    lines.append('        legend_concat_text [label="Concatenation", '
                 'shape=plaintext, style="", fontcolor="#000000"];')
    lines.append('        {rank=same; legend_concat; legend_concat_text;}')
    # last first: the flow runs upwards, so this reads top to bottom
    lines.append('        legend_concat -> legend_output -> legend_warping '
                 '-> legend_latent -> legend_input [style=invis];')
    lines.append('    }')
    return lines


def to_dot(obj, legend=True, rankdir="BT"):
    """
    Writes a model or a latent network as a Graphviz DOT description.

    Parameters
    ----------
    obj
        A `VGPNetwork` from the `models` module, or any node from the `latent`
        module. A model is drawn whole -- coordinates, latent network,
        warpings and output variables; a node is drawn with everything that
        feeds it.
    legend : bool
        Whether to include the legend.
    rankdir : str
        Direction of flow, in Graphviz's terms. The default `"BT"` puts the
        coordinates at the bottom and the variables on top.

    Returns
    -------
    dot : str
        The diagram, to save to a `.dot` file or hand to a Graphviz viewer.
    """
    network = getattr(obj, "latent_network", None)
    model = obj if network is not None else None
    if model is None:
        network = obj

    coordinates = None
    if model is not None:
        coordinates = getattr(model.data, "coordinate_labels", None)

    diagram = _Diagram()
    source = _add_network(diagram, network, coordinates)
    if model is not None:
        _add_head(diagram, model, source)

    lines = ["digraph geoml {",
             "    rankdir=%s;" % rankdir,
             '    node [shape=box, style="filled,rounded", '
             'fontname="Helvetica", fontcolor="#ffffff", penwidth=0];',
             '    edge [fontname="Helvetica", fontsize=10, '
             'color="#000000"];']
    lines += diagram.statements
    if legend:
        lines += _legend()
    lines.append("}")
    return "\n".join(lines)
