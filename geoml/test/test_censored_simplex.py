# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The censored likelihood of a composition on the simplex.

Gate 1 of the plan: every row form against a reference that shares none of
its code. A complete row, zeros included, against a dense grid in the shift
(the formula itself checked by the probabilities of a model's zero
patterns, drawn through the warping's own way back, adding up to one and
matching their frequencies); a row with hidden parts against the complete
rows integrated over every value the hidden parts could take; and the
same with a link that is not affine.
"""
import itertools

import numpy as np
import pytest
import scipy.stats as st
import tensorflow as tf

import geoml.stats.censored as cs
import geoml.warping as wp

D = 3
SCALE = np.array([0.5, 0.3, 0.2])
SD = np.array([0.35, 0.5, 0.6])


def _simplex(alpha, size=D, scale=SCALE):
    warping = wp.PowerSimplex(size, alpha=alpha)
    warping.scale = tf.constant(scale[None, :], tf.float64)
    return warping


def _links(kind, size=D):
    if kind == "none":
        return None
    if kind == "affine":
        link = wp.ZScore(size)
        link.initialize(np.random.default_rng(0).normal(0.3, 1.7, (50, size)))
        return link
    return wp.ChainedWarping(
        wp.SinhArcsinh(size, skewness=0.4, tailweight=1.3), wp.ZScore(size))


def _ours(rows, observed, f, alpha, kind, rule=None, sd=SD, scale=SCALE):
    simplex = _simplex(alpha, len(scale), scale)
    links = _links(kind, len(scale))
    out, _ = cs.log_likelihood(
        tf.constant(rows, tf.float64), tf.constant(observed),
        tf.constant(f, tf.float64), tf.constant(sd, tf.float64),
        cs.Simplex(simplex), cs.Links(links, len(scale)),
        rule or cs.Rule())
    return out.numpy()


# --------------------------------------------------------------------------- #
# the reference: (eq:general) on a dense grid, in NumPy
# --------------------------------------------------------------------------- #
_AFFINE = {}


def _affine(kind, size):
    """`psi(w) = a + s w` for the ZScore a chain ends with, read once."""
    key = (kind, size)
    if key not in _AFFINE:
        link = _links(kind, size)
        if kind == "sinh":
            link = link.warpings[-1]
        ends, _ = link.forward(tf.constant([[0.0] * size, [1.0] * size],
                                           tf.float64))
        ends = ends.numpy()
        _AFFINE[key] = (ends[0], ends[1] - ends[0])
    return _AFFINE[key]


def _psi(kind, w, size=D):
    """The links and their slopes in NumPy, written out by hand."""
    if kind == "none":
        return w, np.ones_like(w)
    a, s = _affine(kind, size)
    if kind == "affine":
        return a + s * w, np.broadcast_to(s, w.shape)
    inner = 1.3 * np.arcsinh(w) - 0.4
    u = np.sinh(inner)
    du = 1.3 * np.cosh(inner) / np.sqrt(1 + w ** 2)
    return a + s * u, s * du


def _reference_many(rows, f, alpha, kind, sd=SD, scale=SCALE):
    """The log-density of complete rows, zeros allowed, each on a grid of
    6001 shifts: (eq:general) written out in NumPy."""
    rows = np.atleast_2d(rows)
    if len(rows) > 256:
        return np.concatenate([
            _reference_many(rows[i:i + 256], f, alpha, kind, sd, scale)
            for i in range(0, len(rows), 256)])
    size = len(scale)
    positive = rows > 0
    ratio = np.where(positive, rows, 1.0) / scale
    b = (ratio ** alpha - 1) / alpha
    log_g = np.where(positive, (1 - alpha) * np.log(np.where(
        positive, rows, 1.0)) + alpha * np.log(scale), 0.0)
    g = np.where(positive, np.exp(log_g), 0.0)
    tau = np.linspace(-12, 12, 6001)
    w = np.where(positive[:, None, :], b[:, None, :] + tau[None, :, None],
                 tau[None, :, None] - 1.0 / alpha)
    value, slope = _psi(kind, w, size)
    z = (value - f) / sd
    log = np.where(positive[:, None, :],
                   st.norm.logpdf(z) - np.log(sd) + np.log(slope),
                   st.norm.logcdf(z)).sum(axis=2)
    top = log.max(axis=1, keepdims=True)
    integral = np.trapezoid(np.exp(log - top), tau, axis=1)
    return np.log(g.sum(axis=1)) - log_g.sum(axis=1) + top[:, 0]         + np.log(integral)


def _reference_complete(row, f, alpha, kind, sd=SD, scale=SCALE):
    return _reference_many(row, f, alpha, kind, sd, scale)[0]


def _segment(R, n=400):
    """Nodes and weights on (0, R), clustered at both ends by the cosine
    substitution, which takes the edges' x**(alpha - 1) at alpha = 1/2."""
    t, v = np.polynomial.legendre.leggauss(n)
    theta = (t + 1) / 2
    x = R * (1 - np.cos(np.pi * theta)) / 2
    return x, v / 2 * R * np.pi * np.sin(np.pi * theta) / 2


def _tetrahedron(R, n=28):
    """Nodes and weights on x > 0, sum(x) < R in three dimensions."""
    u, wu = _segment(R, n)
    out, weights = [], []
    for ui, wi in zip(u, wu):
        vw, wvw = _triangle(R - ui, n)
        out.append(np.concatenate([np.full((len(vw), 1), ui), vw], axis=1))
        weights.append(wi * wvw)
    return np.concatenate(out), np.concatenate(weights)


def _triangle(R, n=120):
    """Nodes and weights on the triangle u, v > 0, u + v < R."""
    u, wu = _segment(R, n)
    out, weights = [], []
    for ui, wi in zip(u, wu):
        v, wv = _segment(R - ui, n)
        out.append(np.stack([np.full(n, ui), v], axis=1))
        weights.append(wi * wv)
    return np.concatenate(out), np.concatenate(weights)


ROWS = [np.array([0.5, 0.3, 0.2]), np.array([0.05, 0.25, 0.7]),
        np.array([0.0, 0.4, 0.6]), np.array([0.7, 0.0, 0.3]),
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]
LATENT = np.array([0.2, -0.3, 0.4])


@pytest.mark.parametrize("kind", ["none", "affine", "sinh"])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_a_complete_row_matches_the_dense_grid(alpha, kind):
    rows = np.stack(ROWS)
    f = np.tile(LATENT[None, :, None], [len(rows), 1, 2])
    f[:, :, 1] += np.array([0.6, -0.2, 0.1])
    ours = _ours(rows, np.ones_like(rows, bool), f, alpha, kind)
    for i, row in enumerate(rows):
        for s in range(2):
            reference = _reference_complete(row, f[i, :, s], alpha, kind)
            assert ours[i, s] == pytest.approx(reference, abs=1e-4), (i, s)


# --------------------------------------------------------------------------- #
# the formula itself: the zero patterns' probabilities against draws
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["affine", "sinh"])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_the_patterns_add_up_to_one_and_match_their_frequencies(alpha,
                                                                kind):
    f = LATENT

    def density(rows):
        return np.exp(_reference_many(np.asarray(rows), f, alpha, kind))

    # vertices: probabilities; edges: densities in one coordinate; the
    # interior: a density in two
    vertices = list(density(np.eye(D)))
    u, wu = _segment(1.0)
    edges = []
    for k in range(D):
        i, j = [p for p in range(D) if p != k]
        rows = np.zeros((len(u), D))
        rows[:, i], rows[:, j] = u, 1 - u
        edges.append(np.sum(wu * density(rows)))
    uv, wuv = _triangle(1.0)
    rows = np.stack([uv[:, 0], uv[:, 1], 1 - uv.sum(axis=1)], axis=1)
    interior = np.sum(wuv * density(rows))
    assert sum(vertices) + sum(edges) + interior == pytest.approx(1, abs=2e-4)

    # the same probabilities as frequencies, drawn through the warping's
    # own way back
    rng = np.random.default_rng(5)
    n = 400_000
    latent = f[None, :] + SD[None, :] * rng.standard_normal((n, D))
    links = _links(kind)
    w = latent if links is None else links.backward(
        tf.constant(latent, tf.float64))
    x = _simplex(alpha).backward(tf.constant(w, tf.float64)).numpy()
    zeros = x <= 1e-12
    for k in range(D):
        frequency = np.mean((zeros.sum(axis=1) == D - 1) & ~zeros[:, k])
        sd = np.sqrt(frequency * (1 - frequency) / n) + 1e-6
        assert abs(frequency - vertices[k]) < 5 * sd, ("vertex", k)
        on_edge = np.mean((zeros.sum(axis=1) == 1) & zeros[:, k])
        sd = np.sqrt(on_edge * (1 - on_edge) / n) + 1e-6
        assert abs(on_edge - edges[k]) < 5 * sd, ("edge", k)


# --------------------------------------------------------------------------- #
# hidden parts against the complete rows, integrated
# --------------------------------------------------------------------------- #
def _reference_hidden(observed_row, hidden, f, alpha, kind, sd=SD,
                      scale=SCALE):
    """The density of the observed parts: every completion of the hidden
    ones, the complete rows integrated over the faces where some of them
    are zero."""
    R = 1 - np.nansum(np.where(hidden, 0, observed_row))
    cols = np.where(hidden)[0]
    m = len(cols)
    base = np.where(hidden, 0.0, observed_row)

    def completed(values):
        rows = np.tile(base, (len(values), 1))
        rows[:, cols] = values
        return np.exp(_reference_many(rows, f, alpha, kind, sd, scale))

    total = 0.0
    for q in range(1, m + 1):
        for face in itertools.combinations(range(m), q):
            if q == 1:
                values = np.zeros((1, m))
                values[0, face[0]] = R
                total += completed(values)[0]
                continue
            nodes, weights = _segment(R) if q == 2 else \
                _triangle(R) if q == 3 else _tetrahedron(R)
            nodes = nodes.reshape(len(weights), q - 1)
            values = np.zeros((len(weights), m))
            values[:, list(face[:-1])] = nodes
            values[:, face[-1]] = R - nodes.sum(axis=1)
            total += np.sum(weights * completed(values))
    return np.log(total)


HIDDEN = [
    # one hidden part: the completed row, exactly
    (np.array([0.3, np.nan, 0.2]), [False, True, False]),
    # two, with a zero observed among the others
    (np.array([0.35, np.nan, np.nan]), [False, True, True]),
    (np.array([0.0, np.nan, np.nan]), [False, True, True]),
]


@pytest.mark.parametrize("kind", ["affine", "sinh"])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
@pytest.mark.parametrize("case", range(len(HIDDEN)))
def test_hidden_parts_are_the_completions_integrated(case, alpha, kind):
    row, hidden = HIDDEN[case]
    hidden = np.array(hidden)
    f = LATENT[None, :, None]
    ours = _ours(np.where(hidden, 1.0, row)[None, :], ~hidden[None, :], f,
                 alpha, kind)[0, 0]
    reference = _reference_hidden(row, hidden, LATENT, alpha, kind)
    assert ours == pytest.approx(reference, abs=1e-3)


SCALE4 = np.array([0.4, 0.3, 0.2, 0.1])
SD4 = np.array([0.4, 0.5, 0.6, 0.45])
LATENT4 = np.array([0.2, -0.3, 0.4, 0.1])


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_three_hidden_parts_are_the_completions_integrated(alpha):
    row = np.array([0.3, np.nan, np.nan, np.nan])
    hidden = np.isnan(row)
    ours = _ours(np.where(hidden, 1.0, row)[None, :], ~hidden[None, :],
                 LATENT4[None, :, None], alpha, "affine", None, SD4,
                 SCALE4)[0, 0]
    reference = _reference_hidden(row, hidden, LATENT4, alpha, "affine",
                                  SD4, SCALE4)
    assert ours == pytest.approx(reference, abs=1e-3)


SCALE5 = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
SD5 = np.array([0.4, 0.5, 0.6, 0.45, 0.55])
LATENT5 = np.array([0.2, -0.3, 0.4, 0.1, -0.1])


def test_four_hidden_parts_are_the_completions_integrated():
    # four positive hidden parts take the Sobol rule, the only one
    # measured approximate; its error at the default count is the
    # benchmark's
    row = np.array([0.25, np.nan, np.nan, np.nan, np.nan])
    hidden = np.isnan(row)
    ours = _ours(np.where(hidden, 1.0, row)[None, :], ~hidden[None, :],
                 LATENT5[None, :, None], 1.0, "affine", None, SD5,
                 SCALE5)[0, 0]
    reference = _reference_hidden(row, hidden, LATENT5, 1.0, "affine",
                                  SD5, SCALE5)
    assert ours == pytest.approx(reference, abs=2e-2)


def test_a_remainder_of_nothing_makes_the_hidden_parts_zeros():
    rows = np.array([[0.4, 0.6, 1.0], [0.4, 0.6, 0.0]])
    observed = np.array([[True, True, False], [True, True, True]])
    f = np.tile(LATENT[None, :, None], [2, 1, 1])
    ours = _ours(rows, observed, f, 0.5, "affine")
    assert ours[0, 0] == pytest.approx(ours[1, 0], abs=1e-12)


def test_rows_with_nothing_observed_or_too_much_hidden_are_not_scored():
    rows = np.ones((2, D))
    observed = np.zeros((2, D), bool)
    observed[1, 0] = True
    rows[1, 0] = 0.2
    simplex = cs.Simplex(_simplex(0.5))
    out, scored = cs.log_likelihood(
        tf.constant(rows), tf.constant(observed),
        tf.constant(np.zeros((2, D, 1))), tf.constant(SD),
        simplex, cs.Links(None, D), cs.Rule(), max_hidden=1)
    assert not scored.numpy().any()
    assert (out.numpy() == 0).all()


def _tiled(links, v, columns, function):
    """Every value in every column, its own column read back: the form
    `Links._whole_rows` replaced, kept as the reference."""
    flat = tf.reshape(v, [-1])
    out = function(tf.tile(flat[:, None], [1, links.size]))
    own = tf.one_hot(tf.reshape(columns, [-1]), links.size,
                     dtype=tf.float64)
    return tf.reshape(tf.reduce_sum(out * own, axis=1), tf.shape(v))


@pytest.mark.parametrize("kind", ["spline", "links"])
def test_each_value_is_read_through_its_own_column_alone(kind):
    """A bent chain sees each value once, in its own column: the same
    numbers and gradients as placing it in every column, with the columns
    unevenly filled and one of them empty."""
    size = 5
    rng = np.random.default_rng(3)
    if kind == "spline":
        chain = wp.ChainedWarping(wp.ZScore(size),
                                  wp.Spline(size, backbone="rq"))
        chain.initialize(rng.gamma(2.0, 1.0, (200, size)))
    else:
        chain = _links("bent", size)
    links = cs.Links(chain, size)
    assert not links.affine
    v = tf.constant(rng.normal(0.0, 1.5, (7, 3, 4)))
    columns = tf.constant(rng.choice([0, 1, 3, 4], (7, 3, 4),
                                     p=[0.6, 0.2, 0.1, 0.1]))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(v)
        new = [links.forward(v, columns)[0], links.forward(v, columns)[1],
               links.backward(v, columns)]
        old = [_tiled(links, v, columns, links._apply),
               _tiled(links, v, columns,
                      lambda x: tf.math.log(_slope(links, x))),
               _tiled(links, v, columns, links._invert)]
    for a, b in zip(new, old):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    for a, b in zip(new, old):
        np.testing.assert_allclose(tape.gradient(a, v).numpy(),
                                   tape.gradient(b, v).numpy(),
                                   rtol=1e-12, atol=1e-14)
    # nothing to read: an empty batch of rows
    empty = tf.zeros((0, 3), tf.float64)
    assert links.backward(empty, tf.zeros((0, 3), tf.int32)).shape == (0, 3)


def _slope(links, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = links._apply(x)
    return tape.gradient(y, x)
