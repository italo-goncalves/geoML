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

A row of shares `x` is the image of a latent row `w` under a simplex
warping's way back, ``x_k = m_k [1 + alpha (w_k - tau)]_+ ** (1/alpha)``,
`tau` making the parts sum to one; after it, increasing links `psi_k` take
`w_k` to the space where the model's latent `f_k` and Gaussian noise live,
``psi_k(w_k) = f_k + sigma_k xi_k``. The way back ignores a shift of the
whole row, so the likelihood of `x` integrates that shift out in one
dimension. A positive part sits on the fibre, ``w_k = b_k + tau`` (`b_k`
the warping's forward value); a zero part lies below the point where it
vanishes, ``w_k <= tau + t0`` (`t0 = -1/alpha`), and scores a probability;
a part not assayed is integrated over every value it could take, given
what the observed parts leave of the whole (`R`).

Everything here is in log space and in float64, and works on whole
batches: `n` rows, `S` latent realizations, `T` nodes in `tau`. Reference:
the research report ``scaled_simplex_entmax.tex`` (project "Alternative
CoDA transformations"), equations (eq:loglik), (eq:missing) and
(eq:general).
"""
import itertools as _itertools

import numpy as _np
import scipy.stats.qmc as _qmc
import tensorflow as _tf
import tensorflow_probability as _tfp

# below this a log-probability counts as the probability zero; a finite
# number, so that a masked branch never sends NaN through the gradient
_LOG_ZERO = -1e4
# how far into a normal's tail a part's latent is followed when a range of
# it is integrated. Far: an improbable row -- an outlier -- has its mass
# there, and a row at -32 nats lost a nat and a half at 8. The lower tail
# keeps its precision this far out, and a range above zero is flipped into it
_TAIL = 30.0
# a remainder below this is no remainder: the parts not assayed are zero
_WHOLE = 1e-10


_NORMAL = _tfp.distributions.Normal(_tf.constant(0.0, _tf.float64),
                                    _tf.constant(1.0, _tf.float64))


def _log_ndtr(z):
    """The standard normal's log-CDF, accurate deep in the lower tail."""
    return _NORMAL.log_cdf(z)


def _ndtr(z):
    return _NORMAL.cdf(z)


def _log1mexp(x):
    """log(1 - exp(x)) for x <= 0, accurately at both ends."""
    x = _tf.minimum(x, -1e-300)
    return _tf.where(x > -0.6931471805599453,
                     _tf.math.log(-_tf.math.expm1(x)),
                     _tf.math.log1p(-_tf.exp(x)))


def _log_ndtr_diff(lo, hi):
    """log(Phi(hi) - Phi(lo)) for lo < hi, from whichever tail is
    accurate."""
    upper = lo > 0
    a = _tf.where(upper, -hi, lo)
    b = _tf.where(upper, -lo, hi)
    log_b = _log_ndtr(b)
    return log_b + _log1mexp(_log_ndtr(a) - log_b)


def _cdf_nodes(lo, hi, nodes):
    """Gauss-Legendre nodes on [lo, hi] in a standard normal's own
    coordinate, placed uniformly in its CDF: `z`, shaped ``lo.shape +
    nodes.shape``, and the log of the CDF's width. The integral of
    ``phi(z) k(z)`` over the range is then ``width * sum(w_g / 2 *
    k(z_g))``.

    The range is clipped to the `_TAIL`, and taken from the lower tail
    (flipped) where it lies above zero, so that the nodes keep their
    precision far out.
    """
    lo = _tf.clip_by_value(lo, -_TAIL, _TAIL)
    hi = _tf.clip_by_value(hi, -_TAIL, _TAIL)
    hi = _tf.maximum(hi, lo + 1e-12)
    upper = lo > 0
    a = _tf.where(upper, -hi, lo)
    b = _tf.where(upper, -lo, hi)
    cdf_a = _ndtr(a)[..., None]
    cdf_b = _ndtr(b)[..., None]
    u = cdf_a + (cdf_b - cdf_a) * nodes
    u = _tf.clip_by_value(u, 1e-300, 1.0 - 1e-16)
    z = _tf.math.ndtri(u)
    z = _tf.where(upper[..., None], -z, z)
    return z, _log_ndtr_diff(lo, hi)


class Simplex:
    """What the likelihood reads off a simplex warping, part by part.

    `columns` in every method names the part each value belongs to, with
    the shape of the values; the scale and the power are gathered for
    them, so a value may be of any part.
    """

    def __init__(self, warping):
        import geoml.warping as _warp
        self.scale = _tf.reshape(_tf.cast(warping.scale, _tf.float64), [-1])
        if isinstance(warping, _warp.PowerSimplex):
            alpha, small, safe = warping._alpha()
            self.alpha = _tf.reshape(alpha, [])
            self.small = _tf.reshape(small, [])
            self.safe = _tf.reshape(safe, [])
            self.threshold = _tf.where(self.small,
                                       _tf.constant(-1e8, _tf.float64),
                                       -1.0 / self.safe)
            self.linear = False
        elif isinstance(warping, _warp.ScaledSimplex):
            self.alpha = _tf.constant(1.0, _tf.float64)
            self.threshold = _tf.constant(0.0, _tf.float64)
            self.linear = True
        else:
            raise TypeError(
                "the censored likelihood reads a simplex warping, "
                "`PowerSimplex` or `ScaledSimplex`, not %s"
                % type(warping).__name__)

    def parts(self, x, columns):
        """`b`, the forward value of positive shares `x`, and `log g`, the
        log of ``dx / db``."""
        m = _tf.gather(self.scale, columns)
        x = _tf.maximum(x, 1e-300)
        log_ratio = _tf.math.log(x / m)
        if self.linear:
            return x / m, _tf.math.log(m) + 0.0 * x
        b = _tf.where(self.small, log_ratio,
                      _tf.math.expm1(self.safe * log_ratio) / self.safe)
        return b, _tf.math.log(m) + (1.0 - self.alpha) * log_ratio

    def share(self, b, columns):
        """The share a part with forward value `b` holds: the way back for
        one part, its threshold already taken out."""
        m = _tf.gather(self.scale, columns)
        if self.linear:
            return m * _tf.maximum(b, 0.0)
        inner = 1.0 + self.safe * b
        safe_inner = _tf.where(inner > 0, inner, 1.0)
        powered = m * _tf.exp(_tf.math.log(safe_inner) / self.safe)
        powered = _tf.where(inner > 0, powered, 0.0)
        return _tf.where(self.small, m * _tf.exp(b), powered)


class Links:
    """The increasing elementwise links after the simplex warping, read
    part by part.

    `links` is a warping, a list of them applied in order, or None. An
    affine chain is read off two points per part, so a value of any part
    costs one multiplication; any other chain is evaluated on the whole
    row, each value placed in every column and its own column kept, which
    costs the row's width more.
    """

    def __init__(self, links, size):
        import geoml.warping as _warp
        if links is None:
            links = []
        elif not isinstance(links, (list, tuple)):
            links = [links]
        flat = []
        for link in links:
            flat.extend(link.warpings if isinstance(link, _warp.ChainedWarping)
                        else [link])
        self.links = flat
        self.size = size
        self.affine = all(_is_affine(link) for link in flat)
        if self.affine:
            ends = self._apply(
                _tf.constant([[0.0] * size, [1.0] * size], _tf.float64))
            self.offset = ends[0]
            self.slope = ends[1] - ends[0]

    def _apply(self, x):
        for link in self.links:
            x = link.forward(x)[0]
        return x

    def _invert(self, z):
        for link in reversed(self.links):
            z = link.backward(z)
        return z

    def _whole_rows(self, v, columns, function):
        # every value in every column, its own column read back
        flat = _tf.reshape(v, [-1])
        tiled = _tf.tile(flat[:, None], [1, self.size])
        out = function(tiled)
        own = _tf.one_hot(_tf.reshape(columns, [-1]), self.size,
                          dtype=_tf.float64)
        return _tf.reshape(_tf.reduce_sum(out * own, axis=1), _tf.shape(v))

    def forward(self, v, columns):
        """``psi(v)`` and ``log psi'(v)``."""
        if self.affine:
            a = _tf.gather(self.offset, columns)
            s = _tf.gather(self.slope, columns)
            return a + s * v, _tf.math.log(s) + 0.0 * v

        def log_slope(x):
            with _tf.GradientTape() as tape:
                tape.watch(x)
                y = self._apply(x)
            # elementwise, so the gradient of the sum is each slope
            return _tf.math.log(tape.gradient(y, x))

        return (self._whole_rows(v, columns, self._apply),
                self._whole_rows(v, columns, log_slope))

    def backward(self, z, columns):
        """``psi^-1(z)``."""
        if self.affine:
            a = _tf.gather(self.offset, columns)
            s = _tf.gather(self.slope, columns)
            return (z - a) / s
        return self._whole_rows(z, columns, self._invert)


def _is_affine(warping):
    import geoml.warping as _warp
    if isinstance(warping, _warp.ChainedWarping):
        return all(_is_affine(link) for link in warping.warpings)
    return bool(getattr(warping, "_AFFINE", False))


def _faces(m):
    """Every non-empty subset of `m` hidden parts that can be positive, as
    tuples of positions."""
    return [q for k in range(1, m + 1)
            for q in _itertools.combinations(range(m), k)]


class Rule:
    """How many nodes each integral takes."""

    def __init__(self, n_tau=24, n_legendre=16, n_sobol=1024, seed=0,
                 n_guide=8):
        y, w = _np.polynomial.hermite.hermgauss(n_tau)
        self.tau_nodes = _tf.constant(y * _np.sqrt(2.0), _tf.float64)
        self.tau_log_weights = _tf.constant(
            _np.log(w) - 0.5 * _np.log(_np.pi), _tf.float64)
        g, v = _np.polynomial.legendre.leggauss(n_legendre)
        self.legendre_nodes = _tf.constant((g + 1.0) / 2.0, _tf.float64)
        self.legendre_log_weights = _tf.constant(_np.log(v / 2.0),
                                                 _tf.float64)
        # the passes that only narrow a window, and the guide values that
        # only place the nodes in `tau`, take fewer
        g, v = _np.polynomial.legendre.leggauss(n_guide)
        self.guide_nodes = _tf.constant((g + 1.0) / 2.0, _tf.float64)
        self.guide_log_weights = _tf.constant(_np.log(v / 2.0), _tf.float64)
        self.n_sobol = n_sobol
        self.seed = seed
        self._sobol = {}

    def sobol(self, dimension):
        """Seeded scrambled Sobol points in the unit cube, fixed per
        dimension, away from its faces."""
        if dimension not in self._sobol:
            points = _qmc.Sobol(dimension, scramble=True,
                                seed=self.seed + dimension).random(
                self.n_sobol)
            self._sobol[dimension] = _tf.constant(
                _np.clip(points, 1e-12, 1.0 - 1e-12), _tf.float64)
        return self._sobol[dimension]


def patterns(y, observed, max_hidden=6):
    """Each row's parts as positive, zero and hidden, the remainder the
    hidden parts share, and which rows are scored.

    A row whose observed parts already make up the whole has its hidden
    parts at zero. A row with no observed part, or more than `max_hidden`
    hidden ones, is not scored.
    """
    observed = _tf.cast(observed, _tf.bool)
    y = _tf.where(observed, y, 0.0)
    remainder = 1.0 - _tf.reduce_sum(y, axis=1)
    whole = remainder <= _WHOLE
    hidden = _tf.logical_not(observed)
    zero = (observed & (y <= 0.0)) | (hidden & whole[:, None])
    hidden = hidden & _tf.logical_not(whole[:, None])
    positive = observed & (y > 0.0)
    n_hidden = _tf.reduce_sum(_tf.cast(hidden, _tf.int32), axis=1)
    scored = _tf.reduce_any(observed, axis=1) & (n_hidden <= max_hidden)
    return positive, zero, hidden, _tf.maximum(remainder, 0.0), scored


def log_likelihood(y, observed, latent, noise_sd, simplex, links, rule,
                   max_hidden=6):
    """The log-likelihood of each row at each latent realization.

    Parameters
    ----------
    y : tensor
        `(n, D)`, shares of the whole, anything where not observed.
    observed : tensor
        `(n, D)`, boolean or 0/1.
    latent : tensor
        `(n, D, S)`, the latent realizations, on the links' side.
    noise_sd : tensor
        `(D,)`, the noise's standard deviation per part.
    simplex : Simplex
    links : Links
    rule : Rule
    max_hidden : int
        The most hidden parts a scored row may have.

    Returns
    -------
    log_lik : tensor
        `(n, S)`, zero on the rows not scored.
    scored : tensor
        `(n,)`, boolean.
    """
    y = _tf.cast(y, _tf.float64)
    n = _tf.shape(y)[0]
    size = y.shape[1]
    positive, zero, hidden, remainder, scored = patterns(
        y, observed, max_hidden)
    columns = _tf.broadcast_to(_tf.range(size)[None, :], _tf.shape(y))
    sd = _tf.reshape(noise_sd, [1, size, 1])
    f = latent

    # the positive parts on the fibre
    b, log_g = simplex.parts(_tf.where(positive, y, 0.5), columns)
    gamma = _tf.reduce_sum(_tf.where(positive, _tf.exp(log_g), 0.0), axis=1)

    # where the integral in `tau` is: the noise's posterior on the shift
    # from the positive parts, linearized through the links (exact for
    # affine ones); a row with none centres on the hidden parts, each
    # holding its scale's share of the remainder
    total_hidden = _tf.reduce_sum(
        _tf.where(hidden, _tf.gather(simplex.scale, columns), 0.0), axis=1)
    guess = remainder[:, None] * _tf.gather(simplex.scale, columns) \
        / _tf.maximum(total_hidden[:, None], 1e-300)
    b_guess, _ = simplex.parts(_tf.where(hidden, guess, 0.5), columns)
    has_positive = _tf.reduce_any(positive, axis=1, keepdims=True)
    anchor = _tf.where(has_positive, positive, hidden)
    anchor = _tf.cast(anchor, _tf.float64)[:, :, None]
    b_anchor = _tf.where(positive, b, b_guess)[:, :, None]
    columns_s = _tf.broadcast_to(columns[:, :, None], _tf.shape(f))
    f_stop = _tf.stop_gradient(f)
    target = _tf.stop_gradient(links.backward(f_stop, columns_s))
    _, log_slope = links.forward(target, columns_s)
    weight = _tf.stop_gradient(_tf.exp(2.0 * log_slope) / sd ** 2) * anchor
    centre = _tf.reduce_sum(weight * (target - b_anchor), axis=1) \
        / _tf.maximum(_tf.reduce_sum(weight, axis=1), 1e-300)
    precision = _tf.reduce_sum(weight, axis=1)
    if not links.affine:
        for _ in range(2):
            at = b_anchor + centre[:, None, :]
            value, log_slope = links.forward(at, columns_s)
            slope = _tf.exp(log_slope) / sd
            residual = (value - f_stop) / sd
            precision = _tf.reduce_sum(anchor * slope ** 2, axis=1)
            centre = centre - _tf.reduce_sum(anchor * slope * residual,
                                             axis=1) \
                / _tf.maximum(precision, 1e-300)
    centre = _tf.stop_gradient(centre)
    width = _tf.stop_gradient(
        1.0 / _tf.sqrt(_tf.maximum(precision, 1e-12)))     # (n, S)

    f_t = f[:, :, :, None]
    sd_t = sd[:, :, :, None]
    sd_flat = _tf.reshape(sd, [-1])
    censored = _tf.where(zero | hidden)                    # (k, 2)

    def on_nodes(centre, width):
        """The nodes in `tau`, the log-density they are placed by, and at
        each: the positive and zero parts' terms, and every part's latent
        where it would vanish, with its log-probability of being zero."""
        tau = centre[:, :, None] + width[:, :, None] * rule.tau_nodes
        log_q = -0.5 * ((tau - centre[:, :, None]) / width[:, :, None])             ** 2 - _tf.math.log(width[:, :, None]) - 0.5 * _np.log(2 * _np.pi)
        at = b[:, :, None, None] + tau[:, None, :, :]
        columns_t = _tf.broadcast_to(columns[:, :, None, None],
                                     _tf.shape(at))
        value, log_slope = links.forward(at, columns_t)
        z = (value - f_t) / sd_t
        log_density = -0.5 * z ** 2 - 0.5 * _np.log(2 * _np.pi)             - _tf.math.log(sd_t) + log_slope
        terms = _tf.reduce_sum(
            _tf.where(positive[:, :, None, None], log_density, 0.0), axis=1)
        # where a part vanishes, and the log-probability it is zero, only
        # for the parts that are zero or hidden: the normal's log-CDF over
        # every part was most of a complete row's cost
        edge = _tf.gather(tau, censored[:, 0]) + simplex.threshold
        col = _tf.broadcast_to(censored[:, 1, None, None], _tf.shape(edge))
        value_edge, _ = links.forward(edge, col)
        z_edge = (value_edge - _tf.gather_nd(f, censored)[:, :, None]) \
            / _tf.gather(sd_flat, censored[:, 1])[:, None, None]
        log_zero = _tf.scatter_nd(censored, _log_ndtr(z_edge),
                                  _tf.shape(at, out_type=_tf.int64))
        terms = terms + _tf.reduce_sum(
            _tf.where(zero[:, :, None, None], log_zero, 0.0), axis=1)
        return tau, log_q, terms, log_zero

    n_hidden = _tf.reduce_sum(_tf.cast(hidden, _tf.int32), axis=1)

    def hidden_terms(tau, log_zero, guide=False):
        """``log h_M(tau)``: the sum constraint's Jacobian where nothing is
        hidden, every face of the hidden parts where something is. A
        `guide` value, which only places the nodes in `tau`, takes each
        integral over the hidden parts in one pass."""
        log_h = _tf.broadcast_to(
            _tf.math.log(_tf.maximum(gamma, 1e-300))[:, None, None],
            _tf.shape(tau))
        for m in range(1, min(max_hidden, size) + 1):
            rows = _tf.where(scored & (n_hidden == m))[:, 0]
            part = _hidden_rows(rows, m, hidden, remainder, gamma, f, sd,
                                tau, log_zero, simplex, links, rule, guide)
            log_h = _tf.tensor_scatter_nd_update(log_h, rows[:, None], part)
        return log_h

    # two adaptive steps: where zeros, hidden parts or a bent link move the
    # integrand off the positive parts' posterior, the nodes follow its own
    # mean and spread. The first pass is three times wider, so that an
    # improbable row's mass, far from where the positive parts put it,
    # falls on a node
    width = 3.0 * width
    for _ in range(2):
        tau, log_q, terms, log_zero = on_nodes(centre, width)
        log_weight = rule.tau_log_weights + terms - log_q \
            + hidden_terms(tau, log_zero, guide=True)
        share = _tf.nn.softmax(log_weight, axis=2)
        mean = _tf.reduce_sum(share * tau, axis=2)
        spread = _tf.sqrt(_tf.maximum(
            _tf.reduce_sum(share * (tau - mean[:, :, None]) ** 2, axis=2),
            1e-24))
        # never narrower than the nodes are apart: where one node holds
        # nearly all the weight, its spread says nothing of the width
        spread = _tf.maximum(spread, width * 2.0
                             / _np.sqrt(rule.tau_nodes.shape[0]))
        centre = _tf.stop_gradient(mean)
        width = _tf.stop_gradient(spread)
    tau, log_q, terms, log_zero = on_nodes(centre, width)
    log_h = hidden_terms(tau, log_zero)

    log_integral = _tf.reduce_logsumexp(
        rule.tau_log_weights + terms + log_h - log_q, axis=2)
    out = log_integral - _tf.reduce_sum(
        _tf.where(positive, log_g, 0.0), axis=1)[:, None]
    return _tf.where(scored[:, None], out, 0.0), scored


def _hidden_rows(rows, m, hidden, remainder, gamma, f, sd, tau, log_zero,
                 simplex, links, rule, guide=False):
    """``log h_M(tau)`` for the rows with `m` hidden parts: every face, a
    subset of them positive and the others zero."""
    hide = _tf.gather(hidden, rows)
    # the hidden columns first, in order
    order = _tf.argsort(_tf.cast(_tf.logical_not(hide), _tf.int32),
                        axis=1, stable=True)
    cols = order[:, :m]                                    # (r, m)
    R = _tf.gather(remainder, rows)                        # (r,)
    gamma_r = _tf.gather(gamma, rows)
    f_r = _tf.gather(_tf.gather(f, rows), cols, batch_dims=1)   # (r,m,S)
    sd_r = _tf.gather(_tf.reshape(sd, [-1]), cols)[:, :, None]  # (r,m,1)
    tau_r = _tf.gather(tau, rows)                          # (r,S,T)
    lz = _tf.gather(_tf.gather(log_zero, rows), cols, batch_dims=1)

    faces = []
    for face in _faces(m):
        rest = [p for p in range(m) if p not in face]
        log_value = _tf.reduce_sum(_tf.gather(lz, rest, axis=1), axis=1) \
            if rest else _tf.zeros_like(tau_r)
        k = _face(face, cols, R, gamma_r, f_r, sd_r, tau_r, simplex, links,
                  rule, guide)
        faces.append(log_value + k)
    return _tf.reduce_logsumexp(_tf.stack(faces, axis=0), axis=0)


def _of_part(p, cols, f_r, sd_r, like):
    """Part `p`'s column, latent and noise, shaped to broadcast with
    `like`, a tensor ``(r, S, ...)``."""
    rank = len(like.shape)
    col = _tf.broadcast_to(_tf.reshape(cols[:, p], [-1] + [1] * (rank - 1)),
                           _tf.shape(like))
    f = _tf.reshape(f_r[:, p, :], _tf.concat(
        [_tf.shape(f_r[:, p, :]), [1] * (rank - 2)], axis=0))
    s = _tf.reshape(sd_r[:, p, :], [-1, 1] + [1] * (rank - 2))
    return col, f, s


def _standard(p, w, cols, f_r, sd_r, links):
    """Part `p`'s latent noise in standard units where its warped value is
    `w`, and the log of its link's slope over the noise's scale there."""
    col, f, s = _of_part(p, cols, f_r, sd_r, w)
    value, log_slope = links.forward(w, col)
    return (value - f) / s, log_slope - _tf.math.log(s)


def _log_normal(z):
    return -0.5 * z ** 2 - 0.5 * _np.log(2 * _np.pi)


def _solved(l, x_l, tau, cols, f_r, sd_r, simplex, links):
    """The solved part's term, ``log p_l(b_l(x_l) + tau) - log g_l(x_l)``,
    and its `g`; `x_l` and `tau` alike shaped ``(r, S, ...)``."""
    col, _, _ = _of_part(l, cols, f_r, sd_r, x_l)
    b_l, log_g_l = simplex.parts(x_l, col)
    z, log_jacobian = _standard(l, b_l + tau, cols, f_r, sd_r, links)
    return _log_normal(z) + log_jacobian - log_g_l, _tf.exp(log_g_l)


def _free(p, z, tau, cols, f_r, sd_r, simplex, links):
    """A free part at its latent noise `z` (one more axis than `tau`): its
    share and its `g`."""
    col, f, s = _of_part(p, cols, f_r, sd_r, z)
    w = links.backward(f + s * z, col)
    x = simplex.share(w - tau[..., None], col)
    _, log_g = simplex.parts(x, col)
    return x, _tf.exp(log_g)


def _range(p, upper, tau, cols, f_r, sd_r, simplex, links):
    """Part `p`'s latent noise, in standard units, from where it vanishes
    to where it holds `upper`."""
    col, _, _ = _of_part(p, cols, f_r, sd_r, tau)
    lo, _ = _standard(p, tau + simplex.threshold, cols, f_r, sd_r, links)
    b, _ = simplex.parts(upper, col)
    hi, _ = _standard(p, b + tau, cols, f_r, sd_r, links)
    lo = _tf.clip_by_value(lo, -_TAIL, _TAIL)
    hi = _tf.maximum(_tf.clip_by_value(hi, -_TAIL, _TAIL), lo + 1e-12)
    return lo, hi


def _adaptive(lo, hi, log_f, rule, passes=3, coarse=False):
    """``log`` of the integral of ``exp(log_f(z))`` over ``[lo, hi]``, `z`
    shaped ``lo.shape + (G,)``, by Gauss-Legendre on a window narrowed to
    the integrand's own mean and spread before the last pass: in an
    improbable row the mass sits deep in a part's tail, where nodes placed
    by the part's own density never reach."""
    a, b = lo, hi
    for step in range(passes):
        if coarse or step < passes - 1:
            nodes, log_weights = rule.guide_nodes, rule.guide_log_weights
        else:
            nodes = rule.legendre_nodes
            log_weights = rule.legendre_log_weights
        z = a[..., None] + (b - a)[..., None] * nodes
        log_terms = log_weights + log_f(z)
        if step == passes - 1:
            break
        share = _tf.nn.softmax(log_terms, axis=-1)
        mean = _tf.reduce_sum(share * z, axis=-1)
        sd = _tf.sqrt(_tf.maximum(_tf.reduce_sum(
            share * (z - mean[..., None]) ** 2, axis=-1), 1e-24))
        sd = _tf.maximum(sd, (b - a) / nodes.shape[0])
        reach = 10.0 if step == 0 else 8.0
        a = _tf.stop_gradient(_tf.maximum(lo, mean - reach * sd))
        b = _tf.stop_gradient(_tf.maximum(_tf.minimum(hi, mean + reach * sd),
                                          a + 1e-12))
    return _tf.math.log(b - a) + _tf.reduce_logsumexp(log_terms, axis=-1)


def _pair(j, l, R, gamma, tau, cols, f_r, sd_r, simplex, links, rule,
          passes=3, coarse=False):
    """``log K`` for two hidden parts sharing `R`, both positive: each in
    turn is the free one, up to half the remainder, the other solved."""
    pieces = []
    for free, solved in ((j, l), (l, j)):
        lo, hi = _range(free, R / 2.0, tau, cols, f_r, sd_r, simplex, links)

        def log_f(z, free=free, solved=solved):
            x, g = _free(free, z, tau, cols, f_r, sd_r, simplex, links)
            x_l = R[..., None] - x
            valid = x_l > 0
            term, g_l = _solved(solved,
                                _tf.where(valid, x_l, R[..., None] / 2),
                                _tf.broadcast_to(tau[..., None],
                                                 _tf.shape(x)),
                                cols, f_r, sd_r, simplex, links)
            log_k = term + _tf.math.log(gamma[..., None] + g + g_l)
            return _log_normal(z) + _tf.where(valid, log_k, _LOG_ZERO)

        pieces.append(_adaptive(lo, hi, log_f, rule, passes, coarse))
    return _tf.reduce_logsumexp(_tf.stack(pieces, axis=0), axis=0)


def _face(face, cols, R, gamma_r, f_r, sd_r, tau_r, simplex, links, rule,
          guide=False):
    """``log K_Q(tau)`` for the face where the hidden parts at positions
    `face` are positive and share the remainder; a `guide` value in one
    coarse pass of each integral."""
    q = len(face)
    R_t = _tf.broadcast_to(R[:, None, None], _tf.shape(tau_r))
    gamma_t = _tf.broadcast_to(gamma_r[:, None, None], _tf.shape(tau_r))
    if q == 1:
        term, g_l = _solved(face[0], R_t, tau_r, cols, f_r, sd_r, simplex,
                            links)
        return term + _tf.math.log(gamma_t + g_l)
    if q == 2:
        return _pair(face[0], face[1], R_t, gamma_t, tau_r, cols, f_r, sd_r,
                     simplex, links, rule, passes=1 if guide else 3,
                     coarse=guide)
    if q == 3:
        # a convolution: the first part over all it could hold, the other
        # two sharing what it leaves
        a = face[0]
        lo, hi = _range(a, R_t, tau_r, cols, f_r, sd_r, simplex, links)

        def log_f(z):
            x, g = _free(a, z, tau_r, cols, f_r, sd_r, simplex, links)
            left = R_t[..., None] - x
            valid = left > 0
            inner = _pair(face[1], face[2],
                          _tf.where(valid, left, R_t[..., None] / 2),
                          gamma_t[..., None] + g,
                          _tf.broadcast_to(tau_r[..., None], _tf.shape(x)),
                          cols, f_r, sd_r, simplex, links, rule,
                          passes=1 if guide else 2, coarse=guide)
            return _log_normal(z) + _tf.where(valid, inner, _LOG_ZERO)

        return _adaptive(lo, hi, log_f, rule, passes=1 if guide else 2,
                         coarse=guide)

    # four or more: each part in turn the largest, solved; the others by
    # seeded Sobol points in the box of their latent noise up to half the
    # remainder, the points outside the piece dropped
    pieces = []
    cube = rule.sobol(q - 1)
    for l in face:
        free = [p for p in face if p != l]
        zs, log_widths = [], 0.0
        for i, p in enumerate(free):
            lo, hi = _range(p, R_t / 2.0, tau_r, cols, f_r, sd_r, simplex,
                            links)
            z, log_width = _cdf_nodes(lo, hi, cube[:, i])
            zs.append(z)
            log_widths = log_widths + log_width
        shares, g_free = [], 0.0
        for p, z in zip(free, zs):
            x, g = _free(p, z, tau_r, cols, f_r, sd_r, simplex, links)
            shares.append(x)
            g_free = g_free + g
        x_l = R_t[..., None] - _tf.add_n(shares)
        valid = (x_l > 0) & (x_l >= _tf.reduce_max(_tf.stack(shares), 0))
        term, g_l = _solved(l, _tf.where(valid, x_l, R_t[..., None] / 2),
                            _tf.broadcast_to(tau_r[..., None],
                                             _tf.shape(x_l)),
                            cols, f_r, sd_r, simplex, links)
        log_k = term + _tf.math.log(gamma_t[..., None] + g_free + g_l)
        log_k = _tf.where(valid, log_k, _LOG_ZERO)
        pieces.append(log_widths - _np.log(rule.n_sobol)
                      + _tf.reduce_logsumexp(log_k, axis=-1))
    return _tf.reduce_logsumexp(_tf.stack(pieces, axis=0), axis=0)
