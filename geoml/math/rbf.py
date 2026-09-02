# geoML - machine learning models for geospatial data
# Copyright (C) 2019  Ítalo Gomes Gonçalves
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

"""Parameter-free radial basis function interpolants for implicit surfaces.

An implicit surface is the zero level set of a scalar field fitted to
points on it, with value zero, and to normals, as its gradient. The field
is a polyharmonic radial basis function expansion with a linear drift: no
range, no trainable parameter, one linear solve. The Hermite form takes the
normals as gradient constraints (Macêdo et al. 2011; Hillier et al. 2014);
the off-surface form of Carr et al. (2001) turns each normal into two
displaced points instead, which is what lets the thin-plate spline and the
linear basis, not twice differentiable at the origin, be used.

In this package's terms the Hermite system is the posterior mean of a
Gaussian process with noise-free value and gradient observations, which is
the potential-field method of Lajaunie et al. (1997) written with a
conditionally positive definite basis instead of a covariance. None of it
is original here; the references are on `HermiteRBF`.
"""
import warnings as _warnings

import numpy as _np
import scipy.linalg as _sla
import scipy.spatial as _spatial
import tensorflow as _tf

import geoml.math.geometry as _geom
import geoml.stats.random as _rnd

# rows of the query-by-centre products held at once when evaluating
_CHUNK_ELEMENTS = 50_000_000
_BASES = ("cubic", "thin_plate", "linear")


def _pairwise(u, v):
    """Differences `u[:, None] - v[None]` and their norms, `[n, m, d]` and
    `[n, m]`."""
    delta = u[:, None, :] - v[None, :, :]
    # the square root's derivative is infinite at zero distance -- a centre
    # against itself on every diagonal, an inducing point on the fault --
    # and meets a zero factor there, which is NaN; floored, the gradient
    # through `maximum` is zero at coincidence, which is the right answer
    r = _tf.sqrt(_tf.maximum(_tf.reduce_sum(delta ** 2, axis=2), 1e-300))
    return delta, r


def _phi(r, basis):
    if basis == "cubic":
        return r ** 3
    if basis == "linear":
        return r
    # thin plate: r^2 log r, zero at the origin
    safe = _tf.where(r > 0, r, _tf.ones_like(r))
    return _tf.where(r > 0, r ** 2 * _tf.math.log(safe), _tf.zeros_like(r))


def _phi_over_r(r, basis):
    """`phi'(r) / r`, the radial factor of the gradient; the thin plate
    and linear bases are singular at a centre and read as zero there."""
    if basis == "cubic":
        return 3.0 * r
    if basis == "linear":
        return _tf.math.divide_no_nan(_tf.ones_like(r), r)
    safe = _tf.where(r > 0, r, _tf.ones_like(r))
    return _tf.where(r > 0, 2.0 * _tf.math.log(safe) + 1.0, _tf.zeros_like(r))


def _value_blocks(u, centres, basis):
    """`phi(|u - x_i|)`, `[n, m]`."""
    _, r = _pairwise(u, centres)
    return _phi(r, basis)


def _gradient_blocks(u, centres, basis):
    """`d/du_a phi(|u - x_i|)`, `[n, m, d]`: the field's gradient carried by
    the value centres."""
    delta, r = _pairwise(u, centres)
    return _phi_over_r(r, basis)[:, :, None] * delta


def _cross_blocks(u, centres):
    """`d/dy_b phi(|u - y|)` at the gradient centres, cubic basis, `[n, m,
    d]`: what a gradient centre contributes to the value."""
    delta, r = _pairwise(u, centres)
    return -3.0 * r[:, :, None] * delta


def _hessian_blocks(u, centres):
    """`d2/du_a dy_b phi(|u - y|)` at the gradient centres, cubic basis,
    `[n, m, d, d]`: what a gradient centre contributes to the gradient.
    `delta_a delta_b / r` vanishes at coincidence, and reads so."""
    delta, r = _pairwise(u, centres)
    outer = delta[:, :, :, None] * delta[:, :, None, :]
    eye = _tf.eye(int(u.shape[1]), dtype=u.dtype)[None, None]
    return -3.0 * (_tf.math.divide_no_nan(outer, r[:, :, None, None])
                   + r[:, :, None, None] * eye)


def solve_hermite(centres, values, gradient_centres=None, gradients=None,
                  basis="cubic"):
    """
    The interpolant's weights, as tensors, for centres given as tensors.

    The same symmetric system `HermiteRBF` solves, assembled and solved in
    TensorFlow so that it can sit inside a traced graph -- what a fault
    network needs, whose older surfaces are refitted on coordinates the
    younger faults' trainable slips restore. Dense, so for centres in the
    hundreds; the standalone class thins a large set first.

    Parameters
    ----------
    centres
        `[n, d]` float64 tensor, in the frame the weights will be used in.
    values
        `[n]`.
    gradient_centres, gradients
        `[m, d]` locations and the gradient constraints at them, or None
        for values alone.
    basis
        As in `HermiteRBF`; gradient constraints need `"cubic"`.

    Returns
    -------
    alpha, beta, drift
        `[n]`, `[m, d]` (None without gradients) and `[d + 1]`.
    """
    centres = _tf.convert_to_tensor(centres, _tf.float64)
    values = _tf.convert_to_tensor(values, _tf.float64)
    n, d = int(centres.shape[0]), int(centres.shape[1])
    k_ff = _value_blocks(centres, centres, basis)
    drift_f = _tf.concat([_tf.ones([n, 1], _tf.float64), centres], axis=1)
    if gradients is None:
        a, rhs, drift = k_ff, values, drift_f
        m = 0
    else:
        if basis != "cubic":
            raise ValueError("gradient constraints need the cubic basis")
        g = _tf.convert_to_tensor(gradient_centres, _tf.float64)
        gradients = _tf.convert_to_tensor(gradients, _tf.float64)
        m = int(g.shape[0])
        k_fg = _tf.reshape(_cross_blocks(centres, g), [n, m * d])
        k_gg = _tf.reshape(
            _tf.transpose(_hessian_blocks(g, g), [0, 2, 1, 3]),
            [m * d, m * d])
        a = _tf.concat([_tf.concat([k_ff, k_fg], axis=1),
                        _tf.concat([_tf.transpose(k_fg), k_gg], axis=1)],
                       axis=0)
        rhs = _tf.concat([values, _tf.reshape(gradients, [-1])], axis=0)
        drift_g = _tf.concat(
            [_tf.zeros([m * d, 1], _tf.float64),
             _tf.tile(_tf.eye(d, dtype=_tf.float64), [m, 1])], axis=1)
        drift = _tf.concat([drift_f, drift_g], axis=0)
    n_rows = n + m * d
    p = d + 1
    system = _tf.concat(
        [_tf.concat([a, drift], axis=1),
         _tf.concat([_tf.transpose(drift), _tf.zeros([p, p], _tf.float64)],
                    axis=1)], axis=0)
    full_rhs = _tf.concat([rhs, _tf.zeros([p], _tf.float64)], axis=0)
    solution = _tf.linalg.solve(system, full_rhs[:, None])[:, 0]
    alpha = solution[:n]
    beta = None if gradients is None \
        else _tf.reshape(solution[n:n_rows], [m, d])
    return alpha, beta, solution[n_rows:]


def field(u, centres, alpha, gradient_centres, beta, drift, basis="cubic"):
    """
    Value and gradient of an interpolant at `u`, given its weights.

    Both `[n, d]` float64 tensors in the frame of `centres`; the functional
    core `HermiteRBF` evaluates through, exposed for a fault network.
    """
    value = _tf.linalg.matvec(_value_blocks(u, centres, basis), alpha)
    grad = _tf.reduce_sum(
        _gradient_blocks(u, centres, basis) * alpha[None, :, None], axis=1)
    if beta is not None:
        value += _tf.reduce_sum(
            _cross_blocks(u, gradient_centres) * beta[None], axis=[1, 2])
        grad += _tf.reduce_sum(
            _hessian_blocks(u, gradient_centres) * beta[None, :, None, :],
            axis=[1, 3])
    value += drift[0] + _tf.linalg.matvec(u, drift[1:])
    grad += drift[None, 1:]
    return value, grad


class HermiteRBF:
    """
    A scalar field through scattered points, with their normals as its
    gradient: the implicit surface is its zero level set.

    Polyharmonic radial basis function interpolant with a linear drift and
    no free parameter. The cubic basis takes the normals as gradient
    constraints (the Hermite form). The thin-plate and linear bases are not
    twice differentiable at the origin, so with them each normal becomes
    two points displaced along it, at `offset` and `-offset`, carrying
    those values (the off-surface form). Both give a field that is negative
    behind the normals and positive ahead of them, close to a signed
    distance near the surface.

    Parameters
    ----------
    points
        The observations, `(n, d)`, on the surface unless `values` says
        otherwise.
    values
        One value per point; zero by default, which is a point on the
        surface.
    normals
        The gradient constraints. One of: an `(n, d)` array aligned with
        the points, rows of NaN where a point carries none; a pair
        `(locations, vectors)` of `(m, d)` arrays, for normals measured
        away from the points; None, to derive one for every point by
        `geometry.point_normals`, oriented toward the surface's concavity;
        or `False`, to fit values alone. A single normal is enough to pin
        the field -- nothing is required per point.
    basis
        `"cubic"` (`r^3`, the default, Hermite), `"thin_plate"`
        (`r^2 log r`) or `"linear"` (`r`), the last two through off-surface
        points.
    transform
        A fixed `geoml.transform` object the coordinates go through before
        distances are measured, for anisotropy. Given normals are mapped
        through its Jacobian; derived ones are found in the transformed
        space directly.
    max_error
        A geometric error budget, in the units of the field. When given,
        the centres are chosen greedily, starting from a subset and adding
        the worst-fitting points until every point is within the budget,
        so a large redundant set is fitted with a fraction of its points.
    offset
        The displacement of the off-surface points, for the thin-plate and
        linear bases; half the median spacing between neighbouring points
        by default.
    k
        Neighbours used to derive normals when none are given.

    Attributes
    ----------
    n_centres
        How many points carry the field.
    max_residual
        The largest value residual over all points after the fit.

    References
    ----------
    Carr, J. C., Beatson, R. K., Cherrie, J. B., Mitchell, T. J., Fright,
    W. R., McCallum, B. C. and Evans, T. R. (2001). Reconstruction and
    representation of 3D objects with radial basis functions. SIGGRAPH
    2001, 67-76.

    Macêdo, I., Gois, J. P. and Velho, L. (2011). Hermite radial basis
    functions implicits. Computer Graphics Forum 30(1), 27-42.

    Hillier, M. J., Schetselaar, E. M., de Kemp, E. A. and Perron, G.
    (2014). Three-dimensional modelling of geological surfaces using
    generalized interpolation with radial basis functions. Mathematical
    Geosciences 46, 931-953.

    Lajaunie, C., Courrioux, G. and Manuel, L. (1997). Foliation fields and
    3D cartography in geology: principles of a method based on potential
    interpolation. Mathematical Geology 29, 571-584.

    Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in
    Sobolev spaces. In: Constructive Theory of Functions of Several
    Variables, Springer, 85-100.

    Wendland, H. (2005). Scattered Data Approximation. Cambridge
    University Press.
    """

    def __init__(self, points, values=None, normals=None, basis="cubic",
                 transform=None, max_error=None, offset=None, k=12):
        if basis not in _BASES:
            raise ValueError("basis must be one of %s" % (_BASES,))
        points = _np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("points must be (n, 2) or (n, 3)")
        n, d = points.shape
        values = _np.zeros(n) if values is None \
            else _np.asarray(values, dtype=float).ravel()
        if len(values) != n:
            raise ValueError("one value per point")
        self.basis = basis
        self.transform = transform
        self.n_dim = d

        # the fit lives in a shifted frame of unit extent: a cubic basis at
        # metre scale puts entries of a million beside entries of one
        # (condition number 1e14 measured on a 50 m sphere), and the
        # scaling costs nothing -- values keep the caller's units and the
        # gradient is scaled back on the way out
        mapped = self._transformed(points)
        self._shift = mapped.mean(axis=0)
        extent = float(_np.max(_np.ptp(mapped, axis=0)))
        self._scale = extent if extent > 0 else 1.0
        working = (mapped - self._shift) / self._scale

        # the gradient constraints: none, derived for every point, given
        # for some of the points (rows of NaN carry none) or given at their
        # own locations as a (locations, vectors) pair -- one is enough to
        # pin the field, the Hermite system asks nothing per point
        if normals is False:
            where, unit = _np.zeros([0, d]), _np.zeros([0, d])
        elif normals is None:
            where = points
            unit = _geom.point_normals(working, k=k, orient="concave") \
                * self._scale
        elif isinstance(normals, (tuple, list)) and len(normals) == 2 \
                and _np.ndim(normals[0]) == 2:
            where = _np.asarray(normals[0], dtype=float)
            vectors = _np.asarray(normals[1], dtype=float)
            if where.shape != vectors.shape or where.shape[1] != d:
                raise ValueError("normals given as (locations, vectors) "
                                 "need two (m, %d) arrays" % d)
            unit = self._map_normals(where, vectors)
        else:
            vectors = _np.asarray(normals, dtype=float)
            if vectors.shape != (n, d):
                raise ValueError("normals aligned with the points must be "
                                 "(n, %d), NaN rows where there is none, or "
                                 "a (locations, vectors) pair" % d)
            has = _np.all(_np.isfinite(vectors), axis=1)
            where = points[has]
            unit = self._map_normals(where, vectors[has])
        if len(where) == 0 and not _np.any(values != 0):
            raise ValueError(
                "every value is zero and there are no normals: the field "
                "would be identically zero. Give a normal (or let them be "
                "derived) or values off the surface")
        self._gradient_points = where
        gradient_working = (self._transformed(where) - self._shift) \
            / self._scale if len(where) else _np.zeros([0, d])

        if basis != "cubic" and len(where):
            # the off-surface form: each normal becomes two displaced
            # points, and the field is fitted to values alone
            direction = unit / _np.linalg.norm(unit, axis=1, keepdims=True)
            if offset is None:
                tree = _spatial.cKDTree(working)
                spacing, _ = tree.query(working, k=2)
                offset = 0.5 * float(_np.median(spacing[:, 1])) * self._scale
            step = offset / self._scale
            working = _np.concatenate(
                [working, gradient_working + step * direction,
                 gradient_working - step * direction])
            at = _np.zeros(len(where))
            values = _np.concatenate([values, at + offset, at - offset])
            gradient_working, unit = _np.zeros([0, d]), _np.zeros([0, d])
        self._points = working
        self._values = values
        self._gradient_working = gradient_working
        self._gradients = unit
        self._fit(max_error)

    # --- coordinates ------------------------------------------------------
    def _transformed(self, x):
        if self.transform is None:
            return _np.asarray(x, dtype=float)
        return _np.asarray(self.transform(_tf.constant(x, _tf.float64)))

    def _jacobian(self, x):
        """`d transform / d x` at each point, `[n, d, d]`."""
        if self.transform is None:
            raise ValueError("no transform to differentiate")
        x = _tf.constant(x, _tf.float64)
        with _tf.GradientTape() as tape:
            tape.watch(x)
            u = self.transform(x)
        return tape.batch_jacobian(u, x)

    def _map_normals(self, points, normals):
        """The gradient constraint in the working frame that makes the
        field's gradient the unit normal in the caller's frame. A gradient
        is a covector: through a transform with Jacobian `J` it goes as
        `J^-T n`, and its length is part of the constraint -- renormalizing
        here would scale the recovered gradient by the transform."""
        norm = _np.linalg.norm(normals, axis=1, keepdims=True)
        if _np.any(norm == 0):
            raise ValueError("a normal has zero length")
        unit = normals / norm
        if self.transform is not None:
            jac = _np.asarray(self._jacobian(points))
            unit = _np.stack([_np.linalg.solve(j.T, n)
                              for j, n in zip(jac, unit)])
        return unit * self._scale

    # --- fitting ----------------------------------------------------------
    def _system(self, rows):
        """The symmetric Hermite system on the value centres `rows` names
        and every gradient constraint."""
        x = _tf.constant(self._points[rows])
        f = self._values[rows]
        n, d = x.shape
        k_ff = _np.asarray(_value_blocks(x, x, self.basis))
        drift_f = _np.concatenate([_np.ones([n, 1]), self._points[rows]],
                                  axis=1)
        m = len(self._gradient_working)
        if m == 0:
            a, rhs, drift = k_ff, f, drift_f
        else:
            g = _tf.constant(self._gradient_working)
            k_fg = _np.asarray(_cross_blocks(x, g)).reshape(n, m * d)
            k_gg = _np.asarray(_hessian_blocks(g, g)) \
                .transpose(0, 2, 1, 3).reshape(m * d, m * d)
            a = _np.block([[k_ff, k_fg], [k_fg.T, k_gg]])
            rhs = _np.concatenate([f, self._gradients.reshape(-1)])
            drift_g = _np.concatenate(
                [_np.zeros([m * d, 1]), _np.tile(_np.eye(d), [m, 1])],
                axis=1)
            drift = _np.concatenate([drift_f, drift_g], axis=0)
        p = drift.shape[1]
        system = _np.block([[a, drift], [drift.T, _np.zeros([p, p])]])
        rhs = _np.concatenate([rhs, _np.zeros(p)])
        return system, rhs

    def _solve(self, rows):
        system, rhs = self._system(rows)
        try:
            solution = _sla.solve(system, rhs, assume_a="sym")
        except _sla.LinAlgError:
            _warnings.warn("the interpolation system is singular (coincident "
                           "or degenerate points); solved in the least-"
                           "squares sense", UserWarning)
            solution = _sla.lstsq(system, rhs)[0]
        n = len(rows)
        d = self.n_dim
        m = len(self._gradient_working)
        self._rows = rows
        self._alpha = solution[:n]
        self._beta = None if m == 0 \
            else solution[n:n + m * d].reshape(m, d)
        self._drift = solution[n + m * d:]

    def _fit(self, max_error):
        n = len(self._points)
        everything = _np.arange(n)
        if max_error is None or n <= 50:
            self._solve(everything)
            self.max_residual = float(_np.max(_np.abs(
                self._evaluate(self._points) - self._values)))
            return
        # Carr et al. (2001, section 4.2): fit a subset, add the worst-
        # fitting points, repeat until every point is within the budget
        rows = _np.sort(_rnd.rng().choice(n, size=max(50, n // 10),
                                          replace=False))
        while True:
            self._solve(rows)
            residual = _np.abs(self._evaluate(self._points) - self._values)
            self.max_residual = float(residual.max())
            missing = _np.setdiff1d(everything, rows)
            bad = missing[residual[missing] > max_error]
            if len(bad) == 0:
                return
            worst = bad[_np.argsort(residual[bad])[::-1]]
            rows = _np.sort(_np.concatenate(
                [rows, worst[:max(1, n // 10)]]))

    @property
    def n_centres(self):
        return len(self._rows)

    @property
    def kept(self):
        """
        Indices of the points that carry the field, into the points given.

        All of them without `max_error`; the greedy choice with it, which
        is what to hand on when a thinned set must be recorded exactly, as
        a fault transform records its observations.
        """
        return self._rows.copy()

    # --- evaluation -------------------------------------------------------
    def _chunks(self, n):
        # inside a traced graph the batch size is unknown: one chunk
        if n is None:
            return [slice(None)]
        per = max(1, _CHUNK_ELEMENTS // (self.n_centres * self.n_dim ** 2))
        return [slice(i, min(i + per, n)) for i in range(0, n, per)]

    def _weights(self):
        """The centres and weights as tensors, in the working frame."""
        if self._beta is None:
            gradient_centres, beta = None, None
        else:
            gradient_centres = _tf.constant(self._gradient_working)
            beta = _tf.constant(self._beta)
        return (_tf.constant(self._points[self._rows]),
                _tf.constant(self._alpha), gradient_centres, beta,
                _tf.constant(self._drift))

    def _field(self, u):
        """Value and gradient of the field at working coordinates `u`, a
        float64 tensor `[n, d]`."""
        centres, alpha, gradient_centres, beta, drift = self._weights()
        return field(u, centres, alpha, gradient_centres, beta, drift,
                     self.basis)

    def to_working(self, x):
        """Coordinates of the caller's frame into the fitted frame, as a
        tensor: what `solve_hermite` and `field` take."""
        return self._working_tensor(x)

    @property
    def gradient_points(self):
        """Where the gradient constraints sit, `(m, d)` in the caller's
        frame; empty without any."""
        return self._gradient_points.copy()

    @property
    def working_gradients(self):
        """The gradient constraints in the fitted frame, as a
        `(locations, vectors)` pair of `(m, d)` arrays, or None."""
        if len(self._gradient_working) == 0:
            return None
        return self._gradient_working.copy(), self._gradients.copy()

    @property
    def scale(self):
        """The factor a gradient in the fitted frame is divided by to be
        one in the caller's frame."""
        return self._scale

    def _evaluate(self, working):
        return _np.concatenate([
            _np.asarray(self._field(_tf.constant(working[part]))[0])
            for part in self._chunks(len(working))])

    def __call__(self, x):
        """
        The field at `x`, `(n, d)`: zero on the surface, negative behind
        the normals, positive ahead of them.
        """
        u = self._working_tensor(x)
        return _tf.concat([self._field(u[part])[0]
                           for part in self._chunks(u.shape[0])], axis=0)

    def evaluate(self, x):
        """
        The field and its gradient at `x`, `(n,)` and `(n, d)`, in the
        original coordinates.
        """
        x = _tf.convert_to_tensor(x, _tf.float64)
        if self.transform is None:
            u = (x - self._shift) / self._scale
            jac = None
        else:
            with _tf.GradientTape() as tape:
                tape.watch(x)
                mapped = self.transform(x)
            jac = tape.batch_jacobian(mapped, x)
            u = (mapped - self._shift) / self._scale
        parts = [self._field(u[part]) for part in self._chunks(u.shape[0])]
        value = _tf.concat([v for v, _ in parts], axis=0)
        grad_u = _tf.concat([g for _, g in parts], axis=0)
        if jac is not None:
            # a gradient is a covector: d s/d x = J^T d s/d u
            grad_u = _tf.linalg.matvec(jac, grad_u, transpose_a=True)
        return value, grad_u / self._scale

    def gradient(self, x):
        """
        The field's gradient at `x`, `(n, d)`, in the original coordinates.
        """
        return self.evaluate(x)[1]

    def _working_tensor(self, x):
        x = _tf.convert_to_tensor(x, _tf.float64)
        mapped = x if self.transform is None else self.transform(x)
        return (mapped - self._shift) / self._scale

    def contour(self, grid, name="implicit"):
        """
        The zero level set of the field over a grid, as a mesh.

        Evaluates the field at the grid's nodes, writes it as the variable
        `name` and contours it at zero, so the result is a `Surface3D` or a
        `Solid3D` as the geometry decides, ready to inspect or export.
        """
        values = _np.asarray(self(_np.asarray(grid.coordinates, dtype=float)))
        if name in grid.variables:
            grid.variables[name].measurements.values[:] = values
        else:
            grid.add_continuous_variable(name, values)
        return grid.variables[name].measurements.get_contour(0.0)
