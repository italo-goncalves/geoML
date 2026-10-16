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
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Why a dilute part's ground value climbs under Laplace noise.

The ground value of a composition is E[closure(exp(clr + eps))], the noise
integrated in fractions. Take the simplest case, two parts, the dilute
one at log-ratio `-a` against the other (`a > 0`): its fraction is the
logistic `sigma(-a + eps)`. A fraction is bounded, so nothing diverges --
the question is how the mean scales with the centre `exp(-a)`.

Split the integral where the logistic turns over (`sigma(z) ~ exp(z)` for
`z < 0`, `~ 1` above) and integrate each piece in closed form (SymPy):

  Laplace, scale b:  E ~ exp(-a)/(2(b+1)) + (exp(-a/b) - exp(-a))/(2(b-1))
                         + exp(-a/b)/2
    b < 1:  E ~ exp(-a) / (1 - b**2)            -- proportional to the centre
    b > 1:  E ~ exp(-a/b) * b / (2 (b - 1))     -- the centre to the power 1/b
  Gaussian, sd s:    E ~ exp(-a + s**2/2)       -- proportional to the centre
                                                   while a > s**2

So under a Laplace wider than one natural-log unit, the mean of a part at
0.04 % is not a multiple of 0.04 % but 0.0004**(1/b): a power of the centre,
1.6 % at b = 1.86. That is the moment generating function's pole (finite
only for b < 1) seen through the closure: bounded, but switched from linear
to a power law. It is the model's own mathematics -- the noise law times
the link -- not an approximation: the exact integral (quadrature) agrees.

Run: python docs/benchmarks/heavy_tail_closure.py
"""
import numpy as np
import sympy as sp
from scipy import integrate, special

a, b, s, e = sp.symbols("a b s epsilon", positive=True)

# the two pieces of E[sigma(-a + eps)] under Laplace(0, b), with
# sigma(z) ~ exp(z) below zero and ~ 1 above
below = sp.integrate(sp.exp(e - a) * sp.exp(e / b) / (2 * b), (e, -sp.oo, 0),
                     conds="none") \
    + sp.integrate(sp.exp(e - a) * sp.exp(-e / b) / (2 * b), (e, 0, a),
                   conds="none")
above = sp.integrate(sp.exp(-e / b) / (2 * b), (e, a, sp.oo), conds="none")
approx = sp.simplify(below + above)
print("Laplace, two-piece integral (b != 1):")
sp.pprint(approx)
# the two regimes, checked at exact values of b against the stated forms
for b_ in (sp.Rational(1, 3), sp.Rational(1, 2), 2, 3):
    if b_ < 1:
        got = sp.limit(approx.subs(b, b_) * sp.exp(a), a, sp.oo)
        want = 1 / (1 - b_ ** 2)
        form = "E * exp(a)"
    else:
        got = sp.limit(approx.subs(b, b_) * sp.exp(a / b_), a, sp.oo)
        want = b_ / (2 * (b_ - 1))
        form = "E * exp(a/b)"
    print("b = %s: %s -> %s, stated %s" % (b_, form, got, sp.nsimplify(want)))

# Gaussian: below zero the lognormal mean, the rest negligible for a > s^2
gauss_below = sp.integrate(sp.exp(e - a) * sp.exp(-e ** 2 / (2 * s ** 2))
                           / (s * sp.sqrt(2 * sp.pi)), (e, -sp.oo, sp.oo))
print("Gaussian, leading term:", sp.simplify(gauss_below))

f_approx = sp.lambdify((a, b), approx, "numpy")


def exact_laplace(a_, b_):
    f = lambda x: special.expit(-a_ + x) * np.exp(-abs(x) / b_) / (2 * b_)
    return integrate.quad(f, -60, 60, points=[0, a_], limit=400)[0]


def exact_gauss(a_, s_):
    f = lambda x: special.expit(-a_ + x) * np.exp(-x ** 2 / (2 * s_ ** 2)) \
        / (s_ * np.sqrt(2 * np.pi))
    return integrate.quad(f, -60, 60, points=[0, a_], limit=400)[0]


centre = 0.0004                    # the waste population's Zn, as a fraction
a_ = float(np.log((1 - centre) / centre))
print("\ncentre %.4f %%, log-ratio a = %.2f" % (100 * centre, a_))
print("%-22s %12s %12s %12s" % ("noise", "exact mean", "closed form",
                                 "x centre"))
# 1.86: the record's waste population at the start (rate 1 in whitened
# units times the axis's clr spread 1.86); 0.715: at epoch 120 (rate 2.6)
for b_ in (0.5, 0.715, 0.9, 1.0 + 1e-6, 1.5, 1.86, 3.0):
    ex = exact_laplace(a_, b_)
    print("%-22s %11.4f%% %11.4f%% %12.1f" % (
        "Laplace b = %.2f" % b_, 100 * ex, 100 * f_approx(a_, b_),
        ex / centre))
for s_ in (0.32, 0.59, 1.0):
    ex = exact_gauss(a_, s_)
    print("%-22s %11.4f%% %11.4f%% %12.1f" % (
        "Gaussian sd = %.2f" % s_, 100 * ex,
        100 * np.exp(-a_ + s_ ** 2 / 2), ex / centre))
