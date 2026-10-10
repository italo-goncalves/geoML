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

"""The Gaussian mixtures the expected kernel reads the Matern family
through.

Usage: python docs/benchmarks/kernel_mixtures.py [components]

For each of the exponential, Matern32 and Matern52 kernels (geoML's
definitions, the distance in ranges), the positive mixture of `components`
Gaussians `sum_q w_q exp(-w_q' d^2)` with weights summing to one -- so the
diagonal of a covariance stays exactly one -- closest to the kernel in the
largest error over [0, 6] ranges: a smoothed maximum minimized from several
starts, then the largest error itself. Prints the rates and weights as they
are pasted into `geoml/latent/network.py` (`_KERNEL_MIXTURES`), and each
kernel's largest error, which bounds the expected kernel's at any range and
any uncertainty since the expectation is linear in the mixture.
"""

import sys

import numpy as np
from scipy import optimize, special

KERNELS = {
    "Exponential": lambda d: np.exp(-3 * d),
    "Matern32": lambda d: (1 + 5 * d) * np.exp(-5 * d),
    "Matern52": lambda d: (1 + 6 * d + 12 * d ** 2) * np.exp(-6 * d),
}
# dense where the kernels bend most: at the origin (the exponential's cusp)
D = np.unique(np.concatenate([[0.0], np.geomspace(1e-5, 0.05, 150),
                              np.linspace(0.05, 2.5, 400),
                              np.linspace(2.5, 6.0, 60)]))


def mixture(p, q):
    rates = np.exp(p[:q])
    weights = special.softmax(p[q:])
    return np.exp(-np.outer(D ** 2, rates)) @ weights, rates, weights


def fit(name, q, seed=0):
    target = KERNELS[name](D)
    rng = np.random.default_rng(seed)
    best = None
    for trial in range(24):
        p = np.concatenate([np.sort(rng.uniform(-1.5, 11.0, q)),
                            rng.normal(0, 0.5, q)])
        for norm in (8, 32, 128):
            def loss(p, norm=norm):
                e = mixture(p, q)[0] - target
                scale = np.abs(e).max() + 1e-300
                return scale * np.mean((np.abs(e) / scale) ** norm) \
                    ** (1 / norm)
            p = optimize.minimize(loss, p, method="L-BFGS-B",
                                  options={"maxiter": 3000}).x
        err = np.abs(mixture(p, q)[0] - target).max()
        if best is None or err < best[0]:
            best = (err, p)

    # the largest error itself: minimize t with |error| <= t everywhere
    def objective(v):
        return v[-1]

    def bounds(v):
        e = mixture(v[:-1], q)[0] - target
        return np.concatenate([v[-1] - e, v[-1] + e])

    v0 = np.append(best[1], best[0])
    polished = optimize.minimize(objective, v0, method="SLSQP",
                                 constraints={"type": "ineq", "fun": bounds},
                                 options={"maxiter": 500, "ftol": 1e-14})
    p = polished.x[:-1]
    err = np.abs(mixture(p, q)[0] - target).max()
    if err > best[0]:
        p, err = best[1], best[0]
    _, rates, weights = mixture(p, q)
    order = np.argsort(rates)
    return rates[order], weights[order], err


def main(argv):
    q = int(argv[0]) if argv else 8
    print("# %d components each, weights summing to one" % q)
    for name in KERNELS:
        rates, weights, err = fit(name, q)
        print("# %s: largest error %.1e on [0, 6] ranges" % (name, err))
        print("_kr.%s: (\n    (%s),\n    (%s))," % (
            name, ", ".join("%.17g" % r for r in rates),
            ", ".join("%.17g" % w for w in weights)))


if __name__ == "__main__":
    main(sys.argv[1:])
