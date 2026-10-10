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

"""A GP at uncertain inputs: what each way of carrying the input's
uncertainty gives, against the exact mixture.

Usage: python docs/benchmarks/uncertain_input_realizations.py

One dimension, a field with a smooth part and a sharp step, a
`GaussianInput` root and a `BasicGP`, trained on known locations and
predicted at locations known to a standard deviation of 0.4. Four ways:

- exact: the mixture over the input, by Monte Carlo (each location's input
  drawn, the posterior taken there);
- expected kernel: the GP with covariance E[k], its variance
  `1 - l^T C l` and its realizations `l(x) (alpha + R eps) + b`;
- noise-like: the same realizations, the second moment's extra variance
  integrated out beside them like a noise term (Gaussian at each location);
- drawn inputs: realization s at location i is `k(x_is, Z)(alpha + R eps_s)
  + b`, `x_is` drawn per location and realization.

Writes `figures/uncertain_input_line.png` (bands and realizations along the
line) and `figures/uncertain_input_marginals.png` (the distribution at a
location on the step and one away from it), and prints the 5% and 95%
quantiles' errors against the exact mixture.
"""

import os

import numpy as np
import tensorflow as tf
from scipy import stats

import geoml

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
INPUT_SD = 0.4
N_REAL = 400


def field(x):
    return np.sin(1.5 * x) + 0.8 * np.tanh(4.0 * (x - 6.0))


def fit():
    geoml.set_seed(3)
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 60))
    data = geoml.data.PointData.from_array(x[:, None], ["X"])
    data.add_continuous_variable("v", field(x) + 0.05 * rng.normal(size=60))
    inducing = geoml.data.inducing.from_grid(data, 0.4)
    root = geoml.latent.GaussianInput(inducing,
                                      transform=geoml.transform.Isotropic(1.0))
    gp = geoml.latent.BasicGP(root, size=1)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), gp,
        options=geoml.models.GPOptions(verbose=False))
    model.train_full(400)
    with model._propagation():
        model._refresh(model.options.jitter)
    return model, root, gp, x


def state(gp):
    z = gp.parent.inducing_points[0].numpy()
    alpha = gp.alpha[0].numpy()[0, :, 0]
    root_r = gp.chol_r[0].numpy()[0]
    inv = gp.cov_smooth_inv[0].numpy()[0]
    bias = float(np.asarray(gp.parameters["bias_0"].get_value()).ravel()[0])
    return z, alpha, root_r, inv, bias


def plain_k(gp, x, z):
    return gp.covariance_matrix(tf.constant(x), tf.constant(z)).numpy()


def main():
    model, root, gp, x_train = fit()
    z, alpha, root_r, inv, bias = state(gp)
    rng = np.random.default_rng(11)
    xq = np.linspace(0, 10, 300)
    u, var = root.propagate(tf.constant(xq[:, None]),
                            tf.constant(np.full([300, 1], INPUT_SD ** 2)))
    u, var = u.numpy(), var.numpy()

    # the expected kernel's moments
    l = gp.expected_covariance(tf.constant(u), tf.constant(var),
                               tf.constant(z), tf.zeros_like(tf.constant(z)),
                               None).numpy()
    mean = l @ alpha + bias
    var_ek = 1.0 - np.einsum("nm,ml,nl->n", l, inv, l)

    # the exact mixture: each location's input drawn, the GP at it
    draws = u + np.sqrt(var) * rng.normal(size=(300, 4000))
    k = plain_k(gp, draws.reshape(-1, 1), z)
    mu_d = (k @ alpha + bias).reshape(300, 4000)
    var_d = np.maximum(1.0 - np.einsum("nm,ml,nl->n", k, inv, k),
                       0.0).reshape(300, 4000)
    exact = mu_d + np.sqrt(var_d) * rng.normal(size=mu_d.shape)
    var_exact = var_d.mean(1) + mu_d.var(1)

    # realizations: the expected kernel's, and at drawn inputs
    eps = rng.normal(size=(z.shape[0], N_REAL))
    coef = alpha[:, None] + root_r @ eps                       # [m, s]
    real_ek = l @ coef + bias                                  # [n, s]
    x_drawn = u + np.sqrt(var) * rng.normal(size=(300, N_REAL))
    real_drawn = np.empty((300, N_REAL))
    for s in range(N_REAL):
        real_drawn[:, s] = plain_k(gp, x_drawn[:, s:s + 1], z) @ coef[:, s] \
            + bias
    # the 5% and 95% quantiles each approach implies. Realizations carry the
    # variance the inducing points explain; the rest of the posterior's is
    # added to the drawn-input ensemble as independent noise, so that its
    # quantiles answer the same question as the others'
    z90 = stats.norm.ppf(0.95)
    q = {
        "exact": np.quantile(exact, [0.05, 0.95], axis=1),
        "expected kernel": np.stack([mean - z90 * np.sqrt(var_ek),
                                     mean + z90 * np.sqrt(var_ek)]),
        "noise-like": np.stack([mean - z90 * np.sqrt(var_exact),
                                mean + z90 * np.sqrt(var_exact)]),
    }
    rest = np.sqrt(np.maximum(var_exact - real_drawn.var(1), 0.0))
    q["drawn inputs"] = np.quantile(
        real_drawn + rest[:, None] * rng.normal(size=real_drawn.shape),
        [0.05, 0.95], axis=1)
    print("5%/95% quantile error against the exact mixture, mean over the "
          "line (latent scale)")
    for name in ("expected kernel", "noise-like", "drawn inputs"):
        err = np.abs(q[name] - q["exact"]).mean(1)
        print("  %-16s 5%%: %.3f   95%%: %.3f" % (name, err[0], err[1]))
    on_step = np.argmin(np.abs(xq - 6.0))
    print("variance at x = 6: exact %.3f, expected kernel %.3f, "
          "expected-kernel realizations %.3f, drawn-input realizations %.3f"
          % (var_exact[on_step], var_ek[on_step], real_ek[on_step].var(),
             real_drawn[on_step].var()))

    figures(xq, x_train, mean, var_ek, var_exact, exact, real_ek,
            real_drawn, q, rng)


def figures(xq, x_train, mean, var_ek, var_exact, exact, real_ek,
            real_drawn, q, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES, exist_ok=True)
    panels = [
        ("exact mixture over the input", None, None),
        ("expected kernel (0.9.0)", var_ek, real_ek),
        ("second moment integrated out like noise", var_exact, real_ek),
        ("realizations at drawn inputs", None, real_drawn),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                             sharey=True, layout="constrained")
    for ax, (title, variance, real) in zip(axes, panels):
        ax.plot(xq, field(xq), color="black", linewidth=1, label="truth")
        ax.plot(x_train, np.full_like(x_train, -2.6), "|", color="black",
                markersize=8)
        ax.fill_between(xq, q["exact"][0], q["exact"][1], color="0.85",
                        label="exact 90% interval")
        if variance is not None:
            z90 = 1.6449
            ax.plot(xq, mean - z90 * np.sqrt(variance), color="C0",
                    linestyle="--")
            ax.plot(xq, mean + z90 * np.sqrt(variance), color="C0",
                    linestyle="--", label="its 90% interval")
        if real is not None:
            for s in range(4):
                ax.plot(xq, real[:, s], color="C%d" % (s + 1),
                        linewidth=0.9)
        else:
            for s in range(4):
                ax.plot(xq, exact[:, s], color="C%d" % (s + 1),
                        linewidth=0.9)
        ax.set_title(title)
        ax.set_ylabel("latent value")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("location (inputs known to a standard deviation of "
                        "%.1f)" % INPUT_SD)
    path = os.path.join(FIGURES, "uncertain_input_line.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("figure:", path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, where in zip(axes, (6.0, 2.0)):
        i = np.argmin(np.abs(xq - where))
        grid = np.linspace(-3, 3, 400)
        ax.hist(exact[i], bins=60, density=True, color="0.8",
                label="exact mixture")
        ax.hist(real_drawn[i], bins=40, density=True, histtype="step",
                color="C3", label="realizations at drawn inputs")
        ax.plot(grid, stats.norm.pdf(grid, mean[i], np.sqrt(var_ek[i])),
                color="C0", label="expected kernel")
        ax.plot(grid, stats.norm.pdf(grid, mean[i], np.sqrt(var_exact[i])),
                color="C1", linestyle="--",
                label="integrated like noise")
        ax.set_title("at x = %.0f%s" % (where, " (on the step)"
                                        if where == 6.0 else ""))
        ax.set_xlabel("latent value")
    axes[0].legend(fontsize=8)
    path = os.path.join(FIGURES, "uncertain_input_marginals.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("figure:", path)


if __name__ == "__main__":
    main()
