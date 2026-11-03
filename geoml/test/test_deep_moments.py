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

"""`GPOptions(deep_moments=)`: the moments of a GP node whose own inducing
points are uncertain, by the quadrature or by the closed forms of the
report on the expected kernel's closure -- the hinge, the blur and the
bound."""
import numpy as np
import pytest
import tensorflow as tf

import geoml
import geoml.latent.network as _net


def test_a_new_model_takes_the_hinge_and_an_older_save_the_quadrature():
    assert geoml.models.GPOptions().deep_moments == "hinge"
    old = geoml.models.GPOptions.__new__(geoml.models.GPOptions)
    assert old.deep_moments == "quadrature"
    with pytest.raises(ValueError, match="deep_moments"):
        geoml.models.GPOptions(deep_moments="exact")


def _two_layers(kernel, form, node=geoml.latent.BasicGP):
    """A `node` of `kernel` reading the coordinates and a GP below them."""
    x = np.random.default_rng(0).uniform(0, 10, (30, 2))
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", np.sin(x[:, 0]))
    root = geoml.latent.BasicInput(geoml.data.inducing.from_kmeans(
        data, 8, seed=0), transform=geoml.transform.Isotropic(5.0))
    leaf = node(geoml.latent.Concatenate(root, geoml.latent.BasicGP(root)),
                kernel=getattr(geoml.kernels, kernel)())
    return geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), leaf,
        options=geoml.models.GPOptions(verbose=False, deep_moments=form))


@pytest.mark.parametrize("kernel, form, says", [
    ("Gaussian", "bound", None),
    ("Gaussian", "hinge", None),
    ("Matern32", "hinge", None),
    ("Matern32", "bound", "36 pairs"),
    ("Exponential", "bound", "36 pairs"),
    ("RationalQuadratic", "bound", "takes the quadrature"),
    ("RationalQuadratic", "hinge", None),
])
def test_the_bound_warns_where_it_is_slow_or_falls_back(kernel, form, says,
                                                        recwarn):
    model = _two_layers(kernel, form)
    found = [str(w.message) for w in recwarn
             if "deep_moments='bound'" in str(w.message)]
    if says is None:
        assert not found
    else:
        assert len(found) == 1 and says in found[0]
    assert model.latent_network._form_for(form) == (
        "quadrature" if (kernel, form) == ("RationalQuadratic", "bound")
        else form)


def test_an_additive_gp_takes_the_quadrature_or_the_bound_s_fallback():
    for form in ("hinge", "blur"):
        with pytest.raises(ValueError, match="AdditiveGP|averages"):
            _two_layers("Gaussian", form, geoml.latent.AdditiveGP)
    with pytest.warns(UserWarning, match="takes the quadrature"):
        model = _two_layers("Gaussian", "bound", geoml.latent.AdditiveGP)
    assert model.latent_network._form_for("bound") == "quadrature"
    _two_layers("Gaussian", "quadrature", geoml.latent.AdditiveGP)


def test_the_bound_s_divergence_majorizes_the_divergence():
    # at the same covariance, variational variances and means, eigenvalue by
    # eigenvalue: equal where the covariance vanishes, above it elsewhere
    rng = np.random.default_rng(4)
    node = geoml.latent.BasicGP(geoml.latent.BasicInput(
        geoml.data.PointData.from_array(rng.normal(size=[6, 2]))), size=2)
    node._kl_jitter = 0.0
    for scale in (0.0, 0.01, 1.0, 100.0):
        a = rng.normal(size=[6, 6])
        cov = scale * a @ a.T
        delta = rng.uniform(0.1, 2.0, size=[2, 6])
        fit = float(np.sum(rng.normal(size=[2, 6, 1]) ** 2))
        stock = sum(0.5 * (- np.trace(np.linalg.solve(cov + np.diag(d), cov))
                           + np.linalg.slogdet(cov + np.diag(d))[1]
                           - np.sum(np.log(d))) for d in delta) + 0.5 * fit
        bound = float(node._majorant_kl(tf.constant(cov), tf.constant(delta),
                                        tf.constant(fit, tf.float64)))
        if scale == 0.0:
            assert bound == pytest.approx(stock, abs=1e-12)
        else:
            assert bound > stock


def _joint_law(rng, n, m, d):
    """A jointly Gaussian input `h` at `n` locations and `m` inducing
    points, dimension by dimension: the moments `_pair_terms` reads, and
    the joint covariance per dimension for drawing from it."""
    mean = rng.normal(size=[n, d])
    points = rng.normal(size=[m, d])
    joint = []
    for _ in range(d):
        a = rng.normal(size=[n + m, n + m]) * 0.4
        joint.append(a @ a.T / (n + m) + 0.05 * np.eye(n + m))
    joint = np.stack(joint)                                  # [d, n+m, n+m]
    var = np.stack([np.diag(s)[:n] for s in joint], -1)
    cov = np.stack([s[:n, n:] for s in joint], -1)
    points_var = np.stack([np.diag(s)[n:] for s in joint], -1)
    points_cov = np.stack([s[n:, n:] for s in joint], -1)
    return dict(mean=mean, var=var, cov=cov, points=points,
                points_var=points_var, points_cov=points_cov), joint


def _psi_monte_carlo(law, joint, scales, weights, alpha, draws, rng):
    """`alphaᵀ Psi alpha` by drawing the input: its mean and its standard
    error, `[size, n]` each."""
    n, d = law["mean"].shape
    centre = np.concatenate([law["mean"], law["points"]], 0)  # [n+m, d]
    eps = np.stack([rng.multivariate_normal(np.zeros(len(s)), s, draws)
                    for s in joint], -1)                     # [draws, n+m, d]
    x = centre[None] + eps
    h, z = x[:, :n], x[:, n:]
    d2 = (h[:, :, None, :] - z[:, None, :, :]) ** 2          # [draws, n, m, d]
    k = sum(w * np.exp(-0.5 * np.sum(d2 / s, -1))
            for w, s in zip(weights, scales))                # [draws, n, m]
    f2 = np.einsum("tnm,sm->tsn", k, alpha) ** 2
    return f2.mean(0), f2.std(0) / np.sqrt(draws)


@pytest.mark.parametrize("kernel", ["Gaussian", "Matern32"])
def test_the_bound_s_second_moment_is_the_mixture_s(kernel):
    rng = np.random.default_rng(2026)
    law, joint = _joint_law(rng, n=4, m=3, d=2)
    alpha = rng.normal(size=[2, 3])
    ranges = tf.constant([[[1.3, 0.8]]], tf.float64)
    scales, pairs = _net._pair_items(getattr(geoml.kernels, kernel)(),
                                     ranges, 2)
    exact = _net._pair_terms(scales, *[tf.constant(law[k]) for k in (
        "mean", "var", "cov", "points", "points_var", "points_cov")],
        tf.constant(alpha), pairs).numpy()
    omega, w, _ = _net._kernel_components(getattr(geoml.kernels, kernel)())
    drawn, error = _psi_monte_carlo(law, joint, scales.numpy(), w.numpy(),
                                    alpha, 400000, rng)
    assert np.all(np.abs(exact - drawn) < 5 * error + 1e-12)


def _scalar(xla, monkeypatch):
    """`sum(upstream * alphaᵀ Psi alpha)` as a function of the input's mean
    and the scales, compiled or not."""
    monkeypatch.setattr(_net, "_quadrature_xla", lambda: xla)
    rng = np.random.default_rng(7)
    law, _ = _joint_law(rng, n=5, m=4, d=3)
    rest = [tf.constant(law[k]) for k in (
        "var", "cov", "points", "points_var", "points_cov")]
    alpha = tf.constant(rng.normal(size=[1, 4]))
    upstream = tf.constant(rng.normal(size=[1, 5]))
    _, pairs = _net._pair_items(geoml.kernels.Gaussian(),
                                tf.ones([1, 1, 3], tf.float64), 3)

    def f(mean, scales):
        return tf.reduce_sum(upstream * _net._pair_terms(
            scales, mean, *rest, alpha, pairs))

    return f, tf.constant(law["mean"]), tf.constant(
        rng.uniform(0.5, 2.0, size=[1, 3]))


@pytest.mark.parametrize("xla", [True, False])
def test_the_second_moment_s_gradient_and_hessian_are_its_own(xla,
                                                              monkeypatch):
    f, mean, scales = _scalar(xla, monkeypatch)
    direction_m = tf.constant(np.random.default_rng(1).normal(
        size=mean.shape))
    direction_s = tf.constant(np.random.default_rng(2).normal(
        size=scales.shape)) * 0.1

    def gradient(m, s):
        with tf.GradientTape() as tape:
            tape.watch([m, s])
            value = f(m, s)
        return tape.gradient(value, [m, s])

    # the gradient against central differences along a direction
    step = 1e-5
    g_m, g_s = gradient(mean, scales)
    slope = float(tf.reduce_sum(g_m * direction_m)
                  + tf.reduce_sum(g_s * direction_s))
    numeric = (float(f(mean + step * direction_m, scales + step * direction_s))
               - float(f(mean - step * direction_m,
                         scales - step * direction_s))) / (2 * step)
    assert abs(slope - numeric) < 1e-7 * max(1.0, abs(numeric))

    # the Hessian along the direction, through the custom gradient, against
    # differences of the gradient
    with tf.GradientTape() as outer:
        outer.watch([mean, scales])
        g_m, g_s = gradient(mean, scales)
        projected = tf.reduce_sum(g_m * direction_m) \
            + tf.reduce_sum(g_s * direction_s)
    h_m, h_s = outer.gradient(projected, [mean, scales])
    up = gradient(mean + step * direction_m, scales + step * direction_s)
    down = gradient(mean - step * direction_m, scales - step * direction_s)
    for exact, a, b in zip((h_m, h_s), up, down):
        numeric = (a - b).numpy() / (2 * step)
        np.testing.assert_allclose(exact.numpy(), numeric, rtol=1e-5,
                                   atol=1e-7)


@pytest.mark.parametrize("xla", [True, False])
def test_the_second_moment_in_chunks_of_rows_is_the_whole(xla, monkeypatch):
    # rows a few at a time, the last chunk padded, both passes: the same
    # value and gradient, eagerly and traced with the rows unknown
    monkeypatch.setattr(_net, "_quadrature_xla", lambda: xla)
    rng = np.random.default_rng(9)
    law, _ = _joint_law(rng, n=11, m=4, d=3)
    args = [tf.constant(law[k]) for k in ("mean", "var", "cov", "points",
                                          "points_var", "points_cov")]
    alpha = tf.constant(rng.normal(size=[2, 4]))
    upstream = tf.constant(rng.normal(size=[2, 11]))
    scales, pairs = _net._pair_items(geoml.kernels.Matern32(),
                                     tf.ones([1, 1, 3], tf.float64), 3)

    def run(mean, var, cov, *rest):
        with tf.GradientTape() as tape:
            tape.watch([scales, mean, var, cov, alpha])
            value = _net._pair_terms(scales, mean, var, cov, *rest, alpha,
                                     pairs)
            total = tf.reduce_sum(value * upstream)
        return [value] + tape.gradient(total, [scales, mean, var, cov,
                                               alpha])

    whole = run(*args)
    # 4 rows a chunk at m = 4, d = 3 and 36 pairs
    monkeypatch.setattr(_net, "_QUADRATURE_MEMORY",
                        4 * _net._PAIR_BYTES * 36 * 4 * 4 * 3)
    chunked = run(*args)
    traced = tf.function(run, input_signature=[
        tf.TensorSpec([None, 3], tf.float64), tf.TensorSpec([None, 3],
                                                           tf.float64),
        tf.TensorSpec([None, 4, 3], tf.float64)] + [
        tf.TensorSpec(a.shape, tf.float64) for a in args[3:]])(*args)
    for a, b, c in zip(whole, chunked, traced):
        np.testing.assert_allclose(b.numpy(), a.numpy(), rtol=1e-12,
                                   atol=1e-14)
        np.testing.assert_allclose(c.numpy(), a.numpy(), rtol=1e-12,
                                   atol=1e-14)


def _node_moments(kernel, law, size=2, seed=3, scale=1.0):
    """A GP node with `size` outputs over `d` inputs, and what
    `_deep_moments` reads at the law's input: its arguments as tensors."""
    rng = np.random.default_rng(seed)
    n, d = law["mean"].shape
    m = law["points"].shape[0]
    root = geoml.latent.BasicInput(
        geoml.data.PointData.from_array(law["points"]))
    node = geoml.latent.BasicGP(root, size=size,
                                kernel=getattr(geoml.kernels, kernel)())
    node.parameters["ranges"].set_value(
        rng.uniform(1.0, 2.0, size=[1, 1, d]))
    t = {k: tf.constant(v * (scale if k in ("var", "cov", "points_var",
                                            "points_cov") else 1.0))
         for k, v in law.items()}
    a = rng.normal(size=[size, m, m])
    smooth_inv = tf.constant(a @ np.transpose(a, [0, 2, 1]) / m
                             + 0.1 * np.eye(m))
    root_r = tf.constant(np.tril(rng.normal(size=[size, m, m])) * 0.2)
    alpha = tf.constant(rng.normal(size=[size, m, 1]))
    l = _net._expected_kernel(node.kernel,
                              node.parameters["ranges"].get_value(),
                              t["mean"], t["var"], t["points"],
                              t["points_var"], t["cov"])
    offset = tf.einsum("nm,sm->sn", l, alpha[..., 0])
    return node, (t["mean"], t["var"], t["cov"], t["points"],
                  t["points_var"], t["points_cov"], smooth_inv, alpha,
                  offset, l, root_r)


@pytest.mark.parametrize("form", ["hinge", "blur", "bound"])
def test_the_outputs_and_the_experts_are_each_their_own(form, monkeypatch):
    # an output's moments are what it gives alone, and a leading axis -- the
    # slots of a pass by expert -- is a stack of separate calls
    monkeypatch.setattr(_net, "_DEEP_MOMENTS", form)
    rng = np.random.default_rng(11)
    laws = [_joint_law(rng, n=5, m=4, d=3)[0] for _ in range(2)]
    calls = [_node_moments("Matern32", law, seed=s)
             for s, law in enumerate(laws)]
    node = calls[0][0]
    whole = [node._deep_moments(*args) for _, args in calls]
    for args, together in zip([c[1] for c in calls], whole):
        for s in range(2):
            alone = node._deep_moments(*(
                args[:6] + tuple(x[s:s + 1] for x in args[6:9]) + args[9:10]
                + (args[10][s:s + 1],)))
            for a, b in zip(together, alone):
                np.testing.assert_allclose(a.numpy()[s:s + 1], b.numpy(),
                                           rtol=1e-12, atol=1e-14)
    # the same node over both laws at once, along a leading axis
    stacked = [tf.stack([c[1][k] for c in calls]) for k in range(11)]
    batched = node._deep_moments(*stacked)
    for k, b in enumerate(batched):
        np.testing.assert_allclose(
            b.numpy(), np.stack([w[k].numpy() for w in whole]),
            rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kernel", ["Gaussian", "Matern32",
                                    "RationalQuadratic"])
def test_the_hinge_and_the_blur_are_the_input_s_average_where_it_is_tight(
        kernel, monkeypatch):
    # at a small input variance the linearized kernel is the conditioned
    # one: the hinge is the average of the clamped conditional variance, and
    # the blur's spread the spread of the conditioned kernel's mean
    rng = np.random.default_rng(5)
    law, joint = _joint_law(rng, n=4, m=3, d=2)
    node, args = _node_moments(kernel, law, size=1, scale=0.01)
    mean, var, cov, points, points_var, _, smooth_inv, alpha = args[:8]
    eps = rng.normal(size=[20000, 4, 2])
    h = mean[:, None, :] + tf.sqrt(var)[:, None, :] * tf.constant(
        np.transpose(eps, [1, 0, 2]))                       # [n, S, d]
    omega, weights, constant = _net._kernel_components(node.kernel)
    k = _net._conditioned_kernel(
        h, mean, var, cov, points, points_var,
        node.parameters["ranges"].get_value() ** 2, omega, weights, constant,
        kernel == "RationalQuadratic").numpy()               # [n, S, m]
    clamped = np.maximum(1.0 - np.einsum("nSi,ij,nSj->nS", k,
                                         smooth_inv[0].numpy(), k), 0.0)
    f = np.einsum("nSm,m->nS", k, alpha[0, :, 0].numpy())
    monkeypatch.setattr(_net, "_DEEP_MOMENTS", "hinge")
    total, explained, _, _ = node._deep_moments(*args)
    spread = total.numpy()[0] - (1.0 - explained.numpy()[0])
    np.testing.assert_allclose(1.0 - explained.numpy()[0],
                               clamped.mean(1), rtol=0.02, atol=1e-4)
    np.testing.assert_allclose(spread, f.var(1), rtol=0.05, atol=1e-6)
    monkeypatch.setattr(_net, "_DEEP_MOMENTS", "blur")
    blur = node._deep_moments(*args)[0].numpy()[0]
    np.testing.assert_allclose(blur - spread, np.maximum(
        1.0 - np.einsum("ni,ij,nj->n", args[9].numpy(), smooth_inv[0].numpy(),
                        args[9].numpy()), 0.0), rtol=1e-12)


@pytest.mark.parametrize("form, low, high", [
    ("hinge", 0.9, 1.1), ("blur", 0.9, 1.1),
    # the bound's spread holds the inducing points' own spread given the
    # input, which draws at the conditioned kernel leave out: measured 0.88
    ("bound", 0.78, 0.97)])
def test_the_realizations_carry_the_variance_the_form_reports(form, low,
                                                              high):
    geoml.set_seed(7)
    x = np.random.default_rng(3).uniform(0, 100, [60, 2])
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 20) + 0.3 * np.cos(x[:, 1] / 15))
    root = geoml.latent.BasicInput(
        geoml.data.inducing.from_kmeans(data, 24, seed=0),
        geoml.transform.Isotropic(40))
    leaf = geoml.latent.BasicGP(geoml.latent.BasicGP(root, size=2), size=1)
    model = geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.Identity(1)), leaf,
        options=geoml.models.GPOptions(verbose=False, deep_moments=form))
    model.train_full(30)
    target = geoml.data.PointData.from_array(
        np.random.default_rng(5).uniform(0, 100, [30, 2]), ["X", "Y"])
    model.predict(target, n_sim=1000)
    sims = np.asarray(target.variables["v"].get_simulations(), dtype=float)
    reported = np.asarray(target.values("v/latent_variance"), dtype=float)
    ratio = float(np.median(np.var(sims, axis=1) / reported.ravel()))
    print("variance ratio under %s: %.3f" % (form, ratio))
    assert low < ratio < high


def _deep_experts(form):
    """A GP reading the coordinates and the GP below them, four experts."""
    geoml.set_seed(1234)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (400, 2))
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", np.sin(x[:, 0] / 15) + np.cos(
        x[:, 1] / 20) + 0.1 * rng.normal(size=400))
    ip = geoml.data.inducing.from_kmeans(data, 160, seed=0)
    root = geoml.latent.BasicInput(geoml.data.inducing.experts(ip, 4, seed=0),
                                   transform=geoml.transform.Isotropic(20.0))
    first = geoml.latent.BasicGP(root, size=1)
    leaf = geoml.latent.BasicGP(geoml.latent.Concatenate(root, first), size=1)
    return geoml.models.VGPNetwork(
        data, "v", geoml.likelihood.Gaussian(geoml.warping.ZScore(1)), leaf,
        options=geoml.models.GPOptions(verbose=False, training_batch_size=100,
                                       deep_moments=form))


@pytest.mark.parametrize("form", ["quadrature", "hinge", "blur", "bound"])
def test_the_slots_train_and_predict_as_the_sets_do(form):
    a, b = _deep_experts(form), _deep_experts(form)
    ra = a.train_by_expert(2, batch_size=50, visits=2, slots=False,
                           weights_every=10, sampling="replacement")
    rb = b.train_by_expert(2, batch_size=50, visits=2, slots=True,
                           weights_every=10, decay="steps",
                           sampling="replacement")
    np.testing.assert_allclose(ra["bound"], rb["bound"], rtol=1e-8)
    # every expert in the slots against every expert by set: the same to
    # rounding
    g = np.linspace(2, 98, 30)
    targets = np.column_stack([c.ravel() for c in np.meshgrid(g, g)])
    pa = geoml.data.PointData.from_array(targets, ["X", "Y"])
    pb = geoml.data.PointData.from_array(targets, ["X", "Y"])
    a.predict(pa, n_sim=5)
    a.predict_by_expert(pb, n_sim=5, coverage=1.0)
    for path in ("v/prediction", "v/latent_variance"):
        np.testing.assert_allclose(pa.values(path), pb.values(path),
                                   rtol=1e-10, atol=1e-12)
